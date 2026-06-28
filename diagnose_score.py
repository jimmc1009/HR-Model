"""
diagnose_score.py
Comprehensive analysis to make the HR score fully monotonic and meaningful.
Uses the ACTUAL scoring functions from hr_picks.py with correct column names
from HR_All_Scores.

Sections:
1. Hit rate by tier — confirm the problem
2. Score component analysis — power vs context by tier
3. Pure power score monotonicity
4. Context score hits vs misses by tier
5. Context cap simulation
6. Power weight analysis
7. Feature separators by tier
"""

import os
import json
import time

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MODEL_START_DATE = "2026-06-09"
MIN_SAMPLE       = 10

# ── Exact constants from hr_picks.py ─────────────────────────────────────
LEAGUE_AVG_HR_PER_PA     = 2.5
LEAGUE_AVG_HR_PER_FB     = 10.0
LEAGUE_AVG_ISO           = 0.155
LEAGUE_AVG_BARREL_7D     = 8.0
LEAGUE_AVG_SEASON_BARREL = 8.0
MIN_PA_FULL              = 150
MIN_BBE_7D_FULL          = 20
MIN_BBE_7D_PARTIAL       = 5

PITCH_MATCHUP_WEIGHT = 1.9
MOMENTUM_WEIGHT      = 0.5
WEATHER_WEIGHT       = 1.0
PLATOON_WEIGHT       = 1.8


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=15):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def regress(value, league_avg, sample, full_sample):
    weight = min(sample / full_sample, 1.0)
    return (value * weight) + (league_avg * (1 - weight))


# ── Exact scoring functions from hr_picks.py ─────────────────────────────
def score_barrel_pct_7d(v, bbe):
    if bbe < MIN_BBE_7D_PARTIAL: return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe, MIN_BBE_7D_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    return 0.0

def score_barrel_pct_5d(v, bbe):
    if bbe < MIN_BBE_7D_PARTIAL: return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe, MIN_BBE_7D_FULL)
    if v >= 20: return 2.0
    if v >= 15: return 1.5
    if v >= 10: return 1.0
    if v >= 6:  return 0.3
    return 0.0

def score_barrel_pct_10d(v, bbe):
    if bbe < MIN_BBE_7D_PARTIAL: return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe, MIN_BBE_7D_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    return 0.0

def score_season_barrel(v, pa):
    v = regress(v, LEAGUE_AVG_SEASON_BARREL, pa, MIN_PA_FULL)
    if v >= 14: return 1.5
    if v >= 11: return 1.0
    if v >= 9:  return 0.6
    if v >= 7:  return 0.3
    return 0.0

def score_hr_per_fb(v, pa):
    v = regress(v, LEAGUE_AVG_HR_PER_FB, pa, MIN_PA_FULL)
    if v >= 20: return 1.5
    if v >= 15: return 1.0
    if v >= 10: return 0.5
    return 0.0

def score_hr_per_pa(v, pa):
    v = regress(v, LEAGUE_AVG_HR_PER_PA, pa, MIN_PA_FULL)
    if v >= 6.0: return 1.5
    if v >= 4.0: return 1.0
    if v >= 2.5: return 0.5
    return 0.0

def score_iso(v, pa):
    v = regress(v, LEAGUE_AVG_ISO, pa, MIN_PA_FULL)
    if v >= 0.300: return 1.0
    if v >= 0.250: return 0.7
    if v >= 0.200: return 0.5
    if v >= 0.175: return 0.3
    if v >= 0.150: return 0.1
    return 0.0

def score_pitcher_barrel(v):
    if v >= 14: return 0.6
    if v >= 11: return 0.4
    if v >= 9:  return 0.2
    if v >= 7:  return 0.1
    return 0.0

def score_pitcher_hr_per_fb(v):
    if v >= 20: return 0.5
    if v >= 15: return 0.3
    if v >= 13: return 0.2
    if v >= 10: return 0.1
    return 0.0

def score_weather(v):
    if v >= 2.5:  return 0.8
    if v >= 1.5:  return 0.6
    if v >= 1.0:  return 0.4
    if v >= 0.5:  return 0.2
    if v >= 0.0:  return 0.0
    if v >= -0.5: return -0.2
    if v >= -1.5: return -0.4
    return -0.6

def hr(sub):
    if len(sub) == 0: return 0.0, 0, 0
    h = int(sub["hit_bool"].sum())
    return round(h / len(sub) * 100, 1), h, len(sub)


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    sh         = with_retry(lambda: gc.open_by_key(sheet_id))
    ws         = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers    = all_values[0]
    rows       = all_values[1:]
    df         = pd.DataFrame(rows, columns=headers)
    df         = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()

    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    print(f"  {len(resolved)} resolved rows from {MODEL_START_DATE}")

    # Coerce all numeric cols
    num_cols = [
        "hr_score", "consensus_odds",
        "barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d", "season_barrel_pct",
        "hr_per_pa", "hr_per_fb", "iso",
        "pa", "bbe_7d", "bbe_5d", "bbe_10d",
        "pitcher_barrel_pct", "pitcher_hr_per_fb",
        "platoon_score", "pitch_matchup_score", "momentum_score",
        "park_hr_factor", "hr_weather_boost",
        "avg_ev_7d", "avg_la_7d",
    ]
    for col in num_cols:
        if col in resolved.columns:
            resolved[col] = resolved[col].apply(safe_float)
        else:
            resolved[col] = 0.0

    resolved["odds_num"] = resolved["consensus_odds"].apply(safe_float)

    # ── Check if pa/bbe cols are populated ───────────────────────────────
    has_bbe = resolved["bbe_7d"].sum() > 0
    has_pa  = resolved["pa"].sum() > 0
    print(f"  pa column populated: {has_pa} (sum={resolved['pa'].sum():.0f})")
    print(f"  bbe_7d column populated: {has_bbe} (sum={resolved['bbe_7d'].sum():.0f})")

    if not has_bbe or not has_pa:
        print("\n  ⚠️  pa/bbe columns not yet populated in HR_All_Scores.")
        print("  These were just added to log_all_scores() — run the main pipeline first.")
        print("  Using raw (unregressed) values as approximation for now...")
        # Use raw values without regression as fallback
        resolved["bbe_7d"]  = 20.0  # assume full sample — will be close enough for structure
        resolved["bbe_5d"]  = 20.0
        resolved["bbe_10d"] = 20.0
        resolved["pa"]      = 150.0

    # ── Compute power and context scores ─────────────────────────────────
    print("Computing power/context scores...")

    def compute_power(r):
        b7  = score_barrel_pct_7d(r["barrel_pct_7d"], r["bbe_7d"])
        b5  = score_barrel_pct_5d(r["barrel_pct_5d"], r["bbe_5d"])
        b10 = score_barrel_pct_10d(r["barrel_pct_10d"], r["bbe_10d"])
        bs  = score_season_barrel(r["season_barrel_pct"], r["pa"])
        fb  = score_hr_per_fb(r["hr_per_fb"], r["pa"])
        pa  = score_hr_per_pa(r["hr_per_pa"], r["pa"])
        iso = score_iso(r["iso"], r["pa"])
        pb  = score_pitcher_barrel(r["pitcher_barrel_pct"])
        pfb = score_pitcher_hr_per_fb(r["pitcher_hr_per_fb"])
        return round(b7 + b5 + b10 + bs + fb + pa + iso + pb + pfb, 3)

    def compute_context(r):
        pm  = min(r["pitch_matchup_score"], 1.0) * PITCH_MATCHUP_WEIGHT
        pl  = max(-2.0, min(2.0, r["platoon_score"]))
        mo  = max(-0.5, min(1.5, r["momentum_score"])) * MOMENTUM_WEIGHT
        wx  = score_weather(r["hr_weather_boost"]) * WEATHER_WEIGHT
        return round(pm + pl + mo + wx, 3)

    resolved["power_score"]   = resolved.apply(compute_power, axis=1)
    resolved["context_score"] = resolved.apply(compute_context, axis=1)
    resolved["recomputed_score"] = resolved["power_score"] + resolved["context_score"]

    # Check alignment with logged score
    score_diff = (resolved["recomputed_score"] - resolved["hr_score"]).abs().mean()
    print(f"  Avg difference between recomputed and logged score: {score_diff:.3f}")
    print(f"  Power score range: {resolved['power_score'].min():.2f} to {resolved['power_score'].max():.2f}")
    print(f"  Context score range: {resolved['context_score'].min():.2f} to {resolved['context_score'].max():.2f}")

    tiers = [
        ("8.5-9",  8.5,  9.0),
        ("9-10",   9.0, 10.0),
        ("10-11", 10.0, 11.0),
        ("11-12", 11.0, 12.0),
        ("12-13", 12.0, 13.0),
        ("13+",   13.0, 99.0),
    ]

    # ── 1. Hit rate by tier ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("1. HIT RATE BY TIER (logged score)")
    print("="*70)
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        r, h, n = hr(sub)
        print(f"  {label:8s}: {h:3d}/{n:3d} = {r:5.1f}%")

    # ── 2. Score composition by tier ─────────────────────────────────────
    print("\n" + "="*70)
    print("2. SCORE COMPOSITION BY TIER — Power vs Context")
    print("="*70)
    print(f"\n  {'Tier':8s} {'Avg Score':>10} {'Avg Power':>10} {'Avg Context':>12} {'Context%':>10} {'Hit Rate':>10}")
    print(f"  {'-'*62}")
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        avg_score   = sub["hr_score"].mean()
        avg_power   = sub["power_score"].mean()
        avg_context = sub["context_score"].mean()
        ctx_pct     = avg_context / avg_score * 100 if avg_score > 0 else 0
        r, h, n     = hr(sub)
        print(f"  {label:8s} {avg_score:>10.2f} {avg_power:>10.2f} {avg_context:>12.2f} {ctx_pct:>9.1f}% {r:>9.1f}%")

    # ── 3. Pure power score monotonicity ─────────────────────────────────
    print("\n" + "="*70)
    print("3. PURE POWER SCORE — Hit rate by power score tier")
    print("   (If monotonic, power alone predicts HRs; if not, context helps)")
    print("="*70)
    power_tiers = [
        ("Power 0-2",  0,  2),
        ("Power 2-3",  2,  3),
        ("Power 3-4",  3,  4),
        ("Power 4-5",  4,  5),
        ("Power 5-6",  5,  6),
        ("Power 6-7",  6,  7),
        ("Power 7-8",  7,  8),
        ("Power 8+",   8, 99),
    ]
    print(f"\n  {'Power Tier':14s} {'Hit Rate':>10} {'N':>6} {'Avg Power':>10}")
    print(f"  {'-'*44}")
    prev = 0
    for label, lo, hi in power_tiers:
        sub = resolved[(resolved["power_score"] >= lo) & (resolved["power_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        r, h, n = hr(sub)
        flag = " ⚠️" if prev > 0 and r < prev - 2 else ""
        prev = r
        print(f"  {label:14s} {r:>9.1f}% {n:>6}  {sub['power_score'].mean():>10.2f}{flag}")

    # ── 4. Context score hits vs misses by tier ───────────────────────────
    print("\n" + "="*70)
    print("4. CONTEXT SCORE BY TIER — Hits vs Misses")
    print("   (Is context inflating misses or deflating hits?)")
    print("="*70)
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        hits   = sub[sub["hit_bool"]]
        misses = sub[~sub["hit_bool"]]
        r, h, n = hr(sub)
        h_ctx = hits["context_score"].mean()
        m_ctx = misses["context_score"].mean()
        h_pow = hits["power_score"].mean()
        m_pow = misses["power_score"].mean()
        h_pl  = hits["platoon_score"].mean()
        m_pl  = misses["platoon_score"].mean()
        h_pm  = hits["pitch_matchup_score"].mean()
        m_pm  = misses["pitch_matchup_score"].mean()
        print(f"\n  {label} ({r:.1f}% hit rate, n={n}):")
        print(f"    Power score   — hits: {h_pow:.2f}  misses: {m_pow:.2f}  diff: {h_pow-m_pow:+.2f}")
        print(f"    Context score — hits: {h_ctx:.2f}  misses: {m_ctx:.2f}  diff: {h_ctx-m_ctx:+.2f}")
        print(f"    Platoon score — hits: {h_pl:.2f}  misses: {m_pl:.2f}  diff: {h_pl-m_pl:+.2f}")
        print(f"    Pitch matchup — hits: {h_pm:.2f}  misses: {m_pm:.2f}  diff: {h_pm-m_pm:+.2f}")

    # ── 5. Context cap simulation ─────────────────────────────────────────
    print("\n" + "="*70)
    print("5. CONTEXT CAP SIMULATION — Does capping context fix monotonicity?")
    print("="*70)
    for cap in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        capped = resolved["power_score"] + resolved["context_score"].clip(upper=cap)
        print(f"\n  Context cap = {cap}:")
        prev_rate = 0
        monotonic = True
        for label, lo, hi in tiers:
            sub = resolved[(capped >= lo) & (capped < hi)]
            if len(sub) < MIN_SAMPLE: continue
            r, h, n = hr(sub)
            flag = " ⚠️ drops" if prev_rate > 0 and r < prev_rate - 2 else ""
            if flag: monotonic = False
            prev_rate = r
            print(f"    {label:8s}: {h:3d}/{n:3d} = {r:5.1f}%{flag}")
        print(f"    Monotonic: {'✅ YES' if monotonic else '❌ NO'}")

    # ── 6. Platoon weight analysis ────────────────────────────────────────
    print("\n" + "="*70)
    print("6. PLATOON WEIGHT ANALYSIS — What weight produces best monotonicity?")
    print("="*70)
    for pl_weight in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        # Recompute context with different platoon weight
        def ctx_with_pl(r, w):
            pm  = min(r["pitch_matchup_score"], 1.0) * PITCH_MATCHUP_WEIGHT
            pl  = max(-2.0, min(2.0, r["platoon_score"])) * w
            mo  = max(-0.5, min(1.5, r["momentum_score"])) * MOMENTUM_WEIGHT
            wx  = score_weather(r["hr_weather_boost"]) * WEATHER_WEIGHT
            return pm + pl + mo + wx

        recomputed = resolved["power_score"] + resolved.apply(lambda r: ctx_with_pl(r, pl_weight), axis=1)
        print(f"\n  Platoon weight = {pl_weight}x:")
        prev_rate = 0
        monotonic = True
        for label, lo, hi in tiers:
            sub = resolved[(recomputed >= lo) & (recomputed < hi)]
            if len(sub) < MIN_SAMPLE: continue
            r, h, n = hr(sub)
            flag = " ⚠️" if prev_rate > 0 and r < prev_rate - 2 else ""
            if flag: monotonic = False
            prev_rate = r
            print(f"    {label:8s}: {h:3d}/{n:3d} = {r:5.1f}%{flag}")
        print(f"    Monotonic: {'✅ YES' if monotonic else '❌ NO'}")

    # ── 7. Feature separators by tier ────────────────────────────────────
    print("\n" + "="*70)
    print("7. FEATURE SEPARATORS BY TIER")
    print("="*70)
    feature_cols = [
        ("power_score",        "Power Score"),
        ("context_score",      "Context Score"),
        ("platoon_score",      "Platoon Score"),
        ("pitch_matchup_score","Pitch Matchup"),
        ("season_barrel_pct",  "Season Barrel%"),
        ("barrel_pct_7d",      "Barrel% 7d"),
        ("hr_per_fb",          "HR/FB%"),
        ("iso",                "ISO"),
        ("pitcher_barrel_pct", "Pitcher Barrel%"),
        ("avg_ev_7d",          "Avg EV 7d"),
        ("momentum_score",     "Momentum"),
        ("hr_weather_boost",   "Weather Boost"),
    ]
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        hits   = sub[sub["hit_bool"]]
        misses = sub[~sub["hit_bool"]]
        r, h, n = hr(sub)
        print(f"\n  {label} ({r:.1f}% hit rate, {n} picks):")
        seps = []
        for col, flabel in feature_cols:
            if col not in sub.columns: continue
            h_avg = hits[col].dropna().mean()
            m_avg = misses[col].dropna().mean()
            if pd.isna(h_avg) or pd.isna(m_avg) or m_avg == 0: continue
            pct = (h_avg - m_avg) / abs(m_avg) * 100
            seps.append((pct, flabel, h_avg, m_avg))
        seps.sort(key=lambda x: abs(x[0]), reverse=True)
        for pct, flabel, h_avg, m_avg in seps[:6]:
            flag = "✅" if pct >= 10 else "❌" if pct <= -10 else "  "
            print(f"    {flag} {flabel:<22} hits={h_avg:.3f}  misses={m_avg:.3f}  ({pct:+.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
