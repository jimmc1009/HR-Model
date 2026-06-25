"""
diagnose_score.py
Comprehensive analysis to make the HR score fully monotonic and meaningful.

Goal: understand why score doesn't descend cleanly in hit rate,
identify what's inflating scores without power backing,
and find the right balance between power and context components.

Sections:
1. Hit rate by tier — confirm the problem
2. Score component analysis — how much does each component contribute by tier
3. Power score vs context score — are players reaching tiers via power or context?
4. Feature separators by tier — what predicts HRs in each tier
5. Context cap analysis — what happens if we limit context contribution
6. Pure power score — if we remove context, does hit rate become monotonic?
7. Optimal weight analysis — what weights produce the most monotonic score
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


def hr(sub):
    if len(sub) == 0:
        return 0.0, 0, 0
    h = int(sub["hit_bool"].sum())
    return round(h / len(sub) * 100, 1), h, len(sub)


# ── Scoring functions (mirrors hr_picks.py) ───────────────────────────────
LEAGUE_AVG_BARREL_7D     = 8.0
LEAGUE_AVG_SEASON_BARREL = 8.0
LEAGUE_AVG_HR_PER_FB     = 10.0
LEAGUE_AVG_HR_PER_PA     = 2.5
LEAGUE_AVG_ISO           = 0.155
MIN_PA_FULL              = 150
MIN_BBE_7D_FULL          = 20
MIN_BBE_7D_PARTIAL       = 5

def regress(value, league_avg, sample, full_sample):
    weight = min(sample / full_sample, 1.0)
    return (value * weight) + (league_avg * (1 - weight))

def score_barrel_7d(v, bbe):
    if bbe < MIN_BBE_7D_PARTIAL: return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe, MIN_BBE_7D_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    return 0.0

def score_barrel_5d(v, bbe):
    if bbe < MIN_BBE_7D_PARTIAL: return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe, MIN_BBE_7D_FULL)
    if v >= 20: return 2.0
    if v >= 15: return 1.5
    if v >= 10: return 1.0
    if v >= 6:  return 0.3
    return 0.0

def score_barrel_10d(v, bbe):
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
        "hr_per_pa", "hr_per_fb", "iso", "pa", "season_bbe",
        "bbe_7d", "bbe_5d", "bbe_10d",
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

    # ── Compute power score and context score for each row ────────────────
    def compute_power_score(r):
        b7  = score_barrel_7d(r["barrel_pct_7d"], r["bbe_7d"])
        b5  = score_barrel_5d(r["barrel_pct_5d"], r["bbe_5d"])
        b10 = score_barrel_10d(r["barrel_pct_10d"], r["bbe_10d"])
        bs  = score_season_barrel(r["season_barrel_pct"], r["pa"])
        fb  = score_hr_per_fb(r["hr_per_fb"], r["pa"])
        pa  = score_hr_per_pa(r["hr_per_pa"], r["pa"])
        iso = score_iso(r["iso"], r["pa"])
        return round(b7 + b5 + b10 + bs + fb + pa + iso, 3)

    print("Computing power/context scores...")
    resolved["power_score"]   = resolved.apply(compute_power_score, axis=1)
    resolved["context_score"] = resolved["hr_score"] - resolved["power_score"]
    resolved["context_ratio"] = resolved["context_score"] / resolved["hr_score"].replace(0, np.nan)

    tiers = [
        ("8.5-9",  8.5,  9.0),
        ("9-10",   9.0, 10.0),
        ("10-11", 10.0, 11.0),
        ("11-12", 11.0, 12.0),
        ("12-13", 12.0, 13.0),
        ("13+",   13.0, 99.0),
    ]

    def get_tier(score):
        for label, lo, hi in tiers:
            if lo <= score < hi:
                return label
        return "Other"

    resolved["tier"] = resolved["hr_score"].apply(get_tier)

    # ── 1. Hit rate by tier ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("1. HIT RATE BY TIER (current score)")
    print("="*70)
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        r, h, n = hr(sub)
        monotonic = ""
        print(f"  {label:8s}: {h:3d}/{n:3d} = {r:5.1f}%  {monotonic}")

    # ── 2. Score component analysis by tier ───────────────────────────────
    print("\n" + "="*70)
    print("2. SCORE COMPOSITION BY TIER — Power vs Context")
    print("="*70)
    print(f"\n  {'Tier':8s} {'Avg Score':>10} {'Avg Power':>10} {'Avg Context':>12} {'Context%':>10} {'Hit Rate':>10}")
    print(f"  {'-'*62}")
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) < MIN_SAMPLE:
            continue
        avg_score   = sub["hr_score"].mean()
        avg_power   = sub["power_score"].mean()
        avg_context = sub["context_score"].mean()
        avg_ctx_pct = (avg_context / avg_score * 100) if avg_score > 0 else 0
        r, h, n     = hr(sub)
        print(f"  {label:8s} {avg_score:>10.2f} {avg_power:>10.2f} {avg_context:>12.2f} {avg_ctx_pct:>9.1f}% {r:>9.1f}%")

    # ── 3. Power score only — is it more monotonic? ───────────────────────
    print("\n" + "="*70)
    print("3. PURE POWER SCORE — Hit rate by power score tier")
    print("   (If this is monotonic, context is the problem)")
    print("="*70)
    power_tiers = [
        ("Power <5",   0,  5),
        ("Power 5-6",  5,  6),
        ("Power 6-7",  6,  7),
        ("Power 7-8",  7,  8),
        ("Power 8-9",  8,  9),
        ("Power 9-10", 9, 10),
        ("Power 10+", 10, 99),
    ]
    print(f"\n  {'Power Tier':12s} {'Hit Rate':>10} {'N':>6}")
    print(f"  {'-'*32}")
    for label, lo, hi in power_tiers:
        sub = resolved[(resolved["power_score"] >= lo) & (resolved["power_score"] < hi)]
        if len(sub) < MIN_SAMPLE:
            continue
        r, h, n = hr(sub)
        print(f"  {label:12s} {r:>9.1f}% {n:>6}")

    # ── 4. Context score distribution within each tier ────────────────────
    print("\n" + "="*70)
    print("4. CONTEXT SCORE BY TIER — Hits vs Misses")
    print("   (High context in a tier = inflated players)")
    print("="*70)
    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) < MIN_SAMPLE:
            continue
        hits   = sub[sub["hit_bool"]]
        misses = sub[~sub["hit_bool"]]
        r, h, n = hr(sub)
        h_ctx = hits["context_score"].mean()
        m_ctx = misses["context_score"].mean()
        h_pow = hits["power_score"].mean()
        m_pow = misses["power_score"].mean()
        print(f"\n  {label} ({r}% hit rate, n={n}):")
        print(f"    Context score — hits: {h_ctx:.2f}  misses: {m_ctx:.2f}  diff: {h_ctx-m_ctx:+.2f}")
        print(f"    Power score   — hits: {h_pow:.2f}  misses: {m_pow:.2f}  diff: {h_pow-m_pow:+.2f}")

    # ── 5. Context cap simulation ─────────────────────────────────────────
    print("\n" + "="*70)
    print("5. CONTEXT CAP SIMULATION")
    print("   What if context score is capped at X points?")
    print("   Does the score become more monotonic?")
    print("="*70)

    for cap in [2.0, 2.5, 3.0, 3.5, 4.0]:
        resolved[f"capped_score_{cap}"] = (
            resolved["power_score"] +
            resolved["context_score"].clip(upper=cap)
        )
        print(f"\n  Context cap = {cap}:")
        prev_rate = 0
        monotonic = True
        for label, lo, hi in tiers:
            capped_lo = lo - (12 - lo) * 0.1  # approximate tier boundaries shift
            sub = resolved[
                (resolved[f"capped_score_{cap}"] >= lo) &
                (resolved[f"capped_score_{cap}"] < hi)
            ]
            if len(sub) < MIN_SAMPLE:
                continue
            r, h, n = hr(sub)
            flag = ""
            if prev_rate > 0 and r < prev_rate - 2:
                flag = " ⚠️ drops"
                monotonic = False
            prev_rate = r
            print(f"    {label:8s}: {h:3d}/{n:3d} = {r:5.1f}%{flag}")
        print(f"    Monotonic: {'✅ YES' if monotonic else '❌ NO'}")

    # ── 6. Power score weight analysis ───────────────────────────────────
    print("\n" + "="*70)
    print("6. WHAT IF WE WEIGHT POWER MORE HEAVILY?")
    print("   score = power * W + context (unchanged)")
    print("="*70)

    for power_weight in [1.0, 1.2, 1.5, 2.0]:
        resolved[f"weighted_score_{power_weight}"] = (
            resolved["power_score"] * power_weight +
            resolved["context_score"]
        )
        # Renormalize to similar range — find what tiers look like
        ws_col = f"weighted_score_{power_weight}"
        print(f"\n  Power weight = {power_weight}x:")
        # Use quantile-based tiers
        q20 = resolved[ws_col].quantile(0.60)
        q40 = resolved[ws_col].quantile(0.70)
        q60 = resolved[ws_col].quantile(0.80)
        q80 = resolved[ws_col].quantile(0.90)

        prev_rate = 0
        monotonic = True
        for t_label, t_lo, t_hi in [
            ("Bottom 40%", 0,    q20),
            ("40-70%",     q20,  q40),
            ("70-80%",     q40,  q60),
            ("80-90%",     q60,  q80),
            ("Top 10%",    q80,  999),
        ]:
            sub = resolved[(resolved[ws_col] >= t_lo) & (resolved[ws_col] < t_hi)]
            if len(sub) < MIN_SAMPLE:
                continue
            r, h, n = hr(sub)
            flag = " ⚠️" if prev_rate > 0 and r < prev_rate - 2 else ""
            if flag:
                monotonic = False
            prev_rate = r
            print(f"    {t_label:12s}: {h:3d}/{n:3d} = {r:5.1f}%{flag}")
        print(f"    Monotonic: {'✅ YES' if monotonic else '❌ NO'}")

    # ── 7. Feature separators across all tiers ────────────────────────────
    print("\n" + "="*70)
    print("7. FEATURE SEPARATORS BY TIER")
    print("   Which features separate hits from misses in each tier?")
    print("="*70)

    feature_cols = [
        ("power_score",        "Power Score"),
        ("context_score",      "Context Score"),
        ("season_barrel_pct",  "Season Barrel%"),
        ("barrel_pct_7d",      "Barrel% 7d"),
        ("hr_per_fb",          "HR/FB%"),
        ("iso",                "ISO"),
        ("platoon_score",      "Platoon Score"),
        ("pitch_matchup_score","Pitch Matchup"),
        ("pitcher_barrel_pct", "Pitcher Barrel%"),
        ("avg_ev_7d",          "Avg EV 7d"),
    ]

    for label, lo, hi in tiers:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) < MIN_SAMPLE:
            continue
        hits   = sub[sub["hit_bool"]]
        misses = sub[~sub["hit_bool"]]
        r, h, n = hr(sub)
        print(f"\n  {label} ({r}% hit rate, {n} picks):")
        seps = []
        for col, flabel in feature_cols:
            if col not in sub.columns:
                continue
            h_avg = hits[col].dropna().mean()
            m_avg = misses[col].dropna().mean()
            if pd.isna(h_avg) or pd.isna(m_avg) or m_avg == 0:
                continue
            pct = (h_avg - m_avg) / abs(m_avg) * 100
            seps.append((pct, flabel, h_avg, m_avg))
        seps.sort(key=lambda x: abs(x[0]), reverse=True)
        for pct, flabel, h_avg, m_avg in seps[:5]:
            flag = "✅" if pct >= 10 else "❌" if pct <= -10 else "  "
            print(f"    {flag} {flabel:<22} hits={h_avg:.3f}  misses={m_avg:.3f}  ({pct:+.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
