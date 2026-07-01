"""
diagnose_ks_score.py
Validates the KS (pitcher strikeout) score the same way diagnose_score.py
validates the HR score.

Sections:
1. Hit rate by score tier (both OVER and UNDER framing)
2. Score component breakdown — pitcher skill vs opponent/context
3. Pure pitcher-skill score monotonicity (excludes opponent factors)
4. Component hits vs misses by tier (OVER direction)
5. Feature separators by tier
6. Score sub-bucket granularity at the top (12+) — same "counter-intuitive
   top tier" check we did for HR
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

MIN_SAMPLE = 10
MIN_IP     = 10


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


# ── Exact scoring functions from ks_picks.py ─────────────────────────────
def score_k_pct(v, ip):
    if ip < MIN_IP: return 0.0
    if v >= 32.0: return 1.5
    if v >= 28.0: return 1.2
    if v >= 25.0: return 0.9
    if v >= 22.0: return 0.6
    if v >= 19.0: return 0.3
    if v >= 16.0: return 0.1
    return 0.0

def score_swstr(v, ip):
    if ip < MIN_IP: return 0.0
    if v >= 16.0: return 1.0
    if v >= 14.0: return 0.7
    if v >= 12.0: return 0.5
    if v >= 10.0: return 0.3
    if v >= 8.0:  return 0.1
    return 0.0

def score_chase(v, ip):
    if ip < MIN_IP: return 0.0
    if v >= 38.0: return 1.5
    if v >= 34.0: return 1.0
    if v >= 30.0: return 0.5
    return 0.0

def score_k_per_9(v, ip):
    if ip < MIN_IP: return 0.0
    if v >= 12.0: return 1.5
    if v >= 10.0: return 1.0
    if v >= 8.5:  return 0.5
    if v >= 7.0:  return 0.2
    return 0.0

def score_velo(v, ip):
    if ip < MIN_IP: return 0.0
    if v >= 97.0: return 1.0
    if v >= 95.0: return 0.6
    if v >= 93.0: return 0.3
    return 0.0

def score_k_per_start_21d(v):
    if v >= 9.0: return 1.5
    if v >= 7.0: return 1.0
    if v >= 5.5: return 0.5
    if v <= 3.0: return -1.0
    if v <= 4.0: return -0.5
    return 0.0

def score_avg_ip(v):
    if v >= 6.5: return  1.3
    if v >= 6.0: return  0.9
    if v >= 5.5: return  0.5
    if v >= 5.0: return  0.1
    if v <= 4.0: return -1.3
    if v <= 4.5: return -0.6
    return 0.0

def score_opp_team_k_pct(v):
    if v >= 28.0: return  1.5
    if v >= 25.0: return  1.0
    if v >= 22.0: return  0.5
    if v >= 20.0: return  0.2
    if v <= 16.0: return -1.0
    if v <= 18.0: return -0.5
    return 0.0

def score_pitcher_whip(v):
    if v <= 0: return 0.0
    if v <= 0.90: return  1.2
    if v <= 1.00: return  0.8
    if v <= 1.10: return  0.3
    if v >= 1.50: return -1.2
    if v >= 1.35: return -0.6
    if v >= 1.20: return -0.3
    return 0.0

def score_pitcher_bb_pct(v):
    if v <= 0: return 0.0
    if v <= 4.0:  return  1.5
    if v <= 6.0:  return  0.8
    if v <= 8.0:  return  0.2
    if v >= 12.0: return -1.5
    if v >= 10.0: return -0.8
    if v >= 9.0:  return -0.3
    return 0.0

def score_park(v):
    if v >= 115: return -0.3
    if v <= 80:  return  0.2
    return 0.0


def hr(sub, col="hit_bool"):
    if len(sub) == 0: return 0.0, 0, 0
    h = int(sub[col].sum())
    return round(h / len(sub) * 100, 1), h, len(sub)


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading KS_All_Scores...")
    sh         = with_retry(lambda: gc.open_by_key(sheet_id))
    ws         = sh.worksheet("KS_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers    = all_values[0]
    rows       = all_values[1:]
    df         = pd.DataFrame(rows, columns=headers)
    df         = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]

    resolved = df[df["over_hit"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    print(f"  {len(resolved)} resolved rows")
    if resolved.empty:
        print("No resolved data — exiting.")
        return

    resolved["hit_bool"]   = resolved["over_hit"].astype(str).str.strip() == "Yes"
    resolved["under_bool"] = resolved["under_hit"].astype(str).str.strip() == "Yes"
    resolved["ks_score"]   = resolved["ks_score"].apply(safe_float)
    resolved["k_line"]     = resolved["k_line"].apply(safe_float)
    resolved["date_dt"]    = pd.to_datetime(resolved["date"], errors="coerce")

    num_cols = [
        "k_pct_season", "swstr_pct", "chase_rate", "k_per_9",
        "fastball_velo", "avg_ip_per_start", "k_per_start_21d",
        "whip_proxy", "bb_pct_season",
        "opp_team_k_pct", "opp_chase_rate", "opp_whiff_rate",
        "over_odds", "under_odds",
    ]
    for col in num_cols:
        if col in resolved.columns:
            resolved[col] = resolved[col].apply(safe_float)
        else:
            resolved[col] = 0.0

    # Approximate IP as gating factor — not directly in KS_All_Scores,
    # so we treat all resolved rows as already passing the IP/GS filter
    # (they wouldn't be scored otherwise per compute_ks_score gating)

    # ── Recompute components ────────────────────────────────────────────
    def compute_skill(r):
        return round(
            score_k_pct(r["k_pct_season"], 999) +
            score_swstr(r["swstr_pct"], 999) +
            score_chase(r["chase_rate"], 999) +
            score_k_per_9(r["k_per_9"], 999) +
            score_velo(r["fastball_velo"], 999) +
            score_k_per_start_21d(r["k_per_start_21d"]) +
            score_avg_ip(r["avg_ip_per_start"]) +
            score_pitcher_whip(r["whip_proxy"]) +
            score_pitcher_bb_pct(r["bb_pct_season"]),
            3
        )

    def compute_opponent(r):
        return round(
            score_opp_team_k_pct(r["opp_team_k_pct"]) +
            score_park(100.0),  # park factor not logged in KS_All_Scores — neutral
            3
        )

    resolved["skill_score"]    = resolved.apply(compute_skill, axis=1)
    resolved["opponent_score"] = resolved.apply(compute_opponent, axis=1)
    resolved["recomputed"]     = resolved["skill_score"] + resolved["opponent_score"]

    score_diff = (resolved["recomputed"] - resolved["ks_score"]).abs().mean()
    print(f"  Avg diff between recomputed and logged score: {score_diff:.3f}")
    print(f"  Skill score range: {resolved['skill_score'].min():.2f} to {resolved['skill_score'].max():.2f}")
    print(f"  Opponent score range: {resolved['opponent_score'].min():.2f} to {resolved['opponent_score'].max():.2f}")

    tiers = [
        ("12+",    12,  999),
        ("10-12",  10,   12),
        ("8-10",    8,   10),
        ("6-8",     6,    8),
        ("4-6",     4,    6),
        ("2-4",     2,    4),
        ("Under 2", 0,    2),
        ("Under 0", -999, 0),
    ]

    # ── 1. Hit rate by tier (OVER direction) ───────────────────────────
    print("\n" + "="*70)
    print("1. OVER HIT RATE BY SCORE TIER")
    print("="*70)
    for label, lo, hi in tiers:
        sub = resolved[(resolved["ks_score"] >= lo) & (resolved["ks_score"] < hi)]
        r, h, n = hr(sub)
        print(f"  {label:8s}: {h:3d}/{n:4d} = {r:5.1f}%")

    # ── 2. Score composition ─────────────────────────────────────────────
    print("\n" + "="*70)
    print("2. SCORE COMPOSITION BY TIER — Skill vs Opponent")
    print("="*70)
    print(f"\n  {'Tier':8s} {'AvgScore':>9} {'AvgSkill':>9} {'AvgOpp':>8} {'Opp%':>7} {'OverHR':>8}")
    print(f"  {'-'*55}")
    for label, lo, hi in tiers:
        sub = resolved[(resolved["ks_score"] >= lo) & (resolved["ks_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        avg_s  = sub["ks_score"].mean()
        avg_sk = sub["skill_score"].mean()
        avg_op = sub["opponent_score"].mean()
        op_pct = avg_op / avg_s * 100 if avg_s != 0 else 0
        r, h, n = hr(sub)
        print(f"  {label:8s} {avg_s:>9.2f} {avg_sk:>9.2f} {avg_op:>8.2f} {op_pct:>6.1f}% {r:>7.1f}%")

    # ── 3. Pure skill score monotonicity ──────────────────────────────────
    print("\n" + "="*70)
    print("3. PURE PITCHER SKILL SCORE — Hit rate by skill tier")
    print("   (Excludes opponent K%/park — does skill alone predict OVER?)")
    print("="*70)
    skill_tiers = [
        ("Skill <2",   -99, 2),
        ("Skill 2-3",  2, 3),
        ("Skill 3-4",  3, 4),
        ("Skill 4-5",  4, 5),
        ("Skill 5-6",  5, 6),
        ("Skill 6-7",  6, 7),
        ("Skill 7+",   7, 99),
    ]
    print(f"\n  {'Skill Tier':12s} {'Hit Rate':>10} {'N':>6}")
    print(f"  {'-'*32}")
    prev = 0
    for label, lo, hi in skill_tiers:
        sub = resolved[(resolved["skill_score"] >= lo) & (resolved["skill_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        r, h, n = hr(sub)
        flag = " ⚠️" if prev > 0 and r < prev - 3 else ""
        prev = r
        print(f"  {label:12s} {r:>9.1f}% {n:>6}{flag}")

    # ── 4. Component hits vs misses by tier ────────────────────────────────
    print("\n" + "="*70)
    print("4. COMPONENT HITS vs MISSES BY TIER")
    print("="*70)
    comp_cols = [
        ("k_pct_season",     "K%"),
        ("swstr_pct",        "SwStr%"),
        ("chase_rate",       "Chase%"),
        ("k_per_9",          "K/9"),
        ("fastball_velo",    "Velo"),
        ("k_per_start_21d",  "K/21d"),
        ("avg_ip_per_start", "Avg IP"),
        ("whip_proxy",       "WHIP"),
        ("bb_pct_season",    "BB%"),
        ("opp_team_k_pct",   "Opp K%"),
        ("opp_chase_rate",   "Opp Chase%"),
        ("opp_whiff_rate",   "Opp Whiff%"),
    ]
    for label, lo, hi in tiers:
        sub = resolved[(resolved["ks_score"] >= lo) & (resolved["ks_score"] < hi)]
        if len(sub) < MIN_SAMPLE: continue
        hits   = sub[sub["hit_bool"]]
        misses = sub[~sub["hit_bool"]]
        r, h, n = hr(sub)
        print(f"\n  {label} ({r:.1f}% OVER, n={n}):")
        seps = []
        for col, clabel in comp_cols:
            h_avg = hits[col].mean()
            m_avg = misses[col].mean()
            if m_avg == 0: continue
            pct = (h_avg - m_avg) / abs(m_avg) * 100
            seps.append((pct, clabel, h_avg, m_avg))
        seps.sort(key=lambda x: abs(x[0]), reverse=True)
        for pct, clabel, h_avg, m_avg in seps[:6]:
            flag = "✅" if pct >= 10 else "❌" if pct <= -10 else "  "
            print(f"    {flag} {clabel:<12} hits={h_avg:.2f}  misses={m_avg:.2f}  ({pct:+.1f}%)")

    # ── 5. Top tier granularity check ───────────────────────────────────
    print("\n" + "="*70)
    print("5. TOP TIER (12+) GRANULARITY — Same check as HR model")
    print("="*70)
    top = resolved[resolved["ks_score"] >= 12]
    print(f"\n  12+ tier: {len(top)} picks")
    for lo, hi in [(12, 13), (13, 14), (14, 15), (15, 16), (16, 99)]:
        sub = top[(top["ks_score"] >= lo) & (top["ks_score"] < hi)]
        if len(sub) < 3: continue
        h_, h, n = hr(sub)
        print(f"  {lo}-{hi}: {h}/{n} = {h_}%  avg_score={sub['ks_score'].mean():.2f}")

    # ── 6. Odds zone check — is the market pricing 12+ correctly? ──────────
    print("\n" + "="*70)
    print("6. 12+ TIER BY OVER ODDS ZONE")
    print("="*70)
    buckets = [
        ("Minus money", -999, -1),
        ("+100-130",     100, 130),
        ("+131-160",     131, 160),
        ("+161+",         161, 999),
    ]
    for label, lo, hi in buckets:
        if lo < 0:
            sub = top[(top["over_odds"] < 0)]
        else:
            sub = top[(top["over_odds"] >= lo) & (top["over_odds"] <= hi)]
        if len(sub) < 3: continue
        r, h, n = hr(sub)
        print(f"  {label:<14}: {h}/{n} = {r}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
