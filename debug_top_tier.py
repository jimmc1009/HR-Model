"""
debug_top_tier.py
Investigates why 13+ tier HR hitters have LOWER scores than misses.
Looks at what's actually driving scores into the 13+ tier and whether
those components predict HRs.
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

def get_gspread_client():
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

    num_cols = [
        "hr_score", "consensus_odds",
        "platoon_score", "pitch_matchup_score", "momentum_score",
        "hr_weather_boost", "park_hr_factor",
        "barrel_pct_7d", "season_barrel_pct", "hr_per_fb",
        "iso", "avg_ev_7d", "avg_la_7d",
    ]
    for col in num_cols:
        if col in resolved.columns:
            resolved[col] = resolved[col].apply(safe_float)
        else:
            resolved[col] = 0.0

    top = resolved[resolved["hr_score"] >= 13].copy()
    hits   = top[top["hit_bool"]]
    misses = top[~top["hit_bool"]]

    print(f"\n13+ tier: {len(top)} picks, {len(hits)} hits ({round(len(hits)/len(top)*100,1)}%)")

    # ── 1. Score distribution ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("1. SCORE DISTRIBUTION — What gets players to 13+?")
    print("="*60)
    print(f"\n  Score range: {top['hr_score'].min():.1f} to {top['hr_score'].max():.1f}")
    print(f"  Avg score — hits: {hits['hr_score'].mean():.2f}  misses: {misses['hr_score'].mean():.2f}")

    # Score sub-buckets
    for lo, hi in [(13, 14), (14, 15), (15, 16), (16, 17), (17, 99)]:
        sub = top[(top["hr_score"] >= lo) & (top["hr_score"] < hi)]
        if len(sub) < 3: continue
        h = int(sub["hit_bool"].sum())
        print(f"  {lo}-{hi}: {h}/{len(sub)} = {round(h/len(sub)*100,1)}%  avg_score={sub['hr_score'].mean():.2f}")

    # ── 2. What components are highest in the top tier ────────────────────
    print("\n" + "="*60)
    print("2. COMPONENT BREAKDOWN — What's driving 13+ scores?")
    print("   Are players hitting because of score or despite it?")
    print("="*60)

    components = [
        ("platoon_score",      "Platoon"),
        ("pitch_matchup_score","Pitch Matchup"),
        ("momentum_score",     "Momentum"),
        ("hr_weather_boost",   "Weather Boost"),
        ("barrel_pct_7d",      "Barrel% 7d"),
        ("season_barrel_pct",  "Season Barrel%"),
        ("hr_per_fb",          "HR/FB%"),
        ("iso",                "ISO"),
        ("avg_ev_7d",          "Avg EV 7d"),
        ("park_hr_factor",     "Park Factor"),
    ]

    print(f"\n  {'Component':<22} {'Hits Avg':>10} {'Miss Avg':>10} {'Diff':>8} {'Signal':>10}")
    print(f"  {'-'*62}")
    for col, label in components:
        if col not in top.columns: continue
        h_avg = hits[col].mean()
        m_avg = misses[col].mean()
        diff  = h_avg - m_avg
        pct   = round(diff / abs(m_avg) * 100, 1) if m_avg != 0 else 0
        flag  = "✅" if pct >= 10 else "❌" if pct <= -10 else "  "
        print(f"  {flag} {label:<22} {h_avg:>10.3f} {m_avg:>10.3f} {diff:>+8.3f} {pct:>+9.1f}%")

    # ── 3. Platoon deep dive ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("3. PLATOON DEEP DIVE — Why are misses getting higher platoon?")
    print("="*60)

    platoon_buckets = [
        ("Negative (<0)",  -99, 0),
        ("Low (0-0.5)",    0, 0.5),
        ("Mid (0.5-1.0)",  0.5, 1.0),
        ("High (1.0-1.5)", 1.0, 1.5),
        ("Elite (1.5+)",   1.5, 99),
    ]
    print(f"\n  {'Platoon Bucket':<20} {'Hit Rate':>10} {'N':>6}")
    print(f"  {'-'*38}")
    for label, lo, hi in platoon_buckets:
        sub = top[(top["platoon_score"] >= lo) & (top["platoon_score"] < hi)]
        if len(sub) < 3: continue
        h = int(sub["hit_bool"].sum())
        print(f"  {label:<20} {round(h/len(sub)*100,1):>9.1f}% {len(sub):>6}")

    # ── 4. Pitch matchup deep dive ────────────────────────────────────────
    print("\n" + "="*60)
    print("4. PITCH MATCHUP DEEP DIVE — Does it help or hurt at 13+?")
    print("="*60)
    pm_buckets = [
        ("Low (<2.0)",   0, 2.0),
        ("Mid (2-3)",    2.0, 3.0),
        ("High (3-4)",   3.0, 4.0),
        ("Elite (4+)",   4.0, 99),
    ]
    print(f"\n  {'PM Bucket':<20} {'Hit Rate':>10} {'N':>6}")
    print(f"  {'-'*38}")
    for label, lo, hi in pm_buckets:
        sub = top[(top["pitch_matchup_score"] >= lo) & (top["pitch_matchup_score"] < hi)]
        if len(sub) < 3: continue
        h = int(sub["hit_bool"].sum())
        print(f"  {label:<20} {round(h/len(sub)*100,1):>9.1f}% {len(sub):>6}")

    # ── 5. Who are the repeat top-tier players ────────────────────────────
    print("\n" + "="*60)
    print("5. MOST FREQUENT 13+ PLAYERS — Are certain players inflating the tier?")
    print("="*60)
    player_counts = top.groupby("player_name").agg(
        appearances=("hit_bool", "count"),
        hits=("hit_bool", "sum"),
        avg_score=("hr_score", "mean"),
        avg_platoon=("platoon_score", "mean"),
        avg_pm=("pitch_matchup_score", "mean"),
    ).sort_values("appearances", ascending=False).head(20)

    print(f"\n  {'Player':<25} {'Apps':>5} {'Hits':>5} {'HR%':>6} {'AvgScore':>9} {'AvgPlatoon':>11} {'AvgPM':>7}")
    print(f"  {'-'*72}")
    for name, row in player_counts.iterrows():
        hr_pct = round(row["hits"] / row["appearances"] * 100, 1)
        print(f"  {name:<25} {int(row['appearances']):>5} {int(row['hits']):>5} {hr_pct:>5.1f}% {row['avg_score']:>9.2f} {row['avg_platoon']:>11.3f} {row['avg_pm']:>7.3f}")

    # ── 6. Odds distribution ──────────────────────────────────────────────
    print("\n" + "="*60)
    print("6. ODDS DISTRIBUTION — Are top tier players being over-priced?")
    print("="*60)
    odds_buckets = [
        ("≤+200",    0,   200),
        ("+201-300", 201, 300),
        ("+301-400", 301, 400),
        ("+401-500", 401, 500),
        ("+500+",    501, 999),
    ]
    print(f"\n  {'Odds Zone':<15} {'Hit Rate':>10} {'N':>6}")
    print(f"  {'-'*33}")
    top["odds_num"] = top["consensus_odds"].apply(safe_float)
    for label, lo, hi in odds_buckets:
        sub = top[(top["odds_num"] > lo) & (top["odds_num"] <= hi)]
        if len(sub) < 3: continue
        h = int(sub["hit_bool"].sum())
        print(f"  {label:<15} {round(h/len(sub)*100,1):>9.1f}% {len(sub):>6}")

    print("\nDone.")

if __name__ == "__main__":
    main()
