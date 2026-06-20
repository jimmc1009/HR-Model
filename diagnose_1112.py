"""
diagnose_1112.py
Analyzes the 11-12 score tier anomaly in HR_All_Scores.
Compares feature averages across 10-11, 11-12, and 12-13 tiers
to identify why 11-12 hits at 7% while adjacent tiers hit 22%+ and 30%+.
"""

import os
import json
import time

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except gspread.exceptions.APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    sh  = with_retry(lambda: gc.open_by_key(sheet_id))
    ws  = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers = all_values[0]
    rows    = all_values[1:]
    df      = pd.DataFrame(rows, columns=headers)
    df      = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    print(f"  {len(df)} total rows loaded")

    # Filter to resolved only
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    print(f"  {len(resolved)} resolved rows")

    if resolved.empty:
        print("No resolved data.")
        return

    # Filter to model start date
    MODEL_START = "2026-06-09"
    resolved["date_dt"] = pd.to_datetime(resolved["date"], errors="coerce")
    resolved = resolved[resolved["date_dt"] >= pd.Timestamp(MODEL_START)].copy()
    print(f"  {len(resolved)} rows after model start date filter ({MODEL_START})")

    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["hr_score"] = resolved["hr_score"].apply(safe_float)
    resolved["odds_num"] = resolved["consensus_odds"].apply(safe_float)

    # Numeric features to compare
    numeric_cols = [
        "barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d", "season_barrel_pct",
        "hr_per_pa", "hr_per_fb", "iso",
        "pitcher_barrel_pct", "pitcher_hr_per_fb",
        "platoon_score", "pitch_matchup_score", "momentum_score",
        "park_hr_factor", "hr_weather_boost", "avg_ev_7d",
        "avg_la_7d", "avg_la_season",
    ]

    for col in numeric_cols:
        if col in resolved.columns:
            resolved[col] = resolved[col].apply(safe_float)
        else:
            resolved[col] = np.nan

    # Define tiers
    tiers = {
        "10-11": resolved[(resolved["hr_score"] >= 10) & (resolved["hr_score"] < 11)],
        "11-12": resolved[(resolved["hr_score"] >= 11) & (resolved["hr_score"] < 12)],
        "12-13": resolved[(resolved["hr_score"] >= 12) & (resolved["hr_score"] < 13)],
        "13+":   resolved[resolved["hr_score"] >= 13],
    }

    print("\n" + "="*70)
    print("HIT RATE BY TIER")
    print("="*70)
    for tier, sub in tiers.items():
        if sub.empty:
            continue
        n    = len(sub)
        hits = int(sub["hit_bool"].sum())
        rate = round(hits / n * 100, 1)
        print(f"  {tier:8s}: {hits:3d}/{n:3d} = {rate}%")

    print("\n" + "="*70)
    print("HIT RATE BY TIER × ODDS ZONE")
    print("="*70)
    odds_zones = [
        ("≤+300",    0,   301),
        ("+301-499", 301, 500),
        ("+500-699", 500, 700),
        ("+700+",    700, 9999),
    ]
    for tier, sub in tiers.items():
        if sub.empty:
            continue
        print(f"\n  {tier}:")
        for oz_label, o_lo, o_hi in odds_zones:
            oz_sub = sub[(sub["odds_num"] >= o_lo) & (sub["odds_num"] < o_hi)]
            if len(oz_sub) < 3:
                continue
            n    = len(oz_sub)
            hits = int(oz_sub["hit_bool"].sum())
            rate = round(hits / n * 100, 1)
            print(f"    {oz_label:10s}: {hits:3d}/{n:3d} = {rate}%")

    print("\n" + "="*70)
    print("FEATURE AVERAGES BY TIER (hits vs misses)")
    print("="*70)

    for tier, sub in tiers.items():
        if sub.empty:
            continue
        hits   = sub[sub["hit_bool"]]
        misses = sub[~sub["hit_bool"]]
        n      = len(sub)
        n_hits = len(hits)
        print(f"\n  {tier} ({n_hits}/{n} = {round(n_hits/n*100,1)}%):")
        print(f"  {'Feature':<25} {'HR Avg':>8} {'No-HR Avg':>10} {'Diff':>8}")
        print(f"  {'-'*55}")
        for col in numeric_cols:
            if col not in sub.columns:
                continue
            h_avg = hits[col].mean()
            m_avg = misses[col].mean()
            if pd.isna(h_avg) or pd.isna(m_avg):
                continue
            diff = h_avg - m_avg
            print(f"  {col:<25} {h_avg:>8.3f} {m_avg:>10.3f} {diff:>+8.3f}")

    print("\n" + "="*70)
    print("SCORE COMPOSITION ANALYSIS — How are 11-12 scores being built?")
    print("="*70)

    # Break down score components for each tier
    component_cols = [
        ("barrel_pct_7d",       "Barrel% 7d"),
        ("barrel_pct_10d",      "Barrel% 10d"),
        ("season_barrel_pct",   "Barrel% Season"),
        ("hr_per_fb",           "HR/FB%"),
        ("hr_per_pa",           "HR/PA%"),
        ("iso",                 "ISO"),
        ("pitcher_barrel_pct",  "Pitcher Barrel%"),
        ("platoon_score",       "Platoon Score"),
        ("pitch_matchup_score", "Pitch Matchup"),
        ("momentum_score",      "Momentum"),
        ("park_hr_factor",      "Park Factor"),
        ("hr_weather_boost",    "Weather Boost"),
    ]

    print(f"\n  {'Feature':<25} {'10-11':>8} {'11-12':>8} {'12-13':>8} {'13+':>8}")
    print(f"  {'-'*60}")
    for col, label in component_cols:
        row_vals = []
        for tier, sub in tiers.items():
            if col in sub.columns:
                val = sub[col].mean()
                row_vals.append(f"{val:>8.3f}" if not pd.isna(val) else f"{'N/A':>8}")
            else:
                row_vals.append(f"{'N/A':>8}")
        print(f"  {label:<25} {' '.join(row_vals)}")

    print("\n" + "="*70)
    print("11-12 TIER — WHAT DO HITTERS LOOK LIKE?")
    print("="*70)

    tier_1112 = tiers["11-12"]
    if not tier_1112.empty:
        # What % of 11-12 players have meaningful barrel% vs low barrel%
        if "barrel_pct_7d" in tier_1112.columns:
            high_barrel = (tier_1112["barrel_pct_7d"] >= 15).sum()
            mid_barrel  = ((tier_1112["barrel_pct_7d"] >= 8) & (tier_1112["barrel_pct_7d"] < 15)).sum()
            low_barrel  = (tier_1112["barrel_pct_7d"] < 8).sum()
            total       = len(tier_1112)
            print(f"\n  Barrel% 7d distribution (n={total}):")
            print(f"    15%+  (elite):  {high_barrel} ({round(high_barrel/total*100,1)}%)")
            print(f"    8-15% (solid):  {mid_barrel}  ({round(mid_barrel/total*100,1)}%)")
            print(f"    <8%   (weak):   {low_barrel}  ({round(low_barrel/total*100,1)}%)")

            # Hit rate by barrel%
            print(f"\n  Hit rate by Barrel% 7d in 11-12 tier:")
            for label, lo, hi in [("15%+", 15, 999), ("8-15%", 8, 15), ("<8%", 0, 8)]:
                sub = tier_1112[(tier_1112["barrel_pct_7d"] >= lo) & (tier_1112["barrel_pct_7d"] < hi)]
                if len(sub) < 3:
                    continue
                h = int(sub["hit_bool"].sum())
                print(f"    {label:6s}: {h}/{len(sub)} = {round(h/len(sub)*100,1)}%")

        # Platoon score distribution
        if "platoon_score" in tier_1112.columns:
            print(f"\n  Platoon score distribution in 11-12 tier:")
            high_plat = (tier_1112["platoon_score"] >= 1.0).sum()
            mid_plat  = ((tier_1112["platoon_score"] >= 0) & (tier_1112["platoon_score"] < 1.0)).sum()
            neg_plat  = (tier_1112["platoon_score"] < 0).sum()
            total     = len(tier_1112)
            print(f"    1.0+  (strong advantage): {high_plat} ({round(high_plat/total*100,1)}%)")
            print(f"    0-1.0 (neutral):           {mid_plat}  ({round(mid_plat/total*100,1)}%)")
            print(f"    <0    (disadvantage):      {neg_plat}  ({round(neg_plat/total*100,1)}%)")

            print(f"\n  Hit rate by platoon score in 11-12 tier:")
            for label, lo, hi in [("1.0+", 1.0, 999), ("0-1.0", 0, 1.0), ("<0", -999, 0)]:
                sub = tier_1112[(tier_1112["platoon_score"] >= lo) & (tier_1112["platoon_score"] < hi)]
                if len(sub) < 3:
                    continue
                h = int(sub["hit_bool"].sum())
                print(f"    {label:6s}: {h}/{len(sub)} = {round(h/len(sub)*100,1)}%")

        # Pitch matchup score distribution
        if "pitch_matchup_score" in tier_1112.columns:
            print(f"\n  Hit rate by pitch matchup score in 11-12 tier:")
            for label, lo, hi in [("1.0+", 1.0, 999), ("0.5-1.0", 0.5, 1.0), ("<0.5", 0, 0.5)]:
                sub = tier_1112[(tier_1112["pitch_matchup_score"] >= lo) & (tier_1112["pitch_matchup_score"] < hi)]
                if len(sub) < 3:
                    continue
                h = int(sub["hit_bool"].sum())
                print(f"    {label:8s}: {h}/{len(sub)} = {round(h/len(sub)*100,1)}%")

    print("\n" + "="*70)
    print("HYPOTHESIS TEST — Does barrel% floor fix 11-12?")
    print("="*70)

    tier_1112 = tiers["11-12"]
    if not tier_1112.empty and "season_barrel_pct" in tier_1112.columns:
        print(f"\n  11-12 tier filtered by minimum season barrel%:")
        for floor in [8, 9, 10, 11, 12]:
            sub = tier_1112[tier_1112["season_barrel_pct"] >= floor]
            if len(sub) < 3:
                continue
            h = int(sub["hit_bool"].sum())
            print(f"    Season barrel% >= {floor}%: {h}/{len(sub)} = {round(h/len(sub)*100,1)}%")

        print(f"\n  11-12 tier filtered by minimum barrel% 7d:")
        for floor in [8, 10, 12, 15]:
            sub = tier_1112[tier_1112["barrel_pct_7d"] >= floor]
            if len(sub) < 3:
                continue
            h = int(sub["hit_bool"].sum())
            print(f"    Barrel% 7d >= {floor}%: {h}/{len(sub)} = {round(h/len(sub)*100,1)}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
