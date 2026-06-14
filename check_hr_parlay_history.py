"""
check_hr_parlay_history.py
Diagnostic: for each day in HR_All_Scores, look at ALL players scoring
10+ and check how many actually hit a HR. Reports days where 3+ hit
(a 3-leg parlay would have been possible), and analyzes what those
winning players had in common (tier, odds range, platoon, features).
"""

import os
import time

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import json

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MIN_SCORE = 10.0
LEG_COUNT = 3


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


def get_score_tier(score: float) -> str:
    if score >= 13:   return "13+"
    if score >= 12:   return "12-13"
    if score >= 11:   return "11-12"
    if score >= 10:   return "10-11"
    return "Under 10"


def get_odds_zone(odds: float) -> str:
    if odds <= 0:    return "No Odds"
    if odds <= 300:  return "<= +300"
    if odds <= 499:  return "+301 to +499"
    if odds <= 699:  return "+500 to +699"
    return "+700+"


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    ws = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())

    headers = all_values[0]
    rows    = all_values[1:]
    df      = pd.DataFrame(rows, columns=headers)

    df["hr_score"] = df["hr_score"].apply(safe_float)
    df["date"]     = df["date"].astype(str).str.strip()
    df["odds_num"] = df["consensus_odds"].apply(lambda x: safe_float(x, 0))

    for col in ["platoon_score", "barrel_pct_7d", "barrel_pct_5d", "season_barrel_pct",
                "hr_per_fb", "hr_per_pa", "pitch_matchup_score", "momentum_score"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)

    # Only resolved rows
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"

    candidates = resolved[resolved["hr_score"] >= MIN_SCORE].copy()
    dates = sorted(candidates["date"].unique())

    print(f"Analyzing {len(dates)} days with score >= {MIN_SCORE} candidates\n")

    days_with_3_hits = 0
    total_days       = 0
    total_candidates = 0
    total_hits       = 0

    winning_players = []  # all players who hit on a 3+-hit day

    for d in dates:
        day_df = candidates[candidates["date"] == d].copy()
        if day_df.empty:
            continue

        total_days       += 1
        total_candidates += len(day_df)

        hits = day_df[day_df["hit_bool"]]
        total_hits += len(hits)

        if len(hits) >= LEG_COUNT:
            days_with_3_hits += 1
            print(f"{d}: {len(hits)} of {len(day_df)} candidates (score>=10) hit")
            for _, r in hits.iterrows():
                winning_players.append(r)
                tier = get_score_tier(r["hr_score"])
                zone = get_odds_zone(r["odds_num"])
                print(f"    - {r['player_name']:25s} score={r['hr_score']:.2f} tier={tier:8s} odds={r['consensus_odds']:>6s} zone={zone}")

    print()
    print("=" * 60)
    print(f"Days with {LEG_COUNT}+ hits among score>=10 candidates: {days_with_3_hits} / {total_days}")
    if total_days:
        print(f"  Rate: {days_with_3_hits/total_days*100:.1f}%")
    print(f"\nOverall: {total_hits} hits / {total_candidates} candidates scoring >= {MIN_SCORE} ({total_hits/total_candidates*100:.1f}% hit rate)" if total_candidates else "")
    print("=" * 60)

    if not winning_players:
        print("\nNo days with 3+ hits among 10+ scores yet — not enough data.")
        return

    wdf = pd.DataFrame(winning_players)

    print("\n--- What winning players (on 3+-hit days) had in common ---\n")

    # Score tier distribution
    wdf["tier"] = wdf["hr_score"].apply(get_score_tier)
    print("Score tier distribution:")
    for tier, n in wdf["tier"].value_counts().items():
        print(f"  {tier:10s}: {n}")

    # Odds zone distribution
    wdf["odds_zone"] = wdf["odds_num"].apply(get_odds_zone)
    print("\nOdds zone distribution:")
    for zone, n in wdf["odds_zone"].value_counts().items():
        print(f"  {zone:14s}: {n}")

    # Average feature values among winners vs all candidates
    print("\nFeature averages — winners (3+-hit days) vs all 10+ candidates:")
    for col in ["platoon_score", "barrel_pct_7d", "barrel_pct_5d", "season_barrel_pct",
                "hr_per_fb", "hr_per_pa", "pitch_matchup_score", "momentum_score"]:
        if col in wdf.columns and col in candidates.columns:
            w_avg = wdf[col].mean()
            c_avg = candidates[col].mean()
            diff  = w_avg - c_avg
            print(f"  {col:22s}: winners={w_avg:7.3f}  all_candidates={c_avg:7.3f}  diff={diff:+.3f}")

    # Platoon matchup text
    if "platoon_matchup" in wdf.columns:
        print("\nPlatoon matchup breakdown (winners):")
        for val, n in wdf["platoon_matchup"].value_counts().head(5).items():
            print(f"  {str(val)[:50]:50s}: {n}")


if __name__ == "__main__":
    main()
