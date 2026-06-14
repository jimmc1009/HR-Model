"""
check_hr_parlay_history.py
Diagnostic: for each day in HR_All_Scores, look at the top-N scored
players and check how many actually hit a HR. Reports the realistic
historical hit rate for "top 3 by score" parlays, plus breaks down
which score tiers / odds ranges tend to produce winning combos.
"""

import os
import time
import unicodedata
from itertools import combinations

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

TOP_N      = 10   # consider top N scored players each day as parlay candidates
LEG_COUNT  = 3    # parlay size


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
    if score >= 9:    return "9-10"
    if score >= 8.5:  return "8.5-9"
    return "Under 8.5"


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

    # Only resolved rows
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"

    dates = sorted(resolved["date"].unique())
    print(f"Analyzing {len(dates)} resolved days, top {TOP_N} players/day, {LEG_COUNT}-leg parlays\n")

    days_with_3_hits = 0
    total_days       = 0

    # Track which tier/odds combos appear in winning parlay legs
    winning_leg_tiers = {}
    all_leg_tiers     = {}

    for d in dates:
        day_df = resolved[resolved["date"] == d].copy()
        if day_df.empty:
            continue

        top = day_df.sort_values("hr_score", ascending=False).head(TOP_N)
        if len(top) < LEG_COUNT:
            continue

        total_days += 1
        hits = top[top["hit_bool"]]

        # Tally tiers for all candidates
        for _, r in top.iterrows():
            tier = get_score_tier(r["hr_score"])
            all_leg_tiers[tier] = all_leg_tiers.get(tier, 0) + 1

        if len(hits) >= LEG_COUNT:
            days_with_3_hits += 1
            print(f"{d}: {len(hits)} of top {TOP_N} hit (3+ leg parlay possible)")
            for _, r in hits.iterrows():
                tier = get_score_tier(r["hr_score"])
                winning_leg_tiers[tier] = winning_leg_tiers.get(tier, 0) + 1
                print(f"    - {r['player_name']:25s} score={r['hr_score']:.2f} odds={r['consensus_odds']}")

    print()
    print("=" * 60)
    print(f"Days with {LEG_COUNT}+ hits in top {TOP_N}: {days_with_3_hits} / {total_days}")
    if total_days:
        print(f"  Rate: {days_with_3_hits/total_days*100:.1f}%")
    print("=" * 60)

    print("\nTier distribution — ALL top-N candidates:")
    for tier, n in sorted(all_leg_tiers.items(), key=lambda x: -x[1]):
        print(f"  {tier:10s}: {n}")

    print("\nTier distribution — WINNING legs (on 3+-hit days):")
    for tier, n in sorted(winning_leg_tiers.items(), key=lambda x: -x[1]):
        win_rate_in_tier = n / all_leg_tiers.get(tier, 1) * 100
        print(f"  {tier:10s}: {n}  (hit rate within tier on these days: {win_rate_in_tier:.1f}%)")

    # ── "Top 3 by score" parlay simulation ──────────────────────────────
    print("\n" + "=" * 60)
    print(f"SIMULATION: pick exactly top {LEG_COUNT} by score each day — would it cash?")
    print("=" * 60)
    cashed = 0
    sim_days = 0
    for d in dates:
        day_df = resolved[resolved["date"] == d].copy()
        if len(day_df) < LEG_COUNT:
            continue
        top3 = day_df.sort_values("hr_score", ascending=False).head(LEG_COUNT)
        sim_days += 1
        if top3["hit_bool"].all():
            cashed += 1
            names = ", ".join(top3["player_name"].tolist())
            print(f"  {d}: CASHED — {names}")

    print(f"\nTop-3-by-score parlay would have cashed {cashed} / {sim_days} days ({cashed/sim_days*100:.1f}%)" if sim_days else "No data")


if __name__ == "__main__":
    main()
