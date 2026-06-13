"""
check_hr_overlap.py
Diagnostic: checks whether players who hit a HR yesterday tend to
dominate today's top-scored HR picks (i.e. is the model just chasing
recent results via momentum/short-window barrel%).

Reads HR_All_Scores, compares each day's top N scores against the
PREVIOUS day's hit_hr == "Yes" players.

Run manually via GitHub Actions (one-off) or locally.
"""

import os
import time
import unicodedata

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

TOP_N = 20  # how many top-scored players per day to check


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


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


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
    df["name_norm"] = df["player_name"].apply(normalize_name)
    df["date"] = df["date"].astype(str).str.strip()

    dates = sorted(df["date"].unique())
    print(f"Found {len(dates)} distinct dates: {dates}\n")

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_df = df[df["date"] == prev_date]
        curr_df = df[df["date"] == curr_date]

        # Yesterday's actual HR hitters
        prev_hr_hitters = set(
            prev_df[prev_df["hit_hr"].astype(str).str.strip() == "Yes"]["name_norm"]
        )

        if not prev_hr_hitters:
            print(f"{curr_date}: no resolved HR hitters from {prev_date} yet — skipping")
            continue

        # Today's top-N scored players
        curr_top = curr_df.sort_values("hr_score", ascending=False).head(TOP_N)
        curr_top_names = set(curr_top["name_norm"])

        overlap = prev_hr_hitters & curr_top_names

        print(f"--- {curr_date} (prev day {prev_date}) ---")
        print(f"  {prev_date} HR hitters (resolved):     {len(prev_hr_hitters)}")
        print(f"  {curr_date} top {TOP_N} scored players: {len(curr_top_names)}")
        print(f"  Overlap:                              {len(overlap)}")
        if overlap:
            overlap_rows = curr_top[curr_top["name_norm"].isin(overlap)]
            for _, r in overlap_rows.iterrows():
                print(f"    - {r['player_name']:25s} score={r['hr_score']:.2f}")
        print()

    # Overall summary across all day-pairs
    total_overlap = 0
    total_possible = 0
    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        prev_df = df[df["date"] == prev_date]
        curr_df = df[df["date"] == curr_date]
        prev_hr_hitters = set(
            prev_df[prev_df["hit_hr"].astype(str).str.strip() == "Yes"]["name_norm"]
        )
        if not prev_hr_hitters:
            continue
        curr_top = curr_df.sort_values("hr_score", ascending=False).head(TOP_N)
        curr_top_names = set(curr_top["name_norm"])
        overlap = prev_hr_hitters & curr_top_names
        total_overlap += len(overlap)
        total_possible += min(len(prev_hr_hitters), TOP_N)

    print("=" * 50)
    print(f"SUMMARY: avg overlap = {total_overlap} players across all day-pairs")
    if total_possible:
        pct = total_overlap / total_possible * 100
        print(f"  As % of (smaller set size summed): {pct:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
