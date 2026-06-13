"""
check_hr_repeat_hitrate.py
Diagnostic: for players who appear in the top-20 scored list on
consecutive days, compare their hit rate vs players appearing
for the first time (not in previous day's top 20).

Helps determine if "hot" players staying at the top are actually
converting, or just sitting there due to short-window stat inertia.
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

TOP_N = 20


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

    df["hr_score"]  = df["hr_score"].apply(safe_float)
    df["name_norm"] = df["player_name"].apply(normalize_name)
    df["date"]      = df["date"].astype(str).str.strip()
    df["hit_bool"]  = df["hit_hr"].astype(str).str.strip() == "Yes"

    # Only keep rows that are resolved (Yes/No) for hit rate calc
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    dates = sorted(df["date"].unique())

    repeat_hits, repeat_total = 0, 0
    new_hits, new_total       = 0, 0

    repeat_names = []
    new_names    = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_df = df[df["date"] == prev_date]
        curr_df = resolved[resolved["date"] == curr_date]  # only resolved for hit rate

        prev_top = prev_df.sort_values("hr_score", ascending=False).head(TOP_N)
        prev_top_names = set(prev_top["name_norm"])

        curr_top = curr_df.sort_values("hr_score", ascending=False).head(TOP_N)

        for _, r in curr_top.iterrows():
            if r["name_norm"] in prev_top_names:
                repeat_total += 1
                if r["hit_bool"]:
                    repeat_hits += 1
                    repeat_names.append((curr_date, r["player_name"], r["hr_score"]))
            else:
                new_total += 1
                if r["hit_bool"]:
                    new_hits += 1
                    new_names.append((curr_date, r["player_name"], r["hr_score"]))

    print("=" * 60)
    print("REPEAT TOP-20 PLAYERS (in top 20 yesterday AND today)")
    print(f"  Total: {repeat_total} | Hits: {repeat_hits} | Hit Rate: {repeat_hits/repeat_total*100:.1f}%" if repeat_total else "  No data")
    print()
    print("NEW TOP-20 PLAYERS (not in yesterday's top 20)")
    print(f"  Total: {new_total} | Hits: {new_hits} | Hit Rate: {new_hits/new_total*100:.1f}%" if new_total else "  No data")
    print("=" * 60)
    print()

    if repeat_total and new_total:
        diff = (repeat_hits/repeat_total) - (new_hits/new_total)
        print(f"Difference (repeat - new): {diff*100:+.1f} percentage points")
        if diff < -0.03:
            print("=> Repeat players underperform new players — possible 'hot hand' overstay issue")
        elif diff > 0.03:
            print("=> Repeat players outperform new players — recency is predictive")
        else:
            print("=> No meaningful difference")

    print()
    print("Repeat-player HITS (converted):")
    for d, n, s in repeat_names:
        print(f"  {d}  {n:25s} score={s:.2f}")

    print()
    print("New-player HITS (converted):")
    for d, n, s in new_names:
        print(f"  {d}  {n:25s} score={s:.2f}")


if __name__ == "__main__":
    main()
