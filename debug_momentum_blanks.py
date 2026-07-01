"""
debug_momentum_blanks.py
Checks today's HR_All_Scores rows to see how many got a real momentum
score vs blank/zero, and diagnoses WHY each blank happened:
  - insufficient PA
  - missing baseline columns (avg_ev_30d, hard_hit_pct_season)
  - insufficient BBE in recent windows
  - split signal (legitimate neutral — 2-of-3 didn't agree)
"""

import os
import json
import time
from datetime import date

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MIN_BBE = 5
MIN_PA  = 50


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

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    ws = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers = all_values[0]
    rows    = all_values[1:]
    df      = pd.DataFrame(rows, columns=headers)

    today_str = date.today().strftime("%Y-%m-%d")
    today = df[df["date"].astype(str).str.strip() == today_str].copy()
    print(f"Today's date: {today_str}")
    print(f"Today's rows: {len(today)}")

    for col in ["pa", "bbe_5d", "bbe_7d", "bbe_10d",
                "barrel_pct_5d", "barrel_pct_7d", "barrel_pct_10d", "season_barrel_pct",
                "hard_hit_pct_7d", "hard_hit_pct_season",
                "avg_ev_7d", "avg_ev_30d", "momentum_score"]:
        if col in today.columns:
            today[col] = today[col].apply(safe_float)
        else:
            today[col] = 0.0

    blank = today[today["momentum_score"] == 0.0]
    nonblank = today[today["momentum_score"] != 0.0]

    print(f"\nMomentum blank/zero: {len(blank)} ({round(len(blank)/len(today)*100,1)}%)")
    print(f"Momentum non-zero:   {len(nonblank)} ({round(len(nonblank)/len(today)*100,1)}%)")

    print("\n" + "="*60)
    print("NON-ZERO MOMENTUM — distribution")
    print("="*60)
    if not nonblank.empty:
        print(f"  Min: {nonblank['momentum_score'].min():.3f}")
        print(f"  Max: {nonblank['momentum_score'].max():.3f}")
        print(f"  Mean: {nonblank['momentum_score'].mean():.3f}")
        print(f"  Std: {nonblank['momentum_score'].std():.3f}")

    print("\n" + "="*60)
    print("BLANK MOMENTUM — diagnosing why (sample of 20)")
    print("="*60)

    reasons = {"low_pa": 0, "missing_baseline": 0, "low_bbe_all_windows": 0, "split_signal_or_neutral": 0}

    for _, r in blank.head(40).iterrows():
        name = r.get("player_name", "?")
        pa = r["pa"]
        ev30 = r["avg_ev_30d"]
        hh_season = r["hard_hit_pct_season"]
        season_barrel = r["season_barrel_pct"]
        bbe5, bbe7, bbe10 = r["bbe_5d"], r["bbe_7d"], r["bbe_10d"]

        reason = []
        if pa < MIN_PA:
            reason.append(f"PA={pa}<{MIN_PA}")
            reasons["low_pa"] += 1
        if ev30 == 0 or hh_season == 0:
            reason.append(f"missing baseline (ev30={ev30}, hh_season={hh_season})")
            reasons["missing_baseline"] += 1
        if bbe5 < MIN_BBE and bbe7 < MIN_BBE and bbe10 < MIN_BBE:
            reason.append(f"all BBE windows <{MIN_BBE} (5d={bbe5},7d={bbe7},10d={bbe10})")
            reasons["low_bbe_all_windows"] += 1
        if not reason:
            reason.append("split signal / metrics disagree — legitimate neutral")
            reasons["split_signal_or_neutral"] += 1

        print(f"  {name:<25} {' | '.join(reason)}")

    print("\n" + "="*60)
    print("SUMMARY OF BLANK REASONS (first 40 blanks)")
    print("="*60)
    for k, v in reasons.items():
        print(f"  {k}: {v}")

    print("\nDone.")


if __name__ == "__main__":
    main()
