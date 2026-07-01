"""
extract_feature_bounds.py
One-time helper: reads HR_All_Scores and reports the real distribution
(percentiles) of each feature that will feed the new percentile-normalized
power and context components. Output is used to set normalization bounds
in the new hr_picks.py so they're grounded in actual data, not guesses.
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


def safe_float(val, default=np.nan):
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

    features = [
        # Power stack
        "season_barrel_pct", "barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d",
        "iso", "hr_per_fb", "hr_per_pa",
        # Context / differentiators
        "platoon_score", "pitch_matchup_score", "hr_weather_boost",
        "park_hr_factor", "avg_la_7d", "avg_la_season",
        # Pitcher side
        "pitcher_barrel_pct", "pitcher_hr_per_fb",
    ]

    print(f"\n  {len(df)} rows from {MODEL_START_DATE}\n")
    print(f"  {'Feature':<22} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p95':>8} {'min':>8} {'max':>8}")
    print(f"  {'-'*82}")

    for col in features:
        if col not in df.columns:
            print(f"  {col:<22} NOT FOUND")
            continue
        vals = df[col].apply(safe_float).dropna()
        if len(vals) < 20:
            print(f"  {col:<22} too few values ({len(vals)})")
            continue
        p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
        print(f"  {col:<22} {p5:>8.2f} {p25:>8.2f} {p50:>8.2f} {p75:>8.2f} {p95:>8.2f} {vals.min():>8.2f} {vals.max():>8.2f}")

    print("\n  Use p5 as normalization floor, p95 as ceiling for each power feature.")
    print("\nDone.")


if __name__ == "__main__":
    main()
