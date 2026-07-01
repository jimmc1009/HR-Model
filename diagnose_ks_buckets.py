"""
diagnose_ks_buckets.py
Checks whether KS score correctly buckets pitchers by actual strikeout
production, independent of betting line or odds.

If the score works, higher score tiers should have higher actual_ks
on average, and the score should correlate with actual_ks numerically.
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
    ws = sh.worksheet("KS_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers = all_values[0]
    rows    = all_values[1:]
    df      = pd.DataFrame(rows, columns=headers)

    resolved = df[df["over_hit"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["ks_score"]  = resolved["ks_score"].apply(safe_float)
    resolved["actual_ks"] = resolved["actual_ks"].apply(lambda x: safe_float(x, -1))
    resolved["projected"] = resolved["projected_ks"].apply(safe_float)
    resolved = resolved[resolved["actual_ks"] >= 0]

    print(f"Resolved rows with actual K count: {len(resolved)}\n")

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

    print("="*60)
    print("AVG ACTUAL K's BY SCORE TIER")
    print("(If working, this should increase monotonically with tier)")
    print("="*60)
    print(f"\n  {'Tier':10s} {'N':>5} {'AvgScore':>9} {'AvgActualK':>11} {'AvgProj':>9}")
    print(f"  {'-'*48}")
    prev = -1
    for label, lo, hi in tiers:
        sub = resolved[(resolved["ks_score"] >= lo) & (resolved["ks_score"] < hi)]
        if len(sub) < 5: continue
        avg_score = sub["ks_score"].mean()
        avg_actual = sub["actual_ks"].mean()
        avg_proj = sub["projected"].mean()
        flag = " ⚠️ drops" if prev >= 0 and avg_actual < prev - 0.3 else ""
        prev = avg_actual
        print(f"  {label:10s} {len(sub):>5} {avg_score:>9.2f} {avg_actual:>11.2f} {avg_proj:>9.2f}{flag}")

    # Correlation
    corr = resolved["ks_score"].corr(resolved["actual_ks"])
    print(f"\n  Correlation (score vs actual K's): {corr:.3f}")

    proj_corr = resolved["projected"].corr(resolved["actual_ks"])
    print(f"  Correlation (projected K's vs actual K's): {proj_corr:.3f}")

    print("\n" + "="*60)
    print("PROJECTION ACCURACY — proj vs actual, by error bucket")
    print("="*60)
    resolved["proj_error"] = resolved["actual_ks"] - resolved["projected"]
    print(f"\n  Mean error (actual - projected): {resolved['proj_error'].mean():.2f}")
    print(f"  Mean absolute error: {resolved['proj_error'].abs().mean():.2f}")
    print(f"  Std of error: {resolved['proj_error'].std():.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
