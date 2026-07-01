"""
test_hand_split.py
Verifies the handedness-split pitch usage is working. Pulls Pitcher_Statcast_2026
and shows, for pitchers with a sweeper (ST) or slider (SL), their usage vs LHH
vs RHH side by side. If the split worked, these should DIFFER — often a lot,
since sweepers/sliders are thrown far more to same-handed batters.
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
    ws = sh.worksheet("Pitcher_Statcast_2026")
    values = with_retry(lambda: ws.get_all_values())
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)

    # Which handedness-split columns exist?
    split_cols = [c for c in headers if c.startswith("pitch_pct_") and (c.endswith("_vs_L") or c.endswith("_vs_R"))]
    print(f"Handedness-split usage columns found: {len(split_cols)}")
    if not split_cols:
        print("  ⚠️ NONE FOUND — pitcher_statcast.py hasn't run with the new code yet,")
        print("     or the run predates the change. Re-run pitcher_statcast on dev first.")
        return
    print(f"  Sample: {split_cols[:10]}")

    # For each pitch type present, show vs_L / vs_R for pitchers who throw it
    print("\n" + "="*70)
    print("SWEEPER (ST) & SLIDER (SL) USAGE: vs LHH vs RHH")
    print("If the split works, these differ — often heavily same-handed.")
    print("="*70)

    for pt in ["ST", "SL", "SI", "CH"]:
        lcol = f"pitch_pct_{pt}_vs_L"
        rcol = f"pitch_pct_{pt}_vs_R"
        if lcol not in df.columns or rcol not in df.columns:
            continue
        print(f"\n  --- {pt} ---")
        print(f"    {'Pitcher':<24} {'vs LHH':>8} {'vs RHH':>8} {'Gap':>8}")
        print(f"    {'-'*52}")
        sub = df.copy()
        sub["_l"] = sub[lcol].apply(safe_float)
        sub["_r"] = sub[rcol].apply(safe_float)
        sub["_gap"] = (sub["_l"] - sub["_r"]).abs()
        sub = sub[(sub["_l"] > 0) | (sub["_r"] > 0)].sort_values("_gap", ascending=False)
        name_col = "pitcher_name" if "pitcher_name" in sub.columns else headers[0]
        for _, r in sub.head(8).iterrows():
            nm = str(r.get(name_col, ""))[:23]
            print(f"    {nm:<24} {r['_l']:>7.1f}% {r['_r']:>7.1f}% {r['_gap']:>7.1f}")

    print("\nDone. Big gaps confirm the handedness split is working.")


if __name__ == "__main__":
    main()
