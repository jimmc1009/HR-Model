"""
debug_contact_columns.py
Quick check: are avg_ev_30d, hard_hit_pct_7d, hard_hit_pct_season
actually populated with varying values in Batter_Statcast_2026 and
in today's HR_All_Scores rows?
"""

import os
import json
import time

import pandas as pd
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

def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()
    sh       = with_retry(lambda: gc.open_by_key(sheet_id))

    print("="*60)
    print("1. Batter_Statcast_2026 — raw column check")
    print("="*60)
    ws = sh.worksheet("Batter_Statcast_2026")
    values = with_retry(lambda: ws.get_all_values())
    headers = values[0]
    rows = values[1:11]  # first 10 players

    for col in ["avg_ev_30d", "hard_hit_pct_7d", "hard_hit_pct_season", "avg_ev_7d"]:
        if col in headers:
            idx = headers.index(col)
            sample = [r[idx] if idx < len(r) else "MISSING" for r in rows]
            print(f"  {col}: {sample}")
        else:
            print(f"  {col}: COLUMN NOT FOUND IN SHEET")

    print("\n" + "="*60)
    print("2. HR_All_Scores — today's rows check")
    print("="*60)
    ws2 = sh.worksheet("HR_All_Scores")
    values2 = with_retry(lambda: ws2.get_all_values())
    headers2 = values2[0]
    all_rows = values2[1:]

    from datetime import date
    today_str = date.today().strftime("%Y-%m-%d")
    today_rows = [r for r in all_rows if len(r) > 0 and r[0].strip() == today_str]

    print(f"  Today's date: {today_str}")
    print(f"  Today's rows found: {len(today_rows)}")

    for col in ["avg_ev_30d", "hard_hit_pct_7d", "hard_hit_pct_season", "avg_ev_7d", "barrel_pct_7d", "season_barrel_pct"]:
        if col in headers2:
            idx = headers2.index(col)
            sample = [r[idx] if idx < len(r) else "SHORT_ROW" for r in today_rows[:10]]
            print(f"  {col} (col idx {idx}): {sample}")
        else:
            print(f"  {col}: COLUMN NOT FOUND IN HR_All_Scores HEADER")

    print(f"\n  Total columns in HR_All_Scores header: {len(headers2)}")
    print(f"  Last 10 headers: {headers2[-10:]}")

if __name__ == "__main__":
    main()
