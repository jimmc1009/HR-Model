"""
migrate_ks_log.py
One-time migration to reorder KS_Picks_Log columns cleanly.
Preserves all existing data.
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

NEW_COLS = [
    "date",
    "rank",
    "pitcher_name",
    "team",
    "k_line",
    "prop_signal",
    "over_odds",
    "under_odds",
    "confidence",
    "bet_side",
    "odds",
    "wager",
    "win",
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet("KS_Picks_Log")

    print("Reading KS_Picks_Log...")
    all_values = ws.get_all_values()

    if not all_values:
        print("Sheet is empty — nothing to migrate.")
        return

    # Use get_all_values to avoid duplicate header error
    headers = all_values[0]
    rows    = all_values[1:]
    print(f"  {len(rows)} rows loaded")
    print(f"  Existing headers: {headers}")

    # Build dataframe from raw values
    df = pd.DataFrame(rows, columns=headers)

    # Drop completely empty rows
    df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)].copy()

    # Add any missing columns as empty
    for col in NEW_COLS:
        if col not in df.columns:
            print(f"  Adding missing column: {col}")
            df[col] = ""

    # Reorder to new column order
    df = df[NEW_COLS].copy()
    df = df.fillna("").replace([np.inf, -np.inf], "")

    print(f"Rewriting KS_Picks_Log with {len(df)} rows and {len(NEW_COLS)} columns...")
    ws.clear()
    time.sleep(2)
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())
    print("Done — KS_Picks_Log migrated successfully.")


if __name__ == "__main__":
    main()
