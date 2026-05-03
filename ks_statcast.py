"""
hrrbi_statcast.py
Reads from Batter_Statcast_2026 (already populated by main.py)
and writes HRRBI_Statcast with the columns needed by hrrbi_picks.py.
No Statcast pull — saves ~30 seconds per pipeline run.
"""

import os
import json

import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HRRBI_COLS = [
    "batter_id", "player_name", "team", "batter_hand",
    "batting_avg", "obp", "iso", "woba", "bb_pct", "k_pct",
    "hard_hit_pct_season", "season_barrel_pct",
    "ld_pct", "gb_pct", "fb_pct",
    "hr_per_pa", "hr_per_fb", "season_hr",
    "avg_bat_order",
    "bbe_7d", "avg_ev_7d", "avg_la_7d", "hard_hit_pct_7d", "barrel_pct_7d", "hr_7d",
    "pa_14d", "hits_14d", "avg_14d",
    "hot_streak", "cold_streak",
    "lhp_start_rate", "rhp_start_rate",
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def read_sheet(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    try:
        ws   = sh.worksheet(name)
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()


def write_sheet(gc: gspread.Client, sheet_id: str, name: str, df: pd.DataFrame) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=1000, cols=60)

    df = df.copy().replace([np.inf, -np.inf], np.nan).fillna("")
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading Batter_Statcast_2026...")
    batters = read_sheet(gc, sheet_id, "Batter_Statcast_2026")

    if batters.empty:
        print("ERROR: Batter_Statcast_2026 is empty — aborting")
        return

    print(f"  {len(batters)} batters loaded")

    keep    = [c for c in HRRBI_COLS if c in batters.columns]
    missing = [c for c in HRRBI_COLS if c not in batters.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")

    hrrbi_df = batters[keep].copy()

    # Rename batting_avg → avg for consistency with hrrbi_picks.py
    if "batting_avg" in hrrbi_df.columns:
        hrrbi_df = hrrbi_df.rename(columns={"batting_avg": "avg"})

    # Numeric coerce
    str_cols = ["batter_id", "player_name", "team", "batter_hand"]
    for col in hrrbi_df.columns:
        if col not in str_cols:
            hrrbi_df[col] = pd.to_numeric(hrrbi_df[col], errors="coerce")

    if "woba" in hrrbi_df.columns:
        hrrbi_df = hrrbi_df.sort_values("woba", ascending=False)

    hrrbi_df = hrrbi_df.reset_index(drop=True)
    print(f"HRRBI_Statcast: {len(hrrbi_df)} batters")

    write_sheet(gc, sheet_id, "HRRBI_Statcast", hrrbi_df)
    print("Written to HRRBI_Statcast")


if __name__ == "__main__":
    main()
