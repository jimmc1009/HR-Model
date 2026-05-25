"""
ks_statcast.py
Reads from Pitcher_Statcast_2026 (already populated by pitcher_statcast.py)
and writes KS_Statcast with columns needed by ks_picks.py.
Team_K_Rates is built by main.py with full chase/whiff data.
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

KS_COLS = [
    "pitcher_name", "pitcher_id", "pitcher_team",
    "opposing_team", "home_team", "pitcher_hand",
    "k_pct_season",
    "bb_pct_season",
    "k_minus_bb",
    "k_per_9",
    "whip_proxy",
    "ks_ip",
    "games_started",
    "avg_ip_per_start",
    "opener_risk",
    "projected_k_baseline",
    "swstr_pct",
    "chase_rate",
    "first_pitch_strike_pct",
    "fastball_velo",
    "swstr_pct_21d",
    "avg_velo_21d",
    "k_last_3",
    "k_per_start_21d",
    "avg_ip_last_3",
    "swstr_trend",
    "velo_trend",
    "season_bbe_allowed",
    "hard_hit_pct_allowed",
    "season_barrel_pct_allowed",
    "avg_ev_allowed",
    "bf",
    "ip",
    "hr_per_fb_allowed",
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def read_sheet(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    try:
        ws         = sh.worksheet(name)
        all_values = ws.get_all_values()
        if not all_values or len(all_values) < 2:
            return pd.DataFrame()
        headers = all_values[0]
        rows    = all_values[1:]
        return pd.DataFrame(rows, columns=headers)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()


def write_sheet(gc: gspread.Client, sheet_id: str, name: str, df: pd.DataFrame) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=500, cols=60)

    df = df.copy().replace([np.inf, -np.inf], np.nan).fillna("")
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading Pitcher_Statcast_2026...")
    pitchers = read_sheet(gc, sheet_id, "Pitcher_Statcast_2026")

    if pitchers.empty:
        print("ERROR: Pitcher_Statcast_2026 is empty — aborting")
        return

    print(f"  {len(pitchers)} pitchers loaded")

    keep    = [c for c in KS_COLS if c in pitchers.columns]
    missing = [c for c in KS_COLS if c not in pitchers.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")

    ks_df = pitchers[keep].copy()

    if "ks_ip" in ks_df.columns:
        ks_df["ip"] = ks_df["ks_ip"]
        ks_df = ks_df.drop(columns=["ks_ip"])

    str_cols = ["pitcher_name", "pitcher_id", "pitcher_team",
                "opposing_team", "home_team", "pitcher_hand"]
    for col in ks_df.columns:
        if col not in str_cols:
            ks_df[col] = pd.to_numeric(ks_df[col], errors="coerce")

    if "k_pct_season" in ks_df.columns:
        ks_df = ks_df.sort_values("k_pct_season", ascending=False)

    ks_df = ks_df.reset_index(drop=True)
    print(f"KS_Statcast: {len(ks_df)} pitchers")

    write_sheet(gc, sheet_id, "KS_Statcast", ks_df)
    print("Written to KS_Statcast")

    # Team_K_Rates is now built by main.py with full chase/whiff data
    # Do not overwrite it here
    print("Team_K_Rates built by main.py — skipping.")


if __name__ == "__main__":
    main()
