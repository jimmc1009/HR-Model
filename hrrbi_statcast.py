"""
hrrbi_statcast.py
Lightweight reader — copies batter data from Batter_Statcast_2026
into HRRBI_Statcast with all columns needed for momentum scoring.
"""

import os
import json
import time

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HRRBI_COLS = [
    "batter_id",
    "player_name",
    "team",
    "batter_hand",
    "pa",
    "avg",
    "obp",
    "iso",
    "woba",
    "bb_pct",
    "k_pct",
    "hard_hit_pct_season",
    "season_barrel_pct",
    "ld_pct",
    "gb_pct",
    "fb_pct",
    "hr_per_pa",
    "hr_per_fb",
    "season_hr",
    "avg_bat_order",
    # 7d metrics
    "bbe_7d",
    "avg_ev_7d",
    "avg_la_7d",
    "hard_hit_pct_7d",
    "barrel_pct_7d",
    "hr_7d",
    # 14d metrics
    "bbe_14d",
    "avg_ev_14d",
    "avg_la_14d",
    "hard_hit_pct_14d",
    "barrel_pct_14d",
    "hr_14d",
    "pa_14d",
    "hits_14d",
    "avg_14d",
    # 30d metrics
    "bbe_30d",
    "avg_ev_30d",
    "hard_hit_pct_30d",
    "barrel_pct_30d",
    # hot/cold
    "hot_streak",
    "cold_streak",
    # start rates
    "lhp_start_rate",
    "rhp_start_rate",
    # speed proxy
    "speed_score",
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except gspread.exceptions.APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    # Read Batter_Statcast_2026
    print("Reading Batter_Statcast_2026...")
    ws      = sh.worksheet("Batter_Statcast_2026")
    data    = with_retry(lambda: ws.get_all_records())
    batters = pd.DataFrame(data)
    print(f"  {len(batters)} batters loaded")

    # Keep only columns that exist
    available = [c for c in HRRBI_COLS if c in batters.columns]
    missing   = [c for c in HRRBI_COLS if c not in batters.columns]
    if missing:
        print(f"  Missing columns (will be skipped): {missing}")

    out = batters[available].copy()

    # Write to HRRBI_Statcast
    try:
        ws_out = sh.worksheet("HRRBI_Statcast")
        with_retry(lambda: ws_out.clear())
    except gspread.WorksheetNotFound:
        ws_out = sh.add_worksheet(title="HRRBI_Statcast", rows=500, cols=len(available) + 5)

    values = [out.columns.tolist()] + out.fillna("").astype(str).values.tolist()
    with_retry(lambda: ws_out.update(values))
    print(f"HRRBI_Statcast: {len(out)} batters")
    print("Written to HRRBI_Statcast")

    # Also read team K rates for ks_statcast
    time.sleep(2)
    print("Reading HRRBI_Statcast for team K rates...")
    ws_hrrbi = sh.worksheet("HRRBI_Statcast")
    hrrbi_df = pd.DataFrame(with_retry(lambda: ws_hrrbi.get_all_records()))

    if not hrrbi_df.empty and "team" in hrrbi_df.columns and "k_pct" in hrrbi_df.columns:
        team_k = hrrbi_df.groupby("team")["k_pct"].mean().reset_index()
        team_k.columns = ["team", "k_pct"]
    elif not hrrbi_df.empty and "team" in hrrbi_df.columns and "k_pct_season" in hrrbi_df.columns:
        team_k = hrrbi_df.groupby("team")["k_pct_season"].mean().reset_index()
        team_k.columns = ["team", "k_pct"]
    else:
        print("  Could not compute team K rates — column missing")
        team_k = pd.DataFrame()

    if not team_k.empty:
        try:
            ws_k = sh.worksheet("Team_K_Rates")
            with_retry(lambda: ws_k.clear())
        except gspread.WorksheetNotFound:
            ws_k = sh.add_worksheet(title="Team_K_Rates", rows=35, cols=5)
        with_retry(lambda: ws_k.update([team_k.columns.tolist()] + team_k.astype(str).values.tolist()))
        print(f"  Team K rates: {len(team_k)} teams")
        print("Written to Team_K_Rates")


if __name__ == "__main__":
    main()
