"""
ks_statcast.py
Reads from Pitcher_Statcast_2026 (already populated by pitcher_statcast.py)
and writes KS_Statcast + Team_K_Rates with columns needed by ks_picks.py.
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

# Columns from Pitcher_Statcast_2026 needed by KS model.
# pitcher_statcast.py now writes all KS metrics directly.
KS_COLS = [
    # Identity
    "pitcher_name", "pitcher_id", "pitcher_team",
    "opposing_team", "home_team", "pitcher_hand",

    # Season K metrics (added by pitcher_statcast.py)
    "k_pct_season",
    "bb_pct_season",
    "k_minus_bb",
    "k_per_9",
    "whip_proxy",
    "ks_ip",           # IP computed for KS model
    "games_started",
    "avg_ip_per_start",
    "opener_risk",
    "projected_k_baseline",

    # Pitch quality (added by pitcher_statcast.py)
    "swstr_pct",
    "chase_rate",
    "first_pitch_strike_pct",
    "fastball_velo",

    # 21-day rolling (added by pitcher_statcast.py)
    "swstr_pct_21d",
    "avg_velo_21d",
    "k_last_3",
    "k_per_start_21d",
    "avg_ip_last_3",
    "swstr_trend",
    "velo_trend",

    # BBE allowed (existing)
    "season_bbe_allowed",
    "hard_hit_pct_allowed",
    "season_barrel_pct_allowed",
    "avg_ev_allowed",

    # Existing pitcher stats useful for context
    "bf",
    "ip",
    "hr_per_fb_allowed",
]

# Columns from Batter_Statcast_2026 for team K rate computation
BATTER_K_COLS = ["batter_id", "team", "k_pct"]


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
        ws = sh.add_worksheet(title=name, rows=500, cols=60)

    df = df.copy().replace([np.inf, -np.inf], np.nan).fillna("")
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())


def build_team_k_rates(batters: pd.DataFrame) -> pd.DataFrame:
    """
    Build team K rates from HRRBI_Statcast batter data.
    Uses k_pct column already computed by main.py.
    """
    if batters.empty or "k_pct" not in batters.columns or "team" not in batters.columns:
        print("  WARNING: Cannot build team K rates — missing k_pct or team column")
        return pd.DataFrame()

    batters = batters.copy()
    batters["k_pct"] = pd.to_numeric(batters["k_pct"], errors="coerce")
    batters = batters[batters["k_pct"].notna() & (batters["team"].astype(str).str.strip() != "")]

    team_k = (
        batters.groupby("team")["k_pct"]
        .mean()
        .reset_index(name="team_k_pct")
    )
    team_k["team_k_pct"] = team_k["team_k_pct"].round(1)
    print(f"  Team K rates: {len(team_k)} teams")
    return team_k


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
        print(f"  WARNING: Missing columns (may affect scoring): {missing}")

    ks_df = pitchers[keep].copy()

    # Rename ks_ip → ip if the KS-specific IP column exists
    if "ks_ip" in ks_df.columns and "ip" not in ks_df.columns:
        ks_df = ks_df.rename(columns={"ks_ip": "ip"})
    elif "ks_ip" in ks_df.columns:
        # Use ks_ip as the authoritative IP for KS model
        ks_df["ip"] = ks_df["ks_ip"]
        ks_df = ks_df.drop(columns=["ks_ip"])

    # Numeric coerce
    str_cols = ["pitcher_name", "pitcher_id", "pitcher_team",
                "opposing_team", "home_team", "pitcher_hand"]
    for col in ks_df.columns:
        if col not in str_cols:
            ks_df[col] = pd.to_numeric(ks_df[col], errors="coerce")

    # Sort by k_pct_season
    if "k_pct_season" in ks_df.columns:
        ks_df = ks_df.sort_values("k_pct_season", ascending=False)

    ks_df = ks_df.reset_index(drop=True)
    print(f"KS_Statcast: {len(ks_df)} pitchers")

    write_sheet(gc, sheet_id, "KS_Statcast", ks_df)
    print("Written to KS_Statcast")

    # ── Team K rates from HRRBI_Statcast ──────────────────────────
    print("Reading HRRBI_Statcast for team K rates...")
    batters = read_sheet(gc, sheet_id, "HRRBI_Statcast")
    if not batters.empty:
        team_k = build_team_k_rates(batters)
        if not team_k.empty:
            write_sheet(gc, sheet_id, "Team_K_Rates", team_k)
            print("Written to Team_K_Rates")
    else:
        print("  WARNING: HRRBI_Statcast empty — Team_K_Rates not updated")


if __name__ == "__main__":
    main()
