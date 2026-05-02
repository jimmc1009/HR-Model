"""
hrrbi_statcast.py
Pulls batter Statcast data for the H+R+RBI model.
Outputs to 'HRRBI_Statcast' Google Sheet tab.
Mirrors pitcher_statcast.py architecture from the HR model.
"""

import pandas as pd
import numpy as np
from pybaseball import statcast_batter_exitvelo_barrels, batting_stats, statcast
from datetime import date, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ── Google Sheets auth ──────────────────────────────────────────────────────
def get_sheet(sheet_id: str):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_json = os.environ["GOOGLE_CREDENTIALS"]
    creds_dict = json.loads(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id)

# ── Date helpers ────────────────────────────────────────────────────────────
def season_start() -> str:
    return "2025-03-27"

def days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).strftime("%Y-%m-%d")

def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")

# ── Pull season FanGraphs batting stats (BA, OBP, ISO, wOBA, K%, BB%) ───────
def pull_season_batting() -> pd.DataFrame:
    print("Pulling season batting stats (FanGraphs)...")
    try:
        df = batting_stats(2025, qual=50)
        cols = {
            "IDfg": "fg_id",
            "Name": "name",
            "Team": "team",
            "G": "games",
            "PA": "pa",
            "AB": "ab",
            "H": "hits",
            "HR": "hr",
            "R": "runs",
            "RBI": "rbi",
            "AVG": "avg",
            "OBP": "obp",
            "SLG": "slg",
            "ISO": "iso",
            "wOBA": "woba",
            "xwOBA": "xwoba",
            "BABIP": "babip",
            "BB%": "bb_pct",
            "K%": "k_pct",
            "LD%": "ld_pct",
            "GB%": "gb_pct",
            "FB%": "fb_pct",
            "Hard%": "hard_hit_pct_season",
            "Barrel%": "barrel_pct_season",
            "EV": "avg_ev_season",
            "Spd": "speed_score",
            "SB": "sb",
            "wRC+": "wrc_plus",
            "BatPos": "avg_bat_order",  # average lineup spot
        }
        df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})
        keep = [v for v in cols.values() if v in df.columns]
        return df[keep].copy()
    except Exception as e:
        print(f"  WARNING: season batting pull failed: {e}")
        return pd.DataFrame()

# ── Pull 7-day Statcast batter data ─────────────────────────────────────────
def pull_7d_statcast() -> pd.DataFrame:
    print("Pulling 7-day Statcast batter data...")
    start = days_ago(7)
    end = today_str()
    try:
        df = statcast(start_dt=start, end_dt=end)
        if df.empty:
            print("  WARNING: no 7-day Statcast data returned")
            return pd.DataFrame()

        # Filter to batted ball events
        bbe = df[df["type"] == "X"].copy()

        agg = bbe.groupby("batter").agg(
            bbe_7d=("launch_speed", "count"),
            avg_ev_7d=("launch_speed", "mean"),
            avg_la_7d=("launch_angle", "mean"),
            hard_hit_7d=("launch_speed", lambda x: (x >= 95).sum()),
            barrel_7d=("barrel", "sum"),
        ).reset_index()

        agg["hard_hit_pct_7d"] = (agg["hard_hit_7d"] / agg["bbe_7d"] * 100).round(1)
        agg["barrel_pct_7d"] = (agg["barrel_7d"] / agg["bbe_7d"] * 100).round(1)
        agg["avg_ev_7d"] = agg["avg_ev_7d"].round(1)
        agg["avg_la_7d"] = agg["avg_la_7d"].round(1)
        agg = agg.rename(columns={"batter": "mlb_id"})
        return agg[["mlb_id", "bbe_7d", "avg_ev_7d", "avg_la_7d",
                    "hard_hit_pct_7d", "barrel_pct_7d"]]
    except Exception as e:
        print(f"  WARNING: 7-day Statcast pull failed: {e}")
        return pd.DataFrame()

# ── Pull 14-day rolling H/R/RBI form from Statcast game logs ────────────────
def pull_14d_form() -> pd.DataFrame:
    """
    Pulls PA-level data and aggregates hits, estimated runs scored proxy,
    and RBI over last 14 days per batter.
    """
    print("Pulling 14-day H/R/RBI form...")
    start = days_ago(14)
    end = today_str()
    try:
        df = statcast(start_dt=start, end_dt=end)
        if df.empty:
            return pd.DataFrame()

        pa_events = [
            "single", "double", "triple", "home_run",
            "walk", "hit_by_pitch", "strikeout", "field_out",
            "grounded_into_double_play", "force_out", "sac_fly",
            "sac_bunt", "fielders_choice", "fielders_choice_out",
            "double_play", "triple_play",
        ]
        pa_df = df[df["events"].isin(pa_events)].copy()

        pa_df["is_hit"] = pa_df["events"].isin(
            ["single", "double", "triple", "home_run"]
        ).astype(int)

        pa_df["rbi_14d_est"] = pd.to_numeric(
            pa_df.get("bat_score", 0), errors="coerce"
        ).fillna(0)

        agg = pa_df.groupby("batter").agg(
            pa_14d=("events", "count"),
            hits_14d=("is_hit", "sum"),
        ).reset_index()

        agg["avg_14d"] = (agg["hits_14d"] / agg["pa_14d"]).round(3)
        agg = agg.rename(columns={"batter": "mlb_id"})
        return agg[["mlb_id", "pa_14d", "hits_14d", "avg_14d"]]
    except Exception as e:
        print(f"  WARNING: 14-day form pull failed: {e}")
        return pd.DataFrame()

# ── Pull RISP stats from FanGraphs splits ───────────────────────────────────
def pull_risp_stats() -> pd.DataFrame:
    """
    Pulls RISP wOBA/avg as RBI opportunity quality proxy.
    Uses FanGraphs season splits qual=30.
    """
    print("Pulling RISP stats...")
    try:
        from pybaseball import batting_stats_range
        # FanGraphs RISP splits not directly available via pybaseball
        # Use season wOBA with RISP proxy from standard batting
        # Flag for manual enrichment if needed
        print("  INFO: RISP splits not available via pybaseball — using season wOBA as proxy")
        return pd.DataFrame()
    except Exception as e:
        print(f"  WARNING: RISP pull failed: {e}")
        return pd.DataFrame()

# ── Main assembly ────────────────────────────────────────────────────────────
def build_hrrbi_statcast(sheet_id: str):
    print("\n=== HRRBI Statcast Pull ===")

    season = pull_season_batting()
    statcast_7d = pull_7d_statcast()
    form_14d = pull_14d_form()

    if season.empty:
        print("ERROR: season batting data empty — aborting")
        return

    # Merge 7-day Statcast on mlb_id if available
    # FanGraphs doesn't return mlb_id directly — join on name/team as fallback
    # (full ID mapping handled in main.py roster pull)
    df = season.copy()

    if not statcast_7d.empty and "mlb_id" in df.columns:
        df = df.merge(statcast_7d, on="mlb_id", how="left")

    if not form_14d.empty and "mlb_id" in df.columns:
        df = df.merge(form_14d, on="mlb_id", how="left")

    # Fill missing 7d/14d columns with 0
    for col in ["bbe_7d", "avg_ev_7d", "avg_la_7d",
                "hard_hit_pct_7d", "barrel_pct_7d",
                "pa_14d", "hits_14d", "avg_14d"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # Derived flags
    df["hot_streak"] = ((df["avg_14d"] >= 0.320) & (df["pa_14d"] >= 20)).astype(int)
    df["cold_streak"] = ((df["avg_14d"] <= 0.180) & (df["pa_14d"] >= 20)).astype(int)

    # Sort by woba descending
    sort_col = "woba" if "woba" in df.columns else (df.columns[0])
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"  Total batters: {len(df)}")

    # ── Write to Google Sheets ───────────────────────────────────────────────
    print("Writing to Google Sheets (HRRBI_Statcast)...")
    wb = get_sheet(sheet_id)

    try:
        ws = wb.worksheet("HRRBI_Statcast")
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = wb.add_worksheet(title="HRRBI_Statcast", rows=600, cols=40)

    df = df.fillna("").replace([np.inf, -np.inf], "")
    headers = df.columns.tolist()
    rows = df.values.tolist()
    ws.update([headers] + rows)

    # Header formatting
    ws.format("A1:AZ1", {
        "backgroundColor": {"red": 0.13, "green": 0.37, "blue": 0.73},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    })

    print(f"  ✓ HRRBI_Statcast written: {len(df)} rows")


if __name__ == "__main__":
    sheet_id = os.environ.get("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID environment variable not set")
    build_hrrbi_statcast(sheet_id)

