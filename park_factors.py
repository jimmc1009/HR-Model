import os
import json
from datetime import date, timedelta

import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials
from pybaseball import statcast

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SEASON_START = "2026-03-26"

# ── Park dimensions for pull hitter scoring ────────────────────────────────
# lf_dist: left field line distance (ft)
# lf_height: left field wall height (ft)
# rf_dist: right field line distance (ft)
# rf_height: right field wall height (ft)
# RHH pull hitters hit to LF — so lf_dist/lf_height are their relevant walls
# LHH pull hitters hit to RF — so rf_dist/rf_height are their relevant walls
# Sources: MLB.com, Ballpark Savvy, verified against 2026 changes (KC renovations)
# ATH = Sutter Health Park (Sacramento — temporary home 2025-2026)

PARK_DIMENSIONS = {
    "AZ":  {"lf_dist": 330, "lf_height": 7.5, "rf_dist": 334, "rf_height": 7.5},
    "ATL": {"lf_dist": 335, "lf_height": 8,   "rf_dist": 325, "rf_height": 8},
    "BAL": {"lf_dist": 333, "lf_height": 7,   "rf_dist": 318, "rf_height": 7},
    "BOS": {"lf_dist": 310, "lf_height": 37,  "rf_dist": 302, "rf_height": 3},
    "CHC": {"lf_dist": 355, "lf_height": 11,  "rf_dist": 353, "rf_height": 11},
    "CWS": {"lf_dist": 330, "lf_height": 8,   "rf_dist": 335, "rf_height": 8},
    "CIN": {"lf_dist": 328, "lf_height": 12,  "rf_dist": 325, "rf_height": 9},
    "CLE": {"lf_dist": 325, "lf_height": 19,  "rf_dist": 325, "rf_height": 9},
    "COL": {"lf_dist": 347, "lf_height": 8,   "rf_dist": 350, "rf_height": 8},
    "DET": {"lf_dist": 342, "lf_height": 8,   "rf_dist": 330, "rf_height": 8},
    "HOU": {"lf_dist": 315, "lf_height": 9,   "rf_dist": 326, "rf_height": 9},
    "KC":  {"lf_dist": 347, "lf_height": 8.5, "rf_dist": 344, "rf_height": 8.5},
    "LAA": {"lf_dist": 347, "lf_height": 8,   "rf_dist": 350, "rf_height": 8},
    "LAD": {"lf_dist": 330, "lf_height": 8,   "rf_dist": 330, "rf_height": 8},
    "MIA": {"lf_dist": 344, "lf_height": 8,   "rf_dist": 335, "rf_height": 8},
    "MIL": {"lf_dist": 342, "lf_height": 8,   "rf_dist": 337, "rf_height": 8},
    "MIN": {"lf_dist": 339, "lf_height": 8,   "rf_dist": 328, "rf_height": 8},
    "NYM": {"lf_dist": 335, "lf_height": 8,   "rf_dist": 330, "rf_height": 8},
    "NYY": {"lf_dist": 318, "lf_height": 8,   "rf_dist": 314, "rf_height": 8},
    "ATH": {"lf_dist": 330, "lf_height": 8,   "rf_dist": 320, "rf_height": 8},
    "PHI": {"lf_dist": 329, "lf_height": 8,   "rf_dist": 330, "rf_height": 8},
    "PIT": {"lf_dist": 325, "lf_height": 6,   "rf_dist": 320, "rf_height": 6},
    "SD":  {"lf_dist": 357, "lf_height": 8,   "rf_dist": 382, "rf_height": 8},
    "SF":  {"lf_dist": 339, "lf_height": 8,   "rf_dist": 309, "rf_height": 25},
    "SEA": {"lf_dist": 331, "lf_height": 8,   "rf_dist": 326, "rf_height": 8},
    "STL": {"lf_dist": 336, "lf_height": 8,   "rf_dist": 335, "rf_height": 8},
    "TB":  {"lf_dist": 315, "lf_height": 9,   "rf_dist": 322, "rf_height": 9},
    "TEX": {"lf_dist": 329, "lf_height": 8,   "rf_dist": 326, "rf_height": 8},
    "TOR": {"lf_dist": 328, "lf_height": 10,  "rf_dist": 328, "rf_height": 10},
    "WSH": {"lf_dist": 336, "lf_height": 8,   "rf_dist": 335, "rf_height": 8},
}

# League average dimensions for normalization
AVG_LF_DIST   = 331.5
AVG_RF_DIST   = 327.0
AVG_WALL_HT   = 9.0


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_season_statcast() -> pd.DataFrame:
    end_dt = date.today() - timedelta(days=1)
    season_start = date.fromisoformat(SEASON_START)

    try:
        print("Attempting bulk Statcast pull...")
        df = statcast(
            start_dt=season_start.strftime("%Y-%m-%d"),
            end_dt=end_dt.strftime("%Y-%m-%d"),
        )
        if df is not None and not df.empty:
            print(f"Bulk pull succeeded: {len(df):,} rows")
            return df
        print("Bulk pull returned empty — falling back to monthly chunks.")
    except Exception as e:
        print(f"Bulk pull failed ({e}) — falling back to monthly chunks.")

    chunks = []
    chunk_start = season_start

    while chunk_start <= end_dt:
        if chunk_start.month == 12:
            chunk_end = date(chunk_start.year + 1, 1, 1) - timedelta(days=1)
        else:
            chunk_end = date(chunk_start.year, chunk_start.month + 1, 1) - timedelta(days=1)

        chunk_end = min(chunk_end, end_dt)
        print(f"Pulling {chunk_start} → {chunk_end}...")

        try:
            chunk_df = statcast(
                start_dt=chunk_start.strftime("%Y-%m-%d"),
                end_dt=chunk_end.strftime("%Y-%m-%d"),
            )
            if chunk_df is not None and not chunk_df.empty:
                chunks.append(chunk_df)
                print(f"  ✓ {len(chunk_df):,} rows")
            else:
                print(f"  ✗ Empty result — skipping.")
        except Exception as e:
            print(f"  ✗ Chunk failed ({e}) — skipping.")

        if chunk_start.month == 12:
            chunk_start = date(chunk_start.year + 1, 1, 1)
        else:
            chunk_start = date(chunk_start.year, chunk_start.month + 1, 1)

    if not chunks:
        print("All chunks failed — returning empty DataFrame.")
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    print(f"Monthly chunking complete: {len(combined):,} total rows")
    return combined


def build_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HR park factor per team from Statcast data."""
    pa_df = df[df["events"].notna()].copy()
    ab_dedupe = [c for c in ["game_pk", "at_bat_number", "home_team"] if c in pa_df.columns]
    pa_df = pa_df.drop_duplicates(subset=ab_dedupe)
    pa_df["is_hr"] = pa_df["events"].astype("string").str.lower().eq("home_run")

    park_stats = (
        pa_df.groupby("home_team", dropna=False)
        .agg(pa=("is_hr", "size"), hr=("is_hr", "sum"))
        .reset_index()
    )
    park_stats = park_stats[park_stats["home_team"].notna()].copy()

    league_hr_rate = park_stats["hr"].sum() / park_stats["pa"].sum()
    print(f"League avg HR rate: {league_hr_rate:.4f}")

    park_stats["hr_rate"] = park_stats["hr"] / park_stats["pa"].replace(0, np.nan)
    park_stats["park_hr_factor"] = (park_stats["hr_rate"] / league_hr_rate * 100).round(1)
    park_stats["small_sample"] = park_stats["pa"] < 50
    park_stats = park_stats.rename(columns={"home_team": "team"})

    return park_stats[["team", "park_hr_factor", "small_sample"]]


def build_park_dimensions_table() -> pd.DataFrame:
    """
    Build a table of park dimensions with pull hitter scores.

    pull_boost_rhh: boost for a RHH pull hitter at this park (positive = HR friendly)
    pull_boost_lhh: boost for a LHH pull hitter at this park (positive = HR friendly)

    Logic:
    - RHH pull to LF: shorter LF dist = boost, taller LF wall = penalty
    - LHH pull to RF: shorter RF dist = boost, taller RF wall = penalty
    - Normalized against league averages
    - Raw score range roughly -2 to +2
    """
    rows = []
    for team, dims in PARK_DIMENSIONS.items():
        lf_dist   = dims["lf_dist"]
        lf_height = dims["lf_height"]
        rf_dist   = dims["rf_dist"]
        rf_height = dims["rf_height"]

        # Distance component: how much shorter/deeper than average (positive = shorter = better for HR)
        lf_dist_score = (AVG_LF_DIST - lf_dist) / 20   # per 20ft deviation
        rf_dist_score = (AVG_RF_DIST - rf_dist) / 20

        # Height component: how much shorter/taller than average (positive = shorter wall = better for HR)
        lf_height_score = (AVG_WALL_HT - lf_height) / 10  # per 10ft deviation
        rf_height_score = (AVG_WALL_HT - rf_height) / 10

        # Combined: distance matters more than wall height (2:1 weighting)
        pull_boost_rhh = round((lf_dist_score * 2 + lf_height_score) / 3, 3)
        pull_boost_lhh = round((rf_dist_score * 2 + rf_height_score) / 3, 3)

        rows.append({
            "team":           team,
            "lf_dist":        lf_dist,
            "lf_height":      lf_height,
            "rf_dist":        rf_dist,
            "rf_height":      rf_height,
            "pull_boost_rhh": pull_boost_rhh,
            "pull_boost_lhh": pull_boost_lhh,
        })

    return pd.DataFrame(rows)


def build_full_park_table(
    park_factors: pd.DataFrame,
    park_dims: pd.DataFrame,
) -> pd.DataFrame:
    """Merge HR park factors with dimension-based pull scores."""
    PARK_NAMES = {
        "AZ":  "Chase Field",
        "ATL": "Truist Park",
        "BAL": "Oriole Park at Camden Yards",
        "BOS": "Fenway Park",
        "CHC": "Wrigley Field",
        "CWS": "Guaranteed Rate Field",
        "CIN": "Great American Ball Park",
        "CLE": "Progressive Field",
        "COL": "Coors Field",
        "DET": "Comerica Park",
        "HOU": "Minute Maid Park",
        "KC":  "Kauffman Stadium",
        "LAA": "Angel Stadium",
        "LAD": "Dodger Stadium",
        "MIA": "loanDepot park",
        "MIL": "American Family Field",
        "MIN": "Target Field",
        "NYM": "Citi Field",
        "NYY": "Yankee Stadium",
        "ATH": "Sutter Health Park",
        "PHI": "Citizens Bank Park",
        "PIT": "PNC Park",
        "SD":  "Petco Park",
        "SF":  "Oracle Park",
        "SEA": "T-Mobile Park",
        "STL": "Busch Stadium",
        "TB":  "Tropicana Field",
        "TEX": "Globe Life Field",
        "TOR": "Rogers Centre",
        "WSH": "Nationals Park",
    }

    combined = park_factors.merge(park_dims, on="team", how="outer")
    combined["park_name"] = combined["team"].map(PARK_NAMES)
    combined["park_hr_factor"] = combined["park_hr_factor"].fillna(100.0)
    combined["small_sample"] = combined["small_sample"].fillna(True)
    combined["pull_boost_rhh"] = combined["pull_boost_rhh"].fillna(0.0)
    combined["pull_boost_lhh"] = combined["pull_boost_lhh"].fillna(0.0)

    col_order = [
        "team", "park_name", "park_hr_factor", "small_sample",
        "lf_dist", "lf_height", "rf_dist", "rf_height",
        "pull_boost_rhh", "pull_boost_lhh",
    ]
    combined = combined[col_order].sort_values("park_hr_factor", ascending=False)

    return combined


def clean_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
            else x
        )
    df = df.fillna("")
    return df


def write_dataframe_to_sheet(
    gc: gspread.Client,
    sheet_id: str,
    worksheet_name: str,
    df: pd.DataFrame,
) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=500, cols=15)

    df = clean_for_sheets(df)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    raw_df = get_season_statcast()
    print(f"Pulled {len(raw_df):,} Statcast rows for 2026 season")

    park_factors = build_park_factors(raw_df)
    park_dims = build_park_dimensions_table()
    full_table = build_full_park_table(park_factors, park_dims)

    print(f"Built park factors for {len(full_table)} parks")
    print(full_table[["team", "park_name", "park_hr_factor", "pull_boost_rhh", "pull_boost_lhh", "small_sample"]].to_string(index=False))

    write_dataframe_to_sheet(gc, sheet_id, "Park_Factors", full_table)
    print("Written to Park_Factors")


if __name__ == "__main__":
    main()
