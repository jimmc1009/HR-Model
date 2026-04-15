import os
import json
from datetime import date, timedelta
from typing import Dict, Set

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

# Home park for each team abbreviation
TEAM_PARK = {
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
    """
    Calculate HR park factor for each ballpark using 2026 season Statcast data.

    Method:
    - Count HRs and PA at each home park (home_team column)
    - Calculate HR rate at each park vs league average
    - Park factor of 100 = league average
    - Above 100 = hitter friendly, below 100 = pitcher friendly
    - Flag parks with fewer than 50 PA as small sample
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Get one row per plate appearance (last pitch of each AB)
    pa_df = df[df["events"].notna()].copy()
    ab_dedupe = [c for c in ["game_pk", "at_bat_number", "home_team"] if c in pa_df.columns]
    pa_df = pa_df.drop_duplicates(subset=ab_dedupe)

    # Flag HRs
    pa_df["is_hr"] = pa_df["events"].astype("string").str.lower().eq("home_run")

    # Aggregate per home team (= per park)
    park_stats = (
        pa_df.groupby("home_team", dropna=False)
        .agg(
            pa=("is_hr", "size"),
            hr=("is_hr", "sum"),
        )
        .reset_index()
    )

    park_stats = park_stats[park_stats["home_team"].notna()].copy()
    park_stats["hr_rate"] = park_stats["hr"] / park_stats["pa"]

    # League average HR rate
    league_hr_rate = park_stats["hr"].sum() / park_stats["pa"].sum()
    print(f"League avg HR rate: {league_hr_rate:.4f}")

    # Park factor: team's HR rate / league avg * 100
    park_stats["park_hr_factor"] = (
        park_stats["hr_rate"] / league_hr_rate * 100
    ).round(1)

    # Small sample flag — fewer than 50 PA at this park
    park_stats["small_sample"] = park_stats["pa"] < 50

    # Add park name
    park_stats["park_name"] = park_stats["home_team"].map(
        lambda t: TEAM_PARK.get(str(t).strip(), "Unknown")
    )

    # Rename and sort
    park_stats = park_stats.rename(columns={"home_team": "team"})
    park_stats = park_stats[[
        "team",
        "park_name",
        "pa",
        "hr",
        "hr_rate",
        "park_hr_factor",
        "small_sample",
    ]]

    park_stats["hr_rate"] = park_stats["hr_rate"].round(4)
    park_stats = park_stats.sort_values("park_hr_factor", ascending=False)

    return park_stats


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
        ws = sh.add_worksheet(title=worksheet_name, rows=500, cols=20)

    df = clean_for_sheets(df)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    raw_df = get_season_statcast()
    print(f"Pulled {len(raw_df):,} Statcast rows for 2026 season")

    park_df = build_park_factors(raw_df)
    print(f"Built park factors for {len(park_df)} parks")
    print(park_df[["team", "park_name", "park_hr_factor", "small_sample"]].to_string(index=False))

    write_dataframe_to_sheet(gc, sheet_id, "Park_Factors", park_df)
    print("Written to Park_Factors")


if __name__ == "__main__":
    main()
