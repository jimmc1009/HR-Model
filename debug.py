import os
import json
from datetime import date, timedelta
from typing import Dict, Set

import pandas as pd
import gspread
import requests
from google.oauth2.service_account import Credentials
from pybaseball import statcast

SEASON_START = "2026-03-26"

def get_today_team_abbrs() -> Set[str]:
    today_str = date.today().strftime("%Y-%m-%d")
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={today_str}&hydrate=probablePitcher"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    teams: Set[str] = set()
    for d in data.get("dates", []):
        for g in d.get("games", []):
            for side in ("away", "home"):
                abbr = g.get("teams", {}).get(side, {}).get("team", {}).get("abbreviation")
                if abbr:
                    teams.add(str(abbr).strip())
    return teams


def debug():
    # Step 1: Check today's teams
    print("=== STEP 1: Today's Teams ===")
    today_teams = get_today_team_abbrs()
    print(f"Teams playing today: {today_teams}")
    print(f"Count: {len(today_teams)}")

    # Step 2: Pull a small slice of Statcast (last 3 days only, fast)
    print("\n=== STEP 2: Statcast Sample Pull (last 3 days) ===")
    end_dt = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=3)
    df = statcast(start_dt=start_dt.strftime("%Y-%m-%d"), end_dt=end_dt.strftime("%Y-%m-%d"))
    print(f"Raw rows returned: {len(df) if df is not None else 'None'}")

    if df is None or df.empty:
        print("ERROR: Statcast returned nothing. Check pybaseball install / network.")
        return

    # Step 3: Check team columns
    print("\n=== STEP 3: Team Columns ===")
    for col in ["batting_team", "home_team", "away_team", "inning_topbot"]:
        if col in df.columns:
            print(f"  '{col}' exists — sample values: {df[col].dropna().unique()[:5].tolist()}")
        else:
            print(f"  '{col}' MISSING from Statcast data")

    # Step 4: Infer batting team and check overlap with today's teams
    print("\n=== STEP 4: Batting Team Inference ===")
    if "batting_team" in df.columns:
        df["team"] = df["batting_team"].fillna("").astype(str)
    elif {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
        df["team"] = df.apply(
            lambda r: str(r["away_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["home_team"]).strip(),
            axis=1,
        )
    else:
        df["team"] = ""

    unique_teams_in_data = set(df["team"].unique())
    print(f"Teams found in Statcast data: {unique_teams_in_data}")
    overlap = today_teams & unique_teams_in_data
    print(f"Overlap with today's teams: {overlap}")

    if not overlap:
        print("WARNING: No overlap — this is why the sheet is blank.")
        print("Possible causes:")
        print("  1. Today's games haven't started yet (Statcast lags by ~1 day)")
        print("  2. Team abbreviations don't match between MLB API and Statcast")
        print("  3. No games today")

    # Step 5: Check BBE filter
    print("\n=== STEP 5: BBE Filter ===")
    bbe = df[
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna() &
        df["launch_angle"].between(-90, 90)
    ]
    print(f"Rows surviving BBE filter: {len(bbe)}")

    # Step 6: Check season_bbe >= 10 filter
    print("\n=== STEP 6: Min BBE Filter (>= 10) ===")
    filtered_to_today = bbe[bbe["team"].isin(today_teams)]
    print(f"BBE rows for today's teams: {len(filtered_to_today)}")
    if not filtered_to_today.empty:
        bbe_counts = filtered_to_today.groupby("batter").size()
        print(f"Batters with >= 10 BBE: {(bbe_counts >= 10).sum()}")
        print(f"Batters with < 10 BBE: {(bbe_counts < 10).sum()}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    debug()
