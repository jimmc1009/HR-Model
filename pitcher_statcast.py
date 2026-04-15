import os
import json
from datetime import date, timedelta
from typing import Dict, List, Set

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


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_today_probable_pitchers() -> Dict[str, dict]:
    """
    Returns {team_abbr: {name, id}} for today's probable starters.
    Falls back to tomorrow if today's games are already underway
    and probables are no longer listed.
    """
    def fetch_probables(date_str: str) -> Dict[str, dict]:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={date_str}&hydrate=probablePitcher"
        )
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        pitchers: Dict[str, dict] = {}
        for d in data.get("dates", []):
            for g in d.get("games", []):
                for side in ("away", "home"):
                    team_info = g.get("teams", {}).get(side, {})
                    abbr = team_info.get("team", {}).get("abbreviation", "")
                    probable = team_info.get("probablePitcher", {})
                    pitcher_id = probable.get("id")
                    pitcher_name = probable.get("fullName", "")
                    if abbr and pitcher_id:
                        pitchers[str(abbr).strip()] = {
                            "name": pitcher_name,
                            "id": int(pitcher_id),
                        }
        return pitchers

    # Try today first
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"Fetching probable pitchers for {today_str}...")
    pitchers = fetch_probables(today_str)

    if pitchers:
        print(f"Found {len(pitchers)} probable pitchers for today.")
        return pitchers

    # Fall back to tomorrow
    tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"No probables for today — trying tomorrow ({tomorrow_str})...")
    pitchers = fetch_probables(tomorrow_str)

    if pitchers:
        print(f"Found {len(pitchers)} probable pitchers for tomorrow.")
        return pitchers

    print("No probable pitchers found for today or tomorrow.")
    return {}


def get_today_matchups(probable_pitchers: Dict[str, dict]) -> Dict[str, str]:
    """
    Returns {team_abbr: opponent_abbr} for today's games.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={today_str}&hydrate=probablePitcher"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    matchups: Dict[str, str] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
            home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
            if away and home:
                matchups[away.strip()] = home.strip()
                matchups[home.strip()] = away.strip()
    return matchups


def get_season_statcast() -> pd.DataFrame:
    """
    Pull all 2026 season Statcast data from Opening Day through yesterday.
    Tries bulk pull first, falls back to monthly chunks.
    """
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


def lookup_player_names(player_ids: List[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    clean_ids = sorted({int(pid) for pid in player_ids if pd.notna(pid)})
    if not clean_ids:
        return out

    for i in range(0, len(clean_ids), 50):
        chunk = clean_ids[i:i + 50]
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        for person in resp.json().get("people", []):
            pid = person.get("id")
            name = person.get("fullName", "")
            if pid and name:
                out[int(pid)] = name
    return out


def filter_bbe_allowed(df: pd.DataFrame, probable_ids: Set[int]) -> pd.DataFrame:
    """
    Filter to one batted ball event per at-bat allowed by today's probable pitchers.
    """
    batted_ball_events = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "sac_fly",
        "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    }

    bbe = df[
        df["pitcher"].isin(probable_ids) &
        df["events"].astype("string").str.lower().isin(batted_ball_events) &
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna() &
        df["launch_angle"].between(-90, 90)
    ].copy()

    dedupe_cols = [
        c for c in ["game_pk", "at_bat_number", "pitcher"]
        if c in bbe.columns
    ]
    if dedupe_cols:
        bbe = bbe.drop_duplicates(subset=dedupe_cols)

    return bbe


def add_pitcher_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add barrel, hard-hit, HR, fly ball flags using official Statcast barrel definition."""
    df = df.copy()

    def is_barrel(row):
        ev = row["launch_speed"]
        la = row["launch_angle"]
        if pd.isna(ev) or pd.isna(la):
            return False
        if ev < 98:
            return False
        min_la = max(26 - (ev - 98), 8)
        max_la = min(30 + (ev - 98), 50)
        return min_la <= la <= max_la

    df["is_barrel"] = df.apply(is_barrel, axis=1)
    df["is_hard_hit"] = df["launch_speed"] >= 95
    df["is_hr"] = df["events"].astype("string").str.lower().eq("home_run")
    df["is_fly_ball"] = df["launch_angle"].between(25, 50, inclusive="both")

    return df


def build_season_stats_pitcher(bbe: pd.DataFrame, full_df: pd.DataFrame, probable_ids: Set[int]) -> pd.DataFrame:
    """Season-level stats allowed per pitcher."""
    # Batters faced = unique at-bats in full dataset
    bf_df = full_df[
        full_df["pitcher"].isin(probable_ids) &
        full_df["events"].notna()
    ].copy()
    bf_dedupe = [c for c in ["game_pk", "at_bat_number", "pitcher"] if c in bf_df.columns]
    bf_df = bf_df.drop_duplicates(subset=bf_dedupe)
    bf_counts = bf_df.groupby("pitcher").size().reset_index(name="bf")

    season = (
        bbe.groupby("pitcher", dropna=False)
        .agg(
            season_bbe_allowed=("launch_speed", "size"),
            season_hr_allowed=("is_hr", "sum"),
            season_fb_allowed=("is_fly_ball", "sum"),
            season_barrel_allowed=("is_barrel", "sum"),
            season_hard_hit_allowed=("is_hard_hit", "sum"),
            avg_ev_allowed=("launch_speed", "mean"),
        )
        .reset_index()
    )

    season = season.merge(bf_counts, on="pitcher", how="left")
    season["hr_per_bf"] = (season["season_hr_allowed"] / season["bf"] * 100).round(2)
    season["hr_per_fb_allowed"] = (
        season["season_hr_allowed"] / season["season_fb_allowed"].replace(0, pd.NA) * 100
    ).round(2)
    season["fb_rate_allowed"] = (
        season["season_fb_allowed"] / season["season_bbe_allowed"] * 100
    ).round(2)
    season["season_barrel_pct_allowed"] = (
        season["season_barrel_allowed"] / season["season_bbe_allowed"] * 100
    ).round(2)
    season["hard_hit_pct_allowed"] = (
        season["season_hard_hit_allowed"] / season["season_bbe_allowed"] * 100
    ).round(2)
    season["avg_ev_allowed"] = season["avg_ev_allowed"].round(2)

    return season


def build_platoon_splits_pitcher(bbe: pd.DataFrame) -> pd.DataFrame:
    """Barrel% and HR rate allowed vs LHH and RHH separately."""
    if "stand" not in bbe.columns:
        return pd.DataFrame(columns=["pitcher"])

    splits = []
    for hand, label in [("L", "vs_lhh"), ("R", "vs_rhh")]:
        sub = bbe[bbe["stand"] == hand]
        grp = (
            sub.groupby("pitcher", dropna=False)
            .agg(
                **{
                    f"{label}_bbe": ("launch_speed", "size"),
                    f"{label}_hr": ("is_hr", "sum"),
                    f"{label}_barrel_pct": ("is_barrel", "mean"),
                }
            )
            .reset_index()
        )
        grp[f"{label}_barrel_pct"] = (grp[f"{label}_barrel_pct"] * 100).round(2)
        grp[f"{label}_hr_rate"] = (
            grp[f"{label}_hr"] / grp[f"{label}_bbe"] * 100
        ).round(2)
