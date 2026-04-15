import os
import json
from datetime import date, timedelta
from typing import Dict, List, Set, Union

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


def fetch_playing_teams(date_str: str) -> Set[str]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    teams: Set[str] = set()
    for d in data.get("dates", []):
        for g in d.get("games", []):
            for side in ("away", "home"):
                abbr = g.get("teams", {}).get(side, {}).get("team", {}).get("abbreviation", "")
                if abbr:
                    teams.add(str(abbr).strip())
    return teams


def get_today_probable_pitchers() -> Dict[str, dict]:
    """
    Tries to get probable pitchers from the MLB API for today and tomorrow.
    Falls back to playing teams, then falls back to all pitchers in season data.
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

    # Try today and tomorrow for probable pitchers
    for days_ahead in (0, 1):
        date_str = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        label = "today" if days_ahead == 0 else "tomorrow"
        print(f"Fetching probable pitchers for {date_str} ({label})...")
        pitchers = fetch_probables(date_str)
        if pitchers:
            print(f"Found {len(pitchers)} probable pitchers for {label}.")
            return pitchers
        print(f"No probables found for {label}.")

    # Try to get playing teams
    print("Falling back to playing teams...")
    playing_teams: Set[str] = set()
    for days_ahead in (0, 1):
        date_str = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        playing_teams |= fetch_playing_teams(date_str)

    if playing_teams:
        print(f"Playing teams found: {playing_teams}")
        return {"_fallback": True, "_teams": playing_teams}

    # Last resort: use all pitchers from season data
    print("No teams found from API — will use all pitchers from season data.")
    return {"_fallback": True, "_teams": "ALL"}


def get_today_matchups() -> Dict[str, str]:
    """
    Returns {team_abbr: opponent_abbr} for today's or tomorrow's games.
    """
    def fetch_matchups(date_str: str) -> Dict[str, str]:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={date_str}"
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

    today_str = date.today().strftime("%Y-%m-%d")
    matchups = fetch_matchups(today_str)
    if matchups:
        return matchups

    tomorrow_str = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    return fetch_matchups(tomorrow_str)


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


def infer_pitching_team(df: pd.DataFrame) -> pd.Series:
    required = {"inning_topbot", "home_team", "away_team"}
    if required.issubset(df.columns):
        return df.apply(
            lambda r: str(r["home_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["away_team"]).strip(),
            axis=1,
        )
    return pd.Series([""] * len(df), index=df.index)


def filter_bbe_allowed(df: pd.DataFrame, probable_ids: Set[int]) -> pd.DataFrame:
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


def build_season_stats_pitcher(
    bbe: pd.DataFrame,
    full_df: pd.DataFrame,
    probable_ids: Set[int],
) -> pd.DataFrame:
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
        splits.append(grp)

    return splits[0].merge(splits[1], on="pitcher", how="outer")


def build_rolling_stats_pitcher(
    bbe: pd.DataFrame,
    windows: List[int] = [7, 14, 30],
) -> pd.DataFrame:
    today = date.today()
    results = []

    for days in windows:
        cutoff = today - timedelta(days=days)
        window_df = bbe[bbe["game_date"] >= pd.Timestamp(cutoff)].copy()

        grp = (
            window_df.groupby("pitcher", dropna=False)
            .agg(
                **{
                    f"bbe_{days}d": ("launch_speed", "size"),
                    f"avg_ev_{days}d": ("launch_speed", "mean"),
                    f"barrel_pct_{days}d": ("is_barrel", "mean"),
                    f"hard_hit_pct_{days}d": ("is_hard_hit", "mean"),
                    f"hr_{days}d": ("is_hr", "sum"),
                }
            )
            .reset_index()
        )

        pct_cols = [f"barrel_pct_{days}d", f"hard_hit_pct_{days}d"]
        grp[pct_cols] = (grp[pct_cols] * 100).round(2)
        grp[f"avg_ev_{days}d"] = grp[f"avg_ev_{days}d"].round(2)
        results.append(grp)

    rolling = results[0]
    for r in results[1:]:
        rolling = rolling.merge(r, on="pitcher", how="outer")

    return rolling


def build_pitcher_full(
    df: pd.DataFrame,
    probable_pitchers: Dict[str, dict],
    matchups: Dict[str, str],
) -> pd.DataFrame:
    if df.empty or not probable_pitchers:
        return pd.DataFrame()

    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitching_team"] = infer_pitching_team(df)

    is_fallback = probable_pitchers.get("_fallback", False)

    if is_fallback:
        playing_teams = probable_pitchers.get("_teams", "ALL")

        if playing_teams == "ALL":
            print("Fallback mode — using ALL pitchers from season data.")
            probable_ids = set(df["pitcher"].dropna().astype(int).unique())
        else:
            print(f"Fallback mode — using pitchers from teams: {playing_teams}")
            team_df = df[df["pitching_team"].isin(playing_teams)].copy()
            probable_ids = set(team_df["pitcher"].dropna().astype(int).unique())

        print(f"Found {len(probable_ids)} pitchers — looking up names...")
        name_map = lookup_player_names(list(probable_ids))
        probable_pitchers = {
            name_map.get(pid, ""): {"name": name_map.get(pid, ""), "id": pid}
            for pid in probable_ids
            if name_map.get(pid)
        }
        print(f"Resolved {len(probable_pitchers)} pitcher names")
    else:
        probable_ids = {v["id"] for v in probable_pitchers.values()}

    print(f"Total pitcher IDs: {len(probable_ids)}")

    bbe = filter_bbe_allowed(df, probable_ids)
    print(f"BBE rows after filtering: {len(bbe)}")

    if bbe.empty:
        print("WARNING: No BBE rows found for pitchers.")
        return pd.DataFrame()

    bbe = add_pitcher_flags(bbe)

    season_stats = build_season_stats_pitcher(bbe, df, probable_ids)
    platoon_splits = build_platoon_splits_pitcher(bbe)
    rolling_stats = build_rolling_stats_pitcher(bbe)

    combined = (
        season_stats
        .merge(platoon_splits, on="pitcher", how="left")
        .merge(rolling_stats, on="pitcher", how="left")
    )

    id_to_name = {v["id"]: v["name"] for v in probable_pitchers.values()}
    id_to_team = {v["id"]: k for k, v in probable_pitchers.items()}

    combined["pitcher_name"] = combined["pitcher"].map(
        lambda x: id_to_name.get(int(x), "") if pd.notna(x) else ""
    )
    combined["pitcher_team"] = combined["pitcher"].map(
        lambda x: id_to_team.get(int(x), "") if pd.notna(x) else ""
    )
    combined["opposing_team"] = combined["pitcher_team"].map(
        lambda t: matchups.get(str(t), "")
    )

    combined = combined[combined["pitcher_name"] != ""].copy()
    combined = combined.rename(columns={"pitcher": "pitcher_id"})

    id_cols = ["pitcher_name", "pitcher_id", "pitcher_team", "opposing_team"]
    other_cols = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + other_cols]

    combined = combined.sort_values(
        by=["hr_per_fb_allowed", "season_barrel_pct_allowed", "avg_ev_allowed"],
        ascending=[False, False, False],
        na_position="last",
    )

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
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=50)

    df = clean_for_sheets(df)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    probable_pitchers = get_today_probable_pitchers()
    print(f"Probable pitchers result: {list(probable_pitchers.keys())[:5]}")

    matchups = get_today_matchups()
    print(f"Matchups found: {len(matchups)}")

    raw_df = get_season_statcast()
    print(f"Pulled {len(raw_df):,} Statcast rows for 2026 season")

    pitcher_df = build_pitcher_full(raw_df, probable_pitchers, matchups)
    print(f"Built {len(pitcher_df)} pitcher rows")

    if pitcher_df.empty:
        print("WARNING: Pitcher DataFrame is empty — nothing to write.")
        return

    write_dataframe_to_sheet(gc, sheet_id, "Pitcher_Statcast_2026", pitcher_df)
    print("Written to Pitcher_Statcast_2026")


if __name__ == "__main__":
    main()
