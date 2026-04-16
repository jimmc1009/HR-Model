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

FASTBALLS = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "CU", "KC", "CS", "SV", "ST"}
OFFSPEED = {"CH", "FS", "FO", "SC"}
KNUCKLEBALL = {"KN"}

ESPN_TO_MLB = {
    "WSH": "WSH", "HOU": "HOU", "MIL": "MIL", "LAD": "LAD",
    "BAL": "BAL", "CIN": "CIN", "NYY": "NYY", "TOR": "TOR",
    "COL": "COL", "OAK": "ATH", "CLE": "CLE", "CHC": "CHC",
    "SEA": "SEA", "PHI": "PHI", "DET": "DET", "ARI": "AZ",
    "TB":  "TB",  "SD":  "SD",  "STL": "STL", "ATL": "ATL",
    "KC":  "KC",  "LAA": "LAA", "NYM": "NYM", "MIA": "MIA",
    "MIN": "MIN", "PIT": "PIT", "TEX": "TEX", "SF":  "SF",
    "CWS": "CWS", "BOS": "BOS", "CHW": "CWS",
}

PITCH_GROUP_MAP = {
    **{p: "fastball" for p in FASTBALLS},
    **{p: "breaking" for p in BREAKING},
    **{p: "offspeed" for p in OFFSPEED},
    **{p: "knuckleball" for p in KNUCKLEBALL},
}


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def fetch_playing_teams(date_str: str) -> Set[str]:
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
    resp = requests.get(url, timeout=15)
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
    def fetch_mlb_probables(date_str: str) -> Dict[str, dict]:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={date_str}&hydrate=probablePitcher"
        )
        resp = requests.get(url, timeout=15)
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
                            "team": str(abbr).strip(),
                        }
        return pitchers

    def fetch_espn_probables() -> Dict[str, dict]:
        today_str = date.today().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today_str}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pitchers: Dict[str, dict] = {}
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    abbr = competitor.get("team", {}).get("abbreviation", "")
                    mlb_abbr = ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip())
                    probable = competitor.get("probables", [])
                    if probable and mlb_abbr:
                        p = probable[0]
                        athlete = p.get("athlete", {})
                        pid = athlete.get("id")
                        name = athlete.get("displayName", "")
                        if pid and name:
                            pitchers[mlb_abbr] = {
                                "name": name,
                                "id": int(pid),
                                "team": mlb_abbr,
                            }
        return pitchers

    def fetch_espn_teams() -> Set[str]:
        today_str = date.today().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today_str}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        teams: Set[str] = set()
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    abbr = competitor.get("team", {}).get("abbreviation", "")
                    if abbr:
                        mlb_abbr = ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip())
                        teams.add(mlb_abbr)
        return teams

    for days_ahead in (0, 1):
        date_str = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        label = "today" if days_ahead == 0 else "tomorrow"
        try:
            print(f"Fetching probable pitchers from MLB API for {label}...")
            pitchers = fetch_mlb_probables(date_str)
            if pitchers:
                print(f"MLB API found {len(pitchers)} probables for {label}.")
                return pitchers
            print(f"MLB API returned no probables for {label}.")
        except Exception as e:
            print(f"MLB API failed for {label}: {e}")

    try:
        print("Trying ESPN API for probable pitchers...")
        pitchers = fetch_espn_probables()
        if pitchers:
            print(f"ESPN API found {len(pitchers)} probable pitchers.")
            pitchers["_espn"] = True
            return pitchers
        print("ESPN API returned no probable pitchers.")
    except Exception as e:
        print(f"ESPN API failed for probables: {e}")

    try:
        print("Trying ESPN API for playing teams...")
        teams = fetch_espn_teams()
        if teams:
            print(f"ESPN API found playing teams: {teams}")
            return {"_fallback": True, "_teams": teams}
        print("ESPN API returned no teams.")
    except Exception as e:
        print(f"ESPN API failed for teams: {e}")

    print("Trying MLB API for playing teams...")
    playing_teams: Set[str] = set()
    for days_ahead in (0, 1):
        date_str = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        try:
            playing_teams |= fetch_playing_teams(date_str)
        except Exception as e:
            print(f"MLB API playing teams failed: {e}")

    if playing_teams:
        print(f"MLB API playing teams found: {playing_teams}")
        return {"_fallback": True, "_teams": playing_teams}

    print("All sources failed — using all pitchers from season data.")
    return {"_fallback": True, "_teams": "ALL"}


def get_today_matchups() -> Dict[str, str]:
    def fetch_mlb_matchups(date_str: str) -> Dict[str, str]:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        resp = requests.get(url, timeout=15)
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

    def fetch_espn_matchups() -> Dict[str, str]:
        today_str = date.today().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today_str}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        matchups: Dict[str, str] = {}
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                competitors = comp.get("competitors", [])
                if len(competitors) == 2:
                    abbr0 = ESPN_TO_MLB.get(
                        competitors[0].get("team", {}).get("abbreviation", ""),
                        competitors[0].get("team", {}).get("abbreviation", "")
                    )
                    abbr1 = ESPN_TO_MLB.get(
                        competitors[1].get("team", {}).get("abbreviation", ""),
                        competitors[1].get("team", {}).get("abbreviation", "")
                    )
                    if abbr0 and abbr1:
                        matchups[abbr0] = abbr1
                        matchups[abbr1] = abbr0
        return matchups

    for days_ahead in (0, 1):
        date_str = (date.today() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        try:
            matchups = fetch_mlb_matchups(date_str)
            if matchups:
                print(f"MLB API matchups: {len(matchups) // 2} games")
                return matchups
        except Exception as e:
            print(f"MLB API matchups failed: {e}")

    try:
        matchups = fetch_espn_matchups()
        if matchups:
            print(f"ESPN API matchups: {len(matchups) // 2} games")
            return matchups
    except Exception as e:
        print(f"ESPN API matchups failed: {e}")

    return {}


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


def build_pitch_mix(df: pd.DataFrame, probable_ids: Set[int]) -> pd.DataFrame:
    """
    For each pitcher calculate:
    1. % of each pitch group (fastball/breaking/offspeed/knuckleball/other)
    2. Top 3 individual pitch types by usage % stored as separate columns
    3. % of each individual pitch type as pitch_pct_XX columns
    """
    if "pitch_type" not in df.columns:
        return pd.DataFrame(columns=["pitcher"])

    pitch_df = df[
        df["pitcher"].isin(probable_ids) &
        df["pitch_type"].notna()
    ].copy()

    pitch_df["pitch_type"] = pitch_df["pitch_type"].astype("string").str.upper().str.strip()
    pitch_df = pitch_df[~pitch_df["pitch_type"].isin(["", "NAN", "NONE", "PO", "UN", "EP"])]

    pitch_df["pitch_group"] = pitch_df["pitch_type"].apply(
        lambda p: PITCH_GROUP_MAP.get(str(p), "other")
    )

    # Total pitches per pitcher
    total = (
        pitch_df.groupby("pitcher")
        .size()
        .reset_index(name="total_pitches")
    )

    # --- Group level percentages ---
    group_counts = (
        pitch_df.groupby(["pitcher", "pitch_group"])
        .size()
        .reset_index(name="count")
    )
    group_counts = group_counts.merge(total, on="pitcher", how="left")
    group_counts["pct"] = (
        group_counts["count"] / group_counts["total_pitches"] * 100
    ).round(2)

    group_pivot = group_counts.pivot_table(
        index="pitcher",
        columns="pitch_group",
        values="pct",
        fill_value=0,
    ).reset_index()
    group_pivot.columns.name = None

    for group in ["fastball", "breaking", "offspeed", "knuckleball", "other"]:
        col = f"pitch_pct_{group}"
        if group in group_pivot.columns:
            group_pivot = group_pivot.rename(columns={group: col})
        else:
            group_pivot[col] = 0.0

    # --- Individual pitch type percentages ---
    individual_counts = (
        pitch_df.groupby(["pitcher", "pitch_type"])
        .size()
        .reset_index(name="count")
    )
    individual_counts = individual_counts.merge(total, on="pitcher", how="left")
    individual_counts["pct"] = (
        individual_counts["count"] / individual_counts["total_pitches"] * 100
    ).round(2)

    # --- Top 3 pitch types per pitcher built DIRECTLY from individual_counts ---
    # Sort by pitcher then descending pct, take top 3 per pitcher
    individual_counts_sorted = individual_counts.sort_values(
        ["pitcher", "pct"], ascending=[True, False]
    )

    top3_records = []
    for pitcher_id, group in individual_counts_sorted.groupby("pitcher"):
        top3 = group.head(3).reset_index(drop=True)
        record = {"pitcher": pitcher_id}
        for i, (_, row) in enumerate(top3.iterrows(), start=1):
            record[f"top_pitch_{i}"] = row["pitch_type"]
            record[f"top_pitch_{i}_pct"] = row["pct"]
        # Fill missing ranks with empty
        for i in range(len(top3) + 1, 4):
            record[f"top_pitch_{i}"] = ""
            record[f"top_pitch_{i}_pct"] = 0.0
        top3_records.append(record)

    top3_df = pd.DataFrame(top3_records)

    # --- Individual pitch type pivot (pitch_pct_FF, pitch_pct_SL etc) ---
    individual_pivot = individual_counts.pivot_table(
        index="pitcher",
        columns="pitch_type",
        values="pct",
        fill_value=0,
    ).reset_index()
    individual_pivot.columns.name = None
    individual_pivot.columns = [
        f"pitch_pct_{c}" if c != "pitcher" else c
        for c in individual_pivot.columns
    ]

    # --- Merge all together ---
    # Start with group pivot + total
    result = group_pivot.merge(total, on="pitcher", how="left")

    # Merge top3 — these columns are ONLY in top3_df, no clash risk
    result = result.merge(top3_df, on="pitcher", how="left")

    # Merge individual pivot — prefix already applied so no clash with top3
    result = result.merge(individual_pivot, on="pitcher", how="left")

    # Ensure pct columns are numeric
    for col in ["top_pitch_1_pct", "top_pitch_2_pct", "top_pitch_3_pct"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).round(2)

    return result


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

    # Capture pitcher handedness
    agg_dict = {
        "season_bbe_allowed": ("launch_speed", "size"),
        "season_hr_allowed":  ("is_hr", "sum"),
        "season_fb_allowed":  ("is_fly_ball", "sum"),
        "season_barrel_allowed": ("is_barrel", "sum"),
        "season_hard_hit_allowed": ("is_hard_hit", "sum"),
        "avg_ev_allowed":     ("launch_speed", "mean"),
    }
    if "p_throws" in bbe.columns:
        agg_dict["pitcher_hand"] = ("p_throws", "first")

    season = (
        bbe.groupby("pitcher", dropna=False)
        .agg(**agg_dict)
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

    if "pitcher_hand" not in season.columns:
        season["pitcher_hand"] = ""

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
    is_espn = probable_pitchers.get("_espn", False)

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

        team_lookup = (
            df[df["pitcher"].isin(probable_ids)]
            .sort_values("game_date")
            .drop_duplicates(subset=["pitcher"], keep="last")
            .set_index("pitcher")["pitching_team"]
            .to_dict()
        )

        probable_pitchers = {
            str(pid): {
                "name": name_map.get(pid, ""),
                "id": pid,
                "team": str(team_lookup.get(pid, "")),
            }
            for pid in probable_ids
            if name_map.get(pid)
        }
        print(f"Resolved {len(probable_pitchers)} pitcher names")
        probable_ids = {v["id"] for v in probable_pitchers.values()}

    elif is_espn:
        print("ESPN mode — resolving pitcher IDs by name from Statcast data...")
        all_pitcher_ids = set(df["pitcher"].dropna().astype(int).unique())
        print(f"Looking up names for {len(all_pitcher_ids)} pitchers in Statcast...")
        name_map = lookup_player_names(list(all_pitcher_ids))

        name_to_statcast_id = {
            v.lower().strip(): k for k, v in name_map.items()
        }

        resolved = {}
        for team_abbr, info in probable_pitchers.items():
            if str(team_abbr).startswith("_"):
                continue
            espn_name = info.get("name", "").lower().strip()
            statcast_id = name_to_statcast_id.get(espn_name)
            if statcast_id:
                resolved[team_abbr] = {
                    "name": info["name"],
                    "id": statcast_id,
                    "team": info.get("team", team_abbr),
                }
            else:
                print(f"  Could not resolve: {info['name']} ({team_abbr})")

        print(f"Resolved {len(resolved)}/{len([k for k in probable_pitchers if not str(k).startswith('_')])} pitchers by name")
        probable_pitchers = resolved
        probable_ids = {v["id"] for v in probable_pitchers.values()}

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
    pitch_mix = build_pitch_mix(df, probable_ids)

    combined = (
        season_stats
        .merge(platoon_splits, on="pitcher", how="left")
        .merge(rolling_stats, on="pitcher", how="left")
        .merge(pitch_mix, on="pitcher", how="left")
    )

    id_to_name = {v["id"]: v["name"] for v in probable_pitchers.values()}
    id_to_team = {v["id"]: v.get("team", "") for v in probable_pitchers.values()}

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

    id_cols = ["pitcher_name", "pitcher_id", "pitcher_team", "opposing_team", "pitcher_hand"]
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
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=100)

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
