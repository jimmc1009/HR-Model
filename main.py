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


def get_today_team_abbrs() -> Set[str]:
    def fetch_mlb_api() -> Set[str]:
        today_str = date.today().strftime("%Y-%m-%d")
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={today_str}&hydrate=probablePitcher"
        )
        resp = requests.get(url, timeout=15)
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

    def fetch_espn_api() -> Set[str]:
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

    try:
        print("Fetching today's teams from MLB API...")
        teams = fetch_mlb_api()
        if teams:
            print(f"MLB API returned {len(teams)} teams: {teams}")
            return teams
        print("MLB API returned no teams.")
    except Exception as e:
        print(f"MLB API failed: {e}")

    try:
        print("Falling back to ESPN API...")
        teams = fetch_espn_api()
        if teams:
            print(f"ESPN API returned {len(teams)} teams: {teams}")
            return teams
        print("ESPN API returned no teams.")
    except Exception as e:
        print(f"ESPN API failed: {e}")

    print("Both APIs failed — returning empty set.")
    return set()


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


def infer_batting_team(df: pd.DataFrame) -> pd.Series:
    if "batting_team" in df.columns:
        return df["batting_team"].fillna("").astype(str)
    required = {"inning_topbot", "home_team", "away_team"}
    if required.issubset(df.columns):
        return df.apply(
            lambda r: str(r["away_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["home_team"]).strip(),
            axis=1,
        )
    return pd.Series([""] * len(df), index=df.index)


def add_statcast_flags(df: pd.DataFrame) -> pd.DataFrame:
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
    df["is_sweet_spot"] = df["launch_angle"].between(8, 32, inclusive="both")
    df["is_hr"] = df["events"].astype("string").str.lower().eq("home_run")
    df["is_fly_ball"] = df["launch_angle"].between(25, 50, inclusive="both")

    if "stand" in df.columns and "hc_x" in df.columns:
        df["is_pull"] = df.apply(
            lambda r: (r["hc_x"] < 125) if str(r.get("stand")) == "R"
            else (r["hc_x"] > 125) if str(r.get("stand")) == "L"
            else False,
            axis=1,
        )
    else:
        df["is_pull"] = False

    if "pitch_type" in df.columns:
        df["pitch_group"] = df["pitch_type"].apply(
            lambda p: PITCH_GROUP_MAP.get(str(p), "other")
        )
    else:
        df["pitch_group"] = "other"

    return df


def filter_bbe(df: pd.DataFrame) -> pd.DataFrame:
    batted_ball_events = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "sac_fly",
        "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    }

    bbe = df[
        df["events"].astype("string").str.lower().isin(batted_ball_events) &
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna() &
        df["launch_angle"].between(-90, 90)
    ].copy()

    dedupe_cols = [
        c for c in ["game_pk", "at_bat_number", "batter"]
        if c in bbe.columns
    ]
    if dedupe_cols:
        bbe = bbe.drop_duplicates(subset=dedupe_cols)

    return bbe


def build_season_stats(bbe: pd.DataFrame, full_df: pd.DataFrame) -> pd.DataFrame:
    pa_df = full_df[full_df["events"].notna()].copy()
    ab_dedupe = [c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
    pa_df = pa_df.drop_duplicates(subset=ab_dedupe)
    pa_counts = pa_df.groupby("batter").size().reset_index(name="pa")

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    bbe = bbe.copy()
    bbe["total_bases"] = bbe["events"].astype("string").str.lower().map(tb_map).fillna(0)
    bbe["is_hit"] = bbe["events"].astype("string").str.lower().isin(
        {"single", "double", "triple", "home_run"}
    )

    season = (
        bbe.groupby("batter", dropna=False)
        .agg(
            season_bbe=("launch_speed", "size"),
            season_hr=("is_hr", "sum"),
            season_fb=("is_fly_ball", "sum"),
            season_pull=("is_pull", "sum"),
            season_barrel=("is_barrel", "sum"),
            total_bases=("total_bases", "sum"),
            hits=("is_hit", "sum"),
            avg_launch_angle=("launch_angle", "mean"),
        )
        .reset_index()
    )

    season = season.merge(pa_counts, on="batter", how="left")
    season["avg_launch_angle"] = season["avg_launch_angle"].round(2)
    season["hr_per_pa"] = (season["season_hr"] / season["pa"] * 100).round(2)
    season["hr_per_fb"] = (
        season["season_hr"] / season["season_fb"].replace(0, pd.NA) * 100
    ).round(2)
    season["fb_rate"] = (season["season_fb"] / season["season_bbe"] * 100).round(2)
    season["pull_rate"] = (season["season_pull"] / season["season_bbe"] * 100).round(2)
    season["season_barrel_pct"] = (
        season["season_barrel"] / season["season_bbe"] * 100
    ).round(2)
    season["iso"] = (
        (season["total_bases"] - season["hits"]) / season["pa"].replace(0, pd.NA)
    ).round(3)

    return season


def build_platoon_splits(bbe: pd.DataFrame) -> pd.DataFrame:
    if "p_throws" not in bbe.columns:
        return pd.DataFrame(columns=["batter"])

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    bbe = bbe.copy()
    bbe["total_bases"] = bbe["events"].astype("string").str.lower().map(tb_map).fillna(0)
    bbe["is_hit"] = bbe["events"].astype("string").str.lower().isin(
        {"single", "double", "triple", "home_run"}
    )

    stand_map = (
        bbe[bbe["stand"].notna()]
        .groupby("batter")["stand"]
        .first()
        .reset_index()
        .rename(columns={"stand": "batter_hand"})
    ) if "stand" in bbe.columns else pd.DataFrame(columns=["batter", "batter_hand"])

    splits = []
    for hand, label in [("L", "vs_lhp"), ("R", "vs_rhp")]:
        sub = bbe[bbe["p_throws"] == hand].copy()

        pa_count = sub.groupby("batter").size().reset_index(name=f"{label}_pa")

        grp = (
            sub.groupby("batter", dropna=False)
            .agg(
                **{
                    f"{label}_bbe":        ("launch_speed", "size"),
                    f"{label}_hr":         ("is_hr", "sum"),
                    f"{label}_barrel_pct": ("is_barrel", "mean"),
                    f"{label}_tb":         ("total_bases", "sum"),
                    f"{label}_hits":       ("is_hit", "sum"),
                }
            )
            .reset_index()
        )

        grp = grp.merge(pa_count, on="batter", how="left")
        grp[f"{label}_barrel_pct"] = (grp[f"{label}_barrel_pct"] * 100).round(2)
        grp[f"{label}_hr_rate"] = (
            grp[f"{label}_hr"] / grp[f"{label}_bbe"] * 100
        ).round(2)
        grp[f"{label}_iso"] = (
            (grp[f"{label}_tb"] - grp[f"{label}_hits"]) /
            grp[f"{label}_pa"].replace(0, pd.NA)
        ).round(3)

        grp = grp.drop(columns=[f"{label}_tb", f"{label}_hits", f"{label}_pa"])
        splits.append(grp)

    result = splits[0].merge(splits[1], on="batter", how="outer")

    if not stand_map.empty:
        result = result.merge(stand_map, on="batter", how="left")
    else:
        result["batter_hand"] = ""

    return result


def build_rolling_stats(bbe: pd.DataFrame, windows: List[int] = [7, 14, 30]) -> pd.DataFrame:
    today = date.today()
    results = []

    for days in windows:
        cutoff = today - timedelta(days=days)
        window_df = bbe[bbe["game_date"] >= pd.Timestamp(cutoff)].copy()

        grp = (
            window_df.groupby("batter", dropna=False)
            .agg(
                **{
                    f"bbe_{days}d":          ("launch_speed", "size"),
                    f"avg_ev_{days}d":       ("launch_speed", "mean"),
                    f"avg_la_{days}d":       ("launch_angle", "mean"),
                    f"barrel_pct_{days}d":   ("is_barrel", "mean"),
                    f"hard_hit_pct_{days}d": ("is_hard_hit", "mean"),
                    f"hr_{days}d":           ("is_hr", "sum"),
                }
            )
            .reset_index()
        )

        pct_cols = [f"barrel_pct_{days}d", f"hard_hit_pct_{days}d"]
        grp[pct_cols] = (grp[pct_cols] * 100).round(2)
        grp[f"avg_ev_{days}d"] = grp[f"avg_ev_{days}d"].round(2)
        grp[f"avg_la_{days}d"] = grp[f"avg_la_{days}d"].round(2)
        results.append(grp)

    rolling = results[0]
    for r in results[1:]:
        rolling = rolling.merge(r, on="batter", how="outer")

    return rolling


def build_pitch_type_splits(bbe: pd.DataFrame) -> pd.DataFrame:
    if "pitch_type" not in bbe.columns:
        return pd.DataFrame(columns=["batter"])

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    bbe = bbe.copy()
    bbe["total_bases"] = bbe["events"].astype("string").str.lower().map(tb_map).fillna(0)
    bbe["is_hit"] = bbe["events"].astype("string").str.lower().isin(
        {"single", "double", "triple", "home_run"}
    )
    bbe["pitch_type"] = bbe["pitch_type"].astype("string").str.upper().str.strip()

    pitch_counts = (
        bbe.groupby(["batter", "pitch_type"])
        .size()
        .reset_index(name="cnt")
    )
    valid_combos = pitch_counts[pitch_counts["cnt"] >= 5].set_index(
        ["batter", "pitch_type"]
    ).index

    results_individual = []
    all_pitch_types = bbe["pitch_type"].dropna().unique()

    for pt in sorted(all_pitch_types):
        if pt in ("", "nan", "None", "OTHER", "PO", "UN", "EP"):
            continue

        sub = bbe[bbe["pitch_type"] == pt].copy()
        if sub.empty:
            continue

        grp = (
            sub.groupby("batter", dropna=False)
            .agg(
                bbe_count=("launch_speed", "size"),
                hr_count=("is_hr", "sum"),
                fb_count=("is_fly_ball", "sum"),
                pull_count=("is_pull", "sum"),
                barrel_count=("is_barrel", "sum"),
                total_bases=("total_bases", "sum"),
                hits=("is_hit", "sum"),
                pa_count=("launch_speed", "size"),
            )
            .reset_index()
        )

        grp = grp[
            grp.apply(lambda r: (r["batter"], pt) in valid_combos, axis=1)
        ].copy()

        if grp.empty:
            continue

        grp[f"iso_vs_{pt}"] = (
            (grp["total_bases"] - grp["hits"]) / grp["pa_count"].replace(0, pd.NA)
        ).round(3)
        grp[f"hr_rate_vs_{pt}"] = (grp["hr_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"fb_rate_vs_{pt}"] = (grp["fb_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"pull_rate_vs_{pt}"] = (grp["pull_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"barrel_pct_vs_{pt}"] = (grp["barrel_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"bbe_vs_{pt}"] = grp["bbe_count"]

        keep_cols = [
            "batter",
            f"iso_vs_{pt}",
            f"hr_rate_vs_{pt}",
            f"fb_rate_vs_{pt}",
            f"pull_rate_vs_{pt}",
            f"barrel_pct_vs_{pt}",
            f"bbe_vs_{pt}",
        ]
        results_individual.append(grp[keep_cols])

    results_group = []
    for group in ["fastball", "breaking", "offspeed", "knuckleball"]:
        sub = bbe[bbe["pitch_group"] == group].copy()
        if sub.empty:
            continue

        grp = (
            sub.groupby("batter", dropna=False)
            .agg(
                bbe_count=("launch_speed", "size"),
                hr_count=("is_hr", "sum"),
                fb_count=("is_fly_ball", "sum"),
                pull_count=("is_pull", "sum"),
                barrel_count=("is_barrel", "sum"),
                total_bases=("total_bases", "sum"),
                hits=("is_hit", "sum"),
                pa_count=("launch_speed", "size"),
            )
            .reset_index()
        )

        grp = grp[grp["bbe_count"] >= 5].copy()
        if grp.empty:
            continue

        grp[f"iso_vs_{group}"] = (
            (grp["total_bases"] - grp["hits"]) / grp["pa_count"].replace(0, pd.NA)
        ).round(3)
        grp[f"hr_rate_vs_{group}"] = (grp["hr_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"fb_rate_vs_{group}"] = (grp["fb_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"pull_rate_vs_{group}"] = (grp["pull_count"] / grp["bbe_count"] * 100).round(2)
        grp[f"barrel_pct_vs_{group}"] = (grp["barrel_count"] / grp["bbe_count"] * 100).round(2)

        keep_cols = [
            "batter",
            f"iso_vs_{group}",
            f"hr_rate_vs_{group}",
            f"fb_rate_vs_{group}",
            f"pull_rate_vs_{group}",
            f"barrel_pct_vs_{group}",
        ]
        results_group.append(grp[keep_cols])

    if not results_individual and not results_group:
        return pd.DataFrame(columns=["batter"])

    if results_individual:
        combined_individual = results_individual[0]
        for r in results_individual[1:]:
            combined_individual = combined_individual.merge(r, on="batter", how="outer")
    else:
        combined_individual = pd.DataFrame(columns=["batter"])

    if results_group:
        combined_group = results_group[0]
        for r in results_group[1:]:
            combined_group = combined_group.merge(r, on="batter", how="outer")
    else:
        combined_group = pd.DataFrame(columns=["batter"])

    if combined_individual.empty:
        return combined_group
    if combined_group.empty:
        return combined_individual

    return combined_individual.merge(combined_group, on="batter", how="outer")


def build_batter_full(df: pd.DataFrame, today_teams: Set[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["team"] = infer_batting_team(df)
    df["game_date"] = pd.to_datetime(df["game_date"])

    if today_teams:
        today_df = df[df["team"].isin(today_teams)].copy()
    else:
        print("No today's teams found — showing all teams from dataset.")
        today_df = df.copy()

    if today_df.empty:
        return pd.DataFrame()

    today_df = add_statcast_flags(today_df)
    bbe = filter_bbe(today_df)

    season_stats = build_season_stats(bbe, today_df)
    platoon_splits = build_platoon_splits(bbe)
    rolling_stats = build_rolling_stats(bbe)
    pitch_type_splits = build_pitch_type_splits(bbe)

    combined = (
        season_stats
        .merge(platoon_splits, on="batter", how="left")
        .merge(rolling_stats, on="batter", how="left")
        .merge(pitch_type_splits, on="batter", how="left")
    )

    team_map = (
        today_df.sort_values("game_date")
        .drop_duplicates(subset=["batter"], keep="last")
        [["batter", "team"]]
    )
    combined = combined.merge(team_map, on="batter", how="left")

    combined["batter"] = combined["batter"].astype("Int64")
    name_map = lookup_player_names(combined["batter"].dropna().astype(int).tolist())
    combined["player_name"] = combined["batter"].map(
        lambda x: name_map.get(int(x), "") if pd.notna(x) else ""
    )

    combined = combined[
        (combined["season_bbe"] >= 10) &
        (combined["player_name"] != "")
    ].copy()

    combined = combined.rename(columns={"batter": "player_id"})

    id_cols = ["player_name", "player_id", "team", "batter_hand"]
    other_cols = [c for c in combined.columns if c not in id_cols]
    combined = combined[id_cols + other_cols]

    combined = combined.sort_values(
        by=["hr_per_pa", "barrel_pct_7d", "avg_ev_7d"],
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

    today_teams = get_today_team_abbrs()
    if not today_teams:
        print("WARNING: No games found for today — defaulting to all teams.")

    raw_df = get_season_statcast()
    print(f"Pulled {len(raw_df):,} Statcast rows for 2026 season")

    batter_df = build_batter_full(raw_df, today_teams)
    print(f"Built {len(batter_df)} batter rows")

    write_dataframe_to_sheet(gc, sheet_id, "Batter_Statcast_2026", batter_df)
    print("Written to Batter_Statcast_2026")


if __name__ == "__main__":
    main()
