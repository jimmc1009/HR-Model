import os
import json
from datetime import date, timedelta
from typing import Dict, List, Set

import pandas as pd
import gspread
import requests
from google.oauth2.service_account import Credentials
from pybaseball import statcast


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_today_team_abbrs() -> Set[str]:
    """
    Pull today's MLB schedule and return only the teams on today's slate.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}&hydrate=probablePitcher"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    teams: Set[str] = set()
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away_abbr = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation")
            home_abbr = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation")
            if away_abbr:
                teams.add(str(away_abbr).strip())
            if home_abbr:
                teams.add(str(home_abbr).strip())
    return teams


def get_last_10_days_statcast() -> pd.DataFrame:
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=10)

    df = statcast(
        start_dt=start_dt.strftime("%Y-%m-%d"),
        end_dt=end_dt.strftime("%Y-%m-%d"),
    )

    if df is None or df.empty:
        return pd.DataFrame()

    return df


def lookup_player_names(player_ids: List[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    clean_ids = sorted({int(pid) for pid in player_ids if pd.notna(pid)})

    if not clean_ids:
        return out

    chunk_size = 50
    for i in range(0, len(clean_ids), chunk_size):
        chunk = clean_ids[i:i + chunk_size]
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for person in data.get("people", []):
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


def build_batter_10d(df: pd.DataFrame, today_teams: Set[str]) -> pd.DataFrame:
    columns = [
        "player_name",
        "player_id",
        "team",
        "bbe",
        "avg_ev",
        "avg_launch_angle",
        "barrel_bbe_pct",
        "hard_hit_pct",
        "sweet_spot_pct",
    ]

    if df.empty:
        return pd.DataFrame(columns=columns)

    # Only real batted-ball events
    bbe = df[
    (df["launch_speed"].notna()) &
    (df["launch_speed"] > 50) &   # removes junk weak readings
    (df["launch_speed"] < 120)    # removes outliers
].copy()

    # Deduplicate at the event level
    dedupe_cols = [c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter"] if c in bbe.columns]
    if dedupe_cols:
        bbe = bbe.drop_duplicates(subset=dedupe_cols)

    # Infer batting team, then keep only today's teams
    bbe["team"] = infer_batting_team(bbe)
    if today_teams:
        bbe = bbe[bbe["team"].isin(today_teams)].copy()

    # Basic Statcast flags
    bbe["is_barrel"] = bbe["launch_speed_angle"].astype("string").eq("6")
    bbe["is_hard_hit"] = bbe["launch_speed"] >= 95
    bbe["is_sweet_spot"] = bbe["launch_angle"].between(8, 32, inclusive="both")

    grouped = (
        bbe.groupby(["batter", "team"], dropna=False)
        .agg(
            bbe=("launch_speed", "size"),
            avg_ev=("launch_speed", "mean"),
            avg_launch_angle=("launch_angle", "mean"),
            barrels=("is_barrel", "sum"),
            hard_hits=("is_hard_hit", "sum"),
            sweet_spots=("is_sweet_spot", "sum"),
        )
        .reset_index()
    )

    grouped["barrel_bbe_pct"] = grouped["barrels"] / grouped["bbe"] * 100
    grouped["hard_hit_pct"] = grouped["hard_hits"] / grouped["bbe"] * 100
    grouped["sweet_spot_pct"] = grouped["sweet_spots"] / grouped["bbe"] * 100

    grouped["batter"] = grouped["batter"].astype("Int64")
    name_map = lookup_player_names(grouped["batter"].dropna().astype(int).tolist())
    grouped["player_name"] = grouped["batter"].map(lambda x: name_map.get(int(x), "") if pd.notna(x) else "")

    result = grouped.rename(columns={"batter": "player_id"})[
        [
            "player_name",
            "player_id",
            "team",
            "bbe",
            "avg_ev",
            "avg_launch_angle",
            "barrel_bbe_pct",
            "hard_hit_pct",
            "sweet_spot_pct",
        ]
    ].copy()

    numeric_cols = [
        "avg_ev",
        "avg_launch_angle",
        "barrel_bbe_pct",
        "hard_hit_pct",
        "sweet_spot_pct",
    ]
    result[numeric_cols] = result[numeric_cols].round(2)

    # Keep believable samples
    result = result[(result["bbe"] >= 5) & (result["player_name"] != "")]
    result = result.sort_values(
        by=["barrel_bbe_pct", "avg_ev", "avg_launch_angle"],
        ascending=[False, False, False],
    )

    return result


def write_dataframe_to_sheet(
    gc: gspread.Client,
    sheet_id: str,
    worksheet_name: str,
    df: pd.DataFrame
) -> None:
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=30)

    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]

    gc = get_gspread_client()
    today_teams = get_today_team_abbrs()
    raw_df = get_last_10_days_statcast()
    batter_df = build_batter_10d(raw_df, today_teams)

    write_dataframe_to_sheet(gc, sheet_id, "Batter_Statcast_10D", batter_df)

    print(f"Wrote {len(batter_df)} batter rows to Batter_Statcast_10D")


if __name__ == "__main__":
    main()
