import os
import json
from datetime import date, timedelta

import pandas as pd
import gspread
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


def build_batter_10d(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "player_name",
            "player_id",
            "team",
            "bbe",
            "avg_ev",
            "barrel_bbe_pct",
            "hard_hit_pct",
            "sweet_spot_pct",
        ])

    # Only keep rows with tracked batted-ball data
    bbe = df[df["launch_speed"].notna()].copy()

    # Deduplicate likely repeated event rows
    dedupe_cols = [c for c in ["game_date", "game_pk", "at_bat_number", "pitch_number", "batter"] if c in bbe.columns]
    if dedupe_cols:
        bbe = bbe.drop_duplicates(subset=dedupe_cols)

    # Infer batting team
    if "batting_team" in bbe.columns:
        bbe["team"] = bbe["batting_team"]
    elif "inning_topbot" in bbe.columns and "home_team" in bbe.columns and "away_team" in bbe.columns:
        bbe["team"] = bbe.apply(
            lambda r: r["away_team"] if str(r["inning_topbot"]).lower().startswith("top") else r["home_team"],
            axis=1
        )
    else:
        bbe["team"] = ""

    # Build Statcast quality-of-contact flags
    bbe["is_barrel"] = bbe["launch_speed_angle"].astype("string").eq("6")
    bbe["is_hard_hit"] = bbe["launch_speed"] >= 95
    bbe["is_sweet_spot"] = bbe["launch_angle"].between(8, 32, inclusive="both")

    grouped = (
        bbe.groupby(["player_name", "batter", "team"], dropna=False)
        .agg(
            bbe=("launch_speed", "size"),
            avg_ev=("launch_speed", "mean"),
            barrels=("is_barrel", "sum"),
            hard_hits=("is_hard_hit", "sum"),
            sweet_spots=("is_sweet_spot", "sum"),
        )
        .reset_index()
    )

    grouped["barrel_bbe_pct"] = grouped["barrels"] / grouped["bbe"] * 100
    grouped["hard_hit_pct"] = grouped["hard_hits"] / grouped["bbe"] * 100
    grouped["sweet_spot_pct"] = grouped["sweet_spots"] / grouped["bbe"] * 100

    result = grouped.rename(columns={"batter": "player_id"})[
        [
            "player_name",
            "player_id",
            "team",
            "bbe",
            "avg_ev",
            "barrel_bbe_pct",
            "hard_hit_pct",
            "sweet_spot_pct",
        ]
    ].copy()

    result["avg_ev"] = result["avg_ev"].round(2)
    result["barrel_bbe_pct"] = result["barrel_bbe_pct"].round(2)
    result["hard_hit_pct"] = result["hard_hit_pct"].round(2)
    result["sweet_spot_pct"] = result["sweet_spot_pct"].round(2)

    result = result[result["bbe"] >= 5].sort_values(
        by=["barrel_bbe_pct", "avg_ev"], ascending=False
    )

    return result


def write_dataframe_to_sheet(gc: gspread.Client, sheet_id: str, worksheet_name: str, df: pd.DataFrame) -> None:
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=20)

    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]

    gc = get_gspread_client()
    raw_df = get_last_10_days_statcast()
    batter_df = build_batter_10d(raw_df)

    write_dataframe_to_sheet(gc, sheet_id, "Batter_Statcast_10D", batter_df)

    print(f"Wrote {len(batter_df)} batter rows to Batter_Statcast_10D")


if __name__ == "__main__":
    main()
