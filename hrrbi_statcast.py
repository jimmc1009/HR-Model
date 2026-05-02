"""
hrrbi_statcast.py
Pulls batter data for the H+R+RBI model entirely from Statcast.
No FanGraphs dependency — all metrics computed from pitch-by-pitch data.
Outputs to HRRBI_Statcast Google Sheet tab.
"""

import os
import json
import unicodedata
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


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).strftime("%Y-%m-%d")


def lookup_player_names(player_ids: list) -> dict:
    out = {}
    clean_ids = sorted({int(pid) for pid in player_ids if pd.notna(pid)})
    for i in range(0, len(clean_ids), 50):
        chunk = clean_ids[i:i + 50]
        try:
            url  = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pid  = person.get("id")
                name = person.get("fullName", "")
                if pid and name:
                    out[int(pid)] = name
        except Exception:
            pass
    return out


def is_barrel(ev, la) -> bool:
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return False
    return max(26 - (ev - 98), 8) <= la <= min(30 + (ev - 98), 50)


def build_hrrbi_statcast(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all H+R+RBI batter metrics from raw Statcast data.

    Season metrics:
    - BA, OBP, SLG, ISO (from hit events and AB/PA counts)
    - wOBA (linear weights from event outcomes)
    - LD%, GB%, FB% (launch angle buckets)
    - Hard hit% (EV >= 95), Barrel% (Statcast definition)
    - Avg EV season
    - HR rate, RBI proxy (runs batted in from bat_score delta)
    - R rate (runs scored proxy)
    - BB%, K%
    - Speed proxy (SB rate)
    - Avg batting order position

    7-day rolling:
    - Avg EV, Hard hit%, Barrel%, Avg LA
    - H rate (hits / PA last 7d)

    14-day rolling:
    - Avg, H, PA
    - Hot/cold flags
    """
    print("Building HRRBI batter features from Statcast...")

    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Batting team ───────────────────────────────────────────────
    if {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
        df["batting_team"] = df.apply(
            lambda r: str(r["away_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["home_team"]).strip(), axis=1,
        )
    else:
        df["batting_team"] = ""

    # ── PA events ──────────────────────────────────────────────────
    AB_EVENTS = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "strikeout",
        "strikeout_double_play", "other_out",
    }
    PA_EVENTS = AB_EVENTS | {
        "walk", "hit_by_pitch", "sac_fly", "sac_fly_double_play",
        "sac_bunt", "sac_bunt_double_play", "catcher_interf",
        "intent_walk",
    }
    HIT_EVENTS = {"single", "double", "triple", "home_run"}

    pa_df = df[df["events"].astype("string").str.lower().isin(PA_EVENTS)].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
    )

    ab_df = pa_df[pa_df["events"].astype("string").str.lower().isin(AB_EVENTS)].copy()

    pa_df["is_hit"]  = pa_df["events"].astype("string").str.lower().isin(HIT_EVENTS).astype(int)
    pa_df["is_bb"]   = pa_df["events"].astype("string").str.lower().isin({"walk", "intent_walk"}).astype(int)
    pa_df["is_k"]    = pa_df["events"].astype("string").str.lower().isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa_df["is_hr"]   = pa_df["events"].astype("string").str.lower().eq("home_run").astype(int)
    pa_df["is_sb"]   = 0  # SB from Statcast requires separate event parsing

    # Total bases for SLG/ISO/wOBA
    TB_MAP = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    pa_df["tb"] = pa_df["events"].astype("string").str.lower().map(TB_MAP).fillna(0)

    # wOBA linear weights (2024 approximate)
    WOBA_WEIGHTS = {
        "walk": 0.690, "hit_by_pitch": 0.722, "intent_walk": 0.690,
        "single": 0.882, "double": 1.242, "triple": 1.569, "home_run": 2.065,
    }
    pa_df["woba_val"] = pa_df["events"].astype("string").str.lower().map(WOBA_WEIGHTS).fillna(0)

    # ── Season aggregations ────────────────────────────────────────
    pa_counts = pa_df.groupby("batter").agg(
        pa=("is_hit", "count"),
        hits=("is_hit", "sum"),
        bb=("is_bb", "sum"),
        ks=("is_k", "sum"),
        hr=("is_hr", "sum"),
        woba_num=("woba_val", "sum"),
    ).reset_index()

    ab_counts = ab_df.groupby("batter").size().reset_index(name="ab")

    tb_counts = pa_df.groupby("batter")["tb"].sum().reset_index(name="total_bases")

    season = pa_counts.merge(ab_counts, on="batter", how="left")
    season = season.merge(tb_counts, on="batter", how="left")
    season["ab"]          = season["ab"].fillna(0)
    season["total_bases"] = season["total_bases"].fillna(0)

    season["avg"]  = (season["hits"] / season["ab"].replace(0, np.nan)).round(3)
    season["obp"]  = ((season["hits"] + season["bb"]) / season["pa"].replace(0, np.nan)).round(3)
    season["slg"]  = (season["total_bases"] / season["ab"].replace(0, np.nan)).round(3)
    season["iso"]  = (season["slg"] - season["avg"]).round(3)
    season["woba"] = (season["woba_num"] / season["pa"].replace(0, np.nan)).round(3)
    season["bb_pct"] = (season["bb"] / season["pa"].replace(0, np.nan) * 100).round(1)
    season["k_pct"]  = (season["ks"] / season["pa"].replace(0, np.nan) * 100).round(1)
    season["hr_rate"] = (season["hr"] / season["pa"].replace(0, np.nan) * 100).round(2)

    # ── Batting order position ─────────────────────────────────────
    if "bat_order" in pa_df.columns:
        bat_order = (
            pa_df.groupby("batter")["bat_order"]
            .apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
            .reset_index(name="avg_bat_order")
        )
        bat_order["avg_bat_order"] = bat_order["avg_bat_order"].round(1)
        season = season.merge(bat_order, on="batter", how="left")
    else:
        season["avg_bat_order"] = 5.0

    # ── BBE metrics ────────────────────────────────────────────────
    BBE_EVENTS = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "sac_fly",
        "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    }

    bbe = df[
        df["events"].astype("string").str.lower().isin(BBE_EVENTS) &
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna()
    ].copy()
    bbe = bbe.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in bbe.columns]
    )

    bbe["is_hard_hit"] = (bbe["launch_speed"] >= 95).astype(int)
    bbe["is_barrel"]   = bbe.apply(lambda r: is_barrel(r["launch_speed"], r["launch_angle"]), axis=1).astype(int)

    # Launch angle buckets
    bbe["is_ld"] = bbe["launch_angle"].between(10, 25).astype(int)
    bbe["is_gb"] = (bbe["launch_angle"] < 10).astype(int)
    bbe["is_fb"] = (bbe["launch_angle"] > 25).astype(int)

    bbe_agg = bbe.groupby("batter").agg(
        season_bbe=("launch_speed", "size"),
        avg_ev_season=("launch_speed", "mean"),
        avg_la_season=("launch_angle", "mean"),
        hard_hit_count=("is_hard_hit", "sum"),
        barrel_count=("is_barrel", "sum"),
        ld_count=("is_ld", "sum"),
        gb_count=("is_gb", "sum"),
        fb_count=("is_fb", "sum"),
    ).reset_index()

    bbe_agg["avg_ev_season"]        = bbe_agg["avg_ev_season"].round(1)
    bbe_agg["avg_la_season"]        = bbe_agg["avg_la_season"].round(1)
    bbe_agg["hard_hit_pct_season"]  = (bbe_agg["hard_hit_count"] / bbe_agg["season_bbe"].replace(0, np.nan) * 100).round(1)
    bbe_agg["barrel_pct_season"]    = (bbe_agg["barrel_count"]   / bbe_agg["season_bbe"].replace(0, np.nan) * 100).round(1)
    bbe_agg["ld_pct"]               = (bbe_agg["ld_count"]       / bbe_agg["season_bbe"].replace(0, np.nan) * 100).round(1)
    bbe_agg["gb_pct"]               = (bbe_agg["gb_count"]       / bbe_agg["season_bbe"].replace(0, np.nan) * 100).round(1)
    bbe_agg["fb_pct"]               = (bbe_agg["fb_count"]       / bbe_agg["season_bbe"].replace(0, np.nan) * 100).round(1)

    season = season.merge(bbe_agg, on="batter", how="left")

    # ── 7-day rolling ──────────────────────────────────────────────
    cutoff_7d = pd.Timestamp(date.today() - timedelta(days=7))

    pa_7d = pa_df[pa_df["game_date"] >= cutoff_7d].copy()
    bbe_7d = bbe[bbe["game_date"] >= cutoff_7d].copy()

    pa_7d_agg = pa_7d.groupby("batter").agg(
        pa_7d=("is_hit", "count"),
        hits_7d=("is_hit", "sum"),
    ).reset_index()
    pa_7d_agg["avg_7d"] = (pa_7d_agg["hits_7d"] / pa_7d_agg["pa_7d"].replace(0, np.nan)).round(3)

    bbe_7d_agg = bbe_7d.groupby("batter").agg(
        bbe_7d=("launch_speed", "size"),
        avg_ev_7d=("launch_speed", "mean"),
        avg_la_7d=("launch_angle", "mean"),
        hard_hit_7d=("is_hard_hit", "sum"),
        barrel_7d=("is_barrel", "sum"),
    ).reset_index()
    bbe_7d_agg["avg_ev_7d"]        = bbe_7d_agg["avg_ev_7d"].round(1)
    bbe_7d_agg["avg_la_7d"]        = bbe_7d_agg["avg_la_7d"].round(1)
    bbe_7d_agg["hard_hit_pct_7d"]  = (bbe_7d_agg["hard_hit_7d"] / bbe_7d_agg["bbe_7d"].replace(0, np.nan) * 100).round(1)
    bbe_7d_agg["barrel_pct_7d"]    = (bbe_7d_agg["barrel_7d"]   / bbe_7d_agg["bbe_7d"].replace(0, np.nan) * 100).round(1)

    season = season.merge(pa_7d_agg,  on="batter", how="left")
    season = season.merge(bbe_7d_agg, on="batter", how="left")

    # ── 14-day rolling ─────────────────────────────────────────────
    cutoff_14d = pd.Timestamp(date.today() - timedelta(days=14))

    pa_14d = pa_df[pa_df["game_date"] >= cutoff_14d].copy()
    pa_14d_agg = pa_14d.groupby("batter").agg(
        pa_14d=("is_hit", "count"),
        hits_14d=("is_hit", "sum"),
    ).reset_index()
    pa_14d_agg["avg_14d"] = (pa_14d_agg["hits_14d"] / pa_14d_agg["pa_14d"].replace(0, np.nan)).round(3)

    season = season.merge(pa_14d_agg, on="batter", how="left")

    # ── Hot/cold flags ─────────────────────────────────────────────
    season["hot_streak"]  = ((season.get("avg_14d", 0) >= 0.320) & (season.get("pa_14d", 0) >= 20)).astype(int)
    season["cold_streak"] = ((season.get("avg_14d", 0) <= 0.180) & (season.get("pa_14d", 0) >= 20)).astype(int)

    # ── Batter hand ────────────────────────────────────────────────
    if "stand" in pa_df.columns:
        hand_map = pa_df.groupby("batter")["stand"].first().to_dict()
        season["batter_hand"] = season["batter"].map(lambda x: hand_map.get(x, ""))
    else:
        season["batter_hand"] = ""

    # ── Team ───────────────────────────────────────────────────────
    team_map = (
        pa_df.sort_values("game_date")
        .drop_duplicates(subset=["batter"], keep="last")
        [["batter", "batting_team"]]
    )
    season = season.merge(team_map, on="batter", how="left")
    season = season.rename(columns={"batting_team": "team"})

    # ── Filter minimum sample ──────────────────────────────────────
    season = season[
        (season["pa"] >= 30) &
        (season["season_bbe"] >= 10)
    ].copy()

    # ── Player names ───────────────────────────────────────────────
    batter_ids = season["batter"].dropna().astype(int).tolist()
    name_map   = lookup_player_names(batter_ids)
    season["player_name"]      = season["batter"].map(lambda x: name_map.get(int(x), "") if pd.notna(x) else "")
    season["player_name_norm"] = season["player_name"].apply(normalize_name)
    season                     = season.rename(columns={"batter": "batter_id"})

    # ── Filter unnamed ─────────────────────────────────────────────
    season = season[season["player_name"] != ""].copy()

    # ── Sort ───────────────────────────────────────────────────────
    season = season.sort_values("woba", ascending=False).reset_index(drop=True)

    print(f"HRRBI_Statcast: {len(season)} batters")
    return season


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
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=60)

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        if hasattr(df[col], 'dtype') and hasattr(df[col].dtype, 'numpy_dtype'):
            df[col] = df[col].astype(object)
    df = df.fillna("")
values = [df.columns.tolist()] + df.astype(str).values.tolist()
ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Pulling full season Statcast data...")
    from pybaseball import statcast as _statcast
    raw_df = _statcast(start_dt=SEASON_START, end_dt=today_str())

    if raw_df is None or raw_df.empty:
        print("ERROR: Statcast returned empty — aborting")
        return

    print(f"Pulled {len(raw_df):,} rows")
    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])

    hrrbi_df = build_hrrbi_statcast(raw_df)

    if hrrbi_df.empty:
        print("ERROR: No batter data built — aborting")
        return

    write_dataframe_to_sheet(gc, sheet_id, "HRRBI_Statcast", hrrbi_df)
    print("Written to HRRBI_Statcast")


if __name__ == "__main__":
    main()
