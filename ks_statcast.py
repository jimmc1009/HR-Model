"""
ks_statcast.py
Pulls pitcher data for the Strikeout model entirely from Statcast.
No FanGraphs dependency — all metrics computed from pitch-by-pitch data.
Outputs to KS_Statcast and Team_K_Rates Google Sheet tabs.
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
MIN_GS       = 3
MIN_IP       = 15


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


def build_ks_statcast(raw_df: pd.DataFrame) -> tuple:
    """
    Computes all K model pitcher metrics from raw Statcast data.

    Season metrics:
    - K%, BB%, K-BB%, K/9 (from PA events)
    - SwStr% (swinging strikes / total pitches)
    - Chase rate / O-Swing% (swings on pitches outside zone)
    - F-Strike% (first pitch called/swinging strike rate)
    - Fastball velocity (avg release_speed on FF/SI/FC)
    - Avg IP per start (outs per game / 3)
    - Opener risk flag (avg IP < 4.0)
    - Hard hit% allowed, Barrel% allowed (BBE metrics)
    - WHIP proxy (H + BB per inning)

    21-day rolling (last 3 starts):
    - SwStr% last 21d
    - K per start last 3 starts
    - Avg velo last 21d
    - Avg IP last 3 starts
    - SwStr trend (21d vs season)
    - Velo trend (21d vs season)

    Also returns team_k_df: team-level strikeout rates for opposing lineup scoring.
    """
    print("Building KS pitcher features from Statcast...")

    df = raw_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # ── Pitching team ──────────────────────────────────────────────
    if {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
        df["pitching_team"] = df.apply(
            lambda r: str(r["home_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["away_team"]).strip(), axis=1,
        )
        df["batting_team"] = df.apply(
            lambda r: str(r["away_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["home_team"]).strip(), axis=1,
        )
    else:
        df["pitching_team"] = ""
        df["batting_team"]  = ""

    # ── Out events for IP calculation ──────────────────────────────
    OUT_EVENTS = {
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "fielders_choice_out", "force_out",
        "sac_fly", "sac_fly_double_play", "sac_bunt",
        "sac_bunt_double_play", "other_out", "strikeout",
        "strikeout_double_play",
    }

    PA_EVENTS = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "strikeout",
        "strikeout_double_play", "other_out",
        "walk", "hit_by_pitch", "sac_fly", "sac_fly_double_play",
        "sac_bunt", "sac_bunt_double_play", "catcher_interf", "intent_walk",
    }

    HIT_EVENTS = {"single", "double", "triple", "home_run"}

    # ── PA-level data ──────────────────────────────────────────────
    pa_df = df[df["events"].astype("string").str.lower().isin(PA_EVENTS)].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "pitcher"] if c in pa_df.columns]
    )

    pa_df["is_k"]    = pa_df["events"].astype("string").str.lower().isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa_df["is_bb"]   = pa_df["events"].astype("string").str.lower().isin({"walk", "intent_walk"}).astype(int)
    pa_df["is_hit"]  = pa_df["events"].astype("string").str.lower().isin(HIT_EVENTS).astype(int)
    pa_df["is_out"]  = pa_df["events"].astype("string").str.lower().isin(OUT_EVENTS).astype(int)

    # ── Season PA aggregation ──────────────────────────────────────
    pa_agg = pa_df.groupby("pitcher").agg(
        batters_faced=("is_k", "count"),
        k_season=("is_k", "sum"),
        bb_season=("is_bb", "sum"),
        hits_allowed=("is_hit", "sum"),
        outs_season=("is_out", "sum"),
        games=("game_pk", "nunique"),
    ).reset_index()

    pa_agg["ip"]         = (pa_agg["outs_season"] / 3).round(2)
    pa_agg["k_pct_season"] = (pa_agg["k_season"] / pa_agg["batters_faced"].replace(0, np.nan) * 100).round(1)
    pa_agg["bb_pct_season"] = (pa_agg["bb_season"] / pa_agg["batters_faced"].replace(0, np.nan) * 100).round(1)
    pa_agg["k_minus_bb"]   = (pa_agg["k_pct_season"] - pa_agg["bb_pct_season"]).round(1)
    pa_agg["k_per_9"]      = (pa_agg["k_season"] / pa_agg["ip"].replace(0, np.nan) * 9).round(1)
    pa_agg["whip_proxy"]   = ((pa_agg["hits_allowed"] + pa_agg["bb_season"]) / pa_agg["ip"].replace(0, np.nan)).round(2)

    # ── Games started and avg IP per start ─────────────────────────
    # Identify starts: game where pitcher recorded most outs (primary pitcher)
    outs_by_game = (
        pa_df[pa_df["is_out"] == 1]
        .groupby(["pitcher", "game_pk"])
        .agg(outs_in_game=("is_out", "sum"))
        .reset_index()
    )

    # Pitcher with most outs in a game = starter
    game_starter = (
        outs_by_game.sort_values("outs_in_game", ascending=False)
        .drop_duplicates(subset=["game_pk"], keep="first")
        .rename(columns={"pitcher": "starter_id"})
    )

    pitcher_starts = (
        outs_by_game.merge(
            game_starter[["game_pk", "starter_id"]],
            on="game_pk",
        )
    )
    pitcher_starts = pitcher_starts[
        pitcher_starts["pitcher"] == pitcher_starts["starter_id"]
    ].copy()

    starts_agg = pitcher_starts.groupby("pitcher").agg(
        games_started=("game_pk", "count"),
        total_outs_started=("outs_in_game", "sum"),
    ).reset_index()
    starts_agg["avg_ip_per_start"] = (
        starts_agg["total_outs_started"] / 3 / starts_agg["games_started"].replace(0, np.nan)
    ).round(2)
    starts_agg["opener_risk"] = (starts_agg["avg_ip_per_start"] < 4.0).astype(int)

    pa_agg = pa_agg.merge(starts_agg, on="pitcher", how="left")
    pa_agg["games_started"]    = pa_agg["games_started"].fillna(0)
    pa_agg["avg_ip_per_start"] = pa_agg["avg_ip_per_start"].fillna(0)
    pa_agg["opener_risk"]      = pa_agg["opener_risk"].fillna(1).astype(int)

    # ── Pitch-level data ───────────────────────────────────────────
    pitch_df = df.copy()
    pitch_df["is_pitch"]  = 1
    pitch_df["is_swstr"]  = pitch_df["description"].astype("string").isin(
        ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
    ).astype(int)
    pitch_df["is_called_strike"] = pitch_df["description"].astype("string").eq("called_strike").astype(int)
    pitch_df["is_swing"]  = pitch_df["description"].astype("string").isin([
        "swinging_strike", "swinging_strike_blocked", "foul_tip",
        "foul", "hit_into_play", "hit_into_play_no_out",
        "hit_into_play_score", "missed_bunt", "foul_bunt",
    ]).astype(int)

    # Out of zone pitches (zones 11-14 = ball zones)
    if "zone" in pitch_df.columns:
        pitch_df["is_out_zone"] = pitch_df["zone"].apply(
            lambda z: int(str(z).strip() in ["11", "12", "13", "14"]) if pd.notna(z) else 0
        )
    else:
        pitch_df["is_out_zone"] = 0

    pitch_df["is_out_zone_swing"] = (pitch_df["is_out_zone"] & pitch_df["is_swing"]).astype(int)

    # First pitch of each PA
    pitch_df["pitch_number"] = pd.to_numeric(pitch_df.get("pitch_number", 1), errors="coerce").fillna(1)
    first_pitch = pitch_df[pitch_df["pitch_number"] == 1].copy()
    first_pitch["is_first_strike"] = first_pitch["description"].astype("string").isin(
        ["called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"]
    ).astype(int)

    # ── Season pitch aggregation ───────────────────────────────────
    pitch_agg = pitch_df.groupby("pitcher").agg(
        total_pitches=("is_pitch", "sum"),
        total_swstr=("is_swstr", "sum"),
        total_out_zone=("is_out_zone", "sum"),
        total_out_zone_swing=("is_out_zone_swing", "sum"),
    ).reset_index()

    pitch_agg["swstr_pct"]  = (pitch_agg["total_swstr"] / pitch_agg["total_pitches"].replace(0, np.nan) * 100).round(1)
    pitch_agg["chase_rate"] = (pitch_agg["total_out_zone_swing"] / pitch_agg["total_out_zone"].replace(0, np.nan) * 100).round(1)

    fps_agg = first_pitch.groupby("pitcher").agg(
        first_pitch_total=("is_pitch", "sum"),
        first_pitch_strikes=("is_first_strike", "sum"),
    ).reset_index()
    fps_agg["first_pitch_strike_pct"] = (fps_agg["first_pitch_strikes"] / fps_agg["first_pitch_total"].replace(0, np.nan) * 100).round(1)

    # ── Fastball velocity ──────────────────────────────────────────
    ff = pitch_df[
        pitch_df["pitch_type"].astype("string").isin(["FF", "SI", "FC"])
    ].copy()
    velo_agg = ff.groupby("pitcher").agg(
        fastball_velo=("release_speed", "mean"),
    ).reset_index()
    velo_agg["fastball_velo"] = velo_agg["fastball_velo"].round(1)

    # ── BBE metrics allowed ────────────────────────────────────────
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
        subset=[c for c in ["game_pk", "at_bat_number", "pitcher"] if c in bbe.columns]
    )

    def is_barrel(ev, la) -> bool:
        if pd.isna(ev) or pd.isna(la) or ev < 98:
            return False
        return max(26 - (ev - 98), 8) <= la <= min(30 + (ev - 98), 50)

    bbe["is_hard_hit"] = (bbe["launch_speed"] >= 95).astype(int)
    bbe["is_barrel"]   = bbe.apply(lambda r: is_barrel(r["launch_speed"], r["launch_angle"]), axis=1).astype(int)

    bbe_agg = bbe.groupby("pitcher").agg(
        season_bbe_allowed=("launch_speed", "size"),
        hard_hit_allowed=("is_hard_hit", "sum"),
        barrel_allowed=("is_barrel", "sum"),
    ).reset_index()

    bbe_agg["hard_hit_pct_against"]      = (bbe_agg["hard_hit_allowed"] / bbe_agg["season_bbe_allowed"].replace(0, np.nan) * 100).round(1)
    bbe_agg["season_barrel_pct_allowed"] = (bbe_agg["barrel_allowed"]   / bbe_agg["season_bbe_allowed"].replace(0, np.nan) * 100).round(1)

    # ── Pitcher hand ───────────────────────────────────────────────
    hand_map = {}
    if "p_throws" in pa_df.columns:
        hand_map = pa_df.groupby("pitcher")["p_throws"].first().to_dict()

    # ── Pitcher team ───────────────────────────────────────────────
    team_map = (
        pa_df.sort_values("game_date")
        .drop_duplicates(subset=["pitcher"], keep="last")
        [["pitcher", "pitching_team"]]
    )

    # ── 21-day rolling ─────────────────────────────────────────────
    cutoff_21d = pd.Timestamp(date.today() - timedelta(days=21))

    pa_21d    = pa_df[pa_df["game_date"] >= cutoff_21d].copy()
    pitch_21d = pitch_df[pitch_df["game_date"] >= cutoff_21d].copy()
    ff_21d    = ff[ff["game_date"] >= cutoff_21d].copy()

    # SwStr% last 21d
    swstr_21d = pitch_21d.groupby("pitcher").agg(
        pitches_21d=("is_pitch", "sum"),
        swstr_21d=("is_swstr", "sum"),
    ).reset_index()
    swstr_21d["swstr_pct_21d"] = (swstr_21d["swstr_21d"] / swstr_21d["pitches_21d"].replace(0, np.nan) * 100).round(1)

    # Avg velo last 21d
    velo_21d = ff_21d.groupby("pitcher").agg(
        avg_velo_21d=("release_speed", "mean"),
    ).reset_index()
    velo_21d["avg_velo_21d"] = velo_21d["avg_velo_21d"].round(1)

    # K per start and avg IP last 3 starts
    ko_21d = pa_21d[pa_21d["is_k"] == 1].copy()
    ko_by_game = ko_21d.groupby(["pitcher", "game_pk"]).size().reset_index(name="ks_in_game")

    ko_last3 = (
        ko_by_game.sort_values("game_pk", ascending=False)
        .groupby("pitcher").head(3)
        .groupby("pitcher")
        .agg(k_last_3=("ks_in_game", "sum"), starts_last_3=("game_pk", "count"))
        .reset_index()
    )
    ko_last3["k_per_start_21d"] = (ko_last3["k_last_3"] / ko_last3["starts_last_3"].replace(0, np.nan)).round(1)

    outs_21d = pa_21d[pa_21d["is_out"] == 1].copy()
    outs_by_game_21d = outs_21d.groupby(["pitcher", "game_pk"]).size().reset_index(name="outs_in_game")
    ip_last3 = (
        outs_by_game_21d.sort_values("game_pk", ascending=False)
        .groupby("pitcher").head(3)
        .groupby("pitcher")
        .agg(total_outs_last_3=("outs_in_game", "sum"), starts_counted=("game_pk", "count"))
        .reset_index()
    )
    ip_last3["avg_ip_last_3"] = (ip_last3["total_outs_last_3"] / 3 / ip_last3["starts_counted"].replace(0, np.nan)).round(2)

    # ── Merge all season metrics ───────────────────────────────────
    result = pa_agg.copy()
    result = result.merge(pitch_agg[["pitcher", "swstr_pct", "chase_rate"]], on="pitcher", how="left")
    result = result.merge(fps_agg[["pitcher", "first_pitch_strike_pct"]], on="pitcher", how="left")
    result = result.merge(velo_agg, on="pitcher", how="left")
    result = result.merge(bbe_agg, on="pitcher", how="left")
    result = result.merge(team_map, on="pitcher", how="left")

    # ── Merge 21-day rolling ───────────────────────────────────────
    result = result.merge(swstr_21d[["pitcher", "swstr_pct_21d"]], on="pitcher", how="left")
    result = result.merge(velo_21d, on="pitcher", how="left")
    result = result.merge(ko_last3[["pitcher", "k_last_3", "k_per_start_21d"]], on="pitcher", how="left")
    result = result.merge(ip_last3[["pitcher", "avg_ip_last_3"]], on="pitcher", how="left")

    # ── Trend columns ──────────────────────────────────────────────
    result["swstr_trend"] = (result["swstr_pct_21d"].fillna(0) - result["swstr_pct"].fillna(0)).round(1)
    result["velo_trend"]  = (result["avg_velo_21d"].fillna(0)  - result["fastball_velo"].fillna(0)).round(1)

    # ── Projected K per start ──────────────────────────────────────
    result["projected_k_baseline"] = (
        result["avg_ip_per_start"].fillna(0) * (result["k_per_9"].fillna(0) / 9)
    ).round(1)

    # ── Pitcher hand and team ──────────────────────────────────────
    result["pitcher_hand"] = result["pitcher"].map(lambda x: hand_map.get(x, ""))

    # ── Filter to starters only ────────────────────────────────────
    result = result[
        (result["games_started"] >= MIN_GS) &
        (result["ip"] >= MIN_IP) &
        (result["opener_risk"] == 0)
    ].copy()

    # ── Player names ───────────────────────────────────────────────
    pitcher_ids = result["pitcher"].dropna().astype(int).tolist()
    name_map    = lookup_player_names(pitcher_ids)
    result["pitcher_name"]      = result["pitcher"].map(lambda x: name_map.get(int(x), "") if pd.notna(x) else "")
    result["pitcher_name_norm"] = result["pitcher_name"].apply(normalize_name)
    result = result.rename(columns={
        "pitcher":       "statcast_id",
        "pitching_team": "team",
    })

    result = result[result["pitcher_name"] != ""].copy()
    result = result.sort_values("k_pct_season", ascending=False).reset_index(drop=True)

    print(f"KS_Statcast: {len(result)} starters")

    # ── Team K rates ───────────────────────────────────────────────
    AB_EVENTS = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "strikeout",
        "strikeout_double_play", "other_out",
    }

    batter_pa = df[df["events"].astype("string").str.lower().isin(AB_EVENTS)].copy()
    batter_pa = batter_pa.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in batter_pa.columns]
    )
    batter_pa["is_k"] = batter_pa["events"].astype("string").str.lower().isin(
        {"strikeout", "strikeout_double_play"}
    ).astype(int)

    if "batting_team" in batter_pa.columns:
        team_k = batter_pa.groupby("batting_team").agg(
            team_pa=("is_k", "count"),
            team_ks=("is_k", "sum"),
        ).reset_index()
        team_k["team_k_pct"] = (team_k["team_ks"] / team_k["team_pa"].replace(0, np.nan) * 100).round(1)
        team_k = team_k.rename(columns={"batting_team": "team"})
        print(f"Team K rates: {len(team_k)} teams")
    else:
        team_k = pd.DataFrame()

    return result, team_k


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
        if hasattr(df[col], "dtype") and hasattr(df[col].dtype, "numpy_dtype"):
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

    ks_df, team_k_df = build_ks_statcast(raw_df)

    if ks_df.empty:
        print("ERROR: No pitcher data built — aborting")
        return

    write_dataframe_to_sheet(gc, sheet_id, "KS_Statcast", ks_df)
    print("Written to KS_Statcast")

    if not team_k_df.empty:
        write_dataframe_to_sheet(gc, sheet_id, "Team_K_Rates", team_k_df)
        print("Written to Team_K_Rates")


if __name__ == "__main__":
    main()
