import os
import json
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple

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

FASTBALLS   = {"FF", "SI", "FC", "FA"}
BREAKING    = {"SL", "CU", "KC", "CS", "SV", "ST"}
OFFSPEED    = {"CH", "FS", "FO", "SC"}
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
    **{p: "fastball"    for p in FASTBALLS},
    **{p: "breaking"    for p in BREAKING},
    **{p: "offspeed"    for p in OFFSPEED},
    **{p: "knuckleball" for p in KNUCKLEBALL},
}

AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "double_play",
    "triple_play", "field_error", "fielders_choice",
    "fielders_choice_out", "force_out", "strikeout",
    "strikeout_double_play", "other_out",
}

OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "double_play",
    "triple_play", "fielders_choice_out", "force_out",
    "sac_fly", "sac_fly_double_play", "sac_bunt",
    "sac_bunt_double_play", "other_out", "strikeout",
    "strikeout_double_play",
}

PA_EVENTS = AB_EVENTS | {
    "walk", "hit_by_pitch", "sac_fly", "sac_fly_double_play",
    "sac_bunt", "sac_bunt_double_play", "catcher_interf", "intent_walk",
}

HIT_EVENTS = {"single", "double", "triple", "home_run"}

WOBA_WEIGHTS = {
    "walk": 0.690, "hit_by_pitch": 0.722, "intent_walk": 0.690,
    "single": 0.882, "double": 1.242, "triple": 1.569, "home_run": 2.065,
}

MIN_BBE_VS_PITCHER = 5

MLB_TEAM_IDS = [
    108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 146, 147, 158,
]


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_season_statcast() -> pd.DataFrame:
    end_dt       = date.today() - timedelta(days=1)
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

    chunks      = []
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


def get_today_confirmed_lineups() -> Dict[str, Set[int]]:
    today_str = date.today().strftime("%Y-%m-%d")
    lineups: Dict[str, Set[int]] = {}

    try:
        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={today_str}&hydrate=lineup,players"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for d in data.get("dates", []):
            for g in d.get("games", []):
                lineups_data = g.get("lineups", {})
                if not lineups_data:
                    continue

                for side, side_key in [("homePlayers", "home"), ("awayPlayers", "away")]:
                    players = lineups_data.get(side, [])
                    if not players:
                        players = g.get("teams", {}).get(side_key, {}).get("battingOrder", [])
                    if not players:
                        continue

                    abbr = g.get("teams", {}).get(side_key, {}).get(
                        "team", {}
                    ).get("abbreviation", "")
                    abbr = ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip())

                    if abbr:
                        player_ids = set()
                        for p in players:
                            if isinstance(p, dict):
                                pid = p.get("id") or p.get("person", {}).get("id")
                            else:
                                pid = p
                            if pid:
                                player_ids.add(int(pid))
                        if player_ids:
                            lineups[abbr] = player_ids

        if lineups:
            confirmed_teams = [t for t, ids in lineups.items() if ids]
            print(f"MLB API confirmed lineups for {len(confirmed_teams)} teams: {confirmed_teams}")
            return lineups

        print("MLB API returned no confirmed lineups yet — trying MLB.com lineup page...")

    except Exception as e:
        print(f"MLB Stats API lineup fetch failed: {e}")

    try:
        schedule_url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&date={today_str}&hydrate=game(content(summary)),linescore"
        )
        resp = requests.get(schedule_url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        game_pks = []
        for d in data.get("dates", []):
            for g in d.get("games", []):
                pk = g.get("gamePk")
                if pk:
                    game_pks.append((
                        pk,
                        ESPN_TO_MLB.get(
                            g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", ""),
                            g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                        ),
                        ESPN_TO_MLB.get(
                            g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", ""),
                            g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                        ),
                    ))

        for game_pk, home_abbr, away_abbr in game_pks:
            try:
                feed_url  = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
                feed_resp = requests.get(feed_url, timeout=15)
                feed_resp.raise_for_status()
                feed      = feed_resp.json()

                live_data = feed.get("liveData", {})
                boxscore  = live_data.get("boxscore", {})
                teams_box = boxscore.get("teams", {})

                for side, abbr in [("home", home_abbr), ("away", away_abbr)]:
                    if not abbr:
                        continue
                    batting_order = teams_box.get(side, {}).get("battingOrder", [])
                    if batting_order:
                        player_ids = set()
                        for pid in batting_order:
                            if pid:
                                player_ids.add(int(pid))
                        if player_ids:
                            lineups[abbr] = player_ids

            except Exception as e:
                continue

        if lineups:
            confirmed_teams = [t for t, ids in lineups.items() if ids]
            print(f"Game feed lineups for {len(confirmed_teams)} teams: {confirmed_teams}")
            return lineups

        print("Game feed returned no lineups yet — trying ESPN...")

    except Exception as e:
        print(f"Game feed lineup fetch failed: {e}")

    try:
        today_espn = date.today().strftime("%Y%m%d")
        url = (
            f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"
            f"/scoreboard?dates={today_espn}"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    abbr   = ESPN_TO_MLB.get(
                        competitor.get("team", {}).get("abbreviation", ""),
                        competitor.get("team", {}).get("abbreviation", "")
                    )
                    roster = competitor.get("roster", [])
                    if roster and abbr:
                        ids = set()
                        for player in roster:
                            pid = player.get("athlete", {}).get("id")
                            if pid:
                                ids.add(int(pid))
                        if ids:
                            lineups[abbr] = ids

        if lineups:
            print(f"ESPN fallback lineups for {len(lineups)} teams.")
            return lineups

    except Exception as e:
        print(f"ESPN lineup fallback failed: {e}")

    print("No confirmed lineups available — lineup filter will not be applied.")
    return {}


def get_active_roster_ids() -> Set[int]:
    active_ids: Set[int] = set()
    failed = 0

    for team_id in MLB_TEAM_IDS:
        try:
            url  = (
                f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
                f"?rosterType=active&season=2026"
            )
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for player in data.get("roster", []):
                pid = player.get("person", {}).get("id")
                if pid:
                    active_ids.add(int(pid))
        except Exception:
            failed += 1

    if active_ids:
        print(f"Active roster check: {len(active_ids)} active players across 30 teams ({failed} teams failed)")
    else:
        print("Active roster check failed for all teams — IL filter will not be applied.")

    return active_ids


def lookup_player_names(player_ids: List[int]) -> Dict[int, str]:
    out       = {}
    clean_ids = sorted({int(pid) for pid in player_ids if pd.notna(pid)})
    if not clean_ids:
        return out
    for i in range(0, len(clean_ids), 50):
        chunk = clean_ids[i:i + 50]
        url   = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"
        try:
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


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def is_barrel(row):
        ev, la = row["launch_speed"], row["launch_angle"]
        if pd.isna(ev) or pd.isna(la) or ev < 98:
            return False
        return max(26 - (ev - 98), 8) <= la <= min(30 + (ev - 98), 50)

    df["is_barrel"]   = df.apply(is_barrel, axis=1)
    df["is_hard_hit"] = df["launch_speed"] >= 95
    df["is_hr"]       = df["events"].astype("string").str.lower().eq("home_run")
    df["is_fly_ball"] = df["launch_angle"].between(20, 50, inclusive="both")

    if "stand" in df.columns and "hc_x" in df.columns:
        df["is_pull"] = df.apply(
            lambda r: (r["hc_x"] < 125) if str(r.get("stand")) == "R"
            else (r["hc_x"] > 125) if str(r.get("stand")) == "L"
            else False, axis=1,
        )
    else:
        df["is_pull"] = False

    if "pitch_type" in df.columns:
        df["pitch_group"] = df["pitch_type"].apply(
            lambda p: PITCH_GROUP_MAP.get(str(p), "other")
        )
    else:
        df["pitch_group"] = "other"

    # Launch angle buckets for HRRBI
    df["is_ld"] = df["launch_angle"].between(10, 25).fillna(False).astype(int) if "launch_angle" in df.columns else 0
    df["is_gb"] = (df["launch_angle"] < 10).fillna(False).astype(int) if "launch_angle" in df.columns else 0

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
    dedupe = [c for c in ["game_pk", "at_bat_number", "batter"] if c in bbe.columns]
    if dedupe:
        bbe = bbe.drop_duplicates(subset=dedupe)
    return bbe


def build_handedness_start_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "p_throws" not in df.columns:
        return pd.DataFrame()

    if "batting_team" not in df.columns:
        if {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
            df = df.copy()
            df["batting_team"] = df.apply(
                lambda r: str(r["away_team"]).strip()
                if str(r["inning_topbot"]).lower().startswith("top")
                else str(r["home_team"]).strip(), axis=1,
            )
        else:
            return pd.DataFrame()

    pa_df = df[df["events"].notna()].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "batter"] if c in pa_df.columns]
    )

    team_vs_lhp = (
        pa_df[pa_df["p_throws"] == "L"]
        .groupby("batting_team")["game_pk"]
        .nunique()
        .reset_index(name="team_games_vs_lhp")
    )
    team_vs_rhp = (
        pa_df[pa_df["p_throws"] == "R"]
        .groupby("batting_team")["game_pk"]
        .nunique()
        .reset_index(name="team_games_vs_rhp")
    )
    batter_vs_lhp = (
        pa_df[pa_df["p_throws"] == "L"]
        .groupby(["batter", "batting_team"])["game_pk"]
        .nunique()
        .reset_index(name="batter_games_vs_lhp")
    )
    batter_vs_rhp = (
        pa_df[pa_df["p_throws"] == "R"]
        .groupby(["batter", "batting_team"])["game_pk"]
        .nunique()
        .reset_index(name="batter_games_vs_rhp")
    )

    starts = batter_vs_lhp.merge(batter_vs_rhp, on=["batter", "batting_team"], how="outer").fillna(0)
    starts = starts.merge(team_vs_lhp, on="batting_team", how="left")
    starts = starts.merge(team_vs_rhp, on="batting_team", how="left")
    starts["team_games_vs_lhp"] = starts["team_games_vs_lhp"].fillna(0)
    starts["team_games_vs_rhp"] = starts["team_games_vs_rhp"].fillna(0)

    starts["lhp_start_rate"] = (
        starts["batter_games_vs_lhp"] /
        starts["team_games_vs_lhp"].replace(0, np.nan)
    ).round(3).fillna(0.5)

    starts["rhp_start_rate"] = (
        starts["batter_games_vs_rhp"] /
        starts["team_games_vs_rhp"].replace(0, np.nan)
    ).round(3).fillna(0.5)

    starts = starts.rename(columns={"batter": "batter_id"})
    print(f"Built handedness start rates for {len(starts)} batters")
    return starts[["batter_id", "batting_team", "batter_games_vs_lhp", "batter_games_vs_rhp",
                    "team_games_vs_lhp", "team_games_vs_rhp",
                    "lhp_start_rate", "rhp_start_rate"]]


def build_hrrbi_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional batter metrics needed by the HRRBI model.
    These are added to Batter_Statcast_2026 to avoid a separate Statcast pull.
    Returns a df keyed on batter ID with extra columns.
    """
    if df.empty:
        return pd.DataFrame()

    # ── PA-level data ──────────────────────────────────────────────
    pa_df = df[df["events"].astype("string").str.lower().isin(PA_EVENTS)].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
    )

    ab_df = df[df["events"].astype("string").str.lower().isin(AB_EVENTS)].copy()
    ab_df = ab_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in ab_df.columns]
    )

    pa_df["is_hit"]  = pa_df["events"].astype("string").str.lower().isin(HIT_EVENTS).astype(int)
    pa_df["is_bb"]   = pa_df["events"].astype("string").str.lower().isin({"walk", "intent_walk"}).astype(int)
    pa_df["is_k"]    = pa_df["events"].astype("string").str.lower().isin({"strikeout", "strikeout_double_play"}).astype(int)
    pa_df["is_hbp"]  = pa_df["events"].astype("string").str.lower().eq("hit_by_pitch").astype(int)

    TB_MAP = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    pa_df["tb"]       = pa_df["events"].astype("string").str.lower().map(TB_MAP).fillna(0)
    pa_df["woba_val"] = pa_df["events"].astype("string").str.lower().map(WOBA_WEIGHTS).fillna(0)

    # ── Season PA agg ──────────────────────────────────────────────
    pa_counts = pa_df.groupby("batter").agg(
        hrrbi_pa=("is_hit", "count"),
        hrrbi_hits=("is_hit", "sum"),
        hrrbi_bb=("is_bb", "sum"),
        hrrbi_k=("is_k", "sum"),
        woba_num=("woba_val", "sum"),
    ).reset_index()

    ab_counts = ab_df.groupby("batter").size().reset_index(name="hrrbi_ab")
    tb_counts = pa_df.groupby("batter")["tb"].sum().reset_index(name="hrrbi_tb")

    result = pa_counts.merge(ab_counts, on="batter", how="left")
    result = result.merge(tb_counts, on="batter", how="left")
    result["hrrbi_ab"] = result["hrrbi_ab"].fillna(0)
    result["hrrbi_tb"] = result["hrrbi_tb"].fillna(0)

    result["woba"]   = (result["woba_num"] / result["hrrbi_pa"].replace(0, np.nan)).round(3)
    result["obp"]    = ((result["hrrbi_hits"] + result["hrrbi_bb"]) / result["hrrbi_pa"].replace(0, np.nan)).round(3)
    result["bb_pct"] = (result["hrrbi_bb"] / result["hrrbi_pa"].replace(0, np.nan) * 100).round(1)
    result["k_pct"]  = (result["hrrbi_k"]  / result["hrrbi_pa"].replace(0, np.nan) * 100).round(1)

    # ── LD%, GB%, FB% from BBE ─────────────────────────────────────
    bbe = filter_bbe(df)
    if not bbe.empty:
        bbe = add_flags(bbe)
        bbe_agg = bbe.groupby("batter").agg(
            ld_pct=("is_ld", "mean"),
            gb_pct=("is_gb", "mean"),
            fb_pct=("is_fly_ball", "mean"),
        ).reset_index()
        bbe_agg["ld_pct"] = (bbe_agg["ld_pct"] * 100).round(1)
        bbe_agg["gb_pct"] = (bbe_agg["gb_pct"] * 100).round(1)
        bbe_agg["fb_pct"] = (bbe_agg["fb_pct"] * 100).round(1)
        result = result.merge(bbe_agg, on="batter", how="left")
    else:
        result["ld_pct"] = 0.0
        result["gb_pct"] = 0.0
        result["fb_pct"] = 0.0

    # ── Batting order ──────────────────────────────────────────────
    if "bat_order" in pa_df.columns:
        bat_order = (
            pa_df.groupby("batter")["bat_order"]
            .apply(lambda x: pd.to_numeric(x, errors="coerce").mean())
            .reset_index(name="avg_bat_order")
        )
        bat_order["avg_bat_order"] = bat_order["avg_bat_order"].round(1)
        result = result.merge(bat_order, on="batter", how="left")
    else:
        result["avg_bat_order"] = 5.0

    # ── 14-day rolling avg ─────────────────────────────────────────
    cutoff_14d = pd.Timestamp(date.today() - timedelta(days=14))
    pa_14d     = pa_df[pa_df["game_date"] >= cutoff_14d].copy()
    pa_14d_agg = pa_14d.groupby("batter").agg(
        pa_14d=("is_hit", "count"),
        hits_14d=("is_hit", "sum"),
    ).reset_index()
    pa_14d_agg["avg_14d"] = (pa_14d_agg["hits_14d"] / pa_14d_agg["pa_14d"].replace(0, np.nan)).round(3)
    result = result.merge(pa_14d_agg, on="batter", how="left")

    # ── Hot/cold flags ─────────────────────────────────────────────
    result["hot_streak"]  = ((result.get("avg_14d", 0) >= 0.320) & (result.get("pa_14d", 0) >= 20)).astype(int)
    result["cold_streak"] = ((result.get("avg_14d", 0) <= 0.180) & (result.get("pa_14d", 0) >= 20)).astype(int)

    # Drop intermediate columns
    drop_cols = ["hrrbi_pa", "hrrbi_hits", "hrrbi_bb", "hrrbi_k", "hrrbi_ab", "hrrbi_tb", "woba_num"]
    result = result.drop(columns=[c for c in drop_cols if c in result.columns])

    result = result.rename(columns={"batter": "batter_id_hrrbi"})
    return result


def build_vs_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    pa_df = df[df["events"].notna()].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
    )

    ab_df = df[df["events"].astype("string").str.lower().isin(AB_EVENTS)].copy()
    ab_df = ab_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in ab_df.columns]
    )

    bbe = filter_bbe(df)
    if bbe.empty:
        return pd.DataFrame()

    bbe = add_flags(bbe)

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    bbe["total_bases"] = bbe["events"].astype("string").str.lower().map(tb_map).fillna(0)
    bbe["is_hit"]      = bbe["events"].astype("string").str.lower().isin(
        {"single", "double", "triple", "home_run"}
    )

    pa_counts = (
        pa_df.groupby(["batter", "pitcher"])
        .size()
        .reset_index(name="bvp_pa")
    )
    ab_counts = (
        ab_df.groupby(["batter", "pitcher"])
        .size()
        .reset_index(name="bvp_ab")
    )

    bvp = (
        bbe.groupby(["batter", "pitcher"], dropna=False)
        .agg(
            bvp_bbe=("launch_speed", "size"),
            bvp_hr=("is_hr", "sum"),
            bvp_barrel=("is_barrel", "sum"),
            bvp_tb=("total_bases", "sum"),
            bvp_hits=("is_hit", "sum"),
        )
        .reset_index()
    )

    bvp = bvp.merge(pa_counts, on=["batter", "pitcher"], how="left")
    bvp = bvp.merge(ab_counts, on=["batter", "pitcher"], how="left")
    bvp["bvp_pa"] = bvp["bvp_pa"].fillna(0)
    bvp["bvp_ab"] = bvp["bvp_ab"].fillna(0)
    bvp = bvp[bvp["bvp_pa"] >= MIN_BBE_VS_PITCHER].copy()

    if bvp.empty:
        return pd.DataFrame()

    bvp["bvp_iso"]        = ((bvp["bvp_tb"] - bvp["bvp_hits"]) / bvp["bvp_ab"].replace(0, np.nan)).round(3)
    bvp["bvp_barrel_pct"] = (bvp["bvp_barrel"] / bvp["bvp_bbe"].replace(0, np.nan) * 100).round(2)
    bvp["bvp_hr_rate"]    = (bvp["bvp_hr"] / bvp["bvp_pa"].replace(0, np.nan) * 100).round(2)
    bvp = bvp.rename(columns={"pitcher": "pitcher_id"})

    return bvp[["batter", "pitcher_id", "bvp_pa", "bvp_hr", "bvp_iso", "bvp_barrel_pct", "bvp_hr_rate"]]


def build_batter_features(df: pd.DataFrame, today_teams: Set[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    if "batting_team" in df.columns:
        df["team"] = df["batting_team"].fillna("").astype(str)
    elif {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
        df["team"] = df.apply(
            lambda r: str(r["away_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["home_team"]).strip(), axis=1,
        )
    else:
        df["team"] = ""

    if today_teams:
        df = df[df["team"].isin(today_teams)].copy()

    if df.empty:
        return pd.DataFrame()

    df  = add_flags(df)
    bbe = filter_bbe(df)

    if bbe.empty:
        return pd.DataFrame()

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    bbe    = bbe.copy()
    bbe["total_bases"] = bbe["events"].astype("string").str.lower().map(tb_map).fillna(0)
    bbe["is_hit"]      = bbe["events"].astype("string").str.lower().isin(
        {"single", "double", "triple", "home_run"}
    )

    pa_df = df[df["events"].notna()].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
    )
    pa_counts = pa_df.groupby("batter").size().reset_index(name="pa")

    ab_df = df[df["events"].astype("string").str.lower().isin(AB_EVENTS)].copy()
    ab_df = ab_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in ab_df.columns]
    )
    ab_counts = ab_df.groupby("batter").size().reset_index(name="ab")

    season = (
        bbe.groupby("batter", dropna=False)
        .agg(
            season_bbe=("launch_speed", "size"),
            season_hr=("is_hr", "sum"),
            season_fb=("is_fly_ball", "sum"),
            season_barrel=("is_barrel", "sum"),
            season_hard_hit=("is_hard_hit", "sum"),
            total_bases=("total_bases", "sum"),
            hits=("is_hit", "sum"),
            avg_launch_angle=("launch_angle", "mean"),
            pull_count=("is_pull", "sum"),
        )
        .reset_index()
    )
    season = season.merge(pa_counts, on="batter", how="left")
    season = season.merge(ab_counts, on="batter", how="left")
    season["ab"] = season["ab"].fillna(0)
    season["pa"] = season["pa"].fillna(0)

    season["hr_per_pa"]         = (season["season_hr"] / season["pa"].replace(0, np.nan) * 100).round(2)
    season["hr_per_fb"]         = (season["season_hr"] / season["season_fb"].replace(0, np.nan) * 100).round(2)
    season["fb_rate"]           = (season["season_fb"] / season["season_bbe"].replace(0, np.nan) * 100).round(2)
    season["season_barrel_pct"] = (season["season_barrel"] / season["season_bbe"].replace(0, np.nan) * 100).round(2)
    season["iso"]               = ((season["total_bases"] - season["hits"]) / season["ab"].replace(0, np.nan)).round(3)
    season["batting_avg"]       = (season["hits"] / season["ab"].replace(0, np.nan)).round(3)
    season["pull_rate"]         = (season["pull_count"] / season["season_bbe"].replace(0, np.nan) * 100).round(2)
    season["avg_launch_angle"]  = season["avg_launch_angle"].round(2)
    season["hard_hit_pct_season"] = (season["season_hard_hit"] / season["season_bbe"].replace(0, np.nan) * 100).round(2)

    today = date.today()

    rolling_parts = []
    for days in [5, 7, 10, 14, 30]:
        cutoff = pd.Timestamp(today - timedelta(days=days))
        w      = bbe[bbe["game_date"] >= cutoff].copy()
        grp    = (
            w.groupby("batter", dropna=False)
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
        grp[f"barrel_pct_{days}d"]   = (grp[f"barrel_pct_{days}d"]   * 100).round(2)
        grp[f"hard_hit_pct_{days}d"] = (grp[f"hard_hit_pct_{days}d"] * 100).round(2)
        grp[f"avg_ev_{days}d"]       = grp[f"avg_ev_{days}d"].round(2)
        grp[f"avg_la_{days}d"]       = grp[f"avg_la_{days}d"].round(2)
        rolling_parts.append(grp)

    combined = season.copy()
    for grp in rolling_parts:
        combined = combined.merge(grp, on="batter", how="left")

    if "p_throws" in bbe.columns and "stand" in bbe.columns:
        stand_map = (
            bbe[bbe["stand"].notna()]
            .groupby("batter")["stand"].first()
            .reset_index()
            .rename(columns={"stand": "batter_hand"})
        )

        for hand, label in [("L", "vs_lhp"), ("R", "vs_rhp")]:
            sub = bbe[bbe["p_throws"] == hand].copy()
            if sub.empty:
                combined[f"{label}_iso"]        = np.nan
                combined[f"{label}_barrel_pct"] = np.nan
                combined[f"{label}_hr_rate"]    = np.nan
                continue

            if "p_throws" in ab_df.columns:
                ab_hand = (
                    ab_df[ab_df["p_throws"] == hand]
                    .groupby("batter")
                    .size()
                    .reset_index(name=f"ab_{label}")
                )
            else:
                ab_hand = pd.DataFrame(columns=["batter", f"ab_{label}"])

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
            grp = grp.merge(ab_hand, on="batter", how="left")
            grp[f"ab_{label}"]         = grp[f"ab_{label}"].fillna(0)
            grp[f"{label}_barrel_pct"] = (grp[f"{label}_barrel_pct"] * 100).round(2)
            grp[f"{label}_hr_rate"]    = (grp[f"{label}_hr"] / grp[f"{label}_bbe"].replace(0, np.nan) * 100).round(2)
            grp[f"{label}_iso"]        = ((grp[f"{label}_tb"] - grp[f"{label}_hits"]) / grp[f"ab_{label}"].replace(0, np.nan)).round(3)
            grp = grp.drop(columns=[f"{label}_tb", f"{label}_hits", f"ab_{label}"])
            combined = combined.merge(grp, on="batter", how="left")

        combined = combined.merge(stand_map, on="batter", how="left")
    else:
        combined["batter_hand"] = ""

    if "pitch_type" in bbe.columns:
        pitch_bbe = bbe[bbe["pitch_type"].notna()].copy()
        pitch_bbe["pitch_type"] = pitch_bbe["pitch_type"].astype("string").str.upper().str.strip()
        pitch_bbe = pitch_bbe[~pitch_bbe["pitch_type"].isin(["", "NAN", "NONE", "PO", "UN", "EP"])]

        if "pitch_type" in ab_df.columns:
            ab_pitch = ab_df[ab_df["pitch_type"].notna()].copy()
            ab_pitch["pitch_type"] = ab_pitch["pitch_type"].astype("string").str.upper().str.strip()
            ab_pitch = ab_pitch[~ab_pitch["pitch_type"].isin(["", "NAN", "NONE", "PO", "UN", "EP"])]
            ab_by_pitch = (
                ab_pitch.groupby(["batter", "pitch_type"])
                .size()
                .reset_index(name="ab_count")
            )
        else:
            ab_by_pitch = pd.DataFrame(columns=["batter", "pitch_type", "ab_count"])

        if not pitch_bbe.empty:
            tb_map2 = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
            pitch_bbe["total_bases"] = pitch_bbe["events"].astype("string").str.lower().map(tb_map2).fillna(0)
            pitch_bbe["is_hit"]      = pitch_bbe["events"].astype("string").str.lower().isin(
                {"single", "double", "triple", "home_run"}
            )

            pitch_grp = (
                pitch_bbe.groupby(["batter", "pitch_type"], dropna=False)
                .agg(
                    bbe_count=("launch_speed", "size"),
                    hr_count=("is_hr", "sum"),
                    barrel_count=("is_barrel", "sum"),
                    total_bases=("total_bases", "sum"),
                    hits=("is_hit", "sum"),
                )
                .reset_index()
            )
            pitch_grp = pitch_grp.merge(ab_by_pitch, on=["batter", "pitch_type"], how="left")
            pitch_grp["ab_count"] = pitch_grp["ab_count"].fillna(0)
            pitch_grp = pitch_grp[pitch_grp["bbe_count"] >= 5].copy()

            if not pitch_grp.empty:
                pitch_grp["iso_val"]        = ((pitch_grp["total_bases"] - pitch_grp["hits"]) / pitch_grp["ab_count"].replace(0, np.nan)).round(3)
                pitch_grp["hr_rate_val"]    = (pitch_grp["hr_count"] / pitch_grp["bbe_count"] * 100).round(2)
                pitch_grp["barrel_pct_val"] = (pitch_grp["barrel_count"] / pitch_grp["bbe_count"] * 100).round(2)

                iso_pivot = pitch_grp.pivot_table(
                    index="batter", columns="pitch_type", values="iso_val", fill_value=np.nan
                ).reset_index()
                iso_pivot.columns = ["batter"] + [f"iso_vs_{c}" for c in iso_pivot.columns if c != "batter"]

                hr_pivot = pitch_grp.pivot_table(
                    index="batter", columns="pitch_type", values="hr_rate_val", fill_value=np.nan
                ).reset_index()
                hr_pivot.columns = ["batter"] + [f"hr_rate_vs_{c}" for c in hr_pivot.columns if c != "batter"]

                barrel_pivot = pitch_grp.pivot_table(
                    index="batter", columns="pitch_type", values="barrel_pct_val", fill_value=np.nan
                ).reset_index()
                barrel_pivot.columns = ["batter"] + [f"barrel_pct_vs_{c}" for c in barrel_pivot.columns if c != "batter"]

                combined = combined.merge(iso_pivot,    on="batter", how="left")
                combined = combined.merge(hr_pivot,     on="batter", how="left")
                combined = combined.merge(barrel_pivot, on="batter", how="left")

    if "pitch_group" in bbe.columns:
        for group in ["fastball", "breaking", "offspeed", "knuckleball"]:
            sub = bbe[bbe["pitch_group"] == group].copy()
            if sub.empty:
                continue
            sub["total_bases"] = sub["events"].astype("string").str.lower().map(tb_map).fillna(0)
            sub["is_hit"]      = sub["events"].astype("string").str.lower().isin(
                {"single", "double", "triple", "home_run"}
            )
            if "pitch_group" in ab_df.columns:
                ab_grp = (
                    ab_df[ab_df["pitch_group"] == group]
                    .groupby("batter")
                    .size()
                    .reset_index(name=f"ab_{group}")
                )
            else:
                ab_grp = pd.DataFrame(columns=["batter", f"ab_{group}"])

            g = (
                sub.groupby("batter", dropna=False)
                .agg(
                    **{
                        f"bbe_{group}":      ("launch_speed", "size"),
                        f"hr_{group}":       ("is_hr", "sum"),
                        f"iso_{group}_tb":   ("total_bases", "sum"),
                        f"iso_{group}_hits": ("is_hit", "sum"),
                        f"hr_rate_{group}":  ("is_hr", "mean"),
                    }
                )
                .reset_index()
            )
            g = g.merge(ab_grp, on="batter", how="left")
            g[f"ab_{group}"]         = g.get(f"ab_{group}", pd.Series(0, index=g.index)).fillna(0)
            g[f"iso_vs_{group}"]     = ((g[f"iso_{group}_tb"] - g[f"iso_{group}_hits"]) / g[f"ab_{group}"].replace(0, np.nan)).round(3)
            g[f"hr_rate_vs_{group}"] = (g[f"hr_rate_{group}"] * 100).round(2)
            drop_cols = [c for c in g.columns if (f"iso_{group}_" in c and ("_tb" in c or "_hits" in c)) or c == f"hr_rate_{group}"]
            g = g.drop(columns=drop_cols, errors="ignore")
            combined = combined.merge(g[["batter", f"iso_vs_{group}", f"hr_rate_vs_{group}"]], on="batter", how="left")

    team_map = (
        df.sort_values("game_date")
        .drop_duplicates(subset=["batter"], keep="last")
        [["batter", "team"]]
    )
    combined = combined.merge(team_map, on="batter", how="left")

    combined = combined[
        (combined["season_bbe"] >= 25) &
        (combined["pa"] >= 30)
    ].copy()

    return combined


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
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=150)

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x
        )
    df = df.fillna("")
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def get_today_teams() -> Set[str]:
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        url  = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        teams: Set[str] = set()
        for d in data.get("dates", []):
            for g in d.get("games", []):
                for side in ("away", "home"):
                    abbr = g.get("teams", {}).get(side, {}).get("team", {}).get("abbreviation", "")
                    if abbr:
                        teams.add(ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip()))
        if teams:
            print(f"MLB API returned {len(teams)} teams.")
            return teams
        print("MLB API returned no teams.")
    except Exception as e:
        print(f"MLB API failed: {e}")

    try:
        espn_str = date.today().strftime("%Y%m%d")
        url  = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={espn_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data  = resp.json()
        teams = set()
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    abbr = competitor.get("team", {}).get("abbreviation", "")
                    if abbr:
                        teams.add(ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip()))
        print(f"ESPN API returned {len(teams)} teams: ***{teams}***")
        return teams
    except Exception as e:
        print(f"ESPN API failed: {e}")

    return set()


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Fetching today's teams...")
    today_teams = get_today_teams()

    print("Pulling Statcast data...")
    raw_df = get_season_statcast()
    if raw_df is None or raw_df.empty:
        print("Statcast returned empty — aborting.")
        return

    raw_df["game_date"] = pd.to_datetime(raw_df["game_date"])
    print(f"Pulled {len(raw_df):,} Statcast rows for 2026 season")

    # ── Build batter features ──────────────────────────────────────
    batter_df = build_batter_features(raw_df, today_teams)
    print(f"Built {len(batter_df)} batter rows")

    if not batter_df.empty:
        batter_ids = batter_df["batter"].dropna().astype(int).tolist()
        name_map   = lookup_player_names(batter_ids)
        batter_df["player_name"] = batter_df["batter"].map(
            lambda x: name_map.get(int(x), "") if pd.notna(x) else ""
        )
        batter_df = batter_df[batter_df["player_name"] != ""].copy()
        batter_df = batter_df.rename(columns={"batter": "batter_id"})

        # ── Handedness start rates ─────────────────────────────────
        print("Building handedness start rates...")
        start_rates = build_handedness_start_rates(raw_df)
        if not start_rates.empty:
            batter_df = batter_df.merge(start_rates, on="batter_id", how="left")
            batter_df["lhp_start_rate"] = batter_df["lhp_start_rate"].fillna(0.5)
            batter_df["rhp_start_rate"] = batter_df["rhp_start_rate"].fillna(0.5)
            print(f"Handedness start rates added for {len(start_rates)} batters")

        # ── HRRBI extra features ───────────────────────────────────
        print("Building HRRBI extra features...")
        hrrbi_extra = build_hrrbi_extra_features(raw_df)
        if not hrrbi_extra.empty:
            batter_df = batter_df.merge(
                hrrbi_extra.rename(columns={"batter_id_hrrbi": "batter_id"}),
                on="batter_id",
                how="left",
            )
            print(f"HRRBI extra features added for {len(hrrbi_extra)} batters")

        write_dataframe_to_sheet(gc, sheet_id, "Batter_Statcast_2026", batter_df)
        print("Written to Batter_Statcast_2026")

    # ── Build BvP history ──────────────────────────────────────────
    print("Building batter vs pitcher history...")
    bvp_df = build_vs_pitcher_stats(raw_df)
    if not bvp_df.empty:
        bvp_batter_ids = bvp_df["batter"].dropna().astype(int).tolist()
        bvp_name_map   = lookup_player_names(bvp_batter_ids)
        bvp_df["player_name"] = bvp_df["batter"].map(
            lambda x: bvp_name_map.get(int(x), "") if pd.notna(x) else ""
        )
        bvp_df = bvp_df.rename(columns={"batter": "batter_id"})
        write_dataframe_to_sheet(gc, sheet_id, "BvP_History", bvp_df)
        print(f"Written {len(bvp_df)} BvP rows to BvP_History")
    else:
        print("No BvP data to write.")

    # ── Active rosters ─────────────────────────────────────────────
    print("Fetching active rosters...")
    active_roster_ids = get_active_roster_ids()
    if active_roster_ids:
        active_roster_df = pd.DataFrame({"player_id": list(active_roster_ids)})
        write_dataframe_to_sheet(gc, sheet_id, "Active_Rosters", active_roster_df)
        print(f"Written {len(active_roster_ids)} active player IDs to Active_Rosters")
    else:
        print("No active roster data to write.")

    # ── Confirmed lineups ──────────────────────────────────────────
    print("Fetching confirmed lineups...")
    lineups = get_today_confirmed_lineups()
    if lineups:
        lineup_rows = []
        for team, player_ids in lineups.items():
            for pid in player_ids:
                lineup_rows.append({"team": team, "player_id": pid})
        lineup_df = pd.DataFrame(lineup_rows)
        write_dataframe_to_sheet(gc, sheet_id, "Confirmed_Lineups", lineup_df)
        print(f"Written {len(lineup_df)} lineup rows to Confirmed_Lineups")
    else:
        print("No confirmed lineups to write.")


if __name__ == "__main__":
    main()
