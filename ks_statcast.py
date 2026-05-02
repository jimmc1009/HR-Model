"""
ks_statcast.py
Pulls pitcher data for the Strikeout model using FanGraphs pitching stats
+ Statcast recent start data.
Outputs to KS_Statcast Google Sheet tab.
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
from pybaseball import pitching_stats, statcast

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SEASON       = 2026
SEASON_START = "2026-03-26"
MIN_IP_FG    = 15


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).strftime("%Y-%m-%d")


def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def pull_fangraphs_pitching() -> pd.DataFrame:
    print("Pulling FanGraphs season pitching stats...")
    try:
        df = pitching_stats(SEASON, qual=MIN_IP_FG)

        rename = {
            "Name":        "pitcher_name",
            "Team":        "team",
            "G":           "games",
            "GS":          "games_started",
            "IP":          "ip",
            "TBF":         "batters_faced",
            "SO":          "k_season",
            "K/9":         "k_per_9",
            "K%":          "k_pct_season",
            "BB%":         "bb_pct_season",
            "K-BB%":       "k_minus_bb",
            "WHIP":        "whip",
            "ERA":         "era",
            "FIP":         "fip",
            "xFIP":        "xfip",
            "SwStr%":      "swstr_pct",
            "O-Swing%":    "chase_rate",
            "Z-Contact%":  "zone_contact_pct",
            "F-Strike%":   "first_pitch_strike_pct",
            "Barrel%":     "barrel_pct_against",
            "Hard%":       "hard_hit_pct_against",
            "GB%":         "gb_pct",
            "FB%":         "fb_pct",
            "HR/FB":       "hr_per_fb",
            "vFA (pfx)":   "fastball_velo",
            "Stuff+":      "stuff_plus",
            "Location+":   "location_plus",
            "Pitching+":   "pitching_plus",
            "ERA-":        "era_minus",
            "FIP-":        "fip_minus",
            "xFIP-":       "xfip_minus",
        }

        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        keep = [v for v in rename.values() if v in df.columns]
        df   = df[keep].copy()

        df["pitcher_name_norm"] = df["pitcher_name"].apply(normalize_name)

        # Convert percentage columns that come as decimals
        for col in ["k_pct_season", "bb_pct_season", "k_minus_bb",
                    "swstr_pct", "chase_rate", "zone_contact_pct",
                    "first_pitch_strike_pct", "barrel_pct_against",
                    "hard_hit_pct_against", "gb_pct", "fb_pct"]:
            if col in df.columns and df[col].max() <= 1.0:
                df[col] = (df[col] * 100).round(1)

        # Avg IP per start
        if "ip" in df.columns and "games_started" in df.columns:
            df["avg_ip_per_start"] = (
                df["ip"] / df["games_started"].replace(0, np.nan)
            ).round(2)
            df["opener_risk"] = (df["avg_ip_per_start"] < 4.0).astype(int)
        else:
            df["avg_ip_per_start"] = 0.0
            df["opener_risk"]      = 0

        # Projected K per start baseline
        if "k_per_9" in df.columns and "avg_ip_per_start" in df.columns:
            df["projected_k_baseline"] = (
                df["avg_ip_per_start"] * (df["k_per_9"] / 9)
            ).round(1)

        print(f"  FanGraphs: {len(df)} pitchers")
        return df.reset_index(drop=True)

    except Exception as e:
        print(f"  WARNING: FanGraphs pitching pull failed: {e}")
        return pd.DataFrame()


def pull_recent_starts_statcast() -> pd.DataFrame:
    """
    Pulls last 21 days of Statcast data to compute:
    - K per start over last 3 starts
    - SwStr% over last 21 days (trend vs season)
    - Average fastball velocity last 21 days (velo trend)
    - Innings pitched per start last 3 starts
    """
    print("Pulling recent starts from Statcast (21 days)...")
    start = days_ago(21)
    end   = today_str()

    try:
        df = statcast(start_dt=start, end_dt=end)
        if df is None or df.empty:
            print("  WARNING: no recent Statcast data")
            return pd.DataFrame()

        # ── SwStr% last 21 days ────────────────────────────────────
        df["is_swinging_strike"] = df["description"].astype("string").isin(
            ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
        ).astype(int)
        df["is_pitch"] = 1

        pitch_agg = (
            df.groupby("pitcher")
            .agg(
                pitches_21d=("is_pitch", "sum"),
                swstr_21d=("is_swinging_strike", "sum"),
            )
            .reset_index()
        )
        pitch_agg["swstr_pct_21d"] = (
            pitch_agg["swstr_21d"] / pitch_agg["pitches_21d"].replace(0, np.nan) * 100
        ).round(1)

        # ── Velocity trend — fastball last 21 days ─────────────────
        ff = df[df["pitch_type"].astype("string").isin(["FF", "SI", "FC"])].copy()
        velo_agg = (
            ff.groupby("pitcher")
            .agg(avg_velo_21d=("release_speed", "mean"))
            .reset_index()
        )
        velo_agg["avg_velo_21d"] = velo_agg["avg_velo_21d"].round(1)

        # ── Ks and innings per start (last 3 starts) ───────────────
        # Get unique game dates per pitcher
        ko_df = df[df["events"].astype("string").str.lower() == "strikeout"].copy()
        ko_df = ko_df.drop_duplicates(
            subset=[c for c in ["game_pk", "at_bat_number", "pitcher"] if c in ko_df.columns]
        )

        ko_by_game = (
            ko_df.groupby(["pitcher", "game_pk"])
            .size()
            .reset_index(name="ks_in_game")
        )

        # Last 3 games per pitcher
        ko_last3 = (
            ko_by_game.sort_values("game_pk", ascending=False)
            .groupby("pitcher")
            .head(3)
            .groupby("pitcher")
            .agg(
                k_last_3=("ks_in_game", "sum"),
                starts_last_3=("game_pk", "count"),
            )
            .reset_index()
        )
        ko_last3["k_per_start_21d"] = (
            ko_last3["k_last_3"] / ko_last3["starts_last_3"].replace(0, np.nan)
        ).round(1)

        # Innings per start last 3
        out_events = {
            "field_out", "grounded_into_double_play", "double_play",
            "triple_play", "fielders_choice_out", "force_out",
            "sac_fly", "sac_fly_double_play", "sac_bunt",
            "sac_bunt_double_play", "other_out", "strikeout",
            "strikeout_double_play",
        }
        outs_df = df[df["events"].astype("string").str.lower().isin(out_events)].copy()
        outs_df = outs_df.drop_duplicates(
            subset=[c for c in ["game_pk", "at_bat_number", "pitcher"] if c in outs_df.columns]
        )

        outs_by_game = (
            outs_df.groupby(["pitcher", "game_pk"])
            .size()
            .reset_index(name="outs_in_game")
        )
        ip_last3 = (
            outs_by_game.sort_values("game_pk", ascending=False)
            .groupby("pitcher")
            .head(3)
            .groupby("pitcher")
            .agg(total_outs_last_3=("outs_in_game", "sum"),
                 starts_counted=("game_pk", "count"))
            .reset_index()
        )
        ip_last3["avg_ip_last_3"] = (
            ip_last3["total_outs_last_3"] / 3 / ip_last3["starts_counted"].replace(0, np.nan)
        ).round(2)

        # ── Merge all recent ───────────────────────────────────────
        recent = pitch_agg.merge(velo_agg, on="pitcher", how="outer")
        recent = recent.merge(ko_last3[["pitcher", "k_last_3", "k_per_start_21d"]], on="pitcher", how="outer")
        recent = recent.merge(ip_last3[["pitcher", "avg_ip_last_3"]], on="pitcher", how="outer")
        recent = recent.rename(columns={"pitcher": "statcast_id"})
        recent = recent.fillna(0.0)

        print(f"  Recent starts: {len(recent)} pitchers")
        return recent

    except Exception as e:
        print(f"  WARNING: recent starts pull failed: {e}")
        return pd.DataFrame()


def pull_pitcher_id_map() -> pd.DataFrame:
    """Map Statcast pitcher IDs to names for FanGraphs merge."""
    print("Building pitcher ID → name map...")
    try:
        df = statcast(start_dt=SEASON_START, end_dt=today_str())
        if df is None or df.empty:
            return pd.DataFrame()

        out_events = {
            "field_out", "grounded_into_double_play", "double_play",
            "triple_play", "fielders_choice_out", "force_out",
            "sac_fly", "strikeout", "strikeout_double_play", "other_out",
        }

        pa_df = df[df["events"].notna()].copy()
        pa_df = pa_df.drop_duplicates(
            subset=[c for c in ["game_pk", "at_bat_number", "pitcher"] if c in pa_df.columns]
        )

        # Get pitcher team
        if {"inning_topbot", "home_team", "away_team"}.issubset(pa_df.columns):
            pa_df["pitching_team"] = pa_df.apply(
                lambda r: str(r["home_team"]).strip()
                if str(r["inning_topbot"]).lower().startswith("top")
                else str(r["away_team"]).strip(), axis=1,
            )
        else:
            pa_df["pitching_team"] = ""

        # Most common team per pitcher
        team_map = (
            pa_df.groupby("pitcher")["pitching_team"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "")
            .reset_index()
            .rename(columns={"pitcher": "statcast_id", "pitching_team": "team_statcast"})
        )

        # Pitcher hand
        hand_map = {}
        if "p_throws" in pa_df.columns:
            hand_map = pa_df.groupby("pitcher")["p_throws"].first().to_dict()

        # Name lookup from MLB API
        pitcher_ids = pa_df["pitcher"].dropna().astype(int).unique().tolist()
        name_map    = {}
        for i in range(0, len(pitcher_ids), 50):
            chunk = pitcher_ids[i:i + 50]
            try:
                url  = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                for person in resp.json().get("people", []):
                    pid  = person.get("id")
                    name = person.get("fullName", "")
                    if pid and name:
                        name_map[int(pid)] = name
            except Exception:
                pass

        team_map["pitcher_name"]      = team_map["statcast_id"].map(lambda x: name_map.get(int(x), "") if pd.notna(x) else "")
        team_map["pitcher_name_norm"] = team_map["pitcher_name"].apply(normalize_name)
        team_map["pitcher_hand"]      = team_map["statcast_id"].map(lambda x: hand_map.get(x, ""))

        print(f"  Pitcher ID map: {len(team_map)} pitchers")
        return team_map

    except Exception as e:
        print(f"  WARNING: pitcher ID map failed: {e}")
        return pd.DataFrame()


def pull_team_strikeout_rates() -> pd.DataFrame:
    """
    Pull team-level K rates from season Statcast batter data.
    Used as opposing lineup K rate in KS model.
    """
    print("Building team strikeout rates...")
    try:
        df = statcast(start_dt=SEASON_START, end_dt=today_str())
        if df is None or df.empty:
            return pd.DataFrame()

        ab_events = {
            "single", "double", "triple", "home_run",
            "field_out", "grounded_into_double_play", "double_play",
            "triple_play", "field_error", "fielders_choice",
            "fielders_choice_out", "force_out", "strikeout",
            "strikeout_double_play", "other_out",
        }

        pa_df = df[df["events"].notna()].copy()
        pa_df = pa_df.drop_duplicates(
            subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
        )

        if {"inning_topbot", "home_team", "away_team"}.issubset(pa_df.columns):
            pa_df["batting_team"] = pa_df.apply(
                lambda r: str(r["away_team"]).strip()
                if str(r["inning_topbot"]).lower().startswith("top")
                else str(r["home_team"]).strip(), axis=1,
            )
        else:
            return pd.DataFrame()

        pa_df["is_k"] = pa_df["events"].astype("string").str.lower().isin(
            {"strikeout", "strikeout_double_play"}
        ).astype(int)

        pa_df = pa_df[pa_df["events"].astype("string").str.lower().isin(ab_events)]

        team_k = (
            pa_df.groupby("batting_team")
            .agg(
                team_pa=("events", "count"),
                team_ks=("is_k", "sum"),
            )
            .reset_index()
        )
        team_k["team_k_pct"] = (
            team_k["team_ks"] / team_k["team_pa"].replace(0, np.nan) * 100
        ).round(1)

        team_k = team_k.rename(columns={"batting_team": "team"})
        print(f"  Team K rates: {len(team_k)} teams")
        return team_k[["team", "team_k_pct"]]

    except Exception as e:
        print(f"  WARNING: team K rate pull failed: {e}")
        return pd.DataFrame()


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
        ws = sh.add_worksheet(title=worksheet_name, rows=500, cols=60)

    df = df.copy().replace([np.inf, -np.inf], np.nan).fillna("")
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    # ── Pull data ──────────────────────────────────────────────────
    fg_df    = pull_fangraphs_pitching()
    id_map   = pull_pitcher_id_map()
    recent   = pull_recent_starts_statcast()
    team_k   = pull_team_strikeout_rates()

    if fg_df.empty:
        print("ERROR: FanGraphs pitching data empty — aborting")
        return

    # ── Merge Statcast ID → FanGraphs via name ─────────────────────
    if not id_map.empty:
        fg_df = fg_df.merge(
            id_map[["statcast_id", "pitcher_name_norm", "team_statcast", "pitcher_hand"]],
            on="pitcher_name_norm",
            how="left",
        )
        if "team" not in fg_df.columns and "team_statcast" in fg_df.columns:
            fg_df["team"] = fg_df["team_statcast"]

    # ── Merge recent Statcast starts ───────────────────────────────
    if not recent.empty and "statcast_id" in fg_df.columns:
        fg_df = fg_df.merge(
            recent,
            on="statcast_id",
            how="left",
        )

    # ── Compute trend columns ──────────────────────────────────────
    if "swstr_pct" in fg_df.columns and "swstr_pct_21d" in fg_df.columns:
        fg_df["swstr_trend"] = (
            fg_df["swstr_pct_21d"].fillna(0) - fg_df["swstr_pct"].fillna(0)
        ).round(1)
    else:
        fg_df["swstr_trend"] = 0.0

    if "fastball_velo" in fg_df.columns and "avg_velo_21d" in fg_df.columns:
        fg_df["velo_trend"] = (
            fg_df["avg_velo_21d"].fillna(0) - fg_df["fastball_velo"].fillna(0)
        ).round(1)
    else:
        fg_df["velo_trend"] = 0.0

    # ── Write team K rates to separate sheet for KS picks ─────────
    if not team_k.empty:
        write_dataframe_to_sheet(gc, sheet_id, "Team_K_Rates", team_k)
        print("Written to Team_K_Rates")

    # ── Sort and write ─────────────────────────────────────────────
    if "k_pct_season" in fg_df.columns:
        fg_df = fg_df.sort_values("k_pct_season", ascending=False)

    fg_df = fg_df.reset_index(drop=True)
    print(f"\nKS_Statcast: {len(fg_df)} pitchers")

    write_dataframe_to_sheet(gc, sheet_id, "KS_Statcast", fg_df)
    print("Written to KS_Statcast")


if __name__ == "__main__":
    main()
