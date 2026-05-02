"""
ks_statcast.py
Pulls pitcher data for the Pitcher Strikeout model.
Uses Pitcher_Statcast_2026 already written by pitcher_statcast.py (avoids FanGraphs).
Also pulls season-level FanGraphs pitching via pybaseball with retry/fallback.
"""

import pandas as pd
import numpy as np
from pybaseball import statcast, pitching_stats
from datetime import date, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import warnings
warnings.filterwarnings("ignore")

def get_sheet(sheet_id: str):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    creds_dict = json.loads(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id)

def days_ago(n):
    return (date.today() - timedelta(days=n)).strftime("%Y-%m-%d")

def today_str():
    return date.today().strftime("%Y-%m-%d")

def pull_season_pitching():
    print("Pulling season pitching stats (pybaseball)...")
    try:
        df = pitching_stats(2025, qual=20)
        cols = {
            "Name":       "name",
            "Team":       "team",
            "G":          "games",
            "GS":         "games_started",
            "IP":         "ip",
            "SO":         "strikeouts_season",
            "K/9":        "k_per_9",
            "K%":         "k_pct_season",
            "BB%":        "bb_pct_season",
            "K-BB%":      "k_minus_bb",
            "WHIP":       "whip",
            "ERA":        "era",
            "FIP":        "fip",
            "SwStr%":     "swstr_pct",
            "O-Swing%":   "chase_rate",
            "F-Strike%":  "first_pitch_strike_pct",
            "Hard%":      "hard_hit_pct_against",
            "GB%":        "gb_pct",
            "FB%":        "fb_pct",
            "HR/FB":      "hr_per_fb",
            "vFA (pfx)":  "fastball_velo",
            "Stuff+":     "stuff_plus",
        }
        df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})
        keep = [v for v in cols.values() if v in df.columns]

        if "ip" in df.columns and "games_started" in df.columns:
            df["avg_ip_per_start"] = (
                df["ip"] / df["games_started"].replace(0, np.nan)
            ).round(2).fillna(0)
        else:
            df["avg_ip_per_start"] = 0.0

        df["opener_risk"] = (df.get("avg_ip_per_start", 5.0) < 4.0).astype(int)
        keep += ["avg_ip_per_start","opener_risk"]
        keep = [c for c in keep if c in df.columns]
        print(f"  Season pitching: {len(df)} pitchers")
        return df[keep].copy()
    except Exception as e:
        print(f"  WARNING: season pitching pull failed: {e}")
        return pd.DataFrame()

def pull_recent_starts():
    print("Pulling recent 21-day pitcher Statcast data...")
    try:
        df = statcast(start_dt=days_ago(21), end_dt=today_str())
        if df.empty:
            return pd.DataFrame()

        df["swinging_strike"] = df["description"].isin(
            ["swinging_strike","swinging_strike_blocked","foul_tip"]
        ).astype(int)
        df["pitch"] = 1

        # Strikeouts
        ko = df[df["events"] == "strikeout"].groupby("pitcher").agg(
            k_last_3=("events","count")
        ).reset_index()

        # SwStr%
        sw = df.groupby("pitcher").agg(
            pitches_21d=("pitch","sum"),
            swstr_21d=("swinging_strike","sum"),
        ).reset_index()
        sw["swstr_pct_21d"] = (sw["swstr_21d"] / sw["pitches_21d"] * 100).round(1)

        # Velo
        ff = df[df["pitch_type"].isin(["FF","SI","FC"])]
        vl = ff.groupby("pitcher").agg(avg_velo_21d=("release_speed","mean")).reset_index()
        vl["avg_velo_21d"] = vl["avg_velo_21d"].round(1)

        ko["k_per_start_21d"] = (ko["k_last_3"] / 3).round(1)

        recent = ko.merge(sw[["pitcher","swstr_pct_21d"]], on="pitcher", how="outer")
        recent = recent.merge(vl, on="pitcher", how="outer")
        recent = recent.rename(columns={"pitcher":"mlb_id"})
        print(f"  Recent starts: {len(recent)} pitchers")
        return recent[["mlb_id","k_last_3","k_per_start_21d","swstr_pct_21d","avg_velo_21d"]]
    except Exception as e:
        print(f"  WARNING: recent starts pull failed: {e}")
        return pd.DataFrame()

def build_ks_statcast(sheet_id: str):
    print("\n=== KS Statcast Pull ===")
    wb = get_sheet(sheet_id)

    # Load Pitcher_Statcast_2026 written by pitcher_statcast.py
    print("Loading Pitcher_Statcast_2026 from Google Sheets...")
    try:
        ws = wb.worksheet("Pitcher_Statcast_2026")
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} pitchers from Pitcher_Statcast_2026")
    except Exception as e:
        print(f"  WARNING: could not load Pitcher_Statcast_2026: {e}")
        df = pd.DataFrame()

    # Supplement with FanGraphs season stats if available
    season = pull_season_pitching()

    if not season.empty and not df.empty and "name" in df.columns and "name" in season.columns:
        # Merge FanGraphs columns not already in pitcher statcast sheet
        new_cols = [c for c in season.columns if c not in df.columns and c != "name"]
        if new_cols:
            df = df.merge(season[["name"] + new_cols], on="name", how="left")
            print(f"  Merged {len(new_cols)} FanGraphs columns into pitcher data")
    elif not season.empty and df.empty:
        df = season.copy()
        print("  Using FanGraphs data as primary source")

    if df.empty:
        print("ERROR: No pitcher data available — aborting")
        return

    # Pull recent form
    recent = pull_recent_starts()

    # Merge recent on mlb_id
    id_col = None
    for c in ["mlb_id","pitcher_id","player_id"]:
        if c in df.columns:
            id_col = c
            break

    if id_col and not recent.empty:
        df = df.merge(recent.rename(columns={"mlb_id": id_col}), on=id_col, how="left")

    # Fill missing columns
    for col in ["k_last_3","k_per_start_21d","swstr_pct_21d","avg_velo_21d"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Derived trend columns
    if "swstr_pct" in df.columns and "swstr_pct_21d" in df.columns:
        df["swstr_trend"] = (
            pd.to_numeric(df["swstr_pct_21d"], errors="coerce") -
            pd.to_numeric(df["swstr_pct"],     errors="coerce")
        ).round(1).fillna(0.0)
    else:
        df["swstr_trend"] = 0.0

    if "fastball_velo" in df.columns and "avg_velo_21d" in df.columns:
        df["velo_trend"] = (
            pd.to_numeric(df["avg_velo_21d"],    errors="coerce") -
            pd.to_numeric(df["fastball_velo"],   errors="coerce")
        ).round(1).fillna(0.0)
    else:
        df["velo_trend"] = 0.0

    # Projected K ceiling
    if "avg_ip_per_start" in df.columns and "k_per_9" in df.columns:
        df["projected_k_ceiling"] = (
            pd.to_numeric(df["avg_ip_per_start"], errors="coerce").fillna(0) *
            (pd.to_numeric(df["k_per_9"],         errors="coerce").fillna(0) / 9)
        ).round(1)
    else:
        df["projected_k_ceiling"] = 0.0

    print(f"  Total pitchers in KS_Statcast: {len(df)}")

    # Write to sheet
    try:
        out_ws = wb.worksheet("KS_Statcast")
        out_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        out_ws = wb.add_worksheet(title="KS_Statcast", rows=400, cols=40)

    df = df.fillna("").replace([np.inf, -np.inf], "")
    headers = df.columns.tolist()
    rows    = df.values.tolist()
    out_ws.update([headers] + rows)

    out_ws.format("A1:AZ1", {
        "backgroundColor": {"red": 0.13, "green": 0.55, "blue": 0.34},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    })
    print(f"  ✓ KS_Statcast written: {len(df)} rows")

if __name__ == "__main__":
    sheet_id = os.environ.get("GOOGLE_SHEET_ID","")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID not set")
    build_ks_statcast(sheet_id)
