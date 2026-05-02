"""
hrrbi_statcast.py
Pulls batter data for H+R+RBI model.
Uses Batter_Statcast_2026 sheet already written by main.py (avoids FanGraphs).
"""

import pandas as pd
import numpy as np
from pybaseball import statcast
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

def pull_14d_form():
    print("Pulling 14-day Statcast form...")
    try:
        df = statcast(start_dt=days_ago(14), end_dt=today_str())
        if df.empty:
            return pd.DataFrame()

        pa_events = [
            "single","double","triple","home_run","walk","hit_by_pitch",
            "strikeout","field_out","grounded_into_double_play","force_out",
            "sac_fly","sac_bunt","fielders_choice","fielders_choice_out",
            "double_play","triple_play",
        ]
        pa_df = df[df["events"].isin(pa_events)].copy()
        pa_df["is_hit"] = pa_df["events"].isin(
            ["single","double","triple","home_run"]
        ).astype(int)

        agg = pa_df.groupby("batter").agg(
            pa_14d=("events","count"),
            hits_14d=("is_hit","sum"),
        ).reset_index()
        agg["avg_14d"] = (agg["hits_14d"] / agg["pa_14d"]).round(3)
        agg = agg.rename(columns={"batter":"mlb_id"})
        print(f"  14-day form: {len(agg)} batters")
        return agg[["mlb_id","pa_14d","hits_14d","avg_14d"]]
    except Exception as e:
        print(f"  WARNING: 14-day form failed: {e}")
        return pd.DataFrame()

def pull_7d_statcast():
    print("Pulling 7-day Statcast BBE data...")
    try:
        df = statcast(start_dt=days_ago(7), end_dt=today_str())
        if df.empty:
            return pd.DataFrame()

        bbe = df[df["type"] == "X"].copy()
        bbe = bbe[bbe["launch_speed"].notna()].copy()

        # barrel: launch_speed >= 98 and launch_angle between 26-30, or
        # higher speed with wider angle range (simplified Statcast definition)
        def is_barrel(row):
            ls = row["launch_speed"]
            la = row["launch_angle"]
            if pd.isna(ls) or pd.isna(la): return 0
            if ls < 98: return 0
            if ls >= 98 and 26 <= la <= 30: return 1
            if ls >= 99 and 25 <= la <= 31: return 1
            if ls >= 100 and 24 <= la <= 33: return 1
            if ls >= 101 and 23 <= la <= 34: return 1
            if ls >= 102 and 22 <= la <= 35: return 1
            if ls >= 103 and 21 <= la <= 36: return 1
            if ls >= 104 and 20 <= la <= 37: return 1
            if ls >= 105 and 19 <= la <= 38: return 1
            if ls >= 106 and 18 <= la <= 39: return 1
            if ls >= 107 and 17 <= la <= 40: return 1
            if ls >= 108 and 16 <= la <= 41: return 1
            if ls >= 109 and 15 <= la <= 42: return 1
            if ls >= 110 and 14 <= la <= 43: return 1
            return 0

        bbe["is_barrel"] = bbe.apply(is_barrel, axis=1)

        agg = bbe.groupby("batter").agg(
            bbe_7d=("launch_speed","count"),
            avg_ev_7d=("launch_speed","mean"),
            avg_la_7d=("launch_angle","mean"),
            hard_hit_7d=("launch_speed", lambda x: (x >= 95).sum()),
            barrel_7d=("is_barrel","sum"),
        ).reset_index()

        agg["hard_hit_pct_7d"] = (agg["hard_hit_7d"] / agg["bbe_7d"] * 100).round(1)
        agg["barrel_pct_7d"]   = (agg["barrel_7d"]   / agg["bbe_7d"] * 100).round(1)
        agg["avg_ev_7d"]       = agg["avg_ev_7d"].round(1)
        agg["avg_la_7d"]       = agg["avg_la_7d"].round(1)
        agg = agg.rename(columns={"batter":"mlb_id"})
        print(f"  7-day BBE: {len(agg)} batters")
        return agg[["mlb_id","bbe_7d","avg_ev_7d","avg_la_7d","hard_hit_pct_7d","barrel_pct_7d"]]
    except Exception as e:
        print(f"  WARNING: 7-day Statcast failed: {e}")
        return pd.DataFrame()

def build_hrrbi_statcast(sheet_id: str):
    print("\n=== HRRBI Statcast Pull ===")
    wb = get_sheet(sheet_id)

    # Load Batter_Statcast_2026 written by main.py — no FanGraphs needed
    print("Loading Batter_Statcast_2026 from Google Sheets...")
    try:
        ws = wb.worksheet("Batter_Statcast_2026")
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} batters")
    except Exception as e:
        print(f"  ERROR: could not load Batter_Statcast_2026: {e}")
        return

    if df.empty:
        print("ERROR: Batter_Statcast_2026 is empty — run main.py first")
        return

    # Numeric coerce key columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(df[col])

    # Pull supplemental data
    form_14d  = pull_14d_form()
    stat_7d   = pull_7d_statcast()

    # Merge on mlb_id if available
    id_col = None
    for c in ["mlb_id","batter_id","player_id"]:
        if c in df.columns:
            id_col = c
            break

    if id_col and not form_14d.empty:
        df = df.merge(form_14d.rename(columns={"mlb_id": id_col}), on=id_col, how="left")

    if id_col and not stat_7d.empty:
        df = df.merge(stat_7d.rename(columns={"mlb_id": id_col}), on=id_col, how="left")

    # Fill missing
    for col in ["bbe_7d","avg_ev_7d","avg_la_7d","hard_hit_pct_7d","barrel_pct_7d",
                "pa_14d","hits_14d","avg_14d"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["hot_streak"]  = ((df["avg_14d"] >= 0.320) & (df["pa_14d"] >= 20)).astype(int)
    df["cold_streak"] = ((df["avg_14d"] <= 0.180) & (df["pa_14d"] >= 20)).astype(int)

    print(f"  Total batters in HRRBI_Statcast: {len(df)}")

    # Write to sheet
    try:
        out_ws = wb.worksheet("HRRBI_Statcast")
        out_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        out_ws = wb.add_worksheet(title="HRRBI_Statcast", rows=600, cols=50)

    df = df.fillna("").replace([np.inf, -np.inf], "")
    headers = df.columns.tolist()
    rows    = df.values.tolist()
    out_ws.update([headers] + rows)

    out_ws.format("A1:AZ1", {
        "backgroundColor": {"red": 0.13, "green": 0.37, "blue": 0.73},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    })
    print(f"  ✓ HRRBI_Statcast written: {len(df)} rows")

if __name__ == "__main__":
    sheet_id = os.environ.get("GOOGLE_SHEET_ID","")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID not set")
    build_hrrbi_statcast(sheet_id)
