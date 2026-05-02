"""
ks_statcast.py
Pulls pitcher Statcast data for the Pitcher Strikeout model.
Outputs to 'KS_Statcast' Google Sheet tab.
Mirrors pitcher_statcast.py architecture from the HR model.
"""

import pandas as pd
import numpy as np
from pybaseball import pitching_stats, statcast_pitcher
from datetime import date, timedelta
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ── Google Sheets auth ──────────────────────────────────────────────────────
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

# ── Date helpers ────────────────────────────────────────────────────────────
def days_ago(n: int) -> str:
    return (date.today() - timedelta(days=n)).strftime("%Y-%m-%d")

def today_str() -> str:
    return date.today().strftime("%Y-%m-%d")

# ── Pull season FanGraphs pitching stats ────────────────────────────────────
def pull_season_pitching() -> pd.DataFrame:
    print("Pulling season pitching stats (FanGraphs)...")
    try:
        df = pitching_stats(2025, qual=20)
        cols = {
            "IDfg": "fg_id",
            "Name": "name",
            "Team": "team",
            "G": "games",
            "GS": "games_started",
            "IP": "ip",
            "TBF": "batters_faced",
            "SO": "strikeouts_season",
            "K/9": "k_per_9",
            "K%": "k_pct_season",
            "BB%": "bb_pct_season",
            "K-BB%": "k_minus_bb",
            "WHIP": "whip",
            "ERA": "era",
            "FIP": "fip",
            "xFIP": "xfip",
            "SwStr%": "swstr_pct",
            "O-Swing%": "chase_rate",
            "Z-Contact%": "zone_contact_pct",
            "F-Strike%": "first_pitch_strike_pct",
            "Barrel%": "barrel_pct_against",
            "Hard%": "hard_hit_pct_against",
            "GB%": "gb_pct",
            "FB%": "fb_pct",
            "HR/FB": "hr_per_fb",
            "vFA (pfx)": "fastball_velo",
            "Stuff+": "stuff_plus",
            "Location+": "location_plus",
            "Pitching+": "pitching_plus",
        }
        df = df.rename(columns={k: v for k, v in cols.items() if k in df.columns})
        keep = [v for v in cols.values() if v in df.columns]

        # Avg innings per start
        if "ip" in df.columns and "games_started" in df.columns:
            df["avg_ip_per_start"] = (
                df["ip"] / df["games_started"].replace(0, np.nan)
            ).round(2)
        else:
            df["avg_ip_per_start"] = 0.0

        # Starter flag (avg IP < 4.0 = opener/bulk risk)
        df["opener_risk"] = (df.get("avg_ip_per_start", 5.0) < 4.0).astype(int)

        keep += ["avg_ip_per_start", "opener_risk"]
        keep = [c for c in keep if c in df.columns]
        return df[keep].copy()
    except Exception as e:
        print(f"  WARNING: season pitching pull failed: {e}")
        return pd.DataFrame()

# ── Pull last 3 starts stats ─────────────────────────────────────────────────
def pull_recent_starts(season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulls last 3 starts K totals and SwStr% trend from Statcast.
    Uses pitcher mlb_id from season_df if available.
    """
    print("Pulling recent starts data...")
    start = days_ago(21)  # ~3 starts worth
    end = today_str()

    try:
        from pybaseball import statcast
        df = statcast(start_dt=start, end_dt=end)
        if df.empty:
            print("  WARNING: no recent Statcast data")
            return pd.DataFrame()

        # Swing and miss events
        df["swinging_strike"] = df["description"].isin(
            ["swinging_strike", "swinging_strike_blocked", "foul_tip"]
        ).astype(int)
        df["pitch"] = 1

        # Strikeout events (PA level)
        ko_events = df[df["events"] == "strikeout"]
        ko_agg = ko_events.groupby("pitcher").agg(
            k_last_3=("events", "count")
        ).reset_index()

        # SwStr% over last 21 days
        swstr_agg = df.groupby("pitcher").agg(
            pitches_21d=("pitch", "sum"),
            swstr_21d=("swinging_strike", "sum"),
        ).reset_index()
        swstr_agg["swstr_pct_21d"] = (
            swstr_agg["swstr_21d"] / swstr_agg["pitches_21d"] * 100
        ).round(1)

        # Velocity trend (last 21d avg fastball velo)
        ff = df[df["pitch_type"].isin(["FF", "SI", "FC"])]
        velo_agg = ff.groupby("pitcher").agg(
            avg_velo_21d=("release_speed", "mean")
        ).reset_index()
        velo_agg["avg_velo_21d"] = velo_agg["avg_velo_21d"].round(1)

        # K per start estimate
        ko_agg["k_per_start_21d"] = (ko_agg["k_last_3"] / 3).round(1)

        # Merge all recent
        recent = ko_agg.merge(swstr_agg[["pitcher", "swstr_pct_21d"]], on="pitcher", how="outer")
        recent = recent.merge(velo_agg, on="pitcher", how="outer")
        recent = recent.rename(columns={"pitcher": "mlb_id"})

        return recent[["mlb_id", "k_last_3", "k_per_start_21d",
                        "swstr_pct_21d", "avg_velo_21d"]]
    except Exception as e:
        print(f"  WARNING: recent starts pull failed: {e}")
        return pd.DataFrame()

# ── Pull umpire zone tendency (if available via Statcast) ────────────────────
def pull_umpire_tendencies() -> pd.DataFrame:
    """
    Approximates umpire zone generosity by called strike rate on pitches
    just off the zone. Used as a game-day modifier in ks_picks.py.
    Note: Umpire assignments not available in advance via free APIs.
    Returns empty — flagged for manual enrichment or paid API.
    """
    print("  INFO: Umpire zone data not available via free API — skipping")
    return pd.DataFrame()

# ── Main assembly ────────────────────────────────────────────────────────────
def build_ks_statcast(sheet_id: str):
    print("\n=== KS Statcast Pull ===")

    season = pull_season_pitching()
    if season.empty:
        print("ERROR: season pitching data empty — aborting")
        return

    recent = pull_recent_starts(season)

    df = season.copy()

    if not recent.empty and "mlb_id" in df.columns:
        df = df.merge(recent, on="mlb_id", how="left")
    elif not recent.empty:
        # Merge on name as fallback if mlb_id not present
        print("  INFO: mlb_id not in season df — recent starts merged by name if available")

    # Fill missing recent columns
    for col in ["k_last_3", "k_per_start_21d", "swstr_pct_21d", "avg_velo_21d"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0.0)

    # SwStr% delta: recent vs season (velocity/form trend)
    if "swstr_pct" in df.columns and "swstr_pct_21d" in df.columns:
        df["swstr_trend"] = (df["swstr_pct_21d"] - df["swstr_pct"]).round(1)
    else:
        df["swstr_trend"] = 0.0

    # Velo trend flag
    if "fastball_velo" in df.columns and "avg_velo_21d" in df.columns:
        df["velo_trend"] = (df["avg_velo_21d"] - df["fastball_velo"]).round(1)
    else:
        df["velo_trend"] = 0.0

    # K ceiling multiplier based on projected innings
    # avg_ip_per_start * (k_per_9 / 9) = projected K ceiling
    if "avg_ip_per_start" in df.columns and "k_per_9" in df.columns:
        df["projected_k_ceiling"] = (
            df["avg_ip_per_start"] * (df["k_per_9"] / 9)
        ).round(1)
    else:
        df["projected_k_ceiling"] = 0.0

    # Sort by k_pct_season descending
    sort_col = "k_pct_season" if "k_pct_season" in df.columns else df.columns[0]
    df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)

    print(f"  Total pitchers: {len(df)}")

    # ── Write to Google Sheets ───────────────────────────────────────────────
    print("Writing to Google Sheets (KS_Statcast)...")
    wb = get_sheet(sheet_id)

    try:
        ws = wb.worksheet("KS_Statcast")
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = wb.add_worksheet(title="KS_Statcast", rows=400, cols=40)

    df = df.fillna("").replace([np.inf, -np.inf], "")
    headers = df.columns.tolist()
    rows = df.values.tolist()
    ws.update([headers] + rows)

    # Header formatting
    ws.format("A1:AZ1", {
        "backgroundColor": {"red": 0.13, "green": 0.55, "blue": 0.34},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    })

    print(f"  ✓ KS_Statcast written: {len(df)} rows")


if __name__ == "__main__":
    sheet_id = os.environ.get("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID environment variable not set")
    build_ks_statcast(sheet_id)

