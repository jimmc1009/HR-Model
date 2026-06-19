"""
hrrbi_statcast.py
Lightweight reader — copies batter data from Batter_Statcast_2026
into HRRBI_Statcast with all columns needed for momentum scoring.

Updated: pulls team runs per game from MLB Stats API and merges onto
each batter row. Momentum weight zeroed out in hrrbi_picks.py.
"""
import os
import json
import time
import requests
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

HRRBI_COLS = [
    "batter_id",
    "player_name",
    "team",
    "batter_hand",
    "pa",
    "avg",
    "obp",
    "iso",
    "batting_avg",
    "woba",
    "bb_pct",
    "k_pct",
    "hard_hit_pct_season",
    "season_barrel_pct",
    "ld_pct",
    "gb_pct",
    "fb_pct",
    "hr_per_pa",
    "hr_per_fb",
    "season_hr",
    "avg_bat_order",
    # 7d metrics
    "bbe_7d",
    "avg_ev_7d",
    "avg_la_7d",
    "hard_hit_pct_7d",
    "barrel_pct_7d",
    "hr_7d",
    # 14d metrics
    "bbe_14d",
    "avg_ev_14d",
    "avg_la_14d",
    "hard_hit_pct_14d",
    "barrel_pct_14d",
    "hr_14d",
    "pa_14d",
    "hits_14d",
    "avg_14d",
    # 30d metrics
    "bbe_30d",
    "avg_ev_30d",
    "hard_hit_pct_30d",
    "barrel_pct_30d",
    # hot/cold
    "hot_streak",
    "cold_streak",
    # start rates
    "lhp_start_rate",
    "rhp_start_rate",
    # speed proxy
    "speed_score",
]

# MLB team abbreviation map — ESPN/API abbr -> our internal abbr
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


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except gspread.exceptions.APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def get_team_runs_per_game() -> dict:
    """
    Fetch season runs scored per game for all MLB teams from the Stats API.
    Returns dict: team_abbr -> runs_per_game (float)
    """
    runs_per_game = {}
    try:
        url  = "https://statsapi.mlb.com/api/v1/teams?sportId=1&season=2026"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        team_ids = {}
        for team in resp.json().get("teams", []):
            tid  = team.get("id")
            abbr = team.get("abbreviation", "")
            abbr = ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip())
            if tid and abbr:
                team_ids[tid] = abbr

        print(f"  Found {len(team_ids)} teams — fetching season stats...")

        stats_url = "https://statsapi.mlb.com/api/v1/teams/stats?sportId=1&season=2026&stats=season&group=hitting"
        resp2     = requests.get(stats_url, timeout=15)
        resp2.raise_for_status()

        for entry in resp2.json().get("stats", []):
            for split in entry.get("splits", []):
                tid   = split.get("team", {}).get("id")
                stat  = split.get("stat", {})
                games = int(stat.get("gamesPlayed", 0) or 0)
                runs  = int(stat.get("runs", 0) or 0)
                abbr  = team_ids.get(tid)
                if abbr and games > 0:
                    runs_per_game[abbr] = round(runs / games, 2)

        print(f"  Team runs/game built for {len(runs_per_game)} teams")

    except Exception as e:
        print(f"  WARNING: Could not fetch team runs/game: {e}")

    return runs_per_game


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    # ── Read Batter_Statcast_2026 ──────────────────────────────────────────
    print("Reading Batter_Statcast_2026...")
    ws      = sh.worksheet("Batter_Statcast_2026")
    data    = with_retry(lambda: ws.get_all_records())
    batters = pd.DataFrame(data)
    print(f"  {len(batters)} batters loaded")

    # Keep only columns that exist
    available = [c for c in HRRBI_COLS if c in batters.columns]
    missing   = [c for c in HRRBI_COLS if c not in batters.columns]
    if missing:
        print(f"  Missing columns (will be skipped): {missing}")

    out = batters[available].copy()

    if "batting_avg" in out.columns:
        out = out.rename(columns={"batting_avg": "avg"})

    # ── Fetch and merge team runs per game ────────────────────────────────
    print("Fetching team runs per game from MLB Stats API...")
    runs_map = get_team_runs_per_game()

    if runs_map:
        team_col = "team" if "team" in out.columns else None
        if team_col:
            out["team_runs_per_game"] = out[team_col].apply(
                lambda t: runs_map.get(str(t).strip(), "")
            )
            filled = (out["team_runs_per_game"] != "").sum()
            print(f"  team_runs_per_game populated for {filled} batters")
        else:
            out["team_runs_per_game"] = ""
            print("  WARNING: No team column found — team_runs_per_game skipped")
    else:
        out["team_runs_per_game"] = ""
        print("  WARNING: No runs data — team_runs_per_game will be empty")

    # ── Write to HRRBI_Statcast ───────────────────────────────────────────
    try:
        ws_out = sh.worksheet("HRRBI_Statcast")
        with_retry(lambda: ws_out.clear())
    except gspread.WorksheetNotFound:
        ws_out = sh.add_worksheet(
            title="HRRBI_Statcast", rows=500, cols=len(available) + 10
        )

    values = [out.columns.tolist()] + out.fillna("").astype(str).values.tolist()
    with_retry(lambda: ws_out.update(values))
    print(f"HRRBI_Statcast: {len(out)} batters written")

    # Team_K_Rates is built by main.py — do not overwrite here.
    print("Team_K_Rates built by main.py — skipping.")


if __name__ == "__main__":
    main()
