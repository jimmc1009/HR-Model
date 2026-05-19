"""
backtest_hr.py
HR model backtest against the 2025 MLB season.
Pulls full season Statcast data once from Baseball Savant,
computes batter and pitcher stats as of each date,
fetches historical HR odds from The Odds API,
checks actual HR results from Statcast events.
Writes results to HR_Backtest and HR_Backtest_Summary sheets.

SETUP:
1. Push this file to dev branch
2. Add to test.yml options list: backtest_hr
3. Add step to test.yml:
   - name: Run backtest_hr.py
     if: inputs.script == 'backtest_hr'
     env:
       GOOGLE_SERVICE_ACCOUNT_JSON: ${{ secrets.GOOGLE_SERVICE_ACCOUNT_JSON }}
       GOOGLE_SHEET_ID: ${{ secrets.GOOGLE_SHEET_ID }}
       ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
     run: python backtest_hr.py
4. Run backtest_hr from test.yml
5. Results appear in HR_Backtest and HR_Backtest_Summary on your main sheet
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import requests
import pybaseball

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ── Date range ────────────────────────────────────────────────────────────
START_DATE    = "2025-03-27"   # 2025 Opening Day
END_DATE      = "2025-09-28"   # 2025 Regular season end
SNAPSHOT_TIME = "T14:00:00Z"   # 10 AM ET — before most games start

# ── Model constants ───────────────────────────────────────────────────────
LEAGUE_AVG_HR_PER_PA     = 2.5
LEAGUE_AVG_HR_PER_FB     = 10.0
LEAGUE_AVG_ISO           = 0.155
LEAGUE_AVG_BARREL_7D     = 11.0
LEAGUE_AVG_SEASON_BARREL = 8.0
LEAGUE_AVG_HARD_HIT_7D   = 40.0
LEAGUE_AVG_EV_7D         = 89.0
MIN_PA_FULL              = 150
MIN_BBE_7D_FULL          = 20

MAX_PER_TEAM         = 2
MAX_PER_GAME         = 2
MAX_CHALK_PICKS      = 3
CHALK_ODDS_THRESHOLD = 310
MIN_BATTING_AVG      = 0.200

EXCLUDED_BOOKS = {"fliff", "espnbet"}


# ── Helpers ───────────────────────────────────────────────────────────────

def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
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


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def regress(value: float, league_avg: float, sample: float, full_sample: float) -> float:
    weight = min(sample / full_sample, 1.0)
    return (value * weight) + (league_avg * (1 - weight))


# ── Scoring functions (simplified core) ──────────────────────────────────

def score_barrel_pct_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < 5: return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 20: return 2.0
    if v >= 15: return 1.5
    if v >= 10: return 1.0
    if v >= 6:  return 0.4
    return 0.0


def score_season_barrel_pct(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_SEASON_BARREL, pa, MIN_PA_FULL)
    if v >= 14: return 1.2
    if v >= 11: return 0.9
    if v >= 9:  return 0.6
    if v >= 7:  return 0.3
    return 0.0


def score_iso(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_ISO, pa, MIN_PA_FULL)
    if v >= 0.300: return 1.2
    if v >= 0.250: return 0.9
    if v >= 0.200: return 0.6
    if v >= 0.175: return 0.4
    if v >= 0.150: return 0.2
    return 0.0


def score_hr_per_pa(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_HR_PER_PA, pa, MIN_PA_FULL)
    if v >= 6.0: return 1.8
    if v >= 4.0: return 1.2
    if v >= 2.5: return 0.6
    return 0.0


def score_hr_per_fb(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_HR_PER_FB, pa, MIN_PA_FULL)
    if v >= 20: return 1.5
    if v >= 15: return 1.0
    if v >= 10: return 0.5
    return 0.0


def score_avg_ev_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < 5: return 0.0
    v = regress(v, LEAGUE_AVG_EV_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 97: return 1.0
    if v >= 94: return 0.6
    if v >= 91: return 0.3
    return 0.0


def score_hard_hit_pct_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < 5: return 0.0
    v = regress(v, LEAGUE_AVG_HARD_HIT_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 55: return 0.8
    if v >= 45: return 0.5
    if v >= 35: return 0.2
    return 0.0


def score_pitcher_barrel_pct(v: float) -> float:
    if v >= 14: return 1.5
    if v >= 11: return 1.0
    if v >= 9:  return 0.6
    if v >= 7:  return 0.3
    return 0.0


def score_pitcher_hr_per_fb(v: float) -> float:
    if v >= 20: return 1.5
    if v >= 15: return 1.0
    if v >= 13: return 0.6
    if v >= 10: return 0.3
    return 0.0


def score_pitcher_hard_hit_pct(v: float) -> float:
    if v >= 45: return 0.8
    if v >= 38: return 0.5
    if v >= 32: return 0.2
    return 0.0


def score_park_factor(v: float) -> float:
    if v >= 20:  return 0.5
    if v >= 10:  return 0.3
    if v >= 0:   return 0.1
    if v >= -10: return -0.1
    return -0.3


def score_pitcher_quality_penalty(barrel_pct: float, hard_hit_pct: float,
                                   hr_per_fb: float, pitcher_bbe: float) -> float:
    sample_weight = min(safe_float(pitcher_bbe) / 80.0, 1.0)
    penalty = 0.0
    if barrel_pct <= 4:   penalty += 0.8
    elif barrel_pct <= 5: penalty += 0.5
    elif barrel_pct <= 6: penalty += 0.3
    if hard_hit_pct <= 30:   penalty += 0.5
    elif hard_hit_pct <= 33: penalty += 0.3
    elif hard_hit_pct <= 36: penalty += 0.15
    if hr_per_fb <= 6:    penalty += 0.5
    elif hr_per_fb <= 8:  penalty += 0.3
    elif hr_per_fb <= 10: penalty += 0.1
    return round(penalty * sample_weight, 3)


# ── Statcast data pull ────────────────────────────────────────────────────

def pull_full_season_statcast(start: str, end: str) -> pd.DataFrame:
    print(f"Pulling full 2025 season Statcast data {start} to {end}...")
    print("  This will take 10-20 minutes — pulling pitch-by-pitch data from Baseball Savant...")
    try:
        pybaseball.cache.enable()
        df = pybaseball.statcast(start_dt=start, end_dt=end)
        if df is None or df.empty:
            print("  WARNING: No Statcast data returned")
            return pd.DataFrame()
        print(f"  Pulled {len(df):,} pitches")
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Normalize batter name
        if "player_name" in df.columns:
            df["batter_norm"] = df["player_name"].apply(
                lambda x: normalize_name(str(x)) if pd.notna(x) else ""
            )
        else:
            df["batter_norm"] = ""

        print(f"  Data ready: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
        return df
    except Exception as e:
        print(f"  ERROR pulling Statcast data: {e}")
        return pd.DataFrame()


def compute_batter_stats_as_of(statcast_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """
    Compute batter stats from all pitches on or before as_of_date.
    Returns one row per batter with season and 7-day rolling stats.
    """
    cutoff    = pd.Timestamp(as_of_date)
    cutoff_7d = cutoff - pd.Timedelta(days=7)
    df_season = statcast_df[statcast_df["game_date"] <= cutoff].copy()

    if df_season.empty:
        return pd.DataFrame()

    results = []

    for batter_norm, group in df_season.groupby("batter_norm"):
        if not batter_norm:
            continue

        player_name = group["player_name"].iloc[0] if "player_name" in group.columns else batter_norm
        pa_rows     = group[group["events"].notna() & (group["events"] != "")]
        pa          = len(pa_rows)

        if pa < 10:
            continue

        # Season stats
        hrs         = (pa_rows["events"] == "home_run").sum()
        hr_per_pa   = round(hrs / pa * 100, 3) if pa > 0 else 0.0
        singles     = (pa_rows["events"] == "single").sum()
        doubles     = (pa_rows["events"] == "double").sum()
        triples     = (pa_rows["events"] == "triple").sum()
        total_bases = singles + doubles * 2 + triples * 3 + hrs * 4
        slg         = total_bases / pa if pa > 0 else 0.0
        hits        = singles + doubles + triples + hrs
        avg         = hits / pa if pa > 0 else 0.0
        iso         = round(slg - avg, 3)

        # BBE — batted ball events
        bbe_rows     = group[group["type"] == "X"]
        bbe          = len(bbe_rows)
        barrel_pct_s = 0.0
        hard_hit_s   = 0.0
        hr_per_fb_s  = 0.0

        if bbe > 0:
            barrels      = bbe_rows[bbe_rows.get("barrel", pd.Series(0, index=bbe_rows.index)).fillna(0) == 1] if "barrel" in bbe_rows.columns else pd.DataFrame()
            barrel_pct_s = round(len(barrels) / bbe * 100, 2) if bbe > 0 else 0.0
            hard_hits    = bbe_rows[bbe_rows["launch_speed"].fillna(0) >= 95] if "launch_speed" in bbe_rows.columns else pd.DataFrame()
            hard_hit_s   = round(len(hard_hits) / bbe * 100, 2) if bbe > 0 else 0.0
            fly_balls    = bbe_rows[bbe_rows.get("bb_type", pd.Series("", index=bbe_rows.index)).fillna("") == "fly_ball"] if "bb_type" in bbe_rows.columns else pd.DataFrame()
            hr_per_fb_s  = round(hrs / len(fly_balls) * 100, 2) if len(fly_balls) > 0 else 0.0

        # 7-day rolling stats
        group_7d  = group[group["game_date"] >= cutoff_7d]
        bbe_7d    = len(group_7d[group_7d["type"] == "X"])
        barrel_7d = 0.0
        hard_7d   = 0.0
        ev_7d     = 0.0

        if bbe_7d > 0:
            bbe_7d_rows  = group_7d[group_7d["type"] == "X"]
            barrels_7d   = bbe_7d_rows[bbe_7d_rows.get("barrel", pd.Series(0, index=bbe_7d_rows.index)).fillna(0) == 1] if "barrel" in bbe_7d_rows.columns else pd.DataFrame()
            barrel_7d    = round(len(barrels_7d) / bbe_7d * 100, 2)
            hard_7d_rows = bbe_7d_rows[bbe_7d_rows["launch_speed"].fillna(0) >= 95] if "launch_speed" in bbe_7d_rows.columns else pd.DataFrame()
            hard_7d      = round(len(hard_7d_rows) / bbe_7d * 100, 2)
            ev_7d        = round(bbe_7d_rows["launch_speed"].dropna().mean(), 1) if "launch_speed" in bbe_7d_rows.columns else 0.0

        results.append({
            "batter_norm":      batter_norm,
            "player_name":      player_name,
            "pa":               pa,
            "batting_avg":      round(avg, 3),
            "iso":              iso,
            "hr_per_pa":        hr_per_pa,
            "hr_per_fb":        hr_per_fb_s,
            "season_barrel_pct": barrel_pct_s,
            "hard_hit_pct_season": hard_hit_s,
            "bbe_season":       bbe,
            "barrel_pct_7d":    barrel_7d,
            "hard_hit_pct_7d":  hard_7d,
            "avg_ev_7d":        ev_7d,
            "bbe_7d":           bbe_7d,
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


def compute_pitcher_stats_as_of(statcast_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """
    Compute pitcher stats from all pitches on or before as_of_date.
    Returns one row per pitcher with barrel%, hard hit%, HR/FB%.
    """
    cutoff    = pd.Timestamp(as_of_date)
    df_season = statcast_df[statcast_df["game_date"] <= cutoff].copy()

    if df_season.empty:
        return pd.DataFrame()

    # Use pitcher column
    if "pitcher" not in df_season.columns:
        return pd.DataFrame()

    results = []

    for pitcher_id, group in df_season.groupby("pitcher"):
        pa_rows = group[group["events"].notna() & (group["events"] != "")]
        bf      = len(pa_rows)

        if bf < 20:
            continue

        bbe_rows = group[group["type"] == "X"]
        bbe      = len(bbe_rows)

        if bbe < 10:
            continue

        hrs = (pa_rows["events"] == "home_run").sum()

        barrel_pct  = 0.0
        hard_hit    = 0.0
        hr_per_fb   = 0.0
        pitcher_name = ""

        # Get pitcher name from MLB Stats API
        try:
            resp = requests.get(
                f"https://statsapi.mlb.com/api/v1/people/{int(pitcher_id)}",
                timeout=10
            )
            if resp.status_code == 200:
                pitcher_name = resp.json().get("people", [{}])[0].get("fullName", "")
        except Exception:
            pass

        if bbe > 0:
            barrels    = bbe_rows[bbe_rows.get("barrel", pd.Series(0, index=bbe_rows.index)).fillna(0) == 1] if "barrel" in bbe_rows.columns else pd.DataFrame()
            barrel_pct = round(len(barrels) / bbe * 100, 2)
            hard_hits  = bbe_rows[bbe_rows["launch_speed"].fillna(0) >= 95] if "launch_speed" in bbe_rows.columns else pd.DataFrame()
            hard_hit   = round(len(hard_hits) / bbe * 100, 2)
            fly_balls  = bbe_rows[bbe_rows.get("bb_type", pd.Series("", index=bbe_rows.index)).fillna("") == "fly_ball"] if "bb_type" in bbe_rows.columns else pd.DataFrame()
            hr_per_fb  = round(hrs / len(fly_balls) * 100, 2) if len(fly_balls) > 0 else 0.0

        results.append({
            "pitcher_id":        int(pitcher_id),
            "pitcher_name":      pitcher_name,
            "pitcher_norm":      normalize_name(pitcher_name) if pitcher_name else "",
            "pitcher_bbe":       bbe,
            "pitcher_bf":        bf,
            "pitcher_barrel_pct": barrel_pct,
            "pitcher_hard_hit":  hard_hit,
            "pitcher_hr_per_fb": hr_per_fb,
        })

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ── Scoring ───────────────────────────────────────────────────────────────

def compute_hr_score(batter: pd.Series, pitcher: pd.Series, park_factor: float = 100.0) -> float:
    pa     = safe_float(batter.get("pa", 0))
    bbe_7d = safe_float(batter.get("bbe_7d", 0))

    score = (
        score_barrel_pct_7d(safe_float(batter.get("barrel_pct_7d")), bbe_7d) +
        score_season_barrel_pct(safe_float(batter.get("season_barrel_pct")), pa) +
        score_iso(safe_float(batter.get("iso")), pa) +
        score_hr_per_pa(safe_float(batter.get("hr_per_pa")), pa) +
        score_hr_per_fb(safe_float(batter.get("hr_per_fb")), pa) +
        score_avg_ev_7d(safe_float(batter.get("avg_ev_7d")), bbe_7d) +
        score_hard_hit_pct_7d(safe_float(batter.get("hard_hit_pct_7d")), bbe_7d) +
        score_pitcher_barrel_pct(safe_float(pitcher.get("pitcher_barrel_pct", 0))) +
        score_pitcher_hr_per_fb(safe_float(pitcher.get("pitcher_hr_per_fb", 0))) +
        score_pitcher_hard_hit_pct(safe_float(pitcher.get("pitcher_hard_hit", 0))) +
        score_park_factor(park_factor - 100) -
        score_pitcher_quality_penalty(
            safe_float(pitcher.get("pitcher_barrel_pct", 0)),
            safe_float(pitcher.get("pitcher_hard_hit", 0)),
            safe_float(pitcher.get("pitcher_hr_per_fb", 0)),
            safe_float(pitcher.get("pitcher_bbe", 0)),
        )
    )
    return round(score, 3)


# ── Odds API ──────────────────────────────────────────────────────────────

def get_historical_events(api_key: str, game_date: str) -> List[dict]:
    snapshot = f"{game_date}{SNAPSHOT_TIME}"
    url      = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key, "date": snapshot}, timeout=15)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception as e:
        print(f"  WARNING: Could not fetch events for {game_date}: {e}")
        return []


def get_historical_hr_odds(api_key: str, event_id: str, game_date: str,
                            home_team: str, away_team: str) -> dict:
    """
    Fetch historical HR odds for one game.
    Returns dict of player_norm -> {odds, home_team, away_team}
    """
    snapshot = f"{game_date}{SNAPSHOT_TIME}"
    url      = f"https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/{event_id}/odds"
    params   = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "batter_home_runs",
        "oddsFormat": "american",
        "date":       snapshot,
    }
    time.sleep(0.3)
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        bookmakers = resp.json().get("data", {}).get("bookmakers", [])

        player_book_odds: Dict[str, Dict[str, int]] = {}
        player_names: Dict[str, str] = {}

        for book in bookmakers:
            if book["key"] in EXCLUDED_BOOKS:
                continue
            for market in book.get("markets", []):
                if market["key"] != "batter_home_runs":
                    continue
                for outcome in market.get("outcomes", []):
                    if outcome.get("name", "").lower() != "over":
                        continue
                    player = outcome.get("description", "").strip()
                    price  = outcome.get("price")
                    if not player or price is None:
                        continue
                    try:
                        price_int = int(float(price))
                        if price_int > 0:
                            price_int = min(price_int, 1000)
                            norm = normalize_name(player)
                            if norm not in player_book_odds:
                                player_book_odds[norm] = {}
                                player_names[norm]     = player
                            if book["key"] not in player_book_odds[norm]:
                                player_book_odds[norm][book["key"]] = price_int
                    except (ValueError, TypeError):
                        pass

        results = {}
        for norm, book_odds in player_book_odds.items():
            if not book_odds:
                continue
            # Filter outliers — remove books >2.5x median
            values = list(book_odds.values())
            median = float(np.median(values))
            clean  = {k: v for k, v in book_odds.items() if v <= median * 2.5}
            if not clean:
                clean = book_odds
            consensus = int(np.median(list(clean.values())))
            results[norm] = {
                "player_name":    player_names[norm],
                "consensus_odds": consensus,
                "home_team":      home_team,
                "away_team":      away_team,
                "num_books":      len(clean),
            }
        return results

    except Exception as e:
        print(f"    WARNING: Could not fetch HR odds for {event_id}: {e}")
        return {}


# ── Results ───────────────────────────────────────────────────────────────

def get_actual_hrs(game_date: str, statcast_df: pd.DataFrame) -> set:
    """
    Return set of (date_str, batter_norm) for all HRs hit on game_date.
    Uses the already-pulled Statcast data — no extra API calls needed.
    """
    cutoff = pd.Timestamp(game_date)
    day_df = statcast_df[
        (statcast_df["game_date"].dt.date == cutoff.date()) &
        (statcast_df["events"] == "home_run")
    ]
    return set(zip(
        day_df["game_date"].dt.strftime("%Y-%m-%d"),
        day_df["batter_norm"]
    ))


# ── Park factors ──────────────────────────────────────────────────────────

# Static 2025 park factors — using current season data as proxy
PARK_FACTORS = {
    "Colorado Rockies":      115,
    "Cincinnati Reds":       108,
    "Texas Rangers":         107,
    "Atlanta Braves":        106,
    "Philadelphia Phillies": 105,
    "New York Yankees":      105,
    "Boston Red Sox":        104,
    "Chicago Cubs":          103,
    "Milwaukee Brewers":     103,
    "Houston Astros":        102,
    "Los Angeles Dodgers":   101,
    "Baltimore Orioles":     100,
    "Toronto Blue Jays":     100,
    "Detroit Tigers":        100,
    "Kansas City Royals":     99,
    "Minnesota Twins":        99,
    "San Diego Padres":       98,
    "Arizona Diamondbacks":   98,
    "St. Louis Cardinals":    97,
    "Washington Nationals":   97,
    "Miami Marlins":          96,
    "Pittsburgh Pirates":     96,
    "Cleveland Guardians":    95,
    "New York Mets":          95,
    "Tampa Bay Rays":         94,
    "Chicago White Sox":      94,
    "Seattle Mariners":       93,
    "San Francisco Giants":   93,
    "Los Angeles Angels":     92,
    "Athletics":              91,
}


# ── Main backtest ─────────────────────────────────────────────────────────

def run_backtest(api_key: str, sheet_id: str, gc: gspread.Client) -> None:
    # Generate date range
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt   = datetime.strptime(END_DATE, "%Y-%m-%d")
    dates    = []
    curr     = start_dt
    while curr <= end_dt:
        dates.append(curr.strftime("%Y-%m-%d"))
        curr += timedelta(days=1)

    print(f"Backtesting {len(dates)} dates from {START_DATE} to {END_DATE}")

    # Pull full season Statcast once — this is the big one
    statcast_df = pull_full_season_statcast(START_DATE, END_DATE)
    if statcast_df.empty:
        print("ERROR: No Statcast data — cannot backtest.")
        return

    # Cache pitcher names to avoid repeated API calls
    pitcher_name_cache: Dict[int, str] = {}

    all_results = []

    for game_date in dates:
        print(f"\n{'='*50}")
        print(f"Processing {game_date}...")

        # Compute batter stats as of this date
        batter_stats = compute_batter_stats_as_of(statcast_df, game_date)
        if batter_stats.empty:
            print(f"  No batter stats as of {game_date} — skipping")
            continue

        # Compute pitcher stats as of this date
        pitcher_stats = compute_pitcher_stats_as_of(statcast_df, game_date)

        print(f"  Batters: {len(batter_stats)} | Pitchers: {len(pitcher_stats)}")

        # Get actual HRs from already-pulled Statcast data — free, no API calls
        actual_hrs = get_actual_hrs(game_date, statcast_df)
        print(f"  Actual HRs today: {len(actual_hrs)}")

        # Get historical events from Odds API
        events = get_historical_events(api_key, game_date)
        if not events:
            print(f"  No events found for {game_date}")
            continue

        day_events = [e for e in events if e.get("commence_time", "")[:10] == game_date]
        if not day_events:
            print(f"  No games on {game_date}")
            continue

        print(f"  Found {len(day_events)} games")

        # Get HR odds for each game
        all_odds: Dict[str, dict] = {}
        for event in day_events:
            event_id  = event["id"]
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            label     = f"{away_team} @ {home_team}"
            odds      = get_historical_hr_odds(api_key, event_id, game_date, home_team, away_team)
            if odds:
                all_odds.update(odds)

        if not all_odds:
            print(f"  No HR odds for {game_date} — skipping")
            continue

        print(f"  Got odds for {len(all_odds)} players")

        # Build picks for this date
        day_picks = []

        for player_norm, odds_data in all_odds.items():
            # Match batter stats
            batter_match = batter_stats[batter_stats["batter_norm"] == player_norm]
            if batter_match.empty:
                continue

            batter = batter_match.iloc[0]

            # Batting average filter
            if safe_float(batter.get("batting_avg", 0)) < MIN_BATTING_AVG:
                continue

            # Match pitcher — we don't know exact pitcher from odds
            # Use a league average pitcher as placeholder
            pitcher = pd.Series({
                "pitcher_barrel_pct": 8.0,
                "pitcher_hard_hit":   38.0,
                "pitcher_hr_per_fb":  10.0,
                "pitcher_bbe":        80.0,
            })

            # Park factor
            home_team   = odds_data.get("home_team", "")
            park_factor = PARK_FACTORS.get(home_team, 100)

            # Score
            hr_score = compute_hr_score(batter, pitcher, park_factor)

            # Check if HR hit
            hit = "Yes" if (game_date, player_norm) in actual_hrs else "No"

            day_picks.append({
                "date":          game_date,
                "player_name":   odds_data["player_name"],
                "player_norm":   player_norm,
                "home_team":     home_team,
                "away_team":     odds_data.get("away_team", ""),
                "consensus_odds": odds_data["consensus_odds"],
                "hr_score":      hr_score,
                "batting_avg":   safe_float(batter.get("batting_avg", 0)),
                "pa":            safe_float(batter.get("pa", 0)),
                "barrel_pct_7d": safe_float(batter.get("barrel_pct_7d", 0)),
                "iso":           safe_float(batter.get("iso", 0)),
                "hit_hr":        hit,
                "num_books":     odds_data["num_books"],
            })

        if not day_picks:
            continue

        # Sort by score and apply diversity cap
        day_df = pd.DataFrame(day_picks)
        day_df = day_df.sort_values("hr_score", ascending=False).reset_index(drop=True)

        # Apply per-team and per-game caps
        selected    = []
        team_counts = {}
        game_counts = {}
        chalk_count = 0

        for _, row in day_df.iterrows():
            if len(selected) >= 10:
                break
            team      = str(row.get("away_team", "")) if row.get("away_team") else "UNK"
            home      = str(row.get("home_team", "UNK"))
            odds_val  = int(row.get("consensus_odds", 9999))
            is_chalk  = odds_val <= CHALK_ODDS_THRESHOLD

            if team_counts.get(team, 0) >= MAX_PER_TEAM:
                continue
            if game_counts.get(home, 0) >= MAX_PER_GAME:
                continue
            if is_chalk and chalk_count >= MAX_CHALK_PICKS:
                continue

            selected.append(row)
            team_counts[team] = team_counts.get(team, 0) + 1
            game_counts[home] = game_counts.get(home, 0) + 1
            if is_chalk:
                chalk_count += 1

        if not selected:
            continue

        picks_df         = pd.DataFrame(selected).reset_index(drop=True)
        picks_df["rank"] = range(1, len(picks_df) + 1)
        all_results.append(picks_df)
        print(f"  Built {len(picks_df)} picks | HRs today: {picks_df['hit_hr'].value_counts().get('Yes', 0)}")

    if not all_results:
        print("No backtest results generated.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    scored   = final_df[final_df["hit_hr"].isin(["Yes", "No"])]
    print(f"\nTotal picks: {len(final_df)}")
    if not scored.empty:
        hit_rate = (scored["hit_hr"] == "Yes").mean() * 100
        print(f"Overall hit rate: {hit_rate:.1f}%")

    write_backtest_results(gc, sheet_id, final_df)


def write_backtest_results(gc: gspread.Client, sheet_id: str, df: pd.DataFrame) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    # Raw results sheet
    try:
        ws = sh.worksheet("HR_Backtest")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HR_Backtest", rows=50000, cols=15)

    col_order = [
        "date", "rank", "player_name", "home_team", "away_team",
        "consensus_odds", "hr_score", "batting_avg", "pa",
        "barrel_pct_7d", "iso", "hit_hr", "num_books",
    ]
    for col in col_order:
        if col not in df.columns:
            df[col] = ""

    out = df[col_order].copy().fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.update([out.columns.tolist()] + out.astype(str).values.tolist()))
    print(f"Written {len(out)} rows to HR_Backtest")

    # Summary sheet
    try:
        ws_sum = sh.worksheet("HR_Backtest_Summary")
        with_retry(lambda: ws_sum.clear())
    except gspread.WorksheetNotFound:
        ws_sum = sh.add_worksheet(title="HR_Backtest_Summary", rows=100, cols=6)

    scored = df[df["hit_hr"].isin(["Yes", "No"])].copy()
    if scored.empty:
        print("No scored picks for summary.")
        return

    scored["hit_bool"]  = scored["hit_hr"] == "Yes"
    scored["rank"]      = pd.to_numeric(scored["rank"],      errors="coerce")
    scored["hr_score"]  = pd.to_numeric(scored["hr_score"],  errors="coerce")
    scored["odds_num"]  = pd.to_numeric(scored["consensus_odds"], errors="coerce")

    rows = [
        ["HR MODEL BACKTEST SUMMARY — 2025 Season", "", "", "", "", ""],
        ["Category", "Total Picks", "HR Count", "Hit Rate %", "Avg Score", ""],
    ]

    def add_row(label, sub):
        if sub.empty: return
        total  = len(sub)
        hits   = int(sub["hit_bool"].sum())
        avg_sc = round(sub["hr_score"].mean(), 2) if not sub["hr_score"].isna().all() else 0
        rows.append([label, total, hits, f"{round(hits/total*100,1)}%", avg_sc, ""])

    add_row("Overall", scored)

    rows.append(["── By Rank ──", "", "", "", "", ""])
    for r in range(1, 11):
        add_row(f"  Rank {r}", scored[scored["rank"] == r])

    rows.append(["── By Score Tier ──", "", "", "", "", ""])
    for label, sub in [
        ("  Score 13+",   scored[scored["hr_score"] >= 13]),
        ("  Score 11-13", scored[(scored["hr_score"] >= 11) & (scored["hr_score"] < 13)]),
        ("  Score 9-11",  scored[(scored["hr_score"] >= 9)  & (scored["hr_score"] < 11)]),
        ("  Score 7-9",   scored[(scored["hr_score"] >= 7)  & (scored["hr_score"] < 9)]),
        ("  Under 7",     scored[scored["hr_score"] < 7]),
    ]:
        add_row(label, sub)

    rows.append(["── By Odds Range ──", "", "", "", "", ""])
    for label, sub in [
        ("  +200 to +299", scored[(scored["odds_num"] >= 200) & (scored["odds_num"] < 300)]),
        ("  +300 to +399", scored[(scored["odds_num"] >= 300) & (scored["odds_num"] < 400)]),
        ("  +400 to +499", scored[(scored["odds_num"] >= 400) & (scored["odds_num"] < 500)]),
        ("  +500 to +699", scored[(scored["odds_num"] >= 500) & (scored["odds_num"] < 700)]),
        ("  +700+",        scored[scored["odds_num"] >= 700]),
    ]:
        add_row(label, sub)

    rows.append(["── By Month ──", "", "", "", "", ""])
    scored["month"] = pd.to_datetime(scored["date"], errors="coerce").dt.strftime("%Y-%m")
    for month in sorted(scored["month"].dropna().unique()):
        add_row(f"  {month}", scored[scored["month"] == month])

    with_retry(lambda: ws_sum.update(rows))
    print(f"Written summary to HR_Backtest_Summary")


def main() -> None:
    api_key  = os.environ["ODDS_API_KEY"]
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print(f"HR Model Backtest — {START_DATE} to {END_DATE}")
    print("=" * 50)
    run_backtest(api_key, sheet_id, gc)
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()

