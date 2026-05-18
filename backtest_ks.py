"""
backtest_ks.py
Precise KS model backtest — March 31 to yesterday.
Pulls full season Statcast pitch-by-pitch data once from Baseball Savant,
computes rolling pitcher stats as of each backtest date,
fetches historical K lines from The Odds API,
and checks actual Ks from MLB Stats API.
Writes results to KS_Backtest and KS_Backtest_Summary sheets.
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

START_DATE    = "2026-03-31"
END_DATE      = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
SNAPSHOT_TIME = "T14:00:00Z"  # 10 AM ET — before most games start

EXCLUDED_BOOKS = {"fliff", "espnbet"}

# KS scoring thresholds
MIN_IP          = 5.0
MIN_GS          = 1
LEAN_OVER_SCORE = 6
LEAN_OVER_EDGE  = 0.25
OVER_SCORE      = 9
OVER_EDGE       = 0.5

# League averages for regression
LEAGUE_AVG_K_PCT  = 22.5
LEAGUE_AVG_SWSTR  = 11.0
LEAGUE_AVG_CHASE  = 30.0
LEAGUE_AVG_FPS    = 59.0
LEAGUE_AVG_VELO   = 93.5
MIN_BF_FULL       = 150


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


def regress(val: float, league_avg: float, sample: float, full_sample: float) -> float:
    w = min(sample / full_sample, 1.0)
    return val * w + league_avg * (1 - w)


# ── Statcast data pull ────────────────────────────────────────────────────

def pull_full_season_statcast(start: str, end: str) -> pd.DataFrame:
    print(f"Pulling full season Statcast data {start} to {end}...")
    print("  This may take 5-10 minutes — pulling pitch-by-pitch data from Baseball Savant...")
    try:
        pybaseball.cache.enable()
        df = pybaseball.statcast(start_dt=start, end_dt=end)
        if df.empty:
            print("  WARNING: No Statcast data returned")
            return pd.DataFrame()
        print(f"  Pulled {len(df):,} pitches")

        # Keep only pitcher-relevant columns
        keep_cols = [
            "game_date", "pitcher", "player_name", "events", "description",
            "type", "strikes", "balls", "outs_when_up",
            "pitch_type", "release_speed", "release_spin_rate",
            "estimated_woba_using_speedangle",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols].copy()

        # Normalize game_date
        df["game_date"] = pd.to_datetime(df["game_date"])

        # Normalize pitcher name
        if "player_name" in df.columns:
            df["pitcher_norm"] = df["player_name"].apply(
                lambda x: normalize_name(str(x)) if pd.notna(x) else ""
            )
        else:
            df["pitcher_norm"] = ""

        print(f"  Statcast data ready: {df['game_date'].min().date()} to {df['game_date'].max().date()}")
        return df

    except Exception as e:
        print(f"  ERROR pulling Statcast data: {e}")
        return pd.DataFrame()


def compute_pitcher_stats_as_of(statcast_df: pd.DataFrame, as_of_date: str) -> pd.DataFrame:
    """
    Compute aggregated pitcher stats from all pitches on or before as_of_date.
    Returns one row per pitcher with computed stats.
    """
    cutoff = pd.Timestamp(as_of_date)
    df     = statcast_df[statcast_df["game_date"] <= cutoff].copy()

    if df.empty:
        return pd.DataFrame()

    results = []

    for pitcher_norm, group in df.groupby("pitcher_norm"):
        if not pitcher_norm:
            continue

        pitcher_name = group["player_name"].iloc[0] if "player_name" in group.columns else pitcher_norm
        total_pitches = len(group)

        # Plate appearances / BF estimate
        # Count events (end of PA)
        pa_rows = group[group["events"].notna() & (group["events"] != "")]
        bf      = len(pa_rows)

        if bf < 10:
            continue

        # Strikeouts
        ks = (pa_rows["events"] == "strikeout").sum()
        k_pct = (ks / bf * 100) if bf > 0 else 0.0

        # SwStr% — swinging strikes / total pitches
        swinging_strikes = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul_tip"
        ])]
        swstr_pct = (len(swinging_strikes) / total_pitches * 100) if total_pitches > 0 else 0.0

        # Chase rate — swings on pitches outside zone
        # Use description for chases
        chase_swings = group[group["description"].isin([
            "swinging_strike", "swinging_strike_blocked", "foul", "hit_into_play"
        ])]
        # Approximate chase as swinging strikes out of zone — use swstr as proxy
        chase_rate = swstr_pct * 2.5  # rough approximation

        # First pitch strikes
        first_pitches = group[(group["balls"] == 0) & (group["strikes"] == 0)]
        fps_strikes   = first_pitches[first_pitches["description"].isin([
            "called_strike", "swinging_strike", "foul", "foul_tip"
        ])]
        fps = (len(fps_strikes) / len(first_pitches) * 100) if len(first_pitches) > 0 else 0.0

        # Fastball velo
        fastballs = group[group["pitch_type"].isin(["FF", "SI", "FC"])]
        velo      = fastballs["release_speed"].mean() if not fastballs.empty else 0.0

        # Games started — approximate from game dates
        game_dates  = group["game_date"].dt.date.unique()
        games_started = len(game_dates)

        # IP estimate — bf / 3 * 0.9 (rough)
        ip = round(bf / 3.5, 1)

        # Rolling 21-day stats
        cutoff_21d = cutoff - pd.Timedelta(days=21)
        group_21d  = group[group["game_date"] >= cutoff_21d]
        pa_21d     = group_21d[group_21d["events"].notna() & (group_21d["events"] != "")]
        bf_21d     = len(pa_21d)

        swstr_21d = 0.0
        velo_21d  = 0.0
        k_per_21d = 0.0
        ip_21d    = 0.0

        if len(group_21d) > 0:
            sw_21d    = group_21d[group_21d["description"].isin([
                "swinging_strike", "swinging_strike_blocked", "foul_tip"
            ])]
            swstr_21d = (len(sw_21d) / len(group_21d) * 100) if len(group_21d) > 0 else 0.0
            fb_21d    = group_21d[group_21d["pitch_type"].isin(["FF", "SI", "FC"])]
            velo_21d  = fb_21d["release_speed"].mean() if not fb_21d.empty else 0.0
            ks_21d    = (pa_21d["events"] == "strikeout").sum() if bf_21d > 0 else 0
            games_21d = len(group_21d["game_date"].dt.date.unique())
            k_per_21d = (ks_21d / games_21d) if games_21d > 0 else 0.0
            ip_21d    = round(bf_21d / 3.5, 1)

        # Last 3 starts Ks
        game_date_list = sorted(group["game_date"].dt.date.unique(), reverse=True)
        last_3_dates   = game_date_list[:3]
        group_last3    = group[group["game_date"].dt.date.isin(last_3_dates)]
        pa_last3       = group_last3[group_last3["events"].notna() & (group_last3["events"] != "")]
        k_last_3       = (pa_last3["events"] == "strikeout").sum()
        ip_last_3      = round(len(pa_last3) / 3.5, 1) if len(last_3_dates) > 0 else 0.0
        avg_ip_last_3  = round(ip_last_3 / len(last_3_dates), 1) if last_3_dates else 0.0

        results.append({
            "pitcher_norm":       pitcher_norm,
            "pitcher_name":       pitcher_name,
            "bf":                 bf,
            "ks_ip":              ip,
            "games_started":      games_started,
            "k_pct_season":       round(k_pct, 2),
            "swstr_pct":          round(swstr_pct, 2),
            "chase_rate":         round(chase_rate, 2),
            "first_pitch_strike_pct": round(fps, 2),
            "fastball_velo":      round(safe_float(velo), 1),
            "swstr_pct_21d":      round(swstr_21d, 2),
            "avg_velo_21d":       round(safe_float(velo_21d), 1),
            "k_last_3":           int(k_last_3),
            "k_per_start_21d":    round(k_per_21d, 2),
            "avg_ip_last_3":      avg_ip_last_3,
            "avg_ip_per_start":   round(ip / games_started, 1) if games_started > 0 else 0.0,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ── KS scoring ────────────────────────────────────────────────────────────

def compute_ks_score(pitcher: pd.Series) -> Tuple[float, float, str]:
    ip    = safe_float(pitcher.get("ks_ip", 0))
    gs    = safe_float(pitcher.get("games_started", 0))
    bf    = safe_float(pitcher.get("bf", 0))

    if ip < MIN_IP or gs < MIN_GS:
        return 0.0, 0.0, "Low"

    k_pct     = safe_float(pitcher.get("k_pct_season", 0))
    swstr     = safe_float(pitcher.get("swstr_pct", 0))
    chase     = safe_float(pitcher.get("chase_rate", 0))
    fps       = safe_float(pitcher.get("first_pitch_strike_pct", 0))
    velo      = safe_float(pitcher.get("fastball_velo", 0))
    avg_ip    = safe_float(pitcher.get("avg_ip_per_start", 0))
    swstr_21d = safe_float(pitcher.get("swstr_pct_21d", 0))
    velo_21d  = safe_float(pitcher.get("avg_velo_21d", 0))
    k_last_3  = safe_float(pitcher.get("k_last_3", 0))
    k_per_21d = safe_float(pitcher.get("k_per_start_21d", 0))
    ip_last_3 = safe_float(pitcher.get("avg_ip_last_3", 0))

    score = 0.0

    # K rate
    k_reg = regress(k_pct, LEAGUE_AVG_K_PCT, bf, MIN_BF_FULL)
    if k_reg >= 32:   score += 3.0
    elif k_reg >= 28: score += 2.0
    elif k_reg >= 25: score += 1.2
    elif k_reg >= 22: score += 0.5
    elif k_reg <= 16: score -= 1.5
    elif k_reg <= 18: score -= 0.8

    # SwStr
    sw_reg = regress(swstr, LEAGUE_AVG_SWSTR, bf, MIN_BF_FULL)
    if sw_reg >= 15:   score += 2.5
    elif sw_reg >= 13: score += 1.5
    elif sw_reg >= 11: score += 0.5
    elif sw_reg <= 8:  score -= 1.5
    elif sw_reg <= 9:  score -= 0.8

    # Chase rate
    ch_reg = regress(chase, LEAGUE_AVG_CHASE, bf, MIN_BF_FULL)
    if ch_reg >= 36:   score += 1.5
    elif ch_reg >= 33: score += 0.8
    elif ch_reg >= 30: score += 0.2
    elif ch_reg <= 24: score -= 1.0
    elif ch_reg <= 27: score -= 0.5

    # First pitch strikes
    fp_reg = regress(fps, LEAGUE_AVG_FPS, bf, MIN_BF_FULL)
    if fp_reg >= 66:   score += 1.0
    elif fp_reg >= 63: score += 0.5
    elif fp_reg <= 54: score -= 0.8
    elif fp_reg <= 57: score -= 0.3

    # Velo
    if velo > 0:
        vl_reg = regress(velo, LEAGUE_AVG_VELO, bf, MIN_BF_FULL)
        if vl_reg >= 97:   score += 1.0
        elif vl_reg >= 95: score += 0.5
        elif vl_reg <= 90: score -= 0.5
        elif vl_reg <= 92: score -= 0.2

    # Recent SwStr trend
    if swstr_21d > 0 and swstr > 0:
        sw_trend = swstr_21d - swstr
        if sw_trend >= 3:    score += 1.5
        elif sw_trend >= 1.5: score += 0.8
        elif sw_trend <= -3:  score -= 1.5
        elif sw_trend <= -1.5: score -= 0.8

    # Velo trend
    if velo_21d > 0 and velo > 0:
        vt = velo_21d - velo
        if vt >= 2:    score += 0.8
        elif vt >= 1:  score += 0.4
        elif vt <= -2: score -= 0.8
        elif vt <= -1: score -= 0.4

    # Recent K rate
    if k_per_21d > 0:
        if k_per_21d >= 9:   score += 1.5
        elif k_per_21d >= 7: score += 0.8
        elif k_per_21d <= 3: score -= 1.0
        elif k_per_21d <= 5: score -= 0.5

    # Projected Ks
    ip_proj = ip_last_3 if ip_last_3 > 0 else avg_ip
    k_rate  = k_reg / 100.0
    bfp_est = ip_proj * 4.3
    proj_k  = round(k_rate * bfp_est, 2)

    # Confidence
    pts = 0
    if bf >= 150:  pts += 2
    elif bf >= 80: pts += 1
    if gs >= 10:   pts += 1
    confidence = "High" if pts >= 3 else "Medium" if pts >= 1 else "Low"

    return round(score, 3), proj_k, confidence


def calc_signal(score: float, proj_k: float, line: float) -> str:
    if line <= 0:
        return "—"
    edge = proj_k - line
    if score >= OVER_SCORE and edge >= OVER_EDGE:
        return f"OVER {line} ✅"
    if score >= LEAN_OVER_SCORE and edge >= LEAN_OVER_EDGE:
        return f"LEAN OVER {line}"
    return "—"


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


def get_historical_ks_odds(api_key: str, event_id: str, game_date: str) -> dict:
    snapshot = f"{game_date}{SNAPSHOT_TIME}"
    url      = f"https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events/{event_id}/odds"
    params   = {
        "apiKey":     api_key,
        "regions":    "us",
        "markets":    "pitcher_strikeouts",
        "oddsFormat": "american",
        "date":       snapshot,
    }
    time.sleep(0.3)
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data       = resp.json()
        bookmakers = data.get("data", {}).get("bookmakers", [])

        pitcher_lines = {}
        for book in bookmakers:
            if book["key"] in EXCLUDED_BOOKS:
                continue
            for market in book.get("markets", []):
                if market["key"] != "pitcher_strikeouts":
                    continue
                for outcome in market.get("outcomes", []):
                    side    = outcome.get("name", "").lower()
                    pitcher = outcome.get("description", "").strip()
                    price   = outcome.get("price")
                    point   = outcome.get("point")
                    if not pitcher or price is None or point is None:
                        continue
                    key = f"{normalize_name(pitcher)}|{point}"
                    if key not in pitcher_lines:
                        pitcher_lines[key] = {
                            "pitcher_name": pitcher,
                            "pitcher_norm": normalize_name(pitcher),
                            "line":         float(point),
                            "over_odds":    [],
                            "under_odds":   [],
                        }
                    try:
                        price_int = int(float(price))
                        if side == "over":
                            pitcher_lines[key]["over_odds"].append(price_int)
                        elif side == "under":
                            pitcher_lines[key]["under_odds"].append(price_int)
                    except (ValueError, TypeError):
                        pass

        results = {}
        for key, d in pitcher_lines.items():
            if not d["over_odds"]:
                continue
            results[d["pitcher_norm"]] = {
                "pitcher_name": d["pitcher_name"],
                "k_line":       d["line"],
                "over_odds":    int(np.median(d["over_odds"])),
                "under_odds":   int(np.median(d["under_odds"])) if d["under_odds"] else None,
                "num_books":    len(d["over_odds"]),
            }
        return results
    except Exception as e:
        print(f"    WARNING: Could not fetch KS odds for {event_id}: {e}")
        return {}


# ── MLB Stats API ─────────────────────────────────────────────────────────

def get_actual_ks(game_date: str) -> Dict[str, int]:
    results = {}
    try:
        url  = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        games = []
        for d in resp.json().get("dates", []):
            for g in d.get("games", []):
                if g.get("status", {}).get("abstractGameState") == "Final":
                    games.append(g.get("gamePk"))

        for pk in games:
            try:
                burl  = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
                br    = requests.get(burl, timeout=15)
                br.raise_for_status()
                bdata = br.json()
                for side in ["home", "away"]:
                    pitchers = bdata.get("teams", {}).get(side, {}).get("pitchers", [])
                    players  = bdata.get("teams", {}).get(side, {}).get("players", {})
                    for pid in pitchers:
                        pkey  = f"ID{pid}"
                        pdata = players.get(pkey, {})
                        name  = pdata.get("person", {}).get("fullName", "")
                        stats = pdata.get("stats", {}).get("pitching", {})
                        ks    = stats.get("strikeOuts")
                        ip    = str(stats.get("inningsPitched", "0"))
                        try:
                            ip_parts  = ip.split(".")
                            ip_float  = int(ip_parts[0]) + (int(ip_parts[1]) / 3 if len(ip_parts) > 1 else 0)
                        except Exception:
                            ip_float = 0.0
                        if name and ks is not None and ip_float >= 3.0:
                            results[normalize_name(name)] = int(ks)
                time.sleep(0.2)
            except Exception:
                pass
    except Exception as e:
        print(f"  WARNING: Could not fetch actual Ks for {game_date}: {e}")
    return results


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

    # Pull full season Statcast once
    statcast_df = pull_full_season_statcast(START_DATE, END_DATE)
    if statcast_df.empty:
        print("ERROR: No Statcast data — cannot backtest.")
        return

    all_results = []

    for game_date in dates:
        print(f"\n{'='*50}")
        print(f"Processing {game_date}...")

        # Compute pitcher stats as of this date
        pitcher_stats = compute_pitcher_stats_as_of(statcast_df, game_date)
        if pitcher_stats.empty:
            print(f"  No pitcher stats as of {game_date}")
            continue

        print(f"  Computed stats for {len(pitcher_stats)} pitchers as of {game_date}")

        # Get historical events
        events = get_historical_events(api_key, game_date)
        if not events:
            print(f"  No events found for {game_date}")
            continue

        day_events = [
            e for e in events
            if e.get("commence_time", "")[:10] == game_date
        ]
        if not day_events:
            print(f"  No games on {game_date}")
            continue

        print(f"  Found {len(day_events)} games")

        # Get actual Ks
        actual_ks = get_actual_ks(game_date)
        print(f"  Got actual Ks for {len(actual_ks)} pitchers")

        # Get odds for each game
        all_odds = {}
        for event in day_events:
            event_id  = event["id"]
            home_team = event.get("home_team", "")
            away_team = event.get("away_team", "")
            label     = f"{away_team} @ {home_team}"
            odds      = get_historical_ks_odds(api_key, event_id, game_date)
            if odds:
                print(f"  ✓ {label}: {len(odds)} pitchers with odds")
                all_odds.update(odds)
            else:
                print(f"  ✗ {label}: no K odds")

        if not all_odds:
            print(f"  No odds data for {game_date} — skipping")
            continue

        # Score each pitcher
        day_picks = []
        for pitcher_norm, odds_data in all_odds.items():
            matches = pitcher_stats[pitcher_stats["pitcher_norm"] == pitcher_norm]

            if matches.empty:
                # Try first name partial match
                first = pitcher_norm.split()[0] if pitcher_norm else ""
                if first:
                    matches = pitcher_stats[pitcher_stats["pitcher_norm"].str.startswith(first, na=False)]

            if matches.empty:
                ks_score   = 0.0
                proj_k     = 0.0
                confidence = "Low"
            else:
                ks_score, proj_k, confidence = compute_ks_score(matches.iloc[0])

            line      = odds_data["k_line"]
            over_odds = odds_data["over_odds"]
            signal    = calc_signal(ks_score, proj_k, line)
            actual    = actual_ks.get(pitcher_norm)
            hit       = "Yes" if actual is not None and actual > line else "No" if actual is not None else "Pending"

            day_picks.append({
                "date":         game_date,
                "pitcher_name": odds_data["pitcher_name"],
                "k_line":       line,
                "over_odds":    over_odds,
                "under_odds":   odds_data.get("under_odds", ""),
                "ks_score":     ks_score,
                "projected_k":  proj_k,
                "prop_signal":  signal,
                "actual_ks":    actual if actual is not None else "",
                "hit":          hit,
                "confidence":   confidence,
                "num_books":    odds_data["num_books"],
            })

        if not day_picks:
            continue

        day_df           = pd.DataFrame(day_picks)
        day_df           = day_df.sort_values("ks_score", ascending=False).reset_index(drop=True)
        day_df["rank"]   = range(1, len(day_df) + 1)
        all_results.append(day_df)
        print(f"  Built {len(day_df)} picks for {game_date}")

    if not all_results:
        print("No backtest results generated.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal backtest picks: {len(final_df)}")

    scored  = final_df[final_df["hit"].isin(["Yes", "No"])]
    signals = scored[scored["prop_signal"] != "—"]
    if not scored.empty:
        print(f"Overall hit rate: {(scored['hit'] == 'Yes').mean() * 100:.1f}%")
    if not signals.empty:
        print(f"Signal hit rate: {(signals['hit'] == 'Yes').mean() * 100:.1f}% ({len(signals)} signals)")

    write_backtest_results(gc, sheet_id, final_df)


def write_backtest_results(gc: gspread.Client, sheet_id: str, df: pd.DataFrame) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    # Raw results
    try:
        ws = sh.worksheet("KS_Backtest")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="KS_Backtest", rows=10000, cols=15)

    col_order = [
        "date", "rank", "pitcher_name", "k_line", "over_odds", "under_odds",
        "ks_score", "projected_k", "prop_signal", "actual_ks", "hit",
        "confidence", "num_books",
    ]
    for col in col_order:
        if col not in df.columns:
            df[col] = ""

    out = df[col_order].copy().fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.update([out.columns.tolist()] + out.astype(str).values.tolist()))
    print(f"Written {len(out)} rows to KS_Backtest")

    # Summary
    try:
        ws_sum = sh.worksheet("KS_Backtest_Summary")
        with_retry(lambda: ws_sum.clear())
    except gspread.WorksheetNotFound:
        ws_sum = sh.add_worksheet(title="KS_Backtest_Summary", rows=100, cols=6)

    scored = df[df["hit"].isin(["Yes", "No"])].copy()
    if scored.empty:
        print("No scored picks for summary.")
        return

    scored["hit_bool"] = scored["hit"] == "Yes"
    scored["rank"]     = pd.to_numeric(scored["rank"],     errors="coerce")
    scored["ks_score"] = pd.to_numeric(scored["ks_score"], errors="coerce")

    rows = [
        ["KS MODEL BACKTEST SUMMARY", "", "", "", "", ""],
        ["Category", "Total", "Hits", "Hit Rate %", "Avg Score", ""],
    ]

    def add_row(label, sub):
        if sub.empty: return
        total  = len(sub)
        hits   = int(sub["hit_bool"].sum())
        avg_sc = round(sub["ks_score"].mean(), 2) if not sub["ks_score"].isna().all() else 0
        rows.append([label, total, hits, f"{round(hits/total*100,1)}%", avg_sc, ""])

    add_row("Overall", scored)
    rows.append(["── By Rank ──", "", "", "", "", ""])
    for r in range(1, 11):
        add_row(f"  Rank {r}", scored[scored["rank"] == r])

    rows.append(["── By Signal ──", "", "", "", "", ""])
    for sig in sorted(scored["prop_signal"].dropna().unique()):
        if str(sig).strip() in ("", "—"): continue
        add_row(f"  {sig}", scored[scored["prop_signal"] == sig])

    rows.append(["── By Score Tier ──", "", "", "", "", ""])
    for label, sub in [
        ("  Score 10+",   scored[scored["ks_score"] >= 10]),
        ("  Score 8-10",  scored[(scored["ks_score"] >= 8)  & (scored["ks_score"] < 10)]),
        ("  Score 6-8",   scored[(scored["ks_score"] >= 6)  & (scored["ks_score"] < 8)]),
        ("  Score 4-6",   scored[(scored["ks_score"] >= 4)  & (scored["ks_score"] < 6)]),
        ("  Under 4",     scored[scored["ks_score"] < 4]),
    ]:
        add_row(label, sub)

    rows.append(["── By Confidence ──", "", "", "", "", ""])
    for tier in ["High", "Medium", "Low"]:
        add_row(f"  {tier}", scored[scored["confidence"] == tier])

    rows.append(["── By Odds Range ──", "", "", "", "", ""])
    scored["over_odds_num"] = pd.to_numeric(scored["over_odds"], errors="coerce")
    for label, sub in [
        ("  -200 or better",  scored[scored["over_odds_num"] >= -200]),
        ("  -201 to -150",    scored[(scored["over_odds_num"] < -150) & (scored["over_odds_num"] >= -200)]),
        ("  -149 to -120",    scored[(scored["over_odds_num"] < -120) & (scored["over_odds_num"] >= -149)]),
        ("  -119 to -100",    scored[(scored["over_odds_num"] < -100) & (scored["over_odds_num"] >= -119)]),
        ("  Plus odds",       scored[scored["over_odds_num"] > 0]),
    ]:
        add_row(label, sub)

    with_retry(lambda: ws_sum.update(rows))
    print(f"Written summary to KS_Backtest_Summary")


def main() -> None:
    api_key  = os.environ["ODDS_API_KEY"]
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print(f"KS Backtest — {START_DATE} to {END_DATE}")
    print("=" * 50)
    run_backtest(api_key, sheet_id, gc)
    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
