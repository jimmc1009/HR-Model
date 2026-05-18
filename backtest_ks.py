"""
backtest_ks.py
Backtests the KS model from March 31 to yesterday.
Pulls historical pitcher_strikeouts odds from The Odds API,
actual K totals from MLB Stats API, and computes model scores
using pybaseball Statcast data.
Writes results to KS_Backtest sheet.
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

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

START_DATE    = "2025-03-31"
END_DATE      = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
SNAPSHOT_TIME = "T11:00:00Z"  # 11 AM UTC = ~7 AM ET, before most games start

EXCLUDED_BOOKS = {"fliff", "espnbet"}

# KS scoring constants
MIN_IP          = 10
MIN_GS          = 2
LEAN_OVER_SCORE = 6
LEAN_OVER_EDGE  = 0.25
OVER_SCORE      = 9
OVER_EDGE       = 0.5


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


# ── Odds API ──────────────────────────────────────────────────────────────

def get_historical_events(api_key: str, game_date: str) -> List[dict]:
    snapshot = f"{game_date}{SNAPSHOT_TIME}"
    url      = "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/events"
    try:
        resp = requests.get(url, params={"apiKey": api_key, "date": snapshot}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
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
    time.sleep(0.5)
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
        for key, data in pitcher_lines.items():
            if not data["over_odds"]:
                continue
            consensus_over  = int(np.median(data["over_odds"]))
            consensus_under = int(np.median(data["under_odds"])) if data["under_odds"] else None
            results[data["pitcher_norm"]] = {
                "pitcher_name":  data["pitcher_name"],
                "k_line":        data["line"],
                "over_odds":     consensus_over,
                "under_odds":    consensus_under,
                "num_books":     len(data["over_odds"]),
            }
        return results
    except Exception as e:
        print(f"    WARNING: Could not fetch KS odds for event {event_id}: {e}")
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
                burl = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
                br   = requests.get(burl, timeout=15)
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
                        ip    = stats.get("inningsPitched", "0")
                        try:
                            ip_float = float(str(ip).replace(".1", ".333").replace(".2", ".667"))
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


# ── Statcast scoring ──────────────────────────────────────────────────────

def regress(val: float, league_avg: float, sample: float, full_sample: float) -> float:
    w = min(sample / full_sample, 1.0)
    return val * w + league_avg * (1 - w)


LEAGUE_AVG_K_PCT   = 22.5
LEAGUE_AVG_SWSTR   = 11.0
LEAGUE_AVG_CHASE   = 30.0
LEAGUE_AVG_FPS     = 59.0
LEAGUE_AVG_VELO    = 93.5
MIN_BF_FULL        = 150
MIN_BBE_7D_FULL    = 30


def compute_ks_score(pitcher: pd.Series, as_of_date: str) -> tuple:
    """
    Compute KS score for a pitcher as of a given date.
    Returns (ks_score, projected_k, confidence)
    """
    ip         = safe_float(pitcher.get("ip", 0))
    gs         = safe_float(pitcher.get("games_started", 0))
    bf         = safe_float(pitcher.get("bf", 0))
    k_pct      = safe_float(pitcher.get("k_pct_season", 0))
    bb_pct     = safe_float(pitcher.get("bb_pct_season", 0))
    swstr      = safe_float(pitcher.get("swstr_pct", 0))
    chase      = safe_float(pitcher.get("chase_rate", 0))
    fps        = safe_float(pitcher.get("first_pitch_strike_pct", 0))
    velo       = safe_float(pitcher.get("fastball_velo", 0))
    avg_ip     = safe_float(pitcher.get("avg_ip_per_start", 0))

    swstr_21d  = safe_float(pitcher.get("swstr_pct_21d", 0))
    velo_21d   = safe_float(pitcher.get("avg_velo_21d", 0))
    k_last_3   = safe_float(pitcher.get("k_last_3", 0))
    k_per_21d  = safe_float(pitcher.get("k_per_start_21d", 0))
    ip_last_3  = safe_float(pitcher.get("avg_ip_last_3", 0))

    if ip < MIN_IP or gs < MIN_GS:
        return 0.0, 0.0, "Low"

    score = 0.0

    # K rate
    k_reg = regress(k_pct, LEAGUE_AVG_K_PCT, bf, MIN_BF_FULL)
    if k_reg >= 32: score += 3.0
    elif k_reg >= 28: score += 2.0
    elif k_reg >= 25: score += 1.2
    elif k_reg >= 22: score += 0.5
    elif k_reg <= 16: score -= 1.5
    elif k_reg <= 18: score -= 0.8

    # SwStr
    sw_reg = regress(swstr, LEAGUE_AVG_SWSTR, bf, MIN_BF_FULL)
    if sw_reg >= 15: score += 2.5
    elif sw_reg >= 13: score += 1.5
    elif sw_reg >= 11: score += 0.5
    elif sw_reg <= 8: score -= 1.5
    elif sw_reg <= 9: score -= 0.8

    # Chase rate
    ch_reg = regress(chase, LEAGUE_AVG_CHASE, bf, MIN_BF_FULL)
    if ch_reg >= 36: score += 1.5
    elif ch_reg >= 33: score += 0.8
    elif ch_reg >= 30: score += 0.2
    elif ch_reg <= 24: score -= 1.0
    elif ch_reg <= 27: score -= 0.5

    # First pitch strikes
    fp_reg = regress(fps, LEAGUE_AVG_FPS, bf, MIN_BF_FULL)
    if fp_reg >= 66: score += 1.0
    elif fp_reg >= 63: score += 0.5
    elif fp_reg <= 54: score -= 0.8
    elif fp_reg <= 57: score -= 0.3

    # Velo
    if velo > 0:
        vl_reg = regress(velo, LEAGUE_AVG_VELO, bf, MIN_BF_FULL)
        if vl_reg >= 97: score += 1.0
        elif vl_reg >= 95: score += 0.5
        elif vl_reg <= 90: score -= 0.5
        elif vl_reg <= 92: score -= 0.2

    # Recent form
    if swstr_21d > 0 and swstr > 0:
        sw_trend = swstr_21d - swstr
        if sw_trend >= 3: score += 1.5
        elif sw_trend >= 1.5: score += 0.8
        elif sw_trend <= -3: score -= 1.5
        elif sw_trend <= -1.5: score -= 0.8

    if velo_21d > 0 and velo > 0:
        vt = velo_21d - velo
        if vt >= 2: score += 0.8
        elif vt >= 1: score += 0.4
        elif vt <= -2: score -= 0.8
        elif vt <= -1: score -= 0.4

    if k_per_21d > 0:
        if k_per_21d >= 9: score += 1.5
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
    if bf >= 150: pts += 2
    elif bf >= 80: pts += 1
    if gs >= 10: pts += 1
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

    # Read pitcher stats from existing sheet
    print("Reading pitcher stats from Pitcher_Statcast_2026 sheet...")
    try:
        sh          = with_retry(lambda: gc.open_by_key(sheet_id))
        ws          = sh.worksheet("Pitcher_Statcast_2026")
        data        = with_retry(lambda: ws.get_all_records())
        statcast_df = pd.DataFrame(data)
        print(f"  Loaded {len(statcast_df)} pitchers")
    except Exception as e:
        print(f"ERROR: Could not read Pitcher_Statcast_2026: {e}")
        return

    if statcast_df.empty:
        print("ERROR: No Statcast data — cannot backtest.")
        return

    if "pitcher_name" in statcast_df.columns:
        statcast_df["pitcher_norm"] = statcast_df["pitcher_name"].apply(normalize_name)
    else:
        print(f"ERROR: No pitcher_name column. Columns: {statcast_df.columns.tolist()}")
        return

    all_results = []

    for game_date in dates:
        print(f"\n{'='*50}")
        print(f"Processing {game_date}...")

        # Get historical events
        events = get_historical_events(api_key, game_date)
        if not events:
            print(f"  No events found for {game_date}")
            continue

        # Filter to games on this date
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

        # Score each pitcher and build picks
        day_picks = []
        for pitcher_norm, odds_data in all_odds.items():
            # Find in statcast
            matches = statcast_df[statcast_df["pitcher_norm"] == pitcher_norm]
            if matches.empty:
                # Try partial match
                matches = statcast_df[statcast_df["pitcher_norm"].str.contains(
                    pitcher_norm.split()[0] if pitcher_norm else "", na=False
                )]

            if matches.empty:
                ks_score  = 0.0
                proj_k    = 0.0
                confidence = "Low"
            else:
                pitcher_row = matches.iloc[0]
                ks_score, proj_k, confidence = compute_ks_score(pitcher_row, game_date)

            line       = odds_data["k_line"]
            over_odds  = odds_data["over_odds"]
            under_odds = odds_data.get("under_odds")
            signal     = calc_signal(ks_score, proj_k, line)
            actual     = actual_ks.get(pitcher_norm)
            hit        = "Yes" if actual is not None and actual > line else "No" if actual is not None else "Pending"

            day_picks.append({
                "date":          game_date,
                "pitcher_name":  odds_data["pitcher_name"],
                "k_line":        line,
                "over_odds":     over_odds,
                "under_odds":    under_odds if under_odds is not None else "",
                "ks_score":      ks_score,
                "projected_k":   proj_k,
                "prop_signal":   signal,
                "actual_ks":     actual if actual is not None else "",
                "hit":           hit,
                "confidence":    confidence,
                "num_books":     odds_data["num_books"],
            })

        if not day_picks:
            continue

        # Rank by score
        day_df = pd.DataFrame(day_picks)
        day_df = day_df.sort_values("ks_score", ascending=False).reset_index(drop=True)
        day_df["rank"] = range(1, len(day_df) + 1)
        all_results.append(day_df)
        print(f"  Built {len(day_df)} picks for {game_date}")

    if not all_results:
        print("No backtest results generated.")
        return

    final_df = pd.concat(all_results, ignore_index=True)
    print(f"\nTotal backtest picks: {len(final_df)}")

    # Summary stats
    scored    = final_df[final_df["hit"].isin(["Yes", "No"])]
    signals   = scored[scored["prop_signal"] != "—"]
    print(f"Scored picks: {len(scored)}")
    if not scored.empty:
        hit_rate = (scored["hit"] == "Yes").mean() * 100
        print(f"Overall hit rate: {hit_rate:.1f}%")
    if not signals.empty:
        sig_rate = (signals["hit"] == "Yes").mean() * 100
        print(f"Signal hit rate: {sig_rate:.1f}% ({len(signals)} signals)")

    # Write to sheet
    write_backtest_results(gc, sheet_id, final_df)


def write_backtest_results(gc: gspread.Client, sheet_id: str, df: pd.DataFrame) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("KS_Backtest")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="KS_Backtest", rows=5000, cols=15)

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
    print(f"Written {len(out)} rows to KS_Backtest sheet")

    # Summary sheet
    try:
        ws_sum = sh.worksheet("KS_Backtest_Summary")
        with_retry(lambda: ws_sum.clear())
    except gspread.WorksheetNotFound:
        ws_sum = sh.add_worksheet(title="KS_Backtest_Summary", rows=100, cols=6)

    scored  = df[df["hit"].isin(["Yes", "No"])].copy()
    scored["hit_bool"] = scored["hit"] == "Yes"
    scored["rank"]     = pd.to_numeric(scored["rank"], errors="coerce")
    scored["ks_score"] = pd.to_numeric(scored["ks_score"], errors="coerce")

    summary_rows = [["KS MODEL BACKTEST SUMMARY", "", "", "", "", ""]]
    summary_rows.append(["Category", "Total", "Hits", "Hit Rate %", "Avg Score", ""])

    def add_row(label, sub):
        if sub.empty: return
        total    = len(sub)
        hits     = int(sub["hit_bool"].sum())
        avg_sc   = round(sub["ks_score"].mean(), 2) if not sub["ks_score"].isna().all() else 0
        summary_rows.append([label, total, hits, f"{round(hits/total*100,1)}%", avg_sc, ""])

    add_row("Overall", scored)
    summary_rows.append(["── By Rank ──", "", "", "", "", ""])
    for r in range(1, 11):
        add_row(f"  Rank {r}", scored[scored["rank"] == r])

    summary_rows.append(["── By Signal ──", "", "", "", "", ""])
    for sig in sorted(scored["prop_signal"].dropna().unique()):
        if str(sig).strip() in ("", "—"): continue
        add_row(f"  {sig}", scored[scored["prop_signal"] == sig])

    summary_rows.append(["── By Score Tier ──", "", "", "", "", ""])
    for label, sub in [
        ("  Score 10+",  scored[scored["ks_score"] >= 10]),
        ("  Score 8-10", scored[(scored["ks_score"] >= 8) & (scored["ks_score"] < 10)]),
        ("  Score 6-8",  scored[(scored["ks_score"] >= 6) & (scored["ks_score"] < 8)]),
        ("  Score 4-6",  scored[(scored["ks_score"] >= 4) & (scored["ks_score"] < 6)]),
        ("  Under 4",    scored[scored["ks_score"] < 4]),
    ]:
        add_row(label, sub)

    summary_rows.append(["── By Confidence ──", "", "", "", "", ""])
    for tier in ["High", "Medium", "Low"]:
        add_row(f"  {tier}", scored[scored["confidence"] == tier])

    with_retry(lambda: ws_sum.update(summary_rows))
    print(f"Written summary to KS_Backtest_Summary sheet")


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
