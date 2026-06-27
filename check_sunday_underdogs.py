"""
check_sunday_underdogs.py
Pulls historical MLB moneylines for every Sunday in 2025 and 2026 seasons
using The Odds API historical endpoint. Calculates P&L for betting every
underdog (+100 or better) $1 flat on Sundays.
Also compares Sunday underdogs vs all-day underdogs for context.
"""

import os
import json
import time
import math
from datetime import date, timedelta

import requests
import pandas as pd
import numpy as np

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
SPORT         = "baseball_mlb"
BASE_URL      = "https://api.the-odds-api.com/v4"
REGION        = "us"
MARKET        = "h2h"
ODDS_FORMAT   = "american"

# Season date ranges
SEASONS = {
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 26), date(2026, 6, 26)),  # up to yesterday
}


def get_sundays(season_start: date, season_end: date):
    """Return all Sundays in the season."""
    sundays = []
    d = season_start
    while d <= season_end:
        if d.weekday() == 6:  # Sunday
            sundays.append(d)
        d += timedelta(days=1)
    return sundays


def get_all_days(season_start: date, season_end: date):
    """Return all days in the season."""
    days = []
    d = season_start
    while d <= season_end:
        days.append(d)
        d += timedelta(days=1)
    return days


def fetch_historical_odds(game_date: date) -> list:
    """Fetch historical odds snapshot for a given date."""
    # Use noon ET snapshot — games are typically posted by then
    timestamp = f"{game_date.strftime('%Y-%m-%d')}T16:00:00Z"
    url = (
        f"{BASE_URL}/historical/sports/{SPORT}/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions={REGION}"
        f"&markets={MARKET}"
        f"&oddsFormat={ODDS_FORMAT}"
        f"&date={timestamp}"
    )
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        elif resp.status_code == 422:
            return []  # No data for this date
        else:
            print(f"  API error {resp.status_code} for {game_date}")
            return []
    except Exception as e:
        print(f"  Request failed for {game_date}: {e}")
        return []


def fetch_scores(game_date: date) -> list:
    """Fetch game scores/results for a given date."""
    url = (
        f"{BASE_URL}/sports/{SPORT}/scores"
        f"?apiKey={ODDS_API_KEY}"
        f"&daysFrom=1"
        f"&dateFormat=iso"
    )
    # For historical, use the scores endpoint with a date filter
    hist_url = (
        f"{BASE_URL}/historical/sports/{SPORT}/scores"
        f"?apiKey={ODDS_API_KEY}"
        f"&date={game_date.strftime('%Y-%m-%d')}T23:59:00Z"
    )
    try:
        resp = requests.get(hist_url, timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", [])
        return []
    except Exception as e:
        print(f"  Scores request failed for {game_date}: {e}")
        return []


def extract_underdog(game: dict) -> tuple:
    """
    From a game dict, extract the underdog team and their odds.
    Returns (underdog_team, favorite_team, underdog_odds, favorite_odds) or None.
    Uses DraftKings or first available bookmaker.
    """
    bookmakers = game.get("bookmakers", [])
    if not bookmakers:
        return None

    # Prefer DraftKings, fall back to first bookmaker
    bk = next((b for b in bookmakers if b["key"] == "draftkings"), bookmakers[0])
    markets = bk.get("markets", [])
    h2h = next((m for m in markets if m["key"] == "h2h"), None)
    if not h2h:
        return None

    outcomes = h2h.get("outcomes", [])
    if len(outcomes) < 2:
        return None

    team_a = outcomes[0]["name"]
    odds_a = outcomes[0]["price"]
    team_b = outcomes[1]["name"]
    odds_b = outcomes[1]["price"]

    # Underdog = higher American odds (more positive or less negative)
    if odds_a >= 100 and odds_a >= odds_b:
        return (team_a, team_b, odds_a, odds_b)
    elif odds_b >= 100 and odds_b >= odds_a:
        return (team_b, team_a, odds_b, odds_a)
    elif odds_a > odds_b:
        return (team_a, team_b, odds_a, odds_b)
    else:
        return (team_b, team_a, odds_b, odds_a)


def did_win(game_id: str, team: str, scores_lookup: dict) -> bool | None:
    """Check if a team won. Returns True/False/None (no result)."""
    result = scores_lookup.get(game_id)
    if not result:
        return None
    completed = result.get("completed", False)
    if not completed:
        return None
    scores = result.get("scores", [])
    if not scores or len(scores) < 2:
        return None
    team_scores = {s["name"]: int(s["score"]) for s in scores if s.get("score")}
    if team not in team_scores:
        return None
    winner = max(team_scores, key=team_scores.get)
    return winner == team


def calc_pnl(odds: float, won: bool, stake: float = 1.0) -> float:
    """Calculate P&L for a bet."""
    if won:
        if odds >= 0:
            return stake * (odds / 100)
        else:
            return stake * (100 / abs(odds))
    else:
        return -stake


def analyze_bets(bets: list, label: str):
    """Print P&L analysis for a list of bets."""
    if not bets:
        print(f"  No bets found for {label}")
        return

    total    = len(bets)
    wins     = sum(1 for b in bets if b["won"])
    win_rate = round(wins / total * 100, 1)
    pnl      = round(sum(b["pnl"] for b in bets), 2)
    roi      = round(pnl / total * 100, 1)
    avg_odds = round(sum(b["odds"] for b in bets) / total, 0)

    print(f"\n  {label}:")
    print(f"    Bets: {total} | Wins: {wins} | Win Rate: {win_rate}%")
    print(f"    Avg Odds: +{avg_odds:.0f}")
    print(f"    P&L (flat $1): {'+' if pnl >= 0 else ''}{pnl}")
    print(f"    ROI: {'+' if roi >= 0 else ''}{roi}%")

    # By season
    for season in [2025, 2026]:
        sub = [b for b in bets if b["season"] == season]
        if not sub: continue
        s_wins = sum(1 for b in sub if b["won"])
        s_pnl  = round(sum(b["pnl"] for b in sub), 2)
        s_roi  = round(s_pnl / len(sub) * 100, 1) if sub else 0
        print(f"    {season}: {s_wins}/{len(sub)} wins, P&L {'+' if s_pnl >= 0 else ''}{s_pnl}, ROI {'+' if s_roi >= 0 else ''}{s_roi}%")


def main():
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        return

    print("MLB Sunday Underdog Moneyline Analysis")
    print("="*60)

    all_bets    = []  # all underdogs on Sundays
    sunday_bets = []  # underdogs at +100 or better on Sundays
    all_day_bets = [] # all underdogs every day (for comparison)

    for season, (season_start, season_end) in SEASONS.items():
        sundays  = get_sundays(season_start, season_end)
        all_days = get_all_days(season_start, season_end)

        print(f"\n{season} Season: {len(sundays)} Sundays, {len(all_days)} total days")
        print(f"  Pulling data... (this may take a few minutes)")

        # Pull odds and scores for each Sunday
        for game_date in sundays:
            print(f"  {game_date}...", end=" ")
            games  = fetch_historical_odds(game_date)
            scores = fetch_historical_odds(game_date)  # we'll use completed flag

            # Get scores for this date
            scores_data = fetch_scores(game_date)
            scores_lookup = {s["id"]: s for s in scores_data}

            day_count = 0
            for game in games:
                result = extract_underdog(game)
                if not result:
                    continue
                underdog, favorite, ud_odds, fav_odds = result

                won = did_win(game["id"], underdog, scores_lookup)
                if won is None:
                    continue  # no result yet

                pnl = calc_pnl(ud_odds, won)
                bet = {
                    "date":     game_date,
                    "season":   season,
                    "day":      "Sunday",
                    "game_id":  game["id"],
                    "underdog": underdog,
                    "favorite": favorite,
                    "odds":     ud_odds,
                    "won":      won,
                    "pnl":      pnl,
                }
                all_bets.append(bet)
                if ud_odds >= 100:
                    sunday_bets.append(bet)
                day_count += 1

            print(f"{day_count} games")
            time.sleep(0.5)  # be nice to the API

        # Pull a sample of non-Sunday days for comparison
        # (every 7th day to avoid too many API calls)
        non_sundays = [d for d in all_days if d.weekday() != 6]
        sample_days = non_sundays[::7]  # every 7th non-Sunday

        print(f"  Pulling {len(sample_days)} sample non-Sunday days for comparison...")
        for game_date in sample_days:
            games       = fetch_historical_odds(game_date)
            scores_data = fetch_scores(game_date)
            scores_lookup = {s["id"]: s for s in scores_data}

            for game in games:
                result = extract_underdog(game)
                if not result:
                    continue
                underdog, favorite, ud_odds, fav_odds = result
                won = did_win(game["id"], underdog, scores_lookup)
                if won is None:
                    continue
                if ud_odds >= 100:
                    all_day_bets.append({
                        "date":     game_date,
                        "season":   season,
                        "day":      game_date.strftime("%A"),
                        "underdog": underdog,
                        "odds":     ud_odds,
                        "won":      won,
                        "pnl":      calc_pnl(ud_odds, won),
                    })
            time.sleep(0.5)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    analyze_bets(sunday_bets, "Sunday Underdogs (+100 or better)")
    analyze_bets(all_day_bets, "Non-Sunday Underdogs (+100 or better) — sample")
    analyze_bets(all_bets, "All Sunday teams (including favorites as underdogs)")

    # Odds breakdown
    if sunday_bets:
        print("\n  Sunday Underdogs by Odds Range:")
        buckets = [
            ("+100 to +130", 100, 130),
            ("+131 to +160", 131, 160),
            ("+161 to +200", 161, 200),
            ("+201 to +250", 201, 250),
            ("+251+",        251, 9999),
        ]
        for label, lo, hi in buckets:
            sub = [b for b in sunday_bets if lo <= b["odds"] <= hi]
            if not sub: continue
            wins = sum(1 for b in sub if b["won"])
            pnl  = round(sum(b["pnl"] for b in sub), 2)
            roi  = round(pnl / len(sub) * 100, 1)
            print(f"    {label:<18}: {wins}/{len(sub)} wins, P&L {'+' if pnl >= 0 else ''}{pnl}, ROI {'+' if roi >= 0 else ''}{roi}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
