"""
check_sunday_underdogs.py
Pulls historical MLB moneylines for every Sunday in 2025 and 2026 seasons
using The Odds API. Calculates P&L for betting every underdog (+100 or better).
Also compares to non-Sunday days for context.
"""

import os
import time
from datetime import date, timedelta

import requests

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORT        = "baseball_mlb"
BASE_URL     = "https://api.the-odds-api.com/v4"

SEASONS = {
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 26), date(2026, 6, 26)),
}


def get_days_of_week(season_start, season_end, weekday):
    """Return all dates matching weekday (0=Mon, 6=Sun) in the season."""
    days = []
    d = season_start
    while d <= season_end:
        if d.weekday() == weekday:
            days.append(d)
        d += timedelta(days=1)
    return days


def fetch_odds_and_results(game_date: date) -> list:
    """
    Use the historical scores endpoint which includes completed results
    and then fetch odds for the same snapshot.
    Returns list of (underdog, odds, won) tuples.
    """
    # Step 1: Get scores/results for this date
    scores_url = (
        f"{BASE_URL}/historical/sports/{SPORT}/scores"
        f"?apiKey={ODDS_API_KEY}"
        f"&date={game_date.strftime('%Y-%m-%d')}T23:59:00Z"
    )
    try:
        resp = requests.get(scores_url, timeout=20)
        if resp.status_code != 200:
            return []
        scores_data = resp.json().get("data", [])
    except Exception as e:
        print(f"    Scores error {game_date}: {e}")
        return []

    if not scores_data:
        return []

    # Build result lookup: game_id -> winner
    results = {}
    for game in scores_data:
        if not game.get("completed"):
            continue
        scores = game.get("scores") or []
        if len(scores) < 2:
            continue
        try:
            score_map = {s["name"]: int(s["score"]) for s in scores}
            winner = max(score_map, key=score_map.get)
            results[game["id"]] = {
                "winner":    winner,
                "home":      game.get("home_team"),
                "away":      game.get("away_team"),
                "home_score": score_map.get(game.get("home_team"), 0),
                "away_score": score_map.get(game.get("away_team"), 0),
            }
        except Exception:
            continue

    if not results:
        return []

    # Step 2: Get odds snapshot for this date (noon ET = 16:00 UTC)
    odds_url = (
        f"{BASE_URL}/historical/sports/{SPORT}/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions=us"
        f"&markets=h2h"
        f"&oddsFormat=american"
        f"&date={game_date.strftime('%Y-%m-%d')}T16:00:00Z"
    )
    try:
        resp = requests.get(odds_url, timeout=20)
        if resp.status_code != 200:
            return []
        odds_data = resp.json().get("data", [])
    except Exception as e:
        print(f"    Odds error {game_date}: {e}")
        return []

    bets = []
    for game in odds_data:
        game_id = game["id"]
        if game_id not in results:
            continue

        result = results[game_id]
        winner = result["winner"]

        # Get odds from DraftKings or first bookmaker
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        bk = next((b for b in bookmakers if b["key"] == "draftkings"), bookmakers[0])
        markets = bk.get("markets", [])
        h2h = next((m for m in markets if m["key"] == "h2h"), None)
        if not h2h:
            continue

        outcomes = h2h.get("outcomes", [])
        if len(outcomes) < 2:
            continue

        team_odds = {o["name"]: o["price"] for o in outcomes}
        teams = list(team_odds.keys())

        # Identify underdog — higher odds value
        if team_odds[teams[0]] >= team_odds[teams[1]]:
            underdog, favorite = teams[0], teams[1]
        else:
            underdog, favorite = teams[1], teams[0]

        ud_odds  = team_odds[underdog]
        fav_odds = team_odds[favorite]
        won      = (winner == underdog)

        bets.append({
            "date":     game_date,
            "underdog": underdog,
            "favorite": favorite,
            "ud_odds":  ud_odds,
            "fav_odds": fav_odds,
            "won":      won,
        })

    return bets


def calc_pnl(odds, won, stake=1.0):
    if won:
        return stake * (odds / 100) if odds >= 0 else stake * (100 / abs(odds))
    return -stake


def print_analysis(bets, label):
    if not bets:
        print(f"\n  {label}: No bets")
        return
    total    = len(bets)
    wins     = sum(1 for b in bets if b["won"])
    win_rate = round(wins / total * 100, 1)
    pnl      = round(sum(b["pnl"] for b in bets), 2)
    roi      = round(pnl / total * 100, 1)
    avg_odds = round(sum(b["ud_odds"] for b in bets) / total)
    print(f"\n  {label}:")
    print(f"    Bets: {total} | Wins: {wins} | Win Rate: {win_rate}%")
    print(f"    Avg Odds: +{avg_odds}")
    print(f"    P&L ($1 flat): {'+'if pnl>=0 else ''}{pnl}")
    print(f"    ROI: {'+'if roi>=0 else ''}{roi}%")

    # By season
    for season in [2025, 2026]:
        sub = [b for b in bets if b["season"] == season]
        if not sub: continue
        sw   = sum(1 for b in sub if b["won"])
        sp   = round(sum(b["pnl"] for b in sub), 2)
        sr   = round(sp / len(sub) * 100, 1)
        print(f"    {season}: {sw}/{len(sub)} wins, P&L {'+'if sp>=0 else ''}{sp}, ROI {'+'if sr>=0 else ''}{sr}%")


def main():
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        return

    print("MLB Sunday Underdog Moneyline Analysis")
    print("="*60)
    print("Using The Odds API historical endpoint")
    print("Betting every underdog (+100 or better) $1 flat\n")

    sunday_bets     = []  # underdogs at +100 or better on Sundays
    all_sunday_bets = []  # all underdogs on Sundays (any odds)
    weekday_bets    = []  # underdogs at +100 or better on non-Sundays (sample)

    for season, (season_start, season_end) in SEASONS.items():
        sundays = get_days_of_week(season_start, season_end, 6)  # Sunday=6
        print(f"{season}: {len(sundays)} Sundays to process")

        for game_date in sundays:
            print(f"  {game_date}...", end=" ", flush=True)
            bets = fetch_odds_and_results(game_date)
            count = 0
            for b in bets:
                b["season"] = season
                b["pnl"]    = calc_pnl(b["ud_odds"], b["won"])
                all_sunday_bets.append(b)
                if b["ud_odds"] >= 100:
                    sunday_bets.append(b)
                    count += 1
            print(f"{count} underdog bets")
            time.sleep(0.3)

        # Sample of non-Sunday days for comparison (every 14th day)
        non_sundays = get_days_of_week(season_start, season_end, 1)  # Mondays
        sample = non_sundays[::2]  # every other Monday
        print(f"  Pulling {len(sample)} non-Sunday days for comparison...")
        for game_date in sample:
            bets = fetch_odds_and_results(game_date)
            for b in bets:
                b["season"] = season
                b["pnl"]    = calc_pnl(b["ud_odds"], b["won"])
                if b["ud_odds"] >= 100:
                    weekday_bets.append(b)
            time.sleep(0.3)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print_analysis(sunday_bets, "Sunday Underdogs (+100 or better)")
    print_analysis(weekday_bets, "Monday sample — Underdogs (+100 or better)")
    print_analysis(all_sunday_bets, "All Sunday underdogs (any odds)")

    # Odds buckets for Sunday
    if sunday_bets:
        print("\n  Sunday Underdogs by Odds Range:")
        buckets = [
            ("+100-130", 100, 130),
            ("+131-160", 131, 160),
            ("+161-200", 161, 200),
            ("+201-250", 201, 250),
            ("+251+",    251, 9999),
        ]
        for label, lo, hi in buckets:
            sub = [b for b in sunday_bets if lo <= b["ud_odds"] <= hi]
            if not sub: continue
            wins = sum(1 for b in sub if b["won"])
            pnl  = round(sum(b["pnl"] for b in sub), 2)
            roi  = round(pnl / len(sub) * 100, 1)
            print(f"    {label:<12}: {wins}/{len(sub)} wins, P&L {'+'if pnl>=0 else ''}{pnl}, ROI {'+'if roi>=0 else ''}{roi}%")

    # Best and worst Sunday underdog teams
    if sunday_bets:
        print("\n  Sunday Underdog Win Rate by Team (5+ bets):")
        from collections import defaultdict
        team_stats = defaultdict(list)
        for b in sunday_bets:
            team_stats[b["underdog"]].append(b)
        team_results = []
        for team, tbets in team_stats.items():
            if len(tbets) < 5: continue
            wins = sum(1 for b in tbets if b["won"])
            pnl  = round(sum(b["pnl"] for b in tbets), 2)
            team_results.append((team, wins, len(tbets), round(wins/len(tbets)*100,1), pnl))
        team_results.sort(key=lambda x: x[3], reverse=True)
        for team, wins, n, wr, pnl in team_results:
            print(f"    {team:<25}: {wins}/{n} = {wr}%, P&L {'+'if pnl>=0 else ''}{pnl}")

    print("\nDone.")


if __name__ == "__main__":
    main()
