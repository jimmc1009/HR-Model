"""
check_sunday_underdogs.py
Pulls historical MLB moneylines from The Odds API and game results from
the MLB Stats API (free). Calculates P&L for betting every underdog
(+100 or better) $1 flat on Sundays. Also compares to other days.
"""

import os
import time
from datetime import date, timedelta
from collections import defaultdict

import requests

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORT        = "baseball_mlb"
BASE_URL     = "https://api.the-odds-api.com/v4"

SEASONS = {
    2025: (date(2025, 3, 27), date(2025, 9, 28)),
    2026: (date(2026, 3, 26), date(2026, 6, 26)),
}


def get_days_of_week(season_start, season_end, weekday):
    days = []
    d = season_start
    while d <= season_end:
        if d.weekday() == weekday:
            days.append(d)
        d += timedelta(days=1)
    return days


def fetch_mlb_results(game_date: date) -> dict:
    """
    Fetch game results from MLB Stats API.
    Returns dict of {(away_team_abbr, home_team_abbr): winner_abbr}
    and also {(away_name, home_name): winner_name}
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={game_date.strftime('%Y-%m-%d')}"
        f"&hydrate=linescore&gameType=R"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    MLB API error {game_date}: {e}")
        return {}

    results = {}
    for d in data.get("dates", []):
        for game in d.get("games", []):
            status = game.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            home = game.get("teams", {}).get("home", {})
            away = game.get("teams", {}).get("away", {})
            home_name  = home.get("team", {}).get("name", "")
            away_name  = away.get("team", {}).get("name", "")
            home_score = home.get("score", 0)
            away_score = away.get("score", 0)
            if home_score is None or away_score is None:
                continue
            winner = home_name if home_score > away_score else away_name
            # Store by both team names for flexible matching
            results[(away_name, home_name)] = winner
            results[(home_name, away_name)] = winner
    return results


def fetch_odds(game_date: date) -> list:
    """Fetch odds snapshot for a date. Returns list of game dicts."""
    url = (
        f"{BASE_URL}/historical/sports/{SPORT}/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions=us"
        f"&markets=h2h"
        f"&oddsFormat=american"
        f"&date={game_date.strftime('%Y-%m-%d')}T16:00:00Z"
    )
    try:
        resp = requests.get(url, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("data", [])
        return []
    except Exception as e:
        print(f"    Odds API error {game_date}: {e}")
        return []


def fuzzy_match_winner(away_team: str, home_team: str, results: dict) -> str | None:
    """
    Match Odds API team names to MLB Stats API team names.
    MLB uses full names like 'New York Yankees', Odds API uses same format.
    Try exact match first, then partial.
    """
    # Exact match
    winner = results.get((away_team, home_team)) or results.get((home_team, away_team))
    if winner:
        return winner

    # Partial match — check if any result key contains both team city/name parts
    for (t1, t2), w in results.items():
        away_words = set(away_team.lower().split())
        home_words = set(home_team.lower().split())
        t1_words   = set(t1.lower().split())
        t2_words   = set(t2.lower().split())

        if (away_words & t1_words and home_words & t2_words) or \
           (away_words & t2_words and home_words & t1_words):
            return w

    return None


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
    for season in [2025, 2026]:
        sub = [b for b in bets if b["season"] == season]
        if not sub: continue
        sw = sum(1 for b in sub if b["won"])
        sp = round(sum(b["pnl"] for b in sub), 2)
        sr = round(sp / len(sub) * 100, 1)
        print(f"    {season}: {sw}/{len(sub)} wins, P&L {'+'if sp>=0 else ''}{sp}, ROI {'+'if sr>=0 else ''}{sr}%")


def process_day(game_date, season):
    """Process one day — return list of bet dicts."""
    results = fetch_mlb_results(game_date)
    if not results:
        return []

    odds_games = fetch_odds(game_date)
    if not odds_games:
        return []

    bets = []
    for game in odds_games:
        away = game.get("away_team", "")
        home = game.get("home_team", "")

        # Get DraftKings or first bookmaker odds
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue
        bk  = next((b for b in bookmakers if b["key"] == "draftkings"), bookmakers[0])
        h2h = next((m for m in bk.get("markets", []) if m["key"] == "h2h"), None)
        if not h2h:
            continue

        outcomes = h2h.get("outcomes", [])
        if len(outcomes) < 2:
            continue

        team_odds = {o["name"]: o["price"] for o in outcomes}
        teams     = list(team_odds.keys())

        # Identify underdog
        if team_odds[teams[0]] >= team_odds[teams[1]]:
            underdog, favorite = teams[0], teams[1]
        else:
            underdog, favorite = teams[1], teams[0]

        ud_odds  = team_odds[underdog]
        fav_odds = team_odds[favorite]

        # Get result
        winner = fuzzy_match_winner(away, home, results)
        if not winner:
            continue

        won = (winner == underdog) or (underdog in winner) or (winner in underdog)

        bets.append({
            "date":     game_date,
            "season":   season,
            "underdog": underdog,
            "favorite": favorite,
            "ud_odds":  ud_odds,
            "fav_odds": fav_odds,
            "won":      won,
            "pnl":      calc_pnl(ud_odds, won),
        })

    return bets


def main():
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        return

    print("MLB Sunday Underdog Moneyline Analysis")
    print("="*60)
    print("Odds: The Odds API | Results: MLB Stats API")
    print("Betting every underdog (+100 or better) $1 flat\n")

    sunday_bets  = []
    weekday_bets = []

    for season, (season_start, season_end) in SEASONS.items():
        sundays = get_days_of_week(season_start, season_end, 6)
        print(f"{season}: {len(sundays)} Sundays")

        for game_date in sundays:
            print(f"  {game_date}...", end=" ", flush=True)
            bets = process_day(game_date, season)
            count = 0
            for b in bets:
                if b["ud_odds"] >= 100:
                    sunday_bets.append(b)
                    count += 1
            print(f"{count} underdog bets ({len(bets)} total games matched)")
            time.sleep(0.3)

        # Comparison: Tuesdays (sample)
        tuesdays = get_days_of_week(season_start, season_end, 1)[::3]
        print(f"  Comparison: {len(tuesdays)} Tuesday samples")
        for game_date in tuesdays:
            bets = process_day(game_date, season)
            for b in bets:
                if b["ud_odds"] >= 100:
                    weekday_bets.append(b)
            time.sleep(0.3)

    # ── Results ───────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print_analysis(sunday_bets, "Sunday Underdogs (+100 or better)")
    print_analysis(weekday_bets, "Tuesday sample — Underdogs (+100 or better)")

    # Odds buckets
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

    # Team breakdown
    if sunday_bets:
        print("\n  Best Sunday Underdog Teams (5+ bets):")
        team_stats = defaultdict(list)
        for b in sunday_bets:
            team_stats[b["underdog"]].append(b)
        team_results = [
            (t, sum(1 for b in v if b["won"]), len(v),
             round(sum(1 for b in v if b["won"])/len(v)*100,1),
             round(sum(b["pnl"] for b in v),2))
            for t, v in team_stats.items() if len(v) >= 5
        ]
        team_results.sort(key=lambda x: x[4], reverse=True)
        for team, wins, n, wr, pnl in team_results[:15]:
            print(f"    {team:<28}: {wins}/{n} = {wr}%, P&L {'+'if pnl>=0 else ''}{pnl}")

    print("\nDone.")


if __name__ == "__main__":
    main()
