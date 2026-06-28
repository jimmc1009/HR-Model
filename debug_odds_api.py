"""
debug_odds_api.py
Quick debug to see what the historical odds and scores endpoints
actually return for a known Sunday with MLB games.
"""

import os
import json
import requests

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORT        = "baseball_mlb"
BASE_URL     = "https://api.the-odds-api.com/v4"

# Test with a known Sunday that had MLB games
TEST_DATE = "2025-06-22"

def main():
    if not ODDS_API_KEY:
        print("ERROR: ODDS_API_KEY not set")
        return

    # Test scores endpoint
    print(f"=== SCORES for {TEST_DATE} ===")
    scores_url = (
        f"{BASE_URL}/historical/sports/{SPORT}/scores"
        f"?apiKey={ODDS_API_KEY}"
        f"&date={TEST_DATE}T23:59:00Z"
    )
    resp = requests.get(scores_url, timeout=20)
    print(f"Status: {resp.status_code}")
    print(f"Remaining requests: {resp.headers.get('x-requests-remaining', 'N/A')}")
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        print(f"Games returned: {len(data)}")
        for g in data[:3]:
            print(f"  ID: {g['id']}")
            print(f"  {g.get('away_team')} @ {g.get('home_team')}")
            print(f"  Completed: {g.get('completed')}")
            print(f"  Scores: {g.get('scores')}")
            print()
    else:
        print(resp.text[:500])

    print()

    # Test odds endpoint
    print(f"=== ODDS for {TEST_DATE} ===")
    odds_url = (
        f"{BASE_URL}/historical/sports/{SPORT}/odds"
        f"?apiKey={ODDS_API_KEY}"
        f"&regions=us"
        f"&markets=h2h"
        f"&oddsFormat=american"
        f"&date={TEST_DATE}T16:00:00Z"
    )
    resp2 = requests.get(odds_url, timeout=20)
    print(f"Status: {resp2.status_code}")
    print(f"Remaining requests: {resp2.headers.get('x-requests-remaining', 'N/A')}")
    if resp2.status_code == 200:
        data2 = resp2.json().get("data", [])
        print(f"Games returned: {len(data2)}")
        for g in data2[:3]:
            print(f"  ID: {g['id']}")
            print(f"  {g.get('away_team')} @ {g.get('home_team')}")
            bks = g.get("bookmakers", [])
            if bks:
                dk = next((b for b in bks if b["key"] == "draftkings"), bks[0])
                h2h = next((m for m in dk.get("markets", []) if m["key"] == "h2h"), None)
                if h2h:
                    for o in h2h.get("outcomes", []):
                        print(f"    {o['name']}: {o['price']}")
            print()
    else:
        print(resp2.text[:500])

    print("Done.")

if __name__ == "__main__":
    main()
