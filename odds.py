import os
import json
import time
import unicodedata
from typing import List

import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
SPORT        = "baseball_mlb"
ODDS_FORMAT  = "american"
MARKETS      = "batter_home_runs"


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def get_today_events() -> List[dict]:
    url    = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events"
    params = {"apiKey": ODDS_API_KEY}
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        print(f"Found {len(events)} MLB events from The Odds API")
        return events
    except Exception as e:
        print(f"Failed to fetch events: {e}")
        return []


def debug_first_event(event_id: str, home_team: str, away_team: str) -> None:
    """Print full raw API response for one game to diagnose missing books."""
    url    = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us,us2",
        "markets":    "h2h",
        "oddsFormat": ODDS_FORMAT,
    }


    print(f"\n=== DEBUG: {away_team} @ {home_team} ===")
    print(f"URL: {url}")
    print(f"Params: {params}")

    try:
        resp = requests.get(url, params=params, timeout=15)
        print(f"Status: {resp.status_code}")
        print(f"Remaining requests header: {resp.headers.get('x-requests-remaining', 'N/A')}")
        print(f"Used requests header: {resp.headers.get('x-requests-used', 'N/A')}")
        data = resp.json()

        bookmakers = data.get("bookmakers", [])
        print(f"Total bookmakers in response: {len(bookmakers)}")
        for bm in bookmakers:
            markets = [m.get("key") for m in bm.get("markets", [])]
            print(f"  {bm.get('key')}: markets={markets}")

    except Exception as e:
        print(f"Debug failed: {e}")


def get_hr_odds_for_event(event_id: str, home_team: str, away_team: str) -> List[dict]:
    url    = f"https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    "us,us2",
        "markets":    MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  Failed for {away_team} @ {home_team}: {e}")
        return []

    rows = []
    for bookmaker in data.get("bookmakers", []):
        book_key = bookmaker.get("key", "")
        for market in bookmaker.get("markets", []):
            if market.get("key") != "batter_home_runs":
                continue
            for outcome in market.get("outcomes", []):
                player_name = outcome.get("description") or outcome.get("name", "")
                price       = outcome.get("price")
                if player_name and price is not None:
                    rows.append({
                        "player_name": player_name,
                        "odds":        int(price),
                        "bookmaker":   book_key,
                        "home_team":   home_team,
                        "away_team":   away_team,
                        "event_id":    event_id,
                    })

    if rows:
        books_found = list({r["bookmaker"] for r in rows})
        print(f"  ✓ {away_team} @ {home_team}: {len(set(r['player_name'] for r in rows))} players, {len(books_found)} books: {books_found}")
    else:
        print(f"  ✗ {away_team} @ {home_team}: no props found")

    return rows


def build_odds_table(events: List[dict]) -> pd.DataFrame:
    all_rows = []

    for event in events:
        event_id  = event.get("id", "")
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        rows      = get_hr_odds_for_event(event_id, home_team, away_team)
        all_rows.extend(rows)
        time.sleep(0.3)

    if not all_rows:
        print("No HR prop odds found across any bookmaker.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Filter out outlier odds above +3000 — novelty lines
    df = df[df["odds"] <= 3000].copy()

    if df.empty:
        print("All odds were filtered out as outliers.")
        return pd.DataFrame()

    # Build consensus odds per player
    best_odds = (
        df.groupby("player_name")
        .agg(
            consensus_odds=("odds", lambda x: int(pd.Series(x).median())),
            best_odds=("odds", "max"),
            worst_odds=("odds", "min"),
            num_books=("bookmaker", "nunique"),
            bookmakers=("bookmaker", lambda x: ",".join(sorted(set(x)))),
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
        )
        .reset_index()
    )

    best_odds["player_name_norm"] = best_odds["player_name"].apply(normalize_name)

    def implied_prob(odds: int) -> float:
        if odds > 0:
            return round(100 / (odds + 100) * 100, 2)
        else:
            return round(abs(odds) / (abs(odds) + 100) * 100, 2)

    best_odds["implied_prob_pct"] = best_odds["consensus_odds"].apply(implied_prob)

    # Sort by implied probability descending (chalk first)
    best_odds = best_odds.sort_values("implied_prob_pct", ascending=False).reset_index(drop=True)

    num_books = df["bookmaker"].nunique()
    print(f"\nBuilt odds table: {len(best_odds)} players across {num_books} bookmakers")
    print(f"Books found: {sorted(df['bookmaker'].unique())}")
    print("\nTop 20 by implied probability:")
    print(best_odds[["player_name", "consensus_odds", "implied_prob_pct", "num_books", "bookmakers"]].head(20).to_string(index=False))

    return best_odds


def write_dataframe_to_sheet(
    gc: gspread.Client,
    sheet_id: str,
    worksheet_name: str,
    df: pd.DataFrame,
) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=500, cols=12)

    df     = df.copy()
    df     = df.replace([np.inf, -np.inf], np.nan).fillna("")
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not set — skipping odds fetch.")
        return

    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Fetching today's MLB events from The Odds API...")
    events = get_today_events()

    if events:
        first = events[0]
        debug_first_event(
            first.get("id", ""),
            first.get("home_team", ""),
            first.get("away_team", "")
        )

    if not events:
        print("No events found — odds sheet will not be updated.")
        return

    print(f"Fetching HR prop odds for {len(events)} games...")
    odds_df = build_odds_table(events)

    if odds_df.empty:
        print("No odds data to write.")
        return

    write_dataframe_to_sheet(gc, sheet_id, "HR_Odds", odds_df)
    print(f"\nWritten {len(odds_df)} player odds to HR_Odds sheet")


if __name__ == "__main__":
    main()

