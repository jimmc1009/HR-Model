import os
import json
import time
import unicodedata
from datetime import date
from typing import Dict, List

import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

EXCLUDED_BOOKS = {"fliff", "espnbet"}

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
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def get_mlb_events(api_key: str) -> List[dict]:
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=15)
    resp.raise_for_status()
    events = resp.json()
    seen = set()
    unique = []
    for e in events:
        if e["id"] not in seen:
            seen.add(e["id"])
            unique.append(e)
    print(f"Found {len(unique)} unique MLB events (from {len(events)} total)")
    return unique


def fetch_props_for_event(
    api_key: str,
    event_id: str,
    event_label: str,
    market: str,
    delay: float = 0.3,
) -> List[dict]:
    url = f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
    params = {
        "apiKey":      api_key,
        "regions":     "us,us2",
        "markets":     market,
        "oddsFormat":  "american",
    }
    time.sleep(delay)
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        bookmakers = resp.json().get("bookmakers", [])
        if bookmakers:
            print(f"    Books returned for {event_label}: {[b['key'] for b in bookmakers]}")
        return bookmakers
    except Exception as e:
        print(f"  ✗ {event_label}: {e}")
        return []


def build_hr_odds(events: List[dict], api_key: str) -> pd.DataFrame:
    print(f"\nFetching HR prop odds for {len(events)} games...")
    
    # Collect per-player, per-book odds across all games
    # Key: player_name_norm, Value: dict of book_key -> best odds
    player_book_odds: Dict[str, Dict[str, int]] = {}
    player_display_names: Dict[str, str] = {}
    player_home_teams: Dict[str, str] = {}
    player_away_teams: Dict[str, str] = {}

    for event in events:
        event_id  = event["id"]
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        label     = f"{away_team} @ {home_team}"

        bookmakers = fetch_props_for_event(api_key, event_id, label, "batter_home_runs")

        if not bookmakers:
            print(f"  ✗ {label}: no props found")
            continue

        book_keys  = [b["key"] for b in bookmakers if b["key"] not in EXCLUDED_BOOKS]
        found_players = set()

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
                    if player and price is not None:
                        try:
                            price_int = int(float(price))
                            if price_int > 0:
                                price_int = min(price_int, 3000)
                                norm = normalize_name(player)
                                # Only store one odds entry per book per player
                                if norm not in player_book_odds:
                                    player_book_odds[norm]     = {}
                                    player_display_names[norm] = player
                                    player_home_teams[norm]    = home_team
                                    player_away_teams[norm]    = away_team
                                # Only add this book if not already seen for this player
                                if book["key"] not in player_book_odds[norm]:
                                    # Cap at 1000 — anything higher is likely a 2+ HR market
                                    if price_int <= 1000:
                                        player_book_odds[norm][book["key"]] = price_int
                                found_players.add(norm)
                        except (ValueError, TypeError):
                            pass

        if found_players:
            print(f"  ✓ {label}: {len(found_players)} players, {len(book_keys)} books: {book_keys}")
        else:
            print(f"  ✗ {label}: no props found")

    if not player_book_odds:
        return pd.DataFrame()

    all_rows = []
    for norm, book_odds in player_book_odds.items():
        odds_list = list(book_odds.values())
        if not odds_list:
            continue
        consensus = int(np.median(odds_list))
        implied   = round(100 / (consensus + 100) * 100, 2)
        all_rows.append({
            "player_name":       player_display_names[norm],
            "player_name_norm":  norm,
            "home_team":         player_home_teams[norm],
            "away_team":         player_away_teams[norm],
            "consensus_odds":    consensus,
            "implied_prob_pct":  implied,
            "num_books":         len(odds_list),
            "book_detail":       str(book_odds),  # shows each book and its odds
        })

    df = pd.DataFrame(all_rows)
    df = df.sort_values("implied_prob_pct", ascending=False).reset_index(drop=True)

    print(f"\nBuilt HR odds table: {len(df)} players across {df['num_books'].max() if not df.empty else 0} bookmakers")
    top20 = df.head(20)[["player_name", "consensus_odds", "implied_prob_pct", "num_books"]]
    print(f"\nTop 20 by implied probability:\n{top20.to_string(index=False)}\n")
    return df


def build_ks_odds(events: List[dict], api_key: str) -> pd.DataFrame:
    print(f"\nFetching pitcher strikeout odds for {len(events)} games...")
    all_rows = []

    for event in events:
        event_id  = event["id"]
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        label     = f"{away_team} @ {home_team}"

        bookmakers = fetch_props_for_event(api_key, event_id, label, "pitcher_strikeouts")

        if not bookmakers:
            print(f"  ✗ {label}: no K props found")
            continue

        book_keys = [b["key"] for b in bookmakers if b["key"] not in EXCLUDED_BOOKS]

        # K props return over/under with a point (line)
        # Structure: {"name": "Over", "description": "Paul Skenes", "price": -130, "point": 6.5}
        pitcher_lines: Dict[str, Dict] = {}

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

                    key = f"{pitcher}|{point}"
                    if key not in pitcher_lines:
                        pitcher_lines[key] = {
                            "pitcher":    pitcher,
                            "line":       float(point),
                            "over_odds":  [],
                            "under_odds": [],
                        }

                    try:
                        price_int = int(float(price))
                        if side == "over":
                            pitcher_lines[key]["over_odds"].append(price_int)
                        elif side == "under":
                            pitcher_lines[key]["under_odds"].append(price_int)
                    except (ValueError, TypeError):
                        pass

        if pitcher_lines:
            print(f"  ✓ {label}: {len(pitcher_lines)} pitcher/line combos, {len(book_keys)} books")
            for key, data in pitcher_lines.items():
                over_odds  = data["over_odds"]
                under_odds = data["under_odds"]
                if not over_odds:
                    continue

                consensus_over  = int(np.median(over_odds))
                consensus_under = int(np.median(under_odds)) if under_odds else None

                # Implied prob from over odds
                if consensus_over >= 0:
                    implied = round(100 / (consensus_over + 100) * 100, 2)
                else:
                    implied = round(abs(consensus_over) / (abs(consensus_over) + 100) * 100, 2)

                all_rows.append({
                    "pitcher_name":      data["pitcher"],
                    "pitcher_name_norm": normalize_name(data["pitcher"]),
                    "home_team":         home_team,
                    "away_team":         away_team,
                    "k_line":            data["line"],
                    "over_odds":         consensus_over,
                    "under_odds":        consensus_under,
                    "implied_prob_over": implied,
                    "num_books":         len(over_odds),
                })
        else:
            print(f"  ✗ {label}: no K props found")

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Keep the most common line per pitcher (books usually agree)
    df = (
        df.sort_values("num_books", ascending=False)
        .drop_duplicates(subset=["pitcher_name_norm"], keep="first")
        .sort_values("implied_prob_over", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nBuilt KS odds table: {len(df)} pitchers")
    print(df[["pitcher_name", "k_line", "over_odds", "under_odds", "num_books"]].head(15).to_string(index=False))
    return df


def build_hrrbi_odds(events: List[dict], api_key: str) -> pd.DataFrame:
    """
    Attempts to pull batter_hits_runs_rbis market.
    Falls back gracefully if unavailable on free tier.
    """
    print(f"\nFetching H+R+RBI prop odds for {len(events)} games...")
    all_rows = []
    found_any = False

    for event in events:
        event_id  = event["id"]
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        label     = f"{away_team} @ {home_team}"

        bookmakers = fetch_props_for_event(
            api_key, event_id, label, "batter_hits_runs_rbis"
        )

        if not bookmakers:
            continue

        player_lines: Dict[str, Dict] = {}

        for book in bookmakers:
            if book["key"] in EXCLUDED_BOOKS:
                continue
            for market in book.get("markets", []):
                if market["key"] != "batter_hits_runs_rbis":
                    continue
                for outcome in market.get("outcomes", []):
                    side   = outcome.get("name", "").lower()
                    player = outcome.get("description", "").strip()
                    price  = outcome.get("price")
                    point  = outcome.get("point")

                    if not player or price is None or point is None:
                        continue

                    key = f"{player}|{point}"
                    if key not in player_lines:
                        player_lines[key] = {
                            "player":     player,
                            "line":       float(point),
                            "over_odds":  [],
                            "under_odds": [],
                        }
                    try:
                        price_int = int(float(price))
                        if side == "over":
                            player_lines[key]["over_odds"].append(price_int)
                        elif side == "under":
                            player_lines[key]["under_odds"].append(price_int)
                    except (ValueError, TypeError):
                        pass

        if player_lines:
            found_any = True
            for key, data in player_lines.items():
                if not data["over_odds"]:
                    continue
                consensus_over  = int(np.median(data["over_odds"]))
                consensus_under = int(np.median(data["under_odds"])) if data["under_odds"] else None

                if consensus_over >= 0:
                    implied = round(100 / (consensus_over + 100) * 100, 2)
                else:
                    implied = round(abs(consensus_over) / (abs(consensus_over) + 100) * 100, 2)

                all_rows.append({
                    "player_name":       data["player"],
                    "player_name_norm":  normalize_name(data["player"]),
                    "home_team":         home_team,
                    "away_team":         away_team,
                    "hrrbi_line":        data["line"],
                    "over_odds":         consensus_over,
                    "under_odds":        consensus_under,
                    "implied_prob_over": implied,
                    "num_books":         len(data["over_odds"]),
                })

    if not found_any:
        print("  INFO: batter_hits_runs_rbis market not available — H+R+RBI odds will be empty")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = (
        df.sort_values("num_books", ascending=False)
        .drop_duplicates(subset=["player_name_norm"], keep="first")
        .sort_values("implied_prob_over", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nBuilt H+R+RBI odds table: {len(df)} players")
    return df


def write_sheet(
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
        ws = sh.add_worksheet(title=worksheet_name, rows=500, cols=20)

    df = df.copy().fillna("").replace([np.inf, -np.inf], "")
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    api_key  = os.environ["ODDS_API_KEY"]
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Fetching today's MLB events from The Odds API...")
    events = get_mlb_events(api_key)

    # ── HR props ──────────────────────────────────────────────────
    hr_df = build_hr_odds(events, api_key)
    if not hr_df.empty:
        write_sheet(gc, sheet_id, "HR_Odds", hr_df)
        print(f"Written {len(hr_df)} player odds to HR_Odds sheet")
    else:
        print("No HR odds data to write.")

    # ── Pitcher K props ───────────────────────────────────────────
    ks_df = build_ks_odds(events, api_key)
    if not ks_df.empty:
        write_sheet(gc, sheet_id, "KS_Odds", ks_df)
        print(f"Written {len(ks_df)} pitcher K odds to KS_Odds sheet")
    else:
        print("No KS odds data to write.")

    # ── H+R+RBI props ─────────────────────────────────────────────
    hrrbi_df = build_hrrbi_odds(events, api_key)
    if not hrrbi_df.empty:
        write_sheet(gc, sheet_id, "HRRBI_Odds", hrrbi_df)
        print(f"Written {len(hrrbi_df)} player H+R+RBI odds to HRRBI_Odds sheet")
    else:
        print("No H+R+RBI odds data to write — sheet not created.")


if __name__ == "__main__":
    main()
