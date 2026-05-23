"""
hr_odds_backtest.py
One-time script to pull FanDuel historical HR prop odds from The Odds API,
match to actual HR outcomes from Statcast, and write analysis to Google Sheets.
Supports resume — skips dates already stored in HR_Odds_History.
"""

import os
import json
import time
import unicodedata
from datetime import date, timedelta

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
from pybaseball import statcast
import requests

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

ODDS_API_KEY    = os.environ["ODDS_API_KEY"]
SPORT           = "baseball_mlb"
REGIONS         = "us"
BOOKMAKER       = "fanduel"
MARKETS         = ["batter_home_runs", "batter_home_runs_alternate"]
SEASON_2025_START = "2025-03-27"
SEASON_2025_END   = "2025-09-28"
SEASON_2026_START = "2026-03-26"
SEASON_2026_END   = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

HISTORY_SHEET   = "HR_Odds_History"
ANALYSIS_SHEET  = "HR_Odds_Analysis"


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


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def read_sheet_raw(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    try:
        sh         = with_retry(lambda: gc.open_by_key(sheet_id))
        ws         = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values or len(all_values) < 2:
            return pd.DataFrame()
        headers = all_values[0]
        rows    = all_values[1:]
        df      = pd.DataFrame(rows, columns=headers)
        df      = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
        return df
    except gspread.WorksheetNotFound:
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read '{name}': {e}")
        return pd.DataFrame()


def write_sheet(gc: gspread.Client, sheet_id: str, name: str, df: pd.DataFrame) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(name)
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=name, rows=50000, cols=20)

    df = df.copy().fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.update(
        [df.columns.tolist()] + df.astype(str).values.tolist()
    ))
    print(f"Written {len(df)} rows to {name}")


# ── Odds API ──────────────────────────────────────────────────────────────

def get_historical_events(date_str: str) -> list:
    """Get all MLB event IDs for a given date."""
    # Use noon UTC as the snapshot time for the date
    timestamp = f"{date_str}T18:00:00Z"
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events"
        f"?apiKey={ODDS_API_KEY}&date={timestamp}"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        events = data.get("data", [])
        remaining = resp.headers.get("x-requests-remaining", "?")
        return events, remaining
    except Exception as e:
        print(f"  WARNING: Could not fetch events for {date_str}: {e}")
        return [], "?"


def get_event_odds(event_id: str, date_str: str, market: str) -> list:
    """Get FanDuel odds for a specific event and market."""
    timestamp = f"{date_str}T18:00:00Z"
    url = (
        f"https://api.the-odds-api.com/v4/historical/sports/{SPORT}/events/{event_id}/odds"
        f"?apiKey={ODDS_API_KEY}&date={timestamp}"
        f"&regions={REGIONS}&markets={market}&bookmakers={BOOKMAKER}&oddsFormat=american"
    )
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data      = resp.json()
        remaining = resp.headers.get("x-requests-remaining", "?")
        return data.get("data", {}), remaining
    except Exception as e:
        print(f"    WARNING: Could not fetch odds for event {event_id} market {market}: {e}")
        return {}, "?"


def parse_player_odds(event_data: dict, market: str, game_date: str) -> list:
    """Extract player HR odds from event odds response."""
    rows = []
    if not event_data:
        return rows

    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")

    for bookmaker in event_data.get("bookmakers", []):
        if bookmaker.get("key") != BOOKMAKER:
            continue
        for mkt in bookmaker.get("markets", []):
            if mkt.get("key") != market:
                continue
            for outcome in mkt.get("outcomes", []):
                name        = outcome.get("description", outcome.get("name", ""))
                bet_type    = outcome.get("name", "")  # Over/Under
                price       = outcome.get("price", None)
                point       = outcome.get("point", 0.5)

                if bet_type.lower() != "over":
                    continue
                if not name or price is None:
                    continue

                rows.append({
                    "game_date":  game_date,
                    "home_team":  home_team,
                    "away_team":  away_team,
                    "player_name": name.strip(),
                    "market":     market,
                    "line":       point,
                    "odds":       int(price),
                })

    return rows


def get_date_range(start: str, end: str) -> list:
    dates  = []
    cursor = date.fromisoformat(start)
    end_dt = date.fromisoformat(end)
    while cursor <= end_dt:
        dates.append(cursor.strftime("%Y-%m-%d"))
        cursor += timedelta(days=1)
    return dates


# ── Statcast HR lookup ────────────────────────────────────────────────────

def build_hr_lookup(start: str, end: str) -> set:
    """Returns set of (date_str, normalized_player_name) for all HR events."""
    print(f"Pulling Statcast data {start} → {end}...")
    try:
        df = statcast(start_dt=start, end_dt=end)
        if df is None or df.empty:
            print("  Statcast returned empty.")
            return set()
        print(f"  Pulled {len(df):,} Statcast rows")
    except Exception as e:
        print(f"  Statcast pull failed: {e}")
        return set()

    df["game_date"] = pd.to_datetime(df["game_date"])
    hr_df           = df[df["events"].astype("string").str.lower() == "home_run"].copy()
    hr_df           = hr_df.drop_duplicates(subset=["game_date", "batter"])

    # Look up player names
    batter_ids = hr_df["batter"].dropna().astype(int).unique().tolist()
    name_map   = {}
    print(f"  Looking up {len(batter_ids)} batter names...")
    for i in range(0, len(batter_ids), 50):
        chunk = batter_ids[i:i + 50]
        try:
            resp = requests.get(
                f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}",
                timeout=30
            )
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pid  = person.get("id")
                name = person.get("fullName", "")
                if pid and name:
                    name_map[int(pid)] = name
        except Exception:
            pass
        time.sleep(0.2)

    hr_lookup = set()
    for _, row in hr_df.iterrows():
        pid       = int(row["batter"]) if pd.notna(row["batter"]) else None
        full_name = name_map.get(pid, "")
        if full_name:
            date_str = row["game_date"].strftime("%Y-%m-%d")
            hr_lookup.add((date_str, normalize_name(full_name)))

    print(f"  Built HR lookup: {len(hr_lookup)} player-date HR events")
    return hr_lookup


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    # ── Load existing data for resume support ─────────────────────────────
    print("Checking existing HR_Odds_History for resume...")
    existing = read_sheet_raw(gc, sheet_id, HISTORY_SHEET)
    already_done_dates = set()
    if not existing.empty and "game_date" in existing.columns:
        already_done_dates = set(existing["game_date"].astype(str).str.strip().unique())
        print(f"  Already have data for {len(already_done_dates)} dates — will skip these")
    else:
        existing = pd.DataFrame()

    # ── Build date list ───────────────────────────────────────────────────
    all_dates = (
        get_date_range(SEASON_2025_START, SEASON_2025_END) +
        get_date_range(SEASON_2026_START, SEASON_2026_END)
    )
    dates_to_pull = [d for d in all_dates if d not in already_done_dates]
    print(f"Dates to pull: {len(dates_to_pull)} (skipping {len(already_done_dates)} already done)")

    if not dates_to_pull:
        print("All dates already pulled — skipping to analysis.")
    else:
        # ── Build Statcast HR lookup ──────────────────────────────────────
        # One bulk pull per season to minimize Statcast calls
        hr_lookup = set()

        pull_2025 = [d for d in dates_to_pull if d <= SEASON_2025_END]
        pull_2026 = [d for d in dates_to_pull if d >= SEASON_2026_START]

        if pull_2025:
            hr_lookup |= build_hr_lookup(pull_2025[0], pull_2025[-1])
        if pull_2026:
            hr_lookup |= build_hr_lookup(pull_2026[0], pull_2026[-1])

        # ── Pull odds date by date ────────────────────────────────────────
        new_rows     = []
        total_dates  = len(dates_to_pull)
        credits_used = 0

        for i, game_date in enumerate(dates_to_pull):
            print(f"[{i+1}/{total_dates}] {game_date}...")

            # Get event IDs for this date
            events, remaining = get_historical_events(game_date)
            credits_used += 1
            time.sleep(0.3)

            if not events:
                print(f"  No events found for {game_date}")
                continue

            print(f"  Found {len(events)} events | Credits remaining: {remaining}")

            date_rows = []
            for event in events:
                event_id = event.get("id")
                if not event_id:
                    continue

                for market in MARKETS:
                    event_data, remaining = get_event_odds(event_id, game_date, market)
                    credits_used += 1
                    time.sleep(0.3)

                    player_rows = parse_player_odds(event_data, market, game_date)
                    date_rows.extend(player_rows)

            if not date_rows:
                continue

            # Deduplicate — if player appears in both markets keep better odds
            date_df = pd.DataFrame(date_rows)
            date_df = date_df.sort_values("odds", ascending=False)
            date_df = date_df.drop_duplicates(subset=["game_date", "player_name"], keep="first")

            # Match to HR outcomes
            date_df["player_norm"] = date_df["player_name"].apply(normalize_name)
            date_df["hit_hr"]      = date_df.apply(
                lambda r: "Yes" if (r["game_date"], r["player_norm"]) in hr_lookup else "No",
                axis=1
            )
            date_df = date_df.drop(columns=["player_norm"])

            new_rows.extend(date_df.to_dict("records"))

            # Write checkpoint every 50 dates to avoid losing progress
            if (i + 1) % 50 == 0:
                print(f"  Checkpoint — writing {len(new_rows)} rows so far...")
                checkpoint_df = pd.DataFrame(new_rows)
                if not existing.empty:
                    checkpoint_df = pd.concat([existing, checkpoint_df], ignore_index=True)
                write_sheet(gc, sheet_id, HISTORY_SHEET, checkpoint_df)
                time.sleep(5)

        print(f"\nTotal API credits used this run: ~{credits_used}")

        # ── Final write of history ────────────────────────────────────────
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            if not existing.empty:
                combined = pd.concat([existing, new_df], ignore_index=True)
            else:
                combined = new_df

            # Final dedup across all data
            combined = combined.drop_duplicates(
                subset=["game_date", "player_name"], keep="first"
            )
            write_sheet(gc, sheet_id, HISTORY_SHEET, combined)
        else:
            print("No new odds data retrieved.")
            combined = existing
    
    # ── Analysis ──────────────────────────────────────────────────────────
    print("\nBuilding analysis...")
    if existing.empty and not new_rows:
        print("No data to analyze.")
        return

    if "combined" not in dir() or combined.empty:
        combined = read_sheet_raw(gc, sheet_id, HISTORY_SHEET)

    if combined.empty:
        print("No data in HR_Odds_History to analyze.")
        return

    combined["odds"]    = pd.to_numeric(combined["odds"], errors="coerce")
    combined["hit_hr"]  = combined["hit_hr"].astype(str).str.strip()
    combined["hit_bool"]= combined["hit_hr"] == "Yes"
    combined["game_date"] = pd.to_datetime(combined["game_date"], errors="coerce")
    combined["season"]  = combined["game_date"].dt.year.astype(str)

    scored = combined[
        combined["hit_hr"].isin(["Yes", "No"]) &
        (pd.to_numeric(combined["line"], errors="coerce") == 0.5)
    ].copy()

    if scored.empty:
        print("No scored rows to analyze.")
        return

    def american_to_profit(odds: float) -> float:
        if pd.isna(odds) or odds <= 0:
            return 0.0
        return odds / 100.0

    scored["profit_if_win"] = scored["odds"].apply(american_to_profit)
    scored["unit_result"]   = scored.apply(
        lambda r: r["profit_if_win"] if r["hit_bool"] else -1.0, axis=1
    )

    analysis_rows = []

    def add_section(label):
        analysis_rows.append({
            "Category": label, "Total Props": "", "Hits": "",
            "Hit Rate %": "", "Units Profit": "", "ROI %": "",
            "Avg Odds": "", "_header": True
        })

    def add_row(label, sub):
        if sub.empty:
            return
        total        = len(sub)
        hits         = int(sub["hit_bool"].sum())
        hit_rate     = round(hits / total * 100, 1)
        units_profit = round(sub["unit_result"].sum(), 2)
        roi          = round(units_profit / total * 100, 1)
        avg_odds     = round(sub["odds"].mean(), 0)
        analysis_rows.append({
            "Category":     label,
            "Total Props":  total,
            "Hits":         hits,
            "Hit Rate %":   f"{hit_rate}%",
            "Units Profit": f"+{units_profit}" if units_profit >= 0 else str(units_profit),
            "ROI %":        f"+{roi}%" if roi >= 0 else f"{roi}%",
            "Avg Odds":     f"+{int(avg_odds)}" if avg_odds >= 0 else str(int(avg_odds)),
            "_header":      False,
        })

    # Overall
    add_row("🏆  Overall", scored)

    # By season
    add_section("── By Season ──")
    for season in sorted(scored["season"].dropna().unique()):
        add_row(f"   {season}", scored[scored["season"] == season])

    # By odds tier
    add_section("── By Odds Tier ──")
    for label, sub in [
        ("   +100 to +149", scored[(scored["odds"] >= 100) & (scored["odds"] < 150)]),
        ("   +150 to +199", scored[(scored["odds"] >= 150) & (scored["odds"] < 200)]),
        ("   +200 to +299", scored[(scored["odds"] >= 200) & (scored["odds"] < 300)]),
        ("   +300 to +399", scored[(scored["odds"] >= 300) & (scored["odds"] < 400)]),
        ("   +400 to +499", scored[(scored["odds"] >= 400) & (scored["odds"] < 500)]),
        ("   +500 to +699", scored[(scored["odds"] >= 500) & (scored["odds"] < 700)]),
        ("   +700+",        scored[scored["odds"] >= 700]),
    ]:
        add_row(label, sub)

    # By market
    add_section("── By Market ──")
    for market in scored["market"].dropna().unique():
        add_row(f"   {market}", scored[scored["market"] == market])

    # By month
    add_section("── By Month ──")
    scored["month"] = scored["game_date"].dt.strftime("%Y-%m")
    for month in sorted(scored["month"].dropna().unique()):
        add_row(f"   {month}", scored[scored["month"] == month])

    # Write analysis
    analysis_df = pd.DataFrame(analysis_rows)
    analysis_df = analysis_df[analysis_df["_header"].astype(str) == "False"].drop(columns=["_header"])
    header_rows = [r for r in analysis_rows if r.get("_header")]

    # Rebuild with headers
    final_rows = []
    for r in analysis_rows:
        if r.get("_header"):
            final_rows.append({
                "Category": r["Category"], "Total Props": "",
                "Hits": "", "Hit Rate %": "", "Units Profit": "",
                "ROI %": "", "Avg Odds": ""
            })
        else:
            final_rows.append({k: v for k, v in r.items() if k != "_header"})

    final_df = pd.DataFrame(final_rows)
    write_sheet(gc, sheet_id, ANALYSIS_SHEET, final_df)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
