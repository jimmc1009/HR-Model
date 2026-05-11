"""
resolve_picks.py
Auto-resolves pending KS and HRRBI picks using the MLB Stats API.
Updates KS_Picks_Log, HRRBI_Picks_Log, KS_Scorecard, HRRBI_Scorecard.
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import pytz
import requests

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

COLOR_BG        = {"red": 0.114, "green": 0.114, "blue": 0.114}
COLOR_BG_ALT    = {"red": 0.149, "green": 0.149, "blue": 0.149}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD      = {"red": 0.800, "green": 0.650, "blue": 0.000}
COLOR_GOLD_DIM  = {"red": 0.200, "green": 0.160, "blue": 0.000}
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_GREEN_DIM = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_RED       = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_RED_DIM   = {"red": 0.200, "green": 0.039, "blue": 0.039}
COLOR_BLUE      = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_BLUE_DIM  = {"red": 0.055, "green": 0.180, "blue": 0.318}
COLOR_PURPLE    = {"red": 0.576, "green": 0.439, "blue": 0.859}
COLOR_PURPLE_DIM= {"red": 0.118, "green": 0.078, "blue": 0.196}
COLOR_TEAL      = {"red": 0.047, "green": 0.450, "blue": 0.353}
COLOR_TEAL_DIM  = {"red": 0.020, "green": 0.120, "blue": 0.090}
COLOR_GREY      = {"red": 0.600, "green": 0.600, "blue": 0.600}
COLOR_HEADER_BG = {"red": 0.078, "green": 0.078, "blue": 0.078}


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


def read_sheet(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    try:
        sh   = with_retry(lambda: gc.open_by_key(sheet_id))
        ws   = sh.worksheet(name)
        data = with_retry(lambda: ws.get_all_records())
        return pd.DataFrame(data)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read sheet '{name}': {e}")
        return pd.DataFrame()


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


def american_odds_to_profit(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return odds / 100.0


# ── MLB Stats API helpers ─────────────────────────────────────────────────

def get_games_for_date(game_date: str) -> list:
    """Returns list of game PKs for a given date (YYYY-MM-DD)."""
    try:
        url  = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={game_date}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        games = []
        for d in resp.json().get("dates", []):
            for g in d.get("games", []):
                games.append(g.get("gamePk"))
        return [g for g in games if g]
    except Exception as e:
        print(f"  WARNING: Could not fetch games for {game_date}: {e}")
        return []


def get_pitcher_ks_for_game(game_pk: int) -> dict:
    """
    Returns dict of {normalized_pitcher_name: actual_ks} for a game.
    """
    result = {}
    try:
        url  = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for side in ["home", "away"]:
            pitchers = data.get("teams", {}).get(side, {}).get("pitchers", [])
            players  = data.get("teams", {}).get(side, {}).get("players", {})
            for pid in pitchers:
                player_key  = f"ID{pid}"
                player_data = players.get(player_key, {})
                full_name   = player_data.get("person", {}).get("fullName", "")
                stats        = player_data.get("stats", {}).get("pitching", {})
                ks           = stats.get("strikeOuts", None)
                if full_name and ks is not None:
                    result[normalize_name(full_name)] = int(ks)
    except Exception as e:
        print(f"  WARNING: Could not fetch pitcher Ks for game {game_pk}: {e}")
    return result


def get_batter_hrrbi_for_game(game_pk: int) -> dict:
    """
    Returns dict of {normalized_batter_name: total_h_r_rbi} for a game.
    """
    result = {}
    try:
        url  = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        for side in ["home", "away"]:
            batters  = data.get("teams", {}).get(side, {}).get("batters", [])
            players  = data.get("teams", {}).get(side, {}).get("players", {})
            for pid in batters:
                player_key  = f"ID{pid}"
                player_data = players.get(player_key, {})
                full_name   = player_data.get("person", {}).get("fullName", "")
                stats        = player_data.get("stats", {}).get("batting", {})
                hits         = stats.get("hits", 0)
                runs         = stats.get("runs", 0)
                rbi          = stats.get("rbi", 0)
                if full_name:
                    result[normalize_name(full_name)] = int(hits) + int(runs) + int(rbi)
    except Exception as e:
        print(f"  WARNING: Could not fetch batter stats for game {game_pk}: {e}")
    return result


def build_ks_results(dates: list) -> dict:
    """
    Returns dict of {(date_str, normalized_pitcher_name): actual_ks}
    for all games on the given dates.
    """
    results = {}
    for game_date in dates:
        print(f"  Fetching KS results for {game_date}...")
        game_pks = get_games_for_date(game_date)
        for pk in game_pks:
            pitcher_ks = get_pitcher_ks_for_game(pk)
            for name_norm, ks in pitcher_ks.items():
                results[(game_date, name_norm)] = ks
        time.sleep(0.5)
    return results


def build_hrrbi_results(dates: list) -> dict:
    """
    Returns dict of {(date_str, normalized_batter_name): total_hrrbi}
    for all games on the given dates.
    """
    results = {}
    for game_date in dates:
        print(f"  Fetching HRRBI results for {game_date}...")
        game_pks = get_games_for_date(game_date)
        for pk in game_pks:
            batter_stats = get_batter_hrrbi_for_game(pk)
            for name_norm, total in batter_stats.items():
                results[(game_date, name_norm)] = total
        time.sleep(0.5)
    return results


# ── Resolution logic ──────────────────────────────────────────────────────

def resolve_ks_log(gc: gspread.Client, sheet_id: str) -> pd.DataFrame:
    today_str = date.today().strftime("%Y-%m-%d")
    log       = read_sheet(gc, sheet_id, "KS_Picks_Log")

    if log.empty:
        print("KS_Picks_Log is empty.")
        return pd.DataFrame()

    pending = log[
        (log["hit"].astype(str).str.strip() == "Pending") &
        (log["date"].astype(str).str.strip() != today_str) &
        (log["date"].astype(str).str.strip() != "")
    ].copy()

    if pending.empty:
        print("No pending KS picks to resolve.")
        return log

    pending_dates = sorted(pending["date"].astype(str).str.strip().unique().tolist())
    print(f"Resolving {len(pending)} pending KS picks across {len(pending_dates)} dates...")

    ks_results = build_ks_results(pending_dates)

    resolved = 0
    for idx, row in log.iterrows():
        if str(row.get("hit", "")).strip() != "Pending":
            continue
        row_date = str(row.get("date", "")).strip()
        if row_date == today_str or not row_date:
            continue

        pitcher_norm = normalize_name(str(row.get("pitcher_name", "")))
        k_line       = safe_float(row.get("k_line", 0))
        actual_ks    = ks_results.get((row_date, pitcher_norm))

        if actual_ks is None:
            log.at[idx, "hit"]       = "No"
            log.at[idx, "actual_ks"] = ""
        else:
            log.at[idx, "actual_ks"] = str(actual_ks)
            bet_side = str(log.at[idx, "bet_side"]).strip().lower() if "bet_side" in log.columns else "over"
            if bet_side == "under":
                log.at[idx, "hit"] = "Yes" if actual_ks <= k_line else "No"
            else:
                log.at[idx, "hit"] = "Yes" if actual_ks > k_line else "No"
        resolved += 1

    print(f"Resolved {resolved} KS picks. Yes: {(log['hit'] == 'Yes').sum()} | No: {(log['hit'] == 'No').sum()}")

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("KS_Picks_Log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="KS_Picks_Log", rows=5000, cols=15)

    log = log.fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([log.columns.tolist()] + log.astype(str).values.tolist()))
    print("KS_Picks_Log updated.")
    return log


def resolve_hrrbi_log(gc: gspread.Client, sheet_id: str) -> pd.DataFrame:
    today_str = date.today().strftime("%Y-%m-%d")
    log       = read_sheet(gc, sheet_id, "HRRBI_Picks_Log")

    if log.empty:
        print("HRRBI_Picks_Log is empty.")
        return pd.DataFrame()

    pending = log[
        (log["hit"].astype(str).str.strip() == "Pending") &
        (log["date"].astype(str).str.strip() != today_str) &
        (log["date"].astype(str).str.strip() != "")
    ].copy()

    if pending.empty:
        print("No pending HRRBI picks to resolve.")
        return log

    pending_dates = sorted(pending["date"].astype(str).str.strip().unique().tolist())
    print(f"Resolving {len(pending)} pending HRRBI picks across {len(pending_dates)} dates...")

    hrrbi_results = build_hrrbi_results(pending_dates)

    resolved = 0
    for idx, row in log.iterrows():
        if str(row.get("hit", "")).strip() != "Pending":
            continue
        row_date = str(row.get("date", "")).strip()
        if row_date == today_str or not row_date:
            continue

        player_norm  = normalize_name(str(row.get("player_name", "")))
        line         = safe_float(row.get("line", 0))
        actual_total = hrrbi_results.get((row_date, player_norm))

        if actual_total is None:
            log.at[idx, "hit"]          = "No"
            log.at[idx, "actual_hrrbi"] = ""
        else:
            log.at[idx, "actual_hrrbi"] = str(actual_total)
            bet_side = str(log.at[idx, "bet_side"]).strip().lower() if "bet_side" in log.columns else "over"
            if bet_side == "under":
                log.at[idx, "hit"] = "Yes" if actual_total <= line else "No"
            else:
                log.at[idx, "hit"] = "Yes" if actual_total > line else "No"
        resolved += 1

    print(f"Resolved {resolved} HRRBI picks. Yes: {(log['hit'] == 'Yes').sum()} | No: {(log['hit'] == 'No').sum()}")

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("HRRBI_Picks_Log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HRRBI_Picks_Log", rows=5000, cols=15)

    log = log.fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([log.columns.tolist()] + log.astype(str).values.tolist()))
    print("HRRBI_Picks_Log updated.")
    return log


# ── Scorecard builder ─────────────────────────────────────────────────────

def build_scorecard(
    gc: gspread.Client,
    sheet_id: str,
    log: pd.DataFrame,
    sheet_name: str,
    hit_col: str,
    score_col: str,
    signal_col: str,
    section_color: dict,
    section_color_dim: dict,
    tab_color: dict,
    title: str,
) -> None:
    if log.empty:
        print(f"No data for {sheet_name} scorecard.")
        return

    scored = log[log[hit_col].astype(str).isin(["Yes", "No"])].copy()
    if scored.empty:
        print(f"No scored picks yet for {sheet_name}.")
        return

    scored["hit_bool"]   = scored[hit_col].astype(str) == "Yes"
    scored["rank_num"]   = pd.to_numeric(scored["rank"], errors="coerce")
    scored["date_dt"]    = pd.to_datetime(scored["date"], errors="coerce")
    scored["score_num"]  = pd.to_numeric(scored[score_col], errors="coerce")

    def is_bet_placed(x) -> bool:
        s = str(x).replace("$", "").strip().lower()
        if s in ("", "no", "nan", "pending"):
            return False
        try:
            float(s)
            return True
        except ValueError:
            return s == "yes"

    def parse_bet_size(x) -> float:
        s = str(x).replace("$", "").strip().lower()
        if s in ("", "yes", "no", "pending"):
            return 1.0
        return safe_float(s, 1.0)

    bet_picks = pd.DataFrame()
    if "bet_placed" in scored.columns and "odds" in scored.columns:
        bet_picks = scored[
            scored["bet_placed"].apply(is_bet_placed) &
            (scored["odds"].astype(str).str.strip() != "")
        ].copy()

        if not bet_picks.empty:
            bet_picks["odds_num"]      = bet_picks["odds"].apply(lambda x: safe_float(str(x).replace("+", "").strip(), 0))
            bet_picks["profit_if_win"] = bet_picks["odds_num"].apply(american_odds_to_profit)
            bet_picks["bet_size"]      = bet_picks["bet_placed"].apply(parse_bet_size)
            bet_picks["unit_result"]   = bet_picks.apply(
                lambda r: (r["profit_if_win"] * r["bet_size"]) if r["hit_bool"] else -r["bet_size"],
                axis=1
            )

    perf_rows  = []
    roi_rows   = []
    score_rows = []

    def add_perf(label, sub, bold=False):
        if sub.empty: return
        total = len(sub)
        hits  = int(sub["hit_bool"].sum())
        perf_rows.append({"label": label, "total": total, "hits": hits, "rate": round(hits / total * 100, 1), "_bold": bold})

    def add_roi(label, sub, bold=False):
        if sub.empty: return
        total        = len(sub)
        hits         = int(sub["hit_bool"].sum())
        profit       = round(sub["unit_result"].sum(), 2)
        roi          = round(profit / total * 100, 1) if total > 0 else 0.0
        roi_rows.append({"label": label, "bets": total, "hits": hits, "rate": round(hits / total * 100, 1), "profit": f"+{profit}" if profit >= 0 else str(profit), "roi": f"+{roi}%" if roi >= 0 else f"{roi}%", "_bold": bold, "_roi_val": roi, "_profit_val": profit})

    def add_score(label, sub, bold=False):
        if sub.empty: return
        total     = len(sub)
        hits      = int(sub["hit_bool"].sum())
        avg_score = round(sub["score_num"].mean(), 2) if not sub["score_num"].isna().all() else 0.0
        score_rows.append({"label": label, "total": total, "hits": hits, "rate": round(hits / total * 100, 1), "avg": avg_score, "_bold": bold})

    # Performance section
    add_perf(f"🏆  Overall", scored, bold=True)
    perf_rows.append({"label": "── By Rank ──", "total": "", "hits": "", "rate": "", "_bold": True, "_header": True})
    for rank in range(1, 11):
        sub = scored[scored["rank_num"] == rank]
        if not sub.empty: add_perf(f"   Rank {rank}", sub)

    perf_rows.append({"label": "── By Confidence ──", "total": "", "hits": "", "rate": "", "_bold": True, "_header": True})
    for tier in ["High", "Medium", "Low"]:
        sub = scored[scored["confidence"].astype(str) == tier]
        if not sub.empty: add_perf(f"   {tier}", sub)

    if signal_col in scored.columns:
        perf_rows.append({"label": "── By Signal ──", "total": "", "hits": "", "rate": "", "_bold": True, "_header": True})
        for sig in scored[signal_col].dropna().unique():
            if str(sig).strip() in ("", "—"):
                continue
            sub = scored[scored[signal_col].astype(str) == str(sig)]
            if not sub.empty: add_perf(f"   {sig}", sub)

    perf_rows.append({"label": "── Rolling ──", "total": "", "hits": "", "rate": "", "_bold": True, "_header": True})
    max_date = scored["date_dt"].max()
    add_perf("   Last 7 Days",  scored[scored["date_dt"] >= max_date - pd.Timedelta(days=7)])
    add_perf("   Last 30 Days", scored[scored["date_dt"] >= max_date - pd.Timedelta(days=30)])

    # ROI section
    if not bet_picks.empty:
        add_roi("💵  All Bets", bet_picks, bold=True)

        roi_rows.append({"label": "── By Confidence ──", "bets": "", "hits": "", "rate": "", "profit": "", "roi": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        for tier in ["High", "Medium", "Low"]:
            sub = bet_picks[bet_picks["confidence"].astype(str) == tier]
            if not sub.empty: add_roi(f"   {tier}", sub)

        roi_rows.append({"label": "── By Signal ──", "bets": "", "hits": "", "rate": "", "profit": "", "roi": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        if signal_col in bet_picks.columns:
            for sig in bet_picks[signal_col].dropna().unique():
                if str(sig).strip() in ("", "—"):
                    continue
                sub = bet_picks[bet_picks[signal_col].astype(str) == str(sig)]
                if not sub.empty: add_roi(f"   {sig}", sub)

        roi_rows.append({"label": "── By Juice ──", "bets": "", "hits": "", "rate": "", "profit": "", "roi": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        favorable = bet_picks[bet_picks["odds_num"] >= -110]
        standard  = bet_picks[(bet_picks["odds_num"] < -110) & (bet_picks["odds_num"] >= -130)]
        heavy     = bet_picks[bet_picks["odds_num"] < -130]
        plus_odds = bet_picks[bet_picks["odds_num"] > 0]
        if not plus_odds.empty:  add_roi("   Plus odds (+100 or better)", plus_odds)
        if not favorable.empty:  add_roi("   Favorable (-110 or better)", favorable)
        if not standard.empty:   add_roi("   Standard (-111 to -130)", standard)
        if not heavy.empty:      add_roi("   Heavy (-131 or worse)", heavy)

        roi_rows.append({"label": "── Rolling ──", "bets": "", "hits": "", "rate": "", "profit": "", "roi": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        max_bet = bet_picks["date_dt"].max()
        add_roi("   Last 7 Days",  bet_picks[bet_picks["date_dt"] >= max_bet - pd.Timedelta(days=7)])
        add_roi("   Last 30 Days", bet_picks[bet_picks["date_dt"] >= max_bet - pd.Timedelta(days=30)])

    # Score tier section
    add_score(f"📈  All Scored Picks", scored, bold=True)
    score_rows.append({"label": "── By Score Tier ──", "total": "", "hits": "", "rate": "", "avg": "", "_bold": True, "_header": True})
    for label, sub in [
        ("   12+",     scored[scored["score_num"] >= 12]),
        ("   10+",     scored[scored["score_num"] >= 10]),
        ("   8+",      scored[scored["score_num"] >= 8]),
        ("   Under 8", scored[scored["score_num"] <  8]),
    ]:
        if not sub.empty: add_score(label, sub)

    # Build sheet values
    all_values = []
    perf_headers  = ["Category", "Total Picks", "Hits", "Hit Rate %"]
    roi_headers   = ["Category", "Bets Placed", "Hits", "Hit Rate %", "Units Profit", "ROI %"]
    score_headers = ["Category", "Total Picks", "Hits", "Hit Rate %", f"Avg Score"]

    all_values.append([f"📊  {title} — MODEL PERFORMANCE", "", "", ""])
    all_values.append(perf_headers)
    perf_start = 2

    for r in perf_rows:
        all_values.append([r["label"], str(r.get("total", "")), str(r.get("hits", "")), f"{r['rate']}%" if r.get("rate", "") != "" else ""])

    all_values.append(["", "", "", ""])
    all_values.append(["", "", "", ""])
    roi_section_start = len(all_values)

    if roi_rows:
        all_values.append(["💰  BETTING ROI", "", "", "", "", ""])
        all_values.append(roi_headers)
        roi_data_start = len(all_values)
        for r in roi_rows:
            all_values.append([r["label"], str(r.get("bets", "")), str(r.get("hits", "")), f"{r['rate']}%" if r.get("rate", "") != "" else "", str(r.get("profit", "")), str(r.get("roi", ""))])
    else:
        all_values.append(["💰  BETTING ROI — No bets placed yet", "", "", "", "", ""])
        roi_data_start = len(all_values)

    all_values.append(["", "", "", "", "", ""])
    all_values.append(["", "", "", "", "", ""])
    score_section_start = len(all_values)

    all_values.append([f"📈  SCORE TIER ANALYSIS", "", "", "", ""])
    all_values.append(score_headers)
    score_data_start = len(all_values)
    for r in score_rows:
        all_values.append([r["label"], str(r.get("total", "")), str(r.get("hits", "")), f"{r['rate']}%" if r.get("rate", "") != "" else "", str(r.get("avg", ""))])

    # Write to sheet
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(sheet_name)
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_name, rows=200, cols=8)

    with_retry(lambda: ws.update(all_values))

    ws_id      = ws.id
    total_rows = len(all_values)
    total_cols = 6
    reqs       = []

    # Base style
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows, "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG, "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10}, "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    # Performance section header
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"backgroundColor": section_color_dim, "textFormat": {"foregroundColor": section_color, "bold": True, "fontFamily": "Roboto", "fontSize": 13}, "verticalAlignment": "MIDDLE"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": 2, "startColumnIndex": 0, "endColumnIndex": 4},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER_BG, "textFormat": {"foregroundColor": section_color, "bold": True, "fontFamily": "Roboto", "fontSize": 10}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
    }})

    for i, r in enumerate(perf_rows):
        row_idx   = perf_start + i
        is_header = r.get("_header", False)
        is_bold   = r.get("_bold", False)
        rate      = r.get("rate", "")
        if is_header:
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 4},
                "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER_BG, "textFormat": {"foregroundColor": COLOR_GREY, "bold": True, "fontSize": 9, "italic": True}, "verticalAlignment": "MIDDLE"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
            }})
        else:
            bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 4},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }})
            if rate != "":
                try:
                    rate_val = float(str(rate).replace("%", ""))
                    rate_bg  = COLOR_GREEN_DIM if rate_val >= 20 else COLOR_RED_DIM if rate_val <= 10 else bg
                    rate_fg  = COLOR_GREEN if rate_val >= 20 else COLOR_RED if rate_val <= 10 else COLOR_WHITE
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 3, "endColumnIndex": 4},
                        "cell": {"userEnteredFormat": {"backgroundColor": rate_bg, "textFormat": {"foregroundColor": rate_fg, "bold": rate_val >= 20 or rate_val <= 10}, "horizontalAlignment": "CENTER"}},
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                    }})
                except Exception:
                    pass
            if is_bold and not is_header:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 1},
                    "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": section_color, "bold": True, "fontSize": 11}}},
                    "fields": "userEnteredFormat(textFormat)",
                }})

    # ROI section
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": roi_section_start, "endRowIndex": roi_section_start + 1, "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_GOLD_DIM, "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True, "fontFamily": "Roboto", "fontSize": 13}, "verticalAlignment": "MIDDLE"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})

    if roi_rows:
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": roi_section_start + 1, "endRowIndex": roi_section_start + 2, "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER_BG, "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True, "fontFamily": "Roboto", "fontSize": 10}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }})
        for i, r in enumerate(roi_rows):
            row_idx    = roi_data_start + i
            is_header  = r.get("_header", False)
            is_bold    = r.get("_bold", False)
            roi_val    = r.get("_roi_val", 0)
            profit_val = r.get("_profit_val", 0)
            if is_header:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": total_cols},
                    "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER_BG, "textFormat": {"foregroundColor": COLOR_GREY, "bold": True, "fontSize": 9, "italic": True}, "verticalAlignment": "MIDDLE"}},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
                }})
            else:
                bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": total_cols},
                    "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                    "fields": "userEnteredFormat(backgroundColor)",
                }})
                roi_fg = COLOR_GREEN if roi_val > 0 else COLOR_RED if roi_val < 0 else COLOR_WHITE
                roi_bg = COLOR_GREEN_DIM if roi_val > 0 else COLOR_RED_DIM if roi_val < 0 else bg
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 5, "endColumnIndex": 6},
                    "cell": {"userEnteredFormat": {"backgroundColor": roi_bg, "textFormat": {"foregroundColor": roi_fg, "bold": True}, "horizontalAlignment": "CENTER"}},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }})
                profit_fg = COLOR_GREEN if profit_val > 0 else COLOR_RED if profit_val < 0 else COLOR_WHITE
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 4, "endColumnIndex": 5},
                    "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": profit_fg, "bold": True}, "horizontalAlignment": "CENTER"}},
                    "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
                }})
                if is_bold:
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 1},
                        "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": COLOR_GOLD, "bold": True, "fontSize": 11}}},
                        "fields": "userEnteredFormat(textFormat)",
                    }})

    # Score tier section
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": score_section_start, "endRowIndex": score_section_start + 1, "startColumnIndex": 0, "endColumnIndex": 5},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_PURPLE_DIM, "textFormat": {"foregroundColor": COLOR_PURPLE, "bold": True, "fontFamily": "Roboto", "fontSize": 13}, "verticalAlignment": "MIDDLE"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": score_section_start + 1, "endRowIndex": score_section_start + 2, "startColumnIndex": 0, "endColumnIndex": 5},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER_BG, "textFormat": {"foregroundColor": COLOR_PURPLE, "bold": True, "fontFamily": "Roboto", "fontSize": 10}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
    }})

    for i, r in enumerate(score_rows):
        row_idx   = score_data_start + i
        is_header = r.get("_header", False)
        is_bold   = r.get("_bold", False)
        rate      = r.get("rate", "")
        if is_header:
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 5},
                "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER_BG, "textFormat": {"foregroundColor": COLOR_GREY, "bold": True, "fontSize": 9, "italic": True}, "verticalAlignment": "MIDDLE"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
            }})
        else:
            bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 5},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }})
            if rate != "":
                try:
                    rate_val = float(str(rate).replace("%", ""))
                    rate_bg  = COLOR_GREEN_DIM if rate_val >= 20 else COLOR_RED_DIM if rate_val <= 10 else bg
                    rate_fg  = COLOR_GREEN if rate_val >= 20 else COLOR_RED if rate_val <= 10 else COLOR_WHITE
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 3, "endColumnIndex": 4},
                        "cell": {"userEnteredFormat": {"backgroundColor": rate_bg, "textFormat": {"foregroundColor": rate_fg, "bold": rate_val >= 20 or rate_val <= 10}, "horizontalAlignment": "CENTER"}},
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                    }})
                except Exception:
                    pass
            if is_bold and not is_header:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 1},
                    "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": COLOR_PURPLE, "bold": True, "fontSize": 11}}},
                    "fields": "userEnteredFormat(textFormat)",
                }})

    # Column widths and row heights
    col_widths = [220, 100, 80, 100, 120, 100]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 0, "endIndex": total_rows},
        "properties": {"pixelSize": 32},
        "fields": "pixelSize",
    }})
    for row_idx in [0, roi_section_start, score_section_start]:
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": row_idx, "endIndex": row_idx + 1},
            "properties": {"pixelSize": 44},
            "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 2}, "tabColorStyle": {"rgbColor": tab_color}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print(f"{sheet_name} scorecard formatted successfully.")
    except APIError as e:
        print(f"{sheet_name} formatting failed: {e}")

    print(f"{sheet_name} updated.")


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("=" * 50)
    print("Resolving KS picks...")
    print("=" * 50)
    ks_log = resolve_ks_log(gc, sheet_id)
    time.sleep(5)

    if not ks_log.empty:
        build_scorecard(
            gc, sheet_id,
            log          = ks_log,
            sheet_name   = "KS_Scorecard",
            hit_col      = "hit",
            score_col    = "ks_score",
            signal_col   = "prop_signal",
            section_color     = COLOR_TEAL,
            section_color_dim = COLOR_TEAL_DIM,
            tab_color    = COLOR_TEAL,
            title        = "PITCHER STRIKEOUTS",
        )
    time.sleep(5)

    print("=" * 50)
    print("Resolving HRRBI picks...")
    print("=" * 50)
    hrrbi_log = resolve_hrrbi_log(gc, sheet_id)
    time.sleep(5)

    if not hrrbi_log.empty:
        build_scorecard(
            gc, sheet_id,
            log          = hrrbi_log,
            sheet_name   = "HRRBI_Scorecard",
            hit_col      = "hit",
            score_col    = "hrrbi_score",
            signal_col   = "prop_signal",
            section_color     = COLOR_BLUE,
            section_color_dim = COLOR_BLUE_DIM,
            tab_color    = COLOR_BLUE,
            title        = "H+R+RBI",
        )

    print("Done.")


if __name__ == "__main__":
    main()
