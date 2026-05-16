"""
resolve_picks.py
Auto-resolves KS and HRRBI win column using MLB Stats API.
Builds KS and HRRBI scorecards from pick logs.
"""

import os
import json
import time
import unicodedata
from datetime import date

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import requests

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

COLOR_BG        = {"red": 0.114, "green": 0.114, "blue": 0.114}
COLOR_BG_ALT    = {"red": 0.149, "green": 0.149, "blue": 0.149}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_GREEN_DIM = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_RED       = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_RED_DIM   = {"red": 0.200, "green": 0.039, "blue": 0.039}
COLOR_BLUE      = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_BLUE_DIM  = {"red": 0.055, "green": 0.180, "blue": 0.318}
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
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read '{name}': {e}")
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


# ── MLB Stats API ─────────────────────────────────────────────────────────

def get_games_for_date(game_date: str) -> list:
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
                stats       = player_data.get("stats", {}).get("pitching", {})
                ks          = stats.get("strikeOuts", None)
                if full_name and ks is not None:
                    result[normalize_name(full_name)] = int(ks)
    except Exception as e:
        print(f"  WARNING: Could not fetch pitcher Ks for game {game_pk}: {e}")
    return result


def get_batter_hrrbi_for_game(game_pk: int) -> dict:
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
                stats       = player_data.get("stats", {}).get("batting", {})
                hits        = stats.get("hits", 0)
                runs        = stats.get("runs", 0)
                rbi         = stats.get("rbi", 0)
                if full_name:
                    result[normalize_name(full_name)] = int(hits) + int(runs) + int(rbi)
    except Exception as e:
        print(f"  WARNING: Could not fetch batter stats for game {game_pk}: {e}")
    return result


def build_ks_results(dates: list) -> dict:
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


# ── Resolution ────────────────────────────────────────────────────────────

def resolve_ks_log(gc: gspread.Client, sheet_id: str) -> pd.DataFrame:
    today_str = date.today().strftime("%Y-%m-%d")
    log       = read_sheet_raw(gc, sheet_id, "KS_Picks_Log")

    if log.empty:
        print("KS_Picks_Log is empty.")
        return pd.DataFrame()

    if "win" not in log.columns:
        print("KS_Picks_Log missing win column.")
        return log

    pending = log[
        (log["win"].astype(str).str.strip() == "") &
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
        if str(row.get("win", "")).strip() != "":
            continue
        row_date = str(row.get("date", "")).strip()
        if row_date == today_str or not row_date:
            continue

        pitcher_norm = normalize_name(str(row.get("pitcher_name", "")))
        k_line       = safe_float(row.get("k_line", 0))
        actual_ks    = ks_results.get((row_date, pitcher_norm))

        if actual_ks is None:
            log.at[idx, "win"] = "No"
        else:
            log.at[idx, "win"] = "Yes" if actual_ks > k_line else "No"
        resolved += 1

    print(f"Resolved {resolved} KS picks. Yes: {(log['win'] == 'Yes').sum()} | No: {(log['win'] == 'No').sum()}")

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("KS_Picks_Log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="KS_Picks_Log", rows=5000, cols=13)

    log = log.fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([log.columns.tolist()] + log.astype(str).values.tolist()))
    print("KS_Picks_Log updated.")
    return log


def resolve_hrrbi_log(gc: gspread.Client, sheet_id: str) -> pd.DataFrame:
    today_str = date.today().strftime("%Y-%m-%d")
    log       = read_sheet_raw(gc, sheet_id, "HRRBI_Picks_Log")

    if log.empty:
        print("HRRBI_Picks_Log is empty.")
        return pd.DataFrame()

    if "win" not in log.columns:
        print("HRRBI_Picks_Log missing win column.")
        return log

    pending = log[
        (log["win"].astype(str).str.strip() == "") &
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
        if str(row.get("win", "")).strip() != "":
            continue
        row_date = str(row.get("date", "")).strip()
        if row_date == today_str or not row_date:
            continue

        player_norm  = normalize_name(str(row.get("player_name", "")))
        line         = safe_float(row.get("line", 0))
        actual_total = hrrbi_results.get((row_date, player_norm))

        if actual_total is None:
            log.at[idx, "win"] = "No"
        else:
            log.at[idx, "win"] = "Yes" if actual_total > line else "No"
        resolved += 1

    print(f"Resolved {resolved} HRRBI picks. Yes: {(log['win'] == 'Yes').sum()} | No: {(log['win'] == 'No').sum()}")

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("HRRBI_Picks_Log")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HRRBI_Picks_Log", rows=5000, cols=13)

    log = log.fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([log.columns.tolist()] + log.astype(str).values.tolist()))
    print("HRRBI_Picks_Log updated.")
    return log


# ── Scorecard ─────────────────────────────────────────────────────────────

def build_scorecard(
    gc: gspread.Client,
    sheet_id: str,
    log: pd.DataFrame,
    sheet_name: str,
    signal_col: str,
    section_color: dict,
    section_color_dim: dict,
    tab_color: dict,
    title: str,
) -> None:
    if log.empty or "win" not in log.columns:
        print(f"{sheet_name}: empty or missing win column — skipping.")
        return

    scored = log[log["win"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if scored.empty:
        print(f"{sheet_name}: no scored picks yet — skipping.")
        return

    scored["hit_bool"] = scored["win"].astype(str).str.strip() == "Yes"
    scored["rank_num"] = pd.to_numeric(scored["rank"], errors="coerce")
    scored["date_dt"]  = pd.to_datetime(scored["date"], errors="coerce")

    rows = []

    def add_row(label, sub, bold=False, header=False):
        if header:
            rows.append({"label": label, "total": "", "hits": "", "rate": "",
                         "_bold": bold, "_header": True})
            return
        if sub.empty:
            return
        total = len(sub)
        hits  = int(sub["hit_bool"].sum())
        rows.append({
            "label": label, "total": total, "hits": hits,
            "rate": round(hits / total * 100, 1),
            "_bold": bold, "_header": False,
        })

    add_row("🏆  Overall", scored, bold=True)

    add_row("── By Rank ──", pd.DataFrame(), bold=True, header=True)
    for rank in range(1, 11):
        sub = scored[scored["rank_num"] == rank]
        if not sub.empty: add_row(f"   Rank {rank}", sub)

    add_row("── By Confidence ──", pd.DataFrame(), bold=True, header=True)
    for tier in ["High", "Medium", "Low"]:
        sub = scored[scored["confidence"].astype(str) == tier]
        if not sub.empty: add_row(f"   {tier}", sub)

    if signal_col in scored.columns:
        add_row("── By Signal ──", pd.DataFrame(), bold=True, header=True)
        for sig in sorted(scored[signal_col].dropna().unique()):
            if str(sig).strip() in ("", "—"):
                continue
            sub = scored[scored[signal_col].astype(str) == str(sig)]
            if not sub.empty: add_row(f"   {sig}", sub)

    add_row("── Rolling ──", pd.DataFrame(), bold=True, header=True)
    max_date = scored["date_dt"].max()
    sub7  = scored[scored["date_dt"] >= max_date - pd.Timedelta(days=7)]
    sub30 = scored[scored["date_dt"] >= max_date - pd.Timedelta(days=30)]
    if not sub7.empty:  add_row("   Last 7 Days", sub7)
    if not sub30.empty: add_row("   Last 30 Days", sub30)

    # Build values
    all_values  = []
    headers_row = ["Category", "Total Picks", "Hits", "Hit Rate %"]

    all_values.append([f"📊  {title} — MODEL PERFORMANCE", "", "", ""])
    all_values.append(headers_row)
    data_start = 2

    for r in rows:
        rate = r.get("rate", "")
        all_values.append([
            r["label"],
            str(r.get("total", "")),
            str(r.get("hits", "")),
            f"{rate}%" if rate != "" else "",
        ])

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(sheet_name)
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=sheet_name, rows=200, cols=6)

    with_retry(lambda: ws.update(all_values))

    ws_id      = ws.id
    total_rows = len(all_values)
    total_cols = 4
    reqs       = []

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": section_color_dim,
            "textFormat": {"foregroundColor": section_color, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 13},
            "verticalAlignment": "MIDDLE",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": 2,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_HEADER_BG,
            "textFormat": {"foregroundColor": section_color, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 10},
            "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
    }})

    for i, r in enumerate(rows):
        row_idx   = data_start + i
        is_header = r.get("_header", False)
        is_bold   = r.get("_bold", False)
        rate      = r.get("rate", "")

        if is_header:
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                          "startColumnIndex": 0, "endColumnIndex": total_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {"foregroundColor": COLOR_GREY, "bold": True,
                                   "fontSize": 9, "italic": True},
                    "verticalAlignment": "MIDDLE",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
            }})
        else:
            bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                          "startColumnIndex": 0, "endColumnIndex": total_cols},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }})
            if rate != "":
                try:
                    rate_val = float(str(rate).replace("%", ""))
                    rate_bg  = COLOR_GREEN_DIM if rate_val >= 20 else COLOR_RED_DIM if rate_val <= 10 else bg
                    rate_fg  = COLOR_GREEN if rate_val >= 20 else COLOR_RED if rate_val <= 10 else COLOR_WHITE
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                                  "startColumnIndex": 3, "endColumnIndex": 4},
                        "cell": {"userEnteredFormat": {
                            "backgroundColor": rate_bg,
                            "textFormat": {"foregroundColor": rate_fg,
                                           "bold": rate_val >= 20 or rate_val <= 10},
                            "horizontalAlignment": "CENTER",
                        }},
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                    }})
                except Exception:
                    pass
            if is_bold and not is_header:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                              "startColumnIndex": 0, "endColumnIndex": 1},
                    "cell": {"userEnteredFormat": {
                        "textFormat": {"foregroundColor": section_color, "bold": True, "fontSize": 11},
                    }},
                    "fields": "userEnteredFormat(textFormat)",
                }})

    col_widths = [220, 100, 80, 100]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS",
                      "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w}, "fields": "pixelSize",
        }})

    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": ws_id, "dimension": "ROWS",
                  "startIndex": 0, "endIndex": total_rows},
        "properties": {"pixelSize": 32}, "fields": "pixelSize",
    }})
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": ws_id, "dimension": "ROWS",
                  "startIndex": 0, "endIndex": 1},
        "properties": {"pixelSize": 44}, "fields": "pixelSize",
    }})

    reqs.append({"updateSheetProperties": {
        "properties": {
            "sheetId": ws_id,
            "gridProperties": {"frozenRowCount": 2},
            "tabColorStyle": {"rgbColor": tab_color},
        },
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print(f"{sheet_name} formatted successfully.")
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
            log               = ks_log,
            sheet_name        = "KS_Scorecard",
            signal_col        = "prop_signal",
            section_color     = COLOR_TEAL,
            section_color_dim = COLOR_TEAL_DIM,
            tab_color         = COLOR_TEAL,
            title             = "PITCHER STRIKEOUTS",
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
            log               = hrrbi_log,
            sheet_name        = "HRRBI_Scorecard",
            signal_col        = "prop_signal",
            section_color     = COLOR_BLUE,
            section_color_dim = COLOR_BLUE_DIM,
            tab_color         = COLOR_BLUE,
            title             = "H+R+RBI",
        )

    print("Done.")


if __name__ == "__main__":
    main()
