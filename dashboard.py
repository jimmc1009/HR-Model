"""
dashboard.py
Builds the "Today's Top Picks" unified dashboard sheet.
"""

import os
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import pytz

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DASHBOARD_SHEET = "Today's Top Picks"

COLOR_BG       = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT   = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD      = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_GREEN     = {"red": 0.118, "green": 0.627, "blue": 0.337}
COLOR_BLUE      = {"red": 0.055, "green": 0.318, "blue": 0.580}
COLOR_TEAL      = {"red": 0.047, "green": 0.450, "blue": 0.353}
COLOR_DARK_RED  = {"red": 0.550, "green": 0.050, "blue": 0.050}
COLOR_HEADER_BG = {"red": 0.055, "green": 0.055, "blue": 0.055}
COLOR_SUBTEXT   = {"red": 0.500, "green": 0.500, "blue": 0.500}
COLOR_BLACK     = {"red": 0.050, "green": 0.050, "blue": 0.050}


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
        sh         = with_retry(lambda: gc.open_by_key(sheet_id))
        ws         = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values:
            return pd.DataFrame()
        start = 1 if all_values and "Last Run" in str(all_values[0]) else 0
        if start >= len(all_values):
            return pd.DataFrame()
        headers = all_values[start]
        rows    = all_values[start + 1:]
        return pd.DataFrame(rows, columns=headers)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read sheet '{name}': {e}")
        return pd.DataFrame()


def has_signal(signal: str) -> bool:
    s = str(signal).strip()
    return s not in ("", "—", "nan") and len(s) > 1


def safe_val(row, col: str, default: str = "") -> str:
    try:
        val = row[col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return str(val) if pd.notna(val) else default
    except Exception:
        return default


def build_rows(hr_df: pd.DataFrame, ks_df: pd.DataFrame, hrrbi_df: pd.DataFrame):
    """
    Returns list of (row_data, row_type) tuples.
    All rows are padded to 6 columns.
    row_type: section_header_hr | section_header_ks | section_header_hrrbi |
              col_header | data_hr | data_ks | data_hrrbi | no_plays | spacer
    """
    N = 6
    E = [""] * N

    def pad(row, n=N):
        return list(row) + [""] * (n - len(row))

    rows = []

    # ── HOME RUN PICKS ────────────────────────────────────────────────────
    rows.append((pad(["🏠  HOME RUN PICKS"]), "section_header_hr"))
    rows.append((pad(["Rank", "Batter", "Team"]), "col_header_hr"))

    if hr_df.empty:
        rows.append((pad(["—", "No picks available", ""]), "no_plays"))
    else:
        top10 = hr_df.head(10)
        for i in range(len(top10)):
            row = top10.iloc[i]
            rank   = str(i + 1)
            batter = safe_val(row, "Batter")
            # Team is column index 3 in Top_HR_Picks to avoid duplicate team cols
            try:
                team = str(top10.iloc[i, 3])
            except Exception:
                team = safe_val(row, "Team")
            rows.append((pad([rank, batter, team]), "data_hr"))

    rows.append((E[:], "spacer"))

    # ── PITCHER STRIKEOUT PLAYS ───────────────────────────────────────────
    rows.append((pad(["⚾  PITCHER STRIKEOUT PLAYS"]), "section_header_ks"))
    rows.append((pad(["Rank", "Pitcher", "Team", "K Line", "Over Odds", "Signal"]), "col_header_ks"))

    if ks_df.empty:
        rows.append((pad(["—", "No plays today"]), "no_plays"))
    else:
        top5  = ks_df.head(5)
        plays = top5[top5.get("Signal", top5.get("prop_signal", pd.Series([""] * len(top5)))).apply(has_signal)] if not top5.empty else pd.DataFrame()

        # Try to find signal column
        sig_col = None
        for c in ["Signal", "prop_signal", "signal"]:
            if c in top5.columns:
                sig_col = c
                break

        if sig_col:
            plays = top5[top5[sig_col].apply(has_signal)]
        else:
            plays = pd.DataFrame()

        if plays.empty:
            rows.append((pad(["—", "No plays today"]), "no_plays"))
        else:
            for i in range(len(plays)):
                row = plays.iloc[i]
                rank      = safe_val(row, "Rank", str(i + 1))
                pitcher   = safe_val(row, "Pitcher")
                team      = safe_val(row, "Team")
                k_line    = safe_val(row, "K Line")
                over_odds = safe_val(row, "Over Odds")
                signal    = safe_val(row, sig_col)
                rows.append((pad([rank, pitcher, team, k_line, over_odds, signal]), "data_ks"))

    rows.append((E[:], "spacer"))

    # ── H+R+RBI PLAYS ─────────────────────────────────────────────────────
    rows.append((pad(["📊  H+R+RBI PLAYS"]), "section_header_hrrbi"))
    rows.append((pad(["Rank", "Player", "Team", "Line", "Over Odds", "Signal"]), "col_header_hrrbi"))

    if hrrbi_df.empty:
        rows.append((pad(["—", "No plays today"]), "no_plays"))
    else:
        top5 = hrrbi_df.head(5)

        sig_col = None
        for c in ["Signal", "prop_signal", "signal"]:
            if c in top5.columns:
                sig_col = c
                break

        if sig_col:
            plays = top5[top5[sig_col].apply(has_signal)]
        else:
            plays = pd.DataFrame()

        if plays.empty:
            rows.append((pad(["—", "No plays today"]), "no_plays"))
        else:
            for i in range(len(plays)):
                row = plays.iloc[i]
                rank      = safe_val(row, "Rank", str(i + 1))
                player    = safe_val(row, "Batter", safe_val(row, "Player"))
                team      = safe_val(row, "Team")
                line      = safe_val(row, "Line")
                over_odds = safe_val(row, "Over Odds")
                signal    = safe_val(row, sig_col)
                rows.append((pad([rank, player, team, line, over_odds, signal]), "data_hrrbi"))

    return rows


def write_dashboard(gc: gspread.Client, sheet_id: str, rows) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(DASHBOARD_SHEET)
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=DASHBOARD_SHEET, rows=200, cols=6)

    data = [row_data for row_data, _ in rows]
    with_retry(lambda: ws.update(data, value_input_option="RAW"))

    ws_id  = ws.id
    n_cols = 6
    reqs   = []

    # ── Global base style ─────────────────────────────────────────────────
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": len(rows) + 5, "startColumnIndex": 0, "endColumnIndex": n_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto", "fontSize": 11, "bold": False},
            "verticalAlignment": "MIDDLE",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    # ── Row-level formatting ──────────────────────────────────────────────
    data_row_count = {"hr": 0, "ks": 0, "hrrbi": 0}

    for row_idx, (row_data, row_type) in enumerate(rows):
        r = row_idx

        if row_type.startswith("section_header"):
            if "hr" in row_type:
                color = COLOR_GOLD
                text_color = COLOR_BLACK
            elif "ks" in row_type:
                color = COLOR_TEAL
                text_color = COLOR_WHITE
            else:
                color = COLOR_BLUE
                text_color = COLOR_WHITE

            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": color,
                    "textFormat": {"foregroundColor": text_color, "bold": True, "fontFamily": "Roboto", "fontSize": 12},
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "mergeType": "MERGE_ALL",
            }})

        elif row_type.startswith("col_header"):
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": True, "fontFamily": "Roboto", "fontSize": 9},
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})

        elif row_type.startswith("data"):
            section = row_type.split("_")[1]
            count   = data_row_count[section]
            bg      = COLOR_BG if count % 2 == 0 else COLOR_BG_ALT
            data_row_count[section] += 1

            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
            }})

            # Rank col — centered, muted
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": 1},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "fontFamily": "Roboto", "fontSize": 11, "bold": True},
                    "horizontalAlignment": "CENTER",
                }},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})

            # Signal col — colored
            if row_type in ("data_ks", "data_hrrbi"):
                signal_val = str(row_data[5]).upper()
                if "OVER" in signal_val and "UNDER" not in signal_val:
                    sig_bg = COLOR_GREEN
                    sig_text = COLOR_WHITE
                elif "UNDER" in signal_val:
                    sig_bg = COLOR_DARK_RED
                    sig_text = COLOR_WHITE
                else:
                    sig_bg = bg
                    sig_text = COLOR_SUBTEXT

                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 5, "endColumnIndex": 6},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": sig_bg,
                        "textFormat": {"foregroundColor": sig_text, "bold": True, "fontFamily": "Roboto", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})

        elif row_type == "no_plays":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_BG_ALT,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "italic": True, "fontFamily": "Roboto", "fontSize": 11},
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
            }})

    # ── Column widths (global, applies to whole sheet) ────────────────────
    col_widths = [55, 180, 70, 70, 95, 180]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    # ── Row heights ───────────────────────────────────────────────────────
    for row_idx, (_, row_type) in enumerate(rows):
        if row_type.startswith("section_header"):
            h = 40
        elif row_type == "spacer":
            h = 16
        elif row_type.startswith("col_header"):
            h = 26
        else:
            h = 32
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": row_idx, "endIndex": row_idx + 1},
            "properties": {"pixelSize": h},
            "fields": "pixelSize",
        }})

    # ── Sheet properties ──────────────────────────────────────────────────
    reqs.append({"updateSheetProperties": {
        "properties": {
            "sheetId": ws_id,
            "gridProperties": {"frozenRowCount": 0},
            "tabColorStyle": {"rgbColor": COLOR_GOLD},
        },
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("Dashboard formatting applied.")
    except APIError as e:
        print(f"Dashboard formatting failed: {e}")


def write_timestamp(gc: gspread.Client, sheet_id: str) -> None:
    et     = pytz.timezone("America/New_York")
    now_et = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")
    sh     = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(DASHBOARD_SHEET)
        with_retry(lambda: ws.insert_row([f"⏱  Last Updated: {now_et}", "", "", "", "", ""], index=1))
    except Exception as e:
        print(f"Dashboard timestamp failed: {e}")
    print(f"Dashboard timestamp written: {now_et}")


def main() -> None:
    time.sleep(5)
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading picks sheets for dashboard...")
    hr_df    = read_sheet(gc, sheet_id, "Top_HR_Picks")
    time.sleep(2)
    ks_df    = read_sheet(gc, sheet_id, "Top_KS_Picks")
    time.sleep(2)
    hrrbi_df = read_sheet(gc, sheet_id, "Top_HRRBI_Picks")

    print(f"HR picks: {len(hr_df)} rows")
    print(f"KS picks: {len(ks_df)} rows")
    print(f"HRRBI picks: {len(hrrbi_df)} rows")

    rows = build_rows(hr_df, ks_df, hrrbi_df)
    write_dashboard(gc, sheet_id, rows)
    time.sleep(3)
    write_timestamp(gc, sheet_id)
    print("Dashboard written to 'Today's Top Picks'")


if __name__ == "__main__":
    main()
