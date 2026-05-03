"""
dashboard.py
Builds the "Today's Top Picks" unified dashboard sheet.
Reads from Top_HR_Picks, Top_KS_Picks, Top_HRRBI_Picks.

Layout (vertical, stacked):
  - HR Picks:    Top 10 — Rank / Batter / Team
  - K Picks:     Top 5 with LEAN/OVER signal only
  - HRRBI Picks: Top 5 with LEAN/OVER/UNDER signal only

If no signals exist in K or HRRBI top 5, shows "No plays today".
"""

import os
import json
import time
from datetime import date, datetime

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
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_BLUE      = {"red": 0.055, "green": 0.318, "blue": 0.580}
COLOR_TEAL      = {"red": 0.075, "green": 0.376, "blue": 0.227}
COLOR_DARK_RED  = {"red": 0.600, "green": 0.050, "blue": 0.050}
COLOR_SECTION   = {"red": 0.055, "green": 0.055, "blue": 0.055}


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
                print(f"  Rate limit hit — waiting {wait}s (attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                print(f"  API error: {e} — retrying in {wait}s...")
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


def has_signal(signal: str) -> bool:
    s = str(signal).strip()
    return s not in ("", "—", "nan") and len(s) > 1


def build_dashboard_rows(hr_df: pd.DataFrame, ks_df: pd.DataFrame, hrrbi_df: pd.DataFrame):
    """
    Returns list of rows and list of row metadata for formatting.
    Each entry: (row_data, row_type)
    row_type: 'section_header' | 'col_header' | 'data' | 'no_plays' | 'spacer'
    """
    rows = []

    # ── HR PICKS ──────────────────────────────────────────────────
    rows.append((["🏠  HOME RUN PICKS", "", ""], "section_header_hr"))
    rows.append((["Rank", "Batter", "Team"], "col_header"))

    if hr_df.empty:
        rows.append((["—", "No picks available", "—"], "no_plays"))
    else:
        # Strip timestamp row if present
        hr_clean = hr_df[~hr_df.iloc[:, 0].astype(str).str.contains("Last Run", na=False)].copy()
        # Rename columns to standard names if needed
        col_map = {}
        for c in hr_clean.columns:
            cl = c.lower().strip()
            if "rank" in cl:       col_map[c] = "rank"
            elif "batter" in cl:   col_map[c] = "batter"
            elif "team" in cl:     col_map[c] = "team"
        hr_clean = hr_clean.rename(columns=col_map)

        top10 = hr_clean.head(10)
        for i, (_, row) in enumerate(top10.iterrows()):
            rows.append((
                [
                    str(row.get("rank", i + 1)),
                    str(row.get("batter", row.get("Batter", ""))),
                    str(row.get("team", row.get("Team", ""))),
                ],
                "data_hr"
            ))

    rows.append((["", "", ""], "spacer"))

    # ── K PICKS ───────────────────────────────────────────────────
    rows.append((["⚾  PITCHER STRIKEOUT PLAYS", "", "", "", "", ""], "section_header_ks"))
    rows.append((["Rank", "Pitcher", "Team", "K Line", "Over Odds", "Signal"], "col_header"))

    if ks_df.empty:
        rows.append((["—", "No plays today", "", "", "", ""], "no_plays"))
    else:
        ks_clean = ks_df[~ks_df.iloc[:, 0].astype(str).str.contains("Last Run", na=False)].copy()

        col_map = {}
        for c in ks_clean.columns:
            cl = c.lower().strip()
            if cl == "rank":                col_map[c] = "rank"
            elif "pitcher" in cl and "name" not in cl: col_map[c] = "pitcher"
            elif cl == "pitcher":           col_map[c] = "pitcher"
            elif cl == "team":              col_map[c] = "team"
            elif "k line" in cl or cl == "k_line": col_map[c] = "k_line"
            elif "over odds" in cl or cl == "over_odds" or cl == "ks_over_odds": col_map[c] = "over_odds"
            elif "signal" in cl:            col_map[c] = "signal"
        ks_clean = ks_clean.rename(columns=col_map)

        top5    = ks_clean.head(5)
        plays   = top5[top5["signal"].apply(has_signal)] if "signal" in top5.columns else pd.DataFrame()

        if plays.empty:
            rows.append((["—", "No plays today", "", "", "", ""], "no_plays"))
        else:
            for i, (_, row) in enumerate(plays.iterrows()):
                rows.append((
                    [
                        str(row.get("rank", i + 1)),
                        str(row.get("pitcher", row.get("Pitcher", ""))),
                        str(row.get("team", row.get("Team", ""))),
                        str(row.get("k_line", row.get("K Line", ""))),
                        str(row.get("over_odds", row.get("Over Odds", ""))),
                        str(row.get("signal", row.get("Signal", ""))),
                    ],
                    "data_ks"
                ))

    rows.append((["", "", "", "", "", ""], "spacer"))

    # ── HRRBI PICKS ───────────────────────────────────────────────
    rows.append((["📊  H+R+RBI PLAYS", "", "", "", "", ""], "section_header_hrrbi"))
    rows.append((["Rank", "Player", "Team", "Line", "Over Odds", "Signal"], "col_header"))

    if hrrbi_df.empty:
        rows.append((["—", "No plays today", "", "", "", ""], "no_plays"))
    else:
        hrrbi_clean = hrrbi_df[~hrrbi_df.iloc[:, 0].astype(str).str.contains("Last Run", na=False)].copy()

        col_map = {}
        for c in hrrbi_clean.columns:
            cl = c.lower().strip()
            if cl == "rank":               col_map[c] = "rank"
            elif "batter" in cl or "player" in cl: col_map[c] = "player"
            elif cl == "team":             col_map[c] = "team"
            elif "line" in cl and "over" not in cl and "under" not in cl: col_map[c] = "line"
            elif "over odds" in cl or "over_odds" in cl or "hrrbi_over" in cl: col_map[c] = "over_odds"
            elif "signal" in cl:           col_map[c] = "signal"
        hrrbi_clean = hrrbi_clean.rename(columns=col_map)

        top5  = hrrbi_clean.head(5)
        plays = top5[top5["signal"].apply(has_signal)] if "signal" in top5.columns else pd.DataFrame()

        if plays.empty:
            rows.append((["—", "No plays today", "", "", "", ""], "no_plays"))
        else:
            for i, (_, row) in enumerate(plays.iterrows()):
                rows.append((
                    [
                        str(row.get("rank", i + 1)),
                        str(row.get("player", row.get("Batter", ""))),
                        str(row.get("team", row.get("Team", ""))),
                        str(row.get("line", row.get("Line", ""))),
                        str(row.get("over_odds", row.get("Over Odds", ""))),
                        str(row.get("signal", row.get("Signal", ""))),
                    ],
                    "data_hrrbi"
                ))

    return rows


def write_dashboard(gc: gspread.Client, sheet_id: str, rows) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws = sh.worksheet(DASHBOARD_SHEET)
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=DASHBOARD_SHEET, rows=200, cols=6)

    # Pad all rows to 6 columns
    padded = []
    for row_data, _ in rows:
        padded.append(row_data + [""] * (6 - len(row_data)))

    with_retry(lambda: ws.update(padded, value_input_option="RAW"))

    ws_id = ws.id
    reqs  = []
    n_cols = 6

    # Base formatting — dark bg, white text, single-line cells
    reqs.append({"repeatCell": {
        "range": {
            "sheetId": ws_id,
            "startRowIndex": 0,
            "endRowIndex": len(rows) + 5,
            "startColumnIndex": 0,
            "endColumnIndex": n_cols,
        },
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {
                "foregroundColor": COLOR_WHITE,
                "fontFamily": "Roboto",
                "fontSize": 11,
            },
            "verticalAlignment": "MIDDLE",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    # Row-level formatting
    for row_idx, (row_data, row_type) in enumerate(rows):
        r = row_idx  # 0-indexed

        if row_type.startswith("section_header"):
            color = COLOR_GOLD if "hr" in row_type else COLOR_TEAL if "ks" in row_type else COLOR_BLUE
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": color,
                    "textFormat": {"foregroundColor": COLOR_BG, "bold": True, "fontFamily": "Roboto", "fontSize": 13},
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
            }})
            # Merge section header across all columns
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "mergeType": "MERGE_ALL",
            }})

        elif row_type == "col_header":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_SECTION,
                    "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True, "fontFamily": "Roboto", "fontSize": 10},
                    "horizontalAlignment": "CENTER",
                    "verticalAlignment": "MIDDLE",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
            }})

        elif row_type.startswith("data"):
            bg = COLOR_BG if row_idx % 2 == 0 else COLOR_BG_ALT
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
            }})

            # Color the signal cell green for OVER/LEAN OVER, red for UNDER
            if row_type in ("data_ks", "data_hrrbi"):
                signal_col = 5
                signal_val = str(row_data[-1]).upper()
                if "OVER" in signal_val and "UNDER" not in signal_val:
                    sig_color = COLOR_GREEN
                elif "UNDER" in signal_val:
                    sig_color = COLOR_DARK_RED
                else:
                    sig_color = COLOR_BG if row_idx % 2 == 0 else COLOR_BG_ALT

                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": signal_col, "endColumnIndex": signal_col + 1},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": sig_color,
                        "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True, "fontFamily": "Roboto", "fontSize": 11},
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }})

        elif row_type == "no_plays":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_BG_ALT,
                    "textFormat": {"foregroundColor": {"red": 0.6, "green": 0.6, "blue": 0.6}, "italic": True, "fontFamily": "Roboto", "fontSize": 11},
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
            }})

        elif row_type == "spacer":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG}},
                "fields": "userEnteredFormat(backgroundColor)",
            }})

    # Column widths
    col_widths = [55, 180, 75, 80, 100, 180]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    # Row heights — uniform 32px, section headers 40px
    for row_idx, (_, row_type) in enumerate(rows):
        h = 40 if row_type.startswith("section_header") else 32
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": row_idx, "endIndex": row_idx + 1},
            "properties": {"pixelSize": h},
            "fields": "pixelSize",
        }})

    # Freeze no rows, no columns
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

    rows = build_dashboard_rows(hr_df, ks_df, hrrbi_df)
    write_dashboard(gc, sheet_id, rows)
    time.sleep(3)
    write_timestamp(gc, sheet_id)
    print("Dashboard written to 'Today's Top Picks'")


if __name__ == "__main__":
    main()
