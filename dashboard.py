"""
resolve_scorecard.py
Reads the Scorecard sheet, calculates P&L for any rows where
Your Odds, Stake, and Result are all filled in, writes P&L back,
and prints a summary by model and overall.
Runs in resolve_and_analyze pipeline before dashboard.
"""

import os
import json
import time

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

COLOR_BG        = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT    = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_GREEN_DIM = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_RED       = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_RED_DIM   = {"red": 0.200, "green": 0.039, "blue": 0.039}
COLOR_GOLD      = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_HEADER_BG = {"red": 0.055, "green": 0.055, "blue": 0.055}
COLOR_GREY      = {"red": 0.600, "green": 0.600, "blue": 0.600}


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=25):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=0.0):
    try:
        f = float(str(val).replace("+", "").replace("$", "").strip())
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def calc_pnl(odds: float, stake: float, result: str) -> float:
    result = str(result).strip().upper()
    if result not in ("W", "L"):
        return 0.0
    if result == "L":
        return round(-stake, 2)
    if odds >= 0:
        return round(stake * odds / 100, 2)
    else:
        return round(stake * 100 / abs(odds), 2)


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading Scorecard...")
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Scorecard")
    except gspread.WorksheetNotFound:
        print("Scorecard sheet not found — skipping.")
        return

    all_values = with_retry(lambda: ws.get_all_values())
    if not all_values or len(all_values) < 2:
        print("Scorecard is empty — skipping.")
        return

    ws_id   = ws.id
    headers = all_values[0]
    rows    = all_values[1:]

    # Find column indices
    def col(name):
        try:
            return headers.index(name)
        except ValueError:
            return None

    c_model     = col("Model")
    c_player    = col("Player / Pitcher")
    c_your_odds = col("Your Odds")
    c_stake     = col("Stake")
    c_result    = col("Result")
    c_pnl       = col("P&L")
    c_date      = col("Date")

    if any(c is None for c in [c_model, c_your_odds, c_stake, c_result, c_pnl]):
        print("Scorecard columns not found — check headers.")
        return

    # Calculate P&L for each resolved row
    updated = 0
    for i, row in enumerate(rows):
        # Pad row if needed
        while len(row) <= c_pnl:
            row.append("")

        odds_raw   = str(row[c_your_odds]).strip()
        stake_raw  = str(row[c_stake]).strip()
        result_raw = str(row[c_result]).strip().upper()

        if not odds_raw or not stake_raw or result_raw not in ("W", "L"):
            row[c_pnl] = ""
            continue

        odds  = safe_float(odds_raw)
        stake = safe_float(stake_raw)
        if odds == 0 or stake == 0:
            continue

        pnl = calc_pnl(odds, stake, result_raw)
        row[c_pnl] = str(pnl)
        updated += 1

    print(f"  P&L calculated for {updated} resolved bets")

    # Write back
    all_updated = [headers] + rows
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update(all_updated, value_input_option="USER_ENTERED"))

    # ── Summary ───────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=headers)
    df["pnl_num"]   = df["P&L"].apply(lambda x: safe_float(x, np.nan))
    df["stake_num"] = df["Stake"].apply(lambda x: safe_float(x, np.nan))
    df["result_clean"] = df["Result"].str.strip().str.upper()
    df["odds_num"]  = df["Your Odds"].apply(safe_float)

    resolved = df[df["result_clean"].isin(["W", "L"])].copy()

    if resolved.empty:
        print("No resolved bets yet.")
        return

    print("\n" + "="*60)
    print("SCORECARD P&L SUMMARY")
    print("="*60)

    def print_model_summary(label, sub):
        if sub.empty:
            return
        bets   = len(sub)
        wins   = int((sub["result_clean"] == "W").sum())
        total_stake = sub["stake_num"].sum()
        total_pnl   = sub["pnl_num"].sum()
        roi         = round(total_pnl / total_stake * 100, 1) if total_stake > 0 else 0
        win_rate    = round(wins / bets * 100, 1)
        print(f"\n  {label}:")
        print(f"    Bets: {bets} | Wins: {wins} | Win Rate: {win_rate}%")
        print(f"    Staked: ${round(total_stake, 2)} | P&L: {'+'if total_pnl>=0 else ''}{round(total_pnl,2)} | ROI: {'+'if roi>=0 else ''}{roi}%")

    # Overall
    print_model_summary("OVERALL", resolved)

    # By model
    for model in ["HR Single", "HR Parlay", "KS", "HRRBI"]:
        sub = resolved[resolved["Model"] == model]
        if not sub.empty:
            print_model_summary(model, sub)

    # Recent — last 7 days
    df["date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    resolved["date_dt"] = pd.to_datetime(resolved["Date"], errors="coerce")
    max_date = resolved["date_dt"].max()
    recent   = resolved[resolved["date_dt"] >= max_date - pd.Timedelta(days=7)]
    if not recent.empty:
        print_model_summary("Last 7 Days", recent)

    # ── Format P&L cells green/red ────────────────────────────────────────
    reqs = []
    for i, row in enumerate(rows):
        row_idx = i + 1
        pnl_val = safe_float(row[c_pnl] if len(row) > c_pnl else "", np.nan)
        if np.isnan(pnl_val):
            continue
        if pnl_val > 0:
            bg, fg = COLOR_GREEN_DIM, COLOR_GREEN
        elif pnl_val < 0:
            bg, fg = COLOR_RED_DIM, COLOR_RED
        else:
            bg, fg = COLOR_BG, COLOR_GREY

        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": c_pnl, "endColumnIndex": c_pnl + 1},
            "cell": {"userEnteredFormat": {
                "backgroundColor": bg,
                "textFormat": {"foregroundColor": fg, "bold": True},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
        }})

        # Color result cell
        result = str(row[c_result]).strip().upper()
        if result == "W":
            r_bg, r_fg = COLOR_GREEN_DIM, COLOR_GREEN
        elif result == "L":
            r_bg, r_fg = COLOR_RED_DIM, COLOR_RED
        else:
            continue
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": c_result, "endColumnIndex": c_result + 1},
            "cell": {"userEnteredFormat": {
                "backgroundColor": r_bg,
                "textFormat": {"foregroundColor": r_fg, "bold": True},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
        }})

    if reqs:
        try:
            with_retry(lambda: sh.batch_update({"requests": reqs}))
            print("\nScorecard P&L formatting applied.")
        except APIError as e:
            print(f"Scorecard formatting failed: {e}")

    print("Scorecard resolve complete.")


if __name__ == "__main__":
    main()
