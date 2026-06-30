"""
hrrbi_analysis.py
Analyzes HRRBI_All_Scores to surface feature separators, score tiers,
signal performance, and rolling trends. Mirrors hr_analysis.py structure.

Updated: granular odds buckets in Score Tier × Odds Zone cross-tab
to distinguish -110 vs -130 vs -155 vs worse, giving real breakeven picture.
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

COLOR_BG         = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT     = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_WHITE      = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_BLUE       = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_BLUE_DIM   = {"red": 0.055, "green": 0.180, "blue": 0.318}
COLOR_GREEN      = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_GREEN_DIM  = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_RED        = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_RED_DIM    = {"red": 0.200, "green": 0.039, "blue": 0.039}
COLOR_GOLD       = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_HEADER_BG  = {"red": 0.055, "green": 0.055, "blue": 0.055}
COLOR_GREY       = {"red": 0.500, "green": 0.500, "blue": 0.500}
COLOR_PURPLE     = {"red": 0.576, "green": 0.439, "blue": 0.859}
COLOR_PURPLE_DIM = {"red": 0.118, "green": 0.078, "blue": 0.196}
COLOR_ORANGE     = {"red": 0.980, "green": 0.502, "blue": 0.059}
COLOR_ORANGE_DIM = {"red": 0.250, "green": 0.118, "blue": 0.000}


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


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


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


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (breakeven %)."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def signal_label(pct_diff: float, threshold_strong: float = 15.0, threshold_positive: float = 7.0) -> str:
    abs_diff = abs(pct_diff)
    if abs_diff >= threshold_strong:
        return "🔥 STRONG +" if pct_diff > 0 else "🔴 STRONG —"
    if abs_diff >= threshold_positive:
        return "✅ Positive" if pct_diff > 0 else "⚠️ Negative"
    return "↔️ Neutral"


def build_analysis(df: pd.DataFrame) -> dict:
    """
    Build all analysis sections from HRRBI_All_Scores.
    Returns dict of section_name -> list of rows.
    """
    resolved = df[df["over_hit"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    if resolved.empty:
        return {}

    resolved["over_bool"]    = resolved["over_hit"].astype(str).str.strip() == "Yes"
    resolved["date_dt"]      = pd.to_datetime(resolved["date"], errors="coerce")
    resolved["hrrbi_score"]  = resolved["hrrbi_score"].apply(safe_float)

    # Line bucket — 1.5 and 2.5 are fundamentally different bets, never blend
    if "hrrbi_line" in resolved.columns:
        resolved["hrrbi_line_num"] = resolved["hrrbi_line"].apply(safe_float)
    else:
        resolved["hrrbi_line_num"] = 0.0

    def line_bucket(v):
        if v <= 0:   return "unknown"
        if v <= 1.5: return "1.5"
        if v <= 2.5: return "2.5"
        return "3.5+"

    resolved["line_bucket"] = resolved["hrrbi_line_num"].apply(line_bucket)

    total_resolved = len(resolved)
    total_over     = int(resolved["over_bool"].sum())
    total_under    = total_resolved - total_over
    over_rate      = round(total_over / total_resolved * 100, 1) if total_resolved > 0 else 0
    days_of_data   = resolved["date_dt"].nunique()

    summary = {
        "total_resolved": total_resolved,
        "over_hits":      total_over,
        "under_hits":     total_under,
        "over_rate":      over_rate,
        "days_of_data":   days_of_data,
    }

    # ── By Score Tier ──────────────────────────────────────────────────────
    score_tiers = []
    tier_defs = [
        ("17+",     17.0, 999),
        ("15-17",   15.0, 17.0),
        ("13-15",   13.0, 15.0),
        ("11-13",   11.0, 13.0),
        ("9-11",     9.0, 11.0),
        ("7-9",      7.0,  9.0),
        ("Under 7",  0.0,  7.0),
    ]
    for label, lo, hi in tier_defs:
        sub = resolved[(resolved["hrrbi_score"] >= lo) & (resolved["hrrbi_score"] < hi)]
        if sub.empty:
            continue
        total = len(sub)
        hits  = int(sub["over_bool"].sum())
        rate  = round(hits / total * 100, 1)
        avg_score = round(sub["hrrbi_score"].mean(), 1)
        score_tiers.append({
            "Score Tier": label,
            "Total":      total,
            "Over Hits":  hits,
            "Over Rate %": f"{rate}%",
            "Avg Score":  avg_score,
        })
        # Line-specific breakdown within tier
        for lb in ["1.5", "2.5"]:
            lsub = sub[sub["line_bucket"] == lb]
            if len(lsub) < 5:
                continue
            ltotal = len(lsub)
            lhits  = int(lsub["over_bool"].sum())
            lrate  = round(lhits / ltotal * 100, 1)
            score_tiers.append({
                "Score Tier": f"  ↳ Line {lb}",
                "Total":      ltotal,
                "Over Hits":  lhits,
                "Over Rate %": f"{lrate}%",
                "Avg Score":  avg_score,
            })

    # ── By Signal ──────────────────────────────────────────────────────────
    signals = []
    if "prop_signal" in resolved.columns:
        for sig_label in ["OVER 1.5 ✅", "LEAN OVER 1.5", "No Signal"]:
            if sig_label == "No Signal":
                sub = resolved[~resolved["prop_signal"].astype(str).str.contains("OVER", na=False)]
            else:
                sub = resolved[resolved["prop_signal"].astype(str).str.contains(sig_label.replace(" ✅", ""), na=False)]
            if sub.empty:
                continue
            total = len(sub)
            hits  = int(sub["over_bool"].sum())
            rate  = round(hits / total * 100, 1)
            signals.append({
                "Signal":    sig_label,
                "Total":     total,
                "Over Hits": hits,
                "Over Rate %": f"{rate}%",
            })

    # ── By Bat Order ───────────────────────────────────────────────────────
    bat_order_rows = []
    if "avg_bat_order" in resolved.columns:
        resolved["bat_order_num"] = resolved["avg_bat_order"].apply(safe_float)
        for label, lo, hi in [
            ("1-2 (Leadoff/2-hole)", 0, 2.5),
            ("3-5 (Middle)",          2.5, 5.5),
            ("6-9 (Bottom)",          5.5, 10),
        ]:
            sub = resolved[(resolved["bat_order_num"] >= lo) & (resolved["bat_order_num"] < hi)]
            if sub.empty:
                continue
            total = len(sub)
            hits  = int(sub["over_bool"].sum())
            rate  = round(hits / total * 100, 1)
            bat_order_rows.append({
                "Bat Order": label,
                "Total":     total,
                "Over Hits": hits,
                "Over Rate %": f"{rate}%",
            })

    # ── Rolling Trends ─────────────────────────────────────────────────────
    rolling = []
    max_date = resolved["date_dt"].max()
    for label, days in [("Last 7 Days", 7), ("Last 14 Days", 14), ("Last 30 Days", 30), ("All Time", 99999)]:
        sub = resolved[resolved["date_dt"] >= max_date - pd.Timedelta(days=days)]
        if sub.empty:
            continue
        total = len(sub)
        hits  = int(sub["over_bool"].sum())
        rate  = round(hits / total * 100, 1)
        rolling.append({
            "Period":    label,
            "Total":     total,
            "Over Hits": hits,
            "Over Rate %": f"{rate}%",
        })

    # ── Feature Separators ────────────────────────────────────────────────
    features = [
        ("hrrbi_score",       "HRRBI Score"),
        ("avg",               "Season AVG"),
        ("woba",              "wOBA"),
        ("obp",               "OBP"),
        ("bb_pct",            "BB%"),
        ("ld_pct",            "LD%"),
        ("gb_pct",            "GB%"),
        ("hard_hit_pct_season", "Hard Hit% (Season)"),
        ("avg_ev_7d",         "Avg EV (7d)"),
        ("hard_hit_pct_7d",   "Hard Hit% (7d)"),
        ("avg_14d",           "AVG (14d)"),
        ("momentum_score",    "Momentum Score"),
        ("avg_bat_order",     "Avg Bat Order"),
        ("opp_whip",          "Opp WHIP"),
        ("opp_k_pct_season",  "Opp K%"),
        ("opp_bb_pct_season", "Opp BB%"),
        ("opp_hard_hit_pct",  "Opp Hard Hit%"),
        ("opp_hr_per_fb",     "Opp HR/FB%"),
        ("opp_fps_pct",       "Opp FPS%"),
        ("opp_babip",         "Opp BABIP"),
        ("game_total",        "Game Total"),
        ("park_hr_factor",    "Park HR Factor"),
    ]

    separators = []
    over_df  = resolved[resolved["over_bool"]]
    under_df = resolved[~resolved["over_bool"]]

    for col, label in features:
        if col not in resolved.columns:
            continue
        resolved[col] = resolved[col].apply(safe_float)
        over_vals  = over_df[col].apply(safe_float).dropna()
        under_vals = under_df[col].apply(safe_float).dropna()
        if len(over_vals) < 5 or len(under_vals) < 5:
            continue
        over_avg  = round(over_vals.mean(), 3)
        under_avg = round(under_vals.mean(), 3)
        diff      = round(over_avg - under_avg, 3)
        pct_diff  = round((diff / under_avg * 100), 1) if under_avg != 0 else 0.0
        separators.append({
            "Feature":        label,
            "Over Avg":       over_avg,
            "Non-Over Avg":   under_avg,
            "Difference":     f"+{diff}" if diff >= 0 else str(diff),
            "% Difference":   f"+{pct_diff}%" if pct_diff >= 0 else f"{pct_diff}%",
            "Signal":         signal_label(pct_diff),
        })

    separators.sort(key=lambda x: abs(safe_float(x["% Difference"].replace("%", "").replace("+", ""))), reverse=True)

    # ── Score Tier × Odds Zone cross-tab ─────────────────────────────────
    # Granular odds buckets so breakeven is meaningful:
    # Minus odds split into -110 or better / -111 to -130 / -131 to -155 / -156 or worse
    # Plus odds kept as before
    if "over_odds" in resolved.columns:
        resolved["odds_num"] = resolved["over_odds"].apply(safe_float)
    else:
        resolved["odds_num"] = 0.0

    tier_odds_rows = []
    hrrbi_tier_defs = [
        ("13-15",  13.0, 15.0),
        ("11-13",  11.0, 13.0),
        ("9-11",    9.0, 11.0),
        ("7-9",     7.0,  9.0),
        ("Under 7", 0.0,  7.0),
    ]

    # Granular odds buckets with breakeven % shown in label
    # Minus odds: lower number = worse odds (more negative)
    # -110 breakeven: 52.4% | -130 breakeven: 56.5% | -156 breakeven: 60.9%
    odds_defs = [
        ("-110 or better (BE: 52.4%)",  -110,    0),   # -110 to -100 (even money)
        ("-111 to -130 (BE: 52-56%)",   -130, -110),
        ("-131 to -155 (BE: 56-61%)",   -155, -130),
        ("-156 or worse (BE: 61%+)",    -999, -155),
        ("+100 to +110 (BE: 47-48%)",    100,  111),
        ("+111 to +120 (BE: 45-47%)",    111,  121),
        ("+121+ (BE: <45%)",             121,  999),
    ]

    # Line is now part of the key — 1.5 and 2.5 are different bets and
    # are NEVER blended together. This matches dashboard.py's calc_hrrbi_value.
    line_buckets_to_show = ["1.5", "2.5"]

    for line_lb in line_buckets_to_show:
        line_sub_all = resolved[resolved["line_bucket"] == line_lb]
        if line_sub_all.empty:
            continue
        for tier_label, t_lo, t_hi in hrrbi_tier_defs:
            tier_sub = line_sub_all[(line_sub_all["hrrbi_score"] >= t_lo) & (line_sub_all["hrrbi_score"] < t_hi)]
            if tier_sub.empty:
                continue
            for odds_label, o_lo, o_hi in odds_defs:
                if o_hi <= 0:
                    sub = tier_sub[
                        (tier_sub["odds_num"] >= o_lo) &
                        (tier_sub["odds_num"] < o_hi)
                    ]
                else:
                    sub = tier_sub[
                        (tier_sub["odds_num"] >= o_lo) &
                        (tier_sub["odds_num"] < o_hi)
                    ]

                if len(sub) < 3:
                    continue

                total    = len(sub)
                hits     = int(sub["over_bool"].sum())
                rate     = round(hits / total * 100, 1)

                avg_odds = sub["odds_num"].mean()
                breakeven = round(american_to_implied(avg_odds) * 100, 1)
                edge      = round(rate - breakeven, 1)
                edge_str  = f"+{edge}%" if edge >= 0 else f"{edge}%"

                tier_odds_rows.append({
                    "Score Tier | Odds": f"Line {line_lb} | {tier_label} | {odds_label}",
                    "Total":     total,
                    "Over Hits": hits,
                    "Over Rate %": f"{rate}%",
                    "Breakeven": f"{breakeven}%",
                    "Edge":      edge_str,
                })

    return {
        "summary":        summary,
        "score_tiers":    score_tiers,
        "signals":        signals,
        "bat_order":      bat_order_rows,
        "rolling":        rolling,
        "separators":     separators,
        "tier_odds_rows": tier_odds_rows,
    }


def write_analysis(gc: gspread.Client, sheet_id: str, analysis: dict) -> None:
    if not analysis:
        print("No analysis data to write.")
        return

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("HRRBI_Analysis")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HRRBI_Analysis", rows=300, cols=8)

    summary = analysis.get("summary", {})
    all_values = []

    # Title row
    all_values.append([
        "📊  HRRBI MODEL ANALYSIS — Up to date",
        "", "", "", "", "", "", "",
    ])
    all_values.append([
        f"Total Resolved: {summary.get('total_resolved', 0)}",
        f"Over Hits: {summary.get('over_hits', 0)}",
        f"Over Rate: {summary.get('over_rate', 0)}%",
        f"Under Rate: {round(100 - summary.get('over_rate', 0), 1)}%",
        f"Days of Data: {summary.get('days_of_data', 0)}",
        "", "", "",
    ])
    all_values.append(["", "", "", "", "", "", "", ""])

    # Score Tiers
    all_values.append(["🎯  BY SCORE TIER", "", "", "", "", "", "", ""])
    all_values.append(["Score Tier", "Total", "Over Hits", "Over Rate %", "Avg Score", "", "", ""])
    for r in analysis.get("score_tiers", []):
        all_values.append([
            r["Score Tier"], str(r["Total"]), str(r["Over Hits"]),
            r["Over Rate %"], str(r["Avg Score"]), "", "", "",
        ])
    all_values.append(["", "", "", "", "", "", "", ""])

    # By Signal
    all_values.append(["📢  BY SIGNAL", "", "", "", "", "", "", ""])
    all_values.append(["Signal", "Total", "Over Hits", "Over Rate %", "", "", "", ""])
    for r in analysis.get("signals", []):
        all_values.append([
            r["Signal"], str(r["Total"]), str(r["Over Hits"]),
            r["Over Rate %"], "", "", "", "",
        ])
    all_values.append(["", "", "", "", "", "", "", ""])

    # By Bat Order
    all_values.append(["📋  BY BAT ORDER", "", "", "", "", "", "", ""])
    all_values.append(["Bat Order", "Total", "Over Hits", "Over Rate %", "", "", "", ""])
    for r in analysis.get("bat_order", []):
        all_values.append([
            r["Bat Order"], str(r["Total"]), str(r["Over Hits"]),
            r["Over Rate %"], "", "", "", "",
        ])
    all_values.append(["", "", "", "", "", "", "", ""])

    # Rolling Trends
    all_values.append(["📈  ROLLING TRENDS", "", "", "", "", "", "", ""])
    all_values.append(["Period", "Total", "Over Hits", "Over Rate %", "", "", "", ""])
    for r in analysis.get("rolling", []):
        all_values.append([
            r["Period"], str(r["Total"]), str(r["Over Hits"]),
            r["Over Rate %"], "", "", "", "",
        ])
    all_values.append(["", "", "", "", "", "", "", ""])

    # Score Tier × Odds Zone — now with Breakeven and Edge columns
    if analysis.get("tier_odds_rows"):
        all_values.append(["💰  SCORE TIER × ODDS ZONE — Granular Breakeven Analysis", "", "", "", "", "", "", ""])
        all_values.append(["Score Tier | Odds", "Total", "Over Hits", "Over Rate %", "Breakeven %", "Edge vs BE", "", ""])
        for r in analysis.get("tier_odds_rows", []):
            all_values.append([
                r["Score Tier | Odds"], str(r["Total"]), str(r["Over Hits"]),
                r["Over Rate %"], r["Breakeven"], r["Edge"], "", "",
            ])
        all_values.append(["", "", "", "", "", "", "", ""])

    # Feature Separators
    all_values.append(["🔬  FEATURE SEPARATORS — Over vs Non-Over", "", "", "", "", "", "", ""])
    all_values.append(["Feature", "Over Avg", "Non-Over Avg", "Difference", "% Difference", "Signal", "", ""])
    for r in analysis.get("separators", []):
        all_values.append([
            r["Feature"], str(r["Over Avg"]), str(r["Non-Over Avg"]),
            r["Difference"], r["% Difference"], r["Signal"], "", "",
        ])

    with_retry(lambda: ws.update(all_values))

    # ── Formatting ────────────────────────────────────────────────────────
    ws_id      = ws.id
    total_rows = len(all_values)
    total_cols = 8
    reqs       = []

    # Base style
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

    # Title row
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BLUE_DIM,
            "textFormat": {"foregroundColor": COLOR_BLUE, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 13},
            "verticalAlignment": "MIDDLE",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})

    # Summary row
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": 2,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_HEADER_BG,
            "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 11},
            "verticalAlignment": "MIDDLE",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})

    # Walk rows for section/col/data formatting
    section_header_rows = []
    col_header_rows     = []
    data_rows_info      = []

    for i, row in enumerate(all_values):
        if i < 3:
            continue
        cell = str(row[0]).strip()
        if any(cell.startswith(s) for s in ["🎯", "📢", "📋", "📈", "🔬", "💰"]):
            section_header_rows.append(i)
        elif cell in ["Score Tier", "Signal", "Bat Order", "Period", "Feature",
                      "Score Tier | Odds", "Score Tier | Odds Zone"]:
            col_header_rows.append(i)
        elif any(str(v).strip() for v in row):
            data_rows_info.append(i)

    # Section headers
    for r in section_header_rows:
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                      "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_BLUE_DIM,
                "textFormat": {"foregroundColor": COLOR_BLUE, "bold": True,
                               "fontFamily": "Roboto", "fontSize": 11},
                "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
        }})

    # Column headers
    for r in col_header_rows:
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                      "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_HEADER_BG,
                "textFormat": {"foregroundColor": COLOR_GREY, "bold": True,
                               "fontFamily": "Roboto", "fontSize": 9},
                "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }})

    # Data rows
    data_counter = 0
    for r in data_rows_info:
        bg = COLOR_BG if data_counter % 2 == 0 else COLOR_BG_ALT
        data_counter += 1
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                      "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)",
        }})

        # Color Over Rate % (col 3)
        rate_str = str(all_values[r][3]).replace("%", "").strip()
        try:
            rate_val = float(rate_str)
            if rate_val >= 60:
                rate_bg = COLOR_GREEN_DIM; rate_fg = COLOR_GREEN
            elif rate_val >= 55:
                rate_bg = {"red": 0.039, "green": 0.120, "blue": 0.039}
                rate_fg = COLOR_GREEN
            elif rate_val >= 50:
                rate_bg = bg; rate_fg = COLOR_WHITE
            elif rate_val <= 40:
                rate_bg = COLOR_RED_DIM; rate_fg = COLOR_RED
            else:
                rate_bg = bg; rate_fg = COLOR_WHITE
            if rate_val >= 55 or rate_val <= 40:
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 3, "endColumnIndex": 4},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": rate_bg,
                        "textFormat": {"foregroundColor": rate_fg,
                                       "bold": rate_val >= 60 or rate_val <= 40},
                        "horizontalAlignment": "CENTER",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }})
        except (ValueError, TypeError):
            pass

        # Color Edge column (col 5) for tier × odds rows
        edge_str = str(all_values[r][5]).strip() if len(all_values[r]) > 5 else ""
        if edge_str and ("+" in edge_str or (edge_str.startswith("-") and "%" in edge_str)):
            try:
                edge_val = float(edge_str.replace("%", "").replace("+", ""))
                if edge_val >= 5:
                    e_bg = COLOR_GREEN_DIM; e_fg = COLOR_GREEN
                elif edge_val >= 0:
                    e_bg = {"red": 0.039, "green": 0.120, "blue": 0.039}
                    e_fg = COLOR_GREEN
                elif edge_val < 0:
                    e_bg = COLOR_RED_DIM; e_fg = COLOR_RED
                else:
                    e_bg = bg; e_fg = COLOR_WHITE
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 5, "endColumnIndex": 6},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": e_bg,
                        "textFormat": {"foregroundColor": e_fg, "bold": abs(edge_val) >= 5},
                        "horizontalAlignment": "CENTER",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                }})
            except (ValueError, TypeError):
                pass

        # Color signal column (col 5) for feature separator rows
        signal_str = str(all_values[r][5]).strip() if len(all_values[r]) > 5 else ""
        if signal_str and "STRONG" in signal_str or "Positive" in signal_str or "Negative" in signal_str:
            if "STRONG +" in signal_str:
                sig_bg = COLOR_GREEN_DIM; sig_fg = COLOR_GREEN
            elif "STRONG —" in signal_str:
                sig_bg = COLOR_RED_DIM; sig_fg = COLOR_RED
            elif "Positive" in signal_str:
                sig_bg = {"red": 0.039, "green": 0.120, "blue": 0.039}
                sig_fg = COLOR_GREEN
            elif "Negative" in signal_str:
                sig_bg = {"red": 0.150, "green": 0.039, "blue": 0.039}
                sig_fg = COLOR_RED
            else:
                sig_bg = bg; sig_fg = COLOR_GREY
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 5, "endColumnIndex": 6},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": sig_bg,
                    "textFormat": {"foregroundColor": sig_fg, "bold": "STRONG" in signal_str},
                    "horizontalAlignment": "CENTER",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
            }})

    # Column widths — expanded for longer odds labels
    col_widths = [320, 90, 90, 100, 110, 110, 80, 80]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS",
                      "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w}, "fields": "pixelSize",
        }})

    # Row heights
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": ws_id, "dimension": "ROWS",
                  "startIndex": 0, "endIndex": total_rows},
        "properties": {"pixelSize": 28}, "fields": "pixelSize",
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
            "tabColorStyle": {"rgbColor": COLOR_BLUE},
        },
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("HRRBI_Analysis formatting applied.")
    except APIError as e:
        print(f"HRRBI_Analysis formatting failed: {e}")

    print("HRRBI_Analysis written.")


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HRRBI_All_Scores...")
    df = read_sheet_raw(gc, sheet_id, "HRRBI_All_Scores")

    if df.empty:
        print("HRRBI_All_Scores is empty — nothing to analyze.")
        return

    resolved = df[df["over_hit"].astype(str).str.strip().isin(["Yes", "No"])]
    print(f"Total scored: {len(df)} | Resolved: {len(resolved)}")

    if resolved.empty:
        print("No resolved picks yet — run resolve_picks.py first.")
        return

    print("Building analysis...")
    analysis = build_analysis(df)

    print("Writing HRRBI_Analysis sheet...")
    write_analysis(gc, sheet_id, analysis)
    print("Done.")


if __name__ == "__main__":
    main()
