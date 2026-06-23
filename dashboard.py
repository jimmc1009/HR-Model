"""
ks_analysis.py
Reads KS_All_Scores daily after resolution and writes feature analysis
to KS_Analysis sheet. Runs automatically at end of pipeline.
"""

import os
import json
import time
from datetime import date

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
COLOR_GOLD_DIM  = {"red": 0.200, "green": 0.160, "blue": 0.000}
COLOR_BLUE      = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_BLUE_DIM  = {"red": 0.055, "green": 0.180, "blue": 0.318}
COLOR_TEAL      = {"red": 0.047, "green": 0.450, "blue": 0.353}
COLOR_TEAL_DIM  = {"red": 0.020, "green": 0.120, "blue": 0.090}
COLOR_PURPLE    = {"red": 0.576, "green": 0.439, "blue": 0.859}
COLOR_PURPLE_DIM= {"red": 0.118, "green": 0.078, "blue": 0.196}
COLOR_GREY      = {"red": 0.600, "green": 0.600, "blue": 0.600}
COLOR_HEADER_BG = {"red": 0.055, "green": 0.055, "blue": 0.055}


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


def read_sheet(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
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


def build_analysis(df: pd.DataFrame) -> dict:
    resolved = df[df["over_hit"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    if resolved.empty:
        return {}

    resolved["over_bool"]  = resolved["over_hit"] == "Yes"
    resolved["under_bool"] = resolved["under_hit"] == "Yes"
    resolved["ks_score"]   = resolved["ks_score"].apply(safe_float)
    resolved["date_dt"]    = pd.to_datetime(resolved["date"], errors="coerce")
    resolved["k_line"]     = resolved["k_line"].apply(safe_float)
    resolved["actual_ks"]  = resolved["actual_ks"].apply(lambda x: safe_float(x, -1))
    resolved["projected"]  = resolved["projected_ks"].apply(safe_float)

    days_of_data = resolved["date_dt"].nunique()
    total        = len(resolved)
    over_hits    = int(resolved["over_bool"].sum())
    under_hits   = int(resolved["under_bool"].sum())
    over_rate    = round(over_hits / total * 100, 1) if total > 0 else 0.0
    under_rate   = round(under_hits / total * 100, 1) if total > 0 else 0.0

    numeric_features = [
        "k_pct_season", "swstr_pct", "chase_rate", "k_per_9",
        "fastball_velo", "avg_ip_per_start", "k_per_start_21d",
        "whip_proxy", "bb_pct_season",
        "opp_team_k_pct", "opp_chase_rate", "opp_whiff_rate",
    ]
    for col in numeric_features:
        if col in resolved.columns:
            resolved[col] = resolved[col].apply(safe_float)
        else:
            resolved[col] = 0.0

    # ── Score tiers — includes Under 0 for negative scores ────────────────
    score_tier_defs = [
        ("12+",    12,  999),
        ("10-12",  10,   12),
        ("8-10",    8,   10),
        ("6-8",     6,    8),
        ("4-6",     4,    6),
        ("2-4",     2,    4),
        ("Under 2", 0,    2),
        ("Under 0", -999, 0),  # negative scores — weak pitchers/extreme underdogs
    ]

    score_tiers = []
    for label, lo, hi in score_tier_defs:
        sub = resolved[(resolved["ks_score"] >= lo) & (resolved["ks_score"] < hi)]
        if sub.empty:
            continue
        n         = len(sub)
        over_h    = int(sub["over_bool"].sum())
        under_h   = int(sub["under_bool"].sum())
        over_r    = round(over_h / n * 100, 1)
        under_r   = round(under_h / n * 100, 1)
        avg_proj  = round(sub["projected"].mean(), 1) if not sub["projected"].isna().all() else 0.0
        score_tiers.append({
            "label": label, "total": n,
            "over_hits": over_h, "over_rate": over_r,
            "under_hits": under_h, "under_rate": under_r,
            "avg_proj": avg_proj,
        })

    # ── Score tier × Line cross-tab — includes Under 0 ───────────────────
    line_vals = [4.5, 5.5, 6.5, 7.5]
    with_line = resolved[resolved["k_line"] > 0].copy()

    score_x_line_rows = []
    for tier_label, lo, hi in score_tier_defs:
        tier_sub = with_line[(with_line["ks_score"] >= lo) & (with_line["ks_score"] < hi)]
        if tier_sub.empty:
            continue
        for line_val in line_vals:
            sub = tier_sub[tier_sub["k_line"] == line_val]
            if sub.empty or len(sub) < 3:
                continue
            n       = len(sub)
            over_h  = int(sub["over_bool"].sum())
            under_h = int(sub["under_bool"].sum())
            over_r  = round(over_h / n * 100, 1)
            under_r = round(under_h / n * 100, 1)
            score_x_line_rows.append({
                "tier":       tier_label,
                "line":       f"O/U {line_val}",
                "total":      n,
                "over_hits":  over_h,
                "over_rate":  over_r,
                "under_hits": under_h,
                "under_rate": under_r,
            })

    # ── By signal ─────────────────────────────────────────────────────────
    signal_rows = []
    for sig_label in ["OVER", "LEAN OVER", "UNDER", "No Signal"]:
        if sig_label == "No Signal":
            sub = resolved[~resolved["prop_signal"].astype(str).str.contains("OVER|UNDER", na=False)]
        elif sig_label == "OVER":
            sub = resolved[resolved["prop_signal"].astype(str).str.contains("✅", na=False)]
        elif sig_label == "UNDER":
            sub = resolved[resolved["prop_signal"].astype(str).str.contains("UNDER", na=False)]
        else:
            sub = resolved[
                resolved["prop_signal"].astype(str).str.contains("LEAN", na=False) &
                ~resolved["prop_signal"].astype(str).str.contains("✅", na=False)
            ]
        if sub.empty:
            continue
        n      = len(sub)
        over_h = int(sub["over_bool"].sum())
        under_h = int(sub["under_bool"].sum())
        over_r = round(over_h / n * 100, 1)
        under_r = round(under_h / n * 100, 1)
        signal_rows.append({
            "label": sig_label, "total": n,
            "over_hits": over_h, "over_rate": over_r,
            "under_hits": under_h, "under_rate": under_r,
        })

    # ── By line ───────────────────────────────────────────────────────────
    line_rows = []
    for line_val in [4.5, 5.5, 6.5, 7.5, 8.5]:
        sub = resolved[resolved["k_line"] == line_val]
        if sub.empty:
            continue
        n       = len(sub)
        over_h  = int(sub["over_bool"].sum())
        under_h = int(sub["under_bool"].sum())
        over_r  = round(over_h / n * 100, 1)
        under_r = round(under_h / n * 100, 1)
        line_rows.append({
            "label": f"Line {line_val}", "total": n,
            "over_hits": over_h, "over_rate": over_r,
            "under_hits": under_h, "under_rate": under_r,
        })

    # ── Projection accuracy ───────────────────────────────────────────────
    proj_rows = []
    if not with_line.empty:
        with_line["proj_edge"] = with_line["projected"] - with_line["k_line"]
        for label, lo, hi in [
            ("Proj +1.5+",    1.5,  99),
            ("Proj +0.5-1.5", 0.5,  1.5),
            ("Proj 0-0.5",    0.0,  0.5),
            ("Proj -0.5-0",  -0.5,  0.0),
            ("Proj -1.5--0.5",-1.5,-0.5),
            ("Proj -1.5-",   -99, -1.5),
        ]:
            sub = with_line[(with_line["proj_edge"] >= lo) & (with_line["proj_edge"] < hi)]
            if sub.empty:
                continue
            n      = len(sub)
            over_h = int(sub["over_bool"].sum())
            over_r = round(over_h / n * 100, 1)
            proj_rows.append({
                "label": label, "total": n,
                "over_hits": over_h, "over_rate": over_r,
            })

    # ── Feature separators ────────────────────────────────────────────────
    over_yes = resolved[resolved["over_bool"]]
    over_no  = resolved[~resolved["over_bool"]]

    feature_labels = {
        "bb_pct_season":    "BB%",
        "avg_ip_per_start": "Avg IP/Start",
        "whip_proxy":       "WHIP",
        "chase_rate":       "Chase Rate%",
        "k_per_9":          "K/9",
        "k_pct_season":     "K% (Season)",
        "opp_team_k_pct":   "Opp Team K%",
        "opp_whiff_rate":   "Opp Whiff Rate%",
        "swstr_pct":        "SwStr%",
        "fastball_velo":    "Fastball Velo",
        "k_per_start_21d":  "K/Start (21d)",
        "opp_chase_rate":   "Opp Chase Rate%",
    }

    feature_separators = []
    for col, label in feature_labels.items():
        if col not in resolved.columns:
            continue
        yes_avg  = round(over_yes[col].mean(), 3) if not over_yes.empty else 0.0
        no_avg   = round(over_no[col].mean(), 3)  if not over_no.empty  else 0.0
        diff     = round(yes_avg - no_avg, 3)
        pct_diff = round((diff / no_avg * 100), 1) if no_avg != 0 else 0.0
        feature_separators.append({
            "label":    label,
            "yes_avg":  yes_avg,
            "no_avg":   no_avg,
            "diff":     diff,
            "pct_diff": pct_diff,
        })

    feature_separators.sort(key=lambda x: abs(x["pct_diff"]), reverse=True)

    # ── Rolling trends ────────────────────────────────────────────────────
    max_date  = resolved["date_dt"].max()
    roll_rows = []
    for label, days in [("Last 7 Days", 7), ("Last 14 Days", 14), ("Last 30 Days", 30), ("All Time", 9999)]:
        cutoff = max_date - pd.Timedelta(days=days) if days < 9999 else pd.Timestamp("2000-01-01")
        sub    = resolved[resolved["date_dt"] >= cutoff]
        if sub.empty:
            continue
        n      = len(sub)
        over_h = int(sub["over_bool"].sum())
        over_r = round(over_h / n * 100, 1)
        roll_rows.append({
            "label": label, "total": n,
            "over_hits": over_h, "over_rate": over_r,
        })

    return {
        "days_of_data":       days_of_data,
        "total":              total,
        "over_hits":          over_hits,
        "under_hits":         under_hits,
        "over_rate":          over_rate,
        "under_rate":         under_rate,
        "score_tiers":        score_tiers,
        "score_x_line":       score_x_line_rows,
        "signal_rows":        signal_rows,
        "line_rows":          line_rows,
        "proj_rows":          proj_rows,
        "feature_separators": feature_separators,
        "roll_rows":          roll_rows,
    }


def write_analysis(gc: gspread.Client, sheet_id: str, analysis: dict) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("KS_Analysis")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="KS_Analysis", rows=300, cols=8)

    ws_id = ws.id
    today = date.today().strftime("%B %d, %Y")

    all_values     = []
    section_starts = {}

    def add_section(title, headers, rows_data, key_fn=None):
        section_starts[title] = len(all_values)
        all_values.append([title] + [""] * (len(headers) - 1))
        all_values.append(headers)
        for r in rows_data:
            if callable(key_fn):
                all_values.append(key_fn(r))
            else:
                all_values.append(r)
        all_values.append([""] * len(headers))

    # ── Header ────────────────────────────────────────────────────────────
    all_values.append([
        f"⚾  KS MODEL ANALYSIS — Updated {today}",
        "", "", "", "", "", "", ""
    ])
    all_values.append([
        f"Total Resolved: {analysis['total']:,}",
        f"Over Hits: {analysis['over_hits']:,}",
        f"Over Rate: {analysis['over_rate']}%",
        f"Under Rate: {analysis['under_rate']}%",
        f"Days of Data: {analysis['days_of_data']}",
        "", "", ""
    ])
    all_values.append([""] * 8)

    # ── Score Tiers ───────────────────────────────────────────────────────
    add_section(
        "🎯  BY SCORE TIER",
        ["Score Tier", "Total", "Over Hits", "Over Rate %", "Under Hits", "Under Rate %", "Avg Proj K", ""],
        analysis["score_tiers"],
        lambda r: [r["label"], r["total"], r["over_hits"], f"{r['over_rate']}%",
                   r["under_hits"], f"{r['under_rate']}%", r["avg_proj"], ""]
    )

    # ── Score Tier × Line Cross-Tab ───────────────────────────────────────
    add_section(
        "📊  SCORE TIER × LINE",
        ["Score Tier", "Line", "Total", "Over Hits", "Over Rate %", "Under Hits", "Under Rate %", ""],
        analysis["score_x_line"],
        lambda r: [r["tier"], r["line"], r["total"], r["over_hits"], f"{r['over_rate']}%",
                   r["under_hits"], f"{r['under_rate']}%", ""]
    )

    # ── By Signal ─────────────────────────────────────────────────────────
    add_section(
        "📡  BY SIGNAL",
        ["Signal", "Total", "Over Hits", "Over Rate %", "Under Hits", "Under Rate %", "", ""],
        analysis["signal_rows"],
        lambda r: [r["label"], r["total"], r["over_hits"], f"{r['over_rate']}%",
                   r["under_hits"], f"{r['under_rate']}%", "", ""]
    )

    # ── By Line ───────────────────────────────────────────────────────────
    add_section(
        "📏  BY LINE",
        ["Line", "Total", "Over Hits", "Over Rate %", "Under Hits", "Under Rate %", "", ""],
        analysis["line_rows"],
        lambda r: [r["label"], r["total"], r["over_hits"], f"{r['over_rate']}%",
                   r["under_hits"], f"{r['under_rate']}%", "", ""]
    )

    # ── Projection Accuracy ───────────────────────────────────────────────
    add_section(
        "🎯  PROJECTION ACCURACY",
        ["Proj Edge vs Line", "Total", "Over Hits", "Over Rate %", "", "", "", ""],
        analysis["proj_rows"],
        lambda r: [r["label"], r["total"], r["over_hits"], f"{r['over_rate']}%", "", "", "", ""]
    )

    # ── Rolling Trends ────────────────────────────────────────────────────
    add_section(
        "📈  ROLLING TRENDS",
        ["Period", "Total", "Over Hits", "Over Rate %", "", "", "", ""],
        analysis["roll_rows"],
        lambda r: [r["label"], r["total"], r["over_hits"], f"{r['over_rate']}%", "", "", "", ""]
    )

    # ── Feature Separators ────────────────────────────────────────────────
    section_starts["features"] = len(all_values)
    all_values.append(["🔬  FEATURE SEPARATORS — Over Hitters vs Non-Over Hitters", "", "", "", "", "", "", ""])
    all_values.append(["Feature", "Over Hitters Avg", "Non-Over Avg", "Difference", "% Difference", "Signal", "", ""])

    for r in analysis["feature_separators"]:
        if r["pct_diff"] >= 15:
            signal = "🔥 STRONG +"
        elif r["pct_diff"] >= 8:
            signal = "✅ Positive"
        elif r["pct_diff"] <= -15:
            signal = "🚨 STRONG —"
        elif r["pct_diff"] <= -8:
            signal = "⚠️ Negative"
        else:
            signal = "↔️ Neutral"
        all_values.append([
            r["label"], r["yes_avg"], r["no_avg"],
            f"+{r['diff']}" if r["diff"] >= 0 else str(r["diff"]),
            f"+{r['pct_diff']}%" if r["pct_diff"] >= 0 else f"{r['pct_diff']}%",
            signal, "", ""
        ])
    all_values.append([""] * 8)

    with_retry(lambda: ws.update(all_values))

    # ── Formatting ────────────────────────────────────────────────────────
    total_rows = len(all_values)
    total_cols = 8
    reqs       = []

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP",
            "horizontalAlignment": "CENTER",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy,horizontalAlignment)",
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows,
                  "startColumnIndex": 0, "endColumnIndex": 1},
        "cell": {"userEnteredFormat": {"horizontalAlignment": "LEFT"}},
        "fields": "userEnteredFormat(horizontalAlignment)",
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": total_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_TEAL_DIM,
            "textFormat": {"foregroundColor": COLOR_TEAL, "bold": True,
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
            "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 11},
            "verticalAlignment": "MIDDLE",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
    }})

    section_colors = {
        "🎯  BY SCORE TIER":        (COLOR_PURPLE, COLOR_PURPLE_DIM),
        "📊  SCORE TIER × LINE":    (COLOR_BLUE,   COLOR_BLUE_DIM),
        "📡  BY SIGNAL":            (COLOR_GREEN,  COLOR_GREEN_DIM),
        "📏  BY LINE":              (COLOR_GOLD,   COLOR_GOLD_DIM),
        "🎯  PROJECTION ACCURACY":  (COLOR_BLUE,   COLOR_BLUE_DIM),
        "📈  ROLLING TRENDS":       (COLOR_GREEN,  COLOR_GREEN_DIM),
        "features":                  (COLOR_PURPLE, COLOR_PURPLE_DIM),
    }

    for title, start_row in section_starts.items():
        color, dim = section_colors.get(title, (COLOR_BLUE, COLOR_BLUE_DIM))
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": start_row, "endRowIndex": start_row + 1,
                      "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": dim,
                "textFormat": {"foregroundColor": color, "bold": True,
                               "fontFamily": "Roboto", "fontSize": 12},
                "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)",
        }})
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": start_row + 1, "endRowIndex": start_row + 2,
                      "startColumnIndex": 0, "endColumnIndex": total_cols},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_HEADER_BG,
                "textFormat": {"foregroundColor": color, "bold": True,
                               "fontFamily": "Roboto", "fontSize": 10},
                "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }})

    for row_idx, row in enumerate(all_values):
        if len(row) >= 4 and str(row[3]).endswith("%"):
            try:
                rate_val = float(str(row[3]).replace("%", ""))
                if rate_val >= 60:
                    rate_bg, rate_fg = COLOR_GREEN_DIM, COLOR_GREEN
                elif rate_val >= 50:
                    rate_bg, rate_fg = COLOR_TEAL_DIM, COLOR_TEAL
                elif rate_val <= 35:
                    rate_bg, rate_fg = COLOR_RED_DIM, COLOR_RED
                else:
                    rate_bg = rate_fg = None
                if rate_bg:
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                                  "startColumnIndex": 3, "endColumnIndex": 4},
                        "cell": {"userEnteredFormat": {
                            "backgroundColor": rate_bg,
                            "textFormat": {"foregroundColor": rate_fg, "bold": True},
                            "horizontalAlignment": "CENTER",
                        }},
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                    }})
            except Exception:
                pass

        if len(row) >= 6 and str(row[5]).endswith("%"):
            try:
                rate_val = float(str(row[5]).replace("%", ""))
                if rate_val >= 60:
                    rate_bg, rate_fg = COLOR_GREEN_DIM, COLOR_GREEN
                elif rate_val >= 50:
                    rate_bg, rate_fg = COLOR_TEAL_DIM, COLOR_TEAL
                elif rate_val <= 35:
                    rate_bg, rate_fg = COLOR_RED_DIM, COLOR_RED
                else:
                    rate_bg = rate_fg = None
                if rate_bg:
                    reqs.append({"repeatCell": {
                        "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                                  "startColumnIndex": 5, "endColumnIndex": 6},
                        "cell": {"userEnteredFormat": {
                            "backgroundColor": rate_bg,
                            "textFormat": {"foregroundColor": rate_fg, "bold": True},
                            "horizontalAlignment": "CENTER",
                        }},
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
                    }})
            except Exception:
                pass

    feature_start = section_starts.get("features", 0) + 2
    for i, r in enumerate(analysis["feature_separators"]):
        row_idx = feature_start + i
        if r["pct_diff"] >= 15:
            fg, bg = COLOR_GREEN, COLOR_GREEN_DIM
        elif r["pct_diff"] >= 8:
            fg, bg = COLOR_TEAL, COLOR_TEAL_DIM
        elif r["pct_diff"] <= -15:
            fg, bg = COLOR_RED, COLOR_RED_DIM
        elif r["pct_diff"] <= -8:
            fg, bg = COLOR_GOLD, COLOR_GOLD_DIM
        else:
            fg, bg = COLOR_GREY, COLOR_BG
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                      "startColumnIndex": 5, "endColumnIndex": 6},
            "cell": {"userEnteredFormat": {
                "backgroundColor": bg,
                "textFormat": {"foregroundColor": fg, "bold": True},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)",
        }})

    for row_idx, row in enumerate(all_values):
        if row_idx < 3:
            continue
        is_section    = any(str(row[0]).startswith(e) for e in ["🎯", "📡", "📏", "📈", "🔬", "📊"])
        is_header_row = row_idx in [s + 1 for s in section_starts.values()]
        is_empty      = not any(str(v).strip() for v in row)
        if not is_section and not is_header_row and not is_empty:
            bg = COLOR_BG if row_idx % 2 == 0 else COLOR_BG_ALT
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1,
                          "startColumnIndex": 0, "endColumnIndex": total_cols},
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }})

    col_widths = [200, 100, 80, 110, 90, 110, 100, 80]
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
            "tabColorStyle": {"rgbColor": COLOR_TEAL},
        },
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("KS_Analysis formatted successfully.")
    except APIError as e:
        print(f"KS_Analysis formatting failed: {e}")

    print("KS_Analysis updated.")


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading KS_All_Scores...")
    df = read_sheet(gc, sheet_id, "KS_All_Scores")

    if df.empty:
        print("KS_All_Scores is empty — skipping analysis.")
        return

    resolved_count = (df["over_hit"].astype(str).str.strip().isin(["Yes", "No"])).sum()
    print(f"  {len(df)} total rows, {resolved_count} resolved")

    if resolved_count < 10:
        print("Not enough resolved data yet — skipping analysis.")
        return

    print("Building KS analysis...")
    analysis = build_analysis(df)

    if not analysis:
        print("Analysis returned empty — skipping.")
        return

    write_analysis(gc, sheet_id, analysis)


if __name__ == "__main__":
    main()
