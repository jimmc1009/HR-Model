"""
hr_analysis.py
Reads HR_All_Scores daily after resolution and writes feature analysis
to HR_Analysis sheet. Runs automatically at end of pipeline.
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

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Model rebuild date — only analyze data from this date forward
# Updated from 2026-06-02 to 2026-06-09 to exclude early dates where
# avg_ev_7d was missing for some players due to Statcast API gaps,
# which was corrupting the 11-12 tier analysis.
MODEL_START_DATE = "2026-06-09"

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
COLOR_PURPLE    = {"red": 0.576, "green": 0.439, "blue": 0.859}
COLOR_PURPLE_DIM= {"red": 0.118, "green": 0.078, "blue": 0.196}
COLOR_TEAL      = {"red": 0.047, "green": 0.450, "blue": 0.353}
COLOR_TEAL_DIM  = {"red": 0.020, "green": 0.120, "blue": 0.090}
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


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def parse_wind(wind_str: str) -> str:
    w = str(wind_str).strip().lower()
    if "out" in w:
        return "Blowing OUT"
    elif "in" in w:
        return "Blowing IN"
    elif "roof" in w or "neutral" in w:
        return "Roof / Neutral"
    elif "calm" in w or "cross" in w:
        return "Calm / Crosswind"
    elif w in ("", "nan", "unknown"):
        return "Unknown"
    return "Calm / Crosswind"


def build_analysis(df: pd.DataFrame) -> dict:
    # ── Filter to current model only ──────────────────────────────────────
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()
    print(f"  Filtered to model start date {MODEL_START_DATE}: {len(df)} rows remaining")

    scored = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    if scored.empty:
        return {}

    scored["hit_bool"]    = scored["hit_hr"] == "Yes"
    scored["hr_score"]    = scored["hr_score"].apply(safe_float)
    scored["date_dt"]     = pd.to_datetime(scored["date"], errors="coerce")
    scored["odds_num"]    = scored["consensus_odds"].apply(lambda x: safe_float(x, 0))
    scored["wind_bucket"] = scored.get("wind_context", pd.Series("", index=scored.index)).apply(parse_wind)

    numeric_features = [
        "barrel_pct_7d", "season_barrel_pct", "barrel_pct_5d", "barrel_pct_10d",
        "avg_ev_7d", "avg_ev_5d", "avg_ev_10d",
        "avg_la_7d", "avg_la_season",
        "iso", "hr_per_pa", "hr_per_fb", "pull_rate",
        "pitcher_barrel_pct", "pitcher_hr_per_fb",
        "pitcher_barrel_vs_lhh", "pitcher_barrel_vs_rhh",
        "park_hr_factor", "hr_weather_boost", "temp_f", "pitch_matchup_score",
        "platoon_score",
    ]
    for col in numeric_features:
        if col in scored.columns:
            # Use NaN for empty/missing so mean() ignores them correctly
            scored[col] = scored[col].apply(lambda x: np.nan if str(x).strip() in ("", "nan", "None") else safe_float(x, np.nan))
        else:
            scored[col] = np.nan

    days_of_data = scored["date_dt"].nunique()
    total        = len(scored)
    hits         = int(scored["hit_bool"].sum())
    hit_rate     = round(hits / total * 100, 1) if total > 0 else 0.0

    # ── Score tiers ───────────────────────────────────────────────────────
    score_tiers = []
    for label, lo, hi in [
        ("13+",   13, 999),
        ("12-13", 12,  13),
        ("11-12", 11,  12),
        ("10-11", 10,  11),
        ("9-10",   9,  10),
        ("8.5-9",  8.5, 9),
        ("Under 8.5", 0, 8.5),
    ]:
        sub = scored[(scored["hr_score"] >= lo) & (scored["hr_score"] < hi)]
        if sub.empty:
            continue
        n    = len(sub)
        h    = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        avg_odds = round(sub[sub["odds_num"] > 0]["odds_num"].mean(), 0) if (sub["odds_num"] > 0).any() else 0
        score_tiers.append({
            "label": label, "total": n, "hits": h,
            "rate": rate, "avg_odds": f"+{int(avg_odds)}" if avg_odds > 0 else "—"
        })

    # ── Odds zones ────────────────────────────────────────────────────────
    odds_zones = []
    for label, lo, hi in [
        ("≤ +300",       0,   301),
        ("+301 to +499", 301, 500),
        ("+500 to +699", 500, 700),
        ("+700+",        700, 9999),
        ("No Odds",      -1,  0),
    ]:
        if label == "No Odds":
            sub = scored[scored["odds_num"] <= 0]
        else:
            sub = scored[(scored["odds_num"] >= lo) & (scored["odds_num"] < hi)]
        if sub.empty:
            continue
        n    = len(sub)
        h    = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        odds_zones.append({"label": label, "total": n, "hits": h, "rate": rate})

    # ── Feature separators ────────────────────────────────────────────────
    hr_yes  = scored[scored["hit_bool"]]
    hr_no   = scored[~scored["hit_bool"]]
    feature_separators = []

    feature_labels = {
        "barrel_pct_7d":          "Barrel% (7d)",
        "season_barrel_pct":      "Barrel% (Season)",
        "avg_ev_7d":              "Avg EV (7d)",
        "avg_la_7d":              "Avg Launch Angle (7d)",
        "iso":                    "ISO",
        "hr_per_pa":              "HR/PA%",
        "hr_per_fb":              "HR/FB%",
        "pull_rate":              "Pull Rate%",
        "pitcher_barrel_pct":     "Pitcher Barrel% Allowed",
        "pitcher_hr_per_fb":      "Pitcher HR/FB%",
        "park_hr_factor":         "Park HR Factor",
        "hr_weather_boost":       "Weather Boost",
        "temp_f":                 "Temperature (°F)",
        "barrel_pct_5d":          "Barrel% (5d)",
        "barrel_pct_10d":         "Barrel% (10d)",
        "avg_ev_5d":              "Avg EV (5d)",
        "avg_ev_10d":             "Avg EV (10d)",
        "pitch_matchup_score":    "Pitch Matchup Score",
        "platoon_score":          "Platoon Score",
    }

    for col, label in feature_labels.items():
        if col not in scored.columns:
            continue
        yes_vals = hr_yes[col].dropna()
        no_vals  = hr_no[col].dropna()
        if len(yes_vals) < 5 or len(no_vals) < 5:
            continue
        yes_avg  = round(yes_vals.mean(), 3)
        no_avg   = round(no_vals.mean(), 3)
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

    # ── Wind analysis ─────────────────────────────────────────────────────
    wind_rows = []
    for bucket in ["Blowing OUT", "Calm / Crosswind", "Blowing IN", "Roof / Neutral", "Unknown"]:
        sub = scored[scored["wind_bucket"] == bucket]
        if sub.empty:
            continue
        n    = len(sub)
        h    = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        wind_rows.append({"label": bucket, "total": n, "hits": h, "rate": rate})

    # ── Rolling trends ────────────────────────────────────────────────────
    max_date  = scored["date_dt"].max()
    roll_rows = []
    for label, days in [("Last 7 Days", 7), ("Last 14 Days", 14), ("Last 30 Days", 30), ("All Time", 9999)]:
        cutoff = max_date - pd.Timedelta(days=days) if days < 9999 else pd.Timestamp(MODEL_START_DATE)
        sub    = scored[scored["date_dt"] >= cutoff]
        if sub.empty:
            continue
        n    = len(sub)
        h    = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        roll_rows.append({"label": label, "total": n, "hits": h, "rate": rate})

    # ── Platoon analysis ──────────────────────────────────────────────────
    platoon_rows = []
    if "platoon_matchup" in scored.columns:
        scored["has_platoon_adv"]     = scored["platoon_matchup"].str.contains("advantage", case=False, na=False)
        scored["has_platoon_dis"]     = scored["platoon_matchup"].str.contains("disadvantage|weakness", case=False, na=False)
        scored["has_platoon_neutral"] = ~scored["has_platoon_adv"] & ~scored["has_platoon_dis"]

        for label, mask in [
            ("Platoon Advantage",             scored["has_platoon_adv"]),
            ("Platoon Neutral",               scored["has_platoon_neutral"]),
            ("Platoon Disadvantage / Weakness", scored["has_platoon_dis"]),
        ]:
            sub = scored[mask]
            if sub.empty:
                continue
            n    = len(sub)
            h    = int(sub["hit_bool"].sum())
            rate = round(h / n * 100, 1)
            platoon_rows.append({"label": label, "total": n, "hits": h, "rate": rate})

    # ── Score Tier × Odds Zone cross-tab ─────────────────────────────────
    tier_odds_rows = []
    tier_defs = [
        ("13+",    13,   999),
        ("12-13",  12,    13),
        ("11-12",  11,    12),
        ("10-11",  10,    11),
        ("9-10",    9,    10),
        ("8.5-9",   8.5,   9),
    ]
    odds_defs = [
        ("≤ +300",       0,   301),
        ("+301 to +499", 301, 500),
        ("+500 to +699", 500, 700),
        ("+700+",        700, 9999),
    ]
    if "odds_num" in scored.columns:
        for tier_label, t_lo, t_hi in tier_defs:
            tier_sub = scored[(scored["hr_score"] >= t_lo) & (scored["hr_score"] < t_hi)]
            if tier_sub.empty:
                continue
            for odds_label, o_lo, o_hi in odds_defs:
                sub = tier_sub[(tier_sub["odds_num"] >= o_lo) & (tier_sub["odds_num"] < o_hi)]
                if len(sub) < 3:
                    continue
                n    = len(sub)
                h    = int(sub["hit_bool"].sum())
                rate = round(h / n * 100, 1)
                tier_odds_rows.append({
                    "label": f"{tier_label} | {odds_label}",
                    "total": n, "hits": h, "rate": rate,
                })

    return {
        "days_of_data":       days_of_data,
        "total":              total,
        "hits":               hits,
        "hit_rate":           hit_rate,
        "score_tiers":        score_tiers,
        "odds_zones":         odds_zones,
        "feature_separators": feature_separators,
        "wind_rows":          wind_rows,
        "roll_rows":          roll_rows,
        "platoon_rows":       platoon_rows,
        "tier_odds_rows":     tier_odds_rows,
    }


def write_analysis(gc: gspread.Client, sheet_id: str, analysis: dict) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("HR_Analysis")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HR_Analysis", rows=300, cols=8)

    ws_id = ws.id
    today = date.today().strftime("%B %d, %Y")

    all_values = []
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
        f"📊  HR MODEL ANALYSIS — Up to date (Model v2 from {MODEL_START_DATE})",
        "", "", "", "", "", "", ""
    ])
    all_values.append([
        f"Total Scored: {analysis['total']:,}",
        f"Hit HRs: {analysis['hits']:,}",
        f"Overall Hit Rate: {analysis['hit_rate']}%",
        f"Days of Data: {analysis['days_of_data']}",
        "", "", "", ""
    ])
    all_values.append([""] * 8)

    # ── Score Tiers ───────────────────────────────────────────────────────
    add_section(
        "🎯  BY SCORE TIER",
        ["Score Tier", "Total Players", "Hit HR", "Hit Rate %", "Avg Odds", "", "", ""],
        analysis["score_tiers"],
        lambda r: [r["label"], r["total"], r["hits"], f"{r['rate']}%", r["avg_odds"], "", "", ""]
    )

    # ── Odds Zones ────────────────────────────────────────────────────────
    add_section(
        "💰  BY ODDS ZONE",
        ["Odds Zone", "Total Players", "Hit HR", "Hit Rate %", "", "", "", ""],
        analysis["odds_zones"],
        lambda r: [r["label"], r["total"], r["hits"], f"{r['rate']}%", "", "", "", ""]
    )

    # ── Wind Analysis ─────────────────────────────────────────────────────
    add_section(
        "🌬️  BY WIND CONDITION",
        ["Wind", "Total Players", "Hit HR", "Hit Rate %", "", "", "", ""],
        analysis["wind_rows"],
        lambda r: [r["label"], r["total"], r["hits"], f"{r['rate']}%", "", "", "", ""]
    )

    # ── Platoon Analysis ──────────────────────────────────────────────────
    add_section(
        "🔄  BY PLATOON MATCHUP",
        ["Platoon", "Total Players", "Hit HR", "Hit Rate %", "", "", "", ""],
        analysis["platoon_rows"],
        lambda r: [r["label"], r["total"], r["hits"], f"{r['rate']}%", "", "", "", ""]
    )

    # ── Score Tier × Odds Zone ────────────────────────────────────────────
    if analysis.get("tier_odds_rows"):
        add_section(
            "💰  SCORE TIER × ODDS ZONE",
            ["Score Tier | Odds Zone", "Total Players", "Hit HR", "Hit Rate %", "", "", "", ""],
            analysis["tier_odds_rows"],
            lambda r: [r["label"], r["total"], r["hits"], f"{r['rate']}%", "", "", "", ""]
        )

    # ── Rolling Trends ────────────────────────────────────────────────────
    add_section(
        "📈  ROLLING TRENDS",
        ["Period", "Total Players", "Hit HR", "Hit Rate %", "", "", "", ""],
        analysis["roll_rows"],
        lambda r: [r["label"], r["total"], r["hits"], f"{r['rate']}%", "", "", "", ""]
    )

    # ── Feature Separators ────────────────────────────────────────────────
    section_starts["features"] = len(all_values)
    all_values.append(["🔬  FEATURE SEPARATORS — HR Hitters vs Non-HR Hitters", "", "", "", "", "", "", ""])
    all_values.append(["Feature", "HR Hitters Avg", "Non-HR Avg", "Difference", "% Difference", "Signal", "", ""])
    for r in analysis["feature_separators"]:
        signal = ""
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
            r["label"],
            r["yes_avg"],
            r["no_avg"],
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
            "backgroundColor": COLOR_BLUE_DIM,
            "textFormat": {"foregroundColor": COLOR_BLUE, "bold": True,
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
        "🎯  BY SCORE TIER":        (COLOR_PURPLE,    COLOR_PURPLE_DIM),
        "💰  BY ODDS ZONE":         (COLOR_GOLD,      COLOR_GOLD_DIM),
        "🌬️  BY WIND CONDITION":    (COLOR_TEAL,      COLOR_TEAL_DIM),
        "🔄  BY PLATOON MATCHUP":   (COLOR_BLUE,      COLOR_BLUE_DIM),
        "💰  SCORE TIER × ODDS ZONE": (COLOR_GOLD,      COLOR_GOLD_DIM),
        "📈  ROLLING TRENDS":       (COLOR_GREEN,     COLOR_GREEN_DIM),
        "features":                  (COLOR_PURPLE,    COLOR_PURPLE_DIM),
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
                if rate_val >= 20:
                    rate_bg = COLOR_GREEN_DIM
                    rate_fg = COLOR_GREEN
                elif rate_val >= 15:
                    rate_bg = COLOR_TEAL_DIM
                    rate_fg = COLOR_TEAL
                elif rate_val <= 10:
                    rate_bg = COLOR_RED_DIM
                    rate_fg = COLOR_RED
                else:
                    continue
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

    feature_start = section_starts.get("features", 0) + 2
    for i, r in enumerate(analysis["feature_separators"]):
        row_idx = feature_start + i
        if r["pct_diff"] >= 15:
            fg = COLOR_GREEN; bg = COLOR_GREEN_DIM
        elif r["pct_diff"] >= 8:
            fg = COLOR_TEAL; bg = COLOR_TEAL_DIM
        elif r["pct_diff"] <= -15:
            fg = COLOR_RED; bg = COLOR_RED_DIM
        elif r["pct_diff"] <= -8:
            fg = COLOR_GOLD; bg = COLOR_GOLD_DIM
        else:
            fg = COLOR_GREY; bg = COLOR_BG
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
        is_section    = any(str(row[0]).startswith(e) for e in ["🎯", "💰", "🌬", "🔄", "📈", "🔬"])
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

    col_widths = [240, 120, 90, 110, 120, 140, 80, 80]
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
            "tabColorStyle": {"rgbColor": COLOR_BLUE},
        },
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("HR_Analysis formatted successfully.")
    except APIError as e:
        print(f"HR_Analysis formatting failed: {e}")

    print("HR_Analysis updated.")


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    df = read_sheet(gc, sheet_id, "HR_All_Scores")

    if df.empty:
        print("HR_All_Scores is empty — skipping analysis.")
        return

    scored_count = (df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])).sum()
    print(f"  {len(df)} total rows, {scored_count} resolved")

    if scored_count < 10:
        print("Not enough resolved data yet — skipping analysis.")
        return

    print("Building analysis...")
    analysis = build_analysis(df)

    if not analysis:
        print("Analysis returned empty — skipping.")
        return

    write_analysis(gc, sheet_id, analysis)


if __name__ == "__main__":
    main()
