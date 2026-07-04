"""
hr_v2_analysis.py
Tracks the SHADOW v2 model's performance by reading HR_All_Scores_v2 and
writing a HR_v2_Analysis tab. Purpose: watch whether real value zones emerge
on the v2 score scale, so you know if/when to switch from v1.

v2 scores compress into roughly 8-17, so tiers here are FINER than v1's.
Shows: hit rate by v2 score tier, by odds zone, and the score×odds cross-tab
(the key view for finding bettable zones). Also flags TRUE breakeven per zone.

Read-only on the model — pure tracking. Writes only to HR_v2_Analysis.
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

MODEL_START_DATE = "2026-07-01"   # v2 shadow began 7/1

COLOR_BG        = {"red": 0.114, "green": 0.114, "blue": 0.114}
COLOR_BG_ALT    = {"red": 0.149, "green": 0.149, "blue": 0.149}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_GREEN_DIM = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_RED       = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_RED_DIM   = {"red": 0.200, "green": 0.039, "blue": 0.039}
COLOR_GREY      = {"red": 0.600, "green": 0.600, "blue": 0.600}
COLOR_HEADER_BG = {"red": 0.078, "green": 0.078, "blue": 0.078}
COLOR_PURPLE    = {"red": 0.576, "green": 0.439, "blue": 0.859}
COLOR_PURPLE_DIM = {"red": 0.118, "green": 0.078, "blue": 0.196}


def get_gspread_client():
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=20):
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
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def read_sheet(gc, sheet_id, name):
    try:
        sh = with_retry(lambda: gc.open_by_key(sheet_id))
        ws = sh.worksheet(name)
        vals = with_retry(lambda: ws.get_all_values())
        if not vals or len(vals) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(vals[1:], columns=vals[0])
        return df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    except gspread.WorksheetNotFound:
        return pd.DataFrame()


def breakeven_for_odds(odds):
    """American odds -> breakeven win prob %."""
    if odds >= 100:
        return round(100 / (odds + 100) * 100, 1)
    elif odds < 0:
        return round(abs(odds) / (abs(odds) + 100) * 100, 1)
    return 100.0


def odds_zone(odds):
    if odds <= 300:
        return "≤+300"
    elif odds <= 499:
        return "+301-499"
    elif odds <= 699:
        return "+500-699"
    return "+700+"


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores_v2...")
    df = read_sheet(gc, sheet_id, "HR_All_Scores_v2")
    if df.empty:
        print("HR_All_Scores_v2 empty or missing — nothing to analyze yet.")
        return

    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    if resolved.empty:
        print("No resolved v2 rows yet — check back after games resolve.")
        return

    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["score"]    = resolved["hr_score"].apply(safe_float)
    resolved["odds"]     = resolved["consensus_odds"].apply(lambda x: safe_float(str(x).replace("+", ""), 0))

    n = len(resolved)
    overall = round(resolved["hit_bool"].mean() * 100, 1)
    print(f"  {n} resolved v2 rows, overall hit rate {overall}%")

    # Confidence flag on sample size
    if n < 100:
        conf = f"⏳ TOO EARLY (n={n}) — need ~300+ for reliable zones"
    elif n < 300:
        conf = f"🌱 BUILDING (n={n}) — directional only"
    elif n < 600:
        conf = f"📊 DEVELOPING (n={n}) — zones starting to mean something"
    else:
        conf = f"✅ SOLID (n={n}) — zones are trustworthy"

    # ── Fine v2 score tiers ──────────────────────────────────────────────
    tiers = [
        ("16+",      16, 99),
        ("15-16",    15, 16),
        ("14-15",    14, 15),
        ("13-14",    13, 14),
        ("12-13",    12, 13),
        ("11-12",    11, 12),
        ("10-11",    10, 11),
        ("9-10",      9, 10),
        ("8-9",       8,  9),
        ("Under 8",  -99, 8),
    ]

    rows = []
    rows.append(["📊  v2 SHADOW MODEL — PERFORMANCE TRACKING", "", "", "", ""])
    rows.append([conf, "", "", "", ""])
    rows.append([f"Overall: {overall}%  ({int(resolved['hit_bool'].sum())}/{n})", "", "", "", ""])
    rows.append(["", "", "", "", ""])

    rows.append(["🎯  BY v2 SCORE TIER", "Total", "Hit HR", "Hit Rate %", "Avg Odds"])
    for label, lo, hi in tiers:
        sub = resolved[(resolved["score"] >= lo) & (resolved["score"] < hi)]
        if len(sub) < 3:
            continue
        h = int(sub["hit_bool"].sum())
        rate = round(h / len(sub) * 100, 1)
        avg_odds = int(sub["odds"].mean()) if len(sub) else 0
        rows.append([label, str(len(sub)), str(h), f"{rate}%", f"+{avg_odds}"])

    rows.append(["", "", "", "", ""])
    rows.append(["💰  BY ODDS ZONE", "Total", "Hit HR", "Hit Rate %", "Breakeven %"])
    for zone in ["≤+300", "+301-499", "+500-699", "+700+"]:
        sub = resolved[resolved["odds"].apply(odds_zone) == zone]
        if len(sub) < 3:
            continue
        h = int(sub["hit_bool"].sum())
        rate = round(h / len(sub) * 100, 1)
        be = breakeven_for_odds(sub["odds"].mean())
        rows.append([zone, str(len(sub)), str(h), f"{rate}%", f"{be}%"])

    rows.append(["", "", "", "", ""])
    rows.append(["🔑  v2 SCORE TIER × ODDS ZONE (bettable-zone finder)", "Total", "Hit HR", "Hit Rate %", "vs Breakeven"])
    for label, lo, hi in tiers:
        tsub = resolved[(resolved["score"] >= lo) & (resolved["score"] < hi)]
        if len(tsub) < 3:
            continue
        for zone in ["≤+300", "+301-499", "+500-699", "+700+"]:
            zsub = tsub[tsub["odds"].apply(odds_zone) == zone]
            if len(zsub) < 3:
                continue
            h = int(zsub["hit_bool"].sum())
            rate = round(h / len(zsub) * 100, 1)
            be = breakeven_for_odds(zsub["odds"].mean())
            edge = round(rate - be, 1)
            flag = "✅" if edge >= 3 else "❌" if edge <= -3 else "≈"
            rows.append([f"{label} | {zone}", str(len(zsub)), str(h), f"{rate}%", f"{edge:+.1f}% {flag}"])

    # ── Feature separators (hit vs miss) — includes v2-specific features ──
    rows.append(["", "", "", "", ""])
    rows.append(["📊  v2 FEATURE SEPARATORS (hit vs miss)", "HR Avg", "Non-HR Avg", "Diff %", "Signal"])

    # vuln lives inside the v2_breakdown JSON blob, not its own column — extract it
    def _extract_vuln(blob):
        try:
            import json as _json
            d = _json.loads(blob) if blob and str(blob).strip() not in ("", "nan", "None") else {}
            return safe_float(d.get("vuln"), np.nan)
        except Exception:
            return np.nan
    if "v2_breakdown" in resolved.columns:
        resolved["vuln_score"] = resolved["v2_breakdown"].apply(_extract_vuln)

    # v2-distinctive features first (power_norm, vuln), then shared ones
    feat_defs = [
        ("power_norm",           "Power Norm (v2)"),
        ("vuln_score",           "Pitcher Vuln (v2)"),
        ("pitch_matchup_score",  "Pitch Matchup (hand-split)"),
        ("platoon_score",        "Platoon Score"),
        ("season_barrel_pct",    "Season Barrel%"),
        ("hr_per_fb",            "HR/FB%"),
        ("iso",                  "ISO"),
        ("barrel_pct_7d",        "Barrel% 7d"),
    ]
    for col, disp in feat_defs:
        if col not in resolved.columns:
            continue
        v = resolved[col].apply(lambda x: safe_float(x, np.nan))
        h_avg = v[resolved["hit_bool"]].mean()
        m_avg = v[~resolved["hit_bool"]].mean()
        if pd.isna(h_avg) or pd.isna(m_avg) or m_avg == 0:
            continue
        diff_pct = round((h_avg - m_avg) / abs(m_avg) * 100, 1)
        if diff_pct >= 15:
            sig = "🔥 STRONG +"
        elif diff_pct >= 3:
            sig = "✅ Positive"
        elif diff_pct <= -3:
            sig = "🔻 Negative"
        else:
            sig = "↔️ Neutral"
        rows.append([disp, f"{h_avg:.3f}", f"{m_avg:.3f}", f"{diff_pct:+.1f}%", sig])

    # Write
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("HR_v2_Analysis")
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HR_v2_Analysis", rows=200, cols=6)

    with_retry(lambda: ws.update(rows))
    print(f"HR_v2_Analysis written ({len(rows)} rows).")
    print("Done.")


if __name__ == "__main__":
    main()
