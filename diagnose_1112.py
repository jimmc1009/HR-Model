"""
diagnose_1112.py
Comprehensive feature analysis for the 11-12 score tier.
Goal: find what separates HR hitters from non-HR hitters within 11-12
so we can fix the scoring formula or add a targeted filter.

Runs:
1. Hit rate comparison across tiers (context)
2. Feature averages — hits vs misses in 11-12
3. Single-feature threshold tests — what floor fixes the tier?
4. Combined feature threshold tests — does combining two features help?
5. Comparison of 11-12 hitters vs 12-13 hitters (what makes a real 12?)
6. 11-12 hits at short odds vs plus odds profile
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

MODEL_START_DATE = "2026-06-09"
MIN_SAMPLE       = 5


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=15):
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


def hr(sub):
    if len(sub) == 0:
        return 0.0, 0, 0
    h = int(sub["hit_bool"].sum())
    return round(h / len(sub) * 100, 1), h, len(sub)


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    sh         = with_retry(lambda: gc.open_by_key(sheet_id))
    ws         = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers    = all_values[0]
    rows       = all_values[1:]
    df         = pd.DataFrame(rows, columns=headers)
    df         = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()
    print(f"  {len(df)} rows after {MODEL_START_DATE} filter")

    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    print(f"  {len(resolved)} resolved rows")

    numeric_cols = [
        "hr_score", "consensus_odds",
        "barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d", "season_barrel_pct",
        "hr_per_pa", "hr_per_fb", "iso",
        "pitcher_barrel_pct", "pitcher_hr_per_fb",
        "platoon_score", "pitch_matchup_score", "momentum_score",
        "park_hr_factor", "hr_weather_boost",
        "avg_ev_7d", "avg_la_7d",
    ]
    for col in numeric_cols:
        if col in resolved.columns:
            resolved[col] = resolved[col].apply(safe_float)
        else:
            resolved[col] = np.nan

    resolved["odds_num"] = resolved["consensus_odds"].apply(safe_float)

    t_1112 = resolved[(resolved["hr_score"] >= 11) & (resolved["hr_score"] < 12)].copy()
    t_1011 = resolved[(resolved["hr_score"] >= 10) & (resolved["hr_score"] < 11)].copy()
    t_1213 = resolved[(resolved["hr_score"] >= 12) & (resolved["hr_score"] < 13)].copy()
    t_13p  = resolved[resolved["hr_score"] >= 13].copy()

    feature_cols = [
        ("season_barrel_pct",  "Season Barrel%"),
        ("barrel_pct_7d",      "Barrel% 7d"),
        ("barrel_pct_5d",      "Barrel% 5d"),
        ("barrel_pct_10d",     "Barrel% 10d"),
        ("hr_per_fb",          "HR/FB%"),
        ("hr_per_pa",          "HR/PA%"),
        ("iso",                "ISO"),
        ("platoon_score",      "Platoon Score"),
        ("pitch_matchup_score","Pitch Matchup"),
        ("pitcher_barrel_pct", "Pitcher Barrel%"),
        ("pitcher_hr_per_fb",  "Pitcher HR/FB%"),
        ("park_hr_factor",     "Park Factor"),
        ("hr_weather_boost",   "Weather Boost"),
        ("avg_ev_7d",          "Avg EV 7d"),
        ("avg_la_7d",          "Avg LA 7d"),
        ("momentum_score",     "Momentum"),
    ]

    # ── 1. Hit rate context ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("1. HIT RATE BY TIER")
    print("="*70)
    for label, sub in [("10-11", t_1011), ("11-12", t_1112), ("12-13", t_1213), ("13+", t_13p)]:
        r, h, n = hr(sub)
        print(f"  {label:8s}: {h:3d}/{n:3d} = {r}%")

    print("\n  11-12 by odds zone:")
    for label, lo, hi in [("<=+300", 0, 301), ("+301-499", 301, 500), ("+500+", 500, 9999)]:
        sub = t_1112[(t_1112["odds_num"] >= lo) & (t_1112["odds_num"] < hi)]
        if len(sub) < 3:
            continue
        r, h, n = hr(sub)
        print(f"    {label:12s}: {h}/{n} = {r}%")

    # ── 2. Feature averages ───────────────────────────────────────────────
    print("\n" + "="*70)
    print("2. FEATURE AVERAGES IN 11-12 — Hits vs Misses")
    print("="*70)
    hits_1112   = t_1112[t_1112["hit_bool"]]
    misses_1112 = t_1112[~t_1112["hit_bool"]]

    separators = []
    print(f"\n  {'Feature':<25} {'Hits':>8} {'Misses':>8} {'Diff':>8} {'%Diff':>7}")
    print(f"  {'-'*60}")
    for col, label in feature_cols:
        if col not in t_1112.columns:
            continue
        h_avg = hits_1112[col].dropna().mean()
        m_avg = misses_1112[col].dropna().mean()
        if pd.isna(h_avg) or pd.isna(m_avg) or m_avg == 0:
            continue
        diff     = h_avg - m_avg
        pct_diff = diff / abs(m_avg) * 100
        flag     = " ✅" if pct_diff >= 10 else " ❌" if pct_diff <= -10 else ""
        print(f"  {label:<25} {h_avg:>8.3f} {m_avg:>8.3f} {diff:>+8.3f} {pct_diff:>+6.1f}%{flag}")
        separators.append((col, label, pct_diff, h_avg, m_avg))

    separators.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  Top separators:")
    for col, label, pct, h, m in separators[:6]:
        direction = "HIGHER in hits" if pct > 0 else "LOWER in hits"
        print(f"    {label:<25} {pct:+.1f}% ({direction})")

    # ── 3. Single feature threshold tests ────────────────────────────────
    print("\n" + "="*70)
    print("3. SINGLE FEATURE THRESHOLD TESTS")
    print("="*70)

    threshold_tests = [
        ("season_barrel_pct",  "Season Barrel%",  [8, 9, 10, 11, 12, 13, 14], ">="),
        ("barrel_pct_7d",      "Barrel% 7d",      [8, 10, 12, 15, 18],        ">="),
        ("barrel_pct_5d",      "Barrel% 5d",      [8, 10, 12, 15],            ">="),
        ("hr_per_fb",          "HR/FB%",           [15, 18, 20, 22, 25],       ">="),
        ("hr_per_pa",          "HR/PA%",           [3, 4, 5, 6],               ">="),
        ("iso",                "ISO",              [0.15, 0.18, 0.20, 0.23],   ">="),
        ("platoon_score",      "Platoon <= (cap)", [0.5, 1.0, 1.5, 2.0],      "<="),
        ("platoon_score",      "Platoon >= (min)", [0.5, 1.0, 1.5],           ">="),
        ("pitch_matchup_score","Pitch Matchup",    [2.0, 2.5, 3.0, 3.5],      ">="),
        ("avg_ev_7d",          "Avg EV 7d",        [85, 88, 90, 92],           ">="),
        ("avg_la_7d",          "Avg LA 7d",        [10, 12, 14, 16],           ">="),
    ]

    baseline_r, baseline_h, baseline_n = hr(t_1112)
    best_single = []

    for col, label, thresholds, direction in threshold_tests:
        if col not in t_1112.columns:
            continue
        print(f"\n  {label} ({direction} threshold) — baseline {baseline_h}/{baseline_n} = {baseline_r}%:")
        for thresh in thresholds:
            if direction == ">=":
                sub = t_1112[t_1112[col] >= thresh]
            else:
                sub = t_1112[t_1112[col] <= thresh]
            if len(sub) < MIN_SAMPLE:
                continue
            r, h, n = hr(sub)
            improvement = round(r - baseline_r, 1)
            flag = " ✅ PROMISING" if r >= 20 and n >= 10 else " 🔥 STRONG" if r >= 25 and n >= 8 else ""
            print(f"    {direction} {thresh:>6}: {h:3d}/{n:3d} = {r}%  ({improvement:+.1f}%){flag}")
            if r >= 20 and n >= MIN_SAMPLE:
                best_single.append((r, n, col, direction, thresh, label))

    # ── 4. Combined threshold tests ───────────────────────────────────────
    print("\n" + "="*70)
    print("4. COMBINED THRESHOLD TESTS")
    print("="*70)
    print(f"  Baseline: {baseline_h}/{baseline_n} = {baseline_r}%\n")

    combos = [
        ("season_barrel_pct", ">=", 10, "hr_per_fb",    ">=", 18),
        ("season_barrel_pct", ">=", 11, "hr_per_fb",    ">=", 20),
        ("season_barrel_pct", ">=", 12, "hr_per_fb",    ">=", 18),
        ("season_barrel_pct", ">=", 10, "hr_per_fb",    ">=", 22),
        ("season_barrel_pct", ">=", 10, "iso",          ">=", 0.18),
        ("season_barrel_pct", ">=", 12, "iso",          ">=", 0.20),
        ("hr_per_fb",         ">=", 18, "iso",          ">=", 0.18),
        ("hr_per_fb",         ">=", 20, "platoon_score","<=", 1.0),
        ("hr_per_fb",         ">=", 18, "platoon_score","<=", 1.5),
        ("season_barrel_pct", ">=", 10, "platoon_score","<=", 1.0),
        ("season_barrel_pct", ">=", 10, "platoon_score","<=", 1.5),
        ("season_barrel_pct", ">=", 10, "avg_ev_7d",    ">=", 88),
        ("hr_per_fb",         ">=", 18, "avg_ev_7d",    ">=", 88),
        ("barrel_pct_7d",     ">=", 12, "hr_per_fb",    ">=", 18),
        ("barrel_pct_7d",     ">=", 12, "season_barrel_pct",">=", 10),
        ("barrel_pct_7d",     ">=", 15, "hr_per_fb",    ">=", 18),
        ("iso",               ">=", 0.18,"platoon_score","<=", 1.0),
        ("hr_per_fb",         ">=", 22, "season_barrel_pct",">=", 10),
        ("avg_la_7d",         ">=", 12, "hr_per_fb",    ">=", 18),
        ("avg_la_7d",         ">=", 12, "season_barrel_pct",">=", 10),
    ]

    combo_results = []
    for c1, d1, t1, c2, d2, t2 in combos:
        if c1 not in t_1112.columns or c2 not in t_1112.columns:
            continue
        m1 = t_1112[c1] >= t1 if d1 == ">=" else t_1112[c1] <= t1
        m2 = t_1112[c2] >= t2 if d2 == ">=" else t_1112[c2] <= t2
        sub = t_1112[m1 & m2]
        if len(sub) < MIN_SAMPLE:
            continue
        r, h, n = hr(sub)
        combo_results.append((r, h, n, c1, d1, t1, c2, d2, t2))

    combo_results.sort(key=lambda x: x[0], reverse=True)
    for r, h, n, c1, d1, t1, c2, d2, t2 in combo_results:
        flag = " 🔥 STRONG" if r >= 25 and n >= 8 else " ✅ PROMISING" if r >= 20 and n >= MIN_SAMPLE else ""
        print(f"  {c1} {d1} {t1} AND {c2} {d2} {t2}: {h}/{n} = {r}%{flag}")

    # ── 5. 11-12 hitters vs 12-13 hitters ────────────────────────────────
    print("\n" + "="*70)
    print("5. PROFILE — 11-12 HR Hitters vs 12-13 HR Hitters")
    print("="*70)
    hits_1213 = t_1213[t_1213["hit_bool"]]
    print(f"\n  {'Feature':<25} {'11-12 Hits':>11} {'12-13 Hits':>11} {'Diff':>8}")
    print(f"  {'-'*58}")
    for col, label in feature_cols:
        if col not in resolved.columns:
            continue
        a = hits_1112[col].dropna().mean()
        b = hits_1213[col].dropna().mean()
        if pd.isna(a) or pd.isna(b):
            continue
        diff = a - b
        flag = " ← weaker" if diff < -1 else " ← stronger" if diff > 1 else ""
        print(f"  {label:<25} {a:>11.3f} {b:>11.3f} {diff:>+8.3f}{flag}")

    # ── 6. Short odds hits vs plus odds hits ──────────────────────────────
    print("\n" + "="*70)
    print("6. 11-12 HITS — Short Odds (≤+300) vs Plus Odds (+301+) Profile")
    print("="*70)
    hits_short = t_1112[t_1112["hit_bool"] & (t_1112["odds_num"] > 0) & (t_1112["odds_num"] <= 300)]
    hits_plus  = t_1112[t_1112["hit_bool"] & (t_1112["odds_num"] > 300)]
    print(f"\n  Short odds hits: {len(hits_short)} | Plus odds hits: {len(hits_plus)}")

    if len(hits_short) >= 3 and len(hits_plus) >= 3:
        print(f"\n  {'Feature':<25} {'≤+300':>9} {'+301+':>9} {'Diff':>8}")
        print(f"  {'-'*55}")
        for col, label in feature_cols:
            if col not in t_1112.columns:
                continue
            s = hits_short[col].dropna().mean()
            p = hits_plus[col].dropna().mean()
            if pd.isna(s) or pd.isna(p):
                continue
            diff = s - p
            flag = " ←" if abs(diff) > 1 else ""
            print(f"  {label:<25} {s:>9.3f} {p:>9.3f} {diff:>+8.3f}{flag}")

    print("\nDone.")


if __name__ == "__main__":
    main()
