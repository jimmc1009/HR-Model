"""
diagnose_hr_tier_granular.py
Step 1: Hit rate by a full granular range of score tiers (0.5-point buckets)
to find where hit rate meaningfully separates from baseline.

Step 2: Once a threshold is chosen, recompute feature separators (hits vs
misses) restricted to ONLY that pool — not the entire scored universe.
This avoids the whole-pool separator problem where low-score non-hitters
trivially inflate every separator.

Run this, review the granular tier table, then tell me the threshold to
use for the restricted-pool separator analysis (or rerun with a different
THRESHOLD value below).
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

# Set this after reviewing Step 1 output, then rerun for Step 2 restricted analysis
THRESHOLD = 10.0


def get_gspread_client():
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

    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["hr_score"] = resolved["hr_score"].apply(safe_float)

    print(f"  {len(resolved)} resolved rows from {MODEL_START_DATE}\n")

    overall_rate = round(resolved["hit_bool"].mean() * 100, 1)

    # ── STEP 1: Granular hit rate by 0.5-point tier ────────────────────────
    print("="*70)
    print("STEP 1 — HIT RATE BY GRANULAR SCORE TIER (0.5-point buckets)")
    print(f"Overall pool hit rate: {overall_rate}%")
    print("="*70)

    min_score = resolved["hr_score"].min()
    max_score = resolved["hr_score"].max()
    bucket_lo = np.floor(min_score * 2) / 2

    print(f"\n  {'Tier':12s} {'N':>6} {'Hits':>6} {'Hit Rate':>10} {'vs Overall':>12}")
    print(f"  {'-'*50}")

    bucket = bucket_lo
    cumulative_rows = []
    while bucket < max_score + 0.5:
        sub = resolved[(resolved["hr_score"] >= bucket) & (resolved["hr_score"] < bucket + 0.5)]
        if len(sub) >= 5:
            n = len(sub)
            h = int(sub["hit_bool"].sum())
            rate = round(h / n * 100, 1)
            vs_overall = round(rate - overall_rate, 1)
            flag = " ✅" if vs_overall >= 5 else " ❌" if vs_overall <= -5 else ""
            print(f"  {bucket:>4.1f}-{bucket+0.5:<4.1f}  {n:>6} {h:>6} {rate:>9.1f}% {vs_overall:>+11.1f}%{flag}")
        bucket += 0.5

    # ── STEP 1b: Cumulative "X+" hit rate — find the real breakpoint ──────
    print("\n" + "="*70)
    print("STEP 1b — CUMULATIVE HIT RATE: score >= X")
    print("(This shows where 'everything above this score' starts beating baseline)")
    print("="*70)
    print(f"\n  {'Threshold':12s} {'N':>6} {'Hits':>6} {'Hit Rate':>10} {'vs Overall':>12}")
    print(f"  {'-'*50}")

    thresholds = np.arange(np.floor(min_score), np.ceil(max_score) + 0.5, 0.5)
    for t in thresholds:
        sub = resolved[resolved["hr_score"] >= t]
        if len(sub) < 10:
            continue
        n = len(sub)
        h = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        vs_overall = round(rate - overall_rate, 1)
        flag = " ✅" if vs_overall >= 5 else ""
        print(f"  >= {t:<9.1f} {n:>6} {h:>6} {rate:>9.1f}% {vs_overall:>+11.1f}%{flag}")

    # ── STEP 2: Restricted-pool feature separators ──────────────────────────
    print("\n" + "="*70)
    print(f"STEP 2 — FEATURE SEPARATORS RESTRICTED TO score >= {THRESHOLD}")
    print("(Hits vs misses computed ONLY within this pool, not the full universe)")
    print("="*70)

    restricted = resolved[resolved["hr_score"] >= THRESHOLD].copy()
    r_rate = round(restricted["hit_bool"].mean() * 100, 1) if not restricted.empty else 0
    print(f"\n  Restricted pool: {len(restricted)} picks, {r_rate}% hit rate")

    numeric_features = [
        "barrel_pct_7d", "season_barrel_pct", "barrel_pct_5d", "barrel_pct_10d",
        "avg_ev_7d", "avg_ev_30d", "avg_la_7d",
        "iso", "hr_per_pa", "hr_per_fb", "pull_rate",
        "pitcher_barrel_pct", "pitcher_hr_per_fb",
        "park_hr_factor", "hr_weather_boost",
        "pitch_matchup_score", "platoon_score", "momentum_score",
        "hard_hit_pct_7d", "hard_hit_pct_season",
    ]
    feature_labels = {
        "barrel_pct_7d":       "Barrel% (7d)",
        "season_barrel_pct":   "Barrel% (Season)",
        "barrel_pct_5d":       "Barrel% (5d)",
        "barrel_pct_10d":      "Barrel% (10d)",
        "avg_ev_7d":           "Avg EV (7d)",
        "avg_ev_30d":          "Avg EV (30d)",
        "avg_la_7d":           "Avg Launch Angle (7d)",
        "iso":                 "ISO",
        "hr_per_pa":           "HR/PA%",
        "hr_per_fb":           "HR/FB%",
        "pull_rate":           "Pull Rate%",
        "pitcher_barrel_pct":  "Pitcher Barrel% Allowed",
        "pitcher_hr_per_fb":   "Pitcher HR/FB%",
        "park_hr_factor":      "Park HR Factor",
        "hr_weather_boost":    "Weather Boost",
        "pitch_matchup_score": "Pitch Matchup Score",
        "platoon_score":       "Platoon Score",
        "momentum_score":      "Momentum Score",
        "hard_hit_pct_7d":     "Hard Hit% (7d)",
        "hard_hit_pct_season": "Hard Hit% (Season)",
    }

    for col in numeric_features:
        if col in restricted.columns:
            restricted[col] = restricted[col].apply(lambda x: np.nan if str(x).strip() in ("", "nan", "None") else safe_float(x, np.nan))
        else:
            restricted[col] = np.nan

    hits   = restricted[restricted["hit_bool"]]
    misses = restricted[~restricted["hit_bool"]]

    seps = []
    for col, label in feature_labels.items():
        if col not in restricted.columns:
            continue
        h_vals = hits[col].dropna()
        m_vals = misses[col].dropna()
        if len(h_vals) < 5 or len(m_vals) < 5:
            continue
        h_avg = round(h_vals.mean(), 3)
        m_avg = round(m_vals.mean(), 3)
        diff  = round(h_avg - m_avg, 3)
        pct   = round((diff / m_avg * 100), 1) if m_avg != 0 else 0.0
        seps.append((pct, label, h_avg, m_avg, len(h_vals), len(m_vals)))

    seps.sort(key=lambda x: abs(x[0]), reverse=True)

    print(f"\n  {'Feature':<24} {'Hits Avg':>10} {'Miss Avg':>10} {'% Diff':>9} {'Signal':>10}")
    print(f"  {'-'*68}")
    for pct, label, h_avg, m_avg, n_h, n_m in seps:
        flag = "🔥 STRONG+" if pct >= 15 else "✅ Pos" if pct >= 7 else "🔴 STRONG-" if pct <= -15 else "⚠️ Neg" if pct <= -7 else "↔️ Neutral"
        print(f"  {label:<24} {h_avg:>10.3f} {m_avg:>10.3f} {pct:>+8.1f}% {flag:>10s}")

    print(f"\n  (n_hits varies per feature based on data availability, typically ~{int(restricted['hit_bool'].sum())} hits, ~{len(restricted)-int(restricted['hit_bool'].sum())} misses in pool)")

    print("\nDone.")


if __name__ == "__main__":
    main()
