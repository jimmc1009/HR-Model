"""
diagnose_pitcher_stage2.py
Stage-two ranker investigation: within the score >= 10 pool, do PITCHER-SIDE
features separate hits from misses? Stage one is batter-centric and spent;
this tests whether the matchup (pitcher vulnerability) is the missing signal.

Key idea: use HANDEDNESS-MATCHED pitcher barrel% (vs LHH for lefty batters,
vs RHH for righty batters) rather than the blended pitcher_barrel_pct.

Validation: split by date — first half finds the signal, second half confirms
it holds out of sample. A signal that only works in-sample is noise.
"""

import os
import json
import time
import re

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
POOL_THRESHOLD   = 10.0


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


def safe_float(val, default=np.nan):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def derive_batter_hand(row):
    """Prefer the logged batter_hand column; fall back to parsing platoon_matchup."""
    bh = str(row.get("batter_hand", "")).strip().upper()
    if bh in ("L", "R", "S"):
        return bh
    pm = str(row.get("platoon_matchup", "")).upper()
    # Look for LHH or RHH in the matchup string
    m = re.search(r"([LR])HH", pm)
    if m:
        return m.group(1)
    return ""


def hand_matched_pitcher_barrel(row):
    """Pitcher barrel% allowed vs THIS batter's handedness."""
    bh = row["batter_hand_derived"]
    if bh == "L":
        return safe_float(row.get("pitcher_barrel_vs_lhh"))
    if bh == "R":
        return safe_float(row.get("pitcher_barrel_vs_rhh"))
    # Switch hitter — use the side matching pitcher hand (bats opposite)
    ph = str(row.get("pitcher_hand", "")).strip().upper()
    if ph == "R":
        return safe_float(row.get("pitcher_barrel_vs_lhh"))
    if ph == "L":
        return safe_float(row.get("pitcher_barrel_vs_rhh"))
    return np.nan


def bucket_hit_rate(df, col, label, buckets):
    print(f"\n  {label}:")
    print(f"    {'Bucket':<16} {'N':>5} {'Hits':>5} {'Hit Rate':>9}")
    print(f"    {'-'*40}")
    for blabel, lo, hi in buckets:
        sub = df[(df[col] >= lo) & (df[col] < hi)]
        if len(sub) < 8:
            continue
        n = len(sub)
        h = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        print(f"    {blabel:<16} {n:>5} {h:>5} {rate:>8.1f}%")


def separator(df, col, label):
    hits = df[df["hit_bool"]][col].dropna()
    miss = df[~df["hit_bool"]][col].dropna()
    if len(hits) < 5 or len(miss) < 5:
        return None
    h_avg = hits.mean()
    m_avg = miss.mean()
    diff = h_avg - m_avg
    pct = (diff / m_avg * 100) if m_avg != 0 else 0
    return (pct, h_avg, m_avg, len(hits), len(miss))


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
    resolved["hr_score"] = resolved["hr_score"].apply(lambda x: safe_float(x, 0))

    # Restrict to stage-two pool
    pool = resolved[resolved["hr_score"] >= POOL_THRESHOLD].copy()
    print(f"  Pool (score >= {POOL_THRESHOLD}): {len(pool)} picks, "
          f"{round(pool['hit_bool'].mean()*100,1)}% hit rate")

    # Derive batter handedness
    pool["batter_hand_derived"] = pool.apply(derive_batter_hand, axis=1)
    hand_known = (pool["batter_hand_derived"] != "").sum()
    print(f"  Batter handedness resolved for {hand_known}/{len(pool)} picks")

    # Numeric pitcher features
    for col in ["pitcher_barrel_pct", "pitcher_hr_per_fb",
                "pitcher_barrel_vs_lhh", "pitcher_barrel_vs_rhh"]:
        if col in pool.columns:
            pool[col] = pool[col].apply(safe_float)
        else:
            pool[col] = np.nan

    pool["hand_matched_barrel"] = pool.apply(hand_matched_pitcher_barrel, axis=1)

    # ── Separator table ─────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"PITCHER-SIDE SEPARATORS WITHIN score >= {POOL_THRESHOLD} POOL")
    print("="*70)
    feats = [
        ("hand_matched_barrel",    "Pitcher Barrel% vs Batter Hand (MATCHED)"),
        ("pitcher_barrel_pct",     "Pitcher Barrel% (blended)"),
        ("pitcher_hr_per_fb",      "Pitcher HR/FB%"),
        ("pitcher_barrel_vs_lhh",  "Pitcher Barrel% vs LHH (raw)"),
        ("pitcher_barrel_vs_rhh",  "Pitcher Barrel% vs RHH (raw)"),
    ]
    print(f"\n  {'Feature':<44} {'Hits':>8} {'Miss':>8} {'% Diff':>8} {'Signal':>10}")
    print(f"  {'-'*82}")
    for col, label in feats:
        r = separator(pool, col, label)
        if r is None:
            continue
        pct, h_avg, m_avg, nh, nm = r
        flag = "🔥 STRONG+" if pct >= 15 else "✅ Pos" if pct >= 7 else "🔴 STRONG-" if pct <= -15 else "⚠️ Neg" if pct <= -7 else "↔️ Neutral"
        print(f"  {label:<44} {h_avg:>8.2f} {m_avg:>8.2f} {pct:>+7.1f}% {flag:>10s}")

    # ── Bucketed hit rate for the matched signal ────────────────────────
    print("\n" + "="*70)
    print("HIT RATE BY HAND-MATCHED PITCHER BARREL% BUCKET")
    print("(Does facing a HR-vulnerable pitcher actually lift the hit rate?)")
    print("="*70)
    bucket_hit_rate(pool, "hand_matched_barrel", "Hand-matched pitcher barrel%", [
        ("Elite pitcher <6", -99, 6),
        ("Good 6-8",          6,  8),
        ("Avg 8-10",          8,  10),
        ("Weak 10-12",        10, 12),
        ("Very weak 12+",     12, 99),
    ])

    bucket_hit_rate(pool, "pitcher_hr_per_fb", "Pitcher HR/FB%", [
        ("Stingy <10",   -99, 10),
        ("Avg 10-13",    10, 13),
        ("Weak 13-16",   13, 16),
        ("Very weak 16+",16, 99),
    ])

    # ── Date-split holdout validation ───────────────────────────────────
    print("\n" + "="*70)
    print("DATE-SPLIT HOLDOUT — does the matched signal hold out of sample?")
    print("="*70)
    pool_sorted = pool.sort_values("date_dt")
    mid = pool_sorted["date_dt"].quantile(0.5)
    first  = pool_sorted[pool_sorted["date_dt"] <= mid]
    second = pool_sorted[pool_sorted["date_dt"] > mid]

    for name, sub in [("FIRST HALF (in-sample)", first), ("SECOND HALF (holdout)", second)]:
        r = separator(sub, "hand_matched_barrel", "matched")
        print(f"\n  {name}: {len(sub)} picks")
        if r:
            pct, h_avg, m_avg, nh, nm = r
            print(f"    Hand-matched barrel: hits={h_avg:.2f} miss={m_avg:.2f} ({pct:+.1f}%)")
        # Also show hit rate for "weak pitcher" subset
        weak = sub[sub["hand_matched_barrel"] >= 10]
        if len(weak) >= 8:
            wr = round(weak["hit_bool"].mean()*100, 1)
            print(f"    Facing weak pitcher (matched barrel >=10): {int(weak['hit_bool'].sum())}/{len(weak)} = {wr}%")
        strong = sub[sub["hand_matched_barrel"] < 8]
        if len(strong) >= 8:
            sr = round(strong["hit_bool"].mean()*100, 1)
            print(f"    Facing strong pitcher (matched barrel <8):  {int(strong['hit_bool'].sum())}/{len(strong)} = {sr}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
