"""
compare_parlay_legcount.py
Compares 2-leg vs 3-leg vs 4-leg HR parlays on the same historical pool,
using the confirmed v1 logic (value-zone pool + platoon + hr_per_fb selector).

Purpose: for small-stakes fun, hit FREQUENCY matters more than theoretical EV.
Shows how often each leg-count actually cashes, plus a "multiple 2-leggers
per day" mode (pair up the top legs into several 2-leg tickets daily).

Reports per format: tickets, hit rate, ROI proxy, and a plain-English
"how often you'd feel a win" read.
"""

import os
import json
import time
from itertools import combinations

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


def get_gspread_client():
    info  = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=20):
    for a in range(retries):
        try:
            return func()
        except Exception:
            if a < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def in_pool_v1(score, odds):
    if score >= 13.0 and odds <= 300:            return True
    if score >= 13.0 and 301 <= odds <= 499:     return True
    if 12.0 <= score < 13.0 and 301 <= odds <= 499: return True
    if 10.0 <= score < 11.0 and 301 <= odds <= 499: return True
    return False


def selector_v1(row):
    return safe_float(row.get("platoon_score", 0)) + safe_float(row.get("hr_per_fb", 0)) / 20


def american_to_decimal(odds):
    if odds >= 100:   return 1 + odds / 100
    if odds <= -100:  return 1 + 100 / abs(odds)
    return 1.0


def rank_pool(day_df):
    """Return the day's pool ranked by selector, diversified by pitcher."""
    pool = day_df[day_df.apply(lambda r: in_pool_v1(r["hr_score"], r["odds_num"]), axis=1)].copy()
    if pool.empty:
        return []
    pool["sel"] = pool.apply(selector_v1, axis=1)
    pool = pool.sort_values("sel", ascending=False).reset_index(drop=True)

    ranked, used = [], set()
    for _, row in pool.iterrows():
        opp = str(row.get("pitcher_name", "")).strip()
        if opp and opp in used:
            continue
        ranked.append(row)
        if opp:
            used.add(opp)
    return ranked


def eval_single_ticket(df, leg_count):
    """One ticket per day: top N legs. Returns (tickets, hits, roi, leg_rate)."""
    days = sorted(df["date"].unique())
    tickets = won = 0
    roi_sum = 0.0
    leg_hits = leg_total = 0

    for d in days:
        ranked = rank_pool(df[df["date"] == d])
        if len(ranked) < leg_count:
            continue
        legs = ranked[:leg_count]
        tickets += 1
        all_hit = True
        dec = 1.0
        for leg in legs:
            leg_total += 1
            hit = str(leg.get("hit_hr", "")).strip() == "Yes"
            if hit:
                leg_hits += 1
            else:
                all_hit = False
            dec *= american_to_decimal(safe_float(leg.get("odds_num", 0)))
        if all_hit:
            won += 1
            roi_sum += (dec - 1)
        else:
            roi_sum -= 1

    hit_rate = round(won / tickets * 100, 1) if tickets else 0
    roi      = round(roi_sum / tickets * 100, 1) if tickets else 0
    leg_rate = round(leg_hits / leg_total * 100, 1) if leg_total else 0
    return tickets, won, hit_rate, roi, leg_rate


def eval_multi_two_leggers(df, max_tickets=3):
    """Several 2-leg tickets per day: pair the top legs into non-overlapping
    2-leg tickets (top1+top2, top3+top4, ...), up to max_tickets/day."""
    days = sorted(df["date"].unique())
    tickets = won = 0
    roi_sum = 0.0

    for d in days:
        ranked = rank_pool(df[df["date"] == d])
        # form non-overlapping pairs from the ranked list
        pairs = []
        i = 0
        while i + 1 < len(ranked) and len(pairs) < max_tickets:
            pairs.append((ranked[i], ranked[i + 1]))
            i += 2
        for a, b in pairs:
            tickets += 1
            dec = american_to_decimal(safe_float(a.get("odds_num", 0))) * \
                  american_to_decimal(safe_float(b.get("odds_num", 0)))
            a_hit = str(a.get("hit_hr", "")).strip() == "Yes"
            b_hit = str(b.get("hit_hr", "")).strip() == "Yes"
            if a_hit and b_hit:
                won += 1
                roi_sum += (dec - 1)
            else:
                roi_sum -= 1

    hit_rate = round(won / tickets * 100, 1) if tickets else 0
    roi      = round(roi_sum / tickets * 100, 1) if tickets else 0
    return tickets, won, hit_rate, roi


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    sh   = with_retry(lambda: gc.open_by_key(sheet_id))
    ws   = sh.worksheet("HR_All_Scores")
    vals = with_retry(lambda: ws.get_all_values())
    df   = pd.DataFrame(vals[1:], columns=vals[0])
    df   = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]

    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()
    df["hr_score"] = df["hr_score"].apply(safe_float)
    df["odds_num"] = df["consensus_odds"].apply(lambda x: safe_float(x, 0))
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    n_days = df["date"].nunique()
    print(f"  {len(df)} resolved rows across {n_days} days\n")

    print("="*62)
    print("LEG-COUNT COMPARISON — one ticket/day, top legs by selector")
    print("="*62)
    print(f"\n  {'Format':<12} {'Tickets':>8} {'Wins':>6} {'Hit%':>7} {'ROI%':>9} {'Leg%':>7}")
    print(f"  {'-'*54}")
    for lc in [2, 3, 4]:
        t, w, hr, roi, lr = eval_single_ticket(df, lc)
        print(f"  {str(lc)+'-leg':<12} {t:>8} {w:>6} {hr:>6.1f}% {roi:>+8.1f}% {lr:>6.1f}%")

    print("\n" + "="*62)
    print("MULTIPLE 2-LEGGERS PER DAY (up to 3 tickets/day)")
    print("="*62)
    for mt in [2, 3]:
        t, w, hr, roi = eval_multi_two_leggers(df, max_tickets=mt)
        wins_per_week = round(w / n_days * 7, 1) if n_days else 0
        print(f"\n  Up to {mt} two-leggers/day:")
        print(f"    Tickets: {t}   Wins: {w}   Hit rate: {hr}%   ROI proxy: {roi:+.1f}%")
        print(f"    ≈ {wins_per_week} winning tickets per week")

    print("\n" + "="*62)
    print("THE FUN READ")
    print("="*62)
    t2, w2, hr2, roi2, _ = eval_single_ticket(df, 2)
    t3, w3, hr3, roi3, _ = eval_single_ticket(df, 3)
    print(f"\n  Single 2-leg/day: cashed {w2}/{t2} days ({hr2}%)")
    print(f"  Single 3-leg/day: cashed {w3}/{t3} days ({hr3}%)")
    print(f"\n  2-leggers hit ~{round(hr2/max(hr3,0.1),1)}x as often as 3-leggers.")
    print("  For small-stakes fun, more frequent cashes = more fun, even if")
    print("  the big 3-leg payout is rarer. ROI proxy is noisy at this sample")
    print("  — use hit frequency to decide, not ROI.")
    print("\nDone.")


if __name__ == "__main__":
    main()
