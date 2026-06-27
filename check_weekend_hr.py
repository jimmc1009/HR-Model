"""
check_weekend_hr.py
Tests the claim that top HR hitters perform worse on weekends.
Splits HR_All_Scores by day of week and compares hit rates.
Also checks if high-scoring players specifically underperform on weekends.
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
    if len(sub) == 0: return 0.0, 0, 0
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

    df["date_dt"]  = pd.to_datetime(df["date"], errors="coerce")
    df["hit_bool"] = df["hit_hr"].astype(str).str.strip() == "Yes"
    df["hr_score"] = df["hr_score"].apply(safe_float)

    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["day_of_week"] = resolved["date_dt"].dt.day_name()
    resolved["day_num"]     = resolved["date_dt"].dt.dayofweek  # 0=Mon, 6=Sun
    resolved["is_weekend"]  = resolved["day_num"].isin([5, 6])  # Sat=5, Sun=6

    clean = resolved[resolved["date_dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()
    print(f"  {len(clean)} resolved rows from {MODEL_START_DATE}")
    print(f"  Date range: {clean['date_dt'].min().date()} to {clean['date_dt'].max().date()}")

    # ── 1. Hit rate by day of week — all players ──────────────────────────
    print("\n" + "="*60)
    print("1. HIT RATE BY DAY OF WEEK — All scored players")
    print("="*60)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_order:
        sub = clean[clean["day_of_week"] == day]
        if len(sub) < 5: continue
        r, h, n = hr(sub)
        weekend = " 📅 WEEKEND" if day in ["Saturday", "Sunday"] else ""
        print(f"  {day:<12}: {h:3d}/{n:4d} = {r:5.1f}%{weekend}")

    # ── 2. Weekday vs Weekend ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("2. WEEKDAY vs WEEKEND — All scored players")
    print("="*60)
    weekday = clean[~clean["is_weekend"]]
    weekend = clean[clean["is_weekend"]]
    wr, wh, wn   = hr(weekday)
    wr2, wh2, wn2 = hr(weekend)
    print(f"  Weekday: {wh}/{wn} = {wr}%")
    print(f"  Weekend: {wh2}/{wn2} = {wr2}%")
    print(f"  Difference: {round(wr2 - wr, 1):+.1f} percentage points")

    # ── 3. Top HR hitters specifically ───────────────────────────────────
    print("\n" + "="*60)
    print("3. TOP HR HITTERS (score 12+) — Weekday vs Weekend")
    print("="*60)
    top = clean[clean["hr_score"] >= 12].copy()
    top_weekday = top[~top["is_weekend"]]
    top_weekend = top[top["is_weekend"]]
    twr, twh, twn   = hr(top_weekday)
    twr2, twh2, twn2 = hr(top_weekend)
    print(f"  Weekday: {twh}/{twn} = {twr}%")
    print(f"  Weekend: {twh2}/{twn2} = {twr2}%")
    print(f"  Difference: {round(twr2 - twr, 1):+.1f} percentage points")

    # ── 4. By score tier × weekend ────────────────────────────────────────
    print("\n" + "="*60)
    print("4. BY SCORE TIER × WEEKDAY vs WEEKEND")
    print("="*60)
    tiers = [
        ("13+",   13, 99),
        ("12-13", 12, 13),
        ("11-12", 11, 12),
        ("10-11", 10, 11),
        ("9-10",   9, 10),
    ]
    print(f"  {'Tier':8s} {'Weekday':>10} {'Weekend':>10} {'Diff':>8}")
    print(f"  {'-'*40}")
    for label, lo, hi in tiers:
        sub = clean[(clean["hr_score"] >= lo) & (clean["hr_score"] < hi)]
        if len(sub) < 10: continue
        wd = sub[~sub["is_weekend"]]
        we = sub[sub["is_weekend"]]
        if len(wd) < 5 or len(we) < 5: continue
        rwd, _, nwd = hr(wd)
        rwe, _, nwe = hr(we)
        diff = round(rwe - rwd, 1)
        flag = " ✅ Weekend better" if diff >= 3 else " ❌ Weekend worse" if diff <= -3 else ""
        print(f"  {label:8s} {rwd:>8.1f}% {rwe:>8.1f}%  {diff:>+6.1f}%{flag}")

    # ── 5. Named top HR hitters ───────────────────────────────────────────
    print("\n" + "="*60)
    print("5. TOP INDIVIDUAL PLAYERS — Weekday vs Weekend")
    print("   (Players with 10+ resolved picks)")
    print("="*60)

    player_stats = []
    for name, grp in clean.groupby("player_name"):
        if len(grp) < 10: continue
        wd = grp[~grp["is_weekend"]]
        we = grp[grp["is_weekend"]]
        if len(wd) < 3 or len(we) < 3: continue
        rwd, hwd, nwd = hr(wd)
        rwe, hwe, nwe = hr(we)
        overall, _, _ = hr(grp)
        diff = round(rwe - rwd, 1)
        player_stats.append((name, overall, rwd, nwd, rwe, nwe, diff))

    player_stats.sort(key=lambda x: abs(x[6]), reverse=True)
    print(f"\n  {'Player':<25} {'Overall':>8} {'Weekday':>8} {'Weekend':>8} {'Diff':>7}")
    print(f"  {'-'*60}")
    for name, overall, rwd, nwd, rwe, nwe, diff in player_stats[:20]:
        flag = " ⬆️" if diff >= 5 else " ⬇️" if diff <= -5 else ""
        print(f"  {name:<25} {overall:>7.1f}% {rwd:>6.1f}%({nwd}) {rwe:>6.1f}%({nwe})  {diff:>+5.1f}%{flag}")

    # ── 6. Statistical significance check ────────────────────────────────
    print("\n" + "="*60)
    print("6. STATISTICAL CONTEXT")
    print("="*60)
    total_days = clean["date_dt"].nunique()
    weekend_days = clean[clean["is_weekend"]]["date_dt"].nunique()
    weekday_days = total_days - weekend_days
    print(f"  Total days of data: {total_days}")
    print(f"  Weekday days: {weekday_days} | Weekend days: {weekend_days}")
    print(f"  Weekend picks: {len(weekend)} | Weekday picks: {len(weekday)}")

    # Simple binomial test approximation
    import math
    if len(weekend) >= 20:
        p_overall = clean["hit_bool"].mean()
        we_hits   = int(weekend["hit_bool"].sum())
        we_n      = len(weekend)
        expected  = p_overall * we_n
        std_err   = math.sqrt(p_overall * (1 - p_overall) * we_n)
        z_score   = (we_hits - expected) / std_err if std_err > 0 else 0
        print(f"\n  Weekend hit rate vs expected:")
        print(f"    Expected hits: {expected:.1f} | Actual hits: {we_hits}")
        print(f"    Z-score: {z_score:.2f}")
        if abs(z_score) >= 1.96:
            print(f"    ⚠️  Statistically significant difference (p < 0.05)")
        elif abs(z_score) >= 1.28:
            print(f"    📊 Marginally significant (p < 0.10) — worth monitoring")
        else:
            print(f"    ✅ Not statistically significant — likely noise")

    print("\nDone.")


if __name__ == "__main__":
    main()
