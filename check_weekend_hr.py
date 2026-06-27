"""
check_weekend_odds.py
Tests whether market-favorite HR hitters (odds <= +300) perform
differently on weekends vs weekdays.
Uses full HR_All_Scores dataset back to May 26.
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

ODDS_THRESHOLD = 300  # "best HR hitters" per market


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
    df["odds_num"] = df["consensus_odds"].apply(safe_float)

    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["day_of_week"] = resolved["date_dt"].dt.day_name()
    resolved["day_num"]     = resolved["date_dt"].dt.dayofweek
    resolved["is_weekend"]  = resolved["day_num"].isin([5, 6])

    print(f"  {len(resolved)} resolved rows")
    print(f"  Date range: {resolved['date_dt'].min().date()} to {resolved['date_dt'].max().date()}")

    # Filter to market favorites — odds must be valid and <= +300
    favorites = resolved[
        (resolved["odds_num"] > 0) &
        (resolved["odds_num"] <= ODDS_THRESHOLD)
    ].copy()

    print(f"  Players at ≤+{ODDS_THRESHOLD} odds: {len(favorites)}")

    # ── 1. Overall favorites hit rate ─────────────────────────────────────
    print("\n" + "="*60)
    print(f"1. MARKET FAVORITES (≤+{ODDS_THRESHOLD}) — Overall")
    print("="*60)
    r, h, n = hr(favorites)
    print(f"  Overall: {h}/{n} = {r}%")
    r2, h2, n2 = hr(resolved[resolved["odds_num"] > ODDS_THRESHOLD])
    print(f"  Longshots (+{ODDS_THRESHOLD+1}+): {h2}/{n2} = {r2}%")

    # ── 2. Weekday vs Weekend ─────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"2. MARKET FAVORITES — Weekday vs Weekend")
    print("="*60)
    fav_weekday = favorites[~favorites["is_weekend"]]
    fav_weekend = favorites[favorites["is_weekend"]]
    rwd, hwd, nwd = hr(fav_weekday)
    rwe, hwe, nwe = hr(fav_weekend)
    diff = round(rwe - rwd, 1)
    print(f"  Weekday: {hwd}/{nwd} = {rwd}%")
    print(f"  Weekend: {hwe}/{nwe} = {rwe}%")
    print(f"  Difference: {diff:+.1f} percentage points")
    if abs(diff) >= 5:
        direction = "BETTER" if diff > 0 else "WORSE"
        print(f"  ⚠️  Market favorites perform {direction} on weekends")
    else:
        print(f"  ✅ No meaningful weekend effect for market favorites")

    # ── 3. By day of week ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"3. MARKET FAVORITES — By Day of Week")
    print("="*60)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in day_order:
        sub = favorites[favorites["day_of_week"] == day]
        if len(sub) < 5: continue
        r, h, n = hr(sub)
        weekend = " 📅" if day in ["Saturday", "Sunday"] else ""
        print(f"  {day:<12}: {h:3d}/{n:3d} = {r:5.1f}%{weekend}")

    # ── 4. Odds buckets × weekend ─────────────────────────────────────────
    print("\n" + "="*60)
    print("4. ODDS BUCKETS — Weekday vs Weekend")
    print("="*60)
    buckets = [
        ("≤+100 (heavy fav)", 0,   100),
        ("+101 to +150",     101,  150),
        ("+151 to +200",     151,  200),
        ("+201 to +250",     201,  250),
        ("+251 to +300",     251,  300),
    ]
    print(f"  {'Odds Bucket':<22} {'Weekday':>10} {'Weekend':>10} {'Diff':>8}")
    print(f"  {'-'*54}")
    for label, lo, hi in buckets:
        sub = resolved[(resolved["odds_num"] > lo) & (resolved["odds_num"] <= hi)]
        if len(sub) < 10: continue
        wd = sub[~sub["is_weekend"]]
        we = sub[sub["is_weekend"]]
        if len(wd) < 5 or len(we) < 5: continue
        rwd, _, nwd = hr(wd)
        rwe, _, nwe = hr(we)
        diff = round(rwe - rwd, 1)
        flag = " ⬆️" if diff >= 5 else " ⬇️" if diff <= -5 else ""
        print(f"  {label:<22} {rwd:>8.1f}%({nwd}) {rwe:>8.1f}%({nwe})  {diff:>+5.1f}%{flag}")

    # ── 5. Top individual favorites ───────────────────────────────────────
    print("\n" + "="*60)
    print(f"5. TOP PLAYERS AT ≤+{ODDS_THRESHOLD} — Weekday vs Weekend")
    print("   (10+ resolved picks at these odds)")
    print("="*60)
    player_stats = []
    for name, grp in favorites.groupby("player_name"):
        if len(grp) < 10: continue
        wd = grp[~grp["is_weekend"]]
        we = grp[grp["is_weekend"]]
        if len(wd) < 3 or len(we) < 3: continue
        rwd, hwd, nwd = hr(wd)
        rwe, hwe, nwe = hr(we)
        overall, _, _ = hr(grp)
        avg_odds = round(grp["odds_num"].mean(), 0)
        diff = round(rwe - rwd, 1)
        player_stats.append((name, overall, avg_odds, rwd, nwd, rwe, nwe, diff))

    player_stats.sort(key=lambda x: abs(x[7]), reverse=True)
    print(f"\n  {'Player':<25} {'AvgOdds':>8} {'Weekday':>10} {'Weekend':>10} {'Diff':>7}")
    print(f"  {'-'*65}")
    for name, overall, avg_odds, rwd, nwd, rwe, nwe, diff in player_stats[:20]:
        flag = " ⬆️" if diff >= 10 else " ⬇️" if diff <= -10 else ""
        print(f"  {name:<25} +{avg_odds:<7.0f} {rwd:>7.1f}%({nwd}) {rwe:>7.1f}%({nwe})  {diff:>+5.1f}%{flag}")

    # ── 6. Statistical significance ───────────────────────────────────────
    print("\n" + "="*60)
    print("6. STATISTICAL SIGNIFICANCE")
    print("="*60)
    import math
    p_overall = favorites["hit_bool"].mean()
    we_hits   = int(fav_weekend["hit_bool"].sum())
    we_n      = len(fav_weekend)
    expected  = p_overall * we_n
    std_err   = math.sqrt(p_overall * (1 - p_overall) * we_n) if we_n > 0 else 1
    z_score   = (we_hits - expected) / std_err
    print(f"  Overall favorite hit rate: {round(p_overall*100,1)}%")
    print(f"  Weekend expected hits: {expected:.1f} | Actual: {we_hits}")
    print(f"  Z-score: {z_score:.2f}")
    if abs(z_score) >= 1.96:
        print(f"  ⚠️  Statistically significant (p < 0.05)")
    elif abs(z_score) >= 1.28:
        print(f"  📊 Marginally significant (p < 0.10) — worth monitoring")
    else:
        print(f"  ✅ Not statistically significant — likely noise")

    print("\nDone.")


if __name__ == "__main__":
    main()
