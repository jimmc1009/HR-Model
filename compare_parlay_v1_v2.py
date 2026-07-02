"""
compare_parlay_v1_v2.py
Head-to-head backtest: builds the 3-leg HR parlay TWO ways on the SAME
resolved games, so the only difference is the selection logic.

  v1 (old): pool = old value zones; selector = platoon + hr_per_fb/20
  v2 (new): pool = score>=8 & plus odds; selector = platoon + power_norm*2

power_norm is computed on the fly from stored features so the v2 selector
can run retroactively across the full history — true apples-to-apples.

Reports, for each: parlays attempted, per-leg hit rate, all-3-hit rate,
and a rough ROI proxy so you can see which selection logic picks better legs.
This does NOT change the model — pure comparison.
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
LEG_COUNT = 3

# v2 percentile bounds (from extract_feature_bounds.py) — for on-the-fly power_norm
V2_BOUNDS = {
    "season_barrel_pct": (0.9, 15.75), "barrel_pct_7d": (0.0, 25.0),
    "barrel_pct_5d": (0.0, 27.0), "barrel_pct_10d": (0.0, 22.0),
    "iso": (0.07, 0.27), "hr_per_fb": (0.0, 28.0), "hr_per_pa": (0.0, 6.0),
}
V2_POWER_BLEND = {
    "season_barrel_pct": 0.30, "hr_per_fb": 0.20, "hr_per_pa": 0.15,
    "iso": 0.15, "barrel_pct_7d": 0.08, "barrel_pct_10d": 0.07, "barrel_pct_5d": 0.05,
}


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


def _fl_label(fl):
    return str(int(fl)) if float(fl).is_integer() else str(fl)


def v2_norm(val, col):
    v = safe_float(val, np.nan)
    if pd.isna(v) or col not in V2_BOUNDS:
        return 0.0
    lo, hi = V2_BOUNDS[col]
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) if hi > lo else 0.0


def compute_power_norm(row):
    """On-the-fly power_norm; use stored value if present (v2 rows)."""
    stored = str(row.get("power_norm", "")).strip()
    if stored not in ("", "nan", "None"):
        return safe_float(stored, 0.0)
    return round(sum(v2_norm(row.get(c), c) * w for c, w in V2_POWER_BLEND.items()), 4)


# ── v1 pool + selector ──────────────────────────────────────────────────
def in_pool_v1(score, odds):
    if score >= 13.0 and odds <= 300:
        return True
    if 10.0 <= score < 11.0 and 301 <= odds <= 499:
        return True
    if score >= 13.0 and 301 <= odds <= 499:
        return True
    if 12.0 <= score < 13.0 and 301 <= odds <= 499:
        return True
    return False


def selector_v1(row):
    return safe_float(row.get("platoon_score", 0)) + safe_float(row.get("hr_per_fb", 0)) / 20


# ── v2 pool + selector ──────────────────────────────────────────────────
def in_pool_v2(score, odds):
    # Plus odds only; the top-20-by-score cut is applied per-day in build_parlay_v2
    return odds >= 100


def selector_v2(row):
    return safe_float(row.get("platoon_score", 0)) + compute_power_norm(row) * 2.0


def build_parlay_v2(day_df, floor=0.0):
    """v2 parlay: plus-odds pool, score>=floor, keep top 20 by score, selector ranks."""
    pool = day_df[(day_df["odds_num"] >= 100) & (day_df["hr_score"] >= floor)].copy()
    if pool.empty:
        return []
    pool = pool.sort_values("hr_score", ascending=False).head(20)
    pool["sel"] = pool.apply(selector_v2, axis=1)
    pool = pool.sort_values("sel", ascending=False).reset_index(drop=True)

    selected, used = [], set()
    for _, row in pool.iterrows():
        if len(selected) >= LEG_COUNT:
            break
        opp = str(row.get("pitcher_name", "")).strip()
        if opp and opp in used:
            continue
        selected.append(row)
        if opp:
            used.add(opp)
    if len(selected) < LEG_COUNT:
        for _, row in pool.iterrows():
            if len(selected) >= LEG_COUNT:
                break
            if not any(s.get("player_name") == row.get("player_name") for s in selected):
                selected.append(row)
    return selected


def build_parlay_v1(day_df):
    in_pool, selector = in_pool_v1, selector_v1
    pool = day_df[day_df.apply(lambda r: in_pool(r["hr_score"], r["odds_num"]), axis=1)].copy()
    if pool.empty:
        return []
    pool["sel"] = pool.apply(selector, axis=1)
    pool = pool.sort_values("sel", ascending=False).reset_index(drop=True)

    selected, used = [], set()
    for _, row in pool.iterrows():
        if len(selected) >= LEG_COUNT:
            break
        opp = str(row.get("pitcher_name", "")).strip()
        if opp and opp in used:
            continue
        selected.append(row)
        if opp:
            used.add(opp)
    if len(selected) < LEG_COUNT:
        for _, row in pool.iterrows():
            if len(selected) >= LEG_COUNT:
                break
            if not any(s.get("player_name") == row.get("player_name") for s in selected):
                selected.append(row)
    return selected


def american_to_decimal(odds):
    if odds >= 100:
        return 1 + odds / 100
    elif odds <= -100:
        return 1 + 100 / abs(odds)
    return 1.0


def evaluate(df, builder, label):
    days = sorted(df["date"].unique())
    total_legs = hit_legs = 0
    parlays = won_parlays = 0
    roi_sum = 0.0

    for d in days:
        day_df = df[df["date"] == d]
        legs = builder(day_df)
        if len(legs) < LEG_COUNT:
            continue
        parlays += 1
        all_hit = True
        dec_product = 1.0
        for leg in legs:
            total_legs += 1
            hit = str(leg.get("hit_hr", "")).strip() == "Yes"
            if hit:
                hit_legs += 1
            else:
                all_hit = False
            dec_product *= american_to_decimal(safe_float(leg.get("odds_num", 0)))
        if all_hit:
            won_parlays += 1
            roi_sum += (dec_product - 1)
        else:
            roi_sum -= 1

    leg_rate = round(hit_legs / total_legs * 100, 1) if total_legs else 0
    win_rate = round(won_parlays / parlays * 100, 1) if parlays else 0
    roi      = round(roi_sum / parlays * 100, 1) if parlays else 0

    print(f"\n  {label}")
    print(f"    Parlays built:     {parlays}")
    print(f"    Per-leg hit rate:  {leg_rate}%  ({hit_legs}/{total_legs})")
    print(f"    All-3-hit rate:    {win_rate}%  ({won_parlays}/{parlays})")
    print(f"    ROI proxy:         {roi:+.1f}% per parlay (flat $1 stake)")
    return {"parlays": parlays, "leg_rate": leg_rate, "win_rate": win_rate, "roi": roi}


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

    df["date"]     = df["date"].astype(str).str.strip()
    df["dt"]       = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()

    df["hr_score"] = df["hr_score"].apply(safe_float)
    df["odds_num"] = df["consensus_odds"].apply(lambda x: safe_float(x, 0))
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    print(f"  {len(df)} resolved rows across {df['date'].nunique()} days")

    print("\n" + "="*64)
    print("PARLAY HEAD-TO-HEAD: v1 (old) vs v2 at multiple score floors")
    print("="*64)

    r1 = evaluate(df, build_parlay_v1, "v1 — OLD zones + old selector")

    floors = [8.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    v2_results = {}
    for fl in floors:
        v2_results[fl] = evaluate(
            df, lambda d, _fl=fl: build_parlay_v2(d, floor=_fl),
            f"v2 — score>={_fl_label(fl)} pool + power_norm selector"
        )

    print("\n" + "="*64)
    print("VERDICT — v1 baseline vs v2 at each floor")
    print("="*64)
    print(f"\n  {'Strategy':<22} {'Parlays':>8} {'Leg%':>8} {'All3%':>8} {'ROI%':>10}")
    print(f"  {'-'*58}")
    print(f"  {'v1 (old)':<22} {r1['parlays']:>8} {r1['leg_rate']:>7.1f}% {r1['win_rate']:>7.1f}% {r1['roi']:>+9.1f}%")
    for fl in floors:
        r = v2_results[fl]
        print(f"  {'v2 floor '+_fl_label(fl):<22} {r['parlays']:>8} {r['leg_rate']:>7.1f}% {r['win_rate']:>7.1f}% {r['roi']:>+9.1f}%")

    print("\n  Per-leg hit rate is the trustworthy metric at this sample.")
    print("  ROI/All3 are noisy with ~22 parlays — don't over-read them.")
    print("  NOTE: most history is old-model scores; floors filter on the")
    print("  old scale for old rows. Directional until v2 data dominates.")
    print("\nDone.")


if __name__ == "__main__":
    main()
