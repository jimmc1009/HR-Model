"""
compare_singles_v1_v2.py
TRUE same-game singles backtest: v1 vs v2.

For every resolved historical row (before 2026-07-01), the stored hr_score IS
the real v1 score as produced live. We recompute the v2 score on the SAME row
using the ACTUAL live function (imported from hr_picks.py — no replication risk).

Then, per day, we take each model's top-N picks and compare their HR hit rate.
This answers: do v2's top picks hit better or worse than v1's, on identical games?

Only the scoring differs — same features, same games, same days.
"""

import os
import json
import time

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

# Import the REAL v2 scoring function so the backtest uses the live formula
import hr_picks

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MODEL_START_DATE = "2026-06-09"
V2_LIVE_DATE     = "2026-07-01"   # rows on/after this already have v2 scores stored
TOP_N_LIST       = [3, 5, 10]     # compare top-3, top-5, top-10 per day


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


def hit_rate_for_topn(df, score_col, n):
    """For each day, take top-n by score_col, pool their hits. Return overall rate."""
    total = hits = 0
    for d, day in df.groupby("date"):
        top = day.sort_values(score_col, ascending=False).head(n)
        total += len(top)
        hits  += int((top["hit_bool"]).sum())
    rate = round(hits / total * 100, 1) if total else 0.0
    return rate, hits, total


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

    df["date"] = df["date"].astype(str).str.strip()
    df["dt"]   = pd.to_datetime(df["date"], errors="coerce")

    # Use only v1-era resolved rows: before V2_LIVE_DATE, so stored score = real v1
    df = df[(df["dt"] >= pd.Timestamp(MODEL_START_DATE)) &
            (df["dt"] < pd.Timestamp(V2_LIVE_DATE))].copy()
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    df["hit_bool"] = df["hit_hr"].astype(str).str.strip() == "Yes"

    print(f"  {len(df)} v1-era resolved rows across {df['date'].nunique()} days")

    # v1 score = stored hr_score
    df["v1_score"] = df["hr_score"].apply(safe_float)

    # v2 score = recompute with the REAL live function on each row
    def recompute_v2(row):
        score, _pn, _bd = hr_picks.compute_score_v2(row)
        return score
    df["v2_score"] = df.apply(recompute_v2, axis=1)

    print("\n" + "="*60)
    print("SINGLES BACKTEST — v1 (stored) vs v2 (recomputed) same games")
    print("="*60)

    overall = round(df["hit_bool"].mean() * 100, 1)
    print(f"\n  Pool baseline hit rate: {overall}%  ({int(df['hit_bool'].sum())}/{len(df)})")

    print(f"\n  {'Top-N/day':<12} {'v1 Hit%':>10} {'v2 Hit%':>10} {'Winner':>10}")
    print(f"  {'-'*46}")
    for n in TOP_N_LIST:
        v1r, v1h, v1t = hit_rate_for_topn(df, "v1_score", n)
        v2r, v2h, v2t = hit_rate_for_topn(df, "v2_score", n)
        winner = "v2" if v2r > v1r else "v1" if v1r > v2r else "tie"
        print(f"  Top {n:<8} {v1r:>9.1f}% {v2r:>9.1f}% {winner:>10}")
        print(f"  {'':12} {f'({v1h}/{v1t})':>10} {f'({v2h}/{v2t})':>10}")

    # Also: correlation / overlap — how different are the top picks?
    print("\n  Pick overlap (are they even choosing different players?):")
    for n in TOP_N_LIST:
        overlap_total = shared = 0
        for d, day in df.groupby("date"):
            v1top = set(day.sort_values("v1_score", ascending=False).head(n)["player_name"])
            v2top = set(day.sort_values("v2_score", ascending=False).head(n)["player_name"])
            if v1top and v2top:
                shared += len(v1top & v2top)
                overlap_total += n
        pct = round(shared / overlap_total * 100, 1) if overlap_total else 0
        print(f"    Top {n}: {pct}% of picks shared between v1 and v2")

    print("\n  Higher hit% wins. Overlap shows how much v2 actually diversifies.")
    print("  NOTE: v2 recomputed from stored features via live compute_score_v2.")
    print("\nDone.")


if __name__ == "__main__":
    main()
