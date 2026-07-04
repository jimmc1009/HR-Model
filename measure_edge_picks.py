"""
measure_edge_picks.py
Isolates HR picks the model flagged as POSITIVE EDGE in the 10+ score bucket
and measures how they actually performed — the honest test of whether the
"profitable zones" hold up.

Method:
  1. Read resolved HR_All_Scores (v1 live scores before 7/1; the real picks).
  2. Build tier x odds-zone hit rates from the SAME data (as the dashboard does).
  3. For each resolved pick with score >= 10, compute the edge the model would
     have shown at bet time = (zone hit rate) - (implied prob from its odds).
  4. Keep only picks with positive edge — the ones you'd actually bet.
  5. Measure those picks' REAL outcomes: hit rate, ROI at their odds, and
     whether they beat breakeven in practice.

IMPORTANT caveat printed in output: this uses in-sample hit rates (the zone
rate is computed from the same games being measured), so it's optimistic —
it answers "did the flagged-edge picks perform" not "will they going forward."
For that, watch it over time / out of sample.
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
MIN_ZONE_SAMPLE  = 15    # zone must have this many picks to trust its rate
MIN_EDGE         = 0.0   # positive edge threshold (0 = any positive)


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


def implied_prob(odds):
    if odds >= 100:
        return 100.0 / (odds + 100.0)
    elif odds <= -100:
        return abs(odds) / (abs(odds) + 100.0)
    return 1.0


def american_profit(odds, stake=1.0):
    """Profit on a win for a given American odds and stake."""
    if odds >= 100:
        return stake * odds / 100.0
    elif odds <= -100:
        return stake * 100.0 / abs(odds)
    return 0.0


def tier_of(score):
    if score >= 13: return "13+"
    if score >= 12: return "12-13"
    if score >= 11: return "11-12"
    if score >= 10: return "10-11"
    return "under10"


def zone_of(odds):
    if odds <= 300:            return "≤+300"
    if 301 <= odds <= 499:     return "+301-499"
    if 500 <= odds <= 699:     return "+500-699"
    return "+700+"


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
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()

    df["score"]    = df["hr_score"].apply(safe_float)
    df["odds"]     = df["consensus_odds"].apply(safe_float)
    df["hit_bool"] = df["hit_hr"].astype(str).str.strip() == "Yes"
    df = df[(df["odds"] != 0)].copy()

    df["tier"] = df["score"].apply(tier_of)
    df["zone"] = df["odds"].apply(zone_of)

    # Build tier x zone hit rates (same as dashboard)
    zone_rates = {}
    for (t, z), sub in df.groupby(["tier", "zone"]):
        if len(sub) >= MIN_ZONE_SAMPLE:
            zone_rates[(t, z)] = sub["hit_bool"].mean()

    # Focus: 10+ bucket
    tenplus = df[df["score"] >= 10.0].copy()
    print(f"\n  {len(tenplus)} resolved 10+ picks total")

    # Compute model's shown edge for each and keep positive-edge ones
    def edge_row(r):
        zr = zone_rates.get((r["tier"], r["zone"]))
        if zr is None:
            return np.nan
        return zr - implied_prob(r["odds"])

    tenplus["edge"] = tenplus.apply(edge_row, axis=1)
    flagged = tenplus[tenplus["edge"] > MIN_EDGE].copy()

    print(f"  {len(flagged)} of them were POSITIVE-EDGE picks (zone rate > implied, zone n>={MIN_ZONE_SAMPLE})")

    if flagged.empty:
        print("\n  No positive-edge 10+ picks with a trustworthy zone sample yet.")
        print("  Zones need 15+ resolved picks before an edge can be computed.")
        return

    # Measure REAL performance of the flagged picks
    n      = len(flagged)
    hits   = int(flagged["hit_bool"].sum())
    rate   = round(hits / n * 100, 1)

    # ROI at flat $1 stake, real American odds
    profit = 0.0
    for _, r in flagged.iterrows():
        if r["hit_bool"]:
            profit += american_profit(r["odds"], 1.0)
        else:
            profit -= 1.0
    roi = round(profit / n * 100, 1)

    avg_odds     = int(flagged["odds"].mean())
    avg_be       = round(implied_prob(flagged["odds"].mean()) * 100, 1)

    print("\n" + "="*58)
    print("PERFORMANCE OF POSITIVE-EDGE 10+ PICKS (in-sample)")
    print("="*58)
    print(f"  Picks:            {n}")
    print(f"  Actual hit rate:  {rate}%   ({hits}/{n})")
    print(f"  Avg odds:         +{avg_odds}  (breakeven {avg_be}%)")
    print(f"  Beat breakeven?   {'YES' if rate > avg_be else 'NO'}  ({rate}% vs {avg_be}%)")
    print(f"  ROI (flat $1):    {roi:+.1f}% per bet")
    print(f"  Net units:        {profit:+.2f}")

    # Break it down by tier x zone so you see WHICH zones carried it
    print("\n  By tier | zone:")
    print(f"    {'tier|zone':<20} {'n':>4} {'hit%':>7} {'be%':>6} {'roi%':>8}")
    for (t, z), sub in flagged.groupby(["tier", "zone"]):
        sn   = len(sub)
        sh_  = int(sub["hit_bool"].sum())
        sr   = round(sh_ / sn * 100, 1)
        sbe  = round(implied_prob(sub["odds"].mean()) * 100, 1)
        sp   = sum(american_profit(o, 1.0) if h else -1.0
                   for o, h in zip(sub["odds"], sub["hit_bool"]))
        sroi = round(sp / sn * 100, 1)
        print(f"    {t+'|'+z:<20} {sn:>4} {sr:>6.1f}% {sbe:>5.1f}% {sroi:>+7.1f}%")

    print("\n  ⚠️  IN-SAMPLE: zone rates come from the same games measured, so")
    print("  these numbers are optimistic. A positive ROI here is necessary")
    print("  but NOT sufficient — the real test is out-of-sample going forward.")
    print("  If even in-sample ROI is negative, the zones aren't real edge.")
    print("\nDone.")


if __name__ == "__main__":
    main()
