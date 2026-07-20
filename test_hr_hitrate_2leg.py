"""
test_hr_hitrate_2leg.py  —  find the combination that CASHES most (ignores edge)

Pure-hit-rate search for a 2-leg parlay: sweeps odds ceilings x tier floors and
reports, out-of-sample, which combo produces the highest 2-leg CASH rate — plus
how often you could actually fill the ticket (fillable days) and the resulting
average payout, so you can see the frequency/payout tradeoff honestly.

This is deliberately edge-blind: legs are ranked by predicted HIT RATE (highest
first), NOT by the selector. Goal is cashes, not value.

Walk-forward: predicted hit rate for each leg = past-only tier x zone rate
(shrunk), so nothing is graded against a lookup that has seen it.

Env: MODEL_START=2026-06-09  MIN_TRAIN_DAYS=10  MIN_ZONE_N=15
Local: CSV=/path/HR_All_Scores.csv python test_hr_hitrate_2leg.py
"""
import os
import json
import math
import numpy as np
import pandas as pd

MODEL_START    = os.environ.get("MODEL_START", "2026-06-09")
MIN_TRAIN_DAYS = int(os.environ.get("MIN_TRAIN_DAYS", "10"))
MIN_ZONE_N     = int(os.environ.get("MIN_ZONE_N", "15"))

# Combinations to sweep: (odds ceiling, tier floor label, min score)
ODDS_CEILINGS = [250, 300, 350]
TIER_FLOORS = [
    ("13-15 only", 13.0, 15.0),   # 13-14 + 14-15, exclude weak 15+
    ("12+",        12.0, 99.0),
    ("10+",        10.0, 99.0),
    ("9+",          9.0, 99.0),
]


def load_scores():
    csv = os.environ.get("CSV")
    if csv:
        print(f"Loading {csv}")
        return pd.read_csv(csv, dtype=str)
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ[os.environ.get("SHEET_ENV_VAR", "GOOGLE_SHEET_ID")])
    vals = sh.worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(vals[1:], columns=vals[0])


def sf(v, d=0.0):
    try:
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return d


def odds_zone(o):
    if o <= 300:  return "le300"
    if o <= 499:  return "301-499"
    if o <= 699:  return "500-699"
    return "700plus"


def score_tier(s):
    if s >= 15:  return "15+"
    if s >= 14:  return "14-15"
    if s >= 13:  return "13-14"
    if s >= 12:  return "12-13"
    if s >= 11:  return "11-12"
    if s >= 10:  return "10-11"
    if s >= 9:   return "9-10"
    if s >= 8.5: return "8.5-9"
    return "below-8.5"


def prep(df):
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START)]
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if df.empty:
        return df
    df["score"] = df.apply(
        lambda r: sf(r.get("hr_score_corrected"))
        if str(r.get("hr_score_corrected", "")).strip() not in ("", "nan", "None")
        else sf(r.get("hr_score")), axis=1)
    df["odds"] = df["consensus_odds"].apply(sf)
    df["hit"] = (df["hit_hr"].astype(str).str.strip() == "Yes").astype(int)
    df["tier"] = df["score"].apply(score_tier)
    df["zone"] = df["odds"].apply(odds_zone)
    df["day"] = df["date_dt"].dt.date
    return df[(df["odds"] > 0) & (df["score"] > 0)].sort_values("date_dt")


def lookup_from(train):
    lut = {}
    if train.empty:
        return lut
    base = train["hit"].mean()
    K = 40.0
    for (t, z), g in train.groupby(["tier", "zone"]):
        if len(g) >= MIN_ZONE_N:
            n = len(g)
            lut[(t, z)] = (n * g["hit"].mean() + K * base) / (n + K)
    return lut


def dec(o):
    return 1 + o / 100.0


def sweep(df):
    days = sorted(df["day"].unique())
    print("=" * 74)
    print("  PURE HIT-RATE 2-LEG SWEEP (walk-forward, edge-blind, ranked by hit%)")
    print("=" * 74)
    print(f"  {'combo':>22} {'fill days':>9} {'tickets':>8} {'leg hit%':>9} "
          f"{'CASH%':>7} {'avg payout':>11} {'ROI/unit':>9}")
    print("  " + "-" * 70)

    results = []
    for ceil in ODDS_CEILINGS:
        for tlabel, tlo, thi in TIER_FLOORS:
            tickets = cashes = leg_hits = leg_n = fill_days = 0
            profit = 0.0
            payouts = []
            for i, d in enumerate(days):
                if i < MIN_TRAIN_DAYS:
                    continue
                lut = lookup_from(df[df["day"] < d])
                g = df[df["day"] == d].copy()
                g = g[(g["odds"] <= ceil) & (g["score"] >= tlo) & (g["score"] < thi)]
                if g.empty:
                    continue
                g["pred"] = g.apply(lambda r: lut.get((r["tier"], r["zone"])), axis=1)
                g = g[g["pred"].notna()]
                if len(g) < 2:
                    continue
                # edge-blind: rank by predicted hit rate, take the 2 highest
                g = g.sort_values("pred", ascending=False).head(2)
                fill_days += 1
                tickets += 1
                won = int(g["hit"].sum() == 2)
                cashes += won
                leg_hits += int(g["hit"].sum()); leg_n += 2
                payout = dec(g.iloc[0]["odds"]) * dec(g.iloc[1]["odds"]) - 1
                payouts.append(payout)
                profit += payout if won else -1.0
            if tickets == 0:
                continue
            cash_rate = cashes / tickets * 100
            leg_rate = leg_hits / leg_n * 100
            avg_pay = np.mean(payouts) * 100 if payouts else 0
            roi = profit / tickets * 100
            results.append((ceil, tlabel, fill_days, tickets, leg_rate, cash_rate, avg_pay, roi))
            print(f"  {f'<=+{ceil} | {tlabel}':>22} {fill_days:9d} {tickets:8d} "
                  f"{leg_rate:8.1f}% {cash_rate:6.1f}% {avg_pay:+10.0f}% {roi:+8.1f}%")

    print("  " + "-" * 70)
    if not results:
        print("  Not enough out-of-sample days / short-odds legs to evaluate.")
        return
    best_cash = max(results, key=lambda r: r[5])
    best_roi = max(results, key=lambda r: r[7])
    print(f"  HIGHEST CASH RATE : <=+{best_cash[0]} | {best_cash[1]} "
          f"-> {best_cash[5]:.1f}% cash on {best_cash[3]} tickets "
          f"({best_cash[2]} fillable days)")
    print(f"  (best ROI combo   : <=+{best_roi[0]} | {best_roi[1]} "
          f"-> {best_roi[7]:+.1f}% ROI, {best_roi[5]:.1f}% cash)")
    print("\n  Tradeoff to weigh: highest CASH% often has fewest fillable days and\n"
          "  smallest payout. Pick the combo whose cash rate AND fill-days you can\n"
          "  live with — that's the one I'll build the pure-hit-rate section around.")
    print("  NOTE: ROI excludes your price-shopping edge; use it to compare combos,\n"
          "  not as a profit forecast. Sample is small — treat as directional.")


def main():
    df = prep(load_scores())
    if df.empty:
        print("No resolved rows after MODEL_START.")
        return
    print(f"Resolved legs since {MODEL_START}: {len(df)} across {df['day'].nunique()} days\n")
    sweep(df)


if __name__ == "__main__":
    main()
