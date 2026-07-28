#!/usr/bin/env python3
"""
test_singles_pnl.py

Flat-stake singles P/L on the +EV buckets only, priced at consensus_odds
(tests the MODEL, not shopping). Reads a resolved HR_All_Scores export.

The +EV buckets (hit rate must beat the price's breakeven, from the
July cross-tab):
    <= +300 & 14-15   (43.9% vs 25.0% breakeven)
    <= +300 & 13-14   (30.6% vs 25.0%)
    +301-499 & 14-15  (31.0% vs ~20%)
Everything else is excluded by design — this is the ~4-5/day defensible set.

Reports per bucket and overall:
    bets, hits, hit rate, units P/L, ROI, and a bootstrap 95% CI on ROI.
Plus the stress test that killed the +28% mirage last time: drop the best
N days and see if the edge survives.

USAGE
    python3 test_singles_pnl.py HR_All_Scores.csv
    python3 test_singles_pnl.py HR_All_Scores.csv --stake 1 --drop-best 3

CAVEATS PRINTED WITH RESULTS.
"""

import sys
import argparse
import numpy as np
import pandas as pd

STAKE = 1.0

# (label, odds_low, odds_high_inclusive, score_low, score_high_exclusive, breakeven)
EV_BUCKETS = [
    ("<=+300 | 14-15",   -10000, 300, 14.0, 15.0, 0.250),
    ("<=+300 | 13-14",   -10000, 300, 13.0, 14.0, 0.250),
    ("+301-499 | 14-15",    301, 499, 14.0, 15.0, 0.200),
]


def american_profit(odds, stake):
    """Profit (not payout) on a winning bet of `stake` at american `odds`."""
    o = float(odds)
    if o >= 0:
        return stake * o / 100.0
    return stake * 100.0 / abs(o)


def bootstrap_roi_ci(unit_results, iters=5000, seed=0):
    a = np.asarray(unit_results, dtype=float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(iters, len(a)), replace=True).mean(axis=1)
    return (np.percentile(means, 2.5) * 100, np.percentile(means, 97.5) * 100)


def load(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    need = ["date", "hr_score", "consensus_odds", "hit_hr"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"FATAL: missing columns: {miss}")
    df["score"] = pd.to_numeric(df["hr_score"], errors="coerce")
    df["odds"] = pd.to_numeric(
        df["consensus_odds"].astype(str).str.replace("+", "", regex=False).str.strip(),
        errors="coerce")
    df["res"] = df["hit_hr"].str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Yes/No only — Void (DNP) and Pending are not bets.
    df = df[df["res"].isin(["yes", "no"])].copy()
    df = df.dropna(subset=["score", "odds", "date"])
    return df


def in_bucket(row, b):
    _, olo, ohi, slo, shi, _ = b
    return (olo <= row["odds"] <= ohi) and (slo <= row["score"] < shi)


def settle(sub, stake):
    hits = (sub["res"] == "yes").sum()
    units = []
    for _, r in sub.iterrows():
        units.append(american_profit(r["odds"], stake) if r["res"] == "yes" else -stake)
    units = np.array(units)
    return hits, units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--stake", type=float, default=STAKE)
    ap.add_argument("--drop-best", type=int, default=3,
                    help="days to drop in the stress test")
    args = ap.parse_args()

    df = load(args.csv)
    print("=" * 74)
    print("SINGLES P/L — +EV BUCKETS ONLY — priced at consensus_odds")
    print(f"stake {args.stake:g}u flat   |   resolved rows in log: {len(df)}")
    print("=" * 74)

    # tag bucket membership
    df["bucket"] = ""
    for b in EV_BUCKETS:
        m = df.apply(lambda r: in_bucket(r, b), axis=1)
        df.loc[m & (df["bucket"] == ""), "bucket"] = b[0]

    bet = df[df["bucket"] != ""].copy()
    if bet.empty:
        print("\nNo rows fall in the +EV buckets. Nothing to settle.")
        return

    ndays = bet["date"].dt.date.nunique()
    print(f"\nbets: {len(bet)}   across {ndays} days   "
          f"(~{len(bet)/max(ndays,1):.1f} per day)\n")

    print(f"{'bucket':<20}{'bets':>6}{'hits':>6}{'rate':>8}"
          f"{'BE':>7}{'units':>9}{'ROI':>9}{'ROI 95% CI':>20}")
    print("-" * 74)

    all_units = []
    for b in EV_BUCKETS:
        sub = bet[bet["bucket"] == b[0]]
        if sub.empty:
            print(f"{b[0]:<20}{'0':>6}{'--':>6}{'--':>8}{b[5]*100:>6.0f}%"
                  f"{'--':>9}{'--':>9}{'(no rows)':>20}")
            continue
        hits, units = settle(sub, args.stake)
        all_units.extend(units.tolist())
        n = len(sub)
        rate = hits / n
        roi = units.sum() / (n * args.stake) * 100
        lo, hi = bootstrap_roi_ci(units / args.stake)
        edge = rate - b[5]
        flag = "" if edge > 0 else "  <-- under BE in-sample"
        print(f"{b[0]:<20}{n:>6}{hits:>6}{rate*100:>7.1f}%{b[5]*100:>6.0f}%"
              f"{units.sum():>+9.2f}{roi:>+8.1f}%   [{lo:>+6.1f},{hi:>+6.1f}]{flag}")

    all_units = np.array(all_units)
    tot_roi = all_units.sum() / (len(all_units) * args.stake) * 100
    lo, hi = bootstrap_roi_ci(all_units / args.stake)
    print("-" * 74)
    print(f"{'OVERALL':<20}{len(all_units):>6}"
          f"{int((bet['res']=='yes').sum()):>6}"
          f"{(bet['res']=='yes').mean()*100:>7.1f}%{'':>7}"
          f"{all_units.sum():>+9.2f}{tot_roi:>+8.1f}%   [{lo:>+6.1f},{hi:>+6.1f}]")

    # ── stress test: drop the best N days ────────────────────────────
    print("\n" + "=" * 74)
    print(f"STRESS TEST — remove the best {args.drop_best} days")
    print("=" * 74)
    daily = (bet.assign(u=all_units)
                .groupby(bet["date"].dt.date)["u"].sum()
                .sort_values(ascending=False))
    if len(daily) <= args.drop_best:
        print("  Not enough distinct days to run this.")
    else:
        drop = daily.head(args.drop_best)
        kept_idx = ~bet["date"].dt.date.isin(drop.index)
        kept = all_units[kept_idx.values]
        kroi = kept.sum() / len(kept) * 100
        print(f"\n  full sample ROI      : {tot_roi:+.1f}%  ({len(all_units)} bets)")
        print(f"  best {args.drop_best} days removed  : {kroi:+.1f}%  ({len(kept)} bets)")
        print(f"  those {args.drop_best} days were worth: {drop.sum():+.2f}u of "
              f"{all_units.sum():+.2f}u total")
        if all_units.sum() > 0 and kept.sum() <= 0:
            print("\n  -> The edge is a hot streak. Same failure mode as the +28%")
            print("     top-10 result that collapsed to -37%. NOT established.")
        elif kroi > 0:
            print("\n  -> Survives removal of its best days. More trustworthy,")
            print("     though still in-sample and small.")

    print("\n" + "=" * 74)
    print("READ BEFORE ACTING")
    print("=" * 74)
    print(f"""
  * IN-SAMPLE. The +EV buckets were CHOSEN from this same cross-tab, so
    these ROIs are optimistic by construction. The only honest test is
    forward: freeze these three buckets, then measure new picks.
  * consensus_odds, not the price you took. This measures the model. If
    you actually bet worse/better than consensus, real P/L differs — and
    that gap is exactly what consensus_at_bet / odds_taken would capture.
  * Pre-void-fix rows are still counted; historical hit rates run slightly
    high until the resolver reprocesses them.
  * n is tiny (13+ was ~11 picks in the whole log). A wide bootstrap CI
    that straddles 0 means 'not yet distinguishable from break-even,' not
    'profitable.'
""")


if __name__ == "__main__":
    main()
