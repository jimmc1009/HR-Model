#!/usr/bin/env python3
"""
test_pitcher_barrel.py

Sizes the pitcher-barrel penalty against reality. Buckets resolved picks by
the pitcher's barrel% allowed to the BATTER'S hand (the exact input the
platoon score uses) and shows the HR rate in each bucket. If low-barrel
pitchers (<=5%) genuinely suppress HRs a lot, the penalty should be steep;
if the gap is mild, the current penalty is about right.

Restricted to hr_score >= 10 (the tier you bet), and also shown for the full
population so selection bias is visible.

For each barrel bucket:
    n, HRs, hit rate, lift vs baseline

Plus the implied "fair" penalty: how many pts each bucket is below the top,
so you can calibrate the (barrel - 8) * SLOPE formula.

USAGE
    python3 test_pitcher_barrel.py            # live sheet
    python3 test_pitcher_barrel.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd


def sf(v, d=np.nan):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return d


def eff_hand(bh, ph):
    if bh == "S":
        return "R" if ph == "L" else "L"
    return bh


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["score"] = df.get("hr_score", "").apply(sf)
    bh = df.get("batter_hand", "").astype(str).str.strip().str.upper().str[:1]
    ph = df.get("pitcher_hand", "").astype(str).str.strip().str.upper().str[:1]
    eff = [eff_hand(b, p) for b, p in zip(bh, ph)]
    lhh = df.get("pitcher_barrel_vs_lhh", "").apply(sf)
    rhh = df.get("pitcher_barrel_vs_rhh", "").apply(sf)
    df["pbar"] = [l if e == "L" else r for e, l, r in zip(eff, lhh, rhh)]
    return df


BUCKETS = [
    ("<=3%  (elite supp)", lambda s: s["pbar"] <= 3),
    ("3-5%  (strong supp)", lambda s: (s["pbar"] > 3) & (s["pbar"] <= 5)),
    ("5-7%  (good)",        lambda s: (s["pbar"] > 5) & (s["pbar"] <= 7)),
    ("7-9%  (average)",     lambda s: (s["pbar"] > 7) & (s["pbar"] <= 9)),
    ("9-11% (vulnerable)",  lambda s: (s["pbar"] > 9) & (s["pbar"] <= 11)),
    ("11-14% (weak)",       lambda s: (s["pbar"] > 11) & (s["pbar"] <= 14)),
    (">14%  (very weak)",   lambda s: s["pbar"] > 14),
]


def report(df, title):
    df = df.dropna(subset=["pbar"])
    df = df[df["pbar"] > 0]
    n_all = len(df)
    if n_all == 0:
        print(f"\n{title}: no rows with pitcher barrel data.")
        return
    base = df["hit"].mean() * 100
    print(f"\n{'='*66}\n{title}  (n={n_all}, baseline {base:.1f}%)\n{'='*66}")
    print(f"{'barrel bucket':<22}{'n':>6}{'HR':>5}{'hit%':>8}{'lift':>8}{'vs 8% ref':>11}")
    print("-" * 66)
    ref_rate = None
    stats = []
    for label, fn in BUCKETS:
        sub = df[fn(df)]
        n = len(sub)
        if n == 0:
            print(f"{label:<22}{0:>6}{'--':>5}{'--':>8}{'--':>8}{'--':>11}")
            stats.append((label, 0, None))
            continue
        r = sub["hit"].mean() * 100
        stats.append((label, n, r))
        if "7-9" in label:
            ref_rate = r
        print(f"{label:<22}{n:>6}{int(sub['hit'].sum()):>5}{r:>7.1f}%{r-base:>+7.1f}", end="")
        print()
    # calibration: pp difference from the ~8% reference bucket
    if ref_rate is not None:
        print(f"\n  vs the 7-9% (~average) reference at {ref_rate:.1f}%:")
        for label, n, r in stats:
            if n >= 15 and r is not None:
                print(f"    {label:<22} {r-ref_rate:+.1f}pp")
        print("\n  This is the REAL suppression curve. The score's penalty for a")
        print("  low-barrel pitcher should roughly match how far below the")
        print("  reference that bucket actually hits — not more (overfit), not")
        print("  less (your concern). Weight n>=30 buckets most.")


def main(src):
    df = load(src)
    report(df[df["score"] >= 10], "HR score >= 10")
    report(df, "FULL population (all scores)")
    print("""
NOTE: if low-barrel buckets sit only mildly below reference, the current
penalty (~ (barrel-8)*0.35) is about right. If they're WAY below (e.g.
-6pp+), the penalty is too soft and the slope should steepen. In-sample.""")


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    v = gspread.authorize(cr).open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(v[1:], columns=v[0])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print("Loaded HR_All_Scores live.\n")
        main(_sheet())
