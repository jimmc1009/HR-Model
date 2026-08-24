#!/usr/bin/env python3
"""
test_pitcher_trends.py

Does a pitcher's RECENT form predict HRs beyond his season line? The score
uses season pitcher stats; this checks whether pitchers trending badly right
now (recent barrel allowed, recent HRs allowed, declining velo) give up more
homers. Tested on the full population AND within good matchups (score>=10) to
see if it adds signal the matchup grading misses.

Signals:
  pitcher_barrel_7d  — barrel% allowed last 7d (higher = getting hit)
  pitcher_hr_7d      — HRs allowed last 7d (higher = getting hit)
  pitcher_velo_trend — velocity trend (negative = declining, often precedes
                       getting hit; a more STABLE signal than 3-game results)

For each, buckets HR rate by the signal and reports lift vs baseline, plus a
correlation. Big monotonic movement = real; flat = noise (leave it).

USAGE
    python3 test_pitcher_trends.py            # live
    python3 test_pitcher_trends.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd


def sf(v, d=np.nan):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return d


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["score"] = df.get("hr_score", "").apply(sf)
    df["pb7"] = df.get("pitcher_barrel_7d", "").apply(sf)
    df["ph7"] = df.get("pitcher_hr_7d", "").apply(sf)
    df["pvt"] = df.get("pitcher_velo_trend", "").apply(sf)
    return df


def bucket_report(df, col, edges, labels, name, base):
    sub = df.dropna(subset=[col])
    if len(sub) < 60:
        print(f"\n  {name}: only {len(sub)} rows with data — too thin.")
        return
    print(f"\n  {name}  (n={len(sub)})")
    print(f"  {'bucket':<22}{'n':>6}{'HR':>5}{'hit%':>8}{'lift':>8}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        b = sub[(sub[col] >= lo) & (sub[col] < hi)]
        n = len(b)
        if n < 15:
            continue
        r = b["hit"].mean() * 100
        flag = "  <<" if abs(r - base) >= 3 and n >= 30 else ""
        print(f"  {labels[i]:<22}{n:>6}{int(b['hit'].sum()):>5}{r:>7.1f}%{r-base:>+7.1f}{flag}")
    c = np.corrcoef(sub[col], sub["hit"])[0, 1] if sub[col].std() > 0 else 0
    print(f"  corr({col}, HR) = {c:+.4f}")


def run(df, title):
    n = len(df)
    if n < 60:
        print(f"\n{title}: {n} rows — too thin.")
        return
    base = df["hit"].mean() * 100
    print(f"\n{'='*60}\n{title}  (n={n}, baseline {base:.1f}%)\n{'='*60}")
    bucket_report(df, "pb7", [-1, 5, 8, 11, 999],
                  ["low <5%", "5-8%", "8-11%", "high >11%"],
                  "pitcher barrel% allowed 7d", base)
    bucket_report(df, "ph7", [-1, 0.5, 1.5, 2.5, 999],
                  ["0 HR", "1 HR", "2 HR", "3+ HR"],
                  "pitcher HRs allowed 7d", base)
    bucket_report(df, "pvt", [-999, -1.0, -0.3, 0.3, 999],
                  ["declining <-1", "-1 to -0.3", "steady", "rising >0.3"],
                  "pitcher velo trend", base)


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved legs.")
    run(df, "FULL POPULATION")
    run(df[df["score"] >= 10], "WITHIN GOOD MATCHUPS (score>=10)")
    print(f"""
{'='*60}
READ
{'='*60}
  A signal is REAL if HR rate rises monotonically as the pitcher looks
  worse (more recent barrels/HRs, declining velo) AND the correlation is
  the right sign, in BOTH groups. Velo_trend is the most stable — a real
  velo decline is a fact, not a 3-game fluke, so trust it more than raw
  hr_7d (which is tiny-sample and noisy). Flat/inconsistent = leave it,
  it's the recency mirage. In-sample; confirm before building.""")


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
