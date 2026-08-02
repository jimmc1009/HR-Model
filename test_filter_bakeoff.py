#!/usr/bin/env python3
"""
test_filter_bakeoff.py

Which filter should pick RR/card legs: the COMBINED platoon+pitch gate
(current: combined >= 3) or a SCORE x ODDS tier gate (the better-sampled
separator)? Head-to-head on resolved data:

  For each filter -> the legs it selects, their actual HR hit rate, the n,
  and the average legs-per-day it yields (fill rate — a filter that hits
  well but finds 2 legs/slate can't fill a 6-leg card).

Filters compared:
  A. COMBINED  : combined(plat+pitch) >= 3, hr_score >= 10, odds <= 499
  B. SCOREODDS : the strong score x odds cells (<=+499 & score 13-15, plus
                 <=+300 & score 11-13) — your best-hitting historical cells
  C. BLEND-ISH : combined >= 1 AND score >= 11 AND odds <= 499 (looser hybrid)

Also reports each filter's lift over the base rate and how often it yields
>=6 legs on a given date (can it fill the card).

USAGE
    python3 test_filter_bakeoff.py            # live
    python3 test_filter_bakeoff.py file.csv
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
    df["odds"] = df.get("consensus_odds", "").apply(sf)
    df["ps"] = df.get("platoon_score", "").apply(sf).fillna(0)
    df["pm"] = df.get("pitch_matchup_score", "").apply(sf).fillna(0)
    df["combo"] = df["ps"].fillna(0) + df["pm"].fillna(0)
    df["date_s"] = df.get("date", "").astype(str).str.strip()
    return df


FILTERS = {
    "A COMBINED (combo>=3, sc>=10, <=499)":
        lambda d: (d["combo"] >= 3) & (d["score"] >= 10) & (d["odds"] > 0) & (d["odds"] <= 499),
    "B SCORExODDS (strong cells)":
        lambda d: (d["odds"] > 0) & (
            ((d["odds"] <= 499) & (d["score"] >= 13) & (d["score"] < 16)) |
            ((d["odds"] <= 300) & (d["score"] >= 11) & (d["score"] < 13))
        ),
    "C HYBRID (combo>=1, sc>=11, <=499)":
        lambda d: (d["combo"] >= 1) & (d["score"] >= 11) & (d["odds"] > 0) & (d["odds"] <= 499),
}


def main(src):
    df = load(src)
    base = df["hit"].mean() * 100
    print(f"Loaded {len(df)} resolved legs, base HR {base:.1f}%\n")
    print(f"{'filter':<40}{'legs':>7}{'HR':>6}{'hit%':>8}{'lift':>8}{'legs/day':>10}{'fill6%':>8}")
    print("-" * 87)

    ndays = df["date_s"].nunique() or 1
    for name, fn in FILTERS.items():
        sub = df[fn(df)]
        n = len(sub)
        if n == 0:
            print(f"{name:<40}{0:>7}")
            continue
        r = sub["hit"].mean() * 100
        per_day = n / ndays
        # how often a date yields >=6 legs (can fill the card)
        by_day = sub.groupby("date_s").size()
        fill6 = (by_day >= 6).mean() * 100 if len(by_day) else 0
        print(f"{name:<40}{n:>7}{int(sub['hit'].sum()):>6}{r:>7.1f}%{r-base:>+7.1f}{per_day:>10.1f}{fill6:>7.0f}%")

    print(f"""
READ ({ndays} days of data):
  hit%  = do the legs this filter picks actually homer more?
  lift  = vs the {base:.1f}% base rate.
  legs/day + fill6% = can it actually FILL a 6-leg card most nights?

  The best filter has BOTH a high hit% AND fills the card. A filter that
  hits 20% but only fills 6 legs 10% of nights is useless for the card;
  one that hits 15% and fills every night may be the practical winner.
  Pick on the combination, not hit% alone. In-sample.""")


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
