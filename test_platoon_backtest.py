#!/usr/bin/env python3
"""
test_platoon_backtest.py

Backtests the platoon FIX against the OLD buggy logic on existing resolved
history — no forward wait needed. Recomputes each row's platoon score both
ways and asks: did the fix change which picks looked good, and do the picks
it newly likes actually homer more?

THE BUG: pitcher barrel% allowed was keyed on the PITCHER's throwing hand
instead of the BATTER's effective hand. So for opposite-hand matchups the
model fetched the wrong pitcher-vs-hand barrel split.

Reports:
  * how many rows the pitcher-barrel term changed, and by how much
  * HR rate of picks the FIX upgraded (old term low -> new term high) vs
    downgraded — if the fix is real, upgraded picks homer more
  * correlation of each version's pitcher-barrel term with actual HR

USAGE
    python3 test_platoon_backtest.py            # live sheet
    python3 test_platoon_backtest.py file.csv
"""

import sys, os, json
import pandas as pd
import numpy as np


def sf(v, d=0.0):
    try:
        x = float(str(v).strip())
        return x if not np.isnan(x) else d
    except (ValueError, TypeError):
        return d


def pitcher_barrel_term(barrel):
    """The score contribution from pitcher barrel% (same thresholds as
    compute_platoon_score, both versions share this — only the INPUT differs)."""
    if barrel <= 0:
        return 0.0
    if barrel >= 14:  return 0.8
    if barrel >= 11:  return 0.4
    if barrel >= 9:   return 0.2
    if barrel <= 4:   return -0.6
    if barrel <= 6:   return -0.3
    return 0.0


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    for c in ["pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct"]:
        df[c] = df.get(c, pd.Series([""]*len(df))).apply(sf) if c in df.columns else 0.0
    df["batter_hand"] = df.get("batter_hand", "").astype(str).str.strip().str.upper()
    df["p_throws"] = df.get("pitcher_hand", "").astype(str).str.strip().str.upper()
    return df


def compute(df):
    lhh = df["pitcher_vs_lhh_barrel_pct"]
    rhh = df["pitcher_vs_rhh_barrel_pct"]

    # OLD (buggy): keyed on pitcher's throwing hand
    old_barrel = np.where(df["p_throws"] == "L", lhh,
                 np.where(df["p_throws"] == "R", rhh, 0.0))
    # NEW (fixed): keyed on batter's effective hand (switch flips vs p_throws)
    eff = np.where(df["batter_hand"] == "S",
                   np.where(df["p_throws"] == "L", "R", "L"),
                   df["batter_hand"])
    new_barrel = np.where(eff == "L", lhh, np.where(eff == "R", rhh, 0.0))

    df["old_term"] = [pitcher_barrel_term(b) for b in old_barrel]
    df["new_term"] = [pitcher_barrel_term(b) for b in new_barrel]
    df["changed"] = df["old_term"] != df["new_term"]
    return df


def main(src):
    df = compute(load(src))
    n = len(df)
    ch = df[df["changed"]]
    print(f"Resolved rows: {n}   base HR rate: {df['hit'].mean()*100:.1f}%")
    print(f"Pitcher-barrel term CHANGED on: {len(ch)} rows ({len(ch)/n*100:.0f}%)")
    print(f"  (these are the opposite-hand matchups the bug mis-scored)\n")

    # among changed rows: did the FIX point toward HRs better than the bug?
    up = df[df["new_term"] > df["old_term"]]   # fix raised the term
    dn = df[df["new_term"] < df["old_term"]]   # fix lowered it
    print("Did the fix's direction match actual HR outcomes?")
    print(f"  fix UPGRADED (new>old): {len(up):>4} rows, HR rate {up['hit'].mean()*100:5.1f}%")
    print(f"  fix DOWNGRADED (new<old):{len(dn):>4} rows, HR rate {dn['hit'].mean()*100:5.1f}%")
    print(f"  unchanged:               {n-len(up)-len(dn):>4} rows, "
          f"HR rate {df[~df['changed']]['hit'].mean()*100:5.1f}%")
    print("  -> if fix is real, UPGRADED should homer MORE than DOWNGRADED.\n")

    # correlation of each version's term with actual HR
    def corr(col):
        a, b = df[col].astype(float), df["hit"].astype(float)
        if a.std() == 0:
            return float("nan")
        return np.corrcoef(a, b)[0, 1]
    print(f"Correlation with actual HR:")
    print(f"  OLD term: {corr('old_term'):+.4f}")
    print(f"  NEW term: {corr('new_term'):+.4f}")
    print("  -> higher (more positive) = better aligned with real HRs.\n")

    # HR rate by term bucket, new version
    print("HR rate by NEW pitcher-barrel term (does a higher term mean more HRs?):")
    for t in sorted(df["new_term"].unique()):
        sub = df[df["new_term"] == t]
        if len(sub) >= 20:
            print(f"  term {t:+.1f}: {sub['hit'].mean()*100:5.1f}%  (n={len(sub)})")


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    gc = gspread.authorize(cr)
    v = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(v[1:], columns=v[0])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print("Loaded HR_All_Scores live.\n")
        main(_sheet())
