#!/usr/bin/env python3
"""
test_platoon_unfiltered.py

The HR_Analysis platoon table showed disadvantage picks homering MORE than
advantage picks (13.7% vs 12.6%) — an inversion. Prime suspect: selection
bias. Only elite bats survive to the pick list *despite* a platoon
disadvantage, so that bucket is stacked with sluggers.

This checks platoon on the FULL HR_All_Scores population (every scored
player, before any pick-list filtering) where that bias is far weaker.
If platoon is real, advantage should out-homer disadvantage HERE.

Buckets three ways so we see it cleanly:
  1. by the platoon_desc TEXT label (same as the analysis tab)
  2. by the numeric platoon_score sign (advantage>0, neutral~0, disadvantage<0)
  3. by score tier — to show whether any platoon effect survives conditioning
     on hr_score (the "already baked in" test)

USAGE
    python3 test_platoon_unfiltered.py            # live sheet
    python3 test_platoon_unfiltered.py file.csv
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
    # prefer corrected platoon score if present
    ps = df.get("platoon_score_corrected", pd.Series([""]*len(df))).apply(sf)
    ps2 = df.get("platoon_score", pd.Series([""]*len(df))).apply(sf)
    df["pscore"] = ps.where(~ps.isna(), ps2)
    df["pmatch"] = df.get("platoon_matchup", df.get("platoon_desc", "")).astype(str)
    return df


def rate(sub):
    n = len(sub)
    return (n, int(sub["hit"].sum()), sub["hit"].mean()*100 if n else float("nan"))


def main(src):
    df = load(src)
    print(f"FULL population: {len(df)} resolved rows, base HR {df['hit'].mean()*100:.1f}%")

    # 1) by TEXT label (same as analysis tab, but unfiltered)
    print("\n" + "="*58)
    print("1) BY TEXT LABEL (platoon_matchup) — full population")
    print("="*58)
    adv = df[df["pmatch"].str.contains("advantage", case=False, na=False)]
    dis = df[df["pmatch"].str.contains("disadvantage|weakness", case=False, na=False)]
    neu = df[~df.index.isin(adv.index) & ~df.index.isin(dis.index)]
    for lab, sub in [("Advantage", adv), ("Neutral", neu), ("Disadvantage", dis)]:
        n, h, r = rate(sub)
        print(f"  {lab:<14} {r:5.1f}%   (n={n}, hr={h})")

    # 2) by NUMERIC platoon score sign
    print("\n" + "="*58)
    print("2) BY NUMERIC platoon_score SIGN — full population")
    print("="*58)
    pos = df[df["pscore"] > 0.3]
    neg = df[df["pscore"] < -0.3]
    mid = df[(df["pscore"] >= -0.3) & (df["pscore"] <= 0.3)]
    for lab, sub in [("score > +0.3 (adv)", pos), ("mid", mid), ("score < -0.3 (dis)", neg)]:
        n, h, r = rate(sub)
        print(f"  {lab:<20} {r:5.1f}%   (n={n}, hr={h})")
    # correlation
    m = df.dropna(subset=["pscore"])
    if len(m) > 30 and m["pscore"].std() > 0:
        c = np.corrcoef(m["pscore"], m["hit"])[0, 1]
        print(f"\n  platoon_score vs HR correlation: {c:+.4f}")
        print("  (positive = higher platoon score -> more HRs, as it should be)")

    # 3) does platoon survive INSIDE a score tier? (baked-in test)
    print("\n" + "="*58)
    print("3) PLATOON EFFECT WITHIN SCORE TIERS (is it already baked in?)")
    print("="*58)
    for lo, hi, lab in [(13, 999, "13+"), (11, 13, "11-13"), (8.5, 11, "8.5-11")]:
        t = df[(df["score"] >= lo) & (df["score"] < hi)]
        if len(t) < 80:
            continue
        a = t[t["pscore"] > 0.3]["hit"].mean()*100 if (t["pscore"] > 0.3).sum() >= 20 else float("nan")
        d = t[t["pscore"] < -0.3]["hit"].mean()*100 if (t["pscore"] < -0.3).sum() >= 20 else float("nan")
        print(f"  {lab:<7} adv {a:5.1f}%  vs  dis {d:5.1f}%   (gap {a-d:+.1f}pp)"
              if not (np.isnan(a) or np.isnan(d))
              else f"  {lab:<7} (too few adv/dis in tier)")

    print("\n" + "="*58)
    print("READ")
    print("="*58)
    print("""  If section 1/2 show Advantage > Disadvantage on the FULL population
  but the pick-list analysis showed the reverse, the inversion was
  SELECTION BIAS (elite bats survive to the list despite a disadvantage),
  not a broken platoon score.

  Section 3: if the adv-vs-dis gap SHRINKS as score rises, platoon is
  already absorbed into hr_score among good picks — real, just upstream.

  If Advantage is STILL below Disadvantage even here, platoon is genuinely
  inverted and needs fixing.""")


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