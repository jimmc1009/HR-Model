#!/usr/bin/env python3
"""
test_pocket_and_inversion.py

Two linked questions:

PART A — THE 9-10 POCKET. In the monotonic dump, 9-10 HR score punched above
tiers 10-14 (21.2% at +301-499, 25.7% at <=+300). What separates the 9-10
hitters who homer from those who don't? If there's a clean separator, that's a
real market blind spot to bet.

PART B — WHY THE VALUE AXIS INVERTS. The score-minus-price "edge" was inverted:
model's +value picks hit 9%, model's -value picks hit 21.5%. Is it just
longshot bias (the +value bucket = longshots the model loves and the market
correctly hates)? Decompose the inversion by odds to see if it's purely a
price-longshot artifact, or something in the score itself.

USAGE
    python3 test_pocket_and_inversion.py            # live
    python3 test_pocket_and_inversion.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, name):
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series([""] * len(df), index=df.index)


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    def score_of(r):
        c = str(r.get("hr_score_corrected", "")).strip()
        return sf(c) if c not in ("", "nan", "None") else sf(r.get("hr_score", ""))
    df["score"] = df.apply(score_of, axis=1)
    df["odds"] = _col(df, "consensus_odds").apply(sf)
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df["date"] = _col(df, "date").str.strip()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df = df.dropna(subset=["score", "odds"])
    df = df[(df["odds"] > 0) & (df["date"] >= MODEL_START)]
    return df


def implied(od):
    return 100/(od+100) if od > 0 else abs(od)/(abs(od)+100)


# candidate separators (whatever's in the sheet)
SEPARATORS = [
    "season_barrel_pct", "barrel_pct_7d", "barrel_pct_10d", "iso", "hr_per_fb",
    "hr_per_pa", "hard_hit_pct_season", "avg_ev_7d", "avg_la_season",
    "pull_rate", "fb_rate", "platoon_score", "pitch_matchup_score",
    "pitcher_barrel_pct", "pitcher_hr_per_fb", "park_hr_factor",
]


def part_a_pocket(df):
    print(f"\n{'='*76}\nPART A — WHAT SEPARATES THE 9-10 POCKET THAT HITS\n{'='*76}")
    pocket = df[(df["score"] >= 9) & (df["score"] < 10)]
    n = len(pocket)
    if n < 50:
        print(f"  only {n} rows in 9-10 — too thin.")
        return
    hr = pocket[pocket["hit"] == 1]
    no = pocket[pocket["hit"] == 0]
    print(f"  9-10 pocket: n={n}, hit {len(hr)} ({len(hr)/n*100:.1f}%), avg odds +{pocket['odds'].mean():.0f}")
    print(f"\n  {'separator':<24}{'HR avg':>10}{'no-HR avg':>11}{'gap':>9}{'effect':>9}")
    print("  " + "-"*62)
    results = []
    for m in SEPARATORS:
        if m not in pocket.columns:
            continue
        yv = pd.to_numeric(hr[m].apply(sf), errors="coerce").dropna()
        nv = pd.to_numeric(no[m].apply(sf), errors="coerce").dropna()
        if len(yv) < 10 or len(nv) < 10:
            continue
        ya, na_ = yv.mean(), nv.mean()
        pooled = pd.concat([yv, nv]).std()
        eff = (ya - na_) / pooled if pooled and pooled > 0 else 0
        results.append((m, ya, na_, ya - na_, eff))
    results.sort(key=lambda x: -abs(x[4]))
    for m, ya, na_, gap, eff in results:
        flag = "  <-- STRONG" if abs(eff) >= 0.3 else "  <- notable" if abs(eff) >= 0.18 else ""
        print(f"  {m:<24}{ya:>10.2f}{na_:>11.2f}{gap:>+9.2f}{eff:>+9.2f}{flag}")
    print("\n  A STRONG separator here = a filter that turns the 9-10 pocket into a")
    print("  real bet ('9-10 AND high [X]'). This is the one place the model may")
    print("  beat the market, so a clean separator is gold.")
    # also: within the pocket, does odds still predict? (is it just cheap 9-10?)
    print(f"\n  9-10 by odds zone:")
    for zl, lo, hi in [("<=+300",0,300),("+301-499",301,499),("+500+",500,99999)]:
        z = pocket[(pocket["odds"]>=lo)&(pocket["odds"]<=hi)]
        if len(z)>=15:
            print(f"    {zl:<10} n={len(z):<4} {z['hit'].mean()*100:.1f}%")


def part_b_inversion(df):
    print(f"\n{'='*76}\nPART B — WHY THE VALUE AXIS INVERTS (longshot bias?)\n{'='*76}")
    # model rate = pooled score-band rate; gap = model - implied
    bands = [(15,99,"15+"),(14,15,"14-15"),(13,14,"13-14"),(12,13,"12-13"),
             (11,12,"11-12"),(10,11,"10-11"),(9,10,"9-10"),(8.5,9,"8.5-9")]
    br = {}
    for lo,hi,lab in bands:
        s = df[(df["score"]>=lo)&(df["score"]<hi)]
        if len(s)>=20: br[lab]=s["hit"].mean()
    def band(sc):
        for lo,hi,lab in bands:
            if lo<=sc<hi: return lab
        return None
    d = df.copy()
    d["mrate"] = d["score"].apply(lambda s: br.get(band(s), np.nan))
    d["impl"] = d["odds"].apply(implied)
    d = d.dropna(subset=["mrate"])
    d["gap"] = d["mrate"] - d["impl"]
    d["valbucket"] = pd.cut(d["gap"], [-9,-0.06,-0.02,0.02,0.06,9],
                            labels=["strong -val","-val","fair","+val","strong +val"])

    # KEY TEST: is the inversion just because +val = longshots? Control for odds.
    print("  Value bucket x odds zone — does +val still underperform WITHIN an")
    print("  odds zone? If the inversion vanishes when odds is held fixed, it was")
    print("  pure longshot bias. If +val still loses within a zone, the score is")
    print("  actively anti-predictive.\n")
    print(f"  {'odds zone':<12}{'val bucket':<14}{'n':>6}{'hit%':>8}")
    print("  " + "-"*40)
    for zl, lo, hi in [("<=+300",0,300),("+301-499",301,499),("+500-699",500,699),("+700+",700,99999)]:
        z = d[(d["odds"]>=lo)&(d["odds"]<=hi)]
        if len(z) < 40:
            continue
        for vb in ["strong +val","+val","fair","-val","strong -val"]:
            b = z[z["valbucket"]==vb]
            if len(b) >= 15:
                print(f"  {zl:<12}{vb:<14}{len(b):>6}{b['hit'].mean()*100:>7.1f}%")
        print()
    # correlation of gap with hit, controlling for odds via partial-ish view
    print("  corr(gap, HR) raw:        ", f"{np.corrcoef(d['gap'],d['hit'])[0,1]:+.4f}")
    print("  corr(odds, HR) raw:       ", f"{np.corrcoef(d['odds'],d['hit'])[0,1]:+.4f}")
    print("  (if gap corr is ~0 once odds is fixed above, inversion = longshot bias)")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} model-era rows, base {df['hit'].mean()*100:.1f}%")
    part_a_pocket(df)
    part_b_inversion(df)
    print(f"""
{'='*76}
READ
{'='*76}
  PART A: a STRONG separator in the 9-10 pocket = a real filter to bet.
  PART B: if +val underperforms even WITHIN each odds zone, your score is
  anti-predictive and you bet WITH the price, full stop. If the inversion
  only exists across odds (vanishes within a zone), it's longshot bias and
  the fix is: never let the model push you onto a longer price than the
  market — cap plays to short/mid odds where the score isn't chasing.""")


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    cr=Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]),scopes=sc)
    v=gspread.authorize(cr).open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    print("Loaded HR_All_Scores live.")
    return pd.DataFrame(v[1:],columns=v[0])


if __name__ == "__main__":
    if len(sys.argv)>=2: main(sys.argv[1])
    else: main(_sheet())
