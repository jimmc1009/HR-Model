#!/usr/bin/env python3
"""
test_monotonic.py

The score tier hit rate is NOT monotonic (15+ dips below 14-15; 9-10 punches
above 12-13). Hypothesis: the score is roughly monotonic in ABILITY, but the
ODDS ZONE is a confound — books overprice elite scores and underprice mid
scores, so raw hit-rate-by-score is distorted by where the market puts each
guy. This finds the gap and the fix.

Three analyses:
 1. WITHIN-ODDS SCORE SWEEP — hold odds zone fixed, sweep score. If the score
    is real, hit rate should climb with score INSIDE a zone. Reveals where
    it's clean and where it breaks.
 2. 15+ SATURATION CHECK — within the 15+ tier, does more score => more HR, or
    is it flat/negative (score saturated, piling redundant points)?
 3. SCORE-vs-PRICE RESIDUAL — the real edge axis. gap = score_implied_rate -
    odds_implied_rate. Bucket by gap, check if HR rate is monotonic in GAP
    (it should be — that's betting the mispricing, not the score).

USAGE
    python3 test_monotonic.py            # live
    python3 test_monotonic.py file.csv
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


SCORE_BANDS = [("15+",15,99),("14-15",14,15),("13-14",13,14),("12-13",12,13),
               ("11-12",11,12),("10-11",10,11),("9-10",9,10),("8.5-9",8.5,9)]
ODDS_ZONES = [("<=+300",0,300),("+301-499",301,499),("+500-699",500,699),("+700+",700,99999)]


def implied(od):
    return 100/(od+100) if od > 0 else abs(od)/(abs(od)+100)


def rate(sub):
    n=len(sub); return (sub["hit"].mean()*100, n) if n else (np.nan,0)


def within_odds_sweep(df):
    print(f"\n{'='*78}\n1. WITHIN-ODDS SCORE SWEEP — is score monotonic INSIDE each odds zone?\n{'='*78}")
    for zl, zlo, zhi in ODDS_ZONES:
        z = df[(df["odds"]>=zlo)&(df["odds"]<=zhi)]
        if len(z) < 40:
            continue
        print(f"\n  {zl}  (n={len(z)}, base {z['hit'].mean()*100:.1f}%)")
        prev=None; flags=[]
        for sl, lo, hi in SCORE_BANDS:
            s = z[(z["score"]>=lo)&(z["score"]<hi)]
            r,n = rate(s)
            if n < 15:
                print(f"    {sl:<8} n={n:<4} --"); continue
            arrow = ""
            if prev is not None:
                arrow = " ^" if r>prev else " v" if r<prev else " ="
            print(f"    {sl:<8} n={n:<4} {r:5.1f}%{arrow}")
            prev=r
    print("\n  -> where you see a clean top-to-bottom descent, the score works.")
    print("     where it zigzags, that zone's pricing is scrambling the signal.")


def saturation_15plus(df):
    print(f"\n{'='*78}\n2. 15+ SATURATION — within 15+, does MORE score => more HR?\n{'='*78}")
    hi = df[df["score"]>=15]
    if len(hi) < 40:
        print("  too few 15+ rows"); return
    # split 15+ into sub-bands
    for lo,hh,lab in [(15,16,"15-16"),(16,17,"16-17"),(17,18,"17-18"),(18,99,"18+")]:
        s = hi[(hi["score"]>=lo)&(hi["score"]<hh)]
        r,n = rate(s)
        if n>=10:
            print(f"    {lab:<7} n={n:<4} {r:5.1f}%   avg odds +{s['odds'].mean():.0f}")
    c = np.corrcoef(hi["score"], hi["hit"])[0,1] if hi["score"].std()>0 else 0
    print(f"\n  corr(score, HR) WITHIN 15+ = {c:+.4f}")
    print("  near-zero/negative => score SATURATES above 15 (piling redundant")
    print("  points, no added HR signal). That's why 15+ dips: extra score isn't")
    print("  real edge, but the book still prices it shorter. Cap the score there.")


def price_residual(df):
    print(f"\n{'='*78}\n3. SCORE-vs-PRICE RESIDUAL — the real edge axis\n{'='*78}")
    # score-implied rate: use the WITHIN-zone score-band rate as the model's
    # estimate for each guy (leave-in, in-sample proxy), then gap vs odds.
    d = df.copy()
    # assign each row its score-band avg hit rate (model's implied) computed
    # pooled across all odds (so it's a pure score estimate)
    band_rate = {}
    for sl,lo,hi in SCORE_BANDS:
        s = d[(d["score"]>=lo)&(d["score"]<hi)]
        if len(s)>=20: band_rate[sl]=s["hit"].mean()
    def band_of(sc):
        for sl,lo,hi in SCORE_BANDS:
            if lo<=sc<hi: return sl
        return None
    d["mrate"] = d["score"].apply(lambda s: band_rate.get(band_of(s), np.nan))
    d["impl"] = d["odds"].apply(implied)
    d = d.dropna(subset=["mrate"])
    d["gap"] = d["mrate"] - d["impl"]   # + = model rate beats price = value
    print(f"{'gap bucket':<20}{'n':>7}{'hit%':>8}{'avg odds':>10}")
    print("-"*50)
    buckets = [("strong +val >+.06",0.06,9),("+val +.02..06",0.02,0.06),
               ("fair -.02..02",-0.02,0.02),("-val -.06..-02",-0.06,-0.02),
               ("strong -val <-.06",-9,-0.06)]
    prev=None; mono=True
    for lab,lo,hi in buckets:
        b = d[(d["gap"]>=lo)&(d["gap"]<hi)] if lo!=-9 else d[d["gap"]<hi]
        if lab.startswith("strong +"): b = d[d["gap"]>=lo]
        r,n = rate(b)
        if n<15:
            print(f"{lab:<20}{n:>7}   --"); continue
        arrow=""
        if prev is not None:
            arrow = " ^" if r>prev else " v"
            if r>prev: mono=False   # expect DESC as value drops
        print(f"{lab:<20}{n:>7}{r:>7.1f}%{'  +'+format(b['odds'].mean(),'.0f'):>10}{arrow}")
        prev=r
    print("\n  This SHOULD be monotonic: more value (score rate > price) => higher")
    print("  HR rate. If it is, bet by GAP not by score — that's your clean tier.")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} model-era rows, base HR {df['hit'].mean()*100:.1f}%")
    within_odds_sweep(df)
    saturation_15plus(df)
    price_residual(df)
    print(f"""
{'='*78}
THE PLAY
{'='*78}
  If within-odds sweeps are cleaner than raw score => odds is the confound;
  bet within a zone, not across. If 15+ corr is ~0 => cap score at ~15, the
  elite dip is saturation the book overpays for. If gap-buckets are monotonic
  => the true tier is SCORE-MINUS-PRICE, and that's the axis to bet with
  confidence. Sift these three and you have your monotonic edge.""")


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
