#!/usr/bin/env python3
"""
test_platoon_diagnose.py

Platoon is inverted (disadvantage homers MORE than advantage across the full
population). This pins down WHERE. Two parts:

PART A — raw components vs HR. For each ingredient, correlate it with actual
HR on the full population. The one(s) with the WRONG sign are the culprit:
  - batter ISO vs the hand faced   (should be +)
  - batter HR-rate vs the hand faced (should be +)
  - batter barrel% vs the hand faced (should be +)
  - pitcher barrel% allowed vs hand  (should be +)
  - pitcher HR9 allowed vs hand      (should be +)
  - the derived platoon_score        (should be +)

PART B — live vs corrected. Compare platoon_score against
platoon_score_corrected (the column the analysis overlays). If corrected is
stale/wrong-signed, the analysis buckets on bad values while the live score
may be fine.

USAGE
    python3 test_platoon_diagnose.py            # live sheet
    python3 test_platoon_diagnose.py file.csv
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
    df["bh"] = df.get("batter_hand", "").astype(str).str.strip().str.upper().str[:1]
    df["ph"] = df.get("pitcher_hand", "").astype(str).str.strip().str.upper().str[:1]
    return df


def corr(a, hit):
    a = pd.to_numeric(a, errors="coerce")
    m = a.notna()
    if m.sum() < 50 or a[m].std() == 0:
        return None, int(m.sum())
    return np.corrcoef(a[m], hit[m])[0, 1], int(m.sum())


def col(df, name):
    return df.get(name, pd.Series([""]*len(df))).apply(sf)


def main(src):
    df = load(src)
    print(f"FULL population: {len(df)} rows, base HR {df['hit'].mean()*100:.1f}%")
    faceL = df["ph"].values == "L"
    eff = np.where(df["bh"].values == "S", np.where(faceL, "R", "L"), df["bh"].values)

    # batter stats vs the hand FACED today
    b_iso = np.where(faceL, col(df, "vs_lhp_iso"), col(df, "vs_rhp_iso"))
    b_hr = np.where(faceL, col(df, "vs_lhp_hr_rate"), col(df, "vs_rhp_hr_rate"))
    b_bar = np.where(faceL, col(df, "vs_lhp_barrel_pct"), col(df, "vs_rhp_barrel_pct"))
    # pitcher stats vs the batter's effective hand
    p_bar = np.where(eff == "L", col(df, "pitcher_barrel_vs_lhh"), col(df, "pitcher_barrel_vs_rhh"))
    p_hr9 = np.where(eff == "L", col(df, "pitcher_vs_lhh_hr9"), col(df, "pitcher_vs_rhh_hr9"))

    print("\n" + "="*62)
    print("PART A — raw component vs HR correlation (should ALL be positive)")
    print("="*62)
    items = [
        ("batter ISO vs hand faced",     pd.Series(b_iso)),
        ("batter HR-rate vs hand faced", pd.Series(b_hr)),
        ("batter barrel% vs hand faced", pd.Series(b_bar)),
        ("pitcher barrel% vs hand",      pd.Series(p_bar)),
        ("pitcher HR9 vs hand",          pd.Series(p_hr9)),
        ("derived platoon_score",        col(df, "platoon_score")),
        ("platoon_score_corrected",      col(df, "platoon_score_corrected")),
    ]
    hit = df["hit"].reset_index(drop=True)
    for name, series in items:
        c, n = corr(series.reset_index(drop=True), hit)
        if c is None:
            print(f"  {name:<30}   (n<50 or constant, n={n})")
        else:
            flag = "  <-- WRONG SIGN" if c < -0.01 else ("  ok" if c > 0.01 else "  ~flat")
            print(f"  {name:<30} {c:+.4f}  (n={n}){flag}")

    print("\n" + "="*62)
    print("PART B — live platoon_score vs platoon_score_corrected")
    print("="*62)
    ls = col(df, "platoon_score")
    cs = col(df, "platoon_score_corrected")
    both = ls.notna() & cs.notna()
    if both.sum() < 30:
        print(f"  Not enough overlap (n={int(both.sum())}). corrected may be absent/blank.")
    else:
        agree = np.corrcoef(ls[both], cs[both])[0, 1]
        print(f"  overlap n={int(both.sum())}")
        print(f"  correlation between the two: {agree:+.4f}")
        print(f"  live mean {ls[both].mean():+.3f}   corrected mean {cs[both].mean():+.3f}")
        if agree < 0:
            print("  -> They are ANTI-correlated: corrected is sign-flipped vs live!")
        elif agree < 0.8:
            print("  -> They diverge: corrected is stale or computed differently.")
        else:
            print("  -> They largely agree.")
        # which one aligns with HR?
        cl, _ = corr(ls[both], hit[both])
        cc, _ = corr(cs[both], hit[both])
        print(f"\n  live platoon_score vs HR:      {cl:+.4f}")
        print(f"  corrected platoon_score vs HR: {cc:+.4f}")
        print("  -> whichever is POSITIVE is the one to trust/keep.")

    print("\n" + "="*62)
    print("READ: any component with WRONG SIGN in Part A is the inversion.")
    print("If only platoon_score_corrected is negative but live is positive,")
    print("the analysis is bucketing on a bad column — fix or drop corrected.")
    print("="*62)


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
