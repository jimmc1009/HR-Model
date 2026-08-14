#!/usr/bin/env python3
"""
test_platoon_split.py

Before swapping the platoon score's ISO split for barrel/HR-rate, TEST which
split actually separates homers from non-homers on resolved data. Avoids
rewiring + rescoring on a hunch.

For each resolved pick, compute the platoon GAP (batter's rate vs THIS pitcher
hand minus vs the OPPOSITE hand) under four definitions:
    ISO split       (current)
    barrel% split
    HR-rate split
    blend (barrel + HR-rate, z-scored)
Then measure separation: mean gap for HR vs non-HR, effect size, and hit rate
in the top vs bottom gap quartile. Whichever split most cleanly sorts homers
is the one to use.

USAGE
    python3 test_platoon_split.py            # live
    python3 test_platoon_split.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"
MIN_BBE = 15
LEAGUE_ISO = 0.155
LEAGUE_BARREL = 8.0     # ~league avg barrel%
LEAGUE_HRRATE = 3.5     # ~league avg HR per 100 PA-ish (units as stored)


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, name):
    return df[name].astype(str) if name in df.columns else pd.Series([""] * len(df), index=df.index)


def regress(value, league, sample, full):
    if np.isnan(value) or sample <= 0:
        return league
    w = min(sample / full, 1.0)
    return value * w + league * (1 - w)


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["date"] = _col(df, "date").str.strip()
    df = df[df["date"] >= MODEL_START]
    df["phand"] = _col(df, "pitcher_hand").str.strip().str.upper()
    for c in ["vs_lhp_iso","vs_rhp_iso","vs_lhp_barrel_pct","vs_rhp_barrel_pct",
              "vs_lhp_hr_rate","vs_rhp_hr_rate","vs_lhp_bbe","vs_rhp_bbe"]:
        df[c] = _col(df, c).apply(sf)
    return df


def gap(df, this_col, opp_col, league, full=25):
    """this = vs the pitcher's hand, opp = vs opposite. Regressed gap."""
    def rowgap(r):
        if r["phand"] == "L":
            this_v, opp_v = r[f"vs_lhp_{this_col}"], r[f"vs_rhp_{opp_col}"]
            this_bbe, opp_bbe = r["vs_lhp_bbe"], r["vs_rhp_bbe"]
        elif r["phand"] == "R":
            this_v, opp_v = r[f"vs_rhp_{this_col}"], r[f"vs_lhp_{opp_col}"]
            this_bbe, opp_bbe = r["vs_rhp_bbe"], r["vs_lhp_bbe"]
        else:
            return np.nan
        if np.isnan(this_v) or np.isnan(opp_v) or min(this_bbe, opp_bbe) < MIN_BBE:
            return np.nan
        return regress(this_v, league, this_bbe, full) - regress(opp_v, league, opp_bbe, full)
    return df.apply(rowgap, axis=1)


def separation(name, g, hit):
    d = pd.DataFrame({"g": g, "hit": hit}).dropna()
    if len(d) < 100:
        print(f"  {name:<18} n={len(d):<5} too few"); return
    hr = d[d["hit"] == 1]["g"]; no = d[d["hit"] == 0]["g"]
    pooled = d["g"].std()
    eff = (hr.mean() - no.mean()) / pooled if pooled > 0 else 0
    # top vs bottom quartile hit rate
    q_hi = d["g"].quantile(0.75); q_lo = d["g"].quantile(0.25)
    top = d[d["g"] >= q_hi]["hit"].mean() * 100
    bot = d[d["g"] <= q_lo]["hit"].mean() * 100
    print(f"  {name:<18} n={len(d):<5} HR-gap {hr.mean():+.3f} vs {no.mean():+.3f}  "
          f"eff {eff:+.3f}  topQ {top:.1f}% botQ {bot:.1f}%  spread {top-bot:+.1f}")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved rows, base {df['hit'].mean()*100:.1f}%")
    print(f"\n{'='*76}\nPLATOON SPLIT SEPARATION — which gap sorts HRs best?\n{'='*76}")
    print("  (eff = effect size; spread = topQ hit% - botQ hit%; bigger = better)")

    iso_g = gap(df, "iso", "iso", LEAGUE_ISO, 25)
    bar_g = gap(df, "barrel_pct", "barrel_pct", LEAGUE_BARREL, 25)
    hr_g  = gap(df, "hr_rate", "hr_rate", LEAGUE_HRRATE, 25)

    # z-score barrel & hr gaps, then blend 50/50
    def z(s):
        s2 = s.dropna()
        m, sd = s2.mean(), s2.std()
        return (s - m) / sd if sd > 0 else s * 0
    blend_g = 0.5 * z(bar_g) + 0.5 * z(hr_g)

    print()
    separation("ISO split (now)", iso_g, df["hit"])
    separation("barrel% split", bar_g, df["hit"])
    separation("HR-rate split", hr_g, df["hit"])
    separation("blend bar+hr", blend_g, df["hit"])

    print(f"""
{'='*76}
READ
{'='*76}
  Biggest 'spread' and 'eff' = the split that best separates homers. If barrel
  or the blend clearly beats ISO, swap the platoon input to it. If ISO is
  within noise of the others, the doubles concern didn't matter much and it's
  not worth the rescore. Coverage (n) matters too — HR-rate splits may have
  fewer usable rows if that column is sparse.""")


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    v = gspread.authorize(cr).open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    print("Loaded HR_All_Scores live.")
    return pd.DataFrame(v[1:], columns=v[0])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main(_sheet())
