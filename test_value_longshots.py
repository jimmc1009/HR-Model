#!/usr/bin/env python3
"""
test_value_longshots.py

Question: in the +400-600 odds range, is there ANY signal that separates the
longshots that homer from the ones that don't? If a signal reliably flags the
"underpriced" guys (model sees it, market hasn't), a value parlay off that
signal makes sense. If nothing separates, the zone is just a coin flip at bad
prices and the parlay is pure lottery.

Tests each candidate signal WITHIN the +400-600 band:
    HR score, park_hr_factor, momentum/form, pitcher vulnerability,
    combined (score+pitch+platoon), and a score+park+form composite.
For each: top-quartile vs bottom-quartile hit rate, and whether the top
quartile clears the zone's breakeven (~+500 => ~16.7%).

USAGE
    python3 test_value_longshots.py            # live
    python3 test_value_longshots.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"
LO, HI = 400, 600          # the target zone


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, n):
    return df[n].astype(str) if n in df.columns else pd.Series([""] * len(df), index=df.index)


def resolve_score(r):
    for c in ("hr_score_recalc", "hr_score_corrected", "hr_score"):
        v = str(r.get(c, "")).strip()
        if v not in ("", "nan", "None"):
            return sf(v)
    return np.nan


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["date"] = _col(df, "date").str.strip()
    df = df[df["date"] >= MODEL_START]
    df["odds"] = _col(df, "consensus_odds").apply(sf)
    df["score"] = df.apply(resolve_score, axis=1)
    for c in ["park_hr_factor", "momentum_score", "barrel_pct_7d", "season_barrel_pct",
              "pitcher_hr_per_fb", "pitcher_barrel_pct", "platoon_score", "pitch_matchup_score",
              "bbe_7d"]:
        df[c] = _col(df, c).apply(sf)
    return df


def sep(name, sub, sig):
    d = pd.DataFrame({"s": sig, "hit": sub["hit"]}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 80:
        print(f"  {name:<26} n={len(d):<5} too few")
        return
    hi_cut = d["s"].quantile(0.75)
    lo_cut = d["s"].quantile(0.25)
    top = d[d["s"] >= hi_cut]["hit"].mean() * 100
    bot = d[d["s"] <= lo_cut]["hit"].mean() * 100
    top_n = (d["s"] >= hi_cut).sum()
    be = 100 / (5.0 + 1) if False else 16.7   # ~+500 breakeven
    flag = "  <-- clears BE!" if top >= 16.7 else ""
    print(f"  {name:<26} n={len(d):<5} topQ {top:.1f}% (n{top_n})  botQ {bot:.1f}%  "
          f"spread {top-bot:+.1f}{flag}")


def main(src):
    df = load(src)
    z = df[(df["odds"] >= LO) & (df["odds"] <= HI)].copy()
    print(f"Loaded {len(df)} resolved rows; +{LO}-{HI} zone n={len(z)}, "
          f"base {z['hit'].mean()*100:.1f}% (breakeven ~16.7%)")
    print(f"\n{'='*78}\nSIGNAL SEPARATION WITHIN +{LO}-{HI} — anything flag the live longshots?\n{'='*78}")
    print("  topQ = hit% of the top-quartile by that signal; want it to clear ~16.7% BE\n")

    sep("HR score", z, z["score"])
    sep("park_hr_factor", z, z["park_hr_factor"])
    sep("momentum_score", z, z["momentum_score"])
    sep("barrel surge (7d-season)", z, z["barrel_pct_7d"] - z["season_barrel_pct"])
    sep("pitcher_hr_per_fb (vuln)", z, z["pitcher_hr_per_fb"])
    sep("pitcher_barrel_pct (vuln)", z, z["pitcher_barrel_pct"])
    sep("combined (plat+pitch)", z, z["platoon_score"] + z["pitch_matchup_score"])

    # composite: z-scored score + park + form
    def zc(s):
        s2 = s.replace([np.inf, -np.inf], np.nan).dropna()
        m, sd = s2.mean(), s2.std()
        return (s - m) / sd if sd > 0 else s * 0
    form = z["barrel_pct_7d"] - z["season_barrel_pct"]
    composite = zc(z["score"]) + zc(z["park_hr_factor"]) + zc(form)
    sep("score+park+form composite", z, composite)

    print(f"""
{'='*78}
READ
{'='*78}
  A signal whose topQ clears ~16.7% (the +500 breakeven) AND has a big spread
  is genuinely flagging underpriced longshots in this zone — build the value
  parlay off THAT. If nothing clears 16.7%, the +{LO}-{HI} zone has no findable
  edge and any parlay there is pure lottery (fine for fun, not for value).
  Small n on topQ = trust it less.""")


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
