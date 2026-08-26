#!/usr/bin/env python3
"""
test_barrel_negative.py

The barrel window functions (5d/7d/10d) currently floor at 0.0 — a guy with
2% recent barrel rate gets the same score as "insufficient sample." But raw
barrel_pct_7d separates HR outcomes strongly (+5.0 spread, +0.196 eff), and
the stale-power test showed 3.2pp difference between fresh vs cold elite
hitters. The fix: let barrel windows go NEGATIVE for genuinely cold recent
production. This tests whether that actually improves separation.

Compares three versions of the barrel 7d scoring:
  A) CURRENT: 0.0 floor (cold and absent are identical)
  B) NEGATIVE: low barrel with sufficient sample subtracts points
  C) SYMMETRIC: full symmetric scoring (same thresholds, mirrored negative)

Same for 5d and 10d windows. Effect size and spread compared head to head.

USAGE
    python3 test_barrel_negative.py            # live
    python3 test_barrel_negative.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"
MIN_BBE_PARTIAL = 5
MIN_BBE_FULL = 20
LEAGUE_AVG_BARREL = 8.0


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, n):
    return df[n].astype(str) if n in df.columns else pd.Series([""] * len(df), index=df.index)


def regress(value, league_avg, sample, full_sample):
    if pd.isna(value) or sample <= 0:
        return league_avg
    w = min(sample / full_sample, 1.0)
    return value * w + league_avg * (1 - w)


# ── THREE VERSIONS OF BARREL 7d SCORING ──────────────────────────────────

def barrel_7d_current(v, bbe):
    """Current: floors at 0.0, cold = absent = zero."""
    if pd.isna(bbe) or bbe < MIN_BBE_PARTIAL:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL, bbe, MIN_BBE_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    return 0.0


def barrel_7d_negative(v, bbe):
    """Negative: same positive thresholds, but low barrel with enough sample
    actively subtracts. Absent sample still returns 0.0 (can't penalize what
    you can't measure)."""
    if pd.isna(bbe) or bbe < MIN_BBE_PARTIAL:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL, bbe, MIN_BBE_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    # NEW: negative returns for genuinely cold recent barrel
    if v < 2:   return -1.0
    if v < 4:   return -0.5
    return 0.0


def barrel_7d_symmetric(v, bbe):
    """Symmetric: mirrors the positive thresholds on the negative side."""
    if pd.isna(bbe) or bbe < MIN_BBE_PARTIAL:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL, bbe, MIN_BBE_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    if v < 2:   return -1.2
    if v < 4:   return -0.5
    if v < 6:   return -0.2
    return 0.0


def separation(name, sig, hit):
    d = pd.DataFrame({"s": sig, "hit": hit}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200:
        print(f"  {name:<28} n={len(d):<6} too few"); return
    hi = d["s"].quantile(0.75); lo = d["s"].quantile(0.25)
    top = d[d["s"] >= hi]["hit"].mean() * 100
    bot = d[d["s"] <= lo]["hit"].mean() * 100
    pooled = d["s"].std()
    eff = (d[d["hit"]==1]["s"].mean() - d[d["hit"]==0]["s"].mean()) / pooled if pooled > 0 else 0
    neg_pct = (d["s"] < -0.01).mean() * 100
    print(f"  {name:<28} n={len(d):<6} topQ {top:5.1f}%  botQ {bot:5.1f}%  "
          f"spread {top-bot:+5.1f}  eff {eff:+.3f}  neg {neg_pct:.0f}%")


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["date"] = _col(df, "date").str.strip()
    df = df[df["date"] >= MODEL_START]
    for c in ["barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d",
              "bbe_7d", "bbe_5d", "bbe_10d", "season_barrel_pct", "pa"]:
        df[c] = _col(df, c).apply(sf)
    return df


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} model-era resolved rows, base {df['hit'].mean()*100:.1f}%")

    has = df[df["bbe_7d"] >= MIN_BBE_PARTIAL].copy()
    print(f"With bbe_7d >= {MIN_BBE_PARTIAL}: {len(has)} rows")

    print(f"\n{'='*78}\nBARREL 7d SCORING: CURRENT vs NEGATIVE vs SYMMETRIC\n{'='*78}")
    print("  (neg% = % of rows that get a negative score under that version)\n")

    has["cur"] = has.apply(lambda r: barrel_7d_current(r["barrel_pct_7d"], r["bbe_7d"]), axis=1)
    has["neg"] = has.apply(lambda r: barrel_7d_negative(r["barrel_pct_7d"], r["bbe_7d"]), axis=1)
    has["sym"] = has.apply(lambda r: barrel_7d_symmetric(r["barrel_pct_7d"], r["bbe_7d"]), axis=1)

    separation("barrel_7d CURRENT (0 floor)", has["cur"], has["hit"])
    separation("barrel_7d NEGATIVE", has["neg"], has["hit"])
    separation("barrel_7d SYMMETRIC", has["sym"], has["hit"])

    # Same comparison but within HIGH-SCORE players (the Buxton case)
    print(f"\n{'='*78}\nSAME TEST — HIGH-SCORE PLAYERS ONLY (top 20%)\n{'='*78}\n")
    for c in ("hr_score_recalc", "hr_score_corrected", "hr_score"):
        if c in has.columns:
            has["sc"] = _col(has, c).apply(sf); break
    sc_cut = has["sc"].quantile(0.80)
    elite = has[has["sc"] >= sc_cut].copy()
    print(f"  High-score pool (score >= {sc_cut:.1f}): {len(elite)} rows\n")

    if len(elite) >= 200:
        separation("barrel_7d CURRENT (elite)", elite["cur"], elite["hit"])
        separation("barrel_7d NEGATIVE (elite)", elite["neg"], elite["hit"])
        separation("barrel_7d SYMMETRIC (elite)", elite["sym"], elite["hit"])

    # How many players would get penalized and by how much
    penalized = has[has["neg"] < 0]
    print(f"\n{'='*78}\nIMPACT: who gets penalized under NEGATIVE version\n{'='*78}")
    print(f"  {len(penalized)} of {len(has)} rows ({len(penalized)/len(has)*100:.1f}%) would receive a penalty")
    if len(penalized) > 0:
        print(f"  avg penalty: {penalized['neg'].mean():.2f}")
        print(f"  their HR rate: {penalized['hit'].mean()*100:.1f}% (vs {has['hit'].mean()*100:.1f}% overall)")
        print(f"  -> penalty group hits {'LESS' if penalized['hit'].mean() < has['hit'].mean() else 'MORE'} "
              f"than average ({'good' if penalized['hit'].mean() < has['hit'].mean() else 'BAD'} — "
              f"penalty is {'justified' if penalized['hit'].mean() < has['hit'].mean() else 'NOT justified'})")

    print(f"""
{'='*78}
READ
{'='*78}
  Compare spread/eff across the 3 versions. If NEGATIVE or SYMMETRIC beats
  CURRENT, the floor-at-zero was hiding real information and letting cold
  players score too high. The IMPACT section confirms whether the guys who'd
  get penalized actually deserve it (hit less than average = justified).
  If NEGATIVE barely changes anything, cold barrel rates aren't informative
  enough to justify the added complexity.""")


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
