#!/usr/bin/env python3
"""
test_progressive_squeeze.py

The conditioning sweep. Tighten the pick pool step by step and, at each step,
measure TWO kinds of separation for every feature:

  [MEAN]  batter-quality separation: HR-hitters' avg feature value vs
          non-HR-hitters' avg (Welch t). These are EXPECTED TO BURN OUT as
          the score climbs -- a high score already means high barrel%/ISO,
          so among top picks everyone has them and they stop distinguishing.
          Watching them fade is confirmation the score did its job.

  [RATE]  situational separation: HR RATE when the factor is present vs
          absent (two-proportion z), computed WITHIN the tightened pool.
          These are what you actually want -- a factor that still splits
          hitters from misses AMONG already-good picks is signal the score
          did NOT capture. If any exist, they should hold or rise as you
          tighten.

The "breakpoint" you're looking for is where these two families cross:
batter-quality goes flat, and situational either emerges or proves it never
did. That crossover is the finding.

TWO SWEEPS, side by side:
  A) score floor climbs:      >=8.5, >=10, >=11, >=12, >=13, >=14
  B) score+odds tighten:      toward your bet rule (13-15 @ <=+499 etc.)

At every step: effect size, n, and significance. As the pool shrinks n
collapses, so a rising point estimate can just be noise -- READ THE Z, not
the headline number. A +14pp at z=1.1 (n=180) is weaker than +6pp at z=4.4
(n=3000).

RAW p-values, no correction. Test count printed at the end.

USAGE
    python3 test_progressive_squeeze.py            # live sheet
    python3 test_progressive_squeeze.py file.csv   # local CSV
"""

import sys
import os
import json
import math
import pandas as pd
import numpy as np

TESTS = [0]

# Batter-quality features (expected to BURN OUT up the ladder)
QUALITY = [
    ("barrel_pct_7d",     "Barrel% 7d"),
    ("season_barrel_pct", "Barrel% season"),
    ("iso",               "ISO"),
    ("hr_per_fb",         "HR/FB%"),
    ("hr_per_pa",         "HR/PA%"),
    ("avg_ev_7d",         "Avg EV 7d"),
    ("avg_la_7d",         "Launch angle 7d"),
    ("pull_rate",         "Pull rate"),
]

# Situational features (what you WANT to emerge) -> (col, label, threshold)
SITU = [
    ("park_hr_factor",     "Park >=105",          105),
    ("park_hr_factor",     "Park >=110",          110),
    ("temp_f",             "Temp >=80F",          80),
    ("temp_f",             "Temp >=85F",          85),
    ("hr_weather_boost",   "Wind out (>0)",       0.01),
    ("hr_weather_boost",   "Strong wind (>=1.5)", 1.5),
    ("pitcher_barrel_pct", "Pitcher brl% >=9",    9),
    ("pitcher_barrel_pct", "Pitcher brl% >=12",   12),
    ("pitcher_hr_per_fb",  "Pitcher HR/FB >=13",  13),
    ("pitcher_hr9",        "Pitcher HR/9 >=1.3",  1.3),
]


def welch_t(a, b):
    a = a.dropna(); b = b.dropna()
    if len(a) < 8 or len(b) < 8:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va/len(a) + vb/len(b))
    if se == 0:
        return None
    return (a.mean() - b.mean()) / se, a.mean(), b.mean()


def two_prop_z(h1, n1, h2, n2):
    if n1 == 0 or n2 == 0:
        return 0.0
    p1, p2 = h1/n1, h2/n2
    p = (h1+h2)/(n1+n2)
    se = math.sqrt(p*(1-p)*(1/n1 + 1/n2))
    return (p1-p2)/se if se > 0 else 0.0


def sig(z):
    a = abs(z)
    return "***" if a >= 2.58 else "**" if a >= 1.96 else "*" if a >= 1.64 else " "


def step_report(pool, label):
    n = len(pool)
    hits = int(pool["hit"].sum())
    rate = hits/n*100 if n else 0
    print(f"\n{'-'*82}\nPOOL: {label}   n={n}  HR rate={rate:.1f}%")
    if n < 40:
        print("  (n<40 — too small to read; shown for continuity only)")
    hr = pool[pool["hit"] == 1]
    no = pool[pool["hit"] == 0]

    print(f"  [MEAN] batter-quality: HR avg vs non-HR avg (Welch t) — expect these to FADE")
    print(f"    {'feature':<22}{'HR avg':>10}{'nonHR':>10}{'t':>7}  sig")
    for col, lab in QUALITY:
        if col not in pool.columns:
            continue
        r = welch_t(hr[col], no[col])
        if r is None:
            continue
        TESTS[0] += 1
        t, ma, mb = r
        print(f"    {lab:<22}{ma:>10.3f}{mb:>10.3f}{t:>+7.2f}  {sig(t)}")

    print(f"  [RATE] situational: HR rate present vs absent (2-prop z) — WANT these to hold/rise")
    print(f"    {'factor':<22}{'present':>9}{'absent':>9}{'Δpp':>7}{'z':>7}  sig   n(pres/abs)")
    situ_rows = []
    for col, lab, thr in SITU:
        if col not in pool.columns or pool[col].isna().all():
            continue
        mask = pool[col] >= thr
        p, a = pool[mask], pool[~mask]
        if len(p) < 12 or len(a) < 12:
            continue
        TESTS[0] += 1
        z = two_prop_z(int(p["hit"].sum()), len(p), int(a["hit"].sum()), len(a))
        situ_rows.append((abs(z), lab, p["hit"].mean(), a["hit"].mean(), len(p), len(a), z))
    for _, lab, pr, ab, npres, nabs, z in sorted(situ_rows, reverse=True):
        print(f"    {lab:<22}{pr*100:>8.1f}%{ab*100:>8.1f}%{(pr-ab)*100:>+7.1f}{z:>+7.2f}  {sig(z)}   {npres}/{nabs}")


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["score"] = pd.to_numeric(df.get("hr_score", ""), errors="coerce")
    df["odds"] = pd.to_numeric(
        df.get("consensus_odds", "").astype(str).str.replace("+", "", regex=False).str.strip(),
        errors="coerce")
    df["res"] = df.get("hit_hr", "").str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    for col, _ in QUALITY:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col, _, _ in SITU:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["score"])


def _read_sheet():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        return None
    if "GOOGLE_SERVICE_ACCOUNT_JSON" not in os.environ or "GOOGLE_SHEET_ID" not in os.environ:
        return None
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    import gspread
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    vals = sh.worksheet("HR_All_Scores").get_all_values()
    if not vals or len(vals) < 2:
        return None
    return pd.DataFrame(vals[1:], columns=vals[0])


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved rows.  odds coverage: {(df['odds']>0).mean()*100:.0f}%")

    print("\n" + "#"*82)
    print("# SWEEP A — SCORE FLOOR CLIMBS (all odds)")
    print("#"*82)
    for floor in [8.5, 10, 11, 12, 13, 14]:
        step_report(df[df["score"] >= floor], f"score >= {floor}")

    print("\n\n" + "#"*82)
    print("# SWEEP B — SCORE + ODDS TIGHTEN TOGETHER (toward the bet rule)")
    print("#"*82)
    for floor, omax, lab in [
        (8.5, 9999, "score>=8.5, any odds"),
        (10,  699,  "score>=10, <=+699"),
        (11,  599,  "score>=11, <=+599"),
        (12,  499,  "score>=12, <=+499"),
        (13,  499,  "score>=13, <=+499"),
        (14,  400,  "score>=14, <=+400"),
    ]:
        sub = df[(df["score"] >= floor) & ((df["odds"] <= omax) | (df["odds"] <= 0))]
        step_report(sub, lab)

    print(f"\n{'#'*82}")
    print(f"# HOW TO READ THIS")
    print(f"{'#'*82}")
    print(f"""
  BURN-OUT (expected, healthy): [MEAN] t-stats for barrel/ISO/EV shrink toward
  0 as the floor climbs. That means the score already captured them — among
  top picks everyone has them, so they stop separating. Not a loss of signal.

  EMERGENCE (what you're hunting): a [RATE] situational factor whose z HOLDS
  or RISES as the pool tightens, and stays significant at the tight end. That
  is a factor adding signal the score missed — a real breakpoint.

  THE TRAP: as the pool shrinks n drops and point estimates swing wildly. A
  big Δpp at the tight end with a small z is NOISE, not emergence. Read the z
  and the n, never the Δpp alone.

  Tests run: {TESTS[0]}  ->  at p<.05 expect ~{TESTS[0]*0.05:.0f} false positives,
  at p<.01 ~{TESTS[0]*0.01:.1f}. A real find is *** that repeats down the sweep
  with a mechanism. Everything here is in-sample: confirm forward before betting.
""")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        d = _read_sheet()
        if d is None:
            print("No CSV arg and no sheet access (need gspread + GOOGLE_* env).")
            sys.exit(1)
        print("Loaded HR_All_Scores live from Google Sheets.\n")
        main(d)
