#!/usr/bin/env python3
"""
test_wind_categories.py

Tests the wind-strength buckets on picks with hr_score >= 10 (the tier you
actually bet). The dashboard's wind table is on ALL scored players; this
restricts to 10+ so you see whether wind separates HRs among viable bets,
not the whole population.

Buckets (same as HR_Analysis):
  Strong OUT (boost >= 2.0)
  Moderate OUT (1.0 to 2.0)
  Mild OUT (0.5 to 1.0)
  Neutral (-0.5 to 0.5)
  Mild IN (-1.5 to -0.5)
  Strong IN (boost <= -1.5)
  Roof / Dome (boost == 0.0)   [only rows explicitly domed]

For each: n, HRs, hit rate, and LIFT vs the 10+ baseline. Also prints a
correlation of boost vs HR within 10+.

USAGE
    python3 test_wind_categories.py            # live sheet
    python3 test_wind_categories.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

HR_FLOOR = 10.0


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
    # prefer explicit weather boost column; fall back to hr_weather_boost
    boost = df.get("weather_boost", pd.Series([""]*len(df))).apply(sf)
    boost2 = df.get("hr_weather_boost", pd.Series([""]*len(df))).apply(sf)
    df["boost"] = boost.where(~boost.isna(), boost2)
    df["is_dome"] = df.get("wind_context", "").astype(str).str.contains("dome|roof", case=False, na=False)
    return df


def main(src):
    df = load(src)
    df = df[df["score"] >= HR_FLOOR].dropna(subset=["boost"])
    n_all = len(df)
    base = df["hit"].mean() * 100
    print(f"HR score >= {HR_FLOOR}: {n_all} resolved picks, baseline HR {base:.1f}%\n")

    buckets = [
        ("Strong OUT (>= 2.0)",   lambda d: d["boost"] >= 2.0),
        ("Moderate OUT (1.0-2.0)", lambda d: (d["boost"] >= 1.0) & (d["boost"] < 2.0)),
        ("Mild OUT (0.5-1.0)",    lambda d: (d["boost"] >= 0.5) & (d["boost"] < 1.0)),
        ("Neutral (-0.5 to 0.5)", lambda d: (d["boost"] > -0.5) & (d["boost"] < 0.5) & ~d["is_dome"]),
        ("Mild IN (-1.5 to -0.5)", lambda d: (d["boost"] > -1.5) & (d["boost"] <= -0.5)),
        ("Strong IN (<= -1.5)",   lambda d: d["boost"] <= -1.5),
        ("Roof / Dome",           lambda d: d["is_dome"]),
    ]

    print(f"{'bucket':<24}{'n':>6}{'HR':>6}{'hit%':>8}{'lift':>8}")
    print("-" * 52)
    for label, fn in buckets:
        sub = df[fn(df)]
        n = len(sub)
        if n == 0:
            print(f"{label:<24}{0:>6}{'--':>6}{'--':>8}{'--':>8}")
            continue
        h = int(sub["hit"].sum())
        r = h / n * 100
        lift = r - base
        flag = "  <<" if abs(lift) >= 3 and n >= 30 else ""
        print(f"{label:<24}{n:>6}{h:>6}{r:>7.1f}%{lift:>+7.1f}{flag}")

    # correlation of boost with HR among 10+
    d2 = df.dropna(subset=["boost"])
    if len(d2) > 30 and d2["boost"].std() > 0:
        c = np.corrcoef(d2["boost"], d2["hit"])[0, 1]
        print(f"\nboost vs HR correlation (score>={HR_FLOOR}): {c:+.4f}")
        print("(positive = more wind-out -> more HRs among viable bets)")

    print(f"""
READ: '<<' flags buckets that move the rate >=3pp off baseline on n>=30.
Strong OUT should sit ABOVE baseline and Strong IN BELOW if wind matters
for the picks you actually bet. Small buckets (low n) are noise — weight
the big ones. This is in-sample on the current log.""")


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
