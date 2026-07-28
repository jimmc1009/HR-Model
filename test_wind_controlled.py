#!/usr/bin/env python3
"""
test_wind_controlled.py

Answers: is the Strong-OUT wind boost a REAL wind effect, or just good
hitters / bandboxes homering anyway? Controls two ways:

1. WITHIN-HITTER — for batters who appear in BOTH wind-out and wind-in games,
   compare the SAME hitter's HR rate out vs in. If the same guy homers more
   with wind out, it's the wind, not his quality.

2. WITHIN-PARK — same idea holding the park constant: does wind-out beat
   wind-in at the SAME park? Separates wind from bandbox.

3. REGRESSION — HR ~ wind_out + wind_in, controlling for hr_score (hitter
   quality proxy) and park_hr_factor. If wind keeps a real coefficient with
   those held constant, the effect is genuine.

Restricted to hr_score >= 11 (the tier you bet).

USAGE
    python3 test_wind_controlled.py            # live sheet
    python3 test_wind_controlled.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

FLOOR = 11.0


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
    boost = df.get("weather_boost", pd.Series([""]*len(df))).apply(sf)
    boost2 = df.get("hr_weather_boost", pd.Series([""]*len(df))).apply(sf)
    df["boost"] = boost.where(~boost.isna(), boost2)
    df["park"] = df.get("park_hr_factor", "").apply(sf)
    df["batter"] = df.get("player_name", "").astype(str).str.strip()
    df = df[df["score"] >= FLOOR].dropna(subset=["boost"])
    df["wind_out"] = (df["boost"] >= 1.0).astype(int)   # meaningful out
    df["wind_in"]  = (df["boost"] <= -1.0).astype(int)  # meaningful in
    df["wind_cat"] = np.where(df["wind_out"] == 1, "OUT",
                       np.where(df["wind_in"] == 1, "IN", "NEU"))
    return df


def within_group(df, key, kname):
    """For entities appearing in BOTH out and in games, compare their own
    out vs in HR rate. Aggregate the paired differences."""
    out = df[df["wind_cat"] == "OUT"].groupby(key)["hit"].agg(["mean", "size"])
    inn = df[df["wind_cat"] == "IN"].groupby(key)["hit"].agg(["mean", "size"])
    both = out.join(inn, lsuffix="_out", rsuffix="_in", how="inner")
    both = both[(both["size_out"] >= 2) & (both["size_in"] >= 2)]
    print(f"\n{'='*58}\nWITHIN-{kname} (same {kname.lower()} in both out & in games)\n{'='*58}")
    if both.empty:
        print(f"  No {kname.lower()}s appear in both wind-out and wind-in with 2+ each.")
        print("  (Sample too small to control this way yet.)")
        return
    both["diff"] = (both["mean_out"] - both["mean_in"]) * 100
    n = len(both)
    avg_diff = both["diff"].mean()
    pos = (both["diff"] > 0).sum()
    print(f"  {n} {kname.lower()}s qualify")
    print(f"  their OWN HR rate:  wind-out {both['mean_out'].mean()*100:.1f}%  "
          f"vs wind-in {both['mean_in'].mean()*100:.1f}%")
    print(f"  avg within-{kname.lower()} diff (out - in): {avg_diff:+.1f}pp")
    print(f"  {pos}/{n} homered more with wind out")
    if avg_diff > 3 and pos > n * 0.55:
        print(f"  -> holds within {kname.lower()}: wind looks REAL, not just quality.")
    elif abs(avg_diff) <= 3:
        print(f"  -> effect SHRINKS when holding {kname.lower()} constant: "
              f"was largely {kname.lower()} quality, not wind.")
    else:
        print(f"  -> mixed / small sample; inconclusive.")


def regression(df):
    print(f"\n{'='*58}\nREGRESSION — wind with hitter-quality & park held constant\n{'='*58}")
    d = df.dropna(subset=["park"]).copy()
    if len(d) < 60:
        print("  too few rows for regression.")
        return
    # simple OLS via numpy: hit ~ 1 + wind_out + wind_in + score + park
    X = np.column_stack([
        np.ones(len(d)), d["wind_out"], d["wind_in"],
        (d["score"] - d["score"].mean()),
        (d["park"].fillna(d["park"].mean()) - d["park"].mean()),
    ])
    y = d["hit"].values.astype(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    names = ["intercept", "wind_OUT", "wind_IN", "hr_score", "park_factor"]
    print("  coefficient (effect on HR probability, all else equal):")
    for nm, b in zip(names, beta):
        tag = ""
        if nm == "wind_OUT":
            tag = "  <- wind-out's OWN effect after controlling for hitter+park"
        if nm == "wind_IN":
            tag = "  <- wind-in's OWN effect"
        print(f"    {nm:<12} {b*100:+.1f}pp{tag}")
    print("\n  If wind_OUT stays clearly positive and wind_IN negative here,")
    print("  the effect survives controlling for hitter quality and park —")
    print("  i.e. it's really the wind. If they collapse toward 0, it wasn't.")


def main(src):
    df = load(src)
    print(f"HR score >= {FLOOR}: {len(df)} resolved picks")
    print(f"  wind-out (>=+1): {int(df['wind_out'].sum())}   "
          f"wind-in (<=-1): {int(df['wind_in'].sum())}   "
          f"neutral: {int((df['wind_cat']=='NEU').sum())}")
    within_group(df, "batter", "HITTER")
    within_group(df, "park", "PARK")
    regression(df)
    print(f"\n{'='*58}\nBOTTOM LINE\n{'='*58}")
    print("""  Trust the within-hitter result most — it's the cleanest control.
  If the same hitters homer more with wind out AND the regression keeps
  a positive wind_OUT coefficient with score+park held constant, the wind
  edge is real and worth weighting. If both shrink, you were seeing good
  hitters / bandboxes, not wind. Small samples early — revisit as data grows.""")


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
