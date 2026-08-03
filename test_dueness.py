#!/usr/bin/env python3
"""
test_dueness.py

Does HR "dueness" predict anything, or is it the gambler's fallacy? A hitter
in a drought (few/no recent HRs) despite a good season rate is "due" — but
that could mean (1) unlucky, regress up (bet him), (2) nothing, independent
events (fallacy), or (3) actually declining (fade him). This checks which.

Proxy for dueness (no exact "PA since last HR" in the data):
  expected recent HRs  = hr_per_pa * pa_14d   (roughly what he "should" have)
  actual recent HRs    = hr_14d
  drought gap          = expected - actual    (positive = "owed" HRs = due)

Buckets resolved picks by that gap and checks the ACTUAL next-game HR rate.
If "due" hitters homer MORE, dueness is real; if SAME, it's the fallacy; if
LESS, the drought signals decline.

Also controls: within score>=10 (so it's not just bad hitters).

USAGE
    python3 test_dueness.py            # live
    python3 test_dueness.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd


def sf(v, d=np.nan):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return d


def _col(df, name):
    """Safely get a column as floats, or NaN series if the column is absent."""
    if name in df.columns:
        return df[name].apply(sf)
    return pd.Series([np.nan] * len(df), index=df.index)


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower() \
        if "hit_hr" in df.columns else ""
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["score"] = _col(df, "hr_score")
    df["hrpa"] = _col(df, "hr_per_pa")
    df["pa14"] = _col(df, "pa_14d")
    df["hr14"] = _col(df, "hr_14d")
    # report which dueness inputs are actually present
    present = {c: (c in df.columns) for c in ["hr_per_pa", "pa_14d", "hr_14d",
                                              "hr_7d", "hr_30d"]}
    print("  dueness columns present:", {k: v for k, v in present.items()})
    return df


def analyze(df, name):
    d = df.dropna(subset=["hrpa", "pa14", "hr14"])
    d = d[(d["hrpa"] > 0) & (d["pa14"] >= 10)]   # need a real rate + real recent sample
    n = len(d)
    if n < 60:
        print(f"\n{name}: only {n} usable rows — too thin.")
        return
    base = d["hit"].mean() * 100
    # expected recent HRs vs actual -> drought gap
    d = d.copy()
    d["expected"] = d["hrpa"] * d["pa14"]
    d["gap"] = d["expected"] - d["hr14"]     # positive = owed HRs = "due"
    print(f"\n{'='*62}\n{name}  (n={n}, base HR {base:.1f}%)\n{'='*62}")
    print(f"{'dueness bucket':<28}{'n':>6}{'HR':>5}{'hit%':>8}{'lift':>8}")
    print("-" * 62)
    buckets = [
        ("very due (gap>=1.5)",   lambda x: x["gap"] >= 1.5),
        ("due (0.7-1.5)",         lambda x: (x["gap"] >= 0.7) & (x["gap"] < 1.5)),
        ("neutral (-0.3..0.7)",   lambda x: (x["gap"] > -0.3) & (x["gap"] < 0.7)),
        ("over-performing (<=-0.3)", lambda x: x["gap"] <= -0.3),
    ]
    for label, fn in buckets:
        b = d[fn(d)]
        m = len(b)
        if m < 15:
            print(f"{label:<28}{m:>6}{'--':>5}{'--':>8}{'--':>8}")
            continue
        r = b["hit"].mean() * 100
        flag = "  <<" if abs(r - base) >= 3 and m >= 30 else ""
        print(f"{label:<28}{m:>6}{int(b['hit'].sum()):>5}{r:>7.1f}%{r-base:>+7.1f}{flag}")
    c = np.corrcoef(d["gap"], d["hit"])[0, 1] if d["gap"].std() > 0 else 0
    print(f"\n  corr(dueness gap, HR) = {c:+.4f}")
    print("  positive = due hitters homer MORE (dueness real / regression);")
    print("  ~zero = the gambler's fallacy (drought means nothing);")
    print("  negative = drought signals decline (fade, don't back).")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved legs.")
    analyze(df, "FULL POPULATION")
    analyze(df[df["score"] >= 10], "WITHIN score>=10 (real bats)")
    print(f"""
{'='*62}
VERDICT GUIDE
{'='*62}
  Almost certainly you'll see ~zero correlation and flat buckets — that's
  the gambler's fallacy confirmed, and dueness should NOT be a signal.
  If 'very due' hitters genuinely homer several pp MORE with real n, that
  would be a surprising regression effect worth a display column (not a
  score input). In-sample; treat any positive result with heavy suspicion
  since 'due' is the most over-narrativized idea in betting.""")


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
