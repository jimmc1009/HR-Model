#!/usr/bin/env python3
"""
build_hr_separators.py

Writes an HR_Separators tab that answers, for each date: what set the HR
hitters apart from the non-HR hitters? For every feature it shows the
HR-hitters' average vs the non-hitters' average, both PER-DAY and on a
ROLLING 14-day window.

Two truths this tab keeps honest:
  * PER-DAY is descriptive only. A single day has a handful of HRs, so daily
    gaps are mostly noise — read them as a diary, not a test. The 't' column
    is blank per-day for exactly this reason.
  * ROLLING 14d has enough sample to mean something. Its 't' (Welch) flags
    which features actually separate hitters over the window. This is where
    real signal shows (spoiler from prior analysis: batter-power separates;
    situational usually doesn't — but the tab lets you watch if that changes).

Runs over all history and, re-run daily, appends the new date each time.

OUTPUT COLUMNS (one row per date × feature):
    date, feature, family, n_hr, n_non,
    day_hr_avg, day_non_avg, day_gap,
    roll_hr_avg, roll_non_avg, roll_gap, roll_t, roll_sig

USAGE
    python3 build_hr_separators.py            # live sheet (reads + writes)
    python3 build_hr_separators.py file.csv   # local CSV, prints only
"""

import sys
import os
import json
import math
import pandas as pd
import numpy as np

ROLL_DAYS = 14

FEATURES = [
    # (column, label, family)
    ("hr_per_pa",             "HR/PA",            "power"),
    ("hr_per_fb",             "HR/FB",            "power"),
    ("season_barrel_pct",     "barrel% season",   "power"),
    ("barrel_pct_7d",         "barrel% 7d",       "power"),
    ("iso",                   "ISO",              "power"),
    ("avg_ev_7d",             "exit velo 7d",     "power"),
    ("pull_rate",             "pull rate",        "power"),
    ("platoon_score",         "platoon score",    "platoon"),
    ("top_pitch_iso_vs_hand", "top-pitch ISO vs hand", "platoon"),
    ("pitch_matchup_score",   "pitch matchup",    "matchup"),
    ("pitcher_barrel_pct",    "pitcher barrel%",  "pitcher"),
    ("pitcher_hr_per_fb",     "pitcher HR/FB",    "pitcher"),
    ("pitcher_hr9",           "pitcher HR/9",     "pitcher"),
    ("pitcher_babip_allowed", "pitcher BABIP",    "pitcher"),
    ("park_hr_factor",        "park HR factor",   "park"),
    ("hr_weather_boost",      "weather boost",    "weather"),
    ("temp_f",                "temperature",      "weather"),
    ("momentum_score",        "momentum",         "context"),
]


def welch_t(a, b):
    a = a.dropna(); b = b.dropna()
    if len(a) < 5 or len(b) < 5:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va/len(a) + vb/len(b))
    if se == 0:
        return None
    return (a.mean() - b.mean()) / se


def sig(t):
    if t is None:
        return ""
    a = abs(t)
    return "***" if a >= 2.58 else "**" if a >= 1.96 else "*" if a >= 1.64 else ""


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["date"] = pd.to_datetime(df.get("date", ""), errors="coerce")
    df = df.dropna(subset=["date"])
    present = [f for f in FEATURES if f[0] in df.columns]
    for col, _, _ in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, present


def build(df, present):
    dates = sorted(df["date"].dropna().unique())
    out = []
    for d in dates:
        day = df[df["date"] == d]
        win = df[(df["date"] > d - pd.Timedelta(days=ROLL_DAYS)) & (df["date"] <= d)]
        d_hr, d_no = day[day["hit"] == 1], day[day["hit"] == 0]
        w_hr, w_no = win[win["hit"] == 1], win[win["hit"] == 0]
        for col, label, fam in present:
            def avg(g):
                v = g[col].dropna()
                return round(v.mean(), 3) if len(v) else ""
            t = welch_t(w_hr[col], w_no[col])
            dhr, dno = avg(d_hr), avg(d_no)
            whr, wno = avg(w_hr), avg(w_no)
            out.append({
                "date": pd.Timestamp(d).strftime("%Y-%m-%d"),
                "feature": label, "family": fam,
                "n_hr": len(d_hr), "n_non": len(d_no),
                "day_hr_avg": dhr, "day_non_avg": dno,
                "day_gap": round(dhr - dno, 3) if dhr != "" and dno != "" else "",
                "roll_hr_avg": whr, "roll_non_avg": wno,
                "roll_gap": round(whr - wno, 3) if whr != "" and wno != "" else "",
                "roll_t": round(t, 2) if t is not None else "",
                "roll_sig": sig(t),
            })
    return pd.DataFrame(out)


def summarize(res):
    """Print which features most consistently separate on the rolling window."""
    if res.empty:
        print("No rows built.")
        return
    print(f"\nRolling {ROLL_DAYS}d separation — how often each feature was "
          f"significant (** or better):")
    rr = res[res["roll_t"] != ""].copy()
    rr["abst"] = rr["roll_t"].astype(float).abs()
    g = rr.groupby("feature").agg(
        days=("date", "nunique"),
        pct_sig=("abst", lambda s: (s >= 1.96).mean() * 100),
        avg_t=("abst", "mean"),
    ).sort_values("pct_sig", ascending=False)
    print(f"  {'feature':<24}{'days':>6}{'% sig':>8}{'avg|t|':>8}")
    for feat, r in g.iterrows():
        print(f"  {feat:<24}{int(r['days']):>6}{r['pct_sig']:>7.0f}%{r['avg_t']:>8.2f}")
    print("\n  Features high on '% sig' are the real separators over time.")
    print("  Prior analysis said batter-power dominates; watch whether any")
    print("  situational feature ever climbs this list.")


def write_tab(gc, sheet_id, res):
    import gspread
    sh = gc.open_by_key(sheet_id)
    header = list(res.columns)
    try:
        ws = sh.worksheet("HR_Separators")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="HR_Separators", rows=len(res)+10, cols=len(header))
    ws.update([header] + res.astype(str).values.tolist())
    print(f"HR_Separators: wrote {len(res)} rows "
          f"({res['date'].nunique()} dates × {res['feature'].nunique()} features).")


def main():
    if len(sys.argv) >= 2:
        df, present = load(sys.argv[1])
        res = build(df, present)
        print(res.head(40).to_string(index=False))
        summarize(res)
        return
    # live
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    vals = sh.worksheet("HR_All_Scores").get_all_values()
    df_raw = pd.DataFrame(vals[1:], columns=vals[0])
    print(f"Loaded HR_All_Scores: {len(df_raw)} rows")
    df, present = load(df_raw)
    print(f"Resolved rows: {len(df)}  |  features present: {len(present)}/{len(FEATURES)}")
    res = build(df, present)
    write_tab(gc, os.environ["GOOGLE_SHEET_ID"], res)
    summarize(res)


if __name__ == "__main__":
    main()
