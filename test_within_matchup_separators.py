#!/usr/bin/env python3
"""
test_within_matchup_separators.py

Version B analysis: WITHIN good matchups, what separates the legs that
homered from the ones that didn't? The model rated these matchups similarly
(good platoon+pitch), yet half homered and half didn't — so any metric that
still separates HR from non-HR HERE is a hidden variable the tier grouping
doesn't capture. That's a candidate refinement.

Runs on two groups:
  1. Great tier: combined platoon+pitch in [3, 4] — your sweet spot
  2. Broader: hr_score >= 10 AND combined >= 1 (any positive matchup)

For each metric, within the group: HR-hitter avg vs non-HR avg, the gap, and
a standardized effect size (gap / pooled std) so metrics on different scales
are comparable. Sorted by |effect|. Big effects = hidden signal.

IMPORTANT: metrics already baked into the score (iso, hr_per_fb, barrel) will
still show up — the point is to find ones that separate MORE than expected
INSIDE a controlled-matchup group, especially ones NOT already weighted.

USAGE
    python3 test_within_matchup_separators.py            # live
    python3 test_within_matchup_separators.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

# metrics to scan — broad net, including ones NOT in the score
METRICS = [
    "iso", "hr_per_pa", "hr_per_fb", "pull_rate", "fb_rate",
    "season_barrel_pct", "barrel_pct_5d", "barrel_pct_7d", "barrel_pct_10d",
    "barrel_pct_14d", "avg_ev_7d", "avg_ev_5d", "avg_ev_10d", "avg_ev_30d",
    "avg_la_7d", "avg_la_season", "hard_hit_pct_7d", "hard_hit_pct_season",
    "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hr9",
    "park_hr_factor", "hr_weather_boost", "temp_f",
    "top_pitch_iso_vs_hand", "woba", "obp", "bb_pct", "k_pct",
    "ld_pct", "gb_pct", "fb_pct", "pa_14d", "avg_bat_order",
]


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
    df["ps"] = df.get("platoon_score", "").apply(sf).fillna(0)
    df["pm"] = df.get("pitch_matchup_score", "").apply(sf).fillna(0)
    df["combo"] = df["ps"].fillna(0) + df["pm"].fillna(0)
    return df


def separate(group, name):
    n = len(group)
    if n < 40:
        print(f"\n{name}: only {n} resolved legs — too thin to trust, skipping.")
        return
    hr = group[group["hit"] == 1]
    no = group[group["hit"] == 0]
    print(f"\n{'='*70}\n{name}  (n={n}, HR {len(hr)}, non-HR {len(no)}, "
          f"rate {len(hr)/n*100:.1f}%)\n{'='*70}")
    print(f"{'metric':<24}{'HR avg':>10}{'non-HR':>10}{'gap':>9}{'effect':>9}")
    print("-" * 70)
    results = []
    for m in METRICS:
        if m not in group.columns:
            continue
        yv = pd.to_numeric(hr[m], errors="coerce").dropna()
        nv = pd.to_numeric(no[m], errors="coerce").dropna()
        if len(yv) < 15 or len(nv) < 15:
            continue
        ya, na_ = yv.mean(), nv.mean()
        gap = ya - na_
        pooled = pd.concat([yv, nv]).std()
        eff = gap / pooled if pooled and pooled > 0 else 0.0
        results.append((m, ya, na_, gap, eff))
    results.sort(key=lambda x: -abs(x[4]))
    for m, ya, na_, gap, eff in results:
        flag = ""
        if abs(eff) >= 0.25:
            flag = "  <-- STRONG"
        elif abs(eff) >= 0.15:
            flag = "  <- notable"
        print(f"{m:<24}{ya:>10.3f}{na_:>10.3f}{gap:>+9.3f}{eff:>+9.2f}{flag}")
    print("\n  effect = gap / pooled_std (standardized). |effect|>=0.25 is a")
    print("  real separator; >=0.15 worth watching. Ones NOT already in the")
    print("  score (pull_rate, fb_rate, k_pct, avg_bat_order, woba...) that")
    print("  show STRONG here are candidate NEW features.")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved legs.")
    great = df[(df["combo"] >= 3) & (df["combo"] < 4)]
    separate(great, "GREAT TIER (combined +3 to +4)")
    broad = df[(df["score"] >= 10) & (df["combo"] >= 1)]
    separate(broad, "BROAD (score>=10, combined>=1)")
    print(f"\n{'='*70}\nREAD: focus on metrics that are STRONG separators here but")
    print("AREN'T already in hr_score/platoon/pitch. Those are the hidden")
    print("variables worth testing as additions. Ones already in the score")
    print("(iso, barrel, hr_per_fb) separating is expected, not new signal.")
    print("In-sample; confirm any candidate on fresh data before building.")


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
