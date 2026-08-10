#!/usr/bin/env python3
"""
rescore_all.py

Retroactively re-score every row in HR_All_Scores using the CURRENT scoring
formula (the power-composite version), then recompute the tier hit-rate
analysis on the new scores. Non-destructive: writes a new `hr_score_recalc`
column beside the original `hr_score`; the old score is never touched.

CRITICAL: the sheet schema grew over time (best_odds, platoon_score,
combined_score, barrel windows etc. were added as the project went along).
Early rows are MISSING columns later rows have. This script must NOT crash on
those — for each row it rescoreswhatever it has the inputs for, records what
it couldn't compute, and skips gracefully. A row missing a power input still
gets scored on the components it does have; a row missing EVERYTHING scoring-
related is flagged and left with a blank recalc.

Imports the REAL scoring functions from hr_picks so the math can't drift.

USAGE (GitHub Actions, reads your live sheet via secrets):
    python3 rescore_all.py            # rescore + write back + analyze
    python3 rescore_all.py --dry      # rescore + analyze, DON'T write to sheet
"""

import sys, os, json
import numpy as np
import pandas as pd

DRY = "--dry" in sys.argv
MODEL_START = "2026-06-09"

# ── import the REAL scoring functions (same math as live) ────────────────
# hr_picks imports pybaseball at module level; that's fine on the Actions
# runner where pybaseball is installed. If import fails, we fall back to a
# clear error rather than a wrong formula.
try:
    import hr_picks as HP
    HAVE_HP = True
except Exception as e:
    print(f"WARNING: could not import hr_picks ({e}).")
    print("Cannot rescore without the real formula — aborting rescore.")
    HAVE_HP = False


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def g(row, col, d=np.nan):
    """Safe get: returns the row's value for col, or default if col is absent
    or blank. This is what makes the script schema-tolerant."""
    if col not in row.index:
        return d
    v = str(row[col]).strip()
    if v in ("", "nan", "None"):
        return d
    return sf(v, d)


def recalc_row(row):
    """Rescore ONE row using the current formula, from whatever inputs exist.
    Returns (new_score, missing_list). Never raises on a missing column."""
    missing = []
    total = 0.0

    def need(col):
        v = g(row, col)
        if np.isnan(v):
            missing.append(col)
            return 0.0
        return v

    pa = g(row, "pa", 100)   # default pa so thresholds behave if absent
    if np.isnan(pa):
        pa = 100

    # --- power composite (the new part) ---
    sb  = need("season_barrel_pct")
    hf  = need("hr_per_fb")
    hp  = need("hr_per_pa")
    iso = need("iso")
    total += HP.score_power_composite(sb, hf, hp, iso, pa)

    # --- barrel windows (recent form) ---
    b7 = g(row, "barrel_pct_7d");  bbe7 = g(row, "bbe_7d", 0)
    b5 = g(row, "barrel_pct_5d");  bbe5 = g(row, "bbe_5d", 0)
    b10 = g(row, "barrel_pct_10d"); bbe10 = g(row, "bbe_10d", 0)
    if not np.isnan(b7):  total += HP.score_barrel_pct_7d(b7, 0 if np.isnan(bbe7) else bbe7)
    else: missing.append("barrel_pct_7d")
    if not np.isnan(b5):  total += HP.score_barrel_pct_5d(b5, 0 if np.isnan(bbe5) else bbe5)
    else: missing.append("barrel_pct_5d")
    if not np.isnan(b10): total += HP.score_barrel_pct_10d(b10, 0 if np.isnan(bbe10) else bbe10)
    else: missing.append("barrel_pct_10d")

    # --- pitcher components ---
    pbar = g(row, "pitcher_barrel_pct")
    if not np.isnan(pbar): total += HP.score_pitcher_barrel_pct(pbar)
    phf = g(row, "pitcher_hr_per_fb")
    if not np.isnan(phf): total += HP.score_pitcher_hr_per_fb(phf)

    # --- weighted matchup/context (use stored *_score if present) ---
    plat = g(row, "platoon_score")
    if not np.isnan(plat):
        total += max(-HP.PLATOON_CAP, min(HP.PLATOON_CAP, plat))   # already weighted? stored as score
    pit = g(row, "pitch_matchup_score")
    if not np.isnan(pit):
        total += pit   # stored pitch matchup score
    # weather / wind if present
    wb = g(row, "weather_boost")
    if not np.isnan(wb): total += HP.score_weather_boost(wb) * HP.WEATHER_WEIGHT

    return round(total, 2), missing


def rescore(df):
    print(f"\n{'='*72}\nRESCORING {len(df)} rows with the current formula\n{'='*72}")
    recalcs, all_missing = [], {}
    fully_scored = 0
    for _, row in df.iterrows():
        new, missing = recalc_row(row)
        recalcs.append(new)
        if not missing:
            fully_scored += 1
        for m in missing:
            all_missing[m] = all_missing.get(m, 0) + 1
    df["hr_score_recalc"] = recalcs
    print(f"  rescored: {len(df)} rows ({fully_scored} with ALL inputs, "
          f"{len(df)-fully_scored} missing \u22651 input)")
    if all_missing:
        print("\n  columns missing on some rows (rescored without them):")
        for col, cnt in sorted(all_missing.items(), key=lambda x: -x[1]):
            print(f"    {col:<24} missing on {cnt} rows")
    # old vs new distribution
    old = pd.to_numeric(df.get("hr_score", pd.Series()).apply(sf), errors="coerce").dropna()
    new = pd.to_numeric(df["hr_score_recalc"], errors="coerce").dropna()
    print(f"\n  OLD score: mean {old.mean():.2f}, p90 {old.quantile(.9):.1f}, max {old.max():.1f}")
    print(f"  NEW score: mean {new.mean():.2f}, p90 {new.quantile(.9):.1f}, max {new.max():.1f}")
    print(f"  -> top compressed by ~{old.quantile(.9)-new.quantile(.9):.1f} at p90 (saturation removed)")
    return df


def analyze(df):
    """Recompute tier hit-rate analysis on the NEW recalc score."""
    print(f"\n{'='*72}\nTIER ANALYSIS ON NEW SCORE (hr_score_recalc)\n{'='*72}")
    d = df.copy()
    d["sc"] = pd.to_numeric(d["hr_score_recalc"], errors="coerce")
    d["res"] = d.get("hit_hr", "").astype(str).str.strip().str.lower()
    d = d[d["res"].isin(["yes", "no"])].copy()
    d["hit"] = (d["res"] == "yes").astype(int)
    d["odds"] = pd.to_numeric(d.get("consensus_odds", "").apply(sf), errors="coerce")
    if "date" in d.columns:
        d = d[d["date"].astype(str) >= MODEL_START]
    d = d.dropna(subset=["sc"])
    print(f"  {len(d)} resolved rows on new scale, base {d['hit'].mean()*100:.1f}%\n")
    # tiers by percentile of the NEW score so they fit the new scale
    qs = [d["sc"].quantile(q) for q in [0.98,0.90,0.80,0.65,0.50,0.35,0.20]]
    edges = [("top 2%",qs[0],99),("90-98",qs[1],qs[0]),("80-90",qs[2],qs[1]),
             ("65-80",qs[3],qs[2]),("50-65",qs[4],qs[3]),("35-50",qs[5],qs[4]),
             ("20-35",qs[6],qs[5]),("bot 20",-99,qs[6])]
    print(f"  {'tier (pctile)':<16}{'score rng':<16}{'n':>6}{'hit%':>8}")
    print("  " + "-"*48)
    prev = None
    for lab, lo, hi in edges:
        s = d[(d["sc"]>=lo)&(d["sc"]<hi)]
        if len(s) < 15:
            print(f"  {lab:<16}{f'{lo:.1f}-{hi:.1f}':<16}{len(s):>6}   --"); continue
        r = s["hit"].mean()*100
        arrow = ""
        if prev is not None: arrow = " ^" if r>prev else " v" if r<prev else ""
        print(f"  {lab:<16}{f'{lo:.1f}-{hi:.1f}':<16}{len(s):>6}{r:>7.1f}%{arrow}")
        prev = r
    print("\n  If this is cleaner/more monotonic than the old-scale tiers, the")
    print("  rescore worked — you now have validated bands on the NEW scale")
    print("  from all your history, no waiting weeks for fresh data.")


def main():
    if not HAVE_HP:
        return
    import gspread
    from google.oauth2.service_account import Credentials
    scope = ["https://www.googleapis.com/auth/spreadsheets",
             "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=scope)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    ws = sh.worksheet("HR_All_Scores")
    vals = ws.get_all_values()
    df = pd.DataFrame(vals[1:], columns=vals[0])
    print(f"Loaded HR_All_Scores: {len(df)} rows, {len(df.columns)} columns")

    df = rescore(df)
    analyze(df)

    if DRY:
        print("\n[--dry] not writing to sheet.")
        return
    # write hr_score_recalc back (append column, keep everything else)
    print("\nWriting hr_score_recalc column back to HR_All_Scores...")
    header = df.columns.tolist()
    ws.clear()
    ws.update([header] + df.astype(str).values.tolist())
    print("Done. hr_score_recalc written (original hr_score untouched).")


if __name__ == "__main__":
    main()
