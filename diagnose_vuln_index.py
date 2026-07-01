"""
diagnose_vuln_index.py
OBSERVATION ONLY — watch the composite pitcher vulnerability index develop
within the score >= 10 pool as data accumulates. Does not change the model;
just shows how things look.

The composite index stacks multiple independent "homer-prone pitcher" signals,
firing higher only when several agree (same philosophy as the batter score):
  - Season HR/9 (handedness-matched when available)
  - Season HR/FB% allowed
  - Fly-ball rate allowed (can't homer on grounders)
  - Season barrel% allowed (handedness-matched)
  - Recent form: barrel% 7d allowed, HR allowed recently
  - BABIP context (luck adjustment)

Each sub-signal is normalized 0-1, then averaged into a 0-10 index.
Reports sample size honestly so you know how much to trust the read.
"""

import os
import json
import time
import re

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MODEL_START_DATE = "2026-06-09"
POOL_THRESHOLD   = 10.0


def get_gspread_client():
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=15):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=np.nan):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def derive_batter_hand(row):
    bh = str(row.get("batter_hand", "")).strip().upper()
    if bh in ("L", "R", "S"):
        return bh
    pm = str(row.get("platoon_matchup", "")).upper()
    m = re.search(r"([LR])HH", pm)
    if m:
        return m.group(1)
    return ""


def norm(val, lo, hi):
    """Normalize value to 0-1 where hi=most vulnerable. NaN-safe."""
    if pd.isna(val):
        return np.nan
    if val <= lo:
        return 0.0
    if val >= hi:
        return 1.0
    return (val - lo) / (hi - lo)


def compute_vuln_index(row):
    """
    Composite vulnerability 0-10. Averages available normalized sub-signals.
    Only counts signals that have data — pitcher with more data gets a
    fuller read, but we don't penalize missing fields.
    """
    bh = row["batter_hand_derived"]

    # Handedness-matched HR/9 and barrel% where possible
    if bh == "L":
        hr9_matched   = safe_float(row.get("pitcher_vs_lhh_hr9"))
        barrel_matched = safe_float(row.get("pitcher_barrel_vs_lhh"))
    elif bh == "R":
        hr9_matched   = safe_float(row.get("pitcher_vs_rhh_hr9"))
        barrel_matched = safe_float(row.get("pitcher_barrel_vs_rhh"))
    else:
        hr9_matched   = safe_float(row.get("pitcher_hr9"))
        barrel_matched = safe_float(row.get("pitcher_barrel_pct"))

    # Fall back to overall if matched missing
    if pd.isna(hr9_matched):
        hr9_matched = safe_float(row.get("pitcher_hr9"))
    if pd.isna(barrel_matched):
        barrel_matched = safe_float(row.get("pitcher_barrel_pct"))

    signals = []

    # HR/9 — league avg ~1.1, vulnerable >1.5, elite <0.8
    s = norm(hr9_matched, 0.8, 2.0)
    if not pd.isna(s): signals.append(s)

    # HR/FB% allowed — avg ~13%, vulnerable >16%
    s = norm(safe_float(row.get("pitcher_hr_per_fb")), 9, 18)
    if not pd.isna(s): signals.append(s)

    # Fly-ball rate allowed — more FB = more HR opportunity
    s = norm(safe_float(row.get("pitcher_fb_rate_allowed")), 30, 48)
    if not pd.isna(s): signals.append(s)

    # Barrel% allowed (matched) — avg ~8%, vulnerable >11%
    s = norm(barrel_matched, 6, 13)
    if not pd.isna(s): signals.append(s)

    # Recent form: barrel% 7d allowed
    s = norm(safe_float(row.get("pitcher_barrel_7d")), 6, 14)
    if not pd.isna(s): signals.append(s)

    if len(signals) < 2:
        return np.nan, len(signals)

    return round(np.mean(signals) * 10, 2), len(signals)


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    sh         = with_retry(lambda: gc.open_by_key(sheet_id))
    ws         = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())
    headers    = all_values[0]
    rows       = all_values[1:]
    df         = pd.DataFrame(rows, columns=headers)
    df         = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]

    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()

    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["hr_score"] = resolved["hr_score"].apply(lambda x: safe_float(x, 0))

    pool = resolved[resolved["hr_score"] >= POOL_THRESHOLD].copy()
    pool["batter_hand_derived"] = pool.apply(derive_batter_hand, axis=1)

    # Check whether the new pitcher columns even exist / are populated
    new_cols = ["pitcher_hr9", "pitcher_fb_rate_allowed", "pitcher_barrel_7d",
                "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9"]
    have_cols = [c for c in new_cols if c in pool.columns]
    populated = {}
    for c in have_cols:
        populated[c] = (pool[c].apply(lambda x: str(x).strip() not in ("", "nan", "None"))).sum()

    print(f"\n  Pool (score >= {POOL_THRESHOLD}): {len(pool)} resolved picks")
    print(f"  New pitcher columns present: {have_cols if have_cols else 'NONE YET'}")
    if populated:
        print("  Populated counts:")
        for c, n in populated.items():
            print(f"    {c}: {n}/{len(pool)}")

    # Compute index
    idx_results = pool.apply(compute_vuln_index, axis=1)
    pool["vuln_index"]   = idx_results.apply(lambda x: x[0])
    pool["vuln_signals"] = idx_results.apply(lambda x: x[1])

    scored = pool[pool["vuln_index"].notna()].copy()
    n_scored = len(scored)

    # How many of the scored rows actually have the NEW pitcher vulnerability
    # data (not just the old fallback columns)? The confidence flag should be
    # based on real new-data rows, otherwise it overstates how proven this is.
    def has_new_data(row):
        for c in ["pitcher_hr9", "pitcher_fb_rate_allowed", "pitcher_barrel_7d"]:
            v = str(row.get(c, "")).strip()
            if v not in ("", "nan", "None"):
                return True
        return False

    n_realdata = int(scored.apply(has_new_data, axis=1).sum()) if n_scored > 0 else 0

    # ── Honest confidence flag — based on REAL new-column data ───────────
    print("\n" + "="*64)
    if n_realdata < 60:
        conf = (f"⏳ TOO EARLY — only {n_realdata} rows have real vulnerability data "
                f"(index is running mostly on old fallback columns; wait for new data)")
    elif n_realdata < 150:
        conf = f"🌱 PRELIMINARY (real n={n_realdata}) — directional only, don't trust yet"
    elif n_realdata < 250:
        conf = f"📊 DEVELOPING (real n={n_realdata}) — starting to be meaningful"
    else:
        conf = f"✅ TESTABLE (real n={n_realdata}) — holdout split is meaningful"
    print(f"  CONFIDENCE: {conf}")
    print(f"  ({n_scored} total scored rows, {n_realdata} with real new-column data)")
    print("="*64)

    if n_scored < 20:
        print("\n  Not enough scored rows yet. Check back after more data accumulates.")
        print("\nDone.")
        return

    base_rate = round(scored["hit_bool"].mean() * 100, 1)
    print(f"\n  Pool hit rate (with index data): {base_rate}%")

    # ── Hit rate by index bucket ────────────────────────────────────────
    print("\n  HIT RATE BY VULNERABILITY INDEX BUCKET:")
    print(f"    {'Index Bucket':<16} {'N':>5} {'Hits':>5} {'Hit Rate':>9}")
    print(f"    {'-'*40}")
    for blabel, lo, hi in [
        ("Low 0-3",    0, 3),
        ("Mid 3-5",    3, 5),
        ("High 5-6.5", 5, 6.5),
        ("Very High 6.5+", 6.5, 99),
    ]:
        sub = scored[(scored["vuln_index"] >= lo) & (scored["vuln_index"] < hi)]
        if len(sub) < 5:
            continue
        n = len(sub)
        h = int(sub["hit_bool"].sum())
        rate = round(h / n * 100, 1)
        lift = round(rate - base_rate, 1)
        print(f"    {blabel:<16} {n:>5} {h:>5} {rate:>8.1f}%  ({lift:+.1f})")

    # ── Separator ───────────────────────────────────────────────────────
    hits = scored[scored["hit_bool"]]["vuln_index"]
    miss = scored[~scored["hit_bool"]]["vuln_index"]
    if len(hits) >= 5 and len(miss) >= 5:
        h_avg = hits.mean(); m_avg = miss.mean()
        pct = (h_avg - m_avg) / m_avg * 100 if m_avg else 0
        print(f"\n  Index separator: hits avg={h_avg:.2f}  misses avg={m_avg:.2f}  ({pct:+.1f}%)")

    # ── Date-split holdout (only if enough data) ────────────────────────
    if n_scored >= 150:
        print("\n  DATE-SPLIT HOLDOUT:")
        ss = scored.sort_values("date_dt")
        mid = ss["date_dt"].quantile(0.5)
        for name, sub in [("First half", ss[ss["date_dt"] <= mid]),
                          ("Holdout   ", ss[ss["date_dt"] > mid])]:
            h = sub[sub["hit_bool"]]["vuln_index"]
            m = sub[~sub["hit_bool"]]["vuln_index"]
            if len(h) >= 5 and len(m) >= 5:
                pct = (h.mean() - m.mean()) / m.mean() * 100 if m.mean() else 0
                hi_sub = sub[sub["vuln_index"] >= 5]
                hi_rate = round(hi_sub["hit_bool"].mean()*100,1) if len(hi_sub) >= 5 else None
                hi_str = f"high-vuln rate={hi_rate}%" if hi_rate is not None else "high-vuln n<5"
                print(f"    {name}: sep {pct:+.1f}%  |  {hi_str}  (n={len(sub)})")
    else:
        print("\n  (Holdout split unlocks at n>=150)")

    print("\nDone.")


if __name__ == "__main__":
    main()
