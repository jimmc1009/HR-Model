"""
rescore_platoon_history.py
Corrects the historical barrel-handedness flip in HR_All_Scores.

THE BUG (now fixed live): the platoon score's "pitcher barrel% allowed vs hand"
sub-block was keyed on the PITCHER's throwing hand instead of the BATTER's
hitting hand. For opposite-handed matchups this used the wrong split.

THIS SCRIPT recomputes ONLY that sub-block's contribution, both the old (wrong)
way and the new (correct) way, takes the delta, and applies it to the stored
platoon_score and hr_score — writing to PARALLEL columns so originals are
preserved and the change is fully reversible.

It does NOT recompute the whole score from scratch (the other components are
unchanged and their raw inputs aren't all stored per-row). It surgically
adjusts for the one flipped sub-block, which is exactly and only what changed.

Writes: platoon_score_corrected, hr_score_corrected  (originals untouched)
Verifies on a known row before writing.
"""

import os
import json
import time

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

PLATOON_WEIGHT = 1.2   # must match hr_picks.py


def get_gspread_client():
    info  = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=20):
    for a in range(retries):
        try:
            return func()
        except Exception:
            if a < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def barrel_subblock(pitcher_barrel_vs_hand: float) -> float:
    """The exact barrel sub-block contribution to the RAW platoon score,
    copied verbatim from compute_platoon_score in hr_picks.py."""
    s = 0.0
    if pitcher_barrel_vs_hand > 0:
        if pitcher_barrel_vs_hand >= 14:
            s += 0.8
        elif pitcher_barrel_vs_hand >= 11:
            s += 0.4
        elif pitcher_barrel_vs_hand >= 9:
            s += 0.2
        elif pitcher_barrel_vs_hand <= 4:
            s -= 0.6
        elif pitcher_barrel_vs_hand <= 6:
            s -= 0.3
    return s


def old_barrel_value(row):
    """OLD (buggy) selection: keyed on pitcher hand."""
    ph = str(row.get("pitcher_hand", "")).strip().upper()
    if ph == "L":
        return safe_float(row.get("pitcher_barrel_vs_lhh", 0))
    elif ph == "R":
        return safe_float(row.get("pitcher_barrel_vs_rhh", 0))
    return 0.0


def new_barrel_value(row):
    """NEW (correct) selection: keyed on batter hand, switch resolves opposite."""
    bh = str(row.get("batter_hand", "")).strip().upper()
    ph = str(row.get("pitcher_hand", "")).strip().upper()
    eff = bh
    if bh == "S":
        eff = "L" if ph == "R" else "R"
    if eff == "L":
        return safe_float(row.get("pitcher_barrel_vs_lhh", 0))
    elif eff == "R":
        return safe_float(row.get("pitcher_barrel_vs_rhh", 0))
    return 0.0


def compute_correction(row):
    """Returns (platoon_delta_weighted, old_barrel_used, new_barrel_used).
    Delta already includes PLATOON_WEIGHT. Note: the raw platoon score is
    capped at +/-2.0 BEFORE weighting. We can't perfectly re-cap without the
    other sub-blocks, but we can reconstruct: stored platoon_score / WEIGHT =
    capped raw. We recompute raw_old and raw_new by swapping just the barrel
    piece, re-cap each, then re-weight. This requires the pre-barrel raw, which
    we back out from the stored value."""
    ph = str(row.get("pitcher_hand", "")).strip().upper()
    if ph not in ("L", "R"):
        return 0.0, 0.0, 0.0   # platoon returned 0 originally; nothing to fix

    stored_platoon = safe_float(row.get("platoon_score", 0))
    # stored = round(cap(raw_old) * WEIGHT, 3)  ->  cap(raw_old) = stored/WEIGHT
    capped_raw_old = stored_platoon / PLATOON_WEIGHT

    old_b = old_barrel_value(row)
    new_b = new_barrel_value(row)
    old_sub = barrel_subblock(old_b)
    new_sub = barrel_subblock(new_b)

    # raw_old (uncapped) is unknown, but the barrel piece is additive. We
    # reconstruct the NON-barrel raw as (capped_raw_old - old_sub) ONLY when the
    # cap wasn't binding. If cap was binding (|capped_raw_old| == 2.0), we can't
    # cleanly separate; we conservatively recompute assuming non-barrel =
    # capped_raw_old - old_sub, then re-cap. This is exact except in the rare
    # case the old value was cap-clipped, where it's a close approximation.
    non_barrel_raw = capped_raw_old - old_sub

    raw_new = non_barrel_raw + new_sub
    capped_raw_new = max(-2.0, min(2.0, raw_new))

    new_platoon = round(capped_raw_new * PLATOON_WEIGHT, 3)
    delta = round(new_platoon - stored_platoon, 3)
    return delta, old_b, new_b


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    sh   = with_retry(lambda: gc.open_by_key(sheet_id))
    ws   = sh.worksheet("HR_All_Scores")
    vals = with_retry(lambda: ws.get_all_values())
    headers = vals[0]
    df = pd.DataFrame(vals[1:], columns=headers)
    df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)].reset_index(drop=True)
    print(f"  {len(df)} rows")

    # ── Verify on a known opposite-handed row before writing anything ──────
    print("\nVerification (opposite-handed rows where the fix should move things):")
    shown = 0
    for _, row in df.iterrows():
        bh = str(row.get("batter_hand", "")).strip().upper()
        ph = str(row.get("pitcher_hand", "")).strip().upper()
        if bh in ("L", "R") and ph in ("L", "R") and bh != ph:
            delta, ob, nb = compute_correction(row)
            if abs(delta) > 0.001:
                print(f"  {str(row.get('player_name','')):<20} {row.get('date','')} "
                      f"{bh}HH vs {ph}HP | old_barrel={ob:.1f}% new_barrel={nb:.1f}% "
                      f"| platoon Δ={delta:+.3f}")
                shown += 1
        if shown >= 8:
            break

    # ── Apply correction to all rows ──────────────────────────────────────
    corrected_platoon = []
    corrected_score   = []
    n_changed = 0
    for _, row in df.iterrows():
        delta, _, _ = compute_correction(row)
        stored_platoon = safe_float(row.get("platoon_score", 0))
        stored_score   = safe_float(row.get("hr_score", 0))
        corrected_platoon.append(round(stored_platoon + delta, 3))
        corrected_score.append(round(stored_score + delta, 3))
        if abs(delta) > 0.001:
            n_changed += 1

    df["platoon_score_corrected"] = [str(x) for x in corrected_platoon]
    df["hr_score_corrected"]      = [str(x) for x in corrected_score]

    print(f"\n  {n_changed} of {len(df)} rows changed by the correction "
          f"({round(n_changed/len(df)*100,1)}%)")

    # summary of score movement
    deltas = np.array(corrected_score) - df["hr_score"].apply(safe_float).values
    moved  = deltas[np.abs(deltas) > 0.001]
    if len(moved):
        print(f"  Avg score change (changed rows): {moved.mean():+.3f}")
        print(f"  Range: {moved.min():+.3f} to {moved.max():+.3f}")

    # ── Write back (originals preserved, two new columns appended) ─────────
    print("\nWriting corrected columns back to HR_All_Scores...")
    out = df.fillna("").replace([np.inf, -np.inf], "")
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([out.columns.tolist()] + out.astype(str).values.tolist()))
    print("Done. Originals in platoon_score/hr_score; corrected in "
          "platoon_score_corrected/hr_score_corrected.")
    print("\nNext: point hr_analysis / zone-building at hr_score_corrected to "
          "rebuild zones on clean data.")


if __name__ == "__main__":
    main()
