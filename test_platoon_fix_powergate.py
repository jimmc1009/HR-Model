#!/usr/bin/env python3
"""
test_platoon_fix_powergate.py

Answers two questions from an HR_All_Scores export alone (no raw feature
sheets needed, since Batter/Pitcher_Statcast are overwritten each run):

  Q1. How is batter_hand encoded for switch hitters?
      If a player shows BOTH "L" and "R" across dates AND it is always
      opposite pitcher_hand, upstream is resolving switch hitters
      dynamically -> batter_hand already IS the effective hand -> the
      handedness fix in hr_picks.py is correct as written.

  Q2. Does the fix have enough blast radius to be backtestable?
      Recomputes the platoon "Piece 2" (pitcher barrel% allowed vs hand)
      the buggy way and the fixed way, derives the hr_score delta, and
      reports how many rows cross a score-tier line. If very few cross,
      the honest verdict is "ship as correctness hygiene, no P/L claim."

USAGE
    python3 test_platoon_fix_powergate.py HR_All_Scores.csv

NOTES / LIMITS
  * Piece 2 is recomputed EXACTLY (both barrel splits are logged).
  * The pre-weight clamp at +/-2.0 is not invertible. Rows logging
    platoon_score == +/-2.4 were clamped; they are flagged and excluded
    from the point estimate rather than guessed at.
  * Platoon enters the final score with coefficient 1.0 (PLATOON_WEIGHT
    is applied inside compute_platoon_score), so:
        delta_score = clip(new_platoon,-2,2) - clip(old_platoon,-2,2)
"""

import sys
import pandas as pd
import numpy as np

PLATOON_WEIGHT = 1.2
CLAMP = 2.0
TIERS = [13.0, 14.0, 15.0]   # the lines that actually matter per the handoff


def safe_float(val, default=0.0):
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def piece2(barrel_pct):
    """Pitcher barrel% allowed vs hand -> raw (pre-weight) contribution."""
    if barrel_pct <= 0:
        return 0.0
    if barrel_pct >= 14: return 0.8
    if barrel_pct >= 11: return 0.4
    if barrel_pct >= 9:  return 0.2
    if barrel_pct <= 4:  return -0.6
    if barrel_pct <= 6:  return -0.3
    return 0.0


def effective_hand(batter_hand, p_throws):
    if batter_hand == "S":
        return "R" if p_throws == "L" else "L"
    return batter_hand


def main(path):
    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]

    need = ["player_name", "batter_hand", "pitcher_hand",
            "pitcher_barrel_vs_lhh", "pitcher_barrel_vs_rhh",
            "platoon_score", "hr_score"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"FATAL: missing columns: {missing}")
        return

    df["bh"] = df["batter_hand"].str.strip().str.upper()
    df["ph"] = df["pitcher_hand"].str.strip().str.upper()

    print("=" * 68)
    print("Q1. BATTER_HAND ENCODING")
    print("=" * 68)
    print("\nbatter_hand value counts:")
    for val, n in df["bh"].value_counts(dropna=False).items():
        show = val if val else "(blank)"
        print(f"   {show:<10} {n:>7}")

    # Players appearing with more than one distinct hand = switch candidates
    valid = df[df["bh"].isin(["L", "R"])]
    per_player = valid.groupby("player_name")["bh"].nunique()
    switchers = per_player[per_player > 1].index.tolist()

    print(f"\nPlayers with BOTH L and R across dates: {len(switchers)}")
    if switchers:
        sub = valid[valid["player_name"].isin(switchers)]
        opposite = (sub["bh"] != sub["ph"]).mean()
        print(f"  Of their rows, batter_hand is opposite pitcher_hand: {opposite:.1%}")
        if opposite > 0.99:
            print("\n  VERDICT: switch hitters ARE resolved upstream to the side")
            print("           they bat from. batter_hand == effective hand.")
            print("           -> The fix in hr_picks.py is CORRECT as written.")
            print("           -> The `batter_hand == \"S\"` branch is dead code")
            print("              (harmless; future-proofs an upstream change).")
        else:
            print("\n  VERDICT: hands vary but NOT strictly opposite the pitcher.")
            print("           Inspect manually before trusting the fix for these.")
        print(f"\n  Sample: {', '.join(switchers[:8])}")
    else:
        print("\n  VERDICT: no player ever changes hands. Switch hitters are")
        print("           either absent or pinned to one side. If MLB switch")
        print("           hitters appear in your pool, the upstream encoding")
        print("           needs checking before this fix is complete.")

    # ── Q2: blast radius + power gate ─────────────────────────────────
    print("\n" + "=" * 68)
    print("Q2. FIX BLAST RADIUS / POWER GATE")
    print("=" * 68)

    d = df[df["ph"].isin(["L", "R"]) & df["bh"].isin(["L", "R", "S"])].copy()
    d["b_lhh"] = d["pitcher_barrel_vs_lhh"].apply(safe_float)
    d["b_rhh"] = d["pitcher_barrel_vs_rhh"].apply(safe_float)
    d["plat"]  = d["platoon_score"].apply(safe_float)
    d["score"] = d["hr_score"].apply(safe_float)

    # OLD (buggy): keyed on pitcher's throwing hand
    d["p2_old"] = np.where(d["ph"] == "L", d["b_lhh"], d["b_rhh"])
    # NEW (fixed): keyed on batter's effective hand
    d["eff"] = [effective_hand(b, p) for b, p in zip(d["bh"], d["ph"])]
    d["p2_new"] = np.where(d["eff"] == "L", d["b_lhh"], d["b_rhh"])

    d["p2_old_s"] = d["p2_old"].apply(piece2)
    d["p2_new_s"] = d["p2_new"].apply(piece2)
    d["p2_delta"] = d["p2_new_s"] - d["p2_old_s"]

    opp = (d["eff"] != d["ph"])
    print(f"\nRows analysed:              {len(d):>7}")
    print(f"  opposite-hand matchups:   {opp.sum():>7}  ({opp.mean():.1%})")
    print(f"  same-hand (fix is a no-op):{(~opp).sum():>7}")

    moved = d["p2_delta"] != 0
    flipped = (d["p2_old_s"] * d["p2_new_s"]) < 0
    print(f"\nPiece-2 value changes:      {moved.sum():>7}  ({moved.mean():.1%} of rows)")
    print(f"  sign flips (worst cases):  {flipped.sum():>7}")

    # Clamp handling: |platoon_score| == 2.4 means the raw sum was clipped
    clamped = np.isclose(d["plat"].abs(), CLAMP * PLATOON_WEIGHT, atol=1e-6)
    print(f"  clamped rows (not invertible, excluded): {clamped.sum():>5}")

    est = d[~clamped].copy()
    raw_old = est["plat"] / PLATOON_WEIGHT
    raw_new = raw_old - est["p2_old_s"] + est["p2_new_s"]
    plat_new = np.clip(raw_new, -CLAMP, CLAMP) * PLATOON_WEIGHT
    est["score_delta"] = (np.clip(plat_new, -CLAMP, CLAMP)
                          - np.clip(est["plat"], -CLAMP, CLAMP))
    est["score_new"] = est["score"] + est["score_delta"]

    nz = est[est["score_delta"] != 0]
    print(f"\nhr_score changes:           {len(nz):>7}")
    if len(nz):
        print(f"  mean |delta|:              {nz['score_delta'].abs().mean():>7.3f}")
        print(f"  max  |delta|:              {nz['score_delta'].abs().max():>7.3f}")

    print("\nTier-line crossings (the thing that actually moves picks):")
    total_cross = 0
    for t in TIERS:
        up   = ((est["score"] < t) & (est["score_new"] >= t)).sum()
        down = ((est["score"] >= t) & (est["score_new"] < t)).sum()
        total_cross += up + down
        print(f"   score {t:>5.1f}   up: {up:>5}   down: {down:>5}")

    print("\n" + "-" * 68)
    if total_cross < 30:
        print("POWER GATE: FAILED.")
        print(f"  Only {total_cross} tier crossings. There is not enough movement to")
        print("  measure a hit-rate or P/L effect. Ship the fix as CORRECTNESS")
        print("  HYGIENE and make no profitability claim from it.")
    else:
        print("POWER GATE: PASSED (marginally).")
        print(f"  {total_cross} tier crossings. A within-zone hit-rate comparison is")
        print("  worth running -- but treat any large P/L swing as noise, not")
        print("  edge, until it survives out-of-sample.")
    print("-" * 68)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
