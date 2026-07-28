#!/usr/bin/env python3
"""
test_parlay_bands.py

Which 2-leg parlays actually profit, at CONSENSUS odds. Groups two ways,
INDEPENDENTLY:

  A) by SCORE band   (13-15, 12-13, ... )
  B) by EDGE band    (the calc_edge value: ✅ positive / ➡️ neutral / ❌ neg,
                      or numeric edge buckets)

For each band it forms all 2-leg combinations of that band's legs on the SAME
day (independent legs, different games where possible), then reports:

  actual cash rate   -- did both legs hit
  implied breakeven  -- from the combined consensus odds
  units / ROI        -- flat 1u per ticket, priced at combined consensus odds
  bootstrap 95% CI on ROI

KEY DISCIPLINE (learned the hard way):
  * Parlays are priced by the PRODUCT of leg odds, and win only if BOTH legs
    hit. A losing leg compounds. A band is +EV only if its cash rate clears
    the combined breakeven -- which needs the leg hit rate well above the
    single-leg breakeven.
  * Same-day legs only (a real parlay is same slate). Optionally same-game
    excluded to avoid correlation.
  * In-sample. A profitable band here is a hypothesis to bet forward.

USAGE
    python3 test_parlay_bands.py            # live sheet
    python3 test_parlay_bands.py file.csv   # local CSV
"""

import sys
import os
import json
import math
import itertools
import pandas as pd
import numpy as np


def dec(odds):
    o = float(odds)
    return 1 + (o/100 if o >= 0 else 100/abs(o))


def implied(odds):
    o = float(odds)
    return 100/(o+100) if o >= 0 else abs(o)/(abs(o)+100)


def bootstrap_ci(units, iters=4000, seed=0):
    a = np.asarray(units, float)
    if len(a) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = rng.choice(a, size=(iters, len(a)), replace=True).mean(axis=1)
    return (np.percentile(m, 2.5)*100, np.percentile(m, 97.5)*100)


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["score"] = pd.to_numeric(df.get("hr_score", ""), errors="coerce")
    df["odds"] = pd.to_numeric(
        df.get("consensus_odds", "").astype(str).str.replace("+", "", regex=False).str.strip(),
        errors="coerce")
    df["res"] = df.get("hit_hr", "").str.strip().str.lower()
    df["date"] = df.get("date", "").astype(str).str.strip()
    df["game"] = df.get("pitcher_name", "").astype(str).str.strip()  # proxy for game/opponent
    df["edge_txt"] = df.get("edge", "").astype(str)
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df = df.dropna(subset=["score", "odds"])
    df = df[(df["odds"] > 0)]
    return df


def score_band(s):
    if s >= 15: return "15+"
    if s >= 14: return "14-15"
    if s >= 13: return "13-14"
    if s >= 12: return "12-13"
    if s >= 11: return "11-12"
    if s >= 10: return "10-11"
    return "<10"


def edge_band(txt):
    t = str(txt)
    if t.startswith("✅"): return "✅ positive"
    if t.startswith("➡"):  return "➡️ neutral"
    if t.startswith("❌"): return "❌ negative"
    return "(no edge)"


def make_combos(pool, legs=2, exclude_same_game=True):
    """All same-day N-leg combos within a pool. Returns list of
    (cash_bool, combined_decimal, combined_implied)."""
    tickets = []
    for _, day in pool.groupby("date"):
        recs = day.to_dict("records")
        for combo in itertools.combinations(recs, legs):
            if exclude_same_game:
                games = [c["game"] for c in combo if c["game"]]
                if len(games) != len(set(games)):   # any two legs share a game
                    continue
            cash = int(all(c["hit"] == 1 for c in combo))
            cdec = 1.0
            cimp = 1.0
            for c in combo:
                cdec *= dec(c["odds"])
                cimp *= implied(c["odds"])
            tickets.append((cash, cdec, cimp))
    return tickets


def make_pairs(pool, exclude_same_game=True):
    return make_combos(pool, legs=2, exclude_same_game=exclude_same_game)


def report(name, bands):
    print(f"\n{'='*84}\n{name}\n{'='*84}")
    print(f"{'band':<16}{'tickets':>8}{'cash%':>8}{'BE%':>7}{'edge':>7}"
          f"{'ROI':>9}{'ROI 95% CI':>20}")
    print("-"*84)
    for label, tickets in bands:
        n = len(tickets)
        if n < 15:
            print(f"{label:<16}{n:>8}{'--':>8}{'--':>7}{'--':>7}{'--':>9}"
                  f"{'(too few)':>20}")
            continue
        cash = sum(t[0] for t in tickets)
        cash_rate = cash / n
        be = np.mean([t[2] for t in tickets])  # avg implied prob = breakeven
        units = []
        for c, cdec, _ in tickets:
            units.append((cdec - 1) if c else -1.0)
        units = np.array(units)
        roi = units.mean() * 100
        lo, hi = bootstrap_ci(units)
        edge = (cash_rate - be) * 100
        star = " +EV" if lo > 0 else ""
        print(f"{label:<16}{n:>8}{cash_rate*100:>7.1f}%{be*100:>6.1f}%"
              f"{edge:>+7.1f}{roi:>+8.1f}%   [{lo:>+6.1f},{hi:>+6.1f}]{star}")


def run_for_legs(df, legs):
    tag = f"{legs}-LEG"
    score_order = ["15+", "14-15", "13-14", "12-13", "11-12", "10-11"]
    bands_a = [(b, make_combos(df[df["sband"] == b], legs=legs)) for b in score_order]
    report(f"A) {tag} PARLAYS BY SCORE BAND  (all legs same band, same day)", bands_a)

    bands_a2 = [
        ("13-15",       make_combos(df[df["score"].between(13, 15, inclusive="left")], legs=legs)),
        ("13-15 ≤+400", make_combos(df[(df["score"].between(13,15,inclusive="left")) & (df["odds"]<=400)], legs=legs)),
        ("14-15 ≤+400", make_combos(df[(df["score"].between(14,15,inclusive="left")) & (df["odds"]<=400)], legs=legs)),
    ]
    report(f"A2) {tag} POOLED GOOD-SCORE PARLAYS", bands_a2)

    bands_b = [(b, make_combos(df[df["eband"] == b], legs=legs))
               for b in ["✅ positive", "➡️ neutral", "❌ negative"]]
    report(f"B) {tag} PARLAYS BY EDGE BAND  (calc_edge label)", bands_b)


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved, priced rows across "
          f"{df['date'].nunique()} days.")

    df["sband"] = df["score"].apply(score_band)
    df["eband"] = df["edge_txt"].apply(edge_band)

    print("\n" + "#"*84 + "\n# TWO-LEG PARLAYS\n" + "#"*84)
    run_for_legs(df, 2)

    print("\n" + "#"*84 + "\n# THREE-LEG PARLAYS\n" + "#"*84)
    run_for_legs(df, 3)

    print(f"\n{'='*84}\nREAD\n{'='*84}")
    print("""  A band is +EV only if cash% > BE% AND the ROI 95% CI clears 0 (' +EV').
  A positive ROI whose CI straddles 0 is NOT established -- it's a hot streak
  candidate, same trap as before. Everything is in-sample and same-day combos
  reuse legs across many tickets, so CIs are optimistic -- treat ' +EV' as
  'worth testing forward,' never as proven. Priced at consensus; your real
  book price differs, and that gap (CLV) is still unmeasured.""")


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
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    vals = sh.worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else None


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
