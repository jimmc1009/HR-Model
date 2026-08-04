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

    def col(name):
        """Return the column as a string Series, or an empty-string Series if absent."""
        if name in df.columns:
            return df[name].astype(str)
        return pd.Series([""] * len(df), index=df.index)

    df["score"] = pd.to_numeric(col("hr_score"), errors="coerce")
    df["odds"] = pd.to_numeric(
        col("consensus_odds").str.replace("+", "", regex=False).str.strip(),
        errors="coerce")
    df["res"] = col("hit_hr").str.strip().str.lower()
    df["date"] = col("date").str.strip()
    df["game"] = col("pitcher_name").str.strip()  # proxy for game/opponent
    df["edge_txt"] = col("edge")
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df = df.dropna(subset=["score", "odds"])
    df = df[(df["odds"] > 0)]
    return df


def odds_band(o):
    if o <= 250:  return "≤+250"
    if o <= 300:  return "+251-300"
    if o <= 400:  return "+301-400"
    if o <= 499:  return "+301-499" if o > 400 else "+301-400"
    if o <= 600:  return "+500-600"
    if o <= 800:  return "+601-800"
    return "+801+"


def odds_band_wide(o):
    """Broader odds bands like the original: ≤300, 301-499, 500-699, 700+."""
    if o <= 300:  return "≤+300"
    if o <= 499:  return "+301-499"
    if o <= 699:  return "+500-699"
    return "+700+"


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


def report_simple(name, bands):
    """Hit rate + average payout per band, no edge/CI. What actually hits and
    what it pays if it does."""
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print(f"{'band':<16}{'tickets':>9}{'hit%':>8}{'avg payout':>14}{'if-win $/25c':>14}")
    print("-"*72)
    for label, tickets in bands:
        n = len(tickets)
        if n < 15:
            print(f"{label:<16}{n:>9}{'--':>8}{'(too few)':>14}")
            continue
        cash = sum(t[0] for t in tickets)
        cash_rate = cash / n
        # average combined decimal payout across the tickets in this band
        avg_dec = np.mean([t[1] for t in tickets])
        avg_american = (avg_dec - 1) * 100
        am_s = f"+{avg_american:.0f}" if avg_american >= 100 else f"{avg_dec:.2f}x"
        # what a 25-cent ticket returns on a win (profit)
        win_profit = 0.25 * (avg_dec - 1)
        print(f"{label:<16}{n:>9}{cash_rate*100:>7.1f}%{am_s:>14}{'$'+format(win_profit,'.2f'):>14}")


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


def sweep_odds_windows(df, legs, width=100, step=50, lo_start=200, hi_end=800):
    """Slide an odds window across the range; for each window, build all N-leg
    parlays whose legs ALL fall in that window, and report hit rate + payout.
    Ranked by hit rate so you can find where parlays actually cash. Payout shown
    alongside so you see the tradeoff (tighter/lower windows hit more, pay less)."""
    print(f"\n{'='*76}\n{legs}-LEG ODDS-WINDOW SWEEP  (width={width}, step={step})\n{'='*76}")
    print(f"{'odds window':<16}{'tickets':>9}{'hit%':>8}{'avg payout':>14}{'$/25c win':>12}")
    print("-"*76)
    results = []
    lo = lo_start
    while lo + width <= hi_end + step:
        hi = lo + width
        pool = df[(df["odds"] >= lo) & (df["odds"] < hi)]
        tickets = make_combos(pool, legs=legs)
        n = len(tickets)
        if n >= 15:
            cash = sum(t[0] for t in tickets)
            rate = cash / n
            avg_dec = np.mean([t[1] for t in tickets])
            results.append((f"+{lo}-{hi}", n, rate, avg_dec))
        lo += step
    # rank by hit rate (Option 1), highest first
    results.sort(key=lambda x: -x[2])
    for label, n, rate, avg_dec in results:
        am = (avg_dec - 1) * 100
        am_s = f"+{am:.0f}"
        win = 0.25 * (avg_dec - 1)
        print(f"{label:<16}{n:>9}{rate*100:>7.1f}%{am_s:>14}{'$'+format(win,'.2f'):>12}")
    if results:
        best = results[0]
        print(f"\n  -> highest hit rate: {best[0]} at {best[2]*100:.1f}% "
              f"(pays ~+{(best[3]-1)*100:.0f}, ${0.25*(best[3]-1):.2f} on 25c)")
    print("  windows overlap (sliding); pick the range whose hit%/payout you like.")


def run_for_legs(df, legs):
    tag = f"{legs}-LEG"

    # ── what you asked for: parlays grouped by ODDS band, hit rate + payout ──
    odds_order = ["≤+300", "+301-499", "+500-699", "+700+"]
    bands_o = [(b, make_combos(df[df["oband_wide"] == b], legs=legs)) for b in odds_order]
    report_simple(f"{tag} PARLAYS BY ODDS BAND — hit rate + payout", bands_o)

    # also the good-score pools, hit rate + payout
    pools = [
        ("13-15",       make_combos(df[df["score"].between(13, 15, inclusive="left")], legs=legs)),
        ("13-15 ≤+400", make_combos(df[(df["score"].between(13,15,inclusive="left")) & (df["odds"]<=400)], legs=legs)),
        ("Great combo",  make_combos(df[df.get("combo_val", 0).between(3, 4, inclusive="left")] if "combo_val" in df.columns else df.iloc[0:0], legs=legs)),
    ]
    report_simple(f"{tag} GOOD-POOL PARLAYS — hit rate + payout", pools)

    score_order = ["15+", "14-15", "13-14", "12-13", "11-12", "10-11"]
    bands_a = [(b, make_combos(df[df["sband"] == b], legs=legs)) for b in score_order]
    report(f"{tag} PARLAYS BY SCORE BAND (with edge/ROI for reference)", bands_a)


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved, priced rows across "
          f"{df['date'].nunique()} days.")

    df["sband"] = df["score"].apply(score_band)
    df["eband"] = df["edge_txt"].apply(edge_band)
    df["oband_wide"] = df["odds"].apply(odds_band_wide)

    print("\n" + "#"*76 + "\n# ODDS-WINDOW SWEEP — find the optimal range\n" + "#"*76)
    sweep_odds_windows(df, 2)
    sweep_odds_windows(df, 3)
    # also sweep within good scores (>=11) since those are the legs you'd use
    good = df[df["score"] >= 11]
    print("\n" + "#"*76 + "\n# SWEEP WITHIN GOOD SCORES (score >= 11)\n" + "#"*76)
    sweep_odds_windows(good, 2)
    sweep_odds_windows(good, 3)

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
