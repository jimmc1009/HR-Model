#!/usr/bin/env python3
"""
test_edge_bands.py

Finds every score x odds cell where the RESOLVED hit rate outpaces the odds'
implied breakeven -- i.e. where the batter/pitcher model was right more often
than the price required. Ranked by edge (hit rate minus breakeven).

No CI gatekeeping (by request) -- raw edge. But n is printed on every row so
you can see which are real vs thin, and a fine odds grid is used so a
promising band isn't hidden inside a coarse bucket.

For each cell:
    hit%      resolved HR rate (from HR_All_Scores)
    BE%       breakeven implied by the cell's avg consensus odds
    edge      hit% - BE%   (positive = the band beats its price)
    ROI       flat-stake ROI betting every pick in the cell as a single
    n         picks in the cell

Also rolls the +edge single-leg cells into 2-leg parlay EV, so you can see
which bands are worth parlaying (edge compounds when legs clear breakeven).

USAGE
    python3 test_edge_bands.py            # live sheet
    python3 test_edge_bands.py file.csv   # local CSV
"""

import sys
import os
import json
import itertools
import pandas as pd
import numpy as np


def implied(odds):
    o = float(odds)
    return 100/(o+100) if o >= 0 else abs(o)/(abs(o)+100)


def dec(odds):
    o = float(odds)
    return 1 + (o/100 if o >= 0 else 100/abs(o))


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["score"] = pd.to_numeric(df.get("hr_score", ""), errors="coerce")
    df["odds"] = pd.to_numeric(
        df.get("consensus_odds", "").astype(str).str.replace("+", "", regex=False).str.strip(),
        errors="coerce")
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df["date"] = df.get("date", "").astype(str).str.strip()
    df["game"] = df.get("pitcher_name", "").astype(str).str.strip()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df = df.dropna(subset=["score", "odds"])
    return df[df["odds"] > 0]


# score bands and a FINE odds grid so promising bands aren't buried
SCORE_BANDS = [(15, 99, "15+"), (14, 15, "14-15"), (13, 14, "13-14"),
               (12, 13, "12-13"), (11, 12, "11-12"), (10, 11, "10-11"),
               (8.5, 10, "8.5-10")]
ODDS_BANDS = [(0, 250, "≤+250"), (251, 300, "+251-300"), (301, 350, "+301-350"),
              (351, 400, "+351-400"), (401, 450, "+401-450"), (451, 500, "+451-500"),
              (501, 600, "+501-600"), (601, 9999, "+601+")]


def cell_stats(sub):
    n = len(sub)
    if n == 0:
        return None
    hit = sub["hit"].mean()
    be = sub["odds"].apply(implied).mean()
    # flat-stake single ROI at each pick's own consensus price
    roi = np.mean([dec(o) - 1 if h else -1 for o, h in zip(sub["odds"], sub["hit"])]) * 100
    return {"n": n, "hit": hit, "be": be, "edge": hit - be, "roi": roi}


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved, priced rows across {df['date'].nunique()} days.\n")

    rows = []
    for slo, shi, slab in SCORE_BANDS:
        for olo, ohi, olab in ODDS_BANDS:
            sub = df[(df["score"] >= slo) & (df["score"] < shi) &
                     (df["odds"] >= olo) & (df["odds"] <= ohi)]
            st = cell_stats(sub)
            if st and st["n"] >= 8:
                st["score"] = slab; st["odds"] = olab
                rows.append(st)

    rows.sort(key=lambda r: r["edge"], reverse=True)

    print("=" * 86)
    print("ALL CELLS RANKED BY EDGE (resolved hit% minus implied breakeven)")
    print("=" * 86)
    print(f"{'score':<8}{'odds':<12}{'n':>5}{'hit%':>8}{'BE%':>8}{'edge':>8}{'ROI':>10}")
    print("-" * 86)
    for r in rows:
        mark = "  <<<" if r["edge"] > 0 else ""
        print(f"{r['score']:<8}{r['odds']:<12}{r['n']:>5}{r['hit']*100:>7.1f}%"
              f"{r['be']*100:>7.1f}%{r['edge']*100:>+7.1f}{r['roi']:>+9.1f}%{mark}")

    winners = [r for r in rows if r["edge"] > 0]
    print("\n" + "=" * 86)
    print(f"POSITIVE-EDGE BANDS: {len(winners)} of {len(rows)} cells")
    print("=" * 86)
    if not winners:
        print("  None. No cell's resolved hit rate beats its price.")
    else:
        print("  These are the bands where hitting outpaces the odds:\n")
        for r in winners:
            print(f"    {r['score']:<7} {r['odds']:<11} "
                  f"hit {r['hit']*100:.1f}% vs BE {r['be']*100:.1f}%  "
                  f"(+{r['edge']*100:.1f}pp, ROI {r['roi']:+.1f}%, n={r['n']})")

    # ── parlay the +edge legs ─────────────────────────────────────────
    if winners:
        print("\n" + "=" * 86)
        print("2 & 3-LEG PARLAYS FROM +EDGE LEGS ONLY  (same day, different games)")
        print("=" * 86)
        # tag every row's cell; keep legs whose cell had +edge
        win_cells = {(r["score"], r["odds"]) for r in winners}

        def which_cell(s, o):
            sl = next((lab for lo, hi, lab in SCORE_BANDS if lo <= s < hi), None)
            ol = next((lab for lo, hi, lab in ODDS_BANDS if lo <= o <= hi), None)
            return (sl, ol)

        df["cell"] = [which_cell(s, o) for s, o in zip(df["score"], df["odds"])]
        legs = df[df["cell"].isin(win_cells)]
        print(f"  eligible +edge legs: {len(legs)} across {legs['date'].nunique()} days")

        def roll(legs_df, k):
            tickets = []
            for _, day in legs_df.groupby("date"):
                recs = day.to_dict("records")
                for combo in itertools.combinations(recs, k):
                    games = [c["game"] for c in combo if c["game"]]
                    if len(games) != len(set(games)):   # any shared game
                        continue
                    cash = int(all(c["hit"] for c in combo))
                    cdec = 1.0; cimp = 1.0
                    for c in combo:
                        cdec *= dec(c["odds"]); cimp *= implied(c["odds"])
                    tickets.append((cash, cdec, cimp))
            return tickets

        for k in (2, 3):
            tickets = roll(legs, k)
            if len(tickets) >= 10:
                n = len(tickets)
                cash = sum(t[0] for t in tickets) / n
                be = np.mean([t[2] for t in tickets])
                roi = np.mean([(cd - 1) if c else -1 for c, cd, _ in tickets]) * 100
                print(f"  {k}-leg: tickets {n:>5}   cash {cash*100:5.1f}%   "
                      f"BE {be*100:5.1f}%   edge {(cash-be)*100:+5.1f}pp   ROI {roi:+.1f}%")
            else:
                print(f"  {k}-leg: only {len(tickets)} combos — too few to summarize.")

    print("\n" + "=" * 86)
    print("READ")
    print("=" * 86)
    print("""  '<<<' marks cells where resolved hit% beat the price. Those are the bands
  where the batter/pitcher model outpaced the odds -- your candidates.
  Raw edge, NO significance filter (as requested): small-n cells will show
  wild edges that are likely noise -- trust rows with n>=30 far more than
  n=8. A band that shows +edge AND has real n AND makes mechanistic sense is
  worth betting forward. In-sample; consensus odds; confirm before scaling.""")


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
