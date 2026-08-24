#!/usr/bin/env python3
"""
test_recency_penalty.py

The scoring has a structural gap: power composite (season-level, no recency)
now carries double weight, but the only recency components (barrel windows)
can only ADD bonus for being hot — they return 0.0 for cold OR absent data,
identical to "mediocre." So a player like Buxton — elite season stats, zero
recent barrels — scores the same as if his recent form were average. There's
no active penalty for demonstrably cold recent performance.

This tests whether a "recency gap" signal (season power minus recent power)
actually separates HR outcomes. If guys whose recent barrels are well below
their season baseline underperform their score, a recency-cold penalty is
justified. If they hit at the same rate regardless, the season stats are
enough and recency doesn't matter for power.

Three candidate penalty signals tested:
  1) barrel_gap = season_barrel - barrel_7d (positive = cold relative to self)
  2) barrel_gap_broad = season_barrel - barrel_10d (wider window)
  3) power_vs_recent = power_composite - recent_barrel_contribution
     (how much of the score is "stale" season vs "live" recent)

USAGE
    python3 test_recency_penalty.py            # live
    python3 test_recency_penalty.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"
MIN_BBE = 5


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, n):
    return df[n].astype(str) if n in df.columns else pd.Series([""] * len(df), index=df.index)


def resolve_score(r):
    for c in ("hr_score_recalc", "hr_score_corrected", "hr_score"):
        v = str(r.get(c, "")).strip()
        if v not in ("", "nan", "None"):
            return sf(v)
    return np.nan


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["date"] = _col(df, "date").str.strip()
    df = df[df["date"] >= MODEL_START]
    for c in ["season_barrel_pct", "barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d",
              "bbe_7d", "bbe_5d", "bbe_10d", "hard_hit_pct_season", "hard_hit_pct_7d",
              "avg_ev_7d", "avg_ev_30d", "pa", "hr_per_fb", "hr_per_pa", "iso"]:
        df[c] = _col(df, c).apply(sf)
    df["score"] = df.apply(resolve_score, axis=1)
    return df


def separation(name, sig, hit, invert=False):
    """If invert=True, HIGHER signal = WORSE (cold), so we check if high-signal
    guys hit LESS (negative spread = the penalty would help)."""
    d = pd.DataFrame({"s": sig, "hit": hit}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 200:
        print(f"  {name:<32} n={len(d):<6} too few"); return
    hi = d["s"].quantile(0.75); lo = d["s"].quantile(0.25)
    top = d[d["s"] >= hi]["hit"].mean() * 100
    bot = d[d["s"] <= lo]["hit"].mean() * 100
    pooled = d["s"].std()
    eff = (d[d["hit"]==1]["s"].mean() - d[d["hit"]==0]["s"].mean()) / pooled if pooled > 0 else 0
    # for inverted signals, we WANT top < bot (cold guys hit less)
    direction = "CONFIRMS penalty" if (invert and top < bot) else \
                "NO penalty signal" if (invert and top >= bot) else ""
    print(f"  {name:<32} n={len(d):<6} topQ {top:5.1f}%  botQ {bot:5.1f}%  "
          f"spread {top-bot:+5.1f}  eff {eff:+.3f}  {direction}")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} model-era resolved rows, base {df['hit'].mean()*100:.1f}%")

    # only players with enough recent sample to measure a meaningful gap
    has_recent = df[(df["bbe_7d"] >= MIN_BBE) & (df["season_barrel_pct"] > 0)].copy()
    print(f"Players with bbe_7d >= {MIN_BBE} and season barrel > 0: {len(has_recent)}")

    print(f"\n{'='*78}\nRECENCY GAP SIGNALS — does being cold vs your own baseline predict fewer HR?\n{'='*78}")
    print("  (inverted: HIGHER gap = COLDER; want topQ < botQ = 'CONFIRMS penalty')\n")

    # 1) barrel gap: season minus 7d (positive = cold)
    has_recent["barrel_gap_7d"] = has_recent["season_barrel_pct"] - has_recent["barrel_pct_7d"]
    separation("barrel_gap (season - 7d)", has_recent["barrel_gap_7d"], has_recent["hit"], invert=True)

    # 2) barrel gap: season minus 10d (wider, more stable)
    has10 = has_recent[has_recent["bbe_10d"] >= MIN_BBE].copy()
    if len(has10) >= 200:
        has10["barrel_gap_10d"] = has10["season_barrel_pct"] - has10["barrel_pct_10d"]
        separation("barrel_gap (season - 10d)", has10["barrel_gap_10d"], has10["hit"], invert=True)

    # 3) just raw recent barrel (not gap — does low recent barrel = fewer HR?)
    separation("raw barrel_pct_7d (low=cold)", has_recent["barrel_pct_7d"], has_recent["hit"], invert=False)

    # 4) hard-hit gap
    has_hh = has_recent[has_recent["hard_hit_pct_season"] > 0].copy()
    if len(has_hh) >= 200:
        has_hh["hh_gap"] = has_hh["hard_hit_pct_season"] - has_hh["hard_hit_pct_7d"]
        separation("hard_hit gap (season - 7d)", has_hh["hh_gap"], has_hh["hit"], invert=True)

    # 5) EV gap
    has_ev = has_recent[(has_recent["avg_ev_30d"] > 0) & (has_recent["avg_ev_7d"] > 0)].copy()
    if len(has_ev) >= 200:
        has_ev["ev_gap"] = has_ev["avg_ev_30d"] - has_ev["avg_ev_7d"]
        separation("EV gap (30d - 7d)", has_ev["ev_gap"], has_ev["hit"], invert=True)

    # THE KEY TEST: among HIGH-SCORE players only (the Buxton case — elite
    # season stats), does the barrel gap separate HR outcomes?
    print(f"\n{'='*78}\nSAME TEST BUT ONLY HIGH-SCORE PLAYERS (top 20% by score)\n{'='*78}")
    print("  This isolates the Buxton case: elite season power, but is he cold?\n")
    sc_cut = has_recent["score"].quantile(0.80)
    elite = has_recent[has_recent["score"] >= sc_cut].copy()
    print(f"  High-score pool (score >= {sc_cut:.1f}): {len(elite)} players")
    if len(elite) >= 200:
        separation("barrel_gap_7d (elite only)", elite["barrel_gap_7d"], elite["hit"], invert=True)
        separation("raw barrel_7d (elite only)", elite["barrel_pct_7d"], elite["hit"], invert=False)
        if "hh_gap" in elite.columns:
            separation("hh_gap (elite only)", elite["hh_gap"], elite["hit"], invert=True)

    # ALSO: players with very LOW recent barrel (< 5%) but high season (> 10%)
    # — the extreme "stale power" case
    stale = has_recent[(has_recent["season_barrel_pct"] >= 10) & (has_recent["barrel_pct_7d"] < 5)].copy()
    print(f"\n{'='*78}\nEXTREME STALE POWER: season barrel >= 10%, recent 7d barrel < 5%\n{'='*78}")
    print(f"  n={len(stale)}, hit rate {stale['hit'].mean()*100:.1f}% vs base {df['hit'].mean()*100:.1f}%")
    if len(stale) >= 30:
        fresh = has_recent[(has_recent["season_barrel_pct"] >= 10) & (has_recent["barrel_pct_7d"] >= 10)].copy()
        print(f"  FRESH POWER (same season >=10%, recent 7d >=10%): n={len(fresh)}, "
              f"hit rate {fresh['hit'].mean()*100:.1f}%")
        print(f"  -> gap: stale {stale['hit'].mean()*100:.1f}% vs fresh {fresh['hit'].mean()*100:.1f}% "
              f"= {fresh['hit'].mean()*100 - stale['hit'].mean()*100:+.1f}pp")

    print(f"""
{'='*78}
READ
{'='*78}
  'CONFIRMS penalty' = cold guys (high gap) hit FEWER HR than hot guys (low
  gap) within the same signal. That justifies an active SUBTRACTION in the
  score for cold recent form, not just "no bonus."
  The EXTREME STALE POWER section is the Buxton test directly: same elite
  season barrel (>=10%), but recent 7d barrel <5% vs >=10%. If the stale
  group hits significantly LESS, that's the concrete evidence for penalizing
  players whose power composite is running on stale season stats.""")


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    v = gspread.authorize(cr).open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    print("Loaded HR_All_Scores live.")
    return pd.DataFrame(v[1:], columns=v[0])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        main(_sheet())
