#!/usr/bin/env python3
"""
test_component_weights.py

Platoon and pitch-matchup tiers are noisy/non-monotonic on current data
(HR_Analysis screenshot: platoon bounces 10-14% with no clean gradient; pitch
similar). Power composite currently has NO weight lever (added at ~1.0
implicitly) while platoon (1.2) and pitch (1.9) carry the two biggest weights
on the board. Before touching any weight, measure each component's actual
separation power head-to-head so reweighting is evidence-based, not a hunch.

For each of platoon_score, pitch_matchup_score, and a recomputed power
composite: top-quartile vs bottom-quartile hit rate and effect size, using
the SAME methodology throughout this project.

USAGE
    python3 test_component_weights.py            # live
    python3 test_component_weights.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"

try:
    import hr_picks as HP
    HAVE_HP = True
except Exception as e:
    print(f"WARNING: could not import hr_picks ({e}); power composite recompute disabled.")
    HAVE_HP = False


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
    for c in ["platoon_score", "pitch_matchup_score", "season_barrel_pct",
              "hr_per_fb", "hr_per_pa", "iso", "pa"]:
        df[c] = _col(df, c).apply(sf)
    return df


def separation(name, sig, hit):
    d = pd.DataFrame({"s": sig, "hit": hit}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 100:
        print(f"  {name:<22} n={len(d):<6} too few"); return None
    hi = d["s"].quantile(0.75); lo = d["s"].quantile(0.25)
    top = d[d["s"] >= hi]["hit"].mean() * 100
    bot = d[d["s"] <= lo]["hit"].mean() * 100
    pooled = d["s"].std()
    eff = (d[d["hit"]==1]["s"].mean() - d[d["hit"]==0]["s"].mean()) / pooled if pooled > 0 else 0
    print(f"  {name:<22} n={len(d):<6} topQ {top:5.1f}%  botQ {bot:5.1f}%  spread {top-bot:+5.1f}  eff {eff:+.3f}")
    return top - bot


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} model-era resolved rows, base {df['hit'].mean()*100:.1f}%")
    print(f"\n{'='*76}\nCOMPONENT SEPARATION HEAD-TO-HEAD\n{'='*76}")
    print("  (bigger spread/eff = more useful signal for the score to weight)\n")

    results = {}
    results["platoon"] = separation("platoon_score", df["platoon_score"], df["hit"])
    results["pitch"] = separation("pitch_matchup_score", df["pitch_matchup_score"], df["hit"])
    results["combined"] = separation("combined (plat+pitch)",
                    df["platoon_score"] + df["pitch_matchup_score"], df["hit"])

    if HAVE_HP:
        pc = df.apply(lambda r: HP.score_power_composite(
            r["season_barrel_pct"], r["hr_per_fb"], r["hr_per_pa"], r["iso"],
            r["pa"] if not pd.isna(r["pa"]) else 100), axis=1)
        results["power"] = separation("power composite (recalc)", pc, df["hit"])
    else:
        print("  power composite         (skipped - hr_picks not importable)")

    sc = df.apply(resolve_score, axis=1)
    results["full_score"] = separation("full HR score (all combined)", sc, df["hit"])

    print("""
==============================================================================
READ
==============================================================================
  Rank by 'spread' (topQ - botQ hit%). Whichever component has the BIGGEST
  spread is doing the most real ranking work and may deserve MORE weight.
  Whichever has the smallest/noisiest spread (matches the flat, bouncing
  HR_Analysis tables) is doing the least and may deserve LESS. Compare each
  to the full score's spread as the ceiling -- no single component should
  exceed it, but the RELATIVE order tells you where to shift weight.""")


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
