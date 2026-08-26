#!/usr/bin/env python3
"""
test_momentum_progressive.py

Before pushing the progressive momentum change live: backtest OLD (hard
2-of-3 agreement gate) vs NEW (continuous directional-consistency taper)
momentum, both recomputed from the SAME stored raw inputs on real resolved
history. Compares separation power (topQ vs botQ hit rate, effect size) so
the decision to switch is evidence-based, not just "the math is smoother."

Also reports COVERAGE: what % of rows get a nonzero momentum signal under
each version, since the old gate discarded real signal on cliff cases.

USAGE
    python3 test_momentum_progressive.py            # live
    python3 test_momentum_progressive.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"
LEAGUE_AVG_SEASON_BARREL = 8.0
MIN_PA_FULL = 150
MIN_BBE = 5
MIN_PA = 50


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").replace("%", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, n):
    return df[n].astype(str) if n in df.columns else pd.Series([""] * len(df), index=df.index)


def regress(value, league_avg, sample, full_sample):
    if pd.isna(value) or sample <= 0:
        return league_avg
    w = min(sample / full_sample, 1.0)
    return value * w + league_avg * (1 - w)


def build_signals(row):
    """Compute the 3 raw normalized signals (barrel/hard-hit/EV delta),
    same as score_momentum_delta's first half — shared by both old and new."""
    pa = row["pa"]
    if pd.isna(pa) or pa < MIN_PA:
        return None
    season_barrel = row["season_barrel_pct"]
    signals = []
    if not pd.isna(season_barrel) and season_barrel > 0:
        season_reg = regress(season_barrel, LEAGUE_AVG_SEASON_BARREL, pa, MIN_PA_FULL)
        deltas = []
        for b, bbe, w in [(row["barrel_pct_5d"], row["bbe_5d"], 0.30),
                          (row["barrel_pct_7d"], row["bbe_7d"], 0.40),
                          (row["barrel_pct_10d"], row["bbe_10d"], 0.30)]:
            if not pd.isna(bbe) and bbe >= MIN_BBE and not pd.isna(b):
                deltas.append((b - season_reg) * w)
        if deltas:
            tot_w = sum(w for b, bbe, w in [(row["barrel_pct_5d"], row["bbe_5d"], 0.30),
                        (row["barrel_pct_7d"], row["bbe_7d"], 0.40),
                        (row["barrel_pct_10d"], row["bbe_10d"], 0.30)]
                        if not pd.isna(bbe) and bbe >= MIN_BBE)
            blended = sum(deltas) / tot_w if tot_w else 0
            signals.append(max(-1.0, min(1.0, blended / 10.0)))
    hh_season = row["hard_hit_pct_season"]
    if not pd.isna(hh_season) and hh_season > 0 and not pd.isna(row["bbe_7d"]) and row["bbe_7d"] >= MIN_BBE:
        hh_delta = row["hard_hit_pct_7d"] - hh_season
        if not pd.isna(hh_delta):
            signals.append(max(-1.0, min(1.0, hh_delta / 15.0)))
    ev30 = row["avg_ev_30d"]
    if not pd.isna(ev30) and ev30 > 0 and not pd.isna(row["bbe_7d"]) and row["bbe_7d"] >= MIN_BBE:
        ev_delta = row["avg_ev_7d"] - ev30
        if not pd.isna(ev_delta):
            signals.append(max(-1.0, min(1.0, ev_delta / 3.0)))
    return signals if signals else None


def momentum_old(signals):
    n = len(signals)
    avg_signal = sum(signals) / n
    positives = sum(1 for s in signals if s > 0.1)
    negatives = sum(1 for s in signals if s < -0.1)
    if n >= 2:
        if positives < 2 and negatives < 2:
            return 0.0
    else:
        if abs(avg_signal) < 0.5:
            return 0.0
    mult = positives / n if avg_signal > 0 else negatives / n
    final = avg_signal * mult
    return max(-1.5, min(1.5, final))


def momentum_new(signals):
    n = len(signals)
    avg_signal = sum(signals) / n
    total_abs = sum(abs(s) for s in signals)
    if total_abs == 0:
        return 0.0
    consistency = abs(sum(signals)) / total_abs
    if n == 1:
        confidence = min(1.0, abs(avg_signal) / 0.5) * 0.7
    else:
        confidence = consistency
    final = avg_signal * confidence
    return max(-1.5, min(1.5, final))


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["date"] = _col(df, "date").str.strip()
    df = df[df["date"] >= MODEL_START]
    for c in ["pa", "season_barrel_pct", "barrel_pct_5d", "barrel_pct_7d", "barrel_pct_10d",
              "bbe_5d", "bbe_7d", "bbe_10d", "hard_hit_pct_season", "hard_hit_pct_7d",
              "avg_ev_30d", "avg_ev_7d"]:
        df[c] = _col(df, c).apply(sf)
    return df


def separation(name, sig, hit):
    d = pd.DataFrame({"s": sig, "hit": hit}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 100:
        print(f"  {name:<20} n={len(d):<6} too few"); return
    hi = d["s"].quantile(0.75); lo = d["s"].quantile(0.25)
    top = d[d["s"] >= hi]["hit"].mean() * 100
    bot = d[d["s"] <= lo]["hit"].mean() * 100
    pooled = d["s"].std()
    eff = (d[d["hit"]==1]["s"].mean() - d[d["hit"]==0]["s"].mean()) / pooled if pooled > 0 else 0
    nonzero = (d["s"].abs() >= 0.001).mean() * 100
    print(f"  {name:<20} n={len(d):<6} topQ {top:5.1f}%  botQ {bot:5.1f}%  spread {top-bot:+5.1f}  "
          f"eff {eff:+.3f}  nonzero {nonzero:.0f}%")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} model-era resolved rows, base {df['hit'].mean()*100:.1f}%")

    old_scores, new_scores = [], []
    for _, row in df.iterrows():
        sig = build_signals(row)
        if sig is None:
            old_scores.append(0.0); new_scores.append(0.0)
        else:
            old_scores.append(momentum_old(sig))
            new_scores.append(momentum_new(sig))
    df["mom_old"] = old_scores
    df["mom_new"] = new_scores

    print(f"\n{'='*78}\nOLD (hard gate) vs NEW (progressive) MOMENTUM — separation head-to-head\n{'='*78}")
    separation("momentum OLD", df["mom_old"], df["hit"])
    separation("momentum NEW", df["mom_new"], df["hit"])

    print(f"""
{'='*78}
READ
{'='*78}
  Bigger spread/eff = better separation. 'nonzero' = % of rows where the
  signal actually fired (not zeroed out) -- NEW should show more coverage
  since it doesn't hard-discard cliff cases.
  If NEW's spread/eff is >= OLD's: switch is evidence-backed, not just
  smoother math -- it's also a better (or equal) predictor.
  If NEW is clearly WORSE: the old hard gate was accidentally filtering
  useful noise, and we should keep the gate (or find a middle ground).""")


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
