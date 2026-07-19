"""
test_hr_calibration_and_random.py

Two HR selection diagnostics in one pass, run against HR_All_Scores:

  1) CALIBRATION — are the model's predicted HR probabilities TRUE numbers?
     Buckets every scored leg by its predicted hit rate (deciles) and compares
     to the actual HR rate in each bucket. Parlays multiply probabilities, so
     miscalibration compounds: a "25%" leg that really hits 18% quietly turns a
     "+EV" parlay into a loser. Perfect calibration sits on the diagonal.

  2) BEAT-RANDOM — does Blend-1's RANKING add anything over the filter?
     Within the qualified parlay pool, compares the legs Blend-1 actually picks
     (top slots) against (a) random draws from the same pool and (b) the pool
     average. If Blend-1's slots don't beat random-from-pool, the ranking is
     decoration and the filter is the whole edge — which is the signal to stop
     tuning the selector and look at correlation instead.

Grades at LEG level (did the pick homer) — the fast-accumulating unit — with a
parlay-level readout flagged as underpowered.

NOTE: this measures SELECTION vs realized HRs only. It does not see the price
you got, so a clean result here still doesn't prove profitability (vig +
shopping edge live outside this data). It answers: is the selection layer sound?

Env knobs:
  SHEET_ENV_VAR   default GOOGLE_SHEET_ID
  MODEL_START     default 2026-06-09  (corrected-data era)
  N_RANDOM_TRIALS default 2000
  MIN_ZONE_N      default 15
Run locally against a CSV instead of Sheets:
  CSV=/path/to/HR_All_Scores.csv python test_hr_calibration_and_random.py
"""
import os
import json
import math
import random
from collections import defaultdict

import numpy as np
import pandas as pd

MODEL_START     = os.environ.get("MODEL_START", "2026-06-09")
N_RANDOM_TRIALS = int(os.environ.get("N_RANDOM_TRIALS", "2000"))
MIN_ZONE_N      = int(os.environ.get("MIN_ZONE_N", "15"))

# 2-leg pool = +301-400 (or <=+300 & 13+); 3-leg pool = +351-500 (or <=+300 & 13+).
# We grade legs in the UNION so the pool matches what the dashboard can draw from.
SLOT_2LEG = [(3, 4), (5, 6)]   # dashboard's proven 2-leg slots
TOP_3LEG  = 3                  # dashboard's 3-leg picks top 3 by selector


# ── Data loading ────────────────────────────────────────────────────────────
def load_scores() -> pd.DataFrame:
    csv = os.environ.get("CSV")
    if csv:
        print(f"Loading {csv}")
        return pd.read_csv(csv, dtype=str)
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ[os.environ.get("SHEET_ENV_VAR", "GOOGLE_SHEET_ID")])
    vals = sh.worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(vals[1:], columns=vals[0])


def sf(v, d=0.0):
    try:
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return d


# ── Shared model logic (mirrors dashboard.py) ───────────────────────────────
def odds_zone(o):
    if o <= 300:  return "le300"
    if o <= 499:  return "301-499"
    if o <= 699:  return "500-699"
    return "700plus"


def score_tier(s):
    if s >= 15:  return "15+"
    if s >= 14:  return "14-15"
    if s >= 13:  return "13-14"
    if s >= 12:  return "12-13"
    if s >= 11:  return "11-12"
    if s >= 10:  return "10-11"
    if s >= 9:   return "9-10"
    if s >= 8.5: return "8.5-9"
    return "below-8.5"


def implied(o):
    return 100.0 / (o + 100.0) if o >= 0 else abs(o) / (abs(o) + 100.0)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START)]
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if df.empty:
        return df
    df["score"] = df.apply(
        lambda r: sf(r.get("hr_score_corrected"))
        if str(r.get("hr_score_corrected", "")).strip() not in ("", "nan", "None")
        else sf(r.get("hr_score")), axis=1)
    df["odds"] = df["consensus_odds"].apply(sf)
    df["hit"] = (df["hit_hr"].astype(str).str.strip() == "Yes").astype(int)
    df["hr_per_fb"] = df.get("hr_per_fb", pd.Series(index=df.index)).apply(sf)
    df["tier"] = df["score"].apply(score_tier)
    df["zone"] = df["odds"].apply(odds_zone)
    df = df[(df["odds"] > 0) & (df["score"] > 0)]
    return df


def build_lookup(df: pd.DataFrame) -> dict:
    """tier x zone -> predicted hit rate (needs MIN_ZONE_N sample)."""
    lut = {}
    for (t, z), g in df.groupby(["tier", "zone"]):
        if len(g) >= MIN_ZONE_N:
            lut[(t, z)] = g["hit"].mean()
    return lut


def predicted_rate(row, lut):
    return lut.get((row["tier"], row["zone"]))


def selector(row, lut):
    pred = predicted_rate(row, lut)
    if pred is None:
        return None
    edge = (pred - implied(row["odds"])) * 100.0   # edge in pct points, like dashboard
    return (row["hr_per_fb"] / 8.0) + (edge * 0.8)


# ── Stats helpers ───────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


# ── 1) CALIBRATION ──────────────────────────────────────────────────────────
def run_calibration(df, lut):
    print("\n" + "=" * 66)
    print("  CALIBRATION — predicted HR% vs actual HR% (leg level)")
    print("=" * 66)
    rows = []
    for _, r in df.iterrows():
        p = predicted_rate(r, lut)
        if p is not None:
            rows.append((p, r["hit"]))
    if not rows:
        print("  No legs with a zone-backed prediction. Need more resolved data.")
        return
    preds = np.array([p for p, _ in rows])
    hits  = np.array([h for _, h in rows])
    n = len(rows)

    print(f"  {n} legs with a zone-backed prediction "
          f"(tier x zone, {MIN_ZONE_N}+ sample).\n")
    print(f"  {'predicted':>10} {'legs':>6} {'actual':>8} {'95% CI':>16}  cal")
    print("  " + "-" * 54)
    edges = [i / 10 for i in range(0, 11)]  # deciles 0..1
    overall_pred, overall_act = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (preds >= lo) & (preds < hi) if hi < 1.0 else (preds >= lo) & (preds <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        act = hits[mask].mean()
        pred_mid = preds[mask].mean()
        lo_ci, hi_ci = wilson(int(hits[mask].sum()), cnt)
        # is predicted inside the actual CI? -> calibrated in that bucket
        cal = "ok" if (lo_ci <= pred_mid <= hi_ci) else ("HIGH" if pred_mid > act else "LOW")
        print(f"  {pred_mid*100:9.1f}% {cnt:6d} {act*100:7.1f}% "
              f"  [{lo_ci*100:4.1f},{hi_ci*100:4.1f}]   {cal}")
        overall_pred.append(pred_mid * cnt)
        overall_act.append(act * cnt)

    tot_pred = sum(overall_pred) / n
    tot_act  = sum(overall_act) / n
    bias = (tot_pred - tot_act) * 100
    # Brier + a simple calibration error
    brier = float(np.mean((preds - hits) ** 2))
    print("  " + "-" * 54)
    print(f"  mean predicted {tot_pred*100:.1f}%  vs  mean actual {tot_act*100:.1f}%"
          f"   (bias {bias:+.1f} pts)")
    print(f"  Brier score {brier:.4f}  (lower = sharper+better; ~0.16 baseline-ish)")
    if abs(bias) <= 1.0:
        print("  READ: probabilities are ~calibrated overall. Parlay pricing is trustworthy.")
    elif bias > 1.0:
        print("  READ: model runs HOT (predicts more HRs than happen). Parlays overpriced by "
              "you\n        -> you'll overpay for legs; true parlay EV below what backtest shows.")
    else:
        print("  READ: model runs COLD (predicts fewer than happen). Legs are underrated;\n"
              "        real edge may be larger than displayed.")


# ── 2) BEAT-RANDOM ──────────────────────────────────────────────────────────
def _pool(df, kind):
    if kind == "2leg":
        m = ((df["odds"].between(301, 400)) | ((df["odds"] <= 300) & (df["score"] >= 13)))
    else:  # 3leg
        m = ((df["odds"].between(351, 500)) | ((df["odds"] <= 300) & (df["score"] >= 13)))
    score_ok = (df["score"].between(9, 10, inclusive="left") |
                df["score"].between(12, 13, inclusive="left") |
                (df["score"] >= 13))
    return df[m & score_ok].copy()


def run_beat_random(df, lut, kind, top_k):
    print("\n" + "=" * 66)
    print(f"  BEAT-RANDOM — {kind} pool: does Blend-1 ranking beat luck?")
    print("=" * 66)
    pool = _pool(df, kind)
    # each day's slate is the natural unit; rank within day, take top_k legs
    picked_hits, picked_n = 0, 0
    per_day_pools = []
    for day, g in pool.groupby(pool["date_dt"].dt.date):
        g = g.copy()
        g["sel"] = g.apply(lambda r: selector(r, lut), axis=1)
        g = g[g["sel"].notna()]
        if len(g) < top_k:
            continue
        g = g.sort_values("sel", ascending=False)
        top = g.head(top_k)
        picked_hits += int(top["hit"].sum())
        picked_n += len(top)
        per_day_pools.append(g["hit"].tolist())

    if picked_n == 0:
        print(f"  Not enough {kind} pool days with {top_k}+ candidates yet.")
        return

    picked_rate = picked_hits / picked_n
    pool_all = [h for day in per_day_pools for h in day]
    pool_rate = float(np.mean(pool_all))

    # random: each trial, draw top_k at random per day, average leg hit rate
    rng = random.Random(42)
    rand_rates = []
    for _ in range(N_RANDOM_TRIALS):
        hh, nn = 0, 0
        for day in per_day_pools:
            k = min(top_k, len(day))
            draw = rng.sample(day, k)
            hh += sum(draw)
            nn += k
        rand_rates.append(hh / nn)
    rand_rates = np.array(rand_rates)
    rand_mean = rand_rates.mean()
    pct_beaten = float((rand_rates < picked_rate).mean())  # share of random draws Blend-1 beats
    p_value = float((rand_rates >= picked_rate).mean())    # one-sided: random >= picked

    lo_ci, hi_ci = wilson(picked_hits, picked_n)
    print(f"  days used: {len(per_day_pools)} | Blend-1 legs graded: {picked_n}\n")
    print(f"  Blend-1 top-{top_k} leg hit rate : {picked_rate*100:5.1f}%  "
          f"[{lo_ci*100:.1f}, {hi_ci*100:.1f}] 95% CI")
    print(f"  pool average (all qualified)    : {pool_rate*100:5.1f}%")
    print(f"  random top-{top_k} draws (mean)      : {rand_mean*100:5.1f}%  "
          f"({N_RANDOM_TRIALS} trials)")
    print(f"  Blend-1 beats {pct_beaten*100:.1f}% of random draws   (p={p_value:.3f})")
    lift = (picked_rate - rand_mean) * 100
    print("  " + "-" * 54)
    if picked_n < 40:
        print(f"  READ: only {picked_n} legs — UNDERPOWERED. Treat as directional, not proof.")
    if p_value <= 0.05 and lift > 0:
        print(f"  READ: ranking adds real signal (+{lift:.1f} pts vs random, significant).\n"
              "        Keep the selector; it's earning its place.")
    elif lift > 0:
        print(f"  READ: ranking looks positive (+{lift:.1f} pts) but not yet significant.\n"
              "        Promising — recheck as the sample grows.")
    else:
        print(f"  READ: ranking does NOT beat random ({lift:+.1f} pts). The FILTER is the edge,\n"
              "        not the ordering. Stop tuning the selector — pull the correlation\n"
              "        lever (same-game / same-day-conditions legs) instead.")


# ── parlay-level secondary read (flagged underpowered) ──────────────────────
def run_parlay_readout(df, lut):
    print("\n" + "=" * 66)
    print("  PARLAY-LEVEL READOUT (secondary — expect small sample)")
    print("=" * 66)
    for kind, k in [("3leg", 3), ("2leg", 2)]:
        pool = _pool(df, kind)
        wins, tickets = 0, 0
        for day, g in pool.groupby(pool["date_dt"].dt.date):
            g = g.copy()
            g["sel"] = g.apply(lambda r: selector(r, lut), axis=1)
            g = g[g["sel"].notna()].sort_values("sel", ascending=False)
            if kind == "3leg":
                legs = g.head(3)
                if len(legs) == 3:
                    tickets += 1
                    wins += int(legs["hit"].sum() == 3)
            else:
                # emulate dashboard slots (3,4)/(5,6)
                for i, j in SLOT_2LEG:
                    if j < len(g):
                        legs = g.iloc[[i, j]]
                        tickets += 1
                        wins += int(legs["hit"].sum() == 2)
        rate = (wins / tickets * 100) if tickets else 0.0
        print(f"  {kind}: {wins}/{tickets} tickets cashed ({rate:.1f}%)  "
              f"-- {'TOO FEW to judge' if tickets < 30 else 'still thin'}")


def main():
    raw = load_scores()
    df = prep(raw)
    if df.empty:
        print("No resolved rows after MODEL_START. Nothing to test yet.")
        return
    print(f"Resolved legs since {MODEL_START}: {len(df)}")
    lut = build_lookup(df)
    print(f"Zone-backed predictions available: {len(lut)} tier x zone cells "
          f"({MIN_ZONE_N}+ sample each)")

    run_calibration(df, lut)
    run_beat_random(df, lut, "3leg", TOP_3LEG)
    run_beat_random(df, lut, "2leg", 6)   # 2-leg pool needs top-6 for slots (3,4)/(5,6)
    run_parlay_readout(df, lut)

    print("\n" + "=" * 66)
    print("  BOTTOM LINE")
    print("=" * 66)
    print("  • Calibration tells you if your parlay PRICES are honest.")
    print("  • Beat-random tells you if your RANKING is worth keeping.")
    print("  • Neither sees the price you got — profitability also needs the")
    print("    consensus-vs-taken logging. Run this again as the sample grows.")


if __name__ == "__main__":
    main()
