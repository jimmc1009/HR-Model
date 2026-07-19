"""
test_hr_calibration_and_random.py  —  WALK-FORWARD (out-of-sample)

Both diagnostics are graded strictly out-of-sample: for each game day D, the
tier x zone hit-rate lookup is built ONLY from resolved legs BEFORE D, then
applied to D's legs. Nothing is ever scored against a lookup that has seen it.
This is the only honest version — an in-sample lookup would grade the data
against itself and force perfect calibration by construction.

  1) CALIBRATION — bucket every out-of-sample leg by its predicted HR% (deciles)
     and compare to the actual HR% in that bucket. Tells you whether your parlay
     PRICES are honest. Miscalibration compounds across parlay legs.

  2) BEAT-RANDOM — within the qualified 2-leg / 3-leg pools, compare the legs
     Blend-1 ranks into its slots against random draws from the same pool and
     the pool average, per day, using only past-data selector values. Tells you
     whether the RANKING earns its keep or the filter is the whole edge.

Grades at LEG level (fast-accumulating). A parlay-level readout is included but
flagged underpowered.

Only sees SELECTION vs realized HRs — not the price you got. A clean pass still
doesn't prove profitability (vig + shopping edge live outside this data).

Env: MODEL_START=2026-06-09  MIN_TRAIN_DAYS=10  MIN_ZONE_N=15  N_RANDOM_TRIALS=2000
Local:  CSV=/path/HR_All_Scores.csv python test_hr_calibration_and_random.py
"""
import os
import json
import math
import random
import numpy as np
import pandas as pd

MODEL_START     = os.environ.get("MODEL_START", "2026-06-09")
MIN_TRAIN_DAYS  = int(os.environ.get("MIN_TRAIN_DAYS", "10"))   # skip early days w/ no history
MIN_ZONE_N      = int(os.environ.get("MIN_ZONE_N", "15"))       # min past sample for a cell
N_RANDOM_TRIALS = int(os.environ.get("N_RANDOM_TRIALS", "2000"))
SLOT_2LEG       = [(3, 4), (5, 6)]   # dashboard's proven 2-leg slots
TOP_3LEG        = 3


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


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


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
    df["day"] = df["date_dt"].dt.date
    return df.sort_values("date_dt")


def lookup_from(train: pd.DataFrame) -> dict:
    """tier x zone -> hit rate, from PAST data only (>= MIN_ZONE_N)."""
    lut = {}
    if train.empty:
        return lut
    for (t, z), g in train.groupby(["tier", "zone"]):
        if len(g) >= MIN_ZONE_N:
            lut[(t, z)] = g["hit"].mean()
    return lut


def build_walk_forward(df: pd.DataFrame):
    """
    For each day (after MIN_TRAIN_DAYS of history), attach to every leg:
      pred  = past-only tier x zone rate (None if no cell yet)
      sel   = hr_per_fb/8 + edge*0.8 using that past-only pred
    Returns the concatenated, out-of-sample-scored frame.
    """
    days = sorted(df["day"].unique())
    out = []
    for i, d in enumerate(days):
        if i < MIN_TRAIN_DAYS:
            continue
        train = df[df["day"] < d]
        lut = lookup_from(train)
        today = df[df["day"] == d].copy()
        preds, sels = [], []
        for _, r in today.iterrows():
            p = lut.get((r["tier"], r["zone"]))
            preds.append(p)
            if p is None:
                sels.append(np.nan)
            else:
                edge = (p - implied(r["odds"])) * 100.0
                sels.append((r["hr_per_fb"] / 8.0) + (edge * 0.8))
        today["pred"] = preds
        today["sel"] = sels
        out.append(today)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


# ── 1) CALIBRATION (out-of-sample) ──────────────────────────────────────────
def run_calibration(wf: pd.DataFrame):
    print("\n" + "=" * 66)
    print("  CALIBRATION (out-of-sample) — predicted HR% vs actual HR%")
    print("=" * 66)
    d = wf[wf["pred"].notna()]
    if d.empty:
        print("  No out-of-sample predictions yet. Need more history.")
        return
    preds = d["pred"].to_numpy(dtype=float)
    hits = d["hit"].to_numpy(dtype=float)
    n = len(d)
    print(f"  {n} legs scored against PAST-ONLY lookups.\n")
    print(f"  {'predicted':>10} {'legs':>6} {'actual':>8} {'95% CI':>16}  cal")
    print("  " + "-" * 54)
    edges = [i / 10 for i in range(0, 11)]
    tp = ta = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (preds >= lo) & (preds < hi) if hi < 1.0 else (preds >= lo) & (preds <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        act = hits[mask].mean()
        pmid = preds[mask].mean()
        lo_ci, hi_ci = wilson(int(hits[mask].sum()), cnt)
        cal = "ok" if (lo_ci <= pmid <= hi_ci) else ("HIGH" if pmid > act else "LOW")
        print(f"  {pmid*100:9.1f}% {cnt:6d} {act*100:7.1f}%   [{lo_ci*100:4.1f},{hi_ci*100:4.1f}]   {cal}")
        tp += pmid * cnt
        ta += act * cnt
    bias = (tp - ta) / n * 100
    brier = float(np.mean((preds - hits) ** 2))
    print("  " + "-" * 54)
    print(f"  mean predicted {tp/n*100:.1f}%  vs  mean actual {ta/n*100:.1f}%   (bias {bias:+.1f} pts)")
    print(f"  Brier {brier:.4f}  (lower = sharper; ~0.16 is coin-flip-ish for these rates)")
    if abs(bias) <= 1.0:
        print("  READ: probabilities hold up out-of-sample. Parlay pricing is honest.")
    elif bias > 1.0:
        print("  READ: runs HOT out-of-sample (predicts more HRs than happen). You overpay\n"
              "        for legs; true parlay EV is below what the in-app hit% implies.")
    else:
        print("  READ: runs COLD (predicts fewer than happen). Legs underrated; real edge\n"
              "        may exceed what's displayed.")
    # per-bucket miscalibration flag
    off = [b for b in ["HIGH", "LOW"]]  # noqa (kept for readability)


# ── 2) BEAT-RANDOM (out-of-sample) ──────────────────────────────────────────
def _pool_mask(df, kind):
    if kind == "2leg":
        m = (df["odds"].between(301, 400)) | ((df["odds"] <= 300) & (df["score"] >= 13))
    else:
        m = (df["odds"].between(351, 500)) | ((df["odds"] <= 300) & (df["score"] >= 13))
    score_ok = (df["score"].between(9, 10, inclusive="left") |
                df["score"].between(12, 13, inclusive="left") |
                (df["score"] >= 13))
    return m & score_ok


def run_beat_random(wf, kind, top_k):
    print("\n" + "=" * 66)
    print(f"  BEAT-RANDOM (out-of-sample) — {kind} pool: does ranking beat luck?")
    print("=" * 66)
    pool = wf[_pool_mask(wf, kind)].copy()
    picked_hits = picked_n = 0
    per_day = []
    for _, g in pool.groupby("day"):
        g = g[g["sel"].notna()]
        if len(g) < top_k:
            continue
        top = g.sort_values("sel", ascending=False).head(top_k)
        picked_hits += int(top["hit"].sum())
        picked_n += len(top)
        per_day.append(g["hit"].tolist())
    if picked_n == 0:
        print(f"  Not enough out-of-sample {kind} days with {top_k}+ candidates yet.")
        return
    picked_rate = picked_hits / picked_n
    pool_rate = float(np.mean([h for day in per_day for h in day]))
    rng = random.Random(42)
    rand = []
    for _ in range(N_RANDOM_TRIALS):
        hh = nn = 0
        for day in per_day:
            k = min(top_k, len(day))
            draw = rng.sample(day, k)
            hh += sum(draw); nn += k
        rand.append(hh / nn)
    rand = np.array(rand)
    p_value = float((rand >= picked_rate).mean())
    beaten = float((rand < picked_rate).mean())
    lo_ci, hi_ci = wilson(picked_hits, picked_n)
    lift = (picked_rate - rand.mean()) * 100
    print(f"  days used: {len(per_day)} | Blend-1 legs graded: {picked_n}\n")
    print(f"  Blend-1 top-{top_k} leg hit rate : {picked_rate*100:5.1f}%  [{lo_ci*100:.1f}, {hi_ci*100:.1f}] 95% CI")
    print(f"  pool average (all qualified)    : {pool_rate*100:5.1f}%")
    print(f"  random top-{top_k} draws (mean)      : {rand.mean()*100:5.1f}%  ({N_RANDOM_TRIALS} trials)")
    print(f"  Blend-1 beats {beaten*100:.1f}% of random draws   (p={p_value:.3f})")
    print("  " + "-" * 54)
    if picked_n < 40:
        print(f"  NOTE: only {picked_n} legs — underpowered; directional.")
    if p_value <= 0.05 and lift > 0:
        print(f"  READ: ranking adds real signal (+{lift:.1f} pts vs random, significant).\n"
              "        Keep the selector — it's earning its place.")
    elif lift > 0:
        print(f"  READ: ranking positive (+{lift:.1f} pts) but not significant. Recheck as it grows.")
    else:
        print(f"  READ: ranking does NOT beat random ({lift:+.1f} pts). The FILTER is the edge,\n"
              "        not the ordering. Stop tuning the selector — try correlation\n"
              "        (same-game / shared-conditions legs) instead.")


def run_parlay_readout(wf):
    print("\n" + "=" * 66)
    print("  PARLAY-LEVEL READOUT (secondary — expect small sample)")
    print("=" * 66)
    for kind, k in [("3leg", 3), ("2leg", 2)]:
        pool = wf[_pool_mask(wf, kind)].copy()
        wins = tickets = 0
        for _, g in pool.groupby("day"):
            g = g[g["sel"].notna()].sort_values("sel", ascending=False)
            if kind == "3leg":
                legs = g.head(3)
                if len(legs) == 3:
                    tickets += 1; wins += int(legs["hit"].sum() == 3)
            else:
                for i, j in SLOT_2LEG:
                    if j < len(g):
                        legs = g.iloc[[i, j]]
                        tickets += 1; wins += int(legs["hit"].sum() == 2)
        rate = (wins / tickets * 100) if tickets else 0.0
        tag = "TOO FEW to judge" if tickets < 30 else "still thin"
        print(f"  {kind}: {wins}/{tickets} tickets cashed ({rate:.1f}%)  -- {tag}")


def main():
    df = prep(load_scores())
    if df.empty:
        print("No resolved rows after MODEL_START.")
        return
    print(f"Resolved legs since {MODEL_START}: {len(df)} across {df['day'].nunique()} days")
    wf = build_walk_forward(df)
    if wf.empty:
        print(f"Not enough history yet (need > {MIN_TRAIN_DAYS} days before grading).")
        return
    print(f"Out-of-sample graded legs: {len(wf)} "
          f"(days {MIN_TRAIN_DAYS+1}..{df['day'].nunique()}, past-only lookups)")

    run_calibration(wf)
    run_beat_random(wf, "3leg", TOP_3LEG)
    run_beat_random(wf, "2leg", 6)
    run_parlay_readout(wf)

    print("\n" + "=" * 66)
    print("  BOTTOM LINE")
    print("=" * 66)
    print("  • Calibration -> are your parlay PRICES honest (out-of-sample)?")
    print("  • Beat-random -> is your RANKING worth keeping?")
    print("  • Neither sees the price you paid. Profitability still needs the")
    print("    consensus-vs-taken logging. Re-run as the sample grows.")


if __name__ == "__main__":
    main()
