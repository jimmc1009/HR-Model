"""
test_hr_parlay_bands.py  —  WALK-FORWARD odds-band sweep for 2-leg & 3-leg parlays

The current bands (2-leg +301-400, 3-leg +351-500) came from IN-SAMPLE ROI/EV
backtests — the same class of test that turned out circular for calibration.
This re-tests them out-of-sample: for each candidate odds band, build the
tier x zone lookup from PAST days only, select each day's legs from that band
using the live selector, grade forward, and tally realized ROI. Nothing is
graded against a lookup or a band choice that has seen it.

For each band it reports, at LEG level and TICKET level:
  • leg hit rate
  • ticket cash rate (all legs hit)
  • ROI per unit at the band's average payout  (the number that actually matters)

Then it prints which band wins out-of-sample for each parlay type, so you can
compare against the deployed +301-400 / +351-500 and see if they hold.

ROI uses realized parlay payout from the legs' own decimal odds. It does NOT
include your price-shopping edge or vig beyond the odds themselves, so treat ROI
as directional-but-honest: it answers "which band is best", not "exact profit".

Env: MODEL_START=2026-06-09  MIN_TRAIN_DAYS=10  MIN_ZONE_N=15
Local: CSV=/path/HR_All_Scores.csv python test_hr_parlay_bands.py
"""
import os
import json
import math
import numpy as np
import pandas as pd

MODEL_START    = os.environ.get("MODEL_START", "2026-06-09")
MIN_TRAIN_DAYS = int(os.environ.get("MIN_TRAIN_DAYS", "10"))
MIN_ZONE_N     = int(os.environ.get("MIN_ZONE_N", "15"))

# Candidate bands to sweep (low, high) inclusive, in American odds.
BANDS = [
    (251, 300),
    (301, 350),
    (301, 400),   # deployed 2-leg
    (351, 450),
    (351, 500),   # deployed 3-leg
    (401, 500),
    (451, 550),
    (501, 600),
]
DEPLOYED = {"2leg": (301, 400), "3leg": (351, 500)}
SLOT_2LEG = [(3, 4), (5, 6)]   # dashboard's 2-leg slots (needs top-6 ranked)
TOP_3LEG  = 3


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
    df["day"] = df["date_dt"].dt.date
    df = df[(df["odds"] > 0) & (df["score"] > 0)]
    # score gate = the qualified pool (9-10 / 12-13 / 13+)
    df["score_ok"] = (df["score"].between(9, 10, inclusive="left") |
                      df["score"].between(12, 13, inclusive="left") |
                      (df["score"] >= 13))
    return df.sort_values("date_dt")


def lookup_from(train):
    lut = {}
    if train.empty:
        return lut, 0.0
    base = train["hit"].mean()
    K = 40.0
    for (t, z), g in train.groupby(["tier", "zone"]):
        if len(g) >= MIN_ZONE_N:
            n = len(g)
            lut[(t, z)] = (n * g["hit"].mean() + K * base) / (n + K)
    return lut, base


def selector(row, lut):
    p = lut.get((row["tier"], row["zone"]))
    if p is None:
        return None
    return (row["hr_per_fb"] / 8.0) + ((p - implied(row["odds"])) * 100.0) * 0.8


def band_legs(day_df, lut, lo, hi):
    """Legs from this day inside [lo,hi] (plus the <=+300 & 13+ carve-out), ranked."""
    m = (day_df["odds"].between(lo, hi)) | ((day_df["odds"] <= 300) & (day_df["score"] >= 13))
    g = day_df[m & day_df["score_ok"]].copy()
    g["sel"] = g.apply(lambda r: selector(r, lut), axis=1)
    g = g[g["sel"].notna()].sort_values("sel", ascending=False)
    return g


def dec(o):
    return 1 + o / 100.0


def sweep(df, kind):
    days = sorted(df["day"].unique())
    print("\n" + "=" * 70)
    print(f"  {kind.upper()} — walk-forward band sweep")
    print("=" * 70)
    print(f"  {'band':>12} {'tickets':>8} {'legs':>6} {'leg hit%':>9} "
          f"{'cash%':>7} {'ROI/unit':>9}")
    print("  " + "-" * 58)

    rows = []
    for lo, hi in BANDS:
        tickets = 0
        cashes = 0
        leg_hits = 0
        leg_n = 0
        profit = 0.0
        for i, d in enumerate(days):
            if i < MIN_TRAIN_DAYS:
                continue
            lut, _ = lookup_from(df[df["day"] < d])
            g = band_legs(df[df["day"] == d], lut, lo, hi)
            if kind == "3leg":
                if len(g) >= 3:
                    legs = g.head(3)
                    tickets += 1
                    won = int(legs["hit"].sum() == 3)
                    cashes += won
                    leg_hits += int(legs["hit"].sum()); leg_n += 3
                    payout = np.prod([dec(o) for o in legs["odds"]]) - 1
                    profit += payout if won else -1.0
            else:  # 2leg — emulate both dashboard slots (3,4)/(5,6)
                for a, b in SLOT_2LEG:
                    if b < len(g):
                        legs = g.iloc[[a, b]]
                        tickets += 1
                        won = int(legs["hit"].sum() == 2)
                        cashes += won
                        leg_hits += int(legs["hit"].sum()); leg_n += 2
                        payout = dec(legs.iloc[0]["odds"]) * dec(legs.iloc[1]["odds"]) - 1
                        profit += payout if won else -1.0
        if tickets == 0:
            continue
        leg_rate = leg_hits / leg_n * 100 if leg_n else 0
        cash_rate = cashes / tickets * 100
        roi = profit / tickets * 100
        tag = "  <- deployed" if (lo, hi) == DEPLOYED[kind] else ""
        rows.append((lo, hi, tickets, roi, cash_rate, leg_rate))
        print(f"  {f'+{lo}-{hi}':>12} {tickets:8d} {leg_n:6d} {leg_rate:8.1f}% "
              f"{cash_rate:6.1f}% {roi:+8.1f}%{tag}")

    print("  " + "-" * 58)
    if not rows:
        print("  Not enough out-of-sample days yet.")
        return
    best = max(rows, key=lambda r: r[3])
    dep = next((r for r in rows if (r[0], r[1]) == DEPLOYED[kind]), None)
    print(f"  BEST band out-of-sample: +{best[0]}-{best[1]}  "
          f"(ROI {best[3]:+.1f}%, {best[2]} tickets)")
    if dep:
        gap = best[3] - dep[3]
        if (best[0], best[1]) == DEPLOYED[kind]:
            print(f"  READ: deployed band +{dep[0]}-{dep[1]} is still best. Holds out-of-sample.")
        elif gap < 5 or best[2] < 30:
            print(f"  READ: +{best[0]}-{best[1]} edges deployed by {gap:.1f} pts but on thin/close\n"
                  f"        evidence ({best[2]} tickets). Not enough to switch — keep deployed.")
        else:
            print(f"  READ: +{best[0]}-{best[1]} beats deployed +{dep[0]}-{dep[1]} by {gap:.1f} ROI pts\n"
                  f"        out-of-sample. Worth a closer look before switching.")
    print("  NOTE: ticket counts are small — ROI here is directional. Re-run as sample grows.")


def main():
    df = prep(load_scores())
    if df.empty:
        print("No resolved rows after MODEL_START.")
        return
    print(f"Resolved legs since {MODEL_START}: {len(df)} across {df['day'].nunique()} days")
    print(f"Walk-forward: first {MIN_TRAIN_DAYS} days are history-only, graded from day {MIN_TRAIN_DAYS+1}.")
    sweep(df, "2leg")
    sweep(df, "3leg")
    print("\n" + "=" * 70)
    print("  Bands are graded out-of-sample with the SAME selector + shrunk")
    print("  lookup the dashboard uses. ROI excludes your shopping edge, so it")
    print("  ranks bands honestly but isn't a profit forecast. Change a band")
    print("  only if a challenger wins clearly AND on a non-trivial sample.")
    print("=" * 70)


if __name__ == "__main__":
    main()
