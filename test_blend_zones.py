#!/usr/bin/env python3
"""
test_blend_zones.py

Re-verify the four proven "blend-1" zones' CURRENT single-leg hit rates (they
were measured 2026-07-12 and may have regressed), then test how 3-leg parlays
built from the pool actually cash — with attention to NOT stacking only cheap
<=+300 chalk (the user wants real payout, not three heavy favorites).

The four zones (score = HR score, odds = American):
    A. 13+   | <=+300           (was 34.3%)
    B. 12-13 | +301-400         (was 31.0%)
    C. 13+   | +301-400         (was 25.5%)
    D. 9-10  | +301-400         (was 22.3%)

Reports current hit rate + breakeven per zone, then 3-leg parlay hit rate +
payout under a few construction rules so we can pick one that pays.

USAGE
    python3 test_blend_zones.py            # live
    python3 test_blend_zones.py file.csv
"""

import sys, os, json, itertools
import numpy as np
import pandas as pd

MODEL_START = "2026-06-09"


def sf(v, d=np.nan):
    try:
        return float(str(v).replace("+", "").strip())
    except (ValueError, TypeError):
        return d


def _col(df, name):
    if name in df.columns:
        return df[name].astype(str)
    return pd.Series([""] * len(df), index=df.index)


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    # corrected score preferred
    def score_of(r):
        c = str(r.get("hr_score_corrected", "")).strip()
        if c not in ("", "nan", "None"):
            return sf(c)
        return sf(r.get("hr_score", ""))
    df["score"] = df.apply(score_of, axis=1)
    df["odds"] = _col(df, "consensus_odds").apply(sf)
    df["res"] = _col(df, "hit_hr").str.strip().str.lower()
    df["date"] = _col(df, "date").str.strip()
    df["game"] = _col(df, "pitcher_name").str.strip()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df = df.dropna(subset=["score", "odds"])
    df = df[df["odds"] > 0]
    # model-era only, matching how the zones were measured
    df = df[df["date"] >= MODEL_START]
    return df


# zone predicates
ZONES = {
    "A 13+ | <=+300":    lambda d: (d["score"] >= 13) & (d["odds"] <= 300),
    "B 12-13 | +301-400": lambda d: (d["score"] >= 12) & (d["score"] < 13) & (d["odds"] >= 301) & (d["odds"] <= 400),
    "C 13+ | +301-400":  lambda d: (d["score"] >= 13) & (d["odds"] >= 301) & (d["odds"] <= 400),
    "D 9-10 | +301-400": lambda d: (d["score"] >= 9) & (d["score"] < 10) & (d["odds"] >= 301) & (d["odds"] <= 400),
}
WAS = {"A 13+ | <=+300": 34.3, "B 12-13 | +301-400": 31.0,
       "C 13+ | +301-400": 25.5, "D 9-10 | +301-400": 22.3}


def breakeven(p):
    if p <= 0:
        return "--"
    return f"+{(1-p)/p*100:.0f}" if p < 0.5 else f"-{p/(1-p)*100:.0f}"


def dec(american):
    return 1 + (american / 100 if american > 0 else 100 / abs(american))


def verify_zones(df):
    print(f"\n{'='*74}\nCURRENT ZONE HIT RATES  (model era, n>=1)  vs 2026-07-12\n{'='*74}")
    print(f"{'zone':<22}{'n':>7}{'HR':>6}{'now%':>8}{'was%':>8}{'drift':>8}{'BE now':>9}")
    print("-"*74)
    pool_mask = pd.Series(False, index=df.index)
    for name, fn in ZONES.items():
        z = df[fn(df)]
        pool_mask |= fn(df)
        n = len(z)
        if n == 0:
            print(f"{name:<22}{0:>7}")
            continue
        hits = int(z["hit"].sum())
        now = hits / n * 100
        was = WAS[name]
        print(f"{name:<22}{n:>7}{hits:>6}{now:>7.1f}%{was:>7.1f}%{now-was:>+7.1f}{breakeven(now/100):>9}")
    return df[pool_mask].copy()


def combos_from(pool, legs, rule=None, exclude_same_game=True):
    """Build same-day N-leg parlays from pool. rule(list_of_recs)->bool filters
    which combos are allowed (e.g. not all <=+300)."""
    out = []
    for _, day in pool.groupby("date"):
        recs = day.to_dict("records")
        for combo in itertools.combinations(recs, legs):
            if exclude_same_game:
                gs = [c["game"] for c in combo if c["game"]]
                if len(gs) != len(set(gs)):
                    continue
            if rule and not rule(combo):
                continue
            cash = int(all(c["hit"] == 1 for c in combo))
            cdec = 1.0
            for c in combo:
                cdec *= dec(c["odds"])
            out.append((cash, cdec))
    return out


def report(title, tickets):
    n = len(tickets)
    if n < 10:
        print(f"{title:<40}{n:>8}   (too few)")
        return
    cash = sum(t[0] for t in tickets)
    rate = cash / n * 100
    avg_dec = np.mean([t[1] for t in tickets])
    am = (avg_dec - 1) * 100
    win = 0.25 * (avg_dec - 1)
    print(f"{title:<40}{n:>8}{rate:>7.1f}%   +{am:<7.0f} ${win:>6.2f}")


def test_parlays(pool):
    print(f"\n{'='*74}\n3-LEG PARLAYS FROM THE POOL — construction rules\n{'='*74}")
    print(f"{'rule':<40}{'tickets':>8}{'hit%':>8}{'payout':>10}{'$/25c':>8}")
    print("-"*74)

    def n_cheap(combo):
        return sum(1 for c in combo if c["odds"] <= 300)

    report("all combos (any mix)", combos_from(pool, 3))
    report("at most 1 leg <=+300 (avoid chalk)",
           combos_from(pool, 3, rule=lambda c: n_cheap(c) <= 1))
    report("no <=+300 legs (all +301-400)",
           combos_from(pool, 3, rule=lambda c: n_cheap(c) == 0))
    report("exactly 1 cheap anchor + 2 longer",
           combos_from(pool, 3, rule=lambda c: n_cheap(c) == 1))
    report("at least 1 leg score>=13",
           combos_from(pool, 3, rule=lambda c: any(x["score"] >= 13 for x in c)))
    report("all legs score>=12",
           combos_from(pool, 3, rule=lambda c: all(x["score"] >= 12 for x in c)))

    print(f"\n{'='*74}\n2-LEG (for reference)\n{'='*74}")
    print(f"{'rule':<40}{'tickets':>8}{'hit%':>8}{'payout':>10}{'$/25c':>8}")
    print("-"*74)
    report("all combos (any mix)", combos_from(pool, 2))
    report("no <=+300 legs", combos_from(pool, 2, rule=lambda c: sum(1 for x in c if x["odds"] <= 300) == 0))


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved model-era rows across {df['date'].nunique()} days.")
    pool = verify_zones(df)
    print(f"\nPool (any of the four zones): {len(pool)} legs across {pool['date'].nunique()} days"
          f"  (~{len(pool)/max(pool['date'].nunique(),1):.1f} legs/day)")
    test_parlays(pool)
    print(f"""
{'='*74}
READ
{'='*74}
  First table: did the four zones hold up since 7-12? Big negative drift =
  regressed to small-sample truth. Still-clearing-breakeven zones stay useful.
  Parlay table: '$/25c' is payout on a win; 'hit%' is how often it cashes.
  You wanted payout (not all <=+300 chalk) -- compare 'no <=+300 legs' and
  'at most 1 cheap' vs 'all combos'. Pick the rule with a hit%/payout you like.
  All in-sample; same-day combos reuse legs so hit% is optimistic.""")


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
