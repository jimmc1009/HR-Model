"""
test_hr_leg_correlation.py

Beat-random said the RANKING adds little inside the qualified pool. The one
parlay-specific lever left is CORRELATION: books price parlays roughly as if
legs are independent, so if the legs you actually stack hit TOGETHER more often
than the independent product predicts, the true P(all hit) beats the parlay
price — a real, mechanism-backed edge that has nothing to do with leg selection.

This measures, out-of-sample against HR_All_Scores, whether HR outcomes are
positively correlated along axes you can actually stack on:

  A) SAME GAME      — two qualified legs in the same game (same home_team/day).
  B) SHARED PARK    — legs sharing a hitter-friendly park (park_hr_factor high).
  C) WIND OUT       — legs on days/games with wind blowing out.
  D) HIGH-TOTAL-ish — proxy via weather boost, if present.

For each axis it compares:
  observed P(both hit | paired on axis)   vs   independence baseline p1*p2
A positive lift means stacking on that axis genuinely beats the book's implied
independence — i.e. build parlays FROM that axis, not from top-ranked legs.

Grades pairs (not parlays) so the sample is usable. Uses realized HRs only —
it shows where correlation EXISTS; turning it into priced edge still needs the
consensus-vs-taken logging.

Env: MODEL_START=2026-06-09  MIN_PAIR_N=30  N_PERM=2000
Local: CSV=/path/HR_All_Scores.csv python test_hr_leg_correlation.py
"""
import os
import json
import math
import random
import itertools
import numpy as np
import pandas as pd

MODEL_START = os.environ.get("MODEL_START", "2026-06-09")
MIN_PAIR_N  = int(os.environ.get("MIN_PAIR_N", "30"))
N_PERM      = int(os.environ.get("N_PERM", "2000"))


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
    df["day"] = df["date_dt"].dt.date
    df["park_hr_factor"] = df.get("park_hr_factor", pd.Series(index=df.index)).apply(lambda x: sf(x, 100))
    df["wind"] = df.get("wind_context", pd.Series(index=df.index)).astype(str).str.lower()
    df["weather"] = df.get("hr_weather_boost", pd.Series(index=df.index)).apply(sf)
    # game key: home_team if present, else park_name, else team-vs-pitcher day bucket
    for c in ["home_team", "park_name", "pitcher_name", "team"]:
        if c not in df.columns:
            df[c] = ""
    df["game_key"] = df["home_team"].astype(str).str.strip()
    df.loc[df["game_key"].isin(["", "nan"]), "game_key"] = df["park_name"].astype(str).str.strip()
    # qualified = the parlay pool (9-10 / 12-13 / 13+ at parlay-ish odds)
    m = ((df["odds"].between(301, 500)) | ((df["odds"] <= 300) & (df["score"] >= 13)))
    score_ok = (df["score"].between(9, 10, inclusive="left") |
                df["score"].between(12, 13, inclusive="left") |
                (df["score"] >= 13))
    df["qualified"] = (m & score_ok & (df["odds"] > 0)).astype(int)
    return df


def _pair_stats(pairs):
    """pairs: list of (h1, h2). Return observed P(both) and independence p1*p2 baseline."""
    n = len(pairs)
    both = sum(1 for a, b in pairs if a and b)
    h1 = np.mean([a for a, _ in pairs])
    h2 = np.mean([b for _, b in pairs])
    return n, both / n, h1 * h2, h1, h2


def _perm_pvalue(pairs, observed_both_rate):
    """Break within-pair pairing by shuffling the second leg; how often does a
    reshuffled (independent) pairing match/exceed observed joint rate?"""
    a = np.array([x for x, _ in pairs])
    b = np.array([y for _, y in pairs])
    rng = random.Random(7)
    idx = list(range(len(b)))
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(idx)
        joint = np.mean(a * b[idx])
        if joint >= observed_both_rate:
            ge += 1
    return ge / N_PERM


def build_pairs(df, axis):
    """Yield (hit1, hit2) for every within-group qualified pair on the axis."""
    q = df[df["qualified"] == 1]
    pairs = []
    if axis == "same_game":
        for (_, _), g in q.groupby(["day", "game_key"]):
            hits = g["hit"].tolist()
            if len(hits) >= 2:
                pairs += list(itertools.combinations(hits, 2))
    elif axis == "hr_park":
        gp = q[q["park_hr_factor"] >= 108]
        for _, g in gp.groupby(["day", "game_key"]):
            hits = g["hit"].tolist()
            if len(hits) >= 2:
                pairs += list(itertools.combinations(hits, 2))
    elif axis == "wind_out":
        wo = q[q["wind"].str.contains("out", na=False)]
        for _, g in wo.groupby(["day", "game_key"]):
            hits = g["hit"].tolist()
            if len(hits) >= 2:
                pairs += list(itertools.combinations(hits, 2))
    elif axis == "high_weather":
        hw = q[q["weather"] >= 1.0]
        for _, g in hw.groupby(["day", "game_key"]):
            hits = g["hit"].tolist()
            if len(hits) >= 2:
                pairs += list(itertools.combinations(hits, 2))
    return pairs


def _park_bucket(pf):
    if pf >= 108:  return "hi"
    if pf <= 92:   return "lo"
    return "mid"


def run_wind_park_controlled(df):
    """
    Is the wind-out correlation real physics, or just park correlation wearing a
    wind label? Compare within the SAME park bucket:
        wind-out same-game pairs   vs   non-wind same-game pairs
    Both share the park's own baseline co-movement, so any REMAINING gap is
    wind-specific. If the wind lift survives here, it isn't a park artifact.
    """
    print("\n" + "-" * 60)
    print("  E) WIND-OUT, PARK-CONTROLLED (wind lift net of park's own")
    print("     baseline correlation — the real physics check)")
    q = df[df["qualified"] == 1].copy()
    q["pk"] = q["park_hr_factor"].apply(_park_bucket)
    q["is_wind"] = q["wind"].str.contains("out", na=False)

    wind_pairs, base_pairs = [], []
    for (_, _, _), g in q.groupby(["day", "game_key", "pk"]):
        hits = g["hit"].tolist()
        if len(hits) < 2:
            continue
        combos = list(itertools.combinations(hits, 2))
        # a game is a wind-out game if its legs are flagged wind-out
        if bool(g["is_wind"].iloc[0]):
            wind_pairs += combos
        else:
            base_pairs += combos

    if len(wind_pairs) < MIN_PAIR_N or len(base_pairs) < MIN_PAIR_N:
        print(f"    too few pairs (wind {len(wind_pairs)}, base {len(base_pairs)}; "
              f"need {MIN_PAIR_N} each). skip.")
        return None

    def excess(pairs):
        n = len(pairs)
        obs = sum(a and b for a, b in pairs) / n
        h1 = np.mean([a for a, _ in pairs]); h2 = np.mean([b for _, b in pairs])
        return n, obs, h1 * h2, (obs - h1 * h2)   # excess over independence

    nw, ow, iw, ew = excess(wind_pairs)
    nb, ob, ib, eb = excess(base_pairs)
    # difference-in-excess: wind excess minus non-wind excess, both within park strata
    did = (ew - eb) * 100

    # permutation: shuffle the wind/non-wind label across games, recompute DiD
    rng = random.Random(11)
    allg = [(list(itertools.combinations(g["hit"].tolist(), 2)), bool(g["is_wind"].iloc[0]))
            for (_, _, _), g in q.groupby(["day", "game_key", "pk"])
            if len(g) >= 2]
    labels = [w for _, w in allg]
    ge = 0
    for _ in range(N_PERM):
        rng.shuffle(labels)
        wp, bp = [], []
        for (combos, _), lab in zip(allg, labels):
            (wp if lab else bp).extend(combos)
        if not wp or not bp:
            continue
        _, ow2, iw2, ew2 = excess(wp)
        _, ob2, ib2, eb2 = excess(bp)
        if (ew2 - eb2) >= (ew - eb):
            ge += 1
    p = ge / N_PERM

    print(f"    wind-out pairs: {nw} | excess over indep {ew*100:+.2f} pts")
    print(f"    non-wind pairs: {nb} | excess over indep {eb*100:+.2f} pts (same park strata)")
    print(f"    wind-specific lift (DiD): {did:+.2f} pts   (p={p:.3f})")
    if p <= 0.05 and did > 0:
        print("    READ: wind lift SURVIVES park control — it's real physics, not a park\n"
              "          artifact. Wind-out stacking is a legitimate correlation edge.")
    elif did > 0:
        print("    READ: wind lift shrinks once park is controlled and is no longer\n"
              "          significant — likely mostly park correlation. Treat with caution.")
    else:
        print("    READ: no wind-specific lift after park control — the earlier wind\n"
              "          signal was park correlation mislabeled. Do NOT build a wind rule.")
    return {"axis": "wind_park_controlled", "n": nw, "lift": did, "p": p}


def run_axis(df, axis, label):
    pairs = build_pairs(df, axis)
    print("\n" + "-" * 60)
    print(f"  {label}")
    if len(pairs) < MIN_PAIR_N:
        print(f"    only {len(pairs)} qualified pairs — too few (need {MIN_PAIR_N}). skip.")
        return None
    n, obs, indep, h1, h2 = _pair_stats(pairs)
    lift = (obs - indep) * 100
    ratio = obs / indep if indep > 0 else float("nan")
    p = _perm_pvalue(pairs, obs)
    print(f"    pairs: {n} | leg hit rates ~{h1*100:.1f}% / {h2*100:.1f}%")
    print(f"    P(both hit) observed      : {obs*100:5.2f}%")
    print(f"    P(both) if independent    : {indep*100:5.2f}%   (p1 x p2)")
    print(f"    lift {lift:+.2f} pts   ratio {ratio:.2f}x   (p={p:.3f})")
    if p <= 0.05 and lift > 0:
        print(f"    READ: legs stacked on '{axis}' hit together MORE than the book assumes.\n"
              f"          Real correlation edge — build parlays from this axis.")
    elif lift > 0:
        print(f"    READ: positive but not significant. Watch as sample grows.")
    else:
        print(f"    READ: no positive correlation here — stacking this axis doesn't beat\n"
              f"          independence. Book prices it fairly.")
    return {"axis": axis, "n": n, "lift": lift, "ratio": ratio, "p": p}


def main():
    df = prep(load_scores())
    if df.empty:
        print("No resolved rows after MODEL_START.")
        return
    q = int(df["qualified"].sum())
    print(f"Resolved legs since {MODEL_START}: {len(df)} | qualified pool legs: {q}")
    print("=" * 60)
    print("  HR LEG CORRELATION — do co-stackable legs hit together")
    print("  more than the book's independence assumption implies?")
    print("=" * 60)

    results = []
    for axis, label in [
        ("same_game",    "A) SAME GAME — two qualified legs, same game"),
        ("hr_park",      "B) HR-FRIENDLY PARK (factor >= 108)"),
        ("wind_out",     "C) WIND BLOWING OUT"),
        ("high_weather", "D) HIGH WEATHER BOOST (>= 1.0)"),
    ]:
        r = run_axis(df, axis, label)
        if r:
            results.append(r)

    # Park-controlled wind check — only meaningful if raw wind looked positive,
    # but run it regardless so the answer is on record.
    wpc = run_wind_park_controlled(df)
    if wpc:
        results.append(wpc)

    print("\n" + "=" * 60)
    print("  BOTTOM LINE")
    print("=" * 60)
    hot = [r for r in results if r["p"] <= 0.05 and r["lift"] > 0]
    if hot:
        for r in sorted(hot, key=lambda x: -x["lift"]):
            print(f"  • '{r['axis']}' shows real positive correlation "
                  f"(+{r['lift']:.2f} pts, {r['ratio']:.2f}x, p={r['p']:.3f}).")
        print("  -> Build parlays by STACKING these axes, not by top-ranking legs.")
        print("     Correlated legs beat the independence-priced parlay line.")
    elif results:
        print("  No axis clears significance yet. Correlation edge is unproven —")
        print("  either it's absent (book prices these well) or the sample is still")
        print("  thin. Re-run as data grows before committing to a stacking rule.")
    else:
        print("  Not enough qualified pairs on any axis yet.")
    print("\n  Reminder: this shows where correlation EXISTS. Converting it to")
    print("  profit still needs the price you actually get vs the parlay line.")


if __name__ == "__main__":
    main()
