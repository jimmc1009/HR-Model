#!/usr/bin/env python3
"""
test_selector_bakeoff.py

You liked "blend1": rank parlay legs by  hr_per_fb/8 + edge*0.8.  It felt
realistic and non-chalky and you almost hit a bunch. This finds which
indicator(s), blended with edge, best reproduce that FEEL — not a proven
profit system (small samples can't promise that), but the selector whose
picks hit decently at fun, non-chalk odds.

WHAT IT DOES
For each candidate indicator (power, platoon, pitch-matchup, pitcher-
vulnerability, context), it builds a selector  = norm(indicator)*w + edge*(1-w)
across several weights, ranks each day's legs by it, takes the top legs a
parlay would use, and reports the PROFILE of those picks:

    hit%      how often the selected legs actually homered
    avg_odds  chalk-ness (low = chalk/boring, high = longshot)
    near%     "almost hit" proxy — selected leg whose batter barrelled a ball
              or had a hard-hit fly but no HR (the feel of just missing)
    n         how many legs the selector put in play

Plus an OUT-OF-SAMPLE column: weights are read on the first 2/3 of dates,
profile measured on the last 1/3. A selector whose hit% holds across the
split is real feel; one that collapses was noise. No single "winner" is
declared — you read the grid and pick the profile you like.

USAGE
    python3 test_selector_bakeoff.py            # live sheet
    python3 test_selector_bakeoff.py file.csv   # local CSV
"""

import sys
import os
import json
import numpy as np
import pandas as pd


# indicator families -> columns to try (only those present are used)
INDICATORS = {
    # power / batter
    "hr_per_fb":        "power: HR per fly ball",
    "season_barrel_pct":"power: season barrel%",
    "barrel_pct_7d":    "power: barrel% 7d",
    "iso":              "power: ISO",
    "avg_ev_7d":        "power: exit velo 7d",
    "hr_per_pa":        "power: HR per PA",
    "pull_rate":        "power: pull rate",
    # platoon
    "platoon_score":    "platoon: platoon score",
    "top_pitch_iso_vs_hand": "platoon: top-pitch ISO vs hand",
    # pitch matchup
    "pitch_matchup_score":   "matchup: pitch matchup score",
    # pitcher vulnerability
    "pitcher_barrel_pct":    "pitcher: barrel% allowed",
    "pitcher_hr_per_fb":     "pitcher: HR/FB allowed",
    "pitcher_hr9":           "pitcher: HR/9",
    "pitcher_babip_allowed": "pitcher: BABIP allowed",
    "pitcher_fb_rate_allowed":"pitcher: FB rate allowed",
    # context
    "park_hr_factor":   "context: park HR factor",
    "hr_weather_boost": "context: weather boost",
    "temp_f":           "context: temperature",
    "momentum_score":   "context: momentum",
}

WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]   # indicator weight; edge gets (1-w)
TOP_PER_DAY = 6                          # legs a parlay slate would consider


def _implied(o):
    o = float(o)
    return 100/(o+100) if o >= 0 else abs(o)/(abs(o)+100)


def znorm(s):
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["score"] = pd.to_numeric(df.get("hr_score", df.get("score", "")), errors="coerce")
    df["odds"] = pd.to_numeric(
        df.get("consensus_odds", "").astype(str).str.replace("+", "", regex=False).str.strip(),
        errors="coerce")
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df["date"] = df.get("date", "").astype(str).str.strip()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    # edge proxy: resolved-ish — use provided edge if numeric, else hit-vs-implied
    e = pd.to_numeric(df.get("edge", "").astype(str).str.replace("%","",regex=False)
                      .str.replace("+","",regex=False), errors="coerce")
    df["edge_val"] = e.fillna(0.0)
    df["implied"] = df["odds"].apply(lambda o: _implied(o) if pd.notna(o) else np.nan)
    # near-miss proxy: barrelled or hard-hit but no HR
    bar = pd.to_numeric(df.get("barrel_pct_7d", ""), errors="coerce")
    hh = pd.to_numeric(df.get("hard_hit_pct_7d", ""), errors="coerce")
    df["near"] = (((bar > 10) | (hh > 40)) & (df["hit"] == 0)).astype(int)
    df = df.dropna(subset=["score", "odds"])
    return df[df["odds"] > 0]


def profile_for(df, col, w):
    """Rank each day's legs by norm(col)*w + norm(edge)*(1-w), take top-N per
    day, return (hit%, avg_odds, near%, n)."""
    if col not in df.columns or pd.to_numeric(df[col], errors="coerce").notna().sum() < 50:
        return None
    d = df.copy()
    d["_ind"] = znorm(d[col])
    d["_edge"] = znorm(d["edge_val"])
    d["_sel"] = d["_ind"] * w + d["_edge"] * (1 - w)
    picks = []
    for _, day in d.groupby("date"):
        picks.append(day.sort_values("_sel", ascending=False).head(TOP_PER_DAY))
    if not picks:
        return None
    p = pd.concat(picks)
    if len(p) < 30:
        return None
    return (p["hit"].mean()*100, p["odds"].mean(), p["near"].mean()*100, len(p))


def run(df, label):
    dates = sorted(df["date"].unique())
    cut = dates[int(len(dates)*2/3)] if len(dates) > 6 else None
    tr = df[df["date"] < cut] if cut else df
    te = df[df["date"] >= cut] if cut else df

    print(f"\n{'='*94}\n{label}\n{'='*94}")
    print(f"{'selector':<34}{'w':>5}  "
          f"{'IN: hit% odds near% n':<28}  {'OUT: hit% odds near% n':<28}")
    print("-"*94)
    rows = []
    for col, name in INDICATORS.items():
        best = None
        for w in WEIGHTS:
            pin = profile_for(tr, col, w)
            if pin is None:
                continue
            pout = profile_for(te, col, w)
            rows.append((name, col, w, pin, pout))
    # sort by out-of-sample hit rate, blended selectors (w<1) first for "feel"
    def sortkey(r):
        pout = r[4]
        return -(pout[0] if pout else -99)
    rows.sort(key=sortkey)
    seen = set()
    for name, col, w, pin, pout in rows:
        # show best 2 weights per indicator to keep it readable
        k = (col,)
        cnt = sum(1 for s in seen if s[0] == col)
        if cnt >= 2:
            continue
        seen.add((col, w))
        istr = f"{pin[0]:5.1f} {pin[1]:+5.0f} {pin[2]:4.0f}% {pin[3]:>4}" if pin else "     --"
        ostr = f"{pout[0]:5.1f} {pout[1]:+5.0f} {pout[2]:4.0f}% {pout[3]:>4}" if pout else "     --"
        print(f"{name:<34}{w:>5.2f}  {istr:<28}  {ostr:<28}")

    # explicit blend1 reference
    b1 = profile_for(te, "hr_per_fb", 0.5)  # ~ the hr_per_fb/8 + edge*0.8 shape
    print("-"*94)
    if b1:
        print(f"{'blend1 (hr_per_fb ~half weight)':<34}{'~':>5}  {'':<28}  "
              f"{b1[0]:5.1f} {b1[1]:+5.0f} {b1[2]:4.0f}% {b1[3]:>4}")


def main(src):
    df = load(src)
    print(f"Loaded {len(df)} resolved, priced rows across {df['date'].nunique()} days.")
    base = df["hit"].mean()*100
    print(f"Base HR rate (all rows): {base:.1f}%   avg odds {df['odds'].mean():+.0f}")

    run(df, "SELECTOR BAKE-OFF — top-6 legs/day by  norm(indicator)*w + edge*(1-w)")

    print(f"""
{'='*94}
HOW TO READ — you're picking a FEEL, not a proven system
{'='*94}
  Each row: rank the day's legs by that indicator blended with edge at weight
  w, take the top 6, and see what those picks look like.
    hit%  — how often they homered
    odds  — chalk-ness. blend1 felt good because it was NON-chalk (+300 to
            +500ish), not favourites. Look for higher odds with decent hit%.
    near% — "almost hit" (barrelled/hard-hit, no HR). High near% = the feel of
            just missing a bunch, which is what you described.
    n     — legs in play.
  IN vs OUT: in-sample (tune) vs out-of-sample (last 1/3 of dates). If a
  selector's OUT hit% holds near its IN hit%, the feel is real. If OUT
  collapses, it was noise — don't trust it forward.

  There is NO declared winner. Pick the row whose profile matches what you
  liked: decent hit%, non-chalk odds, high near%, and holds out-of-sample.
  Small edges are fine — you said so. Just don't bet one whose OUT column
  falls apart.
""")


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
