#!/usr/bin/env python3
"""
test_platoon_progressive.py

Builds a PROGRESSIVE platoon factor and backtests it head-to-head against the
CURRENT threshold-based platoon scoring, on resolved history. Decide from the
numbers whether progressive is worth wiring into hr_score.

CURRENT (threshold): compute_platoon_score gives ±fixed points at ISO-gap and
barrel cutoffs. Coarse.

PROGRESSIVE (this): a continuous factor =
    batter_level_vs_hand  ×  pitcher_vulnerability_vs_hand  ×  confidence
  - batter_level: z-blend of HR-rate, ISO, barrel% vs the hand he faces today
  - pitcher_vuln: z-blend of barrel% + HR-rate allowed to that hand
  - confidence:   shrinks toward 0 when batter BBE-vs-hand or pitcher IP-vs-hand
                  is thin (uses the real per-hand IP now in the data)

Backtest asks: which better separates HR from non-HR? Reports HR rate by
factor quartile, and correlation with actual HR, for OLD vs NEW.

USAGE
    python3 test_platoon_progressive.py            # live sheet
    python3 test_platoon_progressive.py file.csv
"""

import sys, os, json
import numpy as np
import pandas as pd

LEAGUE_ISO = 0.150
MIN_BBE = 15
BBE_FULL = 90
IP_FULL = 40.0   # pitcher IP-vs-hand for full confidence


def sf(v, d=0.0):
    try:
        x = float(str(v).strip()); return x if not np.isnan(x) else d
    except (ValueError, TypeError):
        return d


def regress(val, prior, n, n_full):
    if n <= 0:
        return prior
    w = min(n / n_full, 1.0)
    return val * w + prior * (1 - w)


def znorm(a):
    a = np.asarray(a, float)
    m, s = np.nanmean(a), np.nanstd(a)
    return (a - m) / s if s > 0 else np.zeros_like(a)


def effective_hand(bh, ph):
    if bh == "S":
        return "R" if ph == "L" else "L"
    return bh


# ── OLD: replicate the threshold platoon score (batter ISO gap + pitcher barrel)
def old_platoon(row):
    bh = str(row.get("batter_hand", "")).strip().upper()[:1]
    ph = str(row.get("pitcher_hand", "")).strip().upper()[:1]
    if ph not in ("L", "R"):
        return 0.0
    iso_l, iso_r = sf(row.get("vs_lhp_iso")), sf(row.get("vs_rhp_iso"))
    bbe_l, bbe_r = sf(row.get("vs_lhp_bbe")), sf(row.get("vs_rhp_bbe"))
    if ph == "L":
        iso_this, iso_opp, bbe_this, bbe_opp = iso_l, iso_r, bbe_l, bbe_r
    else:
        iso_this, iso_opp, bbe_this, bbe_opp = iso_r, iso_l, bbe_r, bbe_l
    score = 0.0
    if iso_this > 0 and iso_opp > 0 and min(bbe_this, bbe_opp) >= MIN_BBE:
        it = regress(iso_this, LEAGUE_ISO, bbe_this, BBE_FULL)
        io = regress(iso_opp, LEAGUE_ISO, bbe_opp, BBE_FULL)
        gap = it - io
        if gap >= 0.080:   score += 1.5
        elif gap >= 0.050: score += 0.8
        elif gap >= 0.025: score += 0.3
        elif gap <= -0.200: score -= 2.0
        elif gap <= -0.140: score -= 1.2
        elif gap <= -0.100: score -= 0.6
    eff = effective_hand(bh, ph)
    pb = sf(row.get("pitcher_barrel_vs_lhh")) if eff == "L" else sf(row.get("pitcher_barrel_vs_rhh"))
    if pb > 0:
        if pb >= 14:   score += 0.8
        elif pb >= 11: score += 0.4
        elif pb >= 9:  score += 0.2
        elif pb <= 4:  score -= 0.6
        elif pb <= 6:  score -= 0.3
    return score


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    df["res"] = df.get("hit_hr", "").astype(str).str.strip().str.lower()
    df = df[df["res"].isin(["yes", "no"])].copy()
    df["hit"] = (df["res"] == "yes").astype(int)
    df["bh"] = df.get("batter_hand", "").astype(str).str.strip().str.upper().str[:1]
    df["ph"] = df.get("pitcher_hand", "").astype(str).str.strip().str.upper().str[:1]
    return df


def progressive(df):
    """Vectorized progressive factor per row."""
    n = len(df)
    eff = np.array([effective_hand(b, p) for b, p in zip(df["bh"], df["ph"])])

    def col(name):
        return df.get(name, pd.Series([""]*n)).apply(sf).values

    # batter level vs the hand faced today
    iso_l, iso_r = col("vs_lhp_iso"), col("vs_rhp_iso")
    hr_l, hr_r = col("vs_lhp_hr_rate"), col("vs_rhp_hr_rate")
    bar_l, bar_r = col("vs_lhp_barrel_pct"), col("vs_rhp_barrel_pct")
    bbe_l, bbe_r = col("vs_lhp_bbe"), col("vs_rhp_bbe")
    faceL = df["ph"].values == "L"
    b_iso = np.where(faceL, iso_l, iso_r)
    b_hr = np.where(faceL, hr_l, hr_r)
    b_bar = np.where(faceL, bar_l, bar_r)
    b_bbe = np.where(faceL, bbe_l, bbe_r)
    batter_level = (znorm(b_iso) + znorm(b_hr) + znorm(b_bar)) / 3.0

    # pitcher vulnerability to that hand
    pbar = np.where(eff == "L", col("pitcher_barrel_vs_lhh"), col("pitcher_barrel_vs_rhh"))
    phr9 = np.where(eff == "L", col("pitcher_vs_lhh_hr9"), col("pitcher_vs_rhh_hr9"))
    pip = np.where(eff == "L", col("pitcher_vs_lhh_ip"), col("pitcher_vs_rhh_ip"))
    pitcher_vuln = (znorm(pbar) + znorm(phr9)) / 2.0

    # confidence: shrink when batter BBE or pitcher IP vs hand is thin
    conf_b = np.clip(b_bbe / BBE_FULL, 0, 1)
    conf_p = np.clip(pip / IP_FULL, 0, 1)
    conf = conf_b * conf_p

    return batter_level * pitcher_vuln * conf, batter_level, pitcher_vuln, conf


def report(df, col, name):
    print(f"\n{name}: HR rate by quartile")
    q = pd.qcut(df[col].rank(method="first"), 4, labels=["Q1 low", "Q2", "Q3", "Q4 high"])
    for lab in ["Q1 low", "Q2", "Q3", "Q4 high"]:
        sub = df[q == lab]
        print(f"  {lab:<8} HR {sub['hit'].mean()*100:5.1f}%  (n={len(sub)})")
    c = np.corrcoef(df[col].astype(float), df["hit"].astype(float))[0, 1]
    print(f"  corr with HR: {c:+.4f}")
    return c


def main(src):
    df = load(src)
    print(f"Resolved rows: {len(df)}  base HR {df['hit'].mean()*100:.1f}%")

    df["old"] = [old_platoon(r) for _, r in df.iterrows()]
    df["new"], bl, pv, cf = progressive(df)
    df["new"] = df["new"]

    print("\n" + "="*60)
    print("HEAD TO HEAD — does the platoon factor separate HR from non-HR?")
    print("="*60)
    co = report(df, "old", "OLD threshold platoon")
    cn = report(df, "new", "NEW progressive platoon")

    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)
    print(f"  OLD corr with HR: {co:+.4f}")
    print(f"  NEW corr with HR: {cn:+.4f}")
    if abs(cn) > abs(co) * 1.15:
        print("  -> NEW separates HR meaningfully better. Worth wiring into hr_score.")
    elif abs(cn) > abs(co):
        print("  -> NEW slightly better. Marginal — your call.")
    else:
        print("  -> NEW is NOT better. Keep the current threshold logic.")
    print("\n  (In-sample on history; IP-confidence only sharp on post-fix dates.")
    print("   Both computed from the same resolved rows, so the comparison is fair.)")


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    v = gspread.authorize(cr).open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(v[1:], columns=v[0])


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        main(sys.argv[1])
    else:
        print("Loaded HR_All_Scores live.\n")
        main(_sheet())
