#!/usr/bin/env python3
"""
inspect_matchup.py

Turns "the model took Murakami vs Fried and that's a terrible spot" into a
precise diagnosis. For a named batter and/or pitcher it prints every platoon
input the model had and the score it produced — so you can see exactly which
term failed to fire.

Also runs a WHIFF SCAN: finds matchups where the pitcher is clearly tough on
the batter's hand (low barrel% allowed, low HR9) but the platoon_score did NOT
dock it — the "terrible matchup that slipped through" cases.

Points at whatever GOOGLE_SHEET_ID you give it (set it to your DEV sheet).

USAGE
    python3 inspect_matchup.py                      # whiff scan on whole sheet
    python3 inspect_matchup.py "Murakami"           # one batter, all his rows
    python3 inspect_matchup.py "Murakami" "Fried"    # a specific matchup
    python3 inspect_matchup.py file.csv "Murakami"   # local CSV + filter
"""

import sys, os, json
import pandas as pd
import numpy as np


def sf(v, d=np.nan):
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return d


def load(src):
    df = src if isinstance(src, pd.DataFrame) else pd.read_csv(src, dtype=str)
    df = df.fillna("")
    df.columns = [c.strip() for c in df.columns]
    return df


# columns we care about for a platoon post-mortem
PLATOON_COLS = [
    "date", "player_name", "team", "pitcher_name", "pitcher_hand", "batter_hand",
    "hr_score", "platoon_score", "platoon_matchup", "consensus_odds", "hit_hr",
    "vs_lhp_iso", "vs_rhp_iso", "vs_lhp_hr_rate", "vs_rhp_hr_rate",
    "vs_lhp_barrel_pct", "vs_rhp_barrel_pct", "vs_lhp_bbe", "vs_rhp_bbe",
    "pitcher_barrel_vs_lhh", "pitcher_barrel_vs_rhh",
    "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
    "pitcher_vs_lhh_ip", "pitcher_vs_rhh_ip",
]


def eff_hand(bh, ph):
    if bh == "S":
        return "R" if ph == "L" else "L"
    return bh


def show_row(r):
    bh = str(r.get("batter_hand", "")).strip().upper()[:1]
    ph = str(r.get("pitcher_hand", "")).strip().upper()[:1]
    eff = eff_hand(bh, ph)
    faceL = ph == "L"
    b_iso = sf(r.get("vs_lhp_iso")) if faceL else sf(r.get("vs_rhp_iso"))
    b_hr = sf(r.get("vs_lhp_hr_rate")) if faceL else sf(r.get("vs_rhp_hr_rate"))
    b_bar = sf(r.get("vs_lhp_barrel_pct")) if faceL else sf(r.get("vs_rhp_barrel_pct"))
    b_bbe = sf(r.get("vs_lhp_bbe")) if faceL else sf(r.get("vs_rhp_bbe"))
    p_bar = sf(r.get("pitcher_barrel_vs_lhh")) if eff == "L" else sf(r.get("pitcher_barrel_vs_rhh"))
    p_hr9 = sf(r.get("pitcher_vs_lhh_hr9")) if eff == "L" else sf(r.get("pitcher_vs_rhh_hr9"))
    p_ip = sf(r.get("pitcher_vs_lhh_ip")) if eff == "L" else sf(r.get("pitcher_vs_rhh_ip"))

    print(f"\n{'='*64}")
    print(f"{r.get('player_name','?')} ({bh}HH) vs {r.get('pitcher_name','?')} ({ph}HP)"
          f"  [{r.get('date','?')}]")
    print(f"{'='*64}")
    print(f"  hr_score {r.get('hr_score','?')}   platoon_score {r.get('platoon_score','?')}"
          f"   odds {r.get('consensus_odds','?')}   result {r.get('hit_hr','?')}")
    print(f"  label: {r.get('platoon_matchup','(none)')}")
    print(f"\n  BATTER vs this hand ({'LHP' if faceL else 'RHP'}):")
    print(f"    ISO {b_iso:.3f}   HR-rate {b_hr}   barrel% {b_bar}   (BBE {b_bbe:.0f})")
    print(f"  PITCHER vs {eff}HH:")
    print(f"    barrel% allowed {p_bar}   HR9 {p_hr9}   (IP {p_ip})")

    # sanity flags
    flags = []
    if not np.isnan(p_bar) and p_bar <= 6:
        flags.append(f"pitcher SUPPRESSES {eff}HH (barrel {p_bar}%) — should be a NEGATIVE platoon term")
    if not np.isnan(p_hr9) and p_hr9 <= 0.7:
        flags.append(f"pitcher elite HR9 vs {eff}HH ({p_hr9}) — tough spot")
    ps = sf(r.get("platoon_score"))
    if flags and not np.isnan(ps) and ps >= 0:
        flags.append(f">>> but platoon_score is {ps:+.2f} (NOT docked) — term failed to fire")
    for f in flags:
        print(f"  ⚠️  {f}")


def whiff_scan(df):
    """Find bad matchups the score didn't dock: pitcher tough on the hand,
    but platoon_score >= 0."""
    print("\n" + "="*64)
    print("WHIFF SCAN — tough pitcher matchups the score did NOT dock")
    print("="*64)
    rows = []
    for _, r in df.iterrows():
        bh = str(r.get("batter_hand", "")).strip().upper()[:1]
        ph = str(r.get("pitcher_hand", "")).strip().upper()[:1]
        if ph not in ("L", "R"):
            continue
        eff = eff_hand(bh, ph)
        p_bar = sf(r.get("pitcher_barrel_vs_lhh")) if eff == "L" else sf(r.get("pitcher_barrel_vs_rhh"))
        ps = sf(r.get("platoon_score"))
        if not np.isnan(p_bar) and p_bar <= 5 and not np.isnan(ps) and ps >= 0:
            rows.append((r.get("date",""), r.get("player_name",""), r.get("pitcher_name",""),
                         eff, p_bar, ps, r.get("hr_score",""), r.get("hit_hr","")))
    if not rows:
        print("  none found — the score docked the tough spots it should have.")
        return
    print(f"  {len(rows)} matchups where pitcher barrel-vs-hand ≤5% but platoon_score ≥ 0:\n")
    print(f"  {'date':<11}{'batter':<18}{'pitcher':<18}{'hand':<5}{'pbar':>5}{'pscore':>8}{'score':>7}{'HR':>4}")
    for d, b, p, e, pb, ps, sc, hr in rows[:40]:
        print(f"  {str(d):<11}{str(b)[:17]:<18}{str(p)[:17]:<18}{e:<5}{pb:>5.1f}{ps:>+8.2f}{str(sc):>7}{str(hr):>4}")
    print(f"\n  If these are all platoon_score ~0 on ≤5% barrel pitchers, the")
    print(f"  pitcher-suppression penalty isn't firing hard enough (or at all).")


def main():
    args = sys.argv[1:]
    src = None
    names = []
    for a in args:
        if a.endswith(".csv") and os.path.exists(a):
            src = a
        else:
            names.append(a)
    if src is None:
        src = _sheet()
    df = load(src)
    print(f"Loaded {len(df)} rows.")

    if names:
        sub = df.copy()
        sub = sub[sub["player_name"].astype(str).str.contains(names[0], case=False, na=False)]
        if len(names) > 1:
            sub = sub[sub["pitcher_name"].astype(str).str.contains(names[1], case=False, na=False)]
        if sub.empty:
            print(f"No rows match {names}. Check spelling.")
            return
        for _, r in sub.head(20).iterrows():
            show_row(r)
    else:
        whiff_scan(df)


def _sheet():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    v = gspread.authorize(cr).open_by_key(os.environ["GOOGLE_SHEET_ID"]).worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(v[1:], columns=v[0])


if __name__ == "__main__":
    main()
