"""
validate_model.py
ONE validation run for the HR model. Checks everything against the current
CORRECTED data and prints a prioritized "what needs attention" summary, so you
stop discovering problems one card at a time.

Six checks:
  1. SCORE INTEGRITY   — corrected columns populated? nulls? old vs new gap?
  2. ZONE HIT RATES    — every tier×odds zone vs true breakeven, flagged
  3. FEATURE SEPARATORS— which features actually separate hits, ranked
  4. SELECTOR CHECK    — does parlay selector weight match winning-leg reality?
  5. SINGLES EDGE GATE — are low-score players leaking into edge plays?
  6. CONSISTENCY       — how many rows lack corrected scores (dashboard blind spots)

Reads HR_All_Scores (uses hr_score_corrected / platoon_score_corrected when
present). Read-only — changes nothing. Run after any edit to see what broke.
"""

import os, json, time
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
MODEL_START_DATE = "2026-06-09"


def gc_client():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info, scopes=SCOPES))


def with_retry(fn, retries=4, wait=20):
    for a in range(retries):
        try: return fn()
        except Exception:
            if a < retries-1: time.sleep(wait)
            else: raise


def sf(v, d=0.0):
    try:
        f = float(v); return d if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError): return d


def coalesce(row, corr, orig):
    c = str(row.get(corr, "")).strip()
    return sf(c) if c not in ("", "nan", "None") else sf(row.get(orig))


def breakeven(odds):
    if odds >= 100: return 100.0/(odds+100.0)
    if odds < 0:    return abs(odds)/(abs(odds)+100.0)
    return 1.0


def zone_of(o):
    if o <= 300: return "≤+300"
    if o <= 499: return "+301-499"
    if o <= 699: return "+500-699"
    return "+700+"


def tier_of(s):
    if s >= 13: return "13+"
    if s >= 12: return "12-13"
    if s >= 11: return "11-12"
    if s >= 10: return "10-11"
    if s >= 9:  return "9-10"
    if s >= 8.5:return "8.5-9"
    return "<8.5"


ATTENTION = []  # collect flagged issues for the final summary


def flag(priority, msg):
    ATTENTION.append((priority, msg))


def main():
    sid = os.environ["GOOGLE_SHEET_ID"]
    gc  = gc_client()
    print("Reading HR_All_Scores...")
    sh  = with_retry(lambda: gc.open_by_key(sid))
    ws  = sh.worksheet("HR_All_Scores")
    vals= with_retry(lambda: ws.get_all_values())
    df  = pd.DataFrame(vals[1:], columns=vals[0])
    df  = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)].reset_index(drop=True)

    df["dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()

    has_corr = "hr_score_corrected" in df.columns
    df["score"] = df.apply(lambda r: coalesce(r, "hr_score_corrected", "hr_score"), axis=1) if has_corr \
                  else df["hr_score"].apply(sf)
    df["odds"]  = df["consensus_odds"].apply(lambda x: sf(x, 0))
    df["res"]   = df["hit_hr"].astype(str).str.strip()
    resolved = df[df["res"].isin(["Yes","No"])].copy()
    resolved["hit"] = resolved["res"] == "Yes"

    print(f"  {len(df)} rows total | {len(resolved)} resolved | corrected col present: {has_corr}\n")

    # ══ CHECK 1: SCORE INTEGRITY ══════════════════════════════════════════
    print("="*62); print("1. SCORE INTEGRITY"); print("="*62)
    if not has_corr:
        flag(1, "hr_score_corrected column MISSING — rescore never ran or was overwritten")
        print("  ❌ hr_score_corrected column not found")
    else:
        n_missing = df.apply(lambda r: str(r.get("hr_score_corrected","")).strip() in ("","nan","None"), axis=1).sum()
        pct = round(n_missing/len(df)*100,1)
        print(f"  Rows missing corrected score: {n_missing} ({pct}%)")
        if pct > 5:
            flag(1, f"{pct}% of rows lack hr_score_corrected — new rows may not be populating it")
        old = df["hr_score"].apply(sf); new = df["score"]
        moved = (np.abs(new-old) > 0.001).sum()
        print(f"  Rows where corrected != original: {moved} ({round(moved/len(df)*100,1)}%)")
    print()

    # ══ CHECK 2: ZONE HIT RATES vs BREAKEVEN ══════════════════════════════
    print("="*62); print("2. ZONE HIT RATES vs BREAKEVEN (bettable zones)"); print("="*62)
    resolved["tier"] = resolved["score"].apply(tier_of)
    resolved["zone"] = resolved["odds"].apply(zone_of)
    good_zones = []
    print(f"  {'tier | zone':<20} {'n':>4} {'hit%':>7} {'be%':>6} {'edge':>8}")
    for (t,z), sub in resolved.groupby(["tier","zone"]):
        if len(sub) < 15: continue
        hr = sub["hit"].mean()*100
        be = breakeven(sub["odds"].mean())*100
        edge = hr - be
        mark = "✅" if edge >= 3 else "❌" if edge <= -1 else "≈"
        print(f"  {t+' | '+z:<20} {len(sub):>4} {hr:>6.1f}% {be:>5.1f}% {edge:>+6.1f}% {mark}")
        if edge >= 3 and len(sub) >= 25:
            good_zones.append((t, z, round(hr,1), len(sub), round(edge,1)))
    print(f"\n  → {len(good_zones)} zones clear breakeven by +3% with 25+ sample:")
    for t,z,hr,n,e in sorted(good_zones, key=lambda x:-x[4]):
        print(f"     {t} | {z}: {hr}% (n={n}, +{e}% edge)")
    if not good_zones:
        flag(1, "NO zones clear breakeven with good sample — edge may be gone")
    else:
        flag(3, f"Bettable zones (verify dashboard/parlay use these): " +
              ", ".join(f"{t}|{z}" for t,z,_,_,_ in sorted(good_zones,key=lambda x:-x[4])[:6]))
    print()

    # ══ CHECK 3: FEATURE SEPARATORS ═══════════════════════════════════════
    print("="*62); print("3. FEATURE SEPARATORS (hit vs miss, ranked)"); print("="*62)
    # overlay corrected platoon
    if "platoon_score_corrected" in resolved.columns:
        resolved["platoon_score"] = resolved.apply(
            lambda r: coalesce(r, "platoon_score_corrected", "platoon_score"), axis=1)
    feats = ["iso","season_barrel_pct","hr_per_fb","hr_per_pa","barrel_pct_7d",
             "barrel_pct_5d","platoon_score","pitch_matchup_score","momentum_score"]
    seps = []
    for f in feats:
        if f not in resolved.columns: continue
        v = resolved[f].apply(lambda x: sf(x, np.nan))
        h = v[resolved["hit"]].mean(); m = v[~resolved["hit"]].mean()
        if pd.isna(h) or pd.isna(m) or m == 0: continue
        # Guard against near-zero-denominator blowup: features that center on
        # ~0 (like momentum_score) produce meaningless giant % from tiny diffs.
        # When |miss avg| is tiny, express the gap relative to the feature's
        # spread instead, capped, so it can't fake a "+368%" strongest signal.
        if abs(m) < 0.5:
            spread = v.std()
            pct = ((h - m) / spread * 100) if spread and not pd.isna(spread) else 0.0
            pct = max(-99.9, min(99.9, pct))
        else:
            pct = (h-m)/abs(m)*100
        seps.append((f, round(h,3), round(m,3), round(pct,1)))
    seps.sort(key=lambda x:-x[3])
    print(f"  {'feature':<22} {'hit':>8} {'miss':>8} {'sep%':>8}")
    for f,h,m,p in seps:
        tag = "STRONG" if p>=15 else "pos" if p>=3 else "weak/neutral"
        print(f"  {f:<22} {h:>8.3f} {m:>8.3f} {p:>+7.1f}%  {tag}")
    # flag: strongest separator should be well-weighted
    if seps:
        top = seps[0][0]
        flag(3, f"Strongest separator is '{top}' ({seps[0][3]:+.0f}%) — ensure selector weights it")
    print()

    # ══ CHECK 4: SELECTOR CHECK (parlay winning-leg reality) ══════════════
    print("="*62); print("4. PARLAY SELECTOR vs WINNING-LEG REALITY"); print("="*62)
    # DEPLOYED selector = BLEND 1: hr_per_fb/8 + edge*0.8. hr_per_fb is the
    # primary power driver (no longer buried by /20); edge (value) is the other
    # half. This check confirms hr_per_fb still out-separates platoon so the
    # weighting stays justified.
    plat = resolved["platoon_score"].apply(lambda x: sf(x, np.nan)) if "platoon_score" in resolved.columns else None
    hrfb = resolved["hr_per_fb"].apply(lambda x: sf(x, np.nan)) if "hr_per_fb" in resolved.columns else None
    for name, series, weight_note in [
        ("platoon_score", plat, "Blend1: not in selector (dropped as primary)"),
        ("hr_per_fb",     hrfb, "Blend1: hr_per_fb/8 = primary power driver"),
    ]:
        if series is None: continue
        h = series[resolved["hit"]].mean(); m = series[~resolved["hit"]].mean()
        gap = h - m
        print(f"  {name:<16} winners={h:>7.3f} miss={m:>7.3f} gap={gap:>+7.3f}   [{weight_note}]")
    if plat is not None and hrfb is not None:
        pg = (plat[resolved["hit"]].mean()-plat[~resolved["hit"]].mean())
        hg = (hrfb[resolved["hit"]].mean()-hrfb[~resolved["hit"]].mean())
        pstd = plat.std(); hstd = hrfb.std()
        p_eff = pg/pstd if pstd else 0
        h_eff = hg/hstd if hstd else 0
        print(f"\n  Effect size (gap / std):  platoon={p_eff:+.2f}  hr_per_fb={h_eff:+.2f}")
        if h_eff >= p_eff:
            print("  ✓ hr_per_fb ≥ platoon as a leg separator — Blend-1 weighting justified")
        else:
            flag(2, "platoon now out-separates hr_per_fb — revisit Blend-1 weighting")
            print("  ⚠️  platoon has overtaken hr_per_fb — Blend-1 may need retuning")
    print()

    # ══ CHECK 5: SINGLES EDGE GATE (low-score leak) ═══════════════════════
    print("="*62); print("5. SINGLES EDGE GATE (low-score leak check)"); print("="*62)
    # Deployed dashboard uses a score>=9 floor AND get_hr_score_tier returns
    # "below-8.5" (no hit-rate data) for sub-8.5. So the real leak risk is only
    # sub-9 players clearing breakeven. Check for those specifically.
    zone_rate = {}
    for (t,z), sub in resolved.groupby(["tier","zone"]):
        if len(sub) >= 15: zone_rate[(t,z)] = sub["hit"].mean()
    leaks = 0; leak_ex = []
    for _, r in resolved.iterrows():
        s = r["score"]
        if s >= 9: continue  # dashboard floor is 9 — only sub-9 could leak
        zr = zone_rate.get((tier_of(s), zone_of(r["odds"])))
        if zr is None: continue  # below-8.5 has no zone rate -> can't leak
        if zr - breakeven(r["odds"]) > 0:
            leaks += 1
            if len(leak_ex) < 3:
                leak_ex.append(f"{r.get('player_name','?')} score={s:.1f}")
    if leaks:
        print(f"  ⚠️  {leaks} sub-9 rows would clear breakeven (leak ONLY if floor absent)")
        print(f"     e.g. {', '.join(leak_ex)}")
        flag(2, f"{leaks} sub-9 players clear edge in raw data — confirm dashboard floor=9 is deployed")
    else:
        print("  ✓ No sub-9 leak (floor=9 + below-8.5 tier guard covers it)")
    print()

    # ══ CHECK 6: CONSISTENCY ══════════════════════════════════════════════
    print("="*62); print("6. CONSISTENCY"); print("="*62)
    if has_corr:
        blanks = df.apply(lambda r: str(r.get("hr_score_corrected","")).strip() in ("","nan","None"), axis=1).sum()
        recent = df[df["dt"] >= df["dt"].max() - pd.Timedelta(days=2)]
        recent_blanks = recent.apply(lambda r: str(r.get("hr_score_corrected","")).strip() in ("","nan","None"), axis=1).sum()
        print(f"  Recent (last 2 days) rows missing corrected score: {recent_blanks}/{len(recent)}")
        if recent_blanks > 0:
            flag(1, "Recent rows missing hr_score_corrected — live model not populating it; "
                    "dashboard/analysis will fall back to buggy hr_score for new picks")
        else:
            print("  ✓ Recent rows have corrected scores (live model populating correctly)")
    print()

    # ══ SUMMARY ═══════════════════════════════════════════════════════════
    print("="*62); print("⚠️  WHAT NEEDS ATTENTION (priority order)"); print("="*62)
    if not ATTENTION:
        print("  ✓ Nothing flagged — model looks healthy.")
    else:
        for pri, msg in sorted(ATTENTION, key=lambda x: x[0]):
            tag = {1:"🔴 CRITICAL", 2:"🟠 SHOULD FIX", 3:"🟡 VERIFY"}.get(pri, "•")
            print(f"  {tag}: {msg}")
    print("\nDone.")


if __name__ == "__main__":
    main()
