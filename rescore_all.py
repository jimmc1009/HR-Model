#!/usr/bin/env python3
"""
rescore_all.py  (v2 - faithful)

Re-score every HR_All_Scores row with the CURRENT formula, using ONLY inputs
that are actually stored. Components whose inputs are NOT stored are SKIPPED
(not estimated) - per "if you can't account for it, don't calculate it."

ACCOUNTED FOR: barrel windows 7d/5d/10d, power composite, pitcher_barrel_pct,
pitcher_hr_per_fb, pitch_matchup_score(clip*weight), momentum_score(stored,
clip*weight), platoon_score(clip), hr_weather_boost, wind_context.
SKIPPED (not stored): BVP, pitcher_quality_penalty (no pitcher_bbe_allowed),
pitch_penalty. These are small; recalc runs slightly HIGH vs elite pitchers.

Validation prints recalc-vs-live delta on today's rows (true score known).
Writes hr_score_recalc; hr_score untouched.
USAGE: python3 rescore_all.py [--dry]
"""
import sys, os, json, datetime
import numpy as np
import pandas as pd

DRY = "--dry" in sys.argv
try:
    import hr_picks as HP
except Exception as e:
    print(f"FATAL: cannot import hr_picks ({e}). Aborting."); sys.exit(1)

def sf(v, d=np.nan):
    try: return float(str(v).replace("+","").replace("%","").strip())
    except (ValueError, TypeError): return d

def g(row, col, d=np.nan):
    if col not in row.index: return d
    v = str(row[col]).strip()
    if v in ("","nan","None"): return d
    return sf(v, d)

def recalc_row(row):
    total = 0.0
    pa = g(row,"pa",0.0); pa = 0.0 if np.isnan(pa) else pa
    for fn,col,bcol in [(HP.score_barrel_pct_7d,"barrel_pct_7d","bbe_7d"),
                        (HP.score_barrel_pct_5d,"barrel_pct_5d","bbe_5d"),
                        (HP.score_barrel_pct_10d,"barrel_pct_10d","bbe_10d")]:
        v=g(row,col); bbe=g(row,bcol,0.0)
        if not np.isnan(v): total += fn(v, 0.0 if np.isnan(bbe) else bbe)
    sb=g(row,"season_barrel_pct",0.0); hf=g(row,"hr_per_fb",0.0)
    hp=g(row,"hr_per_pa",0.0); iso=g(row,"iso",0.0)
    # POWER_WEIGHT scales the composite (2026-08-23 rebalance) — must match
    # hr_picks exactly or the rescore diverges from live scores again.
    total += HP.score_power_composite(0.0 if np.isnan(sb) else sb, 0.0 if np.isnan(hf) else hf,
        0.0 if np.isnan(hp) else hp, 0.0 if np.isnan(iso) else iso, pa) * getattr(HP, "POWER_WEIGHT", 1.0)
    pbar=g(row,"pitcher_barrel_pct")
    if not np.isnan(pbar): total += HP.score_pitcher_barrel_pct(pbar)
    phf=g(row,"pitcher_hr_per_fb")
    if not np.isnan(phf): total += HP.score_pitcher_hr_per_fb(phf)
    # SKIP: pitcher_quality_penalty, pitch_penalty, bvp (inputs not stored)
    pm=g(row,"pitch_matchup_score")
    if not np.isnan(pm): total += max(-2.5,min(2.5,pm)) * HP.PITCH_MATCHUP_WEIGHT
    mo=g(row,"momentum_score")
    if not np.isnan(mo): total += max(-1.5,min(1.5,mo)) * HP.MOMENTUM_WEIGHT
    pl=g(row,"platoon_score")
    if not np.isnan(pl): total += max(-HP.PLATOON_CAP,min(HP.PLATOON_CAP,pl))
    wb=g(row,"hr_weather_boost")
    if not np.isnan(wb): total += HP.score_weather_boost(wb) * HP.WEATHER_WEIGHT
    if "wind_context" in row.index:
        wc=str(row["wind_context"]).strip()
        if wc not in ("","nan","None"): total += HP.score_wind_context(wc)
    return round(total,3)

def validate_today(df):
    today = datetime.date.today().isoformat()
    t = df[df["date"].astype(str)==today].copy()
    if t.empty:
        last = sorted(df["date"].astype(str).unique())[-1]
        t = df[df["date"].astype(str)==last].copy(); today=last
    if t.empty: print("  nothing to validate."); return
    t["live"]=pd.to_numeric(t["hr_score"].apply(sf),errors="coerce")
    t["recalc"]=pd.to_numeric(t["hr_score_recalc"],errors="coerce")
    t=t.dropna(subset=["live","recalc"])
    t["delta"]=t["recalc"]-t["live"]
    print(f"\n{'='*72}\nVALIDATION vs TRUE live score ({today}, n={len(t)})\n{'='*72}")
    print(f"  mean delta:   {t['delta'].mean():+.3f}  (recalc minus live)")
    print(f"  median delta: {t['delta'].median():+.3f}")
    print(f"  abs delta:    mean {t['delta'].abs().mean():.3f}, max {t['delta'].abs().max():.3f}")
    print(f"  within 0.25:  {(t['delta'].abs()<=0.25).mean()*100:.0f}% of rows")
    print(f"  within 0.50:  {(t['delta'].abs()<=0.50).mean()*100:.0f}% of rows")
    print("\n  Skipped penalties fire vs elite pitchers -> recalc slightly HIGH there.")
    print("  Small consistent delta = faithful. Big/erratic = a component still wrong.")
    worst=t.reindex(t["delta"].abs().sort_values(ascending=False).index).head(6)
    print(f"\n  biggest deltas:")
    for _,r in worst.iterrows():
        print(f"    {str(r['player_name'])[:20]:<20} live {r['live']:.2f}  recalc {r['recalc']:.2f}  ({r['delta']:+.2f})")

def analyze(df):
    print(f"\n{'='*72}\nTIER ANALYSIS ON NEW SCORE (hr_score_recalc)\n{'='*72}")
    d=df.copy()
    d["sc"]=pd.to_numeric(d["hr_score_recalc"],errors="coerce")
    d["res"]=d.get("hit_hr","").astype(str).str.strip().str.lower()
    d=d[d["res"].isin(["yes","no"])].copy()
    d["hit"]=(d["res"]=="yes").astype(int)
    if "date" in d.columns: d=d[d["date"].astype(str)>="2026-06-09"]
    d=d.dropna(subset=["sc"])
    if len(d)<50: print("  too few resolved rows."); return
    print(f"  {len(d)} resolved rows, base {d['hit'].mean()*100:.1f}%\n")
    qs=[d["sc"].quantile(q) for q in [0.98,0.90,0.80,0.65,0.50,0.35,0.20]]
    edges=[("top 2%",qs[0],99),("90-98",qs[1],qs[0]),("80-90",qs[2],qs[1]),
           ("65-80",qs[3],qs[2]),("50-65",qs[4],qs[3]),("35-50",qs[5],qs[4]),
           ("20-35",qs[6],qs[5]),("bot 20",-99,qs[6])]
    print(f"  {'tier':<10}{'score rng':<14}{'n':>6}{'hit%':>8}")
    print("  "+"-"*40)
    prev=None
    for lab,lo,hi in edges:
        s=d[(d["sc"]>=lo)&(d["sc"]<hi)]
        if len(s)<15: print(f"  {lab:<10}{f'{lo:.1f}-{hi:.1f}':<14}{len(s):>6}   --"); continue
        r=s["hit"].mean()*100
        arrow=" ^" if (prev is not None and r>prev) else " v" if (prev is not None and r<prev) else ""
        print(f"  {lab:<10}{f'{lo:.1f}-{hi:.1f}':<14}{len(s):>6}{r:>7.1f}%{arrow}")
        prev=r

def main():
    import gspread
    from google.oauth2.service_account import Credentials
    scope=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    creds=Credentials.from_service_account_info(json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]),scopes=scope)
    gc=gspread.authorize(creds); sh=gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    ws=sh.worksheet("HR_All_Scores"); vals=ws.get_all_values()
    df=pd.DataFrame(vals[1:],columns=vals[0])
    print(f"Loaded HR_All_Scores: {len(df)} rows, {len(df.columns)} columns")
    df["hr_score_recalc"]=df.apply(recalc_row,axis=1)
    validate_today(df); analyze(df)
    if DRY: print("\n[--dry] not writing."); return
    print("\nWriting hr_score_recalc back...")
    ws.clear(); ws.update([df.columns.tolist()]+df.astype(str).values.tolist())
    print("Done. hr_score_recalc written (hr_score untouched).")

if __name__=="__main__":
    main()
