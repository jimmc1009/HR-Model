#!/usr/bin/env python3
"""
diagnose_hr_tier_granular.py

STEP 1  Hit rate by 0.5-point score buckets — find where hit rate breaks from
        the baseline (whole-pool) rate.
STEP 1b Cumulative "score >= X" hit rate for every half-point threshold — shows
        where "everything above this line" starts clearly beating baseline.
STEP 2  Feature separators computed ONLY within the pool above THRESHOLD, so
        low-score non-hitters can't trivially inflate every separator. Ranked by
        Cohen's d (honest effect size), NOT % difference (which blows up on
        near-zero features like momentum).

Set THRESHOLD below, or override at runtime:  THRESHOLD=11 python diagnose_hr_tier_granular.py
Reads HR_All_Scores, writes nothing.
"""
import os, math, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MODEL_START_DATE="2026-06-09"
THRESHOLD=float(os.environ.get("THRESHOLD","10.0"))

FEATURES=[
    ("iso","ISO"),("season_barrel_pct","Barrel% (season)"),
    ("barrel_pct_5d","Barrel% (5d)"),("barrel_pct_7d","Barrel% (7d)"),
    ("barrel_pct_10d","Barrel% (10d)"),("hr_per_fb","HR/FB%"),("hr_per_pa","HR/PA%"),
    ("hard_hit_pct_7d","HardHit% (7d)"),("hard_hit_pct_season","HardHit% (season)"),
    ("avg_ev_7d","Avg EV (7d)"),("avg_la_7d","Avg LA (7d)"),("pull_rate","Pull Rate"),
    ("park_hr_factor","Park HR Factor"),("pitch_matchup_score","Pitch Matchup"),
    ("hr_weather_boost","Weather Boost"),("platoon_score","Platoon Score"),
    ("pitcher_barrel_pct","Pitcher Barrel%-allowed"),
    ("pitcher_hr_per_fb","Pitcher HR/FB-allowed"),("momentum_score","Momentum"),
]

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))
def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def cohens_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2:return 0.0
    va,vb=a.var(ddof=1),b.var(ddof=1)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)) if (na+nb-2)>0 else 0.0
    return (a.mean()-b.mean())/sp if sp>0 else 0.0

def load(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("HR_All_Scores")
    vals=ws.get_all_values(); df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    df["date_dt"]=pd.to_datetime(df.get("date",""),errors="coerce")
    df=df[df["date_dt"]>=pd.Timestamp(MODEL_START_DATE)].copy()
    def coalesce(r):
        x=str(r.get("hr_score_corrected","")).strip()
        return sf(x) if x not in ("","nan","None") else sf(r.get("hr_score"))
    df["score"]=df.apply(coalesce,axis=1)
    for col,_ in FEATURES:
        df[col]=df.get(col, pd.Series([0]*len(df))).apply(sf)
    res=df[df.get("hit_hr","").astype(str).str.strip().isin(["Yes","No"])].copy()
    res["hit"]=(res["hit_hr"].astype(str).str.strip()=="Yes").astype(int)
    return res

def main():
    gc=gcc(); res=load(gc,os.environ["GOOGLE_SHEET_ID"])
    if res.empty:
        print("No resolved HR rows."); return
    base=res["hit"].mean()*100
    print("="*66)
    print(f"HR TIER GRANULAR  (n={len(res)} resolved, baseline hit {base:.1f}%)")
    print("="*66)

    lo=math.floor(res["score"].min()*2)/2
    hi=math.ceil(res["score"].max()*2)/2
    steps=[lo+0.5*i for i in range(int((hi-lo)/0.5)+1)]

    print("\nSTEP 1 — hit rate by 0.5-pt bucket (flag = >=5pts off baseline):")
    print(f"  {'bucket':<12}{'n':>6}{'hits':>6}{'hit%':>8}{'vs base':>9}")
    for t in steps:
        sub=res[(res["score"]>=t)&(res["score"]<t+0.5)]
        if len(sub)<5: continue
        n=len(sub); h=int(sub["hit"].sum()); rate=h/n*100; d=rate-base
        flag=" ✅" if d>=5 else (" 🔴" if d<=-5 else "")
        print(f"  {t:>5.1f}-{t+0.5:<5.1f}{n:>6}{h:>6}{rate:>7.1f}%{d:>+8.1f}%{flag}")

    print("\nSTEP 1b — cumulative score >= X (where does 'above the line' beat base?):")
    print(f"  {'>= X':<12}{'n':>6}{'hits':>6}{'hit%':>8}{'vs base':>9}")
    for t in steps:
        sub=res[res["score"]>=t]
        if len(sub)<10: continue
        n=len(sub); h=int(sub["hit"].sum()); rate=h/n*100; d=rate-base
        flag=" ✅" if d>=5 else ""
        print(f"  >= {t:<9.1f}{n:>6}{h:>6}{rate:>7.1f}%{d:>+8.1f}%{flag}")

    print("\n"+"="*66)
    print(f"STEP 2 — SEPARATORS WITHIN score >= {THRESHOLD} (Cohen's d)")
    print("="*66)
    r=res[res["score"]>=THRESHOLD].copy()
    if len(r)<20:
        print(f"  Only {len(r)} picks in pool — too few. Lower THRESHOLD."); return
    w=r[r["hit"]==1]; l=r[r["hit"]==0]
    print(f"  Pool: {len(r)} picks, {r['hit'].mean()*100:.1f}% hit "
          f"({len(w)} hits / {len(l)} misses)\n")
    print(f"  {'Feature':22s} {'hit avg':>9s} {'miss avg':>9s} {'diff':>9s} {'Cohen d':>8s}")
    print("  "+"-"*62)
    rows=[]
    for col,label in FEATURES:
        if r[col].std(ddof=0)==0: continue
        a,b=w[col],l[col]
        if len(a)<5 or len(b)<5: continue
        d=cohens_d(a,b)
        rows.append((abs(d),label,a.mean(),b.mean(),a.mean()-b.mean(),d))
    for _,label,wa,la,diff,d in sorted(rows,reverse=True):
        print(f"  {label:22s} {wa:9.3f} {la:9.3f} {diff:+9.3f} {d:+8.3f}")
    print("\n  Cohen d: >|0.2| small, >|0.5| medium, >|0.8| large. Compare to the")
    print("  whole-pool separators — a feature that stays strong HERE (inside the")
    print("  bettable pool) is the real signal; one that fades was riding low scorers.")
    print("Done.")

if __name__=="__main__":
    main()
