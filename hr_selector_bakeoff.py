#!/usr/bin/env python3
"""
hr_selector_bakeoff.py — which selector actually makes money, judged on REAL
parlay tickets at REAL odds (ROI), walk-forward so there's no lookahead.

Why ROI, not leg hit rate: hit rate is biased toward short-odds legs (favorites
hit more but pay less). Grading real tickets at their real prices neutralizes
that — a selector that just farms cheap favorites will show it in the payout.

Walk-forward: for each day, the zone value-table (edge term) and the z-score
stats are built ONLY from prior days. Then each selector ranks that day's pool,
we build the real 3-leg (top1-3) and 2-leg (top4+5, top6+7) tickets diversified
by opposing pitcher, and grade them at actual odds. Nothing peeks ahead.

Reads HR_All_Scores, writes nothing.
"""
import os, math, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MODEL_START_DATE="2026-06-09"
WARMUP_DAYS=14          # need this many prior days before we start grading
MIN_ZONE_N=12           # min sample for a zone rate to be usable

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))
def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def dec(o): return 1+o/100 if o>=100 else (1+100/abs(o) if o<=-100 else 2.0)
def be_pct(o): return 100.0*100.0/(o+100.0) if o>0 else 100.0

def zone_key(s,o):
    if o<=0:return None
    if s>=13.0 and o<=300:return "13+|<=300"
    if s>=13.0 and 301<=o<=400:return "13+|301-400"
    if 12.0<=s<13.0 and 301<=o<=400:return "12-13|301-400"
    if 9.0<=s<10.0 and 301<=o<=400:return "9-10|301-400"
    return None

def load(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("HR_All_Scores")
    vals=ws.get_all_values(); df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    df["date_dt"]=pd.to_datetime(df.get("date",""),errors="coerce")
    df=df[df["date_dt"]>=pd.Timestamp(MODEL_START_DATE)].copy()
    def coalesce(r):
        x=str(r.get("hr_score_corrected","")).strip()
        return sf(x) if x not in ("","nan","None") else sf(r.get("hr_score"))
    rows=[]
    for _,r in df.iterrows():
        if str(r.get("hit_hr","")).strip() not in ("Yes","No"): continue
        score=coalesce(r); odds=sf(r.get("consensus_odds"))
        if zone_key(score,odds) is None: continue
        rows.append({"date":str(r.get("date","")).strip(),
            "pitcher":str(r.get("pitcher_name","")).strip(),
            "player":str(r.get("player_name","")).strip(),
            "score":score,"odds":odds,"win":1 if str(r.get("hit_hr","")).strip()=="Yes" else 0,
            "hr_per_fb":sf(r.get("hr_per_fb")),
            "hard_hit":sf(r.get("hard_hit_pct_season")),
            "pitch_match":sf(r.get("pitch_matchup_score"))})
    return pd.DataFrame(rows)

def zone_rates(train):
    zr={}
    for k in ["13+|<=300","13+|301-400","12-13|301-400","9-10|301-400"]:
        sub=train[train["zk"]==k]
        if len(sub)>=MIN_ZONE_N: zr[k]=sub["win"].mean()*100
    return zr
def edge(r,zr):
    k=r["zk"]
    if k not in zr: return -99.0
    return zr[k]-be_pct(r["odds"])

def main():
    gc=gcc(); legs=load(gc,os.environ["GOOGLE_SHEET_ID"])
    if legs.empty:
        print("No resolved pool legs."); return
    legs["zk"]=legs.apply(lambda r:zone_key(r["score"],r["odds"]),axis=1)
    dates=sorted(legs["date"].unique())

    def z(v,m,s): return (v-m)/s if s else 0.0
    selectors={
        "blend1 (CURRENT)": lambda r,zr,st: r["hr_per_fb"]/8 + edge(r,zr)*0.8,
        "power only":       lambda r,zr,st: r["hr_per_fb"],
        "edge only":        lambda r,zr,st: edge(r,zr),
        "blend1 + hardHit": lambda r,zr,st: r["hr_per_fb"]/8 + edge(r,zr)*0.8 + z(r["hard_hit"],st["hh_m"],st["hh_s"])*st["bp"],
        "blend1 + pitchMatch":lambda r,zr,st: r["hr_per_fb"]/8 + edge(r,zr)*0.8 + z(r["pitch_match"],st["pm_m"],st["pm_s"])*st["bp"],
    }
    acc={name:{"3":[0,0,0.0,0.0],"2":[0,0,0.0,0.0]} for name in selectors}  # formed,won,stake,ret

    for i,d in enumerate(dates):
        if i<WARMUP_DAYS: continue
        train=legs[legs["date"].isin(dates[:i])]
        if len(train)<50: continue
        zr=zone_rates(train)
        st={"hh_m":train["hard_hit"].mean(),"hh_s":train["hard_hit"].std(ddof=0) or 1.0,
            "pm_m":train["pitch_match"].mean(),"pm_s":train["pitch_match"].std(ddof=0) or 1.0,
            "bp":(train["hr_per_fb"]/8).std(ddof=0) or 1.0}
        day=legs[legs["date"]==d]
        if day.empty: continue
        for name,fn in selectors.items():
            dd=day.copy(); dd["sel"]=dd.apply(lambda r:fn(r,zr,st),axis=1)
            dd=dd.sort_values("sel",ascending=False)
            picked=[]; seen=set()
            for _,r in dd.iterrows():
                p=r["pitcher"]
                if p and p in seen: continue
                if p: seen.add(p)
                picked.append(r)
            def grade(ls,slot):
                won=all(x["win"]==1 for x in ls)
                ret=math.prod(dec(x["odds"]) for x in ls) if won else 0.0
                acc[name][slot][0]+=1; acc[name][slot][1]+=int(won)
                acc[name][slot][2]+=1.0; acc[name][slot][3]+=ret
            if len(picked)>=3: grade(picked[0:3],"3")
            for a,b in [(3,4),(5,6)]:
                if b<len(picked): grade([picked[a],picked[b]],"2")

    print("="*72)
    print("HR PARLAY SELECTOR BAKE-OFF — walk-forward, graded at REAL odds (1u/ticket)")
    print("="*72)
    print(f"{len(dates)} days, grading from day {WARMUP_DAYS+1} on (expanding train)\n")
    print(f"  {'Selector':22s} | {'3-leg  form/won  ROI':24s} | {'2-leg  form/won  ROI'}")
    print("  "+"-"*72)
    for name in selectors:
        def fmt(x):
            f,w,st,rt=x; roi=(rt-st)/st*100 if st else 0; hit=(w/f*100 if f else 0)
            return f"{f:3d}/{w:<3d} {hit:4.0f}% {roi:+7.1f}%"
        print(f"  {name:22s} | {fmt(acc[name]['3']):24s} | {fmt(acc[name]['2'])}")
    print("\n  ROI is the honest score (money at real odds, no short-odds bias).")
    print("  2-leg column is the more trustworthy one (more tickets). A selector")
    print("  only wins if it beats blend1 on ROI with a real number of tickets.")
    print("Done.")

if __name__=="__main__":
    main()
