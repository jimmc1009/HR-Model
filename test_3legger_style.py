"""
test_3legger_style.py
Read-only. Finds the optimal 3-leg parlay construction. Since 3-leggers are
high-odds lottery swings (user plays +924 to +2000+), the goal is: maximize
chances of the rare cash AND the payout when it lands.

Tests several construction styles on the same pool (Blend-1 selector, corrected
data), reporting for each: tickets, wins, hit%, avg combined odds, total return,
net units, ROI.

Styles:
  1. TOP3          — top 3 by selector (current)
  2. TOP3 PLUS-ODDS— top 3, but only +301-499 legs (higher combined odds)
  3. HIGHEST-ODDS  — the 3 highest-ODDS legs in the pool (max payout)
  4. HIGHEST-EDGE  — the 3 best-EDGE legs (value)
  5. BALANCED      — 1 low-odds anchor + 2 higher-odds (mix safety & payout)
  6. ALL-9-10 ZONE — 3 legs all from 9-10|+301-499 (your best zone)

Flat 1u/ticket. Small sample — treat ROI as directional.
"""
import os, json, time
from itertools import combinations
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MODEL_START_DATE="2026-06-09"; MIN_SAMPLE=15

def gcc():
    return gspread.authorize(Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]),scopes=SCOPES))
def wr(fn,r=4,w=20):
    for a in range(r):
        try:return fn()
        except Exception:
            if a<r-1:time.sleep(w)
            else:raise
def sf(v,d=0.0):
    try:
        f=float(v);return d if(pd.isna(f)or np.isinf(f))else f
    except(ValueError,TypeError):return d
def coalesce(row,c,o):
    x=str(row.get(c,"")).strip()
    return sf(x) if x not in("","nan","None") else sf(row.get(o))
def be(o):
    if o>=100:return 100.0/(o+100.0)
    if o<0:return abs(o)/(abs(o)+100.0)
    return 1.0
def dec(o):
    if o>=100:return 1+o/100
    if o<=-100:return 1+100/abs(o)
    return 1.0
def zone(o):
    if o<=300:return "≤+300"
    if o<=499:return "+301-499"
    if o<=699:return "+500-699"
    return "+700+"
def tier(s):
    if s>=13:return "13+"
    if s>=12:return "12-13"
    if s>=11:return "11-12"
    if s>=10:return "10-11"
    if s>=9:return "9-10"
    if s>=8.5:return "8.5-9"
    return "<8.5"
def in_pool(s,o):
    if o<=0:return False
    if 9.0<=s<10.0 and 301<=o<=499:return True
    if s>=13.0 and 301<=o<=499:return True
    if 12.0<=s<13.0 and 301<=o<=499:return True
    if s>=13.0 and o<=300:return True
    return False

def main():
    sid=os.environ["GOOGLE_SHEET_ID"];gc=gcc()
    print("Reading HR_All_Scores...")
    sh=wr(lambda:gc.open_by_key(sid));ws=sh.worksheet("HR_All_Scores")
    vals=wr(lambda:ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)]
    df["dt"]=pd.to_datetime(df["date"],errors="coerce")
    df=df[df["dt"]>=pd.Timestamp(MODEL_START_DATE)].copy()
    has="hr_score_corrected" in df.columns
    df["score"]=df.apply(lambda r:coalesce(r,"hr_score_corrected","hr_score"),axis=1) if has else df["hr_score"].apply(sf)
    df["odds"]=df["consensus_odds"].apply(lambda x:sf(x,0))
    df["hr_per_fb"]=df["hr_per_fb"].apply(sf)
    df["res"]=df["hit_hr"].astype(str).str.strip()
    r=df[df["res"].isin(["Yes","No"])].copy()
    r["hit"]=r["res"]=="Yes"

    zr={}
    r["tier"]=r["score"].apply(tier);r["zone"]=r["odds"].apply(zone)
    for (t,z),sub in r.groupby(["tier","zone"]):
        if len(sub)>=MIN_SAMPLE: zr[(t,z)]=sub["hit"].mean()
    def edge(row):
        z=zr.get((tier(row["score"]),zone(row["odds"])))
        return (z-be(row["odds"]))*100 if z is not None else -99
    def selector(row): return row["hr_per_fb"]/8 + edge(row)*0.8

    def pool_of(day):
        pool=day[day.apply(lambda x:in_pool(x["score"],x["odds"]),axis=1)].copy()
        if pool.empty:return pool
        pool["sel"]=pool.apply(selector,axis=1)
        pool["edge"]=pool.apply(edge,axis=1)
        return pool
    def diversify(pool_sorted):
        chosen=[];used=set()
        for _,row in pool_sorted.iterrows():
            pn=str(row.get("pitcher_name","")).strip()
            if pn and pn in used:continue
            chosen.append(row)
            if pn:used.add(pn)
        return chosen

    styles=["1 TOP3","2 TOP3 PLUS","3 HIGH-ODDS","4 HIGH-EDGE","5 BALANCED","6 ALL 9-10"]
    stats={s:{"n":0,"w":0,"ret":0.0,"oddsum":0.0} for s in styles}

    days=sorted(r["date"].unique())
    for d in days:
        pool=pool_of(r[r["date"]==d])
        if pool.empty: continue

        def build(style):
            if style=="1 TOP3":
                return diversify(pool.sort_values("sel",ascending=False))[:3]
            if style=="2 TOP3 PLUS":
                p=pool[(pool["odds"]>=301)&(pool["odds"]<=499)]
                return diversify(p.sort_values("sel",ascending=False))[:3]
            if style=="3 HIGH-ODDS":
                return diversify(pool.sort_values("odds",ascending=False))[:3]
            if style=="4 HIGH-EDGE":
                return diversify(pool.sort_values("edge",ascending=False))[:3]
            if style=="5 BALANCED":
                # 1 lowest-odds anchor + 2 highest-odds
                by_odds=diversify(pool.sort_values("odds"))
                if len(by_odds)<3: return []
                return [by_odds[0], by_odds[-1], by_odds[-2]]
            if style=="6 ALL 9-10":
                p=pool[(pool["score"]>=9)&(pool["score"]<10)&(pool["odds"]>=301)&(pool["odds"]<=499)]
                return diversify(p.sort_values("sel",ascending=False))[:3]
            return []

        for s in styles:
            legs=build(s)
            if len(legs)<3: continue
            st=stats[s]; st["n"]+=1
            combined=1.0; allhit=True
            for leg in legs:
                combined*=dec(leg["odds"])
                if not leg["hit"]: allhit=False
            st["oddsum"]+=combined
            if allhit: st["w"]+=1; st["ret"]+=combined

    print(f"  {len(r)} resolved rows, {len(days)} days\n")
    print("="*76)
    print(f"  {'Style':<14} {'Tix':>4} {'Wins':>5} {'Hit%':>6} {'AvgComboOdds':>13} {'Net':>8} {'ROI%':>8}")
    print("="*76)
    for s in styles:
        st=stats[s]; n=st["n"]; wn=st["w"]
        if n==0: 
            print(f"  {s:<14}    (no qualifying days)")
            continue
        hit=wn/n*100
        avg_combo=(st["oddsum"]/n - 1)*100  # american-ish combined
        net=st["ret"]-n; roi=net/n*100
        print(f"  {s:<14} {n:>4} {wn:>5} {hit:>5.1f}% {'+'+str(int(avg_combo)):>13} {net:>+7.2f}u {roi:>+7.1f}%")

    print("\n  For a high-odds lottery 3-legger: want a style that cashes at all")
    print("  (any wins) with big AvgComboOdds. ROI is very high-variance on 3-")
    print("  leggers (few/no wins) — the payout size when it hits matters most.")
    print("  A style with even 1-2 wins at huge combined odds can be +EV.")
    print("\nDone.")

if __name__=="__main__":
    main()
