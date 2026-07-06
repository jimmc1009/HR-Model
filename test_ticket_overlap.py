"""
test_ticket_overlap.py
Read-only. Compares two ways of building the daily slate of parlay tickets
(one 3-leg + three 2-leg tickets), using the live Blend-1 selector:

  OVERLAP    : 3-leg = top1-3; 2-leggers = top1+2, top3+4, top5+6
               (best players repeat across tickets — concentrated)
  NO-OVERLAP : 3-leg = top1-3; 2-leggers = top4+5, top6+7, top8+9
               (every player appears once — spread out)

The metric that matters for "hitting parlays for fun":
  → % of days where AT LEAST ONE ticket cashed (any of the four)
Also shows each format's per-ticket hit rate and the 3-leg vs 2-leg split.

Uses corrected scores + real edge (zone hit rate − breakeven) for the selector.
"""
import os, json, time
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

    def ranked_pool(day):
        pool=day[day.apply(lambda x:in_pool(x["score"],x["odds"]),axis=1)].copy()
        if pool.empty:return []
        pool["sel"]=pool.apply(selector,axis=1)
        pool=pool.sort_values("sel",ascending=False)
        chosen=[];used=set()
        for _,row in pool.iterrows():
            pn=str(row.get("pitcher_name","")).strip()
            if pn and pn in used:continue
            chosen.append(row)
            if pn:used.add(pn)
        return chosen

    def leg_hit(leg):return leg["hit"]

    days=sorted(r["date"].unique())
    print(f"  {len(r)} resolved rows, {len(days)} days\n")

    for mode in ["OVERLAP","NO-OVERLAP"]:
        days_any_cash=0; days_playable=0
        t3_w=t3_n=0; t2_w=t2_n=0
        for d in days:
            ranked=ranked_pool(r[r["date"]==d])
            if len(ranked)<3:  # need at least a 3-leg
                continue
            days_playable+=1
            # 3-leg always top1-3
            three=ranked[:3]
            t3_hit=all(leg_hit(l) for l in three)
            t3_n+=1; t3_w+=1 if t3_hit else 0
            any_cash=t3_hit
            # 2-leggers
            if mode=="OVERLAP":
                pairs=[(0,1),(2,3),(4,5)]
            else:
                pairs=[(3,4),(5,6),(7,8)]
            for i,j in pairs:
                if j<len(ranked):
                    h=leg_hit(ranked[i]) and leg_hit(ranked[j])
                    t2_n+=1; t2_w+=1 if h else 0
                    any_cash=any_cash or h
            if any_cash: days_any_cash+=1

        print("="*58); print(f"  {mode}"); print("="*58)
        print(f"    Playable days:              {days_playable}")
        print(f"    Days ≥1 ticket cashed:      {days_any_cash} ({round(days_any_cash/days_playable*100,1) if days_playable else 0}%)")
        print(f"    3-leg ticket hit rate:      {t3_w}/{t3_n} = {round(t3_w/t3_n*100,1) if t3_n else 0}%")
        print(f"    2-leg ticket hit rate:      {t2_w}/{t2_n} = {round(t2_w/t2_n*100,1) if t2_n else 0}%")
        print()

    print("  KEY METRIC: 'Days ≥1 ticket cashed' — higher = you feel a win more")
    print("  often. NO-OVERLAP spreads across more players (top4-9 for the")
    print("  2-leggers); OVERLAP concentrates on the best (top1-6). For fun +")
    print("  frequent cashes, higher any-cash% wins even if 2-leg rate dips.")
    print("\nDone.")

if __name__=="__main__":
    main()
