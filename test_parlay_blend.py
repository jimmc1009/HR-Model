"""
test_parlay_blend.py
Read-only. Tests blended parlay leg selectors against the current one, so you
can see the tradeoff between power (hr_per_fb) and value (edge vs breakeven)
BEFORE changing the live parlay.

Compares several selectors on the same historical pool:
  A. CURRENT   : hr_per_fb/6 + platoon*0.6   (power-only, drifts to low-odds stars)
  B. VALUE     : edge only                    (drifts to high-odds longshots)
  C. BLEND 1   : hr_per_fb/8 + edge*0.8       (balanced)
  D. BLEND 2   : hr_per_fb/10 + edge*1.2      (more value-tilted)

For each: leg hit rate, AVG ODDS of picked legs (key — are we off the favorites?),
ROI proxy, and the most-picked players (are they real power bats or randoms?).

Uses corrected scores + zone hit rates computed from the same resolved data.
"""
import os, json, time
from collections import Counter
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MODEL_START_DATE = "2026-06-09"
LEG_COUNT = 3
MIN_SAMPLE = 15

def gcc():
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))
def wr(fn,r=4,w=20):
    for a in range(r):
        try:return fn()
        except Exception:
            if a<r-1:time.sleep(w)
            else:raise
def sf(v,d=0.0):
    try:
        f=float(v);return d if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError,TypeError):return d
def coalesce(row,c,o):
    x=str(row.get(c,"")).strip()
    return sf(x) if x not in ("","nan","None") else sf(row.get(o))
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
    if "platoon_score_corrected" in df.columns:
        df["platoon_score"]=df.apply(lambda r:coalesce(r,"platoon_score_corrected","platoon_score"),axis=1)
    df["odds"]=df["consensus_odds"].apply(lambda x:sf(x,0))
    df["hr_per_fb"]=df["hr_per_fb"].apply(sf)
    df["platoon_score"]=df["platoon_score"].apply(sf)
    df["res"]=df["hit_hr"].astype(str).str.strip()
    r=df[df["res"].isin(["Yes","No"])].copy()
    r["hit"]=r["res"]=="Yes"

    # zone hit rates from resolved data
    zr={}
    r["tier"]=r["score"].apply(tier);r["zone"]=r["odds"].apply(zone)
    for (t,z),sub in r.groupby(["tier","zone"]):
        if len(sub)>=MIN_SAMPLE: zr[(t,z)]=sub["hit"].mean()

    def edge_of(row):
        z=zr.get((tier(row["score"]),zone(row["odds"])))
        return (z-be(row["odds"]))*100 if z is not None else -99  # unknown zone = avoid

    # selectors
    def sel_current(row): return row["hr_per_fb"]/6 + row["platoon_score"]*0.6
    def sel_value(row):   return edge_of(row)
    def sel_blend1(row):  return row["hr_per_fb"]/8 + edge_of(row)*0.8
    def sel_blend2(row):  return row["hr_per_fb"]/10 + edge_of(row)*1.2

    selectors=[("A CURRENT (power)",sel_current),
               ("B VALUE (edge)",sel_value),
               ("C BLEND1 (bal)",sel_blend1),
               ("D BLEND2 (value-tilt)",sel_blend2)]

    days=sorted(r["date"].unique())
    print(f"  {len(r)} resolved rows, {len(days)} days\n")
    print("="*72)
    print(f"  {'Selector':<24} {'Legs':>5} {'Leg%':>7} {'AvgOdds':>8} {'ROI%':>8} {'All3':>6}")
    print("="*72)

    detail={}
    for name,fn in selectors:
        legs_total=legs_hit=0;parlays=won=0;roi=0.0;odds_sum=0.0;picks=Counter()
        for d in days:
            day=r[r["date"]==d].copy()
            pool=day[day.apply(lambda x:in_pool(x["score"],x["odds"]),axis=1)].copy()
            if pool.empty:continue
            pool["sel"]=pool.apply(fn,axis=1)
            pool=pool.sort_values("sel",ascending=False)
            # pitcher-diversify
            chosen=[];used=set()
            for _,row in pool.iterrows():
                pn=str(row.get("pitcher_name","")).strip()
                if pn and pn in used:continue
                chosen.append(row); 
                if pn:used.add(pn)
                if len(chosen)>=LEG_COUNT:break
            if len(chosen)<LEG_COUNT:continue
            parlays+=1;allhit=True;d_dec=1.0
            for leg in chosen:
                legs_total+=1;odds_sum+=leg["odds"]
                picks[leg.get("player_name","?")]+=1
                if leg["hit"]:legs_hit+=1
                else:allhit=False
                d_dec*=dec(leg["odds"])
            if allhit:won+=1;roi+=(d_dec-1)
            else:roi-=1
        lr=legs_hit/legs_total*100 if legs_total else 0
        ao=odds_sum/legs_total if legs_total else 0
        rp=roi/parlays*100 if parlays else 0
        wr_=won/parlays*100 if parlays else 0
        print(f"  {name:<24} {legs_total:>5} {lr:>6.1f}% {ao:>+7.0f} {rp:>+7.1f}% {wr_:>5.1f}%")
        detail[name]=picks

    print("\n  Most-picked players per selector (are they real power bats?):")
    for name,_ in selectors:
        top=detail[name].most_common(5)
        print(f"    {name}: " + ", ".join(f"{p}({c})" for p,c in top))

    print("\n  Read: want higher Leg% AND higher AvgOdds (off the favorites) AND")
    print("  positive ROI, while most-picked stay recognizable power hitters.")
    print("  Pure VALUE (B) will show high odds but may pick randoms; CURRENT (A)")
    print("  low odds/stars. A BLEND that lifts odds without going to randoms wins.")
    print("\nDone.")

if __name__=="__main__":
    main()
