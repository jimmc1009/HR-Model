"""
test_2legger_roi.py
Read-only. Answers ONE question: are the 2-leg parlays profitable?

Builds the three daily 2-leggers (NO-OVERLAP: top4+5, top6+7, top8+9 by the
live Blend-1 selector) and computes REAL ROI using actual combined decimal
odds — do the winning tickets pay enough to cover the losers?

Reports: tickets, wins, hit rate, total staked, total returned, net units,
ROI %, and a breakeven hit-rate reference. Also splits by which pair slot
(top4+5 vs top6+7 vs top8+9) so you see if the better legs carry it.

Flat 1-unit stake per ticket. Uses corrected scores + real edge selector.
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
    if 9.0<=s<10.0 and 301<=o<=400:return True
    if 12.0<=s<13.0 and 301<=o<=400:return True
    if s>=13.0 and 301<=o<=400:return True
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

    days=sorted(r["date"].unique())
    print(f"  {len(r)} resolved rows, {len(days)} days\n")

    slots=[("top4+5",3,4),("top6+7",5,6),("top8+9",7,8)]
    overall={"n":0,"w":0,"staked":0.0,"returned":0.0}
    per_slot={s[0]:{"n":0,"w":0,"ret":0.0} for s in slots}

    for d in days:
        ranked=ranked_pool(r[r["date"]==d])
        for name,i,j in slots:
            if j<len(ranked):
                a,b=ranked[i],ranked[j]
                combined=dec(a["odds"])*dec(b["odds"])
                overall["n"]+=1; overall["staked"]+=1.0
                per_slot[name]["n"]+=1
                if a["hit"] and b["hit"]:
                    overall["w"]+=1; overall["returned"]+=combined
                    per_slot[name]["w"]+=1; per_slot[name]["ret"]+=combined

    n=overall["n"]; w=overall["w"]
    staked=overall["staked"]; returned=overall["returned"]
    net=returned-staked
    roi=net/staked*100 if staked else 0
    hitrate=w/n*100 if n else 0

    print("="*58); print("  2-LEGGER PROFITABILITY (flat 1u/ticket, NO-OVERLAP)"); print("="*58)
    print(f"    Tickets:        {n}")
    print(f"    Wins:           {w}  ({hitrate:.1f}%)")
    print(f"    Staked:         {staked:.0f}u")
    print(f"    Returned:       {returned:.2f}u")
    print(f"    Net:            {net:+.2f}u")
    print(f"    ROI:            {roi:+.1f}%")
    print(f"    {'PROFITABLE ✅' if net>0 else 'UNPROFITABLE ❌'}")

    print("\n  By pair slot:")
    print(f"    {'slot':<8} {'n':>4} {'wins':>5} {'hit%':>7} {'ret':>8} {'net':>8} {'roi%':>8}")
    for name,i,j in slots:
        ps=per_slot[name]; pn=ps["n"]; pw=ps["w"]; pr=ps["ret"]
        pnet=pr-pn; proi=pnet/pn*100 if pn else 0
        phr=pw/pn*100 if pn else 0
        print(f"    {name:<8} {pn:>4} {pw:>5} {phr:>6.1f}% {pr:>7.2f}u {pnet:>+7.2f}u {proi:>+7.1f}%")

    print("\n  Note: small sample (few wins) -> ROI is high-variance. One extra")
    print("  win or miss swings it a lot. Treat as directional. Positive net")
    print("  across ~75 tickets is encouraging; strongly negative is a red flag.")
    print("\nDone.")

if __name__=="__main__":
    main()
