"""
test_2leg_slots.py
Which 2-leg pairing SLOTS are most profitable within an ISOLATED +301-400 pool?

Context: the 3-legger now pulls from a separate +351-500 band, so the +301-400
pool is no longer "shared" — the top legs aren't consumed by the 3-legger.
Question: should the two 2-leggers use the TOP legs (top1+2, top3+4) or skip
down (top4+5, top6+7)? Previously we skipped to avoid overlapping a shared
3-legger; that reason is gone. Test it rather than assume.

Tries several slot schemes on the isolated +301-400 pool, flat 1u/ticket,
Blend-1 selector, pitcher-diversified. Reports ROI per scheme + per pair.
Read-only, corrected data. Small sample → directional; prefer schemes positive
across BOTH pairs, not just highest net.
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
def dec(o):
    o=sf(o)
    if o>=100:return 1+o/100
    if o<=-100:return 1+100/abs(o)
    return 1.0
def be(o):
    o=sf(o)
    if o>=100:return 100.0/(o+100.0)
    if o<=-100:return abs(o)/(abs(o)+100.0)
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

def in_2leg_pool(s,o):
    # isolated +301-400 band (+ 13+ ≤+300 for leg quality)
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
    df["score"]=df.apply(lambda r:coalesce(r,"hr_score_corrected","hr_score"),axis=1)
    df["odds"]=df["consensus_odds"].apply(lambda x:sf(x,0))
    df["hr_per_fb"]=df["hr_per_fb"].apply(sf)
    df["res"]=df["hit_hr"].astype(str).str.strip()
    r=df[df["res"].isin(["Yes","No"])].copy()
    r["hit"]=r["res"]=="Yes"

    zr={}
    r["t"]=r["score"].apply(tier);r["z"]=r["odds"].apply(zone)
    for (t,z),sub in r.groupby(["t","z"]):
        if len(sub)>=MIN_SAMPLE: zr[(t,z)]=sub["hit"].mean()
    def edge(row):
        z=zr.get((tier(row["score"]),zone(row["odds"])))
        return (z-be(row["odds"]))*100 if z is not None else -99
    def selector(row): return row["hr_per_fb"]/8 + edge(row)*0.8

    def ranked(day):
        pool=day[day.apply(lambda x:in_2leg_pool(x["score"],x["odds"]),axis=1)].copy()
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

    # slot schemes to test (0-indexed pairs)
    schemes={
        "top1+2 / top3+4": [(0,1),(2,3)],
        "top4+5 / top6+7 (old)": [(3,4),(5,6)],
        "top1+2 / top4+5":  [(0,1),(3,4)],
        "top1+3 / top2+4":  [(0,2),(1,3)],
        "top1+2 only (single ticket)": [(0,1)],
    }

    def run(pairs):
        slot_stats=[{"n":0,"w":0,"ret":0.0} for _ in pairs]
        tot={"n":0,"w":0,"ret":0.0}
        for d in days:
            rk=ranked(r[r["date"]==d])
            for si,(i,j) in enumerate(pairs):
                if j<len(rk):
                    a,b=rk[i],rk[j]
                    combo=dec(a["odds"])*dec(b["odds"])
                    slot_stats[si]["n"]+=1; tot["n"]+=1
                    if a["hit"] and b["hit"]:
                        slot_stats[si]["w"]+=1; slot_stats[si]["ret"]+=combo
                        tot["w"]+=1; tot["ret"]+=combo
        return tot, slot_stats

    print("="*66)
    print(f"  {'scheme':<30} {'tix':>4} {'wins':>5} {'net':>9} {'roi%':>8}")
    print("="*66)
    verdicts=[]
    for name,pairs in schemes.items():
        tot,slots=run(pairs)
        net=tot["ret"]-tot["n"]; roi=net/tot["n"]*100 if tot["n"] else 0
        # per-slot positivity
        pos_slots=sum(1 for s in slots if (s["ret"]-s["n"])>0)
        verdicts.append((name,net,roi,pos_slots,len(pairs)))
        print(f"  {name:<30} {tot['n']:>4} {tot['w']:>5} {net:>+8.2f}u {roi:>+7.1f}%")
        for si,s in enumerate(slots):
            snet=s["ret"]-s["n"]; sroi=snet/s["n"]*100 if s["n"] else 0
            print(f"      slot{si+1}: n={s['n']:>3} w={s['w']:>2} net={snet:>+7.2f}u roi={sroi:>+7.1f}%")

    print("\n"+"="*66); print("  VERDICT"); print("="*66)
    # prefer: all slots positive, then highest net
    verdicts.sort(key=lambda x:(-(x[3]==x[4]), -x[1]))
    best=verdicts[0]
    print(f"  Best scheme: {best[0]}  (net {best[1]:+.2f}u, ROI {best[2]:+.1f}%, "
          f"{best[3]}/{best[4]} slots positive)")
    print("  Prefer schemes with ALL slots positive over highest raw net.")
    print("  Small sample — one ticket swings it; treat as directional.")
    print("\nDone.")

if __name__=="__main__":
    main()
