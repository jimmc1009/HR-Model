"""
test_3legger_ev.py
Read-only. THEORETICAL EV for the 3-legger — because it wins so rarely (≈0 in
26 days) you can't measure it from actual wins. Instead, for each day's 3-leg
ticket we compute EXPECTED value from the legs' known zone hit rates:

  P(all 3 hit) = p1 * p2 * p3   (each leg's historical tier×odds-zone hit rate)
  payout       = combined decimal odds
  EV per 1u    = P(all hit) * (payout - 1) - (1 - P(all hit))

Averaged across all days, this says whether the 3-legger is +EV IN EXPECTATION,
independent of the tiny actual-win sample. Also reports the realized wins as a
sanity check, and breaks EV down so you see what drives it.

Uses the live logic: +301-400 sweet-spot pool, Blend-1 selector, top-3 legs,
pitcher-diversified. Corrected scores.
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
    if o>=100:return 1+o/100
    if o<=-100:return 1+100/abs(o)
    return 1.0
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
def in_pool(s,o,olo=301,ohi=400):
    if o<=0:return False
    if 9.0<=s<10.0 and olo<=o<=ohi:return True
    if 12.0<=s<13.0 and olo<=o<=ohi:return True
    if s>=13.0 and olo<=o<=ohi:return True
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

    # zone hit rates (each leg's P(hit))
    zr={}
    r["tier"]=r["score"].apply(tier);r["zone"]=r["odds"].apply(zone)
    for (t,z),sub in r.groupby(["tier","zone"]):
        if len(sub)>=MIN_SAMPLE: zr[(t,z)]=sub["hit"].mean()
    def phit(row):
        return zr.get((tier(row["score"]),zone(row["odds"])),None)
    def edge(row):
        p=phit(row); return (p-be(row["odds"]))*100 if p is not None else -99
    def selector(row): return row["hr_per_fb"]/8 + edge(row)*0.8

    def build3(day, olo, ohi):
        pool=day[day.apply(lambda x:in_pool(x["score"],x["odds"],olo,ohi),axis=1)].copy()
        if pool.empty:return []
        pool["sel"]=pool.apply(selector,axis=1)
        pool=pool.sort_values("sel",ascending=False)
        chosen=[];used=set()
        for _,row in pool.iterrows():
            pn=str(row.get("pitcher_name","")).strip()
            if pn and pn in used:continue
            chosen.append(row)
            if pn:used.add(pn)
            if len(chosen)>=3:break
        return chosen

    days=sorted(r["date"].unique())

    def run_band(olo, ohi):
        ev_sum=0.0; n=0; realized_w=0; p_all_sum=0.0; payout_sum=0.0; ev_list=[]
        for d in days:
            legs=build3(r[r["date"]==d], olo, ohi)
            if len(legs)<3:continue
            ps=[phit(l) for l in legs]
            if any(p is None for p in ps):continue
            n+=1
            p_all=ps[0]*ps[1]*ps[2]
            payout=1.0
            for l in legs: payout*=dec(l["odds"])
            ev=p_all*(payout-1)-(1-p_all)
            ev_sum+=ev; ev_list.append(ev)
            p_all_sum+=p_all; payout_sum+=payout
            if all(l["hit"] for l in legs): realized_w+=1
        return dict(n=n, ev=(ev_sum/n if n else 0), pall=(p_all_sum/n if n else 0),
                    payout=(payout_sum/n if n else 0), rw=realized_w,
                    days_pos=sum(1 for e in ev_list if e>0))

    print(f"\nComparing 3-legger EV across odds bands\n")
    bands=[("CURRENT +301-400",301,400),
           ("SWEET SPOT +200-320",200,320),
           ("WIDE +200-400",200,400),
           ("HIGH-ODDS +351-500",351,500)]
    results=[]
    print("="*66)
    print(f"  {'band':<22} {'days':>4} {'P(all3)':>8} {'payout':>9} {'EV/1u':>8}")
    print("="*66)
    for label,olo,ohi in bands:
        z=run_band(olo,ohi)
        if z["n"]==0:
            print(f"  {label:<22}  (no qualifying days)"); continue
        results.append((label,z))
        one_in=int(1/z["pall"]) if z["pall"]>0 else 0
        print(f"  {label:<22} {z['n']:>4} {z['pall']*100:>7.2f}% {(z['payout']-1)*100:>+8.0f} {z['ev']:>+7.3f}u")
        print(f"  {'':<22} ~1 in {one_in}, realized {z['rw']}/{z['n']}, +EV days {z['days_pos']}/{z['n']}")

    if results:
        best=max(results,key=lambda x:x[1]["ev"])
        print("\n"+"="*66); print("  VERDICT"); print("="*66)
        print(f"  Best 3-legger EV band: {best[0]}  (EV {best[1]['ev']:+.3f}u/bet)")
        print("  For a LOTTERY 3-legger, higher-odds bands can win even at a lower")
        print("  P(all 3) because the payout when it lands is so large. Compare to")
        print("  the 2-legger verdict — they may prefer DIFFERENT bands (2-leg wants")
        print("  more frequent cashes, 3-leg wants the jackpot).")
        print("  Small sample + near-zero actual wins → EV is theoretical/directional.")
    print("\nDone.")

if __name__=="__main__":
    main()
