"""
test_blend1_decompose.py
Decomposes the Blend-1 parlay selector (hr_per_fb/8 + edge*0.8) for the
TOP-ranked legs, to see whether the POWER term or the EDGE term is driving
selection. Motivated by: top1+2 pairs losing money — if hr_per_fb/8 swamps
edge*0.8 at the top, the selector effectively ignores odds/edge for its best
picks, choosing raw power at bad prices.

For each day's parlay pool, ranks by Blend-1, then for the top legs shows:
  power_term = hr_per_fb/8, edge_term = edge*0.8, and which dominates.
Aggregates: at each rank, avg power vs edge contribution, and the hit rate of
legs where power dominates vs where edge dominates.

Read-only, corrected data.
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

    days=sorted(r["date"].unique())

    # collect per-rank decomposition
    rank_rows=[]   # (rank, power_term, edge_term, hit, odds)
    for d in days:
        day=r[r["date"]==d]
        pool=day[day.apply(lambda x:in_pool(x["score"],x["odds"]),axis=1)].copy()
        if pool.empty:continue
        pool["power_term"]=pool["hr_per_fb"]/8
        pool["edge_term"]=pool.apply(lambda x: edge(x)*0.8, axis=1)
        pool["sel"]=pool["power_term"]+pool["edge_term"]
        pool=pool.sort_values("sel",ascending=False).reset_index(drop=True)
        # diversify by pitcher
        used=set(); rk=0
        for _,row in pool.iterrows():
            pn=str(row.get("pitcher_name","")).strip()
            if pn and pn in used:continue
            if pn:used.add(pn)
            rank_rows.append((rk,row["power_term"],row["edge_term"],
                              row["hit"],row["odds"]))
            rk+=1

    rd=pd.DataFrame(rank_rows,columns=["rank","power","edge","hit","odds"])

    print(f"  {len(days)} days\n")
    print("="*70)
    print("  BLEND-1 DECOMPOSITION by selector rank (power=hr_per_fb/8, edge=edge*0.8)")
    print("="*70)
    print(f"  {'rank':>4} {'n':>4} {'avgPower':>9} {'avgEdge':>8} {'power>edge%':>11} {'avgOdds':>8} {'hit%':>6}")
    for rk in range(0,8):
        sub=rd[rd["rank"]==rk]
        if len(sub)==0: continue
        pdom=(sub["power"]>sub["edge"]).mean()*100
        print(f"  {rk+1:>4} {len(sub):>4} {sub['power'].mean():>9.2f} {sub['edge'].mean():>8.2f} "
              f"{pdom:>10.0f}% {sub['odds'].mean():>+7.0f} {sub['hit'].mean()*100:>5.1f}%")

    print("\n"+"="*70)
    print("  HIT RATE: legs where POWER dominates vs where EDGE dominates")
    print("="*70)
    power_dom=rd[rd["power"]>rd["edge"]]
    edge_dom=rd[rd["edge"]>=rd["power"]]
    print(f"  Power-dominated legs: n={len(power_dom):>4}  hit={power_dom['hit'].mean()*100:>5.1f}%  "
          f"avgOdds={power_dom['odds'].mean():>+6.0f}")
    print(f"  Edge-dominated legs:  n={len(edge_dom):>4}  hit={edge_dom['hit'].mean()*100:>5.1f}%  "
          f"avgOdds={edge_dom['odds'].mean():>+6.0f}")

    print("\n  READ: if the TOP ranks (1-2) are mostly power-dominated AND their")
    print("  hit% is lower than edge-dominated legs, that's the top1+2 flaw —")
    print("  hr_per_fb/8 is swamping edge*0.8, picking raw power at bad prices.")
    print("  Fix would be rebalancing Blend-1 (smaller power weight or cap it,")
    print("  or bigger edge weight). If power & edge legs hit similarly, no flaw.")
    print("\nDone.")

if __name__=="__main__":
    main()
