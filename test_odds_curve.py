"""
test_odds_curve.py
Finds the TRUE odds sweet spot without arbitrary round-number bands.

Two methods:
  A. FINE BINS (25-wide) — shows the EV *curve* shape across odds, so you can
     see where single-leg edge actually rises, peaks, and falls.
  B. SLIDING WINDOW — slides a fixed-width window across all odds and reports
     the window with the best EV (and hit rate) that meets a sample floor.
     This catches a sweet spot that straddles round-number boundaries.

Single-leg, score>=9 legs (the parlay-eligible pool). Read-only, corrected data.
Enforces sample minimums; flags thin slices.
"""
import os, json, time
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MODEL_START_DATE="2026-06-09"

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

def ev_of(sub):
    """EV per 1u for a set of legs, using each leg's own odds."""
    if len(sub)==0: return 0.0,0.0
    ev=0.0
    for _,r in sub.iterrows():
        ev += (dec(r["odds"])-1) if r["hit"] else -1
    hit=sub["hit"].mean()
    return ev/len(sub), hit

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
    df["res"]=df["hit_hr"].astype(str).str.strip()
    r=df[df["res"].isin(["Yes","No"])].copy()
    r["hit"]=r["res"]=="Yes"
    r=r[(r["score"]>=9.0)&(r["odds"]>0)].copy()
    print(f"  {len(r)} qualifying legs (score>=9, plus odds)\n")

    # ── A. FINE BINS (25-wide) — the EV curve shape ──────────────────────
    print("="*72)
    print("  A. EV CURVE — 25-wide odds bins (see where edge peaks)")
    print("="*72)
    print(f"  {'band':<12} {'n':>4} {'hit%':>7} {'be%':>6} {'EV/1u':>8}  bar")
    edges=list(range(100,1000,25))
    for lo in edges:
        hi=lo+24
        sub=r[(r["odds"]>=lo)&(r["odds"]<=hi)]
        if len(sub)<10:
            continue
        ev,hit=ev_of(sub)
        b=be((lo+hi)/2)*100
        bar=("+"*int(ev*40)) if ev>0 else ("-"*int(-ev*40))
        print(f"  +{lo}-{hi:<7} {len(sub):>4} {hit*100:>6.1f}% {b:>5.1f}% {ev:>+7.2f}u  {bar}")

    # ── B. SLIDING WINDOW — find the true best contiguous range ──────────
    print("\n"+"="*72)
    print("  B. SLIDING WINDOW — best contiguous odds range (100-wide)")
    print("="*72)
    WIN=100; STEP=10; FLOOR=40
    best=[]
    lo=100
    while lo+WIN<=1000:
        hi=lo+WIN
        sub=r[(r["odds"]>=lo)&(r["odds"]<hi)]
        if len(sub)>=FLOOR:
            ev,hit=ev_of(sub)
            best.append((lo,hi,len(sub),hit*100,ev))
        lo+=STEP
    best.sort(key=lambda x:-x[4])
    print(f"  Top ranges by EV (100-wide window, {FLOOR}+ sample):")
    print(f"  {'range':<14} {'n':>4} {'hit%':>7} {'EV/1u':>8}")
    for lo,hi,n,hit,ev in best[:8]:
        print(f"  +{lo}-{hi:<8} {n:>4} {hit:>6.1f}% {ev:>+7.2f}u")

    # also try 150-wide (wider sweet spot)
    print(f"\n  Top ranges by EV (150-wide window, {FLOOR}+ sample):")
    best2=[]; lo=100
    while lo+150<=1000:
        hi=lo+150
        sub=r[(r["odds"]>=lo)&(r["odds"]<hi)]
        if len(sub)>=FLOOR:
            ev,hit=ev_of(sub)
            best2.append((lo,hi,len(sub),hit*100,ev))
        lo+=STEP
    best2.sort(key=lambda x:-x[4])
    print(f"  {'range':<14} {'n':>4} {'hit%':>7} {'EV/1u':>8}")
    for lo,hi,n,hit,ev in best2[:6]:
        print(f"  +{lo}-{hi:<8} {n:>4} {hit:>6.1f}% {ev:>+7.2f}u")

    print("\n  The curve (A) shows the shape; the windows (B) find the true")
    print("  peak range regardless of round numbers. If the best window differs")
    print("  a lot from +301-400, retune the parlay pool to the real sweet spot.")
    print("  Caveat: single-leg EV here; parlay pool uses selector-picked legs,")
    print("  so treat this as WHERE to hunt, not final parlay EV.")
    print("\nDone.")

if __name__=="__main__":
    main()
