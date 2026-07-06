"""
test_odds_sweetspot.py
Read-only. Finds the odds SWEET SPOT for HR legs — where hit rate and payout
multiply to the best expected value. Not max odds (longshots never hit), not
min odds (favorites pay nothing) — the bang-for-buck middle.

For every leg in the value-zone pool (corrected data), buckets by odds band and
reports: n, hit rate, breakeven, edge, and EV per 1u bet (the real bang-for-buck
number = does hit rate beat breakeven, and by how much in $ terms).

Then does the same for 2-leg parlays built WITHIN each odds band, so you see
which single-leg odds range makes the best parlay legs.
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

# odds bands to sweep
BANDS=[("+100-200",100,200),("+201-300",201,300),("+301-400",301,400),
       ("+401-500",401,500),("+501-650",501,650),("+651-850",651,850),
       ("+851-1200",851,1200),("+1201+",1201,999999)]

def band_of(o):
    for name,lo,hi in BANDS:
        if lo<=o<=hi: return name
    return None

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
    df["res"]=df["hit_hr"].astype(str).str.strip()
    r=df[df["res"].isin(["Yes","No"])].copy()
    r["hit"]=r["res"]=="Yes"

    # Only legs that qualify by SCORE (>=9, the model's real floor) — we're
    # asking about odds sweet spot AMONG legitimate plays, not junk.
    r=r[r["score"]>=9.0].copy()
    r["band"]=r["odds"].apply(band_of)
    r=r[r["band"].notna()]

    print(f"  {len(r)} qualifying legs (score>=9)\n")
    print("="*74)
    print("  SINGLE-LEG BANG FOR BUCK by odds band (score>=9 legs)")
    print("="*74)
    print(f"  {'band':<11} {'n':>4} {'hit%':>7} {'be%':>6} {'edge':>7} {'EV/1u':>8}  {'verdict'}")
    best_ev=None; best_band=None
    for name,lo,hi in BANDS:
        sub=r[r["band"]==name]
        if len(sub)<MIN_SAMPLE: 
            if len(sub)>0:
                print(f"  {name:<11} {len(sub):>4}  (thin, <{MIN_SAMPLE})")
            continue
        hr=sub["hit"].mean()
        avg_o=sub["odds"].mean()
        b=be(avg_o)
        edge=(hr-b)*100
        # EV per 1u: hit_rate*payout - (1-hit_rate)*1
        ev=hr*(dec(avg_o)-1) - (1-hr)
        v="✅ +EV" if ev>0 else "❌ -EV"
        print(f"  {name:<11} {len(sub):>4} {hr*100:>6.1f}% {b*100:>5.1f}% {edge:>+6.1f}% {ev:>+7.2f}u  {v}")
        if ev>0 and (best_ev is None or ev>best_ev):
            best_ev=ev; best_band=name

    print()
    if best_band:
        print(f"  → SWEET SPOT (best +EV single-leg band): {best_band} (EV {best_ev:+.2f}u/leg)")
    else:
        print("  → No band is clearly +EV at single-leg level on this sample.")

    # 2-leg parlays built within each band
    print("\n"+"="*74)
    print("  2-LEG PARLAY bang for buck by odds band (both legs same band)")
    print("="*74)
    print(f"  {'band':<11} {'pairs':>6} {'hit%':>7} {'avgOdds':>8} {'EV/1u':>8}  {'verdict'}")
    days=sorted(r["date"].unique())
    for name,lo,hi in BANDS:
        pairs=0; wins=0; ret=0.0; oddsum=0.0
        for d in days:
            legs=r[(r["date"]==d)&(r["band"]==name)]
            legs=legs.drop_duplicates(subset=["player_name"])
            legs=legs.sort_values("score",ascending=False)
            L=legs.to_dict("records")
            # pair sequentially, non-overlapping, diversify by pitcher
            used=set(); picked=[]
            for leg in L:
                pn=str(leg.get("pitcher_name","")).strip()
                if pn and pn in used: continue
                picked.append(leg); 
                if pn: used.add(pn)
            i=0
            while i+1<len(picked):
                a,b2=picked[i],picked[i+1]
                pairs+=1
                combo=dec(a["odds"])*dec(b2["odds"]); oddsum+=combo
                if a["hit"] and b2["hit"]:
                    wins+=1; ret+=combo
                i+=2
        if pairs<8:
            if pairs>0:
                print(f"  {name:<11} {pairs:>6}  (thin)")
            continue
        hr=wins/pairs
        avg_combo=(oddsum/pairs-1)*100
        ev=(ret-pairs)/pairs
        v="✅ +EV" if ev>0 else "❌ -EV"
        print(f"  {name:<11} {pairs:>6} {hr*100:>6.1f}% {'+'+str(int(avg_combo)):>8} {ev:>+7.2f}u  {v}")

    print("\n  Bang for buck = EV/1u (expected profit per unit). Positive = the")
    print("  band pays more than it costs long-run. Sweet spot balances a hit")
    print("  rate high enough to cash with odds high enough to pay. Use the")
    print("  single-leg sweet-spot band to focus your parlay legs.")
    print("\nDone.")

if __name__=="__main__":
    main()
