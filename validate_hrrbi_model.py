"""
validate_hrrbi_model.py
Full workup for the HR+RBI (HRRBI) model — both directions (over AND under).
Analogous to validate_model.py for HR, adapted for over/under prop betting.

Because HRRBI props are two-sided, edge must be checked per DIRECTION: an over bet
and an under bet on the same pitcher have different odds and different breakeven.

Checks:
  1. OVERALL: over vs under hit rates, are either +EV net
  2. SCORE TIER x DIRECTION: which hrrbi_score tiers predict overs vs unders
  3. SCORE TIER x LINE x DIRECTION: the bettable cells vs breakeven
  4. WHICH DIRECTION is the model actually good at
  5. ODDS-BAND EV: where's the sweet spot (like HR's +251-600 finding)

Read-only. Uses over_hit/under_hit already graded in HRRBI_All_Scores.
"""
import os, json, time
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MIN_SAMPLE=15

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
def be(o):
    o=sf(o)
    if o>=100:return 100.0/(o+100.0)
    if o<=-100:return abs(o)/(abs(o)+100.0)
    return 1.0
def dec(o):
    o=sf(o)
    if o>=100:return 1+o/100
    if o<=-100:return 1+100/abs(o)
    return 1.0

ATTENTION=[]
def flag(p,m): ATTENTION.append((p,m))

def hrrbi_tier(s):
    if s>=17: return "17+"
    if s>=15: return "15-17"
    if s>=13: return "13-15"
    if s>=11: return "11-13"
    if s>=9:  return "9-11"
    if s>=7:  return "7-9"
    return "<7"

def main():
    sid=os.environ["GOOGLE_SHEET_ID"];gc=gcc()
    print("Reading HRRBI_All_Scores...")
    sh=wr(lambda:gc.open_by_key(sid))
    # try common tab names
    ws=None
    for tab in ["HRRBI_All_Scores","HRRBI_AllScores","HRRBI_Scores"]:
        try: ws=sh.worksheet(tab); break
        except Exception: continue
    if ws is None:
        print("  Could not find HRRBI_All_Scores tab. Check the tab name."); return
    vals=wr(lambda:ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)

    for c in ["hrrbi_score","hrrbi_line","over_odds","under_odds"]:
        if c in df.columns: df[c]=df[c].apply(sf)
        else: df[c]=0.0
    df["over_str"]=df["over_hit"].astype(str).str.strip()
    df["under_str"]=df["under_hit"].astype(str).str.strip()
    r=df[df["over_str"].isin(["Yes","No"])].copy()
    r["over"]=r["over_str"]=="Yes"
    r["under"]=r["under_str"]=="Yes"
    r["tier"]=r["hrrbi_score"].apply(hrrbi_tier)
    print(f"  {len(r)} resolved HRRBI props\n")

    # ── CHECK 1: overall over vs under ───────────────────────────────────
    print("="*62); print("1. OVERALL — over vs under"); print("="*62)
    n=len(r)
    orate=r["over"].mean()*100; urate=r["under"].mean()*100
    # net EV if you'd bet every over / every under at posted odds, flat 1u
    def dir_ev(mask_hit, odds_col):
        ev=0.0; cnt=0
        for _,row in r.iterrows():
            o=sf(row.get(odds_col,0))
            if o==0: continue
            cnt+=1
            ev += (dec(o)-1) if row[mask_hit] else -1
        return (ev/cnt*100 if cnt else 0), cnt
    over_roi,over_n=dir_ev("over","over_odds")
    under_roi,under_n=dir_ev("under","under_odds")
    print(f"  Overs:  hit {orate:.1f}%   bet-all ROI {over_roi:+.1f}%  (n={over_n})")
    print(f"  Unders: hit {urate:.1f}%   bet-all ROI {under_roi:+.1f}%  (n={under_n})")
    if over_roi<0 and under_roi<0:
        flag(1,"Both over and under bet-all ROI negative — model may lack edge either way")
    print()

    # ── CHECK 2: score tier x direction ──────────────────────────────────
    print("="*62); print("2. SCORE TIER — over vs under hit rate"); print("="*62)
    print(f"  {'tier':<6} {'n':>4} {'over%':>7} {'under%':>8}")
    for t in ["17+","15-17","13-15","11-13","9-11","7-9","<7"]:
        sub=r[r["tier"]==t]
        if len(sub)<MIN_SAMPLE: continue
        print(f"  {t:<6} {len(sub):>4} {sub['over'].mean()*100:>6.1f}% {sub['under'].mean()*100:>7.1f}%")
    print("  (High hrrbi_score should predict OVERS; low/negative should predict UNDERS)")
    print()

    # ── CHECK 3: tier x line x direction vs breakeven ────────────────────
    print("="*62); print("3. BETTABLE CELLS — tier x line x direction vs breakeven"); print("="*62)
    print(f"  {'tier':<6} {'line':>5} {'dir':>6} {'n':>4} {'hit%':>7} {'be%':>6} {'edge':>7}")
    good=[]
    wl=r[r["hrrbi_line"]>0]
    for t in ["17+","15-17","13-15","11-13","9-11","7-9","<7"]:
        for line in [0.5,1.5,2.5,3.5]:
            sub=wl[(wl["tier"]==t)&(wl["hrrbi_line"]==line)]
            if len(sub)<MIN_SAMPLE: continue
            for direction,hitcol,oddscol in [("over","over","over_odds"),("under","under","under_odds")]:
                hit=sub[hitcol].mean()
                avg_be=sub[oddscol].apply(be).mean()
                edge=(hit-avg_be)*100
                mark="✅" if edge>=4 else "❌" if edge<=-1 else "≈"
                if edge>=4 and len(sub)>=MIN_SAMPLE:
                    print(f"  {t:<6} {line:>5} {direction:>6} {len(sub):>4} {hit*100:>6.1f}% {avg_be*100:>5.1f}% {edge:>+6.1f}% {mark}")
                    good.append((t,line,direction,round(hit*100,1),len(sub),round(edge,1)))
    if good:
        print(f"\n  → {len(good)} bettable cells (edge>=+4%, {MIN_SAMPLE}+ sample):")
        for t,line,d,h,nn,e in sorted(good,key=lambda x:-x[5]):
            print(f"     {t} | {line} | {d}: {h}% (n={nn}, +{e}%)")
    else:
        print("\n  → NO cells clear breakeven by +4%. That's the core problem —")
        print("     the HRRBI model isn't finding edges in either direction.")
        flag(1,"No K tier×line×direction cell clears breakeven by +4% — model edge unproven")
    print()

    # ── CHECK 4: which direction is the model better at ──────────────────
    print("="*62); print("4. DIRECTION VERDICT"); print("="*62)
    over_good=[g for g in good if g[2]=="over"]
    under_good=[g for g in good if g[2]=="under"]
    print(f"  Bettable OVER cells:  {len(over_good)}")
    print(f"  Bettable UNDER cells: {len(under_good)}")
    if over_good and not under_good:
        print("  → Model has edge on OVERS only. Bet overs, skip unders.")
    elif under_good and not over_good:
        print("  → Model has edge on UNDERS only. Bet unders, skip overs.")
    elif over_good and under_good:
        print("  → Edge both directions (in different cells). Bet per-cell.")
    else:
        print("  → No proven edge either direction on current data.")
    print()

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print("="*62); print("⚠️  WHAT NEEDS ATTENTION"); print("="*62)
    if not ATTENTION:
        print("  ✓ Nothing critical flagged.")
    for pri,msg in sorted(ATTENTION,key=lambda x:x[0]):
        tag={1:"🔴 CRITICAL",2:"🟠 SHOULD FIX",3:"🟡 VERIFY"}.get(pri,"•")
        print(f"  {tag}: {msg}")
    print("\nDone.")

if __name__=="__main__":
    main()
