#!/usr/bin/env python3
"""
ks_parlay_study.py — find the best KS parlay leg selection, honestly.

KS parlays are FAVORITE parlays: proven cells hit 60-86%, so 2-leggers land
~40-55% and 3-leggers ~25-40% — very different from HR lotteries.

Approach (mirrors the disciplined HR work):
  POOL   = legs whose (tier × line × side × odds-sign) cell is proven +EV on
           TRAIN data (n >= MIN_CELL_N and train edge >= EDGE_MIN), AND whose
           own live price clears that cell's hit rate.
  RANK   = several candidate selectors, compared OUT-OF-SAMPLE.
  TICKET = dedupe by pitcher; 3-leg = top1-3; 2-leg = top4+5 and top6+7.

Split by date: TRAIN = earliest 70% of days, TEST = held-out latest 30%.
Nothing about TEST touches cell selection or ranking stats. Writes nothing.
"""
import os, math, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
TRAIN_FRAC=0.70
MIN_CELL_N=15
EDGE_MIN=4.0

SCORE_TIERS=[("12+",12,999),("10-12",10,12),("8-10",8,10),("6-8",6,8),
             ("4-6",4,6),("2-4",2,4),("Under 2",0,2),("Under 0",-999,0)]
LINE_VALS=[4.5,5.5,6.5,7.5]

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))

def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def dec(o): return 1+o/100 if o>=100 else (1+100/abs(o) if o<=-100 else 2.0)
def be_pct(o):
    if o==0:return 100.0
    return abs(o)/(abs(o)+100)*100 if o<0 else 100/(o+100)*100

def tier_of(s):
    for lab,lo,hi in SCORE_TIERS:
        if lo<=s<hi: return lab
    return None

def load_legs(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("KS_All_Scores")
    vals=ws.get_all_values(); df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    pit_col=next((c for c in ["player_name","pitcher_name","pitcher","name"] if c in df.columns),None)
    def g(r,c):return r.get(c,"")
    rows=[]
    for _,r in df.iterrows():
        if str(g(r,"over_hit")).strip() not in ("Yes","No"): continue
        score=sf(g(r,"ks_score")); line=sf(g(r,"k_line"))
        if line<=0: continue
        tier=tier_of(score)
        if tier is None or line not in LINE_VALS: continue
        proj=sf(g(r,"projected_ks"))
        kstr=sf(g(r,"k_per_start_21d")) or sf(g(r,"k_pct_season"))
        base={"date":str(g(r,"date")).strip(),
              "pitcher":str(g(r,pit_col)).strip() if pit_col else "",
              "tier":tier,"line":line,"proj":proj,"kstr":kstr}
        for side,oc,hc in [("Over","over_odds","over_hit"),("Under","under_odds","under_hit")]:
            odds=sf(g(r,oc))
            if odds==0: continue
            win=1 if str(g(r,hc)).strip()=="Yes" else 0
            pe = (proj-line) if side=="Over" else (line-proj)   # signed toward the bet
            rows.append({**base,"side":side,"odds":odds,"win":win,
                         "proj_edge":pe,"be":be_pct(odds),
                         "sign":"minus" if odds<0 else "plus",
                         "cell":(tier,line,side,"minus" if odds<0 else "plus")})
    return pd.DataFrame(rows)

def proven_cells(train):
    """Cells that cleared EDGE_MIN on TRAIN with enough sample. Returns
    {cell: train_hit_rate}."""
    out={}
    for cell,sub in train.groupby("cell"):
        if len(sub)<MIN_CELL_N: continue
        rate=sub["win"].mean()*100
        avg_be=sub["be"].mean()
        if rate-avg_be>=EDGE_MIN:
            out[cell]=rate
    return out

def zstats(train,col):
    m=train[col].mean(); s=train[col].std(ddof=0) or 1.0
    return m,s

def main():
    gc=gcc(); legs=load_legs(gc,os.environ["GOOGLE_SHEET_ID"])
    if legs.empty:
        print("No resolved KS legs."); return
    dates=sorted(legs["date"].unique())
    if len(dates)<6:
        print("Not enough days for a train/test split."); return
    cut=int(len(dates)*TRAIN_FRAC)
    train_d,test_d=set(dates[:cut]),set(dates[cut:])
    train=legs[legs["date"].isin(train_d)].copy()
    test =legs[legs["date"].isin(test_d)].copy()

    cells=proven_cells(train)
    print("="*64); print("KS PARLAY STUDY"); print("="*64)
    print(f"TRAIN days={len(train_d)} ({len(train)} legs)  TEST days={len(test_d)} ({len(test)} legs)")
    print(f"\nPROVEN CELLS (train edge >= {EDGE_MIN}%, n >= {MIN_CELL_N}):")
    for (t,l,side,sign),rate in sorted(cells.items(),key=lambda x:-x[1]):
        print(f"  {t:8s} O/U {l}  {side:5s} {sign:5s}  train hit {rate:.1f}%")
    if not cells:
        print("  none — not enough proven cells yet."); return

    # eligible TEST legs: cell proven AND own price clears the cell hit rate
    test=test[test["cell"].isin(cells)].copy()
    test["cell_rate"]=test["cell"].map(cells)
    test["value"]=test["cell_rate"]-test["be"]      # edge at the leg's REAL price
    test=test[test["value"]>0].copy()
    print(f"\nEligible TEST legs (proven cell + priced +EV): {len(test)}  "
          f"(hit {test['win'].mean()*100:.1f}%)")

    # z-scored composite inputs (train stats, no leakage)
    pe_m,pe_s=zstats(train,"proj_edge")
    selectors={
        "cell value":       lambda r: r["value"],
        "proj_edge":        lambda r: r["proj_edge"],
        "value + proj (z)": lambda r: (r["value"]-EDGE_MIN)/10 + (r["proj_edge"]-pe_m)/pe_s,
        "k strength":       lambda r: r["kstr"],
    }

    def build(fn):
        res={"3-leg":[0,0,0.0,0.0],"2-leg":[0,0,0.0,0.0]}  # formed,won,stake,ret
        for d in test_d:
            day=test[test["date"]==d].copy()
            if day.empty: continue
            day["sel"]=day.apply(fn,axis=1)
            day=day.sort_values("sel",ascending=False)
            picked=[]; seen=set()
            for _,r in day.iterrows():
                if r["pitcher"] in seen: continue
                seen.add(r["pitcher"]); picked.append(r)
            def ticket(legs_):
                stake=1.0; won=all(l["win"]==1 for l in legs_)
                ret=(math.prod(dec(l["odds"]) for l in legs_)) if won else 0.0
                return won,stake,ret
            if len(picked)>=3:
                w,s,rt=ticket(picked[0:3]); res["3-leg"][0]+=1; res["3-leg"][1]+=w
                res["3-leg"][2]+=s; res["3-leg"][3]+=rt
            for i,j in [(3,4),(5,6)]:
                if j<len(picked):
                    w,s,rt=ticket([picked[i],picked[j]]); res["2-leg"][0]+=1; res["2-leg"][1]+=w
                    res["2-leg"][2]+=s; res["2-leg"][3]+=rt
        return res

    print("\n"+"="*64)
    print("OUT-OF-SAMPLE PARLAY BAKE-OFF (1u/ticket)")
    print("="*64)
    print(f"  {'Selector':18s} | {'3-leg formed/won  ROI':26s} | {'2-leg formed/won  ROI'}")
    print("  "+"-"*74)
    for name,fn in selectors.items():
        r=build(fn)
        def fmt(x):
            f,w,st,rt=x; roi=(rt-st)/st*100 if st else 0
            hit=(w/f*100 if f else 0)
            return f"{f:2d}/{w:<2d} {hit:4.0f}%  {roi:+6.1f}%"
        print(f"  {name:18s} | {fmt(r['3-leg']):26s} | {fmt(r['2-leg'])}")
    print("\n  Small TEST set — treat as directional. Best selector = highest 2-leg")
    print("  ROI with a decent hit rate. That becomes the live KS parlay selector.")
    print("Done.")

if __name__=="__main__":
    main()
