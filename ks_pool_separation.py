#!/usr/bin/env python3
"""
ks_pool_separation.py — inside each strong KS zone, what separated wins from losses.

Finds the strong cells (tier × line × side × odds-sign) on the FULL current data
(hit − breakeven >= EDGE_MIN, n >= MIN_CELL_N), then WITHIN each one compares the
bets that hit vs the bets that missed, feature by feature. Answers: "for this
zone, what did the winners have that the losers didn't?"

CAVEAT baked into the output: these zones are small (~15-55 bets), so once you
split into wins vs losses you're often at single digits per side. Treat any
separator here as a lead to watch, not proof. Reads KS_All_Scores, writes nothing.
"""
import os, math, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
MIN_CELL_N=15
EDGE_MIN=4.0
MIN_SPLIT=5     # need at least this many wins AND losses to attempt a split

SCORE_TIERS=[("12+",12,999),("10-12",10,12),("8-10",8,10),("6-8",6,8),
             ("4-6",4,6),("2-4",2,4),("Under 2",0,2),("Under 0",-999,0)]
LINE_VALS=[4.5,5.5,6.5,7.5]

FEATURES=[
    ("proj_edge","Proj edge (proj−line)"),("ks_score","KS score"),
    ("projected_ks","Projected Ks"),("k_pct_season","K% season"),
    ("swstr_pct","SwStr%"),("chase_rate","Chase%"),("k_per_9","K/9"),
    ("k_per_start_21d","K/start 21d"),("avg_ip_per_start","IP/start"),
    ("fastball_velo","FB velo"),("whip_proxy","WHIP proxy"),
    ("opp_team_k_pct","Opp team K%"),("opp_whiff_rate","Opp whiff%"),
    ("opp_chase_rate","Opp chase%"),("top6_avg_k_pct","Opp top6 K%"),
]

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))
def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def be_pct(o): return abs(o)/(abs(o)+100)*100 if o<0 else 100/(o+100)*100
def tier_of(s):
    for lab,lo,hi in SCORE_TIERS:
        if lo<=s<hi: return lab
    return None
def cohens_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2:return 0.0
    va,vb=a.var(ddof=1),b.var(ddof=1)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)) if (na+nb-2)>0 else 0.0
    return (a.mean()-b.mean())/sp if sp>0 else 0.0

def load_legs(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("KS_All_Scores")
    vals=ws.get_all_values(); df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    for c in ["k_line","projected_ks","over_odds","under_odds"]+[f for f,_ in FEATURES if f not in("proj_edge","ks_score")]+["ks_score"]:
        if c in df.columns: df[c]=df[c].apply(sf)
    df["proj_edge"]=df.get("projected_ks",0)-df.get("k_line",0)
    rows=[]
    for _,r in df.iterrows():
        if str(r.get("over_hit","")).strip() not in ("Yes","No"): continue
        score=sf(r.get("ks_score")); line=sf(r.get("k_line"))
        tier=tier_of(score)
        if tier is None or line not in LINE_VALS: continue
        feats={f:sf(r.get(f)) for f,_ in FEATURES}
        for side,oc,hc in [("Over","over_odds","over_hit"),("Under","under_odds","under_hit")]:
            odds=sf(r.get(oc))
            if odds==0: continue
            win=1 if str(r.get(hc,"")).strip()=="Yes" else 0
            rows.append({**feats,"tier":tier,"line":line,"side":side,"odds":odds,"win":win,
                         "be":be_pct(odds),
                         "cell":(tier,line,side,"minus" if odds<0 else "plus")})
    return pd.DataFrame(rows)

def main():
    gc=gcc(); legs=load_legs(gc,os.environ["GOOGLE_SHEET_ID"])
    if legs.empty:
        print("No resolved KS legs."); return

    # strong cells on FULL data
    strong=[]
    for cell,sub in legs.groupby("cell"):
        if len(sub)<MIN_CELL_N: continue
        hit=sub["win"].mean()*100; be=sub["be"].mean()
        if hit-be>=EDGE_MIN: strong.append((cell,len(sub),hit,be))
    strong.sort(key=lambda x:-(x[2]-x[3]))

    print("="*72)
    print(f"STRONG KS ZONES (full data, hit−BE >= {EDGE_MIN}%, n >= {MIN_CELL_N})")
    print("="*72)
    if not strong:
        print("  none found."); return
    for (t,l,side,sign),n,hit,be in strong:
        print(f"  {t:8s} O/U {l}  {side:5s} {sign:5s}  n={n}  hit {hit:.1f}%  (edge {hit-be:+.1f}%)")

    for (cell,n,hit,be) in strong:
        t,l,side,sign=cell
        sub=legs[legs["cell"]==cell]
        w=sub[sub["win"]==1]; ls=sub[sub["win"]==0]
        print("\n"+"-"*72)
        print(f"ZONE  {t} | O/U {l} | {side} | {sign}   ({len(w)} wins vs {len(ls)} losses)")
        print("-"*72)
        if len(w)<MIN_SPLIT or len(ls)<MIN_SPLIT:
            print(f"  Too few to split ({len(w)}W/{len(ls)}L) — need >= {MIN_SPLIT} each. Skipping.")
            continue
        rows=[]
        for col,label in FEATURES:
            if col not in sub.columns or sub[col].std(ddof=0)==0: continue
            a,b=w[col],ls[col]; d=cohens_d(a,b)
            rows.append((abs(d),label,a.mean(),b.mean(),a.mean()-b.mean(),d))
        print(f"  {'Feature':22s} {'WIN avg':>9s} {'LOSS avg':>9s} {'diff':>9s} {'Cohen d':>8s}")
        for _,label,wa,la,diff,d in sorted(rows,reverse=True)[:8]:
            print(f"  {label:22s} {wa:9.3f} {la:9.3f} {diff:+9.3f} {d:+8.3f}")

    print("\nREAD: big |Cohen d| = that feature pulled wins apart from losses IN")
    print("this zone. But wins/losses counts are small — treat as a lead to watch")
    print("as the zone fills, not a proven filter. Consistent separators appearing")
    print("across MULTIPLE zones are the ones worth trusting.")
    print("Done.")

if __name__=="__main__":
    main()
