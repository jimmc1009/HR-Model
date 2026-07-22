#!/usr/bin/env python3
"""
ks_feature_separation.py — what actually separates a WINNING strikeout bet.

The cell-mining approach failed because each cell had ~30-45 bets. This pools
ALL bets by side and asks the powered question: across every Under (and every
Over), what features distinguish the ones that hit from the ones that missed?
Thousands of bets, not dozens — so the answer is trustworthy.

If a feature cleanly separates winners (big Cohen d) AND its extreme quartile
clears breakeven, THAT feature is a real, poolable edge — the KS equivalent of
HR's score+odds zones. Reads KS_All_Scores, writes nothing.
"""
import os, math, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]

FEATURES=[  # (column, label) — exact KS_All_Scores headers
    ("proj_edge",        "Proj edge (proj−line)"),
    ("ks_score",         "KS score"),
    ("projected_ks",     "Projected Ks"),
    ("k_pct_season",     "K% season"),
    ("swstr_pct",        "SwStr%"),
    ("chase_rate",       "Chase%"),
    ("k_per_9",          "K/9"),
    ("k_per_start_21d",  "K/start 21d"),
    ("avg_ip_per_start", "IP/start"),
    ("fastball_velo",    "FB velo"),
    ("whip_proxy",       "WHIP proxy"),
    ("opp_team_k_pct",   "Opp team K%"),
    ("opp_whiff_rate",   "Opp whiff%"),
    ("opp_chase_rate",   "Opp chase%"),
    ("top6_avg_k_pct",   "Opp top6 K%"),
]

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))

def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def be_pct(o): return abs(o)/(abs(o)+100)*100 if o<0 else 100/(o+100)*100

def load(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("KS_All_Scores")
    vals=ws.get_all_values(); df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    df["k_line"]=df.get("k_line","").apply(lambda x:sf(x))
    df["projected_ks"]=df.get("projected_ks","").apply(lambda x:sf(x))
    df["proj_edge"]=df["projected_ks"]-df["k_line"]
    for col,_ in FEATURES:
        if col in ("proj_edge",): continue
        df[col]=df.get(col, pd.Series([0]*len(df))).apply(lambda x:sf(x))
    for oc in ("over_odds","under_odds"):
        df[oc]=df.get(oc,"").apply(lambda x:sf(x))
    df["over_win"] =(df.get("over_hit","").astype(str).str.strip()=="Yes").astype(int)
    df["under_win"]=(df.get("under_hit","").astype(str).str.strip()=="Yes").astype(int)
    df["resolved"] =df.get("over_hit","").astype(str).str.strip().isin(["Yes","No"])
    return df[df["resolved"]].copy()

def cohens_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2:return 0.0
    va,vb=a.var(ddof=1),b.var(ddof=1)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)) if (na+nb-2)>0 else 0.0
    return (a.mean()-b.mean())/sp if sp>0 else 0.0

def sep_table(res, win_col, side):
    w=res[res[win_col]==1]; l=res[res[win_col]==0]
    n=len(res)
    print("="*70)
    print(f"{side.upper()} SEPARATION — what distinguishes a winning {side} "
          f"(n={n}, hit {len(w)/n*100:.1f}%)")
    print("="*70)
    print(f"  {'Feature':22s} {'win avg':>9s} {'miss avg':>9s} {'diff':>9s} {'Cohen d':>8s} {'r_pb':>7s}")
    print("  "+"-"*68)
    rows=[]
    for col,label in FEATURES:
        if col not in res.columns or res[col].std(ddof=0)==0: continue
        a,b=w[col],l[col]; d=cohens_d(a,b); r=res[win_col].corr(res[col])
        rows.append((abs(d),label,a.mean(),b.mean(),a.mean()-b.mean(),d,r if pd.notna(r) else 0.0))
    for _,label,wa,la,diff,d,r in sorted(rows,reverse=True):
        print(f"  {label:22s} {wa:9.3f} {la:9.3f} {diff:+9.3f} {d:+8.3f} {r:+7.3f}")
    print(f"\n  For {side.upper()}S, a winning bet means the pitcher landed on the {side}"
          f" side.\n  Big |d| = real separator. Sign shows which way winners lean.\n")

def quartile_pool(res, feat, win_col, odds_col, side):
    fr=res[res[feat].notna()].copy()
    if len(fr)<40 or fr[feat].std(ddof=0)==0:
        print(f"  {feat}: not enough spread"); return
    try:
        fr["_q"]=pd.qcut(fr[feat],4,labels=["Q1 low","Q2","Q3","Q4 high"],duplicates="drop")
    except Exception:
        print(f"  {feat}: can't quartile"); return
    print(f"  {side} hit rate by {feat} quartile (+ avg break-even from odds):")
    for lab,sub in fr.groupby("_q",observed=True):
        hit=sub[win_col].mean()*100
        be=sub[odds_col].apply(be_pct).mean()
        edge=hit-be
        flag=" ✅" if edge>=3 else (" ⚠️" if edge>=0 else "")
        print(f"    {str(lab):8s}  hit {hit:5.1f}%   BE {be:5.1f}%   edge {edge:+5.1f}%   n={len(sub)}{flag}")
    print()

def main():
    gc=gcc(); res=load(gc,os.environ["GOOGLE_SHEET_ID"])
    if res.empty:
        print("No resolved KS rows."); return

    sep_table(res,"under_win","under")
    print("  POOL TEST — does a feature carve out profitable UNDERS?")
    quartile_pool(res,"proj_edge","under_win","under_odds","Under")
    quartile_pool(res,"ks_score","under_win","under_odds","Under")

    sep_table(res,"over_win","over")
    print("  POOL TEST — does a feature carve out profitable OVERS?")
    quartile_pool(res,"proj_edge","over_win","over_odds","Over")
    quartile_pool(res,"ks_score","over_win","over_odds","Over")

    print("READ: a quartile whose hit rate clears its own breakeven (✅) on a")
    print("real sample IS a poolable edge — rank within it, like HR zones. If")
    print("every quartile sits below breakeven, the side isn't bettable yet.")
    print("Done.")

if __name__=="__main__":
    main()
