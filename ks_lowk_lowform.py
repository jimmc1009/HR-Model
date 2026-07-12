#!/usr/bin/env python3
"""
ks_lowk_lowform.py — pre-committed, CROSS-VALIDATED test of one idea.

Hypothesis (pre-committed BEFORE this run): a weak strikeout pitcher (low SEASON
K%) who is ALSO pitching badly lately (low recent K/start) is a good UNDER — weak
K-arms get pulled early and never pile up strikeouts, and the line doesn't shade
down enough.

Because this idea was spotted by eyeballing one split, a single held-out window
can't validate it. So we K-FOLD by date: split the season into folds; for each
fold, set the median thresholds on the OTHER folds, then measure the lowK/lowForm
Under edge on the held-out fold. Every bet is graded exactly once, on a fold that
did NOT set its thresholds. Pooled result = honest cross-validated estimate; the
per-fold row shows whether it's robust or one lucky window.
Reads KS_All_Scores, writes nothing.
"""
import os, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
K_FOLDS=4
SEASON_COL="k_pct_season"
RECENT_COL="k_per_start_21d"

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))
def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def be_pct(o): return abs(o)/(abs(o)+100)*100 if o<0 else 100/(o+100)*100

def load_unders(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("KS_All_Scores")
    vals=ws.get_all_values(); df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    rows=[]
    for _,r in df.iterrows():
        if str(r.get("under_hit","")).strip() not in ("Yes","No"): continue
        uo=sf(r.get("under_odds"))
        if uo==0: continue
        rows.append({"date":str(r.get("date","")).strip(),
                     "season":sf(r.get(SEASON_COL)),"recent":sf(r.get(RECENT_COL)),
                     "be":be_pct(uo),
                     "win":1 if str(r.get("under_hit","")).strip()=="Yes" else 0})
    return pd.DataFrame(rows)

def quadlabel(hi,lo):
    if (not hi) and lo:  return "★ lowK / lowForm"
    if hi and lo:        return "  highK / lowForm"
    if hi and not lo:    return "  highK / highForm"
    return "  lowK / highForm"

def report(frame,label):
    n=len(frame)
    if n==0: print(f"  {label:26s}  (n=0)"); return
    hit=frame["win"].mean()*100; be=frame["be"].mean(); edge=hit-be
    flag=" ✅" if edge>=3 else (" ⚠️" if edge>=0 else "")
    print(f"  {label:26s}  hit {hit:5.1f}%   BE {be:5.1f}%   edge {edge:+5.1f}%   n={n}{flag}")

def main():
    gc=gcc(); u=load_unders(gc,os.environ["GOOGLE_SHEET_ID"])
    if u.empty:
        print("No resolved Under legs."); return
    dates=sorted(u["date"].unique())
    if len(dates)<K_FOLDS*2:
        print("Not enough days for K folds."); return

    # contiguous date folds
    fold_of={}
    for i,d in enumerate(dates):
        fold_of[d]=min(i*K_FOLDS//len(dates), K_FOLDS-1)
    u["fold"]=u["date"].map(fold_of)

    print("="*72)
    print("KS UNDER — lowK / lowForm  (pre-committed, %d-fold cross-validated)"%K_FOLDS)
    print("="*72)
    print(f"{len(u)} resolved unders across {len(dates)} days\n")

    graded=[]           # every under, tagged with its held-out quadrant
    print("Per-fold (thresholds set on the OTHER folds):")
    for f in range(K_FOLDS):
        test=u[u["fold"]==f]; train=u[u["fold"]!=f]
        if test.empty or train.empty: continue
        s_med=train["season"].median(); r_med=train["recent"].median()
        t=test.copy()
        t["hi"]=t["season"]>=s_med; t["lo"]=t["recent"]<=r_med
        t["q"]=t.apply(lambda x:quadlabel(x["hi"],x["lo"]),axis=1)
        graded.append(t)
        star=t[t["q"]=="★ lowK / lowForm"]
        rng=f"{test['date'].min()}..{test['date'].max()}"
        report(star, f"fold {f+1} {rng}")

    g=pd.concat(graded,ignore_index=True)
    print("\nPOOLED across folds (each bet graded once, on a held-out fold):")
    for q in ["★ lowK / lowForm","  highK / lowForm","  highK / highForm","  lowK / highForm"]:
        report(g[g["q"]==q], q)
    print("\n  baseline:")
    report(g,"all unders")

    print("\nREAD: the ★ lowK/lowForm line is the one that matters. It's real ONLY")
    print("if it clears breakeven (✅) POOLED, beats the other quadrants, AND is")
    print("green in most folds (not carried by one). If it clears here, it has")
    print("passed a real test — then watch it FORWARD before betting real money.")
    print("Done.")

if __name__=="__main__":
    main()
