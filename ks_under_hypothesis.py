#!/usr/bin/env python3
"""
ks_under_hypothesis.py — test ONE pre-committed idea, out-of-sample.

Hypothesis (from within-zone separation): a genuinely good strikeout pitcher
(high SEASON K%) who is in a recent DIP (low recent K/start) is a good UNDER —
the book's line is inflated by his reputation, current form undercuts it.

Method (no cherry-picking):
  - Split by date: TRAIN = earliest 70% of days, TEST = held-out latest 30%.
  - Define the split points (median season K%, median recent K/start) on TRAIN.
  - The hypothesis pool = season K% >= median AND recent K/start <= median.
  - Report Under hit rate vs breakeven for ALL FOUR quadrants on TEST, so we see
    whether the hypothesis quadrant actually stands out — plus a train-vs-test
    replication check on the hypothesis quadrant itself.
Reads KS_All_Scores, writes nothing.
"""
import os, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
TRAIN_FRAC=0.70
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
                     "odds":uo,"be":be_pct(uo),
                     "win":1 if str(r.get("under_hit","")).strip()=="Yes" else 0})
    return pd.DataFrame(rows)

def quad(row,s_med,r_med):
    hi = row["season"]>=s_med
    lo = row["recent"]<=r_med
    if hi and lo:  return "★ HYPOTHESIS  highK / lowForm"
    if hi and not lo: return "  highK / highForm"
    if (not hi) and lo: return "  lowK / lowForm"
    return "  lowK / highForm"

def summarize(frame,label):
    n=len(frame)
    if n==0:
        print(f"  {label:32s}  (no legs)"); return
    hit=frame["win"].mean()*100; be=frame["be"].mean(); edge=hit-be
    flag=" ✅" if edge>=3 else (" ⚠️" if edge>=0 else "")
    print(f"  {label:32s}  hit {hit:5.1f}%   BE {be:5.1f}%   edge {edge:+5.1f}%   n={n}{flag}")

def main():
    gc=gcc(); u=load_unders(gc,os.environ["GOOGLE_SHEET_ID"])
    if u.empty:
        print("No resolved Under legs."); return
    dates=sorted(u["date"].unique())
    if len(dates)<6:
        print("Not enough days."); return
    cut=int(len(dates)*TRAIN_FRAC)
    train=u[u["date"].isin(set(dates[:cut]))].copy()
    test =u[u["date"].isin(set(dates[cut:]))].copy()
    s_med=train["season"].median(); r_med=train["recent"].median()

    print("="*72)
    print("KS UNDER HYPOTHESIS — high season K% + low recent K/start -> Under")
    print("="*72)
    print(f"TRAIN {len(train)} unders / TEST {len(test)} unders")
    print(f"Split points (from TRAIN): season {SEASON_COL} >= {s_med:.1f} = 'highK',")
    print(f"                           recent {RECENT_COL} <= {r_med:.1f} = 'lowForm'")
    print(f"\nBaseline — ALL test unders:")
    summarize(test,"all unders")

    for fr,tag in [(test,"TEST")]:
        fr=fr.copy(); fr["q"]=fr.apply(lambda x:quad(x,s_med,r_med),axis=1)
        print(f"\n{tag} — Under hit rate by quadrant:")
        order=["★ HYPOTHESIS  highK / lowForm","  highK / highForm",
               "  lowK / lowForm","  lowK / highForm"]
        for q in order:
            summarize(fr[fr["q"]==q],q)

    # replication: hypothesis quadrant on train vs test
    def hypo(fr): return fr[(fr["season"]>=s_med)&(fr["recent"]<=r_med)]
    print("\nReplication check (hypothesis quadrant):")
    summarize(hypo(train),"TRAIN highK/lowForm")
    summarize(hypo(test), "TEST  highK/lowForm")

    print("\nREAD: hypothesis holds only if ★ clears breakeven (✅) on TEST with a")
    print("real n, AND beats the other quadrants, AND train≈test in the replication")
    print("check. If ★ is below breakeven or no better than the rest, the nice")
    print("story doesn't survive contact with held-out data.")
    print("Done.")

if __name__=="__main__":
    main()
