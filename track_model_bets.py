"""
track_model_bets.py
Persistent, self-grading tracker for the dashboard's model picks. Builds a
clean FORWARD track record so the corrected model can prove/disprove itself.

TWO STEPS each run:
  1. LOG today's model picks (singles + parlays) to a persistent tab
     'Model_Bet_Tracker' — APPENDS, never overwrites. Deduped by (date, bet_type,
     player-set) so re-running the same day won't double-log.
  2. RESOLVE any pending logged picks by looking up hit_hr from HR_All_Scores
     (which resolve_picks.py already grades). Then recompute running P&L/ROI.

Bet types tracked: single, 2leg (top4+5, top6+7), 3leg (top1-3).
Parlays graded as all-legs-hit. Uses model/consensus odds as the price proxy.
Flat 1u per ticket.

Designed to run daily AFTER the dashboard + resolve step. Read-mostly; only
writes to its own Model_Bet_Tracker tab, never touches HR_All_Scores.
"""
import os, json, time
from datetime import datetime
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
TRACKER_TAB="Model_Bet_Tracker"
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
    if o<0:return abs(o)/(abs(o)+100.0)
    return 1.0
def tier(s):
    if s>=13:return "13+"
    if s>=12:return "12-13"
    if s>=11:return "11-12"
    if s>=10:return "10-11"
    if s>=9:return "9-10"
    if s>=8.5:return "8.5-9"
    return "<8.5"
def zone(o):
    o=sf(o)
    if o<=300:return "≤+300"
    if o<=499:return "+301-499"
    if o<=699:return "+500-699"
    return "+700+"

# ── pool + selector (mirror dashboard) ────────────────────────────────────
def in_singles(score,odds,zr):
    if score<9.0: return False
    z=zr.get((tier(score),zone(odds)))
    return z is not None and (z-be(odds))>0
def in_parlay_pool(s,o):
    if o<=0:return False
    if 9.0<=s<10.0 and 301<=o<=400:return True
    if 12.0<=s<13.0 and 301<=o<=400:return True
    if s>=13.0 and 301<=o<=400:return True
    if s>=13.0 and o<=300:return True
    return False

def load_hr_scores(gc,sid):
    sh=wr(lambda:gc.open_by_key(sid));ws=sh.worksheet("HR_All_Scores")
    vals=wr(lambda:ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    df["score"]=df.apply(lambda r:coalesce(r,"hr_score_corrected","hr_score"),axis=1)
    df["odds"]=df["consensus_odds"].apply(lambda x:sf(x,0))
    df["hit_str"]=df["hit_hr"].astype(str).str.strip()
    return sh,df

def get_or_create_tracker(sh):
    try:
        ws=sh.worksheet(TRACKER_TAB)
    except gspread.WorksheetNotFound:
        ws=sh.add_worksheet(title=TRACKER_TAB,rows=2000,cols=12)
        ws.update([["date","bet_type","players","odds_each","combined_odds",
                    "result","pnl_units","running_total","logged_at"]])
    return ws

def build_todays_picks(df_today,zr):
    """Return list of picks to log for today: singles + 2leg x2 + 3leg."""
    picks=[]
    # ranked parlay pool (selector = blend1: hr_per_fb/8 + edge*0.8)
    pool=df_today[df_today.apply(lambda r:in_parlay_pool(r["score"],r["odds"]),axis=1)].copy()
    def edge(r):
        z=zr.get((tier(r["score"]),zone(r["odds"])))
        return (z-be(r["odds"]))*100 if z is not None else -99
    if not pool.empty:
        pool["sel"]=pool.apply(lambda r: sf(r.get("hr_per_fb",0))/8 + edge(r)*0.8,axis=1)
        pool=pool.sort_values("sel",ascending=False)
        div=[];used=set()
        for _,row in pool.iterrows():
            pn=str(row.get("pitcher_name","")).strip()
            if pn and pn in used:continue
            div.append(row); 
            if pn:used.add(pn)
        # 3-leg top1-3
        if len(div)>=3:
            legs=div[:3]
            combo=1.0
            for l in legs: combo*=dec(l["odds"])
            picks.append(("3leg",[l["player_name"] for l in legs],
                          [int(sf(l["odds"])) for l in legs],
                          int(round((combo-1)*100))))
        # 2-leg top4+5 and top6+7
        for slot in [(3,4),(5,6)]:
            i,j=slot
            if j<len(div):
                a,b=div[i],div[j]
                combo=dec(a["odds"])*dec(b["odds"])
                picks.append(("2leg",[a["player_name"],b["player_name"]],
                              [int(sf(a["odds"])),int(sf(b["odds"]))],
                              int(round((combo-1)*100))))
    # singles: positive-edge, score>=9
    for _,row in df_today.iterrows():
        if in_singles(row["score"],row["odds"],zr):
            picks.append(("single",[row["player_name"]],[int(sf(row["odds"]))],
                          int(sf(row["odds"]))))
    return picks

def main():
    sid=os.environ["GOOGLE_SHEET_ID"];gc=gcc()
    print("Loading HR_All_Scores...")
    sh,df=load_hr_scores(gc,sid)

    # zone hit rates from resolved rows
    res=df[df["hit_str"].isin(["Yes","No"])].copy()
    res["hit"]=res["hit_str"]=="Yes"
    zr={}
    res["t"]=res["score"].apply(tier);res["z"]=res["odds"].apply(zone)
    for (t,z),sub in res.groupby(["t","z"]):
        if len(sub)>=MIN_SAMPLE: zr[(t,z)]=sub["hit"].mean()

    # today = latest date present
    latest=df["date"].dropna().astype(str).max()
    df_today=df[df["date"].astype(str)==latest].copy()
    print(f"Latest date: {latest} ({len(df_today)} players)")

    ws=get_or_create_tracker(sh)
    existing=wr(lambda:ws.get_all_values())
    ehead=existing[0] if existing else []
    erows=existing[1:] if len(existing)>1 else []
    logged_keys=set((r[0],r[1],r[2]) for r in erows if len(r)>=3)

    # ── STEP 1: log today's picks (dedup) ────────────────────────────────
    picks=build_todays_picks(df_today,zr)
    new_rows=[]
    for bet_type,players,odds_each,combined in picks:
        key=(latest,bet_type,"|".join(str(p) for p in players))
        if key in logged_keys: continue
        new_rows.append([latest,bet_type,"|".join(str(p) for p in players),
                         ",".join(str(o) for o in odds_each),str(combined),
                         "Pending","","",datetime.utcnow().isoformat()])
    if new_rows:
        wr(lambda:ws.append_rows(new_rows))
        print(f"Logged {len(new_rows)} new picks for {latest}")
    else:
        print(f"No new picks to log for {latest} (already logged or none qualify)")

    # ── STEP 2: resolve pending picks via HR_All_Scores hit_hr ───────────
    # outcome lookup: (date, player_norm) -> Yes/No
    def norm(n): return str(n).strip().lower()
    outcome={}
    for _,r in df.iterrows():
        if r["hit_str"] in ("Yes","No"):
            outcome[(str(r["date"]),norm(r["player_name"]))]=r["hit_str"]

    allvals=wr(lambda:ws.get_all_values())
    head=allvals[0]; rows=allvals[1:]
    ci={h:i for i,h in enumerate(head)}
    updates=0
    for ri,row in enumerate(rows):
        if len(row)<len(head): row=row+[""]*(len(head)-len(row))
        if row[ci["result"]].strip()!="Pending": continue
        d=row[ci["date"]]; players=row[ci["players"]].split("|")
        odds_each=[sf(x) for x in row[ci["odds_each"]].split(",") if x!=""]
        # look up each leg
        legs_res=[outcome.get((d,norm(p))) for p in players]
        if any(lr is None for lr in legs_res):
            continue  # not all legs resolved yet — leave pending
        all_hit=all(lr=="Yes" for lr in legs_res)
        bet_type=row[ci["bet_type"]]
        if bet_type=="single":
            pnl=(dec(odds_each[0])-1) if all_hit else -1.0
        else:
            combo=1.0
            for o in odds_each: combo*=dec(o)
            pnl=(combo-1) if all_hit else -1.0
        row[ci["result"]]="Won" if all_hit else "Lost"
        row[ci["pnl_units"]]=f"{pnl:.3f}"
        rows[ri]=row; updates+=1

    # Recompute the running cumulative P&L total across ALL graded rows, in
    # sheet order (chronological), so the running_total column shows the trend
    # after every settled bet — like the Pikkit chart, but for model picks.
    if "running_total" in ci:
        cum=0.0
        for ri,row in enumerate(rows):
            if len(row)<len(head): row=row+[""]*(len(head)-len(row))
            if row[ci["result"]].strip() in ("Won","Lost"):
                cum+=sf(row[ci["pnl_units"]])
                row[ci["running_total"]]=f"{cum:.3f}"
                rows[ri]=row

    if updates or "running_total" in ci:
        wr(lambda:ws.update([head]+rows))
        print(f"Resolved {updates} pending picks; running total updated")
    else:
        print("No newly resolvable pending picks")

    # ── STEP 3: running P&L summary by bet type ──────────────────────────
    graded=[r for r in rows if len(r)>ci["result"] and r[ci["result"]] in ("Won","Lost")]
    print("\n"+"="*56); print("  RUNNING MODEL TRACK RECORD (forward, self-graded)"); print("="*56)
    for bt in ["single","2leg","3leg","ALL"]:
        sub=[r for r in graded if bt=="ALL" or r[ci["bet_type"]]==bt]
        if not sub: continue
        n=len(sub); w=sum(1 for r in sub if r[ci["result"]]=="Won")
        pnl=sum(sf(r[ci["pnl_units"]]) for r in sub)
        roi=pnl/n*100 if n else 0
        print(f"  {bt:<8} n={n:>4}  W={w:>3}  hit={w/n*100:>5.1f}%  P&L={pnl:>+7.2f}u  ROI={roi:>+6.1f}%")
    # Running cumulative total across everything graded
    total_pnl=sum(sf(r[ci["pnl_units"]]) for r in graded)
    print("-"*56)
    print(f"  RUNNING TOTAL P&L (all model picks): {total_pnl:+.2f}u")
    print("="*56)
    print("  This is the CLEAN forward record on corrected picks. Let it")
    print("  build a few weeks before judging. Singles vs parlays shown")
    print("  separately so you can see which (if either) beats the vig.")
    print("\nDone.")

if __name__=="__main__":
    main()
