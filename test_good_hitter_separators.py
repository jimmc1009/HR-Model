"""
test_good_hitter_separators.py
Restricts to LEGIT power hitters (blend: high HR score AND proven season HR
total), then measures what separates HR days from non-HR days — ranking BOTH
core features and bat-tracking metrics head-to-head.

Why: comparing HR hitters vs everyone is muddied by scrubs. Among good hitters
(who all CAN go deep), what actually tips a game into a HR? That's the bet-
relevant signal.

CAVEAT: bat-tracking is SEASON-level (constant per player across days), so it
separates WHICH hitters homer more, not WHICH days. Rolling features
(barrel_7d, ev_7d) capture day-to-day. Output flags this.

Pool = HR score >= SCORE_MIN AND season HR count >= HR_MIN.
Reads bat-tracking.csv (repo root) + HR_All_Scores. Read-only.
"""
import os, json, time
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
BT_CSV=os.environ.get("BT_CSV","bat-tracking.csv")
SCORE_MIN=float(os.environ.get("SCORE_MIN","9"))   # model rates as viable
HR_MIN=int(os.environ.get("HR_MIN","10"))          # proven season power

BAT=["avg_bat_speed","hard_swing_rate","squared_up_per_swing",
     "blast_per_swing","swing_length"]
CORE=["barrel_pct_7d","avg_ev_7d","season_barrel_pct","iso","hr_per_fb",
      "hard_hit_pct_7d","pull_rate"]

def gcc():
    return gspread.authorize(Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]),scopes=SCOPES))
def wr(fn,r=4,w=20):
    for a in range(r):
        try:return fn()
        except Exception:
            if a<r-1:time.sleep(w)
            else:raise
def sf(v,d=np.nan):
    try:
        f=float(v);return d if(pd.isna(f)or np.isinf(f))else f
    except(ValueError,TypeError):return d
def norm_name(n):
    n=str(n).strip().lower()
    if "," in n:
        last,first=n.split(",",1); n=f"{first.strip()} {last.strip()}"
    return " ".join(n.split())

def eff_size(sub,col,mask):
    h=sub[mask][col]; n=sub[~mask][col]
    if h.notna().sum()<25 or n.notna().sum()<25: return None
    sd=sub[col].std()
    return (h.mean()-n.mean())/sd if sd else 0, h.mean(), n.mean()

def main():
    if not os.path.exists(BT_CSV):
        print(f"CSV not found at {BT_CSV}"); return
    bt=pd.read_csv(BT_CSV); bt.columns=[c.strip().strip('"') for c in bt.columns]
    bt["jkey"]=bt["name"].apply(norm_name)
    for m in BAT:
        if m in bt.columns: bt[m]=bt[m].apply(sf)

    sid=os.environ["GOOGLE_SHEET_ID"];gc=gcc()
    sh=wr(lambda:gc.open_by_key(sid));ws=sh.worksheet("HR_All_Scores")
    vals=wr(lambda:ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)]
    df["jkey"]=df["player_name"].apply(norm_name)
    df["score"]=df.apply(lambda r: sf(r.get("hr_score_corrected")) if str(r.get("hr_score_corrected","")).strip() not in ("","nan","None") else sf(r.get("hr_score")),axis=1)
    df["hit_str"]=df["hit_hr"].astype(str).str.strip()
    for c in CORE:
        if c in df.columns: df[c]=df[c].apply(sf)
    r=df[df["hit_str"].isin(["Yes","No"])].copy()
    r["hit"]=r["hit_str"]=="Yes"

    # season HR count per player (from the data itself)
    hrcount=r.groupby("jkey")["hit"].sum().rename("season_hr")
    r=r.merge(hrcount,on="jkey",how="left")

    # join bat-tracking
    r=r.merge(bt[["jkey"]+[x for x in BAT if x in bt.columns]],on="jkey",how="left")

    # ---- GOOD-HITTER POOL ----
    pool=r[(r["score"]>=SCORE_MIN)&(r["season_hr"]>=HR_MIN)].copy()
    print(f"Pool: score>={SCORE_MIN} AND season_hr>={HR_MIN}")
    print(f"  {len(pool)} player-days from {pool['jkey'].nunique()} hitters "
          f"(of {len(r)} total resolved)\n")
    if len(pool)<200:
        print("  ⚠️  Pool small — loosen SCORE_MIN or HR_MIN via env if needed.\n")

    pool_hr=pool["hit"].mean()*100
    print(f"  Pool HR rate: {pool_hr:.1f}%  (vs {r['hit'].mean()*100:.1f}% overall)\n")

    # ---- separators within the good pool: HR day vs non-HR day ----
    print("="*68)
    print("  WITHIN GOOD HITTERS — what separates HR days from non-HR days")
    print("="*68)
    print(f"  {'feature':<22} {'HRday':>9} {'noHR':>9} {'eff':>8} {'type'}")
    rows=[]
    for col in CORE+BAT:
        if col not in pool.columns: continue
        res=eff_size(pool,col,pool["hit"])
        if res is None: continue
        eff,hm,nm=res
        typ="(rolling)" if col in ("barrel_pct_7d","avg_ev_7d","hard_hit_pct_7d") else \
            "(season/bat)" if col in BAT else "(season)"
        rows.append((col,eff,hm,nm,typ))
    for col,eff,hm,nm,typ in sorted(rows,key=lambda x:-abs(x[1])):
        print(f"  {col:<22} {hm:>9.3f} {nm:>9.3f} {eff:>+8.3f} {typ}")

    print("\n  Ranked by |effect|. NOTE: bat-tracking + season features are")
    print("  CONSTANT per player, so they separate WHICH hitters homer, not")
    print("  which days. Only rolling (7d) features truly capture day-to-day.")
    print("  If a bat metric out-ranks rolling ones here, it's telling you the")
    print("  pool still mixes better/worse hitters — tighten the pool. The")
    print("  day-to-day bet signal lives in the rolling features.")

    # ---- secondary: elite vs merely-good (split pool by season_hr median) ----
    print("\n"+"="*68)
    print("  ELITE vs GOOD (split good pool at median season HR)")
    print("="*68)
    med=pool["season_hr"].median()
    pool["elite"]=pool["season_hr"]>med
    print(f"  Median season HR = {med:.0f}; elite > that\n")
    print(f"  {'feature':<22} {'elite':>9} {'good':>9} {'eff':>8}")
    erows=[]
    for col in CORE+BAT:
        if col not in pool.columns: continue
        res=eff_size(pool,col,pool["elite"])
        if res is None: continue
        eff,em,gm=res
        erows.append((col,eff,em,gm))
    for col,eff,em,gm in sorted(erows,key=lambda x:-abs(x[1])):
        print(f"  {col:<22} {em:>9.3f} {gm:>9.3f} {eff:>+8.3f}")
    print("\n  This shows what distinguishes elite power from merely-good — useful")
    print("  for tiering, but remember it's 'who', not 'when'.")
    print("\nDone.")

if __name__=="__main__":
    main()
