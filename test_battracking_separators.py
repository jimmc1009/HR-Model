"""
test_battracking_separators.py
Measures whether Statcast BAT-TRACKING metrics (bat speed, squared-up%, blast%,
swing length, hard-swing rate) separate HR hitters from non-HR — and whether
they add signal BEYOND the existing barrel%/ISO core (correlation check).

Reads a manually-downloaded Savant bat-tracking CSV (season-level) and joins it
to HR_All_Scores by player name, then treats each metric as a season-context
feature and measures separation on resolved HR outcomes.

CSV expected at /mnt/user-data/uploads/bat-tracking.csv (or set BT_CSV env).
Read-only on the sheet.
"""
import os, json, time
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
BT_CSV=os.environ.get("BT_CSV","bat-tracking.csv")

METRICS=["avg_bat_speed","hard_swing_rate","squared_up_per_bat_contact",
         "squared_up_per_swing","blast_per_bat_contact","blast_per_swing",
         "swing_length","whiff_per_swing"]
# existing core features to test redundancy against
CORE=["season_barrel_pct","iso","hr_per_fb","barrel_pct_7d","avg_ev_7d"]

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
    if "," in n:  # "Caminero, Junior" -> "junior caminero"
        last,first=n.split(",",1)
        n=f"{first.strip()} {last.strip()}"
    return " ".join(n.split())

def main():
    # ---- load bat-tracking CSV ----
    if not os.path.exists(BT_CSV):
        print(f"CSV not found at {BT_CSV}. Set BT_CSV env or upload it."); return
    bt=pd.read_csv(BT_CSV)
    bt.columns=[c.strip().strip('"') for c in bt.columns]
    bt["jkey"]=bt["name"].apply(norm_name)
    for m in METRICS:
        if m in bt.columns: bt[m]=bt[m].apply(sf)
    print(f"Bat-tracking rows: {len(bt)}")

    # ---- load HR_All_Scores ----
    sid=os.environ["GOOGLE_SHEET_ID"];gc=gcc()
    sh=wr(lambda:gc.open_by_key(sid));ws=sh.worksheet("HR_All_Scores")
    vals=wr(lambda:ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)]
    df["jkey"]=df["player_name"].apply(norm_name)
    df["hit_str"]=df["hit_hr"].astype(str).str.strip()
    r=df[df["hit_str"].isin(["Yes","No"])].copy()
    r["hit"]=r["hit_str"]=="Yes"
    for c in CORE:
        if c in r.columns: r[c]=r[c].apply(sf)

    # ---- join ----
    m=r.merge(bt[["jkey"]+[x for x in METRICS if x in bt.columns]],on="jkey",how="left")
    matched=m[m[METRICS[0]].notna()]
    print(f"Resolved HR rows: {len(r)} | matched to bat-tracking: {len(matched)} "
          f"({len(matched)/len(r)*100:.0f}%)\n")
    if len(matched)<100:
        print("  ⚠️  Low match rate — name formats may differ. Check a few:")
        unmatched=set(r["jkey"])-set(bt["jkey"])
        for x in list(unmatched)[:8]: print("    HR-only:",x)
        for x in list(set(bt['jkey'])-set(r['jkey']))[:5]: print("    BT-only:",x)

    # ---- separator test ----
    print("="*70)
    print("  BAT-TRACKING SEPARATORS (HR hitters vs non-HR)")
    print("="*70)
    print(f"  {'metric':<28} {'HR avg':>9} {'nonHR':>9} {'eff size':>9} {'signal'}")
    results={}
    for mtr in METRICS:
        if mtr not in matched.columns: continue
        h=matched[matched["hit"]][mtr]; n=matched[~matched["hit"]][mtr]
        if h.notna().sum()<30 or n.notna().sum()<30: continue
        hm,nm=h.mean(),n.mean(); sd=matched[mtr].std()
        eff=(hm-nm)/sd if sd else 0
        results[mtr]=eff
        sig="🔥 STRONG" if abs(eff)>=0.20 else "✓ moderate" if abs(eff)>=0.10 else "· weak"
        print(f"  {mtr:<28} {hm:>9.3f} {nm:>9.3f} {eff:>+9.3f} {sig}")

    # ---- redundancy vs core (correlation) ----
    print("\n"+"="*70)
    print("  REDUNDANCY — corr of each bat metric with existing core features")
    print("  (high corr = echoes what you already have; low = NEW signal)")
    print("="*70)
    for mtr in [x for x in METRICS if x in matched.columns]:
        cors=[]
        for c in CORE:
            if c in matched.columns:
                cc=matched[[mtr,c]].dropna()
                if len(cc)>30:
                    cors.append(f"{c}={cc[mtr].corr(cc[c]):+.2f}")
        print(f"  {mtr:<28} " + "  ".join(cors))

    print("\n  READ: a metric is worth ADDING only if it BOTH separates HR hitters")
    print("  (decent effect size) AND has low correlation with your core features")
    print("  (adds NEW signal, not a barrel% echo). Strong+redundant = skip;")
    print("  moderate+independent = candidate; weak = skip.")
    print("\nDone.")

if __name__=="__main__":
    main()
