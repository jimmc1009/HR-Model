"""
test_scoring_systems.py
Head-to-head: does the OLD normalization scoring or the CURRENT absolute
scoring better predict HRs? Tests the hypothesis that the mid-May switch from
normalization -> absolute broke the model (profitable until ~May 17, downtrend
after).

Reconstructs NORMALIZATION scoring from raw components in HR_All_Scores using
the recovered old WEIGHTS (ranked/z-scored within each day, as the old system
did), and compares it against the stored absolute hr_score. For each system:
hit rate by score-decile and top-N-per-day hit rate — which system's top picks
actually homer more?

Read-only. Uses hit_hr outcomes already in the sheet. Does NOT need the old
hr_picks.py file — rebuilds normalization from raw stats.
"""
import os, json, time
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]

# Recovered OLD normalization weights (from May session, pre-switch)
WEIGHTS = {
    "barrel_pct_7d":        2.0,
    "hr_per_pa":            1.8,
    "hr_per_fb":            1.5,
    "iso":                  1.2,
    "avg_ev_7d":            1.0,
    "hard_hit_pct_7d":      0.8,
    "pitcher_barrel_pct":   1.5,
    "pitcher_hr_per_fb":    1.5,
    "pitcher_hard_hit_pct": 0.8,
    "park_hr_factor":       0.5,
}

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

def main():
    sid=os.environ["GOOGLE_SHEET_ID"];gc=gcc()
    print("Reading HR_All_Scores...")
    sh=wr(lambda:gc.open_by_key(sid));ws=sh.worksheet("HR_All_Scores")
    vals=wr(lambda:ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)

    # absolute score = current stored (corrected)
    df["abs_score"]=df.apply(lambda r: sf(r.get("hr_score_corrected")) if str(r.get("hr_score_corrected","")).strip() not in ("","nan","None") else sf(r.get("hr_score")),axis=1)
    df["hit_str"]=df["hit_hr"].astype(str).str.strip()
    for c in list(WEIGHTS.keys()):
        if c in df.columns: df[c]=df[c].apply(sf)
        else: df[c]=np.nan

    r=df[df["hit_str"].isin(["Yes","No"])].copy()
    r["hit"]=r["hit_str"]=="Yes"
    r["date"]=r["date"].astype(str)

    # ── reconstruct NORMALIZATION score: within each day, z-score each
    # component, weight, sum. That's what the old system did (relative to pool).
    def zscore(s):
        m=s.mean(); sd=s.std()
        return (s-m)/sd if sd and not pd.isna(sd) else s*0
    norm_scores=[]
    for d,day in r.groupby("date"):
        day=day.copy()
        total=pd.Series(0.0,index=day.index)
        for comp,wt in WEIGHTS.items():
            if comp in day.columns and day[comp].notna().any():
                total=total+zscore(day[comp].fillna(day[comp].mean()))*wt
        day["norm_score"]=total
        norm_scores.append(day[["norm_score"]])
    r["norm_score"]=pd.concat(norm_scores)["norm_score"]

    print(f"  {len(r)} resolved rows\n")

    # ── compare: hit rate by within-day RANK for each system ─────────────
    # (both systems are about ranking picks; compare top-N-per-day hit rates)
    print("="*60)
    print("  TOP-N-PER-DAY HIT RATE: absolute vs normalization")
    print("="*60)
    print(f"  {'top N/day':<10} {'ABSOLUTE':>12} {'NORMALIZATION':>15}")
    for topn in [3,5,10,15]:
        abs_hits=abs_tot=norm_hits=norm_tot=0
        for d,day in r.groupby("date"):
            if len(day)<topn: continue
            a=day.nlargest(topn,"abs_score"); n=day.nlargest(topn,"norm_score")
            abs_hits+=a["hit"].sum(); abs_tot+=len(a)
            norm_hits+=n["hit"].sum(); norm_tot+=len(n)
        ah=abs_hits/abs_tot*100 if abs_tot else 0
        nh=norm_hits/norm_tot*100 if norm_tot else 0
        print(f"  {'top '+str(topn):<10} {ah:>11.1f}% {nh:>14.1f}%")

    # ── correlation of each score with actual HR outcome ─────────────────
    print("\n"+"="*60)
    print("  SEPARATION: mean score, HR hitters vs non-HR")
    print("="*60)
    for label,col in [("Absolute","abs_score"),("Normalization","norm_score")]:
        h=r[r["hit"]][col].mean(); m=r[~r["hit"]][col].mean()
        sd=r[col].std()
        eff=(h-m)/sd if sd else 0
        print(f"  {label:<14} HR avg={h:>7.3f}  nonHR avg={m:>7.3f}  effect size={eff:>+.3f}")

    print("\n  Effect size = how many std devs separate HR hitters from non-HR.")
    print("  HIGHER = the system's score better distinguishes who homers.")
    print("  If normalization's effect size and top-N hit rates are clearly")
    print("  higher, the mid-May switch to absolute likely hurt — worth")
    print("  considering a revert or hybrid. If similar/worse, the switch")
    print("  isn't the culprit and the downtrend is something else.")
    print("\n  CAVEAT: normalization here is reconstructed (z-score within day)")
    print("  from raw components — a faithful approximation of the old system,")
    print("  not the exact original code. Directional, not precise.")
    print("\nDone.")

if __name__=="__main__":
    main()
