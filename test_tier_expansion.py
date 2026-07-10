"""
test_tier_expansion.py
Read-only. Tests whether splitting the 13+ tier into finer tiers
(13-14, 14-15, 15-16, 16+) leaves enough resolved picks per tier×odds-zone
to clear the 15-pick minimum the edge filter requires.

If most new buckets fall under 15, the expansion mostly produces "no data"
and isn't worth it yet. If several clear 15 (or 25), expansion adds real
resolution. Uses corrected scores.
"""
import os, json, time
import pandas as pd, numpy as np
import gspread
from google.oauth2.service_account import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
MODEL_START_DATE = "2026-06-09"
MIN_SAMPLE = 15

def gc_client():
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info, scopes=SCOPES))

def with_retry(fn, r=4, w=20):
    for a in range(r):
        try: return fn()
        except Exception:
            if a<r-1: time.sleep(w)
            else: raise

def sf(v,d=0.0):
    try:
        f=float(v); return d if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError,TypeError): return d

def coalesce(row,c,o):
    x=str(row.get(c,"")).strip()
    return sf(x) if x not in ("","nan","None") else sf(row.get(o))

def be(o):
    if o>=100: return 100.0/(o+100.0)
    if o<0: return abs(o)/(abs(o)+100.0)
    return 1.0

def zone(o):
    if o<=300: return "≤+300"
    if o<=499: return "+301-499"
    if o<=699: return "+500-699"
    return "+700+"

def main():
    sid=os.environ["GOOGLE_SHEET_ID"]; gc=gc_client()
    print("Reading HR_All_Scores...")
    sh=with_retry(lambda: gc.open_by_key(sid))
    ws=sh.worksheet("HR_All_Scores")
    vals=with_retry(lambda: ws.get_all_values())
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r: any(str(v).strip() for v in r),axis=1)]
    df["dt"]=pd.to_datetime(df["date"],errors="coerce")
    df=df[df["dt"]>=pd.Timestamp(MODEL_START_DATE)].copy()
    has=("hr_score_corrected" in df.columns)
    df["score"]=df.apply(lambda r: coalesce(r,"hr_score_corrected","hr_score"),axis=1) if has else df["hr_score"].apply(sf)
    df["odds"]=df["consensus_odds"].apply(lambda x: sf(x,0))
    r=df[df["hit_hr"].astype(str).str.strip().isin(["Yes","No"])].copy()
    r["hit"]=r["hit_hr"].astype(str).str.strip()=="Yes"

    print(f"  {len(r)} resolved rows (corrected={has})\n")

    # Current: everything 13+ in one bucket
    print("="*60); print("CURRENT: 13+ as ONE tier"); print("="*60)
    for z in ["≤+300","+301-499","+500-699","+700+"]:
        sub=r[(r["score"]>=13)&(r["odds"].apply(zone)==z)]
        if len(sub)==0: continue
        hr=sub["hit"].mean()*100; b=be(sub["odds"].mean())*100
        mk="✅USABLE" if len(sub)>=MIN_SAMPLE else "⚠️thin"
        print(f"  13+ | {z:<9} n={len(sub):>3}  {hr:>5.1f}%  be={b:>4.1f}%  {mk}")

    # Proposed: split into 13-14,14-15,15-16,16+
    print("\n"+"="*60); print("PROPOSED: split 13+ into finer tiers"); print("="*60)
    tiers=[("13-14",13,14),("14-15",14,15),("15-16",15,16),("16+",16,999)]
    usable=0; thin=0
    for lbl,lo,hi in tiers:
        ts=r[(r["score"]>=lo)&(r["score"]<hi)]
        if len(ts)==0: continue
        print(f"\n  {lbl}  (total resolved: {len(ts)})")
        for z in ["≤+300","+301-499","+500-699","+700+"]:
            zs=ts[ts["odds"].apply(zone)==z]
            if len(zs)==0: continue
            hr=zs["hit"].mean()*100; b=be(zs["odds"].mean())*100
            if len(zs)>=MIN_SAMPLE:
                mk="✅USABLE"; usable+=1
            else:
                mk="⚠️thin (no edge shown)"; thin+=1
            print(f"    {lbl} | {z:<9} n={len(zs):>3}  {hr:>5.1f}%  be={b:>4.1f}%  {mk}")

    print("\n"+"="*60); print("VERDICT"); print("="*60)
    print(f"  Usable buckets after split (n>={MIN_SAMPLE}): {usable}")
    print(f"  Thin buckets (would show no data):          {thin}")
    if usable>=3:
        print("  → Expansion adds real resolution. Worth doing.")
    elif usable>=1:
        print("  → Marginal. A couple usable buckets; rest need more data.")
    else:
        print("  → NOT worth it yet — split mostly yields thin buckets.")
    print("\n  Note: thin buckets aren't harmful (they show 'no data', not fake")
    print("  edge). Downside of expanding early = fewer high-score plays surface")
    print("  until the finer buckets accumulate 15+ picks.")
    print("\nDone.")

if __name__=="__main__":
    main()
