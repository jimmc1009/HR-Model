#!/usr/bin/env python3
"""
feature_separation.py — what actually predicts a HR leg, and whether tweaking
the parlay selector helps.

PART 1  FEATURE SEPARATION (full resolved base — well powered, trustworthy)
  For every leg that resolved Yes/No, compare each feature's average on
  home-run legs vs non-HR legs, with an effect size (Cohen's d) and
  point-biserial correlation. Ranks which signals separate winners.

PART 2  OUT-OF-SAMPLE SELECTOR BAKE-OFF (directional — smaller sample)
  Split by date: TRAIN = earliest 70% of days, TEST = latest 30% (held out).
  The value-zone edge table is built from TRAIN only (no leakage). Several
  candidate selectors then rank each TEST day's pool; we measure the hit rate
  of the legs they'd have picked on days they never saw. This is how you tell
  a real improvement from curve-fitting — the current history-check does NOT
  do this, which is why its numbers are flattered.

Reads HR_All_Scores. Writes nothing. Aggregate output only — no per-day claims.
"""
import os, math, gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
TRAIN_FRAC=0.70
MIN_ZONE_SAMPLE=15

FEATURES=[  # (column, label) — must match HR_All_Scores headers exactly
    ("iso",                "ISO"),
    ("season_barrel_pct",  "Barrel% (season)"),
    ("barrel_pct_5d",      "Barrel% (5d)"),
    ("barrel_pct_7d",      "Barrel% (7d)"),
    ("barrel_pct_10d",     "Barrel% (10d)"),
    ("hr_per_fb",          "HR/FB%"),
    ("hr_per_pa",          "HR/PA%"),
    ("hard_hit_pct_7d",    "HardHit% (7d)"),
    ("hard_hit_pct_season","HardHit% (season)"),
    ("avg_ev_7d",          "Avg EV (7d)"),
    ("avg_ev_5d",          "Avg EV (5d)"),
    ("avg_ev_10d",         "Avg EV (10d)"),
    ("avg_la_7d",          "Avg LA (7d)"),
    ("pull_rate",          "Pull Rate"),
    ("park_hr_factor",     "Park HR Factor"),
    ("pitch_matchup_score","Pitch Matchup"),
    ("hr_weather_boost",   "Weather Boost"),
    ("temp_f",             "Temperature"),
    ("platoon_score",      "Platoon Score"),
    ("top_pitch_iso_vs_hand","Top Pitch ISO vs Hand"),
    ("momentum_score",     "Momentum"),
    # — pitcher (opponent) "gives up HR" signals —
    ("pitcher_hr_per_fb",     "Pitcher HR/FB-allowed"),
    ("pitcher_barrel_pct",    "Pitcher Barrel%-allowed"),
    ("pitcher_hr9",           "Pitcher HR/9"),
    ("pitcher_hr_7d",         "Pitcher HR (7d)"),
    ("pitcher_fb_rate_allowed","Pitcher FB%-allowed"),
    ("pitcher_barrel_7d",     "Pitcher Barrel% (7d)"),
]

def gcc():
    import json
    info=json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    return gspread.authorize(Credentials.from_service_account_info(info,scopes=SCOPES))

def sf(v,d=0.0):
    try:return float(str(v).replace("+","").strip())
    except Exception:return d
def dec(o): return 1+o/100 if o>=100 else 1+100/abs(o)
def be_pct(o): return 100.0*100.0/(o+100.0) if o>0 else 100.0

def zone_key(s,o):
    if o<=0:return None
    if s>=13.0 and o<=300:return "13+|<=300"
    if s>=13.0 and 301<=o<=400:return "13+|301-400"
    if 12.0<=s<13.0 and 301<=o<=400:return "12-13|301-400"
    if 9.0<=s<10.0 and 301<=o<=400:return "9-10|301-400"
    return None

def load(gc,sid):
    sh=gc.open_by_key(sid); ws=sh.worksheet("HR_All_Scores")
    vals=ws.get_all_values()
    df=pd.DataFrame(vals[1:],columns=vals[0])
    df=df[df.apply(lambda r:any(str(v).strip() for v in r),axis=1)].reset_index(drop=True)
    def coalesce(r,c,o):
        x=str(r.get(c,"")).strip()
        return sf(x) if x not in ("","nan","None") else sf(r.get(o))
    df["score"]=df.apply(lambda r:coalesce(r,"hr_score_corrected","hr_score"),axis=1)
    df["odds"] =df["consensus_odds"].apply(lambda x:sf(x,0))
    df["hit_str"]=df["hit_hr"].astype(str).str.strip()
    for col,_ in FEATURES:
        df[col]=df.get(col, pd.Series([0]*len(df))).apply(lambda x:sf(x,0))
    res=df[df["hit_str"].isin(["Yes","No"])].copy()
    res["win"]=(res["hit_str"]=="Yes").astype(int)
    return res

def cohens_d(a,b):
    na,nb=len(a),len(b)
    if na<2 or nb<2:return 0.0
    va,vb=a.var(ddof=1),b.var(ddof=1)
    sp=math.sqrt(((na-1)*va+(nb-1)*vb)/(na+nb-2)) if (na+nb-2)>0 else 0.0
    return (a.mean()-b.mean())/sp if sp>0 else 0.0

def part1(res):
    print("="*68)
    print(f"PART 1 — FEATURE SEPARATION (full base, n={len(res)} resolved legs)")
    print("="*68)
    w=res[res["win"]==1]; l=res[res["win"]==0]
    print(f"  HR legs: {len(w)}   non-HR legs: {len(l)}   base rate: {len(w)/len(res)*100:.1f}%\n")
    print(f"  {'Feature':20s} {'HR avg':>8s} {'noHR avg':>9s} {'diff':>8s} {'Cohen d':>8s} {'r_pb':>7s}")
    print("  "+"-"*62)
    rows=[]
    for col,label in FEATURES:
        a,b=w[col],l[col]
        d=cohens_d(a,b)
        r=res["win"].corr(res[col])
        rows.append((abs(d),label,a.mean(),b.mean(),a.mean()-b.mean(),d,r if pd.notna(r) else 0.0))
    for _,label,wa,la,diff,d,r in sorted(rows,reverse=True):
        print(f"  {label:20s} {wa:8.3f} {la:9.3f} {diff:+8.3f} {d:+8.3f} {r:+7.3f}")
    print("\n  Cohen d: >|0.2| small, >|0.5| medium. r_pb: point-biserial corr with HR.")
    print("  Positive = higher value on HR legs (helps). Near-zero/negative = dead weight.\n")

def _edge(s,o,zr):
    k=zone_key(s,o)
    if k is None or k not in zr:return -99.0
    return zr[k]-be_pct(o)

def zone_rates(train):
    defs={
        "13+|<=300":     (train["score"]>=13)&(train["odds"]>0)&(train["odds"]<=300),
        "13+|301-400":   (train["score"]>=13)&(train["odds"]>=301)&(train["odds"]<=400),
        "12-13|301-400": (train["score"]>=12)&(train["score"]<13)&(train["odds"]>=301)&(train["odds"]<=400),
        "9-10|301-400":  (train["score"]>=9)&(train["score"]<10)&(train["odds"]>=301)&(train["odds"]<=400),
    }
    zr={}
    for k,m in defs.items():
        sub=train[m]
        if len(sub)>=MIN_ZONE_SAMPLE: zr[k]=sub["win"].mean()*100
    return zr

def part2(res):
    print("="*68)
    print("PART 2 — OUT-OF-SAMPLE SELECTOR BAKE-OFF (directional, small test)")
    print("="*68)
    dates=sorted(res["date"].unique())
    if len(dates)<6:
        print("  Not enough distinct days for a train/test split."); return
    cut=int(len(dates)*TRAIN_FRAC)
    train_d,test_d=set(dates[:cut]),set(dates[cut:])
    train=res[res["date"].isin(train_d)].copy()
    test =res[res["date"].isin(test_d)].copy()
    zr=zone_rates(train)
    print(f"  TRAIN days={len(train_d)} ({len(train)} legs)  TEST days={len(test_d)} ({len(test)} legs)")
    print(f"  Zone table built from TRAIN only: "+", ".join(f"{k}={v:.0f}%" for k,v in zr.items())+"\n")

    # pool legs only, per the deployed +301-400 pool
    test=test[test.apply(lambda r:zone_key(r["score"],r["odds"]) is not None,axis=1)].copy()
    pool_rate=test["win"].mean()*100 if len(test) else 0.0

    # train-only stats for scale-free (z-scored) composites — no leakage
    zfeats=["hr_per_fb","iso","season_barrel_pct","barrel_pct_5d"]
    st={f:(train[f].mean(), (train[f].std(ddof=0) or 1.0)) for f in zfeats}
    def z(r,f):
        m,s=st[f]; return (r[f]-m)/s if s else 0.0
    def zpower(r): return sum(z(r,f) for f in zfeats)

    # Pre-committed power-anchor swaps: keep edge*0.8 identical, replace the
    # hr_per_fb/8 power term with season Barrel% or HR/PA%, each scaled so its
    # spread matches hr_per_fb/8's on TRAIN (a fair swap, not a rescale).
    base_pow_std=(train["hr_per_fb"]/8).std(ddof=0) or 1.0
    def kscale(col):
        s=train[col].std(ddof=0) or 1.0
        return base_pow_std/s
    kb=kscale("season_barrel_pct"); kp=kscale("hr_per_pa")

    # pitcher HR-vulnerability composite (z on TRAIN) — positive = HR-prone pitcher
    pv_feats=[c for c in ["pitcher_hr_per_fb","pitcher_barrel_pct","pitcher_hr9"]
              if c in train.columns and train[c].std(ddof=0)>0]
    pst={f:(train[f].mean(),(train[f].std(ddof=0) or 1.0)) for f in pv_feats}
    def pvuln(r):
        if not pv_feats: return 0.0
        return sum((r[f]-pst[f][0])/pst[f][1] for f in pv_feats)/len(pv_feats)

    selectors={
        "power only (hr_per_fb)": lambda r:r["hr_per_fb"],
        "iso only":               lambda r:r["iso"],
        "season barrel only":     lambda r:r["season_barrel_pct"],
        "blend1  (CURRENT)":      lambda r:r["hr_per_fb"]/8 + _edge(r["score"],r["odds"],zr)*0.8,
        "blend: barrel + edge":   lambda r:r["season_barrel_pct"]*kb + _edge(r["score"],r["odds"],zr)*0.8,
        "blend: hr_per_pa + edge":lambda r:r["hr_per_pa"]*kp + _edge(r["score"],r["odds"],zr)*0.8,
        "blend1 + pitchHR(z)":    lambda r:r["hr_per_fb"]/8 + _edge(r["score"],r["odds"],zr)*0.8 + pvuln(r)*base_pow_std,
        "power composite (z)":    zpower,
        "blend1 + platoon*0.8":   lambda r:r["hr_per_fb"]/8 + _edge(r["score"],r["odds"],zr)*0.8 + r["platoon_score"]*0.8,
    }
    # a pool-filter variant: drop negative-platoon legs, then rank with blend1
    gate_name="blend1 + drop platoon<0"

    print(f"  Baseline: ALL pool legs on TEST days hit {pool_rate:.1f}% (n={len(test)})")
    print(f"  Below = hit rate of the TOP-3 legs each selector would pick per test day.\n")
    print(f"  {'Selector':28s} {'top3 legs':>10s} {'hit rate':>9s}")
    print("  "+"-"*50)

    def eval_sel(fn, gate=False):
        hits=tot=0
        for d in test_d:
            day=test[test["date"]==d].copy()
            if gate: day=day[day["platoon_score"]>=0]
            if day.empty: continue
            day["sel"]=day.apply(fn,axis=1)
            top=day.sort_values("sel",ascending=False).head(3)
            hits+=int(top["win"].sum()); tot+=len(top)
        return hits,tot

    for name,fn in selectors.items():
        h,t=eval_sel(fn)
        print(f"  {name:28s} {t:>10d} {(h/t*100 if t else 0):>8.1f}%")
    # gate variant uses blend1 ranking on the filtered pool
    hg,tg=eval_sel(selectors["blend1  (CURRENT)"],gate=True)
    print(f"  {gate_name:28s} {tg:>10d} {(hg/tg*100 if tg else 0):>8.1f}%")

    print("\n  READ WITH CARE: the TEST set is small, so a 1-2 leg swing moves these")
    print("  a lot. Trust PART 1 for what separates; treat PART 2 as a nudge, not")
    print("  proof. A candidate only earns a change if it beats blend1 here AND")
    print("  its feature shows real separation in PART 1.\n")

PITCH_VULN=[
    ("pitcher_hr_per_fb",     "Pitcher HR/FB-allowed"),
    ("pitcher_barrel_pct",    "Pitcher Barrel%-allowed"),
    ("pitcher_hr9",           "Pitcher HR/9"),
    ("pitcher_hr_7d",         "Pitcher HR (7d)"),
    ("pitcher_fb_rate_allowed","Pitcher FB%-allowed"),
]

def _bin_hit(fr, col, title):
    fr=fr[fr[col].notna()]
    if len(fr)<20 or fr[col].std(ddof=0)==0:
        print(f"  {title}: not enough spread / sample"); return
    try:
        fr=fr.assign(_q=pd.qcut(fr[col],4,labels=["Q1 low","Q2","Q3","Q4 high"],duplicates="drop"))
    except Exception:
        print(f"  {title}: not enough spread"); return
    print(f"  {title}")
    for lab,sub in fr.groupby("_q",observed=True):
        print(f"      {str(lab):8s}  hit {sub['win'].mean()*100:5.1f}%   n={len(sub)}")

def part3(res):
    print("="*68)
    print("PART 3 — PITCHER VULNERABILITY -> LEG HIT RATE")
    print("         (does facing a HR-prone pitcher lift the hit rate? Q1 low -> Q4 high)")
    print("="*68)
    for col,label in PITCH_VULN:
        if col in res.columns and res[col].std(ddof=0)>0:
            _bin_hit(res, col, label)
    # combined pitcher HR-vulnerability composite, full base vs within-pool
    zc=[c for c,_ in PITCH_VULN if c in res.columns and res[c].std(ddof=0)>0]
    if zc:
        res=res.copy()
        res["_pvuln"]=sum((res[c]-res[c].mean())/(res[c].std(ddof=0) or 1) for c in zc)
        print("\n  Combined pitcher HR-vulnerability (z-sum of the above):")
        _bin_hit(res, "_pvuln", "  full base")
        pool=res[res.apply(lambda r:zone_key(r["score"],r["odds"]) is not None,axis=1)]
        _bin_hit(pool, "_pvuln", "  within +301-400 pool only")
    print("\n  Rising Q1->Q4 = your hunch holds. If it's flat WITHIN the pool, the")
    print("  eligibility gate already captured it and the blend can't add much.\n")

def main():
    gc=gcc(); sid=os.environ["GOOGLE_SHEET_ID"]
    res=load(gc,sid)
    if res.empty:
        print("No resolved rows in HR_All_Scores."); return
    part1(res)
    part3(res)
    part2(res)
    print("Done.")

if __name__=="__main__":
    main()
