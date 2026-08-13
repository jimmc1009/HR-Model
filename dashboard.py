"""
dashboard.py — HR-only build of "Today's Top Picks".

This is a clean rewrite:
  • Timestamp is now the FIRST row, written and formatted in a single pass
    (previously it was inserted after formatting, which shifted every row
    down by one and corrupted the first data row + header).
  • Every run starts with a full unmerge + format reset so leftover cells
    from the old KS/HRRBI layout can't bleed into the new one.
  • KS and HR+RBI sections are gone (not being bet right now).
  • Score tier lookup splits 13+ into 13-14 / 14-15 / 15+ so each leg is
    credited with its own observed hit rate (15+ underperforms).
  • "Why" shows baseball reasons only — power, barrel, form, platoon,
    park, weather. It no longer repeats the Edge column.
  • Both parlays show individual leg odds AND the combined parlay price.
"""
import os
import json
import time
from datetime import datetime, date as _date
import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import pytz

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
DASHBOARD_SHEET = "Today's Top Picks"

COLOR_BG        = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT    = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD      = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_PURPLE    = {"red": 0.541, "green": 0.165, "blue": 0.557}
COLOR_HEADER_BG = {"red": 0.055, "green": 0.055, "blue": 0.055}
COLOR_SUBTEXT   = {"red": 0.600, "green": 0.600, "blue": 0.600}
COLOR_BLACK     = {"red": 0.050, "green": 0.050, "blue": 0.050}

N_COLS = 15
RESET_ROWS = 400
RESET_COLS = 26


# ── Sheets plumbing ─────────────────────────────────────────────────────────
def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def read_sheet(gc, sheet_id, name) -> pd.DataFrame:
    try:
        sh = with_retry(lambda: gc.open_by_key(sheet_id))
        ws = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values:
            return pd.DataFrame()
        start = 1 if "Last Run" in str(all_values[0]) else 0
        if start >= len(all_values):
            return pd.DataFrame()
        headers = all_values[start]
        rows = all_values[start + 1:]
        df = pd.DataFrame(rows, columns=headers)
        return df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read sheet '{name}': {e}")
        return pd.DataFrame()


def read_sheet_raw(gc, sheet_id, name) -> pd.DataFrame:
    try:
        sh = with_retry(lambda: gc.open_by_key(sheet_id))
        ws = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values or len(all_values) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        return df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    except Exception:
        return pd.DataFrame()


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


# The 8 tier LABELS stay fixed (so all dict keys keep matching), but their
# numeric cutoffs are remapped to PERCENTILES of the score distribution. The
# rescore compressed scores ~2-3 pts, so the old fixed 15/14/13... boundaries
# put everyone in the bottom tiers and capped the blend at ~17%. Percentile
# cutoffs fix that: "15+" now means "top-tier score on WHATEVER scale is live."
_TIER_LABELS = ["15+", "14-15", "13-14", "12-13", "11-12", "10-11", "9-10", "8.5-9"]
_TIER_PCTS   = [98, 90, 80, 65, 50, 35, 20]   # 7 cutpoints -> 8 tiers
_TIER_CUTS_CACHE = {"cuts": None}

def compute_tier_cuts(scores):
    """Return the 7 percentile cutpoints (high->low) that define the 8 tiers."""
    s = pd.Series([x for x in scores if x is not None and not pd.isna(x) and x > -50])
    if len(s) < 50:
        # fallback to old fixed scale if too little data
        return [15, 14, 13, 12, 11, 10, 9]
    return [float(s.quantile(p / 100)) for p in _TIER_PCTS]

def tier_defs_from_cuts(cuts):
    """Build (label, lo, hi) tuples from the 7 cutpoints, matching _TIER_LABELS."""
    bounds = [999] + list(cuts) + [-999]   # 9 boundaries -> 8 tiers
    return [(_TIER_LABELS[i], bounds[i + 1], bounds[i]) for i in range(8)]

def tier_key_from_cuts(score, cuts):
    """Return the tier label for a score given the 7 cutpoints."""
    for label, lo, hi in tier_defs_from_cuts(cuts):
        if lo <= score < hi:
            return label
    return None


def resolve_score(row, default=0.0) -> float:
    """Single source of truth for a row's HR score. Prefers hr_score_recalc
    (the faithful new-scale rescore of all history), then hr_score_corrected,
    then raw hr_score. Everything in the dashboard scores off this so the whole
    board runs on the new monotonic scale."""
    for col in ("hr_score_recalc", "hr_score_corrected", "hr_score"):
        v = str(row.get(col, "")).strip()
        if v not in ("", "nan", "None"):
            return safe_float(v, default)
    return default


def american_to_implied(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> str:
    if prob <= 0 or prob >= 1:
        return "—"
    if prob >= 0.5:
        return str(int(-round((prob / (1 - prob)) * 100)))
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def combined_american(odds_list) -> str:
    """Multiply decimal odds of the legs, return the parlay price in American."""
    dec = 1.0
    for o in odds_list:
        dec *= (1 + o / 100.0)  # legs are positive American here
    return f"+{int(round((dec - 1) * 100))}"


# ── HR hit-rate lookup ──────────────────────────────────────────────────────
def _hr_odds_zone_key(odds: float) -> str:
    if odds <= 300:
        return "le300"
    if 301 <= odds <= 499:
        return "301-499"
    if 500 <= odds <= 699:
        return "500-699"
    return "700plus"


def build_hr_hit_rates(hr_all_scores: pd.DataFrame) -> dict:
    hit_rates = {}
    if hr_all_scores.empty:
        return hit_rates
    MODEL_START_DATE = "2026-06-09"
    df = hr_all_scores.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)]
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if resolved.empty:
        return hit_rates

    resolved["hr_score"] = resolved.apply(resolve_score, axis=1)
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["odds_num"] = resolved["consensus_odds"].apply(safe_float)

    # Global base HR rate — the anchor we shrink small/extreme cells toward.
    base_rate = resolved["hit_bool"].mean()
    # Shrinkage strength: a cell needs ~SHRINK_K samples to earn half its
    # distance from base. Calibration (out-of-sample) showed the high-confidence
    # cells (32%+ predicted) hit ~18% — overrated because a thin, hot cell was
    # trusted at face value. Shrinking pulls those back toward reality without
    # hand-picking which tiers to punish; big stable cells barely move.
    SHRINK_K = 40.0

    def shrunk_rate(sub):
        n = len(sub)
        raw = sub["hit_bool"].mean()
        return (n * raw + SHRINK_K * base_rate) / (n + SHRINK_K)

    # 13+ split into 13-14 / 14-15 / 15+ so each leg is credited with its own
    # observed rate (15+ underperforms the tiers beneath it).
    _cuts = compute_tier_cuts(resolved["hr_score"].tolist())
    _TIER_CUTS_CACHE["cuts"] = _cuts
    tier_defs = tier_defs_from_cuts(_cuts)
    zone_keys = ["le300", "301-499", "500-699", "700plus"]
    for tier_label, lo, hi in tier_defs:
        tier_sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(tier_sub) >= 5:
            hit_rates[tier_label] = shrunk_rate(tier_sub)
        for zk in zone_keys:
            zsub = tier_sub[tier_sub["odds_num"].apply(_hr_odds_zone_key) == zk]
            if len(zsub) >= 15:
                hit_rates[(tier_label, zk)] = shrunk_rate(zsub)

    # ── regressed rates for platoon / pitch / combined tiers (for the blended
    # card ranker). Same shrinkage — small/extreme buckets pulled toward base.
    resolved["ps_val"] = pd.to_numeric(resolved.get("platoon_score", 0), errors="coerce").fillna(0)
    resolved["pm_val"] = pd.to_numeric(resolved.get("pitch_matchup_score", 0), errors="coerce").fillna(0)
    resolved["cb_val"] = resolved["ps_val"] + resolved["pm_val"]
    plat_defs = [("plat_elite", 4, 99), ("plat_strong", 2, 4), ("plat_mild", 0.5, 2),
                 ("plat_neu", -0.5, 0.5), ("plat_mdis", -2, -0.5),
                 ("plat_sdis", -4, -2), ("plat_edis", -99, -4)]
    pitch_defs = [("pit_good", 0.8, 99), ("pit_mild", 0.2, 0.8),
                  ("pit_neu", -0.2, 0.2), ("pit_weak", -99, -0.2)]
    combo_defs = [("cb_elitep", 5, 99), ("cb_elite", 4, 5), ("cb_great", 3, 4),
                  ("cb_good", 1, 3), ("cb_neu", -1, 1), ("cb_bad", -3, -1),
                  ("cb_terrible", -99, -3)]
    for key, lo, hi in plat_defs:
        sub = resolved[(resolved["ps_val"] >= lo) & (resolved["ps_val"] < hi)]
        if len(sub) >= 10:
            hit_rates[key] = shrunk_rate(sub)
    for key, lo, hi in pitch_defs:
        sub = resolved[(resolved["pm_val"] >= lo) & (resolved["pm_val"] < hi)]
        if len(sub) >= 10:
            hit_rates[key] = shrunk_rate(sub)
    for key, lo, hi in combo_defs:
        sub = resolved[(resolved["cb_val"] >= lo) & (resolved["cb_val"] < hi)]
        if len(sub) >= 10:
            hit_rates[key] = shrunk_rate(sub)
    hit_rates["_base"] = base_rate
    hit_rates["_tier_cuts"] = _cuts

    print(f"  HR hit rate lookup: {len(hit_rates)} entries from {len(resolved)} resolved picks "
          f"(base {base_rate*100:.1f}%, shrink k={SHRINK_K:.0f})")
    return hit_rates


def get_hr_score_tier(score: float) -> str:
    cuts = _TIER_CUTS_CACHE.get("cuts") or [15, 14, 13, 12, 11, 10, 9]
    k = tier_key_from_cuts(score, cuts)
    return k if k else "below-8.5"


def calc_hr_value(score: float, odds: float, hit_rates: dict) -> tuple:
    tier = get_hr_score_tier(score)
    zk = _hr_odds_zone_key(odds)
    hit_rate = hit_rates.get((tier, zk))
    if hit_rate is None or odds <= 0:
        return 0.0, "—", False, "No zone data"
    implied = american_to_implied(odds)
    edge = hit_rate - implied
    edge_pct = round(edge * 100, 1)
    edge_str = f"+{edge_pct}%" if edge_pct >= 0 else f"{edge_pct}%"
    return round(hit_rate * 100, 1), implied_to_american(hit_rate), edge > 0, edge_str


# ── Why — baseball reasons only ─────────────────────────────────────────────
def build_why(row) -> str:
    """
    Baseball reasons a pick stands out — power, barrel, form, platoon, park,
    weather. Deliberately does NOT include zone edge (that's the Edge column).
    """
    bits = []

    hr_fb = safe_float(row.get("hr_per_fb", 0))
    if hr_fb >= 18:
        bits.append(f"elite power ({hr_fb:.0f}% HR/FB)")
    elif hr_fb >= 13:
        bits.append(f"strong power ({hr_fb:.0f}% HR/FB)")
    elif hr_fb >= 9:
        bits.append(f"avg power ({hr_fb:.0f}% HR/FB)")

    sb = safe_float(row.get("season_barrel_pct", 0))
    if sb >= 12:
        bits.append(f"elite barrel ({sb:.0f}%)")
    elif sb >= 9:
        bits.append(f"good barrel ({sb:.0f}%)")

    b7 = safe_float(row.get("barrel_pct_7d", 0))
    if b7 >= 18:
        bits.append(f"hot barrel 7d ({b7:.0f}%)")

    mom = str(row.get("momentum_desc", "")).lower()
    if "surging" in mom:
        bits.append("surging form")
    elif "trending up" in mom:
        bits.append("heating up")
    elif "cold" in mom or "trending down" in mom:
        bits.append("cold streak")

    plat = safe_float(row.get("platoon_score", 0))
    if plat >= 1.0:
        bits.append("platoon edge")
    elif plat <= -1.0:
        bits.append("platoon down")

    pf = safe_float(row.get("park_hr_factor", 100))
    if pf >= 110:
        bits.append(f"HR park ({pf:.0f})")
    elif 0 < pf <= 90:
        bits.append(f"tough park ({pf:.0f})")

    wb = safe_float(row.get("hr_weather_boost", 0))
    if wb >= 1.5:
        bits.append("wind out")
    elif wb <= -1.0:
        bits.append("wind in")

    if not bits:
        return f"avg power ({hr_fb:.0f}% HR/FB)" if hr_fb > 0 else "value on odds only"
    return "   ·   ".join(bits[:4])


# ── Row builder (HR only) ───────────────────────────────────────────────────
def build_edge_bands(hr_all_scores: pd.DataFrame, min_n: int = 30) -> list:
    """Compute score×odds bands whose RESOLVED hit rate beats implied
    breakeven, live from HR_All_Scores. Returns a ranked list (best edge
    first) of dicts: {band, s_lo, s_hi, o_lo, o_hi, edge_pp, n, hit}.

    Replaces the old hardcoded band table so the slate reflects current
    resolved data every run instead of a frozen snapshot. Same math as
    test_edge_bands.py. Only bands with n>=min_n AND positive edge qualify.
    """
    if hr_all_scores.empty:
        return []
    MODEL_START_DATE = "2026-06-09"
    df = hr_all_scores.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)]
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if df.empty:
        return []
    df["score"] = df.apply(resolve_score, axis=1)
    df["odds"] = df["consensus_odds"].apply(safe_float)
    df["hit"] = (df["hit_hr"].astype(str).str.strip() == "Yes").astype(int)
    df = df[(df["odds"] > 0) & (df["score"] > 0)]

    score_bands = [(15, 999, "15+"), (14, 15, "14-15"), (13, 14, "13-14"),
                   (12, 13, "12-13"), (11, 12, "11-12"), (10, 11, "10-11"),
                   (8.5, 10, "8.5-10")]
    odds_bands = [(0, 250, "≤+250"), (251, 300, "+251-300"), (301, 350, "+301-350"),
                  (351, 400, "+351-400"), (401, 450, "+401-450"), (451, 500, "+451-500"),
                  (501, 600, "+501-600"), (601, 9999, "+601+")]

    def implied(o):
        return 100/(o+100) if o >= 0 else abs(o)/(abs(o)+100)

    bands = []
    for slo, shi, slab in score_bands:
        for olo, ohi, olab in odds_bands:
            sub = df[(df["score"] >= slo) & (df["score"] < shi) &
                     (df["odds"] >= olo) & (df["odds"] <= ohi)]
            n = len(sub)
            if n < min_n:
                continue
            hit = sub["hit"].mean()
            be = sub["odds"].apply(implied).mean()
            edge = hit - be
            if edge > 0:
                bands.append({
                    "band": f"{slab} @ {olab}", "s_lo": slo, "s_hi": shi,
                    "o_lo": olo, "o_hi": ohi, "edge_pp": edge * 100,
                    "n": n, "hit": hit * 100,
                })
    bands.sort(key=lambda b: -b["edge_pp"])
    return bands


def blended_hit_prob(hr_score, odds, plat, pitch, hit_rates):
    """Blend the regressed hit rates from every validated dimension into one
    expected hit probability for a leg. Overlap-aware: the combined tier
    already contains platoon+pitch, so those get low weight to avoid triple-
    counting the same signal; score×odds is the independent backbone.

    All rates are already shrunk toward base by sample in build_hr_hit_rates,
    so thin/extreme buckets (n=20-something) are automatically pulled toward
    reality — this same code just trusts them more as the samples grow.
    """
    base = hr_rates_get(hit_rates, "_base", 0.12)
    combo = plat + pitch

    _cuts = hr_rates_get(hit_rates, "_tier_cuts", None) or _TIER_CUTS_CACHE.get("cuts") \
            or [15, 14, 13, 12, 11, 10, 9]
    def score_tier_key(s):
        return tier_key_from_cuts(s, _cuts)

    def zone_key(o):
        if o <= 0:      return None
        if o <= 300:    return "le300"
        if o < 500:     return "301-499"
        if o < 700:     return "500-699"
        return "700plus"

    def plat_key(p):
        for k, lo, hi in [("plat_elite",4,99),("plat_strong",2,4),("plat_mild",0.5,2),
                          ("plat_neu",-0.5,0.5),("plat_mdis",-2,-0.5),
                          ("plat_sdis",-4,-2),("plat_edis",-99,-4)]:
            if lo <= p < hi: return k
        return None

    def pitch_key(p):
        for k, lo, hi in [("pit_good",0.8,99),("pit_mild",0.2,0.8),
                          ("pit_neu",-0.2,0.2),("pit_weak",-99,-0.2)]:
            if lo <= p < hi: return k
        return None

    def combo_key(c):
        for k, lo, hi in [("cb_elitep",5,99),("cb_elite",4,5),("cb_great",3,4),
                          ("cb_good",1,3),("cb_neu",-1,1),("cb_bad",-3,-1),
                          ("cb_terrible",-99,-3)]:
            if lo <= c < hi: return k
        return None

    # gather available component rates with weights. score×odds crosstab is the
    # strongest independent cell; fall back to score-tier alone if the crosstab
    # cell is thin/absent.
    st = score_tier_key(hr_score); zk = zone_key(odds)
    comps = []  # (rate, weight)
    cross = hr_rates_get(hit_rates, (st, zk), None) if st and zk else None
    if cross is not None:
        comps.append((cross, 3.0))
    elif st:
        r = hr_rates_get(hit_rates, st, None)
        if r is not None:
            comps.append((r, 2.0))
    # combined tier — the validated backbone of the card
    ck = combo_key(combo)
    rc = hr_rates_get(hit_rates, ck, None) if ck else None
    if rc is not None:
        comps.append((rc, 2.5))
    # platoon & pitch — low weight, already inside combined (avoid triple count)
    pk = plat_key(plat); rp = hr_rates_get(hit_rates, pk, None) if pk else None
    if rp is not None:
        comps.append((rp, 0.6))
    qk = pitch_key(pitch); rq = hr_rates_get(hit_rates, qk, None) if qk else None
    if rq is not None:
        comps.append((rq, 0.6))

    if not comps:
        return base
    wsum = sum(w for _, w in comps)
    return sum(r * w for r, w in comps) / wsum


def hr_rates_get(hit_rates, key, default):
    try:
        v = hit_rates.get(key, default)
        return v if v is not None else default
    except Exception:
        return default


def build_all_band_rates(hr_all_scores: pd.DataFrame, min_n: int = 20) -> dict:
    """Like build_edge_bands but returns EVERY band's hit rate (not only the
    +EV ones), keyed by (s_lo, s_hi, o_lo, o_hi). Used to check each player's
    OWN odds against their band's breakeven — a player priced long enough is
    +EV for himself even if his band averages -EV. Same score/odds/date logic
    as the working edge-bands so it can't diverge."""
    if hr_all_scores.empty:
        return {}
    MODEL_START_DATE = "2026-06-09"
    df = hr_all_scores.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)]
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if df.empty:
        return {}
    df["score"] = df.apply(resolve_score, axis=1)
    df["odds"] = df["consensus_odds"].apply(safe_float)
    df["hit"] = (df["hit_hr"].astype(str).str.strip() == "Yes").astype(int)
    df = df[(df["odds"] > 0) & (df["score"] > 0)]

    score_bands = [(15, 999, "15+"), (14, 15, "14-15"), (13, 14, "13-14"),
                   (12, 13, "12-13"), (11, 12, "11-12"), (10, 11, "10-11"),
                   (8.5, 10, "8.5-10")]
    odds_bands = [(0, 250, "≤+250"), (251, 300, "+251-300"), (301, 350, "+301-350"),
                  (351, 400, "+351-400"), (401, 450, "+401-450"), (451, 500, "+451-500"),
                  (501, 600, "+501-600"), (601, 9999, "+601+")]
    out = {}
    for slo, shi, slab in score_bands:
        for olo, ohi, olab in odds_bands:
            sub = df[(df["score"] >= slo) & (df["score"] < shi) &
                     (df["odds"] >= olo) & (df["odds"] <= ohi)]
            n = len(sub)
            if n < min_n:
                continue
            hit = sub["hit"].mean()
            if hit <= 0:
                continue
            # breakeven American odds from the band's hit rate
            be = -(hit / (1 - hit) * 100) if hit >= 0.5 else ((1 - hit) / hit * 100)
            out[(slo, shi, olo, ohi)] = {
                "band": f"{slab} @ {olab}", "hit": hit * 100, "be": be, "n": n,
            }
    return out


def player_band(score, odds, band_rates):
    """Find the band a player falls into and return its rate/breakeven/n."""
    for (slo, shi, olo, ohi), v in band_rates.items():
        if slo <= score < shi and olo <= odds <= ohi:
            return v
    return None


def build_breakeven_lookup(hr_df):
    """Compute the live breakeven odds for each score-tier x odds-zone cell
    from resolved history — same logic as HR_Analysis, so it auto-updates
    daily as hit rates change. Returns {(tier_label, zone_label): breakeven_am}
    plus the raw hit rate per cell. A leg is +EV if its odds beat its cell's
    breakeven."""
    out = {}
    if hr_df is None or hr_df.empty:
        return out
    d = hr_df.copy()
    if "hit_hr" not in d.columns:
        return out
    d["res"] = d["hit_hr"].astype(str).str.strip()
    d = d[d["res"].isin(["Yes", "No"])]
    if d.empty:
        return out
    d["hit"] = (d["res"] == "Yes").astype(int)
    # use the unified resolver (prefers hr_score_recalc new scale)
    d["sc"] = d.apply(resolve_score, axis=1)
    d["od"] = pd.to_numeric(d["consensus_odds"], errors="coerce") if "consensus_odds" in d.columns else np.nan
    d = d.dropna(subset=["sc", "od"])
    print(f"  [be_debug] resolved={len(d)}, "
          f"od range={d['od'].min():.0f}..{d['od'].max():.0f} "
          f"sc range={d['sc'].min():.1f}..{d['sc'].max():.1f}, "
          f"od<=499={int((d['od']<=499).sum())}, od>0={int((d['od']>0).sum())}")

    _bcuts = _TIER_CUTS_CACHE.get("cuts") or compute_tier_cuts(d["sc"].tolist())
    tier_defs = tier_defs_from_cuts(_bcuts)
    zone_defs = [("le300",0,301),("301-499",301,500),("500-699",500,700),("700plus",700,99999)]

    for zl, zlo, zhi in zone_defs:
        zsub = d[(d["od"] >= zlo) & (d["od"] < zhi)]
        for tl, tlo, thi in tier_defs:
            cell = zsub[(zsub["sc"] >= tlo) & (zsub["sc"] < thi)]
            n = len(cell)
            if n < 8:            # need a minimum sample to trust the breakeven
                continue
            p = cell["hit"].mean()
            if p <= 0:
                continue
            # breakeven American odds from hit rate p
            if p >= 0.5:
                be = -(p / (1 - p) * 100)
            else:
                be = (1 - p) / p * 100
            out[(tl, zl)] = {"be": be, "rate": p * 100, "n": n}
    return out


def _tier_key(s):
    cuts = _TIER_CUTS_CACHE.get("cuts") or [15, 14, 13, 12, 11, 10, 9]
    return tier_key_from_cuts(s, cuts)


def _zone_key(o):
    if o <= 0:    return None
    if o < 301:   return "le300"
    if o < 500:   return "301-499"
    if o < 700:   return "500-699"
    return "700plus"


def american_to_implied(o):
    """Convert American odds to implied win probability (for edge calc)."""
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def reconstruct_heater(hr_df, days_back=7):
    """Rebuild what the Dinger Card would have been each past day from
    HR_All_Scores history and tally how it did. A 'card day' = the top-6
    Great+ legs (combined platoon+pitch >= 3, hr_score >= 10, odds <= 499)
    for that date. Returns a summary dict for a fun streak/heater display."""
    if hr_df is None or hr_df.empty or "date" not in hr_df.columns:
        return None
    d = hr_df.copy()
    d["res"] = d["hit_hr"].astype(str).str.strip()
    d = d[d["res"].isin(["Yes", "No"])]
    if d.empty:
        return None
    d["hit"] = (d["res"] == "Yes").astype(int)
    d["ps"] = pd.to_numeric(d.get("platoon_score", 0), errors="coerce").fillna(0)
    d["pm"] = pd.to_numeric(d.get("pitch_matchup_score", 0), errors="coerce").fillna(0)
    d["sc"] = pd.to_numeric(d.get("hr_score_recalc", d.get("hr_score", 0)), errors="coerce").fillna(
              pd.to_numeric(d.get("hr_score", 0), errors="coerce")).fillna(0)
    d["od"] = pd.to_numeric(d.get("consensus_odds", 0), errors="coerce").fillna(0)
    d["combo"] = d["ps"] + d["pm"]
    d["date_s"] = d["date"].astype(str).str.strip()

    dates = sorted(d["date_s"].unique())[-days_back:]
    night_results = []   # (date, legs_hit, legs_total, best_pair_hit)
    for dt in dates:
        day = d[d["date_s"] == dt]
        card = day[(day["combo"] >= 3) & (day["sc"] >= 10) &
                   (day["od"] > 0) & (day["od"] <= 499)]
        card = card.sort_values("combo", ascending=False).head(6)
        if card.empty:
            continue
        hits = int(card["hit"].sum())
        total = len(card)
        # a "pair cashes" if 2+ legs hit
        pair_cashed = hits >= 2
        night_results.append((dt, hits, total, pair_cashed))

    if not night_results:
        return None

    nights = len(night_results)
    pair_nights = sum(1 for _, _, _, p in night_results if p)
    total_legs = sum(t for _, _, t, _ in night_results)
    total_hits = sum(h for _, h, _, _ in night_results)
    best = max(night_results, key=lambda x: x[1])

    # current streak of pair-cashing nights (from most recent backward)
    streak = 0
    for _, _, _, p in reversed(night_results):
        if p:
            streak += 1
        else:
            break
    cold = 0
    if streak == 0:
        for _, h, _, _ in reversed(night_results):
            if h == 0:
                cold += 1
            else:
                break

    return {
        "nights": nights,
        "pair_nights": pair_nights,
        "leg_rate": round(total_hits / total_legs * 100, 1) if total_legs else 0.0,
        "best_night": best,          # (date, hits, total, pair)
        "streak": streak,
        "cold": cold,
        "recent": night_results[-5:],
    }


def build_rows(hr_df, hr_hit_rates, hr_today, timestamp_str, edge_bands=None, hr_all_scores=None):
    staging = []   # collected slate legs/tickets for the Bet_Staging tab
    def pad(row):
        return list(row) + [""] * (N_COLS - len(row))

    E = pad([])
    rows = []

    rows.append((pad([f"⏱  Last Updated: {timestamp_str}"]), "timestamp"))

    hr_source = hr_today if (hr_today is not None and not hr_today.empty) else hr_df

    # ── +EV SELECTIONS — each player's odds vs THEIR band's breakeven ─────
    # For every player, find their score x odds BAND (same logic as the
    # working edge-bands), get that band's historical hit rate -> breakeven,
    # and show the player if HIS odds beat it. A player priced long enough is
    # +EV for himself even if the band averages -EV. Bands computed fresh from
    # HR_All_Scores each run, so it tracks the data daily.
    # bands need the full HR_All_Scores (with dates); hr_df is the small picks
    # tab. Fall back to hr_df only if hr_all_scores wasn't passed.
    band_source = hr_all_scores if hr_all_scores is not None and not hr_all_scores.empty else hr_df
    band_rates = build_all_band_rates(band_source)

    rows.append((pad(["\U0001F4B0  +EV SELECTIONS — odds beat their band breakeven (live)"]),
                 "section_header_hr"))
    rows.append((pad(["Batter", "Team", "Pitcher", "Score", "Odds", "Book",
                      "Band BE", "Edge", "Band Hit%", "N", "Comb", "Plat", "Form", "Band", "Info"]), "col_header_hr"))

    picks = []
    _diag = {"total": 0, "no_odds": 0, "no_band": 0, "neg_edge": 0, "kept": 0}
    if not hr_source.empty and band_rates:
        for _, row in hr_source.iterrows():
            batter = str(row.get("player_name", "")).strip()
            if not batter or batter == "nan":
                continue
            _diag["total"] += 1
            sc = resolve_score(row)
            # prefer the best available price across your books; fall back to
            # consensus if best_odds isn't present (older rows / no odds).
            _best = str(row.get("best_odds", "")).strip()
            od = safe_float(_best) if _best not in ("", "nan", "None") \
                 else safe_float(row.get("consensus_odds", 0))
            best_book = str(row.get("best_book", "")).strip()
            cons_od = safe_float(row.get("consensus_odds", 0))
            if od <= 0:
                _diag["no_odds"] += 1
                continue
            # DATA-PROVEN GUARDRAILS (test_pocket_and_inversion, test_monotonic):
            # 1) Hard cap at +499. At +500 and longer the model's "value" picks
            #    are anti-predictive (10.4% in the 9-10 pocket, +val 4-6% at
            #    +700). The longshots are a black hole — never bet them.
            # 2) Cap score at 15. corr(score,HR) within 15+ = ~0 — above 15 is
            #    saturation the book overprices, not real edge.
            if od > 499:
                _diag["no_odds"] += 1
                continue
            if sc > 15:
                sc = 15.0
            band = player_band(sc, od, band_rates)
            if not band:
                _diag["no_band"] += 1
                continue
            # +EV for this player if his odds imply LESS than the band's rate
            leg_impl = (100 / (od + 100) if od >= 0 else abs(od) / (abs(od) + 100)) * 100
            edge = band["hit"] - leg_impl        # pp edge of his price vs band rate
            if edge <= 0:
                _diag["neg_edge"] += 1
                continue
            _diag["kept"] += 1
            bh = str(row.get("batter_hand", "")).strip().upper()[:1]
            ph = str(row.get("pitcher_hand", "")).strip().upper()[:1]
            eff = ("R" if ph == "L" else "L") if bh == "S" else bh
            p_bbl = safe_float(row.get(f"pitcher_barrel_vs_{'lhh' if eff=='L' else 'rhh'}", 0))
            b_bbl = safe_float(row.get("season_barrel_pct", 0))
            b_iso = safe_float(row.get(f"vs_{'lhp' if ph=='L' else 'rhp'}_iso", 0))
            info = f"P:{p_bbl:.0f}%bbl \u00b7 B:{b_bbl:.0f}%bbl \u00b7 {b_iso:.3f}v{ph or '?'}HP"
            be = band["be"]
            be_s = f"+{int(round(be))}" if be > 0 else f"-{int(round(abs(be)))}"
            # matchup tier + form for ranking/layering on top of edge
            plat = safe_float(row.get("platoon_score", 0))
            pitch = safe_float(row.get("pitch_matchup_score", 0))
            combo = plat + pitch
            season_bbl = b_bbl
            form = "\u2796"
            for win, bc, blc in [("7d","bbe_7d","barrel_pct_7d"),
                                 ("10d","bbe_10d","barrel_pct_10d"),
                                 ("14d","bbe_14d","barrel_pct_14d")]:
                wbbe = safe_float(row.get(bc, 0)); wbbl = safe_float(row.get(blc, 0))
                if wbbe >= 8:
                    diff = wbbl - season_bbl
                    form = "\U0001F525" if diff >= 4 else "\U0001F9CA" if diff <= -4 else "\u2796"
                    break
            # proven-zone tag: +301-499 value bucket hit 27.8% (model beats
            # market there); short-odds 9-10 pocket hit 21-26%. Float these up.
            in_prime = (301 <= od <= 499) or (9 <= sc < 10 and od <= 499)
            picks.append({"batter":batter,"team":str(row.get("team","")).strip(),
                "pitcher":str(row.get("pitcher_name","")).strip(),
                "sc":sc,"od":od,"be_s":be_s,"edge":edge,"rate":band["hit"],
                "band":band["band"],"info":info,"bn":band.get("n",0),
                "combo":combo,"plat":plat,"form":form,
                "book":best_book,"cons":cons_od,"prime":in_prime})

    # rank: proven-zone plays first (the +301-499 value bucket & 9-10 short-odds
    # pocket, where the model genuinely beats the market), then by edge, then by
    # matchup tier. Longshots already filtered out (+499 cap).
    picks.sort(key=lambda x: (0 if x["prime"] else 1,
                              -round(x["edge"] * 2) / 2, -x["combo"]))
    print(f"  +EV selections: {_diag['kept']} kept of {_diag['total']} legs "
          f"(no_odds={_diag['no_odds']}, no_band={_diag['no_band']}, "
          f"neg_edge={_diag['neg_edge']}); bands={len(band_rates)}")
    if not picks:
        rows.append((pad(["\u2014", "No +EV selections today "
                          "(no player's odds beat their band breakeven)", ""]), "no_plays"))
    else:
        for c in picks:
            rows.append((pad([
                c["batter"], c["team"], c["pitcher"], f"{c['sc']:.1f}",
                f"+{int(c['od'])}", c.get("book","") or "\u2014", c["be_s"],
                f"+{c['edge']:.1f}pp", f"{c['rate']:.1f}%", str(c.get("bn","")),
                f"{c['combo']:+.1f}", f"{c['plat']:+.1f}", c["form"], c["band"], c["info"]]),
                "data_hr_strong"))

    # ── 🔁 5-LEG ROUND ROBINS (by 2s) — always fill ───────────────────────
    # Two versions from the same <=+499 pool:
    #  A) ranked by SCORE×ODDS CELL HIT RATE (the validated monotonic table) —
    #     the sharpest probability estimate, with a sample floor: if a guy's
    #     cell is too thin to trust, fall back to ranking him by score.
    #  B) ranked by BLEND (the current blend) — kept for fun / comparison.
    import itertools as _it
    rr_pool = []
    if not hr_source.empty:
        for _, row in hr_source.iterrows():
            nm = str(row.get("player_name", "")).strip()
            if not nm or nm == "nan":
                continue
            sc = resolve_score(row)
            _b = str(row.get("best_odds", "")).strip()
            od = safe_float(_b) if _b not in ("", "nan", "None") else safe_float(row.get("consensus_odds", 0))
            if od <= 0 or od > 499:      # longshot cap
                continue
            plat = safe_float(row.get("platoon_score", 0))
            pitch = safe_float(row.get("pitch_matchup_score", 0))
            blend = blended_hit_prob(sc, od, plat, pitch, hr_hit_rates)
            # cell hit rate from the validated score×odds table + its sample size
            cell = player_band(sc, od, band_rates) or {}
            cell_rate = cell.get("hit", None)     # % or None if no/thin cell
            cell_n = cell.get("n", 0)
            rr_pool.append({"nm": nm, "team": str(row.get("team", "")).strip(),
                "pit": str(row.get("pitcher_name", "")).strip(), "sc": sc, "od": od,
                "book": str(row.get("best_book", "")).strip(), "blend": blend,
                "cell_rate": cell_rate, "cell_n": cell_n})

    _CELL_FLOOR = 25   # need >=25 resolved in the cell to trust its rate

    def _pick_five(rank_key):
        pool = sorted(rr_pool, key=rank_key)
        five, fteams = [], set()
        for c in pool:
            if len(five) >= 5: break
            if c["team"] in fteams: continue
            five.append(c); fteams.add(c["team"])
        if len(five) < 5:
            for c in pool:
                if len(five) >= 5: break
                if c in five: continue
                five.append(c)
        return five

    def _emit_rr(title, five, prob_col_label, prob_fn):
        rows.append((pad([title]), "section_header_hr"))
        if len(five) < 5:
            rows.append((pad(["—", "Fewer than 5 eligible bats today", ""]), "no_plays"))
            rows.append((E[:], "spacer"))
            return
        rows.append((pad(["Leg", "Batter", "Team", "Pitcher", "Score", "Odds", "Book", prob_col_label]),
                     "col_header_hr"))
        for i, c in enumerate(five, 1):
            rows.append((pad([str(i), c["nm"], c["team"], c["pit"], f"{c['sc']:.1f}",
                f"+{int(c['od'])}", c["book"] or "—", prob_fn(c)]), "data_hr_strong"))
        rows.append((pad([""]), "spacer"))
        rows.append((pad(["  Ten 2-leg parlays (25¢ each = $2.50 total):"]), "col_header_parlay"))
        rows.append((pad(["Combo", "Odds", "Pays", "25¢ wins"]), "col_header_hr"))
        for a, b in _it.combinations(five, 2):
            dec = (1 + (a["od"]/100 if a["od"] > 0 else 100/abs(a["od"]))) * \
                  (1 + (b["od"]/100 if b["od"] > 0 else 100/abs(b["od"])))
            pay = combined_american([a["od"], b["od"]])
            win = 0.25 * (dec - 1)
            combo_name = f"{a['nm'].split()[-1]} + {b['nm'].split()[-1]}"
            rows.append((pad([combo_name, f"+{int(a['od'])}/+{int(b['od'])}", pay, f"${win:.2f}"]),
                "data_hr_strong"))
        rows.append((E[:], "spacer"))

    # A) PRIMARY — rank by cell hit rate (trusted cells first), score as the
    #    fallback for thin/absent cells so a noisy small-sample rate never wins.
    def _cell_key(c):
        trusted = c["cell_rate"] is not None and c["cell_n"] >= _CELL_FLOOR
        # sort: trusted cells first (by rate desc), then everyone else by score
        return (0 if trusted else 1,
                -(c["cell_rate"] if trusted else -999),
                -c["sc"])
    five_cell = _pick_five(_cell_key)
    def _cell_prob(c):
        if c["cell_rate"] is not None and c["cell_n"] >= _CELL_FLOOR:
            return f"{c['cell_rate']:.0f}% (n{c['cell_n']})"
        return f"sc {c['sc']:.1f}"
    _emit_rr("🔁  5-LEG RR — ranked by score×odds hit rate (sharp)",
             five_cell, "Cell Hit%", _cell_prob)

    # B) FUN — rank by blend
    five_blend = _pick_five(lambda c: -c["blend"])
    _emit_rr("🔁  5-LEG RR — ranked by blend (for fun)",
             five_blend, "Blend%", lambda c: f"{c['blend']*100:.1f}%")

    return rows, staging


# ── Dashboard writer ────────────────────────────────────────────────────────
def write_dashboard(gc, sheet_id, rows) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(DASHBOARD_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=DASHBOARD_SHEET, rows=max(RESET_ROWS, len(rows) + 10), cols=RESET_COLS)
    ws_id = ws.id

    # Full reset: unmerge everything + blanket default format so no stale
    # cell (old KS/HRRBI colors, merges, wraps) survives into the new layout.
    reset_reqs = [
        {"unmergeCells": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": RESET_ROWS,
                                    "startColumnIndex": 0, "endColumnIndex": RESET_COLS}}},
        {"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": RESET_ROWS,
                      "startColumnIndex": 0, "endColumnIndex": RESET_COLS},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_BG,
                "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                "verticalAlignment": "MIDDLE", "horizontalAlignment": "LEFT", "wrapStrategy": "CLIP",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,horizontalAlignment,wrapStrategy)",
        }},
    ]
    try:
        with_retry(lambda: sh.batch_update({"requests": reset_reqs}))
    except Exception:
        pass

    with_retry(lambda: ws.clear())
    data = [row_data for row_data, _ in rows]
    with_retry(lambda: ws.update(data, value_input_option="RAW"))

    reqs = []
    data_counts = {}
    for r, (row_data, rtype) in enumerate(rows):
        if rtype == "timestamp":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "italic": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
            }})
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "mergeType": "MERGE_ALL"}})

        elif rtype.startswith("section_header"):
            color = COLOR_PURPLE if "parlay" in rtype else COLOR_GOLD
            text_color = COLOR_WHITE if "parlay" in rtype else COLOR_BLACK
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": color,
                    "textFormat": {"foregroundColor": text_color, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 12},
                    "horizontalAlignment": "LEFT", "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "mergeType": "MERGE_ALL"}})

        elif rtype.startswith("col_header"):
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 9},
                    "horizontalAlignment": "LEFT", "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})

        elif rtype.startswith("data"):
            key = "parlay" if rtype == "data_parlay" else "hr"
            c = data_counts.get(key, 0)
            data_counts[key] = c + 1
            bg = COLOR_BG if c % 2 == 0 else COLOR_BG_ALT
            # whole row: bg + CLIP (names/values stay on one line, no wrapping)
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                    "verticalAlignment": "MIDDLE", "horizontalAlignment": "LEFT", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,horizontalAlignment,wrapStrategy)",
            }})
            # col 0 (Rank/Leg/Ticket) centered dim
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": 1},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "horizontalAlignment": "CENTER"}},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})
            # col 3 (Score) colored by tier
            score_color = COLOR_GREEN if "strong" in rtype else (
                COLOR_GOLD if ("moderate" in rtype or rtype == "data_parlay") else COLOR_WHITE)
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 3, "endColumnIndex": 4},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": score_color, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "horizontalAlignment": "CENTER"}},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})
            # col 6 = Edge (singles) / Payout (parlays) — green accent
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 6, "endColumnIndex": 7},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_GREEN, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11}}},
                "fields": "userEnteredFormat(textFormat)",
            }})

        elif rtype == "no_plays":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_BG_ALT,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "italic": True,
                                   "fontFamily": "Roboto", "fontSize": 11}}},
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
            }})

    # Column widths matched to the +EV layout:
    # Batter Team Pitcher Score Odds Book BandBE Edge BandHit% N Comb Plat Form Band Info
    col_widths = [110, 60, 100, 52, 60, 88, 72, 64, 74, 44, 52, 52, 46, 118, 150]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w}, "fields": "pixelSize",
        }})

    # Row heights
    for r, (_, rtype) in enumerate(rows):
        if rtype == "timestamp":
            h = 30
        elif rtype.startswith("section_header"):
            h = 38
        elif rtype == "spacer":
            h = 12
        elif rtype.startswith("col_header"):
            h = 24
        else:
            h = 44
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": r, "endIndex": r + 1},
            "properties": {"pixelSize": h}, "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 0},
                       "tabColorStyle": {"rgbColor": COLOR_GOLD}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("Dashboard formatting applied.")
    except APIError as e:
        print(f"Dashboard formatting failed: {e}")


# ── Scorecard (HR only) ─────────────────────────────────────────────────────
def write_scorecard(gc, sheet_id, rows_data, today_str) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Scorecard")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Scorecard", rows=1000, cols=11)
    ws_id = ws.id

    existing = with_retry(lambda: ws.get_all_values())
    headers = ["Date", "Model", "Player / Pitcher", "Team", "Score",
               "Direction / Line", "Suggested Odds", "Your Odds", "Stake", "Result", "P&L"]
    if not existing or existing[0] != headers:
        existing = [headers]
    kept = [r for r in existing[1:] if r and str(r[0]).strip() != today_str]

    today_rows = []
    current_model = ""
    for row_data, rtype in rows_data:
        if rtype == "section_header_hr":
            current_model = "HR Single"
        elif rtype == "section_header_parlay":
            current_model = "HR Parlay 2-leg" if "2-LEG" in str(row_data[0]) else "HR Parlay 3-leg"

        if rtype in ("data_hr_strong", "data_hr_moderate", "data_hr_light"):
            name = str(row_data[1]).strip()
            if name and name != "—":
                today_rows.append([today_str, "HR Single", name, str(row_data[2]).strip(),
                                   str(row_data[3]).strip(), "HR", str(row_data[4]).strip(),
                                   "", "", "", ""])
        elif rtype == "data_parlay":
            name = str(row_data[1]).strip()
            if name and name != "—":
                today_rows.append([today_str, current_model, name, str(row_data[2]).strip(),
                                   str(row_data[3]).strip(), "HR", str(row_data[4]).strip(),
                                   "", "", "", ""])

    if not today_rows:
        print("Scorecard: no plays to write today.")
        return

    all_rows = [headers] + kept + today_rows
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update(all_rows, value_input_option="USER_ENTERED"))

    total = len(all_rows)
    reqs = [{"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total,
                  "startColumnIndex": 0, "endColumnIndex": 11},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP", "horizontalAlignment": "LEFT"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy,horizontalAlignment)"}},
        {"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": 11},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_HEADER_BG,
                "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True,
                               "fontFamily": "Roboto", "fontSize": 10},
                "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}},
    ]
    model_colors = {
        "HR Single": COLOR_GOLD,
        "HR Parlay 3-leg": COLOR_PURPLE,
        "HR Parlay 2-leg": {"red": 0.451, "green": 0.227, "blue": 0.620},
    }
    for idx, row in enumerate(all_rows[1:], start=1):
        if len(row) < 2:
            continue
        bg = COLOR_BG if idx % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": idx, "endRowIndex": idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 11},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)"}})
        color = model_colors.get(str(row[1]).strip())
        if color:
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": idx, "endRowIndex": idx + 1,
                          "startColumnIndex": 1, "endColumnIndex": 2},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": color, "bold": True},
                    "horizontalAlignment": "CENTER"}},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})
    for ci in (7, 8, 9):
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": total,
                      "startColumnIndex": ci, "endColumnIndex": ci + 1},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.10, "green": 0.10, "blue": 0.15},
                "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True},
                "horizontalAlignment": "CENTER"}},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"}})
    col_widths = [90, 130, 180, 60, 70, 120, 100, 90, 70, 70, 80]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w}, "fields": "pixelSize"}})
    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1},
                       "tabColorStyle": {"rgbColor": COLOR_GREEN}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle"}})
    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print(f"Scorecard written: {len(today_rows)} plays for {today_str}")
    except APIError as e:
        print(f"Scorecard formatting failed: {e}")


# ── Main ────────────────────────────────────────────────────────────────────
def write_bet_staging(gc, sheet_id, staging, today_str) -> None:
    """Pre-write today's DISPLAYED picks (singles + edge-slate legs) to a
    clean Bet_Staging tab: one row per leg, consensus_at_bet filled, odds_taken
    blank for the user to fill when they place a bet. Combined consensus and
    implied breakeven appear once per parlay ticket (on its first leg),
    computed correctly from the legs' decimal odds. Model_Bet_Tracker is left
    untouched. Today's rows are replaced each run; other dates preserved."""
    header = ["date", "ticket_type", "ticket_id", "player", "team", "score",
              "consensus_at_bet", "odds_taken", "combined_consensus",
              "breakeven_pct", "band", "result"]
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Bet_Staging")
        existing = pd.DataFrame(ws.get_all_records())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Bet_Staging", rows=2000, cols=len(header))
        existing = pd.DataFrame()

    # preserve prior dates (and any odds_taken/result the user already filled)
    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"].astype(str).str.strip() != today_str]
    else:
        existing = pd.DataFrame(columns=header)

    new_rows = []
    for s in staging:
        new_rows.append({
            "date": today_str, "ticket_type": s["ticket_type"],
            "ticket_id": s["ticket_id"], "player": s["player"], "team": s["team"],
            "score": s["score"], "consensus_at_bet": s["consensus_at_bet"],
            "odds_taken": "", "combined_consensus": s["combined_consensus"],
            "breakeven_pct": s["breakeven_pct"], "band": s["band"], "result": "",
        })
    new_df = pd.DataFrame(new_rows, columns=header)
    out = pd.concat([existing.reindex(columns=header), new_df], ignore_index=True) \
        if not existing.empty else new_df
    out = out.fillna("")

    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([header] + out.astype(str).values.tolist()))
    print(f"Bet_Staging: wrote {len(new_rows)} legs for {today_str}")


def main() -> None:
    time.sleep(5)
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    print("Reading HR picks + scores for dashboard...")
    # Tab names default to MAIN's tabs. To test in dev against v1/v2 tabs, set
    # HR_SCORES_TAB / HR_PICKS_TAB env vars in the dev workflow — main runs with
    # neither set and reads the regular tabs, so merging can't break main.
    scores_tab = os.environ.get("HR_SCORES_TAB", "HR_All_Scores")
    picks_tab  = os.environ.get("HR_PICKS_TAB", "Top_HR_Picks")
    print(f"Reading picks from '{picks_tab}', scores from '{scores_tab}'")
    hr_df = read_sheet(gc, sheet_id, picks_tab)
    time.sleep(2)
    hr_all_scores = read_sheet_raw(gc, sheet_id, scores_tab)
    time.sleep(2)

    print(f"HR picks: {len(hr_df)} rows | HR All Scores: {len(hr_all_scores)} rows")
    hr_hit_rates = build_hr_hit_rates(hr_all_scores)
    edge_bands = build_edge_bands(hr_all_scores)
    print(f"Edge bands live: {len(edge_bands)} qualifying (+edge, n>=30)")
    for b in edge_bands[:12]:
        print(f"  {b['band']:<20} edge +{b['edge_pp']:.1f}pp  hit {b['hit']:.1f}%  n={b['n']}")

    today_str = _date.today().strftime("%Y-%m-%d")
    if not hr_all_scores.empty and "date" in hr_all_scores.columns:
        hr_today = hr_all_scores[hr_all_scores["date"].astype(str).str.strip() == today_str].copy()
    else:
        hr_today = pd.DataFrame()
    print(f"HR today's scores: {len(hr_today)} players")

    et = pytz.timezone("America/New_York")
    ts = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")

    rows, staging = build_rows(hr_df, hr_hit_rates, hr_today, ts, edge_bands=edge_bands, hr_all_scores=hr_all_scores)
    write_dashboard(gc, sheet_id, rows)
    # dashboard is top-15 matchup only now; no tickets to stage, so the
    # Bet_Staging write is skipped to avoid overwriting it with blanks.
    # write_bet_staging(gc, sheet_id, staging, today_str)
    time.sleep(3)
    write_scorecard(gc, sheet_id, rows, today_str)
    print("Dashboard written to 'Today's Top Picks'")


if __name__ == "__main__":
    main()
