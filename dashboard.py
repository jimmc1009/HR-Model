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

N_COLS = 8
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

    resolved["hr_score"] = resolved.apply(
        lambda r: safe_float(r.get("hr_score_corrected"))
        if str(r.get("hr_score_corrected", "")).strip() not in ("", "nan", "None")
        else safe_float(r.get("hr_score")),
        axis=1,
    )
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
    tier_defs = [
        ("15+", 15, 999), ("14-15", 14, 15), ("13-14", 13, 14),
        ("12-13", 12, 13), ("11-12", 11, 12),
        ("10-11", 10, 11), ("9-10", 9, 10), ("8.5-9", 8.5, 9),
    ]
    zone_keys = ["le300", "301-499", "500-699", "700plus"]
    for tier_label, lo, hi in tier_defs:
        tier_sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(tier_sub) >= 5:
            hit_rates[tier_label] = shrunk_rate(tier_sub)
        for zk in zone_keys:
            zsub = tier_sub[tier_sub["odds_num"].apply(_hr_odds_zone_key) == zk]
            if len(zsub) >= 15:
                hit_rates[(tier_label, zk)] = shrunk_rate(zsub)
    print(f"  HR hit rate lookup: {len(hit_rates)} entries from {len(resolved)} resolved picks "
          f"(base {base_rate*100:.1f}%, shrink k={SHRINK_K:.0f})")
    return hit_rates


def get_hr_score_tier(score: float) -> str:
    if score >= 15:  return "15+"
    if score >= 14:  return "14-15"
    if score >= 13:  return "13-14"
    if score >= 12:  return "12-13"
    if score >= 11:  return "11-12"
    if score >= 10:  return "10-11"
    if score >= 9:   return "9-10"
    if score >= 8.5: return "8.5-9"
    return "below-8.5"


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
    df["score"] = df.apply(
        lambda r: safe_float(r.get("hr_score_corrected"))
        if str(r.get("hr_score_corrected", "")).strip() not in ("", "nan", "None")
        else safe_float(r.get("hr_score")), axis=1)
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


def build_rows(hr_df, hr_hit_rates, hr_today, timestamp_str, edge_bands=None):
    staging = []   # collected slate legs/tickets for the Bet_Staging tab
    def pad(row):
        return list(row) + [""] * (N_COLS - len(row))

    E = pad([])
    rows = []

    rows.append((pad([f"⏱  Last Updated: {timestamp_str}"]), "timestamp"))

    hr_source = hr_today if (hr_today is not None and not hr_today.empty) else hr_df

    # ── 3 TIERED 5-LEG ROUND ROBINS ──────────────────────────────────────
    # Legs filtered to what your resolved data says works: hr_score >= 10,
    # odds <= +499 (above +500 collapses), pitch score >= +0.2 (Weak/Neutral
    # pitch hits 1.6-7.8%), platoon >= 0 (negative platoon hits 5.8-7%).
    # Survivors ranked by combined platoon+pitch. Best 5 -> RR#1, next 5 ->
    # RR#2, weakest 5 -> RR#3. Concentrating the strongest legs in one ticket
    # maximizes that ticket's cash probability. No player repeats across RRs.
    rows.append((pad(["\U0001F3B0  TIERED 5-LEG ROUND ROBINS — RR#1 strongest, RR#3 weakest"]),
                 "section_header_hr"))

    cand = []
    if not hr_source.empty:
        for _, row in hr_source.iterrows():
            batter = str(row.get("player_name", "")).strip()
            if not batter or batter == "nan":
                continue
            hr_score = safe_float(row.get("hr_score", 0))
            odds = safe_float(row.get("consensus_odds", 0))
            plat = safe_float(row.get("platoon_score", 0))
            pitch = safe_float(row.get("pitch_matchup_score", 0))
            # ── data-driven filter ──
            if hr_score < 10:            continue
            if not (0 < odds <= 499):    continue
            if pitch < 0.2:              continue
            if plat < 0:                 continue
            combined = plat + pitch

            bh = str(row.get("batter_hand", "")).strip().upper()[:1]
            ph = str(row.get("pitcher_hand", "")).strip().upper()[:1]
            eff = ("R" if ph == "L" else "L") if bh == "S" else bh
            p_bbl = safe_float(row.get(f"pitcher_barrel_vs_{'lhh' if eff=='L' else 'rhh'}", 0))
            b_bbl = safe_float(row.get("season_barrel_pct", 0))
            b_iso = safe_float(row.get(f"vs_{'lhp' if ph=='L' else 'rhp'}_iso", 0))
            info = f"P:{p_bbl:.0f}%bbl \u00b7 B:{b_bbl:.0f}%bbl \u00b7 {b_iso:.3f}v{ph or '?'}HP"

            season_bbl = b_bbl
            trend = "\u2796 \u2014"
            for win, bbe_col, bbl_col in [("7d", "bbe_7d", "barrel_pct_7d"),
                                          ("10d", "bbe_10d", "barrel_pct_10d"),
                                          ("14d", "bbe_14d", "barrel_pct_14d")]:
                wbbe = safe_float(row.get(bbe_col, 0))
                wbbl = safe_float(row.get(bbl_col, 0))
                if wbbe >= 8:
                    diff = wbbl - season_bbl
                    icon = "\U0001F525" if diff >= 4 else "\U0001F9CA" if diff <= -4 else "\u2796"
                    trend = f"{icon} {win} {wbbl:.0f}%"
                    break

            cand.append({
                "batter": batter, "team": str(row.get("team", "")).strip(),
                "pitcher": str(row.get("pitcher_name", "")).strip(),
                "opp_pit": str(row.get("pitcher_name", "")).strip(),
                "combined": combined, "plat": plat, "pitch": pitch,
                "odds": odds, "hr_score": hr_score, "info": info, "trend": trend,
            })

    cand.sort(key=lambda x: -x["combined"])

    # RR#1 and RR#2 use the STRICT filter (already applied above building cand).
    # RR#3 gets filled with next-best legs that missed the strict filter, so
    # the dart-throw ticket still fills without diluting the confidence plays.
    # Build the relaxed pool now (score>=10, odds<=499 kept; platoon/pitch gates
    # dropped) for fallback only.
    relaxed = []
    if not hr_source.empty:
        for _, row in hr_source.iterrows():
            batter = str(row.get("player_name", "")).strip()
            if not batter or batter == "nan":
                continue
            hr_score = safe_float(row.get("hr_score", 0))
            odds = safe_float(row.get("consensus_odds", 0))
            if hr_score < 10 or not (0 < odds <= 499):
                continue
            plat = safe_float(row.get("platoon_score", 0))
            pitch = safe_float(row.get("pitch_matchup_score", 0))
            bh = str(row.get("batter_hand", "")).strip().upper()[:1]
            ph = str(row.get("pitcher_hand", "")).strip().upper()[:1]
            eff = ("R" if ph == "L" else "L") if bh == "S" else bh
            p_bbl = safe_float(row.get(f"pitcher_barrel_vs_{'lhh' if eff=='L' else 'rhh'}", 0))
            b_bbl = safe_float(row.get("season_barrel_pct", 0))
            b_iso = safe_float(row.get(f"vs_{'lhp' if ph=='L' else 'rhp'}_iso", 0))
            info = f"P:{p_bbl:.0f}%bbl \u00b7 B:{b_bbl:.0f}%bbl \u00b7 {b_iso:.3f}v{ph or '?'}HP"
            trend = "\u2796 \u2014"
            for win, bbe_col, bbl_col in [("7d", "bbe_7d", "barrel_pct_7d"),
                                          ("10d", "bbe_10d", "barrel_pct_10d"),
                                          ("14d", "bbe_14d", "barrel_pct_14d")]:
                wbbe = safe_float(row.get(bbe_col, 0)); wbbl = safe_float(row.get(bbl_col, 0))
                if wbbe >= 8:
                    diff = wbbl - b_bbl
                    icon = "\U0001F525" if diff >= 4 else "\U0001F9CA" if diff <= -4 else "\u2796"
                    trend = f"{icon} {win} {wbbl:.0f}%"
                    break
            relaxed.append({
                "batter": batter, "team": str(row.get("team", "")).strip(),
                "pitcher": str(row.get("pitcher_name", "")).strip(),
                "opp_pit": str(row.get("pitcher_name", "")).strip(),
                "combined": plat + pitch, "plat": plat, "pitch": pitch,
                "odds": odds, "hr_score": hr_score, "info": info, "trend": trend,
            })
    relaxed.sort(key=lambda x: -x["combined"])

    # select up to 15 unique, pitcher-diversified: strict legs first, then
    # relaxed fallback only to reach 15 (fills RR#3).
    selected = []
    used_pit = set()
    strict_ids = set()
    for c in cand:                     # strict legs first
        if len(selected) >= 15:
            break
        if c["opp_pit"] and c["opp_pit"] in used_pit:
            continue
        selected.append(c)
        strict_ids.add((c["batter"], c["odds"]))
        if c["opp_pit"]:
            used_pit.add(c["opp_pit"])
    n_strict = len(selected)
    for c in relaxed:                  # fallback to fill remaining slots
        if len(selected) >= 15:
            break
        cid = (c["batter"], c["odds"])
        if cid in strict_ids:
            continue
        if c["opp_pit"] and c["opp_pit"] in used_pit:
            continue
        selected.append(c)
        if c["opp_pit"]:
            used_pit.add(c["opp_pit"])

    if len(selected) < 2:
        rows.append((pad(["\u2014", "Not enough qualifying legs today "
                          "(need pitch\u2265+0.2, platoon\u22650, \u2264+499, score\u226510)", ""]),
                     "no_plays"))
    else:
        # assign the 15 ranked legs into 3 RRs of 5, enforcing NO two players
        # from the same team WITHIN a single RR (teams may repeat across RRs).
        # Greedy: walk ranked legs, drop each into the first RR (1->2->3) that
        # isn't full and doesn't already hold that team.
        rr_legs = [[], [], []]
        rr_teams = [set(), set(), set()]
        leftovers = []
        for c in selected:
            placed = False
            for k in range(3):
                if len(rr_legs[k]) < 5 and c["team"] not in rr_teams[k]:
                    rr_legs[k].append(c)
                    rr_teams[k].add(c["team"])
                    placed = True
                    break
            if not placed:
                leftovers.append(c)
        # fill any short RR with leftovers (team dup allowed only as last resort)
        for k in range(3):
            while len(rr_legs[k]) < 5 and leftovers:
                rr_legs[k].append(leftovers.pop(0))

        strict_set = {(c["batter"], c["odds"]) for c in selected[:n_strict]}

        def emit_rr(title, legs):
            n_fb = sum(1 for c in legs if (c["batter"], c["odds"]) not in strict_set)
            if n_fb:
                title += f"  ({n_fb} relaxed-filter)"
            rows.append((pad([title]), "col_header_parlay"))
            rows.append((pad(["#", "Batter", "Team", "Pitcher", "Comb",
                              "Plat", "Pitch", "Odds", "HR", "Info", "Form"]),
                         "col_header_hr"))
            for i, c in enumerate(legs, 1):
                rows.append((pad([
                    str(i), c["batter"], c["team"], c["pitcher"],
                    f"{c['combined']:+.2f}", f"{c['plat']:+.1f}", f"{c['pitch']:+.1f}",
                    f"+{int(c['odds'])}", f"{c['hr_score']:.1f}", c["info"], c["trend"],
                ]), "data_hr_strong"))
            # ── the 10 two-leg combos for this RR (for building at the book) ──
            if len(legs) >= 2:
                import itertools as _it
                pair_list = list(_it.combinations(legs, 2))
                rows.append((pad([f"  \u25b8 {len(pair_list)} PAIRS:"]), "no_plays"))
                for j, (a, b) in enumerate(pair_list, 1):
                    same_game = a["opp_pit"] and a["opp_pit"] == b["opp_pit"]
                    tag = "  (same game)" if same_game else ""
                    rows.append((pad([f"  {j}.", f"{a['batter']} + {b['batter']}",
                                      f"+{int(a['odds'])}/+{int(b['odds'])}{tag}"]),
                                 "data_parlay"))
            rows.append((E[:], "spacer"))

        titles = ["\U0001F3C6 RR#1 — STRONGEST 5",
                  "\U0001F948 RR#2 — MIDDLE 5",
                  "\U0001F949 RR#3 — WEAKEST 5"]
        for title, legs in zip(titles, rr_legs):
            if legs:
                emit_rr(title, legs)

    rows.append((E[:], "spacer"))
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
            # whole row: bg + WRAP (so Why can wrap without clipping)
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                    "verticalAlignment": "MIDDLE", "horizontalAlignment": "LEFT", "wrapStrategy": "WRAP"}},
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

    # Column widths (fit content; Why wraps)
    col_widths = [56, 158, 52, 62, 66, 66, 120, 300]
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

    rows, staging = build_rows(hr_df, hr_hit_rates, hr_today, ts, edge_bands=edge_bands)
    write_dashboard(gc, sheet_id, rows)
    # dashboard is top-15 matchup only now; no tickets to stage, so the
    # Bet_Staging write is skipped to avoid overwriting it with blanks.
    # write_bet_staging(gc, sheet_id, staging, today_str)
    time.sleep(3)
    write_scorecard(gc, sheet_id, rows, today_str)
    print("Dashboard written to 'Today's Top Picks'")


if __name__ == "__main__":
    main()
