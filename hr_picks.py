import os
import json
from datetime import date, timedelta, datetime
from typing import Dict, Set
import time
import unicodedata
import pytz

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
from pybaseball import statcast
import requests

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ── Regression constants ───────────────────────────────────────────────────
LEAGUE_AVG_HR_PER_PA     = 2.5
LEAGUE_AVG_HR_PER_FB     = 10.0
LEAGUE_AVG_ISO           = 0.155
LEAGUE_AVG_BARREL_7D     = 8.0
LEAGUE_AVG_SEASON_BARREL = 8.0
LEAGUE_AVG_HARD_HIT_7D   = 40.0
LEAGUE_AVG_EV_7D         = 88.5

MIN_PA_FULL          = 150
MIN_PA_PARTIAL       = 30
MIN_BBE_7D_FULL      = 20
MIN_BBE_7D_PARTIAL   = 5

# ── Component weights ──────────────────────────────────────────────────────
# CHANGED: Weather 1.2 → 0.3 (wrong direction per analysis, -39.7% separation)
# CHANGED: Pitch matchup 1.2 → 1.2 (keep, insufficient data)
# Platoon, pull/park, momentum, BVP unchanged
PLATOON_BONUS_WEIGHT = 0.8
PITCH_MATCHUP_WEIGHT = 1.2
WEATHER_WEIGHT       = 0.3   # was 1.2 — data shows weather hurts prediction
PULL_PARK_WEIGHT     = 0.6
MOMENTUM_WEIGHT      = 0.4
BVP_WEIGHT           = 0.9

MIN_BATTING_AVG      = 0.200
MAX_PER_TEAM         = 2
MAX_PER_GAME         = 2

# ── Bet filter criteria ────────────────────────────────────────────────────
MIN_SCORE_FLOOR       = 9.5
MAX_CHALK_ODDS        = 300
MIN_VALUE_ODDS        = 500
MID_RANGE_SCORE_FLOOR = 11.0

COLOR_BG        = {"red": 0.114, "green": 0.114, "blue": 0.114}
COLOR_BG_ALT    = {"red": 0.149, "green": 0.149, "blue": 0.149}
COLOR_ACCENT    = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_ACCENT_DIM= {"red": 0.055, "green": 0.180, "blue": 0.318}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD      = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_SILVER    = {"red": 0.753, "green": 0.753, "blue": 0.753}
COLOR_BRONZE    = {"red": 0.804, "green": 0.498, "blue": 0.196}
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_ORANGE    = {"red": 0.980, "green": 0.502, "blue": 0.059}
COLOR_CHALK     = {"red": 0.200, "green": 0.600, "blue": 0.200}
COLOR_VALUE     = {"red": 0.576, "green": 0.439, "blue": 0.859}


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Rate limit hit — waiting {wait}s (attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def read_sheet(gc: gspread.Client, sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    try:
        ws   = sh.worksheet(worksheet_name)
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{worksheet_name}' not found.")
        return pd.DataFrame()


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def regress(value: float, league_avg: float, sample: float, full_sample: float) -> float:
    weight = min(sample / full_sample, 1.0)
    return (value * weight) + (league_avg * (1 - weight))


# ── Batter scoring functions ───────────────────────────────────────────────
# CHANGED: score_barrel_pct_7d max 2.0 → 2.5 (strongest separator at +43-46%)

def score_barrel_pct_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < 5:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 20: return 2.5   # was 2.0
    if v >= 15: return 1.8   # was 1.5
    if v >= 10: return 1.0
    if v >= 6:  return 0.4
    return 0.0


def score_season_barrel_pct(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_SEASON_BARREL, pa, MIN_PA_FULL)
    if v >= 14: return 1.2
    if v >= 11: return 0.9
    if v >=  9: return 0.6
    if v >=  7: return 0.3
    return 0.0


def score_iso(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_ISO, pa, MIN_PA_FULL)
    if v >= 0.300: return 1.2
    if v >= 0.250: return 0.9
    if v >= 0.200: return 0.6
    if v >= 0.175: return 0.4
    if v >= 0.150: return 0.2
    return 0.0


def score_hr_per_pa(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_HR_PER_PA, pa, MIN_PA_FULL)
    if v >= 6.0: return 1.8
    if v >= 4.0: return 1.2
    if v >= 2.5: return 0.6
    return 0.0


def score_hr_per_fb(v: float, pa: float) -> float:
    v = regress(v, LEAGUE_AVG_HR_PER_FB, pa, MIN_PA_FULL)
    if v >= 20: return 1.5
    if v >= 15: return 1.0
    if v >= 10: return 0.5
    return 0.0


# CHANGED: score_avg_ev_7d max 1.0 → 0.5 (only +1.8% separation, weak predictor)
def score_avg_ev_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < 5:
        return 0.0
    v = regress(v, LEAGUE_AVG_EV_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 97: return 0.5   # was 1.0
    if v >= 94: return 0.3   # was 0.6
    if v >= 91: return 0.15  # was 0.3
    return 0.0


def score_hard_hit_pct_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < 5:
        return 0.0
    v = regress(v, LEAGUE_AVG_HARD_HIT_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 55: return 0.8
    if v >= 45: return 0.5
    if v >= 35: return 0.2
    return 0.0


# ── Pitcher scoring functions ──────────────────────────────────────────────
# CHANGED: pitcher barrel max 1.5 → 1.0 (only +10.8% separation vs batter's +43%)
def score_pitcher_barrel_pct(v: float) -> float:
    if v >= 14: return 1.0   # was 1.5
    if v >= 11: return 0.75  # was 1.0
    if v >=  9: return 0.5   # was 0.6
    if v >=  7: return 0.25  # was 0.3
    return 0.0


# CHANGED: pitcher HR/FB max 1.5 → 1.0 (only +9.9% separation)
def score_pitcher_hr_per_fb(v: float) -> float:
    if v >= 20: return 1.0   # was 1.5
    if v >= 15: return 0.7   # was 1.0
    if v >= 13: return 0.4   # was 0.6
    if v >= 10: return 0.2   # was 0.3
    return 0.0


def score_pitcher_hard_hit_pct(v: float) -> float:
    if v >= 45: return 0.8
    if v >= 38: return 0.5
    if v >= 32: return 0.2
    return 0.0


def score_park_factor(v: float) -> float:
    if v >= 20:  return 0.25
    if v >= 10:  return 0.15
    if v >= 0:   return 0.05
    if v >= -10: return -0.1
    return -0.15


def score_pitcher_quality_penalty(
    barrel_pct: float,
    hard_hit_pct: float,
    hr_per_fb: float,
    pitcher_bbe: float,
) -> float:
    sample_weight = min(safe_float(pitcher_bbe) / 80.0, 1.0)
    penalty = 0.0
    if barrel_pct <= 4:   penalty += 0.8
    elif barrel_pct <= 5: penalty += 0.5
    elif barrel_pct <= 6: penalty += 0.3
    if hard_hit_pct <= 30:   penalty += 0.5
    elif hard_hit_pct <= 33: penalty += 0.3
    elif hard_hit_pct <= 36: penalty += 0.15
    if hr_per_fb <= 6:    penalty += 0.5
    elif hr_per_fb <= 8:  penalty += 0.3
    elif hr_per_fb <= 10: penalty += 0.1
    return round(penalty * sample_weight, 3)


def american_odds_to_profit(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return odds / 100.0


def normalize(series: pd.Series) -> pd.Series:
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def compute_momentum_score(row: pd.Series) -> tuple:
    ev_5d      = safe_float(row.get("avg_ev_5d",          0))
    ev_10d     = safe_float(row.get("avg_ev_10d",         0))
    bar_5d     = safe_float(row.get("barrel_pct_5d",      0))
    bar_10d    = safe_float(row.get("barrel_pct_10d",     0))
    hh_5d      = safe_float(row.get("hard_hit_pct_5d",    0))
    hh_10d     = safe_float(row.get("hard_hit_pct_10d",   0))
    la_5d      = safe_float(row.get("avg_la_5d",          0))
    la_10d     = safe_float(row.get("avg_la_10d",         0))
    bbe_5d     = safe_float(row.get("bbe_5d",             0))
    bbe_10d    = safe_float(row.get("bbe_10d",            0))
    pa_14d     = safe_float(row.get("pa_14d",             0))
    avg_14d    = safe_float(row.get("avg_14d",            0))
    avg_season = safe_float(row.get("batting_avg",        0))
    hot        = str(row.get("hot_streak",  "")).strip().lower()
    cold       = str(row.get("cold_streak", "")).strip().lower()

    if bbe_5d < 3 or bbe_10d < 5:
        return 0.0, ""

    score = 0.0
    parts = []

    if ev_5d > 0 and ev_10d > 0:
        ev_delta = ev_5d - ev_10d
        score   += ev_delta / 3.0
        if ev_delta >= 2.0:
            parts.append(f"📈 EV trending up ({ev_10d:.1f}→{ev_5d:.1f} mph)")
        elif ev_delta <= -2.0:
            parts.append(f"📉 EV trending down ({ev_10d:.1f}→{ev_5d:.1f} mph)")

    if bar_5d >= 0 and bar_10d >= 0:
        barrel_delta = bar_5d - bar_10d
        score       += barrel_delta / 8.0
        if barrel_delta >= 8.0:
            parts.append(f"📈 Barrel% trending up ({bar_10d:.1f}→{bar_5d:.1f}%)")
        elif barrel_delta <= -8.0:
            parts.append(f"📉 Barrel% trending down ({bar_10d:.1f}→{bar_5d:.1f}%)")

    if hh_5d > 0 and hh_10d > 0:
        hh_delta = hh_5d - hh_10d
        score   += hh_delta / 15.0
        if hh_delta >= 12.0:
            parts.append(f"💪 Hard hit% surging ({hh_10d:.1f}→{hh_5d:.1f}% last 5d)")
        elif hh_delta <= -12.0:
            parts.append(f"📉 Hard hit% dropping ({hh_10d:.1f}→{hh_5d:.1f}% last 5d)")

    if la_5d != 0 and la_10d != 0:
        optimal_la = 28.0
        dist_5d    = abs(la_5d  - optimal_la)
        dist_10d   = abs(la_10d - optimal_la)
        la_delta   = dist_10d - dist_5d
        score     += la_delta / 15.0
        if la_delta >= 8.0:
            parts.append(f"📐 Launch angle moving toward optimal ({la_10d:.1f}→{la_5d:.1f}°)")
        elif la_delta <= -8.0:
            parts.append(f"📐 Launch angle moving away from optimal ({la_10d:.1f}→{la_5d:.1f}°)")

    if pa_14d >= 20 and avg_season > 0 and avg_14d > 0:
        avg_delta = avg_14d - avg_season
        if avg_delta >= 0.060:
            score += 0.8
            parts.append(f"🔥 AVG trending up (.{int(avg_season*1000)} season → .{int(avg_14d*1000)} last 14d)")
        elif avg_delta >= 0.030:
            score += 0.4
            parts.append(f"📈 AVG trending up (.{int(avg_season*1000)} → .{int(avg_14d*1000)} last 14d)")
        elif avg_delta <= -0.060:
            score -= 0.8
            parts.append(f"❄️ AVG trending down (.{int(avg_season*1000)} season → .{int(avg_14d*1000)} last 14d)")
        elif avg_delta <= -0.030:
            score -= 0.4
            parts.append(f"📉 AVG trending down (.{int(avg_season*1000)} → .{int(avg_14d*1000)} last 14d)")

    if hot in ("1", "true", "yes"):
        score += 0.3
        if not any("trending up" in p or "surging" in p for p in parts):
            parts.append("🔥 On a hot streak")
    elif cold in ("1", "true", "yes"):
        score -= 0.3
        if not any("trending down" in p or "dropping" in p for p in parts):
            parts.append("❄️ On a cold streak")

    return round(score, 3), " | ".join(parts)


def compute_bvp_score(row: pd.Series) -> tuple:
    bvp_pa      = safe_float(row.get("bvp_pa", 0))
    bvp_iso     = safe_float(row.get("bvp_iso", 0))
    bvp_hr_rate = safe_float(row.get("bvp_hr_rate", 0))
    bvp_barrel  = safe_float(row.get("bvp_barrel_pct", 0))
    bvp_hr      = safe_float(row.get("bvp_hr", 0))

    if bvp_pa < 5:
        return 0.0, ""

    score   = (bvp_iso * 3.0) + (bvp_hr_rate / 10.0) + (bvp_barrel / 20.0)
    score   = round(score, 3)
    hr_note = f", {int(bvp_hr)} HR" if bvp_hr > 0 else ""
    desc    = ""

    if bvp_iso >= 0.200:
        desc = f"🔥 Owns this pitcher — ISO {bvp_iso:.3f} in {int(bvp_pa)} PA{hr_note}"
    elif bvp_iso >= 0.150:
        desc = f"✅ Strong vs this pitcher — ISO {bvp_iso:.3f} in {int(bvp_pa)} PA{hr_note}"
    elif bvp_iso <= 0.050 and bvp_pa >= 8:
        desc  = f"⚠️ Struggles vs this pitcher — ISO {bvp_iso:.3f} in {int(bvp_pa)} PA"
        score = -abs(score) * 0.5

    return score, desc


def compute_pull_park_score(row: pd.Series) -> tuple:
    batter_hand = str(row.get("batter_hand", "")).strip().upper()
    pull_rate   = safe_float(row.get("pull_rate", 0))
    season_bbe  = safe_float(row.get("season_bbe", 0))

    if season_bbe < 20:
        pull_rate = 40.0

    if batter_hand == "R":
        park_boost = safe_float(row.get("pull_boost_rhh", 0))
        wall_dist  = safe_float(row.get("lf_dist", 0))
        wall_ht    = safe_float(row.get("lf_height", 0))
        side = "LF"
    elif batter_hand == "L":
        park_boost = safe_float(row.get("pull_boost_lhh", 0))
        wall_dist  = safe_float(row.get("rf_dist", 0))
        wall_ht    = safe_float(row.get("rf_height", 0))
        side = "RF"
    else:
        return 0.0, ""

    if pull_rate <= 0 or wall_dist <= 0:
        return 0.0, ""

    pull_rate_scalar = min(pull_rate / 45.0, 1.2)
    score = round(park_boost * pull_rate_scalar, 3)
    desc  = ""

    if score >= 0.15:
        desc = f"🏟️ Pull hitter ({pull_rate:.0f}% pull) → short {side} ({wall_dist:.0f}ft, {wall_ht:.0f}ft wall)"
    elif score <= -0.15:
        desc = f"🏟️ Pull hitter ({pull_rate:.0f}% pull) → deep {side} ({wall_dist:.0f}ft, {wall_ht:.0f}ft wall)"

    return score, desc


def compute_pitch_matchup_score(row: pd.Series) -> tuple:
    scores        = []
    descriptions  = []
    pitch_penalty = 0.0

    for rank in range(1, 4):
        pitch_type = str(row.get(f"top_pitch_{rank}", "")).strip().upper()
        pitch_pct  = safe_float(row.get(f"top_pitch_{rank}_pct", 0))

        if not pitch_type or pitch_type in ("", "NAN", "NONE"):
            continue

        batter_iso      = safe_float(row.get(f"iso_vs_{pitch_type}", 0))
        batter_hr_rate  = safe_float(row.get(f"hr_rate_vs_{pitch_type}", 0))
        batter_barrel   = safe_float(row.get(f"barrel_pct_vs_{pitch_type}", 0))
        batter_has_data = (batter_iso > 0 or batter_hr_rate > 0)

        pitcher_iso      = safe_float(row.get(f"pitcher_iso_allowed_{pitch_type}", 0))
        pitcher_hr_rate  = safe_float(row.get(f"pitcher_hr_rate_allowed_{pitch_type}", 0))
        pitcher_barrel   = safe_float(row.get(f"pitcher_barrel_pct_allowed_{pitch_type}", 0))
        pitcher_has_data = (pitcher_iso > 0 or pitcher_hr_rate > 0)

        if not batter_has_data and not pitcher_has_data:
            continue

        batter_component  = (batter_iso * 3 + batter_hr_rate / 10 + batter_barrel / 20) if batter_has_data else 0.0
        pitcher_component = (pitcher_iso * 2 + pitcher_hr_rate / 10 + pitcher_barrel / 20) if pitcher_has_data else 0.0
        pitch_score       = (batter_component + pitcher_component) * (pitch_pct / 100)
        scores.append(pitch_score)

        parts = []
        if batter_has_data and pitcher_has_data:
            if batter_iso >= 0.150 and pitch_pct >= 15:
                if pitcher_iso >= 0.150:
                    parts.append(f"✅ {pitch_type} ({pitch_pct:.0f}%) — Batter ISO {batter_iso:.3f} + Pitcher allows {pitcher_iso:.3f} ISO")
                else:
                    parts.append(f"✅ Batter ISO {batter_iso:.3f} vs {pitch_type} ({pitch_pct:.0f}% usage)")
            elif pitcher_iso >= 0.180 and pitch_pct >= 15 and not batter_has_data:
                parts.append(f"✅ {pitch_type} ({pitch_pct:.0f}%) — Pitcher allows {pitcher_iso:.3f} ISO on this pitch")
        elif batter_has_data:
            if batter_iso >= 0.150 and pitch_pct >= 15:
                parts.append(f"✅ ISO {batter_iso:.3f} vs {pitch_type} ({pitch_pct:.0f}% usage)")
        elif pitcher_has_data:
            if pitcher_iso >= 0.180 and pitch_pct >= 15:
                parts.append(f"✅ {pitch_type} ({pitch_pct:.0f}%) — Pitcher allows {pitcher_iso:.3f} ISO")

        if batter_has_data and batter_iso < 0.100 and pitch_pct >= 15:
            pitcher_iso_factor = max(0.0, 1.0 - (pitcher_iso / 0.150)) if pitcher_has_data else 1.0
            penalty_amount     = round((0.100 - batter_iso) * (pitch_pct / 100) * 10 * pitcher_iso_factor, 3)
            pitch_penalty      = max(pitch_penalty, penalty_amount)
            if penalty_amount > 0.02:
                weakness_note = (f" (but pitcher allows {pitcher_iso:.3f} ISO on it)" if pitcher_has_data and pitcher_iso >= 0.100 else "")
                parts.append(f"⚠️ Weak vs {pitch_type} — ISO {batter_iso:.3f} ({pitch_pct:.0f}% usage){weakness_note}")

        if parts:
            descriptions.extend(parts)

    return sum(scores), " + ".join(descriptions), pitch_penalty


def compute_platoon_score(row: pd.Series) -> tuple:
    batter_hand = str(row.get("batter_hand", "")).strip().upper()
    p_throws    = str(row.get("pitcher_hand", "")).strip().upper()

    score   = 0.0
    penalty = 0.0
    parts   = []

    iso_vs_lhp = safe_float(row.get("vs_lhp_iso", 0))
    iso_vs_rhp = safe_float(row.get("vs_rhp_iso", 0))

    if p_throws == "L":
        iso_vs_this      = iso_vs_lhp
        iso_vs_opp       = iso_vs_rhp
        batter_barrel_vs = safe_float(row.get("vs_lhp_barrel_pct", 0))
        batter_hr_vs     = safe_float(row.get("vs_lhp_hr_rate", 0))
        matchup_label    = f"{batter_hand}HH vs LHP"
        start_rate       = safe_float(row.get("lhp_start_rate", 1.0), 1.0)
    elif p_throws == "R":
        iso_vs_this      = iso_vs_rhp
        iso_vs_opp       = iso_vs_lhp
        batter_barrel_vs = safe_float(row.get("vs_rhp_barrel_pct", 0))
        batter_hr_vs     = safe_float(row.get("vs_rhp_hr_rate", 0))
        matchup_label    = f"{batter_hand}HH vs RHP"
        start_rate       = safe_float(row.get("rhp_start_rate", 1.0), 1.0)
    else:
        return 0.0, "", 0.0

    if batter_hand == "R":
        pitcher_barrel_vs   = safe_float(row.get("pitcher_vs_rhh_barrel_pct", 0))
        pitcher_hr_vs       = safe_float(row.get("pitcher_vs_rhh_hr_rate", 0))
        pitcher_hr9_vs      = safe_float(row.get("pitcher_vs_rhh_hr9", 0))
        pitcher_split_label = "vs RHH"
    elif batter_hand == "L":
        pitcher_barrel_vs   = safe_float(row.get("pitcher_vs_lhh_barrel_pct", 0))
        pitcher_hr_vs       = safe_float(row.get("pitcher_vs_lhh_hr_rate", 0))
        pitcher_hr9_vs      = safe_float(row.get("pitcher_vs_lhh_hr9", 0))
        pitcher_split_label = "vs LHH"
    else:
        pitcher_barrel_vs = pitcher_hr_vs = pitcher_hr9_vs = 0.0
        pitcher_split_label = ""

    iso_gap      = iso_vs_opp - iso_vs_this
    has_iso_data = (iso_vs_this > 0 or iso_vs_opp > 0)

    if has_iso_data:
        if iso_gap > 0:
            penalty = round(min(iso_gap * 15.0, 4.5), 3)
            if iso_gap >= 0.150:
                parts.append(f"🚨 Severe platoon weakness ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite")
            elif iso_gap >= 0.100:
                parts.append(f"❌ Platoon disadvantage ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite")
            elif iso_gap >= 0.060:
                parts.append(f"⚠️ Moderate platoon weakness ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand")
        elif iso_gap < 0:
            score += round(min(abs(iso_gap) * 10.0, 3.0), 3)
            parts.append(f"✅ Platoon advantage ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite")
        else:
            parts.append(f"↔️ Neutral platoon ({matchup_label})")

    if start_rate < 0.50:
        start_penalty = round((0.50 - start_rate) * 4.0, 3)
        penalty      += start_penalty
        parts.append(f"⚠️ Rarely starts vs this hand ({start_rate:.0%} start rate)")

    score += pitcher_barrel_vs * 0.06
    score += pitcher_hr_vs     * 0.04
    score += pitcher_hr9_vs    * 0.08

    if pitcher_split_label and pitcher_barrel_vs >= 8:
        parts.append(f"Pitcher {pitcher_barrel_vs:.1f}% barrel allowed {pitcher_split_label}")
    if pitcher_split_label and pitcher_hr9_vs >= 1.5:
        parts.append(f"Pitcher {pitcher_hr9_vs:.2f} HR/9 {pitcher_split_label}")

    score += batter_barrel_vs * 0.04
    score += batter_hr_vs     * 0.03

    return round(score, 3), " | ".join(parts), round(penalty, 3)


def assign_confidence(row: pd.Series) -> str:
    batter_bbe        = safe_float(row.get("season_bbe", 0))
    batter_pa         = safe_float(row.get("pa", 0))
    pitcher_bbe       = safe_float(row.get("pitcher_bbe_allowed", 0))
    small_sample_park = str(row.get("small_sample", "False")).lower() == "true"
    weather_available = str(row.get("wind_context", "")).strip() not in ("", "Unknown", "Weather unavailable")

    points        = 0
    batter_points = 0

    if batter_bbe >= 60 and batter_pa >= 100:
        points += 2
        batter_points = 2
    elif batter_bbe >= 30 and batter_pa >= 50:
        points += 1
        batter_points = 1

    if pitcher_bbe >= 80:   points += 2
    elif pitcher_bbe >= 40: points += 1

    if not small_sample_park: points += 1
    if weather_available:     points += 1

    if points >= 5 and batter_points >= 1:   return "High"
    elif points >= 3 and batter_points >= 1: return "Medium"
    else:                                     return "Low"


def prepare_combined(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
    weather: pd.DataFrame,
    bvp: pd.DataFrame,
    lineups: pd.DataFrame,
    active_roster: pd.DataFrame = None,
) -> pd.DataFrame:
    if batters.empty or pitchers.empty:
        print("Missing batter or pitcher data.")
        return pd.DataFrame()

    pitchers = pitchers.copy()
    pitchers.columns = [c.strip() for c in pitchers.columns]

    pitcher_rename = {
        "pitcher_name":              "opp_pitcher_name",
        "pitcher_team":              "opp_pitcher_team",
        "opposing_team":             "batter_team",
        "home_team":                 "home_team",
        "season_barrel_pct_allowed": "pitcher_barrel_pct",
        "hr_per_fb_allowed":         "pitcher_hr_per_fb",
        "hard_hit_pct_allowed":      "pitcher_hard_hit_pct",
        "avg_ev_allowed":            "pitcher_avg_ev",
        "bf":                        "pitcher_bf",
        "season_bbe_allowed":        "pitcher_bbe_allowed",
        "pitch_pct_fastball":        "pitcher_pct_fastball",
        "pitch_pct_breaking":        "pitcher_pct_breaking",
        "pitch_pct_offspeed":        "pitcher_pct_offspeed",
        "pitch_pct_knuckleball":     "pitcher_pct_knuckleball",
        "vs_lhh_barrel_pct":         "pitcher_vs_lhh_barrel_pct",
        "vs_rhh_barrel_pct":         "pitcher_vs_rhh_barrel_pct",
        "vs_lhh_hr_rate":            "pitcher_vs_lhh_hr_rate",
        "vs_rhh_hr_rate":            "pitcher_vs_rhh_hr_rate",
        "vs_lhh_hr9":                "pitcher_vs_lhh_hr9",
        "vs_rhh_hr9":                "pitcher_vs_rhh_hr9",
        "pitcher_hand":              "pitcher_hand",
        "pitcher_id":                "pitcher_id_num",
        "top_pitch_1":               "top_pitch_1",
        "top_pitch_1_pct":           "top_pitch_1_pct",
        "top_pitch_2":               "top_pitch_2",
        "top_pitch_2_pct":           "top_pitch_2_pct",
        "top_pitch_3":               "top_pitch_3",
        "top_pitch_3_pct":           "top_pitch_3_pct",
    }
    pitchers = pitchers.rename(columns={k: v for k, v in pitcher_rename.items() if k in pitchers.columns})

    if "pitcher_hand" not in pitchers.columns: pitchers["pitcher_hand"] = ""
    if "home_team" not in pitchers.columns:    pitchers["home_team"] = pitchers.get("opp_pitcher_team", "")

    parks   = parks.copy()
    parks.columns = [c.strip() for c in parks.columns]
    parks   = parks.rename(columns={"team": "park_home_team"})

    weather = weather.copy()
    weather.columns = [c.strip() for c in weather.columns]
    weather = weather.rename(columns={"home_team": "weather_home_team"})

    batters = batters.copy()
    batters.columns = [c.strip() for c in batters.columns]
    batters = batters.rename(columns={"team": "batter_team"})

    # ── IL filter ─────────────────────────────────────────────────────────
    if active_roster is not None and not active_roster.empty and "player_id" in active_roster.columns:
        active_ids    = set(active_roster["player_id"].astype(str).str.strip())
        batter_id_col = "batter_id" if "batter_id" in batters.columns else "batter"
        if batter_id_col in batters.columns:
            before = len(batters)
            batters["_active_id_str"] = batters[batter_id_col].astype(str).str.strip()
            batters = batters[batters["_active_id_str"].isin(active_ids)].copy()
            batters = batters.drop(columns=["_active_id_str"])
            print(f"IL filter: {before - len(batters)} players removed, {len(batters)} remaining")
    else:
        print("Active roster unavailable — IL filter skipped.")

    # ── Lineup filter ──────────────────────────────────────────────────────
    if not lineups.empty and "player_id" in lineups.columns and "team" in lineups.columns:
        confirmed_teams = set(lineups["team"].unique())
        lineup_ids      = set(lineups["player_id"].astype(str).str.strip())
        batter_id_col   = "batter_id" if "batter_id" in batters.columns else "batter"

        if batter_id_col in batters.columns:
            before = len(batters)
            batters["_batter_id_str"] = batters[batter_id_col].astype(str).str.strip()

            def lineup_filter(row):
                team = str(row.get("batter_team", "")).strip()
                if team not in confirmed_teams:
                    return True
                return row["_batter_id_str"] in lineup_ids

            batters = batters[batters.apply(lineup_filter, axis=1)].copy()
            batters = batters.drop(columns=["_batter_id_str"])
            print(f"Lineup filter: {before - len(batters)} batters removed, {len(batters)} remaining")
    else:
        print("No confirmed lineups available — lineup filter skipped.")

    dynamic_pitch_cols = [
        c for c in pitchers.columns
        if c.startswith("pitcher_iso_allowed_")
        or c.startswith("pitcher_hr_rate_allowed_")
        or c.startswith("pitcher_barrel_pct_allowed_")
    ]

    pitcher_join_cols = [c for c in [
        "batter_team", "home_team", "opp_pitcher_name", "opp_pitcher_team",
        "pitcher_id_num", "pitcher_barrel_pct", "pitcher_hr_per_fb",
        "pitcher_hard_hit_pct", "pitcher_avg_ev", "pitcher_bf",
        "pitcher_bbe_allowed", "pitcher_pct_fastball", "pitcher_pct_breaking",
        "pitcher_pct_offspeed", "pitcher_pct_knuckleball",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9", "pitcher_hand",
        "top_pitch_1", "top_pitch_1_pct", "top_pitch_2", "top_pitch_2_pct",
        "top_pitch_3", "top_pitch_3_pct",
    ] if c in pitchers.columns] + dynamic_pitch_cols

    combined = batters.merge(pitchers[pitcher_join_cols], on="batter_team", how="inner")

    if combined.empty:
        print("No batter-pitcher matchups found.")
        return pd.DataFrame()

    # ── BvP merge ──────────────────────────────────────────────────────────
    if not bvp.empty and "pitcher_id_num" in combined.columns:
        bvp = bvp.copy()
        bvp.columns = [c.strip() for c in bvp.columns]
        batter_id_col = "batter_id" if "batter_id" in combined.columns else "batter"
        if batter_id_col in combined.columns and "pitcher_id" in bvp.columns:
            combined["_batter_id_merge"]  = combined[batter_id_col].apply(lambda x: safe_float(x, 0))
            combined["_pitcher_id_merge"] = combined["pitcher_id_num"].apply(lambda x: safe_float(x, 0))
            bvp["_batter_id_merge"]       = bvp["batter_id"].apply(lambda x: safe_float(x, 0))
            bvp["_pitcher_id_merge"]      = bvp["pitcher_id"].apply(lambda x: safe_float(x, 0))
            bvp_cols = [c for c in ["_batter_id_merge", "_pitcher_id_merge", "bvp_pa", "bvp_hr", "bvp_iso", "bvp_barrel_pct", "bvp_hr_rate"] if c in bvp.columns]
            combined = combined.merge(bvp[bvp_cols], on=["_batter_id_merge", "_pitcher_id_merge"], how="left")
            combined = combined.drop(columns=["_batter_id_merge", "_pitcher_id_merge"])
            print(f"BvP data matched for {combined['bvp_pa'].notna().sum()} batter-pitcher pairs.")
    else:
        print("No BvP data to merge.")

    # ── Park merge ─────────────────────────────────────────────────────────
    if not parks.empty:
        park_cols = [c for c in ["park_home_team", "park_hr_factor", "park_name", "small_sample", "lf_dist", "lf_height", "rf_dist", "rf_height", "pull_boost_rhh", "pull_boost_lhh"] if c in parks.columns]
        combined  = combined.merge(parks[park_cols], left_on="home_team", right_on="park_home_team", how="left")
    else:
        for col, val in [("park_hr_factor", 100.0), ("park_name", ""), ("small_sample", False), ("lf_dist", 331.0), ("lf_height", 9.0), ("rf_dist", 327.0), ("rf_height", 9.0), ("pull_boost_rhh", 0.0), ("pull_boost_lhh", 0.0)]:
            combined[col] = val

    # ── Weather merge ──────────────────────────────────────────────────────
    if not weather.empty:
        combined = combined.merge(weather[["weather_home_team", "hr_weather_boost", "wind_context", "temp_f"]], left_on="home_team", right_on="weather_home_team", how="left")
        combined["hr_weather_boost"] = combined["hr_weather_boost"].fillna(0.0).apply(safe_float)
        combined["wind_context"]     = combined["wind_context"].fillna("Unknown")
        combined["temp_f"]           = combined["temp_f"].apply(lambda x: 72.0 if str(x).strip() in ("", "0", "0.0") else safe_float(x, 72.0))
    else:
        combined["hr_weather_boost"] = 0.0
        combined["wind_context"]     = ""
        combined["temp_f"]           = 72.0

    # ── Batting average filter ─────────────────────────────────────────────
    if "batting_avg" in combined.columns:
        combined["batting_avg"] = combined["batting_avg"].apply(safe_float)
        before   = len(combined)
        combined = combined[combined["batting_avg"] >= MIN_BATTING_AVG].copy()
        print(f"BA filter: {before - len(combined)} batters removed, {len(combined)} remaining")

    if combined.empty:
        print("No batters remaining after filters.")
        return pd.DataFrame()

    # ── Coerce numeric columns ─────────────────────────────────────────────
    score_cols = [
        "barrel_pct_7d", "season_barrel_pct", "hr_per_pa", "hr_per_fb", "iso",
        "avg_ev_7d", "hard_hit_pct_7d", "avg_launch_angle", "avg_la_7d",
        "avg_ev_5d", "avg_ev_10d", "barrel_pct_5d", "barrel_pct_10d",
        "hard_hit_pct_5d", "hard_hit_pct_10d", "bbe_5d", "bbe_7d", "bbe_10d",
        "pull_rate", "season_bbe", "pa",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_bf", "pitcher_bbe_allowed", "park_hr_factor",
        "lf_dist", "lf_height", "rf_dist", "rf_height",
        "pull_boost_rhh", "pull_boost_lhh",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "vs_lhp_hr_rate", "vs_rhp_hr_rate", "vs_lhp_iso", "vs_rhp_iso",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
        "top_pitch_1_pct", "top_pitch_2_pct", "top_pitch_3_pct",
        "bvp_pa", "bvp_hr", "bvp_iso", "bvp_barrel_pct", "bvp_hr_rate",
        "hr_7d", "season_hr", "season_fb", "season_hard_hit",
        "lhp_start_rate", "rhp_start_rate", "hard_hit_pct_5d", "hard_hit_pct_10d",
        "avg_la_5d", "avg_la_10d",
        "avg_14d", "pa_14d",
        "hot_streak", "cold_streak",
    ]

    dynamic_cols = [c for c in combined.columns if c.startswith("pitcher_iso_allowed_") or c.startswith("pitcher_hr_rate_allowed_") or c.startswith("pitcher_barrel_pct_allowed_")]
    score_cols  += dynamic_cols

    for col in score_cols:
        if col in combined.columns: combined[col] = combined[col].apply(safe_float)
        else:                       combined[col] = 0.0

    combined["park_hr_factor_norm"] = combined["park_hr_factor"] - 100

    # ── Component scores ───────────────────────────────────────────────────
    combined = combined.copy()

    platoon_results = combined.apply(compute_platoon_score, axis=1)
    combined["platoon_score"]   = platoon_results.apply(lambda x: x[0])
    combined["platoon_desc"]    = platoon_results.apply(lambda x: x[1])
    combined["platoon_penalty"] = platoon_results.apply(lambda x: x[2])

    pitch_results = combined.apply(compute_pitch_matchup_score, axis=1)
    combined["pitch_matchup_score"] = pitch_results.apply(lambda x: x[0])
    combined["pitch_matchup_desc"]  = pitch_results.apply(lambda x: x[1])
    combined["pitch_penalty"]       = pitch_results.apply(lambda x: x[2])

    combined["total_penalty"] = combined[["platoon_penalty", "pitch_penalty"]].max(axis=1)

    pull_results = combined.apply(compute_pull_park_score, axis=1)
    combined["pull_park_score"] = pull_results.apply(lambda x: x[0])
    combined["pull_park_desc"]  = pull_results.apply(lambda x: x[1])

    momentum_results = combined.apply(compute_momentum_score, axis=1)
    combined["momentum_score"] = momentum_results.apply(lambda x: x[0])
    combined["momentum_desc"]  = momentum_results.apply(lambda x: x[1])

    bvp_results = combined.apply(compute_bvp_score, axis=1)
    combined["bvp_score"] = bvp_results.apply(lambda x: x[0])
    combined["bvp_desc"]  = bvp_results.apply(lambda x: x[1])

    combined["weather_score"] = combined["hr_weather_boost"].clip(-2, 2) / 2
    combined["confidence"]    = combined.apply(assign_confidence, axis=1)

    # ── Cap context scores ─────────────────────────────────────────────────
    combined["platoon_score_capped"]       = combined["platoon_score"].clip(-2.0, 2.0)
    combined["pitch_matchup_score_capped"] = combined["pitch_matchup_score"].clip(0.0, 1.5)
    combined["pull_park_score_capped"]     = combined["pull_park_score"].clip(-1.0, 1.0)
    combined["momentum_score_capped"]      = combined["momentum_score"].clip(-1.0, 1.0)
    combined["bvp_score_capped"]           = combined["bvp_score"].clip(-0.5, 1.5)

    # ── Absolute scoring with regression ──────────────────────────────────
    combined["score"] = (
        combined.apply(lambda r: score_barrel_pct_7d(safe_float(r["barrel_pct_7d"]), safe_float(r.get("bbe_7d", 0))), axis=1) +
        combined.apply(lambda r: score_season_barrel_pct(safe_float(r["season_barrel_pct"]), safe_float(r["pa"])), axis=1) +
        combined.apply(lambda r: score_iso(safe_float(r["iso"]), safe_float(r["pa"])), axis=1) +
        combined.apply(lambda r: score_hr_per_pa(safe_float(r["hr_per_pa"]), safe_float(r["pa"])), axis=1) +
        combined.apply(lambda r: score_hr_per_fb(safe_float(r["hr_per_fb"]), safe_float(r["pa"])), axis=1) +
        combined.apply(lambda r: score_avg_ev_7d(safe_float(r["avg_ev_7d"]), safe_float(r.get("bbe_7d", 0))), axis=1) +
        combined.apply(lambda r: score_hard_hit_pct_7d(safe_float(r["hard_hit_pct_7d"]), safe_float(r.get("bbe_7d", 0))), axis=1) +
        combined["pitcher_barrel_pct"].apply(score_pitcher_barrel_pct) +
        combined["pitcher_hr_per_fb"].apply(score_pitcher_hr_per_fb) +
        combined["pitcher_hard_hit_pct"].apply(score_pitcher_hard_hit_pct) +
        combined["park_hr_factor_norm"].apply(score_park_factor) +
        combined["weather_score"]                * WEATHER_WEIGHT +
        combined["platoon_score_capped"]         * PLATOON_BONUS_WEIGHT +
        combined["pitch_matchup_score_capped"]   * PITCH_MATCHUP_WEIGHT +
        combined["pull_park_score_capped"]       * PULL_PARK_WEIGHT +
        combined["momentum_score_capped"]        * MOMENTUM_WEIGHT +
        combined["bvp_score_capped"]             * BVP_WEIGHT -
        combined.apply(lambda r: score_pitcher_quality_penalty(
            safe_float(r.get("pitcher_barrel_pct")),
            safe_float(r.get("pitcher_hard_hit_pct")),
            safe_float(r.get("pitcher_hr_per_fb")),
            safe_float(r.get("pitcher_bbe_allowed")),
        ), axis=1) -
        combined["total_penalty"]
    )

    combined["score"] = combined["score"].round(3)
    return combined


def build_reason(row) -> str:
    reasons = []

    barrel_7d = safe_float(row.get("barrel_pct_7d"))
    if barrel_7d >= 15:
        reasons.append(f"🔥 Hot — {barrel_7d:.1f}% barrel rate last 7 days")

    season_barrel = safe_float(row.get("season_barrel_pct"))
    if season_barrel >= 11:
        reasons.append(f"💣 Elite season barrel% ({season_barrel:.1f}%)")

    momentum_desc = str(row.get("momentum_desc", ""))
    if momentum_desc:
        reasons.append(momentum_desc)

    bvp_desc = str(row.get("bvp_desc", ""))
    if bvp_desc:
        reasons.append(bvp_desc)

    hr_pa = safe_float(row.get("hr_per_pa"))
    if hr_pa >= 4:
        reasons.append(f"💪 Strong HR rate ({hr_pa:.1f}% HR/PA)")

    iso = safe_float(row.get("iso"))
    if iso >= 0.200:
        reasons.append(f"⚡ Elite ISO ({iso:.3f})")

    p_barrel = safe_float(row.get("pitcher_barrel_pct"))
    if p_barrel >= 11:
        reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")
    elif p_barrel <= 4:
        reasons.append(f"⚠️ Tough pitcher — only {p_barrel:.1f}% barrels allowed")

    p_hr_fb = safe_float(row.get("pitcher_hr_per_fb"))
    if p_hr_fb >= 15:
        reasons.append(f"🚀 Pitcher HR/FB% is {p_hr_fb:.1f}%")

    platoon_desc = str(row.get("platoon_desc", ""))
    if platoon_desc:
        reasons.append(f"🔄 {platoon_desc}")

    pitch_desc = str(row.get("pitch_matchup_desc", ""))
    if pitch_desc:
        reasons.append(f"🎳 Pitch matchup: {pitch_desc}")

    pull_desc = str(row.get("pull_park_desc", ""))
    if pull_desc:
        reasons.append(pull_desc)

    park_factor = safe_float(row.get("park_hr_factor"), 100)
    park_name   = str(row.get("park_name", ""))
    if park_factor >= 110:
        reasons.append(f"🏟️ HR-friendly park ({park_name}, factor {park_factor:.0f})")
    elif park_factor <= 80:
        reasons.append(f"🏟️ Pitcher-friendly park ({park_name}, factor {park_factor:.0f})")

    boost    = safe_float(row.get("hr_weather_boost"))
    wind_ctx = str(row.get("wind_context", ""))
    temp     = safe_float(row.get("temp_f"), 72)
    if boost >= 1.0:
        reasons.append(f"🌬️ Favorable weather — {wind_ctx}")
    elif boost <= -1.0:
        reasons.append(f"🌬️ Tough weather — {wind_ctx}")
    if temp >= 85:
        reasons.append(f"🌡️ Hot weather ({temp:.0f}°F helps ball carry)")
    elif temp <= 50:
        reasons.append(f"🌡️ Cold weather ({temp:.0f}°F suppresses HRs)")

    if not reasons:
        reasons.append("Solid across multiple factors")

    return " | ".join(reasons)


def odds_qualifies(odds: int, score: float = 0.0) -> bool:
    if odds <= MAX_CHALK_ODDS:
        return True
    if odds >= MIN_VALUE_ODDS:
        return True
    if 300 < odds < 500 and score >= MID_RANGE_SCORE_FLOOR:
        return True
    return False


def build_main_picks(combined: pd.DataFrame, odds_df: pd.DataFrame = None) -> tuple:
    combined = combined.copy()
    combined["reason"] = combined.apply(build_reason, axis=1)

    odds_lookup = {}
    if odds_df is not None and not odds_df.empty and "player_name_norm" in odds_df.columns and "consensus_odds" in odds_df.columns:
        for _, row in odds_df.iterrows():
            norm = str(row["player_name_norm"]).strip()
            try:
                odds_lookup[norm] = int(float(row["consensus_odds"]))
            except (ValueError, TypeError):
                pass
        print(f"Odds lookup built: {len(odds_lookup)} players with odds data")
    else:
        print("No odds data available")

    combined["consensus_odds"] = combined["player_name"].apply(
        lambda n: odds_lookup.get(normalize_name(str(n)), None)
    )

    filtered = combined[combined["score"] >= MIN_SCORE_FLOOR].copy()

    if not filtered.empty and odds_lookup:
        filtered = filtered[filtered.apply(
            lambda row: row.get("consensus_odds") is not None and
            str(row.get("consensus_odds", "")).strip() not in ("", "nan", "None") and
            odds_qualifies(int(float(row["consensus_odds"])), safe_float(row["score"])),
            axis=1
        )].copy()

    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    selected    = []
    team_counts = {}
    game_counts = {}

    for _, row in filtered.iterrows():
        team      = str(row.get("batter_team", ""))
        home_team = str(row.get("home_team", "UNK"))

        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue
        if game_counts.get(home_team, 0) >= MAX_PER_GAME:
            continue

        selected.append(row)
        team_counts[team]      = team_counts.get(team, 0) + 1
        game_counts[home_team] = game_counts.get(home_team, 0) + 1

    if not selected:
        picks = pd.DataFrame()
        print("No qualifying picks today (score ≥11, odds ≤+300 or ≥+500).")
        return picks, odds_lookup

    picks = pd.DataFrame(selected).reset_index(drop=True)
    picks["rank"] = range(1, len(picks) + 1)

    chalk_count   = sum(1 for _, r in picks.iterrows() if r.get("consensus_odds") is not None and int(r["consensus_odds"]) <= MAX_CHALK_ODDS)
    mid_count     = sum(1 for _, r in picks.iterrows() if r.get("consensus_odds") is not None and 300 < int(r["consensus_odds"]) < 500)
    value_count   = sum(1 for _, r in picks.iterrows() if r.get("consensus_odds") is not None and int(r["consensus_odds"]) >= MIN_VALUE_ODDS)
    print(f"Qualifying picks: {len(picks)} total ({chalk_count} chalk ≤+{MAX_CHALK_ODDS}, {mid_count} mid +301-499, {value_count} value ≥+{MIN_VALUE_ODDS})")

    output_cols = {
        "rank":                       "Rank",
        "player_name":                "Batter",
        "batter_hand":                "Bats",
        "batter_team":                "Team",
        "opp_pitcher_name":           "Opposing Pitcher",
        "pitcher_hand":               "Throws",
        "opp_pitcher_team":           "Pitcher Team",
        "park_name":                  "Park",
        "score":                      "HR Score",
        "consensus_odds":             "Consensus Odds",
        "confidence":                 "Confidence",
        "reason":                     "Key Reasons",
        "batting_avg":                "Batting Avg",
        "barrel_pct_7d":              "Barrel% (7d)",
        "season_barrel_pct":          "Barrel% (Season)",
        "barrel_pct_5d":              "Barrel% (5d)",
        "barrel_pct_10d":             "Barrel% (10d)",
        "momentum_desc":              "Momentum",
        "hr_per_pa":                  "HR/PA%",
        "hr_per_fb":                  "HR/FB%",
        "iso":                        "ISO",
        "vs_lhp_iso":                 "ISO vs LHP",
        "vs_rhp_iso":                 "ISO vs RHP",
        "avg_ev_7d":                  "Avg EV (7d)",
        "avg_ev_5d":                  "Avg EV (5d)",
        "avg_ev_10d":                 "Avg EV (10d)",
        "avg_la_7d":                  "Avg Launch Angle (7d)",
        "avg_launch_angle":           "Avg Launch Angle (Season)",
        "pull_rate":                  "Pull Rate%",
        "pull_park_desc":             "Pull Park Matchup",
        "bvp_pa":                     "BvP PA",
        "bvp_hr":                     "BvP HR",
        "bvp_iso":                    "BvP ISO",
        "bvp_desc":                   "BvP Notes",
        "platoon_desc":               "Platoon Matchup",
        "lhp_start_rate":             "LHP Start Rate",
        "rhp_start_rate":             "RHP Start Rate",
        "pitcher_barrel_pct":         "Pitcher Barrel% Allowed",
        "pitcher_hr_per_fb":          "Pitcher HR/FB% Allowed",
        "pitcher_vs_lhh_barrel_pct":  "Pitcher Barrel% vs LHH",
        "pitcher_vs_rhh_barrel_pct":  "Pitcher Barrel% vs RHH",
        "pitcher_vs_lhh_hr9":         "Pitcher HR/9 vs LHH",
        "pitcher_vs_rhh_hr9":         "Pitcher HR/9 vs RHH",
        "top_pitch_1":                "Top Pitch 1",
        "top_pitch_1_pct":            "Top Pitch 1 %",
        "top_pitch_2":                "Top Pitch 2",
        "top_pitch_2_pct":            "Top Pitch 2 %",
        "top_pitch_3":                "Top Pitch 3",
        "top_pitch_3_pct":            "Top Pitch 3 %",
        "pitch_matchup_desc":         "Pitch Matchup",
        "park_hr_factor":             "Park HR Factor",
        "lf_dist":                    "LF Distance",
        "lf_height":                  "LF Wall Height",
        "rf_dist":                    "RF Distance",
        "rf_height":                  "RF Wall Height",
        "hr_weather_boost":           "Weather Boost",
        "wind_context":               "Wind",
        "temp_f":                     "Temp (°F)",
    }

    available = {k: v for k, v in output_cols.items() if k in picks.columns}
    return picks[list(available.keys())].rename(columns=available), odds_lookup


def resolve_pending_picks(gc: gspread.Client, sheet_id: str) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws       = sh.worksheet("Picks_Log")
        existing = pd.DataFrame(ws.get_all_records())
    except gspread.WorksheetNotFound:
        print("Picks_Log not found — skipping resolution.")
        return

    if existing.empty:
        print("Picks_Log is empty — nothing to resolve.")
        return

    today_str = date.today().strftime("%Y-%m-%d")
    pending   = existing[(existing["hit_hr"] == "Pending") & (existing["date"] != today_str) & (existing["date"] != "")].copy()

    if pending.empty:
        print("No pending picks to resolve.")
        return

    pending_dates = sorted(pending["date"].unique())
    print(f"Resolving {len(pending)} pending picks across {len(pending_dates)} dates...")

    try:
        hr_df = statcast(start_dt=pending_dates[0], end_dt=pending_dates[-1])
        if hr_df is None or hr_df.empty:
            print("Statcast returned empty — cannot resolve pending picks.")
            return
        print(f"Pulled {len(hr_df):,} Statcast rows for resolution.")
    except Exception as e:
        print(f"Statcast pull failed: {e}")
        return

    hr_df["game_date"] = pd.to_datetime(hr_df["game_date"])
    hr_df["is_hr"]     = hr_df["events"].astype("string").str.lower().eq("home_run")
    hr_events          = hr_df[hr_df["is_hr"]].copy()

    hr_batter_ids = hr_events["batter"].dropna().astype(int).unique().tolist()
    name_map      = {}
    for i in range(0, len(hr_batter_ids), 50):
        chunk = hr_batter_ids[i:i + 50]
        try:
            resp = requests.get(f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}", timeout=30)
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pid  = person.get("id")
                name = person.get("fullName", "")
                if pid and name:
                    name_map[int(pid)] = name
        except Exception:
            pass

    hr_events = hr_events.copy()
    hr_events["player_name"] = hr_events["batter"].map(lambda x: name_map.get(int(x), "") if pd.notna(x) else "")
    hr_events["date_str"]    = hr_events["game_date"].dt.strftime("%Y-%m-%d")
    hr_lookup = set(zip(hr_events["date_str"], hr_events["player_name"].str.lower().str.strip()))

    resolved_count = 0
    for idx, row in existing.iterrows():
        if row["hit_hr"] != "Pending" or row["date"] == today_str:
            continue
        pick_date = str(row["date"]).strip()
        pick_name = str(row["player_name"]).lower().strip()
        if not pick_date or not pick_name:
            existing.at[idx, "hit_hr"] = "No"
        else:
            existing.at[idx, "hit_hr"] = "Yes" if (pick_date, pick_name) in hr_lookup else "No"
        resolved_count += 1

    print(f"Resolved {resolved_count} picks. Yes: {(existing['hit_hr'] == 'Yes').sum()} | No: {(existing['hit_hr'] == 'No').sum()}")
    ws.clear()
    ws.update([existing.columns.tolist()] + existing.astype(str).values.tolist())
    print("Picks_Log updated with resolved outcomes.")


def log_todays_picks(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame) -> None:
    today_str = date.today().strftime("%Y-%m-%d")
    sh        = gc.open_by_key(sheet_id)

    try:
        ws       = sh.worksheet("Picks_Log")
        existing = pd.DataFrame(ws.get_all_records())
    except gspread.WorksheetNotFound:
        ws       = sh.add_worksheet(title="Picks_Log", rows=5000, cols=60)
        existing = pd.DataFrame()

    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"] != today_str].copy()

    new_rows = []

    if not picks.empty:
        for _, row in picks.iterrows():
            new_rows.append({
                "date":                      today_str,
                "rank":                      str(row.get("Rank", "")),
                "player_name":               str(row.get("Batter", "")),
                "team":                      str(row.get("Team", "")),
                "pitcher_name":              str(row.get("Opposing Pitcher", "")),
                "pitcher_hand":              str(row.get("Throws", "")),
                "pitcher_team":              str(row.get("Pitcher Team", "")),
                "park_name":                 str(row.get("Park", "")),
                "hr_score":                  str(row.get("HR Score", "")),
                "consensus_odds":            str(row.get("Consensus Odds", "")),
                "confidence":                str(row.get("Confidence", "")),
                "hit_hr":                    "Pending",
                "section":                   "Main",
                "odds":                      "",
                "bet_placed":                "",
                "batting_avg":               str(row.get("Batting Avg", "")),
                "barrel_pct_7d":             str(row.get("Barrel% (7d)", "")),
                "season_barrel_pct":         str(row.get("Barrel% (Season)", "")),
                "barrel_pct_5d":             str(row.get("Barrel% (5d)", "")),
                "barrel_pct_10d":            str(row.get("Barrel% (10d)", "")),
                "hr_per_pa":                 str(row.get("HR/PA%", "")),
                "hr_per_fb":                 str(row.get("HR/FB%", "")),
                "iso":                       str(row.get("ISO", "")),
                "iso_vs_lhp":                str(row.get("ISO vs LHP", "")),
                "iso_vs_rhp":                str(row.get("ISO vs RHP", "")),
                "avg_ev_7d":                 str(row.get("Avg EV (7d)", "")),
                "avg_ev_5d":                 str(row.get("Avg EV (5d)", "")),
                "avg_ev_10d":                str(row.get("Avg EV (10d)", "")),
                "avg_la_7d":                 str(row.get("Avg Launch Angle (7d)", "")),
                "avg_la_season":             str(row.get("Avg Launch Angle (Season)", "")),
                "pull_rate":                 str(row.get("Pull Rate%", "")),
                "lhp_start_rate":            str(row.get("LHP Start Rate", "")),
                "rhp_start_rate":            str(row.get("RHP Start Rate", "")),
                "pitcher_barrel_pct":        str(row.get("Pitcher Barrel% Allowed", "")),
                "pitcher_hr_per_fb":         str(row.get("Pitcher HR/FB% Allowed", "")),
                "pitcher_barrel_vs_lhh":     str(row.get("Pitcher Barrel% vs LHH", "")),
                "pitcher_barrel_vs_rhh":     str(row.get("Pitcher Barrel% vs RHH", "")),
                "pitcher_hr9_vs_lhh":        str(row.get("Pitcher HR/9 vs LHH", "")),
                "pitcher_hr9_vs_rhh":        str(row.get("Pitcher HR/9 vs RHH", "")),
                "top_pitch_1":               str(row.get("Top Pitch 1", "")),
                "top_pitch_1_pct":           str(row.get("Top Pitch 1 %", "")),
                "top_pitch_2":               str(row.get("Top Pitch 2", "")),
                "top_pitch_2_pct":           str(row.get("Top Pitch 2 %", "")),
                "top_pitch_3":               str(row.get("Top Pitch 3", "")),
                "top_pitch_3_pct":           str(row.get("Top Pitch 3 %", "")),
                "park_hr_factor":            str(row.get("Park HR Factor", "")),
                "lf_dist":                   str(row.get("LF Distance", "")),
                "lf_height":                 str(row.get("LF Wall Height", "")),
                "rf_dist":                   str(row.get("RF Distance", "")),
                "rf_height":                 str(row.get("RF Wall Height", "")),
                "weather_boost":             str(row.get("Weather Boost", "")),
                "wind":                      str(row.get("Wind", "")),
                "temp_f":                    str(row.get("Temp (°F)", "")),
                "pull_park_matchup":         str(row.get("Pull Park Matchup", "")),
                "platoon_matchup":           str(row.get("Platoon Matchup", "")),
                "pitch_matchup":             str(row.get("Pitch Matchup", "")),
                "momentum":                  str(row.get("Momentum", "")),
                "bvp_pa":                    str(row.get("BvP PA", "")),
                "bvp_hr":                    str(row.get("BvP HR", "")),
                "bvp_iso":                   str(row.get("BvP ISO", "")),
                "bvp_notes":                 str(row.get("BvP Notes", "")),
            })

    if not new_rows:
        print("No qualifying picks to log today.")
        if not existing.empty:
            ws.clear()
            ws.update([existing.columns.tolist()] + existing.astype(str).values.tolist())
        return

    new_df = pd.DataFrame(new_rows)

    if not existing.empty:
        for col in new_df.columns:
            if col not in existing.columns:
                existing[col] = ""

    combined_log = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df

    preserve_cols = ["odds", "bet_placed"]
    preserved     = {col: combined_log[col].copy() for col in preserve_cols if col in combined_log.columns}
    combined_log  = combined_log.replace([np.inf, -np.inf], np.nan).fillna("")
    for col, values in preserved.items():
        combined_log[col] = values.fillna("")

    ws.clear()
    ws.update([combined_log.columns.tolist()] + combined_log.astype(str).values.tolist())
    print(f"Logged {len(new_rows)} picks for {today_str} to Picks_Log")


def update_scorecard(gc: gspread.Client, sheet_id: str) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws_log    = sh.worksheet("Picks_Log")
        picks_log = pd.DataFrame(ws_log.get_all_records())
    except gspread.WorksheetNotFound:
        print("Picks_Log not found — skipping scorecard update.")
        return

    if picks_log.empty:
        print("Picks_Log is empty — skipping scorecard update.")
        return

    scored = picks_log[picks_log["hit_hr"].isin(["Yes", "No"])].copy()
    if scored.empty:
        print("No scored picks yet — skipping scorecard update.")
        return

    scored["hit_hr_bool"] = scored["hit_hr"] == "Yes"
    scored["rank"]        = pd.to_numeric(scored["rank"], errors="coerce")
    scored["date"]        = pd.to_datetime(scored["date"], errors="coerce")
    scored["hr_score"]    = pd.to_numeric(scored["hr_score"], errors="coerce")
    if "odds" in scored.columns:
        scored["odds_num"] = pd.to_numeric(
            scored["odds"].apply(lambda x: str(x).replace("+", "").strip()),
            errors="coerce"
        )
    else:
        scored["odds_num"] = pd.Series(dtype=float, index=scored.index)

    def is_bet_placed(x) -> bool:
        s = str(x).replace("$", "").strip().lower()
        if s in ("", "no", "nan"):
            return False
        try:
            float(s)
            return True
        except ValueError:
            return s == "yes"

    bet_picks = scored[
        scored.get("bet_placed", pd.Series("", index=scored.index)).apply(is_bet_placed) &
        (scored.get("odds", pd.Series("", index=scored.index)).astype(str).str.strip() != "")
    ].copy()

    if not bet_picks.empty:
        bet_picks["odds_num_bet"]  = bet_picks["odds"].apply(lambda x: safe_float(str(x).replace("+", "").strip(), 0))
        bet_picks["profit_if_win"] = bet_picks["odds_num_bet"].apply(american_odds_to_profit)

        def parse_bet_size(x) -> float:
            s = str(x).replace("$", "").strip().lower()
            if s in ("", "yes", "no"):
                return 1.0
            return safe_float(s, 1.0)

        bet_picks["bet_size"]    = bet_picks["bet_placed"].apply(parse_bet_size)
        bet_picks["unit_result"] = bet_picks.apply(
            lambda r: (r["profit_if_win"] * r["bet_size"]) if r["hit_hr_bool"] else -r["bet_size"],
            axis=1
        )

    perf_rows  = []
    roi_rows   = []
    score_rows = []

    def add_perf(label, sub_df, bold=False):
        if sub_df.empty: return
        total = len(sub_df)
        hits  = int(sub_df["hit_hr_bool"].sum())
        perf_rows.append({"label": label, "total_picks": total, "hr_count": hits, "hit_rate_pct": round(hits / total * 100, 1), "_bold": bold})

    def add_roi(label, sub_df, bold=False):
        if sub_df.empty: return
        total        = len(sub_df)
        hits         = int(sub_df["hit_hr_bool"].sum())
        units_profit = round(sub_df["unit_result"].sum(), 2)
        roi          = round(units_profit / total * 100, 1) if total > 0 else 0.0
        roi_rows.append({"label": label, "bets_placed": total, "hr_count": hits, "hit_rate_pct": round(hits / total * 100, 1) if total > 0 else 0.0, "units_wagered": float(total), "units_profit": f"+{units_profit}" if units_profit >= 0 else str(units_profit), "roi_pct": f"+{roi}%" if roi >= 0 else f"{roi}%", "_bold": bold, "_roi_val": roi, "_profit_val": units_profit})

    def add_score(label, sub_df, bold=False):
        if sub_df.empty: return
        total     = len(sub_df)
        hits      = int(sub_df["hit_hr_bool"].sum())
        avg_score = round(sub_df["hr_score"].mean(), 2) if not sub_df["hr_score"].isna().all() else 0.0
        score_rows.append({"label": label, "total_picks": total, "hr_count": hits, "hit_rate_pct": round(hits / total * 100, 1), "avg_score": avg_score, "_bold": bold})

    add_perf("🏆  Overall", scored, bold=True)

    perf_rows.append({"label": "── By Rank ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "_bold": True, "_header": True})
    for rank in range(1, 11):
        sub = scored[scored["rank"] == rank]
        if not sub.empty: add_perf(f"   Rank {rank}", sub)

    perf_rows.append({"label": "── By Confidence ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "_bold": True, "_header": True})
    for tier in ["High", "Medium", "Low"]:
        sub = scored[scored["confidence"] == tier]
        if not sub.empty: add_perf(f"   {tier}", sub)

    perf_rows.append({"label": "── By Odds Zone ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "_bold": True, "_header": True})
    if "consensus_odds" in scored.columns:
        scored["odds_zone_num"] = pd.to_numeric(scored["consensus_odds"], errors="coerce")
        for label, sub in [
            ("   ≤ +300 (Chalk)",   scored[scored["odds_zone_num"] <= 300]),
            ("   +500 to +699",     scored[(scored["odds_zone_num"] >= 500) & (scored["odds_zone_num"] < 700)]),
            ("   +700+",            scored[scored["odds_zone_num"] >= 700]),
        ]:
            if not sub.empty: add_perf(label, sub)

    perf_rows.append({"label": "── Rolling ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "_bold": True, "_header": True})
    max_date = scored["date"].max()
    add_perf("   Last 7 Days",  scored[scored["date"] >= max_date - pd.Timedelta(days=7)])
    add_perf("   Last 30 Days", scored[scored["date"] >= max_date - pd.Timedelta(days=30)])

    if not bet_picks.empty:
        bet_picks["date"] = pd.to_datetime(bet_picks["date"], errors="coerce")
        bet_picks["rank"] = pd.to_numeric(bet_picks["rank"], errors="coerce")
        add_roi("💵  All Bets", bet_picks, bold=True)
        roi_rows.append({"label": "── By Confidence ──", "bets_placed": "", "hr_count": "", "hit_rate_pct": "", "units_wagered": "", "units_profit": "", "roi_pct": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        for tier in ["High", "Medium", "Low"]:
            sub = bet_picks[bet_picks["confidence"] == tier]
            if not sub.empty: add_roi(f"   {tier}", sub)
        roi_rows.append({"label": "── By Odds Zone ──", "bets_placed": "", "hr_count": "", "hit_rate_pct": "", "units_wagered": "", "units_profit": "", "roi_pct": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        bet_picks["odds_num_bet"] = pd.to_numeric(
            bet_picks["odds"].apply(lambda x: str(x).replace("+", "").strip()),
            errors="coerce"
        )
        for label, sub in [
            ("   ≤ +300 (Chalk)",   bet_picks[bet_picks["odds_num_bet"] <= 300]),
            ("   +500 to +699",     bet_picks[(bet_picks["odds_num_bet"] >= 500) & (bet_picks["odds_num_bet"] < 700)]),
            ("   +700+",            bet_picks[bet_picks["odds_num_bet"] >= 700]),
        ]:
            if not sub.empty: add_roi(label, sub)
        roi_rows.append({"label": "── Rolling ──", "bets_placed": "", "hr_count": "", "hit_rate_pct": "", "units_wagered": "", "units_profit": "", "roi_pct": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        max_bet = bet_picks["date"].max()
        add_roi("   Last 7 Days",  bet_picks[bet_picks["date"] >= max_bet - pd.Timedelta(days=7)])
        add_roi("   Last 30 Days", bet_picks[bet_picks["date"] >= max_bet - pd.Timedelta(days=30)])

    add_score("📈  All Scored Picks", scored, bold=True)
    score_rows.append({"label": "── By Score Tier ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "avg_score": "", "_bold": True, "_header": True})
    for label, sub in [
        ("   15+",      scored[scored["hr_score"] >= 15]),
        ("   14+",      scored[scored["hr_score"] >= 14]),
        ("   13+",      scored[scored["hr_score"] >= 13]),
        ("   12+",      scored[scored["hr_score"] >= 12]),
        ("   11+",      scored[scored["hr_score"] >= 11]),
        ("   Under 11", scored[scored["hr_score"] <  11]),
    ]:
        if not sub.empty: add_score(label, sub)

    try:
        ws = sh.worksheet("Scorecard")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Scorecard", rows=200, cols=10)

    ws_id         = ws.id
    perf_headers  = ["Category", "Total Picks", "HR Count", "Hit Rate %"]
    roi_headers   = ["Category", "Bets Placed", "HR Count", "Hit Rate %", "Units Wagered", "Units Profit", "ROI %"]
    score_headers = ["Category", "Total Picks", "HR Count", "Hit Rate %", "Avg HR Score"]

    all_values = []
    all_values.append(["📊  MODEL PERFORMANCE", "", "", ""])
    all_values.append(perf_headers)
    perf_start = 2

    for r in perf_rows:
        all_values.append([r.get("label", ""), str(r.get("total_picks", "")), str(r.get("hr_count", "")), f"{r['hit_rate_pct']}%" if r.get("hit_rate_pct", "") != "" else ""])

    all_values.append(["", "", "", ""])
    all_values.append(["", "", "", ""])
    roi_section_start = len(all_values)

    if roi_rows:
        all_values.append(["💰  BETTING ROI", "", "", "", "", "", ""])
        all_values.append(roi_headers)
        roi_data_start = len(all_values)
        for r in roi_rows:
            all_values.append([r.get("label", ""), str(r.get("bets_placed", "")), str(r.get("hr_count", "")), f"{r['hit_rate_pct']}%" if r.get("hit_rate_pct", "") != "" else "", str(r.get("units_wagered", "")), str(r.get("units_profit", "")), str(r.get("roi_pct", ""))])
    else:
        all_values.append(["💰  BETTING ROI — No bets placed yet", "", "", "", "", "", ""])
        roi_data_start = len(all_values)

    all_values.append(["", "", "", "", "", "", ""])
    all_values.append(["", "", "", "", "", "", ""])
    score_section_start = len(all_values)

    all_values.append(["📈  SCORE TIER ANALYSIS", "", "", "", ""])
    all_values.append(score_headers)
    score_data_start = len(all_values)

    for r in score_rows:
        all_values.append([r.get("label", ""), str(r.get("total_picks", "")), str(r.get("hr_count", "")), f"{r['hit_rate_pct']}%" if r.get("hit_rate_pct", "") != "" else "", str(r.get("avg_score", ""))])

    ws.update(all_values)
    print("Scorecard updated.")


def write_last_run_timestamp(gc, sheet_id: str) -> None:
    """
    No-op — timestamp is now written directly in write_picks_to_sheet as row 1.
    Kept so main() call doesn't break.
    """
    print("Timestamp already written in write_picks_to_sheet — skipping.")
    ws.insert_row([f"⏱  Last Run: {now_et}"], index=1)
    print(f"Last run timestamp written: {now_et}")


def clean_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x)
    df = df.fillna("")
    return df


def write_picks_to_sheet(gc, sheet_id: str, picks, row_count_ref: list) -> int:
    """
    Writes timestamp row 1, headers row 2, data rows 3+.
    Returns row count (data rows only, not counting timestamp or header).
    Stores sheet ws_id in row_count_ref[0] for format_picks_sheet to reuse.
    """
    import pytz
    from datetime import datetime
    import numpy as np

    et     = pytz.timezone("America/New_York")
    now_et = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("Top_HR_Picks")
        ws.clear()
    except Exception:
        ws = sh.add_worksheet(title="Top_HR_Picks", rows=100, cols=55)

    if picks.empty:
        ws.update([[f"⏱  Last Run: {now_et}"],
                   ["No qualifying picks today — score ≥9.5 with odds ≤+300 or ≥+500"]])
        return 0

    picks_clean = clean_for_sheets(picks)
    timestamp_row = [f"⏱  Last Run: {now_et}"] + [""] * (len(picks_clean.columns) - 1)
    header_row    = picks_clean.columns.tolist()
    data_rows     = picks_clean.astype(str).values.tolist()

    all_values = [timestamp_row, header_row] + data_rows
    ws.update(all_values)

    print(f"Written {len(data_rows)} picks to Top_HR_Picks (with timestamp row 1)")
    return len(data_rows)


def format_picks_sheet(gc: gspread.Client, sheet_id: str, row_count: int) -> None:
    if row_count == 0:
        return

    print("Applying Carbon dark mode formatting...")
    sh    = gc.open_by_key(sheet_id)
    ws    = sh.worksheet("Top_HR_Picks")
    ws_id = ws.id

    main_cols  = 50
    total_rows = row_count + 2
    reqs       = []

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows, "startColumnIndex": 0, "endColumnIndex": 55}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG, "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10}, "verticalAlignment": "MIDDLE", "wrapStrategy": "WRAP"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": main_cols}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_ACCENT_DIM, "textFormat": {"foregroundColor": COLOR_ACCENT, "bold": True, "fontFamily": "Roboto", "fontSize": 11}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})

    for i in range(row_count):
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2, "startColumnIndex": 0, "endColumnIndex": main_cols}, "cell": {"userEnteredFormat": {"backgroundColor": bg}}, "fields": "userEnteredFormat(backgroundColor)"}})

    medals = [(1, {"red": 0.18, "green": 0.14, "blue": 0.00}, COLOR_GOLD),
              (2, {"red": 0.14, "green": 0.14, "blue": 0.14}, COLOR_SILVER),
              (3, {"red": 0.16, "green": 0.10, "blue": 0.04}, COLOR_BRONZE)]
    for rank, bg, fg in medals:
        if row_count >= rank:
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": rank, "endRowIndex": rank + 1, "startColumnIndex": 0, "endColumnIndex": main_cols}, "cell": {"userEnteredFormat": {"backgroundColor": bg, "textFormat": {"foregroundColor": fg, "bold": True, "fontSize": 10}}}, "fields": "userEnteredFormat(backgroundColor,textFormat)"}})

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": row_count + 1, "startColumnIndex": 0, "endColumnIndex": 1}, "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 14, "foregroundColor": COLOR_ACCENT}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": row_count + 1, "startColumnIndex": 8, "endColumnIndex": 9}, "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 12, "foregroundColor": COLOR_GREEN}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})

    reqs.append({"updateSheetProperties": {"properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}}, "fields": "gridProperties.frozenRowCount"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 0, "endIndex": 1}, "properties": {"pixelSize": 36}, "fields": "pixelSize"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 1, "endIndex": total_rows}, "properties": {"pixelSize": 58}, "fields": "pixelSize"}})

    col_widths = [50, 160, 50, 55, 175, 55, 75, 175, 75, 95, 80, 380, 75, 90, 90, 75, 75, 200, 75, 75, 65, 80, 80, 85, 85, 85, 100, 100, 75, 75, 75, 220, 60, 60, 70, 200, 220, 120, 120, 130, 130, 90, 90, 90, 75, 90, 75, 90, 75, 240, 75, 65, 65, 65, 65]
    for i, width in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1}, "properties": {"pixelSize": width}, "fields": "pixelSize"}})

    reqs.append({"updateSheetProperties": {"properties": {"sheetId": ws_id, "tabColorStyle": {"rgbColor": COLOR_ACCENT}}, "fields": "tabColorStyle"}})

    try:
        sh.batch_update({"requests": reqs})
        print("Carbon formatting applied successfully!")
    except APIError as e:
        print(f"Formatting failed: {e}")


def log_all_scores(gc: gspread.Client, sheet_id: str, combined: pd.DataFrame) -> None:
    today_str = date.today().strftime("%Y-%m-%d")
    sh        = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws       = sh.worksheet("HR_All_Scores")
        existing = pd.DataFrame(ws.get_all_records())
    except gspread.WorksheetNotFound:
        ws       = sh.add_worksheet(title="HR_All_Scores", rows=10000, cols=40)
        existing = pd.DataFrame()

    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"] != today_str].copy()

    if combined.empty:
        print("No scored players to log to HR_All_Scores.")
        return

    sorted_df = combined.sort_values("score", ascending=False).reset_index(drop=True)
    sorted_df["all_scores_rank"] = range(1, len(sorted_df) + 1)

    new_rows = []
    for _, row in sorted_df.iterrows():
        new_rows.append({
            "date":                     today_str,
            "rank":                     str(row.get("all_scores_rank", "")),
            "player_name":              str(row.get("player_name", "")),
            "team":                     str(row.get("batter_team", "")),
            "pitcher_name":             str(row.get("opp_pitcher_name", "")),
            "pitcher_hand":             str(row.get("pitcher_hand", "")),
            "hr_score":                 str(row.get("score", "")),
            "consensus_odds":           str(row.get("consensus_odds", "") if "consensus_odds" in row.index else ""),
            "barrel_pct_7d":            str(row.get("barrel_pct_7d", "")),
            "season_barrel_pct":        str(row.get("season_barrel_pct", "")),
            "barrel_pct_5d":            str(row.get("barrel_pct_5d", "")),
            "barrel_pct_10d":           str(row.get("barrel_pct_10d", "")),
            "avg_ev_7d":                str(row.get("avg_ev_7d", "")),
            "avg_ev_5d":                str(row.get("avg_ev_5d", "")),
            "avg_ev_10d":               str(row.get("avg_ev_10d", "")),
            "avg_la_7d":                str(row.get("avg_la_7d", "")),
            "avg_la_season":            str(row.get("avg_launch_angle", "")),
            "iso":                      str(row.get("iso", "")),
            "hr_per_pa":                str(row.get("hr_per_pa", "")),
            "hr_per_fb":                str(row.get("hr_per_fb", "")),
            "pull_rate":                str(row.get("pull_rate", "")),
            "platoon_matchup":          str(row.get("platoon_desc", "")),
            "pitch_matchup":            str(row.get("pitch_matchup_desc", "")),
            "pitch_matchup_score":      str(row.get("pitch_matchup_score", "")),
            "pull_park_matchup":        str(row.get("pull_park_desc", "")),
            "pitcher_barrel_pct":       str(row.get("pitcher_barrel_pct", "")),
            "pitcher_hr_per_fb":        str(row.get("pitcher_hr_per_fb", "")),
            "pitcher_barrel_vs_lhh":    str(row.get("pitcher_vs_lhh_barrel_pct", "")),
            "pitcher_barrel_vs_rhh":    str(row.get("pitcher_vs_rhh_barrel_pct", "")),
            "park_hr_factor":           str(row.get("park_hr_factor", "")),
            "weather_boost":            str(row.get("hr_weather_boost", "")),
            "wind":                     str(row.get("wind_context", "")),
            "temp_f":                   str(row.get("temp_f", "")),
            "momentum":                 str(row.get("momentum_desc", "")),
            "hit_hr":                   "Pending",
        })

    if not new_rows:
        print("No rows to log to HR_All_Scores.")
        return

    new_df = pd.DataFrame(new_rows)

    if not existing.empty:
        for col in new_df.columns:
            if col not in existing.columns:
                existing[col] = ""

    combined_log = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined_log = combined_log.fillna("").replace([np.inf, -np.inf], "")

    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([combined_log.columns.tolist()] + combined_log.astype(str).values.tolist()))
    print(f"Logged {len(new_rows)} scored players to HR_All_Scores")


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading data from Google Sheets...")
    batters       = read_sheet(gc, sheet_id, "Batter_Statcast_2026")
    pitchers      = read_sheet(gc, sheet_id, "Pitcher_Statcast_2026")
    parks         = read_sheet(gc, sheet_id, "Park_Factors")
    weather       = read_sheet(gc, sheet_id, "Weather")
    bvp           = read_sheet(gc, sheet_id, "BvP_History")
    lineups       = read_sheet(gc, sheet_id, "Confirmed_Lineups")
    active_roster = read_sheet(gc, sheet_id, "Active_Rosters")
    odds_df       = read_sheet(gc, sheet_id, "HR_Odds")

    print(f"Batters: {len(batters)} rows")
    print(f"Pitchers: {len(pitchers)} rows")
    print(f"Parks: {len(parks)} rows")
    print(f"Weather: {len(weather)} rows")
    print(f"BvP History: {len(bvp)} rows")
    print(f"Confirmed Lineups: {len(lineups)} rows")
    print(f"Active Roster: {len(active_roster)} rows")
    print(f"HR Odds: {len(odds_df)} rows")

    combined = prepare_combined(batters, pitchers, parks, weather, bvp, lineups, active_roster)

    if combined.empty:
        print("WARNING: No combined data — check all sheets have data.")
        return

    picks, odds_lookup = build_main_picks(combined, odds_df)

    print(f"\nQualifying HR Picks (score ≥{MIN_SCORE_FLOOR}, odds ≤+{MAX_CHALK_ODDS} or ≥+{MIN_VALUE_ODDS}):")
    if not picks.empty:
        print(picks[["Rank", "Batter", "Bats", "Team", "Opposing Pitcher", "Throws", "Batting Avg", "Confidence", "HR Score", "Consensus Odds"]].to_string(index=False))
    else:
        print("No qualifying picks today.")

    row_count = write_picks_to_sheet(gc, sheet_id, picks)
    print("Written to Top_HR_Picks")

    format_picks_sheet(gc, sheet_id, row_count)

    resolve_pending_picks(gc, sheet_id)
    log_todays_picks(gc, sheet_id, picks)
    combined["consensus_odds"] = combined["player_name"].apply(
        lambda n: odds_lookup.get(normalize_name(str(n)), "")
    )
    log_all_scores(gc, sheet_id, combined)
    time.sleep(10)
    update_scorecard(gc, sheet_id)
    write_last_run_timestamp(gc, sheet_id)


if __name__ == "__main__":
    main()
