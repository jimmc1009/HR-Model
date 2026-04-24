import os
import json
from datetime import date, timedelta
from typing import Dict, Set

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
LEAGUE_AVG_BARREL_7D     = 11.0
LEAGUE_AVG_SEASON_BARREL = 8.0
LEAGUE_AVG_HARD_HIT_7D   = 40.0
LEAGUE_AVG_EV_7D         = 89.0

MIN_PA_FULL          = 150
MIN_PA_PARTIAL       = 30
MIN_BBE_7D_FULL      = 20
MIN_BBE_7D_PARTIAL   = 5

PLATOON_BONUS_WEIGHT = 0.8
PITCH_MATCHUP_WEIGHT = 1.2
WEATHER_WEIGHT       = 0.4
PULL_PARK_WEIGHT     = 0.6
MOMENTUM_WEIGHT      = 0.8
BVP_WEIGHT           = 0.9

ISO_GAP_SMALL  = 0.030
ISO_GAP_MEDIUM = 0.060
ISO_GAP_LARGE  = 0.100
ISO_GAP_SEVERE = 0.150

MIN_BATTING_AVG = 0.225
MAX_PER_TEAM    = 2

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
COLOR_EV_HEADER = {"red": 0.100, "green": 0.100, "blue": 0.100}


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


# ── Absolute scoring functions ─────────────────────────────────────────────

def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def regress(value: float, league_avg: float, sample: float, full_sample: float) -> float:
    weight = min(sample / full_sample, 1.0)
    return (value * weight) + (league_avg * (1 - weight))


def score_barrel_pct_7d(v: float, bbe_7d: float) -> float:
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 20: return 2.0
    if v >= 15: return 1.5
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


def score_avg_ev_7d(v: float, bbe_7d: float) -> float:
    v = regress(v, LEAGUE_AVG_EV_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 97: return 1.0
    if v >= 94: return 0.6
    if v >= 91: return 0.3
    return 0.0


def score_hard_hit_pct_7d(v: float, bbe_7d: float) -> float:
    v = regress(v, LEAGUE_AVG_HARD_HIT_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 55: return 0.8
    if v >= 45: return 0.5
    if v >= 35: return 0.2
    return 0.0


def score_pitcher_barrel_pct(v: float) -> float:
    if v >= 14: return 1.5
    if v >= 11: return 1.0
    if v >=  9: return 0.6
    if v >=  7: return 0.3
    return 0.0


def score_pitcher_hr_per_fb(v: float) -> float:
    if v >= 20: return 1.5
    if v >= 15: return 1.0
    if v >= 13: return 0.6
    if v >= 10: return 0.3
    return 0.0


def score_pitcher_hard_hit_pct(v: float) -> float:
    if v >= 45: return 0.8
    if v >= 38: return 0.5
    if v >= 32: return 0.2
    return 0.0


def score_park_factor(v: float) -> float:
    if v >= 20:  return 0.5
    if v >= 10:  return 0.3
    if v >= 0:   return 0.1
    if v >= -10: return -0.1
    return -0.3


def score_pitcher_quality_penalty(
    barrel_pct: float,
    hard_hit_pct: float,
    hr_per_fb: float,
    pitcher_bbe: float,
) -> float:
    """
    Multi-factor pitcher quality penalty.
    Each dimension contributes independently.
    Sample weighted — small sample pitchers get reduced penalty.
    """
    sample_weight = min(safe_float(pitcher_bbe) / 80.0, 1.0)

    penalty = 0.0

    # Barrel% suppression
    if barrel_pct <= 4:   penalty += 0.8
    elif barrel_pct <= 5: penalty += 0.5
    elif barrel_pct <= 6: penalty += 0.3

    # Hard hit% suppression
    if hard_hit_pct <= 30:   penalty += 0.5
    elif hard_hit_pct <= 33: penalty += 0.3
    elif hard_hit_pct <= 36: penalty += 0.15

    # HR/FB suppression
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


def normalize_inverted(series: pd.Series) -> pd.Series:
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return 1 - (series - mn) / (mx - mn)


def compute_momentum_score(row: pd.Series) -> tuple:
    ev_5d   = safe_float(row.get("avg_ev_5d",      0))
    ev_10d  = safe_float(row.get("avg_ev_10d",     0))
    bar_5d  = safe_float(row.get("barrel_pct_5d",  0))
    bar_10d = safe_float(row.get("barrel_pct_10d", 0))
    bbe_5d  = safe_float(row.get("bbe_5d",         0))
    bbe_10d = safe_float(row.get("bbe_10d",        0))

    if bbe_5d < 3 or bbe_10d < 5:
        return 0.0, ""

    score = 0.0
    parts = []

    if ev_5d > 0 and ev_10d > 0:
        ev_delta      = ev_5d - ev_10d
        score        += ev_delta / 3.0
        if ev_delta >= 2.0:
            parts.append(f"📈 EV trending up ({ev_10d:.1f}→{ev_5d:.1f} mph)")
        elif ev_delta <= -2.0:
            parts.append(f"📉 EV trending down ({ev_10d:.1f}→{ev_5d:.1f} mph)")

    if bar_5d >= 0 and bar_10d >= 0:
        barrel_delta  = bar_5d - bar_10d
        score        += barrel_delta / 8.0
        if barrel_delta >= 8.0:
            parts.append(f"📈 Barrel% trending up ({bar_10d:.1f}→{bar_5d:.1f}%)")
        elif barrel_delta <= -8.0:
            parts.append(f"📉 Barrel% trending down ({bar_10d:.1f}→{bar_5d:.1f}%)")

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

        batter_component = (
            batter_iso * 3 + batter_hr_rate / 10 + batter_barrel / 20
        ) if batter_has_data else 0.0

        pitcher_component = (
            pitcher_iso * 2 + pitcher_hr_rate / 10 + pitcher_barrel / 20
        ) if pitcher_has_data else 0.0

        pitch_score = (batter_component + pitcher_component) * (pitch_pct / 100)
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
                weakness_note = (
                    f" (but pitcher allows {pitcher_iso:.3f} ISO on it)"
                    if pitcher_has_data and pitcher_iso >= 0.100 else ""
                )
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
    elif p_throws == "R":
        iso_vs_this      = iso_vs_rhp
        iso_vs_opp       = iso_vs_lhp
        batter_barrel_vs = safe_float(row.get("vs_rhp_barrel_pct", 0))
        batter_hr_vs     = safe_float(row.get("vs_rhp_hr_rate", 0))
        matchup_label    = f"{batter_hand}HH vs RHP"
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
        if iso_gap >= ISO_GAP_SEVERE:
            penalty = 2.5
            parts.append(f"🚨 Severe platoon weakness ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite")
        elif iso_gap >= ISO_GAP_LARGE:
            penalty = 1.5
            parts.append(f"❌ Platoon disadvantage ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite")
        elif iso_gap >= ISO_GAP_MEDIUM:
            penalty = 0.7
            parts.append(f"⚠️ Moderate platoon weakness ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand")
        elif iso_gap <= -ISO_GAP_MEDIUM:
            score += 1.0
            parts.append(f"✅ Platoon advantage ({matchup_label}) — ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite")
        else:
            parts.append(f"↔️ Neutral platoon ({matchup_label})")

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
    weather_available = str(row.get("wind_context", "")).strip() not in (
        "", "Unknown", "Weather unavailable"
    )

    points        = 0
    batter_points = 0

    if batter_bbe >= 60 and batter_pa >= 100:
        points += 2
        batter_points = 2
    elif batter_bbe >= 30 and batter_pa >= 50:
        points += 1
        batter_points = 1

    if pitcher_bbe >= 80:
        points += 2
    elif pitcher_bbe >= 40:
        points += 1

    if not small_sample_park:
        points += 1

    if weather_available:
        points += 1

    if points >= 5 and batter_points >= 1:
        return "High"
    elif points >= 3 and batter_points >= 1:
        return "Medium"
    else:
        return "Low"


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
    pitchers = pitchers.rename(
        columns={k: v for k, v in pitcher_rename.items() if k in pitchers.columns}
    )

    if "pitcher_hand" not in pitchers.columns:
        pitchers["pitcher_hand"] = ""
    if "home_team" not in pitchers.columns:
        pitchers["home_team"] = pitchers.get("opp_pitcher_team", "")

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

    # Dynamic pitcher pitch type columns
    dynamic_pitch_cols = [
        c for c in pitchers.columns
        if c.startswith("pitcher_iso_allowed_")
        or c.startswith("pitcher_hr_rate_allowed_")
        or c.startswith("pitcher_barrel_pct_allowed_")
    ]

    pitcher_join_cols = [c for c in [
        "batter_team", "home_team", "opp_pitcher_name", "opp_pitcher_team",
        "pitcher_id_num",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_avg_ev", "pitcher_bf", "pitcher_bbe_allowed",
        "pitcher_pct_fastball", "pitcher_pct_breaking",
        "pitcher_pct_offspeed", "pitcher_pct_knuckleball",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
        "pitcher_hand",
        "top_pitch_1", "top_pitch_1_pct",
        "top_pitch_2", "top_pitch_2_pct",
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
        "hard_hit_pct_5d", "hard_hit_pct_10d",
        "bbe_5d", "bbe_7d", "bbe_10d",
        "pull_rate", "season_bbe", "pa",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_bf", "pitcher_bbe_allowed", "park_hr_factor",
        "lf_dist", "lf_height", "rf_dist", "rf_height",
        "pull_boost_rhh", "pull_boost_lhh",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "vs_lhp_hr_rate", "vs_rhp_hr_rate",
        "vs_lhp_iso", "vs_rhp_iso",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
        "top_pitch_1_pct", "top_pitch_2_pct", "top_pitch_3_pct",
        "bvp_pa", "bvp_hr", "bvp_iso", "bvp_barrel_pct", "bvp_hr_rate",
        "hr_7d", "season_hr", "season_fb", "season_hard_hit",
    ]

    dynamic_cols = [c for c in combined.columns if c.startswith("pitcher_iso_allowed_") or c.startswith("pitcher_hr_rate_allowed_") or c.startswith("pitcher_barrel_pct_allowed_")]
    score_cols  += dynamic_cols

    for col in score_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(safe_float)
        else:
            combined[col] = 0.0

    combined["park_hr_factor_norm"] = combined["park_hr_factor"] - 100

    # ── Component scores ───────────────────────────────────────────────────
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


def build_main_picks(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    combined["reason"] = combined.apply(build_reason, axis=1)

    combined_sorted = combined.sort_values("score", ascending=False)
    combined_sorted["team_count"] = combined_sorted.groupby("batter_team").cumcount()
    capped = combined_sorted[combined_sorted["team_count"] < MAX_PER_TEAM].head(10).copy()
    capped = capped.drop(columns=["team_count"])
    top10  = capped
    top10["rank"] = range(1, len(top10) + 1)

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

    available = {k: v for k, v in output_cols.items() if k in top10.columns}
    return top10[list(available.keys())].rename(columns=available)


def build_ev_subsection(combined: pd.DataFrame, exclude_names: Set[str]) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    combined = combined.copy()
    if "player_name" in combined.columns and exclude_names:
        before   = len(combined)
        combined = combined[~combined["player_name"].isin(exclude_names)].copy()
        excluded = before - len(combined)
        if excluded > 0:
            print(f"EV section: excluded {excluded} players already in top 10")

    if "avg_la_7d" in combined.columns:
        combined["la_for_filter"] = combined["avg_la_7d"].apply(safe_float)
    else:
        combined["la_for_filter"] = combined["avg_launch_angle"].apply(safe_float)

    ev_df = combined[
        (combined["avg_ev_7d"] > 92) &
        (combined["la_for_filter"] < 20) &
        (combined["la_for_filter"] > -90)
    ].copy()

    if ev_df.empty:
        print("No high EV / low launch angle candidates found.")
        return pd.DataFrame()

    ev_df["hard_hit_season_avg"] = ev_df.apply(
        lambda r: safe_float(r.get("season_hard_hit", 0)) / max(safe_float(r.get("season_bbe", 1)), 1) * 100,
        axis=1,
    )
    ev_df["hard_hit_trend"] = ev_df["hard_hit_pct_7d"] - ev_df["hard_hit_season_avg"]
    ev_df["la_season"]      = ev_df["avg_launch_angle"].apply(safe_float)
    ev_df["la_7d"]          = ev_df["avg_la_7d"].apply(safe_float)
    ev_df["la_trend"]       = ev_df["la_7d"] - ev_df["la_season"]
    ev_df["la_hr_gap"]      = (25 - ev_df["la_7d"]).abs()
    ev_df["ev_momentum"]    = ev_df["momentum_score"].apply(safe_float)

    ev_df["ev_score"] = (
        normalize(ev_df["avg_ev_7d"])                * 2.5 +
        normalize(ev_df["hard_hit_pct_7d"])          * 1.5 +
        normalize(ev_df["hard_hit_trend"])           * 0.8 +
        normalize(ev_df["la_trend"])                 * 0.7 +
        normalize(-ev_df["la_hr_gap"])               * 0.6 +
        ev_df["ev_momentum"]                         * 0.8 +
        ev_df["bvp_score_capped"]                    * 0.7 +
        normalize(ev_df["pitcher_barrel_pct"])       * 1.5 +
        normalize(ev_df["pitcher_hr_per_fb"])        * 1.2 +
        normalize(ev_df["pitch_matchup_score_capped"]) * 1.0 +
        normalize(ev_df["platoon_score_capped"])     * 0.8 +
        normalize(ev_df["park_hr_factor_norm"])      * 0.5 +
        ev_df["pull_park_score_capped"]              * 0.4 +
        ev_df["weather_score"]                       * 0.3 -
        ev_df.apply(lambda r: score_pitcher_quality_penalty(
            safe_float(r.get("pitcher_barrel_pct")),
            safe_float(r.get("pitcher_hard_hit_pct")),
            safe_float(r.get("pitcher_hr_per_fb")),
            safe_float(r.get("pitcher_bbe_allowed")),
        ), axis=1) -
        ev_df["total_penalty"]
    ).round(3)

    ev_sorted = ev_df.sort_values("ev_score", ascending=False)
    ev_sorted["team_count"] = ev_sorted.groupby("batter_team").cumcount()
    ev_capped = ev_sorted[ev_sorted["team_count"] < MAX_PER_TEAM].head(5).copy()
    ev_capped = ev_capped.drop(columns=["team_count"])
    top5 = ev_capped
    top5["ev_rank"] = range(1, len(top5) + 1)

    def build_ev_reason(row) -> str:
        reasons = []
        ev    = safe_float(row.get("avg_ev_7d"))
        la_7d = safe_float(row.get("avg_la_7d"))
        la_s  = safe_float(row.get("avg_launch_angle"))
        la    = la_7d if la_7d != 0.0 else la_s
        reasons.append(f"💥 Avg EV {ev:.1f} mph, launch angle {la:.1f}° (7d)")

        la_trend = safe_float(row.get("la_trend"))
        if la_trend >= 3.0:
            reasons.append(f"📐 Launch angle trending up {la_s:.1f}°→{la_7d:.1f}°")
        elif la_7d >= 15 and la_7d < 20:
            reasons.append(f"📐 Launch angle {la_7d:.1f}° — one adjustment from HR zone")

        hh       = safe_float(row.get("hard_hit_pct_7d"))
        hh_trend = safe_float(row.get("hard_hit_trend"))
        if hh >= 50:
            reasons.append(f"🔨 Elite hard hit rate {hh:.1f}% (7d)")
        elif hh >= 40:
            reasons.append(f"🔨 {hh:.1f}% hard hit rate (7d)")
        if hh_trend >= 10:
            hh_season = safe_float(row.get("hard_hit_season_avg"))
            reasons.append(f"📈 Hard hit rate heating up ({hh_season:.1f}% → {hh:.1f}% last 7d)")

        momentum_desc = str(row.get("momentum_desc", ""))
        if momentum_desc:
            reasons.append(momentum_desc)

        bvp_desc = str(row.get("bvp_desc", ""))
        if bvp_desc:
            reasons.append(bvp_desc)

        p_barrel = safe_float(row.get("pitcher_barrel_pct"))
        if p_barrel >= 8:
            reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")

        platoon_desc = str(row.get("platoon_desc", ""))
        if platoon_desc:
            reasons.append(f"🔄 {platoon_desc}")

        pull_desc = str(row.get("pull_park_desc", ""))
        if pull_desc:
            reasons.append(pull_desc)

        pitch_desc = str(row.get("pitch_matchup_desc", ""))
        if pitch_desc:
            reasons.append(f"🎳 {pitch_desc}")

        reasons.append("⬆️ Launch angle correction = HR upside")
        return " | ".join(reasons)

    top5["why"] = top5.apply(build_ev_reason, axis=1)

    output_cols = {
        "ev_rank":                    "Rank",
        "player_name":                "Batter",
        "batter_hand":                "Bats",
        "batter_team":                "Team",
        "opp_pitcher_name":           "Opposing Pitcher",
        "pitcher_hand":               "Throws",
        "opp_pitcher_team":           "Pitcher Team",
        "park_name":                  "Park",
        "ev_score":                   "EV Score",
        "confidence":                 "Confidence",
        "why":                        "Why They're Here",
        "batting_avg":                "Batting Avg",
        "avg_ev_7d":                  "Avg EV (7d)",
        "avg_ev_5d":                  "Avg EV (5d)",
        "avg_ev_10d":                 "Avg EV (10d)",
        "avg_la_7d":                  "Avg Launch Angle (7d)",
        "avg_launch_angle":           "Avg Launch Angle (Season)",
        "la_trend":                   "LA Trend (7d vs Season)",
        "hard_hit_pct_7d":            "Hard Hit% (7d)",
        "hard_hit_season_avg":        "Hard Hit% (Season)",
        "hard_hit_trend":             "Hard Hit% Trend",
        "momentum_desc":              "Momentum",
        "bvp_pa":                     "BvP PA",
        "bvp_hr":                     "BvP HR",
        "bvp_iso":                    "BvP ISO",
        "bvp_desc":                   "BvP Notes",
        "hr_per_fb":                  "HR/FB%",
        "pull_rate":                  "Pull Rate%",
        "pull_park_desc":             "Pull Park Matchup",
        "vs_lhp_iso":                 "ISO vs LHP",
        "vs_rhp_iso":                 "ISO vs RHP",
        "platoon_desc":               "Platoon Matchup",
        "pitcher_barrel_pct":         "Pitcher Barrel% Allowed",
        "pitcher_hr_per_fb":          "Pitcher HR/FB%",
        "pitcher_vs_lhh_barrel_pct":  "Pitcher Barrel% vs LHH",
        "pitcher_vs_rhh_barrel_pct":  "Pitcher Barrel% vs RHH",
        "pitcher_vs_lhh_hr9":         "Pitcher HR/9 vs LHH",
        "pitcher_vs_rhh_hr9":         "Pitcher HR/9 vs RHH",
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

    available = {k: v for k, v in output_cols.items() if k in top5.columns}
    return top5[list(available.keys())].rename(columns=available)


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

    today_str     = date.today().strftime("%Y-%m-%d")
    pending       = existing[(existing["hit_hr"] == "Pending") & (existing["date"] != today_str) & (existing["date"] != "")].copy()

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


def log_todays_picks(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame, ev_section: pd.DataFrame) -> None:
    today_str = date.today().strftime("%Y-%m-%d")
    sh        = gc.open_by_key(sheet_id)

    try:
        ws       = sh.worksheet("Picks_Log")
        existing = pd.DataFrame(ws.get_all_records())
    except gspread.WorksheetNotFound:
        ws       = sh.add_worksheet(title="Picks_Log", rows=5000, cols=25)
        existing = pd.DataFrame()

    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"] != today_str].copy()

    new_rows = []

    def build_log_rows(df: pd.DataFrame, section: str) -> None:
        if df.empty:
            return
        score_col = "HR Score" if section == "Main" else "EV Score"
        for _, row in df.iterrows():
            new_rows.append({
                "date":        today_str,
                "rank":        str(row.get("Rank", "")),
                "player_name": str(row.get("Batter", "")),
                "player_id":   "",
                "team":        str(row.get("Team", "")),
                "pitcher_id":  "",
                "park_name":   str(row.get("Park", "")),
                "hr_score":    str(row.get(score_col, "")),
                "confidence":  str(row.get("Confidence", "")),
                "hit_hr":      "Pending",
                "section":     section,
                "odds":        "",
                "bet_placed":  "",
            })

    build_log_rows(picks,      "Main")
    build_log_rows(ev_section, "EV")

    if not new_rows:
        print("No picks to log.")
        return

    new_df = pd.DataFrame(new_rows)
    for col in ["odds", "bet_placed"]:
        if not existing.empty and col not in existing.columns:
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
    sh = gc.open_by_key(sheet_id)

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

    bet_picks = scored[
        (scored.get("bet_placed", pd.Series("", index=scored.index)).astype(str).str.strip().str.lower() == "yes") &
        (scored.get("odds", pd.Series("", index=scored.index)).astype(str).str.strip() != "")
    ].copy()

    if not bet_picks.empty:
        bet_picks["odds_num"]      = bet_picks["odds"].apply(lambda x: safe_float(str(x).replace("+", "").strip(), 0))
        bet_picks["profit_if_win"] = bet_picks["odds_num"].apply(american_odds_to_profit)
        bet_picks["unit_result"]   = bet_picks.apply(lambda r: r["profit_if_win"] if r["hit_hr_bool"] else -1.0, axis=1)

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
    perf_rows.append({"label": "── By Section ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "_bold": True, "_header": True})
    for section in scored["section"].dropna().unique():
        sub = scored[scored["section"] == section]
        if not sub.empty: add_perf(f"   {section}", sub)
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
        roi_rows.append({"label": "── By Rank ──", "bets_placed": "", "hr_count": "", "hit_rate_pct": "", "units_wagered": "", "units_profit": "", "roi_pct": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        for rank in range(1, 11):
            sub = bet_picks[bet_picks["rank"] == rank]
            if not sub.empty: add_roi(f"   Rank {rank}", sub)
        roi_rows.append({"label": "── By Section ──", "bets_placed": "", "hr_count": "", "hit_rate_pct": "", "units_wagered": "", "units_profit": "", "roi_pct": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        for section in bet_picks["section"].dropna().unique():
            sub = bet_picks[bet_picks["section"] == section]
            if not sub.empty: add_roi(f"   {section}", sub)
        roi_rows.append({"label": "── Rolling ──", "bets_placed": "", "hr_count": "", "hit_rate_pct": "", "units_wagered": "", "units_profit": "", "roi_pct": "", "_bold": True, "_header": True, "_roi_val": 0, "_profit_val": 0})
        max_bet = bet_picks["date"].max()
        add_roi("   Last 7 Days",  bet_picks[bet_picks["date"] >= max_bet - pd.Timedelta(days=7)])
        add_roi("   Last 30 Days", bet_picks[bet_picks["date"] >= max_bet - pd.Timedelta(days=30)])

    add_score("📈  All Scored Picks", scored, bold=True)
    score_rows.append({"label": "── By Score Tier ──", "total_picks": "", "hr_count": "", "hit_rate_pct": "", "avg_score": "", "_bold": True, "_header": True})
    for label, sub in [("   11+", scored[scored["hr_score"] >= 11]), ("   10+", scored[scored["hr_score"] >= 10]), ("   9+", scored[scored["hr_score"] >= 9]), ("   8+", scored[scored["hr_score"] >= 8]), ("   7+", scored[scored["hr_score"] >= 7]), ("   6+", scored[scored["hr_score"] >= 6]), ("   Under 6", scored[scored["hr_score"] < 6])]:
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

    reqs = []
    C_BG        = {"red": 0.114, "green": 0.114, "blue": 0.114}
    C_BG_ALT    = {"red": 0.149, "green": 0.149, "blue": 0.149}
    C_BLUE      = {"red": 0.114, "green": 0.533, "blue": 0.898}
    C_BLUE_DIM  = {"red": 0.055, "green": 0.180, "blue": 0.318}
    C_GOLD      = {"red": 0.800, "green": 0.650, "blue": 0.000}
    C_GOLD_DIM  = {"red": 0.200, "green": 0.160, "blue": 0.000}
    C_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
    C_GREEN_DIM = {"red": 0.039, "green": 0.180, "blue": 0.098}
    C_RED       = {"red": 0.910, "green": 0.259, "blue": 0.259}
    C_RED_DIM   = {"red": 0.200, "green": 0.039, "blue": 0.039}
    C_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
    C_GREY      = {"red": 0.600, "green": 0.600, "blue": 0.600}
    C_HEADER_BG = {"red": 0.078, "green": 0.078, "blue": 0.078}
    C_PURPLE    = {"red": 0.576, "green": 0.439, "blue": 0.859}
    C_PURPLE_DIM= {"red": 0.118, "green": 0.078, "blue": 0.196}

    total_rows = len(all_values)
    total_cols = 7

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows, "startColumnIndex": 0, "endColumnIndex": total_cols}, "cell": {"userEnteredFormat": {"backgroundColor": C_BG, "textFormat": {"foregroundColor": C_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10}, "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": 5}, "cell": {"userEnteredFormat": {"backgroundColor": C_BLUE_DIM, "textFormat": {"foregroundColor": C_BLUE, "bold": True, "fontFamily": "Roboto", "fontSize": 13}, "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": 2, "startColumnIndex": 0, "endColumnIndex": 4}, "cell": {"userEnteredFormat": {"backgroundColor": C_HEADER_BG, "textFormat": {"foregroundColor": C_BLUE, "bold": True, "fontFamily": "Roboto", "fontSize": 10}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})

    for i, r in enumerate(perf_rows):
        row_idx   = perf_start + i
        is_header = r.get("_header", False)
        is_bold   = r.get("_bold", False)
        rate      = r.get("hit_rate_pct", "")
        if is_header:
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 4}, "cell": {"userEnteredFormat": {"backgroundColor": C_HEADER_BG, "textFormat": {"foregroundColor": C_GREY, "bold": True, "fontSize": 9, "italic": True}, "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"}})
        else:
            bg = C_BG if i % 2 == 0 else C_BG_ALT
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 4}, "cell": {"userEnteredFormat": {"backgroundColor": bg}}, "fields": "userEnteredFormat(backgroundColor)"}})
            if rate != "":
                try:
                    rate_val = float(str(rate).replace("%", ""))
                    rate_bg  = C_GREEN_DIM if rate_val >= 15 else C_RED_DIM if rate_val <= 8 else bg
                    rate_fg  = C_GREEN if rate_val >= 15 else C_RED if rate_val <= 8 else C_WHITE
                    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 3, "endColumnIndex": 4}, "cell": {"userEnteredFormat": {"backgroundColor": rate_bg, "textFormat": {"foregroundColor": rate_fg, "bold": rate_val >= 15 or rate_val <= 8}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"}})
                except Exception:
                    pass
            if is_bold and not is_header:
                reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 1}, "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": C_BLUE, "bold": True, "fontSize": 11}}}, "fields": "userEnteredFormat(textFormat)"}})

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": roi_section_start, "endRowIndex": roi_section_start + 1, "startColumnIndex": 0, "endColumnIndex": total_cols}, "cell": {"userEnteredFormat": {"backgroundColor": C_GOLD_DIM, "textFormat": {"foregroundColor": C_GOLD, "bold": True, "fontFamily": "Roboto", "fontSize": 13}, "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"}})

    if roi_rows:
        reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": roi_section_start + 1, "endRowIndex": roi_section_start + 2, "startColumnIndex": 0, "endColumnIndex": total_cols}, "cell": {"userEnteredFormat": {"backgroundColor": C_HEADER_BG, "textFormat": {"foregroundColor": C_GOLD, "bold": True, "fontFamily": "Roboto", "fontSize": 10}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})
        for i, r in enumerate(roi_rows):
            row_idx    = roi_data_start + i
            is_header  = r.get("_header", False)
            is_bold    = r.get("_bold", False)
            roi_val    = r.get("_roi_val", 0)
            profit_val = r.get("_profit_val", 0)
            if is_header:
                reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": total_cols}, "cell": {"userEnteredFormat": {"backgroundColor": C_HEADER_BG, "textFormat": {"foregroundColor": C_GREY, "bold": True, "fontSize": 9, "italic": True}, "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"}})
            else:
                bg = C_BG if i % 2 == 0 else C_BG_ALT
                reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": total_cols}, "cell": {"userEnteredFormat": {"backgroundColor": bg}}, "fields": "userEnteredFormat(backgroundColor)"}})
                if roi_val != 0 or r.get("bets_placed", "") != "":
                    roi_fg = C_GREEN if roi_val > 0 else C_RED if roi_val < 0 else C_WHITE
                    roi_bg = C_GREEN_DIM if roi_val > 0 else C_RED_DIM if roi_val < 0 else bg
                    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 6, "endColumnIndex": 7}, "cell": {"userEnteredFormat": {"backgroundColor": roi_bg, "textFormat": {"foregroundColor": roi_fg, "bold": True}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"}})
                    profit_fg = C_GREEN if profit_val > 0 else C_RED if profit_val < 0 else C_WHITE
                    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 5, "endColumnIndex": 6}, "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": profit_fg, "bold": True}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})
                if is_bold:
                    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 1}, "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": C_GOLD, "bold": True, "fontSize": 11}}}, "fields": "userEnteredFormat(textFormat)"}})

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": score_section_start, "endRowIndex": score_section_start + 1, "startColumnIndex": 0, "endColumnIndex": 5}, "cell": {"userEnteredFormat": {"backgroundColor": C_PURPLE_DIM, "textFormat": {"foregroundColor": C_PURPLE, "bold": True, "fontFamily": "Roboto", "fontSize": 13}, "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": score_section_start + 1, "endRowIndex": score_section_start + 2, "startColumnIndex": 0, "endColumnIndex": 5}, "cell": {"userEnteredFormat": {"backgroundColor": C_HEADER_BG, "textFormat": {"foregroundColor": C_PURPLE, "bold": True, "fontFamily": "Roboto", "fontSize": 10}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})

    for i, r in enumerate(score_rows):
        row_idx   = score_data_start + i
        is_header = r.get("_header", False)
        is_bold   = r.get("_bold", False)
        rate      = r.get("hit_rate_pct", "")
        if is_header:
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 5}, "cell": {"userEnteredFormat": {"backgroundColor": C_HEADER_BG, "textFormat": {"foregroundColor": C_GREY, "bold": True, "fontSize": 9, "italic": True}, "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment)"}})
        else:
            bg = C_BG if i % 2 == 0 else C_BG_ALT
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 5}, "cell": {"userEnteredFormat": {"backgroundColor": bg}}, "fields": "userEnteredFormat(backgroundColor)"}})
            if rate != "":
                try:
                    rate_val = float(str(rate).replace("%", ""))
                    rate_bg  = C_GREEN_DIM if rate_val >= 15 else C_RED_DIM if rate_val <= 8 else bg
                    rate_fg  = C_GREEN if rate_val >= 15 else C_RED if rate_val <= 8 else C_WHITE
                    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 3, "endColumnIndex": 4}, "cell": {"userEnteredFormat": {"backgroundColor": rate_bg, "textFormat": {"foregroundColor": rate_fg, "bold": rate_val >= 15 or rate_val <= 8}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"}})
                except Exception:
                    pass
            if is_bold and not is_header:
                reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": row_idx, "endRowIndex": row_idx + 1, "startColumnIndex": 0, "endColumnIndex": 1}, "cell": {"userEnteredFormat": {"textFormat": {"foregroundColor": C_PURPLE, "bold": True, "fontSize": 11}}}, "fields": "userEnteredFormat(textFormat)"}})

    col_widths = [220, 100, 80, 100, 120, 120, 100]
    for i, width in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1}, "properties": {"pixelSize": width}, "fields": "pixelSize"}})

    reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 0, "endIndex": total_rows}, "properties": {"pixelSize": 32}, "fields": "pixelSize"}})
    for row_idx in [0, roi_section_start, score_section_start]:
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": row_idx, "endIndex": row_idx + 1}, "properties": {"pixelSize": 44}, "fields": "pixelSize"}})
    reqs.append({"updateSheetProperties": {"properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 2}}, "fields": "gridProperties.frozenRowCount"}})
    reqs.append({"updateSheetProperties": {"properties": {"sheetId": ws_id, "tabColorStyle": {"rgbColor": C_GOLD}}, "fields": "tabColorStyle"}})

    try:
        sh.batch_update({"requests": reqs})
        print("Scorecard dashboard formatted successfully!")
    except Exception as e:
        print(f"Scorecard formatting failed: {e}")

    print("Scorecard updated.")


def clean_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x)
    df = df.fillna("")
    return df


def write_picks_to_sheet(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame, ev_section: pd.DataFrame) -> tuple:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("Top_HR_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_HR_Picks", rows=100, cols=55)

    picks_clean = clean_for_sheets(picks)
    ev_clean    = clean_for_sheets(ev_section)
    all_values  = []

    all_values.append(picks_clean.columns.tolist())
    for _, row in picks_clean.iterrows():
        all_values.append(row.astype(str).tolist())

    main_row_count    = len(picks_clean)
    ev_start_row      = len(all_values) + 2
    all_values.append([])
    all_values.append(["⚡  HIGH EXIT VELOCITY — LAUNCH ANGLE UPSIDE PICKS"])
    ev_col_header_row = ev_start_row + 1
    ev_data_row_count = 0

    if not ev_clean.empty:
        all_values.append(ev_clean.columns.tolist())
        for _, row in ev_clean.iterrows():
            all_values.append(row.astype(str).tolist())
        ev_data_row_count = len(ev_clean)

    ws.update(all_values)
    return main_row_count, ev_data_row_count, ev_start_row, ev_col_header_row


def format_picks_sheet(gc: gspread.Client, sheet_id: str, main_row_count: int, ev_data_row_count: int, ev_start_row: int, ev_col_header_row: int) -> None:
    print("Applying Carbon dark mode formatting...")
    sh    = gc.open_by_key(sheet_id)
    ws    = sh.worksheet("Top_HR_Picks")
    ws_id = ws.id

    main_cols  = 50
    ev_cols    = 38
    total_rows = ev_col_header_row + ev_data_row_count + 5
    reqs       = []

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows, "startColumnIndex": 0, "endColumnIndex": 55}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG, "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10}, "verticalAlignment": "MIDDLE", "wrapStrategy": "WRAP"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": main_cols}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_ACCENT_DIM, "textFormat": {"foregroundColor": COLOR_ACCENT, "bold": True, "fontFamily": "Roboto", "fontSize": 11}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})

    for i in range(main_row_count):
        reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2, "startColumnIndex": 0, "endColumnIndex": main_cols}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG}}, "fields": "userEnteredFormat(backgroundColor)"}})

    medals = [(1, {"red": 0.18, "green": 0.14, "blue": 0.00}, COLOR_GOLD), (2, {"red": 0.14, "green": 0.14, "blue": 0.14}, COLOR_SILVER), (3, {"red": 0.16, "green": 0.10, "blue": 0.04}, COLOR_BRONZE)]
    for rank, bg, fg in medals:
        if main_row_count >= rank:
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": rank, "endRowIndex": rank + 1, "startColumnIndex": 0, "endColumnIndex": main_cols}, "cell": {"userEnteredFormat": {"backgroundColor": bg, "textFormat": {"foregroundColor": fg, "bold": True, "fontSize": 10}}}, "fields": "userEnteredFormat(backgroundColor,textFormat)"}})

    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": main_row_count + 1, "startColumnIndex": 0, "endColumnIndex": 1}, "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 14, "foregroundColor": COLOR_ACCENT}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": main_row_count + 1, "startColumnIndex": 8, "endColumnIndex": 9}, "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": 12, "foregroundColor": COLOR_GREEN}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": ev_start_row, "endRowIndex": ev_start_row + 1, "startColumnIndex": 0, "endColumnIndex": ev_cols}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_ACCENT_DIM, "textFormat": {"foregroundColor": COLOR_ORANGE, "bold": True, "fontFamily": "Roboto", "fontSize": 12}, "horizontalAlignment": "LEFT", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})
    reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": ev_col_header_row, "endRowIndex": ev_col_header_row + 1, "startColumnIndex": 0, "endColumnIndex": ev_cols}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_EV_HEADER, "textFormat": {"foregroundColor": COLOR_ORANGE, "bold": True, "fontFamily": "Roboto", "fontSize": 11}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}}, "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}})

    for i in range(ev_data_row_count):
        reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": ev_col_header_row + 1 + i, "endRowIndex": ev_col_header_row + 2 + i, "startColumnIndex": 0, "endColumnIndex": ev_cols}, "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG}}, "fields": "userEnteredFormat(backgroundColor)"}})

    if ev_data_row_count > 0:
        for col_idx, font_size in [(0, 14), (8, 11)]:
            reqs.append({"repeatCell": {"range": {"sheetId": ws_id, "startRowIndex": ev_col_header_row + 1, "endRowIndex": ev_col_header_row + 1 + ev_data_row_count, "startColumnIndex": col_idx, "endColumnIndex": col_idx + 1}, "cell": {"userEnteredFormat": {"textFormat": {"bold": True, "fontSize": font_size, "foregroundColor": COLOR_ORANGE}, "horizontalAlignment": "CENTER"}}, "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})

    reqs.append({"updateSheetProperties": {"properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}}, "fields": "gridProperties.frozenRowCount"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 0, "endIndex": 1}, "properties": {"pixelSize": 42}, "fields": "pixelSize"}})
    reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 1, "endIndex": total_rows}, "properties": {"pixelSize": 58}, "fields": "pixelSize"}})

    col_widths = [50, 160, 50, 55, 175, 55, 75, 175, 75, 80, 380, 75, 90, 90, 75, 75, 200, 75, 75, 65, 80, 80, 85, 85, 85, 100, 100, 75, 220, 60, 60, 70, 200, 220, 120, 120, 130, 130, 90, 90, 90, 75, 90, 75, 90, 75, 240, 75, 65, 65, 65, 65]
    for i, width in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {"range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1}, "properties": {"pixelSize": width}, "fields": "pixelSize"}})

    reqs.append({"updateSheetProperties": {"properties": {"sheetId": ws_id, "tabColorStyle": {"rgbColor": COLOR_ACCENT}}, "fields": "tabColorStyle"}})

    try:
        sh.batch_update({"requests": reqs})
        print("Carbon formatting applied successfully!")
    except APIError as e:
        print(f"Formatting failed: {e}")


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

    print(f"Batters: {len(batters)} rows")
    print(f"Pitchers: {len(pitchers)} rows")
    print(f"Parks: {len(parks)} rows")
    print(f"Weather: {len(weather)} rows")
    print(f"BvP History: {len(bvp)} rows")
    print(f"Confirmed Lineups: {len(lineups)} rows")
    print(f"Active Roster: {len(active_roster)} rows")

    combined = prepare_combined(batters, pitchers, parks, weather, bvp, lineups, active_roster)

    if combined.empty:
        print("WARNING: No combined data — check all sheets have data.")
        return

    picks = build_main_picks(combined)

    exclude_from_ev: Set[str] = set()
    if not picks.empty and "Batter" in picks.columns:
        exclude_from_ev = set(picks["Batter"].dropna().tolist())

    ev_section = build_ev_subsection(combined, exclude_from_ev)

    print(f"Built {len(picks)} main picks")
    print(f"Built {len(ev_section)} EV subsection picks")

    if picks.empty:
        print("WARNING: No picks generated.")
        return

    print("\nTop 10 HR Picks:")
    print(picks[["Rank", "Batter", "Bats", "Team", "Opposing Pitcher", "Throws", "Batting Avg", "Confidence", "HR Score"]].to_string(index=False))

    if not ev_section.empty:
        print("\nHigh EV / Launch Angle Upside:")
        print(ev_section[["Rank", "Batter", "Bats", "Team", "Avg EV (7d)", "Avg Launch Angle (7d)", "Batting Avg", "Confidence"]].to_string(index=False))

    main_row_count, ev_data_row_count, ev_start_row, ev_col_header_row = write_picks_to_sheet(gc, sheet_id, picks, ev_section)
    print("Written to Top_HR_Picks")

    format_picks_sheet(gc, sheet_id, main_row_count=main_row_count, ev_data_row_count=ev_data_row_count, ev_start_row=ev_start_row, ev_col_header_row=ev_col_header_row)

    resolve_pending_picks(gc, sheet_id)
    log_todays_picks(gc, sheet_id, picks, ev_section)
    update_scorecard(gc, sheet_id)


if __name__ == "__main__":
    main()
