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

MIN_PA_FULL        = 150
MIN_BBE_7D_FULL    = 20
MIN_BBE_7D_PARTIAL = 5

# ── Weights (based on feature separator analysis) ──────────────────────────
PITCH_MATCHUP_WEIGHT = 1.0   # capped — only 2 days resolved data
BVP_WEIGHT           = 0.5   # conceptually sound, no separator data yet

# ── Pick criteria ──────────────────────────────────────────────────────────
MIN_SCORE_FLOOR  = 10.0
MIN_ODDS         = 301       # no chalk — ≤+300 hits only 9% (below baseline)
MAX_ODDS         = 699       # hard cap — no +700+ picks
TOP_N            = 15
MAX_PER_TEAM     = 2
MAX_PER_GAME     = 2
MIN_BATTING_AVG  = 0.200

# ── Score tier hit rates (from HR_Analysis, updated as data accumulates) ───
# Used for edge calculation. Update these as more resolved data comes in.
# Tightened buckets as of 2026-06-02. Rates will stabilize with more data.
SCORE_TIER_HIT_RATES = {
    "13+":      0.200,  # 15 picks, 3 HR — 20.0%
    "12-13":    0.176,  # 17 picks, 3 HR — 17.6%
    "11-12":    0.188,  # 32 picks, 6 HR — 18.8%
    "10-11":    0.291,  # 55 picks, 16 HR — 29.1% (best tier)
    "9-10":     0.089,  # 56 picks, 5 HR — 8.9% (below baseline, caution)
    "8.5-9":    0.278,  # 36 picks, 10 HR — 27.8%
}

COLOR_BG         = {"red": 0.114, "green": 0.114, "blue": 0.114}
COLOR_BG_ALT     = {"red": 0.149, "green": 0.149, "blue": 0.149}
COLOR_ACCENT     = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_ACCENT_DIM = {"red": 0.055, "green": 0.180, "blue": 0.318}
COLOR_WHITE      = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD       = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_SILVER     = {"red": 0.753, "green": 0.753, "blue": 0.753}
COLOR_BRONZE     = {"red": 0.804, "green": 0.498, "blue": 0.196}
COLOR_GREEN      = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_GREEN_DIM  = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_RED        = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_RED_DIM    = {"red": 0.200, "green": 0.039, "blue": 0.039}


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


# ── Batter scoring ─────────────────────────────────────────────────────────
# Only components with meaningful separation from feature analysis

def score_barrel_pct_7d(v: float, bbe_7d: float) -> float:
    """Barrel% 7d — +31.8% separation, strongest consistent predictor"""
    if bbe_7d < MIN_BBE_7D_PARTIAL:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe_7d, MIN_BBE_7D_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    return 0.0


def score_barrel_pct_5d(v: float, bbe_5d: float) -> float:
    """Barrel% 5d — +31.8% separation"""
    if bbe_5d < MIN_BBE_7D_PARTIAL:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe_5d, MIN_BBE_7D_FULL)
    if v >= 20: return 2.0
    if v >= 15: return 1.5
    if v >= 10: return 1.0
    if v >= 6:  return 0.3
    return 0.0


def score_barrel_pct_10d(v: float, bbe_10d: float) -> float:
    """Barrel% 10d — +36.3% separation, strongest window"""
    if bbe_10d < MIN_BBE_7D_PARTIAL:
        return 0.0
    v = regress(v, LEAGUE_AVG_BARREL_7D, bbe_10d, MIN_BBE_7D_FULL)
    if v >= 20: return 2.5
    if v >= 15: return 1.8
    if v >= 10: return 1.2
    if v >= 6:  return 0.4
    return 0.0


def score_season_barrel_pct(v: float, pa: float) -> float:
    """Barrel% Season — +31.0% separation"""
    v = regress(v, LEAGUE_AVG_SEASON_BARREL, pa, MIN_PA_FULL)
    if v >= 14: return 1.5
    if v >= 11: return 1.0
    if v >=  9: return 0.6
    if v >=  7: return 0.3
    return 0.0


def score_hr_per_fb(v: float, pa: float) -> float:
    """HR/FB% — +23.8% separation"""
    v = regress(v, LEAGUE_AVG_HR_PER_FB, pa, MIN_PA_FULL)
    if v >= 20: return 1.5
    if v >= 15: return 1.0
    if v >= 10: return 0.5
    return 0.0


def score_hr_per_pa(v: float, pa: float) -> float:
    """HR/PA% — +23.7% separation"""
    v = regress(v, LEAGUE_AVG_HR_PER_PA, pa, MIN_PA_FULL)
    if v >= 6.0: return 1.5
    if v >= 4.0: return 1.0
    if v >= 2.5: return 0.5
    return 0.0


def score_iso(v: float, pa: float) -> float:
    """ISO — +15.0% separation"""
    v = regress(v, LEAGUE_AVG_ISO, pa, MIN_PA_FULL)
    if v >= 0.300: return 1.0
    if v >= 0.250: return 0.7
    if v >= 0.200: return 0.5
    if v >= 0.175: return 0.3
    if v >= 0.150: return 0.1
    return 0.0


# ── Pitcher scoring ────────────────────────────────────────────────────────
# Light weights — only +9% and +8.4% separation

def score_pitcher_barrel_pct(v: float) -> float:
    """Pitcher Barrel% Allowed — +9.0% separation"""
    if v >= 14: return 0.6
    if v >= 11: return 0.4
    if v >=  9: return 0.2
    if v >=  7: return 0.1
    return 0.0


def score_pitcher_hr_per_fb(v: float) -> float:
    """Pitcher HR/FB% Allowed — +8.4% separation"""
    if v >= 20: return 0.5
    if v >= 15: return 0.3
    if v >= 13: return 0.2
    if v >= 10: return 0.1
    return 0.0


def score_pitcher_quality_penalty(
    barrel_pct: float,
    hr_per_fb: float,
    pitcher_bbe: float,
) -> float:
    """Penalty for facing genuinely elite pitchers"""
    sample_weight = min(safe_float(pitcher_bbe) / 80.0, 1.0)
    penalty = 0.0
    if barrel_pct <= 4:   penalty += 0.6
    elif barrel_pct <= 5: penalty += 0.4
    elif barrel_pct <= 6: penalty += 0.2
    if hr_per_fb <= 6:    penalty += 0.4
    elif hr_per_fb <= 8:  penalty += 0.2
    elif hr_per_fb <= 10: penalty += 0.1
    return round(penalty * sample_weight, 3)


# ── Platoon penalty only ───────────────────────────────────────────────────
# Data shows platoon advantage doesn't help (11.1% vs neutral 15.0%)
# But disadvantage clearly hurts (8.5%)

def compute_platoon_penalty(row: pd.Series) -> tuple:
    batter_hand = str(row.get("batter_hand", "")).strip().upper()
    p_throws    = str(row.get("pitcher_hand", "")).strip().upper()

    penalty = 0.0
    parts   = []

    iso_vs_lhp = safe_float(row.get("vs_lhp_iso", 0))
    iso_vs_rhp = safe_float(row.get("vs_rhp_iso", 0))

    if p_throws == "L":
        iso_vs_this = iso_vs_lhp
        iso_vs_opp  = iso_vs_rhp
        label       = f"{batter_hand}HH vs LHP"
        start_rate  = safe_float(row.get("lhp_start_rate", 1.0), 1.0)
    elif p_throws == "R":
        iso_vs_this = iso_vs_rhp
        iso_vs_opp  = iso_vs_lhp
        label       = f"{batter_hand}HH vs RHP"
        start_rate  = safe_float(row.get("rhp_start_rate", 1.0), 1.0)
    else:
        return 0.0, ""

    has_iso_data = (iso_vs_this > 0 or iso_vs_opp > 0)
    if has_iso_data:
        iso_gap = iso_vs_opp - iso_vs_this
        if iso_gap >= 0.200:
            penalty += 2.0
            parts.append(f"🚨 Severe platoon weakness ({label}) — ISO {iso_vs_this:.3f} vs this hand")
        elif iso_gap >= 0.140:
            penalty += 1.2
            parts.append(f"❌ Platoon disadvantage ({label}) — ISO {iso_vs_this:.3f} vs this hand")
        elif iso_gap >= 0.100:
            penalty += 0.6
            parts.append(f"⚠️ Moderate platoon weakness ({label})")

    if start_rate < 0.50:
        start_penalty = round((0.50 - start_rate) * 3.0, 3)
        penalty      += start_penalty
        parts.append(f"⚠️ Rarely starts vs this hand ({start_rate:.0%})")

    return round(penalty, 3), " | ".join(parts)


# ── Pitch matchup score ────────────────────────────────────────────────────

# League average ISO by pitch type — used for regression to the mean
LEAGUE_AVG_ISO_VS_PITCH = 0.150
LEAGUE_AVG_ISO_VS_PITCH_FULL_SAMPLE = 25  # BBE needed for full trust


def regress_pitch_iso(raw_iso: float, bbe_count: float, league_avg: float = LEAGUE_AVG_ISO_VS_PITCH, full_sample: float = LEAGUE_AVG_ISO_VS_PITCH_FULL_SAMPLE) -> float:
    """Regress pitch-specific ISO toward league average based on sample size."""
    if bbe_count <= 0:
        return league_avg
    weight = min(bbe_count / full_sample, 1.0)
    return round((raw_iso * weight) + (league_avg * (1 - weight)), 3)


def compute_pitch_matchup_score(row: pd.Series) -> tuple:
    scores        = []
    descriptions  = []
    pitch_penalty = 0.0

    for rank in range(1, 4):
        pitch_type = str(row.get(f"top_pitch_{rank}", "")).strip().upper()
        pitch_pct  = safe_float(row.get(f"top_pitch_{rank}_pct", 0))

        if not pitch_type or pitch_type in ("", "NAN", "NONE"):
            continue

        # Get raw values
        raw_batter_iso = safe_float(row.get(f"iso_vs_{pitch_type}", 0))
        batter_hr_rate = safe_float(row.get(f"hr_rate_vs_{pitch_type}", 0))
        batter_barrel  = safe_float(row.get(f"barrel_pct_vs_{pitch_type}", 0))

        # Regress batter ISO toward league average based on BBE count
        batter_bbe    = safe_float(row.get(f"bbe_vs_{pitch_type}", 0))
        batter_iso    = regress_pitch_iso(raw_batter_iso, batter_bbe) if raw_batter_iso > 0 else 0.0
        batter_has    = (raw_batter_iso > 0 or batter_hr_rate > 0) and batter_bbe >= 5

        # ISO is already regressed toward league avg in pitcher_statcast.py
        pitcher_iso    = safe_float(row.get(f"pitcher_iso_allowed_{pitch_type}", 0))
        pitcher_hr     = safe_float(row.get(f"pitcher_hr_rate_allowed_{pitch_type}", 0))
        pitcher_barrel = safe_float(row.get(f"pitcher_barrel_pct_allowed_{pitch_type}", 0))
        pitcher_bbe    = safe_float(row.get(f"pitcher_bbe_vs_{pitch_type}", 0))
        pitcher_has    = (pitcher_iso > 0 or pitcher_hr > 0)

        if not batter_has and not pitcher_has:
            continue

        batter_component  = (batter_iso * 3 + batter_hr_rate / 10 + batter_barrel / 20) if batter_has else 0.0
        pitcher_component = (pitcher_iso * 2 + pitcher_hr / 10 + pitcher_barrel / 20) if pitcher_has else 0.0
        pitch_score       = (batter_component + pitcher_component) * (pitch_pct / 100)
        scores.append(pitch_score)

        if batter_has and batter_iso >= 0.200 and pitch_pct >= 15:
            descriptions.append(f"✅ ISO {raw_batter_iso:.3f} vs {pitch_type} ({int(batter_bbe)} BBE, {pitch_pct:.0f}% usage)")
        if pitcher_has and pitcher_iso >= 0.180 and pitch_pct >= 15:
            bbe_note = f" ({int(pitcher_bbe)} BBE)" if pitcher_bbe > 0 else ""
            descriptions.append(f"✅ Pitcher allows {pitcher_iso:.3f} ISO on {pitch_type}{bbe_note}")

        if batter_has and raw_batter_iso < 0.100 and pitch_pct >= 15:
            pitcher_iso_factor = max(0.0, 1.0 - (pitcher_iso / 0.150)) if pitcher_has else 1.0
            pen = round((0.100 - batter_iso) * (pitch_pct / 100) * 10 * pitcher_iso_factor, 3)
            pitch_penalty = max(pitch_penalty, pen)
            if pen > 0.02:
                descriptions.append(f"⚠️ Weak vs {pitch_type} — ISO {raw_batter_iso:.3f} ({int(batter_bbe)} BBE, {pitch_pct:.0f}% usage)")

    return sum(scores), " + ".join(descriptions), pitch_penalty


# ── BvP score ──────────────────────────────────────────────────────────────

def compute_bvp_score(row: pd.Series) -> tuple:
    bvp_pa     = safe_float(row.get("bvp_pa", 0))
    bvp_iso    = safe_float(row.get("bvp_iso", 0))
    bvp_hr     = safe_float(row.get("bvp_hr_rate", 0))
    bvp_barrel = safe_float(row.get("bvp_barrel_pct", 0))
    bvp_hr_cnt = safe_float(row.get("bvp_hr", 0))

    if bvp_pa < 5:
        return 0.0, ""

    score   = (bvp_iso * 2.0) + (bvp_hr / 10.0) + (bvp_barrel / 20.0)
    hr_note = f", {int(bvp_hr_cnt)} HR" if bvp_hr_cnt > 0 else ""
    desc    = ""

    if bvp_iso >= 0.200:
        desc = f"🔥 Owns this pitcher — ISO {bvp_iso:.3f} in {int(bvp_pa)} PA{hr_note}"
    elif bvp_iso >= 0.150:
        desc = f"✅ Strong vs this pitcher — ISO {bvp_iso:.3f} in {int(bvp_pa)} PA{hr_note}"
    elif bvp_iso <= 0.050 and bvp_pa >= 8:
        desc  = f"⚠️ Struggles vs this pitcher — ISO {bvp_iso:.3f} in {int(bvp_pa)} PA"
        score = -abs(score) * 0.5

    return round(score, 3), desc


# ── Edge calculation ───────────────────────────────────────────────────────

def get_score_tier_hit_rate(score: float) -> float:
    if score >= 13:
        return SCORE_TIER_HIT_RATES["13+"]
    if score >= 12:
        return SCORE_TIER_HIT_RATES["12-13"]
    if score >= 11:
        return SCORE_TIER_HIT_RATES["11-12"]
    if score >= 10:
        return SCORE_TIER_HIT_RATES["10-11"]
    if score >= 9:
        return SCORE_TIER_HIT_RATES["9-10"]
    return SCORE_TIER_HIT_RATES["8.5-9"]


def calc_edge(score: float, odds: float) -> str:
    """
    Edge = estimated hit rate (from score tier) - breakeven rate at given odds.
    Positive = value bet. Negative = skip.
    """
    if odds <= 0:
        return "—"
    hit_rate  = get_score_tier_hit_rate(score)
    breakeven = 100.0 / (odds + 100.0)
    edge      = hit_rate - breakeven
    edge_pct  = round(edge * 100, 1)
    if edge_pct >= 3:
        return f"✅ +{edge_pct}%"
    if edge_pct >= 0:
        return f"➡️ +{edge_pct}%"
    return f"❌ {edge_pct}%"


def american_odds_to_profit(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return odds / 100.0


def assign_confidence(row: pd.Series) -> str:
    batter_bbe  = safe_float(row.get("season_bbe", 0))
    batter_pa   = safe_float(row.get("pa", 0))
    pitcher_bbe = safe_float(row.get("pitcher_bbe_allowed", 0))

    points        = 0
    batter_points = 0

    if batter_bbe >= 60 and batter_pa >= 100:
        points += 2; batter_points = 2
    elif batter_bbe >= 30 and batter_pa >= 50:
        points += 1; batter_points = 1

    if pitcher_bbe >= 80:   points += 2
    elif pitcher_bbe >= 40: points += 1

    if points >= 4 and batter_points >= 1:   return "High"
    elif points >= 2 and batter_points >= 1: return "Medium"
    else:                                     return "Low"


def prepare_combined(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
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
    if "home_team" not in pitchers.columns:    pitchers["home_team"] = ""

    batters = batters.copy()
    batters.columns = [c.strip() for c in batters.columns]
    batters = batters.rename(columns={"team": "batter_team"})

    # ── IL filter ──────────────────────────────────────────────────────────
    if active_roster is not None and not active_roster.empty and "player_id" in active_roster.columns:
        active_ids    = set(active_roster["player_id"].astype(str).str.strip())
        batter_id_col = "batter_id" if "batter_id" in batters.columns else "batter"
        if batter_id_col in batters.columns:
            before = len(batters)
            batters["_aid"] = batters[batter_id_col].astype(str).str.strip()
            batters = batters[batters["_aid"].isin(active_ids)].drop(columns=["_aid"])
            print(f"IL filter: {before - len(batters)} removed, {len(batters)} remaining")
    else:
        print("Active roster unavailable — IL filter skipped.")

    # ── Lineup filter ──────────────────────────────────────────────────────
    if not lineups.empty and "player_id" in lineups.columns and "team" in lineups.columns:
        confirmed_teams = set(lineups["team"].unique())
        lineup_ids      = set(lineups["player_id"].astype(str).str.strip())
        batter_id_col   = "batter_id" if "batter_id" in batters.columns else "batter"
        if batter_id_col in batters.columns:
            before = len(batters)
            batters["_bid"] = batters[batter_id_col].astype(str).str.strip()
            def lineup_filter(row):
                team = str(row.get("batter_team", "")).strip()
                if team not in confirmed_teams:
                    return True
                return row["_bid"] in lineup_ids
            batters = batters[batters.apply(lineup_filter, axis=1)].drop(columns=["_bid"])
            print(f"Lineup filter: {before - len(batters)} removed, {len(batters)} remaining")
    else:
        print("No confirmed lineups — lineup filter skipped.")

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
        "pitcher_bbe_allowed", "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
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
            combined["_bm"] = combined[batter_id_col].apply(lambda x: safe_float(x, 0))
            combined["_pm"] = combined["pitcher_id_num"].apply(lambda x: safe_float(x, 0))
            bvp["_bm"]      = bvp["batter_id"].apply(lambda x: safe_float(x, 0))
            bvp["_pm"]      = bvp["pitcher_id"].apply(lambda x: safe_float(x, 0))
            bvp_cols = [c for c in ["_bm", "_pm", "bvp_pa", "bvp_hr", "bvp_iso", "bvp_barrel_pct", "bvp_hr_rate"] if c in bvp.columns]
            combined = combined.merge(bvp[bvp_cols], on=["_bm", "_pm"], how="left")
            combined = combined.drop(columns=["_bm", "_pm"])
            print(f"BvP matched for {combined['bvp_pa'].notna().sum()} pairs.")
    else:
        print("No BvP data to merge.")

    # ── Park merge (kept for display, not scoring) ─────────────────────────
    parks = parks.copy()
    parks.columns = [c.strip() for c in parks.columns]
    parks = parks.rename(columns={"team": "park_home_team"})
    if not parks.empty:
        park_cols = [c for c in ["park_home_team", "park_hr_factor", "park_name"] if c in parks.columns]
        combined  = combined.merge(parks[park_cols], left_on="home_team", right_on="park_home_team", how="left")
    else:
        combined["park_hr_factor"] = 100.0
        combined["park_name"]      = ""

    # ── BA filter ──────────────────────────────────────────────────────────
    if "batting_avg" in combined.columns:
        combined["batting_avg"] = combined["batting_avg"].apply(safe_float)
        before   = len(combined)
        combined = combined[combined["batting_avg"] >= MIN_BATTING_AVG].copy()
        print(f"BA filter: {before - len(combined)} removed, {len(combined)} remaining")

    if combined.empty:
        print("No batters remaining after filters.")
        return pd.DataFrame()

    # ── Coerce numeric columns ─────────────────────────────────────────────
    num_cols = [
        "barrel_pct_7d", "barrel_pct_5d", "barrel_pct_10d", "season_barrel_pct",
        "hr_per_pa", "hr_per_fb", "iso", "pa", "season_bbe",
        "bbe_5d", "bbe_7d", "bbe_10d",
        "vs_lhp_iso", "vs_rhp_iso", "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "vs_lhp_hr_rate", "vs_rhp_hr_rate",
        "lhp_start_rate", "rhp_start_rate",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_bbe_allowed", "pitcher_bf",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
        "top_pitch_1_pct", "top_pitch_2_pct", "top_pitch_3_pct",
        "bvp_pa", "bvp_hr", "bvp_iso", "bvp_barrel_pct", "bvp_hr_rate",
        "batting_avg", "park_hr_factor",
    ]
    dynamic_cols = [c for c in combined.columns
                    if c.startswith("pitcher_iso_allowed_")
                    or c.startswith("pitcher_hr_rate_allowed_")
                    or c.startswith("pitcher_barrel_pct_allowed_")]
    num_cols += dynamic_cols

    for col in num_cols:
        if col in combined.columns: combined[col] = combined[col].apply(safe_float)
        else:                       combined[col] = 0.0

    combined = combined.copy()

    # ── Compute components ─────────────────────────────────────────────────
    platoon_results = combined.apply(compute_platoon_penalty, axis=1)
    combined["platoon_penalty"] = platoon_results.apply(lambda x: x[0])
    combined["platoon_desc"]    = platoon_results.apply(lambda x: x[1])

    pitch_results = combined.apply(compute_pitch_matchup_score, axis=1)
    combined["pitch_matchup_score"] = pitch_results.apply(lambda x: x[0])
    combined["pitch_matchup_desc"]  = pitch_results.apply(lambda x: x[1])
    combined["pitch_penalty"]       = pitch_results.apply(lambda x: x[2])

    bvp_results = combined.apply(compute_bvp_score, axis=1)
    combined["bvp_score"] = bvp_results.apply(lambda x: x[0])
    combined["bvp_desc"]  = bvp_results.apply(lambda x: x[1])

    combined["confidence"] = combined.apply(assign_confidence, axis=1)

    # ── Cap context scores ─────────────────────────────────────────────────
    combined["pitch_matchup_capped"] = combined["pitch_matchup_score"].clip(0.0, 1.0)
    combined["bvp_capped"]           = combined["bvp_score"].clip(-0.5, 1.0)
    combined["total_penalty"]        = (
        combined["platoon_penalty"] + combined["pitch_penalty"]
    ).clip(0, 3.0)

    # ── Final score ────────────────────────────────────────────────────────
    # Components weighted by separator strength, noise removed
    combined["score"] = (
        # Barrel% windows — strongest predictors
        combined.apply(lambda r: score_barrel_pct_7d(r["barrel_pct_7d"], r["bbe_7d"]), axis=1) +
        combined.apply(lambda r: score_barrel_pct_5d(r["barrel_pct_5d"], r["bbe_5d"]), axis=1) +
        combined.apply(lambda r: score_barrel_pct_10d(r["barrel_pct_10d"], r["bbe_10d"]), axis=1) +
        combined.apply(lambda r: score_season_barrel_pct(r["season_barrel_pct"], r["pa"]), axis=1) +
        # Power metrics
        combined.apply(lambda r: score_hr_per_fb(r["hr_per_fb"], r["pa"]), axis=1) +
        combined.apply(lambda r: score_hr_per_pa(r["hr_per_pa"], r["pa"]), axis=1) +
        combined.apply(lambda r: score_iso(r["iso"], r["pa"]), axis=1) +
        # Pitcher side — light weight
        combined["pitcher_barrel_pct"].apply(score_pitcher_barrel_pct) +
        combined["pitcher_hr_per_fb"].apply(score_pitcher_hr_per_fb) -
        combined.apply(lambda r: score_pitcher_quality_penalty(
            r["pitcher_barrel_pct"], r["pitcher_hr_per_fb"], r["pitcher_bbe_allowed"]
        ), axis=1) +
        # Context — capped
        combined["pitch_matchup_capped"] * PITCH_MATCHUP_WEIGHT +
        combined["bvp_capped"]           * BVP_WEIGHT -
        combined["total_penalty"]
    ).round(3)

    return combined


def build_reason(row) -> str:
    reasons = []

    barrel_7d = safe_float(row.get("barrel_pct_7d"))
    if barrel_7d >= 15:
        reasons.append(f"🔥 {barrel_7d:.1f}% barrel rate last 7 days")

    barrel_10d = safe_float(row.get("barrel_pct_10d"))
    if barrel_10d >= 15:
        reasons.append(f"🔥 {barrel_10d:.1f}% barrel rate last 10 days")

    season_barrel = safe_float(row.get("season_barrel_pct"))
    if season_barrel >= 11:
        reasons.append(f"💣 Elite season barrel% ({season_barrel:.1f}%)")

    bvp_desc = str(row.get("bvp_desc", ""))
    if bvp_desc:
        reasons.append(bvp_desc)

    hr_pa = safe_float(row.get("hr_per_pa"))
    if hr_pa >= 4:
        reasons.append(f"💪 {hr_pa:.1f}% HR/PA")

    hr_fb = safe_float(row.get("hr_per_fb"))
    if hr_fb >= 15:
        reasons.append(f"🚀 {hr_fb:.1f}% HR/FB")

    iso = safe_float(row.get("iso"))
    if iso >= 0.200:
        reasons.append(f"⚡ ISO {iso:.3f}")

    p_barrel = safe_float(row.get("pitcher_barrel_pct"))
    if p_barrel >= 11:
        reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")
    elif p_barrel <= 4:
        reasons.append(f"⚠️ Tough pitcher — {p_barrel:.1f}% barrels allowed")

    platoon_desc = str(row.get("platoon_desc", ""))
    if platoon_desc:
        reasons.append(platoon_desc)

    pitch_desc = str(row.get("pitch_matchup_desc", ""))
    if pitch_desc:
        reasons.append(f"🎳 {pitch_desc}")

    park_factor = safe_float(row.get("park_hr_factor"), 100)
    park_name   = str(row.get("park_name", ""))
    if park_factor >= 110:
        reasons.append(f"🏟️ HR-friendly park ({park_name}, {park_factor:.0f})")
    elif park_factor <= 85:
        reasons.append(f"🏟️ Pitcher-friendly park ({park_name}, {park_factor:.0f})")

    if not reasons:
        reasons.append("Solid across multiple power factors")

    return " | ".join(reasons)


def build_main_picks(combined: pd.DataFrame, odds_df: pd.DataFrame = None) -> tuple:
    combined = combined.copy()
    combined["reason"] = combined.apply(build_reason, axis=1)

    # ── Odds lookup ────────────────────────────────────────────────────────
    odds_lookup = {}
    if odds_df is not None and not odds_df.empty and "player_name_norm" in odds_df.columns and "consensus_odds" in odds_df.columns:
        for _, row in odds_df.iterrows():
            norm = str(row["player_name_norm"]).strip()
            try:
                odds_lookup[norm] = int(float(row["consensus_odds"]))
            except (ValueError, TypeError):
                pass
        print(f"Odds lookup built: {len(odds_lookup)} players")
    else:
        print("No odds data available")

    combined["consensus_odds"] = combined["player_name"].apply(
        lambda n: odds_lookup.get(normalize_name(str(n)), None)
    )

    # ── Filter: score floor + hard odds cap ────────────────────────────────
    filtered = combined[combined["score"] >= MIN_SCORE_FLOOR].copy()

    if not filtered.empty and odds_lookup:
        filtered = filtered[filtered.apply(
            lambda row: (
                row.get("consensus_odds") is not None and
                str(row.get("consensus_odds", "")).strip() not in ("", "nan", "None") and
                int(float(row["consensus_odds"])) >= MIN_ODDS and
                int(float(row["consensus_odds"])) <= MAX_ODDS
            ),
            axis=1
        )].copy()

    filtered = filtered.sort_values("score", ascending=False).reset_index(drop=True)

    # ── Diversity cap + TOP_N ──────────────────────────────────────────────
    selected    = []
    team_counts = {}
    game_counts = {}

    for _, row in filtered.iterrows():
        if len(selected) >= TOP_N:
            break
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
        print(f"No qualifying picks (score ≥{MIN_SCORE_FLOOR}, odds ≤+{MAX_ODDS}).")
        return pd.DataFrame(), odds_lookup

    picks = pd.DataFrame(selected).reset_index(drop=True)
    picks["rank"] = range(1, len(picks) + 1)

    # ── Add edge column ────────────────────────────────────────────────────
    picks["edge"] = picks.apply(
        lambda r: calc_edge(safe_float(r["score"]), safe_float(r.get("consensus_odds", 0))),
        axis=1
    )

    print(f"Qualifying picks: {len(picks)} (score ≥{MIN_SCORE_FLOOR}, odds ≤+{MAX_ODDS})")
    pos_edge = picks["edge"].str.startswith("✅").sum()
    neutral_edge = picks["edge"].str.startswith("➡️").sum()
    neg_edge = picks["edge"].str.startswith("❌").sum()
    print(f"Edge breakdown: {pos_edge} positive ✅ | {neutral_edge} neutral ➡️ | {neg_edge} negative ❌")

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
        "edge":                       "Edge",
        "confidence":                 "Confidence",
        "reason":                     "Key Reasons",
        "batting_avg":                "Batting Avg",
        "barrel_pct_7d":              "Barrel% (7d)",
        "barrel_pct_5d":              "Barrel% (5d)",
        "barrel_pct_10d":             "Barrel% (10d)",
        "season_barrel_pct":          "Barrel% (Season)",
        "hr_per_pa":                  "HR/PA%",
        "hr_per_fb":                  "HR/FB%",
        "iso":                        "ISO",
        "vs_lhp_iso":                 "ISO vs LHP",
        "vs_rhp_iso":                 "ISO vs RHP",
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
        print("Picks_Log is empty.")
        return

    today_str = date.today().strftime("%Y-%m-%d")
    pending   = existing[
        (existing["hit_hr"] == "Pending") &
        (existing["date"] != today_str) &
        (existing["date"] != "")
    ].copy()

    if pending.empty:
        print("No pending picks to resolve.")
        return

    pending_dates = sorted(pending["date"].unique())
    print(f"Resolving {len(pending)} pending picks across {len(pending_dates)} dates...")

    try:
        hr_df = statcast(start_dt=pending_dates[0], end_dt=pending_dates[-1])
        if hr_df is None or hr_df.empty:
            print("Statcast returned empty — cannot resolve.")
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

    yes_count = sum(1 for d in pending_dates for _, r in existing.iterrows()
                    if r["date"] == d and r["hit_hr"] == "Yes")
    no_count  = sum(1 for d in pending_dates for _, r in existing.iterrows()
                    if r["date"] == d and r["hit_hr"] == "No")
    print(f"Resolved {resolved_count} picks. Yes: {yes_count} | No: {no_count} (pending dates only)")
    ws.clear()
    ws.update([existing.columns.tolist()] + existing.astype(str).values.tolist())
    print("Picks_Log updated.")


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
                "date":                 today_str,
                "rank":                 str(row.get("Rank", "")),
                "player_name":          str(row.get("Batter", "")),
                "team":                 str(row.get("Team", "")),
                "pitcher_name":         str(row.get("Opposing Pitcher", "")),
                "pitcher_hand":         str(row.get("Throws", "")),
                "pitcher_team":         str(row.get("Pitcher Team", "")),
                "park_name":            str(row.get("Park", "")),
                "hr_score":             str(row.get("HR Score", "")),
                "consensus_odds":       str(row.get("Consensus Odds", "")),
                "edge":                 str(row.get("Edge", "")),
                "confidence":           str(row.get("Confidence", "")),
                "hit_hr":               "Pending",
                "odds":                 "",
                "bet_placed":           "",
                "batting_avg":          str(row.get("Batting Avg", "")),
                "barrel_pct_7d":        str(row.get("Barrel% (7d)", "")),
                "barrel_pct_5d":        str(row.get("Barrel% (5d)", "")),
                "barrel_pct_10d":       str(row.get("Barrel% (10d)", "")),
                "season_barrel_pct":    str(row.get("Barrel% (Season)", "")),
                "hr_per_pa":            str(row.get("HR/PA%", "")),
                "hr_per_fb":            str(row.get("HR/FB%", "")),
                "iso":                  str(row.get("ISO", "")),
                "iso_vs_lhp":           str(row.get("ISO vs LHP", "")),
                "iso_vs_rhp":           str(row.get("ISO vs RHP", "")),
                "pitcher_barrel_pct":   str(row.get("Pitcher Barrel% Allowed", "")),
                "pitcher_hr_per_fb":    str(row.get("Pitcher HR/FB% Allowed", "")),
                "top_pitch_1":          str(row.get("Top Pitch 1", "")),
                "top_pitch_1_pct":      str(row.get("Top Pitch 1 %", "")),
                "top_pitch_2":          str(row.get("Top Pitch 2", "")),
                "top_pitch_2_pct":      str(row.get("Top Pitch 2 %", "")),
                "top_pitch_3":          str(row.get("Top Pitch 3", "")),
                "top_pitch_3_pct":      str(row.get("Top Pitch 3 %", "")),
                "park_hr_factor":       str(row.get("Park HR Factor", "")),
                "platoon_matchup":      str(row.get("Platoon Matchup", "")),
                "pitch_matchup":        str(row.get("Pitch Matchup", "")),
                "bvp_pa":               str(row.get("BvP PA", "")),
                "bvp_hr":               str(row.get("BvP HR", "")),
                "bvp_iso":              str(row.get("BvP ISO", "")),
                "bvp_notes":            str(row.get("BvP Notes", "")),
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
        print("Picks_Log not found — skipping scorecard.")
        return

    if picks_log.empty:
        print("Picks_Log empty — skipping scorecard.")
        return

    scored = picks_log[picks_log["hit_hr"].isin(["Yes", "No"])].copy()
    if scored.empty:
        print("No scored picks yet — skipping scorecard.")
        return

    scored["hit_hr_bool"] = scored["hit_hr"] == "Yes"
    scored["rank"]        = pd.to_numeric(scored["rank"], errors="coerce")
    scored["date"]        = pd.to_datetime(scored["date"], errors="coerce")
    scored["hr_score"]    = pd.to_numeric(scored["hr_score"], errors="coerce")
    scored["odds_zone"]   = pd.to_numeric(scored["consensus_odds"], errors="coerce")

    def is_bet_placed(x) -> bool:
        s = str(x).replace("$", "").strip().lower()
        if s in ("", "no", "nan"): return False
        try:
            float(s); return True
        except ValueError:
            return s == "yes"

    bet_picks = scored[
        scored.get("bet_placed", pd.Series("", index=scored.index)).apply(is_bet_placed) &
        (scored.get("odds", pd.Series("", index=scored.index)).astype(str).str.strip() != "")
    ].copy()

    if not bet_picks.empty:
        bet_picks["odds_num"]      = bet_picks["odds"].apply(lambda x: safe_float(str(x).replace("+", "").strip(), 0))
        bet_picks["profit_if_win"] = bet_picks["odds_num"].apply(american_odds_to_profit)
        def parse_bet(x) -> float:
            s = str(x).replace("$", "").strip().lower()
            if s in ("", "yes", "no"): return 1.0
            return safe_float(s, 1.0)
        bet_picks["bet_size"]    = bet_picks["bet_placed"].apply(parse_bet)
        bet_picks["unit_result"] = bet_picks.apply(
            lambda r: (r["profit_if_win"] * r["bet_size"]) if r["hit_hr_bool"] else -r["bet_size"], axis=1
        )

    perf_rows  = []
    score_rows = []
    roi_rows   = []

    def add_perf(label, sub, bold=False, header=False):
        if header:
            perf_rows.append({"label": label, "total": "", "hits": "", "rate": "", "_header": True})
            return
        if sub.empty: return
        total = len(sub)
        hits  = int(sub["hit_hr_bool"].sum())
        perf_rows.append({"label": label, "total": total, "hits": hits,
                          "rate": round(hits/total*100, 1), "_bold": bold})

    def add_score(label, sub, bold=False, header=False):
        if header:
            score_rows.append({"label": label, "total": "", "hits": "", "rate": "", "avg": "", "_header": True})
            return
        if sub.empty: return
        total    = len(sub)
        hits     = int(sub["hit_hr_bool"].sum())
        avg_scr  = round(sub["hr_score"].mean(), 2) if not sub["hr_score"].isna().all() else 0.0
        score_rows.append({"label": label, "total": total, "hits": hits,
                           "rate": round(hits/total*100, 1), "avg": avg_scr, "_bold": bold})

    def add_roi(label, sub, bold=False, header=False):
        if header:
            roi_rows.append({"label": label, "bets": "", "hits": "", "rate": "", "profit": "", "roi": "", "_header": True})
            return
        if sub.empty: return
        total   = len(sub)
        hits    = int(sub["hit_hr_bool"].sum())
        profit  = round(sub["unit_result"].sum(), 2)
        roi     = round(profit/total*100, 1) if total > 0 else 0.0
        roi_rows.append({"label": label, "bets": total, "hits": hits,
                         "rate": round(hits/total*100, 1) if total > 0 else 0.0,
                         "profit": f"+{profit}" if profit >= 0 else str(profit),
                         "roi": f"+{roi}%" if roi >= 0 else f"{roi}%",
                         "_bold": bold, "_roi_val": roi})

    add_perf("🏆  Overall", scored, bold=True)
    add_perf("── By Rank ──", pd.DataFrame(), header=True)
    for rank in range(1, 16):
        sub = scored[scored["rank"] == rank]
        if not sub.empty: add_perf(f"   Rank {rank}", sub)

    add_perf("── By Confidence ──", pd.DataFrame(), header=True)
    for tier in ["High", "Medium", "Low"]:
        sub = scored[scored["confidence"] == tier]
        if not sub.empty: add_perf(f"   {tier}", sub)

    add_perf("── By Odds Zone ──", pd.DataFrame(), header=True)
    for label, sub in [
        ("   +301 to +399", scored[(scored["odds_zone"] >= 301) & (scored["odds_zone"] <= 399)]),
        ("   +400 to +499", scored[(scored["odds_zone"] >= 400) & (scored["odds_zone"] <= 499)]),
        ("   +500 to +599", scored[(scored["odds_zone"] >= 500) & (scored["odds_zone"] <= 599)]),
        ("   +600 to +699", scored[(scored["odds_zone"] >= 600) & (scored["odds_zone"] <= 699)]),
    ]:
        if not sub.empty: add_perf(label, sub)

    add_perf("── By Edge ──", pd.DataFrame(), header=True)
    if "edge" in scored.columns:
        for label, sub in [
            ("   ✅ Positive Edge", scored[scored["edge"].astype(str).str.startswith("✅")]),
            ("   ➡️ Neutral Edge",  scored[scored["edge"].astype(str).str.startswith("➡️")]),
            ("   ❌ Negative Edge", scored[scored["edge"].astype(str).str.startswith("❌")]),
        ]:
            if not sub.empty: add_perf(label, sub)

    add_perf("── Rolling ──", pd.DataFrame(), header=True)
    max_date = scored["date"].max()
    add_perf("   Last 7 Days",  scored[scored["date"] >= max_date - pd.Timedelta(days=7)])
    add_perf("   Last 30 Days", scored[scored["date"] >= max_date - pd.Timedelta(days=30)])

    add_score("📈  Score Tier Analysis", scored, bold=True)
    add_score("── By Score Tier ──", pd.DataFrame(), header=True)
    for label, sub in [
        ("   13+",     scored[scored["hr_score"] >= 13]),
        ("   12-13",   scored[(scored["hr_score"] >= 12) & (scored["hr_score"] < 13)]),
        ("   11-12",   scored[(scored["hr_score"] >= 11) & (scored["hr_score"] < 12)]),
        ("   10-11",   scored[(scored["hr_score"] >= 10) & (scored["hr_score"] < 11)]),
        ("   9-10",    scored[(scored["hr_score"] >= 9)  & (scored["hr_score"] < 10)]),
        ("   8.5-9",   scored[(scored["hr_score"] >= 8.5) & (scored["hr_score"] < 9)]),
    ]:
        if not sub.empty: add_score(label, sub)

    if not bet_picks.empty:
        bet_picks["date"] = pd.to_datetime(bet_picks["date"], errors="coerce")
        add_roi("💵  All Bets", bet_picks, bold=True)
        add_roi("── By Edge ──", pd.DataFrame(), header=True)
        if "edge" in bet_picks.columns:
            for label, sub in [
                ("   ✅ Positive Edge", bet_picks[bet_picks["edge"].astype(str).str.startswith("✅")]),
                ("   ➡️ Neutral Edge",  bet_picks[bet_picks["edge"].astype(str).str.startswith("➡️")]),
                ("   ❌ Negative Edge", bet_picks[bet_picks["edge"].astype(str).str.startswith("❌")]),
            ]:
                if not sub.empty: add_roi(label, sub)
        add_roi("── Rolling ──", pd.DataFrame(), header=True)
        max_bet = bet_picks["date"].max()
        add_roi("   Last 7 Days",  bet_picks[bet_picks["date"] >= max_bet - pd.Timedelta(days=7)])
        add_roi("   Last 30 Days", bet_picks[bet_picks["date"] >= max_bet - pd.Timedelta(days=30)])

    try:
        ws = sh.worksheet("Scorecard")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Scorecard", rows=200, cols=10)

    all_values = []
    all_values.append(["📊  MODEL PERFORMANCE", "", "", ""])
    all_values.append(["Category", "Total Picks", "HR Count", "Hit Rate %"])
    for r in perf_rows:
        all_values.append([r["label"], str(r.get("total", "")), str(r.get("hits", "")),
                           f"{r['rate']}%" if r.get("rate", "") != "" else ""])

    all_values.append(["", "", "", ""])
    all_values.append(["📈  SCORE TIER ANALYSIS", "", "", "", ""])
    all_values.append(["Category", "Total Picks", "HR Count", "Hit Rate %", "Avg HR Score"])
    for r in score_rows:
        all_values.append([r["label"], str(r.get("total", "")), str(r.get("hits", "")),
                           f"{r['rate']}%" if r.get("rate", "") != "" else "",
                           str(r.get("avg", ""))])

    if roi_rows:
        all_values.append(["", "", "", "", ""])
        all_values.append(["💰  BETTING ROI", "", "", "", "", ""])
        all_values.append(["Category", "Bets", "HR Count", "Hit Rate %", "Units Profit", "ROI %"])
        for r in roi_rows:
            all_values.append([r["label"], str(r.get("bets", "")), str(r.get("hits", "")),
                               f"{r['rate']}%" if r.get("rate", "") != "" else "",
                               str(r.get("profit", "")), str(r.get("roi", ""))])

    ws.update(all_values)
    print("Scorecard updated.")


def clean_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].apply(lambda x: "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else x)
    return df.fillna("")


def write_picks_to_sheet(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame) -> int:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("Top_HR_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_HR_Picks", rows=100, cols=55)

    if picks.empty:
        ws.update([[f"No qualifying picks today — score ≥{MIN_SCORE_FLOOR}, odds ≤+{MAX_ODDS}"]])
        return 0

    picks_clean = clean_for_sheets(picks)
    all_values  = [picks_clean.columns.tolist()]
    for _, row in picks_clean.iterrows():
        all_values.append(row.astype(str).tolist())

    ws.update(all_values)
    print(f"Written {len(picks_clean)} picks to Top_HR_Picks")
    return len(picks_clean)


def write_last_run_timestamp(gc: gspread.Client, sheet_id: str) -> None:
    et     = pytz.timezone("America/New_York")
    now_et = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")
    sh     = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("Top_HR_Picks")
    except gspread.WorksheetNotFound:
        return
    ws.insert_row([f"⏱  Last Run: {now_et}"], index=1)
    print(f"Timestamp written: {now_et}")


def format_picks_sheet(gc: gspread.Client, sheet_id: str, row_count: int) -> None:
    if row_count == 0:
        return

    print("Applying formatting...")
    sh    = gc.open_by_key(sheet_id)
    ws    = sh.worksheet("Top_HR_Picks")
    ws_id = ws.id

    main_cols  = 45
    total_rows = row_count + 2
    reqs       = []

    # Clear any stale merges from previous runs first
    reqs.append({"unmergeCells": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows + 5,
                  "startColumnIndex": 0, "endColumnIndex": 55},
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total_rows,
                  "startColumnIndex": 0, "endColumnIndex": 55},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE", "wrapStrategy": "WRAP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": main_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_ACCENT_DIM,
            "textFormat": {"foregroundColor": COLOR_ACCENT, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 11},
            "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
    }})

    for i in range(row_count):
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2,
                      "startColumnIndex": 0, "endColumnIndex": main_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)",
        }})

    medals = [
        (1, {"red": 0.18, "green": 0.14, "blue": 0.00}, COLOR_GOLD),
        (2, {"red": 0.14, "green": 0.14, "blue": 0.14}, COLOR_SILVER),
        (3, {"red": 0.16, "green": 0.10, "blue": 0.04}, COLOR_BRONZE),
    ]
    for rank, bg, fg in medals:
        if row_count >= rank:
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": rank, "endRowIndex": rank + 1,
                          "startColumnIndex": 0, "endColumnIndex": main_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {"foregroundColor": fg, "bold": True, "fontSize": 10},
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
            }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": row_count + 1,
                  "startColumnIndex": 0, "endColumnIndex": 1},
        "cell": {"userEnteredFormat": {
            "textFormat": {"bold": True, "fontSize": 14, "foregroundColor": COLOR_ACCENT},
            "horizontalAlignment": "CENTER",
        }},
        "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
    }})

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": row_count + 1,
                  "startColumnIndex": 8, "endColumnIndex": 9},
        "cell": {"userEnteredFormat": {
            "textFormat": {"bold": True, "fontSize": 12, "foregroundColor": COLOR_GREEN},
            "horizontalAlignment": "CENTER",
        }},
        "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
    }})

    edge_col_idx = 10
    for i in range(row_count):
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2,
                      "startColumnIndex": edge_col_idx, "endColumnIndex": edge_col_idx + 1},
            "cell": {"userEnteredFormat": {
                "textFormat": {"bold": True, "fontFamily": "Roboto"},
                "horizontalAlignment": "CENTER",
            }},
            "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}},
        "fields": "gridProperties.frozenRowCount",
    }})

    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 0, "endIndex": 1},
        "properties": {"pixelSize": 36}, "fields": "pixelSize",
    }})
    reqs.append({"updateDimensionProperties": {
        "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": 1, "endIndex": total_rows},
        "properties": {"pixelSize": 58}, "fields": "pixelSize",
    }})

    col_widths = [50, 160, 50, 55, 175, 55, 75, 175, 75, 95, 90, 80, 380,
                  90, 90, 90, 90, 75, 75, 65, 80, 80, 75, 75, 75, 100,
                  100, 75, 75, 100, 100, 130, 130, 90, 90, 90, 65, 65, 65,
                  100, 75, 75, 75, 75, 75]
    for i, width in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS",
                      "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": width}, "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "tabColorStyle": {"rgbColor": COLOR_ACCENT}},
        "fields": "tabColorStyle",
    }})

    try:
        sh.batch_update({"requests": reqs})
        print("Formatting applied.")
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
        print("No scored players to log.")
        return

    sorted_df = combined.sort_values("score", ascending=False).reset_index(drop=True)
    sorted_df["all_scores_rank"] = range(1, len(sorted_df) + 1)

    new_rows = []
    for _, row in sorted_df.iterrows():
        new_rows.append({
            "date":              today_str,
            "rank":              str(row.get("all_scores_rank", "")),
            "player_name":       str(row.get("player_name", "")),
            "team":              str(row.get("batter_team", "")),
            "pitcher_name":      str(row.get("opp_pitcher_name", "")),
            "pitcher_hand":      str(row.get("pitcher_hand", "")),
            "hr_score":          str(row.get("score", "")),
            "consensus_odds":    str(row.get("consensus_odds", "") if "consensus_odds" in row.index else ""),
            "barrel_pct_7d":     str(row.get("barrel_pct_7d", "")),
            "barrel_pct_5d":     str(row.get("barrel_pct_5d", "")),
            "barrel_pct_10d":    str(row.get("barrel_pct_10d", "")),
            "season_barrel_pct": str(row.get("season_barrel_pct", "")),
            "hr_per_pa":         str(row.get("hr_per_pa", "")),
            "hr_per_fb":         str(row.get("hr_per_fb", "")),
            "iso":               str(row.get("iso", "")),
            "pitcher_barrel_pct":str(row.get("pitcher_barrel_pct", "")),
            "pitcher_hr_per_fb": str(row.get("pitcher_hr_per_fb", "")),
            "pitch_matchup":     str(row.get("pitch_matchup_desc", "")),
            "pitch_matchup_score":str(row.get("pitch_matchup_score", "")),
            "platoon_matchup":   str(row.get("platoon_desc", "")),
            "park_hr_factor":    str(row.get("park_hr_factor", "")),
            "hit_hr":            "Pending",
        })

    if not new_rows:
        print("No rows to log.")
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
    bvp           = read_sheet(gc, sheet_id, "BvP_History")
    lineups       = read_sheet(gc, sheet_id, "Confirmed_Lineups")
    active_roster = read_sheet(gc, sheet_id, "Active_Rosters")
    odds_df       = read_sheet(gc, sheet_id, "HR_Odds")

    print(f"Batters: {len(batters)} rows")
    print(f"Pitchers: {len(pitchers)} rows")
    print(f"Parks: {len(parks)} rows")
    print(f"BvP History: {len(bvp)} rows")
    print(f"Confirmed Lineups: {len(lineups)} rows")
    print(f"Active Roster: {len(active_roster)} rows")
    print(f"HR Odds: {len(odds_df)} rows")

    # Weather removed — -25% separator, actively hurts prediction
    combined = prepare_combined(batters, pitchers, parks, bvp, lineups, active_roster)

    if combined.empty:
        print("WARNING: No combined data.")
        return

    picks, odds_lookup = build_main_picks(combined, odds_df)

    print(f"\nTop HR Picks (score ≥{MIN_SCORE_FLOOR}, odds ≤+{MAX_ODDS}, top {TOP_N}):")
    if not picks.empty:
        print(picks[["Rank", "Batter", "Bats", "Team", "Opposing Pitcher",
                     "HR Score", "Consensus Odds", "Edge", "Confidence"]].to_string(index=False))
    else:
        print("No qualifying picks today.")

    row_count = write_picks_to_sheet(gc, sheet_id, picks)
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
