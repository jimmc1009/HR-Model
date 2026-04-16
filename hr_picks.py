import os
import json
from datetime import date
from typing import Dict

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

WEIGHTS = {
    "barrel_pct_7d":            2.0,
    "hr_per_pa":                1.8,
    "hr_per_fb":                1.5,
    "iso":                      1.2,
    "avg_ev_7d":                1.0,
    "hard_hit_pct_7d":          0.8,
    "pitcher_barrel_pct":       1.5,
    "pitcher_hr_per_fb":        1.5,
    "pitcher_hard_hit_pct":     0.8,
    "park_hr_factor":           0.5,
    "pitcher_quality_penalty":  1.2,
}

PLATOON_BONUS_WEIGHT = 0.8
PITCH_MATCHUP_WEIGHT = 1.2
WEATHER_WEIGHT = 0.4

ISO_GAP_SMALL    = 0.030
ISO_GAP_MEDIUM   = 0.060
ISO_GAP_LARGE    = 0.100
ISO_GAP_SEVERE   = 0.150

COLOR_BG           = {"red": 0.114, "green": 0.114, "blue": 0.114}
COLOR_BG_ALT       = {"red": 0.149, "green": 0.149, "blue": 0.149}
COLOR_HEADER       = {"red": 0.067, "green": 0.067, "blue": 0.067}
COLOR_ACCENT       = {"red": 0.114, "green": 0.533, "blue": 0.898}
COLOR_ACCENT_DIM   = {"red": 0.055, "green": 0.180, "blue": 0.318}
COLOR_WHITE        = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD         = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_SILVER       = {"red": 0.753, "green": 0.753, "blue": 0.753}
COLOR_BRONZE       = {"red": 0.804, "green": 0.498, "blue": 0.196}
COLOR_GREEN        = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_ORANGE       = {"red": 0.980, "green": 0.502, "blue": 0.059}
COLOR_RED          = {"red": 0.906, "green": 0.298, "blue": 0.235}
COLOR_EV_HEADER    = {"red": 0.100, "green": 0.100, "blue": 0.100}


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def read_sheet(gc: gspread.Client, sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
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


def compute_pitch_matchup_score(row: pd.Series) -> tuple:
    scores = []
    descriptions = []

    for rank in range(1, 4):
        pitch_type = str(row.get(f"top_pitch_{rank}", "")).strip().upper()
        pitch_pct = safe_float(row.get(f"top_pitch_{rank}_pct", 0))

        if not pitch_type or pitch_type in ("", "NAN", "NONE"):
            continue

        iso = safe_float(row.get(f"iso_vs_{pitch_type}", 0))
        hr_rate = safe_float(row.get(f"hr_rate_vs_{pitch_type}", 0))
        barrel = safe_float(row.get(f"barrel_pct_vs_{pitch_type}", 0))

        if iso > 0 or hr_rate > 0:
            pitch_score = (iso * 3 + hr_rate / 10 + barrel / 20) * (pitch_pct / 100)
            scores.append(pitch_score)
            if iso >= 0.200 and pitch_pct >= 20:
                descriptions.append(
                    f"ISO {iso:.3f} vs {pitch_type} ({pitch_pct:.0f}% usage)"
                )

    total_score = sum(scores)
    desc = " + ".join(descriptions) if descriptions else ""
    return total_score, desc


def compute_platoon_score(row: pd.Series) -> tuple:
    """
    Full two-sided platoon matchup using ISO gap analysis + HR/9 by handedness.
    Returns (platoon_score, description, penalty).
    """
    batter_hand = str(row.get("batter_hand", "")).strip().upper()
    p_throws = str(row.get("pitcher_hand", "")).strip().upper()

    score = 0.0
    penalty = 0.0
    parts = []

    iso_vs_lhp = safe_float(row.get("vs_lhp_iso", 0))
    iso_vs_rhp = safe_float(row.get("vs_rhp_iso", 0))

    if p_throws == "L":
        iso_vs_this = iso_vs_lhp
        iso_vs_opp = iso_vs_rhp
        batter_barrel_vs = safe_float(row.get("vs_lhp_barrel_pct", 0))
        batter_hr_vs = safe_float(row.get("vs_lhp_hr_rate", 0))
        matchup_label = f"{batter_hand}HH vs LHP"
    elif p_throws == "R":
        iso_vs_this = iso_vs_rhp
        iso_vs_opp = iso_vs_lhp
        batter_barrel_vs = safe_float(row.get("vs_rhp_barrel_pct", 0))
        batter_hr_vs = safe_float(row.get("vs_rhp_hr_rate", 0))
        matchup_label = f"{batter_hand}HH vs RHP"
    else:
        return 0.0, "", 0.0

    if batter_hand == "R":
        pitcher_barrel_vs = safe_float(row.get("pitcher_vs_rhh_barrel_pct", 0))
        pitcher_hr_vs = safe_float(row.get("pitcher_vs_rhh_hr_rate", 0))
        pitcher_hr9_vs = safe_float(row.get("pitcher_vs_rhh_hr9", 0))
        pitcher_split_label = "vs RHH"
    elif batter_hand == "L":
        pitcher_barrel_vs = safe_float(row.get("pitcher_vs_lhh_barrel_pct", 0))
        pitcher_hr_vs = safe_float(row.get("pitcher_vs_lhh_hr_rate", 0))
        pitcher_hr9_vs = safe_float(row.get("pitcher_vs_lhh_hr9", 0))
        pitcher_split_label = "vs LHH"
    else:
        pitcher_barrel_vs = 0.0
        pitcher_hr_vs = 0.0
        pitcher_hr9_vs = 0.0
        pitcher_split_label = ""

    # ISO gap analysis
    iso_gap = iso_vs_opp - iso_vs_this
    has_iso_data = (iso_vs_this > 0 or iso_vs_opp > 0)

    if has_iso_data:
        if iso_gap >= ISO_GAP_SEVERE:
            penalty = 2.5
            parts.append(
                f"🚨 Severe platoon weakness ({matchup_label}) — "
                f"ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite"
            )
        elif iso_gap >= ISO_GAP_LARGE:
            penalty = 1.5
            parts.append(
                f"❌ Platoon disadvantage ({matchup_label}) — "
                f"ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite"
            )
        elif iso_gap >= ISO_GAP_MEDIUM:
            penalty = 0.7
            parts.append(
                f"⚠️ Moderate platoon weakness ({matchup_label}) — "
                f"ISO {iso_vs_this:.3f} vs this hand"
            )
        elif iso_gap <= -ISO_GAP_MEDIUM:
            score += 1.0
            parts.append(
                f"✅ Platoon advantage ({matchup_label}) — "
                f"ISO {iso_vs_this:.3f} vs this hand vs {iso_vs_opp:.3f} vs opposite"
            )
        else:
            parts.append(f"↔️ Neutral platoon ({matchup_label})")

    # Pitcher vulnerability vs batter handedness
    score += pitcher_barrel_vs * 0.06
    score += pitcher_hr_vs * 0.04
    score += pitcher_hr9_vs * 0.08

    if pitcher_split_label and pitcher_barrel_vs >= 8:
        parts.append(
            f"Pitcher {pitcher_barrel_vs:.1f}% barrel allowed {pitcher_split_label}"
        )
    if pitcher_split_label and pitcher_hr9_vs >= 1.5:
        parts.append(
            f"Pitcher {pitcher_hr9_vs:.2f} HR/9 {pitcher_split_label}"
        )

    # Batter performance vs this hand
    score += batter_barrel_vs * 0.04
    score += batter_hr_vs * 0.03

    score = round(score, 3)
    penalty = round(penalty, 3)
    description = " | ".join(parts) if parts else ""

    return score, description, penalty


def prepare_combined(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
    weather: pd.DataFrame,
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
        "season_barrel_pct_allowed": "pitcher_barrel_pct",
        "hr_per_fb_allowed":         "pitcher_hr_per_fb",
        "hard_hit_pct_allowed":      "pitcher_hard_hit_pct",
        "avg_ev_allowed":            "pitcher_avg_ev",
        "bf":                        "pitcher_bf",
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

    parks = parks.copy()
    parks.columns = [c.strip() for c in parks.columns]
    parks = parks.rename(columns={"team": "home_team"})

    weather = weather.copy()
    weather.columns = [c.strip() for c in weather.columns]
    weather = weather.rename(columns={"home_team": "weather_home_team"})

    batters = batters.copy()
    batters.columns = [c.strip() for c in batters.columns]
    batters = batters.rename(columns={"team": "batter_team"})

    pitcher_join_cols = [c for c in [
        "batter_team", "opp_pitcher_name", "opp_pitcher_team",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_avg_ev", "pitcher_bf",
        "pitcher_pct_fastball", "pitcher_pct_breaking",
        "pitcher_pct_offspeed", "pitcher_pct_knuckleball",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
        "pitcher_hand",
        "top_pitch_1", "top_pitch_1_pct",
        "top_pitch_2", "top_pitch_2_pct",
        "top_pitch_3", "top_pitch_3_pct",
    ] if c in pitchers.columns]

    combined = batters.merge(
        pitchers[pitcher_join_cols],
        on="batter_team",
        how="inner",
    )

    if combined.empty:
        print("No batter-pitcher matchups found.")
        return pd.DataFrame()

    if not parks.empty:
        combined = combined.merge(
            parks[["home_team", "park_hr_factor", "park_name", "small_sample"]],
            left_on="opp_pitcher_team",
            right_on="home_team",
            how="left",
        )
    else:
        combined["park_hr_factor"] = 100.0
        combined["park_name"] = ""
        combined["small_sample"] = False

    if not weather.empty:
        combined = combined.merge(
            weather[["weather_home_team", "hr_weather_boost", "wind_context", "temp_f"]],
            left_on="opp_pitcher_team",
            right_on="weather_home_team",
            how="left",
        )
        combined["hr_weather_boost"] = combined["hr_weather_boost"].fillna(0.0).apply(safe_float)
        combined["wind_context"] = combined["wind_context"].fillna("Unknown")
        combined["temp_f"] = combined["temp_f"].fillna(72.0).apply(safe_float)
    else:
        combined["hr_weather_boost"] = 0.0
        combined["wind_context"] = ""
        combined["temp_f"] = 72.0

    score_cols = [
        "barrel_pct_7d", "hr_per_pa", "hr_per_fb", "iso",
        "avg_ev_7d", "hard_hit_pct_7d", "avg_launch_angle",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_bf", "park_hr_factor",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "vs_lhp_hr_rate", "vs_rhp_hr_rate",
        "vs_lhp_iso", "vs_rhp_iso",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "pitcher_vs_lhh_hr_rate", "pitcher_vs_rhh_hr_rate",
        "pitcher_vs_lhh_hr9", "pitcher_vs_rhh_hr9",
        "top_pitch_1_pct", "top_pitch_2_pct", "top_pitch_3_pct",
    ]
    for col in score_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(safe_float)
        else:
            combined[col] = 0.0

    combined["park_hr_factor_norm"] = combined["park_hr_factor"] - 100

    platoon_results = combined.apply(compute_platoon_score, axis=1)
    combined["platoon_score"] = platoon_results.apply(lambda x: x[0])
    combined["platoon_desc"] = platoon_results.apply(lambda x: x[1])
    combined["platoon_penalty"] = platoon_results.apply(lambda x: x[2])

    pitch_results = combined.apply(compute_pitch_matchup_score, axis=1)
    combined["pitch_matchup_score"] = pitch_results.apply(lambda x: x[0])
    combined["pitch_matchup_desc"] = pitch_results.apply(lambda x: x[1])

    combined["pitcher_quality_penalty"] = (
        normalize_inverted(combined["pitcher_barrel_pct"]) * 0.6 +
        normalize_inverted(combined["pitcher_hard_hit_pct"]) * 0.4
    )

    combined["weather_score"] = combined["hr_weather_boost"].clip(-2, 2) / 2

    combined["score"] = (
        normalize(combined["barrel_pct_7d"])            * WEIGHTS["barrel_pct_7d"] +
        normalize(combined["hr_per_pa"])                 * WEIGHTS["hr_per_pa"] +
        normalize(combined["hr_per_fb"])                 * WEIGHTS["hr_per_fb"] +
        normalize(combined["iso"])                       * WEIGHTS["iso"] +
        normalize(combined["avg_ev_7d"])                 * WEIGHTS["avg_ev_7d"] +
        normalize(combined["hard_hit_pct_7d"])           * WEIGHTS["hard_hit_pct_7d"] +
        normalize(combined["pitcher_barrel_pct"])        * WEIGHTS["pitcher_barrel_pct"] +
        normalize(combined["pitcher_hr_per_fb"])         * WEIGHTS["pitcher_hr_per_fb"] +
        normalize(combined["pitcher_hard_hit_pct"])      * WEIGHTS["pitcher_hard_hit_pct"] +
        normalize(combined["park_hr_factor_norm"])       * WEIGHTS["park_hr_factor"] +
        combined["weather_score"]                        * WEATHER_WEIGHT +
        normalize(combined["platoon_score"])             * PLATOON_BONUS_WEIGHT +
        normalize(combined["pitch_matchup_score"])       * PITCH_MATCHUP_WEIGHT -
        combined["pitcher_quality_penalty"]              * WEIGHTS["pitcher_quality_penalty"] -
        combined["platoon_penalty"]
    )

    combined["score"] = combined["score"].round(3)
    return combined


def build_reason(row) -> str:
    reasons = []

    barrel_7d = safe_float(row.get("barrel_pct_7d"))
    if barrel_7d >= 15:
        reasons.append(f"🔥 Hot — {barrel_7d:.1f}% barrel rate last 7 days")

    hr_pa = safe_float(row.get("hr_per_pa"))
    if hr_pa >= 4:
        reasons.append(f"💪 Strong HR rate ({hr_pa:.1f}% HR/PA)")

    iso = safe_float(row.get("iso"))
    if iso >= 0.200:
        reasons.append(f"⚡ Elite ISO ({iso:.3f})")

    p_barrel = safe_float(row.get("pitcher_barrel_pct"))
    if p_barrel >= 10:
        reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")
    elif p_barrel <= 3:
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

    park_factor = safe_float(row.get("park_hr_factor"), 100)
    park_name = str(row.get("park_name", ""))
    if park_factor >= 110:
        reasons.append(f"🏟️ HR-friendly park ({park_name}, factor {park_factor:.0f})")
    elif park_factor <= 80:
        reasons.append(f"🏟️ Pitcher-friendly park ({park_name}, factor {park_factor:.0f})")

    boost = safe_float(row.get("hr_weather_boost"))
    wind_ctx = str(row.get("wind_context", ""))
    temp = safe_float(row.get("temp_f"), 72)
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

    top10 = combined.nlargest(10, "score").copy()
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
        "reason":                     "Key Reasons",
        "barrel_pct_7d":              "Barrel% (7d)",
        "hr_per_pa":                  "HR/PA%",
        "hr_per_fb":                  "HR/FB%",
        "iso":                        "ISO",
        "vs_lhp_iso":                 "ISO vs LHP",
        "vs_rhp_iso":                 "ISO vs RHP",
        "avg_ev_7d":                  "Avg EV (7d)",
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
        "hr_weather_boost":           "Weather Boost",
        "wind_context":               "Wind",
        "temp_f":                     "Temp (°F)",
    }

    available = {k: v for k, v in output_cols.items() if k in top10.columns}
    return top10[list(available.keys())].rename(columns=available)


def build_ev_subsection(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    ev_df = combined[
        (combined["avg_ev_7d"] > 92) &
        (combined["avg_launch_angle"] < 20) &
        (combined["avg_launch_angle"] > -90)
    ].copy()

    if ev_df.empty:
        print("No high EV / low launch angle candidates found.")
        return pd.DataFrame()

    ev_df["ev_score"] = (
        normalize(ev_df["avg_ev_7d"])           * 2.0 +
        normalize(ev_df["hard_hit_pct_7d"])     * 1.5 +
        normalize(ev_df["pitcher_barrel_pct"])  * 1.5 +
        normalize(ev_df["pitcher_hr_per_fb"])   * 1.2 +
        normalize(ev_df["pitch_matchup_score"]) * 1.0 +
        normalize(ev_df["platoon_score"])       * 0.8 +
        normalize(ev_df["park_hr_factor_norm"]) * 0.5 +
        ev_df["weather_score"]                  * 0.3 -
        ev_df["pitcher_quality_penalty"]        * 1.0 -
        ev_df["platoon_penalty"]
    ).round(3)

    top5 = ev_df.nlargest(5, "ev_score").copy()
    top5["ev_rank"] = range(1, len(top5) + 1)

    def build_ev_reason(row) -> str:
        reasons = []
        ev = safe_float(row.get("avg_ev_7d"))
        la = safe_float(row.get("avg_launch_angle"))
        reasons.append(f"💥 Avg EV {ev:.1f} mph, launch angle {la:.1f}°")

        hh = safe_float(row.get("hard_hit_pct_7d"))
        if hh >= 40:
            reasons.append(f"🔨 {hh:.1f}% hard hit rate")

        p_barrel = safe_float(row.get("pitcher_barrel_pct"))
        if p_barrel >= 8:
            reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")

        platoon_desc = str(row.get("platoon_desc", ""))
        if platoon_desc:
            reasons.append(f"🔄 {platoon_desc}")

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
        "why":                        "Why They're Here",
        "avg_ev_7d":                  "Avg EV (7d)",
        "avg_launch_angle":           "Avg Launch Angle",
        "hard_hit_pct_7d":            "Hard Hit% (7d)",
        "hr_per_fb":                  "HR/FB%",
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
        "hr_weather_boost":           "Weather Boost",
        "wind_context":               "Wind",
        "temp_f":                     "Temp (°F)",
    }

    available = {k: v for k, v in output_cols.items() if k in top5.columns}
    return top5[list(available.keys())].rename(columns=available)


def clean_for_sheets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: "" if (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))
            else x
        )
    df = df.fillna("")
    return df


def write_picks_to_sheet(
    gc: gspread.Client,
    sheet_id: str,
    picks: pd.DataFrame,
    ev_section: pd.DataFrame,
) -> tuple:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("Top_HR_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_HR_Picks", rows=100, cols=35)

    picks_clean = clean_for_sheets(picks)
    ev_clean = clean_for_sheets(ev_section)

    all_values = []

    all_values.append(picks_clean.columns.tolist())
    for _, row in picks_clean.iterrows():
        all_values.append(row.astype(str).tolist())

    main_row_count = len(picks_clean)
    ev_start_row = len(all_values) + 2

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


def format_picks_sheet(
    gc: gspread.Client,
    sheet_id: str,
    main_row_count: int,
    ev_data_row_count: int,
    ev_start_row: int,
    ev_col_header_row: int,
) -> None:
    print("Applying Carbon dark mode formatting...")
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet("Top_HR_Picks")
    ws_id = ws.id

    main_cols = 33
    ev_cols = 28
    total_rows = ev_col_header_row + ev_data_row_count + 2

    reqs = []

    # Full sheet base style
    reqs.append({
        "repeatCell": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": 0,
                "endRowIndex": total_rows,
                "startColumnIndex": 0,
                "endColumnIndex": max(main_cols, ev_cols),
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_BG,
                    "textFormat": {
                        "foregroundColor": COLOR_WHITE,
                        "fontFamily": "Roboto Mono",
                        "fontSize": 10,
                    },
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "WRAP",
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
        }
    })

    # Main picks header
    reqs.append({
        "repeatCell": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": 0,
                "endRowIndex": 1,
                "startColumnIndex": 0,
                "endColumnIndex": main_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_ACCENT_DIM,
                    "textFormat": {
                        "foregroundColor": COLOR_ACCENT,
                        "bold": True,
                        "fontFamily": "Roboto",
                        "fontSize": 11,
                    },
                    "horizontalAlignment": "CENTER",
                    "verticalAlignment": "MIDDLE",
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }
    })

    # Alternating main rows
    for i in range(main_row_count):
        row_idx = i + 1
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({
            "repeatCell": {
                "range": {
                    "sheetId": ws_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": main_cols,
                },
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }
        })

    # Medal rows
    medals = [
        (1, {"red": 0.18, "green": 0.14, "blue": 0.00}, COLOR_GOLD),
        (2, {"red": 0.14, "green": 0.14, "blue": 0.14}, COLOR_SILVER),
        (3, {"red": 0.16, "green": 0.10, "blue": 0.04}, COLOR_BRONZE),
    ]
    for rank, bg, fg in medals:
        if main_row_count >= rank:
            reqs.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws_id,
                        "startRowIndex": rank,
                        "endRowIndex": rank + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": main_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": bg,
                            "textFormat": {
                                "foregroundColor": fg,
                                "bold": True,
                                "fontSize": 10,
                            },
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            })

    # Rank column
    reqs.append({
        "repeatCell": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": 1,
                "endRowIndex": main_row_count + 1,
                "startColumnIndex": 0,
                "endColumnIndex": 1,
            },
            "cell": {
                "userEnteredFormat": {
                    "textFormat": {
                        "bold": True,
                        "fontSize": 14,
                        "foregroundColor": COLOR_ACCENT,
                    },
                    "horizontalAlignment": "CENTER",
                }
            },
            "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
        }
    })

    # HR Score column (col 8)
    reqs.append({
        "repeatCell": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": 1,
                "endRowIndex": main_row_count + 1,
                "startColumnIndex": 8,
                "endColumnIndex": 9,
            },
            "cell": {
                "userEnteredFormat": {
                    "textFormat": {
                        "bold": True,
                        "fontSize": 12,
                        "foregroundColor": COLOR_GREEN,
                    },
                    "horizontalAlignment": "CENTER",
                }
            },
            "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
        }
    })

    # EV section label
    reqs.append({
        "repeatCell": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": ev_start_row,
                "endRowIndex": ev_start_row + 1,
                "startColumnIndex": 0,
                "endColumnIndex": ev_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_ACCENT_DIM,
                    "textFormat": {
                        "foregroundColor": COLOR_ORANGE,
                        "bold": True,
                        "fontFamily": "Roboto",
                        "fontSize": 12,
                    },
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }
    })

    # EV column header
    reqs.append({
        "repeatCell": {
            "range": {
                "sheetId": ws_id,
                "startRowIndex": ev_col_header_row,
                "endRowIndex": ev_col_header_row + 1,
                "startColumnIndex": 0,
                "endColumnIndex": ev_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_EV_HEADER,
                    "textFormat": {
                        "foregroundColor": COLOR_ORANGE,
                        "bold": True,
                        "fontFamily": "Roboto",
                        "fontSize": 11,
                    },
                    "horizontalAlignment": "CENTER",
                    "verticalAlignment": "MIDDLE",
                }
            },
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
        }
    })

    # EV data rows
    for i in range(ev_data_row_count):
        row_idx = ev_col_header_row + 1 + i
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({
            "repeatCell": {
                "range": {
                    "sheetId": ws_id,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": ev_cols,
                },
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }
        })

    # EV rank and score styling
    if ev_data_row_count > 0:
        for col_idx, font_size in [(0, 14), (8, 11)]:
            reqs.append({
                "repeatCell": {
                    "range": {
                        "sheetId": ws_id,
                        "startRowIndex": ev_col_header_row + 1,
                        "endRowIndex": ev_col_header_row + 1 + ev_data_row_count,
                        "startColumnIndex": col_idx,
                        "endColumnIndex": col_idx + 1,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {
                                "bold": True,
                                "fontSize": font_size,
                                "foregroundColor": COLOR_ORANGE,
                            },
                            "horizontalAlignment": "CENTER",
                        }
                    },
                    "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
                }
            })

    # Freeze header
    reqs.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": ws_id,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    })

    # Row heights
    reqs.append({
        "updateDimensionProperties": {
            "range": {
                "sheetId": ws_id,
                "dimension": "ROWS",
                "startIndex": 0,
                "endIndex": 1,
            },
            "properties": {"pixelSize": 42},
            "fields": "pixelSize",
        }
    })
    reqs.append({
        "updateDimensionProperties": {
            "range": {
                "sheetId": ws_id,
                "dimension": "ROWS",
                "startIndex": 1,
                "endIndex": total_rows,
            },
            "properties": {"pixelSize": 58},
            "fields": "pixelSize",
        }
    })

    # Column widths
    col_widths = [
        50,   # Rank
        160,  # Batter
        50,   # Bats
        55,   # Team
        175,  # Opposing Pitcher
        55,   # Throws
        75,   # Pitcher Team
        175,  # Park
        75,   # HR Score
        380,  # Key Reasons
        90,   # Barrel% 7d
        75,   # HR/PA%
        75,   # HR/FB%
        65,   # ISO
        80,   # ISO vs LHP
        80,   # ISO vs RHP
        85,   # Avg EV
        220,  # Platoon Matchup
        120,  # Pitcher Barrel%
        120,  # Pitcher HR/FB%
        130,  # Pitcher Barrel% vs LHH
        130,  # Pitcher Barrel% vs RHH
        90,   # Pitcher HR/9 vs LHH
        90,   # Pitcher HR/9 vs RHH
        90,   # Top Pitch 1
        75,   # Top Pitch 1 %
        90,   # Top Pitch 2
        75,   # Top Pitch 2 %
        90,   # Top Pitch 3
        75,   # Top Pitch 3 %
        240,  # Pitch Matchup
        75,   # Park HR Factor
        75,   # Weather Boost
    ]
    for i, width in enumerate(col_widths):
        reqs.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": ws_id,
                    "dimension": "COLUMNS",
                    "startIndex": i,
                    "endIndex": i + 1,
                },
                "properties": {"pixelSize": width},
                "fields": "pixelSize",
            }
        })

    # Tab color
    reqs.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": ws_id,
                "tabColorStyle": {"rgbColor": COLOR_ACCENT},
            },
            "fields": "tabColorStyle",
        }
    })

    try:
        sh.batch_update({"requests": reqs})
        print("Carbon formatting applied successfully!")
    except APIError as e:
        print(f"Formatting failed: {e}")


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    print("Reading data from Google Sheets...")
    batters = read_sheet(gc, sheet_id, "Batter_Statcast_2026")
    pitchers = read_sheet(gc, sheet_id, "Pitcher_Statcast_2026")
    parks = read_sheet(gc, sheet_id, "Park_Factors")
    weather = read_sheet(gc, sheet_id, "Weather")

    print(f"Batters: {len(batters)} rows")
    print(f"Pitchers: {len(pitchers)} rows")
    print(f"Parks: {len(parks)} rows")
    print(f"Weather: {len(weather)} rows")

    combined = prepare_combined(batters, pitchers, parks, weather)

    if combined.empty:
        print("WARNING: No combined data — check all sheets have data.")
        return

    picks = build_main_picks(combined)
    ev_section = build_ev_subsection(combined)

    print(f"Built {len(picks)} main picks")
    print(f"Built {len(ev_section)} EV subsection picks")

    if picks.empty:
        print("WARNING: No picks generated.")
        return

    print("\nTop 10 HR Picks:")
    print(picks[["Rank", "Batter", "Bats", "Opposing Pitcher", "Throws", "HR Score"]].to_string(index=False))

    if not ev_section.empty:
        print("\nHigh EV / Launch Angle Upside:")
        print(ev_section[["Rank", "Batter", "Bats", "Opposing Pitcher", "Throws", "Avg EV (7d)", "Avg Launch Angle"]].to_string(index=False))

    main_row_count, ev_data_row_count, ev_start_row, ev_col_header_row = write_picks_to_sheet(
        gc, sheet_id, picks, ev_section
    )
    print("Written to Top_HR_Picks")

    format_picks_sheet(
        gc, sheet_id,
        main_row_count=main_row_count,
        ev_data_row_count=ev_data_row_count,
        ev_start_row=ev_start_row,
        ev_col_header_row=ev_col_header_row,
    )


if __name__ == "__main__":
    main()
