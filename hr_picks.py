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

PLATOON_BONUS_WEIGHT = 0.6
PITCH_MATCHUP_WEIGHT = 1.2
WEATHER_WEIGHT = 0.4

# Dark mode color palette
COLOR_BG_DARK      = {"red": 0.102, "green": 0.102, "blue": 0.180}
COLOR_BG_MID       = {"red": 0.067, "green": 0.118, "blue": 0.227}
COLOR_BG_ALT       = {"red": 0.118, "green": 0.118, "blue": 0.196}
COLOR_HEADER       = {"red": 0.059, "green": 0.204, "blue": 0.376}
COLOR_SUBHEADER    = {"red": 0.200, "green": 0.100, "blue": 0.000}
COLOR_GOLD         = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_SILVER       = {"red": 0.753, "green": 0.753, "blue": 0.753}
COLOR_BRONZE       = {"red": 0.804, "green": 0.498, "blue": 0.196}
COLOR_WHITE        = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_ACCENT_BLUE  = {"red": 0.204, "green": 0.596, "blue": 0.859}
COLOR_GREEN        = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_ORANGE       = {"red": 0.945, "green": 0.502, "blue": 0.059}


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
    """Normalize and invert — high values become low scores (for pitcher quality penalty)."""
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


def prepare_combined(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all four data sources into one combined DataFrame
    for use in both main picks and subsection picks.
    """
    if batters.empty or pitchers.empty:
        print("Missing batter or pitcher data.")
        return pd.DataFrame()

    pitchers = pitchers.copy()
    pitchers.columns = [c.strip() for c in pitchers.columns]

    # Explicit rename for all pitcher columns
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
        "top_pitch_1":               "top_pitch_1",
        "top_pitch_1_pct":           "top_pitch_1_pct",
        "top_pitch_2":               "top_pitch_2",
        "top_pitch_2_pct":           "top_pitch_2_pct",
        "top_pitch_3":               "top_pitch_3",
        "top_pitch_3_pct":           "top_pitch_3_pct",
    }
    pitchers = pitchers.rename(columns={k: v for k, v in pitcher_rename.items() if k in pitchers.columns})

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
    else:
        combined["hr_weather_boost"] = 0.0
        combined["wind_context"] = ""
        combined["temp_f"] = 72.0

    # Convert scoring columns to float
    score_cols = [
        "barrel_pct_7d", "hr_per_pa", "hr_per_fb", "iso",
        "avg_ev_7d", "hard_hit_pct_7d", "avg_launch_angle",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_bf", "park_hr_factor", "hr_weather_boost",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "top_pitch_1_pct", "top_pitch_2_pct", "top_pitch_3_pct",
    ]
    for col in score_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(safe_float)
        else:
            combined[col] = 0.0

    combined["park_hr_factor_norm"] = combined["park_hr_factor"] - 100

    # Platoon bonus
    def platoon_bonus(row):
        stand = str(row.get("stand", "")).strip().upper()
        if stand == "L":
            return safe_float(row.get("vs_lhp_barrel_pct", 0))
        elif stand == "R":
            return safe_float(row.get("vs_rhp_barrel_pct", 0))
        return 0.0

    combined["platoon_bonus"] = combined.apply(platoon_bonus, axis=1) \
        if "stand" in combined.columns else 0.0

    # Pitch matchup score
    pitch_matchup_results = combined.apply(compute_pitch_matchup_score, axis=1)
    combined["pitch_matchup_score"] = pitch_matchup_results.apply(lambda x: x[0])
    combined["pitch_matchup_desc"] = pitch_matchup_results.apply(lambda x: x[1])

    # Pitcher quality penalty — elite pitchers drag score down
    # Uses inverted barrel% and inverted hard hit% allowed
    # A pitcher with 0 barrels and low hard hit% gets a HIGH penalty score
    combined["pitcher_quality_penalty"] = (
        normalize_inverted(combined["pitcher_barrel_pct"]) * 0.6 +
        normalize_inverted(combined["pitcher_hard_hit_pct"]) * 0.4
    )

    # Weather as raw additive — can be negative
    # Clamp to reasonable range (-2 to +2) then scale
    combined["weather_score"] = combined["hr_weather_boost"].clip(-2, 2) / 2

    # Composite score
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
        normalize(combined["platoon_bonus"])             * PLATOON_BONUS_WEIGHT +
        normalize(combined["pitch_matchup_score"])       * PITCH_MATCHUP_WEIGHT -
        combined["pitcher_quality_penalty"]              * WEIGHTS["pitcher_quality_penalty"]
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

    platoon = safe_float(row.get("platoon_bonus"))
    if platoon >= 15:
        reasons.append(f"📊 Strong platoon advantage")

    if not reasons:
        reasons.append("Solid across multiple factors")

    return " | ".join(reasons)


def build_ev_subsection(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Top 5 batters with high exit velocity but low launch angle
    who have favorable pitcher matchups.
    Criteria: avg_ev_7d > 92 AND avg_launch_angle < 20
    """
    if combined.empty:
        return pd.DataFrame()

    # Filter to high EV / low launch angle batters
    ev_df = combined[
        (combined["avg_ev_7d"] > 92) &
        (combined["avg_launch_angle"] < 20) &
        (combined["avg_launch_angle"] > -90)  # exclude missing data
    ].copy()

    if ev_df.empty:
        print("No high EV / low launch angle candidates found.")
        return pd.DataFrame()

    # Score based on EV quality + pitcher vulnerability + pitch matchup
    # Don't heavily weight HR rate since launch angle is suppressing it
    ev_df["ev_score"] = (
        normalize(ev_df["avg_ev_7d"])              * 2.0 +
        normalize(ev_df["hard_hit_pct_7d"])        * 1.5 +
        normalize(ev_df["pitcher_barrel_pct"])     * 1.5 +
        normalize(ev_df["pitcher_hr_per_fb"])      * 1.2 +
        normalize(ev_df["pitch_matchup_score"])    * 1.0 +
        normalize(ev_df["park_hr_factor_norm"])    * 0.5 +
        ev_df["weather_score"]                     * 0.3 -
        ev_df["pitcher_quality_penalty"]           * 1.0
    )

    ev_df["ev_score"] = ev_df["ev_score"].round(3)

    top5 = ev_df.nlargest(5, "ev_score").copy()
    top5["ev_rank"] = range(1, len(top5) + 1)

    def build_ev_reason(row) -> str:
        reasons = []
        ev = safe_float(row.get("avg_ev_7d"))
        la = safe_float(row.get("avg_launch_angle"))
        reasons.append(f"💥 Avg EV {ev:.1f} mph but launch angle only {la:.1f}°")

        hh = safe_float(row.get("hard_hit_pct_7d"))
        if hh >= 40:
            reasons.append(f"🔨 {hh:.1f}% hard hit rate")

        p_barrel = safe_float(row.get("pitcher_barrel_pct"))
        if p_barrel >= 8:
            reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")

        pitch_desc = str(row.get("pitch_matchup_desc", ""))
        if pitch_desc:
            reasons.append(f"🎳 {pitch_desc}")

        reasons.append("⬆️ Launch angle correction could mean HR upside")
        return " | ".join(reasons)

    top5["ev_reason"] = top5.apply(build_ev_reason, axis=1)

    output_cols = {
        "ev_rank":            "Rank",
        "player_name":        "Batter",
        "batter_team":        "Team",
        "opp_pitcher_name":   "Opposing Pitcher",
        "park_name":          "Park",
        "ev_score":           "EV Score",
        "ev_reason":          "Why They're Here",
        "avg_ev_7d":          "Avg EV (7d)",
        "avg_launch_angle":   "Avg Launch Angle",
        "hard_hit_pct_7d":    "Hard Hit% (7d)",
        "pitcher_barrel_pct": "Pitcher Barrel% Allowed",
        "pitcher_hr_per_fb":  "Pitcher HR/FB%",
        "pitch_matchup_desc": "Pitch Matchup",
        "park_hr_factor":     "Park HR Factor",
    }

    available = {k: v for k, v in output_cols.items() if k in top5.columns}
    return top5[list(available.keys())].rename(columns=available)


def build_picks(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    combined = combined.copy()
    combined["reason"] = combined.apply(build_reason, axis=1)

    top10 = combined.nlargest(10, "score").copy()
    top10["rank"] = range(1, len(top10) + 1)

    output_cols = {
        "rank":                 "Rank",
        "player_name":          "Batter",
        "batter_team":          "Team",
        "opp_pitcher_name":     "Opposing Pitcher",
        "opp_pitcher_team":     "Pitcher Team",
        "park_name":            "Park",
        "score":                "HR Score",
        "reason":               "Key Reasons",
        "barrel_pct_7d":        "Barrel% (7d)",
        "hr_per_pa":            "HR/PA%",
        "hr_per_fb":            "HR/FB%",
        "iso":                  "ISO",
        "avg_ev_7d":            "Avg EV (7d)",
        "pitcher_barrel_pct":   "Pitcher Barrel% Allowed",
        "pitcher_hr_per_fb":    "Pitcher HR/FB% Allowed",
        "top_pitch_1":          "Pitcher Top Pitch 1",
        "top_pitch_1_pct":      "Top Pitch 1 %",
        "top_pitch_2":          "Pitcher Top Pitch 2",
        "top_pitch_2_pct":      "Top Pitch 2 %",
        "top_pitch_3":          "Pitcher Top Pitch 3",
        "top_pitch_3_pct":      "Top Pitch 3 %",
        "pitch_matchup_desc":   "Pitch Matchup",
        "park_hr_factor":       "Park HR Factor",
        "hr_weather_boost":     "Weather Boost",
        "wind_context":         "Wind",
        "temp_f":               "Temp (°F)",
    }

    available = {k: v for k, v in output_cols.items() if k in top10.columns}
    return top10[list(available.keys())].rename(columns=available)


def format_picks_sheet(
    gc: gspread.Client,
    sheet_id: str,
    num_main_rows: int,
    num_ev_rows: int,
) -> None:
    print("Applying dark mode formatting...")
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet("Top_HR_Picks")
    sheet_id_int = ws.id
    num_cols = 24
    total_rows = num_main_rows + num_ev_rows + 5  # buffer for spacer rows

    requests_body = []

    # Full sheet dark background
    requests_body.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id_int,
                "startRowIndex": 0,
                "endRowIndex": total_rows + 2,
                "startColumnIndex": 0,
                "endColumnIndex": num_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_BG_DARK,
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

    # Main picks header row (row 0)
    requests_body.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id_int,
                "startRowIndex": 0,
                "endRowIndex": 1,
                "startColumnIndex": 0,
                "endColumnIndex": num_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER,
                    "textFormat": {
                        "foregroundColor": COLOR_WHITE,
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

    # Alternating rows for main picks
    for i in range(num_main_rows):
        row_idx = i + 1
        bg = COLOR_BG_MID if i % 2 == 0 else COLOR_BG_ALT
        requests_body.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id_int,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": num_cols,
                },
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }
        })

    # Gold / Silver / Bronze rows
    medal_rows = [
        (1, {"red": 0.200, "green": 0.157, "blue": 0.000}, COLOR_GOLD),
        (2, {"red": 0.150, "green": 0.150, "blue": 0.160}, COLOR_SILVER),
        (3, {"red": 0.180, "green": 0.120, "blue": 0.050}, COLOR_BRONZE),
    ]
    for row_idx, bg, fg in medal_rows:
        if num_main_rows >= row_idx:
            requests_body.append({
                "repeatCell": {
                    "range": {
                        "sheetId": sheet_id_int,
                        "startRowIndex": row_idx,
                        "endRowIndex": row_idx + 1,
                        "startColumnIndex": 0,
                        "endColumnIndex": num_cols,
                    },
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": bg,
                            "textFormat": {"foregroundColor": fg, "bold": True},
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat)",
                }
            })

    # EV subsection header row
    ev_header_row = num_main_rows + 2
    requests_body.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id_int,
                "startRowIndex": ev_header_row,
                "endRowIndex": ev_header_row + 1,
                "startColumnIndex": 0,
                "endColumnIndex": num_cols,
            },
            "cell": {
                "userEnteredFormat": {
                    "backgroundColor": COLOR_SUBHEADER,
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

    # EV subsection data rows
    for i in range(num_ev_rows):
        row_idx = ev_header_row + 1 + i
        bg = COLOR_BG_MID if i % 2 == 0 else COLOR_BG_ALT
        requests_body.append({
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id_int,
                    "startRowIndex": row_idx,
                    "endRowIndex": row_idx + 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": num_cols,
                },
                "cell": {"userEnteredFormat": {"backgroundColor": bg}},
                "fields": "userEnteredFormat(backgroundColor)",
            }
        })

    # Freeze header
    requests_body.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id_int,
                "gridProperties": {"frozenRowCount": 1},
            },
            "fields": "gridProperties.frozenRowCount",
        }
    })

    # Row heights
    requests_body.append({
        "updateDimensionProperties": {
            "range": {
                "sheetId": sheet_id_int,
                "dimension": "ROWS",
                "startIndex": 0,
                "endIndex": 1,
            },
            "properties": {"pixelSize": 40},
            "fields": "pixelSize",
        }
    })
    requests_body.append({
        "updateDimensionProperties": {
            "range": {
                "sheetId": sheet_id_int,
                "dimension": "ROWS",
                "startIndex": 1,
                "endIndex": total_rows + 2,
            },
            "properties": {"pixelSize": 60},
            "fields": "pixelSize",
        }
    })

    # Column widths
    col_widths = [
        50, 160, 60, 180, 80, 180, 80, 400,
        90, 80, 80, 70, 90, 120, 120,
        100, 80, 100, 80, 100, 80, 250, 80, 80,
    ]
    for i, width in enumerate(col_widths):
        requests_body.append({
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id_int,
                    "dimension": "COLUMNS",
                    "startIndex": i,
                    "endIndex": i + 1,
                },
                "properties": {"pixelSize": width},
                "fields": "pixelSize",
            }
        })

    # Rank column styling
    requests_body.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id_int,
                "startRowIndex": 1,
                "endRowIndex": num_main_rows + 1,
                "startColumnIndex": 0,
                "endColumnIndex": 1,
            },
            "cell": {
                "userEnteredFormat": {
                    "textFormat": {
                        "bold": True,
                        "fontSize": 14,
                        "foregroundColor": COLOR_ACCENT_BLUE,
                    },
                    "horizontalAlignment": "CENTER",
                }
            },
            "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
        }
    })

    # HR Score column
    requests_body.append({
        "repeatCell": {
            "range": {
                "sheetId": sheet_id_int,
                "startRowIndex": 1,
                "endRowIndex": num_main_rows + 1,
                "startColumnIndex": 6,
                "endColumnIndex": 7,
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

    # Tab color
    requests_body.append({
        "updateSheetProperties": {
            "properties": {
                "sheetId": sheet_id_int,
                "tabColorStyle": {"rgbColor": COLOR_ACCENT_BLUE},
            },
            "fields": "tabColorStyle",
        }
    })

    try:
        sh.batch_update({"requests": requests_body})
        print("Dark mode formatting applied successfully!")
    except APIError as e:
        print(f"Formatting failed: {e}")


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
) -> None:
    """
    Write main picks and EV subsection to the same sheet
    with a spacer and section label between them.
    """
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet("Top_HR_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_HR_Picks", rows=100, cols=30)

    picks_clean = clean_for_sheets(picks)
    ev_clean = clean_for_sheets(ev_section)

    # Main picks
    values = [picks_clean.columns.tolist()] + picks_clean.astype(str).values.tolist()

    # Spacer row
    values.append([""] * len(picks_clean.columns))

    # EV subsection header label
    ev_label = ["⚡ HIGH EXIT VELOCITY — LAUNCH ANGLE UPSIDE CANDIDATES"] + \
               [""] * (len(picks_clean.columns) - 1)
    values.append(ev_label)

    # EV subsection columns header
    if not ev_clean.empty:
        ev_cols = ev_clean.columns.tolist()
        ev_cols_padded = ev_cols + [""] * max(0, len(picks_clean.columns) - len(ev_cols))
        values.append(ev_cols_padded)
        for _, row in ev_clean.iterrows():
            row_vals = row.astype(str).tolist()
            row_padded = row_vals + [""] * max(0, len(picks_clean.columns) - len(row_vals))
            values.append(row_padded)

    ws.update(values)


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
        print("WARNING: No combined data — check that all sheets have data.")
        return

    picks = build_picks(combined)
    ev_section = build_ev_subsection(combined)

    print(f"Built {len(picks)} main picks")
    print(f"Built {len(ev_section)} EV subsection picks")

    if picks.empty:
        print("WARNING: No picks generated.")
        return

    print("\nTop 10 HR Picks:")
    print(picks[["Rank", "Batter", "Team", "Opposing Pitcher", "HR Score", "Key Reasons"]].to_string(index=False))

    if not ev_section.empty:
        print("\nHigh EV / Launch Angle Upside:")
        print(ev_section[["Rank", "Batter", "Team", "Avg EV (7d)", "Avg Launch Angle"]].to_string(index=False))

    write_picks_to_sheet(gc, sheet_id, picks, ev_section)
    print("Written to Top_HR_Picks")

    format_picks_sheet(
        gc, sheet_id,
        num_main_rows=len(picks),
        num_ev_rows=len(ev_section) + 2,  # +2 for label and column header rows
    )


if __name__ == "__main__":
    main()
