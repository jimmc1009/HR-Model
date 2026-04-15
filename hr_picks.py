import os
import json
from datetime import date
from typing import Dict

import pandas as pd
import numpy as np
import gspread
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
    "hr_weather_boost":         0.4,
}

PLATOON_BONUS_WEIGHT = 0.6
PITCH_MATCHUP_WEIGHT = 1.2


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


def compute_pitch_matchup_score(row: pd.Series) -> tuple:
    """
    Cross-reference batter's ISO vs pitcher's top 3 pitch types.
    Returns (score, description string).
    """
    scores = []
    descriptions = []

    for rank in range(1, 4):
        pitch_type = str(row.get(f"top_pitch_{rank}", "")).strip().upper()
        pitch_pct = safe_float(row.get(f"top_pitch_{rank}_pct", 0))

        if not pitch_type or pitch_type in ("", "NAN", "NONE"):
            continue

        # Look up batter's ISO vs this specific pitch type
        iso_col = f"iso_vs_{pitch_type}"
        hr_rate_col = f"hr_rate_vs_{pitch_type}"
        barrel_col = f"barrel_pct_vs_{pitch_type}"

        iso = safe_float(row.get(iso_col, 0))
        hr_rate = safe_float(row.get(hr_rate_col, 0))
        barrel = safe_float(row.get(barrel_col, 0))

        if iso > 0 or hr_rate > 0:
            # Weight by how often pitcher throws this pitch
            pitch_score = (iso * 3 + hr_rate / 10 + barrel / 20) * (pitch_pct / 100)
            scores.append(pitch_score)

            if iso >= 0.200 and pitch_pct >= 20:
                descriptions.append(
                    f"ISO {iso:.3f} vs {pitch_type} ({pitch_pct:.0f}% usage)"
                )

    total_score = sum(scores)
    desc = " + ".join(descriptions) if descriptions else ""
    return total_score, desc


def build_picks(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    if batters.empty or pitchers.empty:
        print("Missing batter or pitcher data — cannot build picks.")
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

    # Core pitcher cols to join
    pitcher_join_cols = [c for c in [
        "batter_team", "opp_pitcher_name", "opp_pitcher_team",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "pitcher_avg_ev", "pitcher_pct_fastball", "pitcher_pct_breaking",
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
        print("No batter-pitcher matchups found — check that team abbreviations match.")
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
        "avg_ev_7d", "hard_hit_pct_7d",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "park_hr_factor", "hr_weather_boost",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
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

    # Pitch matchup score using individual pitch type ISO
    pitch_matchup_results = combined.apply(compute_pitch_matchup_score, axis=1)
    combined["pitch_matchup_score"] = pitch_matchup_results.apply(lambda x: x[0])
    combined["pitch_matchup_desc"] = pitch_matchup_results.apply(lambda x: x[1])

    # Composite score
    combined["score"] = (
        normalize(combined["barrel_pct_7d"])       * WEIGHTS["barrel_pct_7d"] +
        normalize(combined["hr_per_pa"])            * WEIGHTS["hr_per_pa"] +
        normalize(combined["hr_per_fb"])            * WEIGHTS["hr_per_fb"] +
        normalize(combined["iso"])                  * WEIGHTS["iso"] +
        normalize(combined["avg_ev_7d"])            * WEIGHTS["avg_ev_7d"] +
        normalize(combined["hard_hit_pct_7d"])      * WEIGHTS["hard_hit_pct_7d"] +
        normalize(combined["pitcher_barrel_pct"])   * WEIGHTS["pitcher_barrel_pct"] +
        normalize(combined["pitcher_hr_per_fb"])    * WEIGHTS["pitcher_hr_per_fb"] +
        normalize(combined["pitcher_hard_hit_pct"]) * WEIGHTS["pitcher_hard_hit_pct"] +
        normalize(combined["park_hr_factor_norm"])  * WEIGHTS["park_hr_factor"] +
        normalize(combined["hr_weather_boost"])     * WEIGHTS["hr_weather_boost"] +
        normalize(combined["platoon_bonus"])        * PLATOON_BONUS_WEIGHT +
        normalize(combined["pitch_matchup_score"])  * PITCH_MATCHUP_WEIGHT
    )

    combined["score"] = combined["score"].round(3)

    # Build reason string
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

        boost = safe_float(row.get("hr_weather_boost"))
        wind_ctx = str(row.get("wind_context", ""))
        temp = safe_float(row.get("temp_f"), 72)
        if boost >= 1.0:
            reasons.append(f"🌬️ Favorable weather — {wind_ctx}")
        if temp >= 85:
            reasons.append(f"🌡️ Hot weather ({temp:.0f}°F)")

        platoon = safe_float(row.get("platoon_bonus"))
        if platoon >= 15:
            reasons.append(f"📊 Strong platoon advantage")

        if not reasons:
            reasons.append("Solid across multiple factors")

        return " | ".join(reasons)

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
    result = top10[list(available.keys())].rename(columns=available)

    return result


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


def write_dataframe_to_sheet(
    gc: gspread.Client,
    sheet_id: str,
    worksheet_name: str,
    df: pd.DataFrame,
) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=50, cols=30)

    df = clean_for_sheets(df)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
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

    picks = build_picks(batters, pitchers, parks, weather)
    print(f"Built {len(picks)} picks")

    if picks.empty:
        print("WARNING: No picks generated — check that all sheets have data.")
        return

    print("\nTop 10 HR Picks:")
    print(picks[["Rank", "Batter", "Team", "Opposing Pitcher", "HR Score", "Key Reasons"]].to_string(index=False))

    write_dataframe_to_sheet(gc, sheet_id, "Top_HR_Picks", picks)
    print("\nWritten to Top_HR_Picks")


if __name__ == "__main__":
    main()
