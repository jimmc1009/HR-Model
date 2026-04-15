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

# Scoring weights — how much each factor contributes to the final score
WEIGHTS = {
    "barrel_pct_7d":            2.0,   # recent barrel% is strongest signal
    "hr_per_pa":                1.8,   # season HR rate
    "hr_per_fb":                1.5,   # HR per fly ball
    "iso":                      1.2,   # isolated power
    "avg_ev_7d":                1.0,   # recent exit velocity
    "hard_hit_pct_7d":          0.8,   # recent hard hit%
    "pitcher_barrel_pct":       1.5,   # pitcher allows barrels
    "pitcher_hr_per_fb":        1.5,   # pitcher allows HRs on fly balls
    "pitcher_hard_hit_pct":     0.8,   # pitcher allows hard contact
    "park_hr_factor":           0.5,   # park factor (centered at 100)
    "hr_weather_boost":         0.4,   # weather boost
}

# Platoon bonus — if batter has strong splits vs pitcher handedness
PLATOON_BONUS_WEIGHT = 0.6


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def read_sheet(gc: gspread.Client, sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """Read a worksheet into a DataFrame."""
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{worksheet_name}' not found.")
        return pd.DataFrame()


def safe_float(val, default=0.0) -> float:
    """Safely convert a value to float."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def normalize(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to 0-1 range."""
    mn = series.min()
    mx = series.max()
    if mx == mn:
        return pd.Series([0.5] * len(series), index=series.index)
    return (series - mn) / (mx - mn)


def build_picks(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all four data sources and score each batter-pitcher matchup.
    """
    if batters.empty or pitchers.empty:
        print("Missing batter or pitcher data — cannot build picks.")
        return pd.DataFrame()

    # --- Clean up pitchers ---
    # Keep only one pitcher per opposing team
    # pitcher_team = the team pitching, opposing_team = the team batting against them
    pitchers = pitchers.copy()
    pitchers.columns = [c.strip() for c in pitchers.columns]

    # Rename pitcher columns to avoid clashes with batter columns
    pitcher_cols = {
        "pitcher_name":             "opp_pitcher_name",
        "pitcher_team":             "opp_pitcher_team",
        "opposing_team":            "batter_team",
        "season_barrel_pct_allowed":"pitcher_barrel_pct",
        "hr_per_fb_allowed":        "pitcher_hr_per_fb",
        "hard_hit_pct_allowed":     "pitcher_hard_hit_pct",
        "avg_ev_allowed":           "pitcher_avg_ev",
        "pitch_pct_fastball":       "pitcher_pct_fastball",
        "pitch_pct_breaking":       "pitcher_pct_breaking",
        "pitch_pct_offspeed":       "pitcher_pct_offspeed",
        "vs_lhh_barrel_pct":        "pitcher_vs_lhh_barrel_pct",
        "vs_rhh_barrel_pct":        "pitcher_vs_rhh_barrel_pct",
        "p_throws":                 "pitcher_hand",
    }
    pitchers = pitchers.rename(columns={k: v for k, v in pitcher_cols.items() if k in pitchers.columns})

    # --- Clean up park factors ---
    parks = parks.copy()
    parks.columns = [c.strip() for c in parks.columns]
    parks = parks.rename(columns={"team": "home_team"})

    # --- Clean up weather ---
    weather = weather.copy()
    weather.columns = [c.strip() for c in weather.columns]
    weather = weather.rename(columns={"home_team": "weather_home_team"})

    # --- Clean up batters ---
    batters = batters.copy()
    batters.columns = [c.strip() for c in batters.columns]
    batters = batters.rename(columns={"team": "batter_team"})

    # --- Join batter + pitcher on batter_team ---
    combined = batters.merge(
        pitchers[[c for c in [
            "batter_team",
            "opp_pitcher_name",
            "opp_pitcher_team",
            "pitcher_barrel_pct",
            "pitcher_hr_per_fb",
            "pitcher_hard_hit_pct",
            "pitcher_avg_ev",
            "pitcher_pct_fastball",
            "pitcher_pct_breaking",
            "pitcher_pct_offspeed",
            "pitcher_vs_lhh_barrel_pct",
            "pitcher_vs_rhh_barrel_pct",
            "pitcher_hand",
        ] if c in pitchers.columns]],
        on="batter_team",
        how="inner",
    )

    if combined.empty:
        print("No batter-pitcher matchups found — check that team abbreviations match.")
        return pd.DataFrame()

    # --- Join park factors on opp_pitcher_team (home team = park) ---
    # The park is determined by which team is home
    # We use opp_pitcher_team as a proxy — if pitcher is home team, use their park
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

    # --- Join weather on home team ---
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

    # --- Convert all scoring columns to float ---
    score_cols = [
        "barrel_pct_7d", "hr_per_pa", "hr_per_fb", "iso",
        "avg_ev_7d", "hard_hit_pct_7d",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "park_hr_factor", "hr_weather_boost",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "pitcher_vs_lhh_barrel_pct", "pitcher_vs_rhh_barrel_pct",
        "iso_vs_fastball", "iso_vs_breaking", "iso_vs_offspeed",
        "pitcher_pct_fastball", "pitcher_pct_breaking", "pitcher_pct_offspeed",
    ]
    for col in score_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(safe_float)
        else:
            combined[col] = 0.0

    # --- Normalize park factor (center at 100) ---
    combined["park_hr_factor_norm"] = combined["park_hr_factor"] - 100

    # --- Pitch mix matchup bonus ---
    # Cross-reference batter's ISO vs pitch type with pitcher's pitch mix
    combined["pitch_mix_bonus"] = (
        combined["iso_vs_fastball"] * combined["pitcher_pct_fastball"] / 100 +
        combined["iso_vs_breaking"] * combined["pitcher_pct_breaking"] / 100 +
        combined["iso_vs_offspeed"] * combined["pitcher_pct_offspeed"] / 100
    )

    # --- Platoon bonus ---
    # If pitcher throws left and batter hits better vs LHP, add bonus
    def platoon_bonus(row):
        hand = str(row.get("pitcher_hand", "")).strip().upper()
        if hand == "L":
            return safe_float(row.get("vs_lhp_barrel_pct", 0))
        elif hand == "R":
            return safe_float(row.get("vs_rhp_barrel_pct", 0))
        return 0.0

    if "pitcher_hand" in combined.columns:
        combined["platoon_bonus"] = combined.apply(platoon_bonus, axis=1)
    else:
        combined["platoon_bonus"] = 0.0

    # --- Build composite score ---
    combined["score"] = (
        normalize(combined["barrel_pct_7d"])      * WEIGHTS["barrel_pct_7d"] +
        normalize(combined["hr_per_pa"])           * WEIGHTS["hr_per_pa"] +
        normalize(combined["hr_per_fb"])           * WEIGHTS["hr_per_fb"] +
        normalize(combined["iso"])                 * WEIGHTS["iso"] +
        normalize(combined["avg_ev_7d"])           * WEIGHTS["avg_ev_7d"] +
        normalize(combined["hard_hit_pct_7d"])     * WEIGHTS["hard_hit_pct_7d"] +
        normalize(combined["pitcher_barrel_pct"])  * WEIGHTS["pitcher_barrel_pct"] +
        normalize(combined["pitcher_hr_per_fb"])   * WEIGHTS["pitcher_hr_per_fb"] +
        normalize(combined["pitcher_hard_hit_pct"])* WEIGHTS["pitcher_hard_hit_pct"] +
        normalize(combined["park_hr_factor_norm"]) * WEIGHTS["park_hr_factor"] +
        normalize(combined["hr_weather_boost"])    * WEIGHTS["hr_weather_boost"] +
        normalize(combined["platoon_bonus"])       * PLATOON_BONUS_WEIGHT +
        normalize(combined["pitch_mix_bonus"])     * 0.8
    )

    combined["score"] = combined["score"].round(3)

    # --- Build reason string ---
    def build_reason(row) -> str:
        reasons = []

        # Batter hot streak
        barrel_7d = safe_float(row.get("barrel_pct_7d"))
        if barrel_7d >= 15:
            reasons.append(f"🔥 Hot — {barrel_7d:.1f}% barrel rate last 7 days")

        # Season HR rate
        hr_pa = safe_float(row.get("hr_per_pa"))
        if hr_pa >= 4:
            reasons.append(f"💪 Strong HR rate ({hr_pa:.1f}% HR/PA this season)")

        # ISO
        iso = safe_float(row.get("iso"))
        if iso >= 0.200:
            reasons.append(f"⚡ Elite ISO ({iso:.3f})")

        # Pitcher vulnerable
        p_barrel = safe_float(row.get("pitcher_barrel_pct"))
        if p_barrel >= 10:
            reasons.append(f"🎯 Pitcher allows {p_barrel:.1f}% barrels")

        p_hr_fb = safe_float(row.get("pitcher_hr_per_fb"))
        if p_hr_fb >= 15:
            reasons.append(f"🚀 Pitcher HR/FB% is {p_hr_fb:.1f}%")

        # Pitch mix matchup
        pct_fb = safe_float(row.get("pitcher_pct_fastball"))
        iso_fb = safe_float(row.get("iso_vs_fastball"))
        pct_br = safe_float(row.get("pitcher_pct_breaking"))
        iso_br = safe_float(row.get("iso_vs_breaking"))
        pct_os = safe_float(row.get("pitcher_pct_offspeed"))
        iso_os = safe_float(row.get("iso_vs_offspeed"))

        if pct_fb >= 50 and iso_fb >= 0.200:
            reasons.append(f"🎳 Batter ISO {iso_fb:.3f} vs fastballs, pitcher throws {pct_fb:.0f}% fastballs")
        elif pct_br >= 35 and iso_br >= 0.180:
            reasons.append(f"🎳 Batter ISO {iso_br:.3f} vs breaking balls, pitcher throws {pct_br:.0f}% breaking")
        elif pct_os >= 25 and iso_os >= 0.180:
            reasons.append(f"🎳 Batter ISO {iso_os:.3f} vs offspeed, pitcher throws {pct_os:.0f}% offspeed")

        # Park factor
        park_factor = safe_float(row.get("park_hr_factor"), 100)
        park_name = str(row.get("park_name", ""))
        if park_factor >= 110:
            reasons.append(f"🏟️ HR-friendly park ({park_name}, factor {park_factor:.0f})")

        # Weather
        boost = safe_float(row.get("hr_weather_boost"))
        wind_ctx = str(row.get("wind_context", ""))
        temp = safe_float(row.get("temp_f"), 72)
        if boost >= 1.0:
            reasons.append(f"🌬️ Favorable weather — {wind_ctx}")
        if temp >= 85:
            reasons.append(f"🌡️ Hot weather ({temp:.0f}°F) helps ball carry")

        # Platoon advantage
        platoon = safe_float(row.get("platoon_bonus"))
        hand = str(row.get("pitcher_hand", "")).strip().upper()
        if platoon >= 15 and hand:
            side = "LHP" if hand == "L" else "RHP"
            reasons.append(f"📊 Strong platoon advantage vs {side}")

        if not reasons:
            reasons.append("Solid across multiple factors")

        return " | ".join(reasons)

    combined["reason"] = combined.apply(build_reason, axis=1)

    # --- Select top 10 ---
    top10 = combined.nlargest(10, "score").copy()
    top10["rank"] = range(1, len(top10) + 1)

    # --- Build output table ---
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
