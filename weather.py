import os
import json
from datetime import date, datetime
from typing import Dict, Set

import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ESPN to MLB abbreviation mapping
ESPN_TO_MLB = {
    "WSH": "WSH", "HOU": "HOU", "MIL": "MIL", "LAD": "LAD",
    "BAL": "BAL", "CIN": "CIN", "NYY": "NYY", "TOR": "TOR",
    "COL": "COL", "OAK": "ATH", "CLE": "CLE", "CHC": "CHC",
    "SEA": "SEA", "PHI": "PHI", "DET": "DET", "ARI": "AZ",
    "TB":  "TB",  "SD":  "SD",  "STL": "STL", "ATL": "ATL",
    "KC":  "KC",  "LAA": "LAA", "NYM": "NYM", "MIA": "MIA",
    "MIN": "MIN", "PIT": "PIT", "TEX": "TEX", "SF":  "SF",
    "CWS": "CWS", "BOS": "BOS", "CHW": "CWS",
}

PARKS = [
    {"team": "AZ",  "park": "Chase Field",                 "lat": 33.4453, "lon": -112.0667, "roof": True,  "cf_direction": 0},
    {"team": "ATL", "park": "Truist Park",                 "lat": 33.8908, "lon": -84.4678,  "roof": False, "cf_direction": 25},
    {"team": "BAL", "park": "Oriole Park at Camden Yards", "lat": 39.2838, "lon": -76.6218,  "roof": False, "cf_direction": 55},
    {"team": "BOS", "park": "Fenway Park",                 "lat": 42.3467, "lon": -71.0972,  "roof": False, "cf_direction": 95},
    {"team": "CHC", "park": "Wrigley Field",               "lat": 41.9484, "lon": -87.6553,  "roof": False, "cf_direction": 135},
    {"team": "CWS", "park": "Guaranteed Rate Field",       "lat": 41.8300, "lon": -87.6339,  "roof": False, "cf_direction": 5},
    {"team": "CIN", "park": "Great American Ball Park",    "lat": 39.0979, "lon": -84.5082,  "roof": False, "cf_direction": 20},
    {"team": "CLE", "park": "Progressive Field",           "lat": 41.4962, "lon": -81.6852,  "roof": False, "cf_direction": 310},
    {"team": "COL", "park": "Coors Field",                 "lat": 39.7559, "lon": -104.9942, "roof": False, "cf_direction": 345},
    {"team": "DET", "park": "Comerica Park",               "lat": 42.3390, "lon": -83.0485,  "roof": False, "cf_direction": 340},
    {"team": "HOU", "park": "Minute Maid Park",            "lat": 29.7573, "lon": -95.3555,  "roof": True,  "cf_direction": 0},
    {"team": "KC",  "park": "Kauffman Stadium",            "lat": 39.0517, "lon": -94.4803,  "roof": False, "cf_direction": 10},
    {"team": "LAA", "park": "Angel Stadium",               "lat": 33.8003, "lon": -117.8827, "roof": False, "cf_direction": 5},
    {"team": "LAD", "park": "Dodger Stadium",              "lat": 34.0739, "lon": -118.2400, "roof": False, "cf_direction": 25},
    {"team": "MIA", "park": "loanDepot park",              "lat": 25.7781, "lon": -80.2197,  "roof": True,  "cf_direction": 0},
    {"team": "MIL", "park": "American Family Field",       "lat": 43.0280, "lon": -87.9712,  "roof": True,  "cf_direction": 0},
    {"team": "MIN", "park": "Target Field",                "lat": 44.9817, "lon": -93.2781,  "roof": False, "cf_direction": 330},
    {"team": "NYM", "park": "Citi Field",                  "lat": 40.7571, "lon": -73.8458,  "roof": False, "cf_direction": 355},
    {"team": "NYY", "park": "Yankee Stadium",              "lat": 40.8296, "lon": -73.9262,  "roof": False, "cf_direction": 20},
    {"team": "ATH", "park": "Sutter Health Park",          "lat": 38.5803, "lon": -121.5002, "roof": False, "cf_direction": 30},
    {"team": "PHI", "park": "Citizens Bank Park",          "lat": 39.9061, "lon": -75.1665,  "roof": False, "cf_direction": 350},
    {"team": "PIT", "park": "PNC Park",                    "lat": 40.4469, "lon": -80.0057,  "roof": False, "cf_direction": 30},
    {"team": "SD",  "park": "Petco Park",                  "lat": 32.7073, "lon": -117.1567, "roof": False, "cf_direction": 310},
    {"team": "SF",  "park": "Oracle Park",                 "lat": 37.7786, "lon": -122.3893, "roof": False, "cf_direction": 50},
    {"team": "SEA", "park": "T-Mobile Park",               "lat": 47.5914, "lon": -122.3325, "roof": True,  "cf_direction": 0},
    {"team": "STL", "park": "Busch Stadium",               "lat": 38.6226, "lon": -90.1928,  "roof": False, "cf_direction": 340},
    {"team": "TB",  "park": "Tropicana Field",             "lat": 27.7683, "lon": -82.6534,  "roof": True,  "cf_direction": 0},
    {"team": "TEX", "park": "Globe Life Field",            "lat": 32.7512, "lon": -97.0832,  "roof": True,  "cf_direction": 0},
    {"team": "TOR", "park": "Rogers Centre",               "lat": 43.6414, "lon": -79.3894,  "roof": True,  "cf_direction": 0},
    {"team": "WSH", "park": "Nationals Park",              "lat": 38.8730, "lon": -77.0074,  "roof": False, "cf_direction": 5},
]

# Build lookup by team abbr for fast access
PARK_BY_TEAM = {p["team"]: p for p in PARKS}


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_today_games() -> Dict[str, dict]:
    """
    Returns {home_team_abbr: {away_team, game_time_utc}} for today's games.
    Tries MLB API first, falls back to ESPN.
    Normalizes abbreviations to MLB standard.
    """
    def fetch_mlb_games() -> Dict[str, dict]:
        today_str = date.today().strftime("%Y-%m-%d")
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}&hydrate=venue"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        games: Dict[str, dict] = {}
        for d in data.get("dates", []):
            for g in d.get("games", []):
                home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                game_time = g.get("gameDate", "")
                if home:
                    games[str(home).strip()] = {
                        "away_team": str(away).strip(),
                        "game_time_utc": game_time,
                    }
        return games

    def fetch_espn_games() -> Dict[str, dict]:
        today_str = date.today().strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={today_str}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        games: Dict[str, dict] = {}
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                competitors = comp.get("competitors", [])
                if len(competitors) == 2:
                    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
                    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
                    if home and away:
                        home_abbr = ESPN_TO_MLB.get(
                            home.get("team", {}).get("abbreviation", ""),
                            home.get("team", {}).get("abbreviation", "")
                        )
                        away_abbr = ESPN_TO_MLB.get(
                            away.get("team", {}).get("abbreviation", ""),
                            away.get("team", {}).get("abbreviation", "")
                        )
                        if home_abbr:
                            games[home_abbr] = {
                                "away_team": away_abbr,
                                "game_time_utc": comp.get("date", ""),
                            }
        return games

    try:
        print("Fetching today's games from MLB API...")
        games = fetch_mlb_games()
        if games:
            # Normalize abbreviations
            normalized = {}
            for abbr, info in games.items():
                mlb_abbr = ESPN_TO_MLB.get(abbr, abbr)
                normalized[mlb_abbr] = info
            print(f"MLB API found {len(normalized)} games.")
            return normalized
        print("MLB API returned no games.")
    except Exception as e:
        print(f"MLB API failed: {e}")

    try:
        print("Falling back to ESPN API for games...")
        games = fetch_espn_games()
        if games:
            print(f"ESPN API found {len(games)} games.")
            return games
        print("ESPN API returned no games.")
    except Exception as e:
        print(f"ESPN API failed: {e}")

    return {}


def angle_difference(a: float, b: float) -> float:
    diff = (a - b + 180) % 360 - 180
    return diff


def wind_direction_label(degrees: float) -> str:
    if pd.isna(degrees):
        return "Unknown"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(degrees / 45) % 8
    return dirs[idx]


def hr_weather_boost(
    temp_f: float,
    wind_mph: float,
    wind_dir: float,
    roof: bool,
    cf_direction: int,
) -> float:
    if roof:
        return 0.0

    score = 0.0
    temp_delta = (temp_f - 72) / 10
    score += temp_delta * 0.5

    if not pd.isna(wind_mph) and not pd.isna(wind_dir) and wind_mph > 0:
        wind_to = (wind_dir + 180) % 360
        diff = abs(angle_difference(wind_to, cf_direction))
        alignment = (90 - diff) / 90
        score += alignment * (wind_mph / 10) * 1.5

    return round(score, 2)


def fetch_weather_for_park(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
        f"&temperature_unit=fahrenheit"
        f"&windspeed_unit=mph"
        f"&forecast_days=1"
        f"&timezone=auto"
    )

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("windspeed_10m", [])
        dirs = hourly.get("winddirection_10m", [])

        if not temps:
            return {}

        game_hour = 19
        idx = min(game_hour, len(temps) - 1)

        return {
            "temp_f": round(temps[idx], 1) if temps else None,
            "wind_mph": round(winds[idx], 1) if winds else None,
            "wind_dir_deg": round(dirs[idx], 1) if dirs else None,
        }

    except Exception as e:
        print(f"  Weather fetch failed for ({lat}, {lon}): {e}")
        return {}


def build_weather_table(games: Dict[str, dict]) -> pd.DataFrame:
    """
    Fetch weather for all parks hosting games today.
    If no games found from API, fetch all 30 parks as fallback.
    Also validates that home team abbreviations match our PARKS list
    and corrects any mismatches.
    """
    rows = []

    # Determine which teams to fetch weather for
    if games:
        playing_home_teams = set(games.keys())
        # Validate all teams exist in PARK_BY_TEAM
        unmatched = playing_home_teams - set(PARK_BY_TEAM.keys())
        if unmatched:
            print(f"Warning: No park data for teams: {unmatched} — fetching all parks as fallback")
            playing_home_teams = set(PARK_BY_TEAM.keys())
    else:
        print("No games found — fetching weather for all 30 parks.")
        playing_home_teams = set(PARK_BY_TEAM.keys())

    for park in PARKS:
        team = park["team"]

        if team not in playing_home_teams:
            continue

        print(f"  Fetching weather for {park['park']} ({team})...")
        weather = fetch_weather_for_park(park["lat"], park["lon"])

        temp_f = weather.get("temp_f")
        wind_mph = weather.get("wind_mph")
        wind_dir = weather.get("wind_dir_deg")
        roof = park["roof"]
        cf_direction = park["cf_direction"]

        wind_label = wind_direction_label(wind_dir) if wind_dir is not None else "Unknown"
        boost = hr_weather_boost(
            temp_f or 72.0,
            wind_mph or 0.0,
            wind_dir or 0.0,
            roof,
            cf_direction,
        )

        if roof:
            wind_context = "Roof — weather neutral"
        elif wind_mph and wind_mph >= 5:
            diff = abs(angle_difference(
                (wind_dir + 180) % 360 if wind_dir else 0,
                cf_direction
            ))
            if diff <= 45:
                wind_context = f"Wind blowing OUT at {wind_mph} mph"
            elif diff >= 135:
                wind_context = f"Wind blowing IN at {wind_mph} mph"
            else:
                wind_context = f"Crosswind at {wind_mph} mph"
        else:
            wind_context = "Calm"

        game_info = games.get(team, {})

        rows.append({
            "home_team": team,
            "away_team": game_info.get("away_team", ""),
            "park": park["park"],
            "has_roof": roof,
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "wind_dir_deg": wind_dir,
            "wind_direction": wind_label,
            "wind_context": wind_context,
            "hr_weather_boost": boost,
            "boost_note": "Roof — weather neutral" if roof else (
                "HR friendly" if boost >= 1.0
                else "HR suppressing" if boost <= -1.0
                else "Neutral"
            ),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values("hr_weather_boost", ascending=False)
    return df


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
        ws = sh.add_worksheet(title=worksheet_name, rows=500, cols=20)

    df = clean_for_sheets(df)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    print("Fetching today's games...")
    games = get_today_games()
    print(f"Games today: {len(games)} — home teams: {list(games.keys())}")

    print("Fetching weather for each park...")
    weather_df = build_weather_table(games)
    print(f"Built weather rows for {len(weather_df)} parks")

    write_dataframe_to_sheet(gc, sheet_id, "Weather", weather_df)
    print("Written to Weather")


if __name__ == "__main__":
    main()
