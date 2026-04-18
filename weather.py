import os
import json
import time
from datetime import date
from typing import Dict, List, Set

import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

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

# cf_direction = compass degrees from home plate to center field
# 0/360=North, 90=East, 180=South, 270=West
# Sources: Google Maps satellite, Scoopy Ballpark blog, Andrew Clem Baseball,
#          Hardball Times orientation study. Roof stadiums have cf_direction=0
#          since weather is irrelevant.
#
# Key corrections vs original:
#   CLE: 310 → 0   (faces due North, not NW)
#   CWS: 5  → 135  (faces SE, built across street from old Comiskey)
#   DET: 340 → 150 (faces SSE to give skyline view)
#   PIT: 30 → 113  (faces ESE toward Allegheny River)
#   CIN: 20 → 113  (faces ESE)
#   MIN: 330 → 90  (faces due East)
#   SF:  50 → 90   (faces due East toward bay)
#   STL: 340 → 68  (faces ENE)
#   NYY: 20 → 68   (faces ENE)
#   BAL: 55 → 23   (faces NNE)
#   WSH: 5  → 23   (faces NNE)
#   LAD: 25 → 23   (faces NNE)
#   NYM: 355 → 23  (faces NNE)
#   PHI: 350 → 13  (faces just N of NNE per research)
#   CHC: 135 → 23  (faces NNE, not SE)
#   ATL: 25 → 23   (faces NNE, Truist Park)
#   KC:  10 → 45   (faces NE)
#   BOS: 95 → 45   (faces NE)
#   LAA: 5  → 45   (faces NE)
#   SEA: 0  → 45   (faces NE — roof, irrelevant but corrected)
#   COL: 345 → 0   (faces due North)
#   AZ:  0  → 0    (roof — irrelevant)
#   SD:  310 → 0   (faces due North)
#   ATH: 30 → 68   (Sutter Health Park faces ENE)

PARKS = [
    # Roof stadiums — weather always neutral, cf_direction irrelevant
    {"team": "AZ",  "park": "Chase Field",           "lat": 33.4453, "lon": -112.0667, "roof": True,  "cf_direction": 0},
    {"team": "HOU", "park": "Minute Maid Park",       "lat": 29.7573, "lon": -95.3555,  "roof": True,  "cf_direction": 0},
    {"team": "MIA", "park": "loanDepot park",          "lat": 25.7781, "lon": -80.2197,  "roof": True,  "cf_direction": 0},
    {"team": "MIL", "park": "American Family Field",   "lat": 43.0280, "lon": -87.9712,  "roof": True,  "cf_direction": 0},
    {"team": "SEA", "park": "T-Mobile Park",           "lat": 47.5914, "lon": -122.3325, "roof": True,  "cf_direction": 0},
    {"team": "TB",  "park": "Tropicana Field",         "lat": 27.7683, "lon": -82.6534,  "roof": True,  "cf_direction": 0},
    {"team": "TEX", "park": "Globe Life Field",        "lat": 32.7512, "lon": -97.0832,  "roof": True,  "cf_direction": 0},
    {"team": "TOR", "park": "Rogers Centre",           "lat": 43.6414, "lon": -79.3894,  "roof": True,  "cf_direction": 0},

    # Outdoor stadiums — cf_direction verified from satellite imagery
    # North-facing parks (cf ~0 degrees)
    {"team": "COL", "park": "Coors Field",                    "lat": 39.7559, "lon": -104.9942, "roof": False, "cf_direction": 0},
    {"team": "CLE", "park": "Progressive Field",              "lat": 41.4962, "lon": -81.6852,  "roof": False, "cf_direction": 0},
    {"team": "SD",  "park": "Petco Park",                     "lat": 32.7073, "lon": -117.1567, "roof": False, "cf_direction": 0},

    # NNE-facing parks (cf ~22 degrees)
    {"team": "ATL", "park": "Truist Park",                    "lat": 33.8908, "lon": -84.4678,  "roof": False, "cf_direction": 22},
    {"team": "BAL", "park": "Oriole Park at Camden Yards",    "lat": 39.2838, "lon": -76.6218,  "roof": False, "cf_direction": 22},
    {"team": "CHC", "park": "Wrigley Field",                  "lat": 41.9484, "lon": -87.6553,  "roof": False, "cf_direction": 22},
    {"team": "LAD", "park": "Dodger Stadium",                 "lat": 34.0739, "lon": -118.2400, "roof": False, "cf_direction": 22},
    {"team": "NYM", "park": "Citi Field",                     "lat": 40.7571, "lon": -73.8458,  "roof": False, "cf_direction": 22},
    {"team": "WSH", "park": "Nationals Park",                 "lat": 38.8730, "lon": -77.0074,  "roof": False, "cf_direction": 22},

    # ~13 degrees (just N of NNE)
    {"team": "PHI", "park": "Citizens Bank Park",             "lat": 39.9061, "lon": -75.1665,  "roof": False, "cf_direction": 13},

    # NE-facing parks (cf ~45 degrees)
    {"team": "BOS", "park": "Fenway Park",                    "lat": 42.3467, "lon": -71.0972,  "roof": False, "cf_direction": 45},
    {"team": "KC",  "park": "Kauffman Stadium",               "lat": 39.0517, "lon": -94.4803,  "roof": False, "cf_direction": 45},
    {"team": "LAA", "park": "Angel Stadium",                  "lat": 33.8003, "lon": -117.8827, "roof": False, "cf_direction": 45},

    # ENE-facing parks (cf ~68 degrees)
    {"team": "ATH", "park": "Sutter Health Park",             "lat": 38.5803, "lon": -121.5002, "roof": False, "cf_direction": 68},
    {"team": "NYY", "park": "Yankee Stadium",                 "lat": 40.8296, "lon": -73.9262,  "roof": False, "cf_direction": 68},
    {"team": "STL", "park": "Busch Stadium",                  "lat": 38.6226, "lon": -90.1928,  "roof": False, "cf_direction": 68},

    # East-facing parks (cf ~90 degrees)
    {"team": "MIN", "park": "Target Field",                   "lat": 44.9817, "lon": -93.2781,  "roof": False, "cf_direction": 90},
    {"team": "SF",  "park": "Oracle Park",                    "lat": 37.7786, "lon": -122.3893, "roof": False, "cf_direction": 90},

    # ESE-facing parks (cf ~113 degrees)
    {"team": "CIN", "park": "Great American Ball Park",       "lat": 39.0979, "lon": -84.5082,  "roof": False, "cf_direction": 113},
    {"team": "PIT", "park": "PNC Park",                       "lat": 40.4469, "lon": -80.0057,  "roof": False, "cf_direction": 113},

    # SE-facing parks (cf ~135 degrees) — built to minimize local wind
    {"team": "CWS", "park": "Guaranteed Rate Field",          "lat": 41.8300, "lon": -87.6339,  "roof": False, "cf_direction": 135},

    # SSE-facing parks (cf ~150 degrees) — built for skyline views
    {"team": "DET", "park": "Comerica Park",                  "lat": 42.3390, "lon": -83.0485,  "roof": False, "cf_direction": 150},
]

PARK_BY_TEAM = {p["team"]: p for p in PARKS}
GAME_HOUR = 19

# Wind multipliers — blowing in penalizes more than blowing out rewards
WIND_OUT_MULTIPLIER = 1.75
WIND_IN_MULTIPLIER  = 3.25


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def get_today_games() -> Dict[str, dict]:
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
            normalized = {ESPN_TO_MLB.get(k, k): v for k, v in games.items()}
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


def fetch_weather_batch(parks_to_fetch: List[dict]) -> Dict[str, dict]:
    if not parks_to_fetch:
        return {}

    lats = ",".join(str(p["lat"]) for p in parks_to_fetch)
    lons = ",".join(str(p["lon"]) for p in parks_to_fetch)

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lats}"
        f"&longitude={lons}"
        f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
        f"&temperature_unit=fahrenheit"
        f"&windspeed_unit=mph"
        f"&forecast_days=1"
        f"&timezone=auto"
    )

    results: Dict[str, dict] = {}

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                for i, park_data in enumerate(data):
                    team = parks_to_fetch[i]["team"]
                    hourly = park_data.get("hourly", {})
                    temps = hourly.get("temperature_2m", [])
                    winds = hourly.get("windspeed_10m", [])
                    dirs  = hourly.get("winddirection_10m", [])
                    if temps:
                        idx = min(GAME_HOUR, len(temps) - 1)
                        results[team] = {
                            "temp_f":       round(temps[idx], 1),
                            "wind_mph":     round(winds[idx], 1) if winds else None,
                            "wind_dir_deg": round(dirs[idx],  1) if dirs  else None,
                        }
                print(f"Bulk weather fetch succeeded for {len(results)} parks.")
                return results
            elif isinstance(data, dict):
                team = parks_to_fetch[0]["team"]
                hourly = data.get("hourly", {})
                temps = hourly.get("temperature_2m", [])
                winds = hourly.get("windspeed_10m", [])
                dirs  = hourly.get("winddirection_10m", [])
                if temps:
                    idx = min(GAME_HOUR, len(temps) - 1)
                    results[team] = {
                        "temp_f":       round(temps[idx], 1),
                        "wind_mph":     round(winds[idx], 1) if winds else None,
                        "wind_dir_deg": round(dirs[idx],  1) if dirs  else None,
                    }
                return results

        except requests.exceptions.Timeout:
            wait = 2 ** attempt
            print(f"  Bulk weather timeout (attempt {attempt+1}/3) — retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"  Bulk weather fetch failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    print("Bulk fetch failed — falling back to individual park requests...")
    for park in parks_to_fetch:
        team = park["team"]
        single_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={park['lat']}&longitude={park['lon']}"
            f"&hourly=temperature_2m,windspeed_10m,winddirection_10m"
            f"&temperature_unit=fahrenheit"
            f"&windspeed_unit=mph"
            f"&forecast_days=1"
            f"&timezone=auto"
        )
        for attempt in range(3):
            try:
                resp = requests.get(single_url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                hourly = data.get("hourly", {})
                temps = hourly.get("temperature_2m", [])
                winds = hourly.get("windspeed_10m", [])
                dirs  = hourly.get("winddirection_10m", [])
                if temps:
                    idx = min(GAME_HOUR, len(temps) - 1)
                    results[team] = {
                        "temp_f":       round(temps[idx], 1),
                        "wind_mph":     round(winds[idx], 1) if winds else None,
                        "wind_dir_deg": round(dirs[idx],  1) if dirs  else None,
                    }
                    print(f"  ✓ {park['park']} ({team})")
                break
            except requests.exceptions.Timeout:
                wait = 2 ** attempt
                print(f"  Timeout for {team} (attempt {attempt+1}/3) — retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"  Failed for {team}: {e}")
                break

    return results


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

    # Temperature effect — each 10°F above/below 72°F adds/subtracts 0.5
    temp_delta = (temp_f - 72) / 10
    if temp_delta >= 0:
        score += temp_delta * 0.5   # heat bonus
    else:
        score += temp_delta * 0.8   # cold penalty hits harder

    if not pd.isna(wind_mph) and not pd.isna(wind_dir) and wind_mph > 0:
        wind_to = (wind_dir + 180) % 360
        diff = abs(angle_difference(wind_to, cf_direction))

        # alignment: +1.0 = straight out, -1.0 = straight in, 0 = crosswind
        alignment = (90 - diff) / 90

        # Asymmetric multipliers — wind in penalizes more than wind out rewards
        if alignment >= 0:
            multiplier = WIND_OUT_MULTIPLIER
        else:
            multiplier = WIND_IN_MULTIPLIER

        score += alignment * (wind_mph / 10) * multiplier

    return round(score, 2)


def build_weather_table(games: Dict[str, dict]) -> pd.DataFrame:
    if games:
        playing_home_teams = set(games.keys())
        unmatched = playing_home_teams - set(PARK_BY_TEAM.keys())
        if unmatched:
            print(f"Warning: No park data for teams: {unmatched} — fetching all parks")
            playing_home_teams = set(PARK_BY_TEAM.keys())
    else:
        print("No games found — fetching weather for all 30 parks.")
        playing_home_teams = set(PARK_BY_TEAM.keys())

    outdoor_parks = [
        PARK_BY_TEAM[team] for team in playing_home_teams
        if team in PARK_BY_TEAM and not PARK_BY_TEAM[team]["roof"]
    ]
    roof_parks = [
        PARK_BY_TEAM[team] for team in playing_home_teams
        if team in PARK_BY_TEAM and PARK_BY_TEAM[team]["roof"]
    ]

    print(f"Fetching weather for {len(outdoor_parks)} outdoor parks (batch)...")
    print(f"Skipping weather fetch for {len(roof_parks)} roof stadiums (always neutral).")

    weather_data = fetch_weather_batch(outdoor_parks)

    rows = []

    for park in PARKS:
        team = park["team"]
        if team not in playing_home_teams:
            continue

        roof = park["roof"]
        cf_direction = park["cf_direction"]
        game_info = games.get(team, {})

        if roof:
            rows.append({
                "home_team":        team,
                "away_team":        game_info.get("away_team", ""),
                "park":             park["park"],
                "has_roof":         True,
                "temp_f":           72.0,
                "wind_mph":         0.0,
                "wind_dir_deg":     0.0,
                "wind_direction":   "N/A",
                "wind_context":     "Roof — weather neutral",
                "hr_weather_boost": 0.0,
                "boost_note":       "Roof — weather neutral",
            })
        else:
            w = weather_data.get(team, {})
            temp_f   = w.get("temp_f")
            wind_mph = w.get("wind_mph")
            wind_dir = w.get("wind_dir_deg")

            if not w:
                wind_label   = "Unknown"
                wind_context = "Weather unavailable"
                boost        = 0.0
                boost_note   = "Weather unavailable"
            else:
                wind_label = wind_direction_label(wind_dir) if wind_dir is not None else "Unknown"
                boost = hr_weather_boost(
                    temp_f   or 72.0,
                    wind_mph or 0.0,
                    wind_dir or 0.0,
                    roof,
                    cf_direction,
                )

                if wind_mph and wind_mph >= 5:
                    diff = abs(angle_difference(
                        (wind_dir + 180) % 360 if wind_dir else 0,
                        cf_direction
                    ))
                    if diff <= 45:
                        wind_context = f"Wind blowing OUT at {wind_mph:.0f} mph"
                    elif diff >= 135:
                        wind_context = f"Wind blowing IN at {wind_mph:.0f} mph"
                    else:
                        wind_context = f"Crosswind at {wind_mph:.0f} mph"
                else:
                    wind_context = "Calm"

                boost_note = (
                    "HR friendly"    if boost >= 1.0
                    else "HR suppressing" if boost <= -1.0
                    else "Neutral"
                )

            rows.append({
                "home_team":        team,
                "away_team":        game_info.get("away_team", ""),
                "park":             park["park"],
                "has_roof":         False,
                "temp_f":           temp_f,
                "wind_mph":         wind_mph,
                "wind_dir_deg":     wind_dir,
                "wind_direction":   wind_label,
                "wind_context":     wind_context,
                "hr_weather_boost": boost,
                "boost_note":       boost_note,
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

    print("Fetching weather...")
    weather_df = build_weather_table(games)
    print(f"Built weather rows for {len(weather_df)} parks")

    if not weather_df.empty:
        print(weather_df[[
            "home_team", "park", "temp_f", "wind_mph",
            "wind_context", "hr_weather_boost"
        ]].to_string(index=False))

    write_dataframe_to_sheet(gc, sheet_id, "Weather", weather_df)
    print("Written to Weather")


if __name__ == "__main__":
    main()
