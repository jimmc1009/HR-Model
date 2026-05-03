"""
hrrbi_picks.py
H+R+RBI model — absolute threshold scoring.
Reads from HRRBI_Statcast, Pitcher_Statcast_2026, Park_Factors,
Weather, HRRBI_Odds sheets.
Outputs Top_HRRBI_Picks + HRRBI_Picks_Log + HRRBI_Scorecard.
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Set

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import pytz
import requests

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

LEAGUE_AVG_BA        = 0.248
LEAGUE_AVG_OBP       = 0.318
LEAGUE_AVG_WOBA      = 0.318
LEAGUE_AVG_XWOBA     = 0.318
LEAGUE_AVG_ISO       = 0.155
LEAGUE_AVG_EV_7D     = 89.0
LEAGUE_AVG_HH_7D     = 40.0
LEAGUE_AVG_BARREL_7D = 11.0

MIN_PA               = 80
MIN_BBE_7D           = 5
MIN_BATTING_AVG      = 0.190
MAX_PER_TEAM         = 2
MAX_CHALK_PICKS      = 3
CHALK_ODDS_THRESHOLD = 165
TOP_N                = 10

COLOR_BG     = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_HEADER = {"red": 0.055, "green": 0.318, "blue": 0.580}
COLOR_WHITE  = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GREEN  = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_RED    = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_GOLD   = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_ORANGE = {"red": 0.980, "green": 0.502, "blue": 0.059}


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries: int = 4, wait: int = 25):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                print(f"  Rate limit hit — waiting {wait}s (attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                print(f"  API error: {e} — retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def read_sheet(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    try:
        sh   = with_retry(lambda: gc.open_by_key(sheet_id))
        ws   = sh.worksheet(name)
        data = with_retry(lambda: ws.get_all_records())
        return pd.DataFrame(data)
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read sheet '{name}': {e}")
        return pd.DataFrame()


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def normalize_name(name: str) -> str:
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def get_todays_game_times() -> dict:
    game_times = {}
    try:
        today = date.today().strftime("%Y-%m-%d")
        url   = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=team"
        resp  = requests.get(url, timeout=15)
        resp.raise_for_status()
        for d in resp.json().get("dates", []):
            for game in d.get("games", []):
                home_abbr = game.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                game_time = game.get("gameDate", "")
                if home_abbr and game_time:
                    try:
                        gt = datetime.strptime(game_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
                        game_times[home_abbr.upper()] = gt
                    except Exception:
                        pass
    except Exception as e:
        print(f"  WARNING: Could not fetch game times: {e}")
    return game_times


def filter_started_games(df: pd.DataFrame, game_times: dict, team_col: str = "batter_team") -> pd.DataFrame:
    if not game_times or team_col not in df.columns:
        return df

    now_utc = datetime.now(pytz.utc)
    before  = len(df)

    def game_started(team: str) -> bool:
        gt = game_times.get(str(team).upper())
        if gt is None:
            return False
        return now_utc >= gt

    mask    = ~df[team_col].apply(game_started)
    df      = df[mask].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"Game time filter: {removed} players removed (game already started), {len(df)} remaining")
    return df


def score_avg(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 0.320: return 2.0
    if v >= 0.300: return 1.5
    if v >= 0.280: return 1.0
    if v >= 0.260: return 0.5
    if v >= 0.240: return 0.2
    return 0.0

def score_xwoba(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 0.390: return 2.0
    if v >= 0.360: return 1.5
    if v >= 0.340: return 1.0
    if v >= 0.320: return 0.5
    if v >= 0.300: return 0.2
    return 0.0

def score_woba(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 0.380: return 2.0
    if v >= 0.350: return 1.5
    if v >= 0.330: return 1.0
    if v >= 0.310: return 0.5
    if v >= 0.290: return 0.2
    return 0.0

def score_obp(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 0.400: return 1.5
    if v >= 0.370: return 1.0
    if v >= 0.350: return 0.6
    if v >= 0.330: return 0.3
    return 0.0

def score_iso(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 0.280: return 1.2
    if v >= 0.230: return 0.9
    if v >= 0.190: return 0.6
    if v >= 0.165: return 0.3
    return 0.0

def score_ld_pct(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 28.0: return 1.2
    if v >= 24.0: return 0.8
    if v >= 21.0: return 0.4
    if v >= 18.0: return 0.1
    return 0.0

def score_hard_hit_season(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 52.0: return 0.8
    if v >= 45.0: return 0.5
    if v >= 38.0: return 0.2
    return 0.0

def score_avg_ev_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < MIN_BBE_7D: return 0.0
    if v >= 97.0: return 1.0
    if v >= 94.0: return 0.6
    if v >= 91.0: return 0.3
    return 0.0

def score_hard_hit_7d(v: float, bbe_7d: float) -> float:
    if bbe_7d < MIN_BBE_7D: return 0.0
    if v >= 55.0: return 0.8
    if v >= 45.0: return 0.5
    if v >= 35.0: return 0.2
    return 0.0

def score_rolling_avg(v: float, pa: float) -> float:
    if pa < 20: return 0.0
    if v >= 0.360: return 1.5
    if v >= 0.320: return 1.0
    if v >= 0.290: return 0.5
    if v >= 0.260: return 0.2
    if v <= 0.180: return -1.0
    if v <= 0.210: return -0.5
    return 0.0

def score_bat_order(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v <= 2.0: return 1.5
    if v <= 3.5: return 1.2
    if v <= 5.0: return 1.0
    if v <= 6.5: return 0.4
    return 0.0

def score_speed(v: float) -> float:
    if v >= 7.0: return 0.8
    if v >= 5.5: return 0.4
    if v >= 4.0: return 0.1
    return 0.0

def score_bb_pct(v: float, pa: float) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 14.0: return 0.8
    if v >= 10.0: return 0.5
    if v >=  8.0: return 0.2
    return 0.0

def score_opp_whip(v: float) -> float:
    if v >= 1.50: return 1.2
    if v >= 1.35: return 0.8
    if v >= 1.20: return 0.4
    if v >= 1.10: return 0.1
    if v <= 0.95: return -0.6
    return 0.0

def score_opp_k_pct(v: float) -> float:
    if v >= 30.0: return -1.2
    if v >= 26.0: return -0.8
    if v >= 22.0: return -0.4
    if v <= 16.0: return  0.6
    if v <= 18.0: return  0.3
    return 0.0

def score_opp_hard_hit(v: float) -> float:
    if v >= 45.0: return  0.8
    if v >= 40.0: return  0.4
    if v <= 30.0: return -0.6
    if v <= 35.0: return -0.3
    return 0.0

def score_park_hits(v: float) -> float:
    norm = v - 100
    if norm >= 20: return  0.5
    if norm >= 10: return  0.3
    if norm >=  0: return  0.1
    if norm >= -10: return -0.1
    return -0.3

def score_weather_hits(boost: float) -> float:
    if boost >= 1.5: return  0.5
    if boost >= 0.5: return  0.2
    if boost <= -1.5: return -0.5
    if boost <= -0.5: return -0.2
    return 0.0


def compute_hrrbi_score(row: pd.Series) -> float:
    pa     = safe_float(row.get("pa", 0))
    bbe_7d = safe_float(row.get("bbe_7d", 0))
    pa_14d = safe_float(row.get("pa_14d", 0))

    woba_score = score_woba(safe_float(row.get("woba")), pa)

    s_avg      = score_avg(safe_float(row.get("avg")), pa)
    s_obp      = score_obp(safe_float(row.get("obp")), pa)
    s_iso      = score_iso(safe_float(row.get("iso")), pa)
    s_ld       = score_ld_pct(safe_float(row.get("ld_pct")), pa)
    s_hh_s     = score_hard_hit_season(safe_float(row.get("hard_hit_pct_season")), pa)
    s_ev_7d    = score_avg_ev_7d(safe_float(row.get("avg_ev_7d")), bbe_7d)
    s_hh_7d    = score_hard_hit_7d(safe_float(row.get("hard_hit_pct_7d")), bbe_7d)
    s_roll_avg = score_rolling_avg(safe_float(row.get("avg_14d")), pa_14d)
    s_bat_ord  = score_bat_order(safe_float(row.get("avg_bat_order", 6)), pa)
    s_speed    = score_speed(safe_float(row.get("speed_score")))
    s_bb       = score_bb_pct(safe_float(row.get("bb_pct")), pa)
    s_opp_whip = score_opp_whip(safe_float(row.get("opp_whip", 1.20), 1.20))
    s_opp_k    = score_opp_k_pct(safe_float(row.get("opp_k_pct_season", 22.0), 22.0))
    s_opp_hh   = score_opp_hard_hit(safe_float(row.get("opp_hard_hit_pct", 38.0), 38.0))
    s_park     = score_park_hits(safe_float(row.get("park_hr_factor", 100), 100))
    s_weather  = score_weather_hits(safe_float(row.get("hr_weather_boost", 0)))

    return round(
        s_avg + woba_score + s_obp + s_iso + s_ld + s_hh_s +
        s_ev_7d + s_hh_7d + s_roll_avg +
        s_bat_ord + s_speed + s_bb +
        s_opp_whip + s_opp_k + s_opp_hh +
        s_park + s_weather, 3
    )


def assign_confidence(row: pd.Series) -> str:
    pa      = safe_float(row.get("pa", 0))
    bbe_7d  = safe_float(row.get("bbe_7d", 0))
    opp_bbe = safe_float(row.get("opp_pitcher_bbe", 0))

    pts = 0
    if pa >= 150:  pts += 2
    elif pa >= 80: pts += 1
    if bbe_7d >= 20: pts += 1
    if opp_bbe >= 80:  pts += 2
    elif opp_bbe >= 40: pts += 1

    if pts >= 4: return "High"
    if pts >= 2: return "Medium"
    return "Low"


def calc_prop_signal(row: pd.Series) -> str:
    """
    Generate a betting signal based on the H+R+RBI line.
    Ignores 0.5 lines — too easy, no edge.
    Targets 1.5 as primary signal, 2.5 for elite scores only.
    """
    line  = safe_float(row.get("hrrbi_line", 0))
    score = safe_float(row.get("hrrbi_score", 0))

    if line <= 0:
        return "—"

    # Ignore 0.5 lines — no edge
    if line < 1.0:
        return "—"

    if line >= 2.0:
        # 2.5 line — only flag elite scores
        if score >= 12.0:
            return f"LEAN OVER {line}"
        return "—"

    # 1.5 line — primary target
    if score >= 11.0:
        return f"OVER {line} ✅"
    if score >= 9.0:
        return f"LEAN OVER {line}"
    if score <= 5.0:
        return f"LEAN UNDER {line}"
    return "—"


def build_reason(row: pd.Series) -> str:
    reasons = []

    avg = safe_float(row.get("avg"))
    if avg >= 0.300:
        reasons.append(f"🎯 .{int(avg * 1000)} season avg")

    roll = safe_float(row.get("avg_14d"))
    pa14 = safe_float(row.get("pa_14d"))
    if pa14 >= 20:
        if roll >= 0.320:
            reasons.append(f"🔥 .{int(roll * 1000)} avg last 14 days")
        elif roll <= 0.200:
            reasons.append(f"❄️ .{int(roll * 1000)} avg last 14 days — cold")

    woba = safe_float(row.get("woba"))
    if woba >= 0.360:
        reasons.append(f"⚡ wOBA {woba:.3f}")

    obp = safe_float(row.get("obp"))
    if obp >= 0.370:
        reasons.append(f"👟 OBP {obp:.3f} — gets on base")

    order = safe_float(row.get("avg_bat_order", 6))
    if order <= 2.0:
        reasons.append("📋 Leadoff/2-hole — max run scoring spot")
    elif order <= 5.0:
        reasons.append(f"📋 Bats {order:.1f} — prime RBI spot")

    ld = safe_float(row.get("ld_pct"))
    if ld >= 24.0:
        reasons.append(f"📐 {ld:.1f}% LD% — elite contact quality")

    whip = safe_float(row.get("opp_whip", 1.20), 1.20)
    if whip >= 1.35:
        reasons.append(f"🎯 Opp pitcher WHIP {whip:.2f} — hittable")

    opp_k = safe_float(row.get("opp_k_pct_season", 22.0), 22.0)
    if opp_k <= 18.0:
        reasons.append(f"✅ Opp K% {opp_k:.1f}% — contact-friendly matchup")
    elif opp_k >= 28.0:
        reasons.append(f"⚠️ Opp K% {opp_k:.1f}% — strikeout risk")

    park = safe_float(row.get("park_hr_factor", 100), 100)
    park_name = str(row.get("park_name", ""))
    if park >= 110:
        reasons.append(f"🏟️ Hitter-friendly park ({park_name})")

    if not reasons:
        reasons.append("Solid across multiple factors")

    return " | ".join(reasons)


def prepare_combined(
    batters: pd.DataFrame,
    pitchers: pd.DataFrame,
    parks: pd.DataFrame,
    weather: pd.DataFrame,
    odds_df: pd.DataFrame,
    game_times: dict,
) -> pd.DataFrame:
    if batters.empty:
        print("No HRRBI batter data.")
        return pd.DataFrame()

    batters = batters.copy()
    batters.columns = [c.strip() for c in batters.columns]

    if "player_name" in batters.columns:
        batters["player_name_norm"] = batters["player_name"].apply(normalize_name)
    if "team" in batters.columns:
        batters = batters.rename(columns={"team": "batter_team"})

    batters = filter_started_games(batters, game_times, team_col="batter_team")
    if batters.empty:
        return pd.DataFrame()

    if not pitchers.empty:
        pitchers = pitchers.copy()
        pitchers.columns = [c.strip() for c in pitchers.columns]

        pitcher_rename = {
            "opposing_team":             "batter_team",
            "pitcher_name":              "opp_pitcher_name",
            "pitcher_hand":              "opp_pitcher_hand",
            "season_bbe_allowed":        "opp_pitcher_bbe",
            "hard_hit_pct_allowed":      "opp_hard_hit_pct",
            "season_barrel_pct_allowed": "opp_barrel_pct",
            "hr_per_fb_allowed":         "opp_hr_per_fb",
            "avg_ev_allowed":            "opp_avg_ev",
        }
        pitchers = pitchers.rename(columns={k: v for k, v in pitcher_rename.items() if k in pitchers.columns})

        if "opp_k_pct_season" not in pitchers.columns:
            pitchers["opp_k_pct_season"] = 22.0
        if "opp_whip" not in pitchers.columns:
            pitchers["opp_whip"] = 1.20

        pitcher_cols = [c for c in [
            "batter_team", "opp_pitcher_name", "opp_pitcher_hand",
            "opp_pitcher_bbe", "opp_hard_hit_pct", "opp_barrel_pct",
            "opp_hr_per_fb", "opp_k_pct_season", "opp_whip",
        ] if c in pitchers.columns]

        if "batter_team" in pitchers.columns:
            batters = batters.merge(pitchers[pitcher_cols], on="batter_team", how="left")

    if not parks.empty:
        parks = parks.copy()
        parks.columns = [c.strip() for c in parks.columns]
        park_cols = [c for c in ["team", "park_hr_factor", "park_name"] if c in parks.columns]
        if park_cols:
            batters = batters.merge(
                parks[park_cols],
                left_on="batter_team",
                right_on="team",
                how="left",
            )

    if not weather.empty:
        weather = weather.copy()
        weather.columns = [c.strip() for c in weather.columns]
        weather_cols = [c for c in ["home_team", "hr_weather_boost", "wind_context", "temp_f"] if c in weather.columns]
        if "home_team" in weather.columns:
            batters = batters.merge(
                weather[weather_cols],
                left_on="batter_team",
                right_on="home_team",
                how="left",
            )

    defaults = {
        "opp_whip": 1.20,
        "opp_k_pct_season": 22.0,
        "opp_hard_hit_pct": 38.0,
        "park_hr_factor": 100.0,
        "hr_weather_boost": 0.0,
        "avg_bat_order": 5.0,
        "speed_score": 4.0,
        "ld_pct": 0.0,
        "bb_pct": 0.0,
    }
    for col, val in defaults.items():
        if col not in batters.columns:
            batters[col] = val
        else:
            batters[col] = batters[col].apply(lambda x: safe_float(x, val))

    if "avg" in batters.columns:
        batters["avg"] = batters["avg"].apply(safe_float)
        before  = len(batters)
        batters = batters[batters["avg"] >= MIN_BATTING_AVG].copy()
        print(f"BA filter: {before - len(batters)} removed, {len(batters)} remaining")

    if batters.empty:
        return pd.DataFrame()

    batters["hrrbi_score"] = batters.apply(compute_hrrbi_score, axis=1)
    batters["confidence"]  = batters.apply(assign_confidence, axis=1)
    batters["reason"]      = batters.apply(build_reason, axis=1)

    if not odds_df.empty and "player_name_norm" in odds_df.columns:
        odds_slim = odds_df[["player_name_norm", "hrrbi_line", "over_odds", "under_odds"]].copy()
        odds_slim = odds_slim.rename(columns={
            "over_odds":  "hrrbi_over_odds",
            "under_odds": "hrrbi_under_odds",
        })
        batters = batters.merge(odds_slim, on="player_name_norm", how="left")
    else:
        batters["hrrbi_line"]       = np.nan
        batters["hrrbi_over_odds"]  = np.nan
        batters["hrrbi_under_odds"] = np.nan

    # Generate prop signal after odds merge
    batters["prop_signal"] = batters.apply(calc_prop_signal, axis=1)

    return batters


def apply_diversity_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("hrrbi_score", ascending=False).reset_index(drop=True)
    selected    = []
    team_counts = {}
    chalk_count = 0
    no_odds     = []

    for _, row in df.iterrows():
        odds = row.get("hrrbi_over_odds")
        try:
            odds_val = float(odds)
            has_odds = True
        except (TypeError, ValueError):
            has_odds = False
            no_odds.append(row)
            continue

        if len(selected) >= TOP_N:
            break

        team = str(row.get("batter_team", "UNK"))
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue

        is_chalk = has_odds and abs(odds_val) >= CHALK_ODDS_THRESHOLD and odds_val < 0
        if is_chalk and chalk_count >= MAX_CHALK_PICKS:
            continue

        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1
        if is_chalk:
            chalk_count += 1

    for row in no_odds:
        if len(selected) >= TOP_N:
            break
        team = str(row.get("batter_team", "UNK"))
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue
        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1

    if not selected:
        return pd.DataFrame()

    result = pd.DataFrame(selected).reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def write_picks_to_sheet(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Top_HRRBI_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_HRRBI_Picks", rows=100, cols=15)

    output_cols = {
        "rank":             "Rank",
        "player_name":      "Batter",
        "batter_team":      "Team",
        "opp_pitcher_name": "Opp Pitcher",
        "hrrbi_line":       "Line",
        "hrrbi_over_odds":  "Over Odds",
        "hrrbi_under_odds": "Under Odds",
        "prop_signal":      "Signal",
        "reason":           "Key Reasons",
        "confidence":       "Confidence",
    }

    available = {k: v for k, v in output_cols.items() if k in picks.columns}
    out_df    = picks[list(available.keys())].rename(columns=available)
    out_df    = out_df.copy().replace([np.inf, -np.inf], np.nan).fillna("")

    with_retry(lambda: ws.update([out_df.columns.tolist()] + out_df.astype(str).values.tolist()))

    ws_id  = ws.id
    n_cols = len(out_df.columns)
    reqs   = []

    # Base formatting — no wrap, clip
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": len(out_df) + 2, "startColumnIndex": 0, "endColumnIndex": n_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})

    # Header row
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": n_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_HEADER,
            "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True, "fontFamily": "Roboto", "fontSize": 11},
            "horizontalAlignment": "CENTER",
            "verticalAlignment": "MIDDLE",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
    }})

    # Alternating row colors
    for i in range(len(out_df)):
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2, "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)",
        }})

    # Column widths: Rank, Batter, Team, Opp Pitcher, Line, Over Odds, Under Odds, Signal, Key Reasons, Confidence
    col_widths = [45, 160, 55, 160, 55, 80, 85, 150, 320, 90]
    for i, w in enumerate(col_widths[:n_cols]):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    # Row height — compact
    for i in range(len(out_df) + 1):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": 32},
            "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}, "tabColorStyle": {"rgbColor": COLOR_HEADER}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
    except APIError as e:
        print(f"HRRBI formatting failed: {e}")

    print(f"Written {len(out_df)} H+R+RBI picks to Top_HRRBI_Picks")

def log_picks(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame) -> None:
    today_str = date.today().strftime("%Y-%m-%d")
    sh        = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws       = sh.worksheet("HRRBI_Picks_Log")
        existing = pd.DataFrame(with_retry(lambda: ws.get_all_records()))
    except gspread.WorksheetNotFound:
        ws       = sh.add_worksheet(title="HRRBI_Picks_Log", rows=5000, cols=20)
        existing = pd.DataFrame()

    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"] != today_str].copy()

    new_rows = []
    for _, row in picks.iterrows():
        new_rows.append({
            "date":        today_str,
            "rank":        str(row.get("rank", "")),
            "player_name": str(row.get("player_name", "")),
            "team":        str(row.get("batter_team", "")),
            "hrrbi_score": str(row.get("hrrbi_score", "")),
            "confidence":  str(row.get("confidence", "")),
            "line":        str(row.get("hrrbi_line", "")),
            "over_odds":   str(row.get("hrrbi_over_odds", "")),
            "prop_signal": str(row.get("prop_signal", "")),
            "hit":         "Pending",
            "bet_placed":  "",
            "result":      "",
        })

    if not new_rows:
        return

    new_df       = pd.DataFrame(new_rows)
    combined_log = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined_log = combined_log.fillna("")

    time.sleep(5)
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([combined_log.columns.tolist()] + combined_log.astype(str).values.tolist()))
    print(f"Logged {len(new_rows)} H+R+RBI picks to HRRBI_Picks_Log")


def update_scorecard(gc: gspread.Client, sheet_id: str) -> None:
    time.sleep(15)
    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws_log = sh.worksheet("HRRBI_Picks_Log")
        log    = pd.DataFrame(with_retry(lambda: ws_log.get_all_records()))
    except gspread.WorksheetNotFound:
        print("HRRBI_Picks_Log not found — skipping scorecard")
        return

    if log.empty:
        return

    scored = log[log["hit"].isin(["Yes", "No"])].copy()
    if scored.empty:
        print("No scored H+R+RBI picks yet — skipping scorecard.")
        return

    scored["hit_bool"]  = scored["hit"] == "Yes"
    scored["score_num"] = pd.to_numeric(scored["hrrbi_score"], errors="coerce")

    rows = []

    def add_row(label, sub):
        if sub.empty:
            return
        total = len(sub)
        hits  = int(sub["hit_bool"].sum())
        rows.append([label, total, hits, f"{hits / total * 100:.1f}%"])

    add_row("All Picks", scored)
    for tier, mask in [
        ("Score 12+",     scored["score_num"] >= 12),
        ("Score 10+",     scored["score_num"] >= 10),
        ("Score 8+",      scored["score_num"] >= 8),
        ("Score Under 8", scored["score_num"] <  8),
    ]:
        add_row(tier, scored[mask])

    for conf in ["High", "Medium", "Low"]:
        add_row(f"Confidence: {conf}", scored[scored["confidence"] == conf])

    time.sleep(5)
    try:
        sc_ws = sh.worksheet("HRRBI_Scorecard")
        with_retry(lambda: sc_ws.clear())
    except gspread.WorksheetNotFound:
        sc_ws = sh.add_worksheet(title="HRRBI_Scorecard", rows=50, cols=6)

    with_retry(lambda: sc_ws.update([["Category", "Picks", "Hits", "Hit Rate"]] + rows))
    print("HRRBI_Scorecard updated")


def write_timestamp(gc: gspread.Client, sheet_id: str) -> None:
    et     = pytz.timezone("America/New_York")
    now_et = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")
    sh     = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Top_HRRBI_Picks")
        with_retry(lambda: ws.insert_row([f"⏱  Last Run: {now_et}"], index=1))
    except Exception as e:
        print(f"HRRBI timestamp failed: {e}")
    print(f"H+R+RBI timestamp written: {now_et}")


def main() -> None:
    time.sleep(10)

    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HRRBI data from Google Sheets...")
    batters  = read_sheet(gc, sheet_id, "HRRBI_Statcast")
    time.sleep(3)
    pitchers = read_sheet(gc, sheet_id, "Pitcher_Statcast_2026")
    time.sleep(3)
    parks    = read_sheet(gc, sheet_id, "Park_Factors")
    time.sleep(3)
    weather  = read_sheet(gc, sheet_id, "Weather")
    time.sleep(3)
    odds_df  = read_sheet(gc, sheet_id, "HRRBI_Odds")

    print(f"Batters: {len(batters)} rows")
    print(f"Pitchers: {len(pitchers)} rows")
    print(f"Parks: {len(parks)} rows")
    print(f"Weather: {len(weather)} rows")
    print(f"HRRBI Odds: {len(odds_df)} rows")

    if not odds_df.empty and "player_name" in odds_df.columns:
        odds_df["player_name_norm"] = odds_df["player_name"].apply(normalize_name)

    print("Fetching game start times...")
    game_times = get_todays_game_times()
    print(f"  Found {len(game_times)} game times")

    combined = prepare_combined(batters, pitchers, parks, weather, odds_df, game_times)

    if combined.empty:
        print("WARNING: No combined H+R+RBI data.")
        return

    picks = apply_diversity_cap(combined)

    if picks.empty:
        print("WARNING: No H+R+RBI picks generated.")
        return

    print(f"\nTop {len(picks)} H+R+RBI Picks:")
    print(picks[["rank", "player_name", "batter_team", "hrrbi_score", "confidence"]].to_string(index=False))

    write_picks_to_sheet(gc, sheet_id, picks)
    time.sleep(5)
    log_picks(gc, sheet_id, picks)
    time.sleep(5)
    update_scorecard(gc, sheet_id)
    time.sleep(5)
    write_timestamp(gc, sheet_id)


if __name__ == "__main__":
    main()
