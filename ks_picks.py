"""
ks_picks.py
Pitcher Strikeout model — absolute threshold scoring.
Reads from KS_Statcast, Pitcher_Statcast_2026 (today's probables),
Team_K_Rates, Park_Factors, KS_Odds sheets.
Outputs Top_KS_Picks + KS_Picks_Log + KS_Scorecard.
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime, timedelta

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

# ── Constants ──────────────────────────────────────────────────────────────
LEAGUE_AVG_K_PCT     = 22.5
LEAGUE_AVG_SWSTR     = 11.0
LEAGUE_AVG_CHASE     = 29.0
LEAGUE_AVG_K_PER_9   = 8.8
LEAGUE_AVG_OPP_K_PCT = 22.5

MIN_IP          = 10
MIN_GS          = 2
MAX_PER_TEAM    = 1
MAX_CHALK_PICKS = 4
TOP_N           = 10

COLOR_BG     = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_HEADER = {"red": 0.075, "green": 0.376, "blue": 0.227}
COLOR_WHITE  = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GREEN  = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_RED    = {"red": 0.910, "green": 0.259, "blue": 0.259}


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


def filter_started_games(df: pd.DataFrame, game_times: dict, team_col: str = "team") -> pd.DataFrame:
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
        print(f"Game time filter: {removed} pitchers removed (game already started), {len(df)} remaining")
    return df


# ── Scoring functions ──────────────────────────────────────────────────────

def score_k_pct_season(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 32.0: return 2.5
    if v >= 28.0: return 2.0
    if v >= 26.0: return 1.5
    if v >= 24.0: return 1.0
    if v >= 22.0: return 0.5
    if v >= 20.0: return 0.2
    if v <= 16.0: return -0.5
    return 0.0

def score_swstr_pct(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 15.0: return 2.5
    if v >= 13.5: return 2.0
    if v >= 12.5: return 1.5
    if v >= 11.5: return 1.0
    if v >= 10.5: return 0.5
    if v >=  9.5: return 0.1
    if v <=  8.0: return -0.5
    return 0.0

def score_chase_rate(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 35.0: return 1.5
    if v >= 33.0: return 1.0
    if v >= 31.0: return 0.6
    if v >= 29.0: return 0.2
    if v <= 24.0: return -0.5
    if v <= 26.0: return -0.2
    return 0.0

def score_k_per_9(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 13.0: return 1.5
    if v >= 11.5: return 1.2
    if v >= 10.5: return 0.9
    if v >=  9.5: return 0.6
    if v >=  8.5: return 0.3
    if v <=  6.5: return -0.5
    if v <=  7.5: return -0.2
    return 0.0

def score_k_minus_bb(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 22.0: return 1.2
    if v >= 18.0: return 0.9
    if v >= 15.0: return 0.6
    if v >= 12.0: return 0.3
    if v <=  8.0: return -0.5
    return 0.0

def score_fastball_velo(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 98.0: return 1.5
    if v >= 96.0: return 1.0
    if v >= 94.0: return 0.6
    if v >= 92.0: return 0.2
    if v <= 89.0: return -0.5
    if v <= 91.0: return -0.2
    return 0.0

def score_swstr_trend(v: float) -> float:
    if v >= 3.0:  return 1.5
    if v >= 1.5:  return 1.0
    if v >= 0.5:  return 0.4
    if v <= -3.0: return -1.5
    if v <= -1.5: return -1.0
    if v <= -0.5: return -0.4
    return 0.0

def score_k_per_start_21d(v: float) -> float:
    if v <= 0:   return 0.0
    if v >= 9.0: return 2.0
    if v >= 7.5: return 1.5
    if v >= 6.5: return 1.0
    if v >= 5.5: return 0.5
    if v >= 4.5: return 0.1
    if v <= 3.0: return -1.0
    if v <= 4.0: return -0.5
    return 0.0

def score_velo_trend(v: float) -> float:
    if v >= 1.5:  return 0.8
    if v >= 0.5:  return 0.3
    if v <= -2.0: return -1.0
    if v <= -1.0: return -0.5
    return 0.0

def score_avg_ip_per_start(v: float, gs: float) -> float:
    if gs < MIN_GS: return 0.0
    if v >= 6.0: return 1.5
    if v >= 5.5: return 1.0
    if v >= 5.0: return 0.5
    if v >= 4.5: return 0.2
    if v >= 4.0: return 0.0
    if v <  4.0: return -1.5
    return 0.0

def score_opp_team_k_pct(v: float) -> float:
    if v >= 28.0: return 2.0
    if v >= 26.0: return 1.5
    if v >= 24.0: return 1.0
    if v >= 22.0: return 0.5
    if v >= 20.0: return 0.1
    if v <= 16.0: return -1.5
    if v <= 18.0: return -0.8
    return 0.0

def score_opp_team_bb_pct(v: float) -> float:
    if v >= 12.0: return -0.8
    if v >= 10.0: return -0.4
    if v <=  7.0: return 0.4
    if v <=  8.0: return 0.2
    return 0.0

def score_first_pitch_strike(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 68.0: return 0.8
    if v >= 65.0: return 0.5
    if v >= 62.0: return 0.2
    if v <= 57.0: return -0.5
    if v <= 59.0: return -0.2
    return 0.0

def score_bb_pct_pitcher(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v <=  5.0: return 0.8
    if v <=  7.0: return 0.4
    if v >= 12.0: return -0.8
    if v >= 10.0: return -0.4
    return 0.0

def score_park_k(v: float) -> float:
    norm = v - 100
    if norm >=  8: return  0.5
    if norm >=  4: return  0.3
    if norm >=  0: return  0.0
    if norm <= -15: return -0.5
    if norm <=  -8: return -0.2
    return 0.0


def compute_ks_score(row: pd.Series) -> float:
    ip = safe_float(row.get("ip", 0))
    gs = safe_float(row.get("games_started", 0))

    s_k_pct      = score_k_pct_season(safe_float(row.get("k_pct_season")), ip)
    s_swstr      = score_swstr_pct(safe_float(row.get("swstr_pct")), ip)
    s_chase      = score_chase_rate(safe_float(row.get("chase_rate")), ip)
    s_k_per_9    = score_k_per_9(safe_float(row.get("k_per_9")), ip)
    s_k_minus_bb = score_k_minus_bb(safe_float(row.get("k_minus_bb")), ip)
    s_velo       = score_fastball_velo(safe_float(row.get("fastball_velo")), ip)
    s_swstr_tr   = score_swstr_trend(safe_float(row.get("swstr_trend")))
    s_k_21d      = score_k_per_start_21d(safe_float(row.get("k_per_start_21d")))
    s_velo_tr    = score_velo_trend(safe_float(row.get("velo_trend")))
    s_ip         = score_avg_ip_per_start(safe_float(row.get("avg_ip_per_start")), gs)
    s_opp_k      = score_opp_team_k_pct(safe_float(row.get("opp_team_k_pct", LEAGUE_AVG_OPP_K_PCT), LEAGUE_AVG_OPP_K_PCT))
    s_opp_bb     = score_opp_team_bb_pct(safe_float(row.get("opp_team_bb_pct", 8.5), 8.5))
    s_fps        = score_first_pitch_strike(safe_float(row.get("first_pitch_strike_pct")), ip)
    s_bb_p       = score_bb_pct_pitcher(safe_float(row.get("bb_pct_season")), ip)
    s_park       = score_park_k(safe_float(row.get("park_hr_factor", 100), 100))

    return round(
        s_k_pct + s_swstr + s_chase + s_k_per_9 + s_k_minus_bb +
        s_velo +
        s_swstr_tr + s_k_21d + s_velo_tr +
        s_ip +
        s_opp_k + s_opp_bb +
        s_fps + s_bb_p + s_park, 3
    )


def assign_confidence(row: pd.Series) -> str:
    ip = safe_float(row.get("ip", 0))
    gs = safe_float(row.get("games_started", 0))

    pts = 0
    if ip >= 50:   pts += 2
    elif ip >= 25: pts += 1
    if gs >= 8:    pts += 2
    elif gs >= 4:  pts += 1
    if safe_float(row.get("k_per_start_21d")) > 0:
        pts += 1

    if pts >= 4: return "High"
    if pts >= 2: return "Medium"
    return "Low"


def calc_projected_k(row: pd.Series) -> float:
    avg_ip  = safe_float(row.get("avg_ip_per_start", 5.0), 5.0)
    k_per_9 = safe_float(row.get("k_per_9", LEAGUE_AVG_K_PER_9), LEAGUE_AVG_K_PER_9)
    if avg_ip <= 0 or k_per_9 <= 0:
        return 0.0
    return round(avg_ip * (k_per_9 / 9), 1)


def calc_prop_signal(row: pd.Series) -> str:
    proj  = safe_float(row.get("projected_k_calc", 0))
    score = safe_float(row.get("ks_score", 0))
    line  = safe_float(row.get("k_line", 0))

    if line <= 0:
        return "—"

    edge = proj - line
    if edge >= 1.5 and score >= 10:
        return f"OVER {line} ✅"
    elif edge >= 0.5 and score >= 7:
        return f"LEAN OVER {line}"
    elif edge <= -1.5:
        return f"UNDER {line} consideration"
    return "—"


def build_reason(row: pd.Series) -> str:
    reasons = []

    swstr = safe_float(row.get("swstr_pct"))
    if swstr >= 13.5:
        reasons.append(f"💨 SwStr% {swstr:.1f}% — elite swing and miss")
    elif swstr >= 11.5:
        reasons.append(f"💨 SwStr% {swstr:.1f}%")

    chase = safe_float(row.get("chase_rate"))
    if chase >= 33.0:
        reasons.append(f"🎣 Chase rate {chase:.1f}% — batters chasing off zone")

    velo = safe_float(row.get("fastball_velo"))
    if velo >= 96.0:
        reasons.append(f"🔥 {velo:.1f} mph fastball")

    k21 = safe_float(row.get("k_per_start_21d"))
    if k21 >= 7.0:
        reasons.append(f"📈 {k21:.1f} Ks/start last 3 starts")
    elif k21 > 0 and k21 <= 4.0:
        reasons.append(f"📉 Only {k21:.1f} Ks/start last 3 starts")

    swstr_tr = safe_float(row.get("swstr_trend"))
    if swstr_tr >= 1.5:
        reasons.append(f"📈 SwStr% trending up {swstr_tr:+.1f}%")
    elif swstr_tr <= -1.5:
        reasons.append(f"📉 SwStr% trending down {swstr_tr:.1f}%")

    ip = safe_float(row.get("avg_ip_per_start"))
    if ip >= 6.0:
        reasons.append(f"⏱️ {ip:.1f} avg IP/start — deep into games")
    elif 0 < ip < 4.0:
        reasons.append(f"⚠️ {ip:.1f} avg IP/start — opener risk")

    opp_k = safe_float(row.get("opp_team_k_pct", LEAGUE_AVG_OPP_K_PCT), LEAGUE_AVG_OPP_K_PCT)
    if opp_k >= 25.0:
        reasons.append(f"🎯 Opp K% {opp_k:.1f}% — swing-happy lineup")
    elif opp_k <= 18.0:
        reasons.append(f"⚠️ Opp K% {opp_k:.1f}% — contact lineup")

    if not reasons:
        reasons.append("Solid K profile across multiple factors")

    return " | ".join(reasons)


def prepare_combined(
    pitchers: pd.DataFrame,
    team_k: pd.DataFrame,
    parks: pd.DataFrame,
    odds_df: pd.DataFrame,
    game_times: dict,
) -> pd.DataFrame:
    if pitchers.empty:
        print("No KS pitcher data.")
        return pd.DataFrame()

    pitchers = pitchers.copy()
    pitchers.columns = [c.strip() for c in pitchers.columns]

    if "pitcher_name" in pitchers.columns:
        pitchers["pitcher_name_norm"] = pitchers["pitcher_name"].apply(normalize_name)

    # ── Numeric coerce ─────────────────────────────────────────────
    num_cols = [
        "ip", "games_started", "k_pct_season", "bb_pct_season",
        "k_minus_bb", "k_per_9", "swstr_pct", "chase_rate",
        "first_pitch_strike_pct", "fastball_velo", "avg_ip_per_start",
        "opener_risk", "swstr_trend", "velo_trend", "k_per_start_21d",
        "avg_ip_last_3", "swstr_pct_21d", "avg_velo_21d",
    ]
    for col in num_cols:
        if col in pitchers.columns:
            pitchers[col] = pitchers[col].apply(safe_float)

    # ── Filter starters ────────────────────────────────────────────
    before   = len(pitchers)
    pitchers = pitchers[
        pitchers.get("games_started", pd.Series(0, index=pitchers.index)).apply(safe_float) >= MIN_GS
    ].copy()
    pitchers = pitchers[
        pitchers.get("ip", pd.Series(0, index=pitchers.index)).apply(safe_float) >= MIN_IP
    ].copy()
    if "opener_risk" in pitchers.columns:
        pitchers = pitchers[pitchers["opener_risk"].apply(safe_float) == 0].copy()
    print(f"Starter filter: {before - len(pitchers)} removed, {len(pitchers)} remaining")

    if pitchers.empty:
        return pd.DataFrame()

    # ── Game time filter ───────────────────────────────────────────
    team_col = "pitching_team" if "pitching_team" in pitchers.columns else "pitcher_team"
    if team_col not in pitchers.columns:
        team_col = "team"
    pitchers = filter_started_games(pitchers, game_times, team_col=team_col)
    if pitchers.empty:
        return pd.DataFrame()

    # ── Opposing team K rate merge ─────────────────────────────────
    if not team_k.empty and "team" in team_k.columns:
        team_k = team_k.copy()
        team_k["team_k_pct"] = team_k["team_k_pct"].apply(safe_float)

        opp_col = None
        for col in ["opposing_team", "opp_team"]:
            if col in pitchers.columns:
                opp_col = col
                break

        if opp_col:
            pitchers = pitchers.merge(
                team_k.rename(columns={"team": opp_col, "team_k_pct": "opp_team_k_pct"}),
                on=opp_col,
                how="left",
            )

    if "opp_team_k_pct" not in pitchers.columns:
        pitchers["opp_team_k_pct"] = LEAGUE_AVG_OPP_K_PCT

    # ── Park merge ─────────────────────────────────────────────────
    if not parks.empty:
        parks = parks.copy()
        parks.columns = [c.strip() for c in parks.columns]
        park_cols = [c for c in ["team", "park_hr_factor", "park_name"] if c in parks.columns]

        home_col = None
        for col in ["home_team", "pitching_team", "team"]:
            if col in pitchers.columns:
                home_col = col
                break

        if home_col and park_cols:
            pitchers = pitchers.merge(
                parks[park_cols].rename(columns={"team": home_col}),
                on=home_col,
                how="left",
            )

    # ── Defaults ───────────────────────────────────────────────────
    defaults = {
        "opp_team_k_pct":         LEAGUE_AVG_OPP_K_PCT,
        "opp_team_bb_pct":        8.5,
        "park_hr_factor":         100.0,
        "swstr_trend":            0.0,
        "velo_trend":             0.0,
        "k_per_start_21d":        0.0,
        "avg_ip_last_3":          0.0,
        "chase_rate":             LEAGUE_AVG_CHASE,
        "first_pitch_strike_pct": 62.0,
    }
    for col, val in defaults.items():
        if col not in pitchers.columns:
            pitchers[col] = val
        else:
            pitchers[col] = pitchers[col].apply(lambda x: safe_float(x, val))

    # ── Score ──────────────────────────────────────────────────────
    pitchers["ks_score"]         = pitchers.apply(compute_ks_score, axis=1)
    pitchers["projected_k_calc"] = pitchers.apply(calc_projected_k, axis=1)
    pitchers["confidence"]       = pitchers.apply(assign_confidence, axis=1)

    # ── Odds merge ─────────────────────────────────────────────────
    if not odds_df.empty and "pitcher_name_norm" in odds_df.columns:
        odds_slim = odds_df[["pitcher_name_norm", "k_line", "over_odds", "under_odds"]].copy()
        odds_slim = odds_slim.rename(columns={
            "over_odds":  "ks_over_odds",
            "under_odds": "ks_under_odds",
        })
        pitchers = pitchers.merge(odds_slim, on="pitcher_name_norm", how="left")

        matched = pitchers["pitcher_name_norm"].isin(odds_df["pitcher_name_norm"])
        print(f"Odds matched: {matched.sum()}/{len(pitchers)} pitchers")
        unmatched = pitchers[~matched]["pitcher_name"].tolist()
        if unmatched:
            print(f"Unmatched pitchers: {unmatched}")
    else:
        pitchers["k_line"]        = np.nan
        pitchers["ks_over_odds"]  = np.nan
        pitchers["ks_under_odds"] = np.nan

    # ── Prop signal and reason ─────────────────────────────────────
    pitchers["prop_signal"] = pitchers.apply(calc_prop_signal, axis=1)
    pitchers["reason"]      = pitchers.apply(build_reason, axis=1)

    return pitchers


def apply_diversity_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("ks_score", ascending=False).reset_index(drop=True)
    selected    = []
    team_counts = {}
    chalk_count = 0
    no_odds     = []

    team_col = "pitching_team" if "pitching_team" in df.columns else "pitcher_team"
    if team_col not in df.columns:
        team_col = "team"

    for _, row in df.iterrows():
        odds = row.get("ks_over_odds")
        try:
            odds_val = float(odds)
            has_odds = True
        except (TypeError, ValueError):
            has_odds = False
            no_odds.append(row)
            continue

        if len(selected) >= TOP_N:
            break

        team = str(row.get(team_col, "UNK"))
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue

        is_chalk = has_odds and odds_val <= -200
        if is_chalk and chalk_count >= MAX_CHALK_PICKS:
            continue

        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1
        if is_chalk:
            chalk_count += 1

    for row in no_odds:
        if len(selected) >= TOP_N:
            break
        team = str(row.get(team_col, "UNK"))
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
        ws = sh.worksheet("Top_KS_Picks")
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Top_KS_Picks", rows=100, cols=30)

    if "pitching_team" in picks.columns and "team" not in picks.columns:
        picks = picks.copy()
        picks["team"] = picks["pitching_team"]
    elif "pitcher_team" in picks.columns and "team" not in picks.columns:
        picks = picks.copy()
        picks["team"] = picks["pitcher_team"]

    output_cols = {
        "rank":                   "Rank",
        "pitcher_name":           "Pitcher",
        "team":                   "Team",
        "ip":                     "IP",
        "avg_ip_per_start":       "Avg IP/Start",
        "k_pct_season":           "K%",
        "swstr_pct":              "SwStr%",
        "chase_rate":             "Chase%",
        "k_per_9":                "K/9",
        "fastball_velo":          "FB Velo",
        "k_per_start_21d":        "K/Start (21d)",
        "swstr_trend":            "SwStr Trend",
        "velo_trend":             "Velo Trend",
        "opp_team_k_pct":         "Opp Team K%",
        "projected_k_calc":       "Proj K",
        "k_line":                 "K Line",
        "ks_over_odds":           "Over Odds",
        "ks_under_odds":          "Under Odds",
        "prop_signal":            "Signal",
        "ks_score":               "K Score",
        "confidence":             "Confidence",
        "reason":                 "Key Reasons",
        "park_name":              "Park",
        "park_hr_factor":         "Park Factor",
        "first_pitch_strike_pct": "F-Strike%",
        "bb_pct_season":          "BB%",
        "k_minus_bb":             "K-BB%",
    }

    available = {k: v for k, v in output_cols.items() if k in picks.columns}
    out_df    = picks[list(available.keys())].rename(columns=available)
    out_df    = out_df.copy().replace([np.inf, -np.inf], np.nan).fillna("")

    with_retry(lambda: ws.update([out_df.columns.tolist()] + out_df.astype(str).values.tolist()))

    ws_id = ws.id
    reqs  = []

    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": len(out_df) + 2, "startColumnIndex": 0, "endColumnIndex": len(out_df.columns)},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_BG, "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10}, "verticalAlignment": "MIDDLE", "wrapStrategy": "WRAP"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
    }})
    reqs.append({"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1, "startColumnIndex": 0, "endColumnIndex": len(out_df.columns)},
        "cell": {"userEnteredFormat": {"backgroundColor": COLOR_HEADER, "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True, "fontFamily": "Roboto", "fontSize": 11}, "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
    }})
    for i in range(len(out_df)):
        bg = COLOR_BG if i % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2, "startColumnIndex": 0, "endColumnIndex": len(out_df.columns)},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)",
        }})
    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1}, "tabColorStyle": {"rgbColor": COLOR_HEADER}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
    except APIError as e:
        print(f"KS formatting failed: {e}")

    print(f"Written {len(out_df)} K picks to Top_KS_Picks")


def log_picks(gc: gspread.Client, sheet_id: str, picks: pd.DataFrame) -> None:
    today_str = date.today().strftime("%Y-%m-%d")
    sh        = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws       = sh.worksheet("KS_Picks_Log")
        existing = pd.DataFrame(with_retry(lambda: ws.get_all_records()))
    except gspread.WorksheetNotFound:
        ws       = sh.add_worksheet(title="KS_Picks_Log", rows=5000, cols=20)
        existing = pd.DataFrame()

    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"] != today_str].copy()

    team_col = "pitching_team" if "pitching_temp" in picks.columns else "pitcher_team"
    if team_col not in picks.columns:
        team_col = "team"

    new_rows = []
    for _, row in picks.iterrows():
        new_rows.append({
            "date":         today_str,
            "rank":         str(row.get("rank", "")),
            "pitcher_name": str(row.get("pitcher_name", "")),
            "team":         str(row.get(team_col, "")),
            "ks_score":     str(row.get("ks_score", "")),
            "projected_k":  str(row.get("projected_k_calc", "")),
            "k_line":       str(row.get("k_line", "")),
            "over_odds":    str(row.get("ks_over_odds", "")),
            "under_odds":   str(row.get("ks_under_odds", "")),
            "prop_signal":  str(row.get("prop_signal", "")),
            "confidence":   str(row.get("confidence", "")),
            "actual_ks":    "",
            "hit":          "Pending",
            "bet_placed":   "",
            "result":       "",
        })

    if not new_rows:
        return

    new_df       = pd.DataFrame(new_rows)
    combined_log = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    combined_log = combined_log.fillna("")

    time.sleep(5)
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update([combined_log.columns.tolist()] + combined_log.astype(str).values.tolist()))
    print(f"Logged {len(new_rows)} K picks to KS_Picks_Log")


def update_scorecard(gc: gspread.Client, sheet_id: str) -> None:
    time.sleep(15)
    sh = with_retry(lambda: gc.open_by_key(sheet_id))

    try:
        ws_log = sh.worksheet("KS_Picks_Log")
        log    = pd.DataFrame(with_retry(lambda: ws_log.get_all_records()))
    except gspread.WorksheetNotFound:
        print("KS_Picks_Log not found — skipping scorecard")
        return

    if log.empty:
        return

    scored = log[log["hit"].isin(["Yes", "No"])].copy()
    if scored.empty:
        print("No scored K picks yet — skipping scorecard.")
        return

    scored["hit_bool"]  = scored["hit"] == "Yes"
    scored["score_num"] = pd.to_numeric(scored["ks_score"], errors="coerce")

    rows = []

    def add_row(label, sub):
        if sub.empty:
            return
        total = len(sub)
        hits  = int(sub["hit_bool"].sum())
        rows.append([label, total, hits, f"{hits / total * 100:.1f}%"])

    add_row("All Picks", scored)
    for tier, mask in [
        ("Score 16+",     scored["score_num"] >= 16),
        ("Score 13+",     scored["score_num"] >= 13),
        ("Score 10+",     scored["score_num"] >= 10),
        ("Score 7+",      scored["score_num"] >= 7),
        ("Score Under 7", scored["score_num"] <  7),
    ]:
        add_row(tier, scored[mask])

    for conf in ["High", "Medium", "Low"]:
        add_row(f"Confidence: {conf}", scored[scored["confidence"] == conf])

    time.sleep(5)
    try:
        sc_ws = sh.worksheet("KS_Scorecard")
        with_retry(lambda: sc_ws.clear())
    except gspread.WorksheetNotFound:
        sc_ws = sh.add_worksheet(title="KS_Scorecard", rows=50, cols=6)

    with_retry(lambda: sc_ws.update([["Category", "Picks", "Hits", "Hit Rate"]] + rows))
    print("KS_Scorecard updated")


def write_timestamp(gc: gspread.Client, sheet_id: str) -> None:
    et     = pytz.timezone("America/New_York")
    now_et = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")
    sh     = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Top_KS_Picks")
        with_retry(lambda: ws.insert_row([f"⏱  Last Run: {now_et}"], index=1))
    except Exception as e:
        print(f"KS timestamp failed: {e}")
    print(f"KS timestamp written: {now_et}")


def main() -> None:
    time.sleep(10)

    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading KS data from Google Sheets...")
    pitchers  = read_sheet(gc, sheet_id, "KS_Statcast")
    time.sleep(3)
    probables = read_sheet(gc, sheet_id, "Pitcher_Statcast_2026")
    time.sleep(3)
    team_k    = read_sheet(gc, sheet_id, "Team_K_Rates")
    time.sleep(3)
    parks     = read_sheet(gc, sheet_id, "Park_Factors")
    time.sleep(3)
    odds_df   = read_sheet(gc, sheet_id, "KS_Odds")

    print(f"KS_Statcast: {len(pitchers)} pitchers")
    print(f"Today's probables: {len(probables)} pitchers")
    print(f"Team K Rates: {len(team_k)} rows")
    print(f"Parks: {len(parks)} rows")
    print(f"KS Odds: {len(odds_df)} rows")

    # ── Normalize odds names ───────────────────────────────────────
    if not odds_df.empty and "pitcher_name" in odds_df.columns:
        odds_df["pitcher_name_norm"] = odds_df["pitcher_name"].apply(normalize_name)

    # ── Filter to today's probable starters only ───────────────────
    if not probables.empty and "pitcher_name" in probables.columns and not pitchers.empty:
        probable_norms = set(probables["pitcher_name"].apply(normalize_name).tolist())
        before         = len(pitchers)
        pitchers       = pitchers[
            pitchers["pitcher_name"].apply(normalize_name).isin(probable_norms)
        ].copy()
        print(f"Probable starter filter: {before - len(pitchers)} removed, {len(pitchers)} today's starters")

        if "opposing_team" in probables.columns and "opposing_team" not in pitchers.columns:
            opp_map = dict(zip(
                probables["pitcher_name"].apply(normalize_name),
                probables["opposing_team"],
            ))
            pitchers["opposing_team"] = pitchers["pitcher_name"].apply(
                lambda n: opp_map.get(normalize_name(n), "")
            )

        if "home_team" in probables.columns and "home_team" not in pitchers.columns:
            home_map = dict(zip(
                probables["pitcher_name"].apply(normalize_name),
                probables["home_team"],
            ))
            pitchers["home_team"] = pitchers["pitcher_name"].apply(
                lambda n: home_map.get(normalize_name(n), "")
            )

    # ── Game start times ───────────────────────────────────────────
    print("Fetching game start times...")
    game_times = get_todays_game_times()
    print(f"  Found {len(game_times)} game times")

    combined = prepare_combined(pitchers, team_k, parks, odds_df, game_times)

    if combined.empty:
        print("WARNING: No KS combined data.")
        return

    picks = apply_diversity_cap(combined)

    if picks.empty:
        print("WARNING: No KS picks generated.")
        return

    print(f"\nTop {len(picks)} Pitcher K Picks:")
    display_cols = [c for c in [
        "rank", "pitcher_name", "ks_score",
        "projected_k_calc", "k_line", "prop_signal", "confidence",
    ] if c in picks.columns]
    print(picks[display_cols].to_string(index=False))

    write_picks_to_sheet(gc, sheet_id, picks)
    time.sleep(5)
    log_picks(gc, sheet_id, picks)
    time.sleep(5)
    update_scorecard(gc, sheet_id)
    time.sleep(5)
    write_timestamp(gc, sheet_id)


if __name__ == "__main__":
    main()
