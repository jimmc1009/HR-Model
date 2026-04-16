import os
import json
from datetime import date, timedelta
from typing import Dict, List, Set, Tuple

import pandas as pd
import numpy as np
import gspread
import requests
from google.oauth2.service_account import Credentials
from pybaseball import statcast

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

SEASON_START = "2026-03-26"

FASTBALLS = {"FF", "SI", "FC", "FA"}
BREAKING = {"SL", "CU", "KC", "CS", "SV", "ST"}
OFFSPEED = {"CH", "FS", "FO", "SC"}
KNUCKLEBALL = {"KN"}

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

PITCH_GROUP_MAP = {
    **{p: "fastball" for p in FASTBALLS},
    **{p: "breaking" for p in BREAKING},
    **{p: "offspeed" for p in OFFSPEED},
    **{p: "knuckleball" for p in KNUCKLEBALL},
}

AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "field_out", "grounded_into_double_play", "double_play",
    "triple_play", "field_error", "fielders_choice",
    "fielders_choice_out", "force_out", "strikeout",
    "strikeout_double_play", "other_out",
}

OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "double_play",
    "triple_play", "fielders_choice_out", "force_out",
    "sac_fly", "sac_fly_double_play", "sac_bunt",
    "sac_bunt_double_play", "other_out", "strikeout",
    "strikeout_double_play",
}

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

ISO_GAP_SMALL   = 0.030
ISO_GAP_MEDIUM  = 0.060
ISO_GAP_LARGE   = 0.100
ISO_GAP_SEVERE  = 0.150


# ── Helpers ────────────────────────────────────────────────────────────────

def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


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


def lookup_player_names(player_ids: List[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    clean_ids = sorted({int(pid) for pid in player_ids if pd.notna(pid)})
    if not clean_ids:
        return out
    for i in range(0, len(clean_ids), 50):
        chunk = clean_ids[i:i + 50]
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pid = person.get("id")
                name = person.get("fullName", "")
                if pid and name:
                    out[int(pid)] = name
        except Exception:
            pass
    return out


def get_game_dates() -> List[date]:
    """Return all dates from season start through 2 days ago."""
    end_dt = date.today() - timedelta(days=2)
    season_start = date.fromisoformat(SEASON_START)
    dates = []
    cur = season_start
    while cur <= end_dt:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def get_teams_for_date(game_date: date) -> Set[str]:
    date_str = game_date.strftime("%Y-%m-%d")
    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        teams: Set[str] = set()
        for d in data.get("dates", []):
            for g in d.get("games", []):
                for side in ("away", "home"):
                    abbr = g.get("teams", {}).get(side, {}).get("team", {}).get("abbreviation", "")
                    if abbr:
                        teams.add(str(abbr).strip())
        if teams:
            return teams
    except Exception:
        pass

    try:
        espn_str = game_date.strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={espn_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        teams = set()
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                for competitor in comp.get("competitors", []):
                    abbr = competitor.get("team", {}).get("abbreviation", "")
                    if abbr:
                        mlb_abbr = ESPN_TO_MLB.get(str(abbr).strip(), str(abbr).strip())
                        teams.add(mlb_abbr)
        return teams
    except Exception:
        pass

    return set()


def get_matchups_for_date(game_date: date) -> Dict[str, str]:
    date_str = game_date.strftime("%Y-%m-%d")
    matchups: Dict[str, str] = {}

    try:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for d in data.get("dates", []):
            for g in d.get("games", []):
                away = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                home = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                if away and home:
                    matchups[away.strip()] = home.strip()
                    matchups[home.strip()] = away.strip()
        if matchups:
            return matchups
    except Exception:
        pass

    try:
        espn_str = game_date.strftime("%Y%m%d")
        url = f"https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={espn_str}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for event in data.get("events", []):
            for comp in event.get("competitions", []):
                competitors = comp.get("competitors", [])
                if len(competitors) == 2:
                    abbr0 = ESPN_TO_MLB.get(
                        competitors[0].get("team", {}).get("abbreviation", ""),
                        competitors[0].get("team", {}).get("abbreviation", "")
                    )
                    abbr1 = ESPN_TO_MLB.get(
                        competitors[1].get("team", {}).get("abbreviation", ""),
                        competitors[1].get("team", {}).get("abbreviation", "")
                    )
                    if abbr0 and abbr1:
                        matchups[abbr0] = abbr1
                        matchups[abbr1] = abbr0
        return matchups
    except Exception:
        pass

    return {}


def get_probable_starters(
    full_df: pd.DataFrame,
    game_date: date,
) -> Dict[str, int]:
    day_df = full_df[full_df["game_date"] == pd.Timestamp(game_date)].copy()

    if day_df.empty:
        return {}

    if {"inning_topbot", "home_team", "away_team"}.issubset(day_df.columns):
        day_df["pitching_team"] = day_df.apply(
            lambda r: str(r["home_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["away_team"]).strip(), axis=1,
        )
    else:
        return {}

    pa_dedupe = [c for c in ["game_pk", "at_bat_number", "pitcher"] if c in day_df.columns]
    pa_df = day_df[day_df["events"].notna()].drop_duplicates(subset=pa_dedupe)

    bf_counts = (
        pa_df.groupby(["pitching_team", "pitcher"])
        .size()
        .reset_index(name="bf")
    )

    starters = (
        bf_counts.sort_values("bf", ascending=False)
        .drop_duplicates(subset=["pitching_team"], keep="first")
        .set_index("pitching_team")["pitcher"]
        .to_dict()
    )

    return {team: int(pid) for team, pid in starters.items()}


# ── Feature builders ───────────────────────────────────────────────────────

def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def is_barrel(row):
        ev, la = row["launch_speed"], row["launch_angle"]
        if pd.isna(ev) or pd.isna(la) or ev < 98:
            return False
        return max(26 - (ev - 98), 8) <= la <= min(30 + (ev - 98), 50)

    df["is_barrel"]   = df.apply(is_barrel, axis=1)
    df["is_hard_hit"] = df["launch_speed"] >= 95
    df["is_hr"]       = df["events"].astype("string").str.lower().eq("home_run")
    df["is_fly_ball"] = df["launch_angle"].between(25, 50, inclusive="both")

    if "stand" in df.columns and "hc_x" in df.columns:
        df["is_pull"] = df.apply(
            lambda r: (r["hc_x"] < 125) if str(r.get("stand")) == "R"
            else (r["hc_x"] > 125) if str(r.get("stand")) == "L"
            else False, axis=1,
        )
    else:
        df["is_pull"] = False

    if "pitch_type" in df.columns:
        df["pitch_group"] = df["pitch_type"].apply(
            lambda p: PITCH_GROUP_MAP.get(str(p), "other")
        )
    else:
        df["pitch_group"] = "other"

    return df


def filter_bbe(df: pd.DataFrame) -> pd.DataFrame:
    batted_ball_events = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "sac_fly",
        "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    }
    bbe = df[
        df["events"].astype("string").str.lower().isin(batted_ball_events) &
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna() &
        df["launch_angle"].between(-90, 90)
    ].copy()
    dedupe = [c for c in ["game_pk", "at_bat_number", "batter"] if c in bbe.columns]
    if dedupe:
        bbe = bbe.drop_duplicates(subset=dedupe)
    return bbe


def build_batter_features(
    history_df: pd.DataFrame,
    today_teams: Set[str],
    as_of_date: date,
) -> pd.DataFrame:
    """Build batter features using only data BEFORE as_of_date."""
    df = history_df[history_df["game_date"] < pd.Timestamp(as_of_date)].copy()

    if df.empty:
        return pd.DataFrame()

    if "batting_team" in df.columns:
        df["team"] = df["batting_team"].fillna("").astype(str)
    elif {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
        df["team"] = df.apply(
            lambda r: str(r["away_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["home_team"]).strip(), axis=1,
        )
    else:
        df["team"] = ""

    if today_teams:
        df = df[df["team"].isin(today_teams)].copy()

    if df.empty:
        return pd.DataFrame()

    df = add_flags(df)
    bbe = filter_bbe(df)

    if bbe.empty:
        return pd.DataFrame()

    tb_map = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    bbe = bbe.copy()
    bbe["total_bases"] = bbe["events"].astype("string").str.lower().map(tb_map).fillna(0)
    bbe["is_hit"] = bbe["events"].astype("string").str.lower().isin(
        {"single", "double", "triple", "home_run"}
    )

    pa_df = df[df["events"].notna()].copy()
    pa_df = pa_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_df.columns]
    )
    pa_counts = pa_df.groupby("batter").size().reset_index(name="pa")

    ab_df = df[df["events"].astype("string").str.lower().isin(AB_EVENTS)].copy()
    ab_df = ab_df.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in ab_df.columns]
    )
    ab_counts = ab_df.groupby("batter").size().reset_index(name="ab")

    season = (
        bbe.groupby("batter", dropna=False)
        .agg(
            season_bbe=("launch_speed", "size"),
            season_hr=("is_hr", "sum"),
            season_fb=("is_fly_ball", "sum"),
            season_barrel=("is_barrel", "sum"),
            season_hard_hit=("is_hard_hit", "sum"),
            total_bases=("total_bases", "sum"),
            hits=("is_hit", "sum"),
            avg_launch_angle=("launch_angle", "mean"),
        )
        .reset_index()
    )
    season = season.merge(pa_counts, on="batter", how="left")
    season = season.merge(ab_counts, on="batter", how="left")
    season["ab"] = season["ab"].fillna(0)
    season["pa"] = season["pa"].fillna(0)

    season["hr_per_pa"] = (
        season["season_hr"] / season["pa"].replace(0, np.nan) * 100
    ).round(2)
    season["hr_per_fb"] = (
        season["season_hr"] / season["season_fb"].replace(0, np.nan) * 100
    ).round(2)
    season["season_barrel_pct"] = (
        season["season_barrel"] / season["season_bbe"].replace(0, np.nan) * 100
    ).round(2)
    season["iso"] = (
        (season["total_bases"] - season["hits"]) / season["ab"].replace(0, np.nan)
    ).round(3)
    season["batting_avg"] = (
        season["hits"] / season["ab"].replace(0, np.nan)
    ).round(3)

    cutoff_7d = pd.Timestamp(as_of_date) - timedelta(days=7)
    w7 = bbe[bbe["game_date"] >= cutoff_7d].copy()
    r7 = (
        w7.groupby("batter", dropna=False)
        .agg(
            bbe_7d=("launch_speed", "size"),
            avg_ev_7d=("launch_speed", "mean"),
            avg_la_7d=("launch_angle", "mean"),
            barrel_pct_7d=("is_barrel", "mean"),
            hard_hit_pct_7d=("is_hard_hit", "mean"),
            hr_7d=("is_hr", "sum"),
        )
        .reset_index()
    )
    r7["barrel_pct_7d"] = (r7["barrel_pct_7d"] * 100).round(2)
    r7["hard_hit_pct_7d"] = (r7["hard_hit_pct_7d"] * 100).round(2)
    r7["avg_ev_7d"] = r7["avg_ev_7d"].round(2)
    r7["avg_la_7d"] = r7["avg_la_7d"].round(2)

    combined = season.merge(r7, on="batter", how="left")

    if "p_throws" in bbe.columns and "stand" in bbe.columns:
        stand_map = (
            bbe[bbe["stand"].notna()]
            .groupby("batter")["stand"].first()
            .reset_index()
            .rename(columns={"stand": "batter_hand"})
        )

        for hand, label in [("L", "vs_lhp"), ("R", "vs_rhp")]:
            sub = bbe[bbe["p_throws"] == hand].copy()
            if sub.empty:
                combined[f"{label}_iso"] = np.nan
                combined[f"{label}_barrel_pct"] = np.nan
                combined[f"{label}_hr_rate"] = np.nan
                continue

            if "p_throws" in ab_df.columns:
                ab_hand = (
                    ab_df[ab_df["p_throws"] == hand]
                    .groupby("batter")
                    .size()
                    .reset_index(name=f"ab_{label}")
                )
            else:
                ab_hand = pd.DataFrame(columns=["batter", f"ab_{label}"])

            grp = (
                sub.groupby("batter", dropna=False)
                .agg(
                    **{
                        f"{label}_bbe":        ("launch_speed", "size"),
                        f"{label}_hr":         ("is_hr", "sum"),
                        f"{label}_barrel_pct": ("is_barrel", "mean"),
                        f"{label}_tb":         ("total_bases", "sum"),
                        f"{label}_hits":       ("is_hit", "sum"),
                    }
                )
                .reset_index()
            )
            grp = grp.merge(ab_hand, on="batter", how="left")
            grp[f"ab_{label}"] = grp[f"ab_{label}"].fillna(0)
            grp[f"{label}_barrel_pct"] = (grp[f"{label}_barrel_pct"] * 100).round(2)
            grp[f"{label}_hr_rate"] = (
                grp[f"{label}_hr"] / grp[f"{label}_bbe"].replace(0, np.nan) * 100
            ).round(2)
            grp[f"{label}_iso"] = (
                (grp[f"{label}_tb"] - grp[f"{label}_hits"]) /
                grp[f"ab_{label}"].replace(0, np.nan)
            ).round(3)
            grp = grp.drop(columns=[f"{label}_tb", f"{label}_hits", f"ab_{label}"])
            combined = combined.merge(grp, on="batter", how="left")

        combined = combined.merge(stand_map, on="batter", how="left")
    else:
        combined["batter_hand"] = ""

    team
