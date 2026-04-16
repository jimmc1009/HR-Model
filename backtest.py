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
    """
    Identify the probable starter per team on game_date by finding
    the pitcher with the most batters faced on that day per team.
    Returns {pitching_team: pitcher_id}.
    """
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

    # Deduplicate to one row per plate appearance per pitcher
    pa_dedupe = [c for c in ["game_pk", "at_bat_number", "pitcher"] if c in day_df.columns]
    pa_df = day_df[day_df["events"].notna()].drop_duplicates(subset=pa_dedupe)

    # Count BF per pitcher per team
    bf_counts = (
        pa_df.groupby(["pitching_team", "pitcher"])
        .size()
        .reset_index(name="bf")
    )

    # Take pitcher with most BF per team — that's the starter
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

    team_map = (
        df.sort_values("game_date")
        .drop_duplicates(subset=["batter"], keep="last")
        [["batter", "team"]]
    )
    combined = combined.merge(team_map, on="batter", how="left")

    combined = combined[
        (combined["season_bbe"] >= 10) &
        (combined["pa"] >= 10)
    ].copy()

    return combined


def build_pitcher_features(
    history_df: pd.DataFrame,
    starter_ids: Set[int],
    matchups: Dict[str, str],
    as_of_date: date,
    starters_by_team: Dict[str, int],
) -> pd.DataFrame:
    """Build pitcher features using only data BEFORE as_of_date."""
    df = history_df[history_df["game_date"] < pd.Timestamp(as_of_date)].copy()

    if df.empty or not starter_ids:
        return pd.DataFrame()

    if {"inning_topbot", "home_team", "away_team"}.issubset(df.columns):
        df["pitching_team"] = df.apply(
            lambda r: str(r["home_team"]).strip()
            if str(r["inning_topbot"]).lower().startswith("top")
            else str(r["away_team"]).strip(), axis=1,
        )
    else:
        df["pitching_team"] = ""

    batted_ball_events = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "sac_fly",
        "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    }

    bbe = df[
        df["pitcher"].isin(starter_ids) &
        df["events"].astype("string").str.lower().isin(batted_ball_events) &
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna() &
        df["launch_angle"].between(-90, 90)
    ].copy()

    dedupe = [c for c in ["game_pk", "at_bat_number", "pitcher"] if c in bbe.columns]
    if dedupe:
        bbe = bbe.drop_duplicates(subset=dedupe)

    if bbe.empty:
        return pd.DataFrame()

    bbe = add_flags(bbe)

    bf_df = df[
        df["pitcher"].isin(starter_ids) &
        df["events"].notna()
    ].drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "pitcher"] if c in df.columns]
    )
    bf_counts = bf_df.groupby("pitcher").size().reset_index(name="bf")

    ip_df = df[
        df["pitcher"].isin(starter_ids) &
        df["events"].astype("string").str.lower().isin(OUT_EVENTS)
    ]
    ip_counts = ip_df.groupby("pitcher").size().reset_index(name="outs")
    ip_counts["ip"] = (ip_counts["outs"] / 3).round(2)

    agg_dict = {
        "season_bbe_allowed":      ("launch_speed", "size"),
        "season_hr_allowed":       ("is_hr", "sum"),
        "season_fb_allowed":       ("is_fly_ball", "sum"),
        "season_barrel_allowed":   ("is_barrel", "sum"),
        "season_hard_hit_allowed": ("is_hard_hit", "sum"),
        "avg_ev_allowed":          ("launch_speed", "mean"),
    }
    if "p_throws" in bbe.columns:
        agg_dict["pitcher_hand"] = ("p_throws", "first")

    season = bbe.groupby("pitcher", dropna=False).agg(**agg_dict).reset_index()
    season = season.merge(bf_counts, on="pitcher", how="left")
    season = season.merge(ip_counts[["pitcher", "ip"]], on="pitcher", how="left")
    season["bf"] = season["bf"].fillna(0)
    season["ip"] = season["ip"].fillna(0)

    season["hr_per_fb_allowed"] = (
        season["season_hr_allowed"] / season["season_fb_allowed"].replace(0, np.nan) * 100
    ).round(2)
    season["season_barrel_pct_allowed"] = (
        season["season_barrel_allowed"] / season["season_bbe_allowed"].replace(0, np.nan) * 100
    ).round(2)
    season["hard_hit_pct_allowed"] = (
        season["season_hard_hit_allowed"] / season["season_bbe_allowed"].replace(0, np.nan) * 100
    ).round(2)
    season["avg_ev_allowed"] = season["avg_ev_allowed"].round(2)
    season["hr9"] = (
        season["season_hr_allowed"] / season["ip"].replace(0, np.nan) * 9
    ).round(2)

    if "pitcher_hand" not in season.columns:
        season["pitcher_hand"] = ""

    if "stand" in bbe.columns:
        for hand, label in [("L", "vs_lhh"), ("R", "vs_rhh")]:
            sub_bbe = bbe[bbe["stand"] == hand].copy()
            sub_full = df[
                df["pitcher"].isin(starter_ids) &
                (df["stand"] == hand) &
                df["events"].astype("string").str.lower().isin(OUT_EVENTS)
            ]
            ip_grp = sub_full.groupby("pitcher").size().reset_index(name="outs")
            ip_grp[f"{label}_ip"] = (ip_grp["outs"] / 3).round(2)

            if sub_bbe.empty:
                continue

            grp = (
                sub_bbe.groupby("pitcher", dropna=False)
                .agg(
                    **{
                        f"{label}_bbe":        ("launch_speed", "size"),
                        f"{label}_hr":         ("is_hr", "sum"),
                        f"{label}_barrel_pct": ("is_barrel", "mean"),
                    }
                )
                .reset_index()
            )
            grp = grp.merge(ip_grp[["pitcher", f"{label}_ip"]], on="pitcher", how="left")
            grp[f"{label}_ip"] = grp[f"{label}_ip"].fillna(0)
            grp[f"{label}_barrel_pct"] = (grp[f"{label}_barrel_pct"] * 100).round(2)
            grp[f"{label}_hr_rate"] = (
                grp[f"{label}_hr"] / grp[f"{label}_bbe"].replace(0, np.nan) * 100
            ).round(2)
            grp[f"{label}_hr9"] = (
                grp[f"{label}_hr"] / grp[f"{label}_ip"].replace(0, np.nan) * 9
            ).round(2)
            season = season.merge(grp, on="pitcher", how="left")

    # Map pitcher to their team using starters_by_team
    team_by_pitcher = {v: k for k, v in starters_by_team.items()}
    season["pitcher_team"] = season["pitcher"].map(
        lambda x: team_by_pitcher.get(int(x), "") if pd.notna(x) else ""
    )
    season["opposing_team"] = season["pitcher_team"].map(
        lambda t: matchups.get(str(t), "")
    )

    return season


def compute_platoon_score(row: pd.Series) -> Tuple[float, str, float]:
    batter_hand = str(row.get("batter_hand", "")).strip().upper()
    p_throws = str(row.get("pitcher_hand", "")).strip().upper()

    score = 0.0
    penalty = 0.0

    iso_vs_lhp = safe_float(row.get("vs_lhp_iso", 0))
    iso_vs_rhp = safe_float(row.get("vs_rhp_iso", 0))

    if p_throws == "L":
        iso_vs_this = iso_vs_lhp
        iso_vs_opp = iso_vs_rhp
        batter_barrel_vs = safe_float(row.get("vs_lhp_barrel_pct", 0))
        batter_hr_vs = safe_float(row.get("vs_lhp_hr_rate", 0))
    elif p_throws == "R":
        iso_vs_this = iso_vs_rhp
        iso_vs_opp = iso_vs_lhp
        batter_barrel_vs = safe_float(row.get("vs_rhp_barrel_pct", 0))
        batter_hr_vs = safe_float(row.get("vs_rhp_hr_rate", 0))
    else:
        return 0.0, "", 0.0

    if batter_hand == "R":
        pitcher_barrel_vs = safe_float(row.get("vs_rhh_barrel_pct", 0))
        pitcher_hr_vs = safe_float(row.get("vs_rhh_hr_rate", 0))
        pitcher_hr9_vs = safe_float(row.get("vs_rhh_hr9", 0))
    elif batter_hand == "L":
        pitcher_barrel_vs = safe_float(row.get("vs_lhh_barrel_pct", 0))
        pitcher_hr_vs = safe_float(row.get("vs_lhh_hr_rate", 0))
        pitcher_hr9_vs = safe_float(row.get("vs_lhh_hr9", 0))
    else:
        pitcher_barrel_vs = pitcher_hr_vs = pitcher_hr9_vs = 0.0

    iso_gap = iso_vs_opp - iso_vs_this
    has_iso_data = (iso_vs_this > 0 or iso_vs_opp > 0)

    if has_iso_data:
        if iso_gap >= ISO_GAP_SEVERE:
            penalty = 2.5
        elif iso_gap >= ISO_GAP_LARGE:
            penalty = 1.5
        elif iso_gap >= ISO_GAP_MEDIUM:
            penalty = 0.7
        elif iso_gap <= -ISO_GAP_MEDIUM:
            score += 1.0

    score += pitcher_barrel_vs * 0.06
    score += pitcher_hr_vs * 0.04
    score += pitcher_hr9_vs * 0.08
    score += batter_barrel_vs * 0.04
    score += batter_hr_vs * 0.03

    return round(score, 3), "", round(penalty, 3)


def compute_pitch_matchup_score(row: pd.Series) -> float:
    scores = []
    for rank in range(1, 4):
        pitch_type = str(row.get(f"top_pitch_{rank}", "")).strip().upper()
        pitch_pct = safe_float(row.get(f"top_pitch_{rank}_pct", 0))
        if not pitch_type or pitch_type in ("", "NAN", "NONE"):
            continue
        iso = safe_float(row.get(f"iso_vs_{pitch_type}", 0))
        hr_rate = safe_float(row.get(f"hr_rate_vs_{pitch_type}", 0))
        barrel = safe_float(row.get(f"barrel_pct_vs_{pitch_type}", 0))
        if iso > 0 or hr_rate > 0:
            scores.append((iso * 3 + hr_rate / 10 + barrel / 20) * (pitch_pct / 100))
    return sum(scores)


def score_matchups(combined: pd.DataFrame) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame()

    score_cols = [
        "barrel_pct_7d", "hr_per_pa", "hr_per_fb", "iso",
        "avg_ev_7d", "hard_hit_pct_7d",
        "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
        "park_hr_factor",
        "vs_lhp_barrel_pct", "vs_rhp_barrel_pct",
        "vs_lhp_hr_rate", "vs_rhp_hr_rate",
        "vs_lhp_iso", "vs_rhp_iso",
    ]
    for col in score_cols:
        if col in combined.columns:
            combined[col] = combined[col].apply(safe_float)
        else:
            combined[col] = 0.0

    combined["park_hr_factor_norm"] = combined["park_hr_factor"] - 100
    combined["weather_score"] = 0.0

    platoon_results = combined.apply(compute_platoon_score, axis=1)
    combined["platoon_score"] = platoon_results.apply(lambda x: x[0])
    combined["platoon_penalty"] = platoon_results.apply(lambda x: x[2])
    combined["pitch_matchup_score"] = combined.apply(compute_pitch_matchup_score, axis=1)

    combined["pitcher_quality_penalty"] = (
        normalize_inverted(combined["pitcher_barrel_pct"]) * 0.6 +
        normalize_inverted(combined["pitcher_hard_hit_pct"]) * 0.4
    )

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
        normalize(combined["platoon_score"])             * PLATOON_BONUS_WEIGHT +
        normalize(combined["pitch_matchup_score"])       * PITCH_MATCHUP_WEIGHT -
        combined["pitcher_quality_penalty"]              * WEIGHTS["pitcher_quality_penalty"] -
        combined["platoon_penalty"]
    )

    return combined


def get_actual_hrs(full_df: pd.DataFrame, game_date: date) -> Dict[int, bool]:
    day_df = full_df[
        (full_df["game_date"] == pd.Timestamp(game_date)) &
        (full_df["events"].astype("string").str.lower().eq("home_run"))
    ]
    return {int(pid): True for pid in day_df["batter"].dropna().unique()}


def assign_confidence(row: pd.Series) -> str:
    batter_bbe = safe_float(row.get("season_bbe", 0))
    batter_pa = safe_float(row.get("pa", 0))
    pitcher_bbe = safe_float(row.get("pitcher_bbe_allowed", 0))
    small_sample_park = str(row.get("small_sample", "False")).lower() == "true"

    points = 0

    if batter_bbe >= 60 and batter_pa >= 100:
        points += 2
    elif batter_bbe >= 30 and batter_pa >= 50:
        points += 1

    if pitcher_bbe >= 80:
        points += 2
    elif pitcher_bbe >= 40:
        points += 1

    if not small_sample_park:
        points += 1

    if points >= 4:
        return "High"
    elif points >= 2:
        return "Medium"
    else:
        return "Low"


def build_park_factors(df: pd.DataFrame) -> pd.DataFrame:
    pa_df = df[df["events"].notna()].copy()
    ab_dedupe = [c for c in ["game_pk", "at_bat_number", "home_team"] if c in pa_df.columns]
    pa_df = pa_df.drop_duplicates(subset=ab_dedupe)
    pa_df["is_hr"] = pa_df["events"].astype("string").str.lower().eq("home_run")

    park_stats = (
        pa_df.groupby("home_team", dropna=False)
        .agg(pa=("is_hr", "size"), hr=("is_hr", "sum"))
        .reset_index()
    )
    park_stats = park_stats[park_stats["home_team"].notna()].copy()
    park_stats["hr_rate"] = park_stats["hr"] / park_stats["pa"].replace(0, np.nan)

    league_hr_rate = park_stats["hr"].sum() / park_stats["pa"].sum()
    park_stats["park_hr_factor"] = (
        park_stats["hr_rate"] / league_hr_rate * 100
    ).round(1)
    park_stats["small_sample"] = park_stats["pa"] < 50
    park_stats = park_stats.rename(columns={"home_team": "team"})

    return park_stats[["team", "park_hr_factor", "small_sample"]]


def build_scorecard(picks_log: pd.DataFrame) -> pd.DataFrame:
    if picks_log.empty:
        return pd.DataFrame()

    picks_log = picks_log.copy()
    picks_log["hit_hr_bool"] = picks_log["hit_hr"] == "Yes"
    picks_log["rank"] = pd.to_numeric(picks_log["rank"], errors="coerce")
    picks_log["date"] = pd.to_datetime(picks_log["date"])

    rows = []

    def add_row(category, subcategory, sub_df):
        if sub_df.empty:
            return
        total = len(sub_df)
        hits = sub_df["hit_hr_bool"].sum()
        rows.append({
            "category":     category,
            "subcategory":  subcategory,
            "total_picks":  total,
            "hr_count":     int(hits),
            "hit_rate_pct": round(hits / total * 100, 1),
        })

    add_row("Overall", "All Picks", picks_log)

    for rank in range(1, 11):
        add_row("By Rank", f"Rank {rank}", picks_log[picks_log["rank"] == rank])

    for tier in ["High", "Medium", "Low"]:
        add_row("By Confidence", tier, picks_log[picks_log["confidence"] == tier])

    for section in picks_log["section"].dropna().unique():
        add_row("By Section", section, picks_log[picks_log["section"] == section])

    max_date = picks_log["date"].max()
    add_row("Rolling", "Last 7 Days",  picks_log[picks_log["date"] >= max_date - timedelta(days=7)])
    add_row("Rolling", "Last 30 Days", picks_log[picks_log["date"] >= max_date - timedelta(days=30)])

    return pd.DataFrame(rows)


# ── Main backtest loop ─────────────────────────────────────────────────────

def run_backtest(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    game_dates = get_game_dates()
    print(f"Back-testing {len(game_dates)} dates from {game_dates[0]} to {game_dates[-1]}")

    full_df = full_df.copy()
    full_df["game_date"] = pd.to_datetime(full_df["game_date"])

    all_picks = []
    park_factors = build_park_factors(full_df)

    for i, game_date in enumerate(game_dates):
        print(f"  [{i+1}/{len(game_dates)}] {game_date}...")

        today_teams = get_teams_for_date(game_date)
        if not today_teams:
            continue

        matchups = get_matchups_for_date(game_date)
        if not matchups:
            continue

        batters = build_batter_features(full_df, today_teams, game_date)
        if batters.empty:
            continue

        # Identify one starter per team using BF on game day
        starters_by_team = get_probable_starters(full_df, game_date)
        if not starters_by_team:
            continue

        starter_ids = set(starters_by_team.values())

        pitchers = build_pitcher_features(
            full_df, starter_ids, matchups, game_date, starters_by_team
        )
        if pitchers.empty:
            continue

        pitcher_rename = {
            "pitcher_hand":              "pitcher_hand",
            "season_barrel_pct_allowed": "pitcher_barrel_pct",
            "hr_per_fb_allowed":         "pitcher_hr_per_fb",
            "hard_hit_pct_allowed":      "pitcher_hard_hit_pct",
            "avg_ev_allowed":            "pitcher_avg_ev",
            "season_bbe_allowed":        "pitcher_bbe_allowed",
            "opposing_team":             "batter_team",
            "vs_lhh_barrel_pct":         "vs_lhh_barrel_pct",
            "vs_rhh_barrel_pct":         "vs_rhh_barrel_pct",
            "vs_lhh_hr_rate":            "vs_lhh_hr_rate",
            "vs_rhh_hr_rate":            "vs_rhh_hr_rate",
            "vs_lhh_hr9":                "vs_lhh_hr9",
            "vs_rhh_hr9":                "vs_rhh_hr9",
        }
        pitchers = pitchers.rename(
            columns={k: v for k, v in pitcher_rename.items() if k in pitchers.columns}
        )
        batters = batters.rename(columns={"team": "batter_team"})

        pitcher_cols = [c for c in [
            "batter_team", "pitcher", "pitcher_hand",
            "pitcher_barrel_pct", "pitcher_hr_per_fb", "pitcher_hard_hit_pct",
            "pitcher_avg_ev", "pitcher_bbe_allowed",
            "vs_lhh_barrel_pct", "vs_rhh_barrel_pct",
            "vs_lhh_hr_rate", "vs_rhh_hr_rate",
            "vs_lhh_hr9", "vs_rhh_hr9",
        ] if c in pitchers.columns]

        combined = batters.merge(
            pitchers[pitcher_cols],
            on="batter_team",
            how="inner",
        )

        if combined.empty:
            continue

        combined = combined.merge(
            park_factors[["team", "park_hr_factor", "small_sample"]],
            left_on="batter_team",
            right_on="team",
            how="left",
        )
        combined["park_hr_factor"] = combined["park_hr_factor"].fillna(100.0)
        combined["small_sample"] = combined["small_sample"].fillna(False)
        combined["hr_weather_boost"] = 0.0

        combined = score_matchups(combined)
        if combined.empty:
            continue

        combined["score"] = combined["score"].round(3)
        combined["confidence"] = combined.apply(assign_confidence, axis=1)

        top10 = combined.nlargest(10, "score").copy()
        top10["rank"] = range(1, len(top10) + 1)

        batter_ids = top10["batter"].dropna().astype(int).tolist()
        name_map = lookup_player_names(batter_ids)
        top10["player_name"] = top10["batter"].map(
            lambda x: name_map.get(int(x), str(x)) if pd.notna(x) else ""
        )

        actual_hrs = get_actual_hrs(full_df, game_date)

        for _, row in top10.iterrows():
            batter_id = int(row["batter"]) if pd.notna(row.get("batter")) else None
            hit_hr = actual_hrs.get(batter_id, False) if batter_id else False

            all_picks.append({
                "date":        game_date.strftime("%Y-%m-%d"),
                "rank":        int(row["rank"]),
                "player_name": row.get("player_name", ""),
                "player_id":   batter_id,
                "team":        str(row.get("batter_team", "")),
                "pitcher_id":  int(row["pitcher"]) if pd.notna(row.get("pitcher")) else None,
                "park_name":   str(row.get("park_name", "")),
                "hr_score":    round(float(row["score"]), 3),
                "confidence":  str(row.get("confidence", "")),
                "hit_hr":      "Yes" if hit_hr else "No",
                "section":     "Main",
            })

    if not all_picks:
        return pd.DataFrame(), pd.DataFrame()

    picks_log = pd.DataFrame(all_picks)
    scorecard = build_scorecard(picks_log)

    return picks_log, scorecard


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


def write_sheet(
    gc: gspread.Client,
    sheet_id: str,
    worksheet_name: str,
    df: pd.DataFrame,
    rows: int = 5000,
    cols: int = 20,
) -> None:
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet_name)
        ws.clear()
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=rows, cols=cols)

    df = clean_for_sheets(df)
    values = [df.columns.tolist()] + df.astype(str).values.tolist()
    ws.update(values)


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    print("Pulling full 2026 season Statcast data...")
    end_dt = date.today() - timedelta(days=1)
    season_start = date.fromisoformat(SEASON_START)

    try:
        full_df = statcast(
            start_dt=season_start.strftime("%Y-%m-%d"),
            end_dt=end_dt.strftime("%Y-%m-%d"),
        )
        if full_df is None or full_df.empty:
            print("Statcast returned empty — cannot backtest.")
            return
        print(f"Pulled {len(full_df):,} rows")
    except Exception as e:
        print(f"Statcast pull failed: {e}")
        return

    print("Running backtest...")
    picks_log, scorecard = run_backtest(full_df)

    if picks_log.empty:
        print("No picks generated — backtest produced no results.")
        return

    total = len(picks_log)
    hits = (picks_log["hit_hr"] == "Yes").sum()
    print(f"\nBacktest complete: {total} picks logged")
    print(f"Overall HR hit rate: {hits / total * 100:.1f}%")

    print("\nScorecard:")
    print(scorecard.to_string(index=False))

    write_sheet(gc, sheet_id, "Picks_Log", picks_log, rows=5000, cols=20)
    print("Written to Picks_Log")

    write_sheet(gc, sheet_id, "Scorecard", scorecard, rows=100, cols=10)
    print("Written to Scorecard")


if __name__ == "__main__":
    main()
