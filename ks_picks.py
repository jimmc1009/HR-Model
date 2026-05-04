"""
ks_picks.py
Pitcher strikeout prop model.
Reads from KS_Statcast, Pitcher_Statcast_2026, Team_K_Rates, Park_Factors, KS_Odds.
Outputs Top_KS_Picks + KS_Picks_Log + KS_Scorecard.
"""

import os
import json
import time
import unicodedata
from datetime import date, datetime
from typing import Optional

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

MIN_IP      = 10
MIN_GS      = 2
MAX_PER_TEAM = 1
TOP_N       = 10

COLOR_BG     = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_HEADER = {"red": 0.055, "green": 0.318, "blue": 0.580}
COLOR_WHITE  = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GREEN  = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_RED    = {"red": 0.910, "green": 0.259, "blue": 0.259}
COLOR_GOLD   = {"red": 1.000, "green": 0.843, "blue": 0.000}


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


def score_k_pct(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 32.0: return 3.0
    if v >= 28.0: return 2.5
    if v >= 25.0: return 2.0
    if v >= 22.0: return 1.5
    if v >= 19.0: return 1.0
    if v >= 16.0: return 0.5
    return 0.0


def score_swstr(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 16.0: return 2.5
    if v >= 14.0: return 2.0
    if v >= 12.0: return 1.5
    if v >= 10.0: return 1.0
    if v >= 8.0:  return 0.5
    return 0.0


def score_chase(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 38.0: return 1.5
    if v >= 34.0: return 1.0
    if v >= 30.0: return 0.5
    return 0.0


def score_k_per_9(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 12.0: return 1.5
    if v >= 10.0: return 1.0
    if v >= 8.5:  return 0.5
    if v >= 7.0:  return 0.2
    return 0.0


def score_velo(v: float, ip: float) -> float:
    if ip < MIN_IP: return 0.0
    if v >= 97.0: return 1.0
    if v >= 95.0: return 0.6
    if v >= 93.0: return 0.3
    return 0.0


def score_k_per_start_21d(v: float) -> float:
    if v >= 9.0: return 1.5
    if v >= 7.0: return 1.0
    if v >= 5.5: return 0.5
    if v <= 3.0: return -1.0
    if v <= 4.0: return -0.5
    return 0.0


def score_avg_ip(v: float) -> float:
    if v >= 6.5: return 1.0
    if v >= 6.0: return 0.7
    if v >= 5.5: return 0.4
    if v >= 5.0: return 0.1
    if v <= 4.0: return -1.0
    if v <= 4.5: return -0.5
    return 0.0


def score_opp_k_pct(v: float) -> float:
    if v >= 28.0: return  1.0
    if v >= 24.0: return  0.6
    if v >= 20.0: return  0.2
    if v <= 16.0: return -0.6
    if v <= 18.0: return -0.3
    return 0.0


def score_park(v: float) -> float:
    if v >= 115: return -0.3
    if v <= 80:  return  0.2
    return 0.0


def score_trends(swstr_trend: str, velo_trend: str) -> float:
    s = 0.0
    if "up" in str(swstr_trend).lower():   s += 0.5
    if "down" in str(swstr_trend).lower(): s -= 0.5
    if "up" in str(velo_trend).lower():    s += 0.3
    if "down" in str(velo_trend).lower():  s -= 0.3
    return s


def score_opener_risk(v: float) -> float:
    if v >= 0.5: return -2.0
    return 0.0


def compute_ks_score(row: pd.Series) -> float:
    ip = safe_float(row.get("ip", 0))
    gs = safe_float(row.get("games_started", 0))
    if ip < MIN_IP or gs < MIN_GS:
        return 0.0

    s = (
        score_k_pct(safe_float(row.get("k_pct_season")), ip) +
        score_swstr(safe_float(row.get("swstr_pct")), ip) +
        score_chase(safe_float(row.get("chase_rate")), ip) +
        score_k_per_9(safe_float(row.get("k_per_9")), ip) +
        score_velo(safe_float(row.get("fastball_velo")), ip) +
        score_k_per_start_21d(safe_float(row.get("k_per_start_21d"))) +
        score_avg_ip(safe_float(row.get("avg_ip_per_start"))) +
        score_opp_k_pct(safe_float(row.get("opp_team_k_pct", 22.0), 22.0)) +
        score_park(safe_float(row.get("park_hr_factor", 100), 100)) +
        score_trends(row.get("swstr_trend", ""), row.get("velo_trend", "")) +
        score_opener_risk(safe_float(row.get("opener_risk", 0)))
    )
    return round(s, 2)


def project_ks(row: pd.Series) -> float:
    ip      = safe_float(row.get("avg_ip_per_start", 5.5), 5.5)
    k_per_9 = safe_float(row.get("k_per_9", 7.0), 7.0)
    k_pct   = safe_float(row.get("k_pct_season", 22.0), 22.0)

    method1 = (k_per_9 / 9.0) * ip
    method2 = (k_pct / 100.0) * ip * 3.0
    return round((method1 + method2) / 2, 1)


def calc_prop_signal(row: pd.Series) -> str:
    line  = safe_float(row.get("k_line", 0))
    score = safe_float(row.get("ks_score", 0))
    proj  = safe_float(row.get("projected_k_calc", 0))

    if line <= 0:
        return "—"

    edge = proj - line

    if score >= 10 and edge >= 1.0:
        return f"OVER {line}"
    if score >= 7 and edge >= 0.5:
        return f"LEAN OVER {line}"
    if score <= 3 and edge <= -1.0:
        return f"LEAN UNDER {line}"
    return "—"


def assign_confidence(row: pd.Series) -> str:
    ip = safe_float(row.get("ip", 0))
    gs = safe_float(row.get("games_started", 0))
    if ip >= 40 and gs >= 7: return "High"
    if ip >= 20 and gs >= 4: return "Medium"
    return "Low"


def build_reason(row: pd.Series) -> str:
    reasons = []

    k_pct = safe_float(row.get("k_pct_season"))
    if k_pct >= 28:
        reasons.append(f"🔥 {k_pct:.1f}% K rate — elite strikeout pitcher")
    elif k_pct >= 24:
        reasons.append(f"⚡ {k_pct:.1f}% K rate — above average")

    swstr = safe_float(row.get("swstr_pct"))
    if swstr >= 14:
        reasons.append(f"💫 SwStr% {swstr:.1f}% — elite swing and miss")
    elif swstr >= 12:
        reasons.append(f"🎯 SwStr% {swstr:.1f}% — above average swing and miss")

    chase = safe_float(row.get("chase_rate"))
    if chase >= 34:
        reasons.append(f"🏃 Chase rate {chase:.1f}% — batters chasing off zone")

    velo = safe_float(row.get("fastball_velo"))
    if velo >= 96:
        reasons.append(f"🔥 {velo:.1f} mph fastball")
    elif velo >= 94:
        reasons.append(f"💨 {velo:.1f} mph fastball")

    k21 = safe_float(row.get("k_per_start_21d"))
    if k21 >= 7:
        reasons.append(f"📈 {k21:.1f} K/start last 21 days — hot streak")
    elif k21 <= 4 and k21 > 0:
        reasons.append(f"📉 {k21:.1f} K/start last 21 days — cold stretch")

    avg_ip = safe_float(row.get("avg_ip_per_start"))
    if avg_ip >= 6.0:
        reasons.append(f"⏱️ {avg_ip:.1f} avg IP/start — goes deep")

    opp_k = safe_float(row.get("opp_team_k_pct", 22.0), 22.0)
    if opp_k >= 26:
        reasons.append(f"✅ Opp team K% {opp_k:.1f}% — high strikeout lineup")
    elif opp_k <= 17:
        reasons.append(f"⚠️ Opp team K% {opp_k:.1f}% — contact-heavy lineup")

    swstr_trend = str(row.get("swstr_trend", ""))
    if "up" in swstr_trend.lower():
        reasons.append("📈 SwStr% trending up")
    velo_trend = str(row.get("velo_trend", ""))
    if "up" in velo_trend.lower():
        reasons.append("💪 Velo trending up")
    elif "down" in velo_trend.lower():
        reasons.append("📉 Velo trending down")

    if not reasons:
        reasons.append("Solid across multiple K factors")

    return " | ".join(reasons)


def prepare_picks(
    ks_df: pd.DataFrame,
    pitchers_df: pd.DataFrame,
    team_k_rates: pd.DataFrame,
    parks_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    game_times: dict,
) -> pd.DataFrame:

    if ks_df.empty:
        print("No KS Statcast data.")
        return pd.DataFrame()

    df = ks_df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Always overwrite ip with ks_ip
    if "ks_ip" in df.columns:
        df["ip"] = df["ks_ip"].apply(safe_float)

    # Filter to today's probable starters only
    if not pitchers_df.empty:
        pitchers_df = pitchers_df.copy()
        pitchers_df.columns = [c.strip() for c in pitchers_df.columns]

        if "pitcher_id" in pitchers_df.columns and "pitcher_id" in df.columns:
            try:
                df["pitcher_id"]           = df["pitcher_id"].astype("int64")
                pitchers_df["pitcher_id"]  = pitchers_df["pitcher_id"].astype("int64")
            except Exception:
                pass
            today_ids = set(pitchers_df["pitcher_id"].dropna().unique())
            before    = len(df)
            df        = df[df["pitcher_id"].isin(today_ids)].copy()
            print(f"Probable starter filter: {before - len(df)} removed, {len(df)} today's starters")

    if df.empty:
        return pd.DataFrame()

    # Game time filter
    if game_times and "home_team" in df.columns:
        import pytz as _pytz
        now_utc = datetime.now(_pytz.utc)
        before  = len(df)

        def game_started(team):
            gt = game_times.get(str(team).upper())
            return gt is not None and now_utc >= gt

        df = df[~df["home_team"].apply(game_started)].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"Starter filter: {removed} removed, {len(df)} remaining")

    if df.empty:
        return pd.DataFrame()

    # Merge opponent team K rates
    if not team_k_rates.empty and "opposing_team" in df.columns:
        team_k_rates = team_k_rates.copy()
        team_k_rates.columns = [c.strip() for c in team_k_rates.columns]
        if "team" in team_k_rates.columns and "k_pct" in team_k_rates.columns:
            df = df.merge(
                team_k_rates[["team", "k_pct"]].rename(columns={"k_pct": "opp_team_k_pct", "team": "opposing_team"}),
                on="opposing_team", how="left"
            )

    if not parks_df.empty and "home_team" in df.columns:
        parks_df = parks_df.copy()
        parks_df.columns = [c.strip() for c in parks_df.columns]
        park_cols = [c for c in ["team", "park_hr_factor", "park_name"] if c in parks_df.columns]
        if park_cols:
            df = df.merge(
                parks_df[park_cols],
                left_on="home_team", right_on="team", how="left"
            )

    defaults = {
        "opp_team_k_pct": 22.0,
        "park_hr_factor": 100.0,
        "swstr_trend": "",
        "velo_trend": "",
        "opener_risk": 0.0,
        "k_per_start_21d": 0.0,
        "avg_ip_per_start": 5.5,
        "fastball_velo": 93.0,
        "chase_rate": 30.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            if isinstance(val, float):
                df[col] = df[col].apply(lambda x: safe_float(x, val))

    df["ks_score"]          = df.apply(compute_ks_score, axis=1)
    df["projected_k_calc"]  = df.apply(project_ks, axis=1)
    df["confidence"]        = df.apply(assign_confidence, axis=1)
    df["reason"]            = df.apply(build_reason, axis=1)

    # Merge odds
    if not odds_df.empty:
        odds_df = odds_df.copy()
        odds_df.columns = [c.strip() for c in odds_df.columns]

        if "pitcher_name" in odds_df.columns:
            odds_df["pitcher_name_norm"] = odds_df["pitcher_name"].apply(normalize_name)

        if "pitcher_name" in df.columns:
            df["pitcher_name_norm"] = df["pitcher_name"].apply(normalize_name)

        odds_cols = [c for c in ["pitcher_name_norm", "k_line", "over_odds", "under_odds"] if c in odds_df.columns]
        if "pitcher_name_norm" in odds_cols:
            odds_slim = odds_df[odds_cols].rename(columns={
                "over_odds":  "ks_over_odds",
                "under_odds": "ks_under_odds",
            })
            df = df.merge(odds_slim, on="pitcher_name_norm", how="left")
    else:
        df["k_line"]       = np.nan
        df["ks_over_odds"] = np.nan
        df["ks_under_odds"] = np.nan

    df["prop_signal"] = df.apply(calc_prop_signal, axis=1)

    return df


def apply_diversity_cap(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["ks_score"] > 0].copy()
    df = df.sort_values("ks_score", ascending=False).reset_index(drop=True)

    selected    = []
    team_counts = {}

    for _, row in df.iterrows():
        if len(selected) >= TOP_N:
            break
        team = str(row.get("pitching_team", row.get("team", "UNK")))
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
        ws = sh.add_worksheet(title="Top_KS_Picks", rows=100, cols=15)

    if "pitching_team" in picks.columns and "team" not in picks.columns:
        picks = picks.copy()
        picks["team"] = picks["pitching_team"]
    elif "pitcher_team" in picks.columns and "team" not in picks.columns:
        picks = picks.copy()
        picks["team"] = picks["pitcher_team"]

    output_cols = {
        "rank":             "Rank",
        "pitcher_name":     "Pitcher",
        "team":             "Team",
        "k_pct_season":     "K%",
        "swstr_pct":        "SwStr%",
        "chase_rate":       "Chase%",
        "avg_ip_per_start": "Avg IP/Start",
        "k_line":           "K Line",
        "ks_over_odds":     "Over Odds",
        "projected_k_calc": "Proj K",
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
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": len(out_df) + 2,
                  "startColumnIndex": 0, "endColumnIndex": n_cols},
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
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                  "startColumnIndex": 0, "endColumnIndex": n_cols},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_HEADER,
            "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True,
                           "fontFamily": "Roboto", "fontSize": 11},
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
            "range": {"sheetId": ws_id, "startRowIndex": i + 1, "endRowIndex": i + 2,
                      "startColumnIndex": 0, "endColumnIndex": n_cols},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)",
        }})

    # Column widths: Rank, Pitcher, Team, K%, SwStr%, Chase%, Avg IP/Start, K Line, Over Odds, Proj K, Signal, Key Reasons, Confidence
    col_widths = [45, 160, 55, 55, 65, 65, 90, 65, 90, 60, 150, 300, 90]
    for i, w in enumerate(col_widths[:n_cols]):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS",
                      "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    # Row height — compact
    for i in range(len(out_df) + 1):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS",
                      "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": 32},
            "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1},
                       "tabColorStyle": {"rgbColor": COLOR_HEADER}},
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
        ws       = sh.add_worksheet(title="KS_Picks_Log", rows=5000, cols=15)
        existing = pd.DataFrame()

    if not existing.empty and "date" in existing.columns:
        existing = existing[existing["date"] != today_str].copy()

    new_rows = []
    for _, row in picks.iterrows():
        new_rows.append({
            "date":         today_str,
            "rank":         str(row.get("rank", "")),
            "pitcher_name": str(row.get("pitcher_name", "")),
            "team":         str(row.get("team", row.get("pitching_team", ""))),
            "ks_score":     str(row.get("ks_score", "")),
            "projected_k":  str(row.get("projected_k_calc", "")),
            "k_line":       str(row.get("k_line", "")),
            "over_odds":    str(row.get("ks_over_odds", "")),
            "prop_signal":  str(row.get("prop_signal", "")),
            "confidence":   str(row.get("confidence", "")),
            "hit":          "Pending",
            "actual_ks":    "",
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
        ("Score 10+", scored["score_num"] >= 10),
        ("Score 7+",  scored["score_num"] >= 7),
        ("Score 5+",  scored["score_num"] >= 5),
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
        first_row = ws.row_values(1)
        if first_row and "Last Run" in str(first_row[0]):
            ws.delete_rows(1)
        with_retry(lambda: ws.insert_row([f"⏱  Last Run: {now_et}"], index=1))
    except Exception as e:
        print(f"KS timestamp failed: {e}")
    print(f"KS timestamp written: {now_et}")


def main() -> None:
    time.sleep(10)

    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading KS data from Google Sheets...")
    ks_df        = read_sheet(gc, sheet_id, "KS_Statcast")
    time.sleep(2)
    pitchers_df  = read_sheet(gc, sheet_id, "Pitcher_Statcast_2026")
    time.sleep(2)
    team_k_rates = read_sheet(gc, sheet_id, "Team_K_Rates")
    time.sleep(2)
    parks_df     = read_sheet(gc, sheet_id, "Park_Factors")
    time.sleep(2)
    odds_df      = read_sheet(gc, sheet_id, "KS_Odds")

    print(f"KS_Statcast: {len(ks_df)} pitchers")
    print(f"Today's probables: {len(pitchers_df)} pitchers")
    print(f"Team K Rates: {len(team_k_rates)} rows")
    print(f"Parks: {len(parks_df)} rows")
    print(f"KS Odds: {len(odds_df)} rows")

    print("Fetching game start times...")
    game_times = get_todays_game_times()
    print(f"  Found {len(game_times)} game times")

    picks = prepare_picks(ks_df, pitchers_df, team_k_rates, parks_df, odds_df, game_times)

    if picks.empty:
        print("WARNING: No KS picks generated.")
        return

    picks = apply_diversity_cap(picks)

    if picks.empty:
        print("WARNING: No KS picks after diversity cap.")
        return

    print(f"\nTop {len(picks)} Pitcher K Picks:")
    print(picks[["rank", "pitcher_name", "ks_score", "projected_k_calc", "k_line", "prop_signal", "confidence"]].to_string(index=False))

    write_picks_to_sheet(gc, sheet_id, picks)
    time.sleep(5)
    log_picks(gc, sheet_id, picks)
    time.sleep(5)
    update_scorecard(gc, sheet_id)
    time.sleep(5)
    write_timestamp(gc, sheet_id)


if __name__ == "__main__":
    main()
