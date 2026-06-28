"""
dashboard.py
Builds the "Today's Top Picks" unified dashboard sheet.
Includes KS value finder — compares today's picks against resolved hit rates
to surface value bets where hit rate exceeds breakeven at given odds.

Updated: HR value plays now show score 12+ at ANY odds up to +699
(removed +301 floor — 12-13 | ≤+300 hits 37.5%, 13+ | ≤+300 hits 25.9%)
"""

import os
import json
import time
from datetime import datetime

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials
import pytz

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DASHBOARD_SHEET = "Today's Top Picks"

COLOR_BG        = {"red": 0.086, "green": 0.086, "blue": 0.086}
COLOR_BG_ALT    = {"red": 0.118, "green": 0.118, "blue": 0.118}
COLOR_WHITE     = {"red": 1.000, "green": 1.000, "blue": 1.000}
COLOR_GOLD      = {"red": 1.000, "green": 0.843, "blue": 0.000}
COLOR_GREEN     = {"red": 0.118, "green": 0.627, "blue": 0.337}
COLOR_GREEN_DIM = {"red": 0.039, "green": 0.180, "blue": 0.098}
COLOR_BLUE      = {"red": 0.055, "green": 0.318, "blue": 0.580}
COLOR_TEAL      = {"red": 0.047, "green": 0.450, "blue": 0.353}
COLOR_DARK_RED  = {"red": 0.550, "green": 0.050, "blue": 0.050}
COLOR_ORANGE    = {"red": 0.980, "green": 0.502, "blue": 0.059}
COLOR_ORANGE_DIM= {"red": 0.250, "green": 0.118, "blue": 0.000}
COLOR_HEADER_BG = {"red": 0.055, "green": 0.055, "blue": 0.055}
COLOR_SUBTEXT   = {"red": 0.500, "green": 0.500, "blue": 0.500}
COLOR_BLACK     = {"red": 0.050, "green": 0.050, "blue": 0.050}


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
                print(f"  Rate limit hit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def read_sheet(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    try:
        sh         = with_retry(lambda: gc.open_by_key(sheet_id))
        ws         = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values:
            return pd.DataFrame()
        start = 1 if "Last Run" in str(all_values[0]) else 0
        if start >= len(all_values):
            return pd.DataFrame()
        headers = all_values[start]
        print(f"  Header row ({start}): {headers[:6]}")
        rows = all_values[start + 1:]
        df   = pd.DataFrame(rows, columns=headers)
        df   = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
        return df
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read sheet '{name}': {e}")
        return pd.DataFrame()


def read_sheet_raw(gc: gspread.Client, sheet_id: str, name: str) -> pd.DataFrame:
    """Read sheet without timestamp row stripping."""
    try:
        sh         = with_retry(lambda: gc.open_by_key(sheet_id))
        ws         = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values or len(all_values) < 2:
            return pd.DataFrame()
        headers = all_values[0]
        rows    = all_values[1:]
        df      = pd.DataFrame(rows, columns=headers)
        df      = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
        return df
    except gspread.WorksheetNotFound:
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def has_signal(signal: str) -> bool:
    s = str(signal).strip()
    return s not in ("", "—", "nan") and len(s) > 1


def edge_is_positive_or_neutral(edge_str: str) -> bool:
    s = str(edge_str).strip()
    if not s or s in ("", "nan", "—"):
        return False
    for char in s:
        if char == '+':
            return True
        if char == '-':
            return False
    return False


def safe_val(row, col: str, default: str = "") -> str:
    try:
        if col not in row.index:
            return default
        val = row[col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return str(val).strip()
    except Exception:
        return default


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> str:
    """Convert implied probability to American odds string."""
    if prob <= 0 or prob >= 1:
        return "—"
    if prob >= 0.5:
        odds = -round((prob / (1 - prob)) * 100)
        return str(int(odds))
    else:
        odds = round(((1 - prob) / prob) * 100)
        return f"+{int(odds)}"


# ── KS Hit Rate Lookup ────────────────────────────────────────────────────

def build_ks_hit_rates(ks_all_scores: pd.DataFrame) -> dict:
    """
    Build hit rate lookup from KS_All_Scores resolved data.
    Returns dict keyed by (score_tier, line, direction) -> hit_rate
    direction: 'over' or 'under'

    Score tiers: "12+", "10-12", "8-10", "6-8", "4-6", "2-4", "Under 2"
    """
    hit_rates = {}

    if ks_all_scores.empty:
        return hit_rates

    resolved = ks_all_scores[
        ks_all_scores["over_hit"].astype(str).str.strip().isin(["Yes", "No"])
    ].copy()

    if resolved.empty:
        return hit_rates

    resolved["ks_score"]   = resolved["ks_score"].apply(safe_float)
    resolved["k_line"]     = resolved["k_line"].apply(safe_float)
    resolved["over_bool"]  = resolved["over_hit"].astype(str).str.strip() == "Yes"
    resolved["under_bool"] = resolved["under_hit"].astype(str).str.strip() == "Yes"
    resolved["over_odds"]  = resolved["over_odds"].apply(safe_float) if "over_odds" in resolved.columns else 0.0
    resolved["under_odds"] = resolved["under_odds"].apply(safe_float) if "under_odds" in resolved.columns else 0.0

    tier_defs = [
        ("12+",    12,  999),
        ("10-12",  10,   12),
        ("8-10",    8,   10),
        ("6-8",     6,    8),
        ("4-6",     4,    6),
        ("2-4",     2,    4),
        ("Under 2", 0,    2),
        ("Under 0", -999, 0),
    ]

    line_vals = [4.5, 5.5, 6.5, 7.5, 8.5]
    MIN_PICKS = 15

    for tier_label, lo, hi in tier_defs:
        tier_sub = resolved[(resolved["ks_score"] >= lo) & (resolved["ks_score"] < hi)]

        # Overall tier rates — no line filter
        if len(tier_sub) >= MIN_PICKS:
            hit_rates[(tier_label, "any", "over")]  = (tier_sub["over_bool"].mean(), len(tier_sub))
            hit_rates[(tier_label, "any", "under")] = (tier_sub["under_bool"].mean(), len(tier_sub))

        for line_val in line_vals:
            sub = tier_sub[tier_sub["k_line"] == line_val]
            if len(sub) < MIN_PICKS:
                continue

            # Overall tier × line
            hit_rates[(tier_label, line_val, "over")]  = (sub["over_bool"].mean(), len(sub))
            hit_rates[(tier_label, line_val, "under")] = (sub["under_bool"].mean(), len(sub))

            # Split by odds sign — minus money vs plus money
            # OVER side
            over_minus = sub[sub["over_odds"] < 0]
            over_plus  = sub[sub["over_odds"] > 0]
            if len(over_minus) >= MIN_PICKS:
                hit_rates[(tier_label, line_val, "over", "minus")] = (over_minus["over_bool"].mean(), len(over_minus))
            if len(over_plus) >= MIN_PICKS:
                hit_rates[(tier_label, line_val, "over", "plus")]  = (over_plus["over_bool"].mean(), len(over_plus))

            # UNDER side
            under_minus = sub[sub["under_odds"] < 0]
            under_plus  = sub[sub["under_odds"] > 0]
            if len(under_minus) >= MIN_PICKS:
                hit_rates[(tier_label, line_val, "under", "minus")] = (under_minus["under_bool"].mean(), len(under_minus))
            if len(under_plus) >= MIN_PICKS:
                hit_rates[(tier_label, line_val, "under", "plus")]  = (under_plus["under_bool"].mean(), len(under_plus))

    print(f"  KS hit rate lookup built: {len(hit_rates)} entries from {len(resolved)} resolved picks")
    return hit_rates


def build_hr_hit_rates(hr_all_scores: pd.DataFrame) -> dict:
    """
    Build hit rate lookup from HR_All_Scores resolved data.
    Returns dict keyed by (score_tier, odds_zone) -> hit_rate
    so we can show tier-specific hit rates on the dashboard.
    """
    hit_rates = {}

    if hr_all_scores.empty:
        return hit_rates

    # Filter to model start date — exclude bad early data
    MODEL_START_DATE = "2026-06-09"
    hr_all_scores = hr_all_scores.copy()
    hr_all_scores["date_dt"] = pd.to_datetime(hr_all_scores["date"], errors="coerce")
    hr_all_scores = hr_all_scores[hr_all_scores["date_dt"] >= pd.Timestamp(MODEL_START_DATE)]

    resolved = hr_all_scores[
        hr_all_scores["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])
    ].copy()

    if resolved.empty:
        return hit_rates

    resolved["hr_score"]  = resolved["hr_score"].apply(safe_float)
    resolved["hit_bool"]  = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["odds_num"]  = resolved["consensus_odds"].apply(safe_float)

    tier_defs = [
        ("13+",    13,  999),
        ("12-13",  12,   13),
        ("11-12",  11,   12),
        ("10-11",  10,   11),
        ("9-10",    9,   10),
        ("8.5-9",   8.5,  9),
    ]

    for tier_label, lo, hi in tier_defs:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) >= 5:
            hit_rates[tier_label] = sub["hit_bool"].mean()

    print(f"  HR hit rate lookup built: {len(hit_rates)} tiers from {len(resolved)} resolved picks")
    return hit_rates


def get_hr_score_tier(score: float) -> str:
    if score >= 13:   return "13+"
    if score >= 12:   return "12-13"
    if score >= 11:   return "11-12"
    if score >= 10:   return "10-11"
    if score >= 9:    return "9-10"
    return "8.5-9"


def calc_hr_value(
    score: float,
    odds: float,
    hit_rates: dict,
) -> tuple:
    """
    Returns (hit_rate_pct, breakeven_american, has_value, edge_str)
    """
    tier = get_hr_score_tier(score)
    hit_rate = hit_rates.get(tier)
    if hit_rate is None or odds <= 0:
        return 0.0, "—", False, "No data"

    implied_odds = american_to_implied(odds)
    edge         = hit_rate - implied_odds
    edge_pct     = round(edge * 100, 1)
    has_value    = edge > 0

    breakeven_american = implied_to_american(hit_rate)
    edge_str           = f"+{edge_pct}%" if edge_pct >= 0 else f"{edge_pct}%"

    return round(hit_rate * 100, 1), breakeven_american, has_value, edge_str


def get_score_tier(score: float) -> str:
    if score >= 12:   return "12+"
    if score >= 10:   return "10-12"
    if score >= 8:    return "8-10"
    if score >= 6:    return "6-8"
    if score >= 4:    return "4-6"
    if score >= 2:    return "2-4"
    if score >= 0:    return "Under 2"
    return "Under 0"


def calc_ks_value(
    score: float,
    line: float,
    over_odds: float,
    under_odds: float,
    hit_rates: dict,
) -> tuple:
    """
    Returns (direction, hit_rate, breakeven, has_value, edge_str)
    direction: 'OVER' or 'UNDER'

    Checks BOTH directions for every pick using actual combo-specific hit rates
    (tier × line), then surfaces whichever has genuine positive edge vs real odds.
    Never infers direction from score alone — the data decides.

    Qualification criteria (replaces hard hit rate floor):
      - Minimum 15 picks in the combo for statistical reliability
      - Minimum +5% edge vs breakeven to filter marginal plays
    """
    if line == 0:
        return "UNDER", 0.0, 0.0, False, "—"

    # Validate American odds — must be +100 or better, or -100 or worse
    if over_odds != 0 and abs(over_odds) < 100:
        over_odds = 0
    if under_odds != 0 and abs(under_odds) < 100:
        under_odds = 0

    tier = get_score_tier(score)

    MIN_PICKS    = 15
    MIN_EDGE_PCT = 0.05  # 5% minimum edge

    best_direction = None
    best_edge      = -999
    best_hit_rate  = 0.0
    best_breakeven = 0.0

    for direction, odds in [("UNDER", under_odds), ("OVER", over_odds)]:
        if odds == 0:
            continue

        # Determine odds sign for this direction
        odds_sign = "minus" if odds < 0 else "plus"

        # Prefer tier × line × odds sign, then tier × line, then tier only
        result = hit_rates.get((tier, line, direction.lower(), odds_sign))
        if result is None:
            result = hit_rates.get((tier, line, direction.lower()))
        if result is None:
            result = hit_rates.get((tier, "any", direction.lower()))
        if result is None:
            continue

        hit_rate, n_picks = result

        # Require minimum sample size
        if n_picks < MIN_PICKS:
            continue

        implied = american_to_implied(odds)
        edge    = hit_rate - implied

        # Require minimum edge magnitude
        if edge < MIN_EDGE_PCT:
            continue

        if edge > best_edge:
            best_edge      = edge
            best_direction = direction
            best_hit_rate  = hit_rate
            best_breakeven = implied_to_american(hit_rate)

    if best_direction is None or best_edge <= 0:
        return "UNDER", 0.0, 0.0, False, "No edge"

    edge_pct = round(best_edge * 100, 1)
    edge_str = f"+{edge_pct}%" if edge_pct >= 0 else f"{edge_pct}%"
    return best_direction, round(best_hit_rate * 100, 1), best_breakeven, True, edge_str


def build_hrrbi_hit_rates(hrrbi_all_scores: pd.DataFrame) -> dict:
    """
    Build hit rate lookup from HRRBI_All_Scores resolved data.
    Returns dict keyed by (score_tier, odds_zone) -> (hit_rate, n_picks)
    OVER only — no direction decision needed.
    """
    hit_rates = {}

    if hrrbi_all_scores.empty:
        return hit_rates

    resolved = hrrbi_all_scores[
        hrrbi_all_scores["over_hit"].astype(str).str.strip().isin(["Yes", "No"])
    ].copy()

    if resolved.empty:
        return hit_rates

    resolved["hrrbi_score"] = resolved["hrrbi_score"].apply(safe_float)
    resolved["over_odds"]   = resolved["over_odds"].apply(safe_float)
    resolved["over_bool"]   = resolved["over_hit"].astype(str).str.strip() == "Yes"

    tier_defs = [
        ("13-15",  13,  999),
        ("11-13",  11,   13),
        ("9-11",    9,   11),
        ("7-9",     7,    9),
        ("Under 7", -999,  7),
    ]

    odds_defs = [
        ("minus_heavy",  -999, -156),  # -156 or worse
        ("minus_mid",    -155, -131),  # -131 to -155
        ("minus_light",  -130, -111),  # -111 to -130
        ("minus_best",   -110,    0),  # -110 or better
        ("plus_low",        1,  111),  # +100 to +110
        ("plus_mid",      111,  121),  # +111 to +120
        ("plus_high",     121,  999),  # +121+
    ]

    for tier_label, t_lo, t_hi in tier_defs:
        tier_sub = resolved[(resolved["hrrbi_score"] >= t_lo) & (resolved["hrrbi_score"] < t_hi)]
        if len(tier_sub) >= 15:
            hit_rates[(tier_label, "any")] = (tier_sub["over_bool"].mean(), len(tier_sub))
        for odds_label, o_lo, o_hi in odds_defs:
            if o_lo < 0:
                sub = tier_sub[(tier_sub["over_odds"] >= o_lo) & (tier_sub["over_odds"] < o_hi)]
            else:
                sub = tier_sub[(tier_sub["over_odds"] >= o_lo) & (tier_sub["over_odds"] < o_hi)]
            if len(sub) >= 15:
                hit_rates[(tier_label, odds_label)] = (sub["over_bool"].mean(), len(sub))

    print(f"  HRRBI hit rate lookup built: {len(hit_rates)} entries from {len(resolved)} resolved picks")
    return hit_rates


def get_hrrbi_score_tier(score: float) -> str:
    if score >= 13:  return "13-15"
    if score >= 11:  return "11-13"
    if score >= 9:   return "9-11"
    if score >= 7:   return "7-9"
    return "Under 7"


def get_hrrbi_odds_zone(odds: float) -> str:
    if odds <= -156:  return "minus_heavy"
    if odds <= -131:  return "minus_mid"
    if odds <= -111:  return "minus_light"
    if odds < 0:      return "minus_best"
    if odds <= 110:   return "plus_low"
    if odds <= 120:   return "plus_mid"
    return "plus_high"


def calc_hrrbi_value(
    score: float,
    odds: float,
    hit_rates: dict,
) -> tuple:
    """
    Returns (hit_rate_pct, breakeven_american, has_value, edge_str)
    OVER only — checks score tier × odds zone combo.
    Requires 15+ picks and 5%+ edge.
    """
    if odds == 0:
        return 0.0, "—", False, "No data"

    # Validate American odds — must be +100 or better, or -100 or worse
    # Values between -99 and 99 are malformed data
    if abs(odds) < 100:
        return 0.0, "—", False, "Invalid odds"

    tier      = get_hrrbi_score_tier(score)
    odds_zone = get_hrrbi_odds_zone(odds)

    MIN_PICKS    = 15
    MIN_EDGE_PCT = 0.05

    # Only use tier × odds zone combo — no fallback to tier-only
    # Prevents showing plays based on inflated overall tier rates
    result = hit_rates.get((tier, odds_zone))
    if result is None:
        return 0.0, "—", False, "No data"

    hit_rate, n_picks = result
    if n_picks < MIN_PICKS:
        return 0.0, "—", False, "Small sample"

    implied  = american_to_implied(odds)
    edge     = hit_rate - implied
    if edge < MIN_EDGE_PCT:
        return round(hit_rate * 100, 1), implied_to_american(hit_rate), False, f"{round(edge*100,1)}%"

    edge_pct  = round(edge * 100, 1)
    edge_str  = f"+{edge_pct}%"
    breakeven = implied_to_american(hit_rate)
    return round(hit_rate * 100, 1), breakeven, True, edge_str


def build_rows(
    hr_df: pd.DataFrame,
    ks_df: pd.DataFrame,
    hrrbi_df: pd.DataFrame,
    ks_hit_rates: dict,
    ks_today: pd.DataFrame = None,
    hr_hit_rates: dict = None,
    hr_today: pd.DataFrame = None,
    hrrbi_hit_rates: dict = None,
    hrrbi_today: pd.DataFrame = None,
):
    N = 9  # expanded to 9 cols for KS value section

    def pad(row):
        return list(row) + [""] * (N - len(row))

    E    = pad([])
    rows = []

    # ── HOME RUN VALUE PLAYS ─────────────────────────────────────────────
    # Score 12+ at any odds up to +699
    # Data: 12-13 | ≤+300 = 37.5%, 13+ | ≤+300 = 25.9%
    #       12-13 | +301-499 = 30.0%, 13+ | +301-499 = 29.2%
    #       12-13 | +500-699 = 25.0%
    # All above breakeven — no lower odds floor needed
    rows.append((pad(["🏠  HOME RUN VALUE PLAYS — 13+ ≤+300 | 10-11 +301-499"]), "section_header_hr"))
    rows.append((pad(["Rank", "Batter", "Team", "Score", "Odds", "", "", ""]), "col_header_hr"))

    hr_source = hr_today if (hr_today is not None and not hr_today.empty) else hr_df
    if hr_source.empty or not hr_hit_rates:
        rows.append((pad(["—", "No value plays today", ""]), "no_plays"))
    else:
        hr_value_plays = []

        for _, row in hr_source.iterrows():
            try:
                batter   = str(row.get("player_name", "")).strip()
                if not batter or batter == "nan":
                    continue
                team     = str(row.get("team", "")).strip()
                hr_score = safe_float(row.get("hr_score", 0))
                odds_raw = str(row.get("consensus_odds", "")).strip()
                odds_val = safe_float(odds_raw.replace("+", "")) if odds_raw not in ("", "nan") else 0.0

                if odds_val <= 0 or hr_score <= 0:
                    continue

                hit_rate, breakeven, has_value, edge_str = calc_hr_value(hr_score, odds_val, hr_hit_rates)

                odds_display = f"+{int(odds_val)}" if odds_val > 0 else str(int(odds_val))
                hr_value_plays.append({
                    "batter":    batter,
                    "team":      team,
                    "score":     str(round(hr_score, 1)),
                    "odds":      odds_display,
                    "breakeven": breakeven,
                    "edge":      edge_str,
                    "edge_num":  float(edge_str.replace("%", "").replace("+", "")),
                })
            except Exception:
                continue

        # Filter — four confirmed value zones (data-driven):
        #   13+   | ≤+499  — 30.6% at ≤+300, 25.8% at +301-499
        #   12-13 | ≤+499  — 25.0% at ≤+300, 30.4% at +301-499
        #   11-12 | ≤+300  — 27.3% (plus odds dead: 5.1% at +301-499)
        #   10-11 | +301-499 — 26.9%
        def in_value_zone(score: float, odds: float) -> bool:
            # Only two confirmed profitable zones (data as of Jun 26):
            # 13+ | ≤+300 — 28.6% on 42 picks (breakeven ~25%)
            # 10-11 | +301-499 — 26.0% on 77 picks (breakeven ~25%)
            if score >= 13.0:
                return odds <= 300
            if score >= 10.0 and score < 11.0:
                return 301 <= odds <= 499
            return False

        hr_value_plays = [
            p for p in hr_value_plays
            if in_value_zone(float(p["score"]), safe_float(p["odds"].replace("+", "")))
        ]

        if not hr_value_plays:
            rows.append((pad(["—", "No value plays today — no qualifying picks in value zones", ""]), "no_plays"))
        else:
            hr_value_plays.sort(key=lambda x: float(x["score"]), reverse=True)
            for i, play in enumerate(hr_value_plays):
                score_val = float(play["score"])
                if score_val >= 13:
                    tier_tag = "🔥"
                elif score_val >= 12:
                    tier_tag = "🟢"
                elif score_val >= 11:
                    tier_tag = "🟡"
                else:
                    tier_tag = "⚪"
                rows.append((pad([
                    str(i + 1),
                    play["batter"],
                    play["team"],
                    f"{tier_tag} {play['score']}",
                    play["odds"],
                    "",
                    "",
                    "",
                ]), f"data_hr_{'strong' if score_val >= 13 else 'moderate' if score_val >= 12 else 'light'}"))

    rows.append((E[:], "spacer"))

    # ── 3-LEG HR PARLAY ──────────────────────────────────────────────────
    # Pool definition (data-driven):
    #   10-11 | +301-499  — 26.7% hit rate
    #   12+   | +301-499  — 30.4% hit rate
    #   13+   | up to +499 — 27.6% at ≤+300, 26.9% at +301-499
    # Leg selector: combined score of platoon + season barrel + HR/FB + weather
    # All confirmed positive separators from feature analysis
    rows.append((pad(["🎰  3-LEG HR PARLAY — Best Legs by Combined Selector"]), "section_header_parlay"))
    rows.append((pad(["Leg", "Batter", "Team", "Score", "Odds", "Selector", "", ""]), "col_header_parlay"))

    if hr_source.empty:
        rows.append((pad(["—", "No parlay candidates today", ""]), "no_plays"))
    else:
        parlay_candidates = []

        for _, row in hr_source.iterrows():
            try:
                batter       = str(row.get("player_name", "")).strip()
                if not batter or batter == "nan":
                    continue
                team         = str(row.get("team", "")).strip()
                opp_pit      = str(row.get("pitcher_name", "")).strip()
                hr_score     = safe_float(row.get("hr_score", 0))
                platoon      = safe_float(row.get("platoon_score", 0))
                season_barrel= safe_float(row.get("season_barrel_pct", 0))
                hr_per_fb    = safe_float(row.get("hr_per_fb", 0))
                weather      = safe_float(row.get("hr_weather_boost", 0))
                odds_raw     = str(row.get("consensus_odds", "")).strip()
                odds_val     = safe_float(odds_raw.replace("+", "")) if odds_raw not in ("", "nan") else 0.0

                if odds_val <= 0 or hr_score <= 0:
                    continue

                # Pool filter — three confirmed value zones:
                # 10-11 at +301-499, 12+ at +301-499, 13+ at any up to +499
                in_pool = (
                    # 13+ | ≤+499
                    (hr_score >= 13.0 and odds_val <= 499) or
                    # 12-13 | ≤+499
                    (hr_score >= 12.0 and hr_score < 13.0 and odds_val <= 499) or
                    # 11-12 | ≤+300 only
                    (hr_score >= 11.0 and hr_score < 12.0 and odds_val <= 300) or
                    # 10-11 | +301-499 only
                    (hr_score >= 10.0 and hr_score < 11.0 and 301 <= odds_val <= 499)
                )
                if not in_pool:
                    continue

                # Combined selector — weighted by parlay leg winner analysis
                # Platoon: +1.131 gap (dominant), HR/FB: +1.522 gap
                # Season barrel and weather barely separating — removed
                # Momentum and barrel% 7d both negative — excluded
                selector = (
                    platoon +
                    (hr_per_fb / 20)
                )

                parlay_candidates.append({
                    "batter":   batter,
                    "team":     team,
                    "opp_pit":  opp_pit,
                    "score":    hr_score,
                    "platoon":  platoon,
                    "selector": selector,
                    "odds":     odds_val,
                })
            except Exception:
                continue

        # Rank by combined selector score
        parlay_candidates.sort(key=lambda x: -x["selector"])

        # Pick top 3, diversified by opposing pitcher (different games)
        selected   = []
        used_games = set()
        for c in parlay_candidates:
            if len(selected) >= 3:
                break
            if c["opp_pit"] and c["opp_pit"] in used_games:
                continue
            selected.append(c)
            if c["opp_pit"]:
                used_games.add(c["opp_pit"])

        # Fill remaining slots if diversification left us short
        if len(selected) < 3:
            for c in parlay_candidates:
                if len(selected) >= 3:
                    break
                if c not in selected:
                    selected.append(c)

        if not selected:
            rows.append((pad(["—", "No parlay candidates today — no qualifying picks in value zones", ""]), "no_plays"))
        else:
            for i, c in enumerate(selected):
                odds_display = f"+{int(c['odds'])}"
                rows.append((pad([
                    str(i + 1),
                    c["batter"],
                    c["team"],
                    f"{c['score']:.1f}",
                    odds_display,
                    f"{c['selector']:.3f}",
                    "",
                    "",
                ]), "data_parlay"))

    rows.append((E[:], "spacer"))

    # ── PITCHER STRIKEOUT VALUE PLAYS ─────────────────────────────────────
    rows.append((pad(["⚾  PITCHER STRIKEOUT VALUE PLAYS — Hit Rate vs Breakeven"]), "section_header_ks"))
    rows.append((pad(["Rank", "Pitcher", "Team", "Score", "Line", "Direction", "Odds", "Breakeven", "Edge"]), "col_header_ks"))

    ks_value_source = ks_today if (ks_today is not None and not ks_today.empty) else ks_df
    if ks_value_source.empty or not ks_hit_rates:
        rows.append((pad(["—", "No value plays today"]), "no_plays"))
    else:
        value_plays = []

        sig_col = next((c for c in ["Signal", "prop_signal", "signal"] if c in ks_value_source.columns), None)

        for _, row in ks_value_source.iterrows():
            score      = safe_float(safe_val(row, "KS Score") or safe_val(row, "ks_score"))
            line       = safe_float(safe_val(row, "K Line") or safe_val(row, "k_line"))
            over_odds  = safe_float(safe_val(row, "Over Odds") or safe_val(row, "ks_over_odds") or safe_val(row, "over_odds"))
            under_odds = safe_float(safe_val(row, "Under Odds") or safe_val(row, "ks_under_odds") or safe_val(row, "under_odds"))

            if line == 0:
                continue

            direction, hit_rate, breakeven, has_value, edge_str = calc_ks_value(
                score, line, over_odds, under_odds, ks_hit_rates
            )

            if not has_value:
                continue

            odds_display = over_odds if direction == "OVER" else under_odds
            odds_str     = str(int(odds_display)) if odds_display != 0 else "—"
            if odds_display > 0:
                odds_str = f"+{odds_str}"

            breakeven_american = breakeven
            value_plays.append({
                "rank":       safe_val(row, "Rank"),
                "pitcher":    safe_val(row, "pitcher_name") or safe_val(row, "Pitcher"),
                "team":       safe_val(row, "Team") or safe_val(row, "team"),
                "line":       str(line),
                "direction":  f"{direction} {line} {'✅' if direction == 'OVER' else '🔻'}",
                "odds":       odds_str,
                "breakeven":  breakeven_american,
                "edge":       edge_str,
                "score":      score,
                "score_str":  str(round(score, 1)),
                "direction_raw": direction,
            })

        if not value_plays:
            rows.append((pad(["—", "No value plays today — no edge found vs hit rates"]), "no_plays"))
        else:
            value_plays.sort(key=lambda x: float(x["edge"].replace("%", "").replace("+", "")), reverse=True)
            for i, play in enumerate(value_plays):
                rows.append((pad([
                    str(i + 1),
                    play["pitcher"],
                    play["team"],
                    play["score_str"],
                    play["line"],
                    play["direction"],
                    play["odds"],
                    play["breakeven"],
                    play["edge"],
                ]), f"data_ks_{play['direction_raw'].lower()}"))

    rows.append((E[:], "spacer"))

    # ── ALL KS SIGNALS (reference) ────────────────────────────────────────
    rows.append((pad(["⚾  ALL KS SIGNALS — Reference"]), "section_header_ks_ref"))
    rows.append((pad(["Rank", "Pitcher", "Team", "K Line", "Odds", "Signal", ""]), "col_header_ks"))

    if ks_df.empty:
        rows.append((pad(["—", "No plays today"]), "no_plays"))
    else:
        sig_col = next((c for c in ["Signal", "prop_signal", "signal"] if c in ks_df.columns), None)
        plays   = ks_df[ks_df[sig_col].apply(has_signal)].copy() if sig_col else pd.DataFrame()

        if plays.empty:
            rows.append((pad(["—", "No plays today"]), "no_plays"))
        else:
            for i in range(len(plays)):
                row       = plays.iloc[i]
                rank      = safe_val(row, "Rank", str(i + 1))
                pitcher   = safe_val(row, "Pitcher") or safe_val(row, "pitcher_name")
                team      = safe_val(row, "Team")
                k_line    = safe_val(row, "K Line") or safe_val(row, "k_line")
                over_odds  = safe_val(row, "Over Odds") or safe_val(row, "ks_over_odds")
                under_odds = safe_val(row, "Under Odds") or safe_val(row, "ks_under_odds")
                signal    = safe_val(row, sig_col)
                odds_display = under_odds if "UNDER" in str(signal).upper() else over_odds
                rows.append((pad([rank, pitcher, team, k_line, odds_display, signal, ""]), "data_ks"))

    rows.append((E[:], "spacer"))

    # ── H+R+RBI VALUE PLAYS ───────────────────────────────────────────────
    rows.append((pad(["📊  H+R+RBI VALUE PLAYS — Hit Rate vs Breakeven"]), "section_header_hrrbi"))
    rows.append((pad(["Rank", "Player", "Team", "Score", "Line", "Odds", "Breakeven", "Edge", ""]), "col_header_hrrbi"))

    hrrbi_source = hrrbi_today if (hrrbi_today is not None and not hrrbi_today.empty) else hrrbi_df
    if hrrbi_source.empty or not hrrbi_hit_rates:
        rows.append((pad(["—", "No value plays today"]), "no_plays"))
    else:
        hrrbi_value_plays = []

        for _, row in hrrbi_source.iterrows():
            try:
                player    = str(row.get("player_name", "")).strip()
                if not player or player == "nan":
                    continue
                team      = str(row.get("team", "")).strip()
                score     = safe_float(row.get("hrrbi_score", 0))
                line      = safe_float(row.get("hrrbi_line", 0))
                odds_raw  = str(row.get("over_odds", "")).strip()
                odds_val  = safe_float(odds_raw.replace("+", "")) if odds_raw not in ("", "nan") else 0.0

                if odds_val == 0 or score == 0 or line == 0:
                    continue

                hit_rate, breakeven, has_value, edge_str = calc_hrrbi_value(score, odds_val, hrrbi_hit_rates)

                if not has_value:
                    continue

                odds_display = f"+{int(odds_val)}" if odds_val > 0 else str(int(odds_val))
                hrrbi_value_plays.append({
                    "player":    player,
                    "team":      team,
                    "score":     str(round(score, 1)),
                    "line":      str(line),
                    "odds":      odds_display,
                    "breakeven": breakeven,
                    "edge":      edge_str,
                    "edge_num":  float(edge_str.replace("%", "").replace("+", "")),
                })
            except Exception:
                continue

        if not hrrbi_value_plays:
            rows.append((pad(["—", "No value plays today — no edge found vs hit rates"]), "no_plays"))
        else:
            hrrbi_value_plays.sort(key=lambda x: x["edge_num"], reverse=True)
            for i, play in enumerate(hrrbi_value_plays):
                rows.append((pad([
                    str(i + 1),
                    play["player"],
                    play["team"],
                    play["score"],
                    play["line"],
                    play["odds"],
                    play["breakeven"],
                    play["edge"],
                    "",
                ]), "data_hrrbi_value"))

    return rows


def write_dashboard(gc: gspread.Client, sheet_id: str, rows) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(DASHBOARD_SHEET)
        with_retry(lambda: ws.clear())
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=DASHBOARD_SHEET, rows=200, cols=9)

    reqs_pre = [{"unmergeCells": {
        "range": {"sheetId": ws.id, "startRowIndex": 0, "endRowIndex": 200,
                  "startColumnIndex": 0, "endColumnIndex": 9}
    }}]
    try:
        with_retry(lambda: sh.batch_update({"requests": reqs_pre}))
    except Exception:
        pass

    data = [row_data for row_data, _ in rows]
    with_retry(lambda: ws.update(data, value_input_option="RAW"))

    ws_id  = ws.id
    n_cols = 9
    reqs   = []

    reqs.append({"repeatCell": {
        "range": {
            "sheetId": ws_id, "startRowIndex": 0,
            "endRowIndex": len(rows) + 5,
            "startColumnIndex": 0, "endColumnIndex": n_cols,
        },
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {
                "foregroundColor": COLOR_WHITE,
                "fontFamily": "Roboto", "fontSize": 11, "bold": False,
            },
            "verticalAlignment": "MIDDLE",
            "horizontalAlignment": "LEFT",
            "wrapStrategy": "CLIP",
        }},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,horizontalAlignment,wrapStrategy)",
    }})

    data_row_count = {"hr": 0, "ks": 0, "ks_over": 0, "ks_under": 0, "hrrbi": 0, "hrrbi_value": 0}

    for row_idx, (row_data, row_type) in enumerate(rows):
        r = row_idx

        if row_type.startswith("section_header"):
            if "ks_ref" in row_type:
                color      = {"red": 0.047, "green": 0.250, "blue": 0.250}
                text_color = COLOR_WHITE
            elif "ks" in row_type:
                color      = COLOR_TEAL
                text_color = COLOR_WHITE
            elif "hrrbi" in row_type:
                color      = COLOR_BLUE
                text_color = COLOR_WHITE
            elif "parlay" in row_type:
                color      = {"red": 0.541, "green": 0.165, "blue": 0.557}
                text_color = COLOR_WHITE
            else:
                color      = COLOR_GOLD
                text_color = COLOR_BLACK

            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": color,
                    "textFormat": {
                        "foregroundColor": text_color, "bold": True,
                        "fontFamily": "Roboto", "fontSize": 12,
                    },
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": n_cols},
                "mergeType": "MERGE_ALL",
            }})

        elif row_type.startswith("col_header"):
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {
                        "foregroundColor": COLOR_SUBTEXT, "bold": True,
                        "fontFamily": "Roboto", "fontSize": 9,
                    },
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})

        elif row_type.startswith("data"):
            if "ks_over" in row_type:
                count = data_row_count.get("ks_over", 0)
                data_row_count["ks_over"] = count + 1
            elif "ks_under" in row_type:
                count = data_row_count.get("ks_under", 0)
                data_row_count["ks_under"] = count + 1
            elif "ks" in row_type:
                count = data_row_count.get("ks", 0)
                data_row_count["ks"] = count + 1
            elif "hrrbi_value" in row_type:
                count = data_row_count.get("hrrbi_value", 0)
                data_row_count["hrrbi_value"] = count + 1
            elif "hrrbi" in row_type:
                count = data_row_count.get("hrrbi", 0)
                data_row_count["hrrbi"] = count + 1
            else:
                count = data_row_count.get("hr", 0)
                data_row_count["hr"] = count + 1

            bg = COLOR_BG if count % 2 == 0 else COLOR_BG_ALT

            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {
                        "foregroundColor": COLOR_WHITE,
                        "fontFamily": "Roboto Mono", "fontSize": 11,
                    },
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})

            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": 1},
                "cell": {"userEnteredFormat": {
                    "textFormat": {
                        "foregroundColor": COLOR_SUBTEXT,
                        "fontFamily": "Roboto", "fontSize": 11, "bold": True,
                    },
                    "horizontalAlignment": "CENTER",
                    "verticalAlignment": "MIDDLE",
                }},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment,verticalAlignment)",
            }})

            if "ks_over" in row_type or "ks_under" in row_type:
                direction_val = str(row_data[5]).upper()
                if "OVER" in direction_val:
                    dir_bg = COLOR_GREEN_DIM
                    dir_fg = COLOR_GREEN
                else:
                    dir_bg = COLOR_DARK_RED
                    dir_fg = COLOR_WHITE

                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 5, "endColumnIndex": 6},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": dir_bg,
                        "textFormat": {"foregroundColor": dir_fg, "bold": True,
                                       "fontFamily": "Roboto", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 7, "endColumnIndex": 8},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": COLOR_HEADER_BG,
                        "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": False,
                                       "fontFamily": "Roboto Mono", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 8, "endColumnIndex": 9},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": COLOR_GREEN_DIM,
                        "textFormat": {"foregroundColor": COLOR_GREEN, "bold": True,
                                       "fontFamily": "Roboto", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})

            elif row_type == "data_hrrbi_value":
                # Edge col (7) — green
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 7, "endColumnIndex": 8},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": COLOR_GREEN_DIM,
                        "textFormat": {"foregroundColor": COLOR_GREEN, "bold": True,
                                       "fontFamily": "Roboto", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})
                # Breakeven col (6) — grey informational
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 6, "endColumnIndex": 7},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": COLOR_HEADER_BG,
                        "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": False,
                                       "fontFamily": "Roboto Mono", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})

            elif row_type in ("data_ks", "data_hrrbi"):
                signal_val = str(row_data[5]).upper()
                if "OVER" in signal_val and "UNDER" not in signal_val:
                    sig_bg   = COLOR_GREEN
                    sig_text = COLOR_WHITE
                elif "UNDER" in signal_val:
                    sig_bg   = COLOR_DARK_RED
                    sig_text = COLOR_WHITE
                else:
                    sig_bg   = bg
                    sig_text = COLOR_SUBTEXT

                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 5, "endColumnIndex": 6},
                    "cell": {"userEnteredFormat": {
                        "backgroundColor": sig_bg,
                        "textFormat": {
                            "foregroundColor": sig_text, "bold": True,
                            "fontFamily": "Roboto", "fontSize": 11,
                        },
                        "horizontalAlignment": "CENTER",
                        "verticalAlignment": "MIDDLE",
                    }},
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)",
                }})

            elif row_type in ("data_hr", "data_hr_strong", "data_hr_moderate"):
                # 🔥 = 13+ (green), 🟢 = 12-13 (gold)
                score_col_color = COLOR_GREEN if "strong" in row_type else COLOR_GOLD
                reqs.append({"repeatCell": {
                    "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                              "startColumnIndex": 3, "endColumnIndex": 4},
                    "cell": {"userEnteredFormat": {
                        "textFormat": {"foregroundColor": score_col_color, "bold": True,
                                       "fontFamily": "Roboto", "fontSize": 11},
                        "horizontalAlignment": "CENTER",
                    }},
                    "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
                }})

        elif row_type == "data_parlay":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 3, "endColumnIndex": 4},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "horizontalAlignment": "CENTER",
                }},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 5, "endColumnIndex": 6},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_GREEN, "bold": False,
                                   "fontFamily": "Roboto Mono", "fontSize": 11},
                    "horizontalAlignment": "CENTER",
                }},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})

        elif row_type == "no_plays":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": n_cols},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_BG_ALT,
                    "textFormat": {
                        "foregroundColor": COLOR_SUBTEXT, "italic": True,
                        "fontFamily": "Roboto", "fontSize": 11,
                    },
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "CLIP",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})

    col_widths = [55, 170, 60, 60, 70, 130, 80, 90, 75]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS",
                      "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w},
            "fields": "pixelSize",
        }})

    for row_idx, (_, row_type) in enumerate(rows):
        if row_type.startswith("section_header"):
            h = 40
        elif row_type == "spacer":
            h = 14
        elif row_type.startswith("col_header"):
            h = 24
        else:
            h = 32
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS",
                      "startIndex": row_idx, "endIndex": row_idx + 1},
            "properties": {"pixelSize": h},
            "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {
            "sheetId": ws_id,
            "gridProperties": {"frozenRowCount": 0},
            "tabColorStyle": {"rgbColor": COLOR_GOLD},
        },
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("Dashboard formatting applied.")
    except APIError as e:
        print(f"Dashboard formatting failed: {e}")


def write_timestamp(gc: gspread.Client, sheet_id: str) -> None:
    et     = pytz.timezone("America/New_York")
    now_et = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")
    sh     = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws    = sh.worksheet(DASHBOARD_SHEET)
        ws_id = ws.id
        with_retry(lambda: ws.insert_row(
            [f"⏱  Last Updated: {now_et}", "", "", "", "", "", "", "", ""], index=1
        ))
        reqs = [
            {"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                          "startColumnIndex": 0, "endColumnIndex": 9},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {
                        "foregroundColor": COLOR_SUBTEXT,
                        "fontFamily": "Roboto", "fontSize": 11,
                        "italic": True, "bold": False,
                    },
                    "verticalAlignment": "MIDDLE",
                    "wrapStrategy": "OVERFLOW_CELL",
                }},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
            }},
            {"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                          "startColumnIndex": 0, "endColumnIndex": 9},
                "mergeType": "MERGE_ALL",
            }},
            {"updateDimensionProperties": {
                "range": {"sheetId": ws_id, "dimension": "ROWS",
                          "startIndex": 0, "endIndex": 1},
                "properties": {"pixelSize": 32},
                "fields": "pixelSize",
            }},
        ]
        with_retry(lambda: sh.batch_update({"requests": reqs}))
    except Exception as e:
        print(f"Dashboard timestamp failed: {e}")
    print(f"Dashboard timestamp written: {now_et}")


def main() -> None:
    time.sleep(5)
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading picks sheets for dashboard...")
    hr_df    = read_sheet(gc, sheet_id, "Top_HR_Picks")
    time.sleep(2)
    ks_df    = read_sheet(gc, sheet_id, "Top_KS_Picks")
    time.sleep(2)
    hrrbi_df = read_sheet(gc, sheet_id, "Top_HRRBI_Picks")
    time.sleep(2)

    print("Reading KS_All_Scores for hit rate lookup and value finder...")
    ks_all_scores = read_sheet_raw(gc, sheet_id, "KS_All_Scores")
    time.sleep(2)

    print("Reading HR_All_Scores for hit rate lookup...")
    hr_all_scores = read_sheet_raw(gc, sheet_id, "HR_All_Scores")
    time.sleep(2)

    print("Reading HRRBI_All_Scores for hit rate lookup...")
    hrrbi_all_scores = read_sheet_raw(gc, sheet_id, "HRRBI_All_Scores")
    time.sleep(2)

    print(f"HR picks: {len(hr_df)} rows")
    print(f"KS picks: {len(ks_df)} rows")
    print(f"HRRBI picks: {len(hrrbi_df)} rows")
    print(f"KS All Scores: {len(ks_all_scores)} rows")
    print(f"HRRBI All Scores: {len(hrrbi_all_scores)} rows")

    ks_hit_rates    = build_ks_hit_rates(ks_all_scores)
    hr_hit_rates    = build_hr_hit_rates(hr_all_scores)
    hrrbi_hit_rates = build_hrrbi_hit_rates(hrrbi_all_scores)

    from datetime import date as _date2
    _today_str = _date2.today().strftime("%Y-%m-%d")
    hr_today = hr_all_scores[hr_all_scores["date"].astype(str).str.strip() == _today_str].copy() if not hr_all_scores.empty else pd.DataFrame()
    print(f"HR today's scores for value finder: {len(hr_today)} players")

    from datetime import date as _date
    today_str = _date.today().strftime("%Y-%m-%d")
    ks_today  = ks_all_scores[ks_all_scores["date"].astype(str).str.strip() == today_str].copy() if not ks_all_scores.empty else pd.DataFrame()
    print(f"KS today's scores for value finder: {len(ks_today)} pitchers")

    hrrbi_today = hrrbi_all_scores[hrrbi_all_scores["date"].astype(str).str.strip() == today_str].copy() if not hrrbi_all_scores.empty else pd.DataFrame()
    print(f"HRRBI today's scores for value finder: {len(hrrbi_today)} players")

    rows = build_rows(hr_df, ks_df, hrrbi_df, ks_hit_rates, ks_today, hr_hit_rates, hr_today, hrrbi_hit_rates, hrrbi_today)
    write_dashboard(gc, sheet_id, rows)
    time.sleep(3)
    write_timestamp(gc, sheet_id)
    print("Dashboard written to 'Today's Top Picks'")


if __name__ == "__main__":
    main()
