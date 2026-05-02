"""
hrrbi_picks.py
Hits + Runs + RBI model — absolute threshold scoring system.
Mirrors hr_picks.py architecture exactly.
Outputs Top 10 picks + EV subsection + scorecard to 'Top_HRRBI_Picks' sheet.
"""

import pandas as pd
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import date, datetime
import pytz
import os
import json
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# CONSTANTS — mirror HR model style
# ══════════════════════════════════════════════════════════════════
LEAGUE_AVG_BA               = 0.248
LEAGUE_AVG_OBP              = 0.318
LEAGUE_AVG_WOBA             = 0.318
LEAGUE_AVG_XWOBA            = 0.318
LEAGUE_AVG_ISO              = 0.155
LEAGUE_AVG_HARD_HIT_SEASON  = 38.0
LEAGUE_AVG_BARREL_SEASON    = 8.0
LEAGUE_AVG_EV_SEASON        = 88.5
LEAGUE_AVG_HARD_HIT_7D      = 40.0
LEAGUE_AVG_EV_7D            = 89.0
LEAGUE_AVG_BARREL_7D        = 11.0

MIN_PA                      = 100   # minimum PA for full scoring
MIN_BBE_7D                  = 5     # floor for 7-day batted ball events
MIN_BATTING_AVG             = 0.190 # filter out automatic outs
MAX_PER_TEAM                = 2     # diversity cap
MAX_CHALK_PICKS             = 3
CHALK_ODDS_THRESHOLD        = -140  # H+R+RBI props are listed differently than HRs
TOP_N                       = 10

# ══════════════════════════════════════════════════════════════════
# GOOGLE SHEETS AUTH
# ══════════════════════════════════════════════════════════════════
def get_sheet(sheet_id: str):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_json = os.environ["GOOGLE_CREDENTIALS"]
    creds_dict = json.loads(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id)

# ══════════════════════════════════════════════════════════════════
# ABSOLUTE SCORING FUNCTIONS
# All return float. 7-day functions return 0.0 if bbe_7d < MIN_BBE_7D.
# ══════════════════════════════════════════════════════════════════

# ── Contact Quality ──────────────────────────────────────────────
def score_avg(v: float, pa: int) -> float:
    """Season batting average."""
    if pa < MIN_PA: return 0.0
    if v >= 0.320: return 2.0
    if v >= 0.300: return 1.5
    if v >= 0.280: return 1.0
    if v >= 0.260: return 0.5
    if v >= 0.240: return 0.2
    return 0.0

def score_xwoba(v: float, pa: int) -> float:
    """Expected wOBA — luck-normalized contact quality."""
    if pa < MIN_PA: return 0.0
    if v >= 0.390: return 2.0
    if v >= 0.360: return 1.5
    if v >= 0.340: return 1.0
    if v >= 0.320: return 0.5
    if v >= 0.300: return 0.2
    return 0.0

def score_obp(v: float, pa: int) -> float:
    """On-base percentage — run scoring enabler."""
    if pa < MIN_PA: return 0.0
    if v >= 0.400: return 1.5
    if v >= 0.370: return 1.0
    if v >= 0.350: return 0.6
    if v >= 0.330: return 0.3
    return 0.0

def score_iso(v: float, pa: int) -> float:
    """ISO — extra base hit power, drives RBIs."""
    if pa < MIN_PA: return 0.0
    if v >= 0.280: return 1.2
    if v >= 0.230: return 0.9
    if v >= 0.190: return 0.6
    if v >= 0.165: return 0.3
    return 0.0

def score_hard_hit_pct_season(v: float, pa: int) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 52.0: return 0.8
    if v >= 45.0: return 0.5
    if v >= 38.0: return 0.2
    return 0.0

def score_barrel_pct_season(v: float, pa: int) -> float:
    if pa < MIN_PA: return 0.0
    if v >= 12.0: return 1.0
    if v >= 9.0:  return 0.7
    if v >= 6.0:  return 0.4
    return 0.0

def score_ld_pct(v: float, pa: int) -> float:
    """Line drive % — highest BABIP outcome, best hit predictor."""
    if pa < MIN_PA: return 0.0
    if v >= 28.0: return 1.2
    if v >= 24.0: return 0.8
    if v >= 21.0: return 0.4
    if v >= 18.0: return 0.1
    return 0.0

# ── 7-Day Hot Streak ─────────────────────────────────────────────
def score_avg_ev_7d(v: float, bbe_7d: int) -> float:
    if bbe_7d < MIN_BBE_7D: return 0.0
    if v >= 97.0: return 1.0
    if v >= 94.0: return 0.6
    if v >= 91.0: return 0.3
    return 0.0

def score_hard_hit_pct_7d(v: float, bbe_7d: int) -> float:
    if bbe_7d < MIN_BBE_7D: return 0.0
    if v >= 55.0: return 0.8
    if v >= 45.0: return 0.5
    if v >= 35.0: return 0.2
    return 0.0

def score_barrel_pct_7d(v: float, bbe_7d: int) -> float:
    if bbe_7d < MIN_BBE_7D: return 0.0
    if v >= 20.0: return 1.0
    if v >= 15.0: return 0.7
    if v >= 10.0: return 0.4
    return 0.0

def score_14d_avg(v: float, pa_14d: int) -> float:
    """14-day rolling average — hot/cold form."""
    if pa_14d < 20: return 0.0
    if v >= 0.360: return 1.5
    if v >= 0.320: return 1.0
    if v >= 0.290: return 0.5
    if v >= 0.260: return 0.2
    if v <= 0.180: return -1.0
    if v <= 0.210: return -0.5
    return 0.0

# ── Run Scoring Context ──────────────────────────────────────────
def score_bat_order(v: float, pa: int) -> float:
    """
    Avg lineup spot. Lower = more run opportunities.
    1-2 = leadoff/run scorers, 3-5 = RBI spots.
    Scored as general opportunity — both extremes score well.
    """
    if pa < MIN_PA: return 0.0
    if v <= 2.0: return 1.5   # leadoff/2-hole: max runs
    if v <= 3.5: return 1.2   # 3-hole: both runs + RBI
    if v <= 5.0: return 1.0   # 4-5 hole: max RBI
    if v <= 6.5: return 0.4   # 6-hole: some RBI
    return 0.0                # bottom of order

def score_speed(v: float) -> float:
    """Speed score (FanGraphs Spd) — legs hits + scoring from 2nd."""
    if v >= 7.0: return 0.8
    if v >= 5.5: return 0.4
    if v >= 4.0: return 0.1
    return 0.0

def score_bb_pct(v: float, pa: int) -> float:
    """Walk rate — OBP boost, more baserunner chances."""
    if pa < MIN_PA: return 0.0
    if v >= 14.0: return 0.8
    if v >= 10.0: return 0.5
    if v >= 8.0:  return 0.2
    return 0.0

# ── Pitcher Matchup (opposing pitcher) ───────────────────────────
def score_opp_pitcher_ba_against(v: float) -> float:
    """Opposing pitcher BA against — higher = better for batter."""
    if v >= 0.290: return 1.5
    if v >= 0.270: return 1.0
    if v >= 0.255: return 0.5
    if v >= 0.240: return 0.2
    if v <= 0.210: return -0.8
    if v <= 0.225: return -0.4
    return 0.0

def score_opp_pitcher_whip(v: float) -> float:
    """Opposing pitcher WHIP — more baserunners = more runs/RBI chances."""
    if v >= 1.50: return 1.2
    if v >= 1.35: return 0.8
    if v >= 1.20: return 0.4
    if v >= 1.10: return 0.1
    if v <= 0.95: return -0.6
    return 0.0

def score_opp_pitcher_k_pct(v: float) -> float:
    """Opposing pitcher K% — high K pitcher suppresses H+R+RBI."""
    if v >= 30.0: return -1.2
    if v >= 26.0: return -0.8
    if v >= 22.0: return -0.4
    if v <= 16.0: return 0.6
    if v <= 18.0: return 0.3
    return 0.0

def score_opp_pitcher_gb_pct(v: float) -> float:
    """GB pitcher suppresses extra bases and HR, but not hits."""
    if v >= 55.0: return -0.3  # mild negative for XBH
    if v <= 35.0: return 0.3   # fly ball pitcher = more XBH
    return 0.0

# ── Park Factor ──────────────────────────────────────────────────
def score_park_hits(v: float) -> float:
    """Park factor for hits (100 = neutral)."""
    if v >= 110: return 0.5
    if v >= 105: return 0.3
    if v >= 100: return 0.1
    if v <= 90:  return -0.4
    if v <= 95:  return -0.2
    return 0.0

# ══════════════════════════════════════════════════════════════════
# PLATOON SCORING — mirrors HR model continuous scaling
# ══════════════════════════════════════════════════════════════════
def platoon_score(batter_hand: str, pitcher_hand: str,
                  batter_iso: float, pa: int) -> tuple:
    """
    Returns (score, penalty) for platoon matchup.
    Same-hand = harder. Opposite-hand = easier.
    Uses ISO gap as proxy for platoon split magnitude.
    """
    score = 0.0
    penalty = 0.0

    if not batter_hand or not pitcher_hand:
        return score, penalty

    same_hand = batter_hand == pitcher_hand

    # Estimated ISO gap (proxy — ideally use actual platoon splits)
    iso_gap = 0.04 if same_hand else -0.04

    if iso_gap > 0:
        penalty = round(min(iso_gap * 15.0, 4.5), 3)
    elif iso_gap < 0:
        score += round(min(abs(iso_gap) * 10.0, 3.0), 3)

    return round(score, 3), round(penalty, 3)

# ══════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════
def compute_hrrbi_score(row: pd.Series) -> dict:
    pa = int(row.get("pa", 0))
    bbe_7d = int(row.get("bbe_7d", 0))
    pa_14d = int(row.get("pa_14d", 0))

    # ── Batter contact & quality ─────────────────────────────────
    s_avg             = score_avg(row.get("avg", 0), pa)
    s_xwoba           = score_xwoba(row.get("xwoba", 0), pa)
    s_obp             = score_obp(row.get("obp", 0), pa)
    s_iso             = score_iso(row.get("iso", 0), pa)
    s_hard_season     = score_hard_hit_pct_season(row.get("hard_hit_pct_season", 0), pa)
    s_barrel_season   = score_barrel_pct_season(row.get("barrel_pct_season", 0), pa)
    s_ld              = score_ld_pct(row.get("ld_pct", 0), pa)

    # ── 7-day hot streak ─────────────────────────────────────────
    s_ev_7d           = score_avg_ev_7d(row.get("avg_ev_7d", 0), bbe_7d)
    s_hh_7d           = score_hard_hit_pct_7d(row.get("hard_hit_pct_7d", 0), bbe_7d)
    s_barrel_7d       = score_barrel_pct_7d(row.get("barrel_pct_7d", 0), bbe_7d)
    s_14d_avg         = score_14d_avg(row.get("avg_14d", 0), pa_14d)

    # ── Run/RBI context ───────────────────────────────────────────
    s_bat_order       = score_bat_order(row.get("avg_bat_order", 6), pa)
    s_speed           = score_speed(row.get("speed_score", 0))
    s_bb              = score_bb_pct(row.get("bb_pct", 0), pa)

    # ── Opposing pitcher ─────────────────────────────────────────
    s_opp_ba          = score_opp_pitcher_ba_against(row.get("opp_ba_against", LEAGUE_AVG_BA))
    s_opp_whip        = score_opp_pitcher_whip(row.get("opp_whip", 1.20))
    s_opp_k           = score_opp_pitcher_k_pct(row.get("opp_k_pct", 22.0))
    s_opp_gb          = score_opp_pitcher_gb_pct(row.get("opp_gb_pct", 44.0))

    # ── Park ─────────────────────────────────────────────────────
    s_park            = score_park_hits(row.get("park_factor_hits", 100))

    # ── Platoon ───────────────────────────────────────────────────
    plat_score, plat_penalty = platoon_score(
        row.get("batter_hand", ""),
        row.get("pitcher_hand", ""),
        row.get("iso", 0),
        pa,
    )
    platoon_capped = np.clip(plat_score - plat_penalty, -2.0, 2.0)

    # ── Totals ────────────────────────────────────────────────────
    batter_base = (
        s_avg + s_xwoba + s_obp + s_iso +
        s_hard_season + s_barrel_season + s_ld
    )
    streak_score = s_ev_7d + s_hh_7d + s_barrel_7d + s_14d_avg
    context_score = s_bat_order + s_speed + s_bb
    pitcher_score = s_opp_ba + s_opp_whip + s_opp_k + s_opp_gb
    park_score    = s_park

    total = round(
        batter_base + streak_score + context_score +
        pitcher_score + park_score + platoon_capped, 2
    )

    # ── Confidence tier ───────────────────────────────────────────
    if total >= 14:
        confidence = "Elite"
    elif total >= 12:
        confidence = "High"
    elif total >= 10:
        confidence = "Medium"
    elif total >= 8:
        confidence = "Low"
    else:
        confidence = "Fade"

    # ── Prop signal ───────────────────────────────────────────────
    # H+R+RBI over 1.5 is a common prop line
    prop_signal = "YES" if total >= 10 else ("LEAN" if total >= 8 else "NO")

    return {
        "score": total,
        "confidence": confidence,
        "prop_signal": prop_signal,
        "batter_base": round(batter_base, 2),
        "streak_score": round(streak_score, 2),
        "context_score": round(context_score, 2),
        "pitcher_score": round(pitcher_score, 2),
        "park_score": round(park_score, 2),
        "platoon_score": round(platoon_capped, 2),
        # component detail
        "s_avg": s_avg, "s_xwoba": s_xwoba, "s_obp": s_obp, "s_iso": s_iso,
        "s_ld": s_ld, "s_hard_season": s_hard_season,
        "s_ev_7d": s_ev_7d, "s_hh_7d": s_hh_7d, "s_14d_avg": s_14d_avg,
        "s_bat_order": s_bat_order, "s_speed": s_speed,
        "s_opp_ba": s_opp_ba, "s_opp_whip": s_opp_whip,
        "s_opp_k": s_opp_k, "s_park": s_park,
    }

# ══════════════════════════════════════════════════════════════════
# DIVERSITY CAP — mirrors HR model two-pass system
# ══════════════════════════════════════════════════════════════════
def apply_diversity_cap(df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    Two-pass diversity cap:
    Pass 1: Players with odds, chalk cap enforced
    Pass 2: Players without odds fill remaining slots
    """
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    selected = []
    team_counts = {}
    chalk_count = 0

    has_odds = df[df["hrrbi_odds"].notna() & (df["hrrbi_odds"] != "")]
    no_odds = df[df["hrrbi_odds"].isna() | (df["hrrbi_odds"] == "")]

    for _, row in has_odds.iterrows():
        if len(selected) >= top_n:
            break
        team = row.get("team", "UNK")
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue
        try:
            odds_val = float(row["hrrbi_odds"])
            is_chalk = odds_val <= CHALK_ODDS_THRESHOLD
        except (ValueError, TypeError):
            is_chalk = False
        if is_chalk and chalk_count >= MAX_CHALK_PICKS:
            continue
        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1
        if is_chalk:
            chalk_count += 1

    for _, row in no_odds.iterrows():
        if len(selected) >= top_n:
            break
        team = row.get("team", "UNK")
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue
        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1

    return pd.DataFrame(selected).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
def load_data(wb) -> pd.DataFrame:
    """
    Loads from HRRBI_Statcast sheet and merges with pitcher/park data.
    Falls back gracefully if sheets are missing.
    """
    print("Loading HRRBI_Statcast data...")
    try:
        ws = wb.worksheet("HRRBI_Statcast")
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} batters from HRRBI_Statcast")
    except Exception as e:
        print(f"  ERROR loading HRRBI_Statcast: {e}")
        return pd.DataFrame()

    # Load pitcher stats for matchup scoring
    try:
        pws = wb.worksheet("Pitcher_Statcast")
        pdata = pws.get_all_records()
        pitchers = pd.DataFrame(pdata)
        # Rename pitcher columns to opp_ prefix for batter merge
        pitcher_map = {
            "ba_against": "opp_ba_against",
            "whip": "opp_whip",
            "k_pct": "opp_k_pct",
            "gb_pct": "opp_gb_pct",
            "pitcher_hand": "pitcher_hand",
            "name": "opp_pitcher_name",
        }
        pitchers = pitchers.rename(columns={k: v for k, v in pitcher_map.items() if k in pitchers.columns})
        print(f"  Loaded {len(pitchers)} pitchers")
    except Exception as e:
        print(f"  WARNING: Pitcher_Statcast not loaded: {e} — matchup scoring will use league averages")
        pitchers = pd.DataFrame()

    # Load park factors
    try:
        pkws = wb.worksheet("Park_Factors")
        pkdata = pkws.get_all_records()
        parks = pd.DataFrame(pkdata)
        if "hits_factor" in parks.columns:
            parks = parks.rename(columns={"hits_factor": "park_factor_hits"})
        print(f"  Loaded {len(parks)} park factors")
    except Exception as e:
        print(f"  WARNING: Park_Factors not loaded: {e} — park scoring will use neutral 100")
        parks = pd.DataFrame()

    # Load today's confirmed starters if available
    try:
        lws = wb.worksheet("Lineups")
        ldata = lws.get_all_records()
        lineups = pd.DataFrame(ldata)
        # Expected cols: team, opp_pitcher_id or opp_pitcher_name, game_date
        print(f"  Loaded {len(lineups)} lineup rows")
    except Exception:
        lineups = pd.DataFrame()
        print("  INFO: Lineups sheet not found — pitcher matchup defaults to league avg")

    # Merge pitcher into batter df via team/opponent
    if not pitchers.empty and not lineups.empty:
        # Join lineups → pitchers → batters
        pass  # Full join logic depends on column availability from main.py

    # Add placeholder opp columns if not present
    for col, default in [
        ("opp_ba_against", LEAGUE_AVG_BA),
        ("opp_whip", 1.20),
        ("opp_k_pct", 22.0),
        ("opp_gb_pct", 44.0),
        ("pitcher_hand", ""),
        ("park_factor_hits", 100),
        ("hrrbi_odds", ""),
        ("batter_hand", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    # Numeric coerce
    numeric_cols = [
        "pa", "avg", "obp", "slg", "iso", "woba", "xwoba", "babip",
        "bb_pct", "k_pct", "ld_pct", "gb_pct", "fb_pct",
        "hard_hit_pct_season", "barrel_pct_season", "avg_ev_season",
        "speed_score", "sb", "wrc_plus", "avg_bat_order",
        "bbe_7d", "avg_ev_7d", "avg_la_7d", "hard_hit_pct_7d", "barrel_pct_7d",
        "pa_14d", "hits_14d", "avg_14d",
        "opp_ba_against", "opp_whip", "opp_k_pct", "opp_gb_pct",
        "park_factor_hits",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Filter
    df = df[df.get("avg", pd.Series(dtype=float)) >= MIN_BATTING_AVG] if "avg" in df.columns else df
    df = df[df.get("pa", pd.Series(dtype=float)) >= 50] if "pa" in df.columns else df

    return df.reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# SHEET OUTPUT HELPERS
# ══════════════════════════════════════════════════════════════════
def confidence_color(conf: str) -> dict:
    colors = {
        "Elite":  {"red": 0.13, "green": 0.55, "blue": 0.13},
        "High":   {"red": 0.13, "green": 0.73, "blue": 0.37},
        "Medium": {"red": 0.95, "green": 0.77, "blue": 0.06},
        "Low":    {"red": 0.94, "green": 0.50, "blue": 0.13},
        "Fade":   {"red": 0.80, "green": 0.20, "blue": 0.20},
    }
    return colors.get(conf, {"red": 1, "green": 1, "blue": 1})

def write_picks(wb, picks: pd.DataFrame, ev_picks: pd.DataFrame):
    print("Writing Top_HRRBI_Picks sheet...")
    try:
        ws = wb.worksheet("Top_HRRBI_Picks")
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = wb.add_worksheet(title="Top_HRRBI_Picks", rows=200, cols=30)

    output_cols = [
        "name", "team", "avg_bat_order", "avg", "obp", "iso", "xwoba",
        "ld_pct", "hard_hit_pct_season", "avg_ev_7d", "avg_14d",
        "opp_pitcher_name", "opp_k_pct", "opp_whip",
        "park_factor_hits", "score", "confidence", "prop_signal",
        "hrrbi_odds", "bet_placed", "result",
    ]

    for col in output_cols:
        if col not in picks.columns:
            picks[col] = ""

    # ── Section header ────────────────────────────────────────────
    rows_to_write = []
    rows_to_write.append(["🎯 TOP H+R+RBI PICKS"] + [""] * (len(output_cols) - 1))
    rows_to_write.append(output_cols)

    for _, row in picks.iterrows():
        rows_to_write.append([str(row.get(c, "")) for c in output_cols])

    rows_to_write.append([""] * len(output_cols))

    # ── EV subsection ─────────────────────────────────────────────
    rows_to_write.append(["📊 H+R+RBI EV WATCH"] + [""] * (len(output_cols) - 1))
    rows_to_write.append(output_cols)
    for _, row in ev_picks.iterrows():
        rows_to_write.append([str(row.get(c, "")) for c in output_cols])

    ws.update(rows_to_write)

    # ── Formatting ────────────────────────────────────────────────
    # Title rows
    for title_row in [1, len(picks) + 4]:
        ws.format(f"A{title_row}", {
            "backgroundColor": {"red": 0.13, "green": 0.37, "blue": 0.73},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                           "fontSize": 12},
        })

    # Header rows
    for header_row in [2, len(picks) + 5]:
        ws.format(f"A{header_row}:Z{header_row}", {
            "backgroundColor": {"red": 0.20, "green": 0.20, "blue": 0.20},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })

    # Confidence color coding
    conf_col_idx = output_cols.index("confidence") + 1
    conf_letter = chr(64 + conf_col_idx)
    for i, (_, row) in enumerate(picks.iterrows()):
        sheet_row = i + 3
        color = confidence_color(str(row.get("confidence", "")))
        ws.format(f"{conf_letter}{sheet_row}", {
            "backgroundColor": color,
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })

    print(f"  ✓ Top_HRRBI_Picks written: {len(picks)} picks + {len(ev_picks)} EV")

# ══════════════════════════════════════════════════════════════════
# SCORECARD
# ══════════════════════════════════════════════════════════════════
def update_scorecard(wb):
    """
    Reads historical Top_HRRBI_Picks and computes hit rates by tier.
    Writes to HRRBI_Scorecard tab.
    """
    print("Updating HRRBI_Scorecard...")
    try:
        ws = wb.worksheet("Top_HRRBI_Picks")
        data = ws.get_all_records()
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"  WARNING: could not load picks for scorecard: {e}")
        return

    if df.empty or "result" not in df.columns:
        print("  INFO: No result data yet for scorecard")
        return

    # Filter scored rows only (Main picks with result filled)
    df = df[df["result"].isin(["HIT", "MISS", "1", "0"])].copy()
    if df.empty:
        print("  INFO: No completed picks for scorecard")
        return

    df["hit"] = df["result"].isin(["HIT", "1"]).astype(int)
    df["score_num"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)

    tiers = [
        ("14+",    df["score_num"] >= 14),
        ("12+",    df["score_num"] >= 12),
        ("10+",    df["score_num"] >= 10),
        ("8+",     df["score_num"] >= 8),
        ("Under 8", df["score_num"] < 8),
    ]

    try:
        sc_ws = wb.worksheet("HRRBI_Scorecard")
        sc_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        sc_ws = wb.add_worksheet(title="HRRBI_Scorecard", rows=50, cols=10)

    headers = ["Score Tier", "Picks", "Hits", "Hit Rate", "Notes"]
    sc_rows = [["📊 HRRBI MODEL PERFORMANCE"] + [""] * 4, headers]

    for label, mask in tiers:
        subset = df[mask]
        picks = len(subset)
        hits = subset["hit"].sum()
        rate = f"{hits/picks*100:.1f}%" if picks > 0 else "—"
        note = "✅ Strong" if picks > 0 and hits/picks >= 0.40 else (
               "⚠️ Weak" if picks > 0 and hits/picks < 0.25 else "")
        sc_rows.append([label, picks, hits, rate, note])

    sc_ws.update(sc_rows)
    sc_ws.format("A1:E1", {
        "backgroundColor": {"red": 0.13, "green": 0.37, "blue": 0.73},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    })
    print("  ✓ HRRBI_Scorecard updated")

# ══════════════════════════════════════════════════════════════════
# TIMESTAMP
# ══════════════════════════════════════════════════════════════════
def write_timestamp(ws):
    eastern = pytz.timezone("America/New_York")
    ts = datetime.now(eastern).strftime("Last run: %B %d, %Y %I:%M %p ET")
    try:
        ws.insert_row([ts], index=1)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    sheet_id = os.environ.get("GOOGLE_SHEET_ID", "")
    if not sheet_id:
        raise ValueError("GOOGLE_SHEET_ID environment variable not set")

    wb = get_sheet(sheet_id)
    df = load_data(wb)

    if df.empty:
        print("ERROR: No batter data loaded — aborting")
        return

    print(f"\nScoring {len(df)} batters...")
    score_results = df.apply(compute_hrrbi_score, axis=1)
    score_df = pd.DataFrame(score_results.tolist())
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    # ── Filter & rank ─────────────────────────────────────────────
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # Apply diversity cap for top picks
    top_picks = apply_diversity_cap(df, TOP_N)
    top_picks["section"] = "Main"

    # EV subsection: strong contact, not in top 10
    top_names = set(top_picks.get("name", pd.Series()).tolist())
    ev_candidates = df[~df.get("name", pd.Series()).isin(top_names)].copy()
    ev_picks = ev_candidates[
        (ev_candidates.get("avg_ev_7d", pd.Series(0)) >= 93) &
        (ev_candidates.get("barrel_pct_season", pd.Series(0)) >= 7) &
        (ev_candidates.get("xwoba", pd.Series(0)) >= 0.340) &
        (ev_candidates.get("bbe_7d", pd.Series(0)) >= MIN_BBE_7D)
    ].head(5).copy()
    ev_picks["section"] = "EV"

    print(f"\n{'='*50}")
    print(f"Top {len(top_picks)} H+R+RBI Picks:")
    for _, row in top_picks.iterrows():
        print(f"  {row.get('name','?'):20s}  score={row.get('score',0):5.2f}  "
              f"{row.get('confidence','?'):8s}  {row.get('prop_signal','?')}")
    print(f"{'='*50}\n")

    write_picks(wb, top_picks, ev_picks)

    ws = wb.worksheet("Top_HRRBI_Picks")
    write_timestamp(ws)

    update_scorecard(wb)
    print("✓ HRRBI model complete")

if __name__ == "__main__":
    main()

