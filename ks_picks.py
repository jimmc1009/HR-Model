"""
ks_picks.py
Pitcher Strikeout model — absolute threshold scoring system.
Mirrors hr_picks.py architecture exactly.
Outputs Top 10 picks + EV subsection + scorecard to 'Top_KS_Picks' sheet.
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
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
LEAGUE_AVG_K_PCT            = 22.5
LEAGUE_AVG_SWSTR_PCT        = 11.0
LEAGUE_AVG_CHASE_RATE       = 29.0
LEAGUE_AVG_K_PER_9          = 8.8
LEAGUE_AVG_BB_PCT           = 8.5
LEAGUE_AVG_VELO             = 93.5
LEAGUE_AVG_OPP_K_PCT        = 22.5  # league avg team K rate

MIN_IP                      = 20    # minimum IP for full scoring
MIN_GS                      = 4     # minimum starts
AVG_IP_OPENER_THRESHOLD     = 4.0   # below this = opener/bulk risk
TOP_N                       = 10
MAX_PER_TEAM                = 1     # only 1 pitcher per team (1 start per day)
MAX_CHALK_PICKS             = 4

# ══════════════════════════════════════════════════════════════════
# GOOGLE SHEETS AUTH
# ══════════════════════════════════════════════════════════════════
def get_sheet(sheet_id: str):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    creds_dict = json.loads(creds_json)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(sheet_id)

# ══════════════════════════════════════════════════════════════════
# ABSOLUTE SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

# ── Core K Ability ───────────────────────────────────────────────
def score_k_pct_season(v: float, ip: float) -> float:
    """Season K% — primary baseline."""
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
    """Swinging strike % — #1 K predictor."""
    if ip < MIN_IP: return 0.0
    if v >= 15.0: return 2.5
    if v >= 13.5: return 2.0
    if v >= 12.5: return 1.5
    if v >= 11.5: return 1.0
    if v >= 10.5: return 0.5
    if v >= 9.5:  return 0.1
    if v <= 8.0:  return -0.5
    return 0.0

def score_chase_rate(v: float, ip: float) -> float:
    """O-Swing% — batters chasing off zone = free Ks."""
    if ip < MIN_IP: return 0.0
    if v >= 35.0: return 1.5
    if v >= 33.0: return 1.0
    if v >= 31.0: return 0.6
    if v >= 29.0: return 0.2
    if v <= 24.0: return -0.5
    if v <= 26.0: return -0.2
    return 0.0

def score_k_per_9(v: float, ip: float) -> float:
    """K/9 — rate stat context."""
    if ip < MIN_IP: return 0.0
    if v >= 13.0: return 1.5
    if v >= 11.5: return 1.2
    if v >= 10.5: return 0.9
    if v >= 9.5:  return 0.6
    if v >= 8.5:  return 0.3
    if v <= 6.5:  return -0.5
    if v <= 7.5:  return -0.2
    return 0.0

def score_k_minus_bb(v: float, ip: float) -> float:
    """K-BB% — command + dominance composite."""
    if ip < MIN_IP: return 0.0
    if v >= 22.0: return 1.2
    if v >= 18.0: return 0.9
    if v >= 15.0: return 0.6
    if v >= 12.0: return 0.3
    if v <= 8.0:  return -0.5
    return 0.0

# ── Stuff & Arsenal ──────────────────────────────────────────────
def score_fastball_velo(v: float, ip: float) -> float:
    """Higher velo = more swing and miss."""
    if ip < MIN_IP: return 0.0
    if v >= 98.0: return 1.5
    if v >= 96.0: return 1.0
    if v >= 94.0: return 0.6
    if v >= 92.0: return 0.2
    if v <= 89.0: return -0.5
    if v <= 91.0: return -0.2
    return 0.0

def score_stuff_plus(v: float, ip: float) -> float:
    """Stuff+ (100 = league avg). Composite pitch quality."""
    if ip < MIN_IP or v == 0: return 0.0
    if v >= 125: return 1.5
    if v >= 115: return 1.0
    if v >= 108: return 0.5
    if v >= 100: return 0.1
    if v <= 85:  return -0.5
    if v <= 92:  return -0.2
    return 0.0

# ── Recent Form ───────────────────────────────────────────────────
def score_swstr_trend(v: float) -> float:
    """SwStr% delta: recent 21d vs season. Positive = heating up."""
    if v >= 3.0:  return 1.5
    if v >= 1.5:  return 1.0
    if v >= 0.5:  return 0.4
    if v <= -3.0: return -1.5
    if v <= -1.5: return -1.0
    if v <= -0.5: return -0.4
    return 0.0

def score_k_per_start_21d(v: float) -> float:
    """Avg Ks per start over last 3 starts."""
    if v <= 0: return 0.0
    if v >= 9.0: return 2.0
    if v >= 7.5: return 1.5
    if v >= 6.5: return 1.0
    if v >= 5.5: return 0.5
    if v >= 4.5: return 0.1
    if v <= 3.0: return -1.0
    if v <= 4.0: return -0.5
    return 0.0

def score_velo_trend(v: float) -> float:
    """Recent velo vs season avg. Positive = gaining velocity."""
    if v >= 1.5:  return 0.8
    if v >= 0.5:  return 0.3
    if v <= -2.0: return -1.0
    if v <= -1.0: return -0.5
    return 0.0

# ── Innings / Workload ────────────────────────────────────────────
def score_avg_ip_per_start(v: float, gs: int) -> float:
    """
    Avg innings per start = K ceiling proxy.
    Opener/bulk risk (<4 IP) gets penalized.
    """
    if gs < MIN_GS: return 0.0
    if v >= 6.5: return 1.5
    if v >= 6.0: return 1.0
    if v >= 5.5: return 0.5
    if v >= 5.0: return 0.1
    if v < AVG_IP_OPENER_THRESHOLD: return -2.0  # opener risk hard penalty
    return 0.0

# ── Opponent Batter Profile ───────────────────────────────────────
def score_opp_team_k_pct(v: float) -> float:
    """Opposing team's strikeout rate — higher = more Ks for pitcher."""
    if v >= 28.0: return 2.0
    if v >= 26.0: return 1.5
    if v >= 24.0: return 1.0
    if v >= 22.0: return 0.5
    if v >= 20.0: return 0.1
    if v <= 16.0: return -1.5
    if v <= 18.0: return -0.8
    return 0.0

def score_opp_team_bb_pct(v: float) -> float:
    """
    Opposing team BB rate — patient lineups work counts,
    reduce 2-strike opportunities.
    """
    if v >= 12.0: return -0.8
    if v >= 10.0: return -0.4
    if v <= 7.0:  return 0.4
    if v <= 8.0:  return 0.2
    return 0.0

# ── Park & Environment ────────────────────────────────────────────
def score_park_k_factor(v: float) -> float:
    """Park K factor (100 = neutral). Altitude/foul territory affect Ks."""
    if v >= 108: return 0.5
    if v >= 104: return 0.3
    if v >= 100: return 0.0
    if v <= 90:  return -0.5   # Coors-type suppressor
    if v <= 95:  return -0.2
    return 0.0

# ── Command ───────────────────────────────────────────────────────
def score_first_pitch_strike(v: float, ip: float) -> float:
    """F-Strike% — getting ahead = more 2-strike counts."""
    if ip < MIN_IP: return 0.0
    if v >= 68.0: return 0.8
    if v >= 65.0: return 0.5
    if v >= 62.0: return 0.2
    if v <= 57.0: return -0.5
    if v <= 59.0: return -0.2
    return 0.0

def score_bb_pct_pitcher(v: float, ip: float) -> float:
    """Pitcher BB% — walks = fewer 2-strike counts."""
    if ip < MIN_IP: return 0.0
    if v <= 5.0:  return 0.8
    if v <= 7.0:  return 0.4
    if v <= 9.0:  return 0.0
    if v >= 12.0: return -0.8
    if v >= 10.0: return -0.4
    return 0.0

# ══════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════
def compute_ks_score(row: pd.Series) -> dict:
    ip = float(row.get("ip", 0))
    gs = int(row.get("games_started", 0))

    # ── Core K ability ────────────────────────────────────────────
    s_k_pct         = score_k_pct_season(row.get("k_pct_season", 0), ip)
    s_swstr         = score_swstr_pct(row.get("swstr_pct", 0), ip)
    s_chase         = score_chase_rate(row.get("chase_rate", 0), ip)
    s_k_per_9       = score_k_per_9(row.get("k_per_9", 0), ip)
    s_k_minus_bb    = score_k_minus_bb(row.get("k_minus_bb", 0), ip)

    # ── Stuff ─────────────────────────────────────────────────────
    s_velo          = score_fastball_velo(row.get("fastball_velo", 0), ip)
    s_stuff         = score_stuff_plus(row.get("stuff_plus", 0), ip)

    # ── Recent form ───────────────────────────────────────────────
    s_swstr_trend   = score_swstr_trend(row.get("swstr_trend", 0))
    s_k_21d         = score_k_per_start_21d(row.get("k_per_start_21d", 0))
    s_velo_trend    = score_velo_trend(row.get("velo_trend", 0))

    # ── Innings ───────────────────────────────────────────────────
    s_ip            = score_avg_ip_per_start(row.get("avg_ip_per_start", 0), gs)

    # ── Opponent ──────────────────────────────────────────────────
    s_opp_k         = score_opp_team_k_pct(row.get("opp_team_k_pct", LEAGUE_AVG_OPP_K_PCT))
    s_opp_bb        = score_opp_team_bb_pct(row.get("opp_team_bb_pct", 8.5))

    # ── Park & command ────────────────────────────────────────────
    s_park          = score_park_k_factor(row.get("park_k_factor", 100))
    s_fps           = score_first_pitch_strike(row.get("first_pitch_strike_pct", 0), ip)
    s_bb_pitcher    = score_bb_pct_pitcher(row.get("bb_pct_season", 0), ip)

    # ── Group totals ──────────────────────────────────────────────
    core_k_score    = s_k_pct + s_swstr + s_chase + s_k_per_9 + s_k_minus_bb
    stuff_score     = s_velo + s_stuff
    form_score      = s_swstr_trend + s_k_21d + s_velo_trend
    workload_score  = s_ip
    matchup_score   = s_opp_k + s_opp_bb
    env_score       = s_park + s_fps + s_bb_pitcher

    total = round(
        core_k_score + stuff_score + form_score +
        workload_score + matchup_score + env_score, 2
    )

    # ── Projected K range ─────────────────────────────────────────
    # Simple projection: avg_ip_per_start * (k_per_9 / 9)
    avg_ip = float(row.get("avg_ip_per_start", 5.0))
    k_per_9 = float(row.get("k_per_9", LEAGUE_AVG_K_PER_9))
    projected_k = round(avg_ip * (k_per_9 / 9), 1)

    # ── Confidence tier ───────────────────────────────────────────
    if total >= 16:
        confidence = "Elite"
    elif total >= 13:
        confidence = "High"
    elif total >= 10:
        confidence = "Medium"
    elif total >= 7:
        confidence = "Low"
    else:
        confidence = "Fade"

    # ── Prop signal — most K props are 4.5 or 5.5 ────────────────
    if projected_k >= 7.0 and total >= 13:
        prop_signal = "YES 6.5+"
    elif projected_k >= 6.0 and total >= 10:
        prop_signal = "YES 5.5"
    elif projected_k >= 5.0 and total >= 8:
        prop_signal = "YES 4.5"
    elif total <= 5:
        prop_signal = "NO / FADE"
    else:
        prop_signal = "LEAN"

    return {
        "score": total,
        "confidence": confidence,
        "prop_signal": prop_signal,
        "projected_k": projected_k,
        "core_k_score": round(core_k_score, 2),
        "stuff_score": round(stuff_score, 2),
        "form_score": round(form_score, 2),
        "workload_score": round(workload_score, 2),
        "matchup_score": round(matchup_score, 2),
        "env_score": round(env_score, 2),
        # component detail
        "s_k_pct": s_k_pct, "s_swstr": s_swstr, "s_chase": s_chase,
        "s_k_per_9": s_k_per_9, "s_velo": s_velo, "s_stuff": s_stuff,
        "s_swstr_trend": s_swstr_trend, "s_k_21d": s_k_21d,
        "s_ip": s_ip, "s_opp_k": s_opp_k, "s_park": s_park,
    }

# ══════════════════════════════════════════════════════════════════
# DIVERSITY CAP
# ══════════════════════════════════════════════════════════════════
def apply_diversity_cap(df: pd.DataFrame, top_n: int = TOP_N) -> pd.DataFrame:
    """
    One pitcher per team (natural limit — each team has one starter per day).
    Chalk cap: max MAX_CHALK_PICKS heavy favorites.
    """
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    selected = []
    team_counts = {}
    chalk_count = 0

    for _, row in df.iterrows():
        if len(selected) >= top_n:
            break
        team = row.get("team", "UNK")
        if team_counts.get(team, 0) >= MAX_PER_TEAM:
            continue
        # Chalk = very short odds (heavy favorite)
        try:
            odds_val = float(row.get("ks_odds", 999))
            is_chalk = odds_val <= -200
        except (ValueError, TypeError):
            is_chalk = False
        if is_chalk and chalk_count >= MAX_CHALK_PICKS:
            continue
        selected.append(row)
        team_counts[team] = team_counts.get(team, 0) + 1
        if is_chalk:
            chalk_count += 1

    return pd.DataFrame(selected).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════
def load_data(wb) -> pd.DataFrame:
    print("Loading KS_Statcast data...")
    try:
        ws = wb.worksheet("KS_Statcast")
        data = ws.get_all_records()
        df = pd.DataFrame(data)
        print(f"  Loaded {len(df)} pitchers from KS_Statcast")
    except Exception as e:
        print(f"  ERROR loading KS_Statcast: {e}")
        return pd.DataFrame()

    # Load opposing team K rates from main batter data
    try:
        bws = wb.worksheet("HRRBI_Statcast")
        bdata = bws.get_all_records()
        batters = pd.DataFrame(bdata)
        if not batters.empty and "team" in batters.columns and "k_pct" in batters.columns:
            team_k = batters.groupby("team")["k_pct"].mean().reset_index()
            team_k.columns = ["opp_team", "opp_team_k_pct"]
            print(f"  Loaded team K rates for {len(team_k)} teams")
        else:
            team_k = pd.DataFrame()
    except Exception:
        team_k = pd.DataFrame()
        print("  INFO: Team K rates not loaded — using league average")

    # Load park K factors
    try:
        pkws = wb.worksheet("Park_Factors")
        pkdata = pkws.get_all_records()
        parks = pd.DataFrame(pkdata)
        if "k_factor" in parks.columns:
            parks = parks[["team", "k_factor"]].rename(columns={"k_factor": "park_k_factor"})
        print(f"  Loaded park K factors")
    except Exception:
        parks = pd.DataFrame()
        print("  INFO: Park K factors not loaded — using neutral 100")

    # Add opponent/park columns if lineup data not available
    # These will default to league averages — enriched when main.py provides today's matchups
    for col, default in [
        ("opp_team_k_pct", LEAGUE_AVG_OPP_K_PCT),
        ("opp_team_bb_pct", 8.5),
        ("park_k_factor", 100),
        ("ks_odds", ""),
        ("days_rest", 4),
    ]:
        if col not in df.columns:
            df[col] = default

    # Numeric coerce
    numeric_cols = [
        "ip", "games", "games_started", "k_per_9", "k_pct_season",
        "bb_pct_season", "k_minus_bb", "whip", "era", "fip", "xfip",
        "swstr_pct", "chase_rate", "zone_contact_pct", "first_pitch_strike_pct",
        "barrel_pct_against", "hard_hit_pct_against", "gb_pct", "fb_pct",
        "fastball_velo", "stuff_plus", "avg_ip_per_start", "opener_risk",
        "k_last_3", "k_per_start_21d", "swstr_pct_21d", "avg_velo_21d",
        "swstr_trend", "velo_trend", "projected_k_ceiling",
        "opp_team_k_pct", "opp_team_bb_pct", "park_k_factor", "days_rest",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Filter: must be a starter with meaningful sample
    df = df[df["games_started"] >= MIN_GS] if "games_started" in df.columns else df
    df = df[df["ip"] >= MIN_IP] if "ip" in df.columns else df

    # Exclude openers (hard filter — opener_risk flag)
    df = df[df.get("opener_risk", pd.Series(0)) == 0] if "opener_risk" in df.columns else df

    return df.reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════
# SHEET OUTPUT
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
    print("Writing Top_KS_Picks sheet...")
    try:
        ws = wb.worksheet("Top_KS_Picks")
        ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        ws = wb.add_worksheet(title="Top_KS_Picks", rows=100, cols=30)

    output_cols = [
        "name", "team", "ip", "avg_ip_per_start",
        "k_pct_season", "swstr_pct", "chase_rate", "k_per_9",
        "fastball_velo", "stuff_plus",
        "k_per_start_21d", "swstr_trend", "velo_trend",
        "opp_team_k_pct", "park_k_factor",
        "projected_k", "score", "confidence", "prop_signal",
        "ks_odds", "bet_placed", "result",
    ]

    for col in output_cols:
        if col not in picks.columns:
            picks[col] = ""

    rows_to_write = []
    rows_to_write.append(["⚾ TOP PITCHER STRIKEOUT PICKS"] + [""] * (len(output_cols) - 1))
    rows_to_write.append(output_cols)
    for _, row in picks.iterrows():
        rows_to_write.append([str(row.get(c, "")) for c in output_cols])

    rows_to_write.append([""] * len(output_cols))

    # EV subsection: high SwStr% but not in top picks (may be undervalued)
    rows_to_write.append(["📊 K EV WATCH — High SwStr%, Not in Top Picks"] + [""] * (len(output_cols) - 1))
    rows_to_write.append(output_cols)
    for _, row in ev_picks.iterrows():
        rows_to_write.append([str(row.get(c, "")) for c in output_cols])

    ws.update(rows_to_write)

    # Formatting
    for title_row in [1, len(picks) + 4]:
        ws.format(f"A{title_row}", {
            "backgroundColor": {"red": 0.13, "green": 0.55, "blue": 0.34},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                           "fontSize": 12},
        })

    for header_row in [2, len(picks) + 5]:
        ws.format(f"A{header_row}:Z{header_row}", {
            "backgroundColor": {"red": 0.20, "green": 0.20, "blue": 0.20},
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })

    conf_col_idx = output_cols.index("confidence") + 1
    conf_letter = chr(64 + conf_col_idx)
    for i, (_, row) in enumerate(picks.iterrows()):
        sheet_row = i + 3
        color = confidence_color(str(row.get("confidence", "")))
        ws.format(f"{conf_letter}{sheet_row}", {
            "backgroundColor": color,
            "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
        })

    print(f"  ✓ Top_KS_Picks written: {len(picks)} picks + {len(ev_picks)} EV")

# ══════════════════════════════════════════════════════════════════
# SCORECARD
# ══════════════════════════════════════════════════════════════════
def update_scorecard(wb):
    print("Updating KS_Scorecard...")
    try:
        ws = wb.worksheet("Top_KS_Picks")
        data = ws.get_all_records()
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"  WARNING: could not load picks for scorecard: {e}")
        return

    if df.empty or "result" not in df.columns:
        print("  INFO: No result data yet for scorecard")
        return

    df = df[df["result"].isin(["HIT", "MISS", "1", "0"])].copy()
    if df.empty:
        return

    df["hit"] = df["result"].isin(["HIT", "1"]).astype(int)
    df["score_num"] = pd.to_numeric(df.get("score", 0), errors="coerce").fillna(0)

    tiers = [
        ("16+",    df["score_num"] >= 16),
        ("13+",    df["score_num"] >= 13),
        ("10+",    df["score_num"] >= 10),
        ("7+",     df["score_num"] >= 7),
        ("Under 7", df["score_num"] < 7),
    ]

    try:
        sc_ws = wb.worksheet("KS_Scorecard")
        sc_ws.clear()
    except gspread.exceptions.WorksheetNotFound:
        sc_ws = wb.add_worksheet(title="KS_Scorecard", rows=50, cols=10)

    headers = ["Score Tier", "Picks", "Hits", "Hit Rate", "Notes"]
    sc_rows = [["⚾ PITCHER K MODEL PERFORMANCE"] + [""] * 4, headers]

    for label, mask in tiers:
        subset = df[mask]
        picks = len(subset)
        hits = subset["hit"].sum()
        rate = f"{hits/picks*100:.1f}%" if picks > 0 else "—"
        note = "✅ Strong" if picks > 0 and hits/picks >= 0.50 else (
               "⚠️ Weak" if picks > 0 and hits/picks < 0.30 else "")
        sc_rows.append([label, picks, hits, rate, note])

    sc_ws.update(sc_rows)
    sc_ws.format("A1:E1", {
        "backgroundColor": {"red": 0.13, "green": 0.55, "blue": 0.34},
        "textFormat": {"bold": True, "foregroundColor": {"red": 1, "green": 1, "blue": 1}},
    })
    print("  ✓ KS_Scorecard updated")

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
        print("ERROR: No pitcher data loaded — aborting")
        return

    print(f"\nScoring {len(df)} pitchers...")
    score_results = df.apply(compute_ks_score, axis=1)
    score_df = pd.DataFrame(score_results.tolist())
    df = pd.concat([df.reset_index(drop=True), score_df], axis=1)

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    top_picks = apply_diversity_cap(df, TOP_N)
    top_picks["section"] = "Main"

    # EV: high SwStr% pitchers not in top picks
    top_names = set(top_picks.get("name", pd.Series()).tolist())
    ev_candidates = df[~df.get("name", pd.Series()).isin(top_names)].copy()
    ev_picks = ev_candidates[
        (ev_candidates.get("swstr_pct", pd.Series(0)) >= 13.0) &
        (ev_candidates.get("k_pct_season", pd.Series(0)) >= 24.0) &
        (ev_candidates.get("avg_ip_per_start", pd.Series(0)) >= 5.0)
    ].head(5).copy()
    ev_picks["section"] = "EV"

    print(f"\n{'='*55}")
    print(f"Top {len(top_picks)} Pitcher K Picks:")
    for _, row in top_picks.iterrows():
        print(f"  {row.get('name','?'):22s}  score={row.get('score',0):5.2f}  "
              f"proj_k={row.get('projected_k',0):4.1f}  "
              f"{row.get('confidence','?'):8s}  {row.get('prop_signal','?')}")
    print(f"{'='*55}\n")

    write_picks(wb, top_picks, ev_picks)

    ws = wb.worksheet("Top_KS_Picks")
    write_timestamp(ws)

    update_scorecard(wb)
    print("✓ KS model complete")

if __name__ == "__main__":
    main()

