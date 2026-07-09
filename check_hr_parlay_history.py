"""
check_hr_parlay_history.py
Diagnostic: simulates the 3-leg HR parlay picker against historical data.

RECONCILED to the live dashboard.py parlay build (2026-07-09):
Pool = confirmed value zones narrowed to the +301-400 SWEET SPOT
(test_odds_sweetspot.py); 13+|≤+300 kept for leg quality:
  - 9-10  | +301-400
  - 12-13 | +301-400
  - 13+   | +301-400
  - 13+   | ≤+300

Leg selector — BLEND-1 (dashboard.py:830, validated via test_parlay_blend.py):
  (hr_per_fb / 8.0) + (leg_edge * 0.8)
  where leg_edge = tier×zone hit rate − implied breakeven, in % points.

Picks top 3 by selector score, diversified by opposing pitcher.
Reports simulated parlay hit rate and what winning legs had in common.
"""

import os
import time
import json

import pandas as pd
import numpy as np
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

MODEL_START_DATE = "2026-06-09"
LEG_COUNT        = 3


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info     = json.loads(raw_json)
    creds    = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def with_retry(func, retries=4, wait=15):
    for attempt in range(retries):
        try:
            return func()
        except APIError as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(wait)
            else:
                raise
        except Exception:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (pd.isna(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def get_score_tier(score: float) -> str:
    if score >= 13:   return "13+"
    if score >= 12:   return "12-13"
    if score >= 11:   return "11-12"
    if score >= 10:   return "10-11"
    return "Under 10"


def get_odds_zone(odds: float) -> str:
    if odds <= 0:    return "No Odds"
    if odds <= 300:  return "<= +300"
    if odds <= 499:  return "+301 to +499"
    if odds <= 699:  return "+500 to +699"
    return "+700+"


def american_breakeven(odds: float) -> float:
    """Implied breakeven probability (%) for positive American odds."""
    if odds <= 0:
        return 100.0
    return 100.0 * 100.0 / (odds + 100.0)


def zone_key(score: float, odds: float):
    """Deployed value-zone key for a leg (dashboard.py:810), or None if out
    of pool. Zones narrowed to the +301-400 sweet spot; 13+|≤+300 kept."""
    if odds <= 0:
        return None
    if score >= 13.0 and odds <= 300:
        return "13+|<=300"
    if score >= 13.0 and 301 <= odds <= 400:
        return "13+|301-400"
    if 12.0 <= score < 13.0 and 301 <= odds <= 400:
        return "12-13|301-400"
    if 9.0 <= score < 10.0 and 301 <= odds <= 400:
        return "9-10|301-400"
    return None


def in_parlay_pool(score: float, odds: float) -> bool:
    """Deployed parlay pool = the four value zones above (dashboard.py:810)."""
    return zone_key(score, odds) is not None


def compute_selector(row, zone_rates) -> float:
    """
    BLEND-1 selector — mirrors the live dashboard.py:830 parlay selector:
        (hr_per_fb / 8.0) + (leg_edge * 0.8)
    leg_edge = (tier×zone hit rate − implied breakeven), in % points, using the
    same value-zone hit-rate table the dashboard looks up via calc_hr_value.
    Unknown zone → leg_edge = -99 so it sinks to the bottom, as in dashboard.
    """
    hr_per_fb = safe_float(row.get("hr_per_fb", 0))
    score     = safe_float(row.get("hr_score", 0))
    odds      = safe_float(row.get("odds_num", 0))

    key = zone_key(score, odds)
    if key is None or key not in zone_rates:
        leg_edge = -99.0
    else:
        leg_edge = zone_rates[key] - american_breakeven(odds)

    return (hr_per_fb / 8.0) + (leg_edge * 0.8)


def simulate_parlay(day_df: pd.DataFrame, zone_rates: dict) -> list:
    """
    Simulate the parlay picker for one day.
    Returns list of selected candidate dicts (up to 3).
    """
    pool = day_df[day_df.apply(
        lambda r: in_parlay_pool(r["hr_score"], r["odds_num"]), axis=1
    )].copy()

    if pool.empty:
        return []

    pool["selector"] = pool.apply(lambda r: compute_selector(r, zone_rates), axis=1)
    pool = pool.sort_values("selector", ascending=False).reset_index(drop=True)

    selected   = []
    used_games = set()

    for _, row in pool.iterrows():
        if len(selected) >= LEG_COUNT:
            break
        opp_pit = str(row.get("pitcher_name", "")).strip()
        if opp_pit and opp_pit in used_games:
            continue
        selected.append(row)
        if opp_pit:
            used_games.add(opp_pit)

    # Fill remaining if diversification left us short
    if len(selected) < LEG_COUNT:
        for _, row in pool.iterrows():
            if len(selected) >= LEG_COUNT:
                break
            already = any(
                s.get("player_name") == row.get("player_name")
                for s in selected
            )
            if not already:
                selected.append(row)

    return selected


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    ws = sh.worksheet("HR_All_Scores")
    all_values = with_retry(lambda: ws.get_all_values())

    headers = all_values[0]
    rows    = all_values[1:]
    df      = pd.DataFrame(rows, columns=headers)

    # Use corrected columns (barrel-handedness fix) when present, falling back
    # to originals. Both hr_score and platoon_score changed with the fix, and
    # the parlay selector leans heavily on platoon — so the winning-leg
    # analysis must run on corrected values to be trustworthy.
    def _coalesce(row, corrected_col, orig_col):
        c = str(row.get(corrected_col, "")).strip()
        if c not in ("", "nan", "None"):
            return safe_float(c)
        return safe_float(row.get(orig_col))

    if "hr_score_corrected" in df.columns:
        df["hr_score"] = df.apply(lambda r: _coalesce(r, "hr_score_corrected", "hr_score"), axis=1)
    else:
        df["hr_score"] = df["hr_score"].apply(safe_float)

    df["date"]     = df["date"].astype(str).str.strip()
    df["odds_num"] = df["consensus_odds"].apply(lambda x: safe_float(x, 0))
    df["date_dt"]  = pd.to_datetime(df["date"], errors="coerce")

    # Filter to clean model data only
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)].copy()
    print(f"Filtered to {MODEL_START_DATE}+: {len(df)} rows")

    # Overlay corrected platoon before coercing the feature columns
    if "platoon_score_corrected" in df.columns:
        df["platoon_score"] = df.apply(lambda r: _coalesce(r, "platoon_score_corrected", "platoon_score"), axis=1)

    for col in ["platoon_score", "season_barrel_pct", "hr_per_fb", "hr_weather_boost",
                "barrel_pct_7d", "barrel_pct_5d", "pitch_matchup_score", "momentum_score"]:
        if col in df.columns:
            df[col] = df[col].apply(safe_float)
        else:
            df[col] = 0.0

    # Only resolved rows
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"

    dates = sorted(resolved["date"].unique())
    print(f"Analyzing {len(dates)} days\n")

    # ── Value-zone hit rates (deployed 4 zones) ──────────────────────────
    # These feed the BLEND-1 edge term, mirroring the dashboard's hr_hit_rates
    # lookup. NOTE: computed in-sample for this diagnostic; the live dashboard
    # uses its attached (trailing) table, so live selector edges can differ
    # slightly. Ranking within a day is the same idea either way.
    zone_defs = [
        ("13+|<=300",     "13+   | ≤+300",    (resolved["hr_score"] >= 13) & (resolved["odds_num"] > 0)  & (resolved["odds_num"] <= 300)),
        ("13+|301-400",   "13+   | +301-400", (resolved["hr_score"] >= 13) & (resolved["odds_num"] >= 301) & (resolved["odds_num"] <= 400)),
        ("12-13|301-400", "12-13 | +301-400", (resolved["hr_score"] >= 12) & (resolved["hr_score"] < 13) & (resolved["odds_num"] >= 301) & (resolved["odds_num"] <= 400)),
        ("9-10|301-400",  "9-10  | +301-400", (resolved["hr_score"] >= 9)  & (resolved["hr_score"] < 10) & (resolved["odds_num"] >= 301) & (resolved["odds_num"] <= 400)),
    ]
    zone_rates = {}
    print("=" * 60)
    print("POOL STATS — Value Zone Hit Rates (deployed +301-400 sweet spot)")
    print("=" * 60)
    for key, label, mask in zone_defs:
        sub = resolved[mask]
        n   = len(sub)
        rate = round(sub["hit_bool"].mean() * 100, 1) if n else 0.0
        zone_rates[key] = rate
        h = int(sub["hit_bool"].sum()) if n else 0
        print(f"  {label:20s}: {h}/{n} = {rate}%  (breakeven-edge feeds selector)")
    print()

    # ── Simulate parlay day by day ────────────────────────────────────────
    days_with_parlay      = 0
    days_parlay_hit       = 0
    days_all_3_hit        = 0
    days_no_pool          = 0
    total_legs            = 0
    total_legs_hit        = 0
    winning_leg_data      = []

    print("=" * 60)
    print("SIMULATED PARLAY — Day by Day")
    print("=" * 60)

    for d in dates:
        day_df = resolved[resolved["date"] == d].copy()
        if day_df.empty:
            continue

        selected = simulate_parlay(day_df, zone_rates)

        if not selected:
            days_no_pool += 1
            continue

        days_with_parlay += 1
        legs_hit = sum(1 for s in selected if safe_float(s.get("hit_bool", 0)) == 1.0 or str(s.get("hit_bool", "")) == "True")

        total_legs     += len(selected)
        total_legs_hit += legs_hit

        all_hit = legs_hit == len(selected) and len(selected) == LEG_COUNT

        if all_hit:
            days_parlay_hit += 1
            days_all_3_hit  += 1

        status = "✅ HIT" if all_hit else f"❌ {legs_hit}/{len(selected)}"
        print(f"\n{d} — {status}")
        for s in selected:
            hit_str  = "✅" if (safe_float(s.get("hit_bool", 0)) == 1.0 or str(s.get("hit_bool","")) == "True") else "❌"
            tier     = get_score_tier(s["hr_score"])
            zone     = get_odds_zone(s["odds_num"])
            selector = safe_float(s.get("selector", 0))
            print(f"  {hit_str} {str(s.get('player_name',''))[:22]:22s} score={s['hr_score']:.1f} tier={tier:6s} odds=+{int(s['odds_num'])} zone={zone} selector={selector:.3f}")
            if all_hit or safe_float(s.get("hit_bool", 0)) == 1.0 or str(s.get("hit_bool","")) == "True":
                winning_leg_data.append(s)

    print()
    print("=" * 60)
    print("PARLAY SIMULATION RESULTS")
    print("=" * 60)
    print(f"Days analyzed:          {len(dates)}")
    print(f"Days with parlay pool:  {days_with_parlay}")
    print(f"Days no pool available: {days_no_pool}")
    print(f"")
    print(f"All-3-hit days:         {days_all_3_hit} / {days_with_parlay} = {round(days_all_3_hit/days_with_parlay*100,1) if days_with_parlay else 0}%")
    print(f"Individual leg hit rate:{total_legs_hit} / {total_legs} = {round(total_legs_hit/total_legs*100,1) if total_legs else 0}%")
    print()

    if not winning_leg_data:
        print("No winning legs yet.")
        return

    wdf = pd.DataFrame(winning_leg_data)

    print("=" * 60)
    print("WINNING LEG ANALYSIS")
    print("=" * 60)

    print("\nScore tier distribution:")
    wdf["tier"] = wdf["hr_score"].apply(get_score_tier)
    for tier, n in wdf["tier"].value_counts().items():
        print(f"  {tier:10s}: {n}")

    print("\nOdds zone distribution:")
    wdf["odds_zone"] = wdf["odds_num"].apply(get_odds_zone)
    for zone, n in wdf["odds_zone"].value_counts().items():
        print(f"  {zone:14s}: {n}")

    print("\nFeature averages — winning legs vs full pool:")
    pool_df = resolved[resolved.apply(lambda r: in_parlay_pool(r["hr_score"], r["odds_num"]), axis=1)]
    for col, label in [
        ("platoon_score",     "Platoon Score"),
        ("season_barrel_pct", "Season Barrel%"),
        ("hr_per_fb",         "HR/FB%"),
        ("hr_weather_boost",  "Weather Boost"),
        ("barrel_pct_7d",     "Barrel% 7d"),
        ("pitch_matchup_score","Pitch Matchup"),
        ("momentum_score",    "Momentum"),
    ]:
        if col not in wdf.columns or col not in pool_df.columns:
            continue
        w_avg = wdf[col].mean()
        p_avg = pool_df[col].mean()
        diff  = w_avg - p_avg
        print(f"  {label:20s}: winners={w_avg:7.3f}  pool={p_avg:7.3f}  diff={diff:+.3f}")


if __name__ == "__main__":
    main()
