"""
dashboard.py — HR-only build of "Today's Top Picks".

This is a clean rewrite:
  • Timestamp is now the FIRST row, written and formatted in a single pass
    (previously it was inserted after formatting, which shifted every row
    down by one and corrupted the first data row + header).
  • Every run starts with a full unmerge + format reset so leftover cells
    from the old KS/HRRBI layout can't bleed into the new one.
  • KS and HR+RBI sections are gone (not being bet right now).
  • Score tier lookup splits 13+ into 13-14 / 14-15 / 15+ so each leg is
    credited with its own observed hit rate (15+ underperforms).
  • "Why" shows baseball reasons only — power, barrel, form, platoon,
    park, weather. It no longer repeats the Edge column.
  • Both parlays show individual leg odds AND the combined parlay price.
"""
import os
import json
import time
from datetime import datetime, date as _date
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
COLOR_GREEN     = {"red": 0.180, "green": 0.800, "blue": 0.443}
COLOR_PURPLE    = {"red": 0.541, "green": 0.165, "blue": 0.557}
COLOR_HEADER_BG = {"red": 0.055, "green": 0.055, "blue": 0.055}
COLOR_SUBTEXT   = {"red": 0.600, "green": 0.600, "blue": 0.600}
COLOR_BLACK     = {"red": 0.050, "green": 0.050, "blue": 0.050}

N_COLS = 8
RESET_ROWS = 400
RESET_COLS = 26


# ── Sheets plumbing ─────────────────────────────────────────────────────────
def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
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
        except Exception:
            if attempt < retries - 1:
                time.sleep(wait)
            else:
                raise


def read_sheet(gc, sheet_id, name) -> pd.DataFrame:
    try:
        sh = with_retry(lambda: gc.open_by_key(sheet_id))
        ws = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values:
            return pd.DataFrame()
        start = 1 if "Last Run" in str(all_values[0]) else 0
        if start >= len(all_values):
            return pd.DataFrame()
        headers = all_values[start]
        rows = all_values[start + 1:]
        df = pd.DataFrame(rows, columns=headers)
        return df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    except gspread.WorksheetNotFound:
        print(f"WARNING: Sheet '{name}' not found.")
        return pd.DataFrame()
    except Exception as e:
        print(f"WARNING: Could not read sheet '{name}': {e}")
        return pd.DataFrame()


def read_sheet_raw(gc, sheet_id, name) -> pd.DataFrame:
    try:
        sh = with_retry(lambda: gc.open_by_key(sheet_id))
        ws = sh.worksheet(name)
        all_values = with_retry(lambda: ws.get_all_values())
        if not all_values or len(all_values) < 2:
            return pd.DataFrame()
        df = pd.DataFrame(all_values[1:], columns=all_values[0])
        return df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    except Exception:
        return pd.DataFrame()


def safe_float(val, default=0.0) -> float:
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return default


def american_to_implied(odds: float) -> float:
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def implied_to_american(prob: float) -> str:
    if prob <= 0 or prob >= 1:
        return "—"
    if prob >= 0.5:
        return str(int(-round((prob / (1 - prob)) * 100)))
    return f"+{int(round(((1 - prob) / prob) * 100))}"


def combined_american(odds_list) -> str:
    """Multiply decimal odds of the legs, return the parlay price in American."""
    dec = 1.0
    for o in odds_list:
        dec *= (1 + o / 100.0)  # legs are positive American here
    return f"+{int(round((dec - 1) * 100))}"


# ── HR hit-rate lookup ──────────────────────────────────────────────────────
def _hr_odds_zone_key(odds: float) -> str:
    if odds <= 300:
        return "le300"
    if 301 <= odds <= 499:
        return "301-499"
    if 500 <= odds <= 699:
        return "500-699"
    return "700plus"


def build_hr_hit_rates(hr_all_scores: pd.DataFrame) -> dict:
    hit_rates = {}
    if hr_all_scores.empty:
        return hit_rates
    MODEL_START_DATE = "2026-06-09"
    df = hr_all_scores.copy()
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date_dt"] >= pd.Timestamp(MODEL_START_DATE)]
    resolved = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if resolved.empty:
        return hit_rates

    resolved["hr_score"] = resolved.apply(
        lambda r: safe_float(r.get("hr_score_corrected"))
        if str(r.get("hr_score_corrected", "")).strip() not in ("", "nan", "None")
        else safe_float(r.get("hr_score")),
        axis=1,
    )
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"
    resolved["odds_num"] = resolved["consensus_odds"].apply(safe_float)

    # 13+ split into 13-14 / 14-15 / 15+ so each leg is credited with its own
    # observed rate (15+ underperforms the tiers beneath it).
    tier_defs = [
        ("15+", 15, 999), ("14-15", 14, 15), ("13-14", 13, 14),
        ("12-13", 12, 13), ("11-12", 11, 12),
        ("10-11", 10, 11), ("9-10", 9, 10), ("8.5-9", 8.5, 9),
    ]
    zone_keys = ["le300", "301-499", "500-699", "700plus"]
    for tier_label, lo, hi in tier_defs:
        tier_sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(tier_sub) >= 5:
            hit_rates[tier_label] = tier_sub["hit_bool"].mean()
        for zk in zone_keys:
            zsub = tier_sub[tier_sub["odds_num"].apply(_hr_odds_zone_key) == zk]
            if len(zsub) >= 15:
                hit_rates[(tier_label, zk)] = zsub["hit_bool"].mean()
    print(f"  HR hit rate lookup: {len(hit_rates)} entries from {len(resolved)} resolved picks")
    return hit_rates


def get_hr_score_tier(score: float) -> str:
    if score >= 15:  return "15+"
    if score >= 14:  return "14-15"
    if score >= 13:  return "13-14"
    if score >= 12:  return "12-13"
    if score >= 11:  return "11-12"
    if score >= 10:  return "10-11"
    if score >= 9:   return "9-10"
    if score >= 8.5: return "8.5-9"
    return "below-8.5"


def calc_hr_value(score: float, odds: float, hit_rates: dict) -> tuple:
    tier = get_hr_score_tier(score)
    zk = _hr_odds_zone_key(odds)
    hit_rate = hit_rates.get((tier, zk))
    if hit_rate is None or odds <= 0:
        return 0.0, "—", False, "No zone data"
    implied = american_to_implied(odds)
    edge = hit_rate - implied
    edge_pct = round(edge * 100, 1)
    edge_str = f"+{edge_pct}%" if edge_pct >= 0 else f"{edge_pct}%"
    return round(hit_rate * 100, 1), implied_to_american(hit_rate), edge > 0, edge_str


# ── Why — baseball reasons only ─────────────────────────────────────────────
def build_why(row) -> str:
    """
    Baseball reasons a pick stands out — power, barrel, form, platoon, park,
    weather. Deliberately does NOT include zone edge (that's the Edge column).
    """
    bits = []

    hr_fb = safe_float(row.get("hr_per_fb", 0))
    if hr_fb >= 18:
        bits.append(f"elite power ({hr_fb:.0f}% HR/FB)")
    elif hr_fb >= 13:
        bits.append(f"strong power ({hr_fb:.0f}% HR/FB)")
    elif hr_fb >= 9:
        bits.append(f"avg power ({hr_fb:.0f}% HR/FB)")

    sb = safe_float(row.get("season_barrel_pct", 0))
    if sb >= 12:
        bits.append(f"elite barrel ({sb:.0f}%)")
    elif sb >= 9:
        bits.append(f"good barrel ({sb:.0f}%)")

    b7 = safe_float(row.get("barrel_pct_7d", 0))
    if b7 >= 18:
        bits.append(f"hot barrel 7d ({b7:.0f}%)")

    mom = str(row.get("momentum_desc", "")).lower()
    if "surging" in mom:
        bits.append("surging form")
    elif "trending up" in mom:
        bits.append("heating up")
    elif "cold" in mom or "trending down" in mom:
        bits.append("cold streak")

    plat = safe_float(row.get("platoon_score", 0))
    if plat >= 1.0:
        bits.append("platoon edge")
    elif plat <= -1.0:
        bits.append("platoon down")

    pf = safe_float(row.get("park_hr_factor", 100))
    if pf >= 110:
        bits.append(f"HR park ({pf:.0f})")
    elif 0 < pf <= 90:
        bits.append(f"tough park ({pf:.0f})")

    wb = safe_float(row.get("hr_weather_boost", 0))
    if wb >= 1.5:
        bits.append("wind out")
    elif wb <= -1.0:
        bits.append("wind in")

    if not bits:
        return f"avg power ({hr_fb:.0f}% HR/FB)" if hr_fb > 0 else "value on odds only"
    return "   ·   ".join(bits[:4])


# ── Row builder (HR only) ───────────────────────────────────────────────────
def build_rows(hr_df, hr_hit_rates, hr_today, timestamp_str):
    def pad(row):
        return list(row) + [""] * (N_COLS - len(row))

    E = pad([])
    rows = []

    rows.append((pad([f"⏱  Last Updated: {timestamp_str}"]), "timestamp"))

    hr_source = hr_today if (hr_today is not None and not hr_today.empty) else hr_df

    # ── HR EDGE PLAYS (singles) ──────────────────────────────────────────
    rows.append((pad(["🏠  HOME RUN EDGE PLAYS — beats breakeven vs resolved hit rates"]),
                 "section_header_hr"))
    rows.append((pad(["Rank", "Batter", "Team", "Score", "Odds", "Hit%", "Edge", "Why"]),
                 "col_header_hr"))

    if hr_source.empty or not hr_hit_rates:
        rows.append((pad(["—", "No value plays today", ""]), "no_plays"))
    else:
        plays = []
        for _, row in hr_source.iterrows():
            try:
                batter = str(row.get("player_name", "")).strip()
                if not batter or batter == "nan":
                    continue
                team = str(row.get("team", "")).strip()
                hr_score = safe_float(row.get("hr_score", 0))
                odds_raw = str(row.get("consensus_odds", "")).strip()
                odds_val = safe_float(odds_raw.replace("+", "")) if odds_raw not in ("", "nan") else 0
                if odds_val <= 0 or hr_score <= 0:
                    continue

                hit_rate, _be, has_value, edge_str = calc_hr_value(hr_score, odds_val, hr_hit_rates)
                try:
                    edge_num = float(edge_str.replace("%", "").replace("+", ""))
                except (ValueError, AttributeError):
                    edge_num = -99.0
                in_band = 251 <= odds_val <= 600
                if in_band and edge_num >= 4.0:
                    strength = "🔥 STRONG"
                elif in_band and edge_num >= 2.0:
                    strength = "✓ ok"
                elif not in_band:
                    strength = "⚠ odds"
                else:
                    strength = "· thin"

                plays.append({
                    "batter": batter, "team": team, "score": round(hr_score, 1),
                    "odds": f"+{int(odds_val)}", "hit_rate": hit_rate,
                    "edge": edge_str, "edge_num": edge_num, "strength": strength,
                    "has_value": has_value, "why": build_why(row),
                })
            except Exception:
                continue

        plays = [p for p in plays if p["has_value"] and p["score"] >= 9.0]
        rank_order = {"🔥 STRONG": 0, "✓ ok": 1, "· thin": 2, "⚠ odds": 3}
        plays.sort(key=lambda x: (rank_order.get(x["strength"], 9), -x["edge_num"]))

        if not plays:
            rows.append((pad(["—", "No edge plays today — nothing beats breakeven", ""]), "no_plays"))
        else:
            for i, p in enumerate(plays):
                rtype = ("data_hr_strong" if p["score"] >= 13
                         else "data_hr_moderate" if p["score"] >= 12
                         else "data_hr_light")
                rows.append((pad([
                    str(i + 1), p["batter"], p["team"], f"{p['score']:.1f}",
                    p["odds"], f"{p['hit_rate']:.0f}%",
                    f"{p['edge']} {p['strength']}", p["why"],
                ]), rtype))

    rows.append((E[:], "spacer"))

    # ── Shared parlay candidate pool ─────────────────────────────────────
    parlay_candidates = []
    if not hr_source.empty:
        for _, row in hr_source.iterrows():
            try:
                batter = str(row.get("player_name", "")).strip()
                if not batter or batter == "nan":
                    continue
                hr_score = safe_float(row.get("hr_score", 0))
                odds_raw = str(row.get("consensus_odds", "")).strip()
                odds_val = safe_float(odds_raw.replace("+", "")) if odds_raw not in ("", "nan") else 0
                if odds_val <= 0 or hr_score <= 0:
                    continue

                in_2leg = (301 <= odds_val <= 400) or (odds_val <= 300 and hr_score >= 13)
                in_3leg = (351 <= odds_val <= 500) or (odds_val <= 300 and hr_score >= 13)
                score_ok = (9.0 <= hr_score < 10.0) or (12.0 <= hr_score < 13.0) or (hr_score >= 13.0)
                if not (score_ok and (in_2leg or in_3leg)):
                    continue

                hr_per_fb = safe_float(row.get("hr_per_fb", 0))
                hit, _be, _hv, leg_edge_str = calc_hr_value(hr_score, odds_val, hr_hit_rates)
                try:
                    leg_edge = float(leg_edge_str.replace("%", "").replace("+", ""))
                except (ValueError, AttributeError):
                    leg_edge = -99.0
                selector = (hr_per_fb / 8.0) + (leg_edge * 0.8)

                parlay_candidates.append({
                    "batter": batter,
                    "team": str(row.get("team", "")).strip(),
                    "opp_pit": str(row.get("pitcher_name", "")).strip(),
                    "score": hr_score, "odds": odds_val, "selector": selector,
                    "hit": hit, "in_2leg": in_2leg, "in_3leg": in_3leg,
                    "why": build_why(row),
                })
            except Exception:
                continue
        parlay_candidates.sort(key=lambda x: -x["selector"])

    # ── 3-LEG PARLAY ─────────────────────────────────────────────────────
    rows.append((pad(["🎰  3-LEG HR PARLAY — Jackpot band (+351-500), best legs"]),
                 "section_header_parlay"))
    rows.append((pad(["Leg", "Batter", "Team", "Score", "Odds", "Hit%", "Payout", "Why"]),
                 "col_header_parlay"))

    three_pool = [c for c in parlay_candidates if c.get("in_3leg")]
    selected, used_games = [], set()
    for c in three_pool:
        if len(selected) >= 3:
            break
        if c["opp_pit"] and c["opp_pit"] in used_games:
            continue
        selected.append(c)
        if c["opp_pit"]:
            used_games.add(c["opp_pit"])
    if len(selected) < 3:
        for c in three_pool:
            if len(selected) >= 3:
                break
            if c not in selected:
                selected.append(c)

    if not selected:
        rows.append((pad(["—", "No parlay candidates today", ""]), "no_plays"))
    else:
        payout3 = combined_american([c["odds"] for c in selected]) if len(selected) == 3 else ""
        for i, c in enumerate(selected):
            rows.append((pad([
                str(i + 1), c["batter"], c["team"], f"{c['score']:.1f}",
                f"+{int(c['odds'])}", f"{c['hit']:.0f}%" if c["hit"] else "—",
                payout3 if i == 0 else "", c["why"],
            ]), "data_parlay"))

    rows.append((E[:], "spacer"))

    # ── 2-LEG PARLAYS (x2 tickets) ───────────────────────────────────────
    rows.append((pad(["🎟️  2-LEG HR PARLAYS — 2 tickets (proven +EV slots, +301-400 band)"]),
                 "section_header_parlay"))
    rows.append((pad(["Ticket", "Batter", "Team", "Score", "Odds", "Hit%", "Payout", "Why"]),
                 "col_header_parlay"))

    two_pool = [c for c in parlay_candidates if c.get("in_2leg")]
    diversified, used_g = [], set()
    for c in two_pool:
        if c["opp_pit"] and c["opp_pit"] in used_g:
            continue
        diversified.append(c)
        if c["opp_pit"]:
            used_g.add(c["opp_pit"])
    if len(diversified) < 6:
        for c in two_pool:
            if len(diversified) >= 6:
                break
            if c not in diversified:
                diversified.append(c)

    slot_pairs = [(3, 4), (5, 6)]
    pairs = [(diversified[i], diversified[j]) for i, j in slot_pairs if j < len(diversified)]
    if not pairs:
        rows.append((pad(["—", "Not enough candidates for a 2-legger today", ""]), "no_plays"))
    else:
        for t_idx, (a, b) in enumerate(pairs, start=1):
            payout2 = combined_american([a["odds"], b["odds"]])
            for leg_idx, leg in enumerate((a, b)):
                rows.append((pad([
                    f"#{t_idx}" if leg_idx == 0 else "", leg["batter"], leg["team"],
                    f"{leg['score']:.1f}", f"+{int(leg['odds'])}",
                    f"{leg['hit']:.0f}%" if leg["hit"] else "—",
                    payout2 if leg_idx == 0 else "", leg["why"],
                ]), "data_parlay"))
            if t_idx < len(pairs):
                rows.append((E[:], "spacer"))

    rows.append((E[:], "spacer"))
    return rows


# ── Dashboard writer ────────────────────────────────────────────────────────
def write_dashboard(gc, sheet_id, rows) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet(DASHBOARD_SHEET)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=DASHBOARD_SHEET, rows=max(RESET_ROWS, len(rows) + 10), cols=RESET_COLS)
    ws_id = ws.id

    # Full reset: unmerge everything + blanket default format so no stale
    # cell (old KS/HRRBI colors, merges, wraps) survives into the new layout.
    reset_reqs = [
        {"unmergeCells": {"range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": RESET_ROWS,
                                    "startColumnIndex": 0, "endColumnIndex": RESET_COLS}}},
        {"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": RESET_ROWS,
                      "startColumnIndex": 0, "endColumnIndex": RESET_COLS},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_BG,
                "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                "verticalAlignment": "MIDDLE", "horizontalAlignment": "LEFT", "wrapStrategy": "CLIP",
            }},
            "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,horizontalAlignment,wrapStrategy)",
        }},
    ]
    try:
        with_retry(lambda: sh.batch_update({"requests": reset_reqs}))
    except Exception:
        pass

    with_retry(lambda: ws.clear())
    data = [row_data for row_data, _ in rows]
    with_retry(lambda: ws.update(data, value_input_option="RAW"))

    reqs = []
    data_counts = {}
    for r, (row_data, rtype) in enumerate(rows):
        if rtype == "timestamp":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "italic": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy)",
            }})
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "mergeType": "MERGE_ALL"}})

        elif rtype.startswith("section_header"):
            color = COLOR_PURPLE if "parlay" in rtype else COLOR_GOLD
            text_color = COLOR_WHITE if "parlay" in rtype else COLOR_BLACK
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": color,
                    "textFormat": {"foregroundColor": text_color, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 12},
                    "horizontalAlignment": "LEFT", "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})
            reqs.append({"mergeCells": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "mergeType": "MERGE_ALL"}})

        elif rtype.startswith("col_header"):
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_HEADER_BG,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 9},
                    "horizontalAlignment": "LEFT", "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
            }})

        elif rtype.startswith("data"):
            key = "parlay" if rtype == "data_parlay" else "hr"
            c = data_counts.get(key, 0)
            data_counts[key] = c + 1
            bg = COLOR_BG if c % 2 == 0 else COLOR_BG_ALT
            # whole row: bg + WRAP (so Why can wrap without clipping)
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": bg,
                    "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 11},
                    "verticalAlignment": "MIDDLE", "horizontalAlignment": "LEFT", "wrapStrategy": "WRAP"}},
                "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,horizontalAlignment,wrapStrategy)",
            }})
            # col 0 (Rank/Leg/Ticket) centered dim
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": 1},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "horizontalAlignment": "CENTER"}},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})
            # col 3 (Score) colored by tier
            score_color = COLOR_GREEN if "strong" in rtype else (
                COLOR_GOLD if ("moderate" in rtype or rtype == "data_parlay") else COLOR_WHITE)
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 3, "endColumnIndex": 4},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": score_color, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11},
                    "horizontalAlignment": "CENTER"}},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)",
            }})
            # col 6 = Edge (singles) / Payout (parlays) — green accent
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 6, "endColumnIndex": 7},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": COLOR_GREEN, "bold": True,
                                   "fontFamily": "Roboto", "fontSize": 11}}},
                "fields": "userEnteredFormat(textFormat)",
            }})

        elif rtype == "no_plays":
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": r, "endRowIndex": r + 1,
                          "startColumnIndex": 0, "endColumnIndex": N_COLS},
                "cell": {"userEnteredFormat": {
                    "backgroundColor": COLOR_BG_ALT,
                    "textFormat": {"foregroundColor": COLOR_SUBTEXT, "italic": True,
                                   "fontFamily": "Roboto", "fontSize": 11}}},
                "fields": "userEnteredFormat(backgroundColor,textFormat)",
            }})

    # Column widths (fit content; Why wraps)
    col_widths = [56, 158, 52, 62, 66, 66, 120, 300]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w}, "fields": "pixelSize",
        }})

    # Row heights
    for r, (_, rtype) in enumerate(rows):
        if rtype == "timestamp":
            h = 30
        elif rtype.startswith("section_header"):
            h = 38
        elif rtype == "spacer":
            h = 12
        elif rtype.startswith("col_header"):
            h = 24
        else:
            h = 44
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "ROWS", "startIndex": r, "endIndex": r + 1},
            "properties": {"pixelSize": h}, "fields": "pixelSize",
        }})

    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 0},
                       "tabColorStyle": {"rgbColor": COLOR_GOLD}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle",
    }})

    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print("Dashboard formatting applied.")
    except APIError as e:
        print(f"Dashboard formatting failed: {e}")


# ── Scorecard (HR only) ─────────────────────────────────────────────────────
def write_scorecard(gc, sheet_id, rows_data, today_str) -> None:
    sh = with_retry(lambda: gc.open_by_key(sheet_id))
    try:
        ws = sh.worksheet("Scorecard")
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title="Scorecard", rows=1000, cols=11)
    ws_id = ws.id

    existing = with_retry(lambda: ws.get_all_values())
    headers = ["Date", "Model", "Player / Pitcher", "Team", "Score",
               "Direction / Line", "Suggested Odds", "Your Odds", "Stake", "Result", "P&L"]
    if not existing or existing[0] != headers:
        existing = [headers]
    kept = [r for r in existing[1:] if r and str(r[0]).strip() != today_str]

    today_rows = []
    current_model = ""
    for row_data, rtype in rows_data:
        if rtype == "section_header_hr":
            current_model = "HR Single"
        elif rtype == "section_header_parlay":
            current_model = "HR Parlay 2-leg" if "2-LEG" in str(row_data[0]) else "HR Parlay 3-leg"

        if rtype in ("data_hr_strong", "data_hr_moderate", "data_hr_light"):
            name = str(row_data[1]).strip()
            if name and name != "—":
                today_rows.append([today_str, "HR Single", name, str(row_data[2]).strip(),
                                   str(row_data[3]).strip(), "HR", str(row_data[4]).strip(),
                                   "", "", "", ""])
        elif rtype == "data_parlay":
            name = str(row_data[1]).strip()
            if name and name != "—":
                today_rows.append([today_str, current_model, name, str(row_data[2]).strip(),
                                   str(row_data[3]).strip(), "HR", str(row_data[4]).strip(),
                                   "", "", "", ""])

    if not today_rows:
        print("Scorecard: no plays to write today.")
        return

    all_rows = [headers] + kept + today_rows
    with_retry(lambda: ws.clear())
    with_retry(lambda: ws.update(all_rows, value_input_option="USER_ENTERED"))

    total = len(all_rows)
    reqs = [{"repeatCell": {
        "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": total,
                  "startColumnIndex": 0, "endColumnIndex": 11},
        "cell": {"userEnteredFormat": {
            "backgroundColor": COLOR_BG,
            "textFormat": {"foregroundColor": COLOR_WHITE, "fontFamily": "Roboto Mono", "fontSize": 10},
            "verticalAlignment": "MIDDLE", "wrapStrategy": "CLIP", "horizontalAlignment": "LEFT"}},
        "fields": "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,wrapStrategy,horizontalAlignment)"}},
        {"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 0, "endRowIndex": 1,
                      "startColumnIndex": 0, "endColumnIndex": 11},
            "cell": {"userEnteredFormat": {
                "backgroundColor": COLOR_HEADER_BG,
                "textFormat": {"foregroundColor": COLOR_GOLD, "bold": True,
                               "fontFamily": "Roboto", "fontSize": 10},
                "horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"}},
    ]
    model_colors = {
        "HR Single": COLOR_GOLD,
        "HR Parlay 3-leg": COLOR_PURPLE,
        "HR Parlay 2-leg": {"red": 0.451, "green": 0.227, "blue": 0.620},
    }
    for idx, row in enumerate(all_rows[1:], start=1):
        if len(row) < 2:
            continue
        bg = COLOR_BG if idx % 2 == 0 else COLOR_BG_ALT
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": idx, "endRowIndex": idx + 1,
                      "startColumnIndex": 0, "endColumnIndex": 11},
            "cell": {"userEnteredFormat": {"backgroundColor": bg}},
            "fields": "userEnteredFormat(backgroundColor)"}})
        color = model_colors.get(str(row[1]).strip())
        if color:
            reqs.append({"repeatCell": {
                "range": {"sheetId": ws_id, "startRowIndex": idx, "endRowIndex": idx + 1,
                          "startColumnIndex": 1, "endColumnIndex": 2},
                "cell": {"userEnteredFormat": {
                    "textFormat": {"foregroundColor": color, "bold": True},
                    "horizontalAlignment": "CENTER"}},
                "fields": "userEnteredFormat(textFormat,horizontalAlignment)"}})
    for ci in (7, 8, 9):
        reqs.append({"repeatCell": {
            "range": {"sheetId": ws_id, "startRowIndex": 1, "endRowIndex": total,
                      "startColumnIndex": ci, "endColumnIndex": ci + 1},
            "cell": {"userEnteredFormat": {
                "backgroundColor": {"red": 0.10, "green": 0.10, "blue": 0.15},
                "textFormat": {"foregroundColor": COLOR_WHITE, "bold": True},
                "horizontalAlignment": "CENTER"}},
            "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"}})
    col_widths = [90, 130, 180, 60, 70, 120, 100, 90, 70, 70, 80]
    for i, w in enumerate(col_widths):
        reqs.append({"updateDimensionProperties": {
            "range": {"sheetId": ws_id, "dimension": "COLUMNS", "startIndex": i, "endIndex": i + 1},
            "properties": {"pixelSize": w}, "fields": "pixelSize"}})
    reqs.append({"updateSheetProperties": {
        "properties": {"sheetId": ws_id, "gridProperties": {"frozenRowCount": 1},
                       "tabColorStyle": {"rgbColor": COLOR_GREEN}},
        "fields": "gridProperties.frozenRowCount,tabColorStyle"}})
    try:
        with_retry(lambda: sh.batch_update({"requests": reqs}))
        print(f"Scorecard written: {len(today_rows)} plays for {today_str}")
    except APIError as e:
        print(f"Scorecard formatting failed: {e}")


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    time.sleep(5)
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc = get_gspread_client()

    print("Reading HR picks + scores for dashboard...")
    hr_df = read_sheet(gc, sheet_id, "Top_HR_Picks")
    time.sleep(2)
    hr_all_scores = read_sheet_raw(gc, sheet_id, "HR_All_Scores")
    time.sleep(2)

    print(f"HR picks: {len(hr_df)} rows | HR All Scores: {len(hr_all_scores)} rows")
    hr_hit_rates = build_hr_hit_rates(hr_all_scores)

    today_str = _date.today().strftime("%Y-%m-%d")
    if not hr_all_scores.empty and "date" in hr_all_scores.columns:
        hr_today = hr_all_scores[hr_all_scores["date"].astype(str).str.strip() == today_str].copy()
    else:
        hr_today = pd.DataFrame()
    print(f"HR today's scores: {len(hr_today)} players")

    et = pytz.timezone("America/New_York")
    ts = datetime.now(et).strftime("%B %d, %Y at %I:%M %p ET")

    rows = build_rows(hr_df, hr_hit_rates, hr_today, ts)
    write_dashboard(gc, sheet_id, rows)
    time.sleep(3)
    write_scorecard(gc, sheet_id, rows, today_str)
    print("Dashboard written to 'Today's Top Picks'")


if __name__ == "__main__":
    main()
