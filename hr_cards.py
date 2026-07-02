"""
hr_cards.py
Generates a styled HTML card view of today's HR Value Plays — the same
confirmed value zones shown on the dashboard (13+ <=+300, 10-11 +301-499),
with a "Why this pick" explanation per player using contact quality deltas
and scoring components already logged in HR_All_Scores.

Output: hr_cards.html — saved to repo root, also uploaded as a workflow
artifact. No external hosting required; open the file directly.
"""

import os
import json
import time
from datetime import date

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


def read_sheet_raw(gc, sheet_id, name):
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
    except Exception:
        return pd.DataFrame()


def american_to_implied(odds):
    if odds >= 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def get_hr_score_tier(score):
    if score >= 13:   return "13+"
    if score >= 12:   return "12-13"
    if score >= 11:   return "11-12"
    if score >= 10:   return "10-11"
    if score >= 9:    return "9-10"
    return "8.5-9"


def build_hr_hit_rates(hr_all_scores):
    hit_rates = {}
    if hr_all_scores.empty:
        return hit_rates

    hr_all_scores = hr_all_scores.copy()
    hr_all_scores["date_dt"] = pd.to_datetime(hr_all_scores["date"], errors="coerce")
    hr_all_scores = hr_all_scores[hr_all_scores["date_dt"] >= pd.Timestamp(MODEL_START_DATE)]

    resolved = hr_all_scores[
        hr_all_scores["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])
    ].copy()
    if resolved.empty:
        return hit_rates

    resolved["hr_score"] = resolved["hr_score"].apply(safe_float)
    resolved["hit_bool"] = resolved["hit_hr"].astype(str).str.strip() == "Yes"

    tier_defs = [
        ("13+", 13, 999), ("12-13", 12, 13), ("11-12", 11, 12),
        ("10-11", 10, 11), ("9-10", 9, 10), ("8.5-9", 8.5, 9),
    ]
    for label, lo, hi in tier_defs:
        sub = resolved[(resolved["hr_score"] >= lo) & (resolved["hr_score"] < hi)]
        if len(sub) >= 5:
            hit_rates[label] = (sub["hit_bool"].mean(), len(sub))
    return hit_rates


def in_value_zone(score, odds):
    # Three confirmed value zones — must match dashboard.py exactly:
    #   13+   | ≤+300
    #   13+   | +301-499
    #   10-11 | +301-499
    if score >= 13.0 and odds <= 300:
        return True
    if score >= 13.0 and 301 <= odds <= 499:
        return True
    if 10.0 <= score < 11.0 and 301 <= odds <= 499:
        return True
    return False


def zone_label(score, odds=None):
    if score >= 13.0 and odds is not None and 301 <= odds <= 499:
        return "13+ +301-499"
    if score >= 13.0:
        return "13+ ≤+300"
    return "10-11 +301-499"


def delta_str_html(recent, baseline):
    """Returns (value_str, css_class) for a contact quality delta."""
    r = str(recent).strip()
    b = str(baseline).strip()
    if not r or not b or r in ("0", "0.0", "") or b in ("0", "0.0", ""):
        return "—", "stat-val"
    try:
        d = round(float(r) - float(b), 1)
        cls = "stat-val good" if d > 1 else "stat-val warn" if d < -1 else "stat-val"
        sign = "+" if d >= 0 else ""
        return f"{sign}{d}", cls
    except Exception:
        return "—", "stat-val"


def build_why_text(row, edge_pct, breakeven_pct, hit_rate_pct, n_picks):
    """Builds a human-readable explanation for why this player qualified."""
    platoon  = safe_float(row.get("platoon_score", 0))
    pm       = safe_float(row.get("pitch_matchup_score", 0))
    hr_fb    = safe_float(row.get("hr_per_fb", 0))
    barrel_7d = str(row.get("barrel_pct_7d", "")).strip()
    barrel_season = str(row.get("season_barrel_pct", "")).strip()
    ev_7d    = str(row.get("avg_ev_7d", "")).strip()
    ev_30d   = str(row.get("avg_ev_30d", "")).strip()
    hh_7d    = str(row.get("hard_hit_pct_7d", "")).strip()
    hh_season = str(row.get("hard_hit_pct_season", "")).strip()

    parts = []

    # Contact quality trend
    trend_signals = []
    try:
        if barrel_7d and barrel_season and float(barrel_7d) - float(barrel_season) >= 3:
            trend_signals.append("barrel%")
    except Exception:
        pass
    try:
        if ev_7d and ev_30d and float(ev_7d) - float(ev_30d) >= 1.0:
            trend_signals.append("exit velo")
    except Exception:
        pass
    try:
        if hh_7d and hh_season and float(hh_7d) - float(hh_season) >= 5:
            trend_signals.append("hard hit%")
    except Exception:
        pass

    cold_signals = []
    try:
        if barrel_7d and barrel_season and float(barrel_7d) - float(barrel_season) <= -3:
            cold_signals.append("barrel%")
    except Exception:
        pass
    try:
        if ev_7d and ev_30d and float(ev_7d) - float(ev_30d) <= -1.0:
            cold_signals.append("exit velo")
    except Exception:
        pass
    try:
        if hh_7d and hh_season and float(hh_7d) - float(hh_season) <= -5:
            cold_signals.append("hard hit%")
    except Exception:
        pass

    if len(trend_signals) >= 2:
        parts.append(f"Trending hot — {', '.join(trend_signals)} up vs personal baseline.")
    elif len(cold_signals) >= 2:
        parts.append(f"⚠️ Cold signal — {', '.join(cold_signals)} down vs personal baseline. Score is carried by season power, not current form.")

    if hr_fb >= 20:
        parts.append(f"Elite season power ({hr_fb:.1f}% HR/FB).")
    elif hr_fb >= 14:
        parts.append(f"Above-average power ({hr_fb:.1f}% HR/FB).")

    if platoon >= 1.5:
        parts.append("Strongly favorable platoon matchup.")
    elif platoon >= 0.8:
        parts.append("Favorable platoon matchup.")
    elif platoon < 0:
        parts.append("Unfavorable platoon matchup — score is carried by power, not matchup.")

    if pm >= 3.5:
        parts.append("Strong pitch-type matchup against today's starter.")

    if not parts:
        parts.append("Qualified on score and confirmed zone edge.")

    return " ".join(parts)


def build_cards_html(plays, today_str):
    cards_html = ""
    for i, p in enumerate(plays, 1):
        hot_badge = '<span class="badge conf-high">HOT</span>' if p["is_hot"] else ""
        cold_note = '<div class="cold-flag">⚠️ Cold contact quality this week</div>' if p["is_cold"] else ""

        cards_html += f"""
    <div class="card">
      <div class="card-top">
        <div class="rank">{i}</div>
        <span class="team-badge">{p['team']}</span>
        <span class="player-name">{p['batter']}</span>
        <div class="badges">
          <span class="badge zone">{p['zone']}</span>
          {hot_badge}
        </div>
      </div>
      <div class="matchup">vs <b>{p['pitcher']}</b> &nbsp;·&nbsp; {p['odds']}</div>

      <div class="bar-row">
        <div class="bar-track"><div class="bar-fill" style="width:{p['bar_pct']}%"></div></div>
        <div class="score-num">{p['score']}</div>
      </div>

      <div class="stats-row">
        <div><div class="stat-label">Barrel% 7d</div><div class="{p['barrel_cls']}">{p['barrel_val']}</div></div>
        <div><div class="stat-label">HR/FB%</div><div class="stat-val">{p['hr_fb']}%</div></div>
        <div><div class="stat-label">Platoon</div><div class="stat-val">{p['platoon']}</div></div>
        <div><div class="stat-label">EV Δ7d</div><div class="{p['ev_cls']}">{p['ev_val']}</div></div>
        <div><div class="stat-label">HH% Δ7d</div><div class="{p['hh_cls']}">{p['hh_val']}</div></div>
      </div>
      {cold_note}
      <div class="why">
        <b>Why this pick:</b> {p['why']}
        <div class="why-edge">
          <span>Hit rate this zone: <b style="color:#fff">{p['hit_rate']}%</b> ({p['n_picks']} picks) · Breakeven: {p['breakeven']}%</span>
          <span class="edge-val">{p['edge']}</span>
        </div>
      </div>
    </div>"""
    return cards_html


def build_full_html(plays, today_str):
    n_plays = len(plays)
    cards = build_cards_html(plays, today_str)

    if n_plays == 0:
        cards = '<div class="empty">No qualifying plays today — no players in confirmed value zones.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HR Value Plays — {today_str}</title>
<style>
  :root {{
    --bg: #0a0d12;
    --card-bg: #11161d;
    --card-border: #1f2730;
    --gold: #e8b923;
    --green: #34d399;
    --green-dim: #1a3d31;
    --red: #f0556b;
    --text: #f3f5f7;
    --subtext: #8993a1;
    --teal: #2dd4bf;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    padding: 24px 16px 48px;
  }}
  .frame {{
    max-width: 1180px;
    margin: 0 auto;
    border: 1px solid #2a3340;
    border-radius: 18px;
    padding: 24px 20px 32px;
    background: radial-gradient(1200px 400px at 10% -10%, rgba(232,185,35,0.06), transparent), var(--bg);
  }}
  @media (min-width: 640px) {{ .frame {{ padding: 28px 32px 36px; }} }}
  .eyebrow {{
    display: flex; align-items: center; gap: 8px;
    color: var(--gold); font-size: 12px; letter-spacing: 0.12em;
    text-transform: uppercase; font-weight: 700; margin-bottom: 6px;
  }}
  .eyebrow .dot {{ width: 7px; height: 7px; border-radius: 50%; background: var(--gold); }}
  .header-row {{ display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px; margin-bottom: 4px; }}
  h1 {{ font-size: 26px; margin: 0; letter-spacing: -0.01em; }}
  @media (min-width: 640px) {{ h1 {{ font-size: 34px; }} }}
  .meta {{ color: var(--subtext); font-size: 13px; margin-top: 4px; }}
  .header-right {{ text-align: right; }}
  .header-right .date {{ font-size: 15px; font-weight: 700; }}
  .pill {{
    display: inline-block; margin-top: 8px; padding: 4px 12px;
    border: 1px solid #344050; border-radius: 999px; font-size: 12px; color: var(--subtext);
  }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; margin-top: 26px; }}
  @media (min-width: 800px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
  .card {{ background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 14px; padding: 18px 20px; }}
  .card-top {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  .rank {{
    width: 26px; height: 26px; border-radius: 50%; border: 1px solid #3a4654;
    display: flex; align-items: center; justify-content: center; font-size: 12px; color: var(--subtext); flex-shrink: 0;
  }}
  .team-badge {{ font-size: 11px; font-weight: 800; padding: 3px 7px; border-radius: 5px; background: #1c2733; color: #9db4cc; letter-spacing: 0.03em; }}
  .player-name {{ font-size: 18px; font-weight: 700; margin-left: 2px; }}
  .badges {{ margin-left: auto; display: flex; gap: 6px; flex-shrink: 0; }}
  .badge {{ font-size: 10.5px; font-weight: 700; padding: 3px 8px; border-radius: 999px; white-space: nowrap; }}
  .badge.zone {{ background: var(--green-dim); color: var(--green); border: 1px solid #1f5945; }}
  .badge.conf-high {{ background: #2a2210; color: var(--gold); border: 1px solid #4a3c14; }}
  .matchup {{ color: var(--subtext); font-size: 12.5px; margin-top: 8px; margin-left: 36px; }}
  .matchup b {{ color: #c4cdd6; font-weight: 600; }}
  .bar-row {{ display: flex; align-items: center; gap: 14px; margin-top: 14px; }}
  .bar-track {{ flex: 1; height: 8px; border-radius: 99px; background: #1a212b; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 99px; background: linear-gradient(90deg, #1aa37a, var(--green)); }}
  .score-num {{ font-size: 28px; font-weight: 800; color: var(--green); min-width: 52px; text-align: right; }}
  .stats-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-top: 16px; }}
  .stat-label {{ font-size: 9px; color: var(--subtext); letter-spacing: 0.04em; text-transform: uppercase; }}
  .stat-val {{ font-size: 13px; font-weight: 700; margin-top: 2px; }}
  .stat-val.good {{ color: var(--green); }}
  .stat-val.warn {{ color: var(--red); }}
  .cold-flag {{ margin-top: 10px; font-size: 11.5px; color: var(--red); font-weight: 600; }}
  .why {{ margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--card-border); font-size: 12px; color: #b7c0cb; line-height: 1.5; }}
  .why b {{ color: var(--teal); font-weight: 700; }}
  .why-edge {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 4px; margin-top: 8px; font-size: 11px; color: var(--subtext); }}
  .why-edge .edge-val {{ color: var(--green); font-weight: 700; }}
  .empty {{ padding: 60px 20px; text-align: center; color: var(--subtext); font-size: 14px; }}
</style>
</head>
<body>
<div class="frame">
  <div class="header-row">
    <div>
      <div class="eyebrow"><span class="dot"></span>Dashboard HR Value Plays</div>
      <h1>13+ ≤+300 &nbsp;|&nbsp; 10-11 +301-499</h1>
      <div class="meta">Confirmed profitable zones · Score &amp; contact quality · Generated daily</div>
    </div>
    <div class="header-right">
      <div class="date">{today_str}</div>
      <div class="pill">{n_plays} play{'s' if n_plays != 1 else ''} today</div>
    </div>
  </div>
  <div class="grid">{cards}
  </div>
</div>
</body>
</html>"""


def main():
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    print("Reading HR_All_Scores...")
    hr_all_scores = read_sheet_raw(gc, sheet_id, "HR_All_Scores")

    today_str = date.today().strftime("%Y-%m-%d")
    today = hr_all_scores[hr_all_scores["date"].astype(str).str.strip() == today_str].copy() if not hr_all_scores.empty else pd.DataFrame()
    print(f"Today's scored players: {len(today)}")

    hit_rates = build_hr_hit_rates(hr_all_scores)

    plays = []
    if not today.empty:
        for _, row in today.iterrows():
            batter = str(row.get("player_name", "")).strip()
            if not batter or batter == "nan":
                continue
            team = str(row.get("team", "")).strip()
            score = safe_float(row.get("hr_score", 0))
            odds_raw = str(row.get("consensus_odds", "")).strip()
            odds_val = safe_float(odds_raw.replace("+", "")) if odds_raw not in ("", "nan") else 0.0

            if odds_val <= 0 or score <= 0:
                continue
            if not in_value_zone(score, odds_val):
                continue

            tier = get_hr_score_tier(score)
            result = hit_rates.get(tier)
            if result is None:
                continue
            hit_rate, n_picks = result
            implied = american_to_implied(odds_val)
            edge = hit_rate - implied
            edge_pct = round(edge * 100, 1)
            breakeven_pct = round(implied * 100, 1)

            barrel_val, barrel_cls = delta_str_html(row.get("barrel_pct_7d", ""), row.get("season_barrel_pct", ""))
            ev_val, ev_cls = delta_str_html(row.get("avg_ev_7d", ""), row.get("avg_ev_30d", ""))
            hh_val, hh_cls = delta_str_html(row.get("hard_hit_pct_7d", ""), row.get("hard_hit_pct_season", ""))

            is_hot = barrel_cls == "stat-val good" and ev_cls == "stat-val good"
            is_cold = barrel_cls == "stat-val warn" and ev_cls == "stat-val warn"

            why = build_why_text(row, edge_pct, breakeven_pct, round(hit_rate*100,1), n_picks)

            plays.append({
                "batter": batter,
                "team": team,
                "pitcher": str(row.get("pitcher_name", "")).strip() or "TBD",
                "score": round(score, 1),
                "bar_pct": min(100, round(score / 18 * 100)),
                "odds": f"+{int(odds_val)}",
                "zone": zone_label(score, odds_val),
                "hr_fb": round(safe_float(row.get("hr_per_fb", 0)), 1),
                "platoon": round(safe_float(row.get("platoon_score", 0)), 2),
                "barrel_val": barrel_val, "barrel_cls": barrel_cls,
                "ev_val": ev_val, "ev_cls": ev_cls,
                "hh_val": hh_val, "hh_cls": hh_cls,
                "is_hot": is_hot,
                "is_cold": is_cold,
                "why": why,
                "hit_rate": round(hit_rate*100, 1),
                "n_picks": n_picks,
                "breakeven": breakeven_pct,
                "edge": f"{'+' if edge_pct>=0 else ''}{edge_pct}% edge",
                "edge_num": edge_pct,
            })

    plays.sort(key=lambda x: -x["score"])

    html = build_full_html(plays, today_str)

    out_path = "hr_cards.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"Wrote {len(plays)} cards to {out_path}")


if __name__ == "__main__":
    main()
