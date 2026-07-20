"""
test_top10_daily_pnl.py

Bet 1 unit on each day's TOP 10 hitters by hr_score, over the last month.
Grades at the logged consensus_odds (win = +odds/100 units, loss = -1), sums P/L.
Breaks out hit rate + P/L by score tier so you see where it comes from.

NOTE: grades at consensus (model) odds — not the price you'd actually shop,
so real P/L would differ by your line-shopping edge. This is the honest
"as-logged" number.

Env: MODEL_START defaults to last 31 days.
Local: CSV=/path/HR_All_Scores.csv python test_top10_daily_pnl.py
"""
import os
import json
import math
from datetime import date, timedelta
import numpy as np
import pandas as pd

TOP_N = int(os.environ.get("TOP_N", "10"))
DAYS_BACK = int(os.environ.get("DAYS_BACK", "31"))


def load_scores():
    csv = os.environ.get("CSV")
    if csv:
        print(f"Loading {csv}")
        return pd.read_csv(csv, dtype=str)
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ[os.environ.get("SHEET_ENV_VAR", "GOOGLE_SHEET_ID")])
    vals = sh.worksheet("HR_All_Scores").get_all_values()
    return pd.DataFrame(vals[1:], columns=vals[0])


def sf(v, d=0.0):
    try:
        f = float(v)
        return d if (math.isnan(f) or math.isinf(f)) else f
    except (ValueError, TypeError):
        return d


def score_tier(s):
    if s >= 15:  return "15+"
    if s >= 14:  return "14-15"
    if s >= 13:  return "13-14"
    if s >= 12:  return "12-13"
    if s >= 11:  return "11-12"
    if s >= 10:  return "10-11"
    if s >= 9:   return "9-10"
    return "<9"


def profit(odds, hit):
    if hit:
        return odds / 100.0 if odds > 0 else 100.0 / abs(odds)
    return -1.0


def main():
    df = load_scores()
    df["date_dt"] = pd.to_datetime(df.get("date"), errors="coerce")
    cutoff = pd.Timestamp(date.today() - timedelta(days=DAYS_BACK))
    df = df[df["date_dt"] >= cutoff]
    df = df[df["hit_hr"].astype(str).str.strip().isin(["Yes", "No"])].copy()
    if df.empty:
        print("No resolved rows in the window.")
        return

    df["score"] = df.apply(
        lambda r: sf(r.get("hr_score_corrected"))
        if str(r.get("hr_score_corrected", "")).strip() not in ("", "nan", "None")
        else sf(r.get("hr_score")), axis=1)
    df["odds"] = df["consensus_odds"].apply(sf)
    df["hit"] = (df["hit_hr"].astype(str).str.strip() == "Yes").astype(int)
    df = df[(df["odds"] != 0) & (df["score"] > 0)]
    df["day"] = df["date_dt"].dt.date

    # each day: take the TOP_N by score
    picks = []
    for d, g in df.groupby("day"):
        top = g.sort_values("score", ascending=False).head(TOP_N)
        picks.append(top)
    picks = pd.concat(picks, ignore_index=True)
    picks["pnl"] = picks.apply(lambda r: profit(r["odds"], r["hit"]), axis=1)
    picks["tier"] = picks["score"].apply(score_tier)

    n = len(picks)
    hits = int(picks["hit"].sum())
    total_pnl = picks["pnl"].sum()
    roi = total_pnl / n * 100
    days = picks["day"].nunique()

    print("=" * 60)
    print(f"  DAILY TOP {TOP_N} BY SCORE — 1u each, last {DAYS_BACK} days")
    print("=" * 60)
    print(f"  days:        {days}")
    print(f"  bets:        {n}")
    print(f"  hits:        {hits}  ({hits/n*100:.1f}% hit rate)")
    print(f"  total P/L:   {total_pnl:+.2f} units")
    print(f"  ROI:         {roi:+.1f}% per bet")
    print(f"  avg odds:    {picks['odds'].mean():+.0f}")

    print("\n  By score tier:")
    print(f"  {'tier':>8} {'bets':>5} {'hits':>5} {'hit%':>6} {'P/L':>9} {'ROI':>7}")
    print("  " + "-" * 46)
    for tier in ["15+", "14-15", "13-14", "12-13", "11-12", "10-11", "9-10", "<9"]:
        sub = picks[picks["tier"] == tier]
        if sub.empty:
            continue
        h = int(sub["hit"].sum()); b = len(sub); p = sub["pnl"].sum()
        print(f"  {tier:>8} {b:5d} {h:5d} {h/b*100:5.1f}% {p:+8.2f} {p/b*100:+6.1f}%")

    print("\n  By month-week (are results stable or streaky?):")
    picks["week"] = pd.to_datetime(picks["day"]).dt.isocalendar().week
    for wk, g in picks.groupby("week"):
        p = g["pnl"].sum()
        print(f"    week {wk}: {len(g)} bets, {int(g['hit'].sum())} hits, {p:+.2f}u")

    print("\n  NOTE: graded at consensus odds as logged. Your shopped price would")
    print("  differ. One month is a small sample — treat P/L as directional.")

    # ── RESTRICTED VIEW: only the tiers that actually win (13-14 + 14-15) ──
    # Every test this session converges on these two tiers; 15+ underperforms
    # and 12-13 is a drag. This is the P/L of what you'd actually bet.
    print("\n" + "=" * 60)
    print(f"  RESTRICTED — 13-14 & 14-15 ONLY (1u each, last {DAYS_BACK} days)")
    print("=" * 60)
    good = picks[picks["tier"].isin(["13-14", "14-15"])].copy()
    if good.empty:
        print("  No 13-15 legs in window.")
    else:
        gn = len(good); gh = int(good["hit"].sum()); gp = good["pnl"].sum()
        print(f"  bets:        {gn}")
        print(f"  hits:        {gh}  ({gh/gn*100:.1f}% hit rate)")
        print(f"  total P/L:   {gp:+.2f} units")
        print(f"  ROI:         {gp/gn*100:+.1f}% per bet")
        print(f"  avg odds:    {good['odds'].mean():+.0f}")
        print("\n  By week (still watch for one-week streakiness):")
        for wk, g in good.groupby("week"):
            print(f"    week {wk}: {len(g)} bets, {int(g['hit'].sum())} hits, {g['pnl'].sum():+.2f}u")
        # also show it capped at the daily top-N within these tiers, in case
        # "top 10 including junk tiers" vs "all good-tier legs" differ in volume
        print(f"\n  (for reference: {gn} good-tier bets came out of {n} total top-{TOP_N} bets)")
        print("  Bumped by removing 15+ and 12-13, which lost money as logged.")


if __name__ == "__main__":
    main()
