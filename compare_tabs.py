#!/usr/bin/env python3
"""
compare_tabs.py

Reconciles two tabs that have drifted apart:
  Picks_Log          -- the MODEL's picks, auto-resolved from Statcast
  Model_Bet_Tracker  -- what YOU actually bet (real prices, stakes, parlays)

Matches on date + player and reports:
  1. Coverage: how many rows each tab has, how many overlap
  2. Disagreements: picks in one tab but not the other, and rows where the
     result differs between tabs
  3. P/L side by side: each tab's hit rate / units / ROI on its own terms

Headers are auto-detected (date, player, result, odds, stake variants) and the
mapping is printed so you can confirm it matched the right columns.

USAGE
    python3 compare_tabs.py            # live sheet (needs GOOGLE_* env)
    python3 compare_tabs.py a.csv b.csv   # local: Picks_Log, then Tracker
"""

import sys
import os
import json
import re
import math
import pandas as pd
import numpy as np


# ── column auto-detection ─────────────────────────────────────────────────
CANDIDATES = {
    "date":   ["date", "game_date", "day"],
    "player": ["player", "players", "player_name", "name", "batter"],
    "result": ["result", "hit_hr", "outcome", "won_lost", "status"],
    "odds":   ["odds", "odds_taken", "price", "combined_odds", "american_odds"],
    "stake":  ["stake", "bet_placed", "units", "wager", "bet_size", "risk"],
    "type":   ["bet_type", "type", "market"],
    "consensus": ["consensus_odds", "consensus", "consensus_at_bet"],
}


def detect(cols, kind):
    low = {c.lower().strip(): c for c in cols}
    for cand in CANDIDATES[kind]:
        if cand in low:
            return low[cand]
    # fuzzy contains
    for cand in CANDIDATES[kind]:
        for lc, orig in low.items():
            if cand in lc:
                return orig
    return None


def norm_name(s):
    s = str(s)
    s = re.sub(r"[|,].*$", "", s)  # tracker may store "A|B|C" for parlays
    s = re.sub(r"[^a-z ]", "", s.lower()).strip()
    return s


def norm_result(v):
    s = str(v).strip().lower()
    if s in ("won", "win", "yes", "hit", "w"):
        return "win"
    if s in ("lost", "loss", "no", "miss", "l"):
        return "loss"
    if s in ("void", "push", "dnp", "scratch"):
        return "void"
    if s in ("pending", ""):
        return "pending"
    return s


def american_profit(odds):
    try:
        o = float(str(odds).replace("+", "").strip())
    except (ValueError, TypeError):
        return None
    if o == 0:
        return None
    return o / 100 if o >= 0 else 100 / abs(o)


def summarize(df, cmap, tab):
    print(f"\n{'='*72}\n{tab}: P/L SUMMARY\n{'='*72}")
    r = df["_result"]
    settled = df[r.isin(["win", "loss"])]
    n = len(settled)
    if n == 0:
        print("  No settled (win/loss) rows.")
        return
    wins = (settled["_result"] == "win").sum()
    print(f"  settled: {n}   wins: {wins}   hit rate: {wins/n*100:.1f}%   "
          f"voids: {(r=='void').sum()}   pending: {(r=='pending').sum()}")

    # P/L needs odds + (optional) stake
    if cmap["odds"] is None:
        print("  No odds column — cannot compute P/L for this tab.")
        return
    pnl = 0.0; staked = 0.0; priced = 0
    for _, row in settled.iterrows():
        prof = american_profit(row[cmap["odds"]])
        if prof is None:
            continue
        stake = 1.0
        if cmap["stake"] is not None:
            try:
                sv = float(str(row[cmap["stake"]]).replace("$", "").replace("u", "").strip())
                stake = sv if sv > 0 else 1.0
            except (ValueError, TypeError):
                stake = 1.0
        priced += 1
        staked += stake
        pnl += prof * stake if row["_result"] == "win" else -stake
    if priced == 0:
        print("  Odds column present but empty on settled rows — no P/L yet.")
        return
    print(f"  priced rows: {priced}   units staked: {staked:.1f}   "
          f"P/L: {pnl:+.2f}u   ROI: {pnl/staked*100:+.1f}%")


def prep(df, tab):
    cmap = {k: detect(df.columns, k) for k in CANDIDATES}
    print(f"\n{tab} — {len(df)} rows. Column mapping detected:")
    for k, v in cmap.items():
        print(f"    {k:<10} -> {v}")
    if cmap["date"] is None or cmap["player"] is None:
        print(f"  !! {tab} missing date or player column — cannot match.")
    df = df.copy()
    df["_date"] = df[cmap["date"]].astype(str).str.strip() if cmap["date"] else ""
    df["_player"] = df[cmap["player"]].apply(norm_name) if cmap["player"] else ""
    df["_result"] = df[cmap["result"]].apply(norm_result) if cmap["result"] else "pending"
    df["_key"] = df["_date"] + "|" + df["_player"]
    return df, cmap


def load_sheet(name):
    import gspread
    from google.oauth2.service_account import Credentials
    scopes = ["https://www.googleapis.com/auth/spreadsheets",
              "https://www.googleapis.com/auth/drive"]
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])
    vals = sh.worksheet(name).get_all_values()
    return pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else pd.DataFrame()


def main(picks_df, track_df):
    picks, pmap = prep(picks_df, "Picks_Log")
    track, tmap = prep(track_df, "Model_Bet_Tracker")

    pk = set(picks["_key"]); tk = set(track["_key"])
    both = pk & tk
    only_p = pk - tk
    only_t = tk - pk

    print(f"\n{'='*72}\nCOVERAGE\n{'='*72}")
    print(f"  Picks_Log rows:           {len(picks)}  (unique keys {len(pk)})")
    print(f"  Model_Bet_Tracker rows:   {len(track)}  (unique keys {len(tk)})")
    print(f"  In BOTH:                  {len(both)}")
    print(f"  Only in Picks_Log:        {len(only_p)}  (model flagged, you didn't bet)")
    print(f"  Only in Tracker:          {len(only_t)}  (you bet, not in model log)")
    if pk:
        print(f"  Overlap as % of model picks: {len(both)/len(pk)*100:.0f}%")

    # result disagreements on shared keys
    if both:
        pres = picks.set_index("_key")["_result"].to_dict()
        tres = track.set_index("_key")["_result"].to_dict()
        disagree = [(k, pres.get(k), tres.get(k)) for k in both
                    if pres.get(k) in ("win", "loss") and tres.get(k) in ("win", "loss")
                    and pres.get(k) != tres.get(k)]
        print(f"\n  Result disagreements on shared picks: {len(disagree)}")
        for k, p, t in disagree[:15]:
            d, name = k.split("|", 1)
            print(f"    {d}  {name:<22}  Picks_Log={p:<5}  Tracker={t}")
        if len(disagree) > 15:
            print(f"    ... and {len(disagree)-15} more")

    # a few examples of each 'only' set
    def show_only(keys, src, lim=10):
        for k in list(sorted(keys))[:lim]:
            d, name = k.split("|", 1)
            print(f"    {d}  {name}")
    if only_t:
        print(f"\n  Sample you bet but model didn't log (first 10):")
        show_only(only_t, "t")
    if only_p:
        print(f"\n  Sample model logged but you didn't bet (first 10):")
        show_only(only_p, "p")

    summarize(picks, pmap, "Picks_Log")
    summarize(track, tmap, "Model_Bet_Tracker")

    print(f"\n{'='*72}\nREAD\n{'='*72}")
    print("""  Picks_Log = the model's record (auto-resolved, filtered to its own rule).
  Model_Bet_Tracker = what you actually wagered (real prices/stakes/parlays).
  Low overlap is the point: you've been betting a different pool than the model
  logs. 'Only in Tracker' rows are bets the model never recorded; 'Only in
  Picks_Log' are model picks you skipped. For 'am I profitable', the Tracker
  P/L is the real-world answer; Picks_Log tells you if the *system* would have
  profited on its own rule. Reconcile by making both use the same pick rule.""")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        p = pd.read_csv(sys.argv[1], dtype=str).fillna("")
        t = pd.read_csv(sys.argv[2], dtype=str).fillna("")
        main(p, t)
    else:
        if "GOOGLE_SERVICE_ACCOUNT_JSON" not in os.environ:
            print("Need GOOGLE_* env vars, or pass two CSVs: Picks_Log then Tracker.")
            sys.exit(1)
        print("Loading Picks_Log and Model_Bet_Tracker live from Google Sheets.\n")
        main(load_sheet("Picks_Log"), load_sheet("Model_Bet_Tracker"))
