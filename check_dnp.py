import os
import json
from datetime import date, timedelta

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


def get_gspread_client() -> gspread.Client:
    raw_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    info = json.loads(raw_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)


def normalize_name(name: str) -> str:
    import unicodedata
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    return name.lower().strip()


def main() -> None:
    sheet_id = os.environ["GOOGLE_SHEET_ID"]
    gc       = get_gspread_client()

    # Read Picks_Log
    sh = gc.open_by_key(sheet_id)
    try:
        ws       = sh.worksheet("Picks_Log")
        picks_df = pd.DataFrame(ws.get_all_records())
    except gspread.WorksheetNotFound:
        print("Picks_Log not found.")
        return

    if picks_df.empty:
        print("Picks_Log is empty.")
        return

    # Only check resolved picks (Yes or No) — skip Pending
    resolved = picks_df[picks_df["hit_hr"].isin(["Yes", "No"])].copy()
    if resolved.empty:
        print("No resolved picks to check.")
        return

    resolved["date"] = pd.to_datetime(resolved["date"])
    min_date = resolved["date"].min().strftime("%Y-%m-%d")
    max_date = resolved["date"].max().strftime("%Y-%m-%d")

    print(f"Pulling Statcast from {min_date} to {max_date}...")
    try:
        sc_df = statcast(start_dt=min_date, end_dt=max_date)
        if sc_df is None or sc_df.empty:
            print("Statcast returned empty.")
            return
        print(f"Pulled {len(sc_df):,} rows")
    except Exception as e:
        print(f"Statcast pull failed: {e}")
        return

    sc_df["game_date"] = pd.to_datetime(sc_df["game_date"])

    # Build lookup of (date, batter_id) -> plate appearances
    pa_events = sc_df[sc_df["events"].notna()].copy()
    pa_events = pa_events.drop_duplicates(
        subset=[c for c in ["game_pk", "at_bat_number", "batter"] if c in pa_events.columns]
    )
    pa_lookup = (
        pa_events.groupby(["game_date", "batter"])
        .size()
        .reset_index(name="pa_count")
    )
    pa_lookup["batter"] = pa_lookup["batter"].astype(int)

    # Also build name lookup for fallback matching
    batter_ids = pa_events["batter"].dropna().astype(int).unique().tolist()
    print(f"Looking up names for {len(batter_ids)} batters...")
    name_map: dict = {}
    for i in range(0, len(batter_ids), 50):
        chunk = batter_ids[i:i + 50]
        url   = f"https://statsapi.mlb.com/api/v1/people?personIds={','.join(map(str, chunk))}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            for person in resp.json().get("people", []):
                pid  = person.get("id")
                name = person.get("fullName", "")
                if pid and name:
                    name_map[int(pid)] = normalize_name(name)
        except Exception:
            pass

    # Add normalized name to pa_lookup
    pa_lookup["batter_name_norm"] = pa_lookup["batter"].map(
        lambda x: name_map.get(x, "") if pd.notna(x) else ""
    )

    # Check each pick
    dnp_rows    = []
    played_rows = []

    for idx, row in resolved.iterrows():
        pick_date = pd.Timestamp(row["date"])
        pick_name = normalize_name(str(row.get("player_name", "")))
        player_id = row.get("player_id", "")

        # Try matching by player_id first
        matched_by_id = False
        if player_id and str(player_id).strip() not in ("", "nan"):
            try:
                pid = int(float(str(player_id)))
                day_pa = pa_lookup[
                    (pa_lookup["game_date"] == pick_date) &
                    (pa_lookup["batter"] == pid)
                ]
                if not day_pa.empty:
                    played_rows.append(idx)
                    matched_by_id = True
                else:
                    # ID found but no PA — try name fallback
                    pass
            except Exception:
                pass

        if not matched_by_id:
            # Fallback — match by normalized name
            day_names = pa_lookup[pa_lookup["game_date"] == pick_date]["batter_name_norm"].tolist()
            if pick_name in day_names:
                played_rows.append(idx)
            else:
                dnp_rows.append(idx)

    print(f"\n=== DNP CHECK RESULTS ===")
    print(f"Total resolved picks: {len(resolved)}")
    print(f"Confirmed played:     {len(played_rows)}")
    print(f"Likely DNP/no PA:     {len(dnp_rows)}")

    if dnp_rows:
        print("\nPlayers who likely did NOT play:")
        dnp_df = resolved.loc[dnp_rows][["date", "rank", "player_name", "team", "hit_hr"]].copy()
        dnp_df["date"] = dnp_df["date"].dt.strftime("%Y-%m-%d")
        print(dnp_df.to_string(index=False))

        # Stats with DNPs removed
        played_df  = resolved.loc[played_rows]
        dnp_hits   = resolved.loc[dnp_rows]["hit_hr"].eq("Yes").sum()
        total_dnp  = len(dnp_rows)
        print(f"\n=== ADJUSTED STATS (DNPs removed) ===")
        print(f"Picks after removing DNPs: {len(played_df)}")
        print(f"HRs hit:                   {played_df['hit_hr'].eq('Yes').sum()}")
        hit_rate = played_df['hit_hr'].eq('Yes').mean() * 100
        print(f"Adjusted hit rate:         {hit_rate:.1f}%")
        print(f"DNPs that were marked No:  {total_dnp - dnp_hits}")
        print(f"DNPs that were marked Yes: {dnp_hits} (shouldn't happen but worth checking)")

        # ROI at different odds excluding DNPs
        hits   = played_df['hit_hr'].eq('Yes').sum()
        total  = len(played_df)
        losses = total - hits
        print(f"\n=== ADJUSTED ROI (DNPs removed, avg +534 odds) ===")
        avg_odds = 534
        profit   = round(hits * (avg_odds / 100) - losses, 2)
        roi      = round(profit / total * 100, 1)
        print(f"Net profit: {'+' if profit >= 0 else ''}{profit} units")
        print(f"ROI:        {'+' if roi >= 0 else ''}{roi}%")

    # Write results back to Picks_Log with a new 'played' column
    picks_df["played"] = ""
    for idx in played_rows:
        orig_idx = resolved.loc[idx].name
        picks_df.at[orig_idx, "played"] = "Yes"
    for idx in dnp_rows:
        orig_idx = resolved.loc[idx].name
        picks_df.at[orig_idx, "played"] = "No"

    ws.clear()
    values = [picks_df.columns.tolist()] + picks_df.astype(str).values.tolist()
    ws.update(values)
    print("\nPicks_Log updated with 'played' column.")


if __name__ == "__main__":
    main()
