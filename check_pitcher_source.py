#!/usr/bin/env python3
"""
check_pitcher_source.py

Traces the pitcher barrel%-allowed-by-hand data to its SOURCE. If the
platoon score is dead, the cause is one of:
  (a) the column exists in Pitcher_Statcast_2026 but isn't merged into
      HR_All_Scores, or
  (b) it's never collected at all.
This prints the Pitcher_Statcast_2026 columns and how populated the
by-hand barrel fields are, so we know which.

USAGE  (live only — needs the sheet)
    python3 check_pitcher_source.py
"""

import os, json
import pandas as pd


def main():
    import gspread
    from google.oauth2.service_account import Credentials
    sc = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
    cr = Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=sc)
    gc = gspread.authorize(cr)
    sh = gc.open_by_key(os.environ["GOOGLE_SHEET_ID"])

    ws = sh.worksheet("Pitcher_Statcast_2026")
    vals = ws.get_all_values()
    df = pd.DataFrame(vals[1:], columns=vals[0]).fillna("")
    print(f"Pitcher_Statcast_2026: {len(df)} rows, {len(df.columns)} cols\n")

    print("All columns:")
    for c in df.columns:
        print(f"  {c}")

    print("\nColumns mentioning barrel / hand / vs / lhh / rhh:")
    hits = [c for c in df.columns
            if any(k in c.lower() for k in ["barrel", "lhh", "rhh", "vs_l", "vs_r", "_hand", "platoon"])]
    if not hits:
        print("  NONE — by-hand barrel data is not collected in this tab.")
    for c in hits:
        pop = (df[c].astype(str).str.strip().replace("0", "") != "").mean() * 100
        print(f"  {c:<34} {pop:5.0f}% populated")


if __name__ == "__main__":
    main()
