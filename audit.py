import os
import json
from datetime import date, timedelta

import pandas as pd
from google.oauth2.service_account import Credentials
from pybaseball import statcast

SEASON_START = "2026-03-26"

OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "double_play",
    "triple_play", "fielders_choice_out", "force_out",
    "sac_fly", "sac_fly_double_play", "sac_bunt",
    "sac_bunt_double_play", "other_out", "strikeout",
    "strikeout_double_play",
}

def main():
    end_dt = date.today() - timedelta(days=1)
    season_start = date.fromisoformat(SEASON_START)

    print("Pulling last 7 days of Statcast for audit...")
    df = statcast(
        start_dt=(end_dt - timedelta(days=7)).strftime("%Y-%m-%d"),
        end_dt=end_dt.strftime("%Y-%m-%d"),
    )
    print(f"Total rows: {len(df):,}")

    # --- Check 1: Are events only on final pitch of each AB? ---
    print("\n=== CHECK 1: Events per at-bat ===")
    event_rows = df[df["events"].notna()].copy()
    ab_cols = ["game_pk", "at_bat_number", "pitcher"]
    dupes = event_rows[event_rows.duplicated(subset=ab_cols, keep=False)]
    print(f"Rows with events: {len(event_rows):,}")
    print(f"Duplicate event rows (same game_pk+at_bat+pitcher): {len(dupes):,}")
    if len(dupes) > 0:
        print("Sample duplicates:")
        print(dupes[ab_cols + ["events", "pitch_number"]].head(10).to_string(index=False))
    else:
        print("✅ No duplicates — events only appear on final pitch of each AB")

    # --- Check 2: Out event duplication ---
    print("\n=== CHECK 2: Out events per at-bat ===")
    out_rows = df[df["events"].astype("string").str.lower().isin(OUT_EVENTS)].copy()
    out_dupes = out_rows[out_rows.duplicated(subset=ab_cols, keep=False)]
    print(f"Out event rows: {len(out_rows):,}")
    print(f"Duplicate out rows: {len(out_dupes):,}")
    if len(out_dupes) > 0:
        print("⚠️ Duplicates found in out events!")
        print(out_dupes[ab_cols + ["events", "pitch_number"]].head(10).to_string(index=False))
    else:
        print("✅ No duplicate out events")

    # --- Check 3: HR event duplication ---
    print("\n=== CHECK 3: HR events per at-bat ===")
    hr_rows = df[df["events"].astype("string").str.lower().eq("home_run")].copy()
    hr_dupes = hr_rows[hr_rows.duplicated(subset=ab_cols, keep=False)]
    print(f"HR event rows: {len(hr_rows):,}")
    print(f"Duplicate HR rows: {len(hr_dupes):,}")
    if len(hr_dupes) > 0:
        print("⚠️ Duplicates found in HR events!")
        print(hr_dupes[ab_cols + ["events", "pitch_number"]].head(10).to_string(index=False))
    else:
        print("✅ No duplicate HR events")

    # --- Check 4: BBE deduplication check ---
    print("\n=== CHECK 4: BBE rows before and after dedup ===")
    batted_ball_events = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "triple_play", "field_error", "fielders_choice",
        "fielders_choice_out", "force_out", "sac_fly",
        "sac_fly_double_play", "sac_bunt", "sac_bunt_double_play",
        "other_out",
    }
    bbe = df[
        df["events"].astype("string").str.lower().isin(batted_ball_events) &
        df["launch_speed"].notna() &
        df["launch_speed"].between(50, 120) &
        df["launch_angle"].notna() &
        df["launch_angle"].between(-90, 90)
    ].copy()
    print(f"BBE rows before dedup: {len(bbe):,}")
    batter_dedupe = ["game_pk", "at_bat_number", "batter"]
    bbe_deduped = bbe.drop_duplicates(subset=batter_dedupe)
    print(f"BBE rows after dedup: {len(bbe_deduped):,}")
    print(f"Rows removed by dedup: {len(bbe) - len(bbe_deduped):,}")
    if len(bbe) == len(bbe_deduped):
        print("✅ No BBE duplicates")
    else:
        print("⚠️ BBE duplicates exist — dedup is doing real work")

    # --- Check 5: Platoon split IP sanity check ---
    print("\n=== CHECK 5: IP sanity check for a sample pitcher ===")
    # Pick pitcher with most rows
    top_pitcher = df["pitcher"].value_counts().index[0]
    pitcher_df = df[df["pitcher"] == top_pitcher].copy()
    print(f"Sample pitcher ID: {top_pitcher}")
    print(f"Total rows for pitcher: {len(pitcher_df):,}")

    out_df = pitcher_df[
        pitcher_df["events"].astype("string").str.lower().isin(OUT_EVENTS)
    ]
    print(f"Out rows (pre-dedup): {len(out_df):,}")
    out_deduped = out_df.drop_duplicates(subset=["game_pk", "at_bat_number"])
    print(f"Out rows (post-dedup): {len(out_deduped):,}")
    print(f"Estimated IP (post-dedup): {len(out_deduped) / 3:.1f}")

    # vs RHH
    out_rhh = out_df[out_df["stand"] == "R"].drop_duplicates(subset=["game_pk", "at_bat_number"])
    out_lhh = out_df[out_df["stand"] == "L"].drop_duplicates(subset=["game_pk", "at_bat_number"])
    print(f"IP vs RHH: {len(out_rhh) / 3:.1f}")
    print(f"IP vs LHH: {len(out_lhh) / 3:.1f}")
    print(f"IP vs RHH + LHH: {(len(out_rhh) + len(out_lhh)) / 3:.1f} (should roughly equal total IP)")

    print("\n=== DONE ===")

if __name__ == "__main__":
    main()
