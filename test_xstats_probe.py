"""
test_xstats_probe.py
Probe: can we fetch Statcast EXPECTED stats (xISO, xwOBA, xSLG, xBA) via
pybaseball in this environment? These often out-predict actual stats because
they strip out luck/park. If reachable, we can test them as HR separators with
no new data plumbing.

Stage 1: does statcast_batter_expected_stats exist and return data?
Stage 2: what columns does it give (xwoba, xiso derivable, etc.)?
Also probes bat-tracking (bat speed / swing) availability for completeness.
Read-only.
"""
import sys

print("="*62); print("STATCAST EXPECTED-STATS AVAILABILITY PROBE"); print("="*62)

try:
    import pybaseball as pyb
    print("  ✓ pybaseball imported")
except Exception as e:
    print(f"  ✗ pybaseball import failed: {e}"); sys.exit(0)

# ---- Expected stats ----
print("\n-- Expected stats --")
if hasattr(pyb, "statcast_batter_expected_stats"):
    print("  ✓ statcast_batter_expected_stats EXISTS")
    try:
        df = pyb.statcast_batter_expected_stats(2025, minPA=100)
        print(f"  ✓ returned {len(df)} rows")
        print(f"  Columns: {list(df.columns)}")
        # highlight the useful ones
        useful=[c for c in df.columns if any(k in c.lower() for k in
                ("xwoba","xslg","xba","xiso","woba","slg","iso","barrel","hard","ev"))]
        print(f"  Useful for HR: {useful}")
        if len(df):
            print("  Sample row (first player):")
            print("   ", dict(list(df.iloc[0].items())[:12]))
    except Exception as e:
        print(f"  ✗ call failed: {e}")
else:
    print("  ✗ statcast_batter_expected_stats NOT in this pybaseball version")
    # look for alternatives
    alts=[x for x in dir(pyb) if "expected" in x.lower() or "xstat" in x.lower()]
    print(f"  Alternatives with 'expected': {alts}")

# ---- Percentile / bat tracking ----
print("\n-- Bat tracking (bat speed / swing) --")
bt=[x for x in dir(pyb) if any(k in x.lower() for k in ("bat_track","swing","bat_speed"))]
print(f"  Candidate functions: {bt if bt else 'none found'}")

# ---- What we can always get: statcast search (pitch-level, has launch data) ----
print("\n-- Pitch-level statcast (fallback, always available) --")
print("  statcast() gives launch_speed, launch_angle, estimated_woba_using_speedangle")
print("  per batted ball — xwOBAcon can be aggregated from that if needed.")

print("\n  VERDICT: if expected-stats columns above include xwoba/xslg/xiso,")
print("  we can test them as HR separators immediately (no new data source).")
print("  If not, they'd need a Savant leaderboard CSV fetch added to pipeline.")
print("\nDone.")
