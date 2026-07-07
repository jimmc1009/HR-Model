"""
test_battracking.py
Stage 1: can we even get Statcast bat-tracking data (squared-up%, blast%,
bat speed) in this environment via pybaseball?
Stage 2 (only if stage 1 works): do these metrics separate HR hitters
INDEPENDENTLY of barrel% — i.e. add real signal, or just echo barrel?

Bat-tracking data starts 2024 and lives in a separate Statcast leaderboard,
not the pitch-by-pitch pull. This probes availability first, then tests value.
Read-only. Nothing changes in the model.
"""
import sys

print("="*62)
print("STAGE 1: bat-tracking data availability")
print("="*62)

try:
    import pybaseball as pyb
    print("  ✓ pybaseball imported")
except Exception as e:
    print(f"  ✗ pybaseball import failed: {e}")
    sys.exit(0)

# Look for a bat-tracking / swing leaderboard function
candidates = [x for x in dir(pyb) if any(k in x.lower() for k in
              ("bat_track","swing","statcast_batter_exp","batting"))]
print(f"  Candidate functions: {candidates}")

# Try the known bat-tracking leaderboard endpoint
got = None
for fn_name in ["statcast_batter_bat_tracking", "bat_tracking", "statcast_bat_tracking"]:
    if hasattr(pyb, fn_name):
        try:
            fn = getattr(pyb, fn_name)
            df = fn(2025)
            got = (fn_name, df)
            break
        except Exception as e:
            print(f"  {fn_name} exists but failed: {e}")

if got is None:
    # Fallback: try the leaderboard module directly
    try:
        from pybaseball import statcast
        print("\n  No dedicated bat-tracking function found in this pybaseball")
        print("  version. Bat-tracking (squared-up/blast/bat speed) may need a")
        print("  direct pull from baseballsavant leaderboard CSV instead.")
        print("  → Data source would need to be ADDED to the pipeline first.")
    except Exception as e:
        print(f"  fallback failed: {e}")
    print("\n  VERDICT: bat-tracking not directly available via pybaseball here.")
    print("  To use blast%/squared-up, you'd add a Savant leaderboard fetch")
    print("  (savant.mlb.com/leaderboard/bat-tracking, CSV export) to the")
    print("  pipeline, then join by player+season. Doable but it's new plumbing.")
    print("\nDone.")
    sys.exit(0)

fn_name, df = got
print(f"\n  ✓ Got bat-tracking data via {fn_name}: {len(df)} rows")
print(f"  Columns: {list(df.columns)[:20]}")

# find the relevant columns
cols = {c.lower(): c for c in df.columns}
def find(*keys):
    for k in keys:
        for lc, orig in cols.items():
            if k in lc: return orig
    return None
blast_c   = find("blast")
squared_c = find("squared")
speed_c   = find("bat_speed","swing_speed","avg_bat_speed")
name_c    = find("name","player")

print(f"  blast col: {blast_c} | squared col: {squared_c} | speed col: {speed_c}")
print("\n  STAGE 2 requires joining these to resolved HR outcomes by player.")
print("  If the columns above are populated, next step is a join test — say")
print("  the word and I'll build the separation-vs-barrel analysis.")
print("\nDone.")
