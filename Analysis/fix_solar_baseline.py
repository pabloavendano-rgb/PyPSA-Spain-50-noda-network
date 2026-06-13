#!/usr/bin/env python3
"""
Fix solar ERA5 baseload artifact in the production network file.

ERA5 reanalysis carries a tiny constant diffuse-radiation offset that makes
p_max_pu non-zero during hours that are always dark in Spain. This shows up
as solar producing at night in the dispatch model.

Algorithm (per ES solar node):
  1. Sample anchor dark hours (UTC 02:00 and 03:00 — always dark year-round in Spain)
  2. Identify the floor value (mode of anchor-hour values) → "baseload"
  3. Zero every timestep where p_max_pu <= baseload
  4. Leave daytime values (p_max_pu > baseload) unchanged

Only Spanish solar generators are touched. FR/PT are left as-is.

Run from repo root:
    pixi run python Analysis/fix_solar_baseline.py
"""

import warnings
import pypsa
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

NET_PATH = "resources/networks/base_s_50_elec_2704_fixed.nc"
ANCHOR_HOURS = [2, 3]  # UTC hours guaranteed dark all year in Spain

print(f"Loading: {NET_PATH}")
n = pypsa.Network(NET_PATH)
print(f"  Buses: {len(n.buses)}  Generators: {len(n.generators)}  Snapshots: {len(n.snapshots)}")

# --- identify Spanish solar generators with a time series ----------------
solar_es = n.generators[
    (n.generators.carrier == "solar") &
    (n.generators.bus.str.startswith("ES"))
].index
solar_es_ts = [g for g in solar_es if g in n.generators_t.p_max_pu.columns]
print(f"  ES solar generators with time series: {len(solar_es_ts)}")

# --- diagnostic header ---------------------------------------------------
print(f"\n{'Generator':<32} {'baseload':>10} {'zeroed':>8} {'total':>7} {'%':>6}")
print("-" * 66)

total_zeroed = 0
total_cells = 0
nodes_fixed = 0
nodes_clean = 0

anchor_mask = n.snapshots.hour.isin(ANCHOR_HOURS)

for g in solar_es_ts:
    ts = n.generators_t.p_max_pu[g]
    anchor_vals = ts[anchor_mask]

    unique_anchor = anchor_vals.unique()
    baseload = float(anchor_vals.mode().iloc[0])

    n_cells = len(ts)
    total_cells += n_cells

    if baseload == 0.0:
        nodes_clean += 1
        print(f"  {g:<30} {'0 (clean)':>10}")
        continue

    mask = ts <= baseload
    n_hit = int(mask.sum())
    total_zeroed += n_hit
    nodes_fixed += 1

    n.generators_t.p_max_pu.loc[mask, g] = 0.0
    pct = n_hit / n_cells * 100
    print(f"  {g:<30} {baseload:>10.5f} {n_hit:>8d} {n_cells:>7d} {pct:>5.1f}%")

    if len(unique_anchor) > 2:
        print(f"    ^ WARNING: {len(unique_anchor)} unique values at anchor hours (expected 1)")

print("-" * 66)
if total_cells > 0:
    print(f"  {'TOTAL':<30} {'':>10} {total_zeroed:>8d} {total_cells:>7d} {total_zeroed/total_cells*100:>5.1f}%")
print(f"\n  Nodes fixed: {nodes_fixed}   Nodes already clean: {nodes_clean}")

# --- verify --------------------------------------------------------------
print("\nVerifying anchor hours after fix...")
fail = False
for g in solar_es_ts:
    max_anchor = float(n.generators_t.p_max_pu.loc[anchor_mask, g].max())
    if max_anchor > 0.0:
        print(f"  FAIL {g}: max at anchor hours = {max_anchor:.6f}")
        fail = True
if not fail:
    print("  All ES solar nodes: p_max_pu == 0 at UTC 02:00 and 03:00")

# --- save ----------------------------------------------------------------
if nodes_fixed > 0 or not fail:
    print(f"\nSaving to: {NET_PATH}")
    n.export_to_netcdf(NET_PATH)
    print("Done.")
else:
    print("\nNo changes needed — file not overwritten.")
