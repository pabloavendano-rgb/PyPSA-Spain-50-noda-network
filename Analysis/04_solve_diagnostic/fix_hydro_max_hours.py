#!/usr/bin/env python3
"""Fix hydro max_hours to realistic levels (~23 TWh total energy).

The original network has 55.42 TWh of hydro energy (2.4× real Spanish reservoir
capacity of ~23 TWh). This script scales max_hours proportionally so total
hydro energy ≈ 23 TWh, then updates state_of_charge_initial to 50%.

Usage:
    pixi run python3 Analysis/04_solve_diagnostic/fix_hydro_max_hours.py

This modifies the network file in-place. A backup is created automatically.
"""
import pypsa, pandas as pd, numpy as np, shutil, os, warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NET_PATH = f'{ROOT}/resources/networks/50n_ES_FR_PT.nc'
BACKUP_PATH = NET_PATH.replace('.nc', '_pre_hydrofix.nc')

# ── Backup ──────────────────────────────────────────────────────────────
if not os.path.exists(BACKUP_PATH):
    shutil.copy2(NET_PATH, BACKUP_PATH)
    print(f'Backup: {BACKUP_PATH}')

n = pypsa.Network(NET_PATH)
print(f'Loaded: {len(n.buses)} buses, {len(n.storage_units)} storage units')

# ── Identify ES hydro storage units ────────────────────────────────────
es_hyd = (n.storage_units.bus.astype(str).str.startswith('ES')) & \
         (n.storage_units.carrier == 'hydro')

current_energy = (n.storage_units.loc[es_hyd, 'p_nom'] * 
                  n.storage_units.loc[es_hyd, 'max_hours']).sum()
print(f'\nCurrent hydro energy: {current_energy/1e6:.2f} TWh')
print(f'Target: ~23 TWh (real Spanish reservoir capacity)')

# ── Strategy: proportional scaling to hit 23 TWh ──────────────────────
# This preserves the relative distribution between units while bringing
# total energy in line with reality.
TARGET_TWH = 23.0
scale = TARGET_TWH * 1e6 / current_energy

old_max_hours = n.storage_units.loc[es_hyd, 'max_hours'].copy()
n.storage_units.loc[es_hyd, 'max_hours'] *= scale

new_energy = (n.storage_units.loc[es_hyd, 'p_nom'] * 
              n.storage_units.loc[es_hyd, 'max_hours']).sum()
print(f'Scale factor: {scale:.4f}')
print(f'New hydro energy: {new_energy/1e6:.2f} TWh')

# ── Update state_of_charge_initial to 50% ─────────────────────────────
n.storage_units.loc[es_hyd, 'state_of_charge_initial'] = \
    0.50 * n.storage_units.loc[es_hyd, 'p_nom'] * \
           n.storage_units.loc[es_hyd, 'max_hours']

# ── Report changes ────────────────────────────────────────────────────
print(f'\n{"Unit":30s} {"p_nom":8s} {"Old_mh":8s} {"New_mh":8s} {"Old_En":8s} {"New_En":8s}')
print(f'{"-"*30} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*8}')
for u in n.storage_units.index[es_hyd]:
    r = n.storage_units.loc[u]
    old_mh = old_max_hours[u]
    new_mh = r.max_hours
    old_en = r.p_nom * old_mh / 1e6
    new_en = r.p_nom * new_mh / 1e6
    print(f'{u:30s} {r.p_nom:8.0f} {old_mh:8.0f} {new_mh:8.0f} {old_en:8.3f} {new_en:8.3f}')

# ── Save ──────────────────────────────────────────────────────────────
n.export_to_netcdf(NET_PATH)
print(f'\nSaved: {NET_PATH}')
print('Done.')
