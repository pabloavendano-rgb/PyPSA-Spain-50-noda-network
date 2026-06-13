#!/usr/bin/env python3
"""Build network_interrogation_no_ic.ipynb — hard-coded MCs, ramping, no expansion, no interconnectors, solve + real-world comparison."""
import json, os

cells = []
def md(s): cells.append({"cell_type":"markdown","id":f"md_{len(cells)}","metadata":{},"source":[s]})
def code(s, o=None):
    cells.append({"cell_type":"code","execution_count":None,"id":f"code_{len(cells)}","metadata":{},"outputs":o or [],
                  "source":[s]})

md("""# Network Interrogation — No Interconnectors

**Purpose:** Self-contained script that:
1. Loads the network
2. **Hard-codes** non-linear marginal costs (so you never need to re-run `non_linear_MCs.ipynb`)
3. **Hard-codes** ramping adjustments (so you never need to re-run `07_ramping_adjustments.ipynb`)
4. Verifies hydro marginal cost (opportunity cost) is correct
5. Disables ALL expansion (grid + capacity)
6. **Removes interconnectors** (sets cross-border link `p_nom` to 0)
7. Solves dispatch-only
8. Compares dispatch and price time series to real-world 2024 data

**Workflow:** `Kernel → Restart & Run All`.""")

md("---\n## Section 0 — Parameters  *(edit here)*")

code("""# ================================================================
#  EDIT THESE, then:  Kernel → Restart & Run All
# ================================================================

RANDOM_SEED = 42        # reproducible jitter

# ── VRE floor ────────────────────────────────────────────────────
SOLAR_MC      = 0.10    # solar
WIND_MC       = 0.50    # onwind, offwind, offwind-float

# ── Nuclear  ─────────────────────────────────────────────────────
NUC_MC_LO     = 12.0    # lower bound for per-reactor jitter
NUC_MC_HI     = 18.0    # upper bound
NUC_PMIN      = 0.50    # must-run fraction for all nuclear units

# ── Hydro opportunity costs  (EUR/MWh) ───────────────────────────
ES_HYDRO_OC   = 28.0    # Iberian reservoirs — moderate drought risk
FR_HYDRO_OC   = 22.0    # Large Alpine reservoirs — lower scarcity
PT_HYDRO_OC   = 35.0    # Iberian watershed — higher drought exposure

# ── Biomass (flat) ───────────────────────────────────────────────
BIOMASS_MC    = 40.0

# ── Cogeneration (PT) ────────────────────────────────────────────
COGEN_MC      = 45.0    # industrial CHP, must-run due to heat demand
COGEN_PMIN    = 0.70    # minimum dispatch fraction

# ── CCGT — ES / PT fleet (modern LNG) ────────────────────────────
CCGT_IBERIA_T1 = (52.0, 62.0)   # Tier 1: high efficiency, cheap gas
CCGT_IBERIA_T2 = (62.0, 72.0)   # Tier 2: standard fleet
CCGT_IBERIA_T3 = (72.0, 82.0)   # Tier 3: older / low efficiency

# ── CCGT — FR fleet (aged pipeline-gas) ──────────────────────────
CCGT_FR_T1     = (68.0, 78.0)   # Tier 1 (relative to FR fleet)
CCGT_FR_T2     = (78.0, 88.0)   # Tier 2

# ── Coal (EU-ETS at ~€65/tCO₂ baked in) ─────────────────────────
COAL_MC_LO    = 112.0
COAL_MC_HI    = 118.0

# ── Peakers ──────────────────────────────────────────────────────
OCGT_MC       = 125.0   # PT OCGT
OIL_MC        = 180.0   # FR Fioul turbines

# ── Ramp limits (fraction of p_nom per hour) ─────────────────────
NUC_RAMP_UP   = 0.20
NUC_RAMP_DN   = 0.20
BIO_RAMP_UP   = 0.30
BIO_RAMP_DN   = 0.30
COG_RAMP_UP   = 0.30
COG_RAMP_DN   = 0.30
CCGT_IBERIA_T1_RAMP = (0.80, 0.80)
CCGT_IBERIA_T2_RAMP = (0.65, 0.65)
CCGT_IBERIA_T3_RAMP = (0.50, 0.50)
CCGT_FR_T1_RAMP     = (0.65, 0.65)
CCGT_FR_T2_RAMP     = (0.50, 0.50)
COAL_RAMP_UP  = 0.40
COAL_RAMP_DN  = 0.40
OCGT_RAMP_UP  = 1.00
OCGT_RAMP_DN  = 1.00
OIL_RAMP_UP   = 0.90
OIL_RAMP_DN   = 0.90

# ── Solve period ─────────────────────────────────────────────────
START_DATE = '2024-04-01'
N_DAYS = 91
END_DATE = pd.Timestamp(START_DATE) + pd.Timedelta(days=N_DAYS) - pd.Timedelta(hours=1)
PERIOD_LABEL = f'{START_DATE} ({N_DAYS} days)'
SAVE_SUFFIX = f'{START_DATE}_{N_DAYS}d_no_ic'

# ── Paths ────────────────────────────────────────────────────────
ROOT     = '../..'
NET_PATH = f'{ROOT}/resources/networks/50n_ES_FR_PT.nc'
OMIE_PATH = f'{ROOT}/Analysis/data/Spain_prices.csv'
REE_PATH  = f'{ROOT}/Analysis/data/daily_gen_spain.csv'

CARRIER_COLORS = {
    'solar': '#f9d002', 'onwind': '#235ebc', 'offwind': '#074ede',
    'offwind-float': '#b5e2fa', 'hydro': '#298c81', 'ror': '#3dbfb0',
    'PHS': '#51dbcc', 'nuclear': '#ff8c00', 'CCGT': '#a85522',
    'OCGT': '#d49a6a', 'coal': '#545454', 'biomass': '#baa741',
    'battery': '#e2ff7c', 'oil': '#c9c9c9',
}
COUNTRY_PREFIX = {'ES': 'Spain', 'FR': 'France', 'PT': 'Portugal'}

print('Parameters loaded.')""")

md("---\n## Section 1 — Setup & Load Network")

code("""import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pypsa
import os

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10})
rng = np.random.default_rng(RANDOM_SEED)

pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.2f}'.format)

n = pypsa.Network(NET_PATH)
print(f'Network loaded: {len(n.buses)} buses, {len(n.generators)} generators, '
      f'{len(n.storage_units)} storage_units, {len(n.snapshots)} snapshots')

# ── Country index helpers ─────────────────────────────────────────
bus_cc  = pd.Series(n.buses.index.astype(str).str[:2].values, index=n.buses.index)
gen_cc  = n.generators.bus.map(bus_cc)
su_cc   = n.storage_units.bus.map(bus_cc)

# ── Baseline snapshot ─────────────────────────────────────────────
print('\\n=== BASELINE (before any changes) ===')
for cc in ['ES', 'FR', 'PT']:
    for car in n.generators.loc[gen_cc == cc, 'carrier'].unique():
        m = (gen_cc == cc) & (n.generators['carrier'] == car)
        sub = n.generators.loc[m, ['marginal_cost', 'p_min_pu', 'p_nom']]
        mc_str = f'{sub["marginal_cost"].min():.1f}–{sub["marginal_cost"].max():.1f}'
        print(f'  {cc} {car:14s}: n={m.sum():3d}  MC={mc_str:>14}  '
              f'p_min={sub["p_min_pu"].mean():.2f}  '
              f'cap={sub["p_nom"].sum():.0f} MW')""")

md("---\n## Section 2 — Hard-Code Non-Linear Marginal Costs")

md("### 2.1  VRE floor — solar, wind, offwind")

code("""SOLAR_CARRIERS  = ['solar']
WIND_CARRIERS   = ['onwind', 'offwind', 'offwind-float']

for car in SOLAR_CARRIERS:
    mask = n.generators['carrier'] == car
    if mask.sum():
        n.generators.loc[mask, 'marginal_cost'] = SOLAR_MC
        print(f'  {car:15s}: n={mask.sum():3d}  MC → {SOLAR_MC}')

for car in WIND_CARRIERS:
    mask = n.generators['carrier'] == car
    if mask.sum():
        n.generators.loc[mask, 'marginal_cost'] = WIND_MC
        print(f'  {car:15s}: n={mask.sum():3d}  MC → {WIND_MC}')

print(f'  {"ror":15s}: left at 0.0  (must-dispatch, no fuel cost)')""")

md("### 2.2  Nuclear — per-reactor jitter + must-run")

code("""nuc_mask = n.generators['carrier'] == 'nuclear'
n_nuc    = nuc_mask.sum()

jitter = rng.uniform(NUC_MC_LO, NUC_MC_HI, size=n_nuc)
n.generators.loc[nuc_mask, 'marginal_cost'] = jitter
n.generators.loc[nuc_mask, 'p_min_pu']      = NUC_PMIN

if n_nuc:
    print(f'Nuclear: {n_nuc} units  →  MC range {jitter.min():.2f}–{jitter.max():.2f} EUR/MWh  '
          f'p_min={NUC_PMIN}')
else:
    print('Nuclear: 0 units  (no nuclear generators found)')
print()
for cc in ['ES', 'FR']:
    m = nuc_mask & (gen_cc == cc)
    if m.sum():
        print(f'  {cc} nuclear generators:')
        print(n.generators.loc[m, ['p_nom', 'marginal_cost', 'p_min_pu']].round(2).to_string())""")

md("### 2.3  Biomass & Cogeneration")

code("""# ── Biomass ───────────────────────────────────────────────────────
bio_mask = n.generators['carrier'] == 'biomass'
n.generators.loc[bio_mask, 'marginal_cost'] = BIOMASS_MC
print(f'Biomass  : n={bio_mask.sum()}  MC → {BIOMASS_MC}')

# ── Cogeneration (PT only in this network) ────────────────────────
cog_mask = n.generators['carrier'] == 'cogen'
n.generators.loc[cog_mask, 'marginal_cost'] = COGEN_MC
n.generators.loc[cog_mask, 'p_min_pu']      = COGEN_PMIN
print(f'Cogen    : n={cog_mask.sum()}  MC → {COGEN_MC}  p_min → {COGEN_PMIN}')""")

md("### 2.4  Hydro — opportunity cost")

code("""# ── ES hydro reservoirs (storage units) ──────────────────────────
es_hyd = (su_cc == 'ES') & (n.storage_units['carrier'] == 'hydro')
n.storage_units.loc[es_hyd, 'marginal_cost'] = ES_HYDRO_OC
print(f'ES hydro (storage): n={es_hyd.sum():2d}  '
      f'cap={n.storage_units.loc[es_hyd, "p_nom"].sum():.0f} MW  MC → {ES_HYDRO_OC}')

# ES PHS: leave at 0 — pure pumped storage, arbitrage via efficiency
es_phs = (su_cc == 'ES') & (n.storage_units['carrier'] == 'PHS')
print(f'ES PHS  (storage): n={es_phs.sum():2d}  '
      f'cap={n.storage_units.loc[es_phs, "p_nom"].sum():.0f} MW  MC left at 0.0  (arbitrage unit)')

# ── FR / PT hydro (generators) ────────────────────────────────────
for cc, oc in [('FR', FR_HYDRO_OC), ('PT', PT_HYDRO_OC)]:
    m = (gen_cc == cc) & (n.generators['carrier'] == 'hydro')
    n.generators.loc[m, 'marginal_cost'] = oc
    cap = n.generators.loc[m, 'p_nom'].sum()
    print(f'{cc} hydro (generator): n={m.sum():2d}  cap={cap:.0f} MW  MC → {oc}')""")

md("### 2.5  CCGT — Three-tier efficiency ladder")

code("""def assign_ccgt_tiers(country_code, tier_ranges):
    mask = (gen_cc == country_code) & (n.generators['carrier'] == 'CCGT')
    idx  = n.generators.loc[mask].sort_values('p_nom', ascending=False).index
    n_units = len(idx)
    if n_units == 0:
        return
    n_tiers = len(tier_ranges)
    splits  = np.array_split(idx, n_tiers)
    for tier_idx, (split_idx, (lo, hi)) in enumerate(zip(splits, tier_ranges), start=1):
        mcs = rng.uniform(lo, hi, size=len(split_idx))
        n.generators.loc[split_idx, 'marginal_cost'] = mcs
        print(f'  {country_code} CCGT Tier {tier_idx} '
              f'(n={len(split_idx)}, MC {lo:.0f}–{hi:.0f}): '
              f'assigned {mcs.min():.1f}–{mcs.max():.1f}')
    total_cap = n.generators.loc[mask, 'p_nom'].sum()
    print(f'  {country_code} CCGT total: {n_units} units, {total_cap:.0f} MW')
    print()

print('ES CCGT:')
assign_ccgt_tiers('ES', [CCGT_IBERIA_T1, CCGT_IBERIA_T2, CCGT_IBERIA_T3])
print('PT CCGT:')
assign_ccgt_tiers('PT', [CCGT_IBERIA_T1, CCGT_IBERIA_T2, CCGT_IBERIA_T3])
print('FR CCGT:')
assign_ccgt_tiers('FR', [CCGT_FR_T1, CCGT_FR_T2])""")

md("### 2.6  Coal, OCGT, Oil — price ceiling")

code("""# ── Coal (ES + FR) ────────────────────────────────────────────────
coal_mask = n.generators['carrier'] == 'coal'
n_coal    = coal_mask.sum()
coal_mcs  = rng.uniform(COAL_MC_LO, COAL_MC_HI, size=n_coal)
n.generators.loc[coal_mask, 'marginal_cost'] = coal_mcs
if n_coal:
    print(f'Coal  : n={n_coal}  MC → {coal_mcs.min():.1f}–{coal_mcs.max():.1f} (range {COAL_MC_LO}–{COAL_MC_HI})')
else:
    print(f'Coal  : n={n_coal}  (no coal generators found)')

# ── OCGT (PT only) ────────────────────────────────────────────────
ocgt_mask = n.generators['carrier'] == 'OCGT'
n.generators.loc[ocgt_mask, 'marginal_cost'] = OCGT_MC
print(f'OCGT  : n={ocgt_mask.sum()}  MC → {OCGT_MC}')

# ── Oil / Fioul (FR) ──────────────────────────────────────────────
oil_mask  = n.generators['carrier'] == 'oil'
n.generators.loc[oil_mask, 'marginal_cost'] = OIL_MC
print(f'Oil   : n={oil_mask.sum()}  MC → {OIL_MC}')""")

md("---\n## Section 3 — Verify Hydro MC (Opportunity Cost)")

code("""print('=== HYDRO MC VERIFICATION ===')
print()

# ES hydro storage units
hs = n.storage_units[n.storage_units['carrier'] == 'hydro']
print(f'ES hydro storage units (n={len(hs)}):')
for i, r in hs.iterrows():
    status = 'OK' if abs(r.marginal_cost - ES_HYDRO_OC) < 0.5 else 'MISMATCH'
    print(f'  {i:30s} bus={r.bus:15s} MC={r.marginal_cost:8.2f} (expected {ES_HYDRO_OC}) {status}')

# ES PHS
phs = n.storage_units[n.storage_units['carrier'] == 'PHS']
print(f'\\nES PHS (n={len(phs)}): MC={phs.marginal_cost.unique()} (expected 0.0 — arbitrage unit)')

# FR/PT hydro generators
for cc, expected_oc in [('FR', FR_HYDRO_OC), ('PT', PT_HYDRO_OC)]:
    m = (gen_cc == cc) & (n.generators['carrier'] == 'hydro')
    hg = n.generators.loc[m]
    print(f'\\n{cc} hydro generators (n={len(hg)}):')
    for i, r in hg.iterrows():
        status = 'OK' if abs(r.marginal_cost - expected_oc) < 0.5 else 'MISMATCH'
        print(f'  {i:30s} bus={r.bus:15s} MC={r.marginal_cost:8.2f} (expected {expected_oc}) {status}')

print('\\n=== HYDRO MC VERIFICATION COMPLETE ===')""")

md("---\n## Section 4 — Hard-Code Ramping Adjustments")

md("### 4.1  Nuclear")

code("""nuc = n.generators['carrier'] == 'nuclear'
n.generators.loc[nuc, 'ramp_limit_up']   = NUC_RAMP_UP
n.generators.loc[nuc, 'ramp_limit_down'] = NUC_RAMP_DN
print(f'Nuclear: n={nuc.sum()}  ramp_up={NUC_RAMP_UP}  ramp_dn={NUC_RAMP_DN}')""")

md("### 4.2  Biomass & Cogeneration")

code("""bio = n.generators['carrier'] == 'biomass'
n.generators.loc[bio, 'ramp_limit_up']   = BIO_RAMP_UP
n.generators.loc[bio, 'ramp_limit_down'] = BIO_RAMP_DN
print(f'Biomass: n={bio.sum()}  ramp_up={BIO_RAMP_UP}  ramp_dn={BIO_RAMP_DN}')

cog = n.generators['carrier'] == 'cogen'
n.generators.loc[cog, 'ramp_limit_up']   = COG_RAMP_UP
n.generators.loc[cog, 'ramp_limit_down'] = COG_RAMP_DN
print(f'Cogen  : n={cog.sum()}  ramp_up={COG_RAMP_UP}  ramp_dn={COG_RAMP_DN}')""")

md("### 4.3  CCGT — tiered by p_nom rank")

code("""def assign_ccgt_ramp_tiers(country_code, tier_rates):
    mask = (gen_cc == country_code) & (n.generators['carrier'] == 'CCGT')
    idx  = n.generators.loc[mask].sort_values('p_nom', ascending=False).index
    if len(idx) == 0:
        return
    splits = np.array_split(idx, len(tier_rates))
    for tier_num, (split_idx, (ru, rd)) in enumerate(zip(splits, tier_rates), start=1):
        n.generators.loc[split_idx, 'ramp_limit_up']   = ru
        n.generators.loc[split_idx, 'ramp_limit_down'] = rd
        print(f'  {country_code} CCGT Tier {tier_num} '
              f'(n={len(split_idx)}, p_nom {n.generators.loc[split_idx, "p_nom"].min():.0f}–'
              f'{n.generators.loc[split_idx, "p_nom"].max():.0f} MW): '
              f'ramp {ru}/{rd}')

print('ES CCGT:')
assign_ccgt_ramp_tiers('ES', [CCGT_IBERIA_T1_RAMP, CCGT_IBERIA_T2_RAMP, CCGT_IBERIA_T3_RAMP])
print('PT CCGT:')
assign_ccgt_ramp_tiers('PT', [CCGT_IBERIA_T1_RAMP, CCGT_IBERIA_T2_RAMP, CCGT_IBERIA_T3_RAMP])
print('FR CCGT:')
assign_ccgt_ramp_tiers('FR', [CCGT_FR_T1_RAMP, CCGT_FR_T2_RAMP])""")

md("### 4.4  Coal, OCGT, Oil")

code("""coal = n.generators['carrier'] == 'coal'
n.generators.loc[coal, 'ramp_limit_up']   = COAL_RAMP_UP
n.generators.loc[coal, 'ramp_limit_down'] = COAL_RAMP_DN
print(f'Coal: n={coal.sum()}  ramp_up={COAL_RAMP_UP}  ramp_dn={COAL_RAMP_DN}')

ocgt = n.generators['carrier'] == 'OCGT'
n.generators.loc[ocgt, 'ramp_limit_up']   = OCGT_RAMP_UP
n.generators.loc[ocgt, 'ramp_limit_down'] = OCGT_RAMP_DN
print(f'OCGT: n={ocgt.sum()}  ramp_up={OCGT_RAMP_UP}  ramp_dn={OCGT_RAMP_DN}')

oil = n.generators['carrier'] == 'oil'
n.generators.loc[oil, 'ramp_limit_up']   = OIL_RAMP_UP
n.generators.loc[oil, 'ramp_limit_down'] = OIL_RAMP_DN
print(f'Oil : n={oil.sum()}  ramp_up={OIL_RAMP_UP}  ramp_dn={OIL_RAMP_DN}')

print('\\nVRE / hydro / ror: left as NaN (unconstrained).')""")

md("### 4.5  Set hydro initial state of charge to 50%")

code("""# ── Set hydro initial state of charge to 50% ──────────────────────
es_hyd_soc = (n.storage_units.bus.map(lambda b: str(b).startswith('ES')) &
              (n.storage_units['carrier'] == 'hydro'))
n.storage_units.loc[es_hyd_soc, 'state_of_charge_initial'] = 0.50 * n.storage_units.loc[es_hyd_soc, 'p_nom'] * n.storage_units.loc[es_hyd_soc, 'max_hours']
print(f'  state_of_charge_initial set to 50% for {es_hyd_soc.sum()} ES hydro reservoirs')""")

md("---\n## Section 5 — Verification Table (MCs + Ramp Limits)")

code("""print('=== Updated network summary ===')
summary = (
    n.generators
    .assign(country=gen_cc)
    .groupby(['country', 'carrier'])[['marginal_cost', 'p_min_pu', 'p_nom', 'ramp_limit_up', 'ramp_limit_down']]
    .agg(
        n=('marginal_cost', 'count'),
        MC_min=('marginal_cost', 'min'),
        MC_max=('marginal_cost', 'max'),
        p_min=('p_min_pu', 'mean'),
        cap_MW=('p_nom', 'sum'),
        ramp_up=('ramp_limit_up', 'mean'),
        ramp_dn=('ramp_limit_down', 'mean'),
    )
    .round(2)
)
print(summary.to_string())

print('\\n=== Storage units ===')
su_summary = (
    n.storage_units
    .assign(country=n.storage_units.bus.map(bus_cc))
    .groupby(['country', 'carrier'])[['marginal_cost', 'p_nom']]
    .agg(
        n=('marginal_cost', 'count'),
        MC_min=('marginal_cost', 'min'),
        MC_max=('marginal_cost', 'max'),
        cap_MW=('p_nom', 'sum'),
    )
    .round(2)
)
print(su_summary.to_string())""")

md("---\n## Section 6 — Disable ALL Expansion")

code("""print('Disabling ALL expansion...')
n.generators['p_nom_extendable'] = False
n.links['p_nom_extendable'] = False
n.storage_units['p_nom_extendable'] = False
n.stores['e_nom_extendable'] = False
n.lines['s_nom_extendable'] = False

n_extendable = n.generators['p_nom_extendable'].sum()
n_links_ext = n.links['p_nom_extendable'].sum()
n_stores_ext = n.stores['e_nom_extendable'].sum()
print(f'Extendable generators: {n_extendable} | links: {n_links_ext} | stores: {n_stores_ext}')
assert n_extendable == 0, 'Some generators are still extendable!'
assert n_links_ext == 0, 'Some links are still extendable!'
assert n_stores_ext == 0, 'Some stores are still extendable!'
print('ALL expansion disabled.')""")

md("---\n## Section 7 — Remove Interconnectors")

code("""# Identify cross-border interconnectors (DC_ic export/import carriers)
ic_links = n.links[n.links['carrier'].str.contains('DC_ic', na=False)]
print(f'Interconnector links found: {len(ic_links)}')
for i, r in ic_links.iterrows():
    print(f'  {i:30s} {r.bus0:15s} → {r.bus1:15s}  p_nom={r.p_nom:8.0f} MW  carrier={r.carrier}')

# Set p_nom to 0 to disable all cross-border trade
n.links.loc[ic_links.index, 'p_nom'] = 0
print(f'\\nAll {len(ic_links)} interconnector links set to p_nom=0 — NO cross-border trade.')

# Verify
ic_after = n.links[n.links['carrier'].str.contains('DC_ic', na=False)]
print(f'Verification: {len(ic_after)} interconnectors, total capacity = {ic_after.p_nom.sum():.0f} MW')""")

md("---\n## Section 8 — Slice & Solve Period")

code("""start_time = pd.Timestamp(START_DATE)
end_time = pd.Timestamp(END_DATE)
period_mask = (n.snapshots >= start_time) & (n.snapshots <= end_time)
period_snapshots = n.snapshots[period_mask]
n_snapshots = len(period_snapshots)

print(f'Selected period: {start_time} to {end_time}')
print(f'Snapshots to solve: {n_snapshots} ({n_snapshots/24:.1f} days)')

n_sub = n.copy()
n_sub.set_snapshots(period_snapshots)

print(f'Solving {len(n_sub.snapshots)} snapshots with Gurobi (5 cores, 2h limit)...')

# --- Disable all capacity expansion (dispatch-only) ---
for attr in ['generators', 'links', 'stores', 'storage_units']:
    df = getattr(n, attr)
    if 'p_nom_extendable' in df.columns:
        df.loc[df.p_nom_extendable, 'p_nom_extendable'] = False
if 's_nom_extendable' in n.lines.columns:
    n.lines.loc[n.lines.s_nom_extendable, 's_nom_extendable'] = False

# ── TRANSMISSION CONSTRAINT: Break the copper plate ────────────────────
TRANS_FACTOR = 0.35
es_lines = n.lines[
    n.lines.bus0.astype(str).str[:2].isin(['ES']) &
    n.lines.bus1.astype(str).str[:2].isin(['ES'])
].index
print(f'Scaling {len(es_lines)} ES-ES lines by {TRANS_FACTOR}\u00d7')
n.lines.loc[es_lines, 's_nom'] *= TRANS_FACTOR
n.lines.loc[es_lines, 's_nom_extendable'] = False
n.lines['s_max_pu'] = 0.5
print(f's_max_pu set to {n.lines.s_max_pu.iloc[0]} for all lines')

# --- Set hydro state_of_charge_initial to 50% ---
es_hyd = (n.storage_units.bus.map(lambda b: str(b).startswith('ES')) &
          (n.storage_units['carrier'] == 'hydro'))
n.storage_units.loc[es_hyd, 'state_of_charge_initial'] = (
    0.50 * n.storage_units.loc[es_hyd, 'p_nom'] * n.storage_units.loc[es_hyd, 'max_hours']
)
print(f'Hydro SOC init set to 50% for {es_hyd.sum()} units')

# --- Border Lockdown: isolate Spain from FR/PT ---
cross_border_links = n.links[
    (n.links.bus0.str.contains('ES') & ~n.links.bus1.str.contains('ES')) |
    (~n.links.bus0.str.contains('ES') & n.links.bus1.str.contains('ES'))
].index
n.links.loc[cross_border_links, 'p_nom'] = 0
n.links.loc[cross_border_links, 'p_nom_extendable'] = False
print(f'Disabled {len(cross_border_links)} cross-border links')

res = n_sub.optimize(solver_name='gurobi',
                     solver_options={'OutputFlag': 1, 'TimeLimit': 7200, 'Threads': 5})
print(f'Status: {res[0]} | {res[1]}')
print(f'Objective: {n_sub.objective:.2f} EUR')""")

md("---\n## Section 9 — Save Solved Network")

code("""save_dir = f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/solved_{SAVE_SUFFIX}.nc'
n_sub.export_to_netcdf(save_path)
print(f'Solved network saved to: {save_path}')""")

md("---\n## Section 10 — Dispatch Stack (Model)")

code("""def get_country(bus_name):
    s = str(bus_name)
    if s.startswith('ES'): return 'ES'
    if s.startswith('FR'): return 'FR'
    if s.startswith('PT'): return 'PT'
    return 'OTHER'

gen_p = n_sub.generators_t.p
gen_info = n_sub.generators[['bus', 'carrier']]

gen_by_country = {}
for country_code, country_name in COUNTRY_PREFIX.items():
    country_buses = [b for b in n_sub.buses.index if str(b).startswith(country_code)]
    country_gens = gen_info.index[gen_info['bus'].isin(country_buses)]
    carriers = gen_info.loc[country_gens, 'carrier'].unique()
    carrier_data = {}
    for c in carriers:
        c_gens = country_gens[gen_info.loc[country_gens, 'carrier'] == c]
        carrier_data[c] = gen_p[c_gens].sum(axis=1)
    gen_by_country[country_code] = pd.DataFrame(carrier_data)
    print(f'{country_name}: {len(country_gens)} generators, {len(carriers)} carriers')

all_carriers = {}
for c in gen_info['carrier'].unique():
    c_gens = gen_info.index[gen_info['carrier'] == c]
    all_carriers[c] = gen_p[c_gens].sum(axis=1)
gen_total = pd.DataFrame(all_carriers)

print('\\n--- Total Generation by Carrier (GWh) ---')
summary = gen_total.sum().sort_values(ascending=False)
for carrier, val in summary.items():
    print(f'  {carrier:20s}: {val/1000:8.2f} GWh')
print(f'  {"TOTAL":20s}: {gen_total.sum().sum()/1000:8.2f} GWh')""")

md("---\n## Section 11 — Dispatch Stack Plot")

code("""fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
for idx, (cc, cn) in enumerate(COUNTRY_PREFIX.items()):
    ax = axes[idx]
    df = gen_by_country[cc]
    order = df.sum().sort_values().index
    ax.stackplot(df.index, df[order].T,
                 labels=order,
                 colors=[CARRIER_COLORS.get(c, '#cccccc') for c in order],
                 alpha=0.85)
    ax.set_ylabel('MW')
    ax.set_title(f'{cn} — Dispatch ({PERIOD_LABEL}) — NO INTERCONNECTORS')
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.show()""")

md("---\n## Section 12 — Price Diagnostics")

code("""if hasattr(n_sub, 'buses_t') and 'marginal_price' in n_sub.buses_t:
    p = n_sub.buses_t.marginal_price.copy()
else:
    raise ValueError('No marginal_price found — solve may have failed')

es_buses = [b for b in p.columns if str(b).startswith('ES')]
fr_buses = [b for b in p.columns if str(b).startswith('FR')]
pt_buses = [b for b in p.columns if str(b).startswith('PT')]
p_es = p[es_buses]
p_fr = p[fr_buses]
p_pt = p[pt_buses]

print('=== Price Statistics (EUR/MWh) ===')
stats = {}
for label, df in [('ES (all buses)', p_es), ('FR', p_fr), ('PT', p_pt)]:
    s = df.stack()
    stats[label] = {
        'mean': s.mean(), 'std': s.std(), 'min': s.min(), 'max': s.max(),
        'p5': s.quantile(0.05), 'p25': s.quantile(0.25),
        'p50': s.median(), 'p75': s.quantile(0.75), 'p95': s.quantile(0.95),
    }
    print(f'\\n{label}:')
    for k, v in stats[label].items():
        print(f'  {k:6s} = {v:8.2f}')

# ES mean price (load-weighted)
es_load = n_sub.loads_t.p_set[[c for c in n_sub.loads_t.p_set.columns if str(c).startswith('ES')]]
es_load_total = es_load.sum(axis=1)
es_price_mean = (p_es.mean(axis=1) * es_load_total).sum() / es_load_total.sum()
print(f'\\nES load-weighted mean price: {es_price_mean:.2f} EUR/MWh')
print(f'ES simple mean price:        {p_es.mean(axis=1).mean():.2f} EUR/MWh')

# ──