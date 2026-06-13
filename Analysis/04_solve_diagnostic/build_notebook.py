#!/usr/bin/env python3
"""Build the 04_solve_diagnostic.ipynb notebook programmatically."""
import json, os

cells = []

def md(source):
    cells.append({
        "cell_type": "markdown",
        "id": f"md_{len(cells)}",
        "metadata": {},
        "source": [source]
    })

def code(source, outputs=None):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": f"code_{len(cells)}",
        "metadata": {},
        "outputs": outputs or [],
        "source": source.split("\n") if isinstance(source, str) else source
    })

md("# 4. Solve + Diagnostic — Dispatch Analysis\n\n**Network:** `resources/networks/50n_ES_FR_PT.nc`  \n**Purpose:** Solve a dispatch-only period (ALL expansion OFF), then run comprehensive diagnostics:\n- Marginal price setter identification\n- Price duration curves vs real OMIE data\n- Residual load analysis\n- Dispatch stack vs real REE data\n- Curtailment\n- Interconnector flows\n- Nodal price spread (is the grid copperplate?)\n\n**Workflow:** `Kernel → Restart & Run All`.\n\n---\n\n### ⚠️ CO₂ Price Note\n\n**CO₂ is NOT added to marginal costs in this solve.**  \nThe original network file has CCGT base MC = 54.60 EUR/MWh (fuel + VOM only, no CO₂).  \nIf you want to run with a CO₂ price, uncomment the relevant lines in Section 1b below.\n\nAt 65 EUR/tCO₂: `0.198 tCO₂/MWh_th / 0.58 eff × 65 EUR/t = 22.19 EUR/MWh_e` → CCGT MC = 76.79 EUR/MWh_e.")

md("---\n## Section 0 — Parameters")

code("""import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pypsa

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10})

ROOT = '../..'
NET_PATH = f'{ROOT}/resources/networks/50n_ES_FR_PT.nc'
OMIE_PATH = f'{ROOT}/Analysis/data/Spain_prices.csv'
REE_PATH = f'{ROOT}/Analysis/data/daily_gen_spain.csv'

START_DATE = '2024-04-01'
N_DAYS = 91
END_DATE = pd.Timestamp(START_DATE) + pd.Timedelta(days=N_DAYS) - pd.Timedelta(hours=1)
PERIOD_LABEL = f'{START_DATE} ({N_DAYS} days)'
SAVE_SUFFIX = f'{START_DATE}_{N_DAYS}d'

# ── CO₂ parameters (unused by default — see Section 1b) ──────────
CO2_PRICE = 65.0
CO2_INTENSITY_CCGT = 0.198
CCGT_EFFICIENCY = 0.58
COUNTRY_PREFIX = {'ES': 'Spain', 'FR': 'France', 'PT': 'Portugal'}

CARRIER_COLORS = {
    'solar': '#f9d002', 'onwind': '#235ebc', 'offwind': '#074ede',
    'offwind-float': '#b5e2fa', 'hydro': '#298c81', 'ror': '#3dbfb0',
    'PHS': '#51dbcc', 'nuclear': '#ff8c00', 'CCGT': '#a85522',
    'OCGT': '#d49a6a', 'coal': '#545454', 'biomass': '#baa741',
    'battery': '#e2ff7c', 'oil': '#c9c9c9',
}

pd.set_option('display.max_rows', 300)
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', '{:.2f}'.format)
print('Setup complete.')""")

md("---\n## Section 1 — Load Network & Disable ALL Expansion\n\n> **Note:** Marginal costs, ramp limits, and p_min_pu are baked into the network file by the `non_linear_MCs` and `07_ramping_adjustments` notebooks. This notebook only disables expansion at runtime.")

code("""print(f'Loading network from {NET_PATH}...')
n = pypsa.Network(NET_PATH)
print(f'Network loaded: {len(n.buses)} buses, {len(n.generators)} generators, {len(n.snapshots)} snapshots')

# ── 1a. Disable ALL capacity expansion ───────────────────────────
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

md("### 1b. CO₂ Price (OPTIONAL — uncomment to activate)")

code("""# ── Uncomment the lines below to add a CO₂ price ────────────────
# ccgt_mask = n.generators['carrier'] == 'CCGT'
# n.generators.loc[ccgt_mask, 'marginal_cost'] += CO2_ADD
# print(f'CO2 price added: {CO2_ADD:.2f} EUR/MWh_e to {ccgt_mask.sum()} CCGT generators')
# print(f'CCGT MC range now: {n.generators.loc[ccgt_mask, "marginal_cost"].min():.1f} - '
#       f'{n.generators.loc[ccgt_mask, "marginal_cost"].max():.1f} EUR/MWh')
print('CO₂ price NOT applied (Section 1b is commented out).')""")


md("---\n## Section 2 — Slice & Solve Period")

code("""start_time = pd.Timestamp(START_DATE)
end_time = pd.Timestamp(END_DATE)
period_mask = (n.snapshots >= start_time) & (n.snapshots <= end_time)
period_snapshots = n.snapshots[period_mask]
n_snapshots = len(period_snapshots)

print(f'Selected period: {start_time} to {end_time}')
print(f'Snapshots to solve: {n_snapshots} ({n_snapshots/24:.1f} days)')

n_sub = n.copy()
n_sub.set_snapshots(period_snapshots)

# ── TRANSMISSION CONSTRAINT: Break the copper plate ────────────────────
TRANS_FACTOR = 0.35
es_lines = n_sub.lines[
    n_sub.lines.bus0.astype(str).str[:2].isin(['ES']) &
    n_sub.lines.bus1.astype(str).str[:2].isin(['ES'])
].index
print(f'Scaling {len(es_lines)} ES-ES lines by {TRANS_FACTOR}x')
n_sub.lines.loc[es_lines, 's_nom'] *= TRANS_FACTOR
n_sub.lines.loc[es_lines, 's_nom_extendable'] = False
n_sub.lines['s_max_pu'] = 0.5
print(f's_max_pu set to {n_sub.lines.s_max_pu.iloc[0]} for all lines')

print(f'Solving {len(n_sub.snapshots)} snapshots with Gurobi (5 cores, 2h limit)...')
res = n_sub.optimize(solver_name='gurobi',
                     solver_options={'OutputFlag': 1, 'TimeLimit': 7200, 'Threads': 5})
print(f'Status: {res[0]} | {res[1]}')
print(f'Objective: {n_sub.objective:.2f} EUR')""")

md("---\n## Section 3 — Save Solved Network")

code("""save_dir = f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(save_dir, exist_ok=True)
save_path = f'{save_dir}/solved_{SAVE_SUFFIX}.nc'
n_sub.export_to_netcdf(save_path)
print(f'Solved network saved to: {save_path}')""")

md("---\n## Section 4 — Dispatch Stack (Model)")

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

md("---\n## Section 5 — Dispatch Stack Plot")

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
    ax.set_title(f'{cn} - Dispatch ({PERIOD_LABEL})')
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.show()""")

md("---\n## Section 6 — Price Diagnostics\n\n> **Note on FR/PT prices:** France and Portugal are in separate sub-networks connected via DC links (not AC lines). PyPSA assigns independent marginal prices to each sub-network based on local dispatch. FR/PT prices reflect their own marginal generation costs. The DC interconnectors couple the markets through power flows — when interconnectors are congested (flowing at max capacity), prices differ between countries. When uncongested, prices converge.")

code("""if hasattr(n_sub, 'buses_t') and 'marginal_price' in n_sub.buses_t:
    p = n_sub.buses_t.marginal_price.copy()
else:
    raise ValueError('No marginal_price found - solve may have failed')

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
        'Min': s.min(), 'P5': s.quantile(0.05), 'P25': s.quantile(0.25),
        'Median': s.median(), 'Mean': s.mean(),
        'P75': s.quantile(0.75), 'P95': s.quantile(0.95), 'Max': s.max(), 'Std': s.std(),
    }
    print(f'\\n{label}:')
    for k, v in stats[label].items():
        print(f'  {k:8s}: {v:8.2f}')

print('\\n=== ES Nodal Price Spread ===')
es_spread = p_es.max(axis=1) - p_es.min(axis=1)
print(f'Mean spread: {es_spread.mean():.2f} EUR/MWh')
print(f'Max spread:  {es_spread.max():.2f} EUR/MWh')
print(f'Unique bus prices per snapshot (avg): {p_es.apply(lambda x: x.nunique(), axis=1).mean():.1f}')
print(f'Min unique prices: {p_es.apply(lambda x: x.nunique(), axis=1).min()}')
print(f'Max unique prices: {p_es.apply(lambda x: x.nunique(), axis=1).max()}')

print('\\n=== Interconnector Congestion & Price Coupling ===')
ic_links = n_sub.links[n_sub.links['carrier'].str.contains('DC_ic', na=False)]
print(f'{"Link":25s} {"ES Price":>9} {"FR/PT Price":>11} {"Flow/Cap":>10} {"Congested?":>10}')
print('-' * 67)
for idx in ic_links.index:
    b0 = ic_links.at[idx, 'bus0']
    b1 = ic_links.at[idx, 'bus1']
    cap = ic_links.at[idx, 'p_nom']
    flow = abs(n_sub.links_t.p0.loc[:, idx].mean())
    p_b0 = p[b0].mean() if b0 in p.columns else float('nan')
    p_b1 = p[b1].mean() if b1 in p.columns else float('nan')
    congested = 'YES' if flow / cap > 0.99 else 'no'
    print(f'{idx:25s}: {p_b0:8.2f}  {p_b1:10.2f}  {flow/cap*100:7.1f}%  {congested:>8}')

print()
print('Interpretation:')
print('  - If congested=YES: interconnector is at max capacity, prices differ between countries')
print('  - If congested=no: interconnector has headroom, prices should converge')
print('  - FR nuclear (MC=12 EUR/MWh) sets FR prices low; cheap power flows to ES')
print('  - PT hydro (MC=0 EUR/MWh) sets PT prices at zero; cheap power flows to ES')""")

md("### 6b. Price Duration Curves — Model vs OMIE")

code("""load_es = n_sub.loads_t.p_set[[l for l in n_sub.loads_t.p_set.columns if str(l).startswith('ES')]]
load_total_es = load_es.sum(axis=1)
# Align prices to load buses (p_es has 150 cols incl H2/battery, load_es has 50 cols)
common_buses = [b for b in load_es.columns if b in p_es.columns]
p_es_aligned = p_es[common_buses]
load_aligned = load_es[common_buses]
price_weighted = (p_es_aligned * load_aligned.values).sum(axis=1) / load_total_es

# Load OMIE real prices for the same period
omie = pd.read_csv(OMIE_PATH, parse_dates=['Datetime (Local)'], dayfirst=True)
omie = omie.set_index('Datetime (Local)')['Price (EUR/MWhe)']
omie.index = omie.index.tz_localize(None)
omie_period = omie.loc[start_time:end_time]

# Sort both descending for duration curves
model_sorted = price_weighted.sort_values(ascending=False).reset_index(drop=True)
omie_sorted = omie_period.sort_values(ascending=False).reset_index(drop=True)
x_pct = np.linspace(0, 100, len(model_sorted))

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_pct, model_sorted.values, color='#a85522', linewidth=1.5, label=f'Model (load-weighted)')
ax.plot(x_pct, omie_sorted.values[:len(model_sorted)], color='#235ebc', linewidth=1.5, alpha=0.7, label=f'OMIE (real)')
ax.axhline(price_weighted.mean(), color='#a85522', linestyle='--', alpha=0.5, label=f'Model mean: {price_weighted.mean():.1f}')
ax.axhline(omie_period.mean(), color='#235ebc', linestyle='--', alpha=0.5, label=f'OMIE mean: {omie_period.mean():.1f}')
ax.fill_between(x_pct, 0, model_sorted.values, alpha=0.15, color='#a85522')
ax.fill_between(x_pct, 0, omie_sorted.values[:len(model_sorted)], alpha=0.15, color='#235ebc')
ax.set_xlabel('Hours (%)')
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title(f'Price Duration Curve — Model vs OMIE ({PERIOD_LABEL})')
ax.legend()
ax.set_xlim(0, 100)
plt.tight_layout()
plt.show()

print('=== Price Duration Curve Statistics ===')
print(f'{"":>20} {"Model":>10} {"OMIE":>10} {"Diff":>10}')
print('-' * 52)
for stat_name, func in [('Mean', lambda x: x.mean()), ('Median', lambda x: x.median()),
                         ('P10', lambda x: x.quantile(0.1)),
                         ('P90', lambda x: x.quantile(0.9)), ('Min', lambda x: x.min()),
                         ('Max', lambda x: x.max()), ('Std', lambda x: x.std())]:
    m_val = func(price_weighted)
    o_val = func(omie_period)
    print(f'{stat_name:>20}: {m_val:10.2f} {o_val:10.2f} {m_val - o_val:10.2f}')""")

md("### 6c. Hourly Price Trace — Model vs OMIE")

code("""fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
ax = axes[0]
ax.plot(price_weighted.index, price_weighted.values, label='Model (load-weighted ES)', color='#a85522', linewidth=0.8)
ax.plot(omie_period.index, omie_period.values, label='OMIE (real)', color='#235ebc', linewidth=0.8, alpha=0.7)
ax.set_ylabel('EUR/MWh')
ax.set_title(f'Model vs OMIE - Hourly Prices ({PERIOD_LABEL})')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

ax = axes[1]
aligned = pd.DataFrame({'model': price_weighted, 'omie': omie_period}).dropna()
ax.scatter(aligned['omie'], aligned['model'], s=1, alpha=0.3, color='#a85522')
lims = [min(aligned.min()), max(aligned.max())]
ax.plot(lims, lims, 'k--', alpha=0.3, label='Perfect fit')
ax.set_xlabel('OMIE Price (EUR/MWh)')
ax.set_ylabel('Model Price (EUR/MWh)')
r2 = np.corrcoef(aligned['model'], aligned['omie'])[0,1]**2
ax.set_title(f'Scatter: Model vs OMIE (R² = {r2:.3f})')
ax.legend()
plt.tight_layout()
plt.show()""")

md("---\n## Section 7 — Marginal Price Setter Identification\n\n> **Note:** Only generators with `p_nom > 0` (built capacity) are considered. Zero-capacity VRE (e.g. `offwind-float` at ES buses with 0 MW) is excluded — those are planned but not yet built.")

code("""gen_mc = n_sub.generators['marginal_cost']
gen_carrier = n_sub.generators['carrier']
gen_info = n_sub.generators[['bus', 'carrier', 'p_nom']]

# Only consider generators with built capacity (p_nom > 0)
built_mask = n_sub.generators['p_nom'] > 0
gen_mc_built = gen_mc[built_mask]
gen_carrier_built = gen_carrier[built_mask]
gen_info_built = gen_info[built_mask]

print(f'Total generators: {len(gen_info)}')
print(f'Built generators (p_nom > 0): {built_mask.sum()}')
print(f'Zero-capacity generators excluded: {(~built_mask).sum()}')
print()

marginal_setters = []
for snapshot in p_es.index[:100]:
    for bus in p_es.columns:
        bus_price = p_es.loc[snapshot, bus]
        if np.isnan(bus_price) or bus_price < 0:
            continue
        gens_at_bus = gen_info_built.index[gen_info_built['bus'] == bus]
        if len(gens_at_bus) == 0:
            continue
        mc_at_bus = gen_mc_built[gens_at_bus]
        match = mc_at_bus[(mc_at_bus >= bus_price * 0.99) & (mc_at_bus <= bus_price * 1.01)]
        if len(match) > 0:
            marginal_setters.append({
                'snapshot': snapshot, 'bus': bus, 'price': bus_price,
                'marginal_carrier': gen_carrier_built[match.index[0]], 'marginal_mc': match.iloc[0],
            })

df_setters = pd.DataFrame(marginal_setters)
print('=== Marginal Price Setter (sample: first 100 snapshots) ===')
if len(df_setters) > 0:
    carrier_counts = df_setters['marginal_carrier'].value_counts()
    total = carrier_counts.sum()
    print(f'{"Carrier":>20} {"Count":>8} {"Pct":>8}')
    print('-' * 38)
    for carrier, count in carrier_counts.items():
        print(f'{carrier:>20}: {count:8d} {count/total*100:7.1f}%')
else:
    print('No marginal setters found at 1% tolerance, trying 5%...')
    marginal_setters = []
    for snapshot in p_es.index[:100]:
        for bus in p_es.columns:
            bus_price = p_es.loc[snapshot, bus]
            if np.isnan(bus_price) or bus_price < 0:
                continue
            gens_at_bus = gen_info_built.index[gen_info_built['bus'] == bus]
            if len(gens_at_bus) == 0:
                continue
            mc_at_bus = gen_mc_built[gens_at_bus]
            match = mc_at_bus[(mc_at_bus >= bus_price * 0.95) & (mc_at_bus <= bus_price * 1.05)]
            if len(match) > 0:
                marginal_setters.append({
                    'snapshot': snapshot, 'bus': bus, 'price': bus_price,
                    'marginal_carrier': gen_carrier_built[match.index[0]], 'marginal_mc': match.iloc[0],
                })
    df_setters = pd.DataFrame(marginal_setters)
    if len(df_setters) > 0:
        carrier_counts = df_setters['marginal_carrier'].value_counts()
        total = carrier_counts.sum()
        print(f'{"Carrier":>20} {"Count":>8} {"Pct":>8}')
        print('-' * 38)
        for carrier, count in carrier_counts.items():
            print(f'{carrier:>20}: {count:8d} {count/total*100:7.1f}%')""")

md("### 7b. Gas Dominance Check")

code("""ccgt_gens = gen_info_built.index[gen_info_built['carrier'] == 'CCGT']
ccgt_dispatch = gen_p[ccgt_gens]
ccgt_total = ccgt_dispatch.sum(axis=1)
ccgt_capacity = n_sub.generators.loc[ccgt_gens, 'p_nom'].sum()

print('=== CCGT (Gas) Dominance Check ===')
print(f'Total CCGT capacity: {ccgt_capacity/1000:.1f} GW')
print(f'Mean CCGT dispatch:  {ccgt_total.mean()/1000:.1f} GW')
print(f'Max CCGT dispatch:   {ccgt_total.max()/1000:.1f} GW')
print(f'Min CCGT dispatch:   {ccgt_total.min()/1000:.1f} GW')
print(f'Capacity factor:     {ccgt_total.mean()/ccgt_capacity*100:.1f}%')
if len(df_setters) > 0:
    ccgt_pct = (df_setters['marginal_carrier'] == 'CCGT').mean() * 100
    print(f'CCGT as marginal setter: {ccgt_pct:.1f}% of sampled snapshots')
total_gen = gen_total.sum(axis=1)
ccgt_share = (ccgt_total / total_gen * 100).mean()
print(f'CCGT share of total generation: {ccgt_share:.1f}%')""")

md("---\n## Section 8 — Residual Load Analysis")

code("""es_loads = [l for l in n_sub.loads_t.p_set.columns if str(l).startswith('ES')]
demand_es = n_sub.loads_t.p_set[es_loads].sum(axis=1)
es_buses_list = [b for b in n_sub.buses.index if str(b).startswith('ES')]
es_gens = gen_info.index[gen_info['bus'].isin(es_buses_list)]

solar_mask = gen_info.index[gen_info['carrier'] == 'solar']
solar_es = gen_p[es_gens.intersection(solar_mask)].sum(axis=1) if len(es_gens.intersection(solar_mask)) > 0 else pd.Series(0, index=gen_p.index)
wind_mask = gen_info.index[gen_info['carrier'].isin(['onwind', 'offwind', 'offwind-float'])]
wind_es = gen_p[es_gens.intersection(wind_mask)].sum(axis=1) if len(es_gens.intersection(wind_mask)) > 0 else pd.Series(0, index=gen_p.index)
nuclear_mask = gen_info.index[gen_info['carrier'] == 'nuclear']
nuclear_es = gen_p[es_gens.intersection(nuclear_mask)].sum(axis=1) if len(es_gens.intersection(nuclear_mask)) > 0 else pd.Series(0, index=gen_p.index)

residual = demand_es - solar_es - wind_es - nuclear_es

print('=== Residual Load Analysis (Spain) ===')
print(f'Mean demand:       {demand_es.mean()/1000:.2f} GW')
print(f'Mean solar:        {solar_es.mean()/1000:.2f} GW')
print(f'Mean wind:         {wind_es.mean()/1000:.2f} GW')
print(f'Mean nuclear:      {nuclear_es.mean()/1000:.2f} GW')
print(f'Mean residual:     {residual.mean()/1000:.2f} GW')
print(f'Min residual:      {residual.min()/1000:.2f} GW')
print(f'Max residual:      {residual.max()/1000:.2f} GW')
print(f'Hours with negative residual: {(residual < 0).sum()} / {len(residual)} ({(residual < 0).mean()*100:.1f}%)')

fig, ax = plt.subplots(figsize=(14, 5))
ax.fill_between(demand_es.index, 0, demand_es.values, label='Total Demand', color='gray', alpha=0.3)
ax.fill_between(demand_es.index, 0, solar_es.values, label='Solar', color='#f9d002', alpha=0.5)
ax.fill_between(demand_es.index, solar_es.values, solar_es.values + wind_es.values, label='Wind', color='#235ebc', alpha=0.5)
ax.fill_between(demand_es.index, solar_es.values + wind_es.values, solar_es.values + wind_es.values + nuclear_es.values, label='Nuclear', color='#ff8c00', alpha=0.5)
ax.plot(residual.index, residual.values, label='Residual Load', color='red', linewidth=1)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('MW')
ax.set_title(f'Spain - Residual Load ({PERIOD_LABEL})')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout()
plt.show()""")

md("---\n## Section 9 — Curtailment")

code("""vre_carriers = ['solar', 'onwind', 'offwind', 'offwind-float']
vre_gens = gen_info.index[gen_info['carrier'].isin(vre_carriers)]

if len(vre_gens) > 0:
    vre_available = pd.DataFrame({
        g: n_sub.generators_t.p_max_pu[g] * n_sub.generators.at[g, 'p_nom']
        for g in vre_gens if g in n_sub.generators_t.p_max_pu.columns
    })
    vre_actual = gen_p[vre_gens.intersection(vre_available.columns)]
    vre_curtailed = (vre_available - vre_actual).clip(lower=0)
    
    total_available = vre_available.sum().sum()
    total_actual = vre_actual.sum().sum()
    total_curtailed = vre_curtailed.sum().sum()
    
    print('=== VRE Curtailment ===')
    print(f'Total available VRE: {total_available/1000:.2f} GWh')
    print(f'Total actual VRE:    {total_actual/1000:.2f} GWh')
    print(f'Total curtailed:     {total_curtailed/1000:.2f} GWh')
    print(f'Curtailment rate:    {total_curtailed/total_available*100:.2f}%')
    
    hourly_curtailment = vre_curtailed.sum(axis=1)
    print(f'Hours with curtailment > 0: {(hourly_curtailment > 0).sum()} / {len(hourly_curtailment)}')
    print(f'Max hourly curtailment: {hourly_curtailment.max()/1000:.2f} GWh')
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(hourly_curtailment.index, 0, hourly_curtailment.values/1000, color='red', alpha=0.4, label='Curtailment')
    ax.set_ylabel('GWh')
    ax.set_title(f'VRE Curtailment ({PERIOD_LABEL})')
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
    plt.show()
else:
    print('No VRE generators found with p_max_pu data.')""")

md("---\n## Section 10 — Interconnector Flows\n\n> **Note:** Interconnectors use carrier `DC_ic export` / `DC_ic import` (not plain `DC_ic`).")

code("""ic_links = n_sub.links[n_sub.links['carrier'].str.contains('DC_ic', na=False)]
if len(ic_links) > 0:
    ic_flow = n_sub.links_t.p0[ic_links.index]
    print('=== Interconnector Flows ===')
    for link_name in ic_links.index:
        flow = ic_flow[link_name]
        bus0 = ic_links.at[link_name, 'bus0']
        bus1 = ic_links.at[link_name, 'bus1']
        capacity = ic_links.at[link_name, 'p_nom']
        mean_flow = flow.mean()
        util = abs(mean_flow) / capacity * 100 if capacity > 0 else 0
        direction = f'{bus0} -> {bus1}' if mean_flow > 0 else f'{bus1} -> {bus0}'
        print(f'  {link_name:30s}: cap={capacity:6.0f} MW, mean={mean_flow:7.1f} MW ({util:4.1f}%), net={direction}')
    print(f'\\nTotal net ES export: {ic_flow.sum(axis=1).mean():.0f} MW (mean)')
    fig, ax = plt.subplots(figsize=(14, 4))
    for link_name in ic_links.index:
        ax.plot(ic_flow.index, ic_flow[link_name], label=link_name, linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('MW')
    ax.set_title(f'Interconnector Flows ({PERIOD_LABEL})')
    ax.legend(loc='upper right', fontsize=7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.tight_layout()
    plt.show()
else:
    print('No DC_ic links found.')""")

md("### 10a. DC Link Diagnostic\n\n> **Note:** There is one non-`DC_ic` DC link (`relation/17631956-250-DC`, ES0 5 ↔ ES1 0, 400 MW). All cross-border connections use `DC_ic export`/`DC_ic import` links (11 total). No AC lines cross borders — all 91 AC lines are internal to Spain.")

code("""# ── DC links (non-DC_ic) ────────────────────────────────────────
dc_other = n_sub.links[n_sub.links['carrier'] == 'DC']
print('=== Other DC Links ===')
if len(dc_other) > 0:
    for ln in dc_other.index:
        b0 = dc_other.at[ln, 'bus0']
        b1 = dc_other.at[ln, 'bus1']
        cap = dc_other.at[ln, 'p_nom']
        print(f'  {ln}: {b0} -> {b1}, {cap:.0f} MW')
else:
    print('  None found')

# ── Cross-border AC lines check ──────────────────────────────────
print()
print('=== Cross-Border AC Lines ===')
cross = []
for idx in n_sub.lines.index:
    b0 = n_sub.lines.at[idx, 'bus0']
    b1 = n_sub.lines.at[idx, 'bus1']
    c0 = str(b0)[:2]
    c1 = str(b1)[:2]
    if c0 != c1:
        cross.append((idx, b0, b1, c0, c1))
if cross:
    for idx, b0, b1, c0, c1 in cross:
        print(f'  {idx}: {b0} -> {b1} ({c0}-{c1})')
else:
    print('  No cross-border AC lines — all interconnectors are DC links')

# ── Interconnector capacity summary ──────────────────────────────
print()
print('=== Interconnector Capacity Summary ===')
ic_links = n_sub.links[n_sub.links['carrier'].str.contains('DC_ic', na=False)]
for ln in ic_links.index:
    b0 = ic_links.at[ln, 'bus0']
    b1 = ic_links.at[ln, 'bus1']
    cap = ic_links.at[ln, 'p_nom']
    carrier = ic_links.at[ln, 'carrier']
    print(f'  {carrier:20s} {b0:12s} -> {b1:12s} {cap:6.0f} MW')

# Group by country pair
print()
exports = ic_links[ic_links['carrier'].str.contains('export', na=False)]
imports = ic_links[ic_links['carrier'].str.contains('import', na=False)]
print(f'Total export capacity: {exports["p_nom"].sum():.0f} MW')
print(f'Total import capacity: {imports["p_nom"].sum():.0f} MW')
print(f'Asymmetry (export - import): {exports["p_nom"].sum() - imports["p_nom"].sum():.0f} MW')
print()
print('By country pair:')
for pair in ['PT', 'FR']:
    pair_exp = exports[exports['bus1'].str.startswith(pair)]
    pair_imp = imports[exports['bus0'].str.startswith(pair)]
    print(f'  ES-{pair}: export={pair_exp["p_nom"].sum():.0f} MW, import={pair_imp["p_nom"].sum():.0f} MW')""")

md("---\n## Section 11 — Model vs Real REE Dispatch (Spain)\n\n> **Note:** REE data is daily (columns are dates like `01/jul/24`). Model hourly data is resampled to daily for comparison.")

code("""ree_raw = pd.read_csv(REE_PATH, index_col=0)
ree = ree_raw.T
ree.index = pd.to_datetime(ree.index, format='%d/%m/%y', dayfirst=True, errors='coerce')
ree = ree.dropna(how='all', axis=1)
# Drop rows with failed date parsing
ree = ree.dropna(subset=[ree.columns[0]])

REE_TO_CARRIER = {
    'Hidraulica': 'hydro', 'Nuclear': 'nuclear', 'Carbon': 'coal',
    'Ciclo combinado': 'CCGT', 'Eolica': 'onwind',
    'Solar fotovoltaica': 'solar', 'Turbina de gas': 'OCGT',
    'Motores diesel': 'oil',
}

model_daily = gen_by_country['ES'].resample('D').sum() / 1000
model_daily_period = model_daily.loc[start_time:end_time]

# REE dates are daily — slice by date only
ree_period = ree.loc[start_time.strftime('%Y-%m-%d'):end_time.strftime('%Y-%m-%d')]

print('=== Model vs REE - Spain Daily Generation (GWh) ===')
print(f'{"Technology":>25} {"Model":>10} {"REE":>10} {"Diff":>10} {"Error%":>8}')
print('-' * 65)

comparison = {}
for ree_name, carrier in REE_TO_CARRIER.items():
    # Try to find the column (handle special chars like accents)
    ree_col = [c for c in ree_period.columns if ree_name.lower() in c.lower()]
    if ree_col and carrier in model_daily_period.columns:
        ree_val = ree_period[ree_col[0]].sum()
        model_val = model_daily_period[carrier].sum()
        diff = model_val - ree_val
        err_pct = diff / ree_val * 100 if ree_val > 0 else 0
        comparison[carrier] = {'model': model_val, 'ree': ree_val, 'diff': diff, 'err_pct': err_pct}
        print(f'{ree_col[0]:>25}: {model_val:10.2f} {ree_val:10.2f} {diff:10.2f} {err_pct:7.1f}%')

total_model = sum(v['model'] for v in comparison.values())
total_ree = sum(v['ree'] for v in comparison.values())
total_diff = total_model - total_ree
total_err = total_diff / total_ree * 100 if total_ree > 0 else 0
print('-' * 65)
print(f'{"TOTAL":>25}: {total_model:10.2f} {total_ree:10.2f} {total_diff:10.2f} {total_err:7.1f}%')

fig, ax = plt.subplots(figsize=(12, 5))
carriers_plot = list(comparison.keys())
x = np.arange(len(carriers_plot))
width = 0.35
model_vals = [comparison[c]['model'] for c in carriers_plot]
ree_vals = [comparison[c]['ree'] for c in carriers_plot]
ax.bar(x - width/2, model_vals, width, label='Model', color='#a85522', alpha=0.8)
ax.bar(x + width/2, ree_vals, width, label='REE (real)', color='#235ebc', alpha=0.8)
ax.set_ylabel('Total Generation (GWh)')
ax.set_title(f'Spain - Model vs REE Dispatch ({PERIOD_LABEL})')
ax.set_xticks(x)
ax.set_xticklabels(carriers_plot, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()""")

md("---\n## Section 12 — Summary & Key Findings")

code("""print('=' * 60)
print('  KEY FINDINGS')
print('=' * 60)
print(f'''
1. SOLVE STATUS: {res[0]} | {res[1]}
   Period: {PERIOD_LABEL} ({n_snapshots} snapshots)
   Objective: {n_sub.objective:.2f} EUR

2. PRICE STATISTICS (ES load-weighted):
   Mean: {price_weighted.mean():.1f} EUR/MWh
   Median: {price_weighted.median():.1f} EUR/MWh
   Min: {price_weighted.min():.1f} EUR/MWh
   Max: {price_weighted.max():.1f} EUR/MWh
   Std: {price_weighted.std():.1f} EUR/MWh

3. NODAL SPREAD (ES):
   Mean spread: {es_spread.mean():.2f} EUR/MWh
   Max spread: {es_spread.max():.2f} EUR/MWh
   Avg unique prices/snapshot: {p_es.apply(lambda x: x.nunique(), axis=1).mean():.1f}

4. CCGT DOMINANCE:
   Capacity factor: {ccgt_total.mean()/ccgt_capacity*100:.1f}%
   Share of generation: {ccgt_share:.1f}%
''')

if len(df_setters) > 0:
    ccgt_pct = (df_setters['marginal_carrier'] == 'CCGT').mean() * 100
    print(f'5. CCGT AS MARGINAL SETTER: {ccgt_pct:.1f}% of sampled hours')

print(f'''
6. CURTAILMENT:
   Rate: {total_curtailed/total_available*100:.2f}% (if VRE data available)

7. INTERCONNECTORS:
   Net ES export: {ic_flow.sum(axis=1).mean():.0f} MW mean (if DC_ic links exist)
''')
print('=' * 60)""")

# Build notebook
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.12.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Write
out_path = os.path.join(os.path.dirname(__file__), "04_solve_diagnostic.ipynb")
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook written to {out_path}")
