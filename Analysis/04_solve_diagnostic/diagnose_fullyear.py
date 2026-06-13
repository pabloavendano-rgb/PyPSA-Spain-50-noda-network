#!/usr/bin/env python3
"""Comprehensive full-year diagnostic: ES-only analysis.
Filters everything to Spanish AC buses and generators only.
Interconnectors are OFF so FR/PT are isolated — we ignore them entirely."""
import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates, pypsa, os
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10})

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n=pypsa.Network(f'{ROOT}/solved_networks/04_solve_diagnostic/solved_2024_fullyear_co2.nc')
print(f'Loaded: {len(n.buses)} buses, {len(n.generators)} gens')

sdir=f'{ROOT}/solved_networks/04_solve_diagnostic'

# ── ES-only helpers ────────────────────────────────────────────────────
es_buses = [b for b in n.buses.index if str(b).startswith('ES')]
es_ac    = [b for b in es_buses if not any(s in str(b) for s in [' H2',' battery'])]
es_gens  = [g for g in n.generators.index if str(n.generators.at[g,'bus']).startswith('ES')]
es_sus   = [s for s in n.storage_units.index if str(n.storage_units.at[s,'bus']).startswith('ES')]

print(f'ES AC buses: {len(es_ac)}')
print(f'ES generators: {len(es_gens)}')
print(f'ES storage units: {len(es_sus)}')

# ES-only data slices
g_es = n.generators_t.p[es_gens]
su_es = n.storage_units_t.p[es_sus]
soc_es = n.storage_units_t.state_of_charge[es_sus]
inflow_es = n.storage_units_t.inflow[[s for s in es_sus if s in n.storage_units_t.inflow.columns]]

# ES hydro storage units
es_hyd_sus = [s for s in es_sus if n.storage_units.at[s,'carrier']=='hydro']
hyd_dispatch = su_es[es_hyd_sus]
hyd_soc = soc_es[es_hyd_sus]
hyd_inflow = inflow_es[es_hyd_sus]

# ══════════════════════════════════════════════════════════════════════
# 1. CCGT FLEET SIZE CHECK (ES-only)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('1. CCGT FLEET SIZE (ES-only)')
print('='*60)

ccgt_es = [g for g in es_gens if n.generators.at[g,'carrier']=='CCGT']
ccgt_flex_es = [g for g in es_gens if n.generators.at[g,'carrier']=='CCGT_Flex']
ccgt_mw = n.generators.loc[ccgt_es, 'p_nom'].sum()
ccgt_flex_mw = n.generators.loc[ccgt_flex_es, 'p_nom'].sum()
print(f'Model ES CCGT:       {ccgt_mw:.0f} MW ({len(ccgt_es)} units)')
print(f'Model ES CCGT_Flex:  {ccgt_flex_mw:.0f} MW ({len(ccgt_flex_es)} units)')
print(f'Model ES CCGT total: {ccgt_mw+ccgt_flex_mw:.0f} MW')

# Real ESIOS CCGT capacity (2024)
real_ccgt = pd.read_csv(f'{ROOT}/data_ES/esios/esios_CCGT_capacity_2024.csv')
print(f'Real ES CCGT 2024:   {real_ccgt.iloc[0,1:].astype(float).sum():.0f} MW (NUTS2 sum)')
print(f'Real ES CCGT total:  {real_ccgt["total"].iloc[0]:.0f} MW')

# ══════════════════════════════════════════════════════════════════════
# 2. FULL FLEET COMPARISON (ES-only Model vs Real ESIOS 2024)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('2. FULL FLEET COMPARISON (ES-only)')
print('='*60)

# Model annual generation by carrier (ES generators only)
model_gen_es = g_es.sum().groupby(n.generators.loc[es_gens, 'carrier']).sum() / 1e6  # TWh

# ES CSP dispatch from storage_units (positive = discharge)
csp_sus = [s for s in es_sus if n.storage_units.at[s, 'carrier'] == 'csp']
csp_dispatch = su_es[csp_sus][su_es[csp_sus] > 0].sum().sum() / 1e6  # TWh

# ES hydro dispatch (turbine only, positive = discharge)
hyd_turbine = hyd_dispatch[hyd_dispatch > 0].sum().sum() / 1e6  # TWh
hyd_pump = (-hyd_dispatch[hyd_dispatch < 0]).sum().sum() / 1e6  # TWh

# Real annual generation from daily_gen_spain.csv
real = pd.read_csv(f'{ROOT}/Analysis/data/daily_gen_spain.csv', index_col=0)
real = real.T
spanish_months = {'ene':'01','feb':'02','mar':'03','abr':'04','may':'05','jun':'06',
                  'jul':'07','ago':'08','sep':'09','oct':'10','nov':'11','dic':'12'}
def parse_spanish_date(s):
    parts = str(s).split('/')
    return f'{parts[0]}/{spanish_months[parts[1]]}/{parts[2]}'
real.index = [parse_spanish_date(str(d)) for d in real.index]
real.index = pd.to_datetime(real.index, format='%d/%m/%y', dayfirst=True)
real = real.replace('-', '0').astype(float)

real_total = real['Generación total'].sum() / 1000  # TWh
real_annual = real.drop(columns=['Generación total']).sum() / 1000  # TWh

# Mapping: real ESIOS name → model carrier(s)
tech_map = {
    'Nuclear':              ('nuclear', False),
    'Ciclo combinado':      ('CCGT+CCGT_Flex', True),
    'Carbón':               ('coal', False),
    'Hidráulica':           ('hydro_su', False),
    'Eólica':               ('onwind', False),
    'Solar fotovoltaica':   ('solar', False),
    'Solar térmica':        ('csp', False),
    'Cogeneración':         ('cogen', False),
    'Turbina de gas':       ('OCGT', False),
    'Motores diésel':       ('diesel', False),
    'Fuel + Gas':           ('oil', False),
    'Residuos no renovables': ('biomass', False),
    'Residuos renovables':  ('biomass', False),
    'Otras renovables':     ('biomass', False),
    'Hidroeólica':          ('onwind', False),
}

print(f'\n{"Tech":25s} {"Model ES (TWh)":15s} {"Real ES (TWh)":15s} {"Ratio":8s}')
print('-'*63)
for real_name, (model_car, is_combined) in tech_map.items():
    real_val = real_annual.get(real_name, 0)
    if model_car == 'hydro_su':
        model_val = hyd_turbine
    elif model_car == 'csp':
        model_val = csp_dispatch
    elif is_combined:
        model_val = sum(model_gen_es.get(c, 0) for c in model_car.split('+'))
    else:
        model_val = model_gen_es.get(model_car, 0)
    ratio = model_val / real_val if real_val > 0 else float('inf')
    print(f'{real_name:25s} {model_val:12.2f} TWh  {real_val:12.2f} TWh  {ratio:6.2f}x')

# Total
model_total_es = model_gen_es.sum() + hyd_turbine + csp_dispatch
print(f'\n{"TOTAL":25s} {model_total_es:12.2f} TWh  {real_total:12.2f} TWh  {model_total_es/real_total:6.2f}x')

# ══════════════════════════════════════════════════════════════════════
# 3. SOLAR & WIND CURTAILMENT (ES-only)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('3. CURTAILMENT ANALYSIS (ES-only)')
print('='*60)

for car, label in [('solar','Solar PV'), ('onwind','Onshore Wind'), ('offwind','Offshore Wind')]:
    gens = [g for g in es_gens if n.generators.at[g,'carrier']==car]
    if len(gens) == 0:
        continue
    # Check which columns exist in p_max_pu
    avail_cols = [c for c in gens if c in n.generators_t.p_max_pu.columns]
    if len(avail_cols) == 0:
        print(f'\n{label}: no p_max_pu data available')
        continue
    p_max = n.generators_t.p_max_pu[avail_cols] * n.generators.loc[avail_cols, 'p_nom']
    p_actual = g_es[avail_cols]
    available = p_max.sum(axis=1) / 1e3  # GW
    actual = p_actual.sum(axis=1) / 1e3   # GW
    curtail = available - actual
    curt_hours = (curtail > 0.01).sum()
    curt_twh = curtail.sum() * 1e-3  # GW→TWh (hourly)
    avail_twh = available.sum() * 1e-3
    print(f'\n{label}:')
    print(f'  Available: {avail_twh:.2f} TWh')
    print(f'  Generated: {actual.sum()*1e-3:.2f} TWh')
    print(f'  Curtailed: {curt_twh:.2f} TWh ({curt_twh/avail_twh*100:.1f}%)')
    print(f'  Hours with curtailment >10 MW: {curt_hours}/{len(curtail)} ({curt_hours/len(curtail)*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 4. TRANSMISSION CONSTRAINT ANALYSIS (ES lines only)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('4. TRANSMISSION CONSTRAINT ANALYSIS (ES lines)')
print('='*60)

es_line_idx = [i for i in n.lines.index
               if str(n.lines.at[i,'bus0']).startswith('ES')]
print(f'ES-connected lines: {len(es_line_idx)}/{len(n.lines)}')

p0 = n.lines_t.p0[es_line_idx]
s_nom = n.lines.loc[es_line_idx, 's_nom']
loading = p0.div(s_nom, axis=1)

print(f'\nLine loading stats (ES lines):')
print(f'  Mean: {loading.stack().mean()*100:.1f}%')
print(f'  Max:  {loading.stack().max()*100:.1f}%')

for threshold, label in [(0.90, '>90%'), (0.75, '>75%'), (0.50, '>50%')]:
    n_cong = (loading > threshold).sum().sum()
    total_hours = loading.size
    print(f'  Hours with loading {label}: {n_cong}/{total_hours} ({n_cong/total_hours*100:.1f}%)')

top_congested = loading.max().sort_values(ascending=False).head(10)
print(f'\nTop 10 most congested lines:')
for line_idx, loading_val in top_congested.items():
    l = n.lines.loc[line_idx]
    print(f'  Line {str(line_idx):>5s}  {loading_val*100:5.1f}%  ({l.bus0}→{l.bus1}, {l.s_nom:.0f} MVA)')

# ══════════════════════════════════════════════════════════════════════
# 5. HYDRO DEEP DIVE (ES-only)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('5. HYDRO DEEP DIVE (ES-only)')
print('='*60)

print(f'\nAnnual totals:')
print(f'  Inflow:       {hyd_inflow.sum().sum()/1e6:.2f} TWh')
print(f'  Turbine:      {hyd_turbine:.2f} TWh')
print(f'  Pump:         {hyd_pump:.2f} TWh')
print(f'  SOC initial:  {hyd_soc.iloc[0].sum()/1e6:.2f} TWh')
print(f'  SOC final:    {hyd_soc.iloc[-1].sum()/1e6:.2f} TWh')
print(f'  Net SOC chg:  {(hyd_soc.iloc[-1]-hyd_soc.iloc[0]).sum()/1e6:.2f} TWh')
spill = hyd_inflow.sum().sum() - hyd_turbine*1e6 - (hyd_soc.iloc[-1]-hyd_soc.iloc[0]).sum()
print(f'  Spill:        {spill/1e6:.2f} TWh ({spill/hyd_inflow.sum().sum()*100:.1f}% of inflow)')

# Monthly hydro
print(f'\nMonthly hydro:')
print(f'  {"Month":6s} {"Inflow":10s} {"Turbine":10s} {"Pump":10s} {"Net SOC":10s} {"Spill":10s}')
print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
for m in range(1,13):
    m_mask = hyd_soc.index.month == m
    m_inflow = hyd_inflow.loc[m_mask].sum().sum() / 1e6
    m_turbine = hyd_dispatch.loc[m_mask][hyd_dispatch.loc[m_mask]>0].sum().sum() / 1e6
    m_pump = (-hyd_dispatch.loc[m_mask][hyd_dispatch.loc[m_mask]<0]).sum().sum() / 1e6
    m_soc_chg = (hyd_soc.loc[m_mask].iloc[-1] - hyd_soc.loc[m_mask].iloc[0]).sum() / 1e6 if m_mask.any() else 0
    m_spill = m_inflow - m_turbine - m_soc_chg
    print(f'  Month {m:2d}  {m_inflow:8.2f}  {m_turbine:8.2f}  {m_pump:8.2f}  {m_soc_chg:8.2f}  {m_spill:8.2f}')

# ══════════════════════════════════════════════════════════════════════
# 6. NUCLEAR ANALYSIS (ES-only)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('6. NUCLEAR ANALYSIS (ES-only)')
print('='*60)

nuc_es = [g for g in es_gens if n.generators.at[g,'carrier']=='nuclear']
nuc_p = g_es[nuc_es]
nuc_cf = nuc_p.sum() / (n.generators.loc[nuc_es, 'p_nom'] * len(n.snapshots)) * 100
nuc_pmin = (n.generators.loc[nuc_es, 'p_min_pu'] * n.generators.loc[nuc_es, 'p_nom']).sum()

print(f'\nES Nuclear fleet: {len(nuc_es)} units, {n.generators.loc[nuc_es, "p_nom"].sum():.0f} MW total')
print(f'  p_min_pu: {n.generators.loc[nuc_es, "p_min_pu"].iloc[0]:.2f} (min output: {nuc_pmin:.0f} MW)')
print(f'  Annual generation: {nuc_p.sum().sum()/1e6:.2f} TWh')
print(f'  Avg CF: {nuc_cf.mean():.1f}%')
nuc_total_hourly = nuc_p.sum(axis=1) / 1000  # GW
at_min = (nuc_total_hourly <= nuc_pmin/1000*1.05).sum()
print(f'  Hours at p_min_pu: {at_min}/{len(nuc_total_hourly)} ({at_min/len(nuc_total_hourly)*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 7. EXAMPLE WEEKS PLOTS (ES-only model vs Real)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('7. GENERATING EXAMPLE WEEK PLOTS...')
print('='*60)

# Model hourly dispatch by carrier (ES generators only)
model_hourly = g_es.groupby(n.generators.loc[es_gens, 'carrier'], axis=1).sum() / 1000  # GW
# Add hydro
model_hourly['hydro'] = hyd_dispatch.sum(axis=1) / 1000  # GW

# Real daily dispatch (GWh → GW avg for day)
real_daily = real.drop(columns=['Generación total'])  # GWh/day

real_to_plot = {
    'Nuclear': '#ff8c00',
    'Ciclo combinado': '#b22222',
    'Carbón': '#555555',
    'Hidráulica': '#1e90ff',
    'Eólica': '#32cd32',
    'Solar fotovoltaica': '#ffd700',
    'Cogeneración': '#9370db',
}

model_to_plot = {
    'nuclear': '#ff8c00',
    'CCGT': '#b22222',
    'CCGT_Flex': '#ff6347',
    'coal': '#555555',
    'hydro': '#1e90ff',
    'onwind': '#32cd32',
    'solar': '#ffd700',
    'cogen': '#9370db',
    'OCGT': '#ff69b4',
    'diesel': '#a0522d',
    'oil': '#8b4513',
    'biomass': '#228b22',
    'ror': '#00ced1',
}

# Pick 4 representative weeks
weeks = [
    ('Jan 8-14', pd.Timestamp('2024-01-08'), pd.Timestamp('2024-01-14')),
    ('Apr 8-14', pd.Timestamp('2024-04-08'), pd.Timestamp('2024-04-14')),
    ('Jul 8-14', pd.Timestamp('2024-07-08'), pd.Timestamp('2024-07-14')),
    ('Oct 8-14', pd.Timestamp('2024-10-08'), pd.Timestamp('2024-10-14')),
]

fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex='col')

for idx, (label, start, end) in enumerate(weeks):
    ax_model = axes[idx, 0]
    ax_real = axes[idx, 1]

    # Model hourly stack
    week_model = model_hourly.loc[start:end]
    bottom = np.zeros(len(week_model))
    for car, color in model_to_plot.items():
        if car in week_model.columns:
            vals = week_model[car].values
            ax_model.fill_between(week_model.index, bottom, bottom + vals,
                                  label=car, color=color, alpha=0.8, step='post')
            bottom += vals

    ax_model.set_ylabel('GW')
    ax_model.set_title(f'Model ES: {label}')
    ax_model.legend(loc='upper left', fontsize=7, ncol=2)
    ax_model.set_ylim(0, max(bottom) * 1.1)

    # Real daily bars
    week_real = real_daily.loc[start:end]
    days = np.arange(len(week_real))
    bar_width = 0.8
    bottom_r = np.zeros(len(week_real))
    for rname, color in real_to_plot.items():
        if rname in week_real.columns:
            vals = week_real[rname].values / 24  # GWh/day → GW avg
            ax_real.bar(days, vals, bar_width, bottom=bottom_r,
                        label=rname, color=color, alpha=0.8)
            bottom_r += vals

    ax_real.set_ylabel('GW (daily avg)')
    ax_real.set_title(f'Real ESIOS: {label}')
    ax_real.legend(loc='upper left', fontsize=7, ncol=2)
    ax_real.set_xticks(days)
    ax_real.set_xticklabels([d.strftime('%a') for d in week_real.index], fontsize=8)
    ax_real.set_ylim(0, max(bottom_r) * 1.1)

plt.tight_layout()
plt.savefig(f'{sdir}/fullyear_example_weeks.png', bbox_inches='tight')
print('  Saved: fullyear_example_weeks.png')
plt.close()

# ══════════════════════════════════════════════════════════════════════
# 8. NUCLEAR SENSITIVITY: What if p_min_pu = 0.50?
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('8. NUCLEAR MIN LOAD SENSITIVITY')
print('='*60)

nuc_pmin_current = n.generators.loc[nuc_es, 'p_min_pu'].iloc[0]
nuc_pmin_test = 0.50

nuc_min_current_gw = nuc_pmin / 1000  # GW
nuc_min_test_gw = (nuc_pmin_test * n.generators.loc[nuc_es, 'p_nom']).sum() / 1000  # GW

print(f'\nCurrent p_min_pu={nuc_pmin_current:.2f}:')
print(f'  Nuclear min output: {nuc_min_current_gw:.1f} GW')
print(f'  Hours at/near min: {at_min}/{len(nuc_total_hourly)} ({at_min/len(nuc_total_hourly)*100:.1f}%)')
print(f'  Annual gen: {nuc_p.sum().sum()/1e6:.2f} TWh')

print(f'\nIf p_min_pu={nuc_pmin_test:.2f}:')
print(f'  Nuclear min output: {nuc_min_test_gw:.1f} GW (+{nuc_min_test_gw-nuc_min_current_gw:.1f} GW)')
hours_below = (nuc_total_hourly < nuc_min_test_gw).sum()
print(f'  Hours where current nuc < {nuc_min_test_gw:.1f} GW: {hours_below}/{len(nuc_total_hourly)} ({hours_below/len(nuc_total_hourly)*100:.1f}%)')
forced_twh = (nuc_min_test_gw - nuc_total_hourly[nuc_total_hourly < nuc_min_test_gw]).sum() * 1e-3
print(f'  Additional forced generation: {forced_twh:.2f} TWh')
print(f'  This would displace CCGT/hydro in those hours')

# ══════════════════════════════════════════════════════════════════════
# 9. PRICE vs RESIDUAL LOAD (ES AC buses only)
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('9. PRICE vs RESIDUAL LOAD (ES AC)')
print('='*60)

p = n.buses_t.marginal_price
p_es_ac = p[es_ac]
mean_price = p_es_ac.mean(axis=1)

# Residual load = total ES load - must-run (solar + wind + nuclear min)
load = n.loads_t.p_set
es_load_cols = [c for c in load.columns if str(c).startswith('ES')]
total_load = load[es_load_cols].sum(axis=1) / 1000  # GW

# Must-run: solar + wind (actual) + nuclear minimum
solar_gens = [g for g in es_gens if n.generators.at[g,'carrier']=='solar']
wind_gens = [g for g in es_gens if n.generators.at[g,'carrier']=='onwind']
solar_gen = g_es[solar_gens].sum(axis=1) / 1000 if solar_gens else pd.Series(0, index=g_es.index)
wind_gen = g_es[wind_gens].sum(axis=1) / 1000 if wind_gens else pd.Series(0, index=g_es.index)

residual = total_load - solar_gen - wind_gen - nuc_min_current_gw

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(residual, mean_price, c=mean_price.index.month, cmap='viridis',
                s=5, alpha=0.5, vmin=1, vmax=12)
ax.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax.axvline(0, color='gray', linestyle='--', alpha=0.3)
ax.set_xlabel('Residual Load (GW)')
ax.set_ylabel('Price (EUR/MWh)')
ax.set_title('Price vs Residual Load (ES AC buses, colored by month)')
cbar = plt.colorbar(sc, ax=ax, label='Month', ticks=range(1,13))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{sdir}/fullyear_price_vs_residual.png', bbox_inches='tight')
print('  Saved: fullyear_price_vs_residual.png')
plt.close()

# Summary stats
print(f'\nResidual load stats (ES):')
print(f'  Mean: {residual.mean():.1f} GW')
print(f'  Min:  {residual.min():.1f} GW')
print(f'  Max:  {residual.max():.1f} GW')
print(f'  Hours with residual < 0: {(residual<0).sum()}/{len(residual)} ({(residual<0).mean()*100:.1f}%)')

# ══════════════════════════════════════════════════════════════════════
# 10. ES-ONLY PRICE SUMMARY
# ══════════════════════════════════════════════════════════════════════
print('\n' + '='*60)
print('10. ES-ONLY PRICE SUMMARY')
print('='*60)

print(f'\nES AC bus prices (mean across all ES AC buses):')
print(f'  Mean: {mean_price.mean():.1f} EUR/MWh')
print(f'  Max:  {mean_price.max():.1f} EUR/MWh')
print(f'  Min:  {mean_price.min():.1f} EUR/MWh')
print(f'  Std:  {mean_price.std():.1f} EUR/MWh')

# Compare to OMIE
omie = pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv', parse_dates=['Datetime (UTC)'], dayfirst=True)
omie = omie.set_index('Datetime (UTC)')['Price (EUR/MWhe)']
print(f'\nOMIE 2024:')
print(f'  Mean: {omie.mean():.1f} EUR/MWh')
print(f'  Max:  {omie.max():.1f} EUR/MWh')
print(f'  Min:  {omie.min():.1f} EUR/MWh')
print(f'  Std:  {omie.std():.1f} EUR/MWh')

# Monthly comparison
print(f'\nMonthly price comparison (ES AC vs OMIE):')
print(f'  {"Month":6s} {"Model ES":10s} {"OMIE":10s} {"Ratio":8s} {"Model Max":10s} {"OMIE Max":10s}')
print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*8} {"-"*10} {"-"*10}')
for m in range(1,13):
    m_model = mean_price[mean_price.index.month == m]
    m_omie = omie[omie.index.month == m]
    ratio = m_model.mean() / m_omie.mean() if m_omie.mean() > 0 else float('inf')
    print(f'  Month {m:2d}  {m_model.mean():8.1f}   {m_omie.mean():8.1f}   {ratio:6.2f}x  {m_model.max():8.1f}   {m_omie.max():8.1f}')

print('\n' + '='*60)
print('DIAGNOSTIC COMPLETE')
print('='*60)
