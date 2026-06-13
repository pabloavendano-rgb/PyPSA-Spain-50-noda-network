#!/usr/bin/env python3
"""Quick diagnostic of shoulder_constrained results."""
import warnings, numpy as np, pandas as pd, pypsa, os
warnings.filterwarnings('ignore')

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n=pypsa.Network(f'{ROOT}/solved_networks/04_solve_diagnostic/solved_2024_shoulder_constrained.nc')

# ES AC buses only
es_buses = [b for b in n.buses.index if str(b).startswith('ES')]
es_ac = [b for b in es_buses if not any(s in str(b) for s in [' H2',' battery'])]
print(f'ES AC buses: {len(es_ac)}')

# 1. PRICES
p = n.buses_t.marginal_price
p_ac = p[es_ac]
hourly_spread = p_ac.max(axis=1) - p_ac.min(axis=1)
print(f'\n=== PRICES (ES AC buses, Apr-Jun) ===')
print(f'Mean: {p_ac.stack().mean():.1f} EUR/MWh')
print(f'Max:  {p_ac.stack().max():.1f} EUR/MWh')
print(f'Min:  {p_ac.stack().min():.1f} EUR/MWh')
print(f'Mean hourly spread: {hourly_spread.mean():.1f} EUR/MWh')
print(f'Max hourly spread:  {hourly_spread.max():.1f} EUR/MWh')
print(f'Hours with spread >1: {(hourly_spread>1).sum()}/{len(hourly_spread)} ({(hourly_spread>1).mean()*100:.1f}%)')
print(f'Hours with spread >5: {(hourly_spread>5).sum()}/{len(hourly_spread)} ({(hourly_spread>5).mean()*100:.1f}%)')
print(f'Hours with spread >10: {(hourly_spread>10).sum()}/{len(hourly_spread)} ({(hourly_spread>10).mean()*100:.1f}%)')

# Compare to OMIE
omie=pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv')
omie['datetime']=pd.to_datetime(omie['Datetime (UTC)'],format='%d/%m/%y %H:%M')
omie=omie.set_index('datetime')['Price (EUR/MWhe)']
omie_shoulder=omie[(omie.index>=pd.Timestamp('2024-04-01'))&(omie.index<=pd.Timestamp('2024-06-30 23:00'))]
model_mean=p_ac.mean(axis=1)
print(f'\n=== MODEL vs OMIE ===')
print(f'Model mean: {model_mean.mean():.1f} EUR/MWh')
print(f'OMIE   mean: {omie_shoulder.mean():.1f} EUR/MWh')
print(f'Model max:  {p_ac.stack().max():.1f} EUR/MWh')
print(f'OMIE   max:  {omie_shoulder.max():.1f} EUR/MWh')

# Monthly
print(f'\n=== MONTHLY ===')
print(f'  {"Month":6s} {"Model Mean":10s} {"OMIE Mean":10s} {"Ratio":8s} {"Model Max":10s} {"OMIE Max":10s}')
for m in [4,5,6]:
    m_model = model_mean[model_mean.index.month == m]
    m_omie = omie_shoulder[omie_shoulder.index.month == m]
    ratio = m_model.mean() / m_omie.mean() if m_omie.mean() > 0 else float('inf')
    print(f'  Month {m:2d}  {m_model.mean():8.1f}   {m_omie.mean():8.1f}   {ratio:6.2f}x  {m_model.max():8.1f}   {m_omie.max():8.1f}')

# 2. TRANSMISSION
es_lines = n.lines[
    n.lines.bus0.astype(str).str[:2].isin(['ES']) &
    n.lines.bus1.astype(str).str[:2].isin(['ES'])
].index
loading = n.lines_t.p0.abs().div(n.lines.s_nom, axis=1)
es_line_idx = [l for l in es_lines if l in loading.columns]
print(f'\n=== TRANSMISSION (ES-ES, scaled) ===')
print(f'Max loading: {loading[es_line_idx].max().max()*100:.1f}%')
print(f'Mean loading: {loading[es_line_idx].mean().mean()*100:.1f}%')
print(f'Lines with any hour >90%: {(loading[es_line_idx] > 0.9).any().sum()} / {len(es_line_idx)}')
print(f'Lines with any hour >70%: {(loading[es_line_idx] > 0.7).any().sum()} / {len(es_line_idx)}')
print(f'Lines with any hour >50%: {(loading[es_line_idx] > 0.5).any().sum()} / {len(es_line_idx)}')

max_load = loading[es_line_idx].max().sort_values(ascending=False).head(10)
print('\nTop 10 congested lines:')
for l in max_load.index:
    b0 = n.lines.at[l, 'bus0']
    b1 = n.lines.at[l, 'bus1']
    snom = n.lines.at[l, 's_nom']
    print(f'  {l:6s}: {max_load[l]*100:.1f}% max, {loading[l].mean()*100:.1f}% mean ({b0}→{b1}, {snom:.0f}MW)')

# 3. DISPATCH
g=n.generators_t.p
total_dispatch=g.sum().groupby(n.generators.carrier).sum() / 1e6
print(f'\n=== TOTAL GENERATION (TWh, Apr-Jun) ===')
for car, val in total_dispatch.sort_values(ascending=False).items():
    print(f'  {car:20s}: {val:.2f} TWh')

# Hydro
es_hyd_cols = [c for c in n.storage_units_t.state_of_charge.columns 
               if c in n.storage_units.index and 
               str(n.storage_units.at[c,'bus']).startswith('ES') and
               n.storage_units.at[c,'carrier']=='hydro']
soc = n.storage_units_t.state_of_charge[es_hyd_cols]
print(f'\n=== HYDRO SOC ===')
print(f'Initial SOC: {soc.iloc[0].sum()/1e6:.2f} TWh')
print(f'Final SOC:   {soc.iloc[-1].sum()/1e6:.2f} TWh')
hyd_dispatch = n.storage_units_t.p[es_hyd_cols]
print(f'Hydro dispatched: {hyd_dispatch[hyd_dispatch>0].sum().sum()/1e6:.2f} TWh')
inflow = n.storage_units_t.inflow[es_hyd_cols]
print(f'Hydro inflow:     {inflow.sum().sum()/1e6:.2f} TWh')

# 4. CURTAILMENT
print(f'\n=== CURTAILMENT ===')
for tech in ['solar', 'onwind', 'offwind']:
    avail = n.generators_t.p_max_pu.mul(n.generators.p_nom, axis=1)
    avail_es = avail[[g for g in avail.columns if str(n.generators.at[g,'bus']).startswith('ES') and n.generators.at[g,'carrier']==tech]]
    gen_es = g[[g for g in g.columns if str(n.generators.at[g,'bus']).startswith('ES') and n.generators.at[g,'carrier']==tech]]
    if len(avail_es.columns) > 0:
        avail_sum = avail_es.sum(axis=1)
        gen_sum = gen_es.sum(axis=1)
        curt = (avail_sum - gen_sum).clip(0)
        curt_pct = curt.sum() / avail_sum.sum() * 100
        print(f'  {tech:15s}: {curt.sum()/1e6:.2f} TWh curtailed ({curt_pct:.1f}%)')

# 5. NUCLEAR
nuc_gens = [g for g in n.generators.index if n.generators.at[g,'carrier']=='nuclear' and str(n.generators.at[g,'bus']).startswith('ES')]
if nuc_gens:
    nuc_p = g[nuc_gens]
    print(f'\n=== NUCLEAR ===')
    print(f'Units: {len(nuc_gens)}, Capacity: {n.generators.loc[nuc_gens,"p_nom"].sum():.0f} MW')
    print(f'Generation: {nuc_p.sum().sum()/1e6:.2f} TWh')
    print(f'CF: {nuc_p.sum().sum()/(n.generators.loc[nuc_gens,"p_nom"].sum()*len(n.snapshots))*100:.1f}%')

# 6. RESIDUAL LOAD
print(f'\n=== RESIDUAL LOAD ===')
load_es = n.loads_t.p_set[[l for l in n.loads_t.p_set.columns if str(n.loads.at[l,'bus']).startswith('ES')]]
total_load = load_es.sum(axis=1)
re_techs = ['solar','onwind','offwind','offwind-float']
re_gen = pd.DataFrame()
for t in re_techs:
    tg = [g for g in g.columns if n.generators.at[g,'carrier']==t and str(n.generators.at[g,'bus']).startswith('ES')]
    if tg:
        re_gen[t] = g[tg].sum(axis=1)
total_re = re_gen.sum(axis=1) if len(re_gen.columns) > 0 else pd.Series(0, index=total_load.index)
residual = total_load - total_re
print(f'Mean residual load: {residual.mean():.1f} GW')
print(f'Hours negative: {(residual<0).sum()} / {len(residual)} ({(residual<0).mean()*100:.1f}%)')

print('\nDone!')
