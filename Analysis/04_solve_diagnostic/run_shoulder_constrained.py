#!/usr/bin/env python3
"""3-month shoulder period (Apr-Jun 2024) — SPAIN ONLY, constrained transmission.
Scales down ES-ES line capacities to create realistic congestion.
Uses: CO2 price, hard-coded MCs, ramping, no expansion, no ICs, hydro fix.
FR/PT buses are removed — only ES buses remain."""
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates, pypsa, os
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10})
rng=np.random.default_rng(42)

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n=pypsa.Network(f'{ROOT}/resources/networks/50n_ES_FR_PT.nc')
print(f'Loaded: {len(n.buses)} buses, {len(n.generators)} gens, {len(n.snapshots)} snapshots')

# ── Keep FR/PT buses in network (user preference), zero interconnectors ──
# Interconnectors are zeroed below in the IC section
bus_cc=pd.Series(n.buses.index.astype(str).str[:2].values,index=n.buses.index)
gen_cc=n.generators.bus.map(bus_cc); su_cc=n.storage_units.bus.map(bus_cc)

# ══════════════════════════════════════════════════════════════════════
# TRANSMISSION CONSTRAINT: Scale down ES-ES line capacities
# ══════════════════════════════════════════════════════════════════════
TRANS_FACTOR = 0.35  # Scale factor for ES-ES lines

es_lines = n.lines[
    n.lines.bus0.astype(str).str[:2].isin(['ES']) &
    n.lines.bus1.astype(str).str[:2].isin(['ES'])
].index

print(f'\n=== TRANSMISSION CONSTRAINT ===')
print(f'Scaling {len(es_lines)} ES-ES lines by factor {TRANS_FACTOR}')
print(f'Before: mean={n.lines.loc[es_lines,"s_nom"].mean():.0f} MW, total={n.lines.loc[es_lines,"s_nom"].sum()/1e3:.0f} GW')
n.lines.loc[es_lines, 's_nom'] *= TRANS_FACTOR
n.lines.loc[es_lines, 's_nom_extendable'] = False
print(f'After:  mean={n.lines.loc[es_lines,"s_nom"].mean():.0f} MW, total={n.lines.loc[es_lines,"s_nom"].sum()/1e3:.0f} GW')

# Also set s_max_pu to create thermal limits
n.lines['s_max_pu'] = 0.7  # 70% thermal limit

# ── CO2 price: €65/t ──────────────────────────────────────────────────
co2_price = 65.0
co2_add = {
    'CCGT': 0.348 * co2_price,      # ~22.6
    'CCGT_Flex': 0.532 * co2_price, # ~34.6
    'coal': 0.850 * co2_price,      # ~55.3
    'OCGT': 0.532 * co2_price,      # ~34.6
    'diesel': 0.763 * co2_price,    # ~49.6
    'oil': 0.763 * co2_price,       # ~49.6
}
print(f'\n=== CO2 PRICE: €{co2_price}/t ===')
for car, add in co2_add.items():
    m = n.generators['carrier'] == car
    if m.any():
        old_mc = n.generators.loc[m, 'marginal_cost'].mean()
        n.generators.loc[m, 'marginal_cost'] += add
        new_mc = n.generators.loc[m, 'marginal_cost'].mean()
        print(f'  {car:15s}: +€{add:.1f}/MWh  ({old_mc:.1f} → {new_mc:.1f})')

# ── Hard-code MCs ─────────────────────────────────────────────────────
for car in ['solar']: n.generators.loc[n.generators['carrier']==car,'marginal_cost']=0.10
for car in ['onwind','offwind','offwind-float']: n.generators.loc[n.generators['carrier']==car,'marginal_cost']=0.50
nuc=n.generators['carrier']=='nuclear'
n.generators.loc[nuc,'marginal_cost']=rng.uniform(12,18,size=nuc.sum())
n.generators.loc[nuc,'p_min_pu']=0.10
n.generators.loc[n.generators['carrier']=='biomass','marginal_cost']=40.0
cog=n.generators['carrier']=='cogen'
n.generators.loc[cog,'marginal_cost']=45.0; n.generators.loc[cog,'p_min_pu']=0.70
es_hyd=(su_cc=='ES')&(n.storage_units['carrier']=='hydro')
n.storage_units.loc[es_hyd,'marginal_cost']=28.0
for cc,oc in [('FR',22.0),('PT',35.0)]:
    m=(gen_cc==cc)&(n.generators['carrier']=='hydro')
    n.generators.loc[m,'marginal_cost']=oc

def assign_ccgt_tiers(cc,tiers):
    idx=n.generators.loc[(gen_cc==cc)&(n.generators['carrier']=='CCGT')].sort_values('p_nom',ascending=False).index
    if len(idx)==0: return
    for si,(lo,hi) in zip(np.array_split(idx,len(tiers)),tiers):
        n.generators.loc[si,'marginal_cost']=rng.uniform(lo,hi,size=len(si))
assign_ccgt_tiers('ES',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('PT',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('FR',[(68,78),(78,88)])

# Add CO2 to fossil generators
for car, add in co2_add.items():
    m = n.generators['carrier'] == car
    if m.any():
        n.generators.loc[m, 'marginal_cost'] += add

n.generators.loc[n.generators['carrier']=='coal','marginal_cost']=115.0 + co2_add['coal']
n.generators.loc[n.generators['carrier']=='coal','ramp_limit_up']=0.1
n.generators.loc[n.generators['carrier']=='coal','ramp_limit_down']=0.1
n.generators.loc[n.generators['carrier']=='OCGT','marginal_cost']=125.0 + co2_add['OCGT']
n.generators.loc[n.generators['carrier']=='oil','marginal_cost']=180.0 + co2_add['oil']

# ── CCGT FLEX SPLIT ───────────────────────────────────────────────────
ccgt_es=(gen_cc=='ES')&(n.generators['carrier']=='CCGT')
ccgt_bus_counts_pre=n.generators[ccgt_es.values]['bus'].value_counts()
ccgt_idx=n.generators[ccgt_es.values].sort_values('p_nom',ascending=False).index
flex_pct=0.20
for g in ccgt_idx:
    orig_pnom=n.generators.at[g,'p_nom']
    flex_pnom=orig_pnom*flex_pct
    remain_pnom=orig_pnom*(1-flex_pct)
    n.generators.at[g,'p_nom']=remain_pnom
    bus=n.generators.at[g,'bus']
    n.add('Generator',f'{g}_flex',bus=bus,carrier='CCGT_Flex',
          p_nom=flex_pnom,marginal_cost=rng.uniform(90,105) + co2_add['CCGT_Flex'],
          ramp_limit_up=1.0,ramp_limit_down=1.0,p_min_pu=0)
flex_total=n.generators[n.generators['carrier']=='CCGT_Flex']['p_nom'].sum()
print(f'\n=== CCGT FLEX SPLIT ===')
flex_lo = 90 + co2_add["CCGT_Flex"]
flex_hi = 105 + co2_add["CCGT_Flex"]
print(f"  CCGT_Flex MC: {flex_lo:.0f}-{flex_hi:.0f} EUR/MWh ({flex_total:.0f} MW total)")

# ── ADD PEAKERS ───────────────────────────────────────────────────────
es_loads=n.loads[n.loads.bus.astype(str).str.startswith('ES')].copy()
es_loads['annual']=es_loads.index.map(lambda x: n.loads_t.p_set[x].sum())
load_total=es_loads['annual'].sum()
es_loads['load_share']=es_loads['annual']/load_total
ccgt_bus_share=ccgt_bus_counts_pre/ccgt_bus_counts_pre.sum()
W=pd.Series(0.0,index=es_loads.bus)
for b in W.index:
    W[b]=0.7*es_loads.loc[es_loads.bus==b,'load_share'].values[0]+0.3*ccgt_bus_share.get(b,0)
W=W/W.sum()

peaker_config=[('OCGT',1149,125,145),('diesel',769,160,185)]
for pcar,pnom,plo,phi in peaker_config:
    added=0
    for b in W.sort_values(ascending=False).index:
        if added>=pnom: break
        alloc=round(pnom*W[b])
        if alloc<1: continue
        n.add('Generator',f'{b} {pcar}',bus=b,carrier=pcar,
              p_nom=alloc,marginal_cost=rng.uniform(plo,phi) + co2_add[pcar],
              ramp_limit_up=1.0,ramp_limit_down=1.0,p_min_pu=0)
        added+=alloc
    actual=n.generators[n.generators['carrier']==pcar]['p_nom'].sum()
    print(f'  {pcar}: target={pnom} MW, actual={actual:.0f} MW, MC=€{plo+co2_add[pcar]:.0f}-{phi+co2_add[pcar]:.0f}/MWh')

# ── Ramping ───────────────────────────────────────────────────────────
def ramp(car,up,dn):
    m=n.generators['carrier']==car
    n.generators.loc[m,'ramp_limit_up']=up; n.generators.loc[m,'ramp_limit_down']=dn
ramp('nuclear',0.20,0.20); ramp('biomass',0.30,0.30); ramp('cogen',0.30,0.30)
ramp('OCGT',1.00,1.00); ramp('oil',0.90,0.90); ramp('CCGT_Flex',1.00,1.00); ramp('diesel',1.00,1.00)
def ramp_ccgt(cc,tiers):
    m=(gen_cc==cc)&(n.generators['carrier']=='CCGT')
    idx=n.generators.loc[m].sort_values('p_nom',ascending=False).index
    if len(idx)==0: return
    for si,(ru,rd) in zip(np.array_split(idx,len(tiers)),tiers):
        n.generators.loc[si,'ramp_limit_up']=ru; n.generators.loc[si,'ramp_limit_down']=rd
ramp_ccgt('ES',[(0.80,0.80),(0.65,0.65),(0.50,0.50)])
ramp_ccgt('PT',[(0.80,0.80),(0.65,0.65),(0.50,0.50)])
ramp_ccgt('FR',[(0.65,0.65),(0.50,0.50)])
es_hyd_soc=(n.storage_units.bus.map(lambda b:str(b).startswith('ES'))&(n.storage_units['carrier']=='hydro'))
n.storage_units.loc[es_hyd_soc,'state_of_charge_initial']=0.50*n.storage_units.loc[es_hyd_soc,'p_nom']*n.storage_units.loc[es_hyd_soc,'max_hours']

# ── Disable expansion ─────────────────────────────────────────────────
n.generators['p_nom_extendable']=False; n.links['p_nom_extendable']=False
n.storage_units['p_nom_extendable']=False; n.stores['e_nom_extendable']=False; n.lines['s_nom_extendable']=False

# ── Remove interconnectors ────────────────────────────────────────────
ic=n.links[n.links['carrier'].str.contains('DC_ic',na=False)]
print(f'\nInterconnectors: {len(ic)} → setting p_nom=0')
n.links.loc[ic.index,'p_nom']=0

# ── Load shedding ─────────────────────────────────────────────────────
for b in n.buses.index:
    if str(b).startswith('ES') or str(b).startswith('FR') or str(b).startswith('PT'):
        n.add('Generator',f'{b} load_shedding',bus=b,carrier='load_shedding',
              p_nom=50_000,marginal_cost=3_000,p_min_pu=0)

# ── Restrict to Apr-Jun 2024 ─────────────────────────────────────────
snapshots = n.snapshots
shoulder = snapshots[(snapshots.month >= 4) & (snapshots.month <= 6)]
n.set_snapshots(shoulder)
print(f'\n=== SHOULDER PERIOD: {len(shoulder)} snapshots (Apr-Jun 2024) ===')

# ── Solve ─────────────────────────────────────────────────────────────
print(f'\nSolving {len(n.snapshots)} snapshots...')
res=n.optimize(solver_name='gurobi',solver_options={'OutputFlag':1,'TimeLimit':1800,'Threads':5,'DualReductions':0})
print(f'Status: {res[0]} | Objective: {n.objective:.2e} EUR')

# ── Save ──────────────────────────────────────────────────────────────
sdir=f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(sdir,exist_ok=True)
n.export_to_netcdf(f'{sdir}/solved_2024_shoulder_constrained.nc')
print('Saved.')

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════

# ── 1. PRICE ANALYSIS ────────────────────────────────────────────────
p=n.buses_t.marginal_price
es_b=[b for b in p.columns if str(b).startswith('ES')]
es_ac=[b for b in es_b if not any(suffix in str(b) for suffix in [' H2',' battery'])]
p_ac=p[es_ac]

print('\n' + '='*60)
print('PRICE ANALYSIS')
print('='*60)

print('\n=== ES AC BUSES ONLY ===')
s_ac=p_ac.stack()
print(f'Mean: {s_ac.mean():.1f}, Median: {s_ac.median():.1f}, Max: {s_ac.max():.1f}, Min: {s_ac.min():.1f}')

# Price spread across nodes
print(f'\n=== NODAL PRICE SPREAD ===')
hourly_spread = p_ac.max(axis=1) - p_ac.min(axis=1)
print(f'Mean hourly spread: {hourly_spread.mean():.1f} EUR/MWh')
print(f'Max hourly spread:  {hourly_spread.max():.1f} EUR/MWh')
print(f'Hours with spread >1: {(hourly_spread>1).sum()}/{len(hourly_spread)} ({(hourly_spread>1).mean()*100:.1f}%)')
print(f'Hours with spread >5: {(hourly_spread>5).sum()}/{len(hourly_spread)} ({(hourly_spread>5).mean()*100:.1f}%)')
print(f'Hours with spread >10: {(hourly_spread>10).sum()}/{len(hourly_spread)} ({(hourly_spread>10).mean()*100:.1f}%)')

# Monthly stats
print('\n=== MONTHLY PRICES (AC buses) ===')
for m in [4,5,6]:
    m_p = p_ac.loc[p_ac.index.month == m]
    spread = m_p.max(axis=1) - m_p.min(axis=1)
    print(f'  Month {m:2d}: Mean={m_p.stack().mean():7.1f}, Max={m_p.stack().max():7.1f}, '
          f'Min={m_p.stack().min():7.1f}, Spread={spread.mean():.1f}')

# ── 2. COMPARE TO OMIE ───────────────────────────────────────────────
omie=pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv')
omie['datetime']=pd.to_datetime(omie['Datetime (UTC)'],format='%d/%m/%y %H:%M')
omie=omie.set_index('datetime')['Price (EUR/MWhe)']
omie_shoulder=omie[(omie.index>=pd.Timestamp('2024-04-01'))&(omie.index<=pd.Timestamp('2024-06-30 23:00'))]

print('\n=== MODEL vs REAL (OMIE) — ES AC Buses (Apr-Jun) ===')
model_mean=p_ac.mean(axis=1)
print(f'Model mean: {model_mean.mean():.1f} EUR/MWh')
print(f'OMIE   mean: {omie_shoulder.mean():.1f} EUR/MWh')
print(f'Model max:  {p_ac.stack().max():.1f} EUR/MWh')
print(f'OMIE   max:  {omie_shoulder.max():.1f} EUR/MWh')

# Monthly comparison
print('\n=== MONTHLY COMPARISON ===')
print(f'  {"Month":6s} {"Model Mean":10s} {"OMIE Mean":10s} {"Ratio":8s} {"Model Max":10s} {"OMIE Max":10s}')
print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*8} {"-"*10} {"-"*10}')
for m in [4,5,6]:
    m_model = model_mean[model_mean.index.month == m]
    m_omie = omie_shoulder[omie_shoulder.index.month == m]
    ratio = m_model.mean() / m_omie.mean() if m_omie.mean() > 0 else float('inf')
    print(f'  Month {m:2d}  {m_model.mean():8.1f}   {m_omie.mean():8.1f}   {ratio:6.2f}x  {m_model.max():8.1f}   {m_omie.max():8.1f}')

# ── 3. TRANSMISSION ANALYSIS ─────────────────────────────────────────
print('\n' + '='*60)
print('TRANSMISSION ANALYSIS')
print('='*60)

loading = n.lines_t.p0.abs().div(n.lines.s_nom, axis=1)
es_line_idx = [l for l in es_lines if l in loading.columns]

print(f'\n=== LINE LOADING (ES-ES, post-scaling) ===')
print(f'Max loading: {loading[es_line_idx].max().max()*100:.1f}%')
print(f'Mean loading: {loading[es_line_idx].mean().mean()*100:.1f}%')
print(f'Lines with any hour >90%: {(loading[es_line_idx] > 0.9).any().sum()} / {len(es_line_idx)}')
print(f'Lines with any hour >70%: {(loading[es_line_idx] > 0.7).any().sum()} / {len(es_line_idx)}')

# Top congested lines
max_load = loading[es_line_idx].max().sort_values(ascending=False).head(10)
print('\nTop 10 congested lines:')
for l in max_load.index:
    b0 = n.lines.at[l, 'bus0']
    b1 = n.lines.at[l, 'bus1']
    snom = n.lines.at[l, 's_nom']
    print(f'  {l:6s}: {max_load[l]*100:.1f}% max, {loading[l].mean()*100:.1f}% mean ({b0}→{b1}, {snom:.0f}MW)')

# ── 4. DISPATCH ANALYSIS ─────────────────────────────────────────────
print('\n' + '='*60)
print('DISPATCH ANALYSIS')
print('='*60)

g=n.generators_t.p
total_dispatch=g.sum().groupby(n.generators.carrier).sum() / 1e6
print('\n=== TOTAL GENERATION (TWh, Apr-Jun) ===')
for car, val in total_dispatch.sort_values(ascending=False).items():
    print(f'  {car:20s}: {val:.2f} TWh')

# Storage
su_dispatch=n.storage_units_t.p
su_dispatch_pos=su_dispatch[su_dispatch>0].sum().groupby(n.storage_units.carrier).sum() / 1e6
print('\n=== STORAGE DISPATCH (TWh) ===')
for car, val in su_dispatch_pos.sort_values(ascending=False).items():
    print(f'  {car:20s}: {val:.2f} TWh')

# Hydro
es_hyd_cols = [c for c in n.storage_units_t.state_of_charge.columns 
               if c in n.storage_units.index and 
               str(n.storage_units.at[c,'bus']).startswith('ES') and
               n.storage_units.at[c,'carrier']=='hydro']
soc = n.storage_units_t.state_of_charge[es_hyd_cols]
print(f'\n=== HYDRO SOC DYNAMICS ===')
print(f'Initial SOC: {soc.iloc[0].sum()/1e6:.2f} TWh')
print(f'Final SOC:   {soc.iloc[-1].sum()/1e6:.2f} TWh')
hyd_dispatch = n.storage_units_t.p[es_hyd_cols]
print(f'Hydro dispatched: {hyd_dispatch[hyd_dispatch>0].sum().sum()/1e6:.2f} TWh')
inflow = n.storage_units_t.inflow[es_hyd_cols]
print(f'Hydro inflow:     {inflow.sum().sum()/1e6:.2f} TWh')

# ── 5. COMPARE DISPATCH TO REALITY ───────────────────────────────────
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
real_shoulder = real[(real.index >= pd.Timestamp('2024-04-01')) & (real.index <= pd.Timestamp('2024-06-30'))]

model_gen = n.generators_t.p
model_daily = model_gen.groupby(n.generators.carrier, axis=1).sum().resample('D').sum() / 1000
model_hyd_daily = hyd_dispatch.sum(axis=1).resample('D').sum() / 1000

print('\n=== MODEL vs REAL DISPATCH (GWh/day mean, Apr-Jun) ===')
print(f'  {"Tech":15s} {"Model":10s} {"Real":10s} {"Ratio":8s}')
print(f'  {"-"*15} {"-"*10} {"-"*10} {"-"*8}')
for tech, real_col in [('Hydro','Hidráulica'), ('CCGT','Ciclo combinado'), 
                        ('Nuclear','Nuclear'), ('Solar','Solar fotovoltaica'),
                        ('Wind','Eólica')]:
    if tech == 'Hydro':
        m_val = model_hyd_daily.mean()
    elif tech == 'CCGT':
        m_val = model_daily['CCGT'].mean() if 'CCGT' in model_daily.columns else 0
    elif tech == 'Nuclear':
        m_val = model_daily['nuclear'].mean() if 'nuclear' in model_daily.columns else 0
    elif tech == 'Solar':
        m_val = model_daily['solar'].mean() if 'solar' in model_daily.columns else 0
    elif tech == 'Wind':
        m_val = (model_daily.get('onwind',0) + model_daily.get('offwind',0) + model_daily.get('offwind-float',0)).mean()
    r_val = real_shoulder[real_col].mean()
    ratio = m_val / r_val if r_val > 0 else 0
    print(f'  {tech:15s} {m_val:8.0f} GWh  {r_val:8.0f} GWh  {ratio:6.2f}x')

# ── RESIDUAL LOAD (ES only) ────────────────────────────────────────────
print('\n=== RESIDUAL LOAD (ES only) ===')
load_es = n.loads_t.p_set[[l for l in n.loads_t.p_set.columns if str(n.loads.at[l,'bus']).startswith('ES')]]
total_load = load_es.sum(axis=1) / 1e3  # MW → GW
re_techs = ['solar','onwind','offwind','offwind-float']
re_gen = pd.DataFrame()
for t in re_techs:
    tg = [g for g in n.generators.index if n.generators.at[g,'carrier']==t and str(n.generators.at[g,'bus']).startswith('ES')]
    if tg:
        re_gen[t] = n.generators_t.p[tg].sum(axis=1) / 1e3  # MW → GW
total_re = re_gen.sum(axis=1) if len(re_gen.columns) > 0 else pd.Series(0, index=total_load.index)
residual = total_load - total_re
print(f'Mean residual load: {residual.mean():.1f} GW')
print(f'Hours negative: {(residual<0).sum()} / {len(residual)} ({(residual<0).mean()*100:.1f}%)')

# ── 6. PLOTS — Comprehensive Spanish Analysis ────────────────────────
print('\nGenerating comprehensive Spanish analysis plots...')

# ── 6a. PRICE TIMESERIES (hourly) ────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(model_mean.index, model_mean.values, label=f'Model (trans×{TRANS_FACTOR})', color='steelblue', lw=0.8)
ax.plot(omie_shoulder.index, omie_shoulder.values, label='OMIE 2024', color='darkorange', lw=0.4, alpha=0.5)
ax.fill_between(p_ac.index, p_ac.min(axis=1), p_ac.max(axis=1), alpha=0.15, color='steelblue', label='Nodal range')
ax.set_ylabel('EUR/MWh')
ax.set_title(f'ES Price Timeseries: Constrained (×{TRANS_FACTOR}) vs OMIE — Apr-Jun 2024')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.tight_layout()
plt.savefig(f'{sdir}/ES_price_timeseries.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_price_timeseries.png')

# ── 6b. PRICE DURATION CURVE ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
model_pdc = p_ac.stack().sort_values(ascending=False).values
omie_pdc = omie_shoulder.sort_values(ascending=False).values
ax.plot(np.arange(1, len(model_pdc)+1)/len(model_pdc)*100, model_pdc,
        label=f'Model (mean={model_mean.mean():.1f}, max={model_pdc.max():.0f})', color='steelblue', lw=1.5)
ax.plot(np.arange(1, len(omie_pdc)+1)/len(omie_pdc)*100, omie_pdc,
        label=f'OMIE (mean={omie_shoulder.mean():.1f}, max={omie_pdc.max():.0f})', color='darkorange', lw=1.5, alpha=0.7)
ax.axhline(y=100, color='red', ls='--', lw=0.5, alpha=0.5, label='€100/MWh')
ax.axhline(y=50, color='green', ls='--', lw=0.5, alpha=0.5, label='€50/MWh')
ax.set_xlabel('Exceedance %')
ax.set_ylabel('EUR/MWh')
ax.set_title('ES Price Duration Curve: Model vs OMIE (Apr-Jun 2024)')
ax.legend(fontsize=9)
ax.set_xlim(0, 100)
plt.tight_layout()
plt.savefig(f'{sdir}/ES_price_duration_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_price_duration_curve.png')

# ── 6c. NODAL PRICE SPREAD ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(hourly_spread.index, hourly_spread.values, color='crimson', lw=0.5, alpha=0.7)
ax.axhline(y=1, color='grey', ls='--', lw=0.5, label='€1/MWh')
ax.axhline(y=10, color='orange', ls='--', lw=0.5, label='€10/MWh')
ax.set_ylabel('EUR/MWh')
ax.set_title(f'ES Nodal Price Spread (max-min): Mean €{hourly_spread.mean():.1f}, Max €{hourly_spread.max():.0f}')
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.tight_layout()
plt.savefig(f'{sdir}/ES_nodal_spread.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_nodal_spread.png')

# ── 6d. DAILY DISPATCH: Model vs Real (all techs) ────────────────────
fig, axes = plt.subplots(3, 2, figsize=(18, 14))

tech_map = [('Nuclear','Nuclear'), ('CCGT','Ciclo combinado'),
            ('Solar','Solar fotovoltaica'), ('Wind','Eólica'),
            ('Hydro','Hidráulica')]
colors_daily = {'Nuclear':'purple', 'CCGT':'steelblue', 'Solar':'gold',
                'Wind':'green', 'Hydro':'blue'}

for idx, (tech, real_col) in enumerate(tech_map):
    ax = axes[idx//2, idx%2]
    if tech == 'Hydro':
        m_daily = model_hyd_daily
    elif tech == 'Wind':
        m_daily = (model_daily.get('onwind',0) + model_daily.get('offwind',0) + model_daily.get('offwind-float',0))
    elif tech == 'CCGT':
        m_daily = model_daily['CCGT'] if 'CCGT' in model_daily.columns else pd.Series(0, index=model_hyd_daily.index)
    elif tech == 'Solar':
        m_daily = model_daily['solar'] if 'solar' in model_daily.columns else pd.Series(0, index=model_hyd_daily.index)
    elif tech == 'Nuclear':
        m_daily = model_daily['nuclear'] if 'nuclear' in model_daily.columns else pd.Series(0, index=model_hyd_daily.index)
    r_daily = real_shoulder[real_col]
    ax.plot(m_daily.index, m_daily.values, color=colors_daily[tech], lw=0.8, label=f'Model {tech}')
    ax.plot(r_daily.index, r_daily.values, color=colors_daily[tech], ls='--', lw=0.5, alpha=0.5, label=f'Real {tech}')
    ax.set_ylabel('GWh/day')
    ax.set_title(f'{tech}: Model vs Real (daily)')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# 6d-6: Weekly dispatch comparison
ax = axes[2,0]
model_weekly = pd.DataFrame()
real_weekly = pd.DataFrame()
for tech, real_col in tech_map:
    if tech == 'Hydro':
        m_w = model_hyd_daily.resample('W').sum()
    elif tech == 'Wind':
        m_w = (model_daily.get('onwind',0) + model_daily.get('offwind',0) + model_daily.get('offwind-float',0)).resample('W').sum()
    elif tech == 'CCGT':
        m_w = model_daily['CCGT'].resample('W').sum() if 'CCGT' in model_daily.columns else pd.Series(0, index=model_hyd_daily.resample('W').sum().index)
    elif tech == 'Solar':
        m_w = model_daily['solar'].resample('W').sum() if 'solar' in model_daily.columns else pd.Series(0, index=model_hyd_daily.resample('W').sum().index)
    elif tech == 'Nuclear':
        m_w = model_daily['nuclear'].resample('W').sum() if 'nuclear' in model_daily.columns else pd.Series(0, index=model_hyd_daily.resample('W').sum().index)
    r_w = real_shoulder[real_col].resample('W').sum()
    model_weekly[tech] = m_w
    real_weekly[tech] = r_w

x = np.arange(len(model_weekly))
w = 0.35
for i, tech in enumerate([t[0] for t in tech_map]):
    ax.bar(x - w/2, model_weekly[tech].values, w, label=f'Model {tech}', color=colors_daily[tech], alpha=0.8)
    ax.bar(x + w/2, real_weekly[tech].values, w, label=f'Real {tech}', color=colors_daily[tech], alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels([f'W{k+1}' for k in range(len(model_weekly))], rotation=45, fontsize=8)
ax.set_ylabel('GWh/week')
ax.set_title('Weekly Dispatch: Model vs Real')
ax.legend(fontsize=7, ncol=2)

# 6d-7: Monthly dispatch comparison
ax = axes[2,1]
model_monthly = model_weekly.resample('ME').sum()
real_monthly = real_weekly.resample('ME').sum()
x = np.arange(len(model_monthly))
for i, tech in enumerate([t[0] for t in tech_map]):
    ax.bar(x - w/2, model_monthly[tech].values, w, label=f'Model {tech}', color=colors_daily[tech], alpha=0.8)
    ax.bar(x + w/2, real_monthly[tech].values, w, label=f'Real {tech}', color=colors_daily[tech], alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(['Apr','May','Jun'], fontsize=10)
ax.set_ylabel('GWh/month')
ax.set_title('Monthly Dispatch: Model vs Real')
ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(f'{sdir}/ES_dispatch_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_dispatch_comparison.png')

# ── 6e. WEEKLY DISPATCH STACK (model only) ───────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
model_dispatch = model_gen.groupby(n.generators.carrier, axis=1).sum()
carriers_to_plot = ['nuclear', 'CCGT', 'CCGT_Flex', 'coal', 'OCGT', 'diesel', 'oil',
                    'solar', 'onwind', 'offwind', 'offwind-float', 'hydro', 'biomass', 'cogen']
carriers_to_plot = [c for c in carriers_to_plot if c in model_dispatch.columns]
weekly = model_dispatch[carriers_to_plot].resample('W').mean()
colors = {'nuclear':'purple', 'CCGT':'steelblue', 'CCGT_Flex':'lightblue', 'coal':'black',
          'OCGT':'orange', 'diesel':'brown', 'oil':'darkred',
          'solar':'gold', 'onwind':'green', 'offwind':'teal', 'offwind-float':'cyan',
          'hydro':'blue', 'biomass':'darkgreen', 'cogen':'grey'}
weekly.plot.area(ax=ax, color=[colors.get(c, 'grey') for c in carriers_to_plot],
                 legend=True, alpha=0.8)
ax.set_ylabel('MW (weekly avg)')
ax.set_title('ES Model Dispatch by Carrier (Weekly Avg)')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.savefig(f'{sdir}/ES_dispatch_stack.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_dispatch_stack.png')

# ── 6f. MONTHLY BAR CHART: Model vs Real ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
techs_plot = ['Nuclear','CCGT','Solar','Wind','Hydro']
model_monthly_totals = {}
real_monthly_totals = {}
for tech, real_col in tech_map:
    if tech == 'Hydro':
        m_m = model_hyd_daily.resample('ME').sum()
    elif tech == 'Wind':
        m_m = (model_daily.get('onwind',0) + model_daily.get('offwind',0) + model_daily.get('offwind-float',0)).resample('ME').sum()
    elif tech == 'CCGT':
        m_m = model_daily['CCGT'].resample('ME').sum() if 'CCGT' in model_daily.columns else pd.Series(0, index=model_hyd_daily.resample('ME').sum().index)
    elif tech == 'Solar':
        m_m = model_daily['solar'].resample('ME').sum() if 'solar' in model_daily.columns else pd.Series(0, index=model_hyd_daily.resample('ME').sum().index)
    elif tech == 'Nuclear':
        m_m = model_daily['nuclear'].resample('ME').sum() if 'nuclear' in model_daily.columns else pd.Series(0, index=model_hyd_daily.resample('ME').sum().index)
    r_m = real_shoulder[real_col].resample('ME').sum()
    model_monthly_totals[tech] = m_m
    real_monthly_totals[tech] = r_m

months = ['Apr','May','Jun']
x = np.arange(len(months))
w = 0.35
for i, tech in enumerate(techs_plot):
    m_vals = [model_monthly_totals[tech].iloc[j]/1e3 for j in range(3)]
    r_vals = [real_monthly_totals[tech].iloc[j]/1e3 for j in range(3)]
    ax.bar(x + i*0.22 - 0.5, m_vals, 0.18, label=f'Model {tech}', color=colors_daily[tech], alpha=0.8)
    ax.bar(x + i*0.22 - 0.5 + 0.09, r_vals, 0.18, label=f'Real {tech}', color=colors_daily[tech], alpha=0.3)
ax.set_xticks(x)
ax.set_xticklabels(months)
ax.set_ylabel('GWh/month')
ax.set_title('ES Monthly Generation: Model vs Real (Apr-Jun 2024)')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(f'{sdir}/ES_monthly_bars.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_monthly_bars.png')

# ── 6g. PRICE vs RESIDUAL LOAD ───────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(16, 5))
color1, color2 = 'steelblue', 'crimson'
ax1.plot(model_mean.index, model_mean.values, color=color1, lw=0.8, label='Model Price')
ax1.set_ylabel('EUR/MWh', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax2 = ax1.twinx()
ax2.plot(residual.index, residual.values, color=color2, lw=0.5, alpha=0.5, label='Residual Load')
ax2.set_ylabel('GW', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax1.set_title('ES Price vs Residual Load (Apr-Jun 2024)')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labels1+labels2, fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.tight_layout()
plt.savefig(f'{sdir}/ES_price_vs_residual.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved: ES_price_vs_residual.png')

print('\nAll Spanish analysis plots saved!')
print(f'  Files in: {sdir}/')
print(f'  1. ES_price_timeseries.png     — Hourly price with nodal range')
print(f'  2. ES_price_duration_curve.png  — Price duration curve')
print(f'  3. ES_nodal_spread.png          — Nodal price spread')
print(f'  4. ES_dispatch_comparison.png   — Daily/weekly/monthly dispatch vs real')
print(f'  5. ES_dispatch_stack.png        — Weekly dispatch stack')
print(f'  6. ES_monthly_bars.png          — Monthly bar chart')
print(f'  7. ES_price_vs_residual.png     — Price vs residual load')
