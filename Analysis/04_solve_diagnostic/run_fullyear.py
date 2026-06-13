#!/usr/bin/env python3
"""Run full-year 2024 with CO2 price, MCs, ramping, no expansion, no ICs.
Uses the hydro-fixed network (23 TWh energy cap)."""
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates, pypsa, os
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10})
rng=np.random.default_rng(42)

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n=pypsa.Network(f'{ROOT}/resources/networks/50n_ES_FR_PT.nc')
print(f'Loaded: {len(n.buses)} buses, {len(n.generators)} gens, {len(n.snapshots)} snapshots')

bus_cc=pd.Series(n.buses.index.astype(str).str[:2].values,index=n.buses.index)
gen_cc=n.generators.bus.map(bus_cc); su_cc=n.storage_units.bus.map(bus_cc)

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

# ── Solve: Full Year 2024 ────────────────────────────────────────────
print(f'\nSolving {len(n.snapshots)} snapshots (full year 2024)...')
res=n.optimize(solver_name='gurobi',solver_options={'OutputFlag':1,'TimeLimit':7200,'Threads':5,'DualReductions':0})
print(f'Status: {res[0]} | Objective: {n.objective:.2e} EUR')

# ── Save ──────────────────────────────────────────────────────────────
sdir=f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(sdir,exist_ok=True)
n.export_to_netcdf(f'{sdir}/solved_2024_fullyear_co2.nc')
print('Saved.')

# ══════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════

# ── 1. PRICE ANALYSIS ────────────────────────────────────────────────
p=n.buses_t.marginal_price
es_b=[b for b in p.columns if str(b).startswith('ES')]
p_es=p[es_b]

# AC buses only (not H2 or battery)
es_ac=[b for b in es_b if not any(suffix in str(b) for suffix in [' H2',' battery'])]
p_ac=p[es_ac]

print('\n' + '='*60)
print('PRICE ANALYSIS')
print('='*60)

print('\n=== ALL ES BUSES (incl H2/battery) ===')
s=p_es.stack()
print(f'Mean: {s.mean():.1f}, Median: {s.median():.1f}, Max: {s.max():.1f}, Min: {s.min():.1f}')

print('\n=== ES AC BUSES ONLY ===')
s_ac=p_ac.stack()
print(f'Mean: {s_ac.mean():.1f}, Median: {s_ac.median():.1f}, Max: {s_ac.max():.1f}, Min: {s_ac.min():.1f}')

# Monthly stats
print('\n=== MONTHLY PRICES (AC buses) ===')
for m in range(1,13):
    m_p = p_ac.loc[p_ac.index.month == m]
    print(f'  Month {m:2d}: Mean={m_p.stack().mean():7.1f}, Max={m_p.stack().max():7.1f}, Min={m_p.stack().min():7.1f}')

# Price distribution
mean_p=p_ac.mean(axis=1)
print('\n=== PRICE DISTRIBUTION (AC buses) ===')
for lo, hi, label in [(0, 10, '0-10'), (10, 30, '10-30'), (30, 50, '30-50'),
                       (50, 75, '50-75'), (75, 100, '75-100'), (100, 150, '100-150'),
                       (150, 200, '150-200'), (200, 500, '200-500'), (500, 3000, '500-3000')]:
    count = ((mean_p >= lo) & (mean_p < hi)).sum()
    if count > 0:
        print(f'  {label:10s} EUR/MWh: {count:4d} hours ({count/len(mean_p)*100:5.1f}%)')

# ── 2. COMPARE TO OMIE ───────────────────────────────────────────────
omie=pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv')
omie['datetime']=pd.to_datetime(omie['Datetime (UTC)'],format='%d/%m/%y %H:%M')
omie=omie.set_index('datetime')['Price (EUR/MWhe)']
omie_year=omie[(omie.index>=pd.Timestamp('2024-01-01'))&(omie.index<=pd.Timestamp('2024-12-31 23:00'))]

print('\n=== MODEL vs REAL (OMIE) — ES AC Buses ===')
model_mean=p_ac.mean(axis=1)
print(f'Model mean: {model_mean.mean():.1f} EUR/MWh')
print(f'OMIE   mean: {omie_year.mean():.1f} EUR/MWh')
print(f'Model max:  {p_ac.stack().max():.1f} EUR/MWh')
print(f'OMIE   max:  {omie_year.max():.1f} EUR/MWh')
print(f'Model >50:  {(model_mean>50).sum()}/{len(model_mean)} hours ({(model_mean>50).mean()*100:.1f}%)')
print(f'OMIE   >50:  {(omie_year>50).sum()}/{len(omie_year)} hours ({(omie_year>50).mean()*100:.1f}%)')
print(f'Model >100: {(model_mean>100).sum()}/{len(model_mean)} hours ({(model_mean>100).mean()*100:.1f}%)')
print(f'OMIE   >100: {(omie_year>100).sum()}/{len(omie_year)} hours ({(omie_year>100).mean()*100:.1f}%)')

# Monthly comparison
print('\n=== MONTHLY COMPARISON ===')
print(f'  {"Month":6s} {"Model Mean":10s} {"OMIE Mean":10s} {"Ratio":8s} {"Model Max":10s} {"OMIE Max":10s}')
print(f'  {"-"*6} {"-"*10} {"-"*10} {"-"*8} {"-"*10} {"-"*10}')
for m in range(1,13):
    m_model = model_mean[model_mean.index.month == m]
    m_omie = omie_year[omie_year.index.month == m]
    ratio = m_model.mean() / m_omie.mean() if m_omie.mean() > 0 else float('inf')
    print(f'  Month {m:2d}  {m_model.mean():8.1f}   {m_omie.mean():8.1f}   {ratio:6.2f}x  {m_model.max():8.1f}   {m_omie.max():8.1f}')

# ── 3. DISPATCH ANALYSIS ─────────────────────────────────────────────
print('\n' + '='*60)
print('DISPATCH ANALYSIS')
print('='*60)

# Generator dispatch
g=n.generators_t.p
total_dispatch=g.sum().groupby(n.generators.carrier).sum() / 1e6  # TWh
print('\n=== TOTAL GENERATION (TWh) ===')
for car, val in total_dispatch.sort_values(ascending=False).items():
    print(f'  {car:20s}: {val:.2f} TWh')

# Storage dispatch
su_dispatch=n.storage_units_t.p
su_dispatch_pos=su_dispatch[su_dispatch>0].sum().groupby(n.storage_units.carrier).sum() / 1e6
print('\n=== STORAGE DISPATCH (TWh) ===')
for car, val in su_dispatch_pos.sort_values(ascending=False).items():
    print(f'  {car:20s}: {val:.2f} TWh')

# Hydro SOC dynamics
es_hyd_cols = [c for c in n.storage_units_t.state_of_charge.columns 
               if c in n.storage_units.index and 
               str(n.storage_units.at[c,'bus']).startswith('ES') and
               n.storage_units.at[c,'carrier']=='hydro']
soc = n.storage_units_t.state_of_charge[es_hyd_cols]
print(f'\n=== HYDRO SOC DYNAMICS ===')
print(f'Initial SOC: {soc.iloc[0].sum()/1e6:.2f} TWh')
print(f'Final SOC:   {soc.iloc[-1].sum()/1e6:.2f} TWh')
print(f'Min SOC:     {soc.min(axis=0).sum()/1e6:.2f} TWh')
print(f'Net change:  {(soc.iloc[-1]-soc.iloc[0]).sum()/1e6:.2f} TWh')

# Hydro dispatch
hyd_dispatch = n.storage_units_t.p[es_hyd_cols]
print(f'Hydro dispatched: {hyd_dispatch[hyd_dispatch>0].sum().sum()/1e6:.2f} TWh')
print(f'Hydro charged:    {(-hyd_dispatch[hyd_dispatch<0]).sum().sum()/1e6:.2f} TWh')

# Inflow
inflow = n.storage_units_t.inflow[es_hyd_cols]
print(f'Hydro inflow:     {inflow.sum().sum()/1e6:.2f} TWh')

# ── 4. COMPARE DISPATCH TO REALITY ───────────────────────────────────
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

real_year = real[(real.index >= pd.Timestamp('2024-01-01')) & (real.index <= pd.Timestamp('2024-12-31'))]

# Model daily dispatch by carrier
model_gen = n.generators_t.p
model_daily = model_gen.groupby(n.generators.carrier, axis=1).sum().resample('D').sum() / 1000  # GWh/day

# Model hydro daily
model_hyd_daily = hyd_dispatch.sum(axis=1).resample('D').sum() / 1000  # GWh/day

print('\n=== MODEL vs REAL DISPATCH (GWh/day mean) ===')
print(f'  {"Period":12s} {"Model Hydro":12s} {"Real Hydro":12s} {"Ratio":8s} {"Model CCGT":12s} {"Real CCGT":12s} {"Ratio":8s}')
print(f'  {"-"*12} {"-"*12} {"-"*12} {"-"*8} {"-"*12} {"-"*12} {"-"*8}')
for m in range(1,13):
    m_model_hyd = model_hyd_daily[model_hyd_daily.index.month == m].mean()
    m_real_hyd = real_year.loc[real_year.index.month == m, 'Hidráulica'].mean()
    m_model_ccgt = model_daily.loc[model_daily.index.month == m, 'CCGT'].mean() if 'CCGT' in model_daily.columns else 0
    m_real_ccgt = real_year.loc[real_year.index.month == m, 'Ciclo combinado'].mean()
    hyd_ratio = m_model_hyd / m_real_hyd if m_real_hyd > 0 else 0
    ccgt_ratio = m_model_ccgt / m_real_ccgt if m_real_ccgt > 0 else 0
    print(f'  Month {m:2d}     {m_model_hyd:8.0f} GWh  {m_real_hyd:8.0f} GWh  {hyd_ratio:6.2f}x  {m_model_ccgt:8.0f} GWh  {m_real_ccgt:8.0f} GWh  {ccgt_ratio:6.2f}x')

# ── 5. PLOTS ──────────────────────────────────────────────────────────
print('\nGenerating plots...')

# 5a. Price timeseries
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

# Top: full timeseries
ax1.plot(model_mean.index, model_mean.values, label='Model (AC, CO2=65, hydro=23TWh)', color='steelblue', lw=0.6)
ax1.plot(omie_year.index, omie_year.values, label='OMIE 2024', color='darkorange', lw=0.4, alpha=0.5)
ax1.axhline(y=100, color='red', ls='--', lw=0.5, alpha=0.5)
ax1.set_ylabel('EUR/MWh')
ax1.set_title('ES Price: Model (AC buses, CO₂=€65/t, hydro=23TWh) vs Real OMIE — Full Year 2024')
ax1.legend(fontsize=9)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Bottom: price duration curve
model_pdc = p_ac.stack().sort_values(ascending=False).values
omie_pdc = omie_year.sort_values(ascending=False).values
ax2.plot(np.arange(1, len(model_pdc)+1)/len(model_pdc)*100, model_pdc, 
         label=f'Model (mean={model_mean.mean():.1f}, max={model_pdc.max():.0f})', color='steelblue')
ax2.plot(np.arange(1, len(omie_pdc)+1)/len(omie_pdc)*100, omie_pdc,
         label=f'OMIE (mean={omie_year.mean():.1f}, max={omie_pdc.max():.0f})', color='darkorange', alpha=0.7)
ax2.axhline(y=100, color='red', ls='--', lw=0.5, alpha=0.5)
ax2.set_xlabel('Exceedance %')
ax2.set_ylabel('EUR/MWh')
ax2.set_title('Price Duration Curve')
ax2.legend(fontsize=9)
ax2.set_xlim(0, 100)

plt.tight_layout()
plt.savefig(f'{sdir}/fullyear_price_comparison.png', dpi=150, bbox_inches='tight')
print(f'  Saved: fullyear_price_comparison.png')

# 5b. Dispatch comparison
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Top: Model dispatch stack
model_dispatch = model_gen.groupby(n.generators.carrier, axis=1).sum()
carriers_to_plot = ['nuclear', 'CCGT', 'CCGT_Flex', 'coal', 'OCGT', 'diesel', 'oil',
                    'solar', 'onwind', 'offwind', 'offwind-float', 'hydro', 'biomass', 'cogen']
carriers_to_plot = [c for c in carriers_to_plot if c in model_dispatch.columns]

# Weekly average
weekly = model_dispatch[carriers_to_plot].resample('W').mean()
colors = {'nuclear':'purple', 'CCGT':'steelblue', 'CCGT_Flex':'lightblue', 'coal':'black',
          'OCGT':'orange', 'diesel':'brown', 'oil':'darkred',
          'solar':'gold', 'onwind':'green', 'offwind':'teal', 'offwind-float':'cyan',
          'hydro':'blue', 'biomass':'darkgreen', 'cogen':'grey'}
weekly.plot.area(ax=axes[0], color=[colors.get(c, 'grey') for c in carriers_to_plot],
                 legend=True, alpha=0.8)
axes[0].set_ylabel('MW (weekly avg)')
axes[0].set_title('Model Dispatch by Carrier (Weekly Average) — Full Year 2024')
axes[0].legend(fontsize=8, loc='upper right')

# Bottom: Model vs Real hydro + CCGT daily
model_hydro_daily = model_hyd_daily
model_ccgt_daily = model_daily['CCGT'] if 'CCGT' in model_daily.columns else pd.Series(0, index=model_daily.index)

real_hydro_daily = real_year['Hidráulica']
real_ccgt_daily = real_year['Ciclo combinado']

axes[1].plot(model_hydro_daily.index, model_hydro_daily.values, 'b-', lw=0.8, label='Model Hydro')
axes[1].plot(real_hydro_daily.index, real_hydro_daily.values, 'b--', lw=0.5, alpha=0.5, label='Real Hydro')
axes[1].plot(model_ccgt_daily.index, model_ccgt_daily.values, 'orange', lw=0.8, label='Model CCGT')
axes[1].plot(real_ccgt_daily.index, real_ccgt_daily.values, 'orange', ls='--', lw=0.5, alpha=0.5, label='Real CCGT')
axes[1].set_ylabel('GWh/day')
axes[1].set_title('Daily Dispatch: Model vs Real (Spain 2024)')
axes[1].legend(fontsize=9)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.tight_layout()
plt.savefig(f'{sdir}/fullyear_dispatch_comparison.png', dpi=150, bbox_inches='tight')
print(f'  Saved: fullyear_dispatch_comparison.png')

# 5c. Monthly bar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
model_hyd_monthly = [model_hyd_daily[model_hyd_daily.index.month == m].sum() for m in range(1,13)]
real_hyd_monthly = [real_year.loc[real_year.index.month == m, 'Hidráulica'].sum() for m in range(1,13)]
model_ccgt_monthly = [model_daily.loc[model_daily.index.month == m, 'CCGT'].sum() if 'CCGT' in model_daily.columns else 0 for m in range(1,13)]
real_ccgt_monthly = [real_year.loc[real_year.index.month == m, 'Ciclo combinado'].sum() for m in range(1,13)]

x = np.arange(len(months))
w = 0.35

axes[0].bar(x - w/2, model_hyd_monthly, w, label='Model', color='steelblue')
axes[0].bar(x + w/2, real_hyd_monthly, w, label='Real', color='darkorange', alpha=0.7)
axes[0].set_xticks(x); axes[0].set_xticklabels(months, rotation=45)
axes[0].set_ylabel('GWh/month')
axes[0].set_title('Hydro Dispatch')
axes[0].legend()

axes[1].bar(x - w/2, model_ccgt_monthly, w, label='Model', color='steelblue')
axes[1].bar(x + w/2, real_ccgt_monthly, w, label='Real', color='darkorange', alpha=0.7)
axes[1].set_xticks(x); axes[1].set_xticklabels(months, rotation=45)
axes[1].set_ylabel('GWh/month')
axes[1].set_title('CCGT Dispatch')
axes[1].legend()

plt.tight_layout()
plt.savefig(f'{sdir}/fullyear_monthly_bars.png', dpi=150, bbox_inches='tight')
print(f'  Saved: fullyear_monthly_bars.png')

print('\nDone!')
