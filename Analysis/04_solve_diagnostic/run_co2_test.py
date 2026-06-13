#!/usr/bin/env python3
"""Test: add CO2 price of €65/t to see if price formation fixes."""
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
# Emission factors (tCO2/MWh_th):
#   CCGT: 0.202 tCO2/MWh_th, efficiency ~58% → 0.202/0.58 = 0.348 tCO2/MWh_el
#   Coal: 0.340 tCO2/MWh_th, efficiency ~40% → 0.340/0.40 = 0.850 tCO2/MWh_el
#   OCGT: 0.202 tCO2/MWh_th, efficiency ~38% → 0.202/0.38 = 0.532 tCO2/MWh_el
#   Diesel: 0.267 tCO2/MWh_th, efficiency ~35% → 0.267/0.35 = 0.763 tCO2/MWh_el
co2_price = 65.0  # EUR/tCO2
co2_add = {
    'CCGT': 0.348 * co2_price,      # ~22.6 EUR/MWh
    'CCGT_Flex': 0.532 * co2_price, # ~34.6 EUR/MWh (open-cycle = lower eff)
    'coal': 0.850 * co2_price,      # ~55.3 EUR/MWh
    'OCGT': 0.532 * co2_price,      # ~34.6 EUR/MWh
    'diesel': 0.763 * co2_price,    # ~49.6 EUR/MWh
    'oil': 0.763 * co2_price,       # ~49.6 EUR/MWh
}

print(f'\n=== CO2 PRICE: €{co2_price}/t ===')
for car, add in co2_add.items():
    m = n.generators['carrier'] == car
    if m.any():
        old_mc = n.generators.loc[m, 'marginal_cost'].mean()
        n.generators.loc[m, 'marginal_cost'] += add
        new_mc = n.generators.loc[m, 'marginal_cost'].mean()
        print(f'  {car:15s}: +€{add:.1f}/MWh  (old mean={old_mc:.1f} → new mean={new_mc:.1f})')

# ── 1. Hard-code non-linear MCs (base, then CO2 added above) ──────────
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
    m=(gen_cc==cc)&(n.generators['carrier']=='CCGT')
    idx=n.generators.loc[m].sort_values('p_nom',ascending=False).index
    if len(idx)==0: return
    for si,(lo,hi) in zip(np.array_split(idx,len(tiers)),tiers):
        n.generators.loc[si,'marginal_cost']=rng.uniform(lo,hi,size=len(si))
assign_ccgt_tiers('ES',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('PT',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('FR',[(68,78),(78,88)])

# Now add CO2 to fossil generators
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
print(f"  CCGT_Flex MC: {flex_lo:.0f}-{flex_hi:.0f} EUR/MWh (incl CO2)")

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

# ── Solve ─────────────────────────────────────────────────────────────
start=pd.Timestamp('2024-01-01'); end=pd.Timestamp('2024-01-31 23:00')
snaps=n.snapshots[(n.snapshots>=start)&(n.snapshots<=end)]
print(f'\nSolving {len(snaps)} snapshots ({len(snaps)/24:.0f} days)...')
n_sub=n.copy(); n_sub.set_snapshots(snaps)
res=n_sub.optimize(solver_name='gurobi',solver_options={'OutputFlag':1,'TimeLimit':7200,'Threads':5,'DualReductions':0})
print(f'Status: {res[0]} | Objective: {n_sub.objective:.2f} EUR')

# ── Save ──────────────────────────────────────────────────────────────
sdir=f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(sdir,exist_ok=True)
n_sub.export_to_netcdf(f'{sdir}/solved_2024-01_31d_co2.nc')
print('Saved.')

# ── Price analysis ────────────────────────────────────────────────────
p=n_sub.buses_t.marginal_price
es_b=[b for b in p.columns if str(b).startswith('ES')]
p_es=p[es_b]

print('\n=== ES PRICE STATS (with CO2) ===')
s=p_es.stack()
print(f'Mean: {s.mean():.1f}, Median: {s.median():.1f}, Max: {s.max():.1f}, Min: {s.min():.1f}')
print(f'p5: {s.quantile(0.05):.1f}, p25: {s.quantile(0.25):.1f}, p75: {s.quantile(0.75):.1f}, p95: {s.quantile(0.95):.1f}')

# Price distribution
mean_p=p_es.mean(axis=1)
print('\n=== PRICE DISTRIBUTION ===')
for lo, hi, label in [(0, 10, '0-10'), (10, 30, '10-30'), (30, 50, '30-50'),
                       (50, 75, '50-75'), (75, 100, '75-100'), (100, 150, '100-150'),
                       (150, 200, '150-200'), (200, 500, '200-500')]:
    count = ((mean_p >= lo) & (mean_p < hi)).sum()
    if count > 0:
        print(f'  {label:8s} EUR/MWh: {count:4d} hours ({count/len(mean_p)*100:5.1f}%)')

# Compare to OMIE
omie=pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv')
omie['datetime']=pd.to_datetime(omie['Datetime (UTC)'],format='%d/%m/%y %H:%M')
omie=omie.set_index('datetime')['Price (EUR/MWhe)']
omie_period=omie[(omie.index>=start)&(omie.index<=end)]

print(f'\n=== MODEL vs REAL (OMIE) — ES ===')
print(f'Model mean: {p_es.mean(axis=1).mean():.1f} EUR/MWh')
print(f'OMIE   mean: {omie_period.mean():.1f} EUR/MWh')
print(f'Model max:  {p_es.stack().max():.1f} EUR/MWh')
print(f'OMIE   max:  {omie_period.max():.1f} EUR/MWh')
print(f'Model >50:  {(p_es.mean(axis=1)>50).sum()}/{len(p_es)} hours')
print(f'OMIE   >50:  {(omie_period>50).sum()}/{len(omie_period)} hours')

# ── Plots ─────────────────────────────────────────────────────────────
model_ts=p_es.mean(axis=1)

fig,ax=plt.subplots(figsize=(16,5))
ax.plot(model_ts.index,model_ts.values,label='Model (CO2=65)',color='steelblue',lw=1)
ax.plot(omie_period.index,omie_period.values,label='OMIE 2024',color='darkorange',lw=0.7,alpha=0.7)
ax.set_ylabel('EUR/MWh'); ax.set_title('ES Price: Model (CO2=€65/t) vs Real OMIE — January 2024')
ax.legend(fontsize=9); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout(); plt.savefig(f'{sdir}/price_co2_timeseries.png',dpi=150,bbox_inches='tight')

fig,ax=plt.subplots(figsize=(10,5))
model_pdc=p_es.stack().sort_values(ascending=False).values
omie_pdc=omie_period.sort_values(ascending=False).values
ax.plot(np.arange(1,len(model_pdc)+1)/len(model_pdc)*100,model_pdc,label='Model (CO2=65)',color='steelblue')
ax.plot(np.arange(1,len(omie_pdc)+1)/len(omie_pdc)*100,omie_pdc,label='OMIE 2024',color='darkorange',alpha=0.7)
ax.set_xlabel('Exceedance %'); ax.set_ylabel('EUR/MWh')
ax.set_title('Price Duration Curve — ES (CO2=€65/t)')
ax.legend(); ax.set_xlim(0,100)
plt.tight_layout(); plt.savefig(f'{sdir}/pdc_co2.png',dpi=150,bbox_inches='tight')

print('\nPlots saved.')
