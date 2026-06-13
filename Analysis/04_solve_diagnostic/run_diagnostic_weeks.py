#!/usr/bin/env python3
"""
Diagnostic: Solve 4 weeks (Jan, Apr, Jul, Oct) with updated parameters.

Purpose:
  - Test price formation with current "schedule" (VOLL=3000, hydro SOC=50%, CO2 price)
  - Identify if/where prices hit VOLL (indicating insufficient capacity)
  - Compare shoulder-period prices to OMIE real data
  - Test with and without interconnectors

Usage:
  pixi run python3 Analysis/04_solve_diagnostic/run_diagnostic_weeks.py

Parameters (the "schedule"):
  - VOLL: 3,000 EUR/MWh (load shedding)
  - Hydro SOC initial: 50% of max energy (11.5 TWh)
  - CO2 price: 65 EUR/t
  - CCGT tiers: 52-62 / 62-72 / 72-82 EUR/MWh (base, before CO2)
  - CCGT_Flex: 20% split, MC=90-105 + CO2
  - Peakers: OCGT (1,149 MW, 125-145), Diesel (769 MW, 160-185)
  - Nuclear p_min_pu: 0.10 (guardrail)
  - Ramp limits: nuclear 0.20, CCGT 0.50-0.80, CCGT_Flex 1.0
  - Transmission: s_max_pu=0.7, ES-ES lines scaled by TRANS_FACTOR
  - No expansion, no interconnectors (border lockdown)
"""
import warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.dates as mdates, pypsa, os, sys
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10})
rng=np.random.default_rng(42)

ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n=pypsa.Network(f'{ROOT}/resources/networks/50n_ES_FR_PT.nc')
print(f'Loaded: {len(n.buses)} buses, {len(n.generators)} gens, {len(n.snapshots)} snapshots')

bus_cc=pd.Series(n.buses.index.astype(str).str[:2].values,index=n.buses.index)
gen_cc=n.generators.bus.map(bus_cc); su_cc=n.storage_units.bus.map(bus_cc)

# ══════════════════════════════════════════════════════════════════════
# PARAMETERS (the "schedule")
# ══════════════════════════════════════════════════════════════════════

# Transmission constraint
TRANS_FACTOR = 0.35  # Scale ES-ES lines to create congestion

# CO2 price
CO2_PRICE = 65.0  # EUR/t
co2_add = {
    'CCGT': 0.348 * CO2_PRICE,      # ~22.6
    'CCGT_Flex': 0.532 * CO2_PRICE, # ~34.6
    'coal': 0.850 * CO2_PRICE,      # ~55.3
    'OCGT': 0.532 * CO2_PRICE,      # ~34.6
    'diesel': 0.763 * CO2_PRICE,    # ~49.6
    'oil': 0.763 * CO2_PRICE,       # ~49.6
}

# VOLL
VOLL = 3_000  # EUR/MWh

# ══════════════════════════════════════════════════════════════════════
# 1. TRANSMISSION CONSTRAINT
# ══════════════════════════════════════════════════════════════════════
es_lines = n.lines[
    n.lines.bus0.astype(str).str[:2].isin(['ES']) &
    n.lines.bus1.astype(str).str[:2].isin(['ES'])
].index
print(f'\n=== TRANSMISSION CONSTRAINT ===')
print(f'Scaling {len(es_lines)} ES-ES lines by {TRANS_FACTOR}×')
print(f'Before: mean={n.lines.loc[es_lines,"s_nom"].mean():.0f} MW, total={n.lines.loc[es_lines,"s_nom"].sum()/1e3:.0f} GW')
n.lines.loc[es_lines, 's_nom'] *= TRANS_FACTOR
n.lines.loc[es_lines, 's_nom_extendable'] = False
print(f'After:  mean={n.lines.loc[es_lines,"s_nom"].mean():.0f} MW, total={n.lines.loc[es_lines,"s_nom"].sum()/1e3:.0f} GW')
n.lines['s_max_pu'] = 0.7  # 70% thermal limit

# ══════════════════════════════════════════════════════════════════════
# 2. CO2 PRICE
# ══════════════════════════════════════════════════════════════════════
print(f'\n=== CO2 PRICE: €{CO2_PRICE}/t ===')
for car, add in co2_add.items():
    m = n.generators['carrier'] == car
    if m.any():
        old_mc = n.generators.loc[m, 'marginal_cost'].mean()
        n.generators.loc[m, 'marginal_cost'] += add
        new_mc = n.generators.loc[m, 'marginal_cost'].mean()
        print(f'  {car:15s}: +€{add:.1f}/MWh  ({old_mc:.1f} → {new_mc:.1f})')

# ══════════════════════════════════════════════════════════════════════
# 3. HARD-CODE MARGINAL COSTS
# ══════════════════════════════════════════════════════════════════════
# VRE
for car in ['solar']: n.generators.loc[n.generators['carrier']==car,'marginal_cost']=0.10
for car in ['onwind','offwind','offwind-float']: n.generators.loc[n.generators['carrier']==car,'marginal_cost']=0.50
# Nuclear — guardrail
nuc=n.generators['carrier']=='nuclear'
n.generators.loc[nuc,'marginal_cost']=rng.uniform(12,18,size=nuc.sum())
n.generators.loc[nuc,'p_min_pu']=0.10
# Biomass/Cogen
n.generators.loc[n.generators['carrier']=='biomass','marginal_cost']=40.0
cog=n.generators['carrier']=='cogen'
n.generators.loc[cog,'marginal_cost']=45.0; n.generators.loc[cog,'p_min_pu']=0.70
# Hydro
es_hyd=(su_cc=='ES')&(n.storage_units['carrier']=='hydro')
n.storage_units.loc[es_hyd,'marginal_cost']=28.0
for cc,oc in [('FR',22.0),('PT',35.0)]:
    m=(gen_cc==cc)&(n.generators['carrier']=='hydro')
    n.generators.loc[m,'marginal_cost']=oc

# CCGT tiers (before CO2 — CO2 already added above)
def assign_ccgt_tiers(cc,tiers):
    idx=n.generators.loc[(gen_cc==cc)&(n.generators['carrier']=='CCGT')].sort_values('p_nom',ascending=False).index
    if len(idx)==0: return
    for si,(lo,hi) in zip(np.array_split(idx,len(tiers)),tiers):
        n.generators.loc[si,'marginal_cost']=rng.uniform(lo,hi,size=len(si))
assign_ccgt_tiers('ES',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('PT',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('FR',[(68,78),(78,88)])

# Add CO2 to fossil generators (re-apply after MC reset)
for car, add in co2_add.items():
    m = n.generators['carrier'] == car
    if m.any():
        n.generators.loc[m, 'marginal_cost'] += add

# Coal, OCGT, Oil
n.generators.loc[n.generators['carrier']=='coal','marginal_cost']=115.0 + co2_add['coal']
n.generators.loc[n.generators['carrier']=='coal','ramp_limit_up']=0.1
n.generators.loc[n.generators['carrier']=='coal','ramp_limit_down']=0.1
n.generators.loc[n.generators['carrier']=='OCGT','marginal_cost']=125.0 + co2_add['OCGT']
n.generators.loc[n.generators['carrier']=='oil','marginal_cost']=180.0 + co2_add['oil']

# ══════════════════════════════════════════════════════════════════════
# 4. CCGT FLEX SPLIT — 20% → CCGT_Flex carrier
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# 5. ADD PEAKERS (OCGT + Diesel)
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# 6. RAMP LIMITS
# ══════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════
# 7. HYDRO — SOC initial at 50%
# ══════════════════════════════════════════════════════════════════════
es_hyd_soc=(n.storage_units.bus.map(lambda b:str(b).startswith('ES'))&(n.storage_units['carrier']=='hydro'))
n.storage_units.loc[es_hyd_soc,'state_of_charge_initial']=0.50*n.storage_units.loc[es_hyd_soc,'p_nom']*n.storage_units.loc[es_hyd_soc,'max_hours']
print(f'\n=== HYDRO ===')
print(f'ES hydro SOC initial: {n.storage_units.loc[es_hyd_soc,"state_of_charge_initial"].sum()/1e6:.2f} TWh (50%)')
print(f'ES hydro total energy: {(n.storage_units.loc[es_hyd_soc,"p_nom"]*n.storage_units.loc[es_hyd_soc,"max_hours"]).sum()/1e6:.2f} TWh')

# ══════════════════════════════════════════════════════════════════════
# 8. DISABLE EXPANSION
# ══════════════════════════════════════════════════════════════════════
n.generators['p_nom_extendable']=False; n.links['p_nom_extendable']=False
n.storage_units['p_nom_extendable']=False; n.stores['e_nom_extendable']=False; n.lines['s_nom_extendable']=False

# ══════════════════════════════════════════════════════════════════════
# 9. REMOVE INTERCONNECTORS (border lockdown)
# ══════════════════════════════════════════════════════════════════════
ic=n.links[n.links['carrier'].str.contains('DC_ic',na=False)]
print(f'\nInterconnectors: {len(ic)} → setting p_nom=0')
n.links.loc[ic.index,'p_nom']=0

# ══════════════════════════════════════════════════════════════════════
# 10. ADD LOAD SHEDDING (VOLL)
# ══════════════════════════════════════════════════════════════════════
for b in n.buses.index:
    if str(b).startswith('ES') or str(b).startswith('FR') or str(b).startswith('PT'):
        n.add('Generator',f'{b} load_shedding',bus=b,carrier='load_shedding',
              p_nom=50_000,marginal_cost=VOLL,p_min_pu=0)
print(f'Added load_shedding at VOLL={VOLL} EUR/MWh')

# ══════════════════════════════════════════════════════════════════════
# 11. SOLVE 4 WEEKS (one per season)
# ══════════════════════════════════════════════════════════════════════
# Week 1: Jan 15-21 (winter peak)
# Week 2: Apr 15-21 (spring shoulder)
# Week 3: Jul 15-21 (summer peak)
# Week 4: Oct 15-21 (autumn shoulder)
weeks = [
    ('winter', '2024-01-15', '2024-01-21 23:00'),
    ('spring', '2024-04-15', '2024-04-21 23:00'),
    ('summer', '2024-07-15', '2024-07-21 23:00'),
    ('autumn', '2024-10-15', '2024-10-21 23:00'),
]

sdir=f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(sdir,exist_ok=True)

# Load OMIE data for comparison
omie=pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv')
omie['datetime']=pd.to_datetime(omie['Datetime (UTC)'],format='%d/%m/%y %H:%M')
omie=omie.set_index('datetime')['Price (EUR/MWhe)']

# Store results
results = []

for season_name, start_str, end_str in weeks:
    start=pd.Timestamp(start_str); end=pd.Timestamp(end_str)
    snaps=n.snapshots[(n.snapshots>=start)&(n.snapshots<=end)]
    print(f'\n{"="*60}')
    print(f'SOLVING: {season_name} ({len(snaps)} snapshots, {start.date()} to {end.date()})')
    print(f'{"="*60}')
    
    n_sub=n.copy(); n_sub.set_snapshots(snaps)
    res=n_sub.optimize(solver_name='gurobi',
                       solver_options={'OutputFlag':1,'TimeLimit':1800,'Threads':5,'DualReductions':0})
    status = res[0]
    obj = n_sub.objective
    print(f'Status: {status} | Objective: {obj:.2e} EUR')
    
    # Save
    n_sub.export_to_netcdf(f'{sdir}/solved_{season_name}_week.nc')
    
    # ── PRICE ANALYSIS ──
    p=n_sub.buses_t.marginal_price
    es_b=[b for b in p.columns if str(b).startswith('ES')]
    es_ac=[b for b in es_b if not any(suffix in str(b) for suffix in [' H2',' battery'])]
    p_ac=p[es_ac]
    
    mean_p = p_ac.mean(axis=1)
    model_mean = mean_p.mean()
    model_max = p_ac.stack().max()
    model_min = p_ac.stack().min()
    
    # VOLL check
    voll_hours = (p_ac >= VOLL).sum().sum()
    total_hours = p_ac.size
    
    # Load shedding check
    ls = n_sub.generators[n_sub.generators['carrier']=='load_shedding']
    ls_used = 0
    if len(ls) > 0:
        ls_dispatch = n_sub.generators_t.p[ls.index]
        ls_used = ls_dispatch.sum().sum() / 1000  # MWh
    
    # OMIE comparison
    omie_period = omie[(omie.index>=start)&(omie.index<=end)]
    omie_mean = omie_period.mean() if len(omie_period) > 0 else float('nan')
    omie_max = omie_period.max() if len(omie_period) > 0 else float('nan')
    
    # Price distribution
    dist = {}
    for lo, hi, label in [(0, 10, '0-10'), (10, 30, '10-30'), (30, 50, '30-50'),
                           (50, 75, '50-75'), (75, 100, '75-100'), (100, 150, '100-150'),
                           (150, 200, '150-200'), (200, 500, '200-500'), (500, VOLL, '500-VOLL')]:
        count = ((mean_p >= lo) & (mean_p < hi)).sum()
        if count > 0:
            dist[label] = count
    
    # Dispatch
    g=n_sub.generators_t.p
    es_gens = [g for g in n_sub.generators.index if str(n_sub.generators.at[g,'bus']).startswith('ES')]
    g_es = g[es_gens]
    dispatch = g_es.sum().groupby(n_sub.generators.loc[es_gens,'carrier']).sum() / 1e3  # GWh
    
    # Hydro
    es_hyd_cols = [c for c in n_sub.storage_units_t.state_of_charge.columns 
                   if c in n_sub.storage_units.index and 
                   str(n_sub.storage_units.at[c,'bus']).startswith('ES') and
                   n_sub.storage_units.at[c,'carrier']=='hydro']
    soc = n_sub.storage_units_t.state_of_charge[es_hyd_cols]
    hyd_dispatch = n_sub.storage_units_t.p[es_hyd_cols]
    hyd_gen = hyd_dispatch[hyd_dispatch>0].sum().sum() / 1e3  # GWh
    
    # Curtailment
    curt = {}
    for tech in ['solar', 'onwind']:
        avail = n_sub.generators_t.p_max_pu.mul(n_sub.generators.p_nom, axis=1)
        tech_gens = [g for g in es_gens if n_sub.generators.at[g,'carrier']==tech]
        if len(tech_gens) > 0:
            avail_sum = avail[tech_gens].sum(axis=1)
            gen_sum = g_es[tech_gens].sum(axis=1)
            curt_pct = ((avail_sum - gen_sum).clip(0).sum() / avail_sum.sum() * 100) if avail_sum.sum() > 0 else 0
            curt[tech] = curt_pct
    
    result = {
        'season': season_name,
        'status': status,
        'model_mean': model_mean,
        'model_max': model_max,
        'model_min': model_min,
        'omie_mean': omie_mean,
        'omie_max': omie_max,
        'voll_hours': voll_hours,
        'total_hours': total_hours,
        'ls_used_mwh': ls_used,
        'dispatch': dispatch,
        'hydro_gen_gwh': hyd_gen,
        'curtailment': curt,
        'price_dist': dist,
    }
    results.append(result)
    
    # Print summary
    print(f'\n  Price: Mean={model_mean:.1f}, Max={model_max:.1f}, Min={model_min:.1f} EUR/MWh')
    print(f'  OMIE:  Mean={omie_mean:.1f}, Max={omie_max:.1f} EUR/MWh')
    print(f'  VOLL hits: {voll_hours}/{total_hours} ({(voll_hours/total_hours*100 if total_hours else 0):.1f}%)')
    print(f'  Load shedding: {ls_used:.1f} MWh')
    print(f'  Hydro gen: {hyd_gen:.1f} GWh')
    for tech, pct in curt.items():
        print(f'  {tech} curtailment: {pct:.1f}%')
    print(f'  Dispatch (GWh):')
    for car, val in dispatch.sort_values(ascending=False).items():
        print(f'    {car:20s}: {val:.1f}')

# ══════════════════════════════════════════════════════════════════════
# 12. SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════
print(f'\n{"="*80}')
print('DIAGNOSTIC SUMMARY')
print(f'{"="*80}')
print(f'{"Season":10s} {"Status":10s} {"Model Mean":10s} {"OMIE Mean":10s} {"Ratio":8s} {"Max":8s} {"VOLL hrs":10s} {"LS MWh":10s}')
print(f'{"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*8} {"-"*8} {"-"*10} {"-"*10}')
for r in results:
    ratio = r['model_mean'] / r['omie_mean'] if r['omie_mean'] and r['omie_mean'] > 0 else float('inf')
    voll_str = f"{r['voll_hours']}/{r['total_hours']}"
    print(f'{r["season"]:10s} {r["status"]:10s} {r["model_mean"]:8.1f}   {r["omie_mean"]:8.1f}   {ratio:6.2f}x {r["model_max"]:8.1f} {voll_str:10s} {r["ls_used_mwh"]:8.1f}')

print(f'\nSaved to: {sdir}/solved_*_week.nc')
print('Done.')
