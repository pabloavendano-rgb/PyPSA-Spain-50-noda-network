#!/usr/bin/env python3
"""Dispatch-only solve: hard-coded MCs + ramping + PEAKER STACK AMENDMENT, no expansion, no interconnectors, compare to real-world."""
import warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib.dates as mdates, pypsa, os
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10})
rng=np.random.default_rng(42)

import sys
ROOT=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n=pypsa.Network(f'{ROOT}/resources/networks/50n_ES_FR_PT.nc')
print(f'Loaded: {len(n.buses)} buses, {len(n.generators)} gens, {len(n.snapshots)} snapshots')

bus_cc=pd.Series(n.buses.index.astype(str).str[:2].values,index=n.buses.index)
gen_cc=n.generators.bus.map(bus_cc); su_cc=n.storage_units.bus.map(bus_cc)

# ═══════════════════════════════════════════════════════════════════
#  PEAKER STACK AMENDMENT — 9.3 GW peaker proxy fleet
# ═══════════════════════════════════════════════════════════════════
# Tier 1: Coal (2,061 MW, €115/MWh, ramp=0.1) — Asturias/Almería
# Tier 2: CCGT_Flex (5,250 MW, €90-105/MWh) — 20% of CCGT fleet, open-cycle mode
# Tier 3: OCGT (1,149 MW, €125-145/MWh) — urban fast-ramping
# Tier 4: Diesel (769 MW, €160-185/MWh) — emergency port capacity
# Distribution: 70% load-sink weighted, 30% fuel-infrastructure weighted
# Nuclear guardrail: p_min_pu = 0.1 (not 0.50)

# ── 1. Hard-code non-linear MCs ────────────────────────────────────
# VRE
for car in ['solar']: n.generators.loc[n.generators['carrier']==car,'marginal_cost']=0.10
for car in ['onwind','offwind','offwind-float']: n.generators.loc[n.generators['carrier']==car,'marginal_cost']=0.50
# Nuclear — guardrail: p_min_pu=0.1 prevents full shutdown during solar peaks
nuc=n.generators['carrier']=='nuclear'
n.generators.loc[nuc,'marginal_cost']=rng.uniform(12,18,size=nuc.sum())
n.generators.loc[nuc,'p_min_pu']=0.10  # Nuclear guardrail: min 10% output
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
# CCGT tiers (for remaining 80% CCGT fleet)
def assign_ccgt_tiers(cc,tiers):
    m=(gen_cc==cc)&(n.generators['carrier']=='CCGT')
    idx=n.generators.loc[m].sort_values('p_nom',ascending=False).index
    if len(idx)==0: return
    for si,(lo,hi) in zip(np.array_split(idx,len(tiers)),tiers):
        n.generators.loc[si,'marginal_cost']=rng.uniform(lo,hi,size=len(si))
assign_ccgt_tiers('ES',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('PT',[(52,62),(62,72),(72,82)])
assign_ccgt_tiers('FR',[(68,78),(78,88)])
# Coal — re-label: MC=115, ramp=0.1
n.generators.loc[n.generators['carrier']=='coal','marginal_cost']=115.0
n.generators.loc[n.generators['carrier']=='coal','ramp_limit_up']=0.1
n.generators.loc[n.generators['carrier']=='coal','ramp_limit_down']=0.1
# Existing OCGT/Oil (FR/PT only)
n.generators.loc[n.generators['carrier']=='OCGT','marginal_cost']=125.0
n.generators.loc[n.generators['carrier']=='oil','marginal_cost']=180.0

# ── 1b. CCGT FLEX SPLIT — 20% → CCGT_Flex carrier ─────────────────
ccgt_es=(gen_cc=='ES')&(n.generators['carrier']=='CCGT')
# Capture bus counts BEFORE flex split (index will change after adding new generators)
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
          p_nom=flex_pnom,marginal_cost=rng.uniform(90,105),
          ramp_limit_up=1.0,ramp_limit_down=1.0,p_min_pu=0)
flex_total=n.generators[n.generators['carrier']=='CCGT_Flex']['p_nom'].sum()
print(f'\n=== CCGT FLEX SPLIT ===')
print(f'  Original CCGT: 26,251 MW → 80% CCGT ({26251*0.8:.0f} MW) + 20% CCGT_Flex ({flex_total:.0f} MW)')
print(f'  CCGT_Flex MC: 90-105 EUR/MWh, ramp=1.0 (open-cycle mode)')

# ── 1c. ADD PEAKERS (OCGT + Diesel) — stress-weighted distribution ─
# Load-sink weights (70%): top load nodes
es_loads=n.loads[n.loads.bus.astype(str).str.startswith('ES')].copy()
es_loads['annual']=es_loads.index.map(lambda x: n.loads_t.p_set[x].sum())
load_total=es_loads['annual'].sum()
es_loads['load_share']=es_loads['annual']/load_total

# Fuel-infrastructure weights (30%): nodes with existing CCGT (captured pre-flex-split)
ccgt_bus_share=ccgt_bus_counts_pre/ccgt_bus_counts_pre.sum()

# Combined weight W = 0.7 * load_share + 0.3 * ccgt_bus_share
W=pd.Series(0.0,index=es_loads.bus)
for b in W.index:
    W[b]=0.7*es_loads.loc[es_loads.bus==b,'load_share'].values[0]+0.3*ccgt_bus_share.get(b,0)
W=W/W.sum()  # renormalise

# Add OCGT (1,149 MW, €125-145) + Diesel (769 MW, €160-185)
peaker_config=[('OCGT',1149,125,145),('diesel',769,160,185)]
for pcar,pnom,plo,phi in peaker_config:
    added=0
    for b in W.sort_values(ascending=False).index:
        if added>=pnom: break
        alloc=round(pnom*W[b])
        if alloc<1: continue
        n.add('Generator',f'{b} {pcar}',bus=b,carrier=pcar,
              p_nom=alloc,marginal_cost=rng.uniform(plo,phi),
              ramp_limit_up=1.0,ramp_limit_down=1.0,p_min_pu=0)
        added+=alloc
    actual=n.generators[n.generators['carrier']==pcar]['p_nom'].sum()
    print(f'  {pcar}: target={pnom} MW, actual={actual:.0f} MW, MC=€{plo}-{phi}/MWh')

# ── 2. Hard-code ramping ───────────────────────────────────────────
def ramp(car,up,dn):
    m=n.generators['carrier']==car
    n.generators.loc[m,'ramp_limit_up']=up; n.generators.loc[m,'ramp_limit_down']=dn
ramp('nuclear',0.20,0.20); ramp('biomass',0.30,0.30); ramp('cogen',0.30,0.30)
ramp('OCGT',1.00,1.00); ramp('oil',0.90,0.90); ramp('CCGT_Flex',1.00,1.00); ramp('diesel',1.00,1.00)
# Coal already set to 0.1 above
def ramp_ccgt(cc,tiers):
    m=(gen_cc==cc)&(n.generators['carrier']=='CCGT')
    idx=n.generators.loc[m].sort_values('p_nom',ascending=False).index
    if len(idx)==0: return
    for si,(ru,rd) in zip(np.array_split(idx,len(tiers)),tiers):
        n.generators.loc[si,'ramp_limit_up']=ru; n.generators.loc[si,'ramp_limit_down']=rd
ramp_ccgt('ES',[(0.80,0.80),(0.65,0.65),(0.50,0.50)])
ramp_ccgt('PT',[(0.80,0.80),(0.65,0.65),(0.50,0.50)])
ramp_ccgt('FR',[(0.65,0.65),(0.50,0.50)])
# Hydro SOC 50%
es_hyd_soc=(n.storage_units.bus.map(lambda b:str(b).startswith('ES'))&(n.storage_units['carrier']=='hydro'))
n.storage_units.loc[es_hyd_soc,'state_of_charge_initial']=0.50*n.storage_units.loc[es_hyd_soc,'p_nom']*n.storage_units.loc[es_hyd_soc,'max_hours']

# ── 3. Verify hydro MC ─────────────────────────────────────────────
print('\n=== HYDRO MC ===')
hs=n.storage_units[n.storage_units['carrier']=='hydro']
print(f'ES hydro storage: MC={hs.marginal_cost.unique()} (expected 28.0)')
for cc,exp in [('FR',22.0),('PT',35.0)]:
    hg=n.generators[(gen_cc==cc)&(n.generators['carrier']=='hydro')]
    print(f'{cc} hydro: MC={hg.marginal_cost.unique()} (expected {exp})')

# ── 4. Disable expansion ───────────────────────────────────────────
n.generators['p_nom_extendable']=False; n.links['p_nom_extendable']=False
n.storage_units['p_nom_extendable']=False; n.stores['e_nom_extendable']=False; n.lines['s_nom_extendable']=False

# ── 5. Remove interconnectors ──────────────────────────────────────
ic=n.links[n.links['carrier'].str.contains('DC_ic',na=False)]
print(f'\nInterconnectors: {len(ic)} → setting p_nom=0')
n.links.loc[ic.index,'p_nom']=0

# ── 5b. Add load shedding (VOLL=3000) for feasibility ──────────────
for b in n.buses.index:
    if str(b).startswith('ES') or str(b).startswith('FR') or str(b).startswith('PT'):
        n.add('Generator',f'{b} load_shedding',bus=b,carrier='load_shedding',
              p_nom=50_000,marginal_cost=3_000,p_min_pu=0)
print(f'Added load_shedding generators at VOLL=3000 on {sum(1 for b in n.buses.index if str(b)[:2] in ("ES","FR","PT"))} buses')

# ── 6. Solve dispatch-only (January 2024 — winter peak) ────────────
start=pd.Timestamp('2024-01-01'); end=pd.Timestamp('2024-01-31 23:00')
snaps=n.snapshots[(n.snapshots>=start)&(n.snapshots<=end)]
print(f'\nSolving {len(snaps)} snapshots ({len(snaps)/24:.0f} days)...')
n_sub=n.copy(); n_sub.set_snapshots(snaps)
res=n_sub.optimize(solver_name='gurobi',solver_options={'OutputFlag':1,'TimeLimit':7200,'Threads':5,'DualReductions':0})
print(f'Status: {res[0]} | Objective: {n_sub.objective:.2f} EUR')

# ── 7. Save ─────────────────────────────────────────────────────────
sdir=f'{ROOT}/solved_networks/04_solve_diagnostic'
os.makedirs(sdir,exist_ok=True)
n_sub.export_to_netcdf(f'{sdir}/solved_2024-07-01_31d_no_ic.nc')
print('Saved.')

# ── 8. Dispatch analysis ───────────────────────────────────────────
gen_p=n_sub.generators_t.p; gen_info=n_sub.generators[['bus','carrier']]
cc_map={'ES':'Spain','FR':'France','PT':'Portugal'}
gen_by_cc={}
for cc,name in cc_map.items():
    cb=[b for b in n_sub.buses.index if str(b).startswith(cc)]
    cg=gen_info.index[gen_info['bus'].isin(cb)]
    cd={c:gen_p[c_g].sum(axis=1) for c in gen_info.loc[cg,'carrier'].unique()
        if len(c_g:=cg[gen_info.loc[cg,'carrier']==c])>0}
    gen_by_cc[cc]=pd.DataFrame(cd)
    print(f'{name}: {len(cg)} gens')

# Total generation
all_cars={c:gen_p[gen_info.index[gen_info['carrier']==c]].sum(axis=1) for c in gen_info['carrier'].unique()}
gt=pd.DataFrame(all_cars)
print('\n--- Generation (GWh) ---')
for c,v in gt.sum().sort_values(ascending=False).items(): print(f'  {c:20s}: {v/1000:8.2f}')
print(f'  {"TOTAL":20s}: {gt.sum().sum()/1000:8.2f}')

# ── 9. Dispatch plot ───────────────────────────────────────────────
ccol={'solar':'#f9d002','onwind':'#235ebc','offwind':'#074ede','offwind-float':'#b5e2fa',
      'hydro':'#298c81','ror':'#3dbfb0','PHS':'#51dbcc','nuclear':'#ff8c00','CCGT':'#a85522',
      'CCGT_Flex':'#d49a6a','OCGT':'#e88d4a','coal':'#545454','biomass':'#baa741',
      'battery':'#e2ff7c','oil':'#c9c9c9','diesel':'#8b0000','load_shedding':'#ffcccc'}
fig,axes=plt.subplots(3,1,figsize=(14,10),sharex=True)
for idx,(cc,cn) in enumerate(cc_map.items()):
    ax=axes[idx]; df=gen_by_cc[cc]; order=df.sum().sort_values().index
    ax.stackplot(df.index,df[order].T,labels=order,colors=[ccol.get(c,'#ccc') for c in order],alpha=0.85)
    ax.set_ylabel('MW'); ax.set_title(f'{cn} — Dispatch Jul 2024 — NO INTERCONNECTORS')
    ax.legend(loc='upper left',ncol=3,fontsize=8); ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout(); plt.savefig(f'{sdir}/dispatch_no_ic.png',dpi=150,bbox_inches='tight'); plt.show()

# ── 10. Price diagnostics ──────────────────────────────────────────
p=n_sub.buses_t.marginal_price.copy()
es_b=[b for b in p.columns if str(b).startswith('ES')]
fr_b=[b for b in p.columns if str(b).startswith('FR')]
pt_b=[b for b in p.columns if str(b).startswith('PT')]
p_es=p[es_b]; p_fr=p[fr_b]; p_pt=p[pt_b]

print('\n=== PRICE STATS (EUR/MWh) ===')
for label,df in [('ES',p_es),('FR',p_fr),('PT',p_pt)]:
    s=df.stack()
    print(f'{label}: mean={s.mean():.1f} std={s.std():.1f} p5={s.quantile(0.05):.1f} '
          f'p25={s.quantile(0.25):.1f} p50={s.median():.1f} p75={s.quantile(0.75):.1f} p95={s.quantile(0.95):.1f}')

# Load-weighted ES price
es_load=n_sub.loads_t.p_set[[c for c in n_sub.loads_t.p_set.columns if str(c).startswith('ES')]].sum(axis=1)
lw=(p_es.mean(axis=1)*es_load).sum()/es_load.sum()
print(f'ES load-weighted mean: {lw:.2f} | simple mean: {p_es.mean(axis=1).mean():.2f}')

# Price histogram
fig,axes=plt.subplots(1,2,figsize=(14,4))
for ax,df,label,color in [(axes[0],p_es,'ES','steelblue'),(axes[1],p_fr,'FR','darkred')]:
    ax.hist(df.stack(),bins=80,color=color,edgecolor='white',alpha=0.8)
    ax.set_xlabel('EUR/MWh'); ax.set_ylabel('Frequency')
    ax.set_title(f'{label} Price Distribution — NO INTERCONNECTORS')
    ax.axvline(df.stack().mean(),color='red',ls='--',lw=1,label=f'Mean={df.stack().mean():.1f}')
    ax.axvline(df.stack().median(),color='orange',ls='--',lw=1,label=f'Median={df.stack().median():.1f}')
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{sdir}/price_hist_no_ic.png',dpi=150,bbox_inches='tight'); plt.show()

# ── 11. Compare to real-world OMIE ─────────────────────────────────
omie=pd.read_csv(f'{ROOT}/Analysis/data/Spain_prices.csv')
# European date format DD/MM/YY
omie['datetime']=pd.to_datetime(omie['Datetime (UTC)'],format='%d/%m/%y %H:%M')
omie=omie.set_index('datetime')['Price (EUR/MWhe)']
omie_period=omie[(omie.index>=start)&(omie.index<=end)]
print(f'\n=== MODEL vs REAL (OMIE) — ES ===')
print(f'Model ES mean: {p_es.mean(axis=1).mean():.1f} EUR/MWh')
print(f'OMIE   mean:   {omie_period.mean():.1f} EUR/MWh')
print(f'Model ES std:  {p_es.stack().std():.1f} EUR/MWh')
print(f'OMIE   std:    {omie_period.std():.1f} EUR/MWh')

# Time series comparison
fig,ax=plt.subplots(figsize=(14,5))
model_ts=p_es.mean(axis=1)
ax.plot(model_ts.index,model_ts,label=f'Model ES mean (no IC)',color='steelblue',lw=1)
ax.plot(omie_period.index,omie_period,label='OMIE 2024 (real)',color='darkorange',lw=0.8,alpha=0.7)
ax.set_ylabel('EUR/MWh'); ax.set_title('ES Price: Model (no interconnectors) vs Real OMIE — July 2024')
ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
plt.tight_layout(); plt.savefig(f'{sdir}/price_vs_real_no_ic.png',dpi=150,bbox_inches='tight'); plt.show()

# Price duration curve comparison
fig,ax=plt.subplots(figsize=(10,5))
model_pdc=p_es.stack().sort_values(ascending=False).values
omie_pdc=omie_period.sort_values(ascending=False).values
x=np.arange(1,len(model_pdc)+1)/len(model_pdc)*100
ax.plot(x,model_pdc,label=f'Model (no IC)',color='steelblue')
ax.plot(np.arange(1,len(omie_pdc)+1)/len(omie_pdc)*100,omie_pdc,label='OMIE 2024',color='darkorange',alpha=0.7)
ax.set_xlabel('Exceedance %'); ax.set_ylabel('EUR/MWh'); ax.set_title('Price Duration Curve — ES')
ax.legend(); ax.set_xlim(0,100)
plt.tight_layout(); plt.savefig(f'{sdir}/pdc_comparison_no_ic.png',dpi=150,bbox_inches='tight'); plt.show()

print('\nDone. All plots saved to', sdir)
