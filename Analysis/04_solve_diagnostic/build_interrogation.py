#!/usr/bin/env python3
"""Build network_interrogation.ipynb — network construction & visualisation only."""
import json, os

cells = []
def md(s): cells.append({"cell_type":"markdown","id":f"md_{len(cells)}","metadata":{},"source":[s]})
def code(s, o=None):
    cells.append({"cell_type":"code","execution_count":None,"id":f"code_{len(cells)}","metadata":{},"outputs":o or [],
                  "source":[s]})

md("""# Network Interrogation — 50n_ES_FR_PT.nc

**Purpose:** Visually interrogate the network construction — sub-networks, zero-price buses, hydro/PHS setup, must-run capacity, build quality.

**Workflow:** `Kernel → Restart & Run All`.""")

md("---\n## Section 0 — Imports & Setup")
code("""import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi':130,'font.size':10})
n = pypsa.Network('../../resources/networks/50n_ES_FR_PT.nc')
print(f'{len(n.buses)} buses, {len(n.generators)} gens, {len(n.snapshots)} snapshots')
print(f'{len(n.lines)} lines, {len(n.links)} links, {len(n.stores)} stores, {len(n.storage_units)} storage')""")

md("---\n## Section 1 — Sub-Network Architecture")
code("""print('='*80,'\\nSUB-NETWORK ARCHITECTURE\\n'+'='*80)
for sid in n.sub_networks.index:
    sub=n.sub_networks.loc[sid]; b=n.buses[n.buses['sub_network']==sid]
    g=n.generators[n.generators['bus'].isin(b.index)]; l=n.loads[n.loads['bus'].isin(b.index)]
    s=n.storage_units[n.storage_units['bus'].isin(b.index)]; st=n.stores[n.stores['bus'].isin(b.index)]
    print(f'\\nSub-network: "{sid}" | carrier={sub.carrier} | slack={sub.slack_bus}')
    print(f'  Buses:{len(b):3d} Gens:{len(g):3d} Loads:{len(l):3d} Storage:{len(s):2d} Stores:{len(st):2d}')
    if len(g): [print(f'    {c:15s}: {cnt:2d} units, {g[g.carrier==c].p_nom.sum():8.0f} MW') for c,cnt in g.carrier.value_counts().items()]""")

md("### 1a. Geographic Map — Buses Coloured by Sub-Network")
code("""fig,ax=plt.subplots(figsize=(12,10))
colors={0:'#235ebc',1:'#a85522','':'#cccccc'}
labels={0:'Sub 0 — Mainland ES (49 buses)',1:'Sub 1 — Balearic Is. (1 bus)','':'Sub \"\" — H2/Bat/FR/PT (105)'}
for sid,grp in n.buses.groupby('sub_network'):
    ax.scatter(grp.x,grp.y,c=colors.get(sid,'#ff0000'),s=30,label=labels.get(sid,f'Sub {sid}'),zorder=3,edgecolors='k',linewidths=0.3)
    for i,r in grp.iterrows():
        if sid==1 or (sid==0 and 'ror' in str(n.generators[n.generators['bus']==i].carrier.values)):
            ax.annotate(i.replace('ES0 ','').replace('ES1 ',''),(r.x,r.y),fontsize=5,alpha=0.7)
# Plot lines
for i,r in n.lines.iterrows():
    b0,b1=n.buses.loc[r.bus0],n.buses.loc[r.bus1]
    ax.plot([b0.x,b1.x],[b0.y,b1.y],'#888888',lw=0.3,alpha=0.5)
# Plot DC link
for i,r in n.links[n.links.carrier=='DC'].iterrows():
    b0,b1=n.buses.loc[r.bus0],n.buses.loc[r.bus1]
    ax.plot([b0.x,b1.x],[b0.y,b1.y],'#ff4444',lw=1.5,ls='--',label='DC link' if i==n.links[n.links.carrier=='DC'].index[0] else '')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.set_title('50n_ES_FR_PT — Buses by Sub-Network')
ax.legend(fontsize=8,loc='upper left'); plt.tight_layout(); plt.show()
print('ES1 0 = Balearic Islands (Mallorca). Connected to ES0 5 (Valencia) via 400 MW DC submarine cable.')
print('Empty sub-network (105 buses): 50 H2 + 50 battery + 2 FR + 3 PT — no carrier, no slack → marginal_price=0')""")

md("### 1b. Visualise Sub-Network Composition")
code("""rows=[]
for sid in n.sub_networks.index:
    b=n.buses[n.buses['sub_network']==sid]; g=n.generators[n.generators['bus'].isin(b.index)]
    l=n.loads[n.loads['bus'].isin(b.index)]
    rows.append({'sub':sid if sid else '(empty)','carrier':n.sub_networks.loc[sid,'carrier'] or '(none)',
                 'buses':len(b),'gens':len(g),'loads':len(l),'gen_MW':g.p_nom.sum(),'load_MW':l.p_set.sum() if 'p_set' in l else 0})
df=pd.DataFrame(rows); print(df.to_string(index=False))
fig,ax=plt.subplots(1,3,figsize=(15,4))
for i,(y,t) in enumerate([('buses','Buses'),('gen_MW','Gen (GW)'),('load_MW','Load (GW)')]):
    ax[i].bar(df['sub'],df[y]/[1,1000,1000][i],color=['#235ebc','#a85522','#cccccc']); ax[i].set_title(t); ax[i].tick_params(axis='x',rotation=45)
plt.tight_layout(); plt.show()
print('KEY: Empty-name sub-network has no carrier/slack → H2/battery/FR/PT isolated → marginal_price=0')""")

md("---\n## Section 2 — Zero-Price Bus Analysis")
code("""es_ac=[b for b in n.buses.index if str(b).startswith('ES') and 'H2' not in str(b) and 'battery' not in str(b)]
h2=[b for b in n.buses.index if 'H2' in str(b)]; bat=[b for b in n.buses.index if 'battery' in str(b)]
fr=[b for b in n.buses.index if str(b).startswith('FR')]; pt=[b for b in n.buses.index if str(b).startswith('PT')]
print(f'ES AC:{len(es_ac)} H2:{len(h2)} Bat:{len(bat)} FR:{len(fr)} PT:{len(pt)}')
for lbl,bl in [('ES AC',es_ac),('H2',h2),('Battery',bat),('FR',fr),('PT',pt)]:
    if len(bl): print(f'  {lbl:10s}: sub_networks={list(n.buses.loc[bl,"sub_network"].unique())}')
for lbl,bl in [('H2',h2),('Battery',bat)]:
    ll=n.loads[n.loads['bus'].isin(bl)]
    print(f'  Loads on {lbl}: {len(ll)} (p_set={ll.p_set.sum():.0f} MW)' if len(ll) else f'  Loads on {lbl}: NONE')""")

md("### 2a. Components on H2 & Battery Buses (Sample)")
code("""for lbl,bl in [('H2',h2),('Battery',bat)]:
    if not len(bl): continue
    b0=bl[0]; print(f'\\n{lbl} sample: {b0}')
    g=n.generators[n.generators['bus']==b0]; lk=n.links[(n.links.bus0==b0)|(n.links.bus1==b0)]
    st=n.stores[n.stores['bus']==b0]; su=n.storage_units[n.storage_units['bus']==b0]
    for _,r in g.iterrows(): print(f'  Gen: {r.carrier} p_nom={r.p_nom:.0f} MC={r.marginal_cost:.2f}')
    for _,r in lk.iterrows(): print(f'  Link: {r.bus0}→{r.bus1} carrier={r.carrier} p_nom={r.p_nom:.0f}')
    for _,r in st.iterrows(): print(f'  Store: carrier={r.carrier} e_nom={r.e_nom:.0f} MC={r.marginal_cost:.2f}')""")

md("### 2b. Generators with MC=0")
code("""zm=n.generators[n.generators['marginal_cost']==0]
print(f'Generators with MC=0: {len(zm)}')
for c in zm.carrier.unique():
    s=zm[zm.carrier==c]; print(f'  {c:15s}: {len(s):2d} units, {s.p_nom.sum():8.0f} MW')
    for i,r in s.iterrows(): print(f'      {i:30s} bus={r.bus:15s} p_nom={r.p_nom:6.0f}')
print('KEY: ror (must-take) and PHS (pumping load) have MC=0 → real zero-price sources in AC sub-network')""")

md("---\n## Section 3 — Hydro Modeling")
code("""hg=n.generators[n.generators['carrier']=='hydro']
print(f'Hydro GENERATORS: {len(hg)} units, {hg.p_nom.sum()/1000:.1f} GW')
for i,r in hg.iterrows():
    c='FR' if 'FR' in str(r.bus) else 'PT'; exp={'FR':22,'PT':35}.get(c,0)
    print(f'  {i:30s} bus={r.bus:15s} MC={r.marginal_cost:8.2f} (expected OC={exp}) {"OK" if abs(r.marginal_cost-exp)<1 else "MISMATCH"}')
hs=n.storage_units[n.storage_units['carrier']=='hydro']
print(f'\\nHydro STORAGE: {len(hs)} units, {hs.p_nom.sum()/1000:.1f} GW, {(hs.p_nom*hs.max_hours).sum()/1000:.1f} GWh')
for i,r in hs.iterrows():
    soc_pu=r.state_of_charge_initial/(r.p_nom*r.max_hours) if r.p_nom*r.max_hours>0 else 0
    print(f'  {i:30s} bus={r.bus:15s} MC={r.marginal_cost:8.2f} soc_init={soc_pu:.2f} pu')
if (hs.state_of_charge_initial==0).all(): print('\\nWARNING: All ES hydro reservoirs start EMPTY! Cannot dispatch until charged.')""")

md("### 3a. Hydro Inflow")
code("""if hasattr(n,'storage_units_t') and 'inflow' in n.storage_units_t:
    inf=n.storage_units_t['inflow']
    hc=[c for c in inf.columns if c in hs.index]
    if len(hc):
        hi=inf[hc]; print(f'Inflow: {len(hc)} units, total {hi.sum().sum()/1000:.1f} GWh')
        fig,ax=plt.subplots(figsize=(12,3)); ax.plot(hi.index,hi.sum(axis=1),'#298c81',lw=0.5)
        ax.set_ylabel('MW'); ax.set_title('Total Hydro Inflow'); plt.tight_layout(); plt.show()
else: print('No inflow data in network')""")

md("---\n## Section 4 — PHS (Pumped Storage)")
code("""phs=n.storage_units[n.storage_units['carrier']=='PHS']
print(f'PHS units: {len(phs)}')
if len(phs):
    for i,r in phs.iterrows():
        print(f'  {i:30s} bus={r.bus:15s} p_nom={r.p_nom:6.0f} max_h={r.max_hours:5.1f} MC={r.marginal_cost:.2f} eff={r.efficiency_store:.3f}/{r.efficiency_dispatch:.3f}')
    print(f'Total: {phs.p_nom.sum():.0f} MW, {(phs.p_nom*phs.max_hours).sum():.0f} MWh, MC={phs.marginal_cost.unique()}')
    print('PHS pumps at MC=0 → can set zero prices when marginal')""")

md("---\n## Section 5 — Must-Run Capacity")
code("""nuc=n.generators[n.generators['carrier']=='nuclear']
mr=(nuc.p_nom*nuc.p_min_pu).sum(); tot=nuc.p_nom.sum()
print(f'Nuclear: {len(nuc)} units, {tot/1000:.1f} GW, must-run={mr/1000:.1f} GW (p_min={nuc.p_min_pu.unique()})')
omr=n.generators[(n.generators.p_min_pu>0)&(n.generators.carrier!='nuclear')]
if len(omr):
    for i,r in omr.iterrows(): print(f'  {i:30s} carrier={r.carrier:10s} p_nom={r.p_nom:6.0f} p_min={r.p_min_pu:.2f} must_run={r.p_nom*r.p_min_pu:6.0f}')
load=n.loads_t.p_set; esl=load[[c for c in load.columns if str(c).startswith('ES')]]
print(f'\\nES demand: peak={esl.sum(axis=1).max()/1000:.1f} GW mean={esl.sum(axis=1).mean()/1000:.1f} GW min={esl.sum(axis=1).min()/1000:.1f} GW')
print(f'Must-run/min demand={mr/esl.sum(axis=1).min()*100:.0f}% → nuclear floor suppresses prices')""")

md("---\n## Section 6 — Build Quality Assessment")
code("""issues=[]
# 1. Empty sub-network
es_=[s for s in n.sub_networks.index if s == '' or (isinstance(s, str) and not s)]
if es_:
    eb=n.buses[n.buses['sub_network'].isin(es_)]
    issues.append(f'HIGH: Empty-name sub-network ({len(eb)} buses: {len([b for b in eb.index if "H2" in str(b)])} H2 + {len([b for b in eb.index if "battery" in str(b)])} bat + {len([b for b in eb.index if str(b).startswith("FR")])} FR + {len([b for b in eb.index if str(b).startswith("PT")])} PT) → marginal_price=0')
# 2. Hydro SOC
hs=n.storage_units[n.storage_units['carrier']=='hydro']
if len(hs) and (hs.state_of_charge_initial==0).all():
    issues.append(f'HIGH: All {len(hs)} hydro reservoirs start empty ({(hs.p_nom*hs.max_hours).sum()/1000:.1f} GWh) → cannot dispatch')
# 3. MC=0 generators
zm=n.generators[n.generators['marginal_cost']==0]
if len(zm): issues.append(f'MEDIUM: {len(zm)} gens with MC=0 ({zm.groupby("carrier").p_nom.sum().to_dict()}) → can set zero prices')
# 4. FR/PT in empty sub
if es_:
    fp=n.buses[(n.buses['sub_network'].isin(es_))&(n.buses.index.str.startswith('FR')|n.buses.index.str.startswith('PT'))]
    if len(fp): issues.append(f'HIGH: {len(fp)} FR/PT buses in empty sub-network → prices computed with H2/battery')
# 5. No loads on H2/battery
for lbl,bl in [('H2',[b for b in n.buses.index if 'H2' in str(b)]),('Battery',[b for b in n.buses.index if 'battery' in str(b)])]:
    if len(bl) and len(n.loads[n.loads['bus'].isin(bl)])==0: issues.append(f'MEDIUM: No loads on {lbl} buses ({len(bl)}) → no price signal')
# 6. DC links across sub-networks
for i,r in n.links[n.links.carrier=='DC'].iterrows():
    s0=n.buses.at[r.bus0,'sub_network'] if r.bus0 in n.buses.index else '?'
    s1=n.buses.at[r.bus1,'sub_network'] if r.bus1 in n.buses.index else '?'
    if s0!=s1: issues.append(f'INFO: DC link {i}: {r.bus0}({s0}) ↔ {r.bus1}({s1})')
print('BUILD QUALITY ISSUES FOUND:\\n')
for iss in issues: print(f'  [{iss.split(":")[0]}] {iss.split(":",1)[1].strip()}')
print(f'\\nTotal: {len(issues)} issues')""")

# Assemble notebook
notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12.0"}},
    "nbformat": 4, "nbformat_minor": 5
}
out_path = os.path.join(os.path.dirname(__file__), "network_interrogation.ipynb")
with open(out_path, 'w') as f: json.dump(notebook, f, indent=1)
print(f"Notebook written to {out_path}")
