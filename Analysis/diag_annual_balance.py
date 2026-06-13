#!/usr/bin/env python3
"""
Annual Generation Balance + Border Flow Diagnostic

Three sections:
  1. Annual generation totals — model vs real 2024 (ES carriers)
  2. Border flow totals — per corridor, annual + monthly
  3. CCGT under-dispatch diagnosis — are FR imports substituting for domestic gas?

Usage:
    python3 Analysis/diag_annual_balance.py [path/to/solved.nc]
    (defaults to latest 365d solve)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

REPO = Path(__file__).parent.parent
NC_DEFAULT = REPO / "solved_networks/validation/solved_20240101_365d_20260601.nc"
NC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else NC_DEFAULT

SEP  = "=" * 72
DASH = "-" * 72

def hdr(title): print(f"\n{SEP}\n  {title}\n{SEP}")
def sub(title): print(f"\n  {title}\n  {DASH[:len(title)+2]}")


print(f"Loading {NC_PATH.name} …")
n = pypsa.Network(str(NC_PATH))
snaps = n.snapshots
n_hours = len(snaps)
print(f"  {n_hours} snapshots  ({snaps[0].date()} → {snaps[-1].date()})")

# ── helpers ───────────────────────────────────────────────────────────────────
def gen_twh(carriers, country="ES"):
    if isinstance(carriers, str): carriers = [carriers]
    mask = n.generators["carrier"].isin(carriers)
    if country: mask &= n.generators["bus"].str.startswith(country)
    cols = [g for g in n.generators.index[mask] if g in n.generators_t.p.columns]
    return n.generators_t.p[cols].sum(axis=1).sum() / 1e6 if cols else 0.0

def su_twh(carriers, country="ES", mode="dispatch"):
    if isinstance(carriers, str): carriers = [carriers]
    mask = n.storage_units["carrier"].isin(carriers)
    if country: mask &= n.storage_units["bus"].str.startswith(country)
    idx = n.storage_units.index[mask]
    tbl = n.storage_units_t.p_dispatch if mode == "dispatch" else n.storage_units_t.p_store
    cols = [u for u in idx if u in tbl.columns]
    return tbl[cols].sum(axis=1).sum() / 1e6 if cols else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Annual generation totals
# ─────────────────────────────────────────────────────────────────────────────
hdr("SECTION 1 — ANNUAL GENERATION TOTALS: MODEL vs REAL 2024 (ES)")

real_path = REPO / "Analysis/data/spain_actual_generation_2024.csv"
real = pd.read_csv(real_path, parse_dates=["timestamp"])
real["timestamp"] = pd.to_datetime(real["timestamp"], utc=True).dt.tz_localize(None)
real = real.set_index("timestamp").reindex(snaps)

rows = [
    ("Nuclear",          real["Nuclear"].sum()/1e6,           gen_twh("nuclear")),
    ("Wind (onshore)",   real["Wind"].sum()/1e6,              gen_twh("onwind")),
    ("Solar PV",         real["Solar_PV"].sum()/1e6,          gen_twh("solar")),
    ("Hydro reservoir",  real["Hydro_Reservoir"].sum()/1e6,   su_twh("hydro")),
    ("Hydro RoR",        real["Hydro_River"].sum()/1e6,       gen_twh("ror")),
    ("CCGT+OCGT",        real["CCGT"].sum()/1e6,              gen_twh(["CCGT","CCGT_flex","OCGT"])),
    ("Biomass/Cogen",    real["Cogeneration"].sum()/1e6,      gen_twh("biomass")),
]

total_real  = sum(r[1] for r in rows)
total_model = sum(r[2] for r in rows)

print(f"\n  {'Carrier':<20} {'Real TWh':>10} {'Model TWh':>11} {'Δ TWh':>8} {'Δ %':>7}")
print(f"  {DASH}")
for label, r, m in rows:
    delta = m - r
    pct   = delta / r * 100 if r > 0 else 0
    flag  = " ← HIGH" if pct > 25 else (" ← LOW" if pct < -25 else "")
    print(f"  {label:<20} {r:>10.2f} {m:>11.2f} {delta:>+8.2f} {pct:>+6.1f}%{flag}")
print(f"  {DASH}")
print(f"  {'TOTAL':<20} {total_real:>10.2f} {total_model:>11.2f} "
      f"{total_model-total_real:>+8.2f} {(total_model-total_real)/total_real*100:>+6.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Border flows
# ─────────────────────────────────────────────────────────────────────────────
hdr("SECTION 2 — BORDER FLOWS (positive = import TO ES)")

# Per-corridor breakdown
corridors = {}
for l in n.lines.index:
    b0, b1 = n.lines.at[l, "bus0"], n.lines.at[l, "bus1"]
    if b0[:2] == b1[:2]: continue
    if l not in n.lines_t.p0.columns: continue
    p0   = n.lines_t.p0[l]
    snom = n.lines.at[l, "s_nom"]
    # positive p0 = b0→b1; if b0=ES → export; if b1=ES → import
    sign = -1 if b0.startswith("ES") else +1
    net_twh = sign * p0.sum() / 1e6
    util    = p0.abs().mean() / snom * 100 if snom > 0 else 0
    corridors[l] = {"name": l, "b0": b0, "b1": b1, "net_twh": net_twh,
                    "cap_mw": snom, "util_pct": util, "type": "AC"}

for lk in n.links.index:
    b0, b1 = n.links.at[lk, "bus0"], n.links.at[lk, "bus1"]
    if lk not in n.links_t.p0.columns: continue
    if "battery" in lk or "H2" in lk or "heat" in lk: continue
    if not any(x.startswith(("FR","PT")) for x in [b0, b1]): continue
    if not any(x.startswith("ES") for x in [b0, b1]): continue
    p0   = n.links_t.p0[lk]
    pnom = n.links.at[lk, "p_nom"]
    if pnom <= 0: continue
    sign = -1 if b0.startswith("ES") else +1
    net_twh = sign * p0.sum() / 1e6
    util    = p0.abs().mean() / pnom * 100
    corridors[lk] = {"name": lk, "b0": b0, "b1": b1, "net_twh": net_twh,
                     "cap_mw": pnom, "util_pct": util, "type": "DC"}

sub("Per-corridor annual flows")
print(f"  {'Corridor':<35} {'B0':<12} {'B1':<12} {'Net TWh':>9} {'Cap MW':>8} {'Util%':>7}")
print(f"  {DASH}")
total_net = 0
for k, c in sorted(corridors.items(), key=lambda x: abs(x[1]["net_twh"]), reverse=True):
    direction = "→ES" if c["net_twh"] > 0 else "ES→"
    total_net += c["net_twh"]
    print(f"  {c['name']:<35} {c['b0']:<12} {c['b1']:<12} "
          f"{c['net_twh']:>+9.2f} {c['cap_mw']:>8.0f} {c['util_pct']:>6.1f}%")
print(f"\n  Annual NET to ES: {total_net:+.2f} TWh  "
      f"({'IMPORT' if total_net > 0 else 'EXPORT'} net position)")

# Monthly net IC to ES
net_ic_ts = pd.Series(0.0, index=snaps)
for l in n.lines.index:
    b0, b1 = n.lines.at[l, "bus0"], n.lines.at[l, "bus1"]
    if b0[:2] == b1[:2]: continue
    if l not in n.lines_t.p0.columns: continue
    p0   = n.lines_t.p0[l]
    net_ic_ts += (-p0 if b0.startswith("ES") else p0)
for lk in n.links.index:
    b0, b1 = n.links.at[lk, "bus0"], n.links.at[lk, "bus1"]
    if lk not in n.links_t.p0.columns: continue
    if "battery" in lk or "H2" in lk or "heat" in lk: continue
    if not any(x.startswith(("FR","PT")) for x in [b0, b1]): continue
    if not any(x.startswith("ES") for x in [b0, b1]): continue
    p0   = n.links_t.p0[lk]
    pnom = n.links.at[lk, "p_nom"]
    if pnom <= 0: continue
    net_ic_ts += (-p0 if b0.startswith("ES") else p0)

sub("Monthly net IC to ES (TWh)")
print(f"  {'Mo':>3}  {'Net TWh':>9}  {'Mean MW':>9}  {'Direction'}")
print(f"  {'-'*38}")
for mo in range(1, 13):
    m = snaps[snaps.month == mo]
    twh  = net_ic_ts[m].sum() / 1e6
    mean = net_ic_ts[m].mean()
    direction = "← IMPORT" if twh > 0.5 else ("→ EXPORT" if twh < -0.5 else "balanced")
    print(f"  {mo:>3}  {twh:>+9.3f}  {mean:>+9.0f}  {direction}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — ES CCGT under-dispatch diagnosis
# ─────────────────────────────────────────────────────────────────────────────
hdr("SECTION 3 — ES CCGT UNDER-DISPATCH: ARE FR IMPORTS SUBSTITUTING FOR DOMESTIC GAS?")
print("  Residual = ES_load − (nuclear + VRE + hydro + RoR + biomass)")
print("  If residual ≈ net_IC → imports cover gap and CCGT isn't needed")
print("  If residual > net_IC → CCGT structurally needed\n")

es_load_cols = [l for l in n.loads.index if n.loads.at[l,"bus"].startswith("ES")
                and l in n.loads_t.p_set.columns]
es_load = n.loads_t.p_set[es_load_cols].sum(axis=1)

cheap_carriers = ["nuclear","solar","onwind","ror","biomass"]
es_cheap = pd.Series(0.0, index=snaps)
for c in cheap_carriers:
    gens = n.generators[(n.generators["carrier"]==c) & (n.generators["bus"].str.startswith("ES"))]
    cols = [g for g in gens.index if g in n.generators_t.p.columns]
    if cols: es_cheap += n.generators_t.p[cols].sum(axis=1)
# add hydro storage dispatch
hydro_su = n.storage_units[(n.storage_units["carrier"]=="hydro") &
                            (n.storage_units["bus"].str.startswith("ES"))]
hcols = [u for u in hydro_su.index if u in n.storage_units_t.p_dispatch.columns]
if hcols: es_cheap += n.storage_units_t.p_dispatch[hcols].sum(axis=1)

# ES CCGT
gas_carriers = ["CCGT","CCGT_flex","OCGT","diesel"]
es_ccgt = pd.Series(0.0, index=snaps)
for c in gas_carriers:
    gens = n.generators[(n.generators["carrier"]==c) & (n.generators["bus"].str.startswith("ES"))]
    cols = [g for g in gens.index if g in n.generators_t.p.columns]
    if cols: es_ccgt += n.generators_t.p[cols].sum(axis=1)

residual = es_load - es_cheap

print(f"  {'Mo':>3}  {'Load TWh':>9}  {'Cheap TWh':>10}  {'Residual':>9}  "
      f"{'CCGT TWh':>9}  {'NetIC TWh':>10}  {'IC/Res%':>8}  Analysis")
print(f"  {'-'*88}")

annual_ccgt = 0; annual_ic = 0; annual_res = 0
for mo in range(1, 13):
    m = snaps[snaps.month == mo]
    load_twh = es_load[m].sum() / 1e6
    cheap_twh = es_cheap[m].sum() / 1e6
    res_twh  = residual[m].sum() / 1e6
    ccgt_twh = es_ccgt[m].sum() / 1e6
    ic_twh   = net_ic_ts[m].sum() / 1e6
    ic_pct   = ic_twh / res_twh * 100 if abs(res_twh) > 0.1 else 0
    annual_ccgt += ccgt_twh; annual_ic += ic_twh; annual_res += res_twh

    if ic_pct > 60:
        analysis = "FR imports dominating → ES CCGT suppressed"
    elif ic_pct > 30:
        analysis = "Mix: imports + CCGT sharing residual"
    elif ic_pct < -10:
        analysis = "ES exporting surplus"
    else:
        analysis = "CCGT covering residual"
    print(f"  {mo:>3}  {load_twh:>9.2f}  {cheap_twh:>10.2f}  {res_twh:>+9.2f}  "
          f"{ccgt_twh:>9.2f}  {ic_twh:>+10.2f}  {ic_pct:>7.1f}%  {analysis}")

print(f"  {'-'*88}")
print(f"  {'ANN':>3}  {'':>9}  {'':>10}  {annual_res:>+9.2f}  "
      f"{annual_ccgt:>9.2f}  {annual_ic:>+10.2f}  "
      f"{annual_ic/annual_res*100 if abs(annual_res)>0.1 else 0:>7.1f}%")

# Key finding
ic_frac = annual_ic / annual_res * 100 if abs(annual_res) > 0.1 else 0
print(f"\n  Annual residual: {annual_res:.1f} TWh")
print(f"  FR/PT imports cover {ic_frac:.1f}% of residual demand")
print(f"  ES CCGT covers {annual_ccgt/annual_res*100:.1f}% of residual demand")
if ic_frac > 40:
    print(f"\n  *** IMPORTS ARE SUPPRESSING ES CCGT ***")
    print(f"      FR/PT covering {ic_frac:.0f}% of Spain's residual demand.")
    print(f"      Reducing IC flow (raise FR MC or reduce FR nuclear) would force more ES CCGT dispatch.")
else:
    print(f"\n  ✓ ES CCGT is the dominant residual supplier ({annual_ccgt/annual_res*100:.0f}% of residual).")
