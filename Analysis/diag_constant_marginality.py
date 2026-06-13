#!/usr/bin/env python3
"""
CCGT Constant Marginality Diagnostic

Three-part investigation:
  1. Energy balance — total generation vs total demand across ES+FR+PT.
     A discrepancy here means the model is creating or destroying energy, which
     forces CCGT to compensate.
  2. Nodal CCGT dependence — which ES buses always need local gas generation?
     Node-level pockets caused by internal congestion are invisible in system averages.
  3. Trickle dispatch — with MIP off, does the LP spread tiny dispatch across every
     CCGT, making every unit "always marginal" at near-zero output?

Usage:
    python3 Analysis/diag_constant_marginality.py [path/to/solved.nc]
    (defaults to the latest solved_20240101_364d_20260529.nc)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

REPO = Path(__file__).parent.parent
NC_DEFAULT = REPO / "solved_networks/validation/solved_20240101_364d_20260529.nc"
NC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else NC_DEFAULT

SEP  = "=" * 72
DASH = "-" * 72

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title):
    print(f"\n  {title}\n  {DASH[:len(title)+2]}")


# ─── Load network ─────────────────────────────────────────────────────────────
print(f"Loading {NC_PATH.name} …")
n = pypsa.Network(str(NC_PATH))
snaps = n.snapshots
n_hours = len(snaps)
print(f"  {n_hours} snapshots  |  "
      f"{len(n.generators)} generators  |  "
      f"{len(n.buses)} buses  |  "
      f"{len(n.storage_units)} storage units")


# ─── helpers ──────────────────────────────────────────────────────────────────
def gen_dispatch(carriers, country=None):
    """Sum generator dispatch for given carrier(s) and optional country prefix."""
    if isinstance(carriers, str):
        carriers = [carriers]
    mask = n.generators["carrier"].isin(carriers)
    if country:
        mask &= n.generators["bus"].str.startswith(country)
    cols = [g for g in n.generators.index[mask] if g in n.generators_t.p.columns]
    return n.generators_t.p[cols].sum(axis=1) if cols else pd.Series(0.0, index=snaps)

def su_dispatch(carriers, country=None, mode="dispatch"):
    """Sum storage unit dispatch or store for given carriers."""
    if isinstance(carriers, str):
        carriers = [carriers]
    mask = n.storage_units["carrier"].isin(carriers)
    if country:
        mask &= n.storage_units["bus"].str.startswith(country)
    idx = n.storage_units.index[mask]
    tbl = n.storage_units_t.p_dispatch if mode == "dispatch" else n.storage_units_t.p_store
    cols = [u for u in idx if u in tbl.columns]
    return tbl[cols].sum(axis=1) if cols else pd.Series(0.0, index=snaps)

def load_demand(country):
    cols = [l for l in n.loads.index
            if n.loads.at[l, "bus"].startswith(country)
            and l in n.loads_t.p_set.columns]
    return n.loads_t.p_set[cols].sum(axis=1) if cols else pd.Series(0.0, index=snaps)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Energy balance: total gen vs total demand across ES + FR + PT
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 1 — ENERGY BALANCE: Total Gen vs Total Demand (ES + FR + PT)")

all_carriers = n.generators["carrier"].unique().tolist()
gas_carriers  = [c for c in all_carriers if "CCGT" in c or c in ("OCGT", "diesel")]
vre_carriers  = [c for c in all_carriers if c in ("solar", "onwind", "offwind-ac", "offwind-dc")]
nuke_carriers = [c for c in all_carriers if c == "nuclear"]
ror_carriers  = [c for c in all_carriers if c in ("ror",)]
bio_carriers  = [c for c in all_carriers if c in ("biomass", "load_shedding")]
hydro_su_c    = ["hydro", "PHS"]

results = {}
for country in ["ES", "FR", "PT"]:
    demand = load_demand(country)

    gas    = gen_dispatch(gas_carriers,  country)
    vre    = gen_dispatch(vre_carriers,  country)
    nuke   = gen_dispatch(nuke_carriers, country)
    ror    = gen_dispatch(ror_carriers,  country)
    bio    = gen_dispatch(bio_carriers,  country)
    hydro  = su_dispatch("hydro", country, "dispatch")
    phs_d  = su_dispatch("PHS",   country, "dispatch")
    phs_s  = su_dispatch("PHS",   country, "store")
    total_gen = gas + vre + nuke + ror + bio + hydro + phs_d - phs_s

    gap_mw  = total_gen - demand
    gap_twh = gap_mw.sum() / 1e6

    results[country] = {
        "demand_twh":   demand.sum() / 1e6,
        "gas_twh":      gas.sum() / 1e6,
        "vre_twh":      vre.sum() / 1e6,
        "nuclear_twh":  nuke.sum() / 1e6,
        "hydro_twh":    (hydro + ror).sum() / 1e6,
        "phs_net_twh":  (phs_d - phs_s).sum() / 1e6,
        "bio_twh":      bio.sum() / 1e6,
        "total_gen_twh": total_gen.sum() / 1e6,
        "gap_twh":      gap_twh,
        "gap_pct":      gap_twh / (demand.sum() / 1e6) * 100 if demand.sum() > 0 else 0,
    }

# Print table
print(f"\n  {'Metric':<24} {'ES':>10} {'FR':>10} {'PT':>10} {'TOTAL':>10}")
print(f"  {DASH}")
for key, label in [
    ("demand_twh",    "Demand (TWh)"),
    ("total_gen_twh", "Total Gen (TWh)"),
    ("gap_twh",       "Gap Gen−Demand (TWh)"),
    ("gap_pct",       "Gap % of demand"),
    ("gas_twh",       "  Gas (TWh)"),
    ("vre_twh",       "  VRE (TWh)"),
    ("nuclear_twh",   "  Nuclear (TWh)"),
    ("hydro_twh",     "  Hydro+RoR (TWh)"),
    ("phs_net_twh",   "  PHS net (TWh)"),
    ("bio_twh",       "  Bio/other (TWh)"),
]:
    row = [results[c][key] for c in ["ES", "FR", "PT"]]
    total = sum(row)
    fmt = "10.1f"
    print(f"  {label:<24} {row[0]:{fmt}} {row[1]:{fmt}} {row[2]:{fmt}} {total:{fmt}}")

print()
for country in ["ES", "FR", "PT"]:
    gap = results[country]["gap_twh"]
    pct = results[country]["gap_pct"]
    if abs(gap) > 1.0:
        print(f"  *** {country}: generation EXCEEDS demand by {gap:+.1f} TWh ({pct:+.1f}%) ***")
        print(f"      This excess must leave via interconnectors — if IC capacity is capped,")
        print(f"      the model cannot export, raises curtailment or forces strange dispatch.")
    elif abs(gap) > 0.1:
        print(f"  ⚠  {country}: small imbalance {gap:+.2f} TWh ({pct:+.2f}%) — likely IC flows or losses")
    else:
        print(f"  ✓  {country}: balanced ({gap:+.2f} TWh)")

# System-wide IC flows (lines crossing borders)
sub("Cross-border line flows (net, TWh positive = into ES)")
border_lines = n.lines.index[
    n.lines["bus0"].str.startswith(("FR","PT")) | n.lines["bus1"].str.startswith(("FR","PT")) |
    n.lines["bus0"].str.startswith("ES") & (n.lines["bus1"].str.startswith(("FR","PT"))) |
    n.lines["bus1"].str.startswith("ES") & (n.lines["bus0"].str.startswith(("FR","PT")))
]
for l in border_lines:
    if l not in n.lines_t.p0.columns:
        continue
    p0 = n.lines_t.p0[l]
    b0 = n.lines.at[l, "bus0"]
    # positive p0 = power flowing bus0→bus1
    # if bus0 is FR/PT and bus1 is ES: positive p0 means FR/PT→ES (import to ES)
    if b0.startswith(("FR","PT")):
        net_to_es = p0.sum() / 1e6
    else:
        net_to_es = -p0.sum() / 1e6
    print(f"    {l:<20}: {net_to_es:+.2f} TWh net to ES  (mean {net_to_es*1e6/n_hours:+.0f} MW)")

# INELFE
if "INELFE" in n.links.index and "INELFE" in n.links_t.p0.columns:
    p0 = n.links_t.p0["INELFE"]
    # bus0=ES → positive p0 = ES→FR export (negative for ES)
    net_to_es = -p0.sum() / 1e6
    print(f"    {'INELFE (DC)':<20}: {net_to_es:+.2f} TWh net to ES  (mean {net_to_es*1e6/n_hours:+.0f} MW)")


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Residual demand: does it ever go negative?
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 2 — ES RESIDUAL DEMAND STRUCTURE")
print("  Residual = ES_load − (nuclear + VRE + hydro + PHS_net + RoR + biomass + net_IC)")
print("  If residual > 0 in X% of hours, CCGT must physically run X% of hours.\n")

es_load = load_demand("ES")
es_nuke = gen_dispatch(nuke_carriers, "ES")
es_vre  = gen_dispatch(vre_carriers,  "ES")
es_ror  = gen_dispatch(ror_carriers,  "ES")
es_bio  = gen_dispatch(bio_carriers,  "ES")
es_hydro = su_dispatch("hydro", "ES", "dispatch")
es_phs_d = su_dispatch("PHS",   "ES", "dispatch")
es_phs_s = su_dispatch("PHS",   "ES", "store")
es_must_run = gen_dispatch(["CCGT_must_run"], "ES")

# Net IC to ES
net_ic = pd.Series(0.0, index=snaps)
for l in border_lines:
    if l not in n.lines_t.p0.columns:
        continue
    p0 = n.lines_t.p0[l]
    b0 = n.lines.at[l, "bus0"]
    net_ic += p0 if b0.startswith(("FR","PT")) else -p0
if "INELFE" in n.links.index and "INELFE" in n.links_t.p0.columns:
    net_ic -= n.links_t.p0["INELFE"]

residual = (es_load
            - es_nuke - es_vre - es_ror - es_bio
            - es_hydro - (es_phs_d - es_phs_s)
            - es_must_run
            - net_ic)

ccgt_all = gen_dispatch(gas_carriers, "ES")

print(f"  ES load mean               : {es_load.mean():8.0f} MW")
print(f"  Nuclear                    : {es_nuke.mean():8.0f} MW")
print(f"  VRE (solar+wind)           : {es_vre.mean():8.0f} MW")
print(f"  RoR                        : {es_ror.mean():8.0f} MW")
print(f"  Biomass/must-run gen       : {es_bio.mean():8.0f} MW")
print(f"  Hydro reservoir dispatch   : {es_hydro.mean():8.0f} MW")
print(f"  PHS net                    : {(es_phs_d-es_phs_s).mean():8.0f} MW")
print(f"  CCGT must-run (MC=0)       : {es_must_run.mean():8.0f} MW")
print(f"  Net IC to ES               : {net_ic.mean():8.0f} MW")
print(f"  ──────────────────────────────────────────")
print(f"  Residual (→ CCGT must cover): {residual.mean():8.0f} MW")
print(f"  Actual CCGT dispatch mean  : {ccgt_all.mean():8.0f} MW")

print(f"\n  Residual demand distribution (hours):")
thresholds = [-2000, -500, 0, 500, 1000, 2000, 4000, 6000]
for lo, hi in zip(thresholds, thresholds[1:] + [99999]):
    mask = (residual >= lo) & (residual < hi)
    hrs = mask.sum()
    pct = hrs / n_hours * 100
    bar = "█" * int(pct / 2)
    print(f"    [{lo:6d},{hi:6d}) MW : {hrs:5d} hrs ({pct:5.1f}%)  {bar}")

pct_positive = (residual > 0).mean() * 100
pct_negative = (residual < 0).mean() * 100
print(f"\n  Residual > 0 (CCGT structurally needed) : {pct_positive:.1f}% of hours")
print(f"  Residual < 0 (true surplus, CCGT optional): {pct_negative:.1f}% of hours")
print(f"  Min residual: {residual.min():.0f} MW   Max: {residual.max():.0f} MW")

# In OMIE near-zero hours, is residual negative?
es_price_ts = n.buses_t.marginal_price[
    [b for b in n.buses_t.marginal_price.columns if b.startswith("ES")]
].mean(axis=1)

for label, mask in [
    ("price < €5 (model near-zero)", es_price_ts < 5),
    ("price €5–30",                  (es_price_ts >= 5) & (es_price_ts < 30)),
    ("price €70–100",                (es_price_ts >= 70) & (es_price_ts < 100)),
    ("price > €110",                 es_price_ts >= 110),
]:
    if mask.sum() == 0:
        continue
    print(f"\n  {label} → {mask.sum()} hrs:")
    print(f"    Mean residual  : {residual[mask].mean():+.0f} MW")
    print(f"    Mean CCGT disp : {ccgt_all[mask].mean():.0f} MW")
    print(f"    Mean net IC    : {net_ic[mask].mean():+.0f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Nodal CCGT dependence: which buses always need local gas?
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 3 — NODAL CCGT DEPENDENCE (Bus-level analysis)")
print("  Identifies ES buses where CCGT runs near-constantly —")
print("  a sign of local congestion pockets isolating them from cheap VRE/nuclear.\n")

ccgt_main_mask = (
    n.generators["carrier"].isin(["CCGT", "CCGT_flex"]) &
    n.generators["bus"].str.startswith("ES")
)
ccgt_gens = n.generators[ccgt_main_mask]

bus_rows = []
for bus in sorted(n.buses.index[n.buses.index.str.startswith("ES")]):
    local_ccgt = ccgt_gens[ccgt_gens["bus"] == bus].index.tolist()
    if not local_ccgt:
        continue
    local_cols = [g for g in local_ccgt if g in n.generators_t.p.columns]
    if not local_cols:
        continue

    p_local = n.generators_t.p[local_cols].sum(axis=1)
    p_nom   = ccgt_gens.loc[local_ccgt, "p_nom"].sum()

    # Local cheap gen at this bus
    cheap_at_bus = [g for g in n.generators.index
                    if n.generators.at[g, "bus"] == bus
                    and n.generators.at[g, "carrier"] in
                        ("solar","onwind","nuclear","hydro","ror","biomass")
                    and g in n.generators_t.p.columns]
    cheap_gen_bus = n.generators_t.p[cheap_at_bus].sum(axis=1) if cheap_at_bus else pd.Series(0.0, index=snaps)

    # Local load
    loads_at_bus = [l for l in n.loads.index
                    if n.loads.at[l, "bus"] == bus
                    and l in n.loads_t.p_set.columns]
    load_bus = n.loads_t.p_set[loads_at_bus].sum(axis=1) if loads_at_bus else pd.Series(0.0, index=snaps)

    pct_on      = (p_local > 1.0).mean() * 100
    mean_pu     = (p_local / p_nom).mean() if p_nom > 0 else 0
    pct_trickle = ((p_local > 0.5) & ((p_local / p_nom) < 0.10)).mean() * 100

    bus_rows.append({
        "bus": bus,
        "ccgt_pnom_mw": p_nom,
        "n_units": len(local_ccgt),
        "pct_on": pct_on,
        "mean_pu": mean_pu,
        "pct_trickle": pct_trickle,
        "mean_ccgt_mw": p_local.mean(),
        "mean_cheap_mw": cheap_gen_bus.mean(),
        "mean_load_mw": load_bus.mean(),
        "cheap_cover_pct": cheap_gen_bus.mean() / load_bus.mean() * 100 if load_bus.mean() > 0 else 0,
    })

bus_df = pd.DataFrame(bus_rows).sort_values("pct_on", ascending=False)

print(f"  {'Bus':<10} {'CCGT GW':>8} {'Units':>6} {'%On':>7} {'MeanPU':>8} "
      f"{'%Trickle':>9} {'CheapCov%':>10}")
print(f"  {'-'*66}")
for _, r in bus_df.iterrows():
    flag = " ← POCKET" if r.pct_on > 90 and r.pct_trickle > 30 else ""
    print(f"  {str(r.bus):<10} {r.ccgt_pnom_mw/1000:>8.2f} {int(r.n_units):>6} "
          f"{r.pct_on:>6.1f}% {r.mean_pu:>8.3f} {r.pct_trickle:>8.1f}% "
          f"{r.cheap_cover_pct:>9.1f}%{flag}")

bus_df.to_csv(REPO / "Analysis/validation_output/diag_ccgt_by_bus.csv", index=False)
print(f"\n  Saved: Analysis/validation_output/diag_ccgt_by_bus.csv")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Trickle dispatch: how many generators run at < 5% of p_nom?
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 4 — TRICKLE DISPATCH DETECTION (LP spreading without MIP)")
print("  With MIP off, LP can dispatch every CCGT at 0.5–5% of p_nom,")
print("  making every unit 'on' and every one technically the marginal unit.\n")

trickle_rows = []
for g in ccgt_gens.index:
    if g not in n.generators_t.p.columns:
        continue
    p = n.generators_t.p[g]
    pnom = float(ccgt_gens.at[g, "p_nom"])
    if pnom < 1.0:
        continue
    pu = p / pnom
    pct_off      = (p < 0.5).mean() * 100
    pct_trickle  = ((p >= 0.5) & (pu < 0.05)).mean() * 100   # on but <5% p_nom
    pct_mid      = ((pu >= 0.05) & (pu < 0.50)).mean() * 100
    pct_full     = (pu >= 0.50).mean() * 100
    mean_pu_on   = pu[p >= 0.5].mean() if (p >= 0.5).any() else 0.0
    trickle_rows.append({
        "generator": g,
        "bus": ccgt_gens.at[g, "bus"],
        "carrier": ccgt_gens.at[g, "carrier"],
        "p_nom": pnom,
        "pct_off": pct_off,
        "pct_trickle_lt5": pct_trickle,
        "pct_mid": pct_mid,
        "pct_full_gt50": pct_full,
        "mean_pu_when_on": mean_pu_on,
    })

td_df = pd.DataFrame(trickle_rows).sort_values("pct_trickle_lt5", ascending=False)

n_heavy_trickle = (td_df["pct_trickle_lt5"] > 20).sum()
mean_trickle    = td_df["pct_trickle_lt5"].mean()
print(f"  {len(td_df)} CCGT generators  |  "
      f"{n_heavy_trickle} with >20% trickle hours  |  "
      f"Fleet mean trickle: {mean_trickle:.1f}%\n")

print(f"  {'Generator':<28} {'MW':>6} {'%Off':>6} {'%<5%pu':>8} {'%mid':>6} {'%>50%':>7} {'MeanPU':>7}")
print(f"  {'-'*72}")
for _, r in td_df.head(25).iterrows():
    flag = " ← TRICKLE" if r.pct_trickle_lt5 > 30 else ""
    print(f"  {str(r.generator):<28} {r.p_nom:>6.0f} {r.pct_off:>5.1f}% "
          f"{r.pct_trickle_lt5:>7.1f}% {r.pct_mid:>5.1f}% {r.pct_full_gt50:>6.1f}% "
          f"{r.mean_pu_when_on:>7.3f}{flag}")

td_df.to_csv(REPO / "Analysis/validation_output/diag_ccgt_trickle.csv", index=False)
print(f"\n  Saved: Analysis/validation_output/diag_ccgt_trickle.csv")

# Fleet summary
print(f"\n  Fleet-wide trickle summary:")
print(f"    Hours with ANY CCGT > 0.5 MW : {(ccgt_all > 0.5).mean()*100:.1f}%")
print(f"    Hours with CCGT > 100 MW     : {(ccgt_all > 100).mean()*100:.1f}%")
print(f"    Hours with CCGT > 1000 MW    : {(ccgt_all > 1000).mean()*100:.1f}%")
print(f"    Hours with CCGT > 5000 MW    : {(ccgt_all > 5000).mean()*100:.1f}%")
print(f"    Mean dispatch (all hours)    : {ccgt_all.mean():.0f} MW")
print(f"    Mean dispatch (when > 100MW) : {ccgt_all[ccgt_all>100].mean():.0f} MW")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
hdr("SUMMARY")

total_es_gap = results["ES"]["gap_twh"]
total_fr_gap = results["FR"]["gap_twh"]
total_pt_gap = results["PT"]["gap_twh"]
system_gap   = total_es_gap + total_fr_gap + total_pt_gap

print(f"  System energy gap (ES+FR+PT gen − demand): {system_gap:+.1f} TWh")
if abs(system_gap) > 2.0:
    print(f"  *** LARGE IMBALANCE — model creating/destroying {abs(system_gap):.0f} TWh ***")
    print(f"      This likely forces artificial CCGT dispatch to fill phantom demand.")
else:
    print(f"  ✓ System is broadly balanced (IC flows absorb the country-level gaps)")

print(f"\n  Residual demand (what CCGT must cover) > 0 in {pct_positive:.0f}% of hours")
if pct_positive > 85:
    print(f"  *** HIGH structural CCGT dependence — gas is physically needed most hours ***")
    print(f"      Root cause: demand exceeds cheap gen + IC imports in {pct_positive:.0f}% of hours.")
    print(f"      MIP fixes trickle dispatch but NOT this structural shortfall.")
    print(f"      To fix: need more cheap gen dispatch or more net imports.")
elif pct_positive > 60:
    print(f"  Moderate CCGT dependence. MIP can reduce marginal hours significantly.")
else:
    print(f"  Low structural CCGT need — trickle dispatch is the main driver.")

print(f"\n  Trickle dispatch: {mean_trickle:.0f}% of hours are sub-5%-p_nom dispatch per unit")
if mean_trickle > 20:
    print(f"  *** SIGNIFICANT trickle — MIP would eliminate these hours ***")
    print(f"      With n_top_mip=15, committed CCGTs must run at ≥30% p_nom or be off.")

print(f"\n  Top nodal pockets (>95% CCGT on-time):")
top_pockets = bus_df[bus_df["pct_on"] > 95]
if top_pockets.empty:
    print(f"    None — CCGT on-time is distributed")
else:
    for _, r in top_pockets.iterrows():
        print(f"    {r.bus}: {r.pct_on:.0f}% on, {r.mean_pu:.2f} mean p.u., "
              f"cheap cover {r.cheap_cover_pct:.0f}% of local load")

print()
