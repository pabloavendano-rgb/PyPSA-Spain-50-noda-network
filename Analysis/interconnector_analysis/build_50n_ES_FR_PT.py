"""
Build merged network: ES 50-node + PT proxy (3-bus) + FR proxy (2-bus).

Architecture per corridor (replicating prepare_network.py logic):
    ES_host  --(export link)-->  border_bus  --(country link)-->  proxy_bus
    ES_host  <--(import link)--  border_bus

Carries ALL data from the proxy networks:
  - Buses, internal lines, generators (with p_max_pu time series), loads (with p_set time series)
  - Market price generators are NOT added here — they will be generated at optimisation time

Output: resources/networks/50n_ES_FR_PT.nc
"""

import warnings
import numpy as np
import pandas as pd
import pypsa

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os, pathlib
_ROOT = pathlib.Path(__file__).resolve().parents[2]  # pypsa-spain-master/

ES_PATH  = _ROOT / "resources/networks/base_s_50_elec_ES.nc"
PT_PATH  = _ROOT / "resources/networks/pt_proxy.nc"
FR_PATH  = _ROOT / "resources/networks/fr_proxy.nc"
OUT_PATH = _ROOT / "resources/networks/50n_ES_FR_PT.nc"

# ---------------------------------------------------------------------------
# Corridor definitions
# NTC split: PT total export 3 600 MW / import 2 880 MW (REE 2024)
# distributed by proxy thermal capacity share.
# FR: use proxy es_cap_MW directly (thermal rating, symmetric).
# ---------------------------------------------------------------------------
PT_THERMAL = {"PT_NORTH": 3500, "PT_CENTRE": 3575, "PT_SOUTH": 1787}
PT_TOTAL   = sum(PT_THERMAL.values())  # 8 862 MW
PT_EXP_NTC = 3600   # ES→PT
PT_IMP_NTC = 2880   # PT→ES

CORRIDORS = {
    # --- Portugal ---
    "PT_NORTH": {
        "border_lon": -6.652149, "border_lat": 41.223112,
        "proxy_bus":  "PT_NORTH",
        "p_exp": round(PT_EXP_NTC * PT_THERMAL["PT_NORTH"] / PT_TOTAL),  # ~1 422
        "p_imp": round(PT_IMP_NTC * PT_THERMAL["PT_NORTH"] / PT_TOTAL),  # ~1 137
        "country": "PT",
        "length_km": 36.0,
    },
    "PT_CENTRE": {
        "border_lon": -7.014351, "border_lat": 38.901836,
        "proxy_bus":  "PT_CENTRE",
        "p_exp": round(PT_EXP_NTC * PT_THERMAL["PT_CENTRE"] / PT_TOTAL),  # ~1 452
        "p_imp": round(PT_IMP_NTC * PT_THERMAL["PT_CENTRE"] / PT_TOTAL),  # ~1 161
        "country": "PT",
        "length_km": 42.0,
    },
    "PT_SOUTH": {
        "border_lon": -7.20, "border_lat": 37.50,
        "proxy_bus":  "PT_SOUTH",
        "p_exp": round(PT_EXP_NTC * PT_THERMAL["PT_SOUTH"] / PT_TOTAL),  # ~  726
        "p_imp": round(PT_IMP_NTC * PT_THERMAL["PT_SOUTH"] / PT_TOTAL),  # ~  582
        "country": "PT",
        "length_km": 44.0,
    },
    # --- France ---
    "FR_WEST": {
        "border_lon": -1.90, "border_lat": 43.20,
        "proxy_bus":  "FR_WEST",
        "p_exp": 2793,   # AC thermal (3 circuits)
        "p_imp": 2793,
        "country": "FR",
        "length_km": 22.0,
    },
    "FR_EAST": {
        "border_lon": 2.877482, "border_lat": 42.426598,
        "proxy_bus":  "FR_EAST",
        "p_exp": 3787,   # DC 2 800 + AC 987 MW eastern corridor
        "p_imp": 3787,
        "country": "FR",
        "length_km": 82.0,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def haversine_km(lon0, lat0, lon1, lat1):
    R = 6371.0
    dlat, dlon = np.radians(lat1 - lat0), np.radians(lon1 - lon0)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat0)) * np.cos(np.radians(lat1)) * np.sin(dlon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def closest_es_ac_node(n, lon, lat):
    """Closest Spanish ES0x AC bus to (lon, lat)."""
    ac = n.buses[(n.buses.index.str.contains("ES0")) & (n.buses.carrier == "AC")][["x", "y"]]
    dists = ac.apply(lambda r: haversine_km(lon, lat, r.x, r.y), axis=1)
    return dists.idxmin()


# ---------------------------------------------------------------------------
# Load networks
# ---------------------------------------------------------------------------
print("Loading networks …")
n_es = pypsa.Network(ES_PATH)
n_pt = pypsa.Network(PT_PATH)
n_fr = pypsa.Network(FR_PATH)

# Align snapshots: proxy networks have 8784 (2024 leap year), ES has 8760.
# Use ES snapshots as the master — proxy time series will be reindexed.
snapshots = n_es.snapshots  # 8 760 hourly steps
n_snap = len(snapshots)
print(f"  ES snapshots: {n_snap} ({snapshots[0]} → {snapshots[-1]})")
print(f"  PT snapshots: {len(n_pt.snapshots)}")
print(f"  FR snapshots: {len(n_fr.snapshots)}")

# ---------------------------------------------------------------------------
# Start with the full ES network
# ---------------------------------------------------------------------------
print("Copying ES network …")
n = n_es.copy()

# ---------------------------------------------------------------------------
# Add DC_ic carriers (mirroring prepare_network.py)
# ---------------------------------------------------------------------------
for c in ("DC_ic", "DC_ic export", "DC_ic import"):
    if c not in n.carriers.index:
        n.add("Carrier", c, color=n.carriers.at["AC", "color"])

# ---------------------------------------------------------------------------
# Add proxy buses and internal lines from PT and FR proxy networks
# ---------------------------------------------------------------------------
print("Adding PT and FR proxy buses …")

for proxy_net, country_tag in [(n_pt, "PT"), (n_fr, "FR")]:
    for bus_id, row in proxy_net.buses.iterrows():
        n.add(
            "Bus", bus_id,
            x=row.x, y=row.y,
            v_nom=row.v_nom,
            carrier="AC",
            country=country_tag,
        )

    for line_id, row in proxy_net.lines.iterrows():
        n.add(
            "Line", line_id,
            bus0=row.bus0, bus1=row.bus1,
            s_nom=row.s_nom,
            x=row.x, r=row.r,
            v_nom=row.v_nom,
            num_parallel=row.num_parallel,
            length=row.length,
            type="",
        )

# ---------------------------------------------------------------------------
# Add proxy generators (with their p_max_pu time series)
# ---------------------------------------------------------------------------
print("Adding proxy generators …")

for proxy_net in [n_pt, n_fr]:
    for gen_id, row in proxy_net.generators.iterrows():
        n.add(
            "Generator", gen_id,
            bus=row.bus,
            carrier=row.carrier,
            p_nom=row.p_nom,
            p_nom_extendable=row.p_nom_extendable,
            p_nom_min=row.p_nom_min,
            p_nom_max=row.p_nom_max,
            p_min_pu=row.p_min_pu,
            p_max_pu=row.p_max_pu,
            marginal_cost=row.marginal_cost,
            marginal_cost_quadratic=row.marginal_cost_quadratic,
            efficiency=row.efficiency,
            committable=row.committable,
            ramp_limit_up=row.ramp_limit_up,
            ramp_limit_down=row.ramp_limit_down,
            build_year=row.build_year,
            lifetime=row.lifetime,
            capital_cost=row.capital_cost,
        )

    # Carry over p_max_pu time series (wind/solar/offwind profiles)
    if "p_max_pu" in proxy_net.generators_t:
        for col in proxy_net.generators_t["p_max_pu"].columns:
            if col in n.generators.index:
                # Reindex from proxy snapshots (8784) to ES snapshots (8760)
                # using forward-fill for the extra Feb-29 hours
                ts = proxy_net.generators_t["p_max_pu"][col]
                ts_reindexed = ts.reindex(snapshots, method="ffill")
                n.generators_t["p_max_pu"][col] = ts_reindexed.values

# ---------------------------------------------------------------------------
# Add proxy loads (with their p_set time series)
# ---------------------------------------------------------------------------
print("Adding proxy loads …")

for proxy_net in [n_pt, n_fr]:
    for load_id, row in proxy_net.loads.iterrows():
        n.add(
            "Load", load_id,
            bus=row.bus,
            carrier=row.carrier,
            sign=row.sign,
        )

    # Carry over p_set time series
    if "p_set" in proxy_net.loads_t:
        for col in proxy_net.loads_t["p_set"].columns:
            if col in n.loads.index:
                ts = proxy_net.loads_t["p_set"][col]
                ts_reindexed = ts.reindex(snapshots, method="ffill")
                n.loads_t["p_set"][col] = ts_reindexed.values

# ---------------------------------------------------------------------------
# Add direct interconnector links for each corridor (no border bus)
# Architecture per corridor:
#   ES_host --(export)--> proxy_bus   capacity = p_exp (ES exports to neighbour)
#   ES_host <--(import)-- proxy_bus   capacity = p_imp (ES imports from neighbour)
#
# Both links connect the same two buses in opposite directions, so only
# one can flow at a time — no circular flow possible. The NTC limits are
# enforced by p_nom on each link. Expansion is OFF.
# ---------------------------------------------------------------------------
print("Adding direct interconnector links …")

for cid, info in CORRIDORS.items():
    proxy_bus = info["proxy_bus"]
    es_host   = closest_es_ac_node(n, info["border_lon"], info["border_lat"])

    print(f"  {cid}: ES host = {es_host}, proxy = {proxy_bus}")

    # --- Export link: ES_host → proxy_bus (Spain exports to neighbour) ---
    n.add(
        "Link", f"{cid} export",
        bus0=es_host, bus1=proxy_bus,
        carrier="DC_ic export",
        p_nom=info["p_exp"],
        p_nom_extendable=False,
        efficiency=1.0,
        length=info["length_km"],
        lifetime=50,
        underwater_fraction=0.0,
    )

    # --- Import link: proxy_bus → ES_host (Spain imports from neighbour) ---
    n.add(
        "Link", f"{cid} import",
        bus0=proxy_bus, bus1=es_host,
        carrier="DC_ic import",
        p_nom=info["p_imp"],
        p_nom_extendable=False,
        efficiency=1.0,
        length=info["length_km"],
        lifetime=50,
        underwater_fraction=0.0,
    )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
print(f"Exporting to {OUT_PATH} …")
n.export_to_netcdf(OUT_PATH)
print("Done.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("MERGED NETWORK SUMMARY")
print("=" * 70)
print(f"  Buses           : {len(n.buses)}")
print(f"  AC buses        : {(n.buses.carrier == 'AC').sum()}")
print(f"  Lines           : {len(n.lines)}")
print(f"  Links           : {len(n.links)}")
print(f"  Generators      : {len(n.generators)}")
print(f"  Loads           : {len(n.loads)}")
print(f"  Storage units   : {len(n.storage_units)}")
print(f"  Snapshots       : {len(n.snapshots)}")
print()
print("Interconnector links added:")
ic_links = n.links[n.links.carrier.str.contains("DC_ic")]
print(ic_links[["bus0", "bus1", "carrier", "p_nom"]].to_string())
print()
print("Proxy generators carried over:")
proxy_gens = n.generators[n.generators.index.str.contains("PT_|FR_")]
print(proxy_gens[["bus", "carrier", "p_nom"]].to_string())
print()
print("Proxy loads carried over:")
proxy_loads = n.loads[n.loads.index.str.contains("PT_|FR_")]
print(proxy_loads[["bus", "carrier"]].to_string())
print()
print("Proxy generators with p_max_pu time series:")
for col in n.generators_t["p_max_pu"].columns:
    if "PT_" in col or "FR_" in col:
        print(f"  {col}")
print()
print("Proxy loads with p_set time series:")
for col in n.loads_t["p_set"].columns:
    if "PT_" in col or "FR_" in col:
        print(f"  {col}")
