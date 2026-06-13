"""
Create simplified Portugal (3-bus) and France (2-bus) proxy networks.

Node placement: geographically inside each country at major grid hubs.
Line capacities: calibrated from base_s.nc pre-cluster topology (2024 OSM).
ES interconnectors: NOT included — add manually when merging with high-res ES network.

Outputs:
  resources/networks/pt_proxy.nc
  resources/networks/fr_proxy.nc
  analysis/proxy_networks_map.png
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pypsa

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helper: geographic distance (km) and 380kV line impedance (pu, 100 MVA base)
# ---------------------------------------------------------------------------
V_NOM = 380.0     # kV
S_BASE = 100.0    # MVA
Z_BASE = V_NOM**2 / S_BASE  # = 1444 Ω

# Typical bundled 380kV line parameters per km (single circuit)
X_OHM_PER_KM = 0.37
R_OHM_PER_KM = 0.060


def haversine_km(lon0, lat0, lon1, lat1):
    R = 6371.0
    d_lat = np.radians(lat1 - lat0)
    d_lon = np.radians(lon1 - lon0)
    a = np.sin(d_lat / 2)**2 + np.cos(np.radians(lat0)) * np.cos(np.radians(lat1)) * np.sin(d_lon / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def line_params(lon0, lat0, lon1, lat1, n_parallel):
    """Return (x_pu, r_pu, length_km) for n_parallel 380kV circuits."""
    d = haversine_km(lon0, lat0, lon1, lat1)
    x = (X_OHM_PER_KM * d) / Z_BASE / n_parallel
    r = (R_OHM_PER_KM * d) / Z_BASE / n_parallel
    return round(x, 5), round(r, 6), round(d, 1)


# ---------------------------------------------------------------------------
# PORTUGAL PROXY NETWORK — 3 buses
# ---------------------------------------------------------------------------
# Node positions chosen at major Portuguese grid hubs (inside Portugal)
PT_BUSES = {
    "PT_NORTH": {
        "x": -8.40, "y": 41.10,
        "desc": "Porto / Recarei area — N. Portugal hub",
        "es_corridor": "ES-PT NORTH",
        "es_cap_MW": 3500,     # calibrated NTC for NORTH corridor (OSM thermal ~10 GW, NTC ~3.5 GW)
        "n_es_lines": 3,       # 3 representative 380kV circuits to ES
    },
    "PT_CENTRE": {
        "x": -9.00, "y": 38.70,
        "desc": "Setúbal / Lisbon area — central Portugal hub",
        "es_corridor": "ES-PT CENTRE",
        "es_cap_MW": 3575,     # 2 × 1787 MW
        "n_es_lines": 2,
    },
    "PT_SOUTH": {
        "x": -8.00, "y": 37.60,
        "desc": "Ferreira do Alentejo / Beja area — S. Portugal hub",
        "es_corridor": "ES-PT SOUTH",
        "es_cap_MW": 1787,     # 1 × 1787 MW
        "n_es_lines": 1,
    },
}

# Internal lines (Portuguese 400kV backbone, N–S chain)
# Calibrated from base_s.nc:
#   N–C: 11 lines summing to 13,987 MW — represent as 2 main 400kV circuits (3,575 MW)
#   C–S: 3 lines at 1,787 MW each — represent as 2 main 400kV circuits (3,575 MW)
PT_LINES = [
    {
        "name": "PT_NC",
        "bus0": "PT_NORTH",
        "bus1": "PT_CENTRE",
        "s_nom_MW": 3575,
        "n_parallel": 2,
        "desc": "Porto–Lisbon 400kV backbone (2 circuits)",
    },
    {
        "name": "PT_CS",
        "bus0": "PT_CENTRE",
        "bus1": "PT_SOUTH",
        "s_nom_MW": 3575,
        "n_parallel": 2,
        "desc": "Lisbon–Alentejo 400kV backbone (2 circuits)",
    },
]

n_pt = pypsa.Network()
n_pt.name = "Portugal proxy network (3-bus)"
n_pt.set_snapshots(["2024-01-01 00:00"])

for bus_id, attrs in PT_BUSES.items():
    n_pt.add(
        "Bus", bus_id,
        x=attrs["x"], y=attrs["y"],
        v_nom=V_NOM,
        country="PT",
        carrier="AC",
    )

for line in PT_LINES:
    b0, b1 = line["bus0"], line["bus1"]
    x_pu, r_pu, length = line_params(
        PT_BUSES[b0]["x"], PT_BUSES[b0]["y"],
        PT_BUSES[b1]["x"], PT_BUSES[b1]["y"],
        line["n_parallel"],
    )
    n_pt.add(
        "Line", line["name"],
        bus0=b0, bus1=b1,
        s_nom=line["s_nom_MW"],
        x=x_pu, r=r_pu,
        v_nom=V_NOM,
        num_parallel=line["n_parallel"],
        length=length,
        type="",
    )

pt_path = "../resources/networks/pt_proxy.nc"
n_pt.export_to_netcdf(pt_path)
print(f"Saved: {pt_path}")

# ---------------------------------------------------------------------------
# FRANCE PROXY NETWORK — 2 buses
# ---------------------------------------------------------------------------
# Two nodes: one per Pyrenean corridor (West = Aquitaine, East = Languedoc)
FR_BUSES = {
    "FR_WEST": {
        "x": -0.60, "y": 44.80,
        "desc": "Bordeaux / Aquitaine area — western Pyrenees corridor hub",
        "es_corridor": "ES-FR WEST",
        "es_cap_MW": 2793,     # 3 AC lines (503+503+1787 MW) — existing
        "n_es_lines": 3,       # 3 AC circuits to ES
        "es_dc_cap_MW": 0,     # 0 MW (TYNDP2024_16 not yet built)
    },
    "FR_EAST": {
        "x": 3.80, "y": 43.70,
        "desc": "Montpellier / Languedoc area — eastern Pyrenees corridor hub",
        "es_corridor": "ES-FR EAST",
        "es_cap_MW": 1787,     # 1 AC line
        "n_es_lines": 1,
        "es_dc_cap_MW": 2000,  # 2 × 1000 MW DC (Santa Llogaia–Baixas)
        "n_es_dc_links": 2,
    },
}

# Internal line: FR_WEST ↔ FR_EAST — southern French 400kV arc
# base_s.nc shows 7 cross-lines totalling 8,658 MW between SW and SE France.
# Represent as 3 main 400kV circuits = 3 × 1787 = 5,362 MW
FR_LINES = [
    {
        "name": "FR_WE",
        "bus0": "FR_WEST",
        "bus1": "FR_EAST",
        "s_nom_MW": 5362,
        "n_parallel": 3,
        "desc": "Southern France 400kV arc Aquitaine–Languedoc (3 circuits)",
    },
]

n_fr = pypsa.Network()
n_fr.name = "France proxy network (2-bus)"
n_fr.set_snapshots(["2024-01-01 00:00"])

for bus_id, attrs in FR_BUSES.items():
    n_fr.add(
        "Bus", bus_id,
        x=attrs["x"], y=attrs["y"],
        v_nom=V_NOM,
        country="FR",
        carrier="AC",
    )

for line in FR_LINES:
    b0, b1 = line["bus0"], line["bus1"]
    x_pu, r_pu, length = line_params(
        FR_BUSES[b0]["x"], FR_BUSES[b0]["y"],
        FR_BUSES[b1]["x"], FR_BUSES[b1]["y"],
        line["n_parallel"],
    )
    n_fr.add(
        "Line", line["name"],
        bus0=b0, bus1=b1,
        s_nom=line["s_nom_MW"],
        x=x_pu, r=r_pu,
        v_nom=V_NOM,
        num_parallel=line["n_parallel"],
        length=length,
        type="",
    )

fr_path = "../resources/networks/fr_proxy.nc"
n_fr.export_to_netcdf(fr_path)
print(f"Saved: {fr_path}")

# ---------------------------------------------------------------------------
# Summary printout
# ---------------------------------------------------------------------------
print()
print("=" * 68)
print("PORTUGAL PROXY NETWORK")
print("=" * 68)
print(f"  {'Bus':<12} {'Lon':>6} {'Lat':>6}  Description")
for bus_id, a in PT_BUSES.items():
    print(f"  {bus_id:<12} {a['x']:>6.2f} {a['y']:>6.2f}  {a['desc']}")
print()
print(f"  {'Line':<8} {'From':<12} {'To':<12} {'s_nom':>8}  {'x_pu':>8}  km")
for line in PT_LINES:
    b0, b1 = line["bus0"], line["bus1"]
    x_pu, _, d = line_params(PT_BUSES[b0]["x"], PT_BUSES[b0]["y"],
                              PT_BUSES[b1]["x"], PT_BUSES[b1]["y"],
                              line["n_parallel"])
    print(f"  {line['name']:<8} {b0:<12} {b1:<12} {line['s_nom_MW']:>7} MW  {x_pu:>8.5f}  {d:.0f} km")
print()
print("  ES interconnector stubs (to be added at merge):")
for bus_id, a in PT_BUSES.items():
    print(f"    {bus_id} ← ES  {a['es_cap_MW']:>5} MW  ({a['n_es_lines']} AC circuits)  [{a['es_corridor']}]")

print()
print("=" * 68)
print("FRANCE PROXY NETWORK")
print("=" * 68)
print(f"  {'Bus':<10} {'Lon':>6} {'Lat':>6}  Description")
for bus_id, a in FR_BUSES.items():
    print(f"  {bus_id:<10} {a['x']:>6.2f} {a['y']:>6.2f}  {a['desc']}")
print()
print(f"  {'Line':<7} {'From':<10} {'To':<10} {'s_nom':>8}  {'x_pu':>8}  km")
for line in FR_LINES:
    b0, b1 = line["bus0"], line["bus1"]
    x_pu, _, d = line_params(FR_BUSES[b0]["x"], FR_BUSES[b0]["y"],
                              FR_BUSES[b1]["x"], FR_BUSES[b1]["y"],
                              line["n_parallel"])
    print(f"  {line['name']:<7} {b0:<10} {b1:<10} {line['s_nom_MW']:>7} MW  {x_pu:>8.5f}  {d:.0f} km")
print()
print("  ES interconnector stubs (to be added at merge):")
for bus_id, a in FR_BUSES.items():
    ac = a.get("es_cap_MW", 0)
    dc = a.get("es_dc_cap_MW", 0)
    n_ac = a.get("n_es_lines", 0)
    n_dc = a.get("n_es_dc_links", 0)
    stub = f"{ac} MW AC ({n_ac} circuits)"
    if dc > 0:
        stub += f" + {dc} MW DC ({n_dc} links)"
    print(f"    {bus_id} ← ES  {stub}  [{a['es_corridor']}]")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
proj = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(15, 11), subplot_kw={"projection": proj})
ax.set_extent([-11, 8, 35, 48], crs=proj)
ax.add_feature(cfeature.LAND, facecolor="#f5f5f0", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="#d6eaf8", zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=1.0, edgecolor="#666666", zorder=1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7, zorder=1)
ax.gridlines(draw_labels=True, linewidth=0.3, color="grey", alpha=0.5, linestyle="--")

COLOURS = {"PT": "#1f78b4", "FR": "#e31a1c"}

# Draw lines
def draw_line(ax, x0, y0, x1, y1, colour, lw, label=None):
    ax.plot([x0, x1], [y0, y1], color=colour, lw=lw, zorder=3,
            transform=proj, solid_capstyle="round")
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if label:
        ax.text(mx + 0.2, my + 0.15, label, fontsize=7.5, color=colour,
                transform=proj, zorder=6, ha="left")

for line in PT_LINES:
    b0, b1 = line["bus0"], line["bus1"]
    _, _, d = line_params(PT_BUSES[b0]["x"], PT_BUSES[b0]["y"],
                          PT_BUSES[b1]["x"], PT_BUSES[b1]["y"],
                          line["n_parallel"])
    draw_line(ax,
              PT_BUSES[b0]["x"], PT_BUSES[b0]["y"],
              PT_BUSES[b1]["x"], PT_BUSES[b1]["y"],
              COLOURS["PT"], lw=line["s_nom_MW"] / 800,
              label=f'{line["s_nom_MW"]:,} MW  {d:.0f} km')

for line in FR_LINES:
    b0, b1 = line["bus0"], line["bus1"]
    _, _, d = line_params(FR_BUSES[b0]["x"], FR_BUSES[b0]["y"],
                          FR_BUSES[b1]["x"], FR_BUSES[b1]["y"],
                          line["n_parallel"])
    draw_line(ax,
              FR_BUSES[b0]["x"], FR_BUSES[b0]["y"],
              FR_BUSES[b1]["x"], FR_BUSES[b1]["y"],
              COLOURS["FR"], lw=line["s_nom_MW"] / 800,
              label=f'{line["s_nom_MW"]:,} MW  {d:.0f} km')

# ES-PT stub arrows (dashed, showing where ES lines will connect)
ES_STUBS = {
    "PT_NORTH":  (-6.8, 42.3),
    "PT_CENTRE": (-6.7, 38.3),
    "PT_SOUTH":  (-7.2, 37.5),
    "FR_WEST":   (-1.9, 43.2),
    "FR_EAST":   (2.9,  42.2),
}
for bus_id, es_xy in ES_STUBS.items():
    if bus_id in PT_BUSES:
        bx, by = PT_BUSES[bus_id]["x"], PT_BUSES[bus_id]["y"]
        cap = PT_BUSES[bus_id]["es_cap_MW"]
        col = COLOURS["PT"]
    else:
        bx, by = FR_BUSES[bus_id]["x"], FR_BUSES[bus_id]["y"]
        cap = FR_BUSES[bus_id]["es_cap_MW"] + FR_BUSES[bus_id].get("es_dc_cap_MW", 0)
        col = COLOURS["FR"]
    ax.annotate("", xy=(bx, by), xytext=es_xy,
                xycoords=proj._as_mpl_transform(ax),
                textcoords=proj._as_mpl_transform(ax),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.5,
                                linestyle="dashed", mutation_scale=14),
                zorder=5)
    ax.text((bx + es_xy[0]) / 2 + 0.1, (by + es_xy[1]) / 2,
            f"{cap:,} MW\n(stub→ES)", fontsize=6.5, color=col,
            transform=proj, zorder=6, style="italic")

# Draw buses
for bus_id, a in PT_BUSES.items():
    ax.scatter(a["x"], a["y"], s=160, color=COLOURS["PT"], zorder=7,
               edgecolors="white", linewidths=1.0, transform=proj, marker="o")
    ax.annotate(f'{bus_id}\n({a["x"]:.1f}, {a["y"]:.1f})',
                xy=(a["x"], a["y"]),
                xytext=(a["x"] - 1.8, a["y"] + 0.3),
                fontsize=8, fontweight="bold", color=COLOURS["PT"],
                arrowprops=dict(arrowstyle="-", color=COLOURS["PT"], lw=0.7),
                transform=proj, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=COLOURS["PT"], alpha=0.9, lw=0.8))

for bus_id, a in FR_BUSES.items():
    ax.scatter(a["x"], a["y"], s=160, color=COLOURS["FR"], zorder=7,
               edgecolors="white", linewidths=1.0, transform=proj, marker="s")
    offset = (-2.0, 0.3) if "WEST" in bus_id else (0.3, 0.3)
    ax.annotate(f'{bus_id}\n({a["x"]:.1f}, {a["y"]:.1f})',
                xy=(a["x"], a["y"]),
                xytext=(a["x"] + offset[0], a["y"] + offset[1]),
                fontsize=8, fontweight="bold", color=COLOURS["FR"],
                arrowprops=dict(arrowstyle="-", color=COLOURS["FR"], lw=0.7),
                transform=proj, zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec=COLOURS["FR"], alpha=0.9, lw=0.8))

patches = [
    mpatches.Patch(color=COLOURS["PT"], label="Portugal proxy nodes / lines (●)"),
    mpatches.Patch(color=COLOURS["FR"], label="France proxy nodes / lines (■)"),
    plt.Line2D([0], [0], color="grey", lw=1.5, linestyle="--",
               label="ES interconnector stubs (not yet added)"),
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.95)
ax.set_title(
    "PT & FR proxy networks — 5 nodes, 3 internal lines\n"
    "(ES interconnectors omitted — add at merge)",
    fontsize=12, pad=10,
)

map_path = "proxy_networks_map.png"
fig.savefig(map_path, dpi=160, bbox_inches="tight")
print(f"\nSaved map: analysis/{map_path}")
