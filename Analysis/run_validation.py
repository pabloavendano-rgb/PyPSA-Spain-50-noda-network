"""
Validation suite for the 50-node Spain non-linear merit order model.

Usage (from repo root):
    pixi run python Analysis/run_validation.py

Control the analysis window in config.py → MODEL_CONFIG["validation"]:
    "start_date":  "2024-01-22"   ← any date in 2024
    "n_days":      7              ← number of days to solve

Outputs written to Analysis/validation_output/:
  01_week_overview.png          — Spain dispatch stack + price vs OMIE (first week)
  02_price_duration_curve.png   — Price duration curve, model vs OMIE
  03_capacity_vs_reality.png    — Installed capacity: model vs 2024 REE data
  04_network_map.png            — ES / PT / FR network with interconnectors
  05_curtailment_map.png        — Spain VRE curtailment intensity by node
  06_node_dispatch_pies.png     — Spain generation mix pie at each node
  07_hourly_dispatch.png        — Full hourly dispatch for ES / FR / PT
  08_temporal_comparison.png    — Model vs real dispatch (daily / weekly / monthly)
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib
if sys.platform in ("darwin", "win32") or os.environ.get("DISPLAY"):
    pass
else:
    matplotlib.use("Agg")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG
from refinery import apply_non_linear_refinements, apply_inflow_based_hydro_mc

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─── Global style — Cyber-Industrial ─────────────────────────────────────────
# Deep Slate text on Muted Ice backgrounds; Electric Teal / Burnt Coral accents.
# Inspired by FT / Economist data-journalism aesthetic.

_SLATE   = "#1E252B"   # text, axes, titles
_ICE     = "#F4F6F8"   # axes background (premium matte)
_GRID_C  = "#DDE2E6"   # grid lines

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":            10,
    "text.color":           _SLATE,
    "axes.titlesize":       13,
    "axes.titleweight":     "bold",
    "axes.labelsize":       10,
    "axes.labelcolor":      _SLATE,
    "axes.edgecolor":       _GRID_C,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            True,
    "grid.alpha":           0.45,
    "grid.linewidth":       0.5,
    "grid.linestyle":       "--",
    "grid.color":           _GRID_C,
    "figure.facecolor":     "white",
    "axes.facecolor":       _ICE,
    "lines.linewidth":      1.6,
    "legend.framealpha":    0.93,
    "legend.fontsize":      8,
    "legend.edgecolor":     _GRID_C,
    "xtick.color":          _SLATE,
    "ytick.color":          _SLATE,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
})

_SAVE_DPI = 250

# ─── Carrier colours & ordering ───────────────────────────────────────────────

COLORS = {
    # Baseload / structural (30% — Nordic Blue family)
    "nuclear":        "#457B9D",   # Nordic Blue — calm, institutional
    "coal":           "#4A5568",   # dark slate — muted, receding
    "biomass":        "#8B7355",   # warm bark brown

    # Renewables (graduated teal-green spectrum)
    "hydro":          "#00A896",   # Electric Teal — flexible, premium
    "PHS":            "#48CAE4",   # light sky teal — storage
    "ror":            "#0096C7",   # mid blue-teal — run-of-river
    "onwind":         "#52B788",   # sage green
    "offwind":        "#40916C",   # deeper forest green
    "offwind-float":  "#2D6A4F",   # darkest offshore green

    # Solar
    "solar":          "#F4A261",   # Solar Ochre — soft, warm, sophisticated

    # Gas dispatch — THE accent (thesis story: 10% pop)
    "CCGT":           "#FF6B6B",   # Burnt Coral — primary gas accent
    "CCGT_must_run":  "#FFB347",   # warm amber — industrial gas must-run base
    "CCGT_flex":      "#D64045",   # deeper coral-red — flex/peaker split
    "OCGT":           "#E63946",   # vivid red — open-cycle peakers
    "diesel":         "#9D0208",   # deep crimson — diesel backup
    "oil":            "#6D4C41",   # dark brown — residual oil

    # Alert
    "load_shedding":  "#FF006E",   # hot pink — unmissable VOLL signal
    "other":          "#B0B7C3",   # cool mid-gray
}
CARRIER_ORDER = list(COLORS.keys())

COUNTRY_COLORS = {"ES": "#E63946", "FR": "#457B9D", "PT": "#00A896"}

# Buses to exclude from all ES averages, price stats, and plots.
# ES0 1 is excluded because it has anomalous nodal prices that skew
# load-weighted averages and spatial visualisations.
_EXCLUDE_ES_BUSES: set[str] = {"ES0 1"}

# Spanish → model carrier mapping for daily_gen_spain.csv
_ES_MONTH = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "oct": 10, "nov": 11, "dic": 12,
}
_TECH_TO_CARRIER = {
    "Hidráulica":              "hydro",
    "Hidroeólica":             "hydro",
    "Nuclear":                 "nuclear",
    "Eólica":                  "onwind",
    "Solar fotovoltaica":      "solar",
    "Solar térmica":           "solar",
    "Carbón":                  "coal",
    "Cogeneración":            "biomass",
    "Residuos renovables":     "biomass",
    "Residuos no renovables":  "biomass",
    "Otras renovables":        "biomass",
    "Ciclo combinado":         "CCGT",
    "Motores diésel":          "diesel",
    "Turbina de gas":          "OCGT",    # singular in CSV (was "Turbinas de gas")
    "Turbinas de gas":         "OCGT",    # keep plural form as fallback
    "Turbina de vapor":        "coal",    # singular in CSV (was "Turbinas de vapor")
    "Turbinas de vapor":       "coal",    # keep plural form as fallback
    "Fuel + Gas":              "OCGT",    # legacy fuel+gas turbines
    "Generación no gestionable": "other",
    # "Generación total" is the row-sum sentinel — excluded in _load_real_dispatch
}

_GROUPS = {
    "Nuclear":   ["nuclear"],
    "Hydro":     ["hydro", "ror"],
    "PHS":       ["PHS"],
    "Wind":      ["onwind", "offwind", "offwind-float"],
    "Solar":     ["solar"],
    "CCGT":      ["CCGT", "CCGT_flex"],
    "Gas/Oil":   ["OCGT", "diesel"],
    "Coal/Bio":  ["coal", "biomass", "oil", "other"],
    "Shed":      ["load_shedding"],
}
_GROUP_COLORS = {
    "Nuclear":  "#3B4CC0", "Hydro": "#1E8BC3", "PHS": "#7EC8E3",
    "Wind":     "#2ECC71", "Solar": "#F1C40F",
    "CCGT":     "#E67E22", "Gas/Oil": "#C0392B", "Coal/Bio": "#7F8C8D",
    "Shed":     "#FF00FF",
}


# ─── Cartopy basemap helper ───────────────────────────────────────────────────

def _setup_cartopy_ax(ax, extent=(-10.8, 5.5, 35.0, 44.8)):
    """Apply a consistent Natural Earth basemap to a Cartopy PlateCarree axes.

    Returns the gridlines object so the caller can customise label visibility.
    """
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("10m"),
                   facecolor="#cce5f5", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"),
                   facecolor="#f5ede0", zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale("10m"),
                   facecolor="#cce5f5", zorder=2)
    ax.add_feature(cfeature.RIVERS.with_scale("10m"),
                   edgecolor="#9bbdd4", linewidth=0.35, zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("10m"),
                   edgecolor="#7a93a8", linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"),
                   edgecolor="#8a9bb0", linewidth=0.55,
                   linestyle="--", zorder=3)
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="#aaaaaa",
        alpha=0.5, linestyle=":", crs=ccrs.PlateCarree(),
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8, "color": _SLATE}
    gl.ylabel_style = {"size": 8, "color": _SLATE}
    return gl


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _es_buses(n):
    return n.buses.index[
        n.buses.index.str.startswith("ES")
        & ~n.buses.index.str.contains("H2")
        & ~n.buses.index.str.contains("battery")
        & ~n.buses.index.isin(_EXCLUDE_ES_BUSES)
    ]


def _country_buses(n, prefix):
    return n.buses.index[n.buses.index.str.startswith(prefix)]


def _mean_es_price(n):
    """Load-weighted mean ES nodal price — approximates OMIE single national clearing price."""
    buses = [b for b in _es_buses(n) if b in n.buses_t.marginal_price.columns]
    if not buses:
        return pd.Series(dtype=float)

    weights = {}
    for bus in buses:
        bus_loads = n.loads.index[n.loads["bus"] == bus]
        if len(bus_loads) == 0:
            weights[bus] = 0.0
            continue
        total = 0.0
        for ld in bus_loads:
            if ld in n.loads_t.p_set.columns:
                total += float(n.loads_t.p_set[ld].mean())
            else:
                total += float(n.loads.loc[ld, "p_set"])
        weights[bus] = total

    w = pd.Series(weights, dtype=float)[buses]
    if w.sum() == 0:
        w = pd.Series(1.0, index=buses)
    w = w / w.sum()

    return (n.buses_t.marginal_price[buses] * w).sum(axis=1)


def _dispatch_by_carrier(n, bus_prefix):
    buses = _country_buses(n, bus_prefix)
    gen_mask = n.generators["bus"].isin(buses)
    gen_names = n.generators.index[gen_mask]
    avail_gen = [g for g in gen_names if g in n.generators_t.p.columns]
    if avail_gen:
        carriers = n.generators.loc[avail_gen, "carrier"]
        gen_d = n.generators_t.p[avail_gen].T.groupby(carriers).sum().T
    else:
        gen_d = pd.DataFrame(index=n.snapshots)

    su_mask = n.storage_units["bus"].isin(buses)
    su_names = n.storage_units.index[su_mask]
    p_dis = getattr(n.storage_units_t, "p_dispatch", pd.DataFrame())
    p_str = getattr(n.storage_units_t, "p_store",    pd.DataFrame())
    dis_avail = [s for s in su_names if s in p_dis.columns]
    if dis_avail:
        su_car = n.storage_units.loc[dis_avail, "carrier"]
        net = p_dis[dis_avail].copy()
        str_avail = [s for s in dis_avail if s in p_str.columns]
        if str_avail:
            net[str_avail] -= p_str[str_avail]
        su_d = net.T.groupby(su_car).sum().T
        gen_d = pd.concat([gen_d, su_d], axis=1)

    return gen_d


def _to_daily_gwh(hourly_df):
    return hourly_df.resample("D").sum() / 1000.0


def _to_weekly_gwh(hourly_df):
    return hourly_df.resample("W-MON", label="left", closed="left").sum() / 1000.0


def _to_monthly_gwh(hourly_df):
    return hourly_df.resample("ME").sum() / 1000.0


def _load_omie(cfg, snapshots):
    path = ROOT / cfg["validation"]["omie_csv"]
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["Datetime (UTC)"], format="%d/%m/%y %H:%M")
    price = df["Price (EUR/MWhe)"].reindex(snapshots, method="nearest")
    price.index = snapshots
    return price


def _load_real_prices(cfg, snapshots, country):
    """Load real market prices for a given country from the configured CSV.

    Parameters
    ----------
    cfg : dict
        MODEL_CONFIG dict (must have validation keys for each country).
    snapshots : pd.DatetimeIndex
        Target snapshots to reindex to.
    country : str
        One of "ES", "FR", "PT".

    Returns
    -------
    pd.Series
        Real price series indexed by snapshots, NaN where no match.
    """
    key_map = {"ES": "omie_csv", "FR": "france_prices_csv", "PT": "portugal_prices_csv"}
    csv_key = key_map.get(country)
    if csv_key is None:
        return pd.Series(float("nan"), index=snapshots)
    path = ROOT / cfg["validation"].get(csv_key, "")
    if not path.exists():
        log.warning("Real price CSV not found for %s: %s", country, path)
        return pd.Series(float("nan"), index=snapshots)
    df = pd.read_csv(path)
    df.index = pd.to_datetime(df["Datetime (UTC)"], format="%d/%m/%y %H:%M")
    price = df["Price (EUR/MWhe)"].reindex(snapshots, method="nearest")
    price.index = snapshots
    return price


def _load_real_dispatch(cfg, date_range):
    """Load REE hourly actual generation (spain_actual_generation_2024.csv).

    CSV format: timestamp index (UTC), columns = English carrier names (MW).
    Returns daily GWh DataFrame indexed by date, columns = model carrier names.
    """
    path = ROOT / cfg["validation"]["real_dispatch_csv"]
    df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Filter to date range before resampling
    start = date_range[0].normalize()
    end   = date_range[-1].normalize()
    df = df.loc[(df.index >= start) & (df.index < end + pd.Timedelta(days=1))]

    # Map columns to model carriers using existing _REE_TO_CARRIER mapping
    carrier_map = {col: _REE_TO_CARRIER.get(col, "other") for col in df.columns}
    mapped = df.rename(columns=carrier_map)
    grouped = mapped.T.groupby(level=0).sum().T  # merge any duplicate carrier columns

    # Resample hourly MW → daily GWh
    return grouped.resample("D").sum() / 1000.0


def _group_dispatch(carrier_df):
    grouped = {}
    for grp, carriers in _GROUPS.items():
        cols = [c for c in carriers if c in carrier_df.columns]
        grouped[grp] = carrier_df[cols].sum(axis=1) if cols else pd.Series(0.0, index=carrier_df.index)
    return pd.DataFrame(grouped)


# ─── Real capacity reference (REE data) ───────────────────────────────────────

_REAL_CAPACITY_MW = {
    "nuclear":        7117.0,
    "hydro":         13824.0,   # reservoir + ror only
    "PHS":            3272.0,   # pumped hydro storage
    "wind":          31452.0,
    "solar":         39321.0,   # REE end-2024: 39,321 MW solar PV (up from 34,021 MW)
    "solar_thermal":  2303.0,   # Solar Power Tracker: 53 plants, 2,303 MW CSP
    "CCGT":          24562.0,
    "coal":           1820.0,
    "OCGT":           1149.0,
    "diesel":          769.0,
    "batteries":        23.0,
}
_REAL_CAP_MAP = {
    "nuclear":       ["nuclear"],
    "hydro":         ["hydro", "ror"],
    "PHS":           ["PHS"],
    "wind":          ["onwind", "offwind", "offwind-float"],
    "solar":         ["solar"],
    "solar_thermal": ["csp"],
    "CCGT":          ["CCGT", "CCGT_flex"],
    "coal":          ["coal"],
    "OCGT":          ["OCGT"],
    "diesel":        ["diesel"],
}

# ─── France real 2024 capacity (RTE year-end 2024) ────────────────────────────
# Source: RTE 2024 Bilan Électrique — Filières de production
#   Nucléaire: 61.4 GW | Hydraulique: 25.7 GW | Solaire: 24.3 GW
#   Éolien terrestre: 22.9 GW | Éolien en mer: 1.5 GW
#   Gaz: 12.6 GW | Fioul: 3.0 GW | Charbon: 1.8 GW
#   Thermique renouvelable et déchets: 2.3 GW
_REAL_CAPACITY_MW_FR = {
    "nuclear":   61400.0,
    "hydro":    25700.0,
    "solar":    24300.0,
    "onwind":   22900.0,
    "offwind":   1500.0,
    "CCGT":     12600.0,
    "OCGT":      3000.0,   # fioul (oil) mapped to OCGT
    "coal":      1800.0,
    "biomass":   2300.0,   # renewable thermal & waste
}
_REAL_CAP_MAP_FR = {
    "nuclear":   ["nuclear"],
    "hydro":     ["hydro", "ror"],
    "solar":     ["solar"],
    "onwind":    ["onwind"],
    "offwind":   ["offwind", "offwind-float"],
    "CCGT":      ["CCGT", "CCGT_flex"],
    "OCGT":      ["OCGT"],
    "coal":      ["coal"],
    "biomass":   ["biomass"],
}

# ─── Portugal real 2024 capacity (REN year-end 2024) ──────────────────────────
_REAL_CAPACITY_MW_PT = {
    "hydro":     7600.0,
    "wind":      5700.0,
    "solar":     3600.0,
    "CCGT":      4200.0,
    "OCGT":       500.0,
    "coal":         0.0,
    "biomass":    700.0,
}
_REAL_CAP_MAP_PT = {
    "hydro":     ["hydro", "ror"],
    "wind":      ["onwind", "offwind", "offwind-float"],
    "solar":     ["solar"],
    "CCGT":      ["CCGT", "CCGT_flex"],
    "OCGT":      ["OCGT"],
    "coal":      ["coal"],
    "biomass":   ["biomass"],
}


def _model_capacity_es(n):
    """Return dict of carrier → total MW for ES buses."""
    es_set = set(_es_buses(n))
    cap = {}
    for name, row in n.generators.iterrows():
        if row.get("bus") not in es_set:
            continue
        c = row.get("carrier", "other")
        cap[c] = cap.get(c, 0.0) + float(row["p_nom"])
    for name, row in n.storage_units.iterrows():
        if row.get("bus") not in es_set:
            continue
        c = row.get("carrier", "other")
        cap[c] = cap.get(c, 0.0) + float(row["p_nom"])
    return cap


# ─── Console statistics ───────────────────────────────────────────────────────

_PDC_BINS = [
    (None,  1.0,  "≤ €1   (near-zero)"),
    (1.0,   50.0, "€1–50  (VRE/nuclear floor)"),
    (50.0,  80.0, "€50–80 (hydro/biomass)"),
    (80.0, 120.0, "€80–120 (CCGT base)"),
    (120.0,170.0, "€120–170 (CCGT_flex)"),
    (170.0,300.0, "€170–300 (peakers)"),
    (300.0, None, ">€300  (scarcity/VOLL)"),
]
_NEAR_ZERO_THRESHOLD = 1.0
_NEAR_ZERO_OMIE      = 0.5


# ─── French missing demand ─────────────────────────────────────────────────────

def _add_fr_missing_demand(n, cfg):
    """
    Add weighted non-Spain French export demand to FR load nodes.

    France exports to Italy, Germany, Belgium, UK, Switzerland — not just Spain.
    This function loads an hourly CSV of those missing export flows and distributes
    them across FR_WEST and FR_EAST proportional to each node's annual total demand.

    Returns the modified network (or unchanged if disabled / no FR loads found).
    """
    fd = cfg.get("fr_missing_demand", {})
    if not fd.get("enabled", True):
        log.info("FR missing demand: disabled — skipping")
        return n

    csv_path = ROOT / fd["csv_path"]
    column   = fd.get("column", "FR_net_export")

    if not csv_path.exists():
        log.warning("FR missing demand: %s not found — skipping", csv_path)
        return n

    # Identify French load nodes
    fr_loads = [l for l in n.loads.index if l.startswith("FR") and l in n.loads_t.p_set.columns]
    if not fr_loads:
        log.warning("FR missing demand: no FR loads with time series found — skipping")
        return n

    # Load CSV
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
    if column not in df.columns:
        log.warning("FR missing demand: column '%s' not found in %s — skipping", column, csv_path.name)
        return n

    missing_ts = df[column].copy()
    if missing_ts.index.tz is not None:
        missing_ts.index = missing_ts.index.tz_localize(None)

    # Reindex to network snapshots (handles 8784 vs 8760 leap-day discrepancy)
    missing_ts = missing_ts.reindex(n.snapshots, method="ffill").fillna(0.0)

    # Compute annual total demand per FR node for weighting
    annual_demand = {}
    for l in fr_loads:
        annual_demand[l] = n.loads_t.p_set[l].sum()

    total_fr_demand = sum(annual_demand.values())
    if total_fr_demand <= 0:
        log.warning("FR missing demand: total FR demand is zero — skipping")
        return n

    # Apply weighted demand addition
    total_added_mwh = 0.0
    for l in fr_loads:
        weight = annual_demand[l] / total_fr_demand
        addition = weight * missing_ts
        n.loads_t.p_set[l] += addition.values
        added_mwh = addition.sum() / 1e3  # MWh → GWh
        total_added_mwh += added_mwh
        log.info(
            "FR missing demand: %s gets %.1f%% (%.1f GWh added, weight=%.3f)",
            l, weight * 100, added_mwh, weight,
        )

    log.info(
        "FR missing demand: total %.1f GWh distributed across %d FR load(s)",
        total_added_mwh, len(fr_loads),
    )
    return n


def _curtailment_stats(n):
    vre_carriers = {"solar", "onwind", "offwind", "offwind-float"}
    es_set = set(_es_buses(n))
    total_pot, total_act = 0.0, 0.0
    node_pot, node_act = {}, {}

    for name, row in n.generators.iterrows():
        if row.get("carrier") not in vre_carriers or row.get("bus") not in es_set:
            continue
        if name not in n.generators_t.p.columns:
            continue
        p_nom = float(row["p_nom"])
        if name in n.generators_t.p_max_pu.columns:
            potential = n.generators_t.p_max_pu[name] * p_nom
        else:
            potential = pd.Series(float(row.get("p_max_pu", 1.0)) * p_nom, index=n.snapshots)
        actual = n.generators_t.p[name]
        pot_sum, act_sum = float(potential.sum()), float(actual.sum())
        bus = row["bus"]
        node_pot[bus] = node_pot.get(bus, 0.0) + pot_sum
        node_act[bus] = node_act.get(bus, 0.0) + act_sum
        total_pot += pot_sum
        total_act += act_sum

    curtailed = total_pot - total_act
    total_pct = curtailed / total_pot * 100.0 if total_pot > 0 else 0.0
    max_pct, max_node = 0.0, "n/a"
    for bus in node_pot:
        if node_pot[bus] > 0:
            pct = (node_pot[bus] - node_act[bus]) / node_pot[bus] * 100.0
            if pct > max_pct:
                max_pct, max_node = pct, bus
    return {
        "curtailed_gwh": curtailed / 1000.0,
        "potential_gwh": total_pot / 1000.0,
        "total_pct":     total_pct,
        "max_node":      max_node,
        "max_node_pct":  max_pct,
        "node_pot":      node_pot,
        "node_act":      node_act,
    }


def _pdc_bins(prices):
    n_total = len(prices.dropna())
    rows = []
    for lo, hi, label in _PDC_BINS:
        mask = pd.Series(True, index=prices.index)
        if lo is not None:
            mask &= prices >= lo
        if hi is not None:
            mask &= prices < hi
        count = int(mask.sum())
        pct = count / n_total * 100.0 if n_total > 0 else 0.0
        rows.append((label, count, pct))
    return rows


def _net_es_import_series(n, link_names):
    """Return hourly ES net import MW series for the given link names."""
    result = pd.Series(0.0, index=n.snapshots)
    found = 0
    for ln in link_names:
        if ln not in n.links.index or ln not in n.links_t.p0.columns:
            continue
        p0 = n.links_t.p0[ln]
        b0 = str(n.links.loc[ln, "bus0"]) if "bus0" in n.links.columns else ""
        b1 = str(n.links.loc[ln, "bus1"]) if "bus1" in n.links.columns else ""
        if b1.startswith("ES"):
            result += p0
        elif b0.startswith("ES"):
            result -= p0
        else:
            result += p0
        found += 1
    return result, found


def _net_import_topo(n, country: str) -> pd.Series:
    """ES net import from country (positive = ES imports), topology-agnostic.

    Works with old DC_ic link-pair topology and new AC Line + INELFE topology.
    """
    result = pd.Series(0.0, index=n.snapshots)
    kp0 = getattr(n.links_t, "p0", pd.DataFrame())
    if not kp0.empty and not n.links.empty:
        mask = (
            (n.links.bus0.str.startswith("ES") & n.links.bus1.str.startswith(country)) |
            (n.links.bus0.str.startswith(country) & n.links.bus1.str.startswith("ES"))
        )
        for ln, row in n.links[mask].iterrows():
            if ln not in kp0.columns:
                continue
            p0 = kp0[ln].reindex(n.snapshots, fill_value=0.0)
            if str(row.bus0).startswith("ES"):
                result -= p0
            else:
                result += p0
    lp0 = getattr(n.lines_t, "p0", pd.DataFrame())
    if not lp0.empty and not n.lines.empty:
        mask = (
            (n.lines.bus0.str.startswith("ES") & n.lines.bus1.str.startswith(country)) |
            (n.lines.bus0.str.startswith(country) & n.lines.bus1.str.startswith("ES"))
        )
        for ln, row in n.lines[mask].iterrows():
            if ln not in lp0.columns:
                continue
            p0 = lp0[ln].reindex(n.snapshots, fill_value=0.0)
            if str(row.bus0).startswith("ES"):
                result -= p0
            else:
                result += p0
    return result


def _print_stats(n, omie, model_daily_es, real_daily, start, n_days, cfg=None):
    model_price = _mean_es_price(n)
    aligned = pd.DataFrame({"model": model_price, "omie": omie}).dropna()
    n_hours = len(model_price)

    SEP = "─" * 66
    print(f"\n{SEP}")
    print("  VALIDATION STATISTICS")
    print(f"  Period : {start.date()} → {(start + pd.Timedelta(days=n_days - 1)).date()}"
          f"  ({n_days} day{'s' if n_days > 1 else ''}, {n_hours} hours)")
    print(SEP)

    if aligned.empty or aligned["model"].isna().all():
        print("\n  [WARN] No valid model prices — MIP pass 2 likely failed.")
        print("         Set mip.enabled=False or fix pass-2 LP to recover stats.")
        return

    err = aligned["model"] - aligned["omie"]

    print("\n  Price (Spain load-weighted LMP vs OMIE)  [EUR/MWh]")
    print(f"    {'':22}  {'Model':>8}  {'OMIE':>8}")
    print(f"    {'Mean':22}  {aligned['model'].mean():>8.2f}  {aligned['omie'].mean():>8.2f}")
    print(f"    {'Median':22}  {aligned['model'].median():>8.2f}  {aligned['omie'].median():>8.2f}")
    print(f"    {'p5':22}  {np.percentile(aligned['model'], 5):>8.1f}  {np.percentile(aligned['omie'], 5):>8.1f}")
    print(f"    {'p95':22}  {np.percentile(aligned['model'], 95):>8.1f}  {np.percentile(aligned['omie'], 95):>8.1f}")
    print(f"    {'Max':22}  {aligned['model'].max():>8.1f}  {aligned['omie'].max():>8.1f}")
    print(f"\n    {'Mean bias (model−OMIE)':22}  {err.mean():>+8.2f}")
    print(f"    {'MAE':22}  {err.abs().mean():>8.2f}")
    print(f"    {'RMSE':22}  {np.sqrt((err**2).mean()):>8.2f}")
    print(f"    {'Correlation':22}  {aligned['model'].corr(aligned['omie']):>8.3f}")

    _es_price_cols = [b for b in _es_buses(n) if b in n.buses_t.marginal_price.columns]
    if len(_es_price_cols) > 1:
        _nodal = n.buses_t.marginal_price[_es_price_cols]
        _spread = (_nodal.max(axis=1) - _nodal.min(axis=1)).mean()
        _sigma  = _nodal.std(axis=1).mean()
        print(f"    {'Nodal spread (avg max−min)':22}  {_spread:>8.1f}  (σ={_sigma:.1f} €/MWh — congestion proxy)")

    nz_model = int((aligned["model"] <= _NEAR_ZERO_THRESHOLD).sum())
    nz_omie  = int((aligned["omie"]  <= _NEAR_ZERO_OMIE).sum())
    print(f"\n  Near-zero price hours  (model ≤ €{_NEAR_ZERO_THRESHOLD:.0f},  OMIE ≤ €{_NEAR_ZERO_OMIE:.1f})")
    print(f"    Model : {nz_model:>4}  ({nz_model / n_hours * 100:.1f}% of hours)")
    print(f"    OMIE  : {nz_omie:>4}  ({nz_omie  / n_hours * 100:.1f}% of hours)")

    print(f"\n  Price Frequency Distribution  [hours / % of {n_hours}h]")
    print(f"  {'Band':<26}  {'Model h':>7}  {'Model%':>7}  {'OMIE h':>7}  {'OMIE%':>7}")
    print(f"  {'-'*56}")
    for (label, m_cnt, m_pct), (_, o_cnt, o_pct) in zip(
        _pdc_bins(aligned["model"]), _pdc_bins(aligned["omie"])
    ):
        print(f"  {label:<26}  {m_cnt:>7d}  {m_pct:>6.1f}%  {o_cnt:>7d}  {o_pct:>6.1f}%")

    curt = _curtailment_stats(n)
    print(f"\n  VRE Curtailment (Spain)")
    print(f"    Potential generation : {curt['potential_gwh']:>8.1f} GWh")
    print(f"    Curtailed            : {curt['curtailed_gwh']:>8.1f} GWh  ({curt['total_pct']:.1f}% of potential)")
    print(f"    Worst node           : {curt['max_node']}  ({curt['max_node_pct']:.1f}%)")

    if "load_shedding" in model_daily_es.columns:
        shed_gwh = float(model_daily_es["load_shedding"].sum())
        if shed_gwh > 0.001:
            print(f"\n  *** LOAD SHEDDING: {shed_gwh:.3f} GWh — check feasibility ***")
    else:
        print(f"\n  Load shedding: not modelled (VOLL disabled)")

    print(f"\n  Dispatch over period (Spain)  [GWh]")
    model_g = _group_dispatch(model_daily_es)
    real_g  = _group_dispatch(real_daily) if real_daily is not None else None
    print(f"  {'Group':<12}  {'Model':>8}  {'Real':>8}  {'Diff':>8}  {'%Δ':>7}")
    print(f"  {'-'*50}")
    for grp in _GROUPS:
        m = float(model_g[grp].sum()) if grp in model_g else 0.0
        r = float(real_g[grp].sum()) if (real_g is not None and grp in real_g) else None
        if r is not None:
            diff = m - r
            pct  = (diff / r * 100) if r != 0 else float("nan")
            print(f"  {grp:<12}  {m:>8.1f}  {r:>8.1f}  {diff:>+8.1f}  {pct:>+6.1f}%")
        else:
            print(f"  {grp:<12}  {m:>8.1f}  {'n/a':>8}")

    print(f"\n  Capacity Comparison: Model vs 2024 Real (Spain)  [MW]")
    model_cap = _model_capacity_es(n)
    print(f"  {'Group':<14}  {'Model':>10}  {'Real 2024':>10}  {'Diff':>10}  {'%Δ':>7}")
    print(f"  {'-'*56}")
    for grp, real_mw in _REAL_CAPACITY_MW.items():
        carriers = _REAL_CAP_MAP.get(grp, [grp])
        model_mw = sum(model_cap.get(c, 0.0) for c in carriers)
        diff = model_mw - real_mw
        pct = (diff / real_mw * 100) if real_mw else float("nan")
        print(f"  {grp:<14}  {model_mw:>10.0f}  {real_mw:>10.0f}  {diff:>+10.0f}  {pct:>+6.1f}%")

    fr_net = _net_import_topo(n, "FR")
    pt_net = _net_import_topo(n, "PT")
    if not fr_net.eq(0).all() or not pt_net.eq(0).all():
        print(f"\n  Interconnector Flows (ES net import = positive)  [over {n_days}d period]")
        print(f"  {'Border':<12}  {'Net GWh':>9}  {'Avg MW':>8}  {'Peak Import':>12}  {'Peak Export':>12}  {'% as importer':>14}")
        print(f"  {'-'*73}")
        for label, net_mw in [("ES ↔ FR", fr_net), ("ES ↔ PT", pt_net)]:
            net_gwh  = float(net_mw.sum()) / 1000.0
            avg_mw   = float(net_mw.mean())
            peak_imp = float(net_mw.clip(lower=0).max())
            peak_exp = float(net_mw.clip(upper=0).min())
            pct_imp  = float((net_mw > 0).sum()) / len(net_mw) * 100.0
            print(f"  {label:<12}  {net_gwh:>+9.1f}  {avg_mw:>+8.0f}  {peak_imp:>12.0f}  {peak_exp:>12.0f}  {pct_imp:>13.1f}%")

    print(f"\n{SEP}\n")


def _price_setter_carrier(combined, col_carrier, price):
    """Vectorised: dispatching unit whose MC is closest to the actual bus price.

    Uses closest-MC rather than highest-MC because in an LP the true price-setter
    has MC ≈ bus_price. Highest-MC incorrectly flags ramp-constrained CCGT as the
    setter in low-price hours where VRE/nuclear is actually on the margin.
    """
    mc_arr    = combined.values                # (T, N)  — NaN where not dispatching
    price_arr = price.values[:, np.newaxis]    # (T, 1)
    diff_arr  = np.abs(mc_arr - price_arr)     # (T, N)  — NaN for non-dispatching units
    # Replace NaN with inf so argmin never sees an all-NaN row (np.nanargmin raises on those)
    safe_arr  = np.where(np.isnan(diff_arr), np.inf, diff_arr)
    all_nan   = np.all(~np.isfinite(safe_arr), axis=1)
    best_col  = np.argmin(safe_arr, axis=1)   # safe: argmin of all-inf returns 0
    units     = combined.columns.tolist()
    return pd.Series(
        [col_carrier.get(units[i], "none") if not all_nan[j] else "none"
         for j, i in enumerate(best_col)],
        index=combined.index,
    )


def _print_cost_and_price_setter_table(n):
    """Print MC distribution by technology × country, then price-setter frequency for Spain."""
    SEP = "─" * 66
    DISPATCH_THRESH = 1.0  # MW — generator must exceed this to count as dispatching

    gen = n.generators[["bus", "carrier", "marginal_cost", "p_nom"]].copy()
    gen["country"] = gen["bus"].str[:2]

    su = n.storage_units[["bus", "carrier", "marginal_cost", "p_nom"]].copy()
    su["mc_store"] = n.storage_units.get("marginal_cost_storage", 0.0)
    su["country"]  = su["bus"].str[:2]

    # ── 1. MC distribution ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  MARGINAL COSTS BY TECHNOLOGY × COUNTRY  [EUR/MWh]")
    print(SEP)
    print(f"  {'Cty':<4}  {'Carrier':<16}  {'Min':>7}  {'Mean':>7}  {'Max':>7}  {'N':>3}  {'Notes'}")
    print(f"  {'-'*62}")

    tv_mc = n.generators_t.marginal_cost

    for country in ["ES", "FR", "PT"]:
        rows = []
        for carrier, grp in gen[gen["country"] == country].groupby("carrier"):
            tv_cols = [g for g in grp.index if g in tv_mc.columns]
            if tv_cols:
                vals = tv_mc[tv_cols].values.ravel()
                lo, mean, hi = float(np.nanmin(vals)), float(np.nanmean(vals)), float(np.nanmax(vals))
                note = "tv-MC (daily gas)"
            else:
                mcs = grp["marginal_cost"]
                lo, mean, hi = mcs.min(), mcs.mean(), mcs.max()
                note = ""
            rows.append((mean, carrier, lo, mean, hi, len(grp), note))
        for carrier, grp in su[su["country"] == country].groupby("carrier"):
            mcs = grp["marginal_cost"]
            mc_s = grp["mc_store"].mean()
            note = f"store MC: {mc_s:.1f}" if mc_s > 0 else ""
            rows.append((mcs.mean(), carrier, mcs.min(), mcs.mean(), mcs.max(), len(grp), note))
        rows.sort()  # ascending mean MC
        for i, (_, carrier, lo, mean, hi, n_u, note) in enumerate(rows):
            cty = country if i == 0 else ""
            print(f"  {cty:<4}  {carrier:<16}  {lo:>7.1f}  {mean:>7.1f}  {hi:>7.1f}  {n_u:>3}  {note}")
        if rows:
            print()

    # ── 2. Price setter (Spain) ──────────────────────────────────────────────
    # Exclude must-run carriers (CCGT_must_run, biomass) from price-setter analysis.
    # Must-run units have artificially low MC (€2, €0) to force dispatch, but they
    # are NOT the marginal price setter — they run regardless of price. Including
    # them can falsely flag CCGT_must_run (MC=€2) as the setter in hours where VRE
    # (MC=€0.01) is truly marginal but the bus price lands between the two.
    MUST_RUN_CARRIERS = {"CCGT_must_run", "biomass"}
    es_gen = gen[(gen["country"] == "ES") & (~gen["carrier"].isin(MUST_RUN_CARRIERS))]
    es_su  = su[(su["country"] == "ES") & (~su["carrier"].isin(MUST_RUN_CARRIERS))]
    gen_t  = n.generators_t.p
    su_t   = n.storage_units_t.p_dispatch

    es_gen_cols = [g for g in es_gen.index if g in gen_t.columns]
    es_su_cols  = [s for s in es_su.index  if s in su_t.columns]

    frames = []
    if es_gen_cols:
        active = gen_t[es_gen_cols].where(gen_t[es_gen_cols] > DISPATCH_THRESH)
        mc_df  = _build_tv_mc_df(n, es_gen_cols)
        frames.append(active.where(active.isna(), mc_df))
    if es_su_cols:
        active_su = su_t[es_su_cols].where(su_t[es_su_cols] > DISPATCH_THRESH)
        mc_map_su = es_su.loc[es_su_cols, "marginal_cost"]
        frames.append(active_su.where(active_su.isna(), mc_map_su, axis=1))

    if not frames:
        return

    combined    = pd.concat(frames, axis=1)
    col_carrier = pd.concat([
        es_gen.loc[es_gen_cols, "carrier"],
        es_su.loc[es_su_cols,   "carrier"],
    ]).to_dict()
    es_price = _mean_es_price(n)
    if es_price.empty:
        log.warning("Price setter table: no marginal prices available (MIP pass-2 likely failed) — skipping setter section")
        return

    setter_carrier = _price_setter_carrier(combined, col_carrier, es_price)
    counts         = setter_carrier.value_counts()
    n_snap         = len(n.snapshots)

    print(f"  PRICE SETTER — Spain  (dispatching unit with MC closest to bus price per hour)")
    print(f"  Includes ES generators + reservoir hydro + PHS")
    print(f"  {'Carrier':<18}  {'Hours':>6}  {'% of sim':>9}")
    print(f"  {'-'*38}")
    for carrier, count in counts.items():
        bar_len = int(count / n_snap * 30)
        bar = "█" * bar_len
        print(f"  {carrier:<18}  {count:>6d}  {count/n_snap*100:>8.1f}%  {bar}")
    print(f"  {'─'*38}")
    print(f"  {'Total':<18}  {n_snap:>6d}  {'100.0%':>9}")
    print(f"\n{SEP}\n")


# ─── Plot 13: Cost structure + price setter ───────────────────────────────────

def _plot_cost_and_price_setter(n, out_dir):
    """Two-panel figure: MC table (left) + price-setter bar chart (right)."""
    DISPATCH_THRESH = 1.0  # MW

    gen = n.generators[["bus", "carrier", "marginal_cost", "p_nom"]].copy()
    gen["country"] = gen["bus"].str[:2]
    su = n.storage_units[["bus", "carrier", "marginal_cost", "p_nom"]].copy()
    su["mc_store"] = n.storage_units.get("marginal_cost_storage", 0.0)
    su["country"]  = su["bus"].str[:2]

    # ── Build MC table rows ──────────────────────────────────────────────────
    rows = []
    for country in ["ES", "FR", "PT"]:
        for carrier, grp in gen[gen["country"] == country].groupby("carrier"):
            mcs = grp["marginal_cost"]
            rows.append((country, carrier, mcs.min(), mcs.mean(), mcs.max(), len(mcs), ""))
        for carrier, grp in su[su["country"] == country].groupby("carrier"):
            mcs = grp["marginal_cost"]
            mc_s = grp["mc_store"].mean()
            note = f"store: {mc_s:.1f}" if mc_s > 0 else ""
            rows.append((country, f"{carrier} (su)", mcs.min(), mcs.mean(), mcs.max(), len(grp), note))
    rows.sort(key=lambda r: (r[0], r[3]))  # country then mean MC

    # ── Price setter ─────────────────────────────────────────────────────────
    es_gen = gen[gen["country"] == "ES"]
    es_su  = su[su["country"] == "ES"]
    gen_t  = n.generators_t.p
    su_t   = n.storage_units_t.p_dispatch

    es_gen_cols = [g for g in es_gen.index if g in gen_t.columns]
    es_su_cols  = [s for s in es_su.index  if s in su_t.columns]

    frames = []
    if es_gen_cols:
        active = gen_t[es_gen_cols].where(gen_t[es_gen_cols] > DISPATCH_THRESH)
        frames.append(active.where(active.isna(), es_gen.loc[es_gen_cols, "marginal_cost"], axis=1))
    if es_su_cols:
        active_su = su_t[es_su_cols].where(su_t[es_su_cols] > DISPATCH_THRESH)
        frames.append(active_su.where(active_su.isna(), es_su.loc[es_su_cols, "marginal_cost"], axis=1))

    combined    = pd.concat(frames, axis=1)
    col_carrier = pd.concat([
        es_gen.loc[es_gen_cols, "carrier"],
        es_su.loc[es_su_cols,   "carrier"],
    ]).to_dict()
    es_price = _mean_es_price(n)
    if es_price.empty:
        log.warning("Price setter plot: no marginal prices (MIP pass-2 failed) — skipping price-setter panel")
        setter_carrier = pd.Series(dtype=str)
        counts = pd.Series(dtype=int)
    else:
        setter_carrier = _price_setter_carrier(combined, col_carrier, es_price)
        counts = setter_carrier.value_counts()
    n_snap  = len(n.snapshots)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, max(7, len(rows) * 0.38 + 2)))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.6, 1], wspace=0.35)
    ax_tbl = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    t0 = n.snapshots[0].strftime("%d %b")
    t1 = n.snapshots[-1].strftime("%d %b %Y")
    fig.suptitle(f"Model Cost Structure  [{t0}–{t1}]", fontsize=12, y=1.02)

    # ── Left: MC table ────────────────────────────────────────────────────────
    ax_tbl.axis("off")
    col_labels = ["Cty", "Carrier", "Min", "Mean", "Max", "N", "Notes"]
    cell_text  = [
        [r[0], r[1], f"{r[2]:.1f}", f"{r[3]:.1f}", f"{r[4]:.1f}", str(r[5]), r[6]]
        for r in rows
    ]
    tbl = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.auto_set_column_width(list(range(len(col_labels))))

    # Shade header and alternate rows
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f0f4f8")
        cell.set_edgecolor("#cccccc")

    ax_tbl.set_title("Marginal Costs by Technology × Country  [EUR/MWh]",
                     fontsize=10, pad=8, loc="left")

    # ── Right: price setter bar chart ─────────────────────────────────────────
    carriers = counts.index.tolist()
    pcts     = (counts.values / n_snap * 100).tolist()
    bar_colors = [COLORS.get(c, "#aaaaaa") for c in carriers]

    bars = ax_bar.barh(carriers[::-1], pcts[::-1], color=bar_colors[::-1],
                       alpha=0.85, height=0.55, edgecolor="white", linewidth=0.5)
    ax_bar.set_xlabel("% of simulation hours", fontsize=9)
    ax_bar.set_title("Price Setter Frequency\n(Spain — highest-MC dispatching unit)",
                     fontsize=10, pad=8)
    ax_bar.set_xlim(0, max(pcts) * 1.25 if pcts else 100)
    for bar, pct in zip(bars, pcts[::-1]):
        ax_bar.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{pct:.1f}%", va="center", ha="left", fontsize=8.5)
    ax_bar.grid(axis="x", alpha=0.25)
    ax_bar.spines[["top", "right"]].set_visible(False)

    path = out_dir / "13_cost_structure.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 14: Price setter analysis (ES / FR / PT) ────────────────────────────

def _build_tv_mc_df(n, gen_cols):
    """Return (snapshots × gen_cols) DataFrame of per-hour MCs.

    n.generators_t.marginal_cost holds MIBGAS-derived time-varying MCs written
    by refinery; falls back to static n.generators.marginal_cost for units that
    have fixed MCs (nuclear, hydro, load-shedding).
    """
    tv     = n.generators_t.marginal_cost
    static = n.generators["marginal_cost"]
    mc_df  = pd.DataFrame(index=n.snapshots, columns=gen_cols, dtype=float)
    for g in gen_cols:
        mc_df[g] = tv[g].values if g in tv.columns else static.loc[g]
    return mc_df


def _get_price_setter_series(n, country):
    """Return (mean_price, setter_carrier) Series for one country, one row per snapshot."""
    DISPATCH_THRESH = 1.0  # MW

    buses = _country_buses(n, country)
    price_cols = [b for b in buses if b in n.buses_t.marginal_price.columns]
    if not price_cols:
        return pd.Series(dtype=float), pd.Series(dtype=str)
    if country == "ES":
        price = _mean_es_price(n)
    else:
        price = n.buses_t.marginal_price[price_cols].mean(axis=1)

    gen = n.generators[n.generators["bus"].isin(buses)][["carrier", "marginal_cost"]]
    su  = n.storage_units[n.storage_units["bus"].isin(buses)][["carrier", "marginal_cost"]]

    gen_t = n.generators_t.p
    su_t  = n.storage_units_t.p_dispatch

    gen_cols = [g for g in gen.index if g in gen_t.columns]
    su_cols  = [s for s in su.index  if s in su_t.columns]

    frames, col_carrier = [], {}
    if gen_cols:
        active = gen_t[gen_cols].where(gen_t[gen_cols] > DISPATCH_THRESH)
        mc_df  = _build_tv_mc_df(n, gen_cols)
        frames.append(active.where(active.isna(), mc_df))
        col_carrier.update(gen.loc[gen_cols, "carrier"].to_dict())
    if su_cols:
        active_su = su_t[su_cols].where(su_t[su_cols] > DISPATCH_THRESH)
        frames.append(active_su.where(active_su.isna(), su.loc[su_cols, "marginal_cost"], axis=1))
        col_carrier.update(su.loc[su_cols, "carrier"].to_dict())

    if not frames:
        return price, pd.Series("none", index=n.snapshots)

    combined = pd.concat(frames, axis=1)
    setter   = _price_setter_carrier(combined, col_carrier, price)
    return price, setter


def _plot_price_setter_analysis(n, out_dir, real_prices=None):
    """3-row × 2-col figure: PDC coloured by price-setter + frequency bar, per country.

    Real-market PDC step lines (OMIE data) are overlaid on the LHS scatter plot
    for direct comparison.  Y-axis capped at €170/MWh.

    Parameters
    ----------
    n : pypsa.Network
        Solved network.
    out_dir : Path
        Output directory.
    real_prices : dict[str, pd.Series] or None
        Dict mapping country code → real price Series (indexed by snapshots).
        If None, only model PDC is shown.
    """
    countries = ["ES", "FR", "PT"]
    Y_MAX = 170.0  # cap y-axis at €170/MWh

    fig, axes = plt.subplots(
        3, 2, figsize=(14, 13),
        gridspec_kw={"width_ratios": [3, 1], "hspace": 0.40, "wspace": 0.30},
    )

    t0 = n.snapshots[0].strftime("%d %b")
    t1 = n.snapshots[-1].strftime("%d %b %Y")
    fig.suptitle(f"Price-Setter Analysis — ES / FR / PT  [{t0}–{t1}]",
                 fontsize=13, y=1.01, fontweight="bold")

    # Reference MC lines derived from actual solved time-varying MCs
    _tv_mc  = n.generators_t.marginal_cost
    _ccgt_g = [g for g in n.generators.index
               if n.generators.loc[g, "carrier"] == "CCGT"
               and g.startswith("ES") and g in _tv_mc.columns]
    if _ccgt_g:
        _all_mc  = _tv_mc[_ccgt_g].values.ravel()
        _ccgt_lo = float(np.nanpercentile(_all_mc, 10))
        _ccgt_hi = float(np.nanpercentile(_all_mc, 90))
    else:
        _ccgt_lo, _ccgt_hi = 82.0, 100.0
    _ocgt_g  = [g for g in n.generators.index
                if n.generators.loc[g, "carrier"] == "OCGT"
                and g.startswith("ES") and g in _tv_mc.columns]
    _ocgt_mc = float(np.nanmean(_tv_mc[_ocgt_g].values)) if _ocgt_g else 128.0
    _MC_REFS = {
        "Nuclear ~€15":                (15.0,      ":"),
        "Hydro ~€28":                  (28.0,      "-."),
        f"CCGT low ~€{_ccgt_lo:.0f}":  (_ccgt_lo,  "--"),
        f"CCGT high ~€{_ccgt_hi:.0f}": (_ccgt_hi,  "--"),
        f"OCGT ~€{_ocgt_mc:.0f}":      (_ocgt_mc,  ":"),
    }

    for row, country in enumerate(countries):
        ax_pdc, ax_bar = axes[row, 0], axes[row, 1]
        price, setter = _get_price_setter_series(n, country)

        if price.empty:
            ax_pdc.text(0.5, 0.5, f"No price data for {country}",
                        ha="center", va="center", transform=ax_pdc.transAxes, color="#888")
            ax_bar.axis("off")
            continue

        # ── PDC: sort descending, scatter coloured by setter ─────────────────
        df = pd.DataFrame({"price": price.values, "carrier": setter.values})
        df = df.sort_values("price", ascending=False).reset_index(drop=True)
        pct = np.linspace(0, 100, len(df))

        carriers_present = [c for c in CARRIER_ORDER if (df["carrier"] == c).any()]
        carriers_present += [c for c in df["carrier"].unique() if c not in carriers_present]

        for carrier in carriers_present:
            mask = df["carrier"] == carrier
            ax_pdc.scatter(
                pct[mask.values], df.loc[mask, "price"],
                color=COLORS.get(carrier, "#BBBBBB"), s=6, alpha=0.75,
                label=carrier, rasterized=True, zorder=3,
            )

        # ── Real PDC step line overlay (on same axis) ────────────────────────
        if real_prices is not None and country in real_prices:
            _real = real_prices[country].dropna()
            if len(_real) > 0:
                _real_sorted = np.sort(_real.values)[::-1]
                _x_real = np.linspace(0, 100, len(_real_sorted))
                ax_pdc.step(
                    _x_real, _real_sorted,
                    where="post", lw=1.8, color="#d7191c", alpha=0.75, zorder=4,
                )
                _real_mean = float(np.nanmean(_real_sorted))
                ax_pdc.axhline(_real_mean, color="#d7191c", lw=0.8, ls="--", alpha=0.50, zorder=2)
                ax_pdc.text(101, _real_mean, f"Real €{_real_mean:.0f}",
                            va="center", ha="left", fontsize=7, color="#d7191c")

        # ── Model PDC step line overlay (on same axis) ───────────────────────
        sorted_prices = np.sort(price.dropna().values)[::-1]
        x_pdc = np.linspace(0, 100, len(sorted_prices))
        ax_pdc.step(
            x_pdc, sorted_prices,
            where="post", lw=1.8, color="#2c7bb6", alpha=0.75, zorder=5,
        )
        _mean_p = float(np.nanmean(sorted_prices))
        ax_pdc.axhline(_mean_p, color="#2c7bb6", lw=0.8, ls="--", alpha=0.60, zorder=2)
        ax_pdc.text(101, _mean_p, f"Model €{_mean_p:.0f}",
                    va="center", ha="left", fontsize=7, color="#2c7bb6")

        # MC reference lines
        for label, (mc, ls) in _MC_REFS.items():
            if mc <= Y_MAX * 1.1:
                ax_pdc.axhline(mc, color="#666666", lw=0.7, ls=ls, alpha=0.55, zorder=1)
                ax_pdc.text(100.5, mc, label, va="center", ha="left",
                            fontsize=6, color="#555555", clip_on=False)

        ax_pdc.set_xlim(-1, 100)
        ax_pdc.set_ylim(-5, Y_MAX)
        ax_pdc.set_ylabel("Price (€/MWh)", fontsize=9)
        ax_pdc.set_xlabel("% of simulation hours", fontsize=9)
        ax_pdc.set_title(f"{country} — PDC by Price-Setter", fontsize=10)
        h, l = ax_pdc.get_legend_handles_labels()
        ax_pdc.legend(h, l, loc="upper right", fontsize=7, ncol=2,
                      framealpha=0.92, markerscale=2)
        ax_pdc.grid(axis="y", alpha=0.18)
        ax_pdc.spines[["top", "right"]].set_visible(False)

        # Zero-price / negative annotation
        n_zero  = (df["price"] <= 0.5).sum()
        n_snap  = len(df)
        pct_zero = n_zero / n_snap * 100
        ax_pdc.text(0.01, 0.02, f"≤ €0.5: {pct_zero:.1f}% of hours",
                    transform=ax_pdc.transAxes, fontsize=7.5, color="#cc0000",
                    va="bottom")

        # ── FR import overlay (ES only) ───────────────────────────────────────
        if country == "ES":
            fr_net = _net_import_topo(n, "FR")
            if not fr_net.eq(0).all():
                FR_THRESHOLD = 500.0  # MW — "significant" FR→ES import
                fr_heavy = fr_net > FR_THRESHOLD
                snap_idx = pd.Series(np.arange(len(price)), index=price.index)
                sorted_snaps = price.sort_values(ascending=False).index
                pdc_positions = {snap: i for i, snap in enumerate(sorted_snaps)}
                fr_heavy_pdc = fr_heavy.loc[price.index]
                heavy_snaps = fr_heavy_pdc[fr_heavy_pdc].index
                if len(heavy_snaps) > 0:
                    heavy_pct   = [pdc_positions[s] / len(price) * 100 for s in heavy_snaps if s in pdc_positions]
                    heavy_price = [price.loc[s] for s in heavy_snaps if s in pdc_positions]
                    ax_pdc.scatter(
                        heavy_pct, heavy_price,
                        s=30, facecolors="none", edgecolors="#444444",
                        linewidths=0.8, alpha=0.60, zorder=6,
                        label=f"FR imports >500 MW ({fr_heavy.mean()*100:.0f}% hrs)",
                    )
                    h2, l2 = ax_pdc.get_legend_handles_labels()
                    ax_pdc.legend(h2, l2, loc="upper right", fontsize=7, ncol=2,
                                  framealpha=0.92, markerscale=2)

        # ── Right bar: price-setter frequency ────────────────────────────────
        freq = setter.value_counts()
        carriers_sorted = [c for c in CARRIER_ORDER if c in freq.index]
        carriers_sorted += [c for c in freq.index if c not in carriers_sorted]
        vals   = [freq[c] / n_snap * 100 for c in carriers_sorted]
        colors = [COLORS.get(c, "#BBBBBB") for c in carriers_sorted]

        ys = range(len(carriers_sorted))
        ax_bar.barh(list(ys)[::-1], vals, color=colors, alpha=0.88,
                    height=0.6, edgecolor="white", linewidth=0.5)
        ax_bar.set_yticks(list(ys)[::-1])
        ax_bar.set_yticklabels(carriers_sorted, fontsize=8)
        ax_bar.set_xlabel("% of hours", fontsize=8)
        ax_bar.set_title(f"{country}\nPrice-Setter %", fontsize=9)
        ax_bar.set_xlim(0, max(vals) * 1.3 if vals else 100)
        for i, (v, y_pos) in enumerate(zip(vals, list(ys)[::-1])):
            ax_bar.text(v + 0.5, y_pos, f"{v:.1f}%", va="center", ha="left", fontsize=7.5)
        ax_bar.grid(axis="x", alpha=0.20)
        ax_bar.spines[["top", "right"]].set_visible(False)

    path = out_dir / "14_price_setter_analysis.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_spain_pdc_hd(n, out_dir, omie=None):
    """Standalone high-definition Spain price duration curve coloured by price-setter.

    Large single figure (Spain only) with a non-obscuring FR import overlay and
    an optional OMIE actual-price PDC step line for validation comparison.
    Saves at 300 dpi — zoom in without losing sharpness.
    """
    price, setter = _get_price_setter_series(n, "ES")
    if price.empty:
        log.warning("Spain HD PDC: no ES marginal prices — skipping (pass-2 may have failed)")
        return

    fig, (ax_pdc, ax_bar) = plt.subplots(
        1, 2, figsize=(22, 11),
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.22},
    )

    t0 = n.snapshots[0].strftime("%d %b")
    t1 = n.snapshots[-1].strftime("%d %b %Y")
    fig.suptitle(
        f"Spain — Price Duration Curve by Price-Setter  [{t0}–{t1}]",
        fontsize=16, fontweight="bold", y=1.01,
    )

    # ── Sort PDC descending ───────────────────────────────────────────────────
    df = pd.DataFrame({"price": price.values, "carrier": setter.values},
                      index=price.index)
    df_sorted = df.sort_values("price", ascending=False).reset_index(drop=True)
    pct = np.linspace(0, 100, len(df_sorted))

    # Preserve snapshot index for FR overlay mapping
    sorted_snaps = price.sort_values(ascending=False).index
    pdc_pos = {snap: i / len(price) * 100 for i, snap in enumerate(sorted_snaps)}

    carriers_present = [c for c in CARRIER_ORDER if (df_sorted["carrier"] == c).any()]
    carriers_present += [c for c in df_sorted["carrier"].unique() if c not in carriers_present]

    for carrier in carriers_present:
        mask = df_sorted["carrier"] == carrier
        ax_pdc.scatter(
            pct[mask.values], df_sorted.loc[mask, "price"],
            color=COLORS.get(carrier, "#BBBBBB"), s=12, alpha=0.80,
            label=carrier, rasterized=True, zorder=3,
        )

    # ── MC reference lines ────────────────────────────────────────────────────
    _MC_REFS = {
        "Nuclear ~€15":   (15.0,  ":"),
        "Hydro ~€28":     (28.0,  "-."),
        "CCGT base ~€75": (75.0,  "--"),
        "CCGT peak ~€105":(105.0, "--"),
        "OCGT ~€160":     (160.0, ":"),
    }
    y_max = max(df_sorted["price"].max(), 20)  # used only for MC-ref filter before p98 is computed
    for label, (mc, ls) in _MC_REFS.items():
        if mc <= y_max * 1.1:  # draw all within full range; off-scale ones are clipped by ylim
            ax_pdc.axhline(mc, color="#666666", lw=0.9, ls=ls, alpha=0.50, zorder=1)
            ax_pdc.text(100.8, mc, label, va="center", ha="left",
                        fontsize=8, color="#555555", clip_on=False)

    # ── OMIE actual PDC overlay — one colour step line, independently sorted ──
    if omie is not None:
        omie_clean = omie.dropna()
        if len(omie_clean) > 0:
            omie_sorted = np.sort(omie_clean.values)[::-1]
            x_omie = np.linspace(0, 100, len(omie_sorted))
            ax_pdc.step(
                x_omie, omie_sorted,
                where="post", lw=2.0, color=_SLATE,
                alpha=0.75, zorder=4,
                label="OMIE actual (Spain)",
            )

    # ── FR import overlay — tall dark vertical ticks, zorder below tech scatter
    fr_net = _net_import_topo(n, "FR")
    if not fr_net.eq(0).all():
        FR_THRESHOLD = 500.0
        fr_heavy = fr_net > FR_THRESHOLD
        heavy_snaps = fr_heavy[fr_heavy].index.intersection(price.index)
        if len(heavy_snaps) > 0:
            h_pct   = [pdc_pos[s] for s in heavy_snaps if s in pdc_pos]
            h_price = [price.loc[s] for s in heavy_snaps if s in pdc_pos]
            # Tall dark ticks — visible but non-obscuring (zorder=2, below scatter zorder=3)
            ax_pdc.scatter(
                h_pct, h_price,
                marker="|", s=200, linewidths=1.2,
                color="#111111", alpha=0.50, zorder=2,
                label=f"FR imports >500 MW ({fr_heavy.mean()*100:.0f}% hrs)",
            )

    # Cap y-axis at 98th percentile to prevent VOLL / extreme spikes from
    # collapsing the visible range — extreme hours still exist in data, just off-scale.
    all_prices = [df_sorted["price"]]
    if omie is not None and len(omie.dropna()) > 0:
        all_prices.append(omie.dropna())
    p98 = float(np.percentile(np.concatenate([s.values for s in all_prices]), 98))
    y_top = max(p98 * 1.08, 20.0)
    y_bot = min(df_sorted["price"].min() - 5, -10)

    ax_pdc.set_xlim(-0.5, 100)
    ax_pdc.set_ylim(bottom=y_bot, top=y_top)
    ax_pdc.set_ylabel("Price (€/MWh)", fontsize=12)
    ax_pdc.set_xlabel("% of simulation hours (sorted high → low)", fontsize=12)
    ax_pdc.tick_params(labelsize=11)
    ax_pdc.grid(axis="y", alpha=0.20)
    ax_pdc.spines[["top", "right"]].set_visible(False)

    n_zero   = (df_sorted["price"] <= 0.5).sum()
    pct_zero = n_zero / len(df_sorted) * 100
    ax_pdc.text(0.01, 0.02, f"≤ €0.5: {pct_zero:.1f}% of hours",
                transform=ax_pdc.transAxes, fontsize=9.5, color="#cc0000", va="bottom")

    h_handles, h_labels = ax_pdc.get_legend_handles_labels()
    ax_pdc.legend(h_handles, h_labels, loc="upper right", fontsize=9,
                  ncol=2, framealpha=0.93, markerscale=1.8)

    # ── Right bar: price-setter frequency ────────────────────────────────────
    freq = setter.value_counts()
    n_snap = len(setter)
    carriers_sorted = [c for c in CARRIER_ORDER if c in freq.index]
    carriers_sorted += [c for c in freq.index if c not in carriers_sorted]
    vals   = [freq[c] / n_snap * 100 for c in carriers_sorted]
    colors = [COLORS.get(c, "#BBBBBB") for c in carriers_sorted]

    ys = range(len(carriers_sorted))
    ax_bar.barh(list(ys)[::-1], vals, color=colors, alpha=0.88,
                height=0.65, edgecolor="white", linewidth=0.6)
    ax_bar.set_yticks(list(ys)[::-1])
    ax_bar.set_yticklabels(carriers_sorted, fontsize=11)
    ax_bar.set_xlabel("% of hours", fontsize=11)
    ax_bar.set_title("Price-Setter\nFrequency", fontsize=12)
    ax_bar.set_xlim(0, max(vals) * 1.35 if vals else 100)
    for v, y_pos in zip(vals, list(ys)[::-1]):
        ax_bar.text(v + 0.5, y_pos, f"{v:.1f}%", va="center", ha="left", fontsize=10)
    ax_bar.grid(axis="x", alpha=0.20)
    ax_bar.spines[["top", "right"]].set_visible(False)

    path = out_dir / "15_spain_pdc_setter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _stack_fill(ax, df, colors_map, carrier_order=None):
    """Stacked fill_between for hourly dispatch on ax. Returns (bottom, handles)."""
    ordered = [c for c in (carrier_order or CARRIER_ORDER) if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    pos = df.clip(lower=0)
    bottom = np.zeros(len(pos))
    handles = []
    for c in ordered:
        if c not in pos.columns or pos[c].sum() < 0.01:
            continue
        vals = pos[c].values
        color = colors_map.get(c, "#BBBBBB")
        ax.fill_between(pos.index, bottom, bottom + vals,
                        step="post", color=color, alpha=0.88, label=c)
        handles.append(mpatches.Patch(color=color, label=c))
        bottom = bottom + vals
    return pd.Series(bottom, index=pos.index), handles


def _midnight_lines(ax, snapshots):
    """Draw faint vertical lines at midnight for each day."""
    seen = set()
    for ts in snapshots:
        d = ts.normalize()
        if d not in seen and ts > snapshots[0]:
            ax.axvline(d, color="#aaaaaa", lw=0.45, zorder=0)
            seen.add(d)


# ─── Plot 01: Week overview — dispatch stack + real comparison + price ────────

def _plot_week_overview(n, omie, out_dir, cfg=None):
    """3-panel figure: model stack | REE actual stack | price vs OMIE — middle week."""
    es_dispatch = _dispatch_by_carrier(n, "ES")
    model_price = _mean_es_price(n)

    # Middle week of the simulation (same window as plot 11)
    mid          = len(n.snapshots) // 2
    week_start   = max(0, mid - 84)
    week_end_idx = min(week_start + 168, len(n.snapshots))
    snap_w  = n.snapshots[week_start:week_end_idx]
    disp_w  = es_dispatch.reindex(snap_w).fillna(0.0)
    price_w = model_price.reindex(snap_w)
    omie_w  = omie.reindex(snap_w)

    t0 = snap_w[0].strftime("%d %b")
    t1 = snap_w[-1].strftime("%d %b %Y")

    fig, (ax_mod, ax_real, ax_price) = plt.subplots(
        3, 1, figsize=(14, 11), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.2, 1], "hspace": 0.06},
    )

    # ── Top panel: model stacked dispatch ────────────────────────────────────
    _, _ = _stack_fill(ax_mod, disp_w, COLORS)
    ax_mod.set_ylabel("Generation (MW)")
    ax_mod.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f} GW" if x >= 500 else f"{x:.0f}")
    )
    ax_mod.set_title(f"Spain — Model vs REE Actual: Dispatch & Price  [{t0}–{t1}]",
                     fontsize=12, pad=9)
    h, l = ax_mod.get_legend_handles_labels()
    ax_mod.legend(h[::-1], l[::-1], loc="upper right", ncol=3, fontsize=7.5,
                  framealpha=0.93, bbox_to_anchor=(1.0, 1.0))
    ax_mod.text(0.01, 0.97, "Model (Spain)", transform=ax_mod.transAxes,
                fontsize=9, va="top", color="#333333", fontweight="bold")
    _midnight_lines(ax_mod, snap_w)

    # ── Middle panel: REE actual stacked (same carrier colours / order) ───────
    ree_path = (ROOT / cfg["validation"]["real_dispatch_csv"]) if cfg else None
    if ree_path and ree_path.exists():
        ree_raw = pd.read_csv(ree_path, parse_dates=["timestamp"], index_col="timestamp")
        if ree_raw.index.tz is not None:
            ree_raw.index = ree_raw.index.tz_localize(None)
        ree_slice = ree_raw.loc[snap_w[0]:snap_w[-1]].reindex(snap_w, method="nearest")
        ree_by_carrier = {}
        for col in ree_slice.columns:
            carrier = _REE_TO_CARRIER.get(col, "other")
            vals = ree_slice[col].fillna(0.0)
            ree_by_carrier[carrier] = (
                ree_by_carrier.get(carrier, pd.Series(0.0, index=ree_slice.index)) + vals
            )
        ree_df = pd.DataFrame(ree_by_carrier, index=ree_slice.index)
        _, _ = _stack_fill(ax_real, ree_df, COLORS)
        h2, l2 = ax_real.get_legend_handles_labels()
        ax_real.legend(h2[::-1], l2[::-1], loc="upper right", ncol=3, fontsize=7.5,
                       framealpha=0.93, bbox_to_anchor=(1.0, 1.0))
    else:
        ax_real.text(0.5, 0.5, "REE actual data not available",
                     ha="center", va="center", transform=ax_real.transAxes, color="#888888")
    ax_real.set_ylabel("Generation (MW)")
    ax_real.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f} GW" if x >= 500 else f"{x:.0f}")
    )
    ax_real.text(0.01, 0.97, "REE Actual", transform=ax_real.transAxes,
                 fontsize=9, va="top", color="#333333", fontweight="bold")
    _midnight_lines(ax_real, snap_w)

    # ── Bottom panel: price vs OMIE ───────────────────────────────────────────
    ax_price.plot(price_w.index, price_w.values, lw=2.0, color="#2c7bb6",
                  label="Model — ES avg", zorder=3)
    ax_price.plot(omie_w.index, omie_w.values, lw=1.6, color="#d7191c",
                  ls="--", alpha=0.85, label="OMIE Spain", zorder=2)
    ax_price.fill_between(price_w.index, price_w.values, omie_w.values,
                          alpha=0.10, color="grey")
    ax_price.set_ylabel("Price (EUR/MWh)")
    ax_price.set_ylim(bottom=0)
    ax_price.legend(loc="upper right", fontsize=8)
    ax_price.xaxis.set_major_locator(mdates.HourLocator(byhour=0))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax_price.set_xlim(snap_w[0], snap_w[-1])
    _midnight_lines(ax_price, snap_w)

    path = out_dir / "01_week_overview.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 02: Price duration curve ────────────────────────────────────────────

def _plot_price_duration(n, omie, out_dir):
    """Enhanced price duration curve with threshold bands."""
    model_s = np.sort(_mean_es_price(n).dropna().values)[::-1]
    omie_s  = np.sort(omie.dropna().values)[::-1]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Shaded technology bands
    band_specs = [
        (0,    30,  "#2ECC71", "VRE / nuclear"),
        (30,   80,  "#1E8BC3", "Hydro / baseload"),
        (80,   130, "#E67E22", "CCGT"),
        (130,  210, "#C0392B", "Peakers"),
    ]
    for lo, hi, color, label in band_specs:
        ax.axhspan(lo, hi, alpha=0.06, color=color, zorder=0)

    x_m = np.linspace(0, 100, len(model_s))
    x_o = np.linspace(0, 100, len(omie_s))
    ax.fill_between(x_m, model_s, alpha=0.12, color="#2c7bb6", step="post")
    ax.step(x_m, model_s, lw=2.0, color="#2c7bb6", label="Model — Spain (load-weighted)", where="post")
    ax.step(x_o, omie_s,  lw=1.8, color="#d7191c", ls="--", label="OMIE Spain", where="post")

    for price, label in [(80.0, "CCGT ≈ €80"), (170.0, "Peaker ≈ €170")]:
        ax.axhline(price, color="#888888", ls=":", lw=1.0, alpha=0.8)
        ax.text(74, price + 3, label, fontsize=8, color="#555555", va="bottom")

    bias = _mean_es_price(n).mean() - omie.mean()
    ax.text(0.02, 0.97, f"Mean bias: {bias:+.1f} €/MWh",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

    ax.set_xlabel("% of simulation hours (ranked high → low)")
    ax.set_ylabel("Wholesale Price (EUR/MWh)")
    ax.set_title("Price Duration Curve — Spain")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_ylim(0)

    path = out_dir / "02_price_duration_curve.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_es_pt_joint_pdc(n, out_dir, omie_es=None, omie_pt=None):
    """ES + PT joint price duration curve — reveals MIBEL coupling gaps.

    Both model price series are sorted independently (low → high) and
    overlaid on the same 0–100% axis.  Divergence between the two curves
    indicates interconnector congestion: when PT is above ES, Portugal is
    import-constrained (and vice versa).

    Parameters
    ----------
    n : pypsa.Network
        Solved network.
    out_dir : Path
        Output directory.
    omie_es : pd.Series or None
        OMIE Spain day-ahead prices (for reference overlay).
    omie_pt : pd.Series or None
        OMIE Portugal day-ahead prices (for reference overlay).
    """
    es_buses = [b for b in _country_buses(n, "ES")
                if b in n.buses_t.marginal_price.columns]
    pt_buses = [b for b in _country_buses(n, "PT")
                if b in n.buses_t.marginal_price.columns]

    if not es_buses or not pt_buses:
        log.warning("ES+PT joint PDC: missing bus marginal prices — skipping")
        return

    # Load-weighted mean price for each country
    def _mean_price(buses):
        weights = {}
        for bus in buses:
            bus_loads = n.loads.index[n.loads["bus"] == bus]
            total = 0.0
            for ld in bus_loads:
                if ld in n.loads_t.p_set.columns:
                    total += float(n.loads_t.p_set[ld].mean())
                else:
                    total += float(n.loads.loc[ld, "p_set"])
            weights[bus] = total
        w = pd.Series(weights, dtype=float)[buses]
        if w.sum() == 0:
            w = pd.Series(1.0, index=buses)
        w = w / w.sum()
        return (n.buses_t.marginal_price[buses] * w).sum(axis=1)

    es_price = _mean_price(es_buses)
    pt_price = _mean_price(pt_buses)

    if es_price.empty or pt_price.empty:
        log.warning("ES+PT joint PDC: empty price series — skipping")
        return

    # Sort independently low → high
    es_sorted = np.sort(es_price.dropna().values)
    pt_sorted = np.sort(pt_price.dropna().values)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_es = np.linspace(0, 100, len(es_sorted))
    x_pt = np.linspace(0, 100, len(pt_sorted))

    # Shaded gap between ES and PT curves
    # Interpolate PT onto ES x-axis for fill_between
    pt_interp = np.interp(x_es, x_pt, pt_sorted)
    ax.fill_between(x_es, es_sorted, pt_interp,
                    where=(pt_interp > es_sorted),
                    color="#E63946", alpha=0.10,
                    label="PT > ES (PT import-constrained)")
    ax.fill_between(x_es, es_sorted, pt_interp,
                    where=(pt_interp <= es_sorted),
                    color="#2A9D8F", alpha=0.10,
                    label="PT < ES (PT export-constrained)")

    # Model lines
    ax.plot(x_es, es_sorted, color="#E63946", lw=1.8, label="Model ES", zorder=3)
    ax.plot(x_pt, pt_sorted, color="#2A9D8F", lw=1.8, label="Model PT", zorder=3)

    # OMIE actual overlays
    if omie_es is not None and len(omie_es) > 0:
        omie_es_sorted = np.sort(omie_es.dropna().values)
        x_omie_es = np.linspace(0, 100, len(omie_es_sorted))
        ax.plot(x_omie_es, omie_es_sorted, color="#E63946", lw=1.0,
                ls="--", alpha=0.55, label="OMIE ES actual", zorder=2)
    if omie_pt is not None and len(omie_pt) > 0:
        omie_pt_sorted = np.sort(omie_pt.dropna().values)
        x_omie_pt = np.linspace(0, 100, len(omie_pt_sorted))
        ax.plot(x_omie_pt, omie_pt_sorted, color="#2A9D8F", lw=1.0,
                ls="--", alpha=0.55, label="OMIE PT actual", zorder=2)

    # Congestion statistics (computed on unsorted time series)
    gap = (es_price - pt_price).fillna(0.0)
    CONG_THRESH = 1.0  # €/MWh
    cong_mask = gap.abs() > CONG_THRESH
    n_cong = int(cong_mask.sum())
    pct_cong = n_cong / max(len(gap), 1) * 100
    mean_gap = float(gap[cong_mask].mean()) if n_cong > 0 else 0.0

    # Approximate congestion rent: |gap| × net PT flow
    pt_net = _net_import_topo(n, "PT")
    if pt_net is not None and len(pt_net) == len(gap):
        cong_rent_meur = float((gap.abs() * pt_net.abs() / 1e6).sum())
    else:
        cong_rent_meur = float("nan")

    stats_text = (
        f"Congested: {pct_cong:.1f}% ({n_cong}h)\n"
        f"Mean |gap|: {abs(mean_gap):.1f} €/MWh\n"
        f"Est. cong. rent: €{cong_rent_meur:.2f}M" if not pd.isna(cong_rent_meur)
        else f"Congested: {pct_cong:.1f}% ({n_cong}h)\nMean |gap|: {abs(mean_gap):.1f} €/MWh"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.9))

    ax.set_xlabel("% of simulation hours (low → high)")
    ax.set_ylabel("Wholesale Price (€/MWh)")
    ax.set_title("ES + PT Joint Price Duration Curve — MIBEL Coupling Diagnostic")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)

    path = out_dir / "17_es_pt_joint_pdc.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 19: Transmission line bottleneck map ────────────────────────────────

def _plot_line_bottleneck(n, out_dir):
    """Geographic map of ES transmission lines coloured by congestion frequency.

    Lines coloured from green (rarely congested) to red (>30% of hours at >80%
    loading).  Lines exceeding 10% congestion are annotated.
    Saves 19_line_bottleneck.png.
    """
    if n.lines_t.p0.empty:
        log.warning("line_bottleneck: no lines_t.p0 data — skipping")
        return

    es_lines = n.lines[
        n.lines.bus0.str.startswith("ES") & n.lines.bus1.str.startswith("ES")
    ]
    if es_lines.empty:
        log.warning("line_bottleneck: no internal ES lines found — skipping")
        return

    p0     = n.lines_t.p0.reindex(columns=es_lines.index, fill_value=0.0).abs()
    s_nom  = es_lines.s_nom.clip(lower=1.0)
    load_f = (p0 / s_nom).clip(upper=1.0)
    pct_c  = (load_f > 0.80).mean() * 100   # % hours at >80% loading

    buses = n.buses[
        n.buses.index.str.startswith("ES") &
        n.buses.x.notna() & n.buses.y.notna()
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.set_facecolor(_ICE)
    ax.set_title(
        f"ES Transmission Bottleneck Map — "
        f"{int((pct_c > 10).sum())} lines congested >10 % of hours  "
        f"(threshold: >80 % loading)",
        fontsize=10, fontweight="bold",
    )

    cmap = plt.get_cmap("RdYlGn_r")
    norm = plt.Normalize(0, 30)   # 0–30% hours congested scale

    for line_id, line in es_lines.iterrows():
        if line.bus0 not in buses.index or line.bus1 not in buses.index:
            continue
        b0, b1 = buses.loc[line.bus0], buses.loc[line.bus1]
        pct = float(pct_c.get(line_id, 0.0))
        color = cmap(norm(pct))
        lw = 0.8 + pct / 8          # thicker = more congested
        ax.plot([b0.x, b1.x], [b0.y, b1.y], color=color, lw=lw, zorder=2, alpha=0.85)
        if pct > 10:
            mx, my = (b0.x + b1.x) / 2, (b0.y + b1.y) / 2
            ax.text(mx, my, f"{pct:.0f}%", fontsize=6, ha="center", va="center",
                    color="#7F0000", fontweight="bold", zorder=4)

    # Bus scatter (size ∝ annual mean load)
    load_mean = n.loads_t.p_set.mean() if not n.loads_t.p_set.empty else pd.Series(dtype=float)
    for bus_id, bus in buses.iterrows():
        loads_at = n.loads.index[n.loads.bus == bus_id]
        mw = float(load_mean.reindex(loads_at).sum()) if not loads_at.empty else 0.0
        ax.scatter(bus.x, bus.y, s=max(mw / 60, 6), c=_SLATE,
                   zorder=3, alpha=0.65, linewidths=0)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("% hours at >80 % loading", fontsize=9)

    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "19_line_bottleneck.png"
    fig.savefig(out_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ─── Plot 20: Price-setter breakdown ─────────────────────────────────────────

def _plot_price_setter_breakdown(n, out_dir):
    """Two-panel diagnostic: which carrier sets ES price at which price level.

    Row 1 — stacked histogram over price bins: each bar = hours in that price range,
    stacked by price-setting carrier. Reveals whether CCGT dominates at the right
    price levels (€55–90) vs wrong levels (€0–30 should be VRE/nuclear).

    Row 2 — horizontal bar chart: % hours each carrier is price-setter, sorted
    descending. Reference lines for approximate Spain 2024 actual setter shares.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    try:
        price_series, setter_series = _get_price_setter_series(n, "ES")
    except Exception as exc:
        log.warning("_plot_price_setter_breakdown: could not get setter series: %s", exc)
        return

    if setter_series.empty:
        log.warning("_plot_price_setter_breakdown: empty setter series, skipping")
        return

    # Price bins and carrier colours (reuse TECH_STYLE where available)
    bin_edges  = [-1, 10, 30, 50, 70, 90, 120, 9999]
    bin_labels = ["≤10", "10–30", "30–50", "50–70", "70–90", "90–120", ">120"]

    carriers_ordered = setter_series.value_counts().index.tolist()
    cmap_colors = plt.cm.get_cmap("tab20", max(len(carriers_ordered), 1))
    carrier_color = {c: cmap_colors(i) for i, c in enumerate(carriers_ordered)}
    for c, color in COLORS.items():
        if c in carrier_color:
            carrier_color[c] = color

    # Build matrix: rows=price_bin, cols=carrier, values=hours
    price_bin_s = pd.cut(price_series, bins=bin_edges, labels=bin_labels)
    matrix = {}
    for carrier in carriers_ordered:
        mask = setter_series == carrier
        bin_counts = price_bin_s[mask].value_counts().reindex(bin_labels, fill_value=0)
        matrix[carrier] = bin_counts

    matrix_df = pd.DataFrame(matrix, index=bin_labels)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.6, 1]})
    fig.patch.set_facecolor("#F7F9FB")
    ax1.set_facecolor("#F7F9FB")
    ax2.set_facecolor("#F7F9FB")

    # ── Row 1: stacked bar by price bin ──────────────────────────────────────
    bottoms = pd.Series(0.0, index=bin_labels)
    for carrier in carriers_ordered:
        vals = matrix_df[carrier]
        ax1.bar(bin_labels, vals, bottom=bottoms, color=carrier_color[carrier],
                label=carrier, edgecolor="white", linewidth=0.4)
        bottoms = bottoms + vals

    ax1.set_xlabel("Price bin (€/MWh)", fontsize=9)
    ax1.set_ylabel("Hours", fontsize=9)
    ax1.set_title("ES price-setting carrier by price level", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7, ncol=2,
               framealpha=0.9, title="Price-setter")
    ax1.tick_params(labelsize=8)

    # ── Row 2: % hours each carrier sets price ────────────────────────────────
    total_h = max(len(setter_series), 1)
    pct_series = setter_series.value_counts() / total_h * 100

    colors_r2 = [carrier_color.get(c, "#999") for c in pct_series.index]
    bars = ax2.barh(pct_series.index, pct_series.values, color=colors_r2,
                    edgecolor="white", linewidth=0.4)

    # Reference lines for Spain 2024 actuals
    ref = {"CCGT": (45, 55), "solar": (10, 18), "onwind": (5, 12), "nuclear": (8, 14), "hydro": (8, 14)}
    for carrier, (lo, hi) in ref.items():
        if carrier in pct_series.index:
            ax2.plot([lo, hi], [carrier, carrier], color="#C0392B", lw=3, solid_capstyle="round",
                     alpha=0.7, zorder=5)

    ax2.set_xlabel("% of hours as price-setter", fontsize=9)
    ax2.set_title("Price-setter share (red bars = Spain 2024 target range)", fontsize=10)
    ax2.axvline(50, color="#999", lw=0.8, ls="--", alpha=0.5)
    ax2.tick_params(labelsize=8)
    ax2.invert_yaxis()

    fig.suptitle(
        f"ES Price-Setter Breakdown  ·  {len(setter_series)} hours",
        fontsize=11, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_path = out_dir / "20_price_setter_breakdown.png"
    fig.savefig(out_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ─── Plot 03: Installed capacity vs reality ────────────────────────────────────

def _plot_capacity_vs_reality(n, out_dir):
    """Horizontal grouped bar chart comparing model capacity with 2024 REE data."""
    model_cap = _model_capacity_es(n)

    groups   = list(_REAL_CAPACITY_MW.keys())
    model_gw = []
    real_gw  = []
    for grp in groups:
        carriers = _REAL_CAP_MAP.get(grp, [grp])
        model_gw.append(sum(model_cap.get(c, 0.0) for c in carriers) / 1000.0)
        real_gw.append(_REAL_CAPACITY_MW[grp] / 1000.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(groups))
    h = 0.36

    bars_m = ax.barh(y + h / 2, model_gw, h, color="#2c7bb6", alpha=0.88, label="Model")
    bars_r = ax.barh(y - h / 2, real_gw,  h, color="#d7191c", alpha=0.88, label="Real 2024 (REE)")

    x_max = max(max(model_gw), max(real_gw)) * 1.25
    for bar, val in zip(bars_m, model_gw):
        ax.text(val + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color="#2c7bb6", fontweight="bold")
    for bar, val in zip(bars_r, real_gw):
        ax.text(val + x_max * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color="#d7191c", fontweight="bold")

    # % diff annotation on right
    for i, (m, r) in enumerate(zip(model_gw, real_gw)):
        if r > 0:
            pct = (m - r) / r * 100
            color = "#2c7bb6" if abs(pct) < 10 else "#d7191c"
            ax.text(x_max * 0.98, y[i], f"{pct:+.0f}%",
                    va="center", ha="right", fontsize=7.5, color=color, style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([g.title() for g in groups], fontsize=9)
    ax.set_xlabel("Installed Capacity (GW)")
    ax.set_title("Installed Capacity: Model vs Reality — Spain")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, x_max)
    ax.invert_yaxis()
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    path = out_dir / "03_capacity_vs_reality.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 04: Full network map — ES / PT / FR ─────────────────────────────────

def _plot_network_map(n, out_dir):
    """Geographic map of the full ES / PT / FR transmission network."""
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection=proj)

    # Basemap — Natural Earth features (50m resolution balances detail vs speed)
    ax.add_feature(cfeature.OCEAN.with_scale("50m"),     facecolor="#cce5ff", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("50m"),      facecolor="#f2ede3", zorder=0)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, edgecolor="#666666", zorder=1)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"),   linewidth=0.4, edgecolor="#999999",
                   linestyle="--", zorder=1)

    # AC lines
    for _, line in n.lines.iterrows():
        b0, b1 = line.bus0, line.bus1
        if b0 not in n.buses.index or b1 not in n.buses.index:
            continue
        x0, y0 = float(n.buses.loc[b0, "x"]), float(n.buses.loc[b0, "y"])
        x1, y1 = float(n.buses.loc[b1, "x"]), float(n.buses.loc[b1, "y"])
        c0, c1 = b0[:2], b1[:2]
        if c0 == "ES" and c1 == "ES":
            color, lw, alpha, zorder = "#cc3333", 0.7, 0.55, 2
        elif c0 == "FR" and c1 == "FR":
            color, lw, alpha, zorder = "#3366cc", 0.55, 0.45, 2
        elif c0 == "PT" and c1 == "PT":
            color, lw, alpha, zorder = "#33aa55", 0.55, 0.45, 2
        else:
            color, lw, alpha, zorder = "#ff8800", 1.8, 0.90, 4
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=zorder, transform=proj)

    # DC links — cross-country interconnectors (orange dashed) + intra-national HVDC (purple dashed)
    # Same-country non-DC links (batteries, electrolyzers, PHS chargers) are skipped.
    # Cross-country links are all interconnectors and are drawn regardless of carrier.
    for _, link in n.links.iterrows():
        b0 = link.get("bus0", "")
        b1 = link.get("bus1", "")
        carrier = link.get("carrier", "")
        if not (isinstance(b0, str) and isinstance(b1, str)):
            continue
        if b0 not in n.buses.index or b1 not in n.buses.index:
            continue
        same_country = b0[:2] == b1[:2]
        is_hvdc = carrier == "DC"
        if same_country and not is_hvdc:
            continue  # skip battery chargers, electrolyzers, PHS, etc. within same country
        x0, y0 = float(n.buses.loc[b0, "x"]), float(n.buses.loc[b0, "y"])
        x1, y1 = float(n.buses.loc[b1, "x"]), float(n.buses.loc[b1, "y"])
        if same_country:
            # intra-national HVDC (Balearics cable: ES0 5 → ES1 0, carrier='DC')
            ax.plot([x0, x1], [y0, y1], color="#8833cc", lw=2.0,
                    ls="--", alpha=0.90, zorder=4, transform=proj)
        else:
            # cross-country interconnectors (FR-ES, PT-ES)
            ax.plot([x0, x1], [y0, y1], color="#ff8800", lw=2.2,
                    ls="--", alpha=0.92, zorder=4, transform=proj)

    # Nodes — FR and PT first (behind ES)
    for prefix in ("FR", "PT", "ES"):
        buses = [
            b for b in n.buses.index
            if b.startswith(prefix) and "H2" not in b and "battery" not in b
        ]
        if not buses:
            continue
        xs = n.buses.loc[buses, "x"].astype(float).values
        ys = n.buses.loc[buses, "y"].astype(float).values
        color = COUNTRY_COLORS[prefix]
        size  = 90 if prefix == "ES" else 55
        ax.scatter(xs, ys, s=size, c=color, edgecolors="white", linewidths=1.0,
                   zorder=6, alpha=0.95, transform=proj,
                   label={"ES": "Spain", "FR": "France", "PT": "Portugal"}[prefix])
        if prefix == "ES":
            for bus, x, y in zip(buses, xs, ys):
                short = bus.replace("ES", "").strip()
                ax.annotate(short, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=5.5, color="#111111", zorder=7,
                            alpha=0.85)

    # Legend with line types
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([0], [0], color="#cc3333", lw=1.5, label="ES transmission (AC)"),
        plt.Line2D([0], [0], color="#3366cc", lw=1.2, label="FR transmission (AC)"),
        plt.Line2D([0], [0], color="#33aa55", lw=1.2, label="PT transmission (AC)"),
        plt.Line2D([0], [0], color="#ff8800", lw=2.0, label="Interconnector (AC)"),
        plt.Line2D([0], [0], color="#ff8800", lw=2.0, ls="--", label="Interconnector (DC)"),
        plt.Line2D([0], [0], color="#8833cc", lw=2.0, ls="--", label="Island HVDC (DC)"),
    ]
    labels += [
        "ES transmission (AC)", "FR transmission (AC)", "PT transmission (AC)",
        "Interconnector (AC)", "Interconnector (DC)", "Island HVDC (DC)",
    ]
    ax.legend(handles, labels, loc="lower left", fontsize=8, framealpha=0.93,
              edgecolor="#dddddd", ncol=2)

    ax.set_title("Iberian Peninsula & France — Transmission Network")
    all_x = n.buses["x"].dropna().astype(float)
    all_y = n.buses["y"].dropna().astype(float)
    ax.set_extent([all_x.min() - 0.8, all_x.max() + 0.8,
                   all_y.min() - 0.8, all_y.max() + 0.8], crs=proj)
    ax.gridlines(alpha=0.20)

    path = out_dir / "04_network_map.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 05: Curtailment intensity map ───────────────────────────────────────

def _plot_curtailment_map(n, out_dir):
    """Spain VRE curtailment intensity map — rate (%) and absolute (GWh)."""
    curt = _curtailment_stats(n)
    node_pot = curt["node_pot"]
    node_act = curt["node_act"]

    if not node_pot:
        log.warning("Curtailment map: no VRE generators — skipping")
        return

    nodes    = list(node_pot.keys())
    pcts     = [(node_pot[b] - node_act[b]) / node_pot[b] * 100.0 for b in nodes]
    abs_curt = [(node_pot[b] - node_act[b]) / 1000.0 for b in nodes]

    coords = n.buses.loc[nodes, ["x", "y"]].astype(float)
    pcts_s = pd.Series(pcts, index=nodes)
    abs_s  = pd.Series(abs_curt, index=nodes)

    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, 2, figsize=(18, 8),
        subplot_kw={"projection": proj},
    )
    fig.patch.set_facecolor("white")

    es_line_idx = n.lines.index[
        n.lines["bus0"].str.startswith("ES") & n.lines["bus1"].str.startswith("ES")
    ]

    panels = [
        (axes[0], pcts_s, "Curtailment Rate  (% of potential)", "YlOrRd", "%"),
        (axes[1], abs_s,  "Absolute Curtailment  (GWh)", "OrRd", "GWh"),
    ]
    for i, (ax, vals, title, cmap, unit) in enumerate(panels):
        gl = _setup_cartopy_ax(ax)
        gl.left_labels = (i == 0)

        # Transmission skeleton
        for _, line in n.lines.loc[es_line_idx].iterrows():
            b0, b1 = line.bus0, line.bus1
            if b0 not in n.buses.index or b1 not in n.buses.index:
                continue
            ax.plot(
                [n.buses.loc[b0, "x"], n.buses.loc[b1, "x"]],
                [n.buses.loc[b0, "y"], n.buses.loc[b1, "y"]],
                color="#b0bccc", lw=0.6, alpha=0.65, zorder=4,
                transform=proj,
            )

        vmax = float(vals.max()) * 1.05 if vals.max() > 0 else 1.0
        sizes = (vals / max(vals.max(), 1) * 500 + 60).clip(lower=60)
        sc = ax.scatter(
            coords["x"], coords["y"],
            c=vals.values, cmap=cmap, s=sizes.values,
            vmin=0, vmax=vmax,
            edgecolors="white", linewidths=1.0, zorder=6, alpha=0.92,
            transform=proj,
        )
        cb = plt.colorbar(sc, ax=ax, shrink=0.72, pad=0.03, aspect=20)
        cb.set_label(unit, fontsize=9)
        cb.ax.tick_params(labelsize=8)

        # Annotate top-3 worst nodes
        for bus in vals.nlargest(3).index:
            x, y = coords.loc[bus, "x"], coords.loc[bus, "y"]
            short = bus.replace("ES", "").strip()
            ax.annotate(
                f"{short}\n{vals[bus]:.1f}{unit}",
                xy=(x, y), xytext=(10, 10),
                textcoords="offset points",
                fontsize=7.5, color="#1a1a2e", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="none"),
                zorder=8,
                xycoords=proj._as_mpl_transform(ax),
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color=_SLATE)

    t0 = n.snapshots[0].strftime("%d %b %Y")
    t1 = n.snapshots[-1].strftime("%d %b %Y")
    fig.suptitle(
        f"Spain — VRE Curtailment by Node  [{t0} – {t1}]",
        fontsize=14, fontweight="bold", y=1.01, color=_SLATE,
    )

    path = out_dir / "05_curtailment_map.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 06: Dispatch by node — geographic pie charts ────────────────────────

def _plot_node_dispatch_pies(n, out_dir):
    """Generation mix pies at each node — ES, FR, PT — on a Natural Earth basemap."""
    # ── Collect per-bus dispatch mix for ES + FR + PT ──────────────────────────
    prefixes = ("ES", "FR", "PT")
    all_buses = [
        b for b in n.buses.index
        if any(b.startswith(p) for p in prefixes)
        and "H2" not in b and "battery" not in b
    ]

    total_by_bus: dict[str, float] = {}
    mix_by_bus:   dict[str, pd.Series] = {}
    for bus in all_buses:
        gens  = n.generators.index[n.generators["bus"] == bus]
        avail = [g for g in gens if g in n.generators_t.p.columns]
        # also include storage dispatch (hydro StorageUnits)
        sus   = n.storage_units.index[n.storage_units["bus"] == bus]
        avail_su = [s for s in sus if s in n.storage_units_t.p_dispatch.columns]

        parts = []
        if avail:
            carriers = n.generators.loc[avail, "carrier"]
            parts.append(n.generators_t.p[avail].T.groupby(carriers).sum().sum(axis=1))
        if avail_su:
            carriers_su = n.storage_units.loc[avail_su, "carrier"]
            parts.append(n.storage_units_t.p_dispatch[avail_su].T.groupby(carriers_su).sum().sum(axis=1))
        if not parts:
            continue
        by_carrier = pd.concat(parts).groupby(level=0).sum()
        total = float(by_carrier.sum())
        if total < 1.0:
            continue
        total_by_bus[bus] = total
        mix_by_bus[bus]   = by_carrier[by_carrier > 0]

    if not total_by_bus:
        log.warning("Node dispatch pies: no data — skipping")
        return

    coords    = n.buses.loc[list(total_by_bus.keys()), ["x", "y"]].astype(float)
    max_total = max(total_by_bus.values())

    # ── Figure: Cartopy PlateCarree with wider extent to show FR+PT ────────────
    proj    = ccrs.PlateCarree()
    EXTENT  = (-10.8, 8.5, 35.0, 46.8)   # (lon_min, lon_max, lat_min, lat_max)
    fig, ax = plt.subplots(figsize=(15, 11), subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    _setup_cartopy_ax(ax, extent=EXTENT)

    # Transmission skeleton (ES internal lines)
    es_line_idx = n.lines.index[
        n.lines["bus0"].str.startswith("ES") & n.lines["bus1"].str.startswith("ES")
    ]
    for _, line in n.lines.loc[es_line_idx].iterrows():
        b0, b1 = line.bus0, line.bus1
        if b0 not in n.buses.index or b1 not in n.buses.index:
            continue
        ax.plot(
            [n.buses.loc[b0, "x"], n.buses.loc[b1, "x"]],
            [n.buses.loc[b0, "y"], n.buses.loc[b1, "y"]],
            color="#b0bccc", lw=0.55, alpha=0.60, zorder=4, transform=proj,
        )

    # Country label watermarks
    for label, lon, lat in [("SPAIN", -3.5, 40.2), ("FRANCE", 2.5, 45.5), ("PORTUGAL", -8.2, 39.5)]:
        ax.text(lon, lat, label, transform=proj,
                fontsize=11, color="#c8d0da", fontweight="bold",
                ha="center", va="center", alpha=0.55, zorder=4,
                style="italic")

    # ── Pie placement (figure-fraction math via Cartopy extent) ────────────────
    lon_min, lon_max, lat_min, lat_max = EXTENT
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min
    ax_pos = ax.get_position()

    for bus in total_by_bus:
        lon = coords.loc[bus, "x"]
        lat = coords.loc[bus, "y"]
        mix   = mix_by_bus[bus]
        total = total_by_bus[bus]

        x_frac = (lon - lon_min) / lon_range
        y_frac = (lat - lat_min) / lat_range
        fig_x  = ax_pos.x0 + x_frac * ax_pos.width
        fig_y  = ax_pos.y0 + y_frac * ax_pos.height

        # Scale pies: ES nodes get full range; FR/PT slightly smaller max
        country = bus[:2]
        base = 0.052 if country == "ES" else 0.058
        scale = (total / max_total) ** 0.42
        size  = base * scale + 0.010   # minimum readable size

        pie_ax = fig.add_axes([fig_x - size / 2, fig_y - size / 2, size, size])
        values = mix.values
        colors = [COLORS.get(c, "#BBBBBB") for c in mix.index]
        pie_ax.pie(
            values, colors=colors, startangle=90,
            wedgeprops={"linewidth": 0.4, "edgecolor": "white"},
        )
        pie_ax.set_aspect("equal")

        short = (bus.replace("ES", "").replace("FR_", "FR ").replace("PT_", "PT ")
                    .strip().replace("  ", " "))
        fs = 5.5 if country == "ES" else 6.5
        pie_ax.set_title(short, fontsize=fs, pad=1.8,
                         color=COUNTRY_COLORS.get(country, "#333333"),
                         fontweight="bold")

    # ── Country legend patches ──────────────────────────────────────────────────
    country_patches = [
        mpatches.Patch(color=COUNTRY_COLORS["ES"], label="Spain (ES)"),
        mpatches.Patch(color=COUNTRY_COLORS["FR"], label="France (FR)"),
        mpatches.Patch(color=COUNTRY_COLORS["PT"], label="Portugal (PT)"),
    ]
    all_carriers = set()
    for mix in mix_by_bus.values():
        all_carriers.update(mix.index)
    carrier_handles = [
        mpatches.Patch(color=COLORS.get(c, "#BBB"), label=c)
        for c in CARRIER_ORDER if c in all_carriers
    ]
    leg1 = ax.legend(handles=country_patches, loc="lower left",
                     fontsize=8, framealpha=0.92, edgecolor=_GRID_C,
                     title="Country", title_fontsize=8,
                     bbox_to_anchor=(0.01, 0.01))
    ax.add_artist(leg1)
    ax.legend(handles=carrier_handles, loc="lower right",
              fontsize=7.5, framealpha=0.92, edgecolor=_GRID_C,
              ncol=2, title="Carrier", title_fontsize=8,
              bbox_to_anchor=(0.99, 0.01))

    t0 = n.snapshots[0].strftime("%d %b %Y")
    t1 = n.snapshots[-1].strftime("%d %b %Y")
    ax.set_title(
        f"ES / FR / PT — Generation Mix by Node  [{t0} – {t1}]\n"
        "(pie area ∝ total dispatch over period)",
        fontsize=12, fontweight="bold", pad=10, color=_SLATE,
    )

    path = out_dir / "06_node_dispatch_pies.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 07: Full hourly dispatch — ES / FR / PT ─────────────────────────────

def _plot_hourly_dispatch(n, out_dir):
    """Full-period hourly dispatch stacked fill for ES, FR, PT."""
    es_price = _mean_es_price(n)
    panels   = [("ES", "Spain"), ("FR", "France"), ("PT", "Portugal")]
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)

    for ax, (pfx, label) in zip(axes, panels):
        dispatch = _dispatch_by_carrier(n, pfx)
        if dispatch.empty:
            ax.set_title(f"{label} — no data")
            continue
        _stack_fill(ax, dispatch, COLORS)
        neg = dispatch.clip(upper=0)
        if neg.sum().sum() < -0.1:
            ax.fill_between(dispatch.index, neg.sum(axis=1).values, 0,
                            step="post", color="lightblue", alpha=0.5, label="Charging")
        ax.set_ylabel("MW")
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.20)
        h, lbl = ax.get_legend_handles_labels()
        if h:
            ax.legend(h[::-1], lbl[::-1], loc="upper right", fontsize=7, ncol=2,
                      framealpha=0.80)
        if pfx == "ES" and len(es_price):
            ax2 = ax.twinx()
            ax2.plot(es_price.index, es_price.values, color="black", lw=1.2,
                     alpha=0.75, label="ES price")
            ax2.set_ylabel("EUR/MWh", color="black")
            ax2.tick_params(axis="y", labelcolor="black")
            ax2.legend(loc="upper left", fontsize=7, framealpha=0.80)

    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].tick_params(axis="x", rotation=20)
    fig.suptitle("Hourly Dispatch by Country", fontsize=12)
    fig.tight_layout()

    path = out_dir / "07_hourly_dispatch.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 08: Temporal comparison — model vs real dispatch ────────────────────

def _draw_stacked_bars(ax, grp_df, title, ylabel):
    """Stacked bar chart of grouped dispatch (positive + negative) with date x-axis."""
    x = np.arange(len(grp_df))
    bottom_pos = np.zeros(len(grp_df))
    bottom_neg = np.zeros(len(grp_df))
    legend_added = set()
    for grp in _GROUPS:
        if grp not in grp_df:
            continue
        vals  = grp_df[grp].values
        pos   = np.clip(vals, 0, None)
        neg   = np.clip(vals, None, 0)
        color = _GROUP_COLORS[grp]
        if pos.sum() > 0.01:
            label = grp if grp not in legend_added else "_nolegend_"
            ax.bar(x, pos, bottom=bottom_pos, width=0.75,
                   color=color, label=label, alpha=0.90)
            legend_added.add(grp)
            bottom_pos += pos
        if neg.sum() < -0.01:
            label = grp if grp not in legend_added else "_nolegend_"
            ax.bar(x, neg, bottom=bottom_neg, width=0.75,
                   color=color, label=label, alpha=0.70)
            legend_added.add(grp)
            bottom_neg += neg
    if bottom_neg.min() < 0:
        ax.axhline(0, color="black", lw=0.6)
    date_labels = [str(d.date()) for d in grp_df.index]
    ax.set_xticks(x)
    ax.set_xticklabels(date_labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)


def _plot_temporal_comparison(n, real_daily, n_days, out_dir):
    """Model vs real dispatch at daily / weekly / monthly resolution."""
    es_hourly = _dispatch_by_carrier(n, "ES")
    if es_hourly.empty:
        log.warning("Temporal comparison: no ES dispatch — skipping")
        return

    resolutions = [("daily", _to_daily_gwh(es_hourly), real_daily, "Daily GWh")]
    if n_days >= 7:
        rw = real_daily.resample("W-MON", label="left", closed="left").sum() if real_daily is not None else None
        resolutions.append(("weekly", _to_weekly_gwh(es_hourly), rw, "Weekly GWh"))
    if n_days >= 28:
        rm = real_daily.resample("ME").sum() if real_daily is not None else None
        resolutions.append(("monthly", _to_monthly_gwh(es_hourly), rm, "Monthly GWh"))

    for res_name, model_df, real_df, ylabel in resolutions:
        model_grp = _group_dispatch(model_df)
        real_grp  = _group_dispatch(real_df) if real_df is not None else None

        n_panels = 2 if real_grp is not None else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels + 2, 5), sharey=True)
        if n_panels == 1:
            axes = [axes]

        _draw_stacked_bars(axes[0], model_grp, f"Model — Spain {res_name.capitalize()}", ylabel)
        if real_grp is not None:
            _draw_stacked_bars(axes[1], real_grp, f"Real (REE) — Spain {res_name.capitalize()}", ylabel)

        fig.suptitle(f"Spain Dispatch: Model vs Real — {res_name.capitalize()}", fontsize=12)
        fig.tight_layout()
        path = out_dir / f"08_{res_name}_comparison.png"
        fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)


# ─── Plot 08b/08c: Temporal dispatch — PT and FR (model vs real) ─────────────

_COUNTRY_NAMES = {"ES": "Spain", "FR": "France", "PT": "Portugal"}

# Carrier name mappings from real-data sources to model carrier names
_FR_CARRIER_MAP = {
    "Nuclear":                "nuclear",
    "Hydropower":             "hydro",
    "Wind":                   "onwind",
    "Solar":                  "solar",
    "Fossil-fired thermal":   "CCGT",
    "Renewable thermal and waste": "biomass",
    "Autre":                  None,   # skip — misc imports/other
    "Total generation":       None,
}
_PT_CARRIER_MAP = {
    "Nuclear":        "nuclear",
    "Hydro":          "hydro",
    "Wind":           "onwind",
    "Solar":          "solar",
    "Gas":            "CCGT",
    "Bioenergy":      "biomass",
    "Coal":           "coal",
    "Other fossil":   "diesel",
    "Total generation": None,
}

# Path to the interconnector_analysis folder (relative to Analysis/)
_IC_DATA_DIR = Path(__file__).parent / "interconnector_analysis"


def _load_real_fr_dispatch_monthly(snapshots):
    """Parse FR_monthy_gen_breakdown.csv → GWh/month DataFrame by model carrier."""
    path = _IC_DATA_DIR / "FR_monthy_gen_breakdown.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = [c.strip().strip('"') for c in df.columns]
        # Value column uses tab as decimal separator (French locale): "35\t925..." → 35.925
        val_col = [c for c in df.columns if "Valeur" in c or "TWh" in c][0]
        def _parse_fr_val(v):
            if pd.isna(v):
                return float("nan")
            s = str(v).replace("\t", ".").strip()
            try:
                return float(s)
            except ValueError:
                return float("nan")
        df["_gwh"] = df[val_col].apply(_parse_fr_val) * 1000.0   # TWh → GWh
        df["_carrier"] = df["Filière"].str.strip().map(_FR_CARRIER_MAP)
        df = df[df["_carrier"].notna()].copy()
        # Date: "1/1/24" = day/month/year
        df["_date"] = pd.to_datetime(df["Date"].str.strip(), format="%d/%m/%y", errors="coerce")
        df = df.dropna(subset=["_date"])
        df = df.set_index("_date")
        # Filter to simulation window months
        start, end = snapshots[0].to_pydatetime(), snapshots[-1].to_pydatetime()
        df = df[(df.index >= start.replace(day=1)) & (df.index <= end)]
        pivot = df.pivot_table(index=df.index, columns="_carrier", values="_gwh", aggfunc="sum")
        return pivot
    except Exception as exc:
        log.warning("Could not load FR real dispatch: %s", exc)
        return pd.DataFrame()


def _load_real_pt_dispatch_monthly(snapshots):
    """Parse PTGAL_monthy_gen_breakdown.csv → GWh/month DataFrame by model carrier."""
    path = _IC_DATA_DIR / "PTGAL_monthy_gen_breakdown.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["_carrier"] = df["series"].map(_PT_CARRIER_MAP)
        df = df[df["_carrier"].notna()].copy()
        df["_date"] = pd.to_datetime(df["full_date"], errors="coerce")
        df = df.dropna(subset=["_date"])
        df["_gwh"] = df["generation_twh"] * 1000.0   # TWh → GWh
        df = df.set_index("_date")
        start, end = snapshots[0].to_pydatetime(), snapshots[-1].to_pydatetime()
        df = df[(df.index >= start.replace(day=1)) & (df.index <= end)]
        pivot = df.pivot_table(index=df.index, columns="_carrier", values="_gwh", aggfunc="sum")
        return pivot
    except Exception as exc:
        log.warning("Could not load PT real dispatch: %s", exc)
        return pd.DataFrame()


def _plot_country_temporal_dispatch(n, n_days, out_dir):
    """Daily + weekly stacked dispatch for PT and FR — model vs real side-by-side."""
    real_loaders = {
        "FR": _load_real_fr_dispatch_monthly,
        "PT": _load_real_pt_dispatch_monthly,
    }
    for prefix, code in [("PT", "08b"), ("FR", "08c")]:
        hourly = _dispatch_by_carrier(n, prefix)
        if hourly.empty:
            log.warning("Country temporal dispatch [%s]: no dispatch data — skipping", prefix)
            continue
        cname  = _COUNTRY_NAMES.get(prefix, prefix)
        real_m = real_loaders[prefix](n.snapshots)   # GWh/month, may be empty

        resolutions = [("daily", _to_daily_gwh(hourly), "Daily GWh")]
        if n_days >= 7:
            resolutions.append(("weekly", _to_weekly_gwh(hourly), "Weekly GWh"))
        # monthly excluded — real data is already monthly so weekly is the right comparison level

        for res_name, model_df, ylabel in resolutions:
            model_grp = _group_dispatch(model_df)

            # Real panel: group monthly GWh by carrier group (available for all resolutions)
            real_grp = None
            if not real_m.empty:
                real_grp = _group_dispatch(real_m)

            # Normalise real monthly → weekly-average GWh so both panels share y-axis
            real_grp_norm = None
            if real_grp is not None:
                weeks_per_month = pd.Series(
                    {ts: (pd.Timestamp(ts.year, ts.month, 1) +
                          pd.offsets.MonthEnd(0)).day / 7.0
                     for ts in real_grp.index},
                    dtype=float,
                )
                real_grp_norm = real_grp.div(weeks_per_month, axis=0)

            n_panels = 2 if real_grp_norm is not None else 1
            panel_w  = max(10, len(model_df) * 0.55 + 3)
            fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, 5),
                                     sharey=(n_panels == 2))
            if n_panels == 1:
                axes = [axes]

            _draw_stacked_bars(axes[0], model_grp,
                               f"Model — {cname} {res_name.capitalize()}", ylabel)
            if real_grp_norm is not None:
                _draw_stacked_bars(axes[1], real_grp_norm,
                                   f"Real — {cname} (monthly ÷ weeks)", ylabel)

            fig.suptitle(
                f"{cname} Dispatch: Model ({res_name}) vs Real (monthly avg, same GWh/week scale)",
                fontsize=11,
            )
            fig.tight_layout()
            path = out_dir / f"{code}_{res_name}_{prefix.lower()}.png"
            fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved %s", path)


# ─── Plot 09: Merit order — Spain detail + ES/FR/PT combined ─────────────────

def _build_merit_order(n, country_prefix):
    buses = n.buses.index[n.buses.index.str.startswith(country_prefix)]
    gens = n.generators.loc[
        n.generators["bus"].isin(buses),
        ["p_nom", "marginal_cost", "carrier"]
    ].copy()
    gens["name"] = gens.index
    return gens.sort_values("marginal_cost", ascending=True)


def _draw_merit_staircase(ax, mo, lw_h=3.0, lw_v=1.5, alpha=1.0, color_override=None):
    prev_cum, prev_mc = 0.0, 0.0
    for gen_name, row in mo.iterrows():
        mc, p, carrier = row["marginal_cost"], row["p_nom"], row["carrier"]
        color = (color_override or {}).get(gen_name) or COLORS.get(carrier, "#BBBBBB")
        ax.plot([prev_cum, prev_cum],     [prev_mc, mc], color=color, lw=lw_v, alpha=alpha)
        ax.plot([prev_cum, prev_cum + p], [mc, mc],      color=color, lw=lw_h,
                alpha=alpha, solid_capstyle="butt")
        prev_cum += p
        prev_mc   = mc
    return prev_cum


def _plot_merit_order_es(n, out_dir):
    """Spain merit order staircase with average dispatch overlay."""
    mo_full = _build_merit_order(n, "ES")
    if mo_full.empty:
        log.warning("Merit order ES: no generators — skipping")
        return

    mo = mo_full[mo_full["carrier"] != "load_shedding"].copy()

    dispatch = {
        g: float(n.generators_t.p[g].mean())
        for g in mo["name"] if g in n.generators_t.p.columns
    }

    fig, ax = plt.subplots(figsize=(14, 7))

    # Background price bands with labelled legend
    band_defs = [
        (0, 30, "#2ECC71",  "Low MC"),
        (30, 80, "#1E8BC3", "Mid MC"),
        (80, 130, "#E67E22","High MC"),
        (130, 220, "#C0392B","Very high MC"),
    ]
    for lo, hi, color, _ in band_defs:
        ax.axhspan(lo, hi, alpha=0.05, color=color, zorder=0)

    # Build CCGT tier color map: continuous gradient from light coral → deep red
    ccgt_rows = mo[mo["carrier"] == "CCGT"].sort_values("marginal_cost")
    color_override = {}
    if not ccgt_rows.empty:
        n_ccgt = len(ccgt_rows)
        # Use a proper continuous colormap from light pink to dark crimson
        ccgt_cmap = plt.cm.ScalarMappable(
            norm=plt.Normalize(0, max(n_ccgt - 1, 1)),
            cmap=LinearSegmentedColormap.from_list(
                "ccgt_tier", ["#FFCCCC", "#FF6B6B", "#C0392B", "#7B241C"]
            )
        )
        for i, gen_name in enumerate(ccgt_rows.index):
            color_override[gen_name] = ccgt_cmap.to_rgba(i)[:3]  # RGB tuple

    x_max = _draw_merit_staircase(ax, mo, color_override=color_override)

    # Dispatch overlay (dashed, same colour per carrier)
    mo_d = mo.copy()
    mo_d["dispatch_mw"] = mo_d["name"].map(dispatch).fillna(0.0)
    prev_cum, prev_mc = 0.0, 0.0
    for _, row in mo_d.iterrows():
        mc, p_d = row["marginal_cost"], row["dispatch_mw"]
        if p_d < 0.1:
            continue
        color = COLORS.get(row["carrier"], "#BBBBBB")
        ax.plot([prev_cum, prev_cum],       [prev_mc, mc], color=color, lw=0.8, alpha=0.55)
        ax.plot([prev_cum, prev_cum + p_d], [mc, mc],      color=color, lw=1.8, ls="--", alpha=0.75)
        prev_cum += p_d
        prev_mc   = mc

    # Dynamic reference lines from actual solved time-varying MCs
    _tv = n.generators_t.marginal_cost
    _es_ccgt_tv = [g for g in n.generators.index
                   if n.generators.loc[g, "carrier"] == "CCGT"
                   and n.generators.loc[g, "bus"].startswith("ES")
                   and g in _tv.columns]
    if _es_ccgt_tv:
        _mc_vals = _tv[_es_ccgt_tv].values.ravel()
        _ccgt_lo_r = float(np.nanpercentile(_mc_vals, 10))
        _ccgt_hi_r = float(np.nanpercentile(_mc_vals, 90))
    else:
        _ccgt_lo_r, _ccgt_hi_r = 82.0, 100.0
    _es_flex_tv = [g for g in n.generators.index
                   if n.generators.loc[g, "carrier"] == "CCGT_flex"
                   and n.generators.loc[g, "bus"].startswith("ES")
                   and g in _tv.columns]
    _flex_r = float(np.nanmean(_tv[_es_flex_tv].values)) if _es_flex_tv else 115.0
    _es_ocgt_tv = [g for g in n.generators.index
                   if n.generators.loc[g, "carrier"] == "OCGT"
                   and n.generators.loc[g, "bus"].startswith("ES")
                   and g in _tv.columns]
    _ocgt_r = float(np.nanmean(_tv[_es_ocgt_tv].values)) if _es_ocgt_tv else 128.0

    for price, label in [
        (_ccgt_lo_r, f"CCGT T1 ~€{_ccgt_lo_r:.0f}"),
        (_ccgt_hi_r, f"CCGT T3 ~€{_ccgt_hi_r:.0f}"),
        (_flex_r,    f"CCGT flex ~€{_flex_r:.0f}"),
        (_ocgt_r,    f"OCGT ~€{_ocgt_r:.0f}"),
    ]:
        ax.axhline(price, color="#999999", ls=":", lw=1.0, alpha=0.75)
        ax.text(x_max * 0.97, price + 2, label, fontsize=8, color="#555555",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    # Legend: replace generic CCGT with 3 tier entries + background band legend
    carriers = [c for c in mo["carrier"].unique() if c != "CCGT"]
    handles  = [plt.Line2D([0],[0], color=COLORS.get(c,"#BBB"), lw=3, label=c) for c in carriers]
    if not ccgt_rows.empty:
        handles += [
            plt.Line2D([0],[0], color="#FF9999", lw=3, label="CCGT T1 (efficient)"),
            plt.Line2D([0],[0], color="#FF6B6B", lw=3, label="CCGT T2 (standard)"),
            plt.Line2D([0],[0], color="#C0392B", lw=3, label="CCGT T3 (flex)"),
        ]
    handles += [plt.Line2D([0],[0], color="grey", lw=1.8, ls="--", label="Avg dispatch (dashed)")]
    # Add background band legend entries
    for lo, hi, color, label in band_defs:
        handles.append(plt.Rectangle((0,0), 1, 1, color=color, alpha=0.2, label=label))
    ax.legend(handles=handles, loc="upper left", fontsize=7.5, framealpha=0.93, ncol=2)

    total_gw = mo["p_nom"].sum() / 1000
    disp_gw  = sum(dispatch.values()) / 1000
    ax.text(0.98, 0.97, f"Installed: {total_gw:.1f} GW\nAvg dispatched: {disp_gw:.1f} GW",
            transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

    ax.set_xlabel("Cumulative Installed Capacity (MW)")
    ax.set_ylabel("Marginal Cost (EUR/MWh)")
    ax.set_title("Spain — Merit Order Supply Stack")
    ax.grid(axis="y", alpha=0.3, ls=":")
    y_top = max(float(mo["marginal_cost"].max()) * 1.12, 220.0)
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(0, y_top)

    path = out_dir / "09_merit_order_spain.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _plot_merit_order_combined(n, out_dir):
    """Merit order staircases for ES / FR / PT side by side."""
    countries = [("ES", "Spain"), ("FR", "France"), ("PT", "Portugal")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    all_carriers = set()

    band_defs = [
        (0, 30, "#2ECC71",  "Low MC"),
        (30, 80, "#1E8BC3", "Mid MC"),
        (80, 130, "#E67E22","High MC"),
        (130, 220, "#C0392B","Very high MC"),
    ]

    for ax, (pfx, label) in zip(axes, countries):
        mo = _build_merit_order(n, pfx)
        if mo.empty:
            ax.set_title(f"{label} — no data")
            continue
        all_carriers.update(mo["carrier"].unique())
        for lo, hi, color, _ in band_defs:
            ax.axhspan(lo, hi, alpha=0.05, color=color, zorder=0)

        # CCGT tier gradient for ES only (FR/PT may not have CCGT tiers)
        color_override = {}
        if pfx == "ES":
            ccgt_rows = mo[mo["carrier"] == "CCGT"].sort_values("marginal_cost")
            if not ccgt_rows.empty:
                n_ccgt = len(ccgt_rows)
                ccgt_cmap = plt.cm.ScalarMappable(
                    norm=plt.Normalize(0, max(n_ccgt - 1, 1)),
                    cmap=LinearSegmentedColormap.from_list(
                        "ccgt_tier", ["#FFCCCC", "#FF6B6B", "#C0392B", "#7B241C"]
                    )
                )
                for i, gen_name in enumerate(ccgt_rows.index):
                    color_override[gen_name] = ccgt_cmap.to_rgba(i)[:3]

        x_max = _draw_merit_staircase(ax, mo, lw_h=2.5, lw_v=1.0,
                                       color_override=color_override if color_override else None)
        ax.set_xlabel("Cumulative MW")
        ax.set_title(f"{label}  ({mo['p_nom'].sum() / 1000:.1f} GW)", fontsize=11)
        ax.set_xlim(0, x_max * 1.02)
        ax.grid(axis="y", alpha=0.2, ls=":")

    axes[0].set_ylabel("Marginal Cost (EUR/MWh)")
    axes[0].set_ylim(0, 230)

    # Shared legend at bottom with carrier colors + band labels
    carrier_order = ["nuclear","hydro","PHS","ror","biomass","coal",
                     "CCGT","CCGT_flex","OCGT","solar","onwind","offwind",
                     "diesel","oil","other"]
    handles = []
    for c in carrier_order:
        if c in all_carriers:
            handles.append(plt.Line2D([0],[0], color=COLORS.get(c,"#BBB"), lw=3, label=c))
    for _, _, color, label in band_defs:
        handles.append(plt.Rectangle((0,0), 1, 1, color=color, alpha=0.2, label=label))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8),
               fontsize=7.5, framealpha=0.93, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Merit Order Comparison — ES / FR / PT", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)

    path = out_dir / "10_merit_order_combined.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 11: First-week model vs REE hourly actuals ─────────────────────────

_REE_TO_CARRIER = {
    "Nuclear":         "nuclear",
    "Hydro_Reservoir": "hydro",
    "Hydro_River":     "ror",
    "Solar_PV":        "solar",
    "Wind":            "onwind",
    "CCGT":            "CCGT",
    "Coal":            "coal",
    "Cogeneration":    "biomass",
    "Other":           "other",
}


def _plot_week_vs_ree_hourly(n, out_dir):
    """3-panel figure: model stack (top) | REE actual stack (mid) | solar zoom (bottom)."""
    ree_path = ROOT / "Analysis/data/spain_actual_generation_2024.csv"
    if not ree_path.exists():
        log.warning("REE hourly data not found at %s — skipping plot 11", ree_path)
        return

    # Model dispatch — Spain, middle 7 days of simulation
    mid           = len(n.snapshots) // 2
    week_start    = max(0, mid - 84)          # 3.5 days before centre
    week_end_idx  = min(week_start + 168, len(n.snapshots))
    snap_w        = n.snapshots[week_start:week_end_idx]
    es_disp       = _dispatch_by_carrier(n, "ES").reindex(snap_w).fillna(0.0)

    # REE hourly data
    ree_raw = pd.read_csv(ree_path, parse_dates=["timestamp"], index_col="timestamp")
    ree_raw.index = ree_raw.index.tz_localize(None) if ree_raw.index.tz is not None else ree_raw.index
    ree_slice = ree_raw.loc[snap_w[0]:snap_w[-1]].reindex(snap_w, method="nearest")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1], "hspace": 0.08})
    ax_mod, ax_ree, ax_sol = axes

    t0 = snap_w[0].strftime("%d %b")
    t1 = snap_w[-1].strftime("%d %b %Y")

    # ── top: model stacked ──
    _, _ = _stack_fill(ax_mod, es_disp, COLORS)
    ax_mod.set_ylabel("Generation (MW)")
    ax_mod.set_title(f"Model vs REE Hourly — Middle Week  [{t0}–{t1}]", fontsize=12, pad=8)
    h, l = ax_mod.get_legend_handles_labels()
    ax_mod.legend(h[::-1], l[::-1], loc="upper right", ncol=3, fontsize=7, framealpha=0.93)
    ax_mod.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f} GW" if x >= 500 else f"{x:.0f}"))
    _midnight_lines(ax_mod, snap_w)
    ax_mod.text(0.01, 0.97, "Model (Spain)", transform=ax_mod.transAxes,
                fontsize=9, va="top", color="#333333", fontweight="bold")

    # ── mid: REE stacked — map to model carriers then use _stack_fill for consistent order ──
    ree_by_carrier = {}
    for col in ree_slice.columns:
        carrier = _REE_TO_CARRIER.get(col, "other")
        vals    = ree_slice[col].fillna(0.0)
        ree_by_carrier[carrier] = ree_by_carrier.get(carrier, pd.Series(0.0, index=ree_slice.index)) + vals
    ree_df = pd.DataFrame(ree_by_carrier, index=ree_slice.index)
    _, _ = _stack_fill(ax_ree, ree_df, COLORS)
    ax_ree.set_ylabel("Generation (MW)")
    h, l = ax_ree.get_legend_handles_labels()
    ax_ree.legend(h[::-1], l[::-1], loc="upper right", ncol=3, fontsize=7, framealpha=0.93)
    ax_ree.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1000:.0f} GW" if x >= 500 else f"{x:.0f}"))
    _midnight_lines(ax_ree, snap_w)
    ax_ree.text(0.01, 0.97, "REE Actual", transform=ax_ree.transAxes,
                fontsize=9, va="top", color="#333333", fontweight="bold")

    # ── bottom: solar comparison ──
    model_solar = es_disp.get("solar", pd.Series(0.0, index=snap_w))
    ree_solar   = ree_slice.get("Solar_PV", pd.Series(0.0, index=snap_w)).fillna(0.0)
    ax_sol.fill_between(snap_w, model_solar.values, step="post",
                        color=COLORS["solar"], alpha=0.65, label="Model solar")
    ax_sol.plot(ree_slice.index, ree_solar.values, drawstyle="steps-post",
                color="#B7950B", lw=1.4, ls="--", label="REE Solar_PV")
    ax_sol.set_ylabel("Solar (MW)")
    ax_sol.legend(loc="upper right", fontsize=8)
    ax_sol.set_ylim(bottom=-50)
    ax_sol.axhline(0, color="#999999", lw=0.6)
    _midnight_lines(ax_sol, snap_w)
    ax_sol.text(0.01, 0.97, "Solar detail", transform=ax_sol.transAxes,
                fontsize=9, va="top", color="#333333", fontweight="bold")

    ax_sol.xaxis.set_major_locator(mdates.HourLocator(byhour=0))
    ax_sol.xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
    ax_sol.set_xlim(snap_w[0], snap_w[-1])

    path = out_dir / "11_week_vs_ree_hourly.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Plot 12: Interconnector flows — ES↔FR and ES↔PT ─────────────────────────

def _load_real_fr_flows(csv_path, snapshots):
    """Load ENTSO-E FR↔ES hourly flows (FR_to_ES, ES_to_FR columns, UTC index).

    Net ES import = FR_to_ES - ES_to_FR.  Returns series or None.
    """
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        net = (df["FR_to_ES"] - df["ES_to_FR"]).reindex(snapshots)
        return net
    except Exception as exc:
        log.warning("Could not load real FR flows from %s: %s", csv_path, exc)
        return None


def _load_real_balance_csv(csv_path, snapshots):
    """Load an interconnector_analysis balance CSV (REE format).

    Format: id, name, geoid, geoname, value, datetime
    - value: plain decimal float (new) or European notation "2.143,291" (old)
    - datetime is CET (UTC+1) — converted to UTC for alignment with model snapshots
    - Sign convention: positive = ES net importing, negative = ES net exporting

    Returns series aligned to snapshots, or None if file missing/unparseable.
    """
    try:
        df = pd.read_csv(csv_path).dropna(subset=["datetime", "value"])
        df = df.drop_duplicates(subset=["datetime"])
        raw = df["value"]
        if pd.api.types.is_numeric_dtype(raw):
            df["value_mw"] = raw.astype(float)
        else:
            df["value_mw"] = (
                raw.astype(str)
                   .str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False)
                   .astype(float)
            )
        df.index = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
        net = df["value_mw"].reindex(snapshots, method="nearest",
                                     tolerance=pd.Timedelta("75min"))
        return net
    except Exception as exc:
        log.warning("Could not load real balance flows from %s: %s", csv_path, exc)
        return None


def _draw_ic_panels(axes_col, net_mw, snap_w, color, col_title):
    """Draw the 4 time-resolution panels for one interconnector series into axes_col."""

    def _bar_colors(vals):
        return [color if v >= 0 else "#d62728" for v in vals]

    # Row 0: hourly MW (first week)
    ax = axes_col[0]
    hw = net_mw.reindex(snap_w).fillna(0.0)
    ax.fill_between(snap_w, hw.values, step="post",
                    where=hw.values >= 0, color=color,     alpha=0.70, label="ES import")
    ax.fill_between(snap_w, hw.values, step="post",
                    where=hw.values <  0, color="#d62728", alpha=0.70, label="ES export")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title(f"{col_title}\nFirst week — hourly MW", fontsize=9)
    ax.set_ylabel("MW")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.tick_params(axis="x", rotation=25, labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.20)

    # Row 1: daily GWh
    ax = axes_col[1]
    daily = net_mw.resample("D").sum() / 1000.0
    xd    = np.arange(len(daily))
    ax.bar(xd, daily.values, color=_bar_colors(daily.values), alpha=0.85, width=0.8)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Daily GWh", fontsize=9)
    ax.set_ylabel("GWh")
    step = max(1, len(daily) // 8)
    ax.set_xticks(xd[::step])
    ax.set_xticklabels([str(d.date()) for d in daily.index[::step]],
                       rotation=35, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.20)

    # Row 2: weekly GWh
    ax = axes_col[2]
    weekly = net_mw.resample("W-MON", label="left", closed="left").sum() / 1000.0
    xw     = np.arange(len(weekly))
    ax.bar(xw, weekly.values, color=_bar_colors(weekly.values), alpha=0.85, width=0.6)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Weekly GWh", fontsize=9)
    ax.set_ylabel("GWh")
    ax.set_xticks(xw)
    ax.set_xticklabels([str(d.date()) for d in weekly.index],
                       rotation=35, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.20)

    # Row 3: monthly GWh
    ax = axes_col[3]
    monthly = net_mw.resample("ME").sum() / 1000.0
    xm      = np.arange(len(monthly))
    ax.bar(xm, monthly.values, color=_bar_colors(monthly.values), alpha=0.85, width=0.5)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_title("Monthly GWh", fontsize=9)
    ax.set_ylabel("GWh")
    ax.set_xticks(xm)
    ax.set_xticklabels([d.strftime("%b %Y") for d in monthly.index],
                       rotation=35, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.20)

    import_patch = mpatches.Patch(color=color,     label="ES net import")
    export_patch = mpatches.Patch(color="#d62728", label="ES net export")
    axes_col[3].legend(handles=[import_patch, export_patch], fontsize=7, loc="upper right")


def _plot_interconnector_flows(n, cfg, out_dir):
    """One PNG per border: ES↔FR and ES↔PT, each with 4 time-resolution panels.

    Where real ENTSO-E data is available, a side-by-side comparison column is added.
    Positive = ES importing; negative = ES exporting.
    """
    val_cfg = cfg.get("validation", {})

    fr_net = _net_import_topo(n, "FR")
    pt_net = _net_import_topo(n, "PT")

    if fr_net.eq(0).all() and pt_net.eq(0).all():
        log.warning("Interconnector flows: no link/line time-series found — skipping")
        return

    week_end = min(7 * 24, len(n.snapshots))
    snap_w   = n.snapshots[:week_end]

    # ── Load real flow data ───────────────────────────────────────────────────
    fr_csv = val_cfg.get("real_flows_fr_csv")
    pt_csv = val_cfg.get("real_flows_pt_csv")
    real_fr = _load_real_fr_flows(fr_csv, n.snapshots) if fr_csv else None
    real_pt = _load_real_balance_csv(pt_csv, n.snapshots) if pt_csv else None

    # ── One figure per border ─────────────────────────────────────────────────
    for label, model_mw, real_mw, color, fname, suptitle in [
        ("ES ↔ FR", fr_net, real_fr, "#1f77b4", "12a_interconnector_flows_FR.png",
         "ES ↔ FR  —  Net Flows  (positive = ES imports)"),
        ("ES ↔ PT", pt_net, real_pt, "#2ca02c", "12b_interconnector_flows_PT.png",
         "ES ↔ PT  —  Net Flows  (positive = ES imports)"),
    ]:
        n_cols = 2 if real_mw is not None else 1
        fig, axes = plt.subplots(4, n_cols, figsize=(10 * n_cols, 18),
                                 gridspec_kw={"hspace": 0.55, "wspace": 0.35})
        if n_cols == 1:
            axes = axes[:, None]
        fig.suptitle(suptitle, fontsize=12, y=1.01)
        _draw_ic_panels(axes[:, 0], model_mw, snap_w, color, "Model")
        if real_mw is not None:
            _draw_ic_panels(axes[:, 1], real_mw, snap_w, color, "REE / ENTSO-E Actual")

        # Equalise y-axes row-by-row: symmetric ±max so import/export are comparable
        for row in range(4):
            row_axes = axes[row, :]
            ylims = [ax.get_ylim() for ax in row_axes]
            max_abs = max(abs(lim) for yl in ylims for lim in yl)
            for ax in row_axes:
                ax.set_ylim(-max_abs, max_abs)

        path = out_dir / fname
        fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", path)


# ─── Plot 18: IC export/import technology composition ─────────────────────────

def _country_dispatch_by_carrier(n, prefix: str) -> dict:
    """Return {carrier: pd.Series(hourly MW)} for all generators+storage in country prefix."""
    out = {}
    gen_idx = n.generators.index[n.generators.bus.str.startswith(prefix)]
    if gen_idx.size > 0 and not n.generators_t.p.empty:
        p_gen = n.generators_t.p.reindex(columns=gen_idx, fill_value=0.0)
        for carrier, grp in p_gen.groupby(n.generators.loc[gen_idx, "carrier"], axis=1):
            out[carrier] = grp.sum(axis=1)

    su_idx = n.storage_units.index[n.storage_units.bus.str.startswith(prefix)]
    if su_idx.size > 0 and "p_dispatch" in n.storage_units_t.keys():
        p_su = n.storage_units_t.p_dispatch.reindex(columns=su_idx, fill_value=0.0)
        for carrier, grp in p_su.groupby(n.storage_units.loc[su_idx, "carrier"], axis=1):
            out[carrier] = out.get(carrier, pd.Series(0.0, index=n.snapshots)) + grp.sum(axis=1)
    return out


def _plot_ic_tech_composition(n, out_dir):
    """IC export/import tech composition — what carrier drives cross-border flows.

    For FR and PT: shows the generation mix of the source country in hours when
    it exports heavily to Spain vs. balanced vs. Spain exporting to it.
    Electricity is fungible; this is a proxy based on energy balance.
    Saves 18_ic_tech_composition.png.
    """
    import matplotlib.gridspec as gridspec

    THRESH = 500.0   # MW threshold for export/import classification
    CARRIERS = ["nuclear", "hydro", "ror", "onwind", "offwind-ac",
                "offwind-float", "solar", "CCGT", "CCGT_flex", "OCGT",
                "coal", "biomass", "CCGT_must_run", "other"]

    rows = [("FR", "FR→ES (nuclear?)"), ("PT", "PT→ES (hydro?)")]
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    for row_idx, (country, subtitle) in enumerate(rows):
        dispatch = _country_dispatch_by_carrier(n, country)
        ic_net   = _net_import_topo(n, country)   # positive = ES imports

        if not dispatch or ic_net.eq(0).all():
            for col_idx in range(3):
                ax = fig.add_subplot(gs[row_idx, col_idx])
                ax.text(0.5, 0.5, f"No {country} data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.axis("off")
            continue

        masks = {
            f"{country}→ES\n(>500 MW)": ic_net > THRESH,
            "Balanced\n(±500 MW)":      ic_net.abs() <= THRESH,
            f"ES→{country}\n(>500 MW)": ic_net < -THRESH,
        }
        carriers_present = [c for c in CARRIERS if c in dispatch]

        # ── Panel 0: weekly dispatch stacked bars + IC flow overlay ──────────
        ax0 = fig.add_subplot(gs[row_idx, 0])
        ax0r = ax0.twinx()
        snap = n.snapshots
        week_ms = pd.Timedelta("7D")
        freq = "W" if (snap[-1] - snap[0]) > week_ms else "D"
        bottoms = pd.Series(0.0, index=pd.date_range(snap[0], snap[-1], freq=freq))

        for carrier in carriers_present:
            weekly = dispatch[carrier].resample(freq).mean() / 1e3  # GW
            weekly = weekly.reindex(bottoms.index, fill_value=0.0)
            color = COLORS.get(carrier, "#999")
            ax0.bar(bottoms.index, weekly, bottom=bottoms, color=color,
                    width=pd.Timedelta("6D") if freq == "W" else pd.Timedelta("22H"),
                    label=carrier, alpha=0.85)
            bottoms += weekly

        ic_weekly = ic_net.resample(freq).mean() / 1e3
        ic_weekly = ic_weekly.reindex(bottoms.index, fill_value=0.0)
        ax0r.plot(bottoms.index, ic_weekly, color=COUNTRY_COLORS.get(country, "#333"),
                  lw=1.8, label="IC net flow", zorder=5)
        ax0r.axhline(0, color="#888", lw=0.6, ls="--")
        ax0r.set_ylabel("IC net flow (GW, +ve=→ES)", fontsize=8,
                        color=COUNTRY_COLORS.get(country, "#333"))
        ax0.set_ylabel("Generation (GW)", fontsize=8)
        ax0.set_title(f"{country} generation + IC flow", fontsize=9)
        ax0.tick_params(axis="x", rotation=30)
        handles = [mpatches.Patch(color=COLORS.get(c, "#999"), label=c) for c in carriers_present]
        ax0.legend(handles=handles, fontsize=6, loc="upper left", ncol=2,
                   framealpha=0.85)

        # ── Panel 1: categorical mean dispatch by flow direction ─────────────
        ax1 = fig.add_subplot(gs[row_idx, 1])
        group_labels = list(masks.keys())
        bottom_arr = np.zeros(len(group_labels))
        for carrier in carriers_present:
            vals = [
                float(dispatch[carrier][m].mean()) if m.any() else 0.0
                for m in masks.values()
            ]
            color = COLORS.get(carrier, "#999")
            ax1.barh(group_labels, vals, left=bottom_arr, color=color,
                     label=carrier, height=0.55, alpha=0.88)
            bottom_arr += np.array(vals)
        counts = [int(m.sum()) for m in masks.values()]
        for i, (lbl, cnt) in enumerate(zip(group_labels, counts)):
            ax1.text(bottom_arr[i] * 1.01, i, f"n={cnt}h",
                     va="center", fontsize=7, color=_SLATE)
        ax1.set_xlabel("Mean hourly dispatch (MW)", fontsize=8)
        ax1.set_title(f"{country} mix by flow direction", fontsize=9)
        ax1.legend(fontsize=6, loc="lower right", ncol=2, framealpha=0.85)

        # ── Panel 2: scatter — dominant carrier vs IC net flow ───────────────
        ax2 = fig.add_subplot(gs[row_idx, 2])
        dom_carrier = "nuclear" if country == "FR" else "hydro"
        if dom_carrier in dispatch:
            dom_mw = dispatch[dom_carrier]
            hour_of_day = snap.hour
            sc = ax2.scatter(dom_mw, ic_net, c=hour_of_day, cmap="plasma",
                             s=4, alpha=0.35, rasterized=True)
            plt.colorbar(sc, ax=ax2, label="Hour of day", shrink=0.75)
            # regression line
            valid = dom_mw.dropna().index.intersection(ic_net.dropna().index)
            if len(valid) > 20:
                x, y = dom_mw[valid].values, ic_net[valid].values
                # Guard against degenerate data (all same value, NaN, inf) that crashes polyfit
                x_finite = np.isfinite(x).all() and np.isfinite(y).all()
                x_varies = np.ptp(x) > 1e-6 and np.ptp(y) > 1e-6
                if x_finite and x_varies:
                    try:
                        m_coef, b_coef = np.polyfit(x, y, 1)
                        x_line = np.linspace(x.min(), x.max(), 100)
                        ax2.plot(x_line, m_coef * x_line + b_coef,
                                 color="#E63946", lw=1.2, alpha=0.8, label=f"r={np.corrcoef(x, y)[0,1]:.2f}")
                        ax2.legend(fontsize=8)
                    except np.linalg.LinAlgError:
                        pass  # skip regression line if SVD fails
        else:
            ax2.text(0.5, 0.5, f"No {dom_carrier}\ndata", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=9, color="#888")
        ax2.axhline(0, color="#aaa", lw=0.7, ls="--")
        ax2.axhline(THRESH,  color="#aaa", lw=0.5, ls=":")
        ax2.axhline(-THRESH, color="#aaa", lw=0.5, ls=":")
        ax2.set_xlabel(f"{dom_carrier.title()} dispatch (MW)", fontsize=8)
        ax2.set_ylabel("IC net flow (MW, +ve=ES imports)", fontsize=8)
        ax2.set_title(f"{country}: {dom_carrier} vs IC flow", fontsize=9)

    fig.suptitle(
        "IC Export/Import Technology Composition\n"
        "Generation mix by flow direction — proxy for what carrier drives cross-border flow",
        fontsize=11, fontweight="bold",
    )
    out_path = out_dir / "18_ic_tech_composition.png"
    fig.savefig(out_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", out_path)


# ─── Monthly hydro marginal cost profiles ────────────────────────────────────
# Overrides the static SOC-tiered MC from _apply_hydro() with time-varying
# monthly values.  This prevents the "hydro ripple effect" where reservoirs
# with a cheap static MC dump all water in January and sit empty for the rest
# of the year.
#
# Profiles are defined in config["hydro"]["monthly_mc"] as:
#   {"ES": {1: 45, 2: 30, ..., 12: 45}, "PT": {...}, "FR": {...}}
# Month keys are 1–12 (January–December).  Values in EUR/MWh.
#
# ─── Storage-unit ramp constraints (extra_functionality) ─────────────────────

def _add_hydro_min_dispatch(n, snapshots, cfg):
    """Flat minimum dispatch for hydro reservoirs (ecological flow / caudal ecológico).

    Adds the LP constraint for ES, FR, and PT hydro storage units:
        p_dispatch_i[t]  >=  p_min_pu_i[t] × p_nom_i × (e_i[t-1] / e_nom_i)

    The relaxation factor is SOC fraction (e_prev / e_nom), always in [0, 1]:
      • SOC = 100% → floor = floor_da × 1.0 = floor_da  (full ecological flow)
      • SOC = 50%  → floor = floor_da × 0.5             (half floor)
      • SOC = 10%  → floor = floor_da × 0.1             (drought conservation)
      • SOC = 0%   → floor = 0                           (empty — no obligation)

    This is SAFER than the drought_soc formulation (floor_da × e_prev / (drought_soc × e_nom))
    which gives floor > floor_da when SOC > drought_soc, potentially exceeding p_nom
    for units with small max_hours. The SOC-fraction formulation is always bounded
    by [0, floor_da] and cannot create infeasibility from the floor alone.

    p_min_pu can be either:
      • A scalar (float) — applied uniformly across all hours
      • A monthly dict {month: value} — interpolated to hourly resolution
        Values are the minimum dispatch fraction derived from real REE data
        (P5 of daily hydro dispatch / p_nom), so the constraint mirrors real
        ecological flow obligations.

    Config key (per country, under hydro.per_country.<CC>.min_dispatch):
        "p_min_pu":       0.05 or {1: 0.20, 2: 0.18, ...}  # ecological flow floor
        "drought_soc":    0.10    # below this SOC fraction, relax floor linearly to 0
    """
    import xarray as xr
    import numpy as np

    hydro_cfg = cfg.get("hydro", {})
    per_country = hydro_cfg.get("per_country", {})

    for country in ("ES", "FR", "PT"):
        min_cfg = per_country.get(country, {}).get("min_dispatch", {})
        if not min_cfg:
            continue

        drought_soc  = float(min_cfg.get("drought_soc",   0.10))
        p_min_pu_raw = min_cfg.get("p_min_pu", 0.05)

        # ── Filter: only reservoir units with meaningful storage ──
        # Skip units with max_hours < 6 (run-of-river, pumped-storage, or tiny
        # reservoirs) — these cannot sustain ecological flow obligations and
        # applying min-dispatch to them creates infeasibilities (the floor
        # drains their tiny storage, conflicting with terminal SOC targets).
        country_hydro_all = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith(country)
        ]
        if country_hydro_all.empty:
            continue

        max_h_all = n.storage_units.loc[country_hydro_all, "max_hours"].values.astype(float)
        # Skip units with max_hours < 50h — small reservoirs cannot sustain ecological
        # flow obligations. The min-dispatch floor (p_min_pu × p_nom) drains their
        # tiny storage in hours, creating infeasibilities. Real ecological flow
        # obligations apply to large seasonal reservoirs, not small daily ones.
        # ES0 47 hydro (p_nom=1,980 MW, max_hours=6.8h) was causing infeasibility:
        #   floor = 0.143 × 1,980 = 283 MW → drains 13,430 MWh in 47h
        #   but window is 216h and inflow is only 49 MW → impossible.
        country_hydro = country_hydro_all[max_h_all >= 50.0]

        if country_hydro.empty:
            log.info("Extra: hydro %s min-dispatch — all units have max_hours < 50h, skipping", country)
            continue

        skipped = country_hydro_all[~country_hydro_all.isin(country_hydro)]
        if len(skipped):
            log.info(
                "Extra: hydro %s min-dispatch — skipped %d small units (max_hours < 50h): %s",
                country, len(skipped), ", ".join(skipped),
            )

        p_nom_vals = n.storage_units.loc[country_hydro, "p_nom"].values.astype(float)
        max_hours  = n.storage_units.loc[country_hydro, "max_hours"].values.astype(float)
        e_nom_vals = p_nom_vals * max_hours

        # ── Resolve p_min_pu to hourly series ──
        if isinstance(p_min_pu_raw, dict):
            # Monthly profile: build hourly series from month mapping
            snap = snapshots[1:]  # first snapshot has no e_prev
            month_idx = snap.month  # pandas Series of month numbers
            p_min_pu_series = np.array([float(p_min_pu_raw.get(m, 0.05)) for m in month_idx])
        else:
            p_min_pu_series = float(p_min_pu_raw)  # scalar

        try:
            p_var = n.model["StorageUnit-p_dispatch"].sel(name=list(country_hydro))
            e_var = n.model["StorageUnit-state_of_charge"].sel(name=list(country_hydro))
        except KeyError as err:
            log.warning("Hydro min-dispatch [%s] skipped — model variable not found: %s", country, err)
            continue

        snaps  = snapshots[1:]
        p_curr = p_var.sel(snapshot=snaps)
        e_prev = e_var.shift(snapshot=1).sel(snapshot=snaps)

        # ── Build floor: p_min_pu[t] × p_nom_i ──
        e_nom_da = xr.DataArray(
            e_nom_vals, dims=["name"], coords={"name": list(country_hydro)}
        )
        if isinstance(p_min_pu_raw, dict):
            floor_mw_t = np.outer(p_min_pu_series, p_nom_vals)  # shape (T, n_units)
            floor_da = xr.DataArray(
                floor_mw_t,
                dims=["snapshot", "name"],
                coords={"snapshot": snaps, "name": list(country_hydro)},
            )
        else:
            floor_mw = p_min_pu_series * p_nom_vals
            floor_da = xr.DataArray(
                floor_mw, dims=["name"], coords={"name": list(country_hydro)}
            )

        # ── Single constraint with SOC-proportional relaxation ──
        # Constraint: p_dispatch[t] >= floor_da × e[t-1] / e_nom
        #
        # This uses SOC fraction (e_prev / e_nom) as the relaxation factor,
        # which is always in [0, 1] so the floor never exceeds floor_da.
        #
        #   SOC = 100% → floor = floor_da × 1.0 = floor_da  (full ecological flow)
        #   SOC = 50%  → floor = floor_da × 0.5             (half floor)
        #   SOC = 10%  → floor = floor_da × 0.1             (drought conservation)
        #   SOC = 0%   → floor = 0                           (empty — no obligation)
        #
        # This is SAFER than the drought_soc formulation (floor_da × e_prev / (drought_soc × e_nom))
        # which gives floor > floor_da when SOC > drought_soc, potentially exceeding p_nom
        # for units with small max_hours. The SOC-fraction formulation is always bounded
        # by [0, floor_da] and cannot create infeasibility from the floor alone.
        n.model.add_constraints(
            p_curr >= floor_da * e_prev / e_nom_da,
            name=f"SU-hydro-{country}-min-dispatch",
        )

        # ── Diagnostic ──
        if isinstance(p_min_pu_raw, dict):
            pct_str = f"monthly profile [{min(p_min_pu_raw.values())*100:.0f}–{max(p_min_pu_raw.values())*100:.0f}%]"
        else:
            pct_str = f"{p_min_pu_raw*100:.1f}%"
        log.info(
            "Extra: hydro %s min-dispatch ≥ %s p_nom (drought relax < %.0f%% SOC) on %d units",
            country, pct_str, drought_soc * 100, len(country_hydro),
        )


def _add_su_ramp_constraints(n, snapshots, cfg):
    """Add linopy ramp-rate constraints for all hydro reservoir storage units and PHS.

    PyPSA 1.1.x StorageUnits have no native ramp_limit columns, so we inject
    these as custom LP constraints via extra_functionality.

    Hydro reservoirs (ES/FR/PT): use the global hydro.ramp_limit_pu from config.
    PHS: uses a separate phs.ramp_limit_dispatch key (if present).
    """
    import xarray as xr

    def _add_ramp(su_names, ramp_frac, label):
        su_list = list(su_names)
        if not su_list:
            return
        p_nom = n.storage_units.loc[su_list, "p_nom"]
        max_ramp = xr.DataArray(
            (ramp_frac * p_nom).values,
            dims=["name"],
            coords={"name": su_list},
        )
        p_disp = n.model["StorageUnit-p_dispatch"].sel(name=su_list)
        diff = (p_disp - p_disp.shift(snapshot=1)).sel(snapshot=snapshots[1:])
        n.model.add_constraints(diff <= max_ramp,  name=f"SU-{label}-ramp-up")
        n.model.add_constraints(-diff <= max_ramp, name=f"SU-{label}-ramp-dn")
        log.info("Extra: %s ramp≤%.0f%%/h on %d storage units", label, ramp_frac * 100, len(su_list))

    # ── Global hydro reservoir ramp (all countries) ──────────────────────────
    # Uses the top-level hydro.ramp_limit_pu from config (default: 0.20).
    # Applies to ALL hydro storage units regardless of country.
    hydro_ramp = cfg.get("hydro", {}).get("ramp_limit_pu")
    if hydro_ramp is not None:
        hydro_su = n.storage_units.index[n.storage_units["carrier"] == "hydro"]
        _add_ramp(hydro_su, hydro_ramp, "hydro-all")

    # ── PHS ramp (separate key) ──────────────────────────────────────────────
    phs_ramp = cfg.get("phs", {}).get("ramp_limit_dispatch")
    if phs_ramp is not None:
        phs_units = n.storage_units.index[n.storage_units["carrier"] == "PHS"]
        _add_ramp(phs_units, phs_ramp, "PHS")


# ─── VRE Bottleneck Diagnostic ───────────────────────────────────────────────

def add_vre_bottleneck_diagnostics(
    n: "pypsa.Network",
    output_df: "pd.DataFrame",
    cfg: dict | None = None,
    cong_threshold: float = 0.95,
    top_n: int = 5,
) -> "pd.DataFrame":
    """Identify hours where VRE should theoretically clear the market but CCGT does instead,
    and trace which transmission lines are congested during those "trapped renewable" hours.

    Accounting identity per snapshot t
    ────────────────────────────────────
      VRE_Potential(t)     = Σ p_max_pu_i(t) × p_nom_i   for ES solar + onwind
      Inflexible_Floor(t)  = biomass_p_nom + CCGT_must_run_p_nom
                           + nuclear_p_min_pu × ES_nuclear_p_nom
      Net_Import(t)        = FR_import(t) + PT_import(t)   [positive = into ES]
      Residual(t)          = ES_load(t) - VRE_Potential(t) - Net_Import(t)

      Theoretically VRE-marginal  ←→  Residual(t) < Inflexible_Floor(t)
      "Trapped renewable" hour    ←→  theory=VRE  AND  actual setter ∈ {CCGT, CCGT_flex}

    New columns appended to output_df
    ───────────────────────────────────
      vre_potential_MW          total dispatchable ES solar + onwind
      inflex_floor_MW           biomass + must_run CCGT + nuclear p_min floor
      net_import_total_MW       FR + PT combined net imports
      theoretical_residual_MW   load - VRE_potential - imports
      residual_margin_MW        Residual - Floor  (negative → VRE should dominate)
      theoretically_vre_hour    bool: Residual < Floor
      trapped_vre_hour          bool: theory=VRE AND actual setter is CCGT/CCGT_flex

    Parameters
    ──────────
    n               Solved PyPSA network
    output_df       Per-hour diagnostic DataFrame (index = n.snapshots).
                    Must contain 'price_setter' column for mismatch detection.
    cfg             MODEL_CONFIG dict; used for nuclear p_min_pu, trans_factor,
                    s_max_pu.  Defaults to empty dict (reasonable fallbacks apply).
    cong_threshold  Fraction of rated capacity above which a line is congested (0.95).
    top_n           Number of most-congested lines to surface in the summary.

    Returns
    ───────
    output_df with new diagnostic columns in-place (copy).
    Also prints a formatted console summary.
    """
    if cfg is None:
        cfg = {}

    snaps = n.snapshots
    gen   = n.generators

    # ── 1. VRE Potential ─────────────────────────────────────────────────────
    # Maximum possible output of all ES solar + onwind at every hour.
    # Time-varying p_max_pu (capacity factor profiles) multiplied by p_nom.
    vre_carriers = {"solar", "onwind"}
    es_vre = gen.index[
        gen["carrier"].isin(vre_carriers) & gen["bus"].str.startswith("ES")
    ]
    tv_pmax = getattr(n.generators_t, "p_max_pu", pd.DataFrame())

    vre_pot = pd.Series(0.0, index=snaps)
    for g in es_vre:
        p_nom = float(gen.at[g, "p_nom"])
        if g in tv_pmax.columns:
            vre_pot += tv_pmax[g].reindex(snaps, fill_value=0.0) * p_nom
        else:
            vre_pot += float(gen.at[g, "p_max_pu"]) * p_nom

    # ── 2. Inflexible Baseload Floor ──────────────────────────────────────────
    # Three components that must always be absorbed by the grid regardless of VRE:
    #   (a) Biomass + CCGT_must_run  — MC=0, forced full output every hour
    #   (b) Nuclear p_min            — minimum stable generation, cannot ramp below
    must_run_carriers = {"biomass", "CCGT_must_run"}
    es_mr = gen.index[
        gen["carrier"].isin(must_run_carriers) & gen["bus"].str.startswith("ES")
    ]
    mr_mw = float(gen.loc[es_mr, "p_nom"].sum())

    nuc_p_min = (cfg.get("nuclear", {})
                    .get("per_country", {})
                    .get("ES", {})
                    .get("p_min_pu", 0.40))
    es_nuc = gen.index[
        (gen["carrier"] == "nuclear") & gen["bus"].str.startswith("ES")
    ]
    nuc_floor_mw = nuc_p_min * float(gen.loc[es_nuc, "p_nom"].sum())

    inflex_floor = pd.Series(mr_mw + nuc_floor_mw, index=snaps)

    # ── 3. Net Imports (FR + PT) ──────────────────────────────────────────────
    # Positive = power flowing INTO Spain.  Uses the topology-agnostic helper
    # that handles both the old DC_ic Link pairs and the new AC Line topology.
    net_import = _net_import_topo(n, "FR") + _net_import_topo(n, "PT")

    # ── 4. ES Load ────────────────────────────────────────────────────────────
    es_loads = n.loads.index[n.loads["bus"].str.startswith("ES")]
    if not n.loads_t.p_set.empty:
        load_cols = [l for l in es_loads if l in n.loads_t.p_set.columns]
        es_load_t = (n.loads_t.p_set[load_cols]
                       .sum(axis=1)
                       .reindex(snaps, fill_value=0.0))
    else:
        es_load_t = pd.Series(float(n.loads.loc[es_loads, "p_set"].sum()),
                              index=snaps)

    # ── 5. Theoretical Residual Demand ────────────────────────────────────────
    # After VRE and imports are subtracted, what's left for dispatchable units?
    residual = es_load_t - vre_pot - net_import

    # ── 6. Price-Setter Series ────────────────────────────────────────────────
    # Prefer the pre-computed column in output_df; fall back to network solve.
    if "price_setter" in output_df.columns:
        setter = output_df["price_setter"].reindex(snaps)
    else:
        _, setter = _get_price_setter_series(n, "ES")

    gas_setters = {"CCGT", "CCGT_flex", "OCGT", "diesel"}
    vre_setters = {"solar", "onwind", "offwind", "ror"}

    theoretically_vre = residual < inflex_floor
    ccgt_actually_set = setter.isin(gas_setters)
    vre_actually_set  = setter.isin(vre_setters)
    trapped_vre       = theoretically_vre & ccgt_actually_set

    trapped_snaps = snaps[trapped_vre.reindex(snaps, fill_value=False).values]

    # ── 7. Congestion in Trapped Hours ────────────────────────────────────────
    # For every internal ES line and every ES↔FR/PT border circuit, count how
    # many of the "trapped VRE" hours it ran at ≥ cong_threshold of rated capacity.
    #
    # Effective capacities:
    #   Internal ES lines   : s_nom × trans_factor × s_max_pu  (both scalers apply)
    #   Border AC Lines     : s_nom × s_max_pu                 (no trans_factor — added
    #                                                            by refinery at rated values)
    #   Border DC Links     : p_nom                            (PyPSA Link, no s_nom)

    trans_factor = cfg.get("transmission", {}).get("trans_factor", 1.0)
    s_max_pu     = cfg.get("transmission", {}).get("s_max_pu",     1.0)

    congestion_counts: dict[str, dict] = {}   # name → {count, bus0, bus1, category}

    lp0 = getattr(n.lines_t, "p0", pd.DataFrame())
    if not lp0.empty and not n.lines.empty and len(trapped_snaps) > 0:
        for ln, row in n.lines.iterrows():
            b0, b1 = str(row.bus0), str(row.bus1)
            es0, es1 = b0.startswith("ES"), b1.startswith("ES")

            if not (es0 or es1):
                continue                        # neither end in Spain — skip
            if ln not in lp0.columns:
                continue

            flows = lp0[ln].reindex(trapped_snaps, fill_value=0.0).abs()

            if es0 and es1:
                # Internal ES line — both scalers apply
                cap = float(row.s_nom) * trans_factor * s_max_pu
                cat = "internal"
            else:
                # Border line — only s_max_pu (refinery sets the intended s_nom)
                cap = float(row.s_nom) * s_max_pu
                cat = "border_AC"

            cnt = int((flows >= cong_threshold * cap).sum())
            if cnt > 0:
                congestion_counts[ln] = {
                    "count": cnt,
                    "bus0": b0, "bus1": b1,
                    "category": cat,
                    "capacity_mw": cap,
                }

    kp0 = getattr(n.links_t, "p0", pd.DataFrame())
    if not kp0.empty and not n.links.empty and len(trapped_snaps) > 0:
        for lk, row in n.links.iterrows():
            b0, b1 = str(row.bus0), str(row.bus1)
            es0, es1 = b0.startswith("ES"), b1.startswith("ES")
            if not (es0 or es1):
                continue
            if lk not in kp0.columns:
                continue

            flows = kp0[lk].reindex(trapped_snaps, fill_value=0.0).abs()
            cap   = float(row.p_nom)
            cnt   = int((flows >= cong_threshold * cap).sum())
            if cnt > 0:
                congestion_counts[lk] = {
                    "count": cnt,
                    "bus0": b0, "bus1": b1,
                    "category": "border_DC",
                    "capacity_mw": cap,
                }

    top_lines = sorted(congestion_counts.items(),
                       key=lambda x: -x[1]["count"])[:top_n]

    # ── 8. Append Columns to output_df ───────────────────────────────────────
    out = output_df.copy()
    idx = out.index

    out["vre_potential_MW"]       = vre_pot.reindex(idx)
    out["inflex_floor_MW"]        = inflex_floor.reindex(idx)
    out["net_import_total_MW"]    = net_import.reindex(idx)
    out["theoretical_residual_MW"]= residual.reindex(idx)
    out["residual_margin_MW"]     = (residual - inflex_floor).reindex(idx)
    out["theoretically_vre_hour"] = theoretically_vre.reindex(idx).astype(bool)
    out["trapped_vre_hour"]       = trapped_vre.reindex(idx).astype(bool)

    # ── 9. Console Summary ────────────────────────────────────────────────────
    n_total   = len(snaps)
    n_theory  = int(theoretically_vre.sum())
    n_actual  = int(vre_actually_set.sum())
    n_trapped = int(trapped_vre.sum())

    margin_trapped = (residual - inflex_floor)[trapped_vre.reindex(snaps, fill_value=False)]

    print("\n" + "─" * 76)
    print("  VRE BOTTLENECK DIAGNOSTIC")
    print("─" * 76)
    print(f"  {'Period:':<34} {snaps[0].date()} → {snaps[-1].date()}"
          f"  ({n_total} hours)")
    print()
    print(f"  {'Inflexible floor (nuclear p_min + must-run):':<44}"
          f"  {mr_mw + nuc_floor_mw:>7.0f} MW")
    print(f"  {'VRE mean potential:':<44}  {vre_pot.mean():>7.0f} MW")
    print(f"  {'Mean net import (FR+PT):':<44}  {net_import.mean():>+7.0f} MW")
    print()
    print(f"  {'Theoretical VRE-marginal hours:':<44}"
          f"  {n_theory:>5} / {n_total}  ({100*n_theory/n_total:.1f}%)")
    print(f"  {'Actual VRE price-setting hours:':<44}"
          f"  {n_actual:>5} / {n_total}  ({100*n_actual/n_total:.1f}%)")
    print(f"  {'\"Trapped\" renewable hours (theory≠actual):':<44}"
          f"  {n_trapped:>5} / {n_total}  ({100*n_trapped/n_total:.1f}%)")
    if n_trapped > 0:
        print(f"  {'Residual margin in trapped hours:':<44}"
              f"  mean {margin_trapped.mean():>+7.0f} MW"
              f"  (median {margin_trapped.median():>+7.0f} MW)")

    if top_lines:
        print()
        print(f"  TOP {top_n} CONGESTED LINES IN TRAPPED HOURS  "
              f"(threshold ≥{cong_threshold:.0%} of capacity)")
        print(f"  {'Line/Link':<22} {'Cat':<10} {'Cap MW':>7}  "
              f"{'Congested':>10}  {'% of trapped':>14}  Route")
        print("  " + "─" * 74)
        for name, info in top_lines:
            pct = 100 * info["count"] / max(n_trapped, 1)
            print(f"  {str(name):<22} {info['category']:<10}"
                  f" {info['capacity_mw']:>7.0f}"
                  f"  {info['count']:>6} hrs"
                  f"  {pct:>12.1f}%"
                  f"  {info['bus0']} → {info['bus1']}")
    else:
        print("\n  No lines congested at threshold during trapped hours.")

    print("─" * 76 + "\n")

    return out


# ─── VRE price-setter diagnostic (Groups A–F) ────────────────────────────────

def _print_vre_price_setter_diagnostic(n, omie, cfg=None):
    """Groups A–F: Why doesn't VRE set price in low-OMIE hours?

    A — Supply cover: does must-take supply ≥ demand in real low-price hours?
    B — LP attribution: is VRE curtailed (→ could be marginal) or fully dispatched?
    C — Forced thermal floors: CCGT_must_run, nuclear p_min, regular CCGT in low hours
    D — Interconnector: ES vs FR shadow price, IC congestion, FR/PT SOC summary
    E — CCGT MC structure: p10/p50/p90 by carrier, high-price hour attribution
    F — Hydro SOC by month: when do reservoirs deplete?
    """
    if cfg is None:
        cfg = {}

    snaps  = n.snapshots
    gen    = n.generators
    gen_t  = n.generators_t.p
    es_set = set(_es_buses(n))
    SEP    = "─" * 76

    omie_aligned = omie.reindex(snaps)
    low_mask  = omie_aligned.le(5.0).fillna(False)
    low_hours = snaps[low_mask.values]

    print(f"\n{SEP}")
    print("  VRE PRICE-SETTER DIAGNOSTIC  (Groups A–F)")
    print(f"  Reference: OMIE ≤ €5 hours in model period")
    print(SEP)
    if len(low_hours) == 0:
        print("  No OMIE hours ≤ €5 in this period — Groups A/B not applicable.")
        print("  (Run with a full-year solve for complete diagnostic)")
        print()
    else:
        print(f"  Low-OMIE reference hours (OMIE ≤ €5): {len(low_hours)} / {len(snaps)}")
        print()

    # ── Shared building blocks ────────────────────────────────────────────────
    es_load_idx = n.loads.index[n.loads["bus"].str.startswith("ES")]
    if not n.loads_t.p_set.empty:
        load_cols = [l for l in es_load_idx if l in n.loads_t.p_set.columns]
        es_load_t = n.loads_t.p_set[load_cols].sum(axis=1).reindex(snaps, fill_value=0.0)
    else:
        es_load_t = pd.Series(float(n.loads.loc[es_load_idx, "p_set"].sum()), index=snaps)

    def _es_carrier_dispatch(carrier):
        cg = [g for g in gen.index
              if gen.at[g, "carrier"] == carrier
              and gen.at[g, "bus"] in es_set and g in gen_t.columns]
        return gen_t[cg].sum(axis=1).reindex(snaps, fill_value=0.0) if cg else pd.Series(0.0, index=snaps)

    vre_carriers = {"solar", "onwind", "ror"}
    vre_by_c = {c: _es_carrier_dispatch(c) for c in vre_carriers}
    vre_total_t = sum(vre_by_c.values())

    nuclear_t   = _es_carrier_dispatch("nuclear")
    biomass_t   = _es_carrier_dispatch("biomass")
    ccgt_mr_t   = _es_carrier_dispatch("CCGT_must_run")
    ccgt_t      = _es_carrier_dispatch("CCGT")
    ccgt_flex_t = _es_carrier_dispatch("CCGT_flex")
    must_run_t  = nuclear_t + biomass_t + ccgt_mr_t

    net_fr = _net_import_topo(n, "FR")
    net_pt = _net_import_topo(n, "PT")
    net_imp_t  = net_fr + net_pt
    must_take_t = vre_total_t + must_run_t + net_imp_t

    # ── GROUP A ──────────────────────────────────────────────────────────────
    if len(low_hours) > 0:
        def _lm(s):
            return float(s.reindex(low_hours).mean())

        n_covered = int((must_take_t.reindex(low_hours) >= es_load_t.reindex(low_hours)).sum())

        print("  GROUP A — Supply cover in low-OMIE hours")
        print(f"  {'Component':<28}  {'All-hrs mean MW':>16}  {'Low-OMIE mean MW':>17}")
        print("  " + "─" * 65)
        rows = [
            ("ES demand",            es_load_t),
            ("VRE (solar+wind+ror)", vre_total_t),
            ("  solar",              vre_by_c["solar"]),
            ("  onwind",             vre_by_c["onwind"]),
            ("  ror",                vre_by_c["ror"]),
            ("Must-run (nuc+bio+mr)",must_run_t),
            ("Net import (FR+PT)",   net_imp_t),
            ("Must-take total",      must_take_t),
        ]
        for label, s in rows:
            print(f"  {label:<28}  {float(s.mean()):>+16.0f}  {_lm(s):>+17.0f}")
        pct_cov = 100.0 * n_covered / len(low_hours)
        print(f"\n  A6. Must-take ≥ demand in low-OMIE hours: {n_covered} / {len(low_hours)}"
              f"  ({pct_cov:.1f}%)")
        if pct_cov >= 50:
            print("      → Model HAS the supply in most low-OMIE hours.")
            print("        Root cause: LP attribution (Groups B–E), NOT a capacity gap.")
        else:
            print("      → Model LACKS must-take supply in majority of low-OMIE hours.")
            print("        Root cause: capacity / inflow / import gap (Groups D & F).")
        print()

    # ── GROUP B ──────────────────────────────────────────────────────────────
    tv_pmax = getattr(n.generators_t, "p_max_pu", pd.DataFrame())
    vre_sw_gens = [g for g in gen.index
                   if gen.at[g, "carrier"] in {"solar", "onwind"}
                   and gen.at[g, "bus"] in es_set]
    vre_pot = pd.Series(0.0, index=snaps)
    for g in vre_sw_gens:
        p_nom = float(gen.at[g, "p_nom"])
        cf = tv_pmax[g].reindex(snaps, fill_value=0.0) if g in tv_pmax.columns else float(gen.at[g, "p_max_pu"])
        vre_pot += cf * p_nom if not isinstance(cf, float) else cf * p_nom

    if len(low_hours) > 0 and vre_sw_gens:
        vre_actual_sw = sum(_es_carrier_dispatch(c) for c in ("solar", "onwind"))
        curt_mw   = (vre_pot - vre_actual_sw).clip(lower=0.0)
        curt_frac = curt_mw / vre_pot.clip(lower=1.0)

        low_curt  = curt_frac.reindex(low_hours)
        n_curt_h  = int((low_curt > 0.01).sum())
        n_full_h  = len(low_hours) - n_curt_h

        es_price_s = _mean_es_price(n)
        _, setter  = _get_price_setter_series(n, "ES")

        fd_snaps = low_hours[(low_curt <= 0.01).values]
        if len(fd_snaps) > 0 and not setter.empty:
            fd_setter_cts = setter.reindex(fd_snaps).value_counts()
        else:
            fd_setter_cts = pd.Series(dtype=int)

        print("  GROUP B — VRE marginal status in low-OMIE hours")
        print(f"  Hours with ≥1 VRE curtailed   (→ VRE could be LP-marginal): "
              f"{n_curt_h:>5}  ({100*n_curt_h/len(low_hours):.1f}%)")
        print(f"  Hours with VRE fully dispatched (→ VRE CANNOT be LP-marginal): "
              f"{n_full_h:>5}  ({100*n_full_h/len(low_hours):.1f}%)")
        if n_full_h > 0 and not fd_setter_cts.empty:
            print(f"  Price setter in fully-dispatched low-OMIE hours:")
            for carrier, cnt in fd_setter_cts.head(5).items():
                print(f"    {carrier:<20}  {cnt:>5} hrs  ({100*cnt/n_full_h:.1f}%)")
            top = fd_setter_cts.index[0] if len(fd_setter_cts) > 0 else ""
            if top in {"CCGT", "CCGT_flex", "nuclear"}:
                print("    → Must-run / thermal floor is pushing CCGT onto the margin even")
                print("      when VRE is fully dispatched (see Group C).")
        print()

    # ── GROUP C ──────────────────────────────────────────────────────────────
    nuc_p_min = (cfg.get("nuclear", {}).get("per_country", {})
                    .get("ES", {}).get("p_min_pu", 0.55))
    es_nuc_gens = [g for g in gen.index
                   if gen.at[g, "carrier"] == "nuclear" and gen.at[g, "bus"] in es_set]
    nuc_p_nom   = float(gen.loc[es_nuc_gens, "p_nom"].sum()) if es_nuc_gens else 0.0
    nuc_floor   = nuc_p_min * nuc_p_nom
    ccgt_mr_tgt = float(cfg.get("ccgt_must_run", {}).get("target_mw", 1400.0))

    print("  GROUP C — Forced thermal floors")
    print(f"  Nuclear p_min floor:          {nuc_floor:>7.0f} MW"
          f"  (p_min_pu={nuc_p_min:.2f} × {nuc_p_nom:.0f} MW)")
    print(f"  Nuclear actual mean:          {float(nuclear_t.mean()):>7.0f} MW"
          f"  (above floor by {float(nuclear_t.mean()) - nuc_floor:>+.0f} MW)")
    print(f"  Biomass fixed mean:           {float(biomass_t.mean()):>7.0f} MW")
    print(f"  CCGT_must_run target:         {ccgt_mr_tgt:>7.0f} MW")
    print(f"  CCGT_must_run actual mean:    {float(ccgt_mr_t.mean()):>7.0f} MW")
    print(f"  Must-run total mean:          {float(must_run_t.mean()):>7.0f} MW")
    if len(low_hours) > 0:
        ccgt_in_low = float((ccgt_t + ccgt_flex_t).reindex(low_hours).mean())
        print(f"  Regular CCGT mean in low-OMIE hours: {ccgt_in_low:>7.0f} MW")
        if ccgt_in_low > 500:
            print("  → Regular CCGT dispatching meaningfully in low-OMIE hours.")
            print("    Likely: nuclear floor squeezes CCGT onto margin (nodal pockets or trickle).")
    print()

    # ── GROUP D ──────────────────────────────────────────────────────────────
    mp      = getattr(n.buses_t, "marginal_price", pd.DataFrame())
    fr_cols = [b for b in n.buses.index if b.startswith("FR") and b in mp.columns]
    pt_cols = [b for b in n.buses.index if b.startswith("PT") and b in mp.columns]
    es_p    = _mean_es_price(n)
    fr_p    = mp[fr_cols].mean(axis=1) if fr_cols else pd.Series(dtype=float)

    print("  GROUP D — Interconnector shadow prices")
    if len(low_hours) > 0 and not es_p.empty and not fr_p.empty:
        es_lo = float(es_p.reindex(low_hours).mean())
        fr_lo = float(fr_p.reindex(low_hours).mean())
        diff  = es_lo - fr_lo
        print(f"  In low-OMIE hours — mean shadow prices:")
        print(f"    ES: {es_lo:>+7.1f} €/MWh   FR: {fr_lo:>+7.1f} €/MWh"
              f"   Δ(ES−FR): {diff:>+7.1f} €/MWh")
        if diff > 5:
            print("    → ES > FR: LP should be importing. Check IC congestion:")
        elif diff < -5:
            print("    → ES < FR: ES has surplus; exporting keeps ES price low.")
        else:
            print("    → ES ≈ FR: IC not a binding constraint in these hours.")

    s_max_pu = cfg.get("transmission", {}).get("s_max_pu", 0.9)
    lp0 = getattr(n.lines_t, "p0", pd.DataFrame())
    kp0 = getattr(n.links_t, "p0", pd.DataFrame())
    border_cong = {}
    if len(low_hours) > 0:
        for ln, row in n.lines.iterrows():
            b0, b1 = str(row.bus0), str(row.bus1)
            is_border = (b0.startswith("ES") and not b1.startswith("ES")) or \
                        (b1.startswith("ES") and not b0.startswith("ES"))
            if not is_border or ln not in lp0.columns:
                continue
            cap   = float(row.s_nom) * s_max_pu
            flows = lp0[ln].reindex(low_hours, fill_value=0.0).abs()
            pct   = 100.0 * float((flows >= 0.95 * cap).sum()) / len(low_hours)
            if pct > 5:
                border_cong[ln] = (pct, cap, b0, b1)
        for lk, row in n.links.iterrows():
            b0, b1 = str(row.bus0), str(row.bus1)
            is_border = (b0.startswith("ES") and not b1.startswith("ES")) or \
                        (b1.startswith("ES") and not b0.startswith("ES"))
            if not is_border or lk not in kp0.columns:
                continue
            cap   = float(row.p_nom)
            flows = kp0[lk].reindex(low_hours, fill_value=0.0).abs()
            pct   = 100.0 * float((flows >= 0.95 * cap).sum()) / len(low_hours)
            if pct > 5:
                border_cong[lk] = (pct, cap, b0, b1)

    if border_cong:
        print(f"  Border IC congestion (≥95% cap) in low-OMIE hours:")
        for name, (pct, cap, b0, b1) in sorted(border_cong.items(), key=lambda x: -x[1][0]):
            print(f"    {str(name):<24}  {pct:>5.1f}% hrs  {cap:.0f} MW  {b0}→{b1}")
    elif len(low_hours) > 0:
        print("  Border IC: no circuits at ≥95% capacity in low-OMIE hours.")

    soc_raw = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())
    print("  FR/PT/ES hydro SOC:")
    for country in ("FR", "PT", "ES"):
        su_h = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith(country)
        ]
        su_in = [s for s in su_h if s in soc_raw.columns]
        if not su_in:
            continue
        e_nom = float((n.storage_units.loc[su_in, "p_nom"] *
                       n.storage_units.loc[su_in, "max_hours"]).sum())
        if e_nom <= 0:
            continue
        soc_pct = soc_raw[su_in].sum(axis=1) / e_nom * 100.0
        monthly = soc_pct.resample("ME").mean()
        if monthly.empty:
            end_val = float(soc_pct.iloc[-1])
            print(f"    {country}: end={end_val:.1f}%  (period too short for monthly)")
            continue
        end_val = float(soc_pct.iloc[-1])
        min_val = float(monthly.min())
        min_mon = monthly.idxmin().strftime("%b")
        summary = "  ".join(f"{m.strftime('%b')}:{v:.0f}%" for m, v in monthly.items())
        print(f"    {country}: {summary}   end={end_val:.1f}%  min={min_val:.1f}% ({min_mon})")
        if end_val < 5.0:
            print(f"      ⚠ {country} nearly empty at period end — forces extra gas in final months")
    print()

    # ── GROUP E ──────────────────────────────────────────────────────────────
    tv_mc     = getattr(n.generators_t, "marginal_cost", pd.DataFrame())
    es_ccgt_g = [g for g in gen.index
                 if gen.at[g, "carrier"] in {"CCGT", "CCGT_flex"}
                 and gen.at[g, "bus"] in es_set]

    print("  GROUP E — CCGT marginal cost distribution")
    for carrier in ("CCGT", "CCGT_flex"):
        cg_tv = [g for g in es_ccgt_g
                 if gen.at[g, "carrier"] == carrier and g in tv_mc.columns]
        cg_st = [g for g in es_ccgt_g
                 if gen.at[g, "carrier"] == carrier and g not in tv_mc.columns]
        if cg_tv:
            vals = tv_mc[cg_tv].values.ravel()
            vals = vals[np.isfinite(vals)]
            p10, p50, p90 = np.percentile(vals, [10, 50, 90]) if len(vals) > 0 else (0, 0, 0)
            print(f"  {carrier:<16}  n={len(cg_tv):>3} (tv-MC)  "
                  f"p10={p10:>7.1f}  p50={p50:>7.1f}  p90={p90:>7.1f}  €/MWh")
        elif cg_st:
            mcs = gen.loc[cg_st, "marginal_cost"].values
            print(f"  {carrier:<16}  n={len(cg_st):>3} (static)  "
                  f"min={mcs.min():>7.1f}  mean={mcs.mean():>7.1f}  max={mcs.max():>7.1f}  €/MWh")

    es_p_series = _mean_es_price(n)
    if not es_p_series.empty:
        _, setter = _get_price_setter_series(n, "ES")
        high_snaps = snaps[(es_p_series > 120.0).fillna(False).values]
        if len(high_snaps) > 0:
            hp_cts = setter.reindex(high_snaps).value_counts()
            print(f"\n  Price setter in ES price > €120 hours ({len(high_snaps)} hrs):")
            for carrier, cnt in hp_cts.head(5).items():
                print(f"    {carrier:<20}  {cnt:>5} hrs  ({100*cnt/len(high_snaps):.1f}%)")
    print()

    # ── GROUP F ──────────────────────────────────────────────────────────────
    print("  GROUP F — Hydro SOC depletion (monthly detail)")
    any_f = False
    for country in ("ES", "FR", "PT"):
        su_h  = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith(country)
        ]
        su_in = [s for s in su_h if s in soc_raw.columns]
        if not su_in:
            continue
        e_nom = float((n.storage_units.loc[su_in, "p_nom"] *
                       n.storage_units.loc[su_in, "max_hours"]).sum())
        if e_nom <= 0:
            continue
        any_f = True
        soc_pct = soc_raw[su_in].sum(axis=1) / e_nom * 100.0
        monthly = soc_pct.resample("ME").mean()
        if monthly.empty:
            print(f"  {country}: period too short for monthly breakdown")
            continue
        danger = monthly[monthly < 10.0]
        vals_s = "  ".join(
            f"{'⚠' if v < 10 else ''}{m.strftime('%b')}:{v:.0f}%"
            for m, v in monthly.items()
        )
        print(f"  {country}: {vals_s}")
        if not danger.empty:
            mons = ", ".join(m.strftime("%b") for m in danger.index)
            print(f"    ⚠ SOC < 10% in: {mons} — thermal dispatch forced in these months")
    if not any_f:
        print("  No hydro reservoir state_of_charge data available.")

    print(f"\n{SEP}\n")


def _plot_vre_diagnostic(n, omie, out_dir):
    """4-panel diagnostic: supply cover in low-OMIE hours, VRE utilisation scatter,
    monthly hydro SOC by country, CCGT MC distribution.
    Saved as 21_vre_price_setter_diag.png.
    """
    snaps  = n.snapshots
    gen    = n.generators
    gen_t  = n.generators_t.p
    es_set = set(_es_buses(n))

    omie_al   = omie.reindex(snaps)
    low_mask  = omie_al.le(5.0).fillna(False)
    low_hours = snaps[low_mask.values]

    def _es_c(carrier):
        cg = [g for g in gen.index
              if gen.at[g, "carrier"] == carrier
              and gen.at[g, "bus"] in es_set and g in gen_t.columns]
        return gen_t[cg].sum(axis=1).reindex(snaps, fill_value=0.0) if cg else pd.Series(0.0, index=snaps)

    vre_solar  = _es_c("solar")
    vre_wind   = _es_c("onwind")
    vre_ror    = _es_c("ror")
    vre_tot    = vre_solar + vre_wind + vre_ror
    nuclear_t  = _es_c("nuclear")
    biomass_t  = _es_c("biomass")
    ccgt_mr_t  = _es_c("CCGT_must_run")
    must_run_t = nuclear_t + biomass_t + ccgt_mr_t
    net_imp_t  = _net_import_topo(n, "FR") + _net_import_topo(n, "PT")

    es_load_idx = n.loads.index[n.loads["bus"].str.startswith("ES")]
    if not n.loads_t.p_set.empty:
        load_cols = [l for l in es_load_idx if l in n.loads_t.p_set.columns]
        es_load_t = n.loads_t.p_set[load_cols].sum(axis=1).reindex(snaps, fill_value=0.0)
    else:
        es_load_t = pd.Series(float(n.loads.loc[es_load_idx, "p_set"].sum()), index=snaps)

    # VRE potential
    tv_pmax = getattr(n.generators_t, "p_max_pu", pd.DataFrame())
    vre_sw_gens = [g for g in gen.index
                   if gen.at[g, "carrier"] in {"solar", "onwind"}
                   and gen.at[g, "bus"] in es_set]
    vre_pot = pd.Series(0.0, index=snaps)
    for g in vre_sw_gens:
        p_nom = float(gen.at[g, "p_nom"])
        cf = tv_pmax[g].reindex(snaps, fill_value=0.0) if g in tv_pmax.columns else float(gen.at[g, "p_max_pu"])
        vre_pot += cf * p_nom if not isinstance(cf, float) else cf * p_nom

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    t0 = snaps[0].strftime("%d %b")
    t1 = snaps[-1].strftime("%d %b %Y")
    fig.suptitle(f"VRE Price-Setter Diagnostic  [{t0}–{t1}]",
                 fontsize=13, fontweight="bold")

    ax_sup, ax_vre, ax_soc, ax_mc = axes.flatten()

    # ── Panel 1: supply components — overall vs low-OMIE hours ───────────────
    components = {
        "VRE":        vre_tot,
        "Nuclear":    nuclear_t,
        "Biomass":    biomass_t,
        "CCGT_must_run": ccgt_mr_t,
        "Net import": net_imp_t,
    }
    comp_colors = {
        "VRE":           COLORS.get("solar", "#F4A261"),
        "Nuclear":       COLORS.get("nuclear", "#457B9D"),
        "Biomass":       COLORS.get("biomass", "#8B7355"),
        "CCGT_must_run": COLORS.get("CCGT_must_run", "#FFB347"),
        "Net import":    "#A8D8EA",
    }
    x = np.arange(2)
    labels = ["All hours", "Low-OMIE hrs\n(OMIE ≤ €5)"]
    bottom_all = np.zeros(2)
    for comp, s in components.items():
        all_mean  = float(s.mean())
        low_mean  = float(s.reindex(low_hours).mean()) if len(low_hours) > 0 else 0.0
        heights   = np.array([all_mean, low_mean])
        ax_sup.bar(x, heights, bottom=bottom_all, color=comp_colors[comp],
                   label=comp, alpha=0.85, edgecolor="white", linewidth=0.5)
        bottom_all += heights

    dem_all = float(es_load_t.mean())
    dem_low = float(es_load_t.reindex(low_hours).mean()) if len(low_hours) > 0 else 0.0
    ax_sup.plot(x, [dem_all, dem_low], "k^--", ms=9, lw=1.6, label="ES demand", zorder=5)
    ax_sup.set_xticks(x)
    ax_sup.set_xticklabels(labels)
    ax_sup.set_ylabel("Mean MW")
    ax_sup.set_title("A: Supply components vs ES demand")
    ax_sup.legend(fontsize=8, loc="upper right")
    if len(low_hours) > 0:
        n_cov = int((
            (vre_tot + must_run_t + net_imp_t).reindex(low_hours) >= es_load_t.reindex(low_hours)
        ).sum())
        ax_sup.text(0.02, 0.97,
                    f"Must-take ≥ demand in\n{n_cov}/{len(low_hours)} low-OMIE hrs",
                    transform=ax_sup.transAxes, fontsize=8, va="top",
                    color="#333333", bbox=dict(fc="white", ec="none", alpha=0.7))

    # ── Panel 2: VRE potential vs actual scatter ──────────────────────────────
    es_p_series = _mean_es_price(n)
    price_vals  = es_p_series.reindex(snaps).fillna(np.nan).values
    vre_pot_v   = vre_pot.values / 1e3     # GW
    vre_act_v   = (vre_solar + vre_wind).values / 1e3

    price_bins = [0, 10, 50, 100, 300]
    bin_labels = ["≤€10", "€10–50", "€50–100", ">€100"]
    bin_colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    for i, (lo, hi, lbl, col) in enumerate(zip(price_bins, price_bins[1:], bin_labels, bin_colors)):
        mask = (price_vals >= lo) & (price_vals < hi)
        ax_vre.scatter(vre_pot_v[mask], vre_act_v[mask],
                       s=8, alpha=0.55, color=col, label=lbl, rasterized=True)

    max_pot = max(float(vre_pot.max()) / 1e3, 1.0)
    ax_vre.plot([0, max_pot], [0, max_pot], "k--", lw=1.0, alpha=0.5, label="y=x (no curtail)")
    ax_vre.set_xlabel("VRE potential (solar+wind, GW)")
    ax_vre.set_ylabel("VRE actual dispatch (GW)")
    ax_vre.set_title("B: VRE utilisation by price band")
    ax_vre.legend(fontsize=8, markerscale=2)

    # ── Panel 3: Monthly hydro SOC % ES / FR / PT ─────────────────────────────
    soc_raw = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())
    country_cfg = [("ES", "#457B9D"), ("FR", "#E67E22"), ("PT", "#27AE60")]
    has_soc = False
    for country, col in country_cfg:
        su_h  = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith(country)
        ]
        su_in = [s for s in su_h if s in soc_raw.columns]
        if not su_in:
            continue
        e_nom = float((n.storage_units.loc[su_in, "p_nom"] *
                       n.storage_units.loc[su_in, "max_hours"]).sum())
        if e_nom <= 0:
            continue
        soc_pct = soc_raw[su_in].sum(axis=1) / e_nom * 100.0
        monthly = soc_pct.resample("ME").mean()
        if monthly.empty:
            ax_soc.plot(soc_pct.resample("D").mean().index,
                        soc_pct.resample("D").mean().values,
                        lw=1.4, color=col, label=country)
        else:
            ax_soc.plot(monthly.index, monthly.values, "o-",
                        lw=1.8, ms=5, color=col, label=country)
        has_soc = True

    if has_soc:
        ax_soc.axhspan(0, 5,  color="#e74c3c", alpha=0.12, zorder=0)
        ax_soc.axhspan(5, 10, color="#f39c12", alpha=0.10, zorder=0)
        ax_soc.axhline(10, color="#f39c12", lw=0.8, ls="--", alpha=0.6)
        ax_soc.axhline(5,  color="#e74c3c", lw=0.8, ls="--", alpha=0.6)
        ax_soc.set_ylim(0, 100)
        ax_soc.set_ylabel("Mean SOC (%)")
        ax_soc.set_xlabel("Month")
        ax_soc.set_title("F: Monthly hydro SOC by country")
        ax_soc.legend(fontsize=9)
        ax_soc.text(0.98, 0.06, "⚠ <10% danger zone",
                    transform=ax_soc.transAxes, fontsize=7, ha="right",
                    color="#e74c3c")
    else:
        ax_soc.text(0.5, 0.5, "No hydro SOC data", ha="center", va="center",
                    transform=ax_soc.transAxes, color="#888")
        ax_soc.set_title("F: Monthly hydro SOC by country")

    # ── Panel 4: CCGT MC distribution ────────────────────────────────────────
    tv_mc = getattr(n.generators_t, "marginal_cost", pd.DataFrame())
    mc_data, mc_labels, mc_colors = [], [], []
    for carrier, col in [("CCGT", COLORS.get("CCGT", "#FF6B6B")),
                          ("CCGT_flex", COLORS.get("CCGT_flex", "#D64045"))]:
        cg = [g for g in gen.index
              if gen.at[g, "carrier"] == carrier
              and gen.at[g, "bus"] in es_set and g in tv_mc.columns]
        if cg:
            vals = tv_mc[cg].values.ravel()
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) > 0:
                mc_data.append(vals)
                mc_labels.append(carrier)
                mc_colors.append(col)

    if mc_data:
        bparts = ax_mc.violinplot(mc_data, showmedians=True, showextrema=True)
        for patch, col in zip(bparts["bodies"], mc_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        ax_mc.set_xticks(range(1, len(mc_labels) + 1))
        ax_mc.set_xticklabels(mc_labels)
    else:
        ax_mc.text(0.5, 0.5, "No time-varying CCGT MC stored",
                   ha="center", va="center", transform=ax_mc.transAxes, color="#888")

    # Overlay mean ES price in low-OMIE hours + overall
    if not es_p_series.empty:
        mean_all = float(es_p_series.mean())
        ax_mc.axhline(mean_all, color=_SLATE, lw=1.4, ls="--", alpha=0.8,
                      label=f"ES mean price €{mean_all:.0f}")
        if len(low_hours) > 0:
            mean_low = float(es_p_series.reindex(low_hours).mean())
            ax_mc.axhline(mean_low, color="#2ecc71", lw=1.4, ls=":",
                          label=f"Low-OMIE mean €{mean_low:.0f}")
        ax_mc.legend(fontsize=8)

    ax_mc.set_ylabel("Marginal cost (€/MWh)")
    ax_mc.set_title("E: CCGT MC distribution vs ES price")

    fig.tight_layout()
    path = out_dir / "21_vre_price_setter_diag.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── BESS fleet injection ─────────────────────────────────────────────────────

def _add_bess_fleet(n, cfg):
    """Add BESS fleet from CSV, mapping each project to its closest ES network bus.

    Enabled only when cfg["bess_fleet"]["enabled"] is True (2030 scenario).
    Canary Islands projects (lat ≤ canary_lat_max) are silently dropped.

    Technology → carrier:
        Li-ion / Stand-alone → battery
        PHS                  → PHS_new
        Termico              → thermal_storage
    """
    fleet = cfg.get("bess_fleet", {})
    if not fleet.get("enabled", False):
        log.info("BESS fleet: disabled — set bess_fleet.enabled=True for 2030 scenario")
        return

    csv_path  = ROOT / fleet["csv_path"]
    lat_max   = float(fleet.get("canary_lat_max", 29.5))
    soc_frac  = float(fleet.get("soc_initial_fraction", 0.50))
    tech_map  = fleet.get("technology_map", {
        "Li-ion":      "battery",
        "Stand-alone": "battery",
        "PHS":         "PHS_new",
        "Termico":     "thermal_storage",
    })

    df = pd.read_csv(csv_path)
    n_total = len(df)
    df = df[df["lat"] > lat_max].copy()
    log.info("BESS fleet: %d / %d projects retained after Canary Islands filter",
             len(df), n_total)

    # Ensure required carriers exist in the network
    for carrier in set(tech_map.values()):
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier)

    # Use pre-computed closest_bus column if present; fall back to haversine at runtime.
    if "closest_bus" not in df.columns:
        log.warning("BESS fleet: 'closest_bus' column missing — computing haversine on the fly")
        es_buses  = n.buses[n.buses["country"] == "ES"]
        bus_lons  = es_buses["x"].values
        bus_lats  = es_buses["y"].values
        bus_names = es_buses.index.values

        def _haversine_closest(lat, lon):
            dlat = np.radians(bus_lats - lat)
            dlon = np.radians(bus_lons - lon)
            a = (np.sin(dlat / 2) ** 2
                 + np.cos(np.radians(lat)) * np.cos(np.radians(bus_lats))
                 * np.sin(dlon / 2) ** 2)
            return bus_names[np.argmin(a)]

        df["closest_bus"] = df.apply(
            lambda r: _haversine_closest(float(r["lat"]), float(r["long"])), axis=1
        )

    counts: dict[str, int] = {}
    skipped = 0
    for _, row in df.iterrows():
        carrier = tech_map.get(str(row["technology"]).strip())
        if carrier is None:
            skipped += 1
            continue

        bus       = str(row["closest_bus"])
        raw_name  = str(row["name"]).strip()
        unit_name = raw_name
        suffix    = 1
        while unit_name in n.storage_units.index:
            unit_name = f"{raw_name}_{suffix}"
            suffix += 1

        p_nom     = float(row["p_nom"])
        max_hours = float(row["max_hours"])
        cyclic    = str(row["cyclic_state_of_charge"]).strip().upper() != "FALSE"

        n.add(
            "StorageUnit", unit_name,
            bus                    = bus,
            carrier                = carrier,
            p_nom                  = p_nom,
            max_hours              = max_hours,
            efficiency_store       = float(row["efficiency_store"]),
            efficiency_dispatch    = float(row["efficiency_dispatch"]),
            standing_loss          = float(row["standing_loss"]),
            marginal_cost          = float(row["marginal_cost"]),
            cyclic_state_of_charge = cyclic,
            state_of_charge_initial = soc_frac * p_nom * max_hours,
        )
        counts[carrier] = counts.get(carrier, 0) + 1

    total = sum(counts.values())
    log.info(
        "BESS fleet: added %d storage units  (%s)",
        total,
        ", ".join(f"{c}={v}" for c, v in sorted(counts.items())),
    )
    if skipped:
        log.warning("BESS fleet: %d rows skipped (unmapped technology)", skipped)


def _plot_bess_map(n, cfg, out_dir):
    """BESS project map — Natural Earth basemap, typed markers, proportional node bubbles.

    Always generated when called so geography can be checked before activating the fleet.
    """
    from matplotlib.lines import Line2D

    fleet    = cfg.get("bess_fleet", {})
    csv_path = ROOT / fleet.get("csv_path", "Analysis/data/2024_batteries.csv")
    lat_max  = float(fleet.get("canary_lat_max", 29.5))
    tech_map = fleet.get("technology_map", {
        "Li-ion": "battery", "Stand-alone": "battery",
        "PHS": "PHS_new", "Termico": "thermal_storage",
    })

    df = pd.read_csv(csv_path)
    df = df[df["lat"] > lat_max].copy()
    if "closest_bus" not in df.columns:
        log.warning("BESS map: closest_bus column missing — skipping map")
        return

    df["carrier"] = df["technology"].map(tech_map).fillna("unknown")

    agg    = df.groupby("closest_bus")["p_nom"].sum().rename("total_mw")
    bus_xy = n.buses[["x", "y"]]

    # Per-technology: (marker, fill, edge)
    TECH_STYLE: dict[str, tuple] = {
        "battery":         ("o",  "#1E88E5", "#0D47A1"),   # circle — Li-ion blue
        "PHS_new":         ("^",  "#43A047", "#1B5E20"),   # triangle — pumped hydro green
        "thermal_storage": ("D",  "#FB8C00", "#E65100"),   # diamond — thermal orange
        "unknown":         ("s",  "#9E9E9E", "#616161"),   # square  — grey
    }

    proj   = ccrs.PlateCarree()
    EXTENT = (-10.5, 5.2, 35.5, 44.5)
    fig, ax = plt.subplots(figsize=(13, 10), subplot_kw={"projection": proj})
    fig.patch.set_facecolor("white")
    _setup_cartopy_ax(ax, extent=EXTENT)

    # ── Transmission skeleton (ES only) ─────────────────────────────────────────
    es_line_idx = n.lines.index[
        n.lines["bus0"].str.startswith("ES") & n.lines["bus1"].str.startswith("ES")
    ]
    for _, line in n.lines.loc[es_line_idx].iterrows():
        b0, b1 = line.bus0, line.bus1
        if b0 not in n.buses.index or b1 not in n.buses.index:
            continue
        ax.plot(
            [n.buses.loc[b0, "x"], n.buses.loc[b1, "x"]],
            [n.buses.loc[b0, "y"], n.buses.loc[b1, "y"]],
            color="#b8c4d0", lw=0.55, alpha=0.65, zorder=4, transform=proj,
        )

    # ── All ES bus dots ──────────────────────────────────────────────────────────
    es_buses_df = n.buses[n.buses.index.str.startswith("ES")]
    ax.scatter(
        es_buses_df["x"], es_buses_df["y"],
        s=14, color="#9aaabb", edgecolors="#6b7c93",
        linewidths=0.5, zorder=5, transform=proj,
    )

    # ── Individual project markers ───────────────────────────────────────────────
    for carrier, grp in df.groupby("carrier"):
        mrk, fill, edge = TECH_STYLE.get(carrier, TECH_STYLE["unknown"])
        ax.scatter(
            grp["long"], grp["lat"],
            s=70, marker=mrk, color=fill, edgecolors=edge,
            linewidths=0.9, alpha=0.82, zorder=6, transform=proj,
        )

    # ── Node aggregate bubbles ───────────────────────────────────────────────────
    max_mw = agg.max() if not agg.empty else 1.0
    for bus_name, total_mw in agg.items():
        if bus_name not in bus_xy.index:
            continue
        bx, by = bus_xy.loc[bus_name, "x"], bus_xy.loc[bus_name, "y"]
        # Sqrt scaling with generous minimum for legibility
        size = (total_mw / max_mw) ** 0.55 * 1800 + 120

        ax.scatter(
            bx, by, s=size,
            color="#1E88E5", alpha=0.30,
            edgecolors="#0D47A1", linewidths=1.8,
            zorder=7, transform=proj,
        )
        # Bold MW label inside bubble
        ax.text(
            bx, by, f"{total_mw:.0f}",
            transform=proj, zorder=8,
            fontsize=7.5, ha="center", va="center",
            fontweight="bold", color="#0D47A1",
        )

    # ── Legend ───────────────────────────────────────────────────────────────────
    tech_labels = {
        "battery":         "Li-ion Battery",
        "PHS_new":         "Pumped Hydro (new)",
        "thermal_storage": "Thermal Storage",
        "unknown":         "Other / Unknown",
    }
    legend_elems = [
        Line2D([0], [0], marker=TECH_STYLE[k][0], color="w",
               markerfacecolor=TECH_STYLE[k][1], markeredgecolor=TECH_STYLE[k][2],
               markersize=9, label=tech_labels[k])
        for k in ("battery", "PHS_new", "thermal_storage", "unknown")
        if k in df["carrier"].values
    ]
    legend_elems.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#1E88E5", markeredgecolor="#0D47A1",
               markersize=12, alpha=0.45,
               label="Node total  (bubble ∝ MW)")
    )
    ax.legend(
        handles=legend_elems, loc="lower left",
        fontsize=9, framealpha=0.93, edgecolor=_GRID_C,
        title="Technology", title_fontsize=9,
    )

    # ── Title + subtitle ─────────────────────────────────────────────────────────
    total_mw_all = df["p_nom"].sum()
    n_proj = len(df)
    n_nodes = len(agg)
    ax.set_title(
        "Spain 2030 BESS Pipeline — Project Locations & Node Aggregation",
        fontsize=13, fontweight="bold", pad=12, color=_SLATE,
    )
    fig.text(
        0.5, 0.01,
        f"{total_mw_all:.0f} MW total  ·  {n_proj} projects  ·  {n_nodes} grid nodes  |"
        f"  bubble label = aggregated node MW",
        ha="center", va="bottom", fontsize=8.5, color="#666666", style="italic",
    )

    out_path = out_dir / "bess_node_map.png"
    fig.savefig(out_path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("BESS map saved → %s", out_path)


def _plot_hydro_soc_trajectory(n, out_dir):
    """Hydro SOC trajectories per reservoir in absolute energy (GWh or MWh).

    All subplots share a common y-axis scaled to the largest reservoir in the country,
    so the visual weight of each panel is proportional to its actual energy content.
    A small pond at full capacity appears near the bottom; a large draining reservoir
    fills the plot. Each reservoir's own capacity ceiling is marked with a dotted line.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    soc_raw = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())

    def _plot_country_reservoirs(prefix, filename, line_color, title_suffix):
        su_idx = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith(prefix)
        ]
        if su_idx.empty:
            return
        su_in_soc = [g for g in su_idx if g in soc_raw.columns]
        if not su_in_soc:
            log.warning("_plot_hydro_soc_trajectory [%s]: no reservoirs in state_of_charge table", prefix)
            return

        e_nom = n.storage_units.loc[su_in_soc, "p_nom"] * n.storage_units.loc[su_in_soc, "max_hours"]
        # Auto-scale: use GWh when largest reservoir > 500 MWh, else MWh
        use_gwh = e_nom.max() > 500.0
        scale = 1e3 if use_gwh else 1.0
        unit_lbl = "GWh" if use_gwh else "MWh"

        soc_energy = soc_raw[su_in_soc]  # MWh

        # Sort by e_nom descending so largest reservoirs come first
        order = e_nom.sort_values(ascending=False).index.tolist()
        soc_energy = soc_energy[order]
        e_nom = e_nom[order]

        # Shared y-axis: all subplots use the same scale so large reservoirs
        # visually dominate and small ones appear proportionally small.
        max_cap_scaled = float(e_nom.max()) / scale
        y_top = max_cap_scaled * 1.05

        n_res = len(order)
        n_cols = min(4, n_res)
        n_rows = (n_res + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 2.8),
                                 squeeze=False, sharey=True)
        axes_flat = axes.flatten()

        # Tier band thresholds relative to the shared y-top (largest reservoir)
        tier_bands_shared = [
            (0.60 * max_cap_scaled, y_top,                "#c6e0b4", "≥60% — €15"),
            (0.40 * max_cap_scaled, 0.60 * max_cap_scaled, "#d9ead3", "40–60% — €35"),
            (0.25 * max_cap_scaled, 0.40 * max_cap_scaled, "#ffe699", "25–40% — €55"),
            (0.10 * max_cap_scaled, 0.25 * max_cap_scaled, "#f4cccc", "10–25% — €80"),
            (0.0,                   0.10 * max_cap_scaled, "#e06666", "<10% — €150"),
        ]

        for idx, su_name in enumerate(order):
            ax = axes_flat[idx]
            cap_scaled = float(e_nom[su_name]) / scale

            for lo_e, hi_e, color, _ in tier_bands_shared:
                ax.axhspan(lo_e, hi_e, color=color, alpha=0.25, zorder=0)
            for frac in [0.10, 0.25, 0.40, 0.60]:
                ax.axhline(frac * max_cap_scaled, color="#666", lw=0.4, ls="--", alpha=0.4, zorder=1)

            ax.plot(soc_energy.index, soc_energy[su_name] / scale, lw=0.7, color=line_color, zorder=2)

            # Dotted line at this reservoir's own capacity ceiling
            ax.axhline(cap_scaled, color="#333", lw=0.8, ls=":", alpha=0.6, zorder=1)

            short_name = su_name.split(" ")[-1] if " " in su_name else su_name
            ax.set_title(f"{short_name}\n({cap_scaled:.0f} {unit_lbl})", fontsize=6.5, pad=2)
            ax.set_ylabel(unit_lbl, fontsize=5.5)
            ax.set_ylim(0, y_top)
            ax.set_xlim(soc_energy.index[0], soc_energy.index[-1])
            ax.tick_params(axis="both", labelsize=5)
            ax.tick_params(axis="x", rotation=25)

        for idx in range(n_res, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        legend_elements = [
            Patch(facecolor="#c6e0b4", alpha=0.5, label="≥60% — €15/MWh"),
            Patch(facecolor="#d9ead3", alpha=0.5, label="40–60% — €35/MWh"),
            Patch(facecolor="#ffe699", alpha=0.5, label="25–40% — €55/MWh"),
            Patch(facecolor="#f4cccc", alpha=0.5, label="10–25% — €80/MWh"),
            Patch(facecolor="#e06666", alpha=0.5, label="<10% — €150/MWh"),
        ]
        fig.legend(handles=legend_elements, loc="upper right", fontsize=6,
                   framealpha=0.9, ncol=1, title="SOC tier → MC")

        total_cap_gwh = float(e_nom.sum()) / 1e3
        fig.suptitle(
            f"{prefix} hydro reservoir SOC — individual {unit_lbl} axes | "
            f"{n_res} units, {total_cap_gwh:.0f} GWh total",
            fontsize=9, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Hydro SOC trajectory [%s] → %s", prefix, out_path.name)

    _plot_country_reservoirs("ES", "hydro_soc_trajectory.png",  "#1f77b4", "")
    _plot_country_reservoirs("FR", "hydro_soc_fr.png",          "#E67E22", "")
    _plot_country_reservoirs("PT", "hydro_soc_pt.png",          "#27AE60", "")

    # ── Capacity bar chart: GWh capacity per ES reservoir with SOC fill ───────
    try:
        su_es = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith("ES")
        ]
        if not su_es.empty:
            soc_raw = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())
            e_nom_es = (n.storage_units.loc[su_es, "p_nom"] *
                        n.storage_units.loc[su_es, "max_hours"]) / 1e3   # GWh
            e_nom_es = e_nom_es.sort_values(ascending=True)
            names = [s.split(" ")[-1] if " " in s else s for s in e_nom_es.index]

            soc_init, soc_end = {}, {}
            for su in e_nom_es.index:
                if su in soc_raw.columns:
                    soc_init[su] = float(soc_raw[su].iloc[0])  / 1e3  # GWh
                    soc_end[su]  = float(soc_raw[su].iloc[-1]) / 1e3  # GWh

            fig, ax = plt.subplots(figsize=(9, max(4, len(e_nom_es) * 0.35 + 1)))
            y = np.arange(len(e_nom_es))
            ax.barh(y, e_nom_es.values, color="#BDE0FE", edgecolor="#888", linewidth=0.5,
                    label="Total capacity")
            init_vals = [soc_init.get(s, float("nan")) for s in e_nom_es.index]
            end_vals  = [soc_end.get(s, float("nan"))  for s in e_nom_es.index]
            ax.barh(y, init_vals, color="#457B9D", alpha=0.7, label="Start SOC")
            ax.barh(y, end_vals,  color="#E63946", alpha=0.7, label="End SOC")
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=7)
            ax.set_xlabel("GWh")
            ax.set_title("ES Hydro — Reservoir Capacity & SOC (start vs end)", fontsize=10)
            ax.legend(fontsize=8, loc="lower right")
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            cap_path = out_dir / "hydro_capacity_reservoirs.png"
            fig.savefig(cap_path, dpi=_SAVE_DPI, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved %s", cap_path)
    except Exception as exc:
        log.warning("Hydro capacity bar chart: %s", exc)


# ─── CSV exports ─────────────────────────────────────────────────────────────

def _save_csv_outputs(n, out_dir, cfg, omie=None, real_prices=None):
    """Save key simulation results as CSVs alongside the validation PNGs.

    Files written to out_dir:
      hourly_nodal_prices.csv       — ES bus LMPs (€/MWh) per snapshot
      hourly_national_prices.csv    — ES/FR/PT mean model price + OMIE actual
      hourly_dispatch_national.csv  — dispatch by carrier×country (MW)
      capacity_by_carrier.csv       — installed MW (and GWh for storage) by carrier×country
      vre_curtailment_by_carrier.csv— hourly curtailment by VRE carrier (MW) + total
      price_error_hourly.csv        — model vs OMIE price error with hour/month columns
      line_loading_stats.csv        — % hours congested per internal ES line
      battery_arbitrage.csv         — battery charge/discharge/revenue (if BESS present)
    """
    snap = n.snapshots
    mp   = n.buses_t.marginal_price
    log.info("Saving CSV outputs to %s …", out_dir)

    # 1 — Hourly nodal prices (ES buses only)
    try:
        es_buses = [b for b in mp.columns if str(b).startswith("ES")]
        if es_buses:
            mp[es_buses].to_csv(out_dir / "hourly_nodal_prices.csv")
            log.info("  hourly_nodal_prices.csv (%d buses)", len(es_buses))
    except Exception as exc:
        log.warning("hourly_nodal_prices: %s", exc)

    # 2 — Hourly national prices (ES/FR/PT model + OMIE)
    try:
        fr_buses = [b for b in mp.columns if str(b).startswith("FR")]
        pt_buses = [b for b in mp.columns if str(b).startswith("PT")]
        price_df = pd.DataFrame(index=snap)
        price_df["model_ES"] = mp[es_buses].mean(axis=1) if es_buses else np.nan
        price_df["model_FR"] = mp[fr_buses].mean(axis=1) if fr_buses else np.nan
        price_df["model_PT"] = mp[pt_buses].mean(axis=1) if pt_buses else np.nan
        if omie is not None:
            price_df["omie_ES"] = omie.reindex(snap)
        if real_prices and "PT" in real_prices and real_prices["PT"] is not None:
            price_df["omie_PT"] = real_prices["PT"].reindex(snap)
        if real_prices and "FR" in real_prices and real_prices["FR"] is not None:
            price_df["epex_FR"] = real_prices["FR"].reindex(snap)
        price_df.to_csv(out_dir / "hourly_national_prices.csv")
        log.info("  hourly_national_prices.csv")
    except Exception as exc:
        log.warning("hourly_national_prices: %s", exc)

    # 3 — Hourly dispatch national (all carriers, all three countries)
    try:
        frames = []
        for prefix in ("ES", "FR", "PT"):
            gen_idx = n.generators.index[n.generators.bus.str.startswith(prefix)]
            su_idx  = n.storage_units.index[n.storage_units.bus.str.startswith(prefix)]

            if gen_idx.size > 0 and not n.generators_t.p.empty:
                p_gen = n.generators_t.p.reindex(columns=gen_idx, fill_value=0.0)
                for c, grp in p_gen.groupby(n.generators.loc[gen_idx, "carrier"], axis=1):
                    frames.append(grp.sum(axis=1).rename(f"{prefix}_{c}"))

            if su_idx.size > 0 and "p_dispatch" in n.storage_units_t.keys():
                p_su = n.storage_units_t.p_dispatch.reindex(columns=su_idx, fill_value=0.0)
                for c, grp in p_su.groupby(n.storage_units.loc[su_idx, "carrier"], axis=1):
                    frames.append(grp.sum(axis=1).rename(f"{prefix}_{c}_dispatch"))

        if frames:
            pd.concat(frames, axis=1).fillna(0.0).to_csv(
                out_dir / "hourly_dispatch_national.csv")
            log.info("  hourly_dispatch_national.csv (%d columns)", len(frames))
    except Exception as exc:
        log.warning("hourly_dispatch_national: %s", exc)

    # 4 — Capacity by carrier × country
    try:
        rows = []
        for prefix in ("ES", "FR", "PT"):
            gen = n.generators[n.generators.bus.str.startswith(prefix)]
            for c, grp in gen.groupby("carrier"):
                rows.append({"country": prefix, "carrier": c,
                             "type": "generator",
                             "installed_MW": round(grp.p_nom.sum(), 1)})
            su = n.storage_units[n.storage_units.bus.str.startswith(prefix)]
            for c, grp in su.groupby("carrier"):
                rows.append({"country": prefix, "carrier": c,
                             "type": "storage",
                             "installed_MW":  round(grp.p_nom.sum(), 1),
                             "installed_GWh": round((grp.p_nom * grp.max_hours).sum() / 1e3, 2)})
        pd.DataFrame(rows).to_csv(out_dir / "capacity_by_carrier.csv", index=False)
        log.info("  capacity_by_carrier.csv")
    except Exception as exc:
        log.warning("capacity_by_carrier: %s", exc)

    # 5 — VRE curtailment by carrier
    try:
        vre_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc", "offwind-float"]
        es_gen = n.generators[n.generators.bus.str.startswith("ES")]
        curt_cols = {}
        for c in vre_carriers:
            gens = es_gen.index[es_gen.carrier == c]
            if gens.empty:
                continue
            p_max_cols = [g for g in gens if g in n.generators_t.p_max_pu.columns]
            if not p_max_cols:
                continue
            potential = (n.generators_t.p_max_pu[p_max_cols] *
                         n.generators.loc[p_max_cols, "p_nom"]).sum(axis=1)
            actual    = n.generators_t.p.reindex(columns=gens, fill_value=0.0).sum(axis=1)
            curt_cols[c] = (potential - actual).clip(lower=0.0)
        if curt_cols:
            curt_df = pd.DataFrame(curt_cols, index=snap)
            curt_df["total_curtailed_MW"] = curt_df.sum(axis=1)
            curt_df.to_csv(out_dir / "vre_curtailment_by_carrier.csv")
            log.info("  vre_curtailment_by_carrier.csv (%d VRE carriers)", len(curt_cols))
    except Exception as exc:
        log.warning("vre_curtailment_by_carrier: %s", exc)

    # 6 — Price error hourly
    try:
        if es_buses and omie is not None:
            es_mean = mp[es_buses].mean(axis=1)
            err_df  = pd.DataFrame({
                "model_ES_price": es_mean,
                "omie_price":     omie.reindex(snap),
            }, index=snap)
            err_df["price_error"]     = err_df["model_ES_price"] - err_df["omie_price"]
            err_df["price_error_pct"] = (err_df["price_error"]
                                         / err_df["omie_price"].replace(0.0, np.nan) * 100)
            err_df["hour"]  = err_df.index.hour
            err_df["month"] = err_df.index.month
            err_df.to_csv(out_dir / "price_error_hourly.csv")
            log.info("  price_error_hourly.csv")
    except Exception as exc:
        log.warning("price_error_hourly: %s", exc)

    # 7 — Line loading stats (ES internal lines)
    try:
        if not n.lines_t.p0.empty and not n.lines.empty:
            es_lines = n.lines[
                n.lines.bus0.str.startswith("ES") & n.lines.bus1.str.startswith("ES")
            ]
            if not es_lines.empty:
                p0_abs   = n.lines_t.p0.reindex(columns=es_lines.index, fill_value=0.0).abs()
                s_nom    = es_lines.s_nom.clip(lower=1.0)
                load_f   = (p0_abs / s_nom).clip(upper=1.0)
                stats_df = pd.DataFrame({
                    "bus0":           es_lines.bus0,
                    "bus1":           es_lines.bus1,
                    "s_nom_MW":       es_lines.s_nom,
                    "mean_loading":   load_f.mean().round(4),
                    "p90_loading":    load_f.quantile(0.90).round(4),
                    "pct_gt80pct":    ((load_f > 0.80).mean() * 100).round(2),
                    "pct_gt95pct":    ((load_f > 0.95).mean() * 100).round(2),
                })
                stats_df.sort_values("pct_gt80pct", ascending=False).to_csv(
                    out_dir / "line_loading_stats.csv")
                log.info("  line_loading_stats.csv (%d internal ES lines)", len(es_lines))
    except Exception as exc:
        log.warning("line_loading_stats: %s", exc)

    # 8 — Battery arbitrage (conditional on BESS/PHS present)
    try:
        batt_carriers = {"battery", "PHS_new"}
        batt_su = n.storage_units.index[n.storage_units.carrier.isin(batt_carriers)]
        if not batt_su.empty and "p_dispatch" in n.storage_units_t.keys():
            es_mean_price = mp[es_buses].mean(axis=1) if es_buses else pd.Series(np.nan, index=snap)
            frames_b = {}
            for su in batt_su:
                bus = n.storage_units.at[su, "bus"]
                price_at = mp[bus] if bus in mp.columns else es_mean_price
                p_d = n.storage_units_t.p_dispatch.get(su, pd.Series(0.0, index=snap))
                p_s = n.storage_units_t.p_store.get(su, pd.Series(0.0, index=snap))
                frames_b[f"{su}_dispatch_MW"] = p_d
                frames_b[f"{su}_store_MW"]    = p_s
                frames_b[f"{su}_price"]       = price_at
                frames_b[f"{su}_revenue_eur"] = (p_d - p_s) * price_at
            pd.DataFrame(frames_b, index=snap).to_csv(out_dir / "battery_arbitrage.csv")
            log.info("  battery_arbitrage.csv (%d units)", len(batt_su))
    except Exception as exc:
        log.warning("battery_arbitrage: %s", exc)

    log.info("CSV exports complete.")


def _save_validation_stats_txt(n, out_dir, cfg, omie=None, real_prices=None):
    """Write a structured statistics summary to validation_output/validation_stats.txt.

    Covers the same ground as the dashboard AI-prompt export but computed directly
    from the PyPSA network object — useful for thesis appendices and reproducibility.
    """
    import datetime as _dt_stats
    snap   = n.snapshots
    mp     = n.buses_t.marginal_price
    es_buses = [b for b in mp.columns if str(b).startswith("ES")]
    fr_buses = [b for b in mp.columns if str(b).startswith("FR")]
    pt_buses = [b for b in mp.columns if str(b).startswith("PT")]

    # Use load-weighted mean (same as _mean_es_price) — raw spatial mean pulls toward
    # generator/VRE buses which have near-zero shadow prices and are unrepresentative.
    def _load_weighted_price(prefix):
        buses = [b for b in mp.columns if str(b).startswith(prefix)]
        if not buses:
            return pd.Series(dtype=float)
        weights = {}
        for bus in buses:
            bus_loads = n.loads.index[n.loads["bus"] == bus]
            total = 0.0
            for ld in bus_loads:
                if ld in n.loads_t.p_set.columns:
                    total += float(n.loads_t.p_set[ld].mean())
                else:
                    total += float(n.loads.loc[ld, "p_set"])
            weights[bus] = total
        w = pd.Series(weights, dtype=float)[buses]
        if w.sum() == 0:
            w = pd.Series(1.0, index=buses)
        w = w / w.sum()
        return (mp[buses] * w).sum(axis=1)

    price_es = _mean_es_price(n)
    price_fr = _load_weighted_price("FR")
    price_pt = _load_weighted_price("PT")

    def _s(val, fmt=".1f"):
        try:
            return format(float(val), fmt)
        except Exception:
            return "n/a"

    lines = [
        "# PyPSA-Spain Validation Statistics",
        f"Generated: {_dt_stats.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Period:    {snap[0].strftime('%Y-%m-%d')} → {snap[-1].strftime('%Y-%m-%d')} ({len(snap)} hours)",
        "",
    ]

    # ── 1. Price accuracy ────────────────────────────────────────────────────
    lines += ["## 1. ES Price Accuracy"]
    if omie is not None and not omie.empty:
        omie_aligned = omie.reindex(snap)
        err = price_es - omie_aligned
        corr = price_es.corr(omie_aligned)
        lines += [
            f"  Model mean:    {_s(price_es.mean())} EUR/MWh",
            f"  OMIE mean:     {_s(omie_aligned.mean())} EUR/MWh",
            f"  Mean error:    {_s(err.mean(), '+.1f')} EUR/MWh",
            f"  MAE:           {_s(err.abs().mean())} EUR/MWh",
            f"  RMSE:          {_s((err**2).mean()**0.5)} EUR/MWh",
            f"  Correlation:   {_s(corr, '.3f')}",
            f"  Price p10/p50/p90: {_s(price_es.quantile(0.1))} / "
            f"{_s(price_es.quantile(0.5))} / {_s(price_es.quantile(0.9))} EUR/MWh",
            f"  OMIE  p10/p50/p90: {_s(omie_aligned.quantile(0.1))} / "
            f"{_s(omie_aligned.quantile(0.5))} / {_s(omie_aligned.quantile(0.9))} EUR/MWh",
            f"  Zero-price hours: model={int((price_es < 1).sum())}  OMIE={int((omie_aligned < 1).sum())}",
        ]
    else:
        lines.append("  (no OMIE reference loaded)")
    lines.append("")

    # ── 2. FR/PT price comparison ────────────────────────────────────────────
    lines += ["## 2. Border Country Prices"]
    for cname, pseries, rprice_key in [("FR", price_fr, "FR"), ("PT", price_pt, "PT")]:
        actual_p = (real_prices or {}).get(rprice_key)
        if pseries.empty:
            lines.append(f"  {cname}: no model price (buses absent)")
        else:
            model_mean = _s(pseries.mean())
            actual_str = _s(actual_p.reindex(snap).mean()) if actual_p is not None else "n/a"
            lines.append(f"  {cname}: model={model_mean}  actual={actual_str} EUR/MWh")
    lines.append("")

    # ── 3. ES dispatch breakdown ──────────────────────────────────────────────
    lines += ["## 3. ES Dispatch Breakdown (GWh)"]
    try:
        es_gen_idx = n.generators.index[n.generators.bus.str.startswith("ES")]
        es_su_idx  = n.storage_units.index[n.storage_units.bus.str.startswith("ES")]
        p_gen = n.generators_t.p.reindex(columns=es_gen_idx, fill_value=0.0)
        rows_disp = {}
        for c, grp in p_gen.groupby(n.generators.loc[es_gen_idx, "carrier"], axis=1):
            rows_disp[c] = grp.sum(axis=1).sum() / 1e3  # GWh
        if "p_dispatch" in n.storage_units_t.keys() and es_su_idx.size > 0:
            p_su = n.storage_units_t.p_dispatch.reindex(columns=es_su_idx, fill_value=0.0)
            for c, grp in p_su.groupby(n.storage_units.loc[es_su_idx, "carrier"], axis=1):
                rows_disp[f"{c} (dispatch)"] = grp.sum(axis=1).sum() / 1e3
        total_gwh = sum(rows_disp.values())
        for carrier, gwh in sorted(rows_disp.items(), key=lambda x: -x[1]):
            lines.append(f"  {carrier:<20} {gwh:>8.1f} GWh  ({gwh/total_gwh*100:.1f}%)")
        lines.append(f"  {'TOTAL':<20} {total_gwh:>8.1f} GWh")
    except Exception as exc:
        lines.append(f"  (dispatch breakdown error: {exc})")
    lines.append("")

    # ── 4. VRE curtailment ────────────────────────────────────────────────────
    lines += ["## 4. ES VRE Curtailment"]
    try:
        vre_carriers = ["solar", "onwind", "offwind-ac", "offwind-dc"]
        es_gen = n.generators[n.generators.bus.str.startswith("ES")]
        for c in vre_carriers:
            gens = es_gen[es_gen.carrier == c].index
            if gens.empty:
                continue
            potential = (n.generators_t.p_max_pu[gens] * n.generators.loc[gens, "p_nom"]).sum(axis=1)
            actual    = n.generators_t.p[gens].sum(axis=1)
            curtailed = (potential - actual).clip(lower=0)
            curt_gwh  = float(curtailed.sum()) / 1e3
            pot_gwh   = float(potential.sum()) / 1e3
            pot_sum   = float(potential.sum())
            rate      = curtailed.sum() / pot_sum * 100 if pot_sum > 0 else float("nan")
            lines.append(f"  {c:<15} potential={pot_gwh:.1f} GWh  curtailed={curt_gwh:.1f} GWh  rate={_s(rate)}%")
    except Exception as exc:
        lines.append(f"  (curtailment error: {exc})")
    lines.append("")

    # ── 4b. Price-setter breakdown ────────────────────────────────────────────
    lines += ["## 4b. Price-Setter by Carrier (% hours)"]
    try:
        price_series, setter_series = _get_price_setter_series(n, "ES")
        total_h = max(len(setter_series), 1)
        for carrier in setter_series.value_counts().head(12).index:
            mask   = setter_series == carrier
            n_hrs  = int(mask.sum())
            pct    = n_hrs / total_h * 100
            mean_p = float(price_series[mask].mean()) if mask.any() else float("nan")
            lines.append(f"  {carrier:<22} {n_hrs:>5}h  {pct:>5.1f}%  mean={mean_p:.1f} €/MWh")
        # Price-bin × setter matrix for top-3
        price_bins  = [-999, 10, 30, 50, 70, 90, 120, 9999]
        bin_labels  = ["≤10", "10–30", "30–50", "50–70", "70–90", "90–120", ">120"]
        price_bin_s = pd.cut(price_series, bins=price_bins, labels=bin_labels)
        top_setters = setter_series.value_counts().head(3).index.tolist()
        lines.append("  --- price-bin breakdown (top 3 setters) ---")
        for setter in top_setters:
            smask      = setter_series == setter
            bin_counts = price_bin_s[smask].value_counts()
            parts      = [f"{lbl}:{int(bin_counts.get(lbl, 0))}h" for lbl in bin_labels]
            lines.append(f"  {setter:<18}: {'  '.join(parts)}")
        lines.append("  [Spain 2024 actual: CCGT ~45–55%, VRE ~15–25%, Nuclear ~10–15%, Hydro ~10–15%]")
    except Exception as exc:
        lines.append(f"  (price-setter error: {exc})")
    lines.append("")

    # ── 5. Interconnector flows ───────────────────────────────────────────────
    lines += ["## 5. Interconnector Flows"]
    val_cfg = cfg.get("validation", {})
    fr_csv = val_cfg.get("real_flows_fr_csv")
    pt_csv = val_cfg.get("real_flows_pt_csv")
    real_fr = _load_real_fr_flows(fr_csv, snap) if fr_csv else None
    real_pt = _load_real_balance_csv(pt_csv, snap) if pt_csv else None
    for ctry, real_series in [("FR", real_fr), ("PT", real_pt)]:
        try:
            net = _net_import_topo(n, ctry)
            if net.empty:
                continue
            mean_mw = float(net.mean())
            pct_exp = float((net > 0).mean() * 100)  # >0 = country exports to ES
            pct_imp = float((net < 0).mean() * 100)  # <0 = ES exports to country
            lines.append(
                f"  {ctry}↔ES: mean_net={mean_mw:+.0f} MW  "
                f"({pct_exp:.0f}% hrs {ctry}→ES, {pct_imp:.0f}% hrs ES→{ctry})"
            )
            # Real flow comparison
            if real_series is not None and not real_series.empty:
                real_mean = float(real_series.mean())
                err = mean_mw - real_mean
                lines.append(
                    f"  {ctry}↔ES real: mean_net={real_mean:+.0f} MW  "
                    f"(model error={err:+.0f} MW)"
                )
        except Exception:
            pass
    lines.append("")

    # ── 6. Hydro SOC summary ──────────────────────────────────────────────────
    lines += ["## 6. Hydro SOC (end of simulation)"]
    soc_t = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())
    for pfx in ("ES", "FR", "PT"):
        su_idx = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith(pfx)
        ]
        if su_idx.empty or soc_t.empty:
            continue
        cols = [c for c in su_idx if c in soc_t.columns]
        if not cols:
            continue
        e_nom = (n.storage_units.loc[cols, "p_nom"] * n.storage_units.loc[cols, "max_hours"]).sum()
        soc_end = float(soc_t[cols].iloc[-1].sum())
        inflow_t = getattr(n.storage_units_t, "inflow", pd.DataFrame())
        inflow_cols = [c for c in cols if c in inflow_t.columns]
        inflow_gwh = float(inflow_t[inflow_cols].sum().sum()) / 1e3 if inflow_cols else float("nan")
        lines.append(
            f"  {pfx}: total e-cap={e_nom/1e3:.0f} GWh  "
            f"end_SOC={soc_end/1e3:.0f} GWh ({soc_end/e_nom*100:.0f}%)  "
            f"inflow={inflow_gwh:.0f} GWh (window)"
        )
    lines.append("")

    # ── 6b. Monthly hydro SOC, dispatch & uplift ──────────────────────────────
    lines += ["## 6b. Monthly Hydro — SOC, Dispatch & Uplift over Real (GWh)"]
    try:
        snap = n.snapshots
        months = snap.to_series().dt.month
        month_labels = snap.to_series().dt.strftime("%Y-%m").unique()

        # Model dispatch: storage units (p_dispatch) + generators (ror)
        es_su_idx = n.storage_units.index[
            (n.storage_units["carrier"] == "hydro") &
            n.storage_units["bus"].str.startswith("ES")
        ]
        es_gen_idx = n.generators.index[
            (n.generators["carrier"] == "ror") &
            n.generators["bus"].str.startswith("ES")
        ]
        model_hydro_mw = pd.Series(0.0, index=snap)
        if not es_su_idx.empty and "p_dispatch" in n.storage_units_t:
            su_dispatch = n.storage_units_t.p_dispatch.reindex(columns=es_su_idx, fill_value=0.0)
            model_hydro_mw += su_dispatch.sum(axis=1)
        if not es_gen_idx.empty:
            gen_p = n.generators_t.p.reindex(columns=es_gen_idx, fill_value=0.0)
            model_hydro_mw += gen_p.sum(axis=1)
        model_hydro_gwh = model_hydro_mw.resample("ME").sum() / 1e3  # GWh/month

        # Real ES hydro dispatch (daily GWh from REE → resample monthly)
        real_hydro_gwh = pd.Series(dtype=float)
        try:
            real_daily = _load_real_dispatch(cfg, snap)
            if real_daily is not None and "hydro" in real_daily.columns:
                real_hydro_gwh = real_daily["hydro"].resample("ME").sum()
        except Exception:
            pass

        # SOC trajectory: start-of-month and end-of-month
        soc_t = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())
        es_soc_cols = [c for c in es_su_idx if c in soc_t.columns] if not soc_t.empty else []

        # Header
        lines.append(f"  {'Month':<10}  {'SOC_start':>9}  {'SOC_end':>9}  "
                     f"{'Dispatch':>9}  {'Real_hydro':>10}  {'Uplift':>8}  {'Uplift%':>7}")
        lines.append(f"  {'-'*70}")

        for ml in month_labels:
            mask = snap.to_series().dt.strftime("%Y-%m") == ml
            if not mask.any():
                continue
            month_snaps = snap[mask]
            month_num = month_snaps[0].month

            # SOC at start and end of this calendar month
            soc_start = float("nan")
            soc_end   = float("nan")
            if es_soc_cols:
                # Find first and last snapshot of this calendar month
                first_in_month = month_snaps[0]
                last_in_month  = month_snaps[-1]
                # SOC at first hour of month (or closest prior)
                prior = soc_t[es_soc_cols].loc[:first_in_month]
                if not prior.empty:
                    soc_start = float(prior.iloc[-1].sum()) / 1e3  # GWh
                soc_end = float(soc_t[es_soc_cols].loc[last_in_month].sum()) / 1e3  # GWh

            # Model dispatch this month
            disp = model_hydro_gwh.get(month_snaps[-1], float("nan"))
            if pd.isna(disp):
                # Fallback: sum hourly
                disp = float(model_hydro_mw[month_snaps].sum()) / 1e3

            # Real hydro this month
            real_val = float(real_hydro_gwh.get(month_snaps[-1], float("nan"))) if not real_hydro_gwh.empty else float("nan")

            # Uplift
            if not pd.isna(disp) and not pd.isna(real_val) and real_val != 0:
                uplift = disp - real_val
                uplift_pct = (uplift / real_val) * 100
            else:
                uplift = float("nan")
                uplift_pct = float("nan")

            lines.append(
                f"  {ml:<10}  {_s(soc_start):>9}  {_s(soc_end):>9}  "
                f"{_s(disp):>9}  {_s(real_val):>10}  {_s(uplift,'+.1f'):>8}  {_s(uplift_pct,'+.1f'):>6}%"
            )

        # Total row
        total_disp = float(model_hydro_gwh.sum()) if not model_hydro_gwh.empty else float("nan")
        total_real = float(real_hydro_gwh.sum()) if not real_hydro_gwh.empty else float("nan")
        total_uplift = total_disp - total_real if not pd.isna(total_disp) and not pd.isna(total_real) else float("nan")
        total_uplift_pct = (total_uplift / total_real * 100) if not pd.isna(total_uplift) and not pd.isna(total_real) and total_real != 0 else float("nan")
        lines.append(f"  {'TOTAL':<10}  {'':>9}  {'':>9}  "
                     f"{_s(total_disp):>9}  {_s(total_real):>10}  {_s(total_uplift,'+.1f'):>8}  {_s(total_uplift_pct,'+.1f'):>6}%")
    except Exception as exc:
        lines.append(f"  (monthly hydro error: {exc})")
    lines.append("")

    # ── 7. Capacity summary ────────────────────────────────────────────────────
    lines += ["## 7. Installed Capacity (ES only, GW)"]
    try:
        es_gen = n.generators[n.generators.bus.str.startswith("ES")]
        for c, grp in es_gen.groupby("carrier"):
            lines.append(f"  {c:<20} {grp.p_nom.sum()/1e3:.2f} GW ({len(grp)} units)")
        es_su = n.storage_units[n.storage_units.bus.str.startswith("ES")]
        for c, grp in es_su.groupby("carrier"):
            gwh = (grp.p_nom * grp.max_hours).sum() / 1e3
            lines.append(f"  {c:<20} {grp.p_nom.sum()/1e3:.2f} GW  {gwh:.1f} GWh storage")
    except Exception as exc:
        lines.append(f"  (capacity error: {exc})")
    lines.append("")

    # ── 8. Line congestion ────────────────────────────────────────────────────
    lines += ["## 8. Internal ES Line Congestion"]
    try:
        if not n.lines_t.p0.empty:
            es_lines = n.lines[
                n.lines.bus0.str.startswith("ES") & n.lines.bus1.str.startswith("ES")
            ]
            if not es_lines.empty:
                load_frac = (n.lines_t.p0[es_lines.index].abs() / es_lines.s_nom).clip(upper=1.0)
                pct_gt80  = (load_frac > 0.80).mean() * 100
                n_congested = int((pct_gt80 > 10).sum())
                lines.append(f"  {len(es_lines)} internal ES lines  ({n_congested} congested >10% of hours)")
                top5 = pct_gt80.sort_values(ascending=False).head(5)
                for lid, pct in top5.items():
                    b0, b1 = n.lines.at[lid, "bus0"], n.lines.at[lid, "bus1"]
                    lines.append(f"    {lid}: {b0}→{b1}  {pct:.0f}% hrs >80%")
        else:
            lines.append("  (no line flow data)")
    except Exception as exc:
        lines.append(f"  (line congestion error: {exc})")
    lines.append("")

    txt = "\n".join(lines)
    try:
        out_path = out_dir / "validation_stats.txt"
        out_path.write_text(txt, encoding="utf-8")
        log.info("Validation stats saved → %s", out_path.name)
    except Exception as exc:
        log.warning("validation_stats.txt: %s", exc)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = MODEL_CONFIG
    val = cfg["validation"]

    start  = pd.Timestamp(val["start_date"])
    n_days = int(val["n_days"])
    end    = start + pd.Timedelta(hours=n_days * 24 - 1)
    log.info("Analysis window: %s → %s  (%d days)", start.date(), end.date(), n_days)

    out_dir = ROOT / val["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    net_path = ROOT / val["network_path"]
    log.info("Loading %s", net_path.name)
    n = pypsa.Network(str(net_path))

    # French missing demand (non-Spain export routes)
    n = _add_fr_missing_demand(n, cfg)

    # Refine
    log.info("Applying refinements...")
    n = apply_non_linear_refinements(n, cfg)

    # Monthly hydro MC profiles — overrides static SOC-tiered MC with time-varying
    # monthly values to prevent the "hydro ripple effect" (reservoirs dumping all
    # water in January and sitting empty).  Must run after refinements so the
    # storage_units_t.marginal_cost DataFrame is populated.
    apply_inflow_based_hydro_mc(n, cfg)

    # Hydro inflow diagnostic — verify inflows are non-zero before slicing
    _inflow_t = getattr(n.storage_units_t, "inflow", pd.DataFrame())
    for _pfx in ("ES", "FR", "PT"):
        _su_idx = n.storage_units.index[n.storage_units["bus"].str.startswith(_pfx)]
        _cols = [c for c in _inflow_t.columns if c in _su_idx]
        if _cols:
            _total_gwh = float(_inflow_t[_cols].sum().sum()) / 1e3
            _nonzero   = int((_inflow_t[_cols].sum(axis=0) > 0).sum())
            log.info(
                "Inflow diagnostic [%s]: %d hydro units, %d with inflow, %.1f GWh total (full year)",
                _pfx, len(_su_idx), _nonzero, _total_gwh,
            )
        else:
            log.warning("Inflow diagnostic [%s]: %d hydro units but NONE in inflow table", _pfx, len(_su_idx))

    _add_bess_fleet(n, cfg)

    # BESS geography map — always generated as reference (reads CSV directly)
    _plot_bess_map(n, cfg, out_dir)

    # Slice
    snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
    if len(snap) == 0:
        log.error("No snapshots in [%s, %s] — check start_date", start, end)
        sys.exit(1)
    n.set_snapshots(snap)
    log.info("Sliced to %d snapshots (%s → %s)", len(snap), snap[0], snap[-1])

    # Save pre-solve (post-refinement) network — captures exact LP/MILP problem for
    # reproducibility without re-running the full refinement pipeline.
    import datetime as _dt_ps
    _today_ps = _dt_ps.date.today().strftime("%Y%m%d")
    presolved_dir = ROOT / "solved_networks" / "presolved"
    presolved_dir.mkdir(parents=True, exist_ok=True)
    presolved_path = presolved_dir / f"presolved_{start.strftime('%Y%m%d')}_{n_days}d_{_today_ps}.nc"
    n.export_to_netcdf(str(presolved_path))
    log.info("Pre-solve network saved → %s", presolved_path.name)

    # Solve
    solver_opts = val.get("solver_options", {})
    log.info("Solving with %s (threads=%s)...", val["solver"], solver_opts.get("Threads", "all"))

    def extra_functionality(n, snapshots):
        _add_su_ramp_constraints(n, snapshots, cfg)
        _add_hydro_min_dispatch(n, snapshots, cfg)

    def _write_iis(label=""):
        """Compute Gurobi IIS and write the .ilp file so infeasible constraints are visible."""
        try:
            m = n.model.solver_model
            m.computeIIS()
            iis_path = out_dir / f"infeasible{'_' + label if label else ''}.ilp"
            m.write(str(iis_path))
            log.error("IIS written → %s  (open in any text editor — each line is a conflicting constraint)", iis_path)

            # ── Diagnostic: map IIS constraint indices to PyPSA constraint names ──
            try:
                import gurobipy as gp
                iis_constrs = m.getConstrs()
                iis_info = []
                for c in iis_constrs:
                    if c.IISConstr:
                        cname = c.ConstrName
                        sense = "≤" if c.Sense == "<" else "≥" if c.Sense == ">" else "="
                        iis_info.append(f"  {cname}: {sense} {c.RHS:.4f}  (IIS={c.IISConstr})")
                if iis_info:
                    log.error("IIS constraint details (%d in conflict):\n%s", len(iis_info), "\n".join(iis_info))
                else:
                    log.error("IIS: no constraints in conflict (check variable bounds)")
                # ── Dump ALL IIS constraint names with their linopy/PyPSA names ──
                for c in iis_constrs:
                    if c.IISConstr:
                        cname = c.ConstrName
                        log.error("IIS constraint: idx=%s name='%s' sense=%s rhs=%.4f",
                                  c.index, cname, c.Sense, c.RHS)
                        # Try to find the linopy constraint by name
                        try:
                            if cname in n.model.constraints:
                                ln_con = n.model.constraints[cname]
                                log.error("  → linopy constraint: %s", cname)
                                # Show the first few terms of the constraint expression
                                log.error("  → expr: %s", str(ln_con)[:200])
                        except Exception:
                            pass
                # ── Map linopy constraint names to Gurobi indices ──
                target_indices = {189590, 588613, 451189}
                try:
                    # Dump ALL linopy constraint names that contain key terms
                    for ln_name in list(n.model.constraints.keys()):
                        if any(kw in ln_name for kw in ['StorageUnit', 'hydro', 'min-dispatch', 'spill', 'state_of_charge']):
                            log.error("LINOPY CONSTRAINT: %s", ln_name)
                    # Also search for the specific names
                    for ln_name in list(n.model.constraints.keys()):
                        if ln_name in ('c189590', 'c588613'):
                            ln_con = n.model.constraints[ln_name]
                            log.error("LINOPY MATCH: name='%s'", ln_name)
                            log.error("  expr: %s", str(ln_con)[:300])
                except Exception as e:
                    log.warning("Linopy→Gurobi mapping failed: %s", e)
                    
                for c in iis_constrs:
                    if c.ConstrName in ('c189590', 'c588613'):
                        log.error("GUROBI MATCH: idx=%s name='%s' sense=%s rhs=%.4f IIS=%s",
                                  c.index, c.ConstrName, c.Sense, c.RHS, c.IISConstr)
            except Exception as map_exc:
                log.warning("IIS constraint mapping failed: %s", map_exc)

            # ── Diagnostic: dump storage unit state at infeasibility ──
            _su_df = n.storage_units[["carrier", "p_nom", "max_hours", "state_of_charge_initial"]].copy()
            _su_df["e_nom"] = _su_df["p_nom"] * _su_df["max_hours"]
            _su_df["soc0_ratio"] = _su_df["state_of_charge_initial"] / _su_df["e_nom"].replace(0, float("nan"))
            _bad = _su_df[_su_df["state_of_charge_initial"] > _su_df["e_nom"]]
            if not _bad.empty:
                log.error("IIS context — %d unit(s) with soc0 > e_nom:\n%s", len(_bad), _bad.to_string())
            # Also show units with e_nom < 100 MWh (small units most likely to cause IIS)
            _small = _su_df[_su_df["e_nom"] < 100]
            if not _small.empty:
                log.error("IIS context — small units (e_nom < 100 MWh):\n%s", _small.to_string())
        except Exception as iis_exc:
            log.warning("IIS unavailable (%s) — check solver log for status", iis_exc)

    # ── Single LP solve ──────────────────────────────────────────────────────
    try:
        status, cond = n.optimize(snapshots=n.snapshots, solver_name=val["solver"],
                                   solver_options=solver_opts,
                                   extra_functionality=extra_functionality)
    except Exception as exc:
        log.error("Solve failed: %s", exc)
        _write_iis("full")
        raise
    log.info("Solve returned: status=%r condition=%r  objective=%.2f M€",
             status, cond, n.objective / 1e6)
    if status != "ok":
        log.error("Non-ok status=%r condition=%r — aborting", status, cond)
        _write_iis("full")
        raise RuntimeError(f"Infeasible/failed solve: {cond}")

    # Save solved network
    import datetime as _dt
    _today = _dt.date.today().strftime("%Y%m%d")
    solved_dir = ROOT / "solved_networks" / "validation"
    solved_dir.mkdir(parents=True, exist_ok=True)
    solved_path = solved_dir / f"solved_{start.strftime('%Y%m%d')}_{n_days}d_{_today}.nc"
    n.export_to_netcdf(str(solved_path))
    log.info("Saved solved network → %s", solved_path.name)

    # Reference data
    omie       = _load_omie(cfg, n.snapshots)
    real_daily = _load_real_dispatch(cfg, n.snapshots)
    model_daily_es = _to_daily_gwh(_dispatch_by_carrier(n, "ES"))

    # Console statistics
    _print_stats(n, omie, model_daily_es, real_daily, start, n_days, cfg)
    _print_cost_and_price_setter_table(n)

    # VRE bottleneck diagnostic — runs on a minimal stub DataFrame so it can
    # compute price-setter internally and print the congestion summary.
    _diag_stub = pd.DataFrame(index=n.snapshots)
    add_vre_bottleneck_diagnostics(n, _diag_stub, cfg=cfg)
    _print_vre_price_setter_diagnostic(n, omie, cfg=cfg)

    # Plots — all saved to out_dir, overwritten each run
    _plot_week_overview(n, omie, out_dir, cfg=cfg)  # 01 — model vs REE dispatch + price, middle week
    _plot_price_duration(n, omie, out_dir)          # 02 — price duration / frequency curve
    _plot_capacity_vs_reality(n, out_dir)           # 03 — model vs REE installed capacity
    _plot_network_map(n, out_dir)                   # 04 — ES/FR/PT geographic network map
    _plot_curtailment_map(n, out_dir)               # 05 — Spain VRE curtailment by node
    _plot_node_dispatch_pies(n, out_dir)            # 06 — Spain generation mix pies per node
    _plot_hourly_dispatch(n, out_dir)               # 07 — full-period dispatch ES/FR/PT
    _plot_temporal_comparison(n, real_daily, n_days, out_dir)      # 08  — ES model vs real
    _plot_country_temporal_dispatch(n, n_days, out_dir)            # 08b/c — PT + FR model
    _plot_merit_order_es(n, out_dir)                # 09 — Spain merit order + dispatch overlay
    _plot_merit_order_combined(n, out_dir)          # 10 — ES/FR/PT merit orders side by side
    _plot_week_vs_ree_hourly(n, out_dir)            # 11 — middle week model vs REE hourly actuals
    _plot_interconnector_flows(n, cfg, out_dir)     # 12 — ES↔FR and ES↔PT flows, 4 resolutions
    _plot_cost_and_price_setter(n, out_dir)         # 13 — MC table + price-setter frequency
    real_prices = {c: _load_real_prices(cfg, n.snapshots, c) for c in ["ES", "FR", "PT"]}
    _plot_price_setter_analysis(n, out_dir, real_prices=real_prices)  # 14 — PDC coloured by setter + frequency, ES/FR/PT
    _plot_spain_pdc_hd(n, out_dir, omie=omie)        # 15 — Spain-only HD PDC, 300 dpi, OMIE overlay + FR ticks
    _plot_hydro_soc_trajectory(n, out_dir)            # 16 — ES hydro SOC trajectories with tier bands
    _plot_es_pt_joint_pdc(n, out_dir,
                          omie_es=real_prices.get("ES"),
                          omie_pt=real_prices.get("PT"))  # 17 — ES+PT joint PDC, MIBEL coupling diagnostic
    _plot_ic_tech_composition(n, out_dir)            # 18 — IC export/import tech composition
    _plot_line_bottleneck(n, out_dir)                # 19 — ES transmission bottleneck map
    _plot_price_setter_breakdown(n, out_dir)         # 20 — price-setter by carrier × price bin
    _plot_vre_diagnostic(n, omie, out_dir)           # 21 — Groups A–F VRE price-setter diagnostic
    _save_csv_outputs(n, out_dir, cfg,
                      omie=omie, real_prices=real_prices)  # CSV exports
    _save_validation_stats_txt(n, out_dir, cfg,
                               omie=omie, real_prices=real_prices)  # stats summary txt

    SEP = "─" * 60
    print(f"\n{SEP}")
    print("  OUTPUT FILES  (all in Analysis/validation_output/)")
    print(SEP)
    descriptions = {
        "01_week_overview.png":        "Model vs REE dispatch stack + price vs OMIE — middle week",
        "02_price_duration_curve.png": "Price frequency / duration curve",
        "03_capacity_vs_reality.png":  "Installed capacity: model vs 2024 REE",
        "04_network_map.png":          "ES / FR / PT geographic network + interconnectors",
        "05_curtailment_map.png":      "Spain VRE curtailment intensity by node",
        "06_node_dispatch_pies.png":   "Spain generation mix pie at each node",
        "07_hourly_dispatch.png":      "Full-period hourly dispatch ES / FR / PT",
        "08_daily_comparison.png":          "ES model vs real daily dispatch",
        "08_weekly_comparison.png":         "ES model vs real weekly dispatch",
        "08_monthly_comparison.png":        "ES model vs real monthly dispatch",
        "08b_daily_pt.png":    "PT model vs real daily dispatch",
        "08b_weekly_pt.png":   "PT model vs real weekly dispatch (real=monthly GWh)",
        "08c_daily_fr.png":    "FR model vs real daily dispatch",
        "08c_weekly_fr.png":   "FR model vs real weekly dispatch (real=monthly GWh)",
        "09_merit_order_spain.png":    "Spain merit order + avg dispatch overlay",
        "10_merit_order_combined.png": "Merit order comparison ES / FR / PT",
        "11_week_vs_ree_hourly.png":   "First week model vs REE hourly actuals (+ solar zoom)",
        "12a_interconnector_flows_FR.png": "ES↔FR net flows — 4 time resolutions (model vs ENTSO-E)",
        "12b_interconnector_flows_PT.png": "ES↔PT net flows — 4 time resolutions (model only)",
        "13_cost_structure.png":       "MC table by technology × country + price-setter frequency",
        "14_price_setter_analysis.png":"PDC coloured by price-setter + frequency bar — ES/FR/PT",
        "15_spain_pdc_setter.png":     "Spain HD PDC by price-setter — 300 dpi, non-obscuring FR overlay",
        "16_hydro_soc_trajectory.png": "ES hydro reservoir SOC trajectories with MC tier bands",
        "17_es_pt_joint_pdc.png":  "ES+PT joint PDC — MIBEL coupling diagnostic, shaded gap + congestion stats",
        "18_ic_tech_composition.png": "IC export/import tech composition — carrier mix by flow direction (FR/PT)",
        "19_line_bottleneck.png":  "ES transmission bottleneck map — % hours at >80% loading per line",
        "21_vre_price_setter_diag.png": "Groups A–F VRE diagnostic: supply cover / utilisation / hydro SOC / CCGT MC",
        "hourly_nodal_prices.csv":    "ES bus LMPs (€/MWh) per snapshot",
        "hourly_national_prices.csv": "ES/FR/PT model price + OMIE/EPEX actual (€/MWh)",
        "hourly_dispatch_national.csv": "Dispatch by carrier × country (MW) per snapshot",
        "capacity_by_carrier.csv":    "Installed MW and GWh by carrier × country (static)",
        "vre_curtailment_by_carrier.csv": "Hourly VRE curtailment by carrier + total (MW)",
        "price_error_hourly.csv":     "Model vs OMIE price error with hour/month columns",
        "line_loading_stats.csv":     "ES internal line congestion statistics (% hours >80/95%)",
        "battery_arbitrage.csv":      "Battery charge/discharge/revenue per unit (if BESS enabled)",
    }
    for fname, desc in descriptions.items():
        exists = (out_dir / fname).exists()
        tick   = "✓" if exists else "–"
        print(f"  {tick}  {fname:<36}  {desc}")
    print(SEP + "\n")


if __name__ == "__main__":
    main()
