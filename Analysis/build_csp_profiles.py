"""
build_csp_profiles.py — Standalone script: generate CSP inflow profiles (MW_e)
aggregated to the 50-node network buses, scaled to ~5,000 GWh/year.

Reads ``Analysis/data/CSP_Spain.csv`` (53 plants, 2,303 MW total),
loads the ERA5 cutout, maps each plant to its closest ES network bus via
haversine, aggregates capacity per bus, and runs ``atlite.Cutout.csp()``
to produce one capacity-factor time series per bus.

The CF is then converted to MW_electrical-equivalent inflow by scaling so
that the annual sum across all buses equals CSP_TARGET_GWH (5,000 GWh).
This represents the solar heat collected by the parabolic trough field,
already converted to electrical-equivalent MW (post-turbine), so
efficiency_dispatch=1.0 in the StorageUnit.

Usage
-----
    pixi run python Analysis/build_csp_profiles.py

Output
------
    Analysis/data/csp_profiles.nc
        Dimensions: (bus: ~20–30, time: N)
        Variables:
            inflow    (bus, time)  float32  — solar inflow in MW_e (post-turbine eq.)
            capacity  (bus)        float32  — turbine nameplate MW per bus
            n_plants  (bus)        int32    — number of plants mapped to this bus
"""

import logging
import sys
from pathlib import Path

import atlite
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from scipy.sparse import csr_matrix

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
CSV_PATH     = ROOT / "Analysis" / "data" / "CSP_Spain.csv"
CUTOUT_DIR   = ROOT / "data" / "cutout" / "archive" / "v1.0"
NETWORK_PATH = ROOT / "resources" / "networks" / "base_s_50_elec_2704_fixed.nc"
OUTPUT       = ROOT / "Analysis" / "data" / "csp_profiles.nc"
CONFIG_PATH  = ROOT / "Analysis" / "config.py"

log = logging.getLogger(__name__)


def _haversine_closest(lat, lon, bus_lats, bus_lons, bus_names):
    """Return the name of the bus closest to (lat, lon) via haversine."""
    dlat = np.radians(bus_lats - lat)
    dlon = np.radians(bus_lons - lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat)) * np.cos(np.radians(bus_lats))
         * np.sin(dlon / 2) ** 2)
    return bus_names[np.argmin(a)]


def _load_target_gwh():
    """Read target_gwh from the solar_thermal section of Analysis/config.py."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("analysis_config", CONFIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    # Don't add to sys.modules — we just want the dict
    spec.loader.exec_module(mod)
    cfg = mod.MODEL_CONFIG.get("solar_thermal", {})
    return float(cfg.get("target_gwh", 5000.0))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    CSP_TARGET_GWH = _load_target_gwh()
    log.info("CSP target annual inflow: %.0f GWh_e", CSP_TARGET_GWH)

    # ── 1. Load plant list ────────────────────────────────────────────────────
    plants = pd.read_csv(CSV_PATH)
    n_plants = len(plants)
    total_mw = plants["Capacity (MW)"].sum()
    log.info("Loaded %d CSP plants (%.0f MW total) from %s",
             n_plants, total_mw, CSV_PATH)

    # ── 2. Load base network → get ES bus coordinates ─────────────────────────
    log.info("Loading base network: %s", NETWORK_PATH)
    n = pypsa.Network(str(NETWORK_PATH))
    es_buses = n.buses[n.buses["country"] == "ES"]
    bus_names = es_buses.index.values
    bus_lats = es_buses["y"].values.astype(float)
    bus_lons = es_buses["x"].values.astype(float)
    log.info("Found %d ES buses in the 50-node network", len(es_buses))

    # ── 3. Map each plant to closest ES bus ───────────────────────────────────
    plants["closest_bus"] = plants.apply(
        lambda r: _haversine_closest(
            float(r["Latitude"]), float(r["Longitude"]),
            bus_lats, bus_lons, bus_names,
        ),
        axis=1,
    )

    # Aggregate capacity per bus
    bus_agg = (
        plants.groupby("closest_bus")
        .agg(capacity=("Capacity (MW)", "sum"), n_plants=("Capacity (MW)", "count"))
        .sort_values("capacity", ascending=False)
    )
    n_buses = len(bus_agg)
    log.info("Mapped %d plants to %d buses", n_plants, n_buses)
    for bus_name, row in bus_agg.head(5).iterrows():
        log.info("  %s: %.0f MW (%d plants)", bus_name, row["capacity"], row["n_plants"])
    if n_buses > 5:
        log.info("  ... and %d more buses", n_buses - 5)

    # ── 4. Locate and load cutout ─────────────────────────────────────────────
    cutout_files = sorted(CUTOUT_DIR.glob("*.nc"))
    if not cutout_files:
        log.error("No cutout files found in %s", CUTOUT_DIR)
        sys.exit(1)

    cutout_path = str(cutout_files[0])
    log.info("Loading cutout: %s", cutout_path)
    cutout = atlite.Cutout(cutout_path)

    # ── 5. Build bus-level aggregation matrix ─────────────────────────────────
    # For each bus, we need the capacity-weighted average CF of all plants
    # mapped to it.  We build an N_bus × S matrix where each row has the
    # capacity weight for each grid cell that contains plants mapped to that bus.
    #
    # The cutout grid is a flat GeoDataFrame with one row per (y, x) cell,
    # ordered as y-major (all x for y=33.0, then all x for y=33.3, ...).
    # We snap each plant to the nearest grid cell and build a sparse matrix
    # mapping (bus → grid_cell) with capacity weights.
    grid = cutout.grid                     # flat GeoDataFrame, shape (N_spatial,)
    grid_x = grid["x"].values              # flat array of x coords
    grid_y = grid["y"].values              # flat array of y coords
    n_spatial = len(grid)

    # Unique sorted x/y for searchsorted-based snapping
    ux = np.sort(grid["x"].unique())
    uy = np.sort(grid["y"].unique())
    nx = len(ux)

    # Snap each plant to the nearest grid cell using searchsorted
    plant_lons = plants["Longitude"].values
    plant_lats = plants["Latitude"].values
    ix = np.searchsorted(ux, plant_lons, side="left")
    ix = np.clip(ix, 0, nx - 2)
    ix = np.where(plant_lons - ux[ix] > ux[ix + 1] - plant_lons, ix + 1, ix)
    iy = np.searchsorted(uy, plant_lats, side="left")
    iy = np.clip(iy, 0, len(uy) - 2)
    iy = np.where(plant_lats - uy[iy] > uy[iy + 1] - plant_lats, iy + 1, iy)

    # Convert (iy, ix) to flat index into the grid GeoDataFrame (y-major order)
    spatial_idx = iy * nx + ix

    # Sanity check: all indices must be within [0, n_spatial)
    assert spatial_idx.min() >= 0 and spatial_idx.max() < n_spatial, (
        f"spatial_idx out of bounds: [{spatial_idx.min()}, {spatial_idx.max()}] "
        f"vs n_spatial={n_spatial}"
    )

    # Build bus → index mapping
    bus_order = bus_agg.index.tolist()
    bus_to_idx = {b: i for i, b in enumerate(bus_order)}

    # For each plant, add its capacity weight to the correct (bus, grid_cell)
    plant_weights = plants["Capacity (MW)"].values
    plant_bus_idx = np.array([bus_to_idx[b] for b in plants["closest_bus"]])
    bus_capacity = bus_agg["capacity"].values  # total MW per bus

    # Weight = plant_cap / bus_total_cap  (so matrix row sums to 1.0 per bus)
    weights = plant_weights / bus_capacity[plant_bus_idx]

    rows = plant_bus_idx
    cols = spatial_idx
    M = csr_matrix((weights, (rows, cols)), shape=(n_buses, n_spatial))

    # Wrap as xr.DataArray for atlite's alignment check
    spatial_index = pd.MultiIndex.from_frame(grid[["x", "y"]])
    M_xr = xr.DataArray(
        M.toarray(),
        dims=("bus", "spatial"),
        coords={
            "bus": bus_order,
            "spatial": spatial_index,
        },
    )

    # ── 6. Run atlite CSP conversion ──────────────────────────────────────────
    log.info("Running atlite csp() for %d buses (parabolic trough, lossless) ...",
             n_buses)

    # csp() → convert_and_aggregate with matrix=M_xr, per_unit=True
    # Returns capacity factor time series [0–1] per bus
    # The matrix row sums to 1.0, so per_unit=True gives the capacity-weighted
    # average CF across all plants mapped to that bus.
    csp_cf = cutout.csp(
        installation="lossless_installation",
        technology="parabolic trough",
        matrix=M_xr,
        index=pd.Index(bus_order, name="bus"),
        per_unit=True,
        show_progress=True,
    )
    log.info("CSP CF computed: shape = %s, range = [%.4f, %.4f]",
             csp_cf.shape, float(csp_cf.min()), float(csp_cf.max()))

    # ── 7. Scale CF → MW_electrical-equivalent inflow ─────────────────────────
    # The CF from atlite lossless is DNI/1000, representing the fraction of
    # 1 kW/m² that hits the collector.  We convert to MW_electrical-equivalent
    # by scaling so that the annual sum across all buses = CSP_TARGET_GWH.
    #
    # This lumps together: solar field area, turbine efficiency (~40%),
    # and the solar multiple (~1.3–1.5).  The result is in MW_electrical-
    # equivalent — already post-turbine — so the StorageUnit can use
    # efficiency_dispatch=1.0.
    #
    # inflow_mw[t, bus] = cf[t, bus] * scale_factor
    # where scale_factor is chosen so that sum(inflow) = CSP_TARGET_GWH * 1000 MWh
    cf_values = csp_cf.values  # shape (time, bus)
    total_cf_sum = float(cf_values.sum())  # sum across all (t, bus)
    target_mwh = CSP_TARGET_GWH * 1000.0
    scale_factor = target_mwh / total_cf_sum if total_cf_sum > 0 else 0.0

    inflow_mw = cf_values * scale_factor  # MW_electrical-equivalent

    log.info(
        "CSP inflow scaled: total CF sum = %.1f, scale_factor = %.2f, "
        "target = %.0f GWh, achieved = %.0f GWh",
        total_cf_sum, scale_factor, CSP_TARGET_GWH,
        float(inflow_mw.sum()) / 1000.0,
    )

    # ── 8. Build output dataset ───────────────────────────────────────────────
    ds = xr.Dataset(
        {
            "inflow":   (["time", "bus"], inflow_mw.astype(np.float32)),
            "capacity": (["bus"], bus_agg["capacity"].values.astype(np.float32)),
            "n_plants": (["bus"], bus_agg["n_plants"].values.astype(np.int32)),
        },
        coords={
            "bus": bus_order,
            "time": csp_cf.time.values,
        },
    )
    ds["inflow"].attrs = {
        "units": "MW_e",
        "description": (
            "CSP solar inflow in MW_electrical-equivalent (post-turbine). "
            "Scaled from atlite lossless CF so annual sum = 5,000 GWh."
        ),
    }
    ds["capacity"].attrs = {"units": "MW", "description": "Turbine nameplate capacity per bus"}
    ds.attrs = {
        "source": str(CSV_PATH),
        "network": str(NETWORK_PATH),
        "cutout": cutout_path,
        "n_plants": n_plants,
        "n_buses": n_buses,
        "total_capacity_mw": total_mw,
        "target_gwh": CSP_TARGET_GWH,
        "technology": "parabolic trough (lossless, scaled to 5 TWh)",
    }

    # ── 9. Save ───────────────────────────────────────────────────────────────
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUTPUT)
    log.info("Saved profiles to %s", OUTPUT)
    log.info(
        "Summary: %d buses, %d plants, %.0f MW turbine, %d time steps, "
        "mean inflow = %.1f MW_e, annual = %.0f GWh",
        n_buses, n_plants, total_mw, len(ds.time),
        float(inflow_mw.mean(axis=0).sum()),
        float(inflow_mw.sum()) / 1000.0,
    )


if __name__ == "__main__":
    main()
