"""
Data extraction and solve utilities for the PyPSA-Spain dashboard.

Wraps run_validation.py helpers, handles serialisation to/from Dash dcc.Store,
and runs fresh Gurobi solves in a background thread.
"""

import copy
import logging
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).parent.parent
_IC_DIR = ROOT / "Analysis" / "interconnector_analysis"
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG
from refinery import apply_non_linear_refinements
from run_validation import (
    _add_fr_missing_demand,
    _add_su_ramp_constraints,
    _add_hydro_min_dispatch,
    _dispatch_by_carrier,
    _es_buses,
    _get_price_setter_series,
    _load_omie,
    _net_es_import_series,
    _mean_es_price,
)

log = logging.getLogger(__name__)

SOLVED_DIR = ROOT / "solved_networks" / "validation"
BASE_NET   = ROOT / MODEL_CONFIG["validation"]["network_path"]
OMIE_CSV   = ROOT / MODEL_CONFIG["validation"]["omie_csv"]
REE_CSV    = ROOT / "data" / "validation" / "spain_actual_generation_2024.csv"

# REE (ENTSO-E) column → model carrier name (for colour consistency)
_REE_TO_CARRIER = {
    "Nuclear":         "nuclear",
    "Wind":            "onwind",
    "Solar_PV":        "solar",
    "Hydro_Reservoir": "hydro",
    "Hydro_River":     "ror",
    "CCGT":            "CCGT",
    "Coal":            "coal",
    "Cogeneration":    "cogen",    # gas-fired industrial CHP (not solid biomass)
    "Other":           "other",
}

# ── Solve state shared between thread and Dash callbacks ──────────────────────
_solve_state = {"status": "idle", "data": None, "error": None}
_solve_lock  = threading.Lock()


def _parse_entsoe_ic(path) -> pd.Series:
    """Parse ENTSOE hourly interconnector balance CSV → UTC-indexed MW series.

    Handles two formats:
      - New (plain decimal):   value column already float64, e.g. 2576.27
      - Old (European locale): string "2.576,272" = 2576.272 MW
    Sign convention: positive = exporting country sends to Spain (Spain imports).
    Deduplicates on UTC timestamp (DST transitions create duplicate rows).
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    raw = df["value"]
    if pd.api.types.is_numeric_dtype(raw):
        df["mw"] = raw.astype(float)
    else:
        df["mw"] = (
            raw.astype(str)
            .str.replace(".", "", regex=False)   # remove thousands dot
            .str.replace(",", ".", regex=False)  # decimal comma → point
            .astype(float)
        )
    df.index = pd.to_datetime(df["datetime"], utc=True)
    # Drop NaT (malformed rows) and duplicate UTC timestamps (DST fold)
    df = df[df.index.notna()]
    df = df[~df.index.duplicated(keep="first")]
    return df["mw"].sort_index()


def list_solved_networks() -> list[dict]:
    """Return [{label, value}] sorted newest-first for the network dropdown."""
    files = sorted(SOLVED_DIR.glob("solved_*.nc"),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"label": f.name, "value": str(f)} for f in files]


# ── Data extraction ───────────────────────────────────────────────────────────

def load_and_extract(nc_path: str) -> dict:
    """Load a pre-solved .nc file and extract all dashboard time-series."""
    n = pypsa.Network(str(nc_path))
    return _extract(n)


def _net_import_auto(n: "pypsa.Network", country: str) -> "pd.Series":
    """ES net import from country (positive = ES imports), topology-agnostic.

    Works with both old DC_ic link-pair topology and new AC Line + INELFE topology.
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


def _extract(n: pypsa.Network) -> dict:
    """Extract time-series from a solved network into a JSON-serialisable dict."""
    dispatch_es = _dispatch_by_carrier(n, "ES").copy()
    price_es, setter_es = _get_price_setter_series(n, "ES")

    fr_net = _net_import_auto(n, "FR")
    pt_net = _net_import_auto(n, "PT")

    # Node-level prices for spatial spread analysis
    es_bus_cols = [b for b in _es_buses(n) if b in n.buses_t.marginal_price.columns]
    bus_prices  = (n.buses_t.marginal_price[es_bus_cols]
                   if es_bus_cols else pd.DataFrame())

    # OMIE actual prices (best-effort)
    omie = None
    try:
        omie = _load_omie(MODEL_CONFIG, n.snapshots)
    except Exception:
        pass

    # France and Portugal actual market prices (same CSV format, UTC column)
    def _load_market_price(csv_key: str) -> pd.Series | None:
        path = ROOT / MODEL_CONFIG["validation"].get(csv_key, "")
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
            df.index = pd.to_datetime(
                df["Datetime (UTC)"], format="%d/%m/%y %H:%M", dayfirst=True
            )
            # Align to network snapshots, allow ±90 min tolerance for DST edge cases
            _snaps_utc = n.snapshots.tz_localize("UTC") if n.snapshots.tz is None else n.snapshots
            raw = df["Price (EUR/MWhe)"]
            raw.index = raw.index.tz_localize("UTC")
            aligned = raw.reindex(_snaps_utc, method="nearest",
                                   tolerance=pd.Timedelta("90min"))
            return pd.Series(aligned.values, index=n.snapshots, dtype=float)
        except Exception:
            log.debug("Market price load failed for %s", csv_key, exc_info=True)
            return None

    omie_fr = _load_market_price("france_prices_csv")
    omie_pt = _load_market_price("portugal_prices_csv")

    # ── Split CCGT aggregate into low / mid / high MC tranches ────────────────
    # This replaces the single "CCGT" column with three shaded bands so the
    # dispatch chart shows which cost-tier of gas is actually running.
    ccgt_bounds: dict = {}
    es_ccgt_idx = n.generators.index[
        (n.generators.carrier == "CCGT") &
        n.generators.bus.str.startswith("ES")
    ]
    if len(es_ccgt_idx) > 0 and "p" in n.generators_t:
        mc      = n.generators.loc[es_ccgt_idx, "marginal_cost"]
        mc_lo   = float(mc.quantile(0.33))
        mc_hi   = float(mc.quantile(0.67))
        gp      = n.generators_t.p

        def _tranche_sum(idx):
            cols = [g for g in idx if g in gp.columns]
            return gp[cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0) if cols else pd.Series(0.0, index=n.snapshots)

        dispatch_es["CCGT_lo"]  = _tranche_sum(es_ccgt_idx[mc <= mc_lo])
        dispatch_es["CCGT_mid"] = _tranche_sum(es_ccgt_idx[(mc > mc_lo) & (mc <= mc_hi)])
        dispatch_es["CCGT_hi"]  = _tranche_sum(es_ccgt_idx[mc > mc_hi])
        dispatch_es = dispatch_es.drop(columns=["CCGT"], errors="ignore")
        ccgt_bounds = {"lo": mc_lo, "hi": mc_hi,
                       "lo_cap": float(n.generators.loc[es_ccgt_idx[mc <= mc_lo], "p_nom"].sum()),
                       "mid_cap": float(n.generators.loc[es_ccgt_idx[(mc > mc_lo) & (mc <= mc_hi)], "p_nom"].sum()),
                       "hi_cap": float(n.generators.loc[es_ccgt_idx[mc > mc_hi], "p_nom"].sum())}

    # ── Installed capacity by carrier (generators + storage, ES only) ─────────
    cap: dict = {}
    # Generators — exclude load_shedding backstop
    es_gen = n.generators[
        n.generators.bus.str.startswith("ES") &
        (n.generators.carrier != "load_shedding")
    ]
    # Merge CCGT tranches back into named buckets for capacity chart
    for carrier, grp in es_gen.groupby("carrier"):
        if carrier == "CCGT":
            # Use actual MC-tranche split if we have it
            if ccgt_bounds:
                mc_g = n.generators.loc[grp.index, "marginal_cost"]
                cap["CCGT_lo"]  = float(grp.loc[mc_g <= ccgt_bounds["lo"],  "p_nom"].sum())
                cap["CCGT_mid"] = float(grp.loc[(mc_g > ccgt_bounds["lo"]) & (mc_g <= ccgt_bounds["hi"]), "p_nom"].sum())
                cap["CCGT_hi"]  = float(grp.loc[mc_g > ccgt_bounds["hi"],  "p_nom"].sum())
            else:
                cap["CCGT"] = float(grp["p_nom"].sum())
        else:
            cap[carrier] = float(grp["p_nom"].sum())
    # Storage units
    es_su = n.storage_units[n.storage_units.bus.str.startswith("ES")]
    for carrier, grp in es_su.groupby("carrier"):
        cap[carrier] = float(grp["p_nom"].sum())

    # ── ES total load (demand) ─────────────────────────────────────────────────
    es_load: list = []
    try:
        es_load_idx = n.loads[n.loads.bus.str.startswith("ES")].index
        if len(es_load_idx) > 0 and "p_set" in n.loads_t:
            es_load = n.loads_t.p_set[es_load_idx].sum(axis=1).reindex(n.snapshots).tolist()
    except Exception:
        pass

    # ── Spatial map: bus coords, line topology, per-snapshot loadings ─────────
    map_meta: dict          = {}
    line_loadings_raw: dict = {}
    link_flows_raw: dict    = {}
    try:
        # Only AC transmission buses — exclude virtual storage/H2/heat buses
        # (PyPSA creates named sub-buses like "ES0 1 battery" that have no real coords)
        _virtual_suffixes = (" battery", " H2", " heat", " gas", " water tanks")
        es_buses_df = n.buses[
            n.buses.index.str.startswith("ES") &
            ~n.buses.index.str.endswith(tuple(_virtual_suffixes))
        ]
        buses_dict: dict = {
            bid: {"lon": float(row.x), "lat": float(row.y)}
            for bid, row in es_buses_df.iterrows()
            if pd.notna(row.x) and pd.notna(row.y)
        }
        # Full bus coordinate lookup (needed for FR/PT link endpoints)
        all_coords: dict = {
            bid: {"lon": float(n.buses.at[bid, "x"]), "lat": float(n.buses.at[bid, "y"])}
            for bid in n.buses.index
            if pd.notna(n.buses.at[bid, "x"]) and pd.notna(n.buses.at[bid, "y"])
        }
        # ES–ES transmission lines
        es_lines_df = n.lines[
            n.lines.bus0.str.startswith("ES") & n.lines.bus1.str.startswith("ES")
        ]
        lines_dict: dict = {}
        for lid, row in es_lines_df.iterrows():
            b0, b1 = row.bus0, row.bus1
            if b0 in buses_dict and b1 in buses_dict:
                lines_dict[str(lid)] = {
                    "bus0": b0, "bus1": b1,
                    "s_nom": float(row.s_nom),
                    "x0": buses_dict[b0]["lon"], "y0": buses_dict[b0]["lat"],
                    "x1": buses_dict[b1]["lon"], "y1": buses_dict[b1]["lat"],
                }
        # Border interconnector links — DC Links (INELFE, Balearic, legacy DC_ic)
        links_dict: dict = {}
        if hasattr(n, "links") and not n.links.empty:
            border_mask = (
                (n.links.bus0.str.startswith("ES") &
                 (n.links.bus1.str.startswith("FR") | n.links.bus1.str.startswith("PT"))) |
                ((n.links.bus0.str.startswith("FR") | n.links.bus0.str.startswith("PT")) &
                 n.links.bus1.str.startswith("ES"))
            )
            for lid, row in n.links[border_mask].iterrows():
                c0, c1 = all_coords.get(row.bus0), all_coords.get(row.bus1)
                if c0 and c1:
                    links_dict[str(lid)] = {
                        "bus0": row.bus0, "bus1": row.bus1,
                        "p_nom": float(getattr(row, "p_nom", 0)),
                        "tech": "DC",
                        "x0": c0["lon"], "y0": c0["lat"],
                        "x1": c1["lon"], "y1": c1["lat"],
                    }

        # Border interconnector AC Lines (ES↔FR/PT after topology change to AC Lines)
        _ac_border = (
            (n.lines.bus0.str.startswith("ES") &
             (n.lines.bus1.str.startswith("FR") | n.lines.bus1.str.startswith("PT"))) |
            ((n.lines.bus0.str.startswith("FR") | n.lines.bus0.str.startswith("PT")) &
             n.lines.bus1.str.startswith("ES"))
        )
        for lid, row in n.lines[_ac_border].iterrows():
            c0, c1 = all_coords.get(row.bus0), all_coords.get(row.bus1)
            if c0 and c1:
                links_dict[str(lid)] = {
                    "bus0": row.bus0, "bus1": row.bus1,
                    "p_nom": float(row.s_nom),
                    "tech": "AC",
                    "x0": c0["lon"], "y0": c0["lat"],
                    "x1": c1["lon"], "y1": c1["lat"],
                }

        map_meta = {"buses": buses_dict, "lines": lines_dict, "links": links_dict}

        # Per-bus, per-carrier generation dispatch (for hover tooltips on the map)
        # Only buses that have at least one generator; only carriers with dispatch > 0
        bus_gen_raw: dict = {}
        if not n.generators_t.p.empty:
            gp = n.generators_t.p
            es_gens = n.generators[
                n.generators.bus.isin(buses_dict) &
                (n.generators.carrier != "load_shedding")
            ]
            for bus_id, grp in es_gens.groupby("bus"):
                bus_gen_raw[bus_id] = {}
                for carrier, cgrp in grp.groupby("carrier"):
                    cols = [g for g in cgrp.index if g in gp.columns]
                    if cols:
                        dispatch = (
                            gp[cols].sum(axis=1).clip(lower=0)
                            .reindex(n.snapshots, fill_value=0.0)
                        )
                        if dispatch.max() > 1.0:  # skip if never dispatches
                            bus_gen_raw[bus_id][carrier] = dispatch.tolist()
                if not bus_gen_raw[bus_id]:
                    del bus_gen_raw[bus_id]

        # Per-bus installed capacity (static) — used in hover panel
        bus_cap_raw: dict = {}
        _cap_gens = n.generators[
            n.generators.bus.isin(buses_dict) &
            (n.generators.carrier != "load_shedding")
        ]
        for bus_id, grp in _cap_gens.groupby("bus"):
            caps = {}
            for carrier, cgrp in grp.groupby("carrier"):
                total = float(cgrp["p_nom"].sum())
                if total > 0.1:
                    caps[carrier] = total
            if caps:
                bus_cap_raw[bus_id] = caps

        # Per-bus electricity demand per snapshot — used in hover panel
        bus_load_raw: dict = {}
        if not n.loads_t.p.empty:
            _lp = n.loads_t.p
            for bus_id in buses_dict:
                _lids = n.loads.index[n.loads.bus == bus_id]
                _cols = [l for l in _lids if l in _lp.columns]
                if _cols:
                    _demand = (
                        _lp[_cols].sum(axis=1)
                        .reindex(n.snapshots, fill_value=0.0)
                    )
                    if _demand.max() > 0.1:
                        bus_load_raw[bus_id] = _demand.tolist()

        # Per-snapshot line loading fractions (|p0| / s_nom, clipped 0–1)
        lines_p0 = getattr(n.lines_t, "p0", pd.DataFrame())
        if not lines_p0.empty:
            for lid, ld in lines_dict.items():
                if lid in lines_p0.columns and ld["s_nom"] > 0:
                    line_loadings_raw[lid] = (
                        (lines_p0[lid].abs() / ld["s_nom"]).clip(0, 1)
                        .reindex(n.snapshots, fill_value=0.0).tolist()
                    )
        # Per-snapshot border link flows (MW) — DC Links from links_t.p0
        links_p0 = getattr(n.links_t, "p0", pd.DataFrame())
        if not links_p0.empty:
            for lid in links_dict:
                if lid in links_p0.columns:
                    link_flows_raw[lid] = (
                        links_p0[lid].reindex(n.snapshots, fill_value=0.0).tolist()
                    )
        # AC border Lines from lines_t.p0 (new topology: PT/FR corridors as Lines)
        if not lines_p0.empty:
            for lid in links_dict:
                if lid not in link_flows_raw and lid in lines_p0.columns:
                    link_flows_raw[lid] = (
                        lines_p0[lid].reindex(n.snapshots, fill_value=0.0).tolist()
                    )
    except Exception:
        log.debug("Map data extraction failed", exc_info=True)

    # Serialise as flat lists keyed by ISO timestamp strings — avoids pandas
    # JSON index confusion and keeps the store compact.
    ts_strs = [str(s) for s in n.snapshots]

    dispatch_fr = _dispatch_by_carrier(n, "FR")
    dispatch_pt = _dispatch_by_carrier(n, "PT")

    # REE actual dispatch — align to model snapshots (2024 only; silently skip otherwise)
    ree_actual: dict = {}
    try:
        snaps_utc = (n.snapshots.tz_localize("UTC")
                     if n.snapshots.tz is None else n.snapshots.tz_convert("UTC"))
        ree = pd.read_csv(REE_CSV, index_col=0, parse_dates=True)
        if ree.index.tz is None:
            ree.index = ree.index.tz_localize("UTC")
        ree = ree.rename(columns=_REE_TO_CARRIER)
        ree = ree.reindex(snaps_utc, method="nearest",
                          tolerance=pd.Timedelta("35min")).fillna(0.0)
        ree_actual = {c: ree[c].tolist() for c in ree.columns}
    except Exception:
        pass

    # ── Extended diagnostics ──────────────────────────────────────────────────
    _gp     = n.generators_t.p if not n.generators_t.p.empty else pd.DataFrame(index=n.snapshots)
    _pmax_t = getattr(n.generators_t, "p_max_pu", pd.DataFrame())
    _tv_mc  = n.generators_t.marginal_cost

    # VRE potential vs actual (ES wind + solar)
    # Include all VRE carriers for per-technology curtailment breakdown
    _vre_carriers_all = {"solar", "onwind", "offwind-ac", "offwind-dc", "offwind-float"}
    _es_vre_all = n.generators.index[
        n.generators.bus.str.startswith("ES") &
        n.generators["carrier"].isin(_vre_carriers_all)
    ]
    # Per-technology potential and actual
    _vre_tech_pot: dict[str, pd.Series] = {}
    _vre_tech_act: dict[str, pd.Series] = {}
    for _carrier in sorted(_vre_carriers_all):
        _tech_gens = [g for g in _es_vre_all if n.generators.at[g, "carrier"] == _carrier]
        if not _tech_gens:
            continue
        _pot_parts = []
        for _g in _tech_gens:
            _pn = float(n.generators.at[_g, "p_nom"])
            if not _pmax_t.empty and _g in _pmax_t.columns:
                _pot_parts.append(_pmax_t[_g] * _pn)
            else:
                _pot_parts.append(pd.Series(_pn * float(n.generators.at[_g, "p_max_pu"]),
                                            index=n.snapshots))
        _vre_tech_pot[_carrier] = (pd.concat(_pot_parts, axis=1).sum(axis=1)
                                    if _pot_parts else pd.Series(0.0, index=n.snapshots))
        _act_cols = [_g for _g in _tech_gens if _g in _gp.columns]
        _vre_tech_act[_carrier] = (_gp[_act_cols].clip(lower=0).sum(axis=1)
                                    if _act_cols else pd.Series(0.0, index=n.snapshots))
    # Aggregate total VRE potential/actual (backward-compatible)
    _vre_potential = (pd.concat(list(_vre_tech_pot.values()), axis=1).sum(axis=1)
                      if _vre_tech_pot else pd.Series(0.0, index=n.snapshots))
    _vre_actual = (pd.concat(list(_vre_tech_act.values()), axis=1).sum(axis=1)
                   if _vre_tech_act else pd.Series(0.0, index=n.snapshots))
    # Per-technology curtailment (MWh → GWh for monthly aggregation)
    _vre_tech_curtail_gwh: dict[str, list] = {}
    for _carrier in _vre_tech_pot:
        _curt_mwh = (_vre_tech_pot[_carrier] - _vre_tech_act[_carrier]).clip(lower=0)
        _vre_tech_curtail_gwh[_carrier] = _curt_mwh.tolist()

    # Must-run stack (nuclear + biomass + CCGT_must_run at ES)
    _must_carriers = {"nuclear", "biomass", "CCGT_must_run"}
    _es_must = n.generators.index[
        n.generators.bus.str.startswith("ES") &
        n.generators["carrier"].isin(_must_carriers)
    ]
    _must_cols = [_g for _g in _es_must if _g in _gp.columns]
    _must_run  = (_gp[_must_cols].clip(lower=0).sum(axis=1)
                  if _must_cols else pd.Series(0.0, index=n.snapshots))

    # Time-varying mean CCGT MC (ES)
    _es_ccgt_all = n.generators.index[
        n.generators.bus.str.startswith("ES") & (n.generators["carrier"] == "CCGT")
    ]
    _tv_ccgt_cols = [_g for _g in _es_ccgt_all if _g in _tv_mc.columns]
    _ccgt_mc_t    = (_tv_mc[_tv_ccgt_cols].mean(axis=1).reindex(n.snapshots)
                     if _tv_ccgt_cols else pd.Series(float("nan"), index=n.snapshots))

    # Per-hour MC of the price-setting carrier (vectorised by carrier)
    _es_non_voll = n.generators[
        n.generators.bus.str.startswith("ES") &
        (n.generators["carrier"] != "load_shedding")
    ]
    _carrier_mc_t: dict = {}
    for _c, _grp in _es_non_voll.groupby("carrier"):
        _tc = [_g for _g in _grp.index if _g in _tv_mc.columns]
        if _tc:
            _carrier_mc_t[_c] = _tv_mc[_tc].mean(axis=1).reindex(n.snapshots)
        else:
            _mc_val = float(n.generators.loc[_grp.index, "marginal_cost"].mean())
            _carrier_mc_t[_c] = pd.Series(_mc_val, index=n.snapshots)
    _setter_mc_t = pd.Series(float("nan"), index=n.snapshots)
    for _c, _mc_s in _carrier_mc_t.items():
        _mask = setter_es == _c
        _setter_mc_t[_mask] = _mc_s[_mask].values

    # FR nuclear / hydro and PT hydro dispatch
    _fr_nuc_cols = [_g for _g in n.generators.index
                    if n.generators.at[_g, "bus"].startswith("FR")
                    and n.generators.at[_g, "carrier"] == "nuclear"
                    and _g in _gp.columns]
    _fr_hyd_cols = [_g for _g in n.generators.index
                    if n.generators.at[_g, "bus"].startswith("FR")
                    and n.generators.at[_g, "carrier"] in ("hydro", "ror")
                    and _g in _gp.columns]
    _pt_hyd_cols = [_g for _g in n.generators.index
                    if n.generators.at[_g, "bus"].startswith("PT")
                    and n.generators.at[_g, "carrier"] in ("hydro", "ror")
                    and _g in _gp.columns]
    _fr_nuclear_t = (_gp[_fr_nuc_cols].clip(lower=0).sum(axis=1)
                     if _fr_nuc_cols else pd.Series(0.0, index=n.snapshots))
    _fr_hydro_t   = (_gp[_fr_hyd_cols].clip(lower=0).sum(axis=1)
                     if _fr_hyd_cols else pd.Series(0.0, index=n.snapshots))
    _pt_hydro_t   = (_gp[_pt_hyd_cols].clip(lower=0).sum(axis=1)
                     if _pt_hyd_cols else pd.Series(0.0, index=n.snapshots))

    # FR / PT shadow prices (mean across all buses of each country)
    _mp = n.buses_t.marginal_price
    _fr_bus_cols = [b for b in _mp.columns if b.startswith("FR")]
    _pt_bus_cols = [b for b in _mp.columns if b.startswith("PT")]
    _fr_price_t  = (_mp[_fr_bus_cols].mean(axis=1).reindex(n.snapshots)
                    if _fr_bus_cols else pd.Series(float("nan"), index=n.snapshots))
    _pt_price_t  = (_mp[_pt_bus_cols].mean(axis=1).reindex(n.snapshots)
                    if _pt_bus_cols else pd.Series(float("nan"), index=n.snapshots))

    # FR / PT total load per snapshot
    _fr_load_idx = n.loads.index[n.loads.bus.str.startswith("FR")]
    _pt_load_idx = n.loads.index[n.loads.bus.str.startswith("PT")]
    _lp_set = getattr(n.loads_t, "p_set", pd.DataFrame())
    _fr_load_t = (_lp_set[[l for l in _fr_load_idx if l in _lp_set.columns]].sum(axis=1)
                  .reindex(n.snapshots, fill_value=0.0)
                  if not _lp_set.empty and len(_fr_load_idx)
                  else pd.Series(0.0, index=n.snapshots))
    _pt_load_t = (_lp_set[[l for l in _pt_load_idx if l in _lp_set.columns]].sum(axis=1)
                  .reindex(n.snapshots, fill_value=0.0)
                  if not _lp_set.empty and len(_pt_load_idx)
                  else pd.Series(0.0, index=n.snapshots))

    # FR wind + solar dispatch (used to compute FR surplus)
    _fr_wind_cols  = [g for g in n.generators.index
                      if n.generators.at[g, "bus"].startswith("FR")
                      and n.generators.at[g, "carrier"] in ("onwind", "offwind-ac", "offwind-dc")
                      and g in _gp.columns]
    _fr_solar_cols = [g for g in n.generators.index
                      if n.generators.at[g, "bus"].startswith("FR")
                      and n.generators.at[g, "carrier"] == "solar"
                      and g in _gp.columns]
    _fr_wind_t  = (_gp[_fr_wind_cols].clip(lower=0).sum(axis=1)
                   if _fr_wind_cols else pd.Series(0.0, index=n.snapshots))
    _fr_solar_t = (_gp[_fr_solar_cols].clip(lower=0).sum(axis=1)
                   if _fr_solar_cols else pd.Series(0.0, index=n.snapshots))

    # FR generation surplus vs load (positive = France has excess → exports)
    _fr_total_gen_t = _fr_nuclear_t + _fr_hydro_t + _fr_wind_t + _fr_solar_t
    _fr_surplus_t   = _fr_total_gen_t - _fr_load_t

    # ── ES wind and solar dispatch (Step 4 curtailment cross-analysis) ──────
    _es_wind_carriers = {"onwind", "offwind-ac", "offwind-dc"}
    _es_wind_cols  = [g for g in n.generators.index
                      if n.generators.at[g, "bus"].startswith("ES")
                      and n.generators.at[g, "carrier"] in _es_wind_carriers
                      and g in _gp.columns]
    _es_solar_cols = [g for g in n.generators.index
                      if n.generators.at[g, "bus"].startswith("ES")
                      and n.generators.at[g, "carrier"] == "solar"
                      and g in _gp.columns]
    _es_wind_t  = (_gp[_es_wind_cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0)
                   if _es_wind_cols else pd.Series(0.0, index=n.snapshots))
    _es_solar_t = (_gp[_es_solar_cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0)
                   if _es_solar_cols else pd.Series(0.0, index=n.snapshots))

    # ── CCGT MC by tier (Step 3: sort ES CCGTs by mean MC → natural tier groups) ─
    _ccgt_tier_mc: dict[str, list] = {}
    if _tv_ccgt_cols:
        _mc_means_sorted = _tv_mc[_tv_ccgt_cols].mean().sort_values()
        _n_tiers_cfg = 6  # must match ccgt_efficiency_tiers ES tier count in config
        _chunk = max(1, len(_mc_means_sorted) // _n_tiers_cfg)
        for _ti in range(_n_tiers_cfg):
            _tc = _mc_means_sorted.index[_ti * _chunk: (_ti + 1) * _chunk].tolist()
            if _tc:
                _ccgt_tier_mc[f"T{_ti + 1}"] = _tv_mc[_tc].mean(axis=1).reindex(n.snapshots).tolist()

    # Implied MIBGAS from cheapest CCGT tier MC (back-calculation from T1 MC)
    # MIBGAS(t) ≈ (T1_MC(t) − VOM) × η_T1_mid − CO₂_intensity × CO₂_price
    _mibgas_t = pd.Series(float("nan"), index=n.snapshots)
    if "T1" in _ccgt_tier_mc:
        _t1_mc_s   = pd.Series(_ccgt_tier_mc["T1"], index=n.snapshots)
        _eta_t1    = 0.585  # midpoint of (0.57, 0.60) T1 range
        _vom_ccgt  = float(MODEL_CONFIG.get("gas_vom", {}).get("CCGT", 3.0))
        _co2_price = float(MODEL_CONFIG.get("co2_price", 60.0))
        _co2_th    = float(MODEL_CONFIG.get("gas_co2_intensity_th", 0.202))
        _mibgas_t  = (_t1_mc_s - _vom_ccgt) * _eta_t1 - _co2_th * _co2_price

    # ── Border IC congestion & congestion rent ────────────────────────────────
    _CONG_THR = 0.98
    _lp0_ext = getattr(n.lines_t, "p0", pd.DataFrame())
    _kp0_ext = getattr(n.links_t, "p0", pd.DataFrame())

    def _sat_count(line_names, link_names):
        count = pd.Series(0, index=n.snapshots, dtype=int)
        for _ln in line_names:
            if _ln in _lp0_ext.columns and n.lines.loc[_ln, "s_nom"] > 0:
                count += (_lp0_ext[_ln].abs() / n.lines.loc[_ln, "s_nom"] >= _CONG_THR).astype(int)
        for _lk in link_names:
            if _lk in _kp0_ext.columns and n.links.loc[_lk, "p_nom"] > 0:
                count += (_kp0_ext[_lk].abs() / n.links.loc[_lk, "p_nom"] >= _CONG_THR).astype(int)
        return count

    _fr_ac_names = [nm for nm in ("FR_WEST", "FR_EAST_AC") if nm in n.lines.index]
    _pt_ac_names = [nm for nm in ("PT_NORTH", "PT_CENTRE", "PT_SOUTH") if nm in n.lines.index]
    _fr_dc_names = [lk for lk in n.links.index if
                    (n.links.loc[lk, "bus0"].startswith("ES") and n.links.loc[lk, "bus1"].startswith("FR")) or
                    (n.links.loc[lk, "bus0"].startswith("FR") and n.links.loc[lk, "bus1"].startswith("ES"))]
    _pt_dc_names = [lk for lk in n.links.index if
                    (n.links.loc[lk, "bus0"].startswith("ES") and n.links.loc[lk, "bus1"].startswith("PT")) or
                    (n.links.loc[lk, "bus0"].startswith("PT") and n.links.loc[lk, "bus1"].startswith("ES"))]

    _fr_ic_sat_t    = _sat_count(_fr_ac_names, _fr_dc_names)
    _pt_ic_sat_t    = _sat_count(_pt_ac_names, _pt_dc_names)
    _fr_rent_t      = (_fr_price_t - price_es).abs().reindex(n.snapshots)
    _pt_rent_t      = (_pt_price_t - price_es).abs().reindex(n.snapshots)

    # Internal ES line congestion (how many ES-ES lines at ≥98%)
    _es_int_lines = n.lines[n.lines.bus0.str.startswith("ES") & n.lines.bus1.str.startswith("ES")]
    _internal_cong_t = _sat_count(list(_es_int_lines.index), [])

    # ── Supply stack headroom (CCGT + OCGT + dispatching hydro) ──────────────
    _flex_carriers = {"CCGT", "CCGT_flex", "OCGT", "hydro"}
    _es_flex_idx   = n.generators.index[
        n.generators.bus.str.startswith("ES") &
        n.generators["carrier"].isin(_flex_carriers)
    ]
    _flex_up_t  = pd.Series(0.0, index=n.snapshots)
    _flex_dn_t  = pd.Series(0.0, index=n.snapshots)
    _pmin_t_ext = getattr(n.generators_t, "p_min_pu", pd.DataFrame())
    if not _gp.empty:
        for _g in _es_flex_idx:
            if _g not in _gp.columns:
                continue
            _pn  = float(n.generators.at[_g, "p_nom"])
            _p   = _gp[_g].clip(lower=0).reindex(n.snapshots, fill_value=0.0)
            _pmx = ((_pmax_t[_g] * _pn).reindex(n.snapshots)
                    if not _pmax_t.empty and _g in _pmax_t.columns
                    else float(n.generators.at[_g, "p_max_pu"]) * _pn)
            _pmn = ((_pmin_t_ext[_g] * _pn).reindex(n.snapshots)
                    if not _pmin_t_ext.empty and _g in _pmin_t_ext.columns
                    else float(n.generators.at[_g, "p_min_pu"]) * _pn)
            _online = _p > 0.01
            _flex_up_t += (_pmx - _p).clip(lower=0) * _online.astype(float)
            _flex_dn_t += (_p - _pmn).clip(lower=0) * _online.astype(float)

    # ── ES hydro reservoir SoC + inflow ──────────────────────────────────────
    _hydro_su   = n.storage_units[n.storage_units["carrier"] == "hydro"]
    _soc_raw    = getattr(n.storage_units_t, "state_of_charge", pd.DataFrame())
    _infl_raw   = getattr(n.storage_units_t, "inflow",          pd.DataFrame())
    _es_h_cols  = [g for g in _hydro_su.index
                   if n.storage_units.at[g, "bus"].startswith("ES") and g in _soc_raw.columns]
    _hydro_soc_gwh_t = (
        _soc_raw[_es_h_cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0) / 1e3
        if _es_h_cols else pd.Series(0.0, index=n.snapshots)
    )
    _infl_cols   = [g for g in _es_h_cols if g in _infl_raw.columns]
    _hydro_infl_gwh_t = (
        _infl_raw[_infl_cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0) / 1e3
        if _infl_cols else pd.Series(0.0, index=n.snapshots)
    )

    # ── FR / PT hydro dispatch (generators, no reservoir model) ──────────────
    # FR and PT hydro are simple generators — no SoC exists. We expose their
    # hourly dispatch so the dashboard can show whether the model is dispatching
    # them with realistic seasonality (it currently doesn't for PT in summer).
    _gen_p = getattr(n.generators_t, "p", pd.DataFrame())
    _hg_mask = n.generators["carrier"] == "hydro"
    def _country_hydro_mw(prefix):
        cols = [g for g in n.generators.index[_hg_mask]
                if n.generators.at[g, "bus"].startswith(prefix) and g in _gen_p.columns]
        return (
            _gen_p[cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0)
            if cols else pd.Series(0.0, index=n.snapshots)
        )
    _fr_hydro_mw_t = _country_hydro_mw("FR")
    _pt_hydro_mw_t = _country_hydro_mw("PT")

    # FR / PT hydro storage-unit dispatch + inflow (post-refinery conversion)
    # After _hydro_convert_gens_to_storage, FR/PT reservoir hydro lives in
    # storage_units, not generators.  Extract separately so the dashboard can
    # show inflow vs dispatch seasonality.
    _su_p_raw      = getattr(n.storage_units_t, "p", pd.DataFrame())
    _hydro_su_all  = n.storage_units[n.storage_units["carrier"].isin(["hydro", "ror"])]
    _fr_su_idx     = [g for g in _hydro_su_all.index if n.storage_units.at[g, "bus"].startswith("FR")]
    _pt_su_idx     = [g for g in _hydro_su_all.index if n.storage_units.at[g, "bus"].startswith("PT")]

    def _su_series(idx_list, src):
        cols = [g for g in idx_list if g in src.columns]
        return (src[cols].sum(axis=1).reindex(n.snapshots, fill_value=0.0) / 1e3
                if cols else pd.Series(0.0, index=n.snapshots))

    _fr_su_hydro_gw_t = _su_series(_fr_su_idx, _su_p_raw)
    _pt_su_hydro_gw_t = _su_series(_pt_su_idx, _su_p_raw)
    _fr_infl_gwh_t    = _su_series(_fr_su_idx, _infl_raw)
    _pt_infl_gwh_t    = _su_series(_pt_su_idx, _infl_raw)

    # FR / PT reservoir SOC — same StorageUnits, read from state_of_charge
    _fr_su_in_soc = [g for g in _fr_su_idx if g in _soc_raw.columns]
    _pt_su_in_soc = [g for g in _pt_su_idx if g in _soc_raw.columns]
    _fr_soc_gwh_t = (
        _soc_raw[_fr_su_in_soc].sum(axis=1).reindex(n.snapshots, fill_value=0.0) / 1e3
        if _fr_su_in_soc else pd.Series(0.0, index=n.snapshots)
    )
    _pt_soc_gwh_t = (
        _soc_raw[_pt_su_in_soc].sum(axis=1).reindex(n.snapshots, fill_value=0.0) / 1e3
        if _pt_su_in_soc else pd.Series(0.0, index=n.snapshots)
    )

    # ── MIP startup events (ES CCGT + OCGT) ──────────────────────────────────
    _su_df      = getattr(n.generators_t, "start_up", pd.DataFrame())
    _su_carriers = {"CCGT", "CCGT_flex", "OCGT"}
    _su_gens    = [g for g in n.generators.index
                   if n.generators.at[g, "carrier"] in _su_carriers
                   and n.generators.at[g, "bus"].startswith("ES")
                   and not _su_df.empty and g in _su_df.columns]
    _startups_t = (_su_df[_su_gens].sum(axis=1).reindex(n.snapshots, fill_value=0.0)
                   if _su_gens else pd.Series(0.0, index=n.snapshots))
    # Startup costs: Pass-2 LP zeros start_up_cost on committable gens to recover
    # dual prices. Read from n.meta["startup_costs_eur"] if available (preserved
    # before Pass-2 overwrites it), falling back to the network column.
    _su_cost_lookup = n.meta.get("startup_costs_eur", {}) if hasattr(n, "meta") else {}
    # Fallback for networks solved before the meta fix: read costs from MODEL_CONFIG.
    # Pass-2 LP zeros start_up_cost on all committable gens, so the column is useless.
    if not _su_cost_lookup:
        _mip_cfg = MODEL_CONFIG.get("mip", {})
        _ccgt_su_cost = float(_mip_cfg.get("CCGT", {}).get("start_up_cost", 0))
        _ocgt_su_cost = float(_mip_cfg.get("OCGT", {}).get("start_up_cost", 0))
        for _g in _su_gens:
            _gc = n.generators.at[_g, "carrier"] if _g in n.generators.index else ""
            if "CCGT" in _gc:
                _su_cost_lookup[_g] = _ccgt_su_cost
            elif _gc == "OCGT":
                _su_cost_lookup[_g] = _ocgt_su_cost
    _su_cost_t  = pd.Series(0.0, index=n.snapshots)
    for _g in _su_gens:
        _c = _su_cost_lookup.get(_g, None)
        if _c is None:
            _c = (float(n.generators.at[_g, "start_up_cost"])
                  if "start_up_cost" in n.generators.columns else 0.0)
        if _c > 0:
            _su_cost_t += _su_df[_g].reindex(n.snapshots, fill_value=0.0) * _c
    _es_load_su = (n.loads_t.p_set[[l for l in n.loads.index
                                     if n.loads.loc[l, "bus"].startswith("ES")
                                     and l in n.loads_t.p_set.columns]]
                   .sum(axis=1).reindex(n.snapshots, fill_value=1.0)
                   if not n.loads_t.p_set.empty else pd.Series(1.0, index=n.snapshots))
    _startup_eur_mwh_t = _su_cost_t / _es_load_su.clip(lower=1)

    # ── Actual ENTSOE interconnector flows ────────────────────────────────────
    _actual_fr_t = pd.Series(0.0, index=n.snapshots)
    _actual_pt_t = pd.Series(0.0, index=n.snapshots)
    try:
        _snaps_utc = (n.snapshots.tz_localize("UTC")
                      if n.snapshots.tz is None else n.snapshots.tz_convert("UTC"))
        _fr_raw = _parse_entsoe_ic(_IC_DIR / "2024_FR_ES_balance_hourly.csv")
        _pt_raw = _parse_entsoe_ic(_IC_DIR / "2024_PT_ES_balance_hourly.csv")
        _fr_al  = _fr_raw.reindex(_snaps_utc, method="nearest",
                                   tolerance=pd.Timedelta("75min"))
        _pt_al  = _pt_raw.reindex(_snaps_utc, method="nearest",
                                   tolerance=pd.Timedelta("75min"))
        _actual_fr_t = pd.Series(_fr_al.values, index=n.snapshots)
        _actual_pt_t = pd.Series(_pt_al.values, index=n.snapshots)
        log.info("Actual IC flows loaded: FR %d valid, PT %d valid",
                 _fr_al.notna().sum(), _pt_al.notna().sum())
    except Exception:
        log.debug("Actual IC flow loading failed", exc_info=True)

    # ── VRE Price-Formation Bottleneck Diagnostics ────────────────────────────
    # Per-hour accounting: when should VRE clear the market but gas actually does?
    _es_load_pfm = pd.Series(es_load, index=n.snapshots, dtype=float) if es_load else pd.Series(0.0, index=n.snapshots)

    _bmr_carriers_pfm = {"biomass", "CCGT_must_run"}
    _es_bmr_pfm = n.generators.index[
        n.generators.bus.str.startswith("ES") &
        n.generators["carrier"].isin(_bmr_carriers_pfm)
    ]
    _bmr_mw_pfm = float(n.generators.loc[_es_bmr_pfm, "p_nom"].sum()) if len(_es_bmr_pfm) else 0.0

    _nuc_pmin_pfm = (MODEL_CONFIG.get("nuclear", {})
                     .get("per_country", {}).get("ES", {}).get("p_min_pu", 0.40))
    _es_nuc_pfm = n.generators.index[
        n.generators.bus.str.startswith("ES") &
        (n.generators["carrier"] == "nuclear")
    ]
    _nuc_floor_pfm = _nuc_pmin_pfm * float(n.generators.loc[_es_nuc_pfm, "p_nom"].sum()) if len(_es_nuc_pfm) else 0.0
    _inflex_floor_pfm = pd.Series(_bmr_mw_pfm + _nuc_floor_pfm, index=n.snapshots)

    _net_import_pfm   = fr_net + pt_net
    _residual_pfm     = _es_load_pfm - _vre_potential - _net_import_pfm
    _res_margin_pfm   = _residual_pfm - _inflex_floor_pfm

    _gas_setters_pfm  = {"CCGT", "CCGT_flex", "OCGT", "diesel"}
    _theory_vre_pfm   = _residual_pfm < _inflex_floor_pfm
    _trapped_vre_pfm  = _theory_vre_pfm & setter_es.isin(_gas_setters_pfm)

    # Congestion in trapped VRE hours
    _trapped_snaps_pfm = n.snapshots[_trapped_vre_pfm.reindex(n.snapshots, fill_value=False).values]
    _tf_pfm  = MODEL_CONFIG.get("transmission", {}).get("trans_factor", 1.0)
    _smx_pfm = MODEL_CONFIG.get("transmission", {}).get("s_max_pu", 1.0)
    _lp0_pfm = getattr(n.lines_t, "p0", pd.DataFrame())
    _kp0_pfm = getattr(n.links_t, "p0", pd.DataFrame())
    _cong_dict_pfm: dict = {}
    if len(_trapped_snaps_pfm) > 0:
        for _ln, _row in n.lines.iterrows():
            _b0, _b1 = str(_row.bus0), str(_row.bus1)
            if not (_b0.startswith("ES") or _b1.startswith("ES")):
                continue
            if _ln not in _lp0_pfm.columns:
                continue
            _flows_pfm = _lp0_pfm[_ln].reindex(_trapped_snaps_pfm, fill_value=0.0).abs()
            _cap_pfm = (float(_row.s_nom) * _tf_pfm * _smx_pfm
                        if _b0.startswith("ES") and _b1.startswith("ES")
                        else float(_row.s_nom) * _smx_pfm)
            _cnt_pfm = int((_flows_pfm >= 0.95 * _cap_pfm).sum())
            if _cnt_pfm > 0:
                _cong_dict_pfm[_ln] = {"count": _cnt_pfm, "bus0": _b0, "bus1": _b1}
        for _lk, _row in n.links.iterrows():
            _b0, _b1 = str(_row.bus0), str(_row.bus1)
            if not (_b0.startswith("ES") or _b1.startswith("ES")):
                continue
            if _lk not in _kp0_pfm.columns:
                continue
            _flows_pfm = _kp0_pfm[_lk].reindex(_trapped_snaps_pfm, fill_value=0.0).abs()
            _cnt_pfm   = int((_flows_pfm >= 0.95 * float(_row.p_nom)).sum())
            if _cnt_pfm > 0:
                _cong_dict_pfm[_lk] = {"count": _cnt_pfm, "bus0": _b0, "bus1": _b1}
    _cong_top_pfm = [
        {"line": k, "count": v["count"], "bus0": v["bus0"], "bus1": v["bus1"]}
        for k, v in sorted(_cong_dict_pfm.items(), key=lambda x: -x[1]["count"])[:8]
    ]

    # Time-weighted average: plain arithmetic mean of nodal shadow prices
    price_tw = (bus_prices.mean(axis=1) if not bus_prices.empty
                else pd.Series(0.0, index=n.snapshots))

    # Annual mean load per bus (for load-map bubble sizes)
    bus_load_annual = {
        bid: float(sum(vals) / max(len(vals), 1))
        for bid, vals in bus_load_raw.items()
        if vals and bid in buses_dict
    }

    return {
        "timestamps":   ts_strs,
        "dispatch_es":  {c: dispatch_es[c].tolist() for c in dispatch_es.columns},
        "dispatch_fr":  {c: dispatch_fr[c].tolist() for c in dispatch_fr.columns},
        "dispatch_pt":  {c: dispatch_pt[c].tolist() for c in dispatch_pt.columns},
        "price_es":     price_es.tolist() if not price_es.empty else [],
        "setter_es":    setter_es.astype(str).tolist() if not setter_es.empty else [],
        "fr_net":       fr_net.tolist() if not fr_net.empty else [],
        "pt_net":       pt_net.tolist() if not pt_net.empty else [],
        "bus_prices":   {b: bus_prices[b].tolist() for b in bus_prices.columns}
                        if not bus_prices.empty else {},
        "omie":         omie.tolist() if omie is not None else None,
        "ree_actual":   ree_actual,
        "es_load":      es_load,
        "ccgt_bounds":  ccgt_bounds,
        "capacity":      cap,
        "map_meta":      map_meta,
        "line_loadings": line_loadings_raw,
        "link_flows":    link_flows_raw,
        "bus_gen":       bus_gen_raw,
        "bus_cap":       bus_cap_raw,
        "bus_load":      bus_load_raw,
        # Extended diagnostics
        "vre_potential_es": _vre_potential.tolist(),
        "vre_actual_es":    _vre_actual.tolist(),
        "must_run_es":      _must_run.tolist(),
        "ccgt_mc_t":        _ccgt_mc_t.tolist(),
        "setter_mc_t":      _setter_mc_t.tolist(),
        "fr_nuclear_t":     _fr_nuclear_t.tolist(),
        "fr_hydro_t":       _fr_hydro_t.tolist(),
        "pt_hydro_t":       _pt_hydro_t.tolist(),
        "actual_fr_t":      _actual_fr_t.tolist(),
        "actual_pt_t":      _actual_pt_t.tolist(),
        "fr_price_t":       _fr_price_t.tolist(),
        "pt_price_t":       _pt_price_t.tolist(),
        "fr_load_t":        _fr_load_t.tolist(),
        "pt_load_t":        _pt_load_t.tolist(),
        "fr_wind_t":        _fr_wind_t.tolist(),
        "fr_solar_t":       _fr_solar_t.tolist(),
        "fr_surplus_t":     _fr_surplus_t.tolist(),
        # LP constraint mechanics
        "fr_ic_sat_t":       _fr_ic_sat_t.tolist(),
        "pt_ic_sat_t":       _pt_ic_sat_t.tolist(),
        "fr_rent_t":         _fr_rent_t.tolist(),
        "pt_rent_t":         _pt_rent_t.tolist(),
        "internal_cong_t":   _internal_cong_t.tolist(),
        "flex_up_t":         _flex_up_t.tolist(),
        "flex_dn_t":         _flex_dn_t.tolist(),
        "hydro_soc_gwh":     _hydro_soc_gwh_t.tolist(),
        "hydro_inflow_gwh":  _hydro_infl_gwh_t.tolist(),
        "fr_soc_gwh":        _fr_soc_gwh_t.tolist(),
        "pt_soc_gwh":        _pt_soc_gwh_t.tolist(),
        "fr_hydro_mw":       _fr_hydro_mw_t.tolist(),
        "pt_hydro_mw":       _pt_hydro_mw_t.tolist(),
        "fr_su_hydro_gw":    _fr_su_hydro_gw_t.tolist(),
        "pt_su_hydro_gw":    _pt_su_hydro_gw_t.tolist(),
        "fr_infl_gwh":       _fr_infl_gwh_t.tolist(),
        "pt_infl_gwh":       _pt_infl_gwh_t.tolist(),
        "startups_t":        _startups_t.tolist(),
        "startup_eur_mwh_t": _startup_eur_mwh_t.tolist(),
        # Actual market prices (day-ahead wholesale) for FR and PT
        "omie_fr": omie_fr.tolist() if omie_fr is not None else None,
        "omie_pt": omie_pt.tolist() if omie_pt is not None else None,
        # Step 3 — CCGT tier MC and implied MIBGAS
        "es_wind_t":    _es_wind_t.tolist(),
        "es_solar_t":   _es_solar_t.tolist(),
        "ccgt_tier_mc": _ccgt_tier_mc,          # dict {tier: [hourly values]}
        "mibgas_t":     _mibgas_t.tolist(),
        # Curtailment by technology (hourly MW curtailed per carrier)
        "vre_tech_curtail_mw": {k: v for k, v in _vre_tech_curtail_gwh.items()},
        # VRE price-formation bottleneck
        "pfm_inflex_floor":  _inflex_floor_pfm.tolist(),
        "pfm_net_import":    _net_import_pfm.tolist(),
        "pfm_residual":      _residual_pfm.tolist(),
        "pfm_res_margin":    _res_margin_pfm.tolist(),
        "pfm_theory_vre":    _theory_vre_pfm.astype(int).tolist(),
        "pfm_trapped_vre":   _trapped_vre_pfm.astype(int).tolist(),
        "pfm_cong_top":      _cong_top_pfm,
        # Price construction method variants
        "price_tw_t":        price_tw.tolist(),
        "bus_load_annual":   bus_load_annual,
    }


def deserialise(data: dict) -> dict:
    """Reconstruct pandas objects from the JSON store dict."""
    ts = pd.to_datetime(data["timestamps"])

    def _s(key, default=float):
        vals = data.get(key, [])
        return pd.Series(vals, index=ts, dtype=default) if vals else pd.Series(dtype=default)

    def _df(key):
        raw = data.get(key, {})
        return pd.DataFrame(raw, index=ts) if raw else pd.DataFrame(index=ts)

    omie_vals = data.get("omie")
    omie = pd.Series(omie_vals, index=ts) if omie_vals else None

    es_load_vals = data.get("es_load", [])
    es_load = pd.Series(es_load_vals, index=ts, dtype=float) if es_load_vals else pd.Series(dtype=float)

    return {
        "timestamps":  data["timestamps"],
        "dispatch_es": _df("dispatch_es"),
        "dispatch_fr": _df("dispatch_fr"),
        "dispatch_pt": _df("dispatch_pt"),
        "ree_actual":  _df("ree_actual"),
        "price_es":    _s("price_es"),
        "setter_es":   _s("setter_es", str),
        "fr_net":      _s("fr_net"),
        "pt_net":      _s("pt_net"),
        "bus_prices":  _df("bus_prices"),
        "omie":        omie,
        "es_load":     es_load,
        "ccgt_bounds":   data.get("ccgt_bounds", {}),
        "capacity":      data.get("capacity", {}),
        "map_meta":      data.get("map_meta", {}),
        "line_loadings": data.get("line_loadings", {}),
        "link_flows":    data.get("link_flows", {}),
        "bus_gen":       data.get("bus_gen", {}),
        "bus_cap":       data.get("bus_cap", {}),
        "bus_load":      data.get("bus_load", {}),
        # Extended diagnostics
        "vre_potential_es": _s("vre_potential_es"),
        "vre_actual_es":    _s("vre_actual_es"),
        "must_run_es":      _s("must_run_es"),
        "ccgt_mc_t":        _s("ccgt_mc_t"),
        "setter_mc_t":      _s("setter_mc_t"),
        "fr_nuclear_t":     _s("fr_nuclear_t"),
        "fr_hydro_t":       _s("fr_hydro_t"),
        "pt_hydro_t":       _s("pt_hydro_t"),
        "actual_fr_t":      _s("actual_fr_t"),
        "actual_pt_t":      _s("actual_pt_t"),
        "fr_price_t":       _s("fr_price_t"),
        "pt_price_t":       _s("pt_price_t"),
        "fr_load_t":        _s("fr_load_t"),
        "pt_load_t":        _s("pt_load_t"),
        "fr_wind_t":        _s("fr_wind_t"),
        "fr_solar_t":       _s("fr_solar_t"),
        "fr_surplus_t":     _s("fr_surplus_t"),
        # LP constraint mechanics
        "fr_ic_sat_t":       _s("fr_ic_sat_t", int),
        "pt_ic_sat_t":       _s("pt_ic_sat_t", int),
        "fr_rent_t":         _s("fr_rent_t"),
        "pt_rent_t":         _s("pt_rent_t"),
        "internal_cong_t":   _s("internal_cong_t", int),
        "flex_up_t":         _s("flex_up_t"),
        "flex_dn_t":         _s("flex_dn_t"),
        "hydro_soc_gwh":     _s("hydro_soc_gwh"),
        "hydro_inflow_gwh":  _s("hydro_inflow_gwh"),
        "fr_soc_gwh":        _s("fr_soc_gwh"),
        "pt_soc_gwh":        _s("pt_soc_gwh"),
        "fr_hydro_mw":       _s("fr_hydro_mw"),
        "pt_hydro_mw":       _s("pt_hydro_mw"),
        "fr_su_hydro_gw":    _s("fr_su_hydro_gw"),
        "pt_su_hydro_gw":    _s("pt_su_hydro_gw"),
        "fr_infl_gwh":       _s("fr_infl_gwh"),
        "pt_infl_gwh":       _s("pt_infl_gwh"),
        "startups_t":        _s("startups_t"),
        "startup_eur_mwh_t": _s("startup_eur_mwh_t"),
        # Actual market prices
        "omie_fr": pd.Series(data["omie_fr"], index=ts) if data.get("omie_fr") else None,
        "omie_pt": pd.Series(data["omie_pt"], index=ts) if data.get("omie_pt") else None,
        # VRE price-formation bottleneck
        "pfm_inflex_floor": _s("pfm_inflex_floor"),
        "pfm_net_import":   _s("pfm_net_import"),
        "pfm_residual":     _s("pfm_residual"),
        "pfm_res_margin":   _s("pfm_res_margin"),
        "pfm_theory_vre":   _s("pfm_theory_vre", int),
        "pfm_trapped_vre":  _s("pfm_trapped_vre", int),
        "pfm_cong_top":     data.get("pfm_cong_top", []),
        # Step 3/4 additions
        "es_wind_t":    _s("es_wind_t"),
        "es_solar_t":   _s("es_solar_t"),
        "mibgas_t":     _s("mibgas_t"),
        "ccgt_tier_mc": {k: pd.Series(v, index=ts)
                         for k, v in data.get("ccgt_tier_mc", {}).items()},
        # Curtailment by technology
        "vre_tech_curtail_mw": {k: pd.Series(v, index=ts, dtype=float)
                                for k, v in data.get("vre_tech_curtail_mw", {}).items()},
        # Price construction method variants
        "price_tw_t":        _s("price_tw_t"),
        "bus_load_annual":   data.get("bus_load_annual", {}),
    }


# ── Solve worker ──────────────────────────────────────────────────────────────

def _solve_worker(overrides: dict) -> None:
    """Background thread: build fresh network with overrides → Gurobi → extract."""
    global _solve_state
    try:
        cfg = copy.deepcopy(MODEL_CONFIG)
        val = cfg["validation"]

        # Apply slider overrides to the config copy
        if "co2_price"        in overrides: cfg["co2_price"]                                 = float(overrides["co2_price"])
        if "ic_factor"        in overrides: cfg["borders"]["ic_factor"]                      = float(overrides["ic_factor"])
        if "fr_nuclear_pmin"  in overrides: cfg["nuclear"]["per_country"]["FR"]["p_min_pu"]  = float(overrides["fr_nuclear_pmin"])
        if "fr_hydro_pmax"    in overrides: cfg["hydro"]["per_country"]["FR"]["p_max_pu"]    = float(overrides["fr_hydro_pmax"])
        if "phs_pmax"         in overrides: cfg["phs"]["p_max_pu"]                           = float(overrides["phs_pmax"])
        if "trans_factor"     in overrides: cfg["transmission"]["trans_factor"]              = float(overrides["trans_factor"])
        if "mip_enabled"      in overrides: cfg["mip"]["enabled"]                            = bool(overrides["mip_enabled"])
        if "ccgt_must_run_mw" in overrides:
            mw = float(overrides["ccgt_must_run_mw"])
            cfg["ccgt_must_run"]["target_mw"] = mw
            cfg["ccgt_must_run"]["enabled"]   = mw > 0

        n_days = int(overrides.get("n_days", val["n_days"]))
        start  = pd.Timestamp(val["start_date"])
        end    = start + pd.Timedelta(hours=n_days * 24 - 1)

        log.info("Solve thread: loading base network…")
        n = pypsa.Network(str(BASE_NET))
        n = _add_fr_missing_demand(n, cfg)
        n = apply_non_linear_refinements(n, cfg)

        snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
        n.set_snapshots(snap)
        log.info("Solve thread: %d snapshots, optimising…", len(snap))

        solver_opts = dict(val.get("solver_options", {}))

        def extra_fn(nn, snapshots):
            _add_su_ramp_constraints(nn, snapshots, cfg)
            _add_hydro_min_dispatch(nn, snapshots, cfg)

        n.optimize(solver_name="gurobi", solver_options=solver_opts,
                   extra_functionality=extra_fn)

        # MIP two-pass: fix commitment, re-solve LP for dual prices
        if cfg.get("mip", {}).get("enabled", False) and "status" in n.generators_t:
            status = n.generators_t["status"]
            committable_idx = n.generators.index[
                n.generators.get("committable", pd.Series(dtype=bool)).astype(bool)
            ]
            committable_idx = [g for g in committable_idx if g in status.columns]
            if committable_idx:
                if "p_max_pu" not in n.generators_t or n.generators_t["p_max_pu"].empty:
                    n.generators_t["p_max_pu"] = pd.DataFrame(index=n.snapshots)
                for g in committable_idx:
                    n.generators_t["p_max_pu"][g] = status[g].astype(float)
                if "p_min_pu" in n.generators_t:
                    drop_cols = [g for g in committable_idx
                                 if g in n.generators_t["p_min_pu"].columns]
                    if drop_cols:
                        n.generators_t["p_min_pu"].drop(columns=drop_cols, inplace=True)
                # Preserve startup costs before zeroing (Pass-2 LP needs them at 0)
                n.meta["startup_costs_eur"] = {
                    g: float(n.generators.at[g, "start_up_cost"])
                    for g in committable_idx
                    if "start_up_cost" in n.generators.columns
                }
                n.generators.loc[committable_idx, "committable"]   = False
                n.generators.loc[committable_idx, "start_up_cost"] = 0.0
                n.generators.loc[committable_idx, "p_min_pu"]      = 0.0
                lp_opts = {k: v for k, v in solver_opts.items() if k != "MIPGap"}
                lp_opts["DualReductions"] = 0
                n.optimize(solver_name="gurobi", solver_options=lp_opts,
                           extra_functionality=extra_fn)

        log.info("Solve thread: extracting results…")
        result = _extract(n)
        with _solve_lock:
            _solve_state.update({"status": "done", "data": result, "error": None})
        log.info("Solve thread: done.")

    except Exception as exc:
        log.exception("Solve thread failed")
        with _solve_lock:
            _solve_state.update({"status": "error", "data": None, "error": str(exc)})


def start_solve(overrides: dict) -> None:
    """Launch a background solve thread (non-blocking)."""
    with _solve_lock:
        _solve_state.update({"status": "running", "data": None, "error": None})
    t = threading.Thread(target=_solve_worker, args=(overrides,), daemon=True)
    t.start()


def poll_solve_result() -> tuple[str, dict | None, str | None]:
    """Return (status, data, error) without side effects."""
    with _solve_lock:
        return _solve_state["status"], _solve_state["data"], _solve_state["error"]


def clear_solve_state() -> None:
    with _solve_lock:
        _solve_state["status"] = "idle"
