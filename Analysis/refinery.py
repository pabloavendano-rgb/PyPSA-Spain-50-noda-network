"""
apply_non_linear_refinements(n, config) — idempotent in-memory refinements.

All operations check whether they have already been applied before mutating.
The source .nc file is NEVER written — all changes live in the network object.

Execution order matters for CO₂ logic:
  0. Merit-order splits (FR/PT aggregated fleets → tiered sub-units)
     Must run first so CCGT tiering sees the split units and FR skips on threshold.
  1. CCGT tiering + CO₂ adder (must precede flex split so flex sees updated parent)
  2. CCGT_Flex split
  3. CCGT must-run carve (industrial CHP proxy, MC=0)
  4. Peaker fleet
  5. Nuclear constraints
  6. Hydro parameters (incl. capacity scaler)
  7. PHS operational friction
  8. Border restoration
  9. Transmission limits
 10. VOLL load-shedding generators
 11. Lock capacities → dispatch-only solve
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def _load_mibgas_ts(config, snapshots):
    """Broadcast MIBGAS daily gas prices to network hourly snapshots.

    Applies mibgas_multiplier (default 1.0) to scale the fuel price before
    MC computation — enables gas shock simulation via Snakemake wildcard.

    Returns a Series indexed by snapshots (same TZ as snapshots).
    Each day's gas price is forward-filled to cover all hours within that day.
    """
    raw_path = config["gas_prices_csv"]
    csv_path = Path(__file__).parent.parent / raw_path
    df = pd.read_csv(csv_path)
    date_col  = df.columns[0]   # "Delivery day"
    price_col = df.columns[1]   # "MIBGAS PVB Last Price Index Day-Ahead [EUR/MWh]"
    df["_date"] = pd.to_datetime(df[date_col], dayfirst=True)
    daily = df.set_index("_date")[price_col].astype(float)

    # Strip TZ from snapshots for date matching (MIBGAS index is tz-naive)
    snap_naive = snapshots.tz_localize(None) if snapshots.tz is not None else snapshots

    # ffill broadcasts each day's price to all intra-day hours
    mibgas = daily.reindex(snap_naive, method="ffill")

    if mibgas.isna().any():
        n_na = int(mibgas.isna().sum())
        log.warning("MIBGAS: %d snapshots have no price data — backfilling", n_na)
        mibgas = mibgas.bfill().ffill()

    # ── Fuel price multiplier ──────────────────────────────────────────────
    # Scales MIBGAS before MC computation so ALL gas generators (CCGT, CCGT_flex,
    # OCGT) inherit the multiplier through their η-based MC formulas.
    # MC(t) = (mibgas_mult × MIBGAS(t) + 0.202 × co2_price) / η + VOM
    # This is physically correct: fuel price scales before η conversion, so
    # less-efficient units get proportionally larger absolute €/MWh increases.
    mibgas_mult = config.get("mibgas_multiplier", 1.0)
    if mibgas_mult != 1.0:
        mibgas = mibgas * mibgas_mult
        log.info("MIBGAS multiplier: ×%.2f applied (mean fuel price now €%.0f/MWh_th)",
                 mibgas_mult, float(mibgas.mean()))

    return pd.Series(mibgas.values, index=snapshots, dtype=float)


def _remove_es_offwind(n):
    """Remove all Spanish offshore wind generators (onshore-only Spanish grid 2024)."""
    offwind_carriers = {"offwind", "offwind-float"}
    countries = _gen_countries(n)
    mask = n.generators["carrier"].isin(offwind_carriers) & (countries == "ES")
    if not mask.any():
        return
    to_remove = n.generators.index[mask].tolist()
    for gen in to_remove:
        n.remove("Generator", gen)
    log.info("Removed %d ES offshore wind generators (carriers: %s)", len(to_remove), offwind_carriers)


def _apply_coal_override(n, config):
    """Zero out ES coal capacity for 2024 — all Spanish coal plants closed by Jun 2023.

    As Pontes (1,400 MW, A Coruña) closed June 2023; Compostilla II and others
    were retired earlier. No coal generation occurred in Spain during 2024.
    Setting p_nom=0 removes coal from dispatch without deleting the carrier.
    """
    cfg = config.get("coal", {})
    if not cfg.get("disable_es", False):
        return
    countries = _gen_countries(n)
    coal_mask = (n.generators["carrier"] == "coal") & (countries == "ES")
    if not coal_mask.any():
        log.info("Coal override: no ES coal generators found")
        return
    total_mw = n.generators.loc[coal_mask, "p_nom"].sum()
    n.generators.loc[coal_mask, "p_nom"]     = 0.0
    n.generators.loc[coal_mask, "p_nom_min"] = 0.0
    log.info(
        "Coal override: zeroed %d ES coal generators (%.0f MW) — all plants closed by Jun 2023",
        coal_mask.sum(), total_mw,
    )


# ─── Public entry point ───────────────────────────────────────────────────────

def apply_non_linear_refinements(n, config):
    rng = np.random.default_rng(config["nuclear"]["random_seed"])
    _remove_es_offwind(n)                       # step 0a — Spain has no offshore wind in 2024
    _apply_coal_override(n, config)             # step 0b — zero 2024 coal (all plants closed Jun 2023)
    _apply_generator_splits(n, config, rng)   # step 0 — must precede CCGT tiering

    # ── Flex carving BEFORE tranching ────────────────────────────────────────
    # CRITICAL ORDERING: flex must be carved from the original CCGT p_nom BEFORE
    # the efficiency tranche split. If we tranche first, each _base/_mid/_peak
    # slice would get its own flex generator (93 flex units instead of 31).
    # Flex carving first keeps the flex count at 31 (one per ES CCGT).
    _apply_ccgt_flex(n, config, rng)

    # ── Efficiency tranches + MIBGAS MCs ─────────────────────────────────────
    # After flex carving, the remaining CCGT capacity is split into efficiency
    # tranches (_base/_mid/_peak) with decreasing η (increasing MC). This creates
    # a piecewise-linear cost ramp that mimics quadratic behavior without QP.
    # Combined with MIP commitment, this prevents trickle-dispatch.
    _apply_ccgt_tiers_and_co2(n, config, rng)
    _apply_ccgt_must_run(n, config)
    _apply_biomass_correction(n, config)
    _apply_peakers(n, config)
    _apply_nuclear(n, config, rng)
    _apply_vre_mc(n, config)               # step 5a — VRE price-setter (after nuclear, before hydro)
    _apply_solar_capacity_scaler(n, config) # step 5b — scale ES solar to match REE end-2024 (39.3 GW)
    _apply_solar_thermal(n, config)         # step 5c — add CSP plants from ERA5 profiles
    _apply_hydro(n, config)
    _apply_fr_pt_ror(n, config)     # add FR/PT RoR missing from base PyPSA-Eur 50-node network
    _apply_phs(n, config)
    _restore_borders(n, config)
    _apply_border_ac_dc(n, config)
    _apply_transmission(n, config)
    _apply_wind_availability(n, config)  # after transmission, before VOLL
    _apply_voll(n, config)
    _lock_capacities(n)                 # must be last — prevents investment solve
    return n


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _country(bus_name):
    for prefix in ("ES", "FR", "PT"):
        if bus_name.startswith(prefix):
            return prefix
    return None


def _gen_countries(n):
    return n.generators["bus"].map(_country)


# ─── 0. Merit-order splits — FR/PT aggregated fleets ─────────────────────────

def _apply_generator_splits(n, config, rng):
    """Split highly aggregated FR/PT generators into capacity-weighted tiers.

    Each tier gets a random MC drawn from its configured range, giving the
    merit-order curve realistic staircase resolution instead of flat blocks.

    Works on static generators only (no time-varying p_max_pu columns needed
    for nuclear/hydro/CCGT — only VRE has those, and they are not split here).

    Idempotent: skips carriers/countries where split sub-units already exist
    (detected by the '_t1' suffix on any matching generator name).

    FR CCGTs are no longer split here — they now use MIBGAS-based time-varying MCs
    via ccgt_efficiency_tiers.FR in _apply_ccgt_tiers_and_co2. The old merit_splits.CCGT.FR
    entry has been removed from config. Only FR nuclear and hydro remain in merit_splits.
    """
    splits_cfg = config.get("merit_splits", {})
    if not splits_cfg:
        return

    countries_series = _gen_countries(n)

    for carrier, country_map in splits_cfg.items():
        for country, tiers in country_map.items():

            # Validate fractions
            total_frac = sum(t["fraction"] for t in tiers)
            if abs(total_frac - 1.0) > 1e-6:
                log.warning(
                    "Merit splits [%s/%s]: fractions sum to %.4f ≠ 1.0 — skipping",
                    carrier, country, total_frac,
                )
                continue

            mask = (n.generators["carrier"] == carrier) & (countries_series == country)
            if not mask.any():
                log.info("Merit splits [%s/%s]: no generators found — skipping", carrier, country)
                continue

            # Idempotency: skip if any split sub-unit already exists
            if any(name.endswith("_t1") for name in n.generators.loc[mask].index):
                log.info("Merit splits [%s/%s]: already split — skipping", carrier, country)
                continue

            originals = n.generators.loc[mask].copy()
            new_rows = {}

            for orig_name, row in originals.iterrows():
                for i, tier in enumerate(tiers):
                    frac = tier["fraction"]
                    mc   = float(rng.uniform(tier["mc_lo"], tier["mc_hi"]))
                    new_name = f"{orig_name}_t{i + 1}"

                    # Inherit all scalar properties; override p_nom and marginal_cost
                    new_props = row.to_dict()
                    new_props["p_nom"]         = row["p_nom"] * frac
                    new_props["p_nom_min"]      = row.get("p_nom_min", 0.0) * frac
                    new_props["marginal_cost"]  = mc
                    new_rows[new_name] = new_props

            # Drop originals from generators DataFrame
            n.generators.drop(index=originals.index, inplace=True)

            # Add tiered sub-units using n.add to keep the component registry consistent
            for new_name, props in new_rows.items():
                ramp_up = props.get("ramp_limit_up", np.nan)
                ramp_dn = props.get("ramp_limit_down", np.nan)
                kwargs = dict(
                    bus           = props["bus"],
                    carrier       = props["carrier"],
                    p_nom         = props["p_nom"],
                    p_nom_min     = props["p_nom_min"],
                    p_min_pu      = props.get("p_min_pu", 0.0),
                    p_max_pu      = props.get("p_max_pu", 1.0),
                    marginal_cost = props["marginal_cost"],
                )
                if not np.isnan(ramp_up):
                    kwargs["ramp_limit_up"]   = ramp_up
                    kwargs["ramp_limit_down"]  = ramp_dn
                n.add("Generator", new_name, **kwargs)

            log.info(
                "Merit splits [%s/%s]: %d generators → %d sub-units "
                "(%.0f MW total, %d tiers, MC ranges %s)",
                carrier, country,
                len(originals), len(new_rows),
                originals["p_nom"].sum(),
                len(tiers),
                ", ".join(f"€{t['mc_lo']:.0f}–{t['mc_hi']:.0f}" for t in tiers),
            )


# ─── 1. CCGT tiering (MIBGAS-based time-varying MC) ──────────────────────────

def _apply_ccgt_tiers_and_co2(n, config, rng):
    """Assign time-varying marginal costs to ES/PT/FR CCGTs using MIBGAS gas prices.

    Each unit draws an efficiency η from its tier range (larger plant → better η).
    MC(t) = (MIBGAS(t) + gas_co2_intensity_th × co2_price) / η + VOM

    FR CCGTs now use the same MIBGAS-based approach as ES (via ccgt_efficiency_tiers.FR),
    replacing the old static merit_splits MCs that didn't respond to gas price fluctuations.
    The old FR merit_splits entry has been removed from config — FR CCGTs no longer get
    the "_t1" suffix from _apply_generator_splits, so they fall through to this pipeline.
    """
    efficiency_tiers = config.get("ccgt_efficiency_tiers", {})
    if not efficiency_tiers:
        log.warning("CCGT tiers: 'ccgt_efficiency_tiers' not in config — skipping")
        return

    mibgas      = _load_mibgas_ts(config, n.snapshots)
    co2_price   = config["co2_price"]
    gas_co2_th  = config["gas_co2_intensity_th"]   # 0.202 tCO₂/MWh_th
    vom         = config.get("gas_vom", {}).get("CCGT", 3.0)
    fr_gas_mult = float(config.get("fr_gas_multiplier", 1.0))
    countries   = _gen_countries(n)

    # Ensure generators_t.marginal_cost DataFrame exists with snapshot index
    if len(n.generators_t.marginal_cost) == 0:
        n.generators_t["marginal_cost"] = pd.DataFrame(index=n.snapshots)

    for country, tier_eta_ranges in efficiency_tiers.items():
        mask = (n.generators["carrier"] == "CCGT") & (countries == country)
        if not mask.any():
            continue

        # Drop any pre-existing time-varying MC columns for this country's CCGT fleet.
        # PyPSA-EUR base networks ship CCGTs with fuel-cost time-series; those values
        # must be replaced by the MIBGAS pipeline — never silently skipped.
        existing_tv = [g for g in n.generators.loc[mask].index
                       if g in n.generators_t.marginal_cost.columns]
        if existing_tv:
            n.generators_t.marginal_cost.drop(columns=existing_tv, inplace=True)
            log.info("CCGT %s: cleared %d pre-existing time-varying MC columns; applying MIBGAS", country, len(existing_tv))

        # Skip countries where merit_splits already set the MC (CO₂ baked in).
        # Detection: merit_splits renames generators with "_t1" / "_t2" / "_t3" suffixes.
        # Using name-suffix is more robust than the old MC threshold check, which broke
        # when FR CCGT MCs were lowered to realistic values (close to ES base-network MCs).
        if any(name.endswith("_t1") for name in n.generators.loc[mask].index):
            log.info("CCGT %s: already split by merit_splits (_t1 suffix found) — skipping MIBGAS", country)
            continue

        # Sort by p_nom descending: largest = most modern = best η (lowest tier index)
        idx = n.generators.loc[mask].sort_values("p_nom", ascending=False).index

        # Split by cumulative MW (not generator count) so each tier gets equal capacity.
        # Equal-count split biases T1 toward large generators that dominate total MW,
        # leaving T2/T3 with so little capacity they rarely dispatch.
        p_noms   = n.generators.loc[idx, "p_nom"].values
        cum_mw   = np.cumsum(p_noms)
        total_mw = cum_mw[-1]
        n_tiers  = len(tier_eta_ranges)
        splits   = []
        prev     = 0
        for i in range(n_tiers - 1):
            cut = int(np.searchsorted(cum_mw, total_mw * (i + 1) / n_tiers, side="right"))
            cut = max(cut, prev + 1)   # ensure at least one generator per tier
            splits.append(idx[prev:cut])
            prev = cut
        splits.append(idx[prev:])      # last tier gets remainder

        gas_input = mibgas * fr_gas_mult if (country == "FR" and fr_gas_mult != 1.0) else mibgas
        if country == "FR" and fr_gas_mult != 1.0:
            log.info(
                "CCGT FR: applying fr_gas_multiplier=%.2f (mean fuel price €%.1f → €%.1f/MWh_th)",
                fr_gas_mult, float(mibgas.mean()), float(gas_input.mean()),
            )

        for split_idx, (eta_lo, eta_hi) in zip(splits, tier_eta_ranges):
            for gen_name in split_idx:
                eta  = float(rng.uniform(eta_lo, eta_hi))
                mc_ts = (gas_input + gas_co2_th * co2_price) / eta + vom
                n.generators_t.marginal_cost[gen_name] = mc_ts.values
                # Store mean as static MC for display / solver fallback
                n.generators.loc[gen_name, "marginal_cost"] = float(mc_ts.mean())

        # ─── Efficiency tranche splitting ────────────────────────────────────────
        # After assigning MIBGAS-based MCs, split each CCGT into _base/_mid/_peak
        # tranches with decreasing η (increasing MC). This creates a piecewise-linear
        # cost ramp that prevents trickle-dispatch.
        # Formula: MC_new = (MC_old - VOM) / η_mult + VOM  (since MC ∝ 1/η)
        # Only applies to countries listed in ccgt_tranches.countries (default: ["ES"]).
        tranche_cfg = config.get("ccgt_tranches", {})
        if tranche_cfg.get("enabled", False):
            tranche_countries = tranche_cfg.get("countries", ["ES"])
            if country not in tranche_countries:
                log.info(
                    "CCGT %s: skipping efficiency tranches (not in ccgt_tranches.countries=%s)",
                    country, tranche_countries,
                )
            else:
                tranches = tranche_cfg["tranches"]
                total_frac = sum(t["capacity_fraction"] for t in tranches)
                if abs(total_frac - 1.0) > 0.01:
                    raise ValueError(
                        f"ccgt_tranches fractions sum to {total_frac}, expected 1.0"
                    )

                lp_gens = list(idx)
                for gen_name in lp_gens:
                    base_p_nom     = float(n.generators.at[gen_name, "p_nom"])
                    base_mc_tv     = n.generators_t.marginal_cost[gen_name].copy()
                    bus            = n.generators.at[gen_name, "bus"]
                    carrier        = n.generators.at[gen_name, "carrier"]
                    ramp_up        = float(n.generators.at[gen_name, "ramp_limit_up"])
                    ramp_dn        = float(n.generators.at[gen_name, "ramp_limit_down"])

                    for t in tranches:
                        suffix   = t["suffix"]
                        eta_mult = t["eta_multiplier"]
                        cap_frac = t["capacity_fraction"]

                        new_name      = f"{gen_name}{suffix}"
                        new_p_nom     = base_p_nom * cap_frac
                        new_mc_tv     = (base_mc_tv - vom) / eta_mult + vom
                        new_mc_static = float(new_mc_tv.mean())

                        n.add("Generator", new_name,
                              bus=bus,
                              carrier=carrier,
                              p_nom=new_p_nom,
                              marginal_cost=new_mc_static,
                              ramp_limit_up=ramp_up,
                              ramp_limit_down=ramp_dn)
                        n.generators_t.marginal_cost[new_name] = new_mc_tv.values

                    # Remove original generator
                    n.generators.drop(gen_name, inplace=True)
                    if gen_name in n.generators_t.marginal_cost.columns:
                        n.generators_t.marginal_cost.drop(columns=[gen_name], inplace=True)

                idx = pd.Index([
                    f"{gen_name}{t['suffix']}"
                    for gen_name in lp_gens for t in tranches
                ])

                log.info(
                    "CCGT %s: split %d LP units into %d tranches each → %d tranche generators total",
                    country, len(lp_gens), len(tranches), len(lp_gens) * len(tranches),
                )

        log.info(
            "CCGT %s: MIBGAS-based time-varying MCs → %d units "
            "(η tiers %s, mean MC ≈ €%.0f–%.0f)",
            country, mask.sum(),
            ", ".join(f"{lo:.2f}–{hi:.2f}" for lo, hi in tier_eta_ranges),
            n.generators_t.marginal_cost[idx[-1]].mean(),   # worst tier
            n.generators_t.marginal_cost[idx[0]].mean(),    # best tier
        )

    # ─── Override base-network p_nom_min / ramp limits ────────────────────────
    # PyPSA-EUR ships CCGTs with p_nom_min = p_nom (must-run at full capacity).
    # This prevents flexible dispatch — the solver must either run them at 100%
    # or not at all. We override to the config-defined minimum so they can ramp.
    ccgt_cfg = config.get("ccgt", {})
    per_country = ccgt_cfg.get("per_country", {})
    countries = _gen_countries(n)

    for country, override in per_country.items():
        mask = (n.generators["carrier"] == "CCGT") & (countries == country)
        if not mask.any():
            continue
        p_min = override.get("p_min_pu", ccgt_cfg.get("p_min_pu", 0.0))
        r_up  = override.get("ramp_limit_up", ccgt_cfg.get("ramp_limit_up", 1.0))
        r_dn  = override.get("ramp_limit_down", ccgt_cfg.get("ramp_limit_down", 1.0))
        n.generators.loc[mask, "p_nom_min"] = n.generators.loc[mask, "p_nom"] * p_min
        n.generators.loc[mask, "ramp_limit_up"]   = r_up
        n.generators.loc[mask, "ramp_limit_down"]  = r_dn
        log.info(
            "CCGT ops [%s]: p_min_pu=%.2f, ramp_up=%.2f, ramp_dn=%.2f → %d units",
            country, p_min, r_up, r_dn, mask.sum(),
        )

    # Apply top-level defaults to any CCGTs not covered by per-country (safety net)
    default_mask = (n.generators["carrier"] == "CCGT")
    for country in set(countries[default_mask]) - set(per_country.keys()):
        m = default_mask & (countries == country)
        if m.any():
            p_min = ccgt_cfg.get("p_min_pu", 0.0)
            r_up  = ccgt_cfg.get("ramp_limit_up", 1.0)
            r_dn  = ccgt_cfg.get("ramp_limit_down", 1.0)
            n.generators.loc[m, "p_nom_min"] = n.generators.loc[m, "p_nom"] * p_min
            n.generators.loc[m, "ramp_limit_up"]   = r_up
            n.generators.loc[m, "ramp_limit_down"]  = r_dn
            log.info(
                "CCGT ops [%s]: fallback p_min_pu=%.2f, ramp=%.2f → %d units",
                country, p_min, r_up, m.sum(),
            )


# ─── 2. CCGT_Flex split ───────────────────────────────────────────────────────

def _apply_ccgt_flex(n, config, rng):
    """Carve a low-efficiency flex slice from each ES CCGT with MIBGAS-based MC.

    CCGT_flex represents units operating in partial-load / open-cycle mode:
    η 0.38–0.44, always more expensive than regular CCGT (η 0.46–0.60).
    MC(t) = (MIBGAS(t) + 0.202 × co2_price) / η + VOM_flex
    """
    if (n.generators["carrier"] == "CCGT_flex").any():
        log.info("CCGT_Flex: already present — skipping")
        return

    cfg        = config["ccgt_flex"]
    frac       = cfg["capacity_fraction"]
    ramp       = cfg["ramp_limit_pu"]
    mibgas     = _load_mibgas_ts(config, n.snapshots)
    co2_price  = config["co2_price"]
    gas_co2_th = config["gas_co2_intensity_th"]
    vom        = config.get("gas_vom", {}).get("CCGT_flex", 3.0)

    # Support either multi-tier ("efficiency_tiers") or single-range ("efficiency_range")
    eta_tiers = cfg.get("efficiency_tiers")
    if eta_tiers is None:
        single_range = cfg["efficiency_range"]
        eta_tiers = [single_range]   # wrap for uniform loop below

    countries = _gen_countries(n)
    es_ccgt   = n.generators[(n.generators["carrier"] == "CCGT") & (countries == "ES")]

    if "CCGT_flex" not in n.carriers.index:
        n.add("Carrier", "CCGT_flex", nice_name="CCGT Flex", color="#E74C3C")

    if len(n.generators_t.marginal_cost) == 0:
        n.generators_t["marginal_cost"] = pd.DataFrame(index=n.snapshots)

    n_tiers   = len(eta_tiers)
    added_mw  = 0.0
    for gen_name, row in es_ccgt.iterrows():
        tier_pnom = row["p_nom"] * frac / n_tiers   # each tier gets equal slice
        if tier_pnom < 1.0:
            continue
        for t_idx, (eta_lo, eta_hi) in enumerate(eta_tiers):
            eta       = float(rng.uniform(eta_lo, eta_hi))
            mc_ts     = (mibgas + gas_co2_th * co2_price) / eta + vom
            flex_name = f"CCGT_Flex_T{t_idx + 1}_{gen_name}"
            n.add(
                "Generator",
                flex_name,
                bus=row["bus"],
                carrier="CCGT_flex",
                p_nom=tier_pnom,
                marginal_cost=float(mc_ts.mean()),
                ramp_limit_up=ramp,
                ramp_limit_down=ramp,
            )
            n.generators_t.marginal_cost[flex_name] = mc_ts.values
        # Carve frac off the parent CCGT regardless of tier count
        n.generators.loc[gen_name, "p_nom"] *= (1.0 - frac)
        if n.generators.loc[gen_name, "p_nom_min"] > 0:
            n.generators.loc[gen_name, "p_nom_min"] *= (1.0 - frac)
        added_mw += tier_pnom * n_tiers

    log.info(
        "CCGT_Flex: %.0f MW split across %d ES CCGTs × %d tiers (η %s, VOM=€%.0f)",
        added_mw, len(es_ccgt), n_tiers,
        " / ".join(f"{lo:.2f}–{hi:.2f}" for lo, hi in eta_tiers),
        vom,
    )


# ─── 3. CCGT must-run (industrial CHP proxy) ─────────────────────────────────

def _apply_ccgt_must_run(n, config):
    cfg = config.get("ccgt_must_run", {})
    if not cfg.get("enabled", False):
        return
    if any(g.endswith("_must_run") for g in n.generators.index):
        log.info("CCGT_must_run: already present — skipping")
        return

    target_mw = float(cfg["target_mw"])
    mc        = float(cfg["marginal_cost"])
    p_min_pu  = float(cfg.get("p_min_pu", 0.0))
    ramp      = float(cfg.get("ramp_limit_pu", 1.0))
    color     = cfg.get("color", "#1E252B")

    countries = _gen_countries(n)
    es_ccgt = n.generators[(n.generators["carrier"] == "CCGT") & (countries == "ES")].copy()

    total_ccgt_mw = es_ccgt["p_nom"].sum()
    if total_ccgt_mw < 1.0:
        log.warning("CCGT_must_run: no ES CCGT capacity found — skipping")
        return

    alpha = target_mw / total_ccgt_mw  # proportional carve fraction

    if "CCGT_must_run" not in n.carriers.index:
        n.add("Carrier", "CCGT_must_run", nice_name="CCGT Must-Run", color=color)

    added_mw = 0.0
    for gen_id, row in es_ccgt.iterrows():
        p_chunk = row["p_nom"] * alpha
        if p_chunk < 0.5:
            continue
        # Shrink parent proportionally (keeps net capacity neutral)
        n.generators.loc[gen_id, "p_nom"] -= p_chunk
        if n.generators.loc[gen_id, "p_nom_min"] > 0:
            n.generators.loc[gen_id, "p_nom_min"] -= p_chunk
        n.add(
            "Generator",
            f"{gen_id}_must_run",
            bus=row["bus"],
            carrier="CCGT_must_run",
            p_nom=p_chunk,
            marginal_cost=mc,
            committable=False,
            p_min_pu=p_min_pu,
            ramp_limit_up=ramp,
            ramp_limit_down=ramp,
        )
        added_mw += p_chunk

    after_ccgt_mw = n.generators.loc[es_ccgt.index, "p_nom"].sum()
    after_mr_mw   = n.generators[n.generators["carrier"] == "CCGT_must_run"]["p_nom"].sum()
    assert abs((after_ccgt_mw + after_mr_mw) - total_ccgt_mw) < 1.0, (
        f"CCGT_must_run net-neutrality violated: {after_ccgt_mw:.0f} + {after_mr_mw:.0f} ≠ {total_ccgt_mw:.0f}"
    )
    log.info(
        "CCGT_must_run: carved %.0f MW (target %.0f) from %d ES CCGTs at MC=€%.0f — total CCGT=%.0f MR=%.0f",
        added_mw, target_mw, len(es_ccgt), mc, after_ccgt_mw, after_mr_mw,
    )


# ─── 3b. Biomass / cogeneration — constant must-run injection ────────────────

def _apply_biomass_correction(n, config):
    """Scale ES biomass fleet and configure as a constant zero-MC must-run injection.

    p_min_pu = p_max_pu = 1.0 forces the LP to dispatch exactly p_nom every hour.
    MC = 0 ensures it is always dispatched first (heat obligation proxy).
    This removes cogeneration from the optimization and treats it as a fixed
    600 MW baseload offset — equivalent to the CCGT_must_run approach.
    Idempotent: checks p_nom vs target before rescaling.
    """
    cfg = config.get("biomass", {})
    if not cfg.get("enabled", False):
        return

    target_mw = float(cfg["target_mw"])
    mc_target  = float(cfg.get("marginal_cost", 40.0))
    p_min_pu   = float(cfg.get("p_min_pu", 0.0))

    countries = _gen_countries(n)
    es_bio    = n.generators[(n.generators["carrier"] == "biomass") & (countries == "ES")]
    if es_bio.empty:
        log.warning("Biomass: no ES biomass generators found — skipping")
        return

    current_mw = es_bio["p_nom"].sum()
    if abs(current_mw - target_mw) > 5.0:
        scale = target_mw / current_mw
        n.generators.loc[es_bio.index, "p_nom"] *= scale
        log.info("Biomass: scaled %.0f → %.0f MW (×%.3f)", current_mw, target_mw, scale)
    else:
        log.info("Biomass: already at target %.0f MW — skipping capacity scale", target_mw)

    n.generators.loc[es_bio.index, "marginal_cost"] = mc_target
    n.generators.loc[es_bio.index, "p_min_pu"]      = p_min_pu
    n.generators.loc[es_bio.index, "p_max_pu"]      = 1.0   # explicit ceiling at p_nom
    n.generators.loc[es_bio.index, "p_nom_min"]     = (
        n.generators.loc[es_bio.index, "p_nom"] * p_min_pu
    )
    dispatch_mw = n.generators.loc[es_bio.index, "p_nom"].sum() * p_min_pu
    log.info(
        "Biomass: constant must-run %.0f MW, MC=€%.0f/MWh, p_min_pu=%.1f, %d plants",
        dispatch_mw, mc_target, p_min_pu, len(es_bio),
    )


# ─── 4. Peaker fleet ──────────────────────────────────────────────────────────

def _apply_peakers(n, config):
    existing = [g for g in n.generators.index if g.startswith(("OCGT_pk_", "Diesel_pk_"))]
    if existing:
        log.info("Peakers: already present (%d generators) — skipping", len(existing))
        return

    lw = config["peakers"]["load_weight"]
    cw = config["peakers"]["ccgt_weight"]
    countries = _gen_countries(n)

    es_buses = n.buses.index[n.buses.index.str.startswith("ES")]

    # Load share: mean hourly demand per ES bus
    es_loads = n.loads[n.loads["bus"].isin(es_buses)]
    if len(es_loads) == 0:
        log.warning("Peakers: no ES loads found — placing by CCGT share only")
        load_by_bus = pd.Series(0.0, index=es_buses)
    else:
        available_cols = [c for c in es_loads.index if c in n.loads_t.p_set.columns]
        if available_cols:
            mean_p = n.loads_t.p_set[available_cols].mean()
        else:
            mean_p = n.loads.loc[es_loads.index, "p_set"] if "p_set" in n.loads.columns else pd.Series(1.0, index=es_loads.index)
        load_by_bus = (
            pd.DataFrame({"bus": es_loads["bus"], "p": mean_p.reindex(es_loads.index, fill_value=0.0)})
            .groupby("bus")["p"]
            .sum()
        )

    # CCGT share: total installed capacity per ES bus
    es_ccgt = n.generators[
        (n.generators["carrier"].isin(["CCGT", "CCGT_flex"])) & (countries == "ES")
    ]
    ccgt_by_bus = es_ccgt.groupby("bus")["p_nom"].sum()

    load_s = load_by_bus.reindex(es_buses, fill_value=0.0)
    ccgt_s = ccgt_by_bus.reindex(es_buses, fill_value=0.0)
    if load_s.sum() > 0:
        load_s = load_s / load_s.sum()
    if ccgt_s.sum() > 0:
        ccgt_s = ccgt_s / ccgt_s.sum()

    weight = lw * load_s + cw * ccgt_s
    if weight.sum() > 0:
        weight = weight / weight.sum()

    mibgas     = _load_mibgas_ts(config, n.snapshots)
    co2_price  = config["co2_price"]
    gas_co2_th = config["gas_co2_intensity_th"]

    if len(n.generators_t.marginal_cost) == 0:
        n.generators_t["marginal_cost"] = pd.DataFrame(index=n.snapshots)

    for prefix, pk_cfg in config["peakers"].items():
        if prefix in ("load_weight", "ccgt_weight"):
            continue
        carrier  = pk_cfg["carrier"]
        total_mw = pk_cfg["total_mw"]
        if carrier not in n.carriers.index:
            n.add("Carrier", carrier, nice_name=carrier.capitalize())

        # Gas peakers (OCGT): time-varying MC from MIBGAS + efficiency
        if "eta" in pk_cfg:
            eta      = pk_cfg["eta"]
            vom      = pk_cfg.get("vom", 0.0)
            mc_ts    = (mibgas + gas_co2_th * co2_price) / eta + vom
            static_mc = float(mc_ts.mean())
            use_tv   = True
        else:
            # Non-gas peakers (diesel): static MC with CO₂ adder
            co2_adder = config["co2_price"] * config["co2_intensity"].get(carrier, 0.0)
            static_mc = pk_cfg["base_mc"] + co2_adder + pk_cfg.get("vom", 0.0)
            mc_ts     = None
            use_tv    = False

        count = 0
        for bus in es_buses:
            bus_mw = total_mw * float(weight.get(bus, 0.0))
            if bus_mw < 0.5:
                continue
            gen_name = f"{prefix}_{bus}"
            n.add(
                "Generator",
                gen_name,
                bus=bus,
                carrier=carrier,
                p_nom=bus_mw,
                marginal_cost=static_mc,
            )
            if use_tv:
                n.generators_t.marginal_cost[gen_name] = mc_ts.values
            count += 1

        if use_tv:
            log.info(
                "Peakers %s: %.0f MW across %d ES nodes (η=%.2f, mean MC=€%.0f, time-varying)",
                prefix, total_mw, count, pk_cfg["eta"], static_mc,
            )
        else:
            log.info(
                "Peakers %s: %.0f MW across %d ES nodes (MC=€%.0f, static)",
                prefix, total_mw, count, static_mc,
            )


# ─── 4. Nuclear constraints ───────────────────────────────────────────────────

def _apply_nuclear(n, config, rng):
    cfg = config["nuclear"]
    per_country = cfg.get("per_country", {})
    mask = n.generators["carrier"] == "nuclear"
    if not mask.any():
        return

    # Countries whose nuclear MCs were already set by _apply_generator_splits
    # should not be overwritten with the flat mc_range jitter.
    split_nuclear_countries = set(config.get("merit_splits", {}).get("nuclear", {}).keys())

    default_lo, default_hi = cfg["mc_range"]
    countries = _gen_countries(n)
    for gen_idx in n.generators.loc[mask].index:
        country = countries.loc[gen_idx]
        if country in split_nuclear_countries:
            continue   # MC already set per-tier in _apply_generator_splits — skip
        lo, hi = per_country.get(country, {}).get("mc_range", (default_lo, default_hi))
        n.generators.loc[gen_idx, "marginal_cost"] = float(rng.uniform(lo, hi))

    for country, override in per_country.items():
        cmask = mask & (countries == country)
        if not cmask.any():
            continue
        p_min   = override.get("p_min_pu", cfg["p_min_pu"])
        ramp    = override.get("ramp_limit_pu", cfg["ramp_limit_pu"])
        p_max   = override.get("p_max_pu", 1.0)
        lo, hi  = override.get("mc_range", (default_lo, default_hi))
        n.generators.loc[cmask, "p_min_pu"] = p_min
        n.generators.loc[cmask, "p_max_pu"] = p_max
        n.generators.loc[cmask, "ramp_limit_up"]   = ramp
        n.generators.loc[cmask, "ramp_limit_down"]  = ramp
        n.generators.loc[cmask, "p_nom_min"] = n.generators.loc[cmask, "p_nom"] * p_min
        log.info(
            "Nuclear [%s]: p_min=%.2f, p_max=%.2f, ramp=%.2f, MC [%.0f–%.0f] → %d units",
            country, p_min, p_max, ramp, lo, hi, cmask.sum(),
        )

    # Apply defaults for any nuclear units not covered by per_country
    remaining = mask & ~countries.isin(list(per_country.keys()))
    if remaining.any():
        n.generators.loc[remaining, "p_min_pu"] = cfg["p_min_pu"]
        n.generators.loc[remaining, "ramp_limit_up"]   = cfg["ramp_limit_pu"]
        n.generators.loc[remaining, "ramp_limit_down"]  = cfg["ramp_limit_pu"]
        n.generators.loc[remaining, "p_nom_min"] = n.generators.loc[remaining, "p_nom"] * cfg["p_min_pu"]

    log.info(
        "Nuclear: MC jitter applied to %d units (default [%.0f–%.0f]; per-country overrides where set)",
        mask.sum(), default_lo, default_hi,
    )


def _apply_vre_mc(n, config):
    """Set non-zero marginal cost for VRE generators (solar, wind) to act as price setters.

    Rationale: MC=0 means VRE never clears the merit order — under congestion the LP
    cannot distinguish local from remote VRE. A small positive MC (0.25 €/MWh) preserves
    near-zero dispatch cost while giving the solver a tiebreaker that favours local
    generation when line capacity is the binding constraint.
    """
    cfg = config.get("vre", {})
    mc_map = cfg.get("marginal_cost", {})
    if not mc_map:
        return

    countries = _gen_countries(n)
    for carrier, mc in mc_map.items():
        mask = (n.generators["carrier"] == carrier) & (countries == "ES")
        if not mask.any():
            continue
        n.generators.loc[mask, "marginal_cost"] = float(mc)
        log.info(
            "VRE MC [%s / ES]: %.2f €/MWh → %d generators (%.0f MW)",
            carrier, mc, mask.sum(), n.generators.loc[mask, "p_nom"].sum(),
        )


# ─── 5b. Solar capacity scaler ─────────────────────────────────────────────────
def _apply_solar_capacity_scaler(n, config):
    """Scale ES solar p_nom to match REE end-2024 installed capacity.

    ESIOS CSV reports 32,350 MW of solar PV (early 2024 snapshot).
    REE end-2024 reports 39,321 MW — a 1.22× increase over the year.
    Scaling p_nom (rather than p_max_pu) is equivalent for dispatch but
    keeps the capacity comparison in _plot_capacity_vs_reality() honest.

    Idempotent: checks for a marker attribute before applying.
    """
    cfg = config.get("vre", {})
    factor = cfg.get("solar_capacity_scaler", None)
    if factor is None or factor == 1.0:
        return

    # Idempotency marker — int, not bool (NetCDF4 rejects b1 dtype)
    if getattr(n, "_solar_scaler_applied", 0):
        log.info("Solar capacity scaler: already applied — skipping")
        return

    solar_mask = n.generators["carrier"] == "solar"
    if not solar_mask.any():
        log.warning("Solar capacity scaler: no solar generators found — skipping")
        return

    gen_countries = _gen_countries(n)
    es_mask = solar_mask & (gen_countries == "ES")
    if not es_mask.any():
        log.warning("Solar capacity scaler: no ES solar generators found — skipping")
        return

    gens = es_mask.index[es_mask.values]
    before = n.generators.loc[gens, "p_nom"].sum()
    n.generators.loc[gens, "p_nom"] *= factor
    after = n.generators.loc[gens, "p_nom"].sum()
    n._solar_scaler_applied = 1

    log.info(
        "Solar capacity scaler [ES]: p_nom × %.2f → %d generators "
        "(%.0f → %.0f MW, +%.0f MW)",
        factor, len(gens), before, after, after - before,
    )


# ─── 5c. Solar thermal (CSP) ────────────────────────────────────────────────────

def _apply_solar_thermal(n, config):
    """Add CSP StorageUnits with TES from pre-computed atlite inflow profiles.

    Reads ``Analysis/data/csp_profiles.nc`` (generated by
    ``Analysis/build_csp_profiles.py``), which contains per-bus inflow time
    series (MW_electrical-equivalent, post-turbine) aggregated from the 53 CSP
    plants (2,303 MW total) in the ``CSP_Spain.csv`` database.

    For each bus in the profile file, adds a single StorageUnit with:
        carrier              = csp
        p_nom                = turbine rating (MW_e)
        inflow               = solar heat collected (MW_e, post-turbine)
        max_hours            = 7.5  (thermal storage at p_nom)
        efficiency_store     = 0.99 (heat-to-tank, near-lossless)
        efficiency_dispatch  = 1.0  (turbine conversion baked into inflow)
        standing_loss        = 0.001 (0.1%/h thermal decay)
        marginal_cost        = 0.5  (€/MWh — tiny, lets LP dispatch freely)
        marginal_cost_storage = 0.0 (€/MWh — no cost to store)
        p_min_pu             = 0.0  (no must-run on the turbine)
        cyclic_state_of_charge = False (rolling horizon handles SOC)
        initial_soc          = 0.50

    Idempotent: checks for a marker attribute before applying.
    """
    cfg = config.get("solar_thermal", {})
    if not cfg.get("enabled", False):
        return

    if getattr(n, "_solar_thermal_applied", 0):
        log.info("Solar thermal: already applied — skipping")
        return

    # Ensure carrier exists
    if "csp" not in n.carriers.index:
        n.add("Carrier", "csp")

    # Resolve paths relative to the project root
    root = Path(__file__).resolve().parent.parent
    profile_path = root / "Analysis" / cfg["profile_path"]

    if not profile_path.exists():
        log.warning(
            "Solar thermal: profile file not found at %s — "
            "run 'pixi run python Analysis/build_csp_profiles.py' first. Skipping.",
            profile_path,
        )
        return

    # Read config parameters
    mc              = float(cfg.get("marginal_cost", 0.5))
    mc_storage      = float(cfg.get("marginal_cost_storage", 0.0))
    max_hours       = float(cfg.get("max_hours", 7.5))
    eta_store       = float(cfg.get("efficiency_store", 0.99))
    eta_dispatch    = float(cfg.get("efficiency_dispatch", 1.0))
    standing_loss   = float(cfg.get("standing_loss", 0.001))
    initial_soc     = float(cfg.get("initial_soc", 0.50))
    p_min_pu        = float(cfg.get("p_min_pu", 0.0))

    # Load pre-computed profiles
    ds = xr.open_dataset(profile_path)
    buses_in_profile = ds["bus"].values.tolist()
    log.info(
        "Solar thermal: loading profiles for %d buses (%.0f MW turbine total)",
        len(buses_in_profile), float(ds["capacity"].sum()),
    )

    # Align time index: the profile may have a different time range than the
    # network.  Reindex to the network's snapshots, filling missing with 0.
    snapshots = n.snapshots
    inflow_da = ds["inflow"]  # (bus, time) in MW_e
    inflow_aligned = inflow_da.interp(
        time=snapshots, method="linear", kwargs={"fill_value": 0.0}
    )

    added = 0
    for bus_name in buses_in_profile:
        if bus_name not in n.buses.index:
            log.debug("Solar thermal: bus %s not in network — skipping", bus_name)
            continue

        cap_mw  = float(ds["capacity"].sel(bus=bus_name).values)
        inflow  = inflow_aligned.sel(bus=bus_name).values  # numpy array, len = n_snapshots

        su_name = f"{bus_name} CSP"
        suffix = 1
        while su_name in n.storage_units.index:
            su_name = f"{bus_name} CSP_{suffix}"
            suffix += 1

        n.add(
            "StorageUnit", su_name,
            bus                    = bus_name,
            carrier                = "csp",
            p_nom                  = cap_mw,
            inflow                 = inflow,
            max_hours              = max_hours,
            efficiency_store       = eta_store,
            efficiency_dispatch    = eta_dispatch,
            standing_loss          = standing_loss,
            marginal_cost          = mc,
            marginal_cost_storage  = mc_storage,
            p_min_pu               = p_min_pu,
            cyclic_state_of_charge = False,
            initial_soc            = initial_soc * cap_mw * max_hours,  # MWh
        )
        added += 1

    n._solar_thermal_applied = 1
    log.info(
        "Solar thermal: added %d StorageUnits (%.0f MW turbine, %.1f h storage) "
        "from %d-profile buses",
        added, float(ds["capacity"].sum()), max_hours, len(buses_in_profile),
    )


# ─── 5d. Hydro parameters ──────────────────────────────────────────────────────

def _monthly_mwh_to_hourly_mw(monthly_mwh: pd.Series, snapshots: pd.DatetimeIndex) -> pd.Series:
    """
    Spread monthly total inflow (MWh/month) to a constant hourly MW value for
    each snapshot that falls within that month.  Months not covered by snapshots
    are ignored; snapshots not covered by any monthly entry get 0 MW inflow.

    Handles UTC-aware monthly index vs tz-naive network snapshots automatically.
    """
    snaps_naive = snapshots.tz_localize(None) if snapshots.tz is not None else snapshots
    result = np.zeros(len(snapshots), dtype=float)
    for ts, total_mwh in monthly_mwh.items():
        if pd.isna(total_mwh):
            continue
        ts_naive = ts.tz_localize(None) if ts.tzinfo is not None else ts
        mask = (snaps_naive.year == ts_naive.year) & (snaps_naive.month == ts_naive.month)
        n_hours = mask.sum()
        if n_hours > 0:
            result[mask] = float(total_mwh) / n_hours
    return pd.Series(result, index=snapshots)


def _resolve_initial_soc(n, cfg) -> float:
    """Return the initial SOC fraction for the simulation start month.

    Priority order:
      1. ``initial_soc_monthly`` — real observed reservoir fill by month (preferred).
         Set in config.hydro.per_country.<CTY>.initial_soc_monthly (keys 1-12).
      2. ``terminal_soc_monthly`` — terminal SOC targets (used as fallback if no
         dedicated initial table is present).
      3. ``initial_soc`` — flat scalar fallback (default 0.50).

    Uses n.snapshots[0] to identify the start month.
    """
    snap0 = n.snapshots[0]
    ts    = pd.Timestamp(snap0)
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    month = ts.month

    fallback = float(cfg.get("initial_soc", 0.50))

    # 1. Dedicated initial SOC table (real observed data)
    init_monthly = cfg.get("initial_soc_monthly", {})
    if init_monthly and month in init_monthly:
        return float(init_monthly[month])

    # 2. Terminal SOC targets as proxy
    term_monthly = cfg.get("terminal_soc_monthly", {})
    if term_monthly and month in term_monthly:
        return float(term_monthly[month])

    return fallback


def _hydro_convert_gens_to_storage(n, cfg):
    """Convert FR/PT hydro generators to storage_units so they get SOC tracking
    and SOC-tiered MCs identical to ES reservoirs.  Must be called before the
    main mask is defined in _apply_hydro so new units are included automatically.
    """
    per_country = cfg.get("per_country", {})
    hydro_gen_mask = n.generators["carrier"] == "hydro"
    if not hydro_gen_mask.any():
        return
    gen_cntry = _gen_countries(n)

    # Load ENTSO-E monthly inflow CSV once (shared across countries).
    # Column format: "{bus}_hydro"  e.g. FR_WEST_hydro, PT_NORTH_hydro.
    inflow_csv_path = cfg.get("inflow_csv")
    inflow_df: pd.DataFrame = pd.DataFrame()
    if inflow_csv_path:
        p = Path(inflow_csv_path)
        if p.exists():
            inflow_df = pd.read_csv(p, index_col=0, parse_dates=True)
            log.info("Hydro: loaded inflow CSV %s  (%d months × %d nodes)", p.name, len(inflow_df), len(inflow_df.columns))
        else:
            log.warning("Hydro: inflow_csv not found: %s — falling back to inflow_pu", inflow_csv_path)

    # ── Phase 1: collect all generators to convert BEFORE any n.remove() calls ──
    # n.generators shrinks as generators are removed, making stale boolean masks
    # misalign on subsequent country iterations (IndexError: size 349 vs 347).
    # Snapshot everything needed while the DataFrame is still at its original size.
    to_convert: dict[str, dict] = {}
    for country, ctry_cfg in per_country.items():
        if not ctry_cfg.get("convert_to_storage", False):
            continue
        cmask = hydro_gen_mask & (gen_cntry == country)
        if not cmask.any():
            continue
        gen_names = n.generators.index[cmask].tolist()
        bus_pnom: dict[str, float] = {}
        for g in gen_names:
            b = n.generators.loc[g, "bus"]
            bus_pnom[b] = bus_pnom.get(b, 0.0) + float(n.generators.loc[g, "p_nom"])
        to_convert[country] = {
            "gen_names":  gen_names,
            "bus_pnom":   bus_pnom,
            "max_h":      ctry_cfg.get("max_hours", cfg.get("max_hours", 1700)),
            "inflow_pu":  ctry_cfg.get("inflow_pu", 0.0),
            "p_max_val":  ctry_cfg.get("p_max_pu", 1.0),
            "gen_rows":   {g: n.generators.loc[g].copy() for g in gen_names},
        }

    # ── Phase 2: convert (n.generators mutated here) ───────────────────────────
    for country, info in to_convert.items():
        max_h     = info["max_h"]
        inflow_pu = info["inflow_pu"]
        p_max_val = info["p_max_val"]
        bus_pnom  = info["bus_pnom"]
        # Resolve initial SOC per country: use country-level monthly table if present,
        # fall back to global ES table so changing start date auto-adjusts all countries.
        ctry_cfg = per_country.get(country, {})
        if "terminal_soc_monthly" in ctry_cfg or "initial_soc" in ctry_cfg:
            initial_soc = _resolve_initial_soc(n, ctry_cfg)
        else:
            initial_soc = _resolve_initial_soc(n, cfg)

        for gen_name in info["gen_names"]:
            gen   = info["gen_rows"][gen_name]
            gen   = n.generators.loc[gen_name]
            p_nom = float(gen["p_nom"])
            e_cap = p_nom * max_h

            n.add(
                "StorageUnit",
                name=gen_name,
                bus=gen["bus"],
                carrier="hydro",
                p_nom=p_nom,
                max_hours=max_h,
                p_max_pu=p_max_val,
                p_min_pu=0.0,               # no grid charging — inflow fills reservoir
                efficiency_dispatch=1.0,
                efficiency_store=0.0,        # unused (p_min_pu=0 prevents storing)
                cyclic_state_of_charge=False,
                state_of_charge_initial=initial_soc * e_cap,
                marginal_cost=0.0,           # overwritten by SOC tiers below
            )

            # ── Inflow: CSV (preferred) or constant inflow_pu (fallback) ──────
            node_key = f"{gen['bus']}_hydro"
            if not inflow_df.empty and node_key in inflow_df.columns:
                # Split node-level monthly MWh by this generator's p_nom share.
                gen_share  = p_nom / bus_pnom[gen["bus"]]
                node_monthly = inflow_df[node_key] * gen_share
                hourly_mw  = _monthly_mwh_to_hourly_mw(node_monthly, n.snapshots)
                if "inflow" not in n.storage_units_t or n.storage_units_t.inflow.empty:
                    n.storage_units_t["inflow"] = pd.DataFrame(index=n.snapshots)
                n.storage_units_t.inflow[gen_name] = hourly_mw.values
                annual_gwh = float(hourly_mw.sum()) / 1000
                log.info(
                    "Hydro: converted %s [%s] → StorageUnit "
                    "(p_nom=%.0f MW, max_h=%.0f, share=%.3f, CSV inflow=%.0f GWh/yr, soc0=%.0f%%)",
                    gen_name, country, p_nom, max_h, gen_share, annual_gwh, initial_soc * 100,
                )
            elif inflow_pu > 0:
                if "inflow" not in n.storage_units_t or n.storage_units_t.inflow.empty:
                    n.storage_units_t["inflow"] = pd.DataFrame(index=n.snapshots)
                n.storage_units_t.inflow[gen_name] = p_nom * inflow_pu
                log.info(
                    "Hydro: converted %s [%s] → StorageUnit "
                    "(p_nom=%.0f MW, max_h=%.0f, inflow_pu=%.3f constant, soc0=%.0f%%)",
                    gen_name, country, p_nom, max_h, inflow_pu, initial_soc * 100,
                )
            else:
                log.info(
                    "Hydro: converted %s [%s] → StorageUnit "
                    "(p_nom=%.0f MW, max_h=%.0f, no inflow, soc0=%.0f%%)",
                    gen_name, country, p_nom, max_h, initial_soc * 100,
                )

            n.remove("Generator", gen_name)


def _redistribute_hydro_inflow(n, cfg, mask):
    """Daisy-chain surplus inflow from oversupplied pondage → larger reservoirs.

    ERA5 distributes runoff proportionally by p_nom (add_electricity.py:777).
    Small pondage units (e.g. ES0 36, 42 MW) get 1,564 GWh/yr inflow while
    large reservoirs (e.g. ES0 37, 1,354 MW) get only 16 GWh/yr.  The LP
    spills the excess because the turbine is 36× undersized.

    This function keeps enough inflow for the source to run at ~90% CF and
    redirects the remainder to a geographically-close target with spare
    turbine and storage capacity.

    Config section (optional — sensible defaults below):
        "inflow_redistribution": {
            "enabled": true,
            "target_cf": 0.90,       # source keeps enough for this CF
            "pairs": [
                {"source": "ES0 36 hydro", "target": "ES0 37 hydro"},
                ...
            ]
        }
    """
    inflow_cfg = cfg.get("inflow_redistribution", {})
    if not inflow_cfg.get("enabled", False):
        return

    pairs = inflow_cfg.get("pairs", [])
    if not pairs:
        log.warning("Hydro inflow redistribution enabled but no pairs defined — skipping")
        return

    target_cf = inflow_cfg.get("target_cf", 0.90)
    inflow_t = n.storage_units_t.get("inflow", pd.DataFrame())
    if inflow_t.empty:
        log.warning("Hydro inflow redistribution: no inflow time series found — skipping")
        return

    total_redirected = 0.0
    for pair in pairs:
        src = pair["source"]
        tgt = pair["target"]

        if src not in inflow_t.columns:
            log.warning("  Source '%s' not in inflow columns — skipping", src)
            continue
        if tgt not in inflow_t.columns:
            log.warning("  Target '%s' not in inflow columns — skipping", tgt)
            continue

        # Source turbine capacity
        src_p_nom = float(n.storage_units.loc[src, "p_nom"]) if src in n.storage_units.index else 0.0
        if src_p_nom <= 0:
            log.warning("  Source '%s' has p_nom=%.0f — skipping", src, src_p_nom)
            continue

        # Keep enough inflow for source to run at target_cf
        src_keep_mw = src_p_nom * target_cf
        src_inflow = inflow_t[src].copy()
        src_surplus = src_inflow - src_keep_mw
        # Only redirect positive surplus (don't steal from already-undersupplied)
        src_surplus = src_surplus.clip(lower=0.0)

        redirected_gwh = src_surplus.sum() * 1e-3  # MW → GWh
        if redirected_gwh < 0.001:
            continue

        # Apply: reduce source inflow, add to target
        inflow_t[src] = inflow_t[src] - src_surplus
        inflow_t[tgt] = inflow_t[tgt] + src_surplus

        total_redirected += redirected_gwh
        log.info(
            "  %s → %s: keep %.0f MW (%.0f%% CF), redirect %.1f GWh/yr "
            "(src p_nom=%.0f MW, tgt p_nom=%.0f MW)",
            src, tgt, src_keep_mw, target_cf * 100,
            redirected_gwh, src_p_nom,
            float(n.storage_units.loc[tgt, "p_nom"]) if tgt in n.storage_units.index else 0.0,
        )

    if total_redirected > 0:
        log.info(
            "Hydro inflow redistribution: %.1f GWh/yr redirected across %d pairs",
            total_redirected, len(pairs),
        )


def _apply_hydro(n, config):
    cfg = config["hydro"]

    # Convert any FR/PT generators flagged with convert_to_storage=True.
    # Must happen before mask is defined so they are included in SOC-tier logic.
    _hydro_convert_gens_to_storage(n, cfg)

    cap_hours = cfg["max_hours"]
    mask = n.storage_units["carrier"] == "hydro"
    if not mask.any():
        return

    # ── Ramp limits on hydro generators (carrier='hydro', not storage_units) ──
    # Only ES ror remains as a generator after conversion. FR/PT are now storage_units.
    hydro_gen_mask = n.generators["carrier"] == "hydro"
    if hydro_gen_mask.any():
        per_country = cfg.get("per_country", {})
        default_ramp = cfg.get("ramp_limit_pu", None)
        gen_countries = _gen_countries(n)
        for country, override in per_country.items():
            cmask = hydro_gen_mask & (gen_countries == country)
            if not cmask.any():
                continue
            ramp = override.get("ramp_limit_pu", default_ramp)
            if ramp is not None:
                n.generators.loc[cmask, "ramp_limit_up"]   = ramp
                n.generators.loc[cmask, "ramp_limit_down"]  = ramp
                log.info(
                    "Hydro gen [%s]: ramp_limit=%.2f → %d units",
                    country, ramp, cmask.sum(),
                )
            p_max = override.get("p_max_pu", None)
            if p_max is not None:
                n.generators.loc[cmask, "p_max_pu"] = float(p_max)
                log.info(
                    "Hydro gen [%s]: p_max_pu=%.2f → effective cap %.0f MW",
                    country, p_max, n.generators.loc[cmask, "p_nom"].sum() * p_max,
                )
        # Apply default ramp to any remaining hydro generators
        remaining = hydro_gen_mask & ~gen_countries.isin(list(per_country.keys()))
        if remaining.any() and default_ramp is not None:
            n.generators.loc[remaining, "ramp_limit_up"]   = default_ramp
            n.generators.loc[remaining, "ramp_limit_down"]  = default_ramp

    # Scale ES hydro capacity to match REE 2024 installed figures
    scaler = cfg.get("capacity_scaler", 1.0)
    if scaler is not None and scaler != 1.0:
        es_hydro_su = mask & n.storage_units["bus"].str.startswith("ES")
        if es_hydro_su.any():
            before = n.storage_units.loc[es_hydro_su, "p_nom"].sum()
            n.storage_units.loc[es_hydro_su, "p_nom"] *= scaler
            after = n.storage_units.loc[es_hydro_su, "p_nom"].sum()
            log.info(
                "Hydro: capacity_scaler=%.3f → ES reservoir %.0f → %.0f MW (-%.0f MW)",
                scaler, before, after, before - after,
            )
        ror_mask = (n.generators["carrier"] == "ror") & n.generators["bus"].str.startswith("ES")
        if ror_mask.any():
            # Use ror_capacity_scaler if set; otherwise fall back to capacity_scaler.
            # ERA5 p_max_pu CFs for ES ror are ~24% vs real 44% — p_nom is scaled up to
            # compensate since the underlying ERA5 profiles can't easily be recalibrated.
            ror_scaler = cfg.get("ror_capacity_scaler", scaler)
            before_r = n.generators.loc[ror_mask, "p_nom"].sum()
            # Undo the capacity_scaler already applied above, then apply ror_scaler
            # (capacity_scaler block only touched StorageUnits; ror Generators are separate)
            n.generators.loc[ror_mask, "p_nom"] *= ror_scaler
            after_r = n.generators.loc[ror_mask, "p_nom"].sum()
            log.info(
                "Hydro: ror_capacity_scaler=%.3f → ES ror %.0f → %.0f MW (+%.0f MW)",
                ror_scaler, before_r, after_r, after_r - before_r,
            )

    # Cap only reservoirs ABOVE the limit. Never inflate small pondage units
    # (some have max_hours < 10 — forcing them to 1200 would invent fictitious storage).
    current_mh = n.storage_units.loc[mask, "max_hours"].copy()
    to_cap = mask & (n.storage_units["max_hours"] > cap_hours)
    to_keep = mask & (n.storage_units["max_hours"] <= cap_hours)

    if to_cap.any():
        n.storage_units.loc[to_cap, "max_hours"] = cap_hours
        log.info(
            "Hydro: capped %d large reservoirs at %d h (max was %.0f h, "
            "total capped e-capacity %.0f GWh → %.0f GWh)",
            to_cap.sum(), cap_hours, current_mh[to_cap].max(),
            (n.storage_units.loc[to_cap, "p_nom"] * current_mh[to_cap]).sum() / 1000,
            (n.storage_units.loc[to_cap, "p_nom"] * cap_hours).sum() / 1000,
        )
    if to_keep.any():
        log.info(
            "Hydro: kept %d smaller reservoirs at original max_hours (range %.0f–%.0f h)",
            to_keep.sum(), current_mh[to_keep].min(), current_mh[to_keep].max(),
        )

    # Disable cyclic constraint so state_of_charge_initial is actually enforced.
    # (PyPSA-EUR ships with cyclic=True, which silently ignores the initial value.)
    n.storage_units.loc[mask, "cyclic_state_of_charge"] = False

    # Override efficiency_dispatch if set in config (base network ships with 0.9).
    eff = cfg.get("efficiency_dispatch", None)
    if eff is not None:
        n.storage_units.loc[mask, "efficiency_dispatch"] = float(eff)
        log.info("Hydro: efficiency_dispatch → %.2f for %d reservoirs", float(eff), mask.sum())

    # Small spill_cost activates PyPSA's spill variable so the LP prices implicit overflow.
    # Without it, pondage units (max_hours 3-10h) shed inflow silently — spill shows up in
    # post-solve diagnostics but is invisible to the optimiser.  1 €/MWh is below any MC
    # tier so it does not distort dispatch merit order.
    sc = cfg.get("spill_cost", None)
    if sc is not None and float(sc) > 0:
        n.storage_units.loc[mask, "spill_cost"] = float(sc)
        log.info("Hydro: spill_cost → %.2f €/MWh for %d reservoirs", float(sc), mask.sum())

    # Zero inflow for p_nom=0 units — ERA5 base-network artifacts with no dispatch capacity.
    # Example: ES0 27 hydro has p_nom=0 in PyPSA-Eur (zero-capacity placeholder at the PT border
    # bus) but retains a non-zero ERA5 inflow series (3.057 TWh/yr). Without this guard, all
    # inflow spills — wasting 42% of total ES hydro inflow as phantom spill.
    _inflow_t = n.storage_units_t.get("inflow", pd.DataFrame())
    if not _inflow_t.empty:
        zero_cap = mask & (n.storage_units["p_nom"] < 1.0)
        cols_to_zero = [u for u in n.storage_units.index[zero_cap] if u in _inflow_t.columns]
        if cols_to_zero:
            n.storage_units_t.inflow[cols_to_zero] = 0.0
            log.info(
                "Hydro: zeroed inflow on %d p_nom<1MW units to prevent phantom spill: %s",
                len(cols_to_zero), cols_to_zero,
            )

    # ── Daisy-chain inflow redistribution ──────────────────────────────────
    # Redirect surplus inflow from small oversupplied units (pondage with huge
    # ERA5 inflow) to larger downstream reservoirs with spare turbine capacity.
    #
    # Root cause: ERA5 distributes runoff proportionally by p_nom within each
    # country (add_electricity.py:777).  Small pondage units get far more
    # inflow-per-MW than large reservoirs — ES0 36 (42 MW) gets 1,564 GWh/yr
    # inflow while ES0 37 (1,354 MW) gets only 16 GWh/yr.
    #
    # The LP then spills the excess because the turbine is 36× undersized.
    # This function keeps enough inflow for the source to run at ~90% CF and
    # redirects the rest to a geographically-close target with spare capacity.
    _redistribute_hydro_inflow(n, cfg, mask)

    # Set initial SOC based on (now updated) max_hours
    e_cap = n.storage_units.loc[mask, "p_nom"] * n.storage_units.loc[mask, "max_hours"]
    n.storage_units.loc[mask, "state_of_charge_initial"] = _resolve_initial_soc(n, cfg) * e_cap

    # Set flat fallback MC — overwritten by apply_inflow_based_hydro_mc().
    mc_f = float(cfg.get("marginal_cost", 15.0))
    n.storage_units.loc[mask, "marginal_cost"] = mc_f

    # Flat MC for hydro generators (FR/PT reservoir gen + ES ror).
    # Generators have no SOC — use the global marginal_cost_gen floor.
    mc_gen_default = cfg.get("marginal_cost_gen", None)
    if mc_gen_default is not None and hydro_gen_mask.any():
        n.generators.loc[hydro_gen_mask, "marginal_cost"] = float(mc_gen_default)
        log.info(
            "Hydro: marginal_cost_gen=%.1f applied to %d generators (FR/PT/ES ror)",
            float(mc_gen_default), hydro_gen_mask.sum(),
        )

    # Per-country MC overrides for hydro generators.
    # Applied after the global marginal_cost_gen floor so country overrides win.
    if hydro_gen_mask.any():
        gen_cntry_mc = _gen_countries(n)
        for ctry, ctry_cfg in cfg.get("per_country", {}).items():
            ctry_mask = hydro_gen_mask & (gen_cntry_mc == ctry)
            if not ctry_mask.any():
                continue
            use_flex = ctry_cfg.get("use_flexible_mc", False)
            if use_flex:
                flex_mc = float(ctry_cfg.get("flexible_mc", mc_gen_default or 6.0))
                n.generators.loc[ctry_mask, "marginal_cost"] = flex_mc
                log.info(
                    "Hydro gen [%s]: MC=%.1f (flexible_mc mode, overrides marginal_cost_gen=%.1f)",
                    ctry, flex_mc, mc_gen_default or 6.0,
                )

    # Scale inflow time series if present
    inflow_t = n.storage_units_t.get("inflow", pd.DataFrame())
    if not inflow_t.empty:
        hydro_cols = [c for c in inflow_t.columns if c in mask.index[mask]]
        if hydro_cols:
            n.storage_units_t.inflow[hydro_cols] *= cfg["inflow_multiplier"]
            log.info("Hydro: inflow × %.2f applied to %d reservoirs", cfg["inflow_multiplier"], len(hydro_cols))


# ─── 5b. FR/PT run-of-river generators ───────────────────────────────────────

def _apply_fr_pt_ror(n, config):
    """Add FR and PT run-of-river generators from hourly dispatch CSV.

    The base PyPSA-Eur 50-node network has NO ror generators for France or Portugal.
    PyPSA-Eur's 50-node aggregation collapsed all French/Portuguese hydro into two
    aggregate 'hydro' generators per country, losing the RoR component entirely.

    Real FR RoR = 1,985 MW mean (17.4 TWh/yr); real PT RoR = 413 MW mean (3.6 TWh/yr).
    Without these, FR/PT generation is ~21 TWh/yr short, forcing more CCGT dispatch
    and distorting the ES↔FR/PT price differential and interconnector flows.

    Source CSV: data_ES/hydro/generation_ror_hourly_2024.csv
      Columns: FR_WEST_hydro, FR_EAST_hydro, PT_NORTH_hydro, PT_CENTRE_hydro, PT_SOUTH_hydro
      Values in MW (actual hourly dispatch); buses mapped by stripping '_hydro' suffix.

    Implementation:
      p_nom = max(hourly series) per corridor (rated capacity)
      p_max_pu = actual_dispatch / p_nom (time-varying availability)
      marginal_cost = hydro.marginal_cost_gen (default 6 €/MWh — same as ES ror)

    Idempotent: skips if FR/PT ror generators already present.
    """
    cfg = config.get("fr_pt_ror", {})
    if not cfg.get("enabled", True):
        log.info("FR/PT RoR: disabled in config — skipping")
        return

    # Idempotency: check for existing FR/PT ror generators
    existing_frpt_ror = n.generators[
        (n.generators["carrier"] == "ror") &
        (n.generators["bus"].str.startswith(("FR", "PT")))
    ]
    if not existing_frpt_ror.empty:
        log.info("FR/PT RoR: already present (%d generators) — skipping", len(existing_frpt_ror))
        return

    # Locate CSV
    csv_rel = cfg.get("csv_path", "data_ES/hydro/generation_ror_hourly_2024.csv")
    csv_path = Path(csv_rel)
    if not csv_path.exists():
        csv_path = Path(__file__).parent.parent / csv_rel
    if not csv_path.exists():
        log.warning("FR/PT RoR: CSV not found at %s — skipping", csv_rel)
        return

    ror_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    # Align index to network snapshots (strip timezone for matching)
    snap_naive = n.snapshots.tz_localize(None) if n.snapshots.tz is not None else n.snapshots
    ror_naive = pd.DatetimeIndex([
        pd.Timestamp(t).tz_localize(None) if pd.Timestamp(t).tzinfo else pd.Timestamp(t)
        for t in ror_df.index
    ])
    ror_df = ror_df.copy()
    ror_df.index = ror_naive
    ror_aligned = ror_df.reindex(snap_naive, method="ffill").bfill().fillna(0.0)

    mc_ror = float(config.get("hydro", {}).get("marginal_cost_gen", 6.0))

    if "ror" not in n.carriers.index:
        n.add("Carrier", "ror", nice_name="Run of River", color="#4FC3F7")

    added = []
    for col in ror_aligned.columns:
        bus = col.replace("_hydro", "")
        if bus not in n.buses.index:
            log.debug("FR/PT RoR: bus '%s' not in network — skipping column %s", bus, col)
            continue
        series = ror_aligned[col]
        p_nom = float(series.max())
        if p_nom < 1.0:
            continue
        p_max_pu = (series / p_nom).clip(0.0, 1.0)
        gen_name = f"{bus}_ror"
        n.add("Generator", gen_name,
              bus=bus,
              carrier="ror",
              p_nom=p_nom,
              marginal_cost=mc_ror,
              p_min_pu=0.0)
        if len(n.generators_t.p_max_pu) == 0:
            n.generators_t["p_max_pu"] = pd.DataFrame(index=n.snapshots)
        n.generators_t.p_max_pu[gen_name] = p_max_pu.values
        added.append((gen_name, p_nom, float(series.mean())))

    if added:
        total_mean = sum(mw for _, _, mw in added)
        log.info(
            "FR/PT RoR: added %d generators (total mean %.0f MW, ~%.1f TWh/yr) — %s",
            len(added), total_mean, total_mean * len(snap_naive) / 1e6,
            "  ".join(f"{nm}({pn:.0f}MW cap,{mw:.0f}MW mean)" for nm, pn, mw in added),
        )
    else:
        log.warning("FR/PT RoR: no generators added — check CSV column names vs bus names")


# ─── 5d. Inflow-based hydro water values ─────────────────────────────────────

def apply_inflow_based_hydro_mc(n, config):
    """Set time-varying marginal costs for reservoir hydro driven by inflow data.

    Replaces soc_mc_tiers and monthly_mc with a log-space inverse mapping:
      high inflow → low MC (dispatch freely)
      low  inflow → high MC (defend water)

    ES uses ERA5 hourly inflow from n.storage_units_t.inflow (29 reservoirs).
    FR/PT use monthly river-discharge proxies (Loire/Tagus) from a CSV, expanded
    to hourly by forward-filling each month's value.  Log normalization is
    scale-invariant so only the seasonal shape matters.

    Called from run_validation.py after apply_non_linear_refinements().
    """
    cfg = config.get("hydro", {})
    inflow_mc_cfg = cfg.get("inflow_mc", {})
    if not inflow_mc_cfg.get("enabled", False):
        log.info("Hydro inflow MC: disabled — skipping")
        return n

    country_cfgs = inflow_mc_cfg.get("countries", {
        "ES": {"window_days":  7, "mc_min": 15.0, "mc_max": 80.0},
        "FR": {"window_days": 28, "mc_min": 20.0, "mc_max": 75.0},
        "PT": {"window_days": 14, "mc_min": 20.0, "mc_max": 80.0},
    })

    # Load river-proxy CSV for FR/PT (monthly discharge → hourly forward-fill).
    proxy_csv_rel = inflow_mc_cfg.get("river_proxy_csv", "")
    proxy_df: pd.DataFrame = pd.DataFrame()
    if proxy_csv_rel:
        proxy_path = Path(__file__).parent.parent / proxy_csv_rel
        if proxy_path.exists():
            raw = pd.read_csv(proxy_path)
            _month_map = {m: i + 1 for i, m in enumerate([
                "January","February","March","April","May","June",
                "July","August","September","October","November","December",
            ])}
            raw_months = raw.iloc[:, 0].map(_month_map)
            proxy_df = pd.DataFrame(index=raw_months)
            for col_i in range(1, len(raw.columns)):
                vals = pd.to_numeric(
                    raw.iloc[:, col_i].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
                proxy_df[col_i] = vals.values
            log.info("Hydro inflow MC: loaded river proxy CSV %s (%d months, %d columns)",
                     proxy_path.name, len(proxy_df), len(proxy_df.columns))
        else:
            log.warning("Hydro inflow MC: river_proxy_csv not found at %s", proxy_path)

    mask = n.storage_units["carrier"] == "hydro"
    inflow_t = n.storage_units_t.get("inflow", pd.DataFrame())
    snapshots = n.snapshots
    snap_months = snapshots.to_series().dt.month

    # Ensure time-varying marginal_cost DataFrame exists
    if "marginal_cost" not in n.storage_units_t or n.storage_units_t.marginal_cost.empty:
        n.storage_units_t["marginal_cost"] = pd.DataFrame(index=snapshots)

    for country, ccfg in country_cfgs.items():
        mc_min     = float(ccfg["mc_min"])
        mc_max     = float(ccfg["mc_max"])
        window_h   = int(ccfg["window_days"]) * 24
        proxy_col  = ccfg.get("river_proxy_col")

        cmask = mask & n.storage_units["bus"].str.startswith(country)
        reservoirs = n.storage_units.index[cmask].tolist()
        if not reservoirs:
            log.info("Hydro inflow MC [%s]: no reservoirs — skipping", country)
            continue

        # ── Build inflow proxy series ────────────────────────────────────────
        if proxy_col is not None and not proxy_df.empty and proxy_col in proxy_df.columns:
            # River proxy: expand monthly discharge to hourly by forward-fill
            monthly_vals = proxy_df[proxy_col].clip(lower=1.0)
            inflow_proxy = snap_months.map(monthly_vals.to_dict()).astype(float).clip(lower=1.0)
            inflow_proxy.index = snapshots
            source = f"river proxy col={proxy_col}"
        else:
            # ERA5 reservoir inflow: sum across all country reservoirs
            res_with_inflow = [r for r in reservoirs if r in inflow_t.columns]
            if not res_with_inflow:
                log.warning("Hydro inflow MC [%s]: no inflow data and no river proxy — skipping", country)
                continue
            inflow_proxy = inflow_t[res_with_inflow].sum(axis=1).clip(lower=1.0)
            source = f"ERA5 sum ({len(res_with_inflow)} reservoirs)"

        # ── Smooth (trailing rolling mean, min_periods=1 handles warmup) ────
        smooth = inflow_proxy.rolling(window=window_h, min_periods=1).mean().clip(lower=1.0)

        # ── Log-space normalization ──────────────────────────────────────────
        p10 = float(smooth.quantile(0.10))
        p90 = float(smooth.quantile(0.90))
        p10 = max(p10, 1.0)
        p90 = max(p90, p10 + 1e-6)

        log_range = np.log(p90) - np.log(p10)
        norm = ((np.log(smooth) - np.log(p10)) / log_range).clip(0.0, 1.0)
        mc_series = mc_max - (mc_max - mc_min) * norm

        # Align to snapshots (handles tz-aware edge cases)
        mc_series = mc_series.reindex(snapshots).ffill().bfill()

        # ── Write time-varying MC ────────────────────────────────────────────
        res_with_inflow_col = [r for r in reservoirs if r in inflow_t.columns]
        for res in res_with_inflow_col:
            n.storage_units_t.marginal_cost[res] = mc_series.values
            n.storage_units.loc[res, "marginal_cost"] = 0.0

        # Reservoirs without inflow columns (shouldn't exist for ES; possible for FR/PT
        # if only proxy is used): apply flat mean MC
        for res in reservoirs:
            if res not in res_with_inflow_col:
                n.storage_units.loc[res, "marginal_cost"] = float(mc_series.mean())

        # If using river proxy, ALL reservoirs get the time-varying series
        if proxy_col is not None:
            for res in reservoirs:
                n.storage_units_t.marginal_cost[res] = mc_series.values
                n.storage_units.loc[res, "marginal_cost"] = 0.0

        # ── Sanity checks ────────────────────────────────────────────────────
        mc_arr   = mc_series.values
        inf_arr  = smooth.reindex(snapshots).ffill().bfill().values
        corr     = float(np.corrcoef(inf_arr, mc_arr)[0, 1]) if len(inf_arr) > 1 else 0.0
        h_floor  = int((mc_arr <= mc_min + 0.5).sum())
        h_ceil   = int((mc_arr >= mc_max - 0.5).sum())
        monthly_mean = {
            m: round(float(mc_series[snap_months == m].mean()), 1)
            for m in range(1, 13)
            if int((snap_months == m).sum()) > 0
        }
        log.info(
            "Hydro inflow MC [%s] source=%s: "
            "inflow p10/p50/p90=%.0f/%.0f/%.0f; "
            "MC p10/p50/p90=%.1f/%.1f/%.1f €/MWh; "
            "corr=%.3f; floor=%dh ceil=%dh; "
            "monthly=%s",
            country, source,
            float(np.percentile(inf_arr, 10)),
            float(np.percentile(inf_arr, 50)),
            float(np.percentile(inf_arr, 90)),
            float(np.percentile(mc_arr, 10)),
            float(np.percentile(mc_arr, 50)),
            float(np.percentile(mc_arr, 90)),
            corr, h_floor, h_ceil,
            monthly_mean,
        )

    return n


# ─── 6. PHS operational friction ─────────────────────────────────────────────

def _apply_phs(n, config):
    cfg = config.get("phs", {})
    mask = n.storage_units["carrier"] == "PHS"
    if not mask.any():
        log.info("PHS: no PHS storage units found — skipping")
        return

    mc = cfg.get("marginal_cost")
    if mc is not None:
        n.storage_units.loc[mask, "marginal_cost"] = mc
        log.info("PHS: marginal_cost=€%.2f/MWh on %d units", mc, mask.sum())

    mc_store = cfg.get("marginal_cost_storage")
    if mc_store is not None:
        n.storage_units.loc[mask, "marginal_cost_storage"] = mc_store
        log.info("PHS: marginal_cost_storage=€%.2f/MWh on %d units", mc_store, mask.sum())

    p_max = cfg.get("p_max_pu")
    if p_max is not None:
        n.storage_units.loc[mask, "p_max_pu"] = p_max
        log.info("PHS: p_max_pu=%.2f (caps dispatch+store to %.0f%% p_nom) on %d units",
                 p_max, p_max * 100, mask.sum())

    log.info(
        "PHS: total %.0f MW constrained (MC dispatch=€%.2f, store=€%.2f, p_max_pu=%.2f)",
        n.storage_units.loc[mask, "p_nom"].sum(),
        cfg.get("marginal_cost", 0.0),
        cfg.get("marginal_cost_storage", 0.0),
        cfg.get("p_max_pu", 1.0),
    )


# ─── 7. Border restoration ────────────────────────────────────────────────────

def _restore_borders(n, config):
    borders = {k: v for k, v in config["borders"].items() if k != "ic_factor"}
    ic_factor = config["borders"].get("ic_factor", 1.0)

    changed = 0
    for link_name, p_nom in borders.items():
        if link_name not in n.links.index:
            log.warning("Border link '%s' not found — skipping", link_name)
            continue
        if not np.isclose(n.links.loc[link_name, "p_nom"], p_nom):
            n.links.loc[link_name, "p_nom"] = p_nom
            changed += 1

    if changed:
        log.info("Borders: restored %d link p_nom values to open-border NTC", changed)
    else:
        log.info("Borders: all p_nom already at open-border values — no change")

    # Apply interconnector shrink factor AFTER restoring NTC values.
    # ic_factor < 1.0 simulates missing export routes (e.g. Italy/Germany for France)
    # without altering the generation fleet.
    if ic_factor is not None and 0 < ic_factor < 1:
        n_links = 0
        total_before = 0.0
        total_after = 0.0
        for link_name in borders:
            if link_name not in n.links.index:
                continue
            total_before += n.links.loc[link_name, "p_nom"]
            n.links.loc[link_name, "p_nom"] *= ic_factor
            total_after += n.links.loc[link_name, "p_nom"]
            n_links += 1
        log.info(
            "Borders: ic_factor=%.2f applied to %d links "
            "(total %.0f → %.0f MW, -%.0f MW)",
            ic_factor, n_links, total_before, total_after,
            total_before - total_after,
        )


# ─── 8. Cross-border AC/DC topology ──────────────────────────────────────────

def _apply_border_ac_dc(n, config):
    """Replace DC_ic cross-border Link pairs with physically accurate AC Lines + INELFE DC Link.

    Physical reality:
      ES–PT (all 3 corridors): 400 kV AC overhead lines → PyPSA Line, carrier='AC'
      ES–FR western axis:      400 kV + 220 kV AC overhead → PyPSA Line, carrier='AC'
      ES–FR eastern Vic–Baixas: 400 kV AC single circuit → PyPSA Line, carrier='AC'
      INELFE (ES0 43 ↔ FR_EAST): two ±320 kV VSC HVDC cables → PyPSA Link, carrier='DC'

    x values (Ω) are calibrated to the internal ES line convention from PyPSA-Eur:
      x = 0.35 rad × V_nom² / s_nom  (at rated capacity the angle spread ≈ 20°).
    ic_factor from config["borders"] scales all s_nom and p_nom uniformly.
    """
    ic_factor = config.get("borders", {}).get("ic_factor", 1.0)

    # ── 1. Remove existing DC_ic Link pairs for PT and FR borders ────────────
    dc_ic_names = [
        "PT_NORTH export", "PT_NORTH import",
        "PT_CENTRE export", "PT_CENTRE import",
        "PT_SOUTH export",  "PT_SOUTH import",
        "FR_WEST export",   "FR_WEST import",
        "FR_EAST export",   "FR_EAST import",
    ]
    removed = []
    for name in dc_ic_names:
        if name in n.links.index:
            n.remove("Link", name)
            removed.append(name)
    if removed:
        log.info("Border AC/DC: removed %d DC_ic links (%s...)", len(removed), removed[0])
    else:
        log.info("Border AC/DC: no DC_ic links found to remove (already converted)")

    # ── 2. Add AC Lines for PT and FR western + FR eastern AC component ──────
    ac_cfg = config.get("border_ac_lines", {})
    added_lines = []
    for line_name, params in ac_cfg.items():
        if line_name in n.lines.index:
            log.debug("Border AC/DC: AC Line '%s' already exists — skipping add", line_name)
            continue
        bus0   = params["bus0"]
        bus1   = params["bus1"]
        s_nom  = params["s_nom"] * ic_factor
        x      = params["x"]
        r      = params.get("r", round(x * 0.12, 2))
        marginal_cost = params.get("marginal_cost", 0.0)
        n.add("Line", line_name,
              bus0=bus0, bus1=bus1,
              s_nom=s_nom, x=x, r=r,
              carrier="AC",
              marginal_cost=marginal_cost)
        added_lines.append(
            f"{line_name}({s_nom:.0f}MW,x={x}Ω,mc={marginal_cost:.1f}€/MWh)"
            if marginal_cost
            else f"{line_name}({s_nom:.0f}MW,x={x}Ω)"
        )
    if added_lines:
        log.info("Border AC/DC: added %d AC Lines: %s", len(added_lines), ", ".join(added_lines))

    # ── 3. Add INELFE HVDC bidirectional DC Link ─────────────────────────────
    inelfe = config.get("inelfe", {})
    if not inelfe.get("enabled", False):
        log.info("Border AC/DC: INELFE disabled in config — skipping")
        return

    if "INELFE" in n.links.index:
        log.info("Border AC/DC: INELFE link already present — skipping add")
        return

    p_nom = inelfe["p_nom"] * ic_factor
    marginal_cost = inelfe.get("marginal_cost", 0.0)
    n.add("Link", "INELFE",
          bus0=inelfe["bus0"],
          bus1=inelfe["bus1"],
          p_nom=p_nom,
          efficiency=inelfe["efficiency"],
          p_min_pu=inelfe["p_min_pu"],
          marginal_cost=marginal_cost,
          carrier="DC")
    log.info(
        "Border AC/DC: added INELFE DC Link %.0f MW (eff=%.2f, p_min_pu=%.1f, "
        "marginal_cost=%.1f) %s ↔ %s",
        p_nom, inelfe["efficiency"], inelfe["p_min_pu"],
        marginal_cost,
        inelfe["bus0"], inelfe["bus1"],
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    ac_total = sum(
        config["border_ac_lines"][k]["s_nom"] * ic_factor
        for k in config.get("border_ac_lines", {})
    )
    log.info(
        "Border AC/DC: total AC Line capacity %.0f MW + INELFE %.0f MW = %.0f MW "
        "(ic_factor=%.2f)",
        ac_total, p_nom, ac_total + p_nom, ic_factor,
    )



# ─── 9. Transmission limits ───────────────────────────────────────────────────

def _apply_transmission(n, config):
    cfg = config["transmission"]
    es_lines = n.lines.index[
        n.lines["bus0"].str.startswith("ES") & n.lines["bus1"].str.startswith("ES")
    ]
    if len(es_lines) == 0:
        log.warning("Transmission: no internal ES lines found")
        return

    # 1. Scale s_nom by trans_factor (reduces thermal capacity → more congestion)
    tf = cfg["trans_factor"]
    if tf is not None and 0 < tf < 1:
        original_sum = n.lines.loc[es_lines, "s_nom"].sum()
        n.lines.loc[es_lines, "s_nom"] *= tf
        new_sum = n.lines.loc[es_lines, "s_nom"].sum()
        log.info(
            "Transmission: trans_factor=%.2f applied to %d ES lines "
            "(s_nom %.0f → %.0f MVA, -%.0f MVA)",
            tf, len(es_lines), original_sum, new_sum, original_sum - new_sum,
        )
    else:
        log.info("Transmission: trans_factor=%.2f — no scaling applied", tf)

    # 2. Set s_max_pu thermal limit
    n.lines.loc[es_lines, "s_max_pu"] = cfg["s_max_pu"]
    log.info(
        "Transmission: s_max_pu=%.2f set on %d internal ES lines",
        cfg["s_max_pu"], len(es_lines),
    )

    # 3. HVDC cable losses — applies only to carrier=="DC" links.
    # This is the Balearic Islands cable (ES0 5 → ES1 0, 400 MW).
    # FR/PT interconnectors use carrier "DC_ic export/import" and represent AC
    # overhead lines modelled with directional capacity bounds — they are excluded.
    dc_eff = cfg.get("dc_loss_efficiency")
    if dc_eff is not None and 0 < dc_eff < 1:
        dc_mask = n.links["carrier"] == "DC"
        dc_idx  = n.links.index[dc_mask]
        if dc_idx.empty:
            log.info("Transmission: no carrier='DC' links found — skipping DC loss")
        else:
            n.links.loc[dc_idx, "efficiency"] = dc_eff
            log.info(
                "Transmission: DC loss efficiency=%.3f applied to %d HVDC link(s): %s",
                dc_eff, len(dc_idx), list(dc_idx),
            )


# ─── 8b. Wind availability scaler ────────────────────────────────────────────

def _apply_wind_availability(n, config):
    """Scale onwind p_nom by per-country factors.

    ERA5 2023 cutout overestimates FR/PT wind speeds for Feb-Apr 2024.
    Scaling p_nom is equivalent to scaling p_max_pu (dispatch = p_nom × p_max_pu × pu)
    but is more physically intuitive ("less installed capacity") and avoids
    in-place time-series mutation edge cases.

    Idempotent: checks for a marker attribute before applying.
    """
    cfg = config.get("wind_availability", {})
    per_country = cfg.get("per_country", {})
    if not per_country:
        return

    # Idempotency marker — stored as int (not bool) so PyPSA's NetCDF export
    # doesn't trip over the b1 dtype that NetCDF4 refuses to serialise.
    if getattr(n, "_wind_avail_applied", 0):
        log.info("Wind availability: already applied — skipping")
        return

    onwind_mask = n.generators["carrier"] == "onwind"
    if not onwind_mask.any():
        log.warning("Wind availability: no onwind generators found — skipping")
        return

    gen_countries = _gen_countries(n)
    total_affected = 0
    for country, factor in per_country.items():
        cmask = onwind_mask & (gen_countries == country)
        if not cmask.any():
            log.warning("Wind availability [%s]: no onwind generators found", country)
            continue
        gens = cmask.index[cmask.values]   # .values → numpy bool array, safe for Index indexing
        before = n.generators.loc[gens, "p_nom"].sum()
        n.generators.loc[gens, "p_nom"] *= factor
        after = n.generators.loc[gens, "p_nom"].sum()
        total_affected += len(gens)
        log.info(
            "Wind availability [%s]: p_nom x %.2f → %d generators "
            "(%.0f → %.0f MW, -%.0f MW)",
            country, factor, len(gens), before, after, before - after,
        )

    if total_affected:
        n._wind_avail_applied = 1   # int, not bool — NetCDF4 rejects b1 dtype


# ─── 9. VOLL / load shedding ─────────────────────────────────────────────────

def _apply_voll(n, config):
    voll = config.get("voll")
    if voll is None:
        log.info("VOLL: disabled (config['voll']=None) — skipping")
        return
    if (n.generators["carrier"] == "load_shedding").any():
        log.info("VOLL: load-shedding generators already present — skipping")
        return
    if "load_shedding" not in n.carriers.index:
        n.add("Carrier", "load_shedding", nice_name="Load Shedding", color="#FF00FF")
    model_buses = n.buses.index[
        (
            n.buses.index.str.startswith("ES") |
            n.buses.index.str.startswith("FR") |
            n.buses.index.str.startswith("PT")
        ) &
        ~n.buses.index.str.contains("battery|H2", regex=True)
    ]
    for bus in model_buses:
        n.add(
            "Generator",
            f"VOLL_{bus}",
            bus=bus,
            carrier="load_shedding",
            p_nom=10_000.0,
            marginal_cost=voll,
        )
    log.info("VOLL: added load-shedding at %d buses (ES/FR/PT) (€%.0f/MWh)", len(model_buses), voll)


# ─── 10. Lock capacities (dispatch-only solve) [renumbered from 11] ──────────


def _lock_capacities(n):
    """Set all extendable flags to False so n.optimize() is dispatch-only.

    PyPSA-EUR networks ship with p_nom_extendable=True on generators (VRE, H2,
    batteries, etc.) and links. Without locking, Gurobi would expand them —
    turning a dispatch validation into an investment run.
    """
    total = 0
    for df in (n.generators, n.lines, n.links, n.storage_units, n.stores):
        for col in ("p_nom_extendable", "s_nom_extendable", "e_nom_extendable"):
            if col in df.columns and df[col].any():
                total += int(df[col].sum())
                df[col] = False
    if total:
        log.info("Capacities locked: disabled expansion on %d components → dispatch-only solve", total)
    else:
        log.info("Capacities locked: all components already non-extendable")
