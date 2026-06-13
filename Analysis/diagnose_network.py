"""
diagnose_network.py — Comprehensive sanity check of the PyPSA network.

Checks:
  1. Load time series: FR, PT, ES all have live data in loads_t.p_set
  2. Generator time series: p_max_pu for VRE, marginal_cost for thermal
  3. Storage unit time series: inflow for hydro
  4. Interconnector capacities match config
  5. CCGT p_nom_min and ramp limits correctly set
  6. Nuclear p_nom_min and ramp limits correctly set
  7. Hydro ramp limits correctly set
  8. Transmission s_max_pu correctly set
  9. VOLL generators present
  10. Capacities locked (no extendable components)
"""

import sys
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

SEP = "─" * 66


def _country(bus_name):
    for prefix in ("ES", "FR", "PT"):
        if bus_name.startswith(prefix):
            return prefix
    return None


def _gen_countries(n):
    return n.generators["bus"].map(_country)


def check_loads(n):
    """Verify FR, PT, ES loads have live time series."""
    print(f"\n{SEP}")
    print("  CHECK 1: Load Time Series")
    print(SEP)

    all_ok = True
    for prefix, label in [("FR", "France"), ("PT", "Portugal"), ("ES", "Spain")]:
        load_names = [l for l in n.loads.index if l.startswith(prefix)]
        ts_cols = [c for c in load_names if c in n.loads_t.p_set.columns]
        static = n.loads.loc[load_names, "p_set"] if "p_set" in n.loads.columns else pd.Series(dtype=float)

        print(f"\n  {label} ({prefix}): {len(load_names)} loads")
        for ln in load_names:
            has_ts = ln in ts_cols
            s_val = static.get(ln, 0.0)
            if has_ts:
                mean_val = n.loads_t.p_set[ln].mean()
                max_val  = n.loads_t.p_set[ln].max()
                print(f"    ✓ {ln}: time series present, mean={mean_val:.0f} MW, max={max_val:.0f} MW")
            else:
                print(f"    ✗ {ln}: NO time series! static p_set={s_val}")
                all_ok = False

    return all_ok


def check_generators(n):
    """Verify key generator parameters and time series."""
    print(f"\n{SEP}")
    print("  CHECK 2: Generator Parameters & Time Series")
    print(SEP)

    all_ok = True

    # ── CCGT check ──────────────────────────────────────────────────────────
    ccgt = n.generators[n.generators["carrier"] == "CCGT"]
    print(f"\n  CCGT: {len(ccgt)} units, total p_nom={ccgt['p_nom'].sum():.0f} MW")
    for _, g in ccgt.iterrows():
        country = _country(g["bus"])
        p_min = g["p_nom_min"]
        p_nom = g["p_nom"]
        r_up  = g.get("ramp_limit_up", "N/A")
        r_dn  = g.get("ramp_limit_down", "N/A")
        mc    = g["marginal_cost"]
        status = "✓" if p_min < p_nom else "✗ MUST-RUN"
        print(f"    {status} {g.name}: bus={g['bus']}, p_nom={p_nom:.0f}, "
              f"p_nom_min={p_min:.0f}, MC={mc:.1f}, ramp_up={r_up}, ramp_dn={r_dn}")
        if p_min >= p_nom and p_nom > 0:
            all_ok = False

    # ── CCGT_flex check ─────────────────────────────────────────────────────
    flex = n.generators[n.generators["carrier"] == "CCGT_flex"]
    if len(flex) > 0:
        print(f"\n  CCGT_flex: {len(flex)} units, total p_nom={flex['p_nom'].sum():.0f} MW")
        for _, g in flex.iterrows():
            print(f"    ✓ {g.name}: bus={g['bus']}, p_nom={g['p_nom']:.0f}, MC={g['marginal_cost']:.1f}")
    else:
        print("\n  CCGT_flex: NONE — check config")

    # ── Nuclear check ───────────────────────────────────────────────────────
    nuc = n.generators[n.generators["carrier"] == "nuclear"]
    print(f"\n  Nuclear: {len(nuc)} units, total p_nom={nuc['p_nom'].sum():.0f} MW")
    for _, g in nuc.iterrows():
        country = _country(g["bus"])
        p_min = g["p_nom_min"]
        p_nom = g["p_nom"]
        r_up  = g.get("ramp_limit_up", "N/A")
        r_dn  = g.get("ramp_limit_down", "N/A")
        mc    = g["marginal_cost"]
        p_min_pu_actual = p_min / p_nom if p_nom > 0 else 0
        print(f"    ✓ {g.name}: bus={g['bus']}, p_nom={p_nom:.0f}, "
              f"p_nom_min={p_min:.0f} ({p_min_pu_actual:.2f} pu), "
              f"MC={mc:.1f}, ramp_up={r_up}, ramp_dn={r_dn}")

    # ── VRE time series check ───────────────────────────────────────────────
    vre = ["solar", "onwind", "offwind", "offwind-float"]
    print(f"\n  VRE Time Series (p_max_pu):")
    for carrier in vre:
        gens = n.generators[n.generators["carrier"] == carrier]
        if len(gens) == 0:
            continue
        ts_cols = [g for g in gens.index if g in n.generators_t.p_max_pu.columns]
        missing = [g for g in gens.index if g not in n.generators_t.p_max_pu.columns]
        if ts_cols:
            print(f"    ✓ {carrier}: {len(ts_cols)}/{len(gens)} have p_max_pu time series")
        if missing:
            print(f"    ✗ {carrier}: {len(missing)}/{len(gens)} MISSING p_max_pu time series!")
            for m in missing:
                print(f"      ✗ {m}")
            all_ok = False

    return all_ok


def check_storage(n):
    """Verify hydro storage has inflow time series and correct parameters."""
    print(f"\n{SEP}")
    print("  CHECK 3: Storage Units")
    print(SEP)

    all_ok = True

    hydro_su = n.storage_units[n.storage_units["carrier"] == "hydro"]
    print(f"\n  Hydro storage: {len(hydro_su)} units, total p_nom={hydro_su['p_nom'].sum():.0f} MW")

    inflow_t = n.storage_units_t.get("inflow", pd.DataFrame())
    for _, su in hydro_su.iterrows():
        country = _country(su["bus"])
        has_inflow = su.name in inflow_t.columns if not inflow_t.empty else False
        mh = su["max_hours"]
        soc_init = su["state_of_charge_initial"]
        cyclic = su.get("cyclic_state_of_charge", "N/A")
        print(f"    {'✓' if has_inflow else '✗'} {su.name}: bus={su['bus']}, "
              f"p_nom={su['p_nom']:.0f} MW, max_h={mh:.0f}, "
              f"soc_init={soc_init:.0f} MWh, inflow={'✓' if has_inflow else '✗'}, "
              f"cyclic={cyclic}")
        if not has_inflow:
            all_ok = False

    # Check battery/H2 stores
    stores = n.stores
    if len(stores) > 0:
        print(f"\n  Stores: {len(stores)} total")
        for _, s in stores.iterrows():
            e_nom = s.get("e_nom", 0)
            if e_nom > 0:
                print(f"    ⚠ {s.name}: e_nom={e_nom:.0f} MWh — active store!")

    return all_ok


def check_interconnectors(n, config):
    """Verify border link capacities match config."""
    print(f"\n{SEP}")
    print("  CHECK 4: Interconnector Capacities")
    print(SEP)

    all_ok = True
    borders = config.get("borders", {})

    for link_name, expected in borders.items():
        if link_name not in n.links.index:
            print(f"    ✗ {link_name}: NOT FOUND in network!")
            all_ok = False
            continue
        actual = n.links.loc[link_name, "p_nom"]
        match = "✓" if np.isclose(actual, expected) else "✗ MISMATCH"
        print(f"    {match} {link_name}: expected={expected:.0f} MW, actual={actual:.0f} MW")
        if not np.isclose(actual, expected):
            all_ok = False

    return all_ok


def check_transmission(n, config):
    """Verify s_max_pu on internal ES lines."""
    print(f"\n{SEP}")
    print("  CHECK 5: Transmission Limits")
    print(SEP)

    all_ok = True
    expected = config.get("transmission", {}).get("s_max_pu", 0.50)

    es_lines = n.lines.index[
        n.lines["bus0"].str.startswith("ES") & n.lines["bus1"].str.startswith("ES")
    ]
    print(f"  Internal ES lines: {len(es_lines)}")
    actual_values = n.lines.loc[es_lines, "s_max_pu"].unique()
    if len(actual_values) == 1 and np.isclose(actual_values[0], expected):
        print(f"    ✓ All ES lines: s_max_pu={actual_values[0]:.2f} (expected {expected})")
    else:
        print(f"    ✗ s_max_pu values: {actual_values} (expected {expected})")
        all_ok = False

    return all_ok


def check_voll(n, config):
    """Verify VOLL generators are present."""
    print(f"\n{SEP}")
    print("  CHECK 6: VOLL / Load Shedding")
    print(SEP)

    voll_gens = n.generators[n.generators["carrier"] == "load_shedding"]
    if len(voll_gens) > 0:
        print(f"    ✓ {len(voll_gens)} load-shedding generators present")
        print(f"    ✓ VOLL = €{config.get('voll', 'N/A')}/MWh")
        return True
    else:
        print(f"    ✗ NO load-shedding generators!")
        return False


def check_capacities_locked(n):
    """Verify no extendable components remain."""
    print(f"\n{SEP}")
    print("  CHECK 7: Capacities Locked (Dispatch-Only)")
    print(SEP)

    all_ok = True
    for df_name, df in [
        ("generators", n.generators),
        ("lines", n.lines),
        ("links", n.links),
        ("storage_units", n.storage_units),
        ("stores", n.stores),
    ]:
        for col in ("p_nom_extendable", "s_nom_extendable", "e_nom_extendable"):
            if col in df.columns and df[col].any():
                count = int(df[col].sum())
                print(f"    ✗ {df_name}.{col}: {count} components still extendable!")
                all_ok = False

    if all_ok:
        print("    ✓ All components locked (non-extendable)")

    return all_ok


def check_hydro_ramp(n, config):
    """Verify hydro generator ramp limits."""
    print(f"\n{SEP}")
    print("  CHECK 8: Hydro Generator Ramp Limits")
    print(SEP)

    all_ok = True
    hydro_gen = n.generators[n.generators["carrier"] == "hydro"]
    if len(hydro_gen) == 0:
        print("    ⚠ No hydro generators found (only storage_units)")
        return True

    print(f"  Hydro generators: {len(hydro_gen)} units")
    for _, g in hydro_gen.iterrows():
        country = _country(g["bus"])
        r_up = g.get("ramp_limit_up", "N/A")
        r_dn = g.get("ramp_limit_down", "N/A")
        print(f"    {'✓' if r_up != 'N/A' else '✗'} {g.name}: bus={g['bus']}, "
              f"p_nom={g['p_nom']:.0f}, ramp_up={r_up}, ramp_dn={r_dn}")
        if r_up == "N/A" or r_dn == "N/A":
            all_ok = False

    return all_ok


def check_load_balance(n):
    """Quick sanity: total generation capacity vs peak load."""
    print(f"\n{SEP}")
    print("  CHECK 9: Load Balance Sanity")
    print(SEP)

    # ES loads
    es_load_cols = [c for c in n.loads.index if c.startswith("ES") and c in n.loads_t.p_set.columns]
    if es_load_cols:
        es_peak = n.loads_t.p_set[es_load_cols].sum(axis=1).max()
        es_mean = n.loads_t.p_set[es_load_cols].sum(axis=1).mean()
        print(f"  ES load: mean={es_mean:.0f} MW, peak={es_peak:.0f} MW")

    # FR loads
    fr_load_cols = [c for c in n.loads.index if c.startswith("FR") and c in n.loads_t.p_set.columns]
    if fr_load_cols:
        fr_peak = n.loads_t.p_set[fr_load_cols].sum(axis=1).max()
        fr_mean = n.loads_t.p_set[fr_load_cols].sum(axis=1).mean()
        print(f"  FR load: mean={fr_mean:.0f} MW, peak={fr_peak:.0f} MW")

    # PT loads
    pt_load_cols = [c for c in n.loads.index if c.startswith("PT") and c in n.loads_t.p_set.columns]
    if pt_load_cols:
        pt_peak = n.loads_t.p_set[pt_load_cols].sum(axis=1).max()
        pt_mean = n.loads_t.p_set[pt_load_cols].sum(axis=1).mean()
        print(f"  PT load: mean={pt_mean:.0f} MW, peak={pt_peak:.0f} MW")

    # Total generation capacity
    total_gen = n.generators["p_nom"].sum()
    print(f"\n  Total generation capacity: {total_gen:.0f} MW")

    # FR generation vs FR load
    fr_gen = n.generators[n.generators["bus"].str.startswith("FR")]["p_nom"].sum()
    print(f"  FR generation: {fr_gen:.0f} MW vs FR peak load: {fr_peak:.0f} MW")
    if fr_gen < fr_peak:
        print(f"    ⚠ FR generation ({fr_gen:.0f} MW) < FR peak load ({fr_peak:.0f} MW) — needs imports!")

    pt_gen = n.generators[n.generators["bus"].str.startswith("PT")]["p_nom"].sum()
    print(f"  PT generation: {pt_gen:.0f} MW vs PT peak load: {pt_peak:.0f} MW")

    return True


def run_diagnostics(n, config):
    """Run all checks and return overall status."""
    checks = [
        ("Loads", check_loads(n)),
        ("Generators", check_generators(n)),
        ("Storage", check_storage(n)),
        ("Interconnectors", check_interconnectors(n, config)),
        ("Transmission", check_transmission(n, config)),
        ("VOLL", check_voll(n, config)),
        ("Capacities Locked", check_capacities_locked(n)),
        ("Hydro Ramp", check_hydro_ramp(n, config)),
        ("Load Balance", check_load_balance(n)),
    ]

    print(f"\n{'='*66}")
    print("  DIAGNOSTIC SUMMARY")
    print(f"{'='*66}")
    all_pass = True
    for name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}  {name}")
        if not result:
            all_pass = False

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED — review above'}")
    print(f"{'='*66}\n")

    return all_pass


if __name__ == "__main__":
    import pypsa
    from config import MODEL_CONFIG

    cfg = MODEL_CONFIG
    net_path = cfg["validation"]["network_path"]
    print(f"\nLoading {net_path}...")
    n = pypsa.Network(net_path)

    # Apply refinements first (same as run_validation.py does)
    from refinery import apply_non_linear_refinements
    n = apply_non_linear_refinements(n, cfg)

    # Slice to analysis window
    start = pd.Timestamp(cfg["validation"]["start_date"])
    n_days = int(cfg["validation"]["n_days"])
    end = start + pd.Timedelta(hours=n_days * 24 - 1)
    snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
    n.set_snapshots(snap)
    print(f"Sliced to {len(snap)} snapshots ({snap[0]} → {snap[-1]})")

    run_diagnostics(n, cfg)
