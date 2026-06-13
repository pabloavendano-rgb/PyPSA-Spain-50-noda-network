#!/usr/bin/env python3
"""
Diagnostic: investigate hydro inflow and interconnector behavior
in a recently solved network.

Usage:
    pixi run python Analysis/diag_hydro_inflow.py
"""

import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
SOLVED_DIR = "solved_networks/validation"
# Pick the most recent 120d run from April (has the monthly MC fix applied)
SOLVED_NC = "solved_20240401_120d_20260527.nc"

# Also check the 364d run for comparison
SOLVED_NC_364D = "solved_20240101_364d_20260527.nc"

# Diagnostic CSV (from the 364d run that showed the issues)
DIAG_CSV = "Analysis/validation_output/pypsa_diag_20240101 (3).csv"

# Real flow CSVs
FR_FLOWS_CSV = "Interconnector_ENTSOE_pull/data/FR_ES_cross_border_flows_2024.csv"
PT_FLOWS_CSV = "Analysis/interconnector_analysis/2024_PT_ES_balance_hourly.csv"


def investigate_hydro_inflow(n, label=""):
    """Check if hydro storage units have inflow data."""
    print(f"\n{'='*70}")
    print(f"  HYDRO INFLOW INVESTIGATION  [{label}]")
    print(f"{'='*70}")

    # Find hydro storage units
    hydro_su = n.storage_units[n.storage_units["carrier"] == "hydro"]
    print(f"\n  Hydro storage units: {len(hydro_su)}")
    if hydro_su.empty:
        print("  ⚠ NO HYDRO STORAGE UNITS FOUND")
        return

    # Check inflow
    has_inflow = "inflow" in n.storage_units_t
    print(f"  storage_units_t.inflow exists: {has_inflow}")

    if has_inflow:
        inflow = n.storage_units_t["inflow"]
        # Filter to hydro units only
        hydro_inflow = inflow[hydro_su.index]
        print(f"  Inflow shape: {hydro_inflow.shape}")
        print(f"  Inflow dtypes: {hydro_inflow.dtypes.iloc[0]}")
        print(f"\n  Inflow summary (MW):")
        print(f"    Total (sum over all units): {hydro_inflow.sum().sum():.1f} MWh over period")
        print(f"    Mean per hour (all units):  {hydro_inflow.sum(axis=1).mean():.2f} MW")
        print(f"    Max per hour (all units):   {hydro_inflow.sum(axis=1).max():.2f} MW")
        print(f"    Min per hour (all units):   {hydro_inflow.sum(axis=1).min():.2f} MW")
        print(f"    Zero-inflow hours:          {(hydro_inflow.sum(axis=1) == 0).sum()} / {len(hydro_inflow)}")

        # Per-country breakdown
        for country in ["ES", "FR", "PT"]:
            country_units = [u for u in hydro_su.index if str(hydro_su.loc[u, "bus"]).startswith(country)]
            if not country_units:
                continue
            ci = hydro_inflow[country_units]
            print(f"\n  [{country}] {len(country_units)} units:")
            print(f"    Total inflow:     {ci.sum().sum():.1f} MWh")
            print(f"    Mean hourly:      {ci.sum(axis=1).mean():.2f} MW")
            print(f"    Max hourly:       {ci.sum(axis=1).max():.2f} MW")
            print(f"    Zero-inflow hrs:  {(ci.sum(axis=1) == 0).sum()} / {len(ci)}")
            # Show top 5 units by total inflow
            top5 = ci.sum().sort_values(ascending=False).head(5)
            print(f"    Top 5 units by total inflow (MWh):")
            for u, v in top5.items():
                bus = hydro_su.loc[u, "bus"]
                p_nom = hydro_su.loc[u, "p_nom"]
                max_h = hydro_su.loc[u, "max_hours"]
                print(f"      {u:<40} bus={bus:<12} p_nom={p_nom:>8.1f} max_h={max_h:>6.1f}  inflow={v:>10.1f}")
    else:
        print("  ⚠ NO INFLOW DATA — storage_units_t has no 'inflow' key")
        print(f"  Available keys: {list(n.storage_units_t.keys())}")

    # Check state_of_charge
    has_soc = "state_of_charge" in n.storage_units_t
    print(f"\n  storage_units_t.state_of_charge exists: {has_soc}")
    if has_soc:
        soc = n.storage_units_t["state_of_charge"]
        hydro_soc = soc[hydro_su.index]
        print(f"  SOC shape: {hydro_soc.shape}")
        # Per-country SOC trajectory
        for country in ["ES", "FR", "PT"]:
            country_units = [u for u in hydro_su.index if str(hydro_su.loc[u, "bus"]).startswith(country)]
            if not country_units:
                continue
            cs = hydro_soc[country_units]
            e_nom = hydro_su.loc[country_units, "p_nom"] * hydro_su.loc[country_units, "max_hours"]
            soc_ratio = cs.div(e_nom.replace(0, float("nan")), axis=1)
            print(f"\n  [{country}] SOC ratio (state_of_charge / e_nom):")
            print(f"    Start: {soc_ratio.iloc[0].mean():.3f}  End: {soc_ratio.iloc[-1].mean():.3f}")
            print(f"    Min:   {soc_ratio.min().min():.3f}  Max: {soc_ratio.max().max():.3f}")
            # Monthly mean SOC ratio
            months = soc_ratio.index.to_series().dt.month
            monthly_mean = soc_ratio.groupby(months).mean().mean(axis=1)
            print(f"    Monthly mean SOC ratio:")
            for m, v in monthly_mean.items():
                print(f"      Month {m:>2d}: {v:.3f}")

    # Check marginal_cost
    has_mc = "marginal_cost" in n.storage_units_t
    print(f"\n  storage_units_t.marginal_cost exists: {has_mc}")
    if has_mc:
        mc = n.storage_units_t["marginal_cost"]
        # Only select columns that actually exist (handles pre-fix networks)
        existing_cols = [u for u in hydro_su.index if u in mc.columns]
        if not existing_cols:
            print("  ⚠ No hydro units found in marginal_cost columns (pre-fix network)")
            return
        hydro_mc = mc[existing_cols]
        if not hydro_mc.empty:
            print(f"  MC shape: {hydro_mc.shape}")
            # Per-country MC stats
            for country in ["ES", "FR", "PT"]:
                country_units = [u for u in hydro_su.index if str(hydro_su.loc[u, "bus"]).startswith(country)]
                if not country_units:
                    continue
                cm = hydro_mc[country_units]
                print(f"\n  [{country}] MC stats (€/MWh):")
                print(f"    Mean: {cm.mean().mean():.1f}  Min: {cm.min().min():.1f}  Max: {cm.max().max():.1f}")
                # Monthly mean MC
                months = cm.index.to_series().dt.month
                monthly_mc = cm.groupby(months).mean().mean(axis=1)
                print(f"    Monthly mean MC:")
                for m, v in monthly_mc.items():
                    print(f"      Month {m:>2d}: {v:.1f} €/MWh")


def investigate_interconnectors(n, label=""):
    """Check interconnector topology and flow patterns."""
    print(f"\n{'='*70}")
    print(f"  INTERCONNECTOR INVESTIGATION  [{label}]")
    print(f"{'='*70}")

    # ── Lines ────────────────────────────────────────────────────────────────
    print(f"\n  ── Lines (AC) ──")
    # Find lines connecting ES to FR/PT
    for border, country_code in [("FR", "FR"), ("PT", "PT")]:
        border_lines = n.lines[
            n.lines["bus0"].str.contains(country_code) | n.lines["bus1"].str.contains(country_code)
        ]
        # Filter to those where the other bus is in ES
        es_fr_lines = border_lines[
            ((border_lines["bus0"].str.contains("ES")) & (border_lines["bus1"].str.contains(country_code))) |
            ((border_lines["bus1"].str.contains("ES")) & (border_lines["bus0"].str.contains(country_code)))
        ]
        if es_fr_lines.empty:
            print(f"    ES↔{country_code} AC lines: NONE")
        else:
            print(f"    ES↔{country_code} AC lines ({len(es_fr_lines)}):")
            for idx, row in es_fr_lines.iterrows():
                print(f"      {idx:<30}  {row['bus0']:<15} ↔ {row['bus1']:<15}  "
                      f"s_nom={row['s_nom']:>8.1f} MW  s_nom_ext={row.get('s_nom_extendable', False)}")
                # Check congestion
                if "s_nom_opt" in n.lines:
                    s_opt = n.lines.at[idx, "s_nom_opt"]
                    print(f"        s_nom_opt={s_opt:>8.1f} MW")

    # ── Links ────────────────────────────────────────────────────────────────
    print(f"\n  ── Links (DC / HVDC) ──")
    for border, country_code in [("FR", "FR"), ("PT", "PT")]:
        border_links = n.links[
            n.links["bus0"].str.contains(country_code) | n.links["bus1"].str.contains(country_code)
        ]
        es_border_links = border_links[
            ((border_links["bus0"].str.contains("ES")) & (border_links["bus1"].str.contains(country_code))) |
            ((border_links["bus1"].str.contains("ES")) & (border_links["bus0"].str.contains(country_code)))
        ]
        if es_border_links.empty:
            print(f"    ES↔{country_code} links: NONE")
        else:
            print(f"    ES↔{country_code} links ({len(es_border_links)}):")
            for idx, row in es_border_links.iterrows():
                print(f"      {idx:<30}  {row['bus0']:<15} → {row['bus1']:<15}  "
                      f"p_nom={row['p_nom']:>8.1f} MW  p_min_pu={row.get('p_min_pu', 'N/A')}")

    # ── Flow analysis ────────────────────────────────────────────────────────
    print(f"\n  ── Flow Analysis ──")

    # Use _net_import_topo logic: sum links_t.p0 + lines_t.p0 for ES↔country
    for border, country_code in [("FR", "FR"), ("PT", "PT")]:
        # Lines: flow from bus0 to bus1. ES net import = flow into ES buses.
        border_lines = n.lines[
            ((n.lines["bus0"].str.contains("ES")) & (n.lines["bus1"].str.contains(country_code))) |
            ((n.lines["bus1"].str.contains("ES")) & (n.lines["bus0"].str.contains(country_code)))
        ]
        line_flow = pd.Series(0.0, index=n.snapshots)
        for idx, row in border_lines.iterrows():
            if idx in n.lines_t["p0"].columns:
                flow = n.lines_t["p0"][idx]
                # If bus0 is ES, p0 > 0 means ES exports → subtract
                if "ES" in str(row["bus0"]):
                    line_flow -= flow
                else:
                    line_flow += flow

        # Links: p0 is flow from bus0 to bus1
        border_links = n.links[
            ((n.links["bus0"].str.contains("ES")) & (n.links["bus1"].str.contains(country_code))) |
            ((n.links["bus1"].str.contains("ES")) & (n.links["bus0"].str.contains(country_code)))
        ]
        link_flow = pd.Series(0.0, index=n.snapshots)
        for idx, row in border_links.iterrows():
            if idx in n.links_t["p0"].columns:
                flow = n.links_t["p0"][idx]
                # If bus0 is ES, p0 > 0 means ES exports → subtract
                if "ES" in str(row["bus0"]):
                    link_flow -= flow
                else:
                    link_flow += flow

        net_import = line_flow + link_flow

        print(f"\n    ES↔{country_code} net import (positive = import to ES):")
        print(f"      Mean:   {net_import.mean():>8.1f} MW")
        print(f"      Std:    {net_import.std():>8.1f} MW")
        print(f"      Min:    {net_import.min():>8.1f} MW")
        print(f"      Max:    {net_import.max():>8.1f} MW")
        print(f"      Median: {net_import.median():>8.1f} MW")

        # Congestion analysis
        # For lines: congested if |flow| / s_nom > 0.99
        # For links: congested if |flow| / p_nom > 0.99
        congested_hours = pd.Series(False, index=n.snapshots)
        for idx, row in border_lines.iterrows():
            if idx in n.lines_t["p0"].columns:
                s_nom = row.get("s_nom_opt", row["s_nom"])
                if s_nom > 0:
                    congested_hours |= (n.lines_t["p0"][idx].abs() / s_nom > 0.99)
        for idx, row in border_links.iterrows():
            if idx in n.links_t["p0"].columns:
                p_nom = row.get("p_nom_opt", row["p_nom"])
                if p_nom > 0:
                    congested_hours |= (n.links_t["p0"][idx].abs() / p_nom > 0.99)

        n_congested = congested_hours.sum()
        total_hours = len(congested_hours)
        print(f"      Congested hours: {n_congested} / {total_hours} ({100*n_congested/total_hours:.1f}%)")

        # Monthly congestion breakdown
        months = congested_hours.index.to_series().dt.month
        monthly_cong = congested_hours.groupby(months).mean() * 100
        print(f"      Monthly congestion (%):")
        for m, pct in monthly_cong.items():
            print(f"        Month {m:>2d}: {pct:>5.1f}%")


def check_diagnostic_csv():
    """Check the diagnostic CSV for the reported unit multiplier errors."""
    import os
    path = DIAG_CSV
    if not os.path.exists(path):
        print(f"\n  ⚠ Diagnostic CSV not found: {path}")
        return

    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC CSV CHECK")
    print(f"{'='*70}")

    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    print(f"\n  Columns: {list(df.columns)}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")

    # Check the columns that were reported as having unit multiplier errors
    # From earlier analysis: col 8 = es_load_MW, col 9 = fr_import_MW, col 10 = pt_import_MW
    # Let's find them by name
    for col in ["es_load_MW", "fr_import_MW", "pt_import_MW",
                "actual_fr_import_MW", "actual_pt_import_MW",
                "fr_import_error_MW", "pt_import_error_MW"]:
        if col in df.columns:
            print(f"\n  {col}:")
            print(f"    Mean: {df[col].mean():>10.1f}")
            print(f"    Std:  {df[col].std():>10.1f}")
            print(f"    Min:  {df[col].min():>10.1f}")
            print(f"    Max:  {df[col].max():>10.1f}")
            # Check for absurd values
            absurd = df[col].abs() > 10000
            if absurd.any():
                n_absurd = absurd.sum()
                print(f"    ⚠ {n_absurd} hours with |value| > 10,000 MW")
                print(f"    Examples:")
                for idx in df[absurd].index[:5]:
                    print(f"      {idx}: {df.loc[idx, col]:.1f}")


def main():
    import os
    import pypsa

    # ── 1. Load solved network (120d from April — has monthly MC fix) ──────
    path_120d = os.path.join(SOLVED_DIR, SOLVED_NC)
    if os.path.exists(path_120d):
        log.info("Loading solved network: %s", path_120d)
        n = pypsa.Network(path_120d)
        investigate_hydro_inflow(n, "120d Apr–Jul (with monthly MC fix)")
        investigate_interconnectors(n, "120d Apr–Jul (with monthly MC fix)")
    else:
        log.warning("Solved network not found: %s", path_120d)

    # ── 2. Also load the 364d run for comparison ──────────────────────────
    path_364d = os.path.join(SOLVED_DIR, SOLVED_NC_364D)
    if os.path.exists(path_364d):
        log.info("\nLoading 364d solved network: %s", path_364d)
        n364 = pypsa.Network(path_364d)
        investigate_hydro_inflow(n364, "364d Jan–Dec (BEFORE monthly MC fix)")
        investigate_interconnectors(n364, "364d Jan–Dec (BEFORE monthly MC fix)")
    else:
        log.warning("364d solved network not found: %s", path_364d)

    # ── 3. Check diagnostic CSV ───────────────────────────────────────────
    check_diagnostic_csv()

    print("\n✅ Diagnostic complete.")


if __name__ == "__main__":
    main()
