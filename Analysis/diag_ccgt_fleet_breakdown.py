"""
Quick diagnostic: print CCGT fleet breakdown with MIBGAS cost contribution.

Usage:
    pixi run python Analysis/diag_ccgt_fleet_breakdown.py

Prints:
  - ES CCGT fleet: units, capacities, efficiency tiers, MC ranges
  - MIBGAS price statistics (mean, min, max)
  - Cost construction breakdown per tier (fuel + CO₂ + VOM)
  - CCGT_flex fleet
  - CCGT_must_run fleet
  - Total gas fleet summary
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "Analysis"))

from config import MODEL_CONFIG
from refinery import apply_non_linear_refinements, _load_mibgas_ts

logging.basicConfig(level=logging.WARNING, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

SEP = "═" * 72


def _es_buses(n):
    return [b for b in n.buses.index if str(b).startswith("ES")]


def print_ccgt_fleet_breakdown(n, config):
    """Print detailed CCGT fleet breakdown with MIBGAS cost contribution."""
    cfg = config
    co2_price = cfg["co2_price"]
    gas_co2_th = cfg["gas_co2_intensity_th"]
    co2_adder = gas_co2_th * co2_price  # €13.13/MWh_th
    mibgas_mult = cfg.get("mibgas_multiplier", 1.0)

    # ── Load MIBGAS series ──────────────────────────────────────────────────
    mibgas = _load_mibgas_ts(cfg, n.snapshots)
    mibgas_mean = float(mibgas.mean())
    mibgas_min = float(mibgas.min())
    mibgas_max = float(mibgas.max())

    print(f"\n{SEP}")
    print("  CCGT FLEET BREAKDOWN — MIBGAS Cost Contribution")
    print(SEP)

    # ── MIBGAS price stats ──────────────────────────────────────────────────
    print(f"\n  MIBGAS PVB Day-Ahead Index  (multiplier: ×{mibgas_mult:.1f})")
    print(f"    Mean:  €{mibgas_mean:>6.2f}/MWh_th")
    print(f"    Min:   €{mibgas_min:>6.2f}/MWh_th")
    print(f"    Max:   €{mibgas_max:>6.2f}/MWh_th")
    print(f"    CO₂ adder:  {gas_co2_th} × €{co2_price:.0f} = €{co2_adder:.2f}/MWh_th")
    print(f"    Fuel+CO₂ mean: €{mibgas_mean + co2_adder:.2f}/MWh_th")

    # ── Per-carrier breakdown ───────────────────────────────────────────────
    es_mask = n.generators["bus"].apply(lambda b: str(b).startswith("ES"))

    for carrier in ["CCGT", "CCGT_flex", "CCGT_must_run", "OCGT"]:
        mask = (n.generators["carrier"] == carrier) & es_mask
        gens = n.generators[mask]
        if len(gens) == 0:
            continue

        total_mw = float(gens["p_nom"].sum())
        n_units = len(gens)

        print(f"\n  ── {carrier}: {n_units} units, {total_mw:>8,.0f} MW total ──")

        if carrier == "CCGT":
            tiers = cfg.get("ccgt_efficiency_tiers", {}).get("ES", [])
            vom = cfg.get("gas_vom", {}).get("CCGT", 5.0)
            print(f"    VOM: €{vom}/MWh_e")
            print(f"    Efficiency tiers (η range → MC range at mean MIBGAS=€{mibgas_mean:.1f}):")
            for i, (eta_lo, eta_hi) in enumerate(tiers):
                mc_lo = (mibgas_mean + co2_adder) / eta_hi + vom
                mc_hi = (mibgas_mean + co2_adder) / eta_lo + vom
                print(f"      T{i+1}: η {eta_lo:.2f}–{eta_hi:.2f}  →  MC €{mc_lo:.0f}–{mc_hi:.0f}/MWh_e")
                print(f"           (€{mibgas_mean + co2_adder:.1f}/η + €{vom})")

            # Tranche breakdown
            tranche_cfg = cfg.get("ccgt_tranches", {})
            if tranche_cfg.get("enabled", False):
                print(f"\n    Efficiency tranches (capacity split):")
                for t in tranche_cfg["tranches"]:
                    print(f"      {t['suffix']}: η×{t['eta_multiplier']:.2f}, {t['capacity_fraction']*100:.0f}% of capacity")

        elif carrier == "CCGT_flex":
            flex_cfg = cfg.get("ccgt_flex", {})
            vom = cfg.get("gas_vom", {}).get("CCGT_flex", 8.0)
            frac = flex_cfg.get("capacity_fraction", 0.20)
            eta_tiers = flex_cfg.get("efficiency_tiers", [(0.25, 0.32)])
            print(f"    Carve fraction: {frac*100:.0f}% of each ES CCGT")
            print(f"    VOM: €{vom}/MWh_e")
            for i, (eta_lo, eta_hi) in enumerate(eta_tiers):
                mc_lo = (mibgas_mean + co2_adder) / eta_hi + vom
                mc_hi = (mibgas_mean + co2_adder) / eta_lo + vom
                print(f"      Tier {i+1}: η {eta_lo:.2f}–{eta_hi:.2f}  →  MC €{mc_lo:.0f}–{mc_hi:.0f}/MWh_e")

        elif carrier == "CCGT_must_run":
            mr_cfg = cfg.get("ccgt_must_run", {})
            mc = mr_cfg.get("marginal_cost", 2.0)
            print(f"    MC: €{mc}/MWh_e (below nuclear, above VRE — must-run CHP proxy)")
            print(f"    Target: {mr_cfg.get('target_mw', 2000):.0f} MW")

        elif carrier == "OCGT":
            vom = cfg.get("gas_vom", {}).get("OCGT", 15.0)
            eta = cfg.get("peakers", {}).get("OCGT", {}).get("eta", 0.30)
            mc_mean = (mibgas_mean + co2_adder) / eta + vom
            print(f"    η: {eta:.2f}, VOM: €{vom}/MWh_e")
            print(f"    Mean MC: €{mc_mean:.0f}/MWh_e at MIBGAS=€{mibgas_mean:.1f}")

        # Top 5 units by capacity
        sorted_gens = gens.sort_values("p_nom", ascending=False)
        print(f"\n    Top 5 units by capacity:")
        print(f"    {'Name':<30} {'Bus':<10} {'p_nom':>8} {'MC_mean':>8}")
        print(f"    {'─'*56}")
        for gen_name, row in sorted_gens.head(5).iterrows():
            mc = row["marginal_cost"]
            print(f"    {gen_name:<30} {row['bus']:<10} {row['p_nom']:>8.0f} {mc:>8.1f}")
        if len(sorted_gens) > 5:
            print(f"    ... and {len(sorted_gens) - 5} more units")

    # ── Total gas fleet summary ─────────────────────────────────────────────
    print(f"\n  ── Total Gas Fleet Summary (ES) ──")
    total_gas_mw = 0
    for carrier in ["CCGT", "CCGT_flex", "CCGT_must_run", "OCGT"]:
        mask = (n.generators["carrier"] == carrier) & es_mask
        mw = float(n.generators[mask]["p_nom"].sum())
        if mw > 0:
            total_gas_mw += mw
            print(f"    {carrier:<20} {mw:>8,.0f} MW")
    print(f"    {'─'*30}")
    print(f"    {'Total gas fleet':<20} {total_gas_mw:>8,.0f} MW")

    # ── Cost construction formula ───────────────────────────────────────────
    print(f"\n  ── Cost Construction Formula ──")
    print(f"    MC_e(t) = (MIBGAS(t) × {mibgas_mult:.1f} + {gas_co2_th} × €{co2_price}) / η + VOM")
    print(f"           = (MIBGAS(t) + €{co2_adder:.1f}) / η + VOM")
    print(f"")
    print(f"    At mean MIBGAS=€{mibgas_mean:.1f}:")
    print(f"      Fuel+CO₂ = €{mibgas_mean + co2_adder:.1f}/MWh_th")
    print(f"      CCGT T1 (η=0.72): MC = €{(mibgas_mean + co2_adder)/0.72 + 5:.0f}/MWh_e")
    print(f"      CCGT T4 (η=0.42): MC = €{(mibgas_mean + co2_adder)/0.42 + 5:.0f}/MWh_e")
    print(f"      CCGT_flex (η=0.28): MC = €{(mibgas_mean + co2_adder)/0.28 + 8:.0f}/MWh_e")
    print(f"      OCGT (η=0.30): MC = €{(mibgas_mean + co2_adder)/0.30 + 15:.0f}/MWh_e")

    print(f"\n{SEP}\n")


if __name__ == "__main__":
    cfg = MODEL_CONFIG
    val = cfg["validation"]

    start = pd.Timestamp(val["start_date"])
    n_days = int(val["n_days"])
    end = start + pd.Timedelta(hours=n_days * 24 - 1)

    import pypsa

    net_path = ROOT / val["network_path"]
    print(f"Loading {net_path.name}...")
    n = pypsa.Network(str(net_path))

    # Apply refinements to get the full CCGT fleet with MIBGAS MCs
    print("Applying refinements...")
    n = apply_non_linear_refinements(n, cfg)

    # Slice to analysis window
    snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
    n.set_snapshots(snap)

    print_ccgt_fleet_breakdown(n, cfg)
