#!/usr/bin/env python3
"""
Standalone diagnostic: load a solved network and run the two new diagnostics:

  D6 · CCGT Export vs Domestic Decomposition
  D7 · Seasonal Curtailment Analysis

Usage:
  pixi run python Analysis/diag_export_curtailment.py

Optionally specify a solved network path:
  pixi run python Analysis/diag_export_curtailment.py --net solved_networks/validation/solved_20240101_365d_20260602.nc
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pypsa

# ── Ensure Analysis/ is on sys.path ────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from config import MODEL_CONFIG

ROOT = Path(__file__).parent.parent
from refinery import apply_non_linear_refinements
from run_validation import (
    _add_fr_missing_demand,
    _apply_fr_demand_scaler,
    _add_bess_fleet,
    _mean_es_price,
    _net_import_topo,
    _get_price_setter_series,
)

log = logging.getLogger(__name__)

_SEP  = "=" * 72
_SEP2 = "─" * 72


def _bold(s):
    return f"\033[1m{s}\033[0m"


def _es_price_series(n, snaps=None):
    """Load-weighted mean ES price for each snapshot."""
    mp = _mean_es_price(n)
    if snaps is not None:
        mp = mp.reindex(snaps)
    return mp.dropna()


# ═══════════════════════════════════════════════════════════════════════════
#  D6 · CCGT Export vs Domestic Decomposition
# ═══════════════════════════════════════════════════════════════════════════

def print_ccgt_export_decomposition(n, omie_series=None):
    """Decompose CCGT dispatch into domestic-serving vs export-driven components.

    Algorithm
    ---------
    For each hour:
      domestic_shortfall = load − must_run − VRE − hydro − FR_net − PT_net

    If shortfall > 0:
      CCGT serves domestic demand first; remainder = export-driven
    If shortfall ≤ 0:
      ALL CCGT dispatch is export-driven or LP trickle

    Also computes a counterfactual: cap FR exports at historical mean (344 MW
    import to ES, i.e. ES exports to FR capped at 0) and PT exports at ~200 MW
    mean, then recompute how much CCGT dispatch would be avoided.
    """
    snaps = n.snapshots
    es_buses = [b for b in n.buses.index if str(b).startswith("ES")]

    # ── Load ────────────────────────────────────────────────────────────────
    load_t = n.loads_t.p_set.filter(like="ES").sum(axis=1).reindex(snaps)

    # Must-run: CCGT_must_run + biomass + nuclear
    must_run_cols = [g for g in n.generators.index
                     if ("_must_run" in g or "biomass" in g or "nuclear" in g)
                     and str(g).startswith("ES")]
    must_t = n.generators_t.p[must_run_cols].sum(axis=1).reindex(snaps) if must_run_cols else pd.Series(0.0, index=snaps)

    # VRE: solar + onwind + offwind
    vre_cols = [g for g in n.generators.index
                if any(c in g for c in ("solar", "onwind", "offwind"))
                and str(g).startswith("ES")]
    vre_t = n.generators_t.p[vre_cols].sum(axis=1).reindex(snaps) if vre_cols else pd.Series(0.0, index=snaps)

    # Hydro: storage units + generators (ror + reservoir)
    hyd_su = [su for su in n.storage_units.index if str(su).startswith("ES")]
    hyd_gen = [g for g in n.generators.index
               if "hydro" in g and str(g).startswith("ES")]
    hyd_t = pd.DataFrame(0.0, index=snaps, columns=[0]).sum(axis=1)
    if hyd_su:
        hyd_t += n.storage_units_t.p[hyd_su].sum(axis=1).reindex(snaps).fillna(0.0)
    if hyd_gen:
        hyd_t += n.generators_t.p[hyd_gen].sum(axis=1).reindex(snaps).fillna(0.0)

    # FR net import (positive = ES importing from FR)
    fr_t = _net_import_topo(n, "FR").reindex(snaps)
    pt_t = _net_import_topo(n, "PT").reindex(snaps)

    # CCGT dispatch
    ccgt_cols = [g for g in n.generators.index
                 if ("CCGT" in g or "OCGT" in g) and str(g).startswith("ES")]
    gas_t = n.generators_t.p[ccgt_cols].sum(axis=1).reindex(snaps) if ccgt_cols else pd.Series(0.0, index=snaps)

    # Price setter
    if omie_series is not None:
        _, setter_t = _get_price_setter_series(n, "ES")
        setter_t = setter_t.reindex(snaps)
    else:
        setter_t = pd.Series("unknown", index=snaps)

    # ── Decompose ────────────────────────────────────────────────────────────
    domestic_supply_before_gas = must_t + vre_t + hyd_t + fr_t + pt_t
    domestic_shortfall = load_t - domestic_supply_before_gas

    # Debug: understand sign convention and distribution
    print(f"\n  [DEBUG] FR net import: mean={fr_t.mean():+.0f} MW, min={fr_t.min():+.0f}, max={fr_t.max():+.0f}")
    print(f"  [DEBUG] PT net import: mean={pt_t.mean():+.0f} MW, min={pt_t.min():+.0f}, max={pt_t.max():+.0f}")
    print(f"  [DEBUG] Domestic shortfall: mean={domestic_shortfall.mean():+.0f} MW, "
          f"min={domestic_shortfall.min():+.0f}, max={domestic_shortfall.max():+.0f}")
    neg_short = (domestic_shortfall <= 0).sum()
    print(f"  [DEBUG] Hours with shortfall ≤ 0: {neg_short} / {len(snaps)} ({100*neg_short/len(snaps):.1f}%)")
    ccgt_running = (gas_t > 0).sum()
    neg_short_and_ccgt = ((domestic_shortfall <= 0) & (gas_t > 0)).sum()
    print(f"  [DEBUG] Hours with CCGT running: {ccgt_running} / {len(snaps)}")
    print(f"  [DEBUG] Hours with shortfall ≤ 0 AND CCGT running: {neg_short_and_ccgt} / {ccgt_running}")

    domestic_ccgt_t = pd.Series(0.0, index=snaps, dtype=float)
    export_ccgt_t   = pd.Series(0.0, index=snaps, dtype=float)

    for h in snaps:
        ccgt_h = float(gas_t.loc[h])
        if ccgt_h <= 0:
            continue
        short_h = float(domestic_shortfall.loc[h])
        if short_h > 0:
            dom_h = min(short_h, ccgt_h)
            exp_h = ccgt_h - dom_h
        else:
            dom_h = 0.0
            exp_h = ccgt_h
        domestic_ccgt_t.loc[h] = dom_h
        export_ccgt_t.loc[h]   = exp_h

    # ── Export-only hours ────────────────────────────────────────────────────
    export_only_mask = (gas_t > 0) & (domestic_shortfall <= 0)
    export_only_hrs  = int(export_only_mask.sum())
    export_only_gwh  = float(gas_t[export_only_mask].sum()) / 1e3

    # ── Monthly breakdown ────────────────────────────────────────────────────
    months = sorted(set(snaps.to_period("M")))

    # ── Counterfactual: cap FR exports ───────────────────────────────────────
    # Historical mean: FR imports ~344 MW from ES (i.e. ES exports ~344 MW to FR)
    # Cap: FR_net_import ≥ -344 (i.e. ES exports to FR ≤ 344 MW)
    fr_capped = fr_t.clip(lower=-344.0)
    pt_capped = pt_t.clip(lower=-200.0)  # PT imports ~200 MW mean from ES

    cf_domestic_supply = must_t + vre_t + hyd_t + fr_capped + pt_capped
    cf_shortfall = load_t - cf_domestic_supply

    cf_domestic_ccgt = pd.Series(0.0, index=snaps, dtype=float)
    cf_export_ccgt   = pd.Series(0.0, index=snaps, dtype=float)
    for h in snaps:
        ccgt_h = float(gas_t.loc[h])
        if ccgt_h <= 0:
            continue
        short_h = float(cf_shortfall.loc[h])
        if short_h > 0:
            dom_h = min(short_h, ccgt_h)
            exp_h = ccgt_h - dom_h
        else:
            dom_h = 0.0
            exp_h = ccgt_h
        cf_domestic_ccgt.loc[h] = dom_h
        cf_export_ccgt.loc[h]   = exp_h

    total_export_reduction = (export_ccgt_t - cf_export_ccgt).clip(lower=0.0)
    cf_ccgt_t = (gas_t - total_export_reduction).clip(lower=0.0)
    cf_ccgt_gwh = float(cf_ccgt_t.sum()) / 1e3
    actual_ccgt_gwh = float(gas_t.sum()) / 1e3

    # Counterfactual price setter (rough: CCGT sets price if >100 MW)
    cf_gas_setter_hrs = int((cf_ccgt_t > 100.0).sum())
    if omie_series is not None:
        gas_setter_mask = setter_t.isin({"CCGT", "CCGT_flex", "OCGT"})
        actual_gas_setter_hrs = int(gas_setter_mask.sum())
    else:
        actual_gas_setter_hrs = 0

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  D6 · CCGT EXPORT VS DOMESTIC DECOMPOSITION")
    print(_SEP2)
    print(f"  Method: domestic_shortfall = load − must_run − VRE − hydro − FR_net − PT_net")
    print(f"  When shortfall > 0: CCGT serves domestic demand first; remainder = export-driven")
    print(f"  When shortfall ≤ 0: ALL CCGT dispatch is export-driven or LP trickle\n")

    # Table 1: Annual aggregates
    dom_gwh = float(domestic_ccgt_t.sum()) / 1e3
    exp_gwh = float(export_ccgt_t.sum()) / 1e3
    total_gwh = dom_gwh + exp_gwh
    print(f"  {'Component':<35}  {'GWh':>8}  {'% of total':>10}")
    print(f"  {'─'*55}")
    print(f"  {'Domestic-serving CCGT':<35}  {dom_gwh:>8.0f}  {100*dom_gwh/max(total_gwh,1):>9.1f}%")
    print(f"  {'Export-driven CCGT':<35}  {exp_gwh:>8.0f}  {100*exp_gwh/max(total_gwh,1):>9.1f}%")
    print(f"  {'Total CCGT dispatch':<35}  {total_gwh:>8.0f}  {'100.0%':>10}")

    # Table 2: Export-only hours
    print(f"\n  {'Metric':<45}  {'Value':>10}")
    print(f"  {'─'*57}")
    print(f"  {'Hours where CCGT runs solely for exports (domestic shortfall ≤ 0)':<45}  {export_only_hrs:>5d} / {len(snaps)}")
    print(f"  {'CCGT GWh in those hours':<45}  {export_only_gwh:>8.0f} GWh")
    if export_only_hrs > 0 and omie_series is not None:
        export_only_setter = int(setter_t[export_only_mask].isin({"CCGT", "CCGT_flex", "OCGT"}).sum())
        print(f"  {'CCGT is price-setter in those hours':<45}  {export_only_setter:>5d} / {export_only_hrs}  ({100*export_only_setter/max(export_only_hrs,1):.0f}%)")

    # Table 3: Monthly breakdown
    print(f"\n  {'Month':<8}  {'CCGT':>7}  {'Domestic':>9}  {'Export':>7}  {'Exp%':>5}  "
          f"{'FR_export':>9}  {'PT_export':>9}  {'FR_import':>9}  {'PT_import':>9}")
    print(f"  {'─'*75}")
    for p in months:
        mask_m = snaps.to_period("M") == p
        sl     = snaps[mask_m]
        n_hrs  = int(mask_m.sum())
        if n_hrs == 0:
            continue
        ccgt_m  = float(gas_t.reindex(sl).sum()) / 1e3
        dom_m   = float(domestic_ccgt_t.reindex(sl).sum()) / 1e3
        exp_m   = float(export_ccgt_t.reindex(sl).sum()) / 1e3
        exp_pct = 100 * exp_m / max(ccgt_m, 0.001)
        fr_exp_m = float(fr_t.reindex(sl).clip(upper=0).sum()) / 1e3  # negative = ES exports to FR
        pt_exp_m = float(pt_t.reindex(sl).clip(upper=0).sum()) / 1e3
        fr_imp_m = float(fr_t.reindex(sl).clip(lower=0).sum()) / 1e3
        pt_imp_m = float(pt_t.reindex(sl).clip(lower=0).sum()) / 1e3
        print(f"  {str(p):<8}  {ccgt_m:>7.0f}  {dom_m:>9.0f}  {exp_m:>7.0f}  {exp_pct:>4.0f}%  "
              f"{fr_exp_m:>9.0f}  {pt_exp_m:>9.0f}  {fr_imp_m:>9.0f}  {pt_imp_m:>9.0f}")

    # Table 4: Counterfactual
    print(f"\n  {'─'*57}")
    print(f"  Counterfactual: cap FR exports at 344 MW, PT exports at 200 MW")
    print(f"  {'─'*57}")
    print(f"  {'Metric':<45}  {'Actual':>10}  {'Counterfactual':>15}")
    print(f"  {'─'*72}")
    print(f"  {'Total CCGT dispatch (GWh)':<45}  {actual_ccgt_gwh:>8.0f}        {cf_ccgt_gwh:>8.0f}")
    print(f"  {'Reduction (GWh)':<45}  {'':>10}  {actual_ccgt_gwh - cf_ccgt_gwh:>8.0f}")
    print(f"  {'Reduction (%)':<45}  {'':>10}  {100*(actual_ccgt_gwh-cf_ccgt_gwh)/max(actual_ccgt_gwh,1):>7.1f}%")
    if omie_series is not None:
        print(f"  {'Hours where CCGT >100 MW (price-setter proxy)':<45}  {actual_gas_setter_hrs:>5d} / {len(snaps)}        {cf_gas_setter_hrs:>5d} / {len(snaps)}")

    # Table 5: Top-level interpretation
    exp_share = 100 * exp_gwh / max(total_gwh, 1)
    print(f"\n  {'─'*57}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*57}")
    if exp_share > 30:
        print(f"  ⚠  {exp_share:.0f}% of CCGT dispatch is export-driven — FR/PT modelling is the primary lever.")
        print(f"     Fix: tighten FR border capacity or use historical flow caps in the base case.")
    elif exp_share > 10:
        print(f"  ⚡  {exp_share:.0f}% of CCGT dispatch is export-driven — meaningful but not dominant.")
        print(f"     Both export modelling AND domestic must_run MC fix are needed.")
    else:
        print(f"  ✓  Only {exp_share:.0f}% of CCGT dispatch is export-driven — domestic dynamics dominate.")
        print(f"     Fix: must_run MC (0.0 → 2.0) + MIP ON + MCQ are the right levers.")

    print()


# ═══════════════════════════════════════════════════════════════════════════
#  D7 · Seasonal Curtailment Analysis
# ═══════════════════════════════════════════════════════════════════════════

def print_curtailment_seasonal(n):
    """Break down VRE curtailment by season, node region, and carrier.

    Also reports FR border congestion frequency per season to explain the
    northern Spain spring/autumn curtailment pattern.
    """
    snaps = n.snapshots

    # ── Helper: season from datetime ─────────────────────────────────────────
    def _season(dt):
        m = dt.month
        if 3 <= m <= 5:
            return "Spring"
        elif 6 <= m <= 8:
            return "Summer"
        elif 9 <= m <= 11:
            return "Autumn"
        else:
            return "Winter"

    # ── Helper: region from bus name ─────────────────────────────────────────
    # ES bus naming convention: ES{nn} {city/region}
    # North coast: ES0 24 (Basque Country), ES0 22 (Galicia), ES0 23 (Asturias/Cantabria)
    # North: ES0 21 (Aragon/Navarre/La Rioja), ES0 25 (Catalonia)
    # Centre: ES0 30 (Madrid/Castile-La Mancha), ES0 31 (Castile-Leon)
    # South centre: ES0 41 (Extremadura), ES0 42 (Andalusia north)
    # South: ES0 51 (Andalusia south), ES0 52 (Murcia), ES0 53 (Valencia)
    def _region(bus):
        b = str(bus)
        if any(x in b for x in ("0 24", "0 22", "0 23")):
            return "north_coast"
        elif any(x in b for x in ("0 21", "0 25")):
            return "north"
        elif any(x in b for x in ("0 30", "0 31")):
            return "centre"
        elif any(x in b for x in ("0 41", "0 42")):
            return "south_centre"
        elif any(x in b for x in ("0 51", "0 52", "0 53")):
            return "south"
        return "other"

    # ── VRE curtailment ──────────────────────────────────────────────────────
    # PyPSA stores curtailment as `generators_t.p` minus `generators_t.p_max_pu * p_nom`
    # But more directly: curtailment = p_max_pu - p for VRE generators
    vre_carriers = {"solar", "onwind", "offwind"}
    es_vre = [g for g in n.generators.index
              if n.generators.loc[g, "carrier"] in vre_carriers
              and str(g).startswith("ES")]

    rows = []
    for g in es_vre:
        bus = n.generators.loc[g, "bus"]
        carrier = n.generators.loc[g, "carrier"]
        region = _region(bus)
        p_max = n.generators_t.p_max_pu[g] * n.generators.loc[g, "p_nom"]
        p_actual = n.generators_t.p[g]
        curtail = (p_max - p_actual).clip(lower=0)
        for h in snaps:
            rows.append({
                "snap": h,
                "season": _season(h),
                "region": region,
                "carrier": carrier,
                "curtail_mw": float(curtail.loc[h]),
                "p_max_mw": float(p_max.loc[h]),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"\n{_SEP2}")
        print("  D7 · SEASONAL CURTAILMENT ANALYSIS")
        print(_SEP2)
        print("  No VRE generators found — skipping.")
        return

    # Aggregate by season × region × carrier
    grp = df.groupby(["season", "region", "carrier"]).agg(
        curtail_gwh=("curtail_mw", lambda x: x.sum() / 1e3),
        p_max_gwh=("p_max_mw", lambda x: x.sum() / 1e3),
    )
    grp["curtail_pct"] = 100 * grp["curtail_gwh"] / grp["p_max_gwh"].replace(0, float("nan"))

    # ── FR border congestion ─────────────────────────────────────────────────
    # Find FR↔ES AC lines and INELFE DC links
    fr_es_lines = [ln for ln in n.lines.index
                   if any(b in str(n.lines.loc[ln, "bus0"]) for b in ("FR", "ES"))
                   and any(b in str(n.lines.loc[ln, "bus1"]) for b in ("FR", "ES"))]
    fr_es_links = [lk for lk in n.links.index
                   if any(b in str(n.links.loc[lk, "bus0"]) for b in ("FR", "ES"))
                   and any(b in str(n.links.loc[lk, "bus1"]) for b in ("FR", "ES"))]

    congestion = {}
    for season in ["Spring", "Summer", "Autumn", "Winter"]:
        season_snaps = [h for h in snaps if _season(h) == season]
        if not season_snaps:
            continue
        n_cong = 0
        n_total = len(season_snaps)
        for ln in fr_es_lines:
            s_max = n.lines.loc[ln, "s_max_pu"] * n.lines.loc[ln, "s_nom"]
            if s_max <= 0:
                continue
            flow = n.lines_t.p0[ln].reindex(season_snaps).abs()
            n_cong += int((flow >= 0.95 * s_max).sum())
        for lk in fr_es_links:
            s_max = n.links.loc[lk, "p_nom"]
            if s_max <= 0:
                continue
            flow = n.links_t.p0[lk].reindex(season_snaps).abs()
            n_cong += int((flow >= 0.95 * s_max).sum())
        congestion[season] = (n_cong, n_total * max(len(fr_es_lines) + len(fr_es_links), 1))

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  D7 · SEASONAL CURTAILMENT ANALYSIS")
    print(_SEP2)

    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    for season in season_order:
        sub = grp.loc[season] if season in grp.index else None
        if sub is None or sub.empty:
            continue
        total_curtail = sub["curtail_gwh"].sum()
        total_pmax = sub["p_max_gwh"].sum()
        total_pct = 100 * total_curtail / max(total_pmax, 1)

        print(f"\n  {'─'*57}")
        print(f"  {season.upper():^57}")
        print(f"  {'─'*57}")
        print(f"  Total curtailment: {total_curtail:>8.0f} GWh  ({total_pct:>5.1f}% of available)")

        # By region
        print(f"\n  {'Region':<16}  {'Carrier':<10}  {'Curtail (GWh)':>14}  {'Available (GWh)':>15}  {'%':>5}")
        print(f"  {'─'*64}")
        for region in ["north_coast", "north", "centre", "south_centre", "south"]:
            try:
                reg_sub = sub.loc[region]
            except KeyError:
                continue
            if isinstance(reg_sub, pd.DataFrame):
                for carrier in ["solar", "onwind", "offwind"]:
                    if carrier in reg_sub.index:
                        r = reg_sub.loc[carrier]
                        print(f"  {region:<16}  {carrier:<10}  {r['curtail_gwh']:>10.0f} GWh  {r['p_max_gwh']:>10.0f} GWh  {r['curtail_pct']:>4.1f}%")
            else:
                print(f"  {region:<16}  {reg_sub.name:<10}  {reg_sub['curtail_gwh']:>10.0f} GWh  {reg_sub['p_max_gwh']:>10.0f} GWh  {reg_sub['curtail_pct']:>4.1f}%")

        # FR border congestion
        if season in congestion:
            n_cong, n_possible = congestion[season]
            cong_pct = 100 * n_cong / max(n_possible, 1)
            print(f"\n  FR border congestion (≥95% of capacity): {n_cong:>4d} / {n_possible:>4d}  ({cong_pct:>4.1f}%)")

    # Summary interpretation
    print(f"\n  {'─'*57}")
    print(f"  INTERPRETATION")
    print(f"  {'─'*57}")
    # Aggregate by season
    seas_agg = df.groupby("season").agg(
        curtail_gwh=("curtail_mw", lambda x: x.sum() / 1e3),
        p_max_gwh=("p_max_mw", lambda x: x.sum() / 1e3),
    )
    seas_agg["curtail_pct"] = 100 * seas_agg["curtail_gwh"] / seas_agg["p_max_gwh"].replace(0, float("nan"))
    for s in season_order:
        if s in seas_agg.index:
            r = seas_agg.loc[s]
            print(f"  {s:<10}  {r['curtail_gwh']:>8.0f} GWh  ({r['curtail_pct']:>4.1f}%)")

    # Check if spring/autumn > summer
    spring_pct = seas_agg.loc["Spring", "curtail_pct"] if "Spring" in seas_agg.index else 0
    summer_pct = seas_agg.loc["Summer", "curtail_pct"] if "Summer" in seas_agg.index else 0
    autumn_pct = seas_agg.loc["Autumn", "curtail_pct"] if "Autumn" in seas_agg.index else 0
    winter_pct = seas_agg.loc["Winter", "curtail_pct"] if "Winter" in seas_agg.index else 0

    if spring_pct > summer_pct * 1.5 and autumn_pct > summer_pct * 1.5:
        print(f"\n  ✓ Pattern confirmed: Spring ({spring_pct:.0f}%) and Autumn ({autumn_pct:.0f}%)")
        print(f"    have significantly higher curtailment than Summer ({summer_pct:.0f}%).")
        print(f"    This is physically consistent: high wind + moderate demand + FR border congestion.")
    elif spring_pct > summer_pct * 1.2:
        print(f"\n  ⚡ Moderate seasonal pattern: Spring ({spring_pct:.0f}%) > Summer ({summer_pct:.0f}%).")
    else:
        print(f"\n  ✓ Curtailment is relatively uniform across seasons.")
        print(f"    Spring: {spring_pct:.0f}% | Summer: {summer_pct:.0f}% | Autumn: {autumn_pct:.0f}% | Winter: {winter_pct:.0f}%")

    print()


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Export & curtailment diagnostics on a solved network.")
    parser.add_argument("--net", type=str, default=None,
                        help="Path to solved .nc file (default: use config validation network_path)")
    args = parser.parse_args()

    cfg = MODEL_CONFIG
    val = cfg["validation"]

    if args.net:
        net_path = Path(args.net)
    else:
        net_path = ROOT / val["network_path"]

    if not net_path.exists():
        log.error("Network not found: %s", net_path)
        sys.exit(1)

    print(f"\n{_SEP}")
    print(_bold(f"  EXPORT & CURTAILMENT DIAGNOSTIC"))
    print(f"  Loading: {net_path.name}")
    print(_SEP)

    n = pypsa.Network(str(net_path))

    # Apply refinements (same as run_diagnostic does)
    n = _add_fr_missing_demand(n, cfg)
    n = _apply_fr_demand_scaler(n, cfg)
    log.info("Applying refinements ...")
    n = apply_non_linear_refinements(n, cfg)
    _add_bess_fleet(n, cfg)

    # Slice to configured window
    start = pd.Timestamp(val["start_date"])
    n_days = int(val["n_days"])
    end = start + pd.Timedelta(hours=n_days * 24 - 1)
    snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
    n.set_snapshots(snap)
    log.info("Sliced to %d snapshots", len(snap))

    # Get OMIE price series for price-setter analysis
    omie = _es_price_series(n)

    # Run diagnostics
    print_ccgt_export_decomposition(n, omie)
    print_curtailment_seasonal(n)

    print(f"\n{_SEP}")
    print(_bold("  DIAGNOSTIC COMPLETE"))
    print(_SEP)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
