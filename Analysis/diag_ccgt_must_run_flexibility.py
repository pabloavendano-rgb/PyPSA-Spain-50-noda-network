#!/usr/bin/env python3
"""
Q1: Is CCGT_must_run ACTUALLY FLEXIBLE?
Q2: CCGT linear MC audit — η, tier, MC stats for each ES CCGT generator.

Usage:
    pixi run python Analysis/diag_ccgt_must_run_flexibility.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Analysis"))

from run_validation import _load_omie, _get_price_setter_series, _es_buses
from refinery import apply_non_linear_refinements, _load_mibgas_ts
from config import MODEL_CONFIG

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_SEP = "─" * 72

cfg = MODEL_CONFIG
val = cfg["validation"]

# ── Load most recent solved network ──────────────────────────────────────────
net_dir = ROOT / "solved_networks" / "validation"
nc_files = sorted(net_dir.glob("solved_*.nc"), key=lambda p: p.stat().st_mtime)
if not nc_files:
    raise FileNotFoundError("No solved networks found")
net_path = nc_files[-1]
log.info("Loading %s", net_path.name)
n = pypsa.Network(str(net_path))

# Apply refinements (needed for MC time series, must-run gen creation)
n = apply_non_linear_refinements(n, cfg)

start = pd.Timestamp(val["start_date"])
n_days = int(val["n_days"])
end = start + pd.Timedelta(hours=n_days * 24 - 1)
snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
n.set_snapshots(snap)
log.info("Snapshots: %d  (%s → %s)", len(snap), start.date(), end.date())

# ── Load OMIE for low-price filtering ────────────────────────────────────────
omie = _load_omie(cfg, n.snapshots)

# ═══════════════════════════════════════════════════════════════════════════════
# Q1: CCGT_must_run FLEXIBILITY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{_SEP}")
print("  Q1 — CCGT_must_run FLEXIBILITY DIAGNOSTIC")
print(_SEP)

# Identify must-run generators
mr_gens = n.generators.index[
    n.generators["carrier"] == "CCGT_must_run"
].tolist()
log.info("CCGT_must_run generators: %d", len(mr_gens))

# Get dispatch
gp = n.generators_t.p if not n.generators_t.p.empty else pd.DataFrame(index=n.snapshots)
mr_cols = [g for g in mr_gens if g in gp.columns]
if not mr_cols:
    print("  ⚠ No CCGT_must_run generators found in dispatch")
else:
    mr_dispatch = gp[mr_cols].clip(lower=0)
    mr_total = mr_dispatch.sum(axis=1)

    # (a) All hours
    print(f"\n  (a) ALL HOURS ({len(n.snapshots)} snapshots)")
    print(f"      Mean CCGT_must_run dispatch: {mr_total.mean():.0f} MW")
    print(f"      Min / Max: {mr_total.min():.0f} / {mr_total.max():.0f} MW")
    print(f"      Std:        {mr_total.std():.0f} MW")
    print(f"      p10 / p90:  {mr_total.quantile(0.10):.0f} / {mr_total.quantile(0.90):.0f} MW")
    print(f"      Flat at target_mw? {abs(mr_total.mean() - 1400) < 50}  (target_mw=1400)")

    # (b) Low-OMIE hours where must-take >= demand
    must_carriers = {"nuclear", "biomass", "CCGT_must_run"}
    all_mr_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(must_carriers)
    ]
    all_mr_cols = [g for g in all_mr_gens if g in gp.columns]
    must_take = gp[all_mr_cols].clip(lower=0).sum(axis=1).reindex(n.snapshots, fill_value=0.0)

    # ES load
    es_load = n.loads_t.p_set[
        [c for c in n.loads_t.p_set.columns if c.startswith("ES")]
    ].sum(axis=1).reindex(n.snapshots)

    # Low-OMIE: omie < 50 €/MWh
    low_omie_mask = omie.reindex(n.snapshots) < 50.0
    must_ge_demand = must_take >= es_load

    combined_mask = low_omie_mask & must_ge_demand
    n_combined = combined_mask.sum()
    print(f"\n  (b) LOW-OMIE (<€50) + MUST-TAKE >= DEMAND")
    print(f"      Hours: {n_combined} / {len(n.snapshots)}  ({n_combined/max(1,len(n.snapshots))*100:.1f}%)")
    if n_combined > 0:
        mr_low = mr_total[combined_mask]
        print(f"      Mean CCGT_must_run dispatch: {mr_low.mean():.0f} MW")
        print(f"      Min / Max: {mr_low.min():.0f} / {mr_low.max():.0f} MW")
        print(f"      p10 / p90:  {mr_low.quantile(0.10):.0f} / {mr_low.quantile(0.90):.0f} MW")
        print(f"      Flat at target_mw? {abs(mr_low.mean() - 1400) < 50}")
        print(f"      If ~1400 MW → PINNED (bug: LP can't turn it down)")
        print(f"      If <1400 MW → LP IS turning it down (constraint binds elsewhere)")

    # (c) Hours where ES bus price < €5
    es_buses = _es_buses(n)
    es_price = n.buses_t.marginal_price[es_buses].mean(axis=1).reindex(n.snapshots)
    low_price_mask = es_price < 5.0
    n_low_price = low_price_mask.sum()
    print(f"\n  (c) ES BUS PRICE < €5/MWh")
    print(f"      Hours: {n_low_price} / {len(n.snapshots)}  ({n_low_price/max(1,len(n.snapshots))*100:.1f}%)")
    if n_low_price > 0:
        mr_low_price = mr_total[low_price_mask]
        print(f"      Mean CCGT_must_run dispatch: {mr_low_price.mean():.0f} MW")
        print(f"      Min / Max: {mr_low_price.min():.0f} / {mr_low_price.max():.0f} MW")
        print(f"      p10 / p90:  {mr_low_price.quantile(0.10):.0f} / {mr_low_price.quantile(0.90):.0f} MW")
        print(f"      Flat at target_mw? {abs(mr_low_price.mean() - 1400) < 50}")

    # (d) Individual generator stats
    print(f"\n  (d) INDIVIDUAL CCGT_must_run GENERATORS")
    print(f"      {'Generator':<35} {'Bus':<12} {'p_nom':>8} {'Mean MW':>8} {'CF%':>6} {'p10':>8} {'p90':>8}")
    print(f"      {'─'*35} {'─'*12} {'─'*8} {'─'*8} {'─'*6} {'─'*8} {'─'*8}")
    for g in mr_cols:
        p_nom = float(n.generators.at[g, "p_nom"]) if g in n.generators.index else 0
        bus = str(n.generators.at[g, "bus"]) if g in n.generators.index else "?"
        s = mr_dispatch[g]
        mean_mw = s.mean()
        cf = mean_mw / p_nom * 100 if p_nom > 0 else 0
        p10 = s.quantile(0.10)
        p90 = s.quantile(0.90)
        print(f"      {g:<35} {bus:<12} {p_nom:>8.0f} {mean_mw:>8.0f} {cf:>5.1f}% {p10:>8.0f} {p90:>8.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Q2: CCGT LINEAR MC AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n\n{_SEP}")
print("  Q2 — CCGT LINEAR MC AUDIT")
print(_SEP)

# Identify ES CCGT generators (not must_run, not flex)
es_ccgt = n.generators.index[
    (n.generators["bus"].str.startswith("ES")) &
    (n.generators["carrier"] == "CCGT")
].tolist()
log.info("ES CCGT generators: %d", len(es_ccgt))

# Get marginal_cost time series
mc_t = n.generators_t.marginal_cost if hasattr(n, "generators_t") and not n.generators_t.marginal_cost.empty else None

# Get MIBGAS time series for hand-recalculation
mibgas_ts = _load_mibgas_ts(cfg, n.snapshots)
mibgas_mean = float(mibgas_ts.mean())
co2_price = cfg.get("co2_price", 65.0)
gas_co2_th = cfg.get("co2_intensity", {}).get("gas", 0.202)

# Config tier info
tiers = cfg.get("ccgt_tiers", {}).get("ES", [])
tier_labels = [f"T{i+1}" for i in range(len(tiers))]

print(f"\n  MIBGAS mean (×{cfg.get('mibgas_multiplier', 1.0)}): €{mibgas_mean:.1f}/MWh")
print(f"  CO₂ price: €{co2_price}/t  ×  {gas_co2_th} t/MWh_th = €{co2_price * gas_co2_th:.1f}/MWh_th")
print(f"  VOM: €{cfg.get('ccgt', {}).get('vom', 1.0):.1f}/MWh")
print(f"  MCQ α: {cfg.get('gas_mcq_alpha', {}).get('CCGT', 0.0)}")
print(f"  use_mcq: {cfg.get('use_mcq', False)}")
print()

# Header
hdr = (f"  {'Generator':<35} {'Bus':<12} {'Tier':<6} {'η_lo':>5} {'η_hi':>5} "
       f"{'MC_mean':>8} {'MC_p10':>8} {'MC_p90':>8} {'MCQ':>8} {'VOM':>6} "
       f"{'p_nom':>8} {'MeanMW':>8}")
print(hdr)
print("  " + "─" * len(hdr))

for g in es_ccgt:
    bus = str(n.generators.at[g, "bus"])
    p_nom = float(n.generators.at[g, "p_nom"])
    
    # Efficiency — check if it has a time-varying efficiency
    eta_col = f"{g}_eta"
    if eta_col in n.generators_t:
        eta_mean = float(n.generators_t[eta_col].mean())
        eta_lo = float(n.generators_t[eta_col].min())
        eta_hi = float(n.generators_t[eta_col].max())
    else:
        # Static efficiency from marginal_cost
        mc_val = float(n.generators.at[g, "marginal_cost"])
        eta_mean = (mibgas_mean + gas_co2_th * co2_price) / (mc_val - cfg.get("ccgt", {}).get("vom", 1.0)) if mc_val > 0 else 0
        eta_lo = eta_mean
        eta_hi = eta_mean
    
    # Determine tier
    tier_idx = -1
    for i, (lo, hi) in enumerate(tiers):
        if lo <= eta_mean <= hi:
            tier_idx = i
            break
    tier_label = tier_labels[tier_idx] if tier_idx >= 0 else "??"
    
    # MC time series
    if mc_t is not None and g in mc_t.columns:
        mc_series = mc_t[g]
        mc_mean = float(mc_series.mean())
        mc_p10 = float(mc_series.quantile(0.10))
        mc_p90 = float(mc_series.quantile(0.90))
    else:
        mc_mean = float(n.generators.at[g, "marginal_cost"])
        mc_p10 = mc_mean
        mc_p90 = mc_mean
    
    # MCQ value
    mcq = float(n.generators.at[g, "marginal_cost_quadratic"]) if "marginal_cost_quadratic" in n.generators.columns else 0.0
    
    # VOM
    vom = float(n.generators.at[g, "vom"]) if "vom" in n.generators.columns else cfg.get("ccgt", {}).get("vom", 1.0)
    
    # Mean dispatch
    if g in gp.columns:
        mean_mw = float(gp[g].clip(lower=0).mean())
    else:
        mean_mw = 0.0
    
    print(f"  {g:<35} {bus:<12} {tier_label:<6} {eta_lo:>5.3f} {eta_hi:>5.3f} "
          f"{mc_mean:>8.1f} {mc_p10:>8.1f} {mc_p90:>8.1f} {mcq:>8.2f} {vom:>6.1f} "
          f"{p_nom:>8.0f} {mean_mw:>8.0f}")

# ── Hand-recalculation for one unit ──────────────────────────────────────────
print(f"\n\n  HAND RECALCULATION (first ES CCGT)")
if es_ccgt:
    g = es_ccgt[0]
    bus = str(n.generators.at[g, "bus"])
    p_nom = float(n.generators.at[g, "p_nom"])
    
    # Get efficiency
    eta_col = f"{g}_eta"
    if eta_col in n.generators_t:
        eta_mean = float(n.generators_t[eta_col].mean())
    else:
        mc_static = float(n.generators.at[g, "marginal_cost"])
        eta_mean = (mibgas_mean + gas_co2_th * co2_price) / (mc_static - cfg.get("ccgt", {}).get("vom", 1.0))
    
    vom = float(n.generators.at[g, "vom"]) if "vom" in n.generators.columns else cfg.get("ccgt", {}).get("vom", 1.0)
    
    # Hand compute
    hand_mc = (mibgas_mean + gas_co2_th * co2_price) / eta_mean + vom
    
    # Model MC
    if mc_t is not None and g in mc_t.columns:
        model_mc_mean = float(mc_t[g].mean())
    else:
        model_mc_mean = float(n.generators.at[g, "marginal_cost"])
    
    print(f"\n  Generator: {g}")
    print(f"  Bus:       {bus}")
    print(f"  p_nom:     {p_nom:.0f} MW")
    print(f"  η_mean:    {eta_mean:.4f}")
    print(f"  VOM:       €{vom:.1f}/MWh")
    print(f"\n  Formula:   (MIBGAS_mean + CO₂ × price) / η + VOM")
    print(f"           = ({mibgas_mean:.1f} + {gas_co2_th:.3f} × {co2_price:.0f}) / {eta_mean:.4f} + {vom:.1f}")
    print(f"           = ({mibgas_mean:.1f} + {gas_co2_th * co2_price:.1f}) / {eta_mean:.4f} + {vom:.1f}")
    print(f"           = {mibgas_mean + gas_co2_th * co2_price:.1f} / {eta_mean:.4f} + {vom:.1f}")
    print(f"           = €{hand_mc:.1f}/MWh")
    print(f"\n  Model MC mean: €{model_mc_mean:.1f}/MWh")
    print(f"  Difference:    €{model_mc_mean - hand_mc:.1f}/MWh  ({'✓ MATCH' if abs(model_mc_mean - hand_mc) < 1.0 else '✗ MISMATCH'})")
    
    if abs(model_mc_mean - hand_mc) >= 1.0:
        print(f"\n  ⚠ MISMATCH DETECTED — possible causes:")
        print(f"     1. η values differ from what I computed (check refinery.py tier assignment)")
        print(f"     2. MCQ is active (α={cfg.get('gas_mcq_alpha', {}).get('CCGT', 0.0)})")
        print(f"     3. MIBGAS multiplier not applied correctly")
        print(f"     4. CO₂ price differs from config value")
        print(f"     5. VOM differs from config value")

# ── Summary: fleet composition ──────────────────────────────────────────────
print(f"\n\n{_SEP}")
print("  FLEET COMPOSITION SUMMARY")
print(_SEP)

# Count by tier
tier_counts = {}
tier_capacity = {}
for g in es_ccgt:
    bus = str(n.generators.at[g, "bus"])
    p_nom = float(n.generators.at[g, "p_nom"])
    eta_col = f"{g}_eta"
    if eta_col in n.generators_t:
        eta_mean = float(n.generators_t[eta_col].mean())
    else:
        mc_val = float(n.generators.at[g, "marginal_cost"])
        eta_mean = (mibgas_mean + gas_co2_th * co2_price) / (mc_val - cfg.get("ccgt", {}).get("vom", 1.0)) if mc_val > 0 else 0
    
    tier_idx = -1
    for i, (lo, hi) in enumerate(tiers):
        if lo <= eta_mean <= hi:
            tier_idx = i
            break
    label = tier_labels[tier_idx] if tier_idx >= 0 else "??"
    tier_counts[label] = tier_counts.get(label, 0) + 1
    tier_capacity[label] = tier_capacity.get(label, 0) + p_nom

print(f"\n  {'Tier':<8} {'Count':>6} {'Capacity':>10} {'% of Fleet':>10}")
print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*10}")
for label in tier_labels:
    cnt = tier_counts.get(label, 0)
    cap = tier_capacity.get(label, 0)
    pct = cap / sum(tier_capacity.values()) * 100 if sum(tier_capacity.values()) > 0 else 0
    print(f"  {label:<8} {cnt:>6} {cap:>10.0f} MW {pct:>9.1f}%")
print(f"  {'─'*8} {'─'*6} {'─'*10} {'─'*10}")
print(f"  {'Total':<8} {len(es_ccgt):>6} {sum(tier_capacity.values()):>10.0f} MW {'100.0%':>10}")

# ── Key: what to look for ────────────────────────────────────────────────────
print(f"\n\n{_SEP}")
print("  KEY — WHAT TO LOOK FOR")
print(_SEP)
print("""
  Q1 — CCGT_must_run Flexibility
  ───────────────────────────────
  ✓ If mean dispatch in (b) ≈ 1400 MW → PINNED. The LP cannot turn it down
    even when there's zero residual demand. This is the bug.
    
  ✓ If mean dispatch in (b) < 1400 MW but flat at 1400 in (a) → LP IS
    turning it down in low-demand hours, but some other constraint
    (ramp limits? reserve requirements?) forces it back up in normal hours.
    
  ✓ If mean dispatch varies significantly (std > 100 MW) → must_run IS
    flexible and the LP dispatches it economically. The trickle-dispatch
    problem is elsewhere (MIP off, MCQ off, or tier structure).

  Q2 — CCGT Linear MC Audit
  ──────────────────────────
  ✓ Expected MC range by tier (MIBGAS=36×1.4=50.4, CO₂=60):
      T1 (η 0.72-0.80):  €61-68  ← cheap, modern CCGTs
      T2 (η 0.60-0.72):  €68-81  ← mid fleet
      T3 (η 0.38-0.52):  €92-126 ← cliff jump (+€11 above T2)
      T4 (η 0.24-0.36):  €132-197 ← expensive
      T5 (η 0.16-0.24):  €197-294 ← scarcity/peaking
      
  ✓ Red flags:
      - MCQ > 0 for base CCGTs (should be 0 since α=0.0)
      - MC_mean far outside expected tier range
      - p10-p90 spread very narrow (< €5) → no time variation
      - Hand recalculation mismatch > €1 → formula bug
      - All CCGTs in same tier → tier assignment broken
      - Mean dispatch near p_nom for all units → no economic dispatch
      - Large fleet fraction in T4/T5 with zero dispatch → expensive
        capacity that never runs (your observation — let's check!)
""")
