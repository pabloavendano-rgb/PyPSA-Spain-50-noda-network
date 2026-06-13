#!/usr/bin/env python3
"""
CCGT Trickle Dispatch & Price-Setting Diagnostic

Five-section investigation of whether LP trickle dispatch (CCGT running at <5%
of p_nom without MIP commitment) is the mechanism forcing prices to CCGT
marginal cost across the network.

Sections:
  1. Fleet-wide trickle quantification — how pervasive is trickle by generator and country?
  2. Does trickle set the price? — price distributions in trickle vs non-trickle hours
  3. Bus-level pocket classification — structural congestion vs LP-spread artefact
  4. Residual demand structure — why is trickle persistent at each bus?
  5. PDC impact — how much of the price duration curve is trickle-driven?

Usage:
    python3 Analysis/diag_ccgt_trickle_price.py [path/to/solved.nc]
    (defaults to solved_20240101_180d_20260601.nc)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import pypsa

REPO = Path(__file__).parent.parent
NC_DEFAULT = REPO / "solved_networks/validation/solved_20240101_180d_20260601.nc"
NC_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else NC_DEFAULT
OUT_DIR  = REPO / "Analysis/validation_output"

SEP  = "=" * 72
DASH = "-" * 72

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def sub(title):
    print(f"\n  {title}\n  {DASH[:len(title)+2]}")


# ─── Load network ─────────────────────────────────────────────────────────────
print(f"Loading {NC_PATH.name} …")
n = pypsa.Network(str(NC_PATH))
snaps   = n.snapshots
n_hours = len(snaps)
print(f"  {n_hours} snapshots  |  "
      f"{len(n.generators)} generators  |  "
      f"{len(n.buses)} buses")

# Identify CCGT/flex generators by country
ccgt_carriers = ["CCGT", "CCGT_flex"]
ccgt_mask = n.generators["carrier"].isin(ccgt_carriers)
all_ccgt  = n.generators[ccgt_mask]

def country_prefix(bus_series):
    return bus_series.str[:2]

# Nodal prices (only real generation buses — skip H2/battery buses)
real_es_buses = [b for b in n.buses.index
                 if b.startswith("ES") and not any(x in b for x in
                    ["H2","battery","heat","biogas"])]
price_cols_es = [b for b in real_es_buses if b in n.buses_t.marginal_price.columns]
es_price_ts   = n.buses_t.marginal_price[price_cols_es].mean(axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Fleet-wide trickle quantification
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 1 — FLEET-WIDE TRICKLE QUANTIFICATION")
print("  Trickle = generator is on (p > 0.5 MW) but dispatching < 5% of p_nom.")
print("  With MIP off, LP can spread dispatch across every CCGT simultaneously.\n")

trickle_rows = []
for country in ["ES", "FR", "PT"]:
    country_ccgt = all_ccgt[all_ccgt["bus"].str.startswith(country)]
    for g in country_ccgt.index:
        if g not in n.generators_t.p.columns:
            continue
        p    = n.generators_t.p[g]
        pnom = float(n.generators.at[g, "p_nom"])
        if pnom < 1.0:
            continue
        pu = p / pnom
        pct_off      = (p  <  0.5).mean() * 100
        pct_trickle  = ((p >= 0.5) & (pu < 0.05)).mean() * 100
        pct_mid      = ((pu >= 0.05) & (pu < 0.50)).mean() * 100
        pct_full     = (pu >= 0.50).mean() * 100
        mean_pu_on   = pu[p >= 0.5].mean() if (p >= 0.5).any() else 0.0
        trickle_mwh  = p[(p >= 0.5) & (pu < 0.05)].sum()
        total_mwh    = p.sum()
        trickle_rows.append({
            "generator": g,
            "bus": n.generators.at[g, "bus"],
            "carrier": n.generators.at[g, "carrier"],
            "country": country,
            "p_nom": pnom,
            "pct_off": pct_off,
            "pct_trickle": pct_trickle,
            "pct_mid": pct_mid,
            "pct_full": pct_full,
            "mean_pu_on": mean_pu_on,
            "trickle_mwh": trickle_mwh,
            "total_mwh": total_mwh,
        })

td_df = pd.DataFrame(trickle_rows).sort_values("pct_trickle", ascending=False)

n_heavy = (td_df["pct_trickle"] > 20).sum()
mean_tr  = td_df["pct_trickle"].mean()
total_trickle_twh = td_df["trickle_mwh"].sum() / 1e6
total_disp_twh    = td_df["total_mwh"].sum() / 1e6
trickle_share_pct = total_trickle_twh / total_disp_twh * 100 if total_disp_twh > 0 else 0

print(f"  {len(td_df)} CCGT/flex generators  |  "
      f"{n_heavy} with >20% trickle hours  |  fleet mean trickle: {mean_tr:.1f}%")
print(f"  Total trickle MWh (180d): {total_trickle_twh:.2f} TWh  "
      f"({trickle_share_pct:.1f}% of all CCGT dispatch)")

sub("Country breakdown")
for country in ["ES", "FR", "PT"]:
    sub_df = td_df[td_df["country"] == country]
    if sub_df.empty:
        continue
    print(f"  {country}: {len(sub_df)} generators  "
          f"mean trickle {sub_df['pct_trickle'].mean():.1f}%  "
          f"max {sub_df['pct_trickle'].max():.1f}%  "
          f"trickle TWh {sub_df['trickle_mwh'].sum()/1e6:.3f}")

sub("Top 25 generators by trickle%")
print(f"  {'Generator':<30} {'MW':>6} {'Ctry':>5} {'%Off':>6} {'%<5%pu':>8} "
      f"{'%mid':>6} {'%>50%':>7} {'MeanPU_on':>9}")
print(f"  {DASH}")
for _, r in td_df.head(25).iterrows():
    flag = " ← TRICKLE" if r.pct_trickle > 30 else ""
    print(f"  {str(r.generator):<30} {r.p_nom:>6.0f} {r.country:>5} "
          f"{r.pct_off:>5.1f}% {r.pct_trickle:>7.1f}% {r.pct_mid:>5.1f}% "
          f"{r.pct_full:>6.1f}% {r.mean_pu_on:>9.3f}{flag}")


# ─── Hourly trickle aggregates (used in all later sections) ───────────────────
# Per-generator trickle boolean: on (<5% p_nom)
trickle_mw_ts = pd.Series(0.0, index=snaps)
for g in td_df["generator"]:
    p    = n.generators_t.p[g]
    pnom = n.generators.at[g, "p_nom"]
    pu   = p / pnom
    trickle_mw_ts += p.where((p >= 0.5) & (pu < 0.05), 0.0)

is_trickle_hour = trickle_mw_ts > 50   # >50 MW of trickle system-wide


# ─────────────────────────────────────────────────────────────────────────────
# PART 2 — Does trickle set the price?
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 2 — DOES TRICKLE SET THE PRICE?")
print("  Compares ES mean price in trickle vs non-trickle hours.")
print("  If trickle is marginal, trickle-hour prices should cluster near CCGT MC.\n")

# Mean CCGT MC across active generators (time-varying if available)
ccgt_mc_cols = [g for g in all_ccgt.index
                if g in n.generators_t.marginal_cost.columns]
if ccgt_mc_cols:
    ccgt_mc_ts = n.generators_t.marginal_cost[ccgt_mc_cols].mean(axis=1)
else:
    static_mc = all_ccgt["marginal_cost"].mean()
    ccgt_mc_ts = pd.Series(static_mc, index=snaps)
    print(f"  (No time-varying MC found — using static mean €{static_mc:.1f}/MWh)")

es_ccgt_mc = ccgt_mc_ts  # alias

trickle_hrs   = is_trickle_hour.sum()
no_trickle_hrs = (~is_trickle_hour).sum()
pct_trickle_hrs = trickle_hrs / n_hours * 100

print(f"  Trickle hours (>50 MW system trickle): {trickle_hrs} / {n_hours} ({pct_trickle_hrs:.1f}%)")
print(f"  Non-trickle hours: {no_trickle_hrs} ({100-pct_trickle_hrs:.1f}%)")

sub("Price statistics by trickle regime")
for label, mask in [("Trickle hours  ", is_trickle_hour),
                    ("Non-trickle hrs", ~is_trickle_hour)]:
    if mask.sum() == 0:
        continue
    prices = es_price_ts[mask]
    mc_gap = (prices - es_ccgt_mc[mask]).abs()
    pct_near_mc = (mc_gap < 10).mean() * 100
    print(f"  {label}: mean €{prices.mean():.1f}  median €{prices.median():.1f}  "
          f"p5 €{prices.quantile(0.05):.1f}  p95 €{prices.quantile(0.95):.1f}  "
          f"|price−CCGT_MC|<€10: {pct_near_mc:.1f}%")

sub("Price near CCGT MC (|price − MC| < 10 €/MWh) — by price band")
for lo, hi, label in [
    (0,   30,  "€0–30  (near-zero)"),
    (30,  60,  "€30–60"),
    (60,  90,  "€60–90"),
    (90,  120, "€90–120 (CCGT range)"),
    (120, 999, ">€120  (high)"),
]:
    band_mask = (es_price_ts >= lo) & (es_price_ts < hi)
    if band_mask.sum() == 0:
        continue
    tr_in_band = (band_mask & is_trickle_hour).sum()
    pct_tr     = tr_in_band / band_mask.sum() * 100
    near_mc    = (band_mask & ((es_price_ts - es_ccgt_mc).abs() < 10)).sum()
    pct_near   = near_mc / band_mask.sum() * 100
    print(f"  {label}: {band_mask.sum():5d} hrs  {pct_tr:5.1f}% are trickle hrs  "
          f"{pct_near:5.1f}% within €10 of CCGT MC")


# ─────────────────────────────────────────────────────────────────────────────
# PART 3 — Bus-level pocket classification
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 3 — BUS-LEVEL POCKET CLASSIFICATION")
print("  Structural pocket: isolated by congestion — local CCGT physically needed.")
print("  LP-spread pocket:  cheap gen available but LP dispatched trickle CCGT anyway.")
print("  OK:                CCGT on-time low or cheap generation adequate.\n")

# Line utilisation per bus: fraction of hours where ALL connecting lines are at >80% s_nom
def line_utilisation_at_bus(bus):
    """Mean max-utilisation across lines connected to this bus."""
    connected = n.lines[(n.lines["bus0"] == bus) | (n.lines["bus1"] == bus)].index
    utils = []
    for l in connected:
        if l not in n.lines_t.p0.columns:
            continue
        s_nom = n.lines.at[l, "s_nom"]
        if s_nom <= 0:
            continue
        p0 = n.lines_t.p0[l].abs()
        # also check p1 if available
        pu = p0 / s_nom
        utils.append(pu)
    # also check links (DC cables like Mallorca)
    dc_links = n.links[(n.links["bus0"] == bus) | (n.links["bus1"] == bus)]
    dc_links = dc_links[dc_links["carrier"] == "DC"]
    for lk in dc_links.index:
        if lk not in n.links_t.p0.columns:
            continue
        p_nom = n.links.at[lk, "p_nom"]
        if p_nom <= 0:
            continue
        p0 = n.links_t.p0[lk].abs()
        utils.append(p0 / p_nom)
    if not utils:
        return pd.Series(0.0, index=snaps)
    return pd.concat(utils, axis=1).max(axis=1)  # worst line at this bus

# Per-bus classification — ES buses with CCGT only
es_ccgt_mask = ccgt_mask & n.generators["bus"].str.startswith("ES")
es_ccgt_gens = n.generators[es_ccgt_mask]

bus_class_rows = []
for bus in sorted(set(es_ccgt_gens["bus"])):
    local_ccgt = es_ccgt_gens[es_ccgt_gens["bus"] == bus].index.tolist()
    local_cols = [g for g in local_ccgt if g in n.generators_t.p.columns]
    if not local_cols:
        continue

    p_local = n.generators_t.p[local_cols].sum(axis=1)
    p_nom   = es_ccgt_gens.loc[local_ccgt, "p_nom"].sum()

    pct_on      = (p_local > 1.0).mean() * 100
    pct_trickle = ((p_local > 0.5) & ((p_local / p_nom) < 0.05)).mean() * 100

    # Cheap gen at bus
    cheap_at = [g for g in n.generators.index
                if n.generators.at[g, "bus"] == bus
                and n.generators.at[g, "carrier"] in
                    ("solar","onwind","nuclear","ror","biomass","hydro")
                and g in n.generators_t.p.columns]
    cheap_gen = n.generators_t.p[cheap_at].sum(axis=1) if cheap_at else pd.Series(0.0, index=snaps)

    # Also add storage dispatch
    cheap_su = [u for u in n.storage_units.index
                if n.storage_units.at[u, "bus"] == bus
                and n.storage_units.at[u, "carrier"] in ("hydro", "PHS")
                and u in n.storage_units_t.p_dispatch.columns]
    if cheap_su:
        cheap_gen = cheap_gen + n.storage_units_t.p_dispatch[cheap_su].sum(axis=1)

    # Local load
    load_at = [l for l in n.loads.index
               if n.loads.at[l, "bus"] == bus and l in n.loads_t.p_set.columns]
    load_ts = n.loads_t.p_set[load_at].sum(axis=1) if load_at else pd.Series(0.0, index=snaps)

    cheap_cover = cheap_gen.mean() / load_ts.mean() * 100 if load_ts.mean() > 0 else 0.0

    # Line/cable congestion
    max_line_util = line_utilisation_at_bus(bus)
    pct_congested = (max_line_util > 0.80).mean() * 100

    # Classify
    if pct_on > 90 and cheap_cover < 30 and pct_congested > 50:
        category = "STRUCTURAL"
    elif pct_trickle > 30 and cheap_cover > 60:
        category = "LP-SPREAD"
    elif pct_on > 60 and cheap_cover < 50:
        category = "BORDERLINE"
    else:
        category = "OK"

    bus_class_rows.append({
        "bus": bus,
        "p_nom_mw": p_nom,
        "pct_on": pct_on,
        "pct_trickle": pct_trickle,
        "cheap_cover_pct": cheap_cover,
        "pct_congested": pct_congested,
        "category": category,
    })

bc_df = pd.DataFrame(bus_class_rows).sort_values("pct_trickle", ascending=False)

cat_counts = bc_df["category"].value_counts()
print(f"  Bus classification summary (ES buses with CCGT):")
for cat in ["STRUCTURAL", "LP-SPREAD", "BORDERLINE", "OK"]:
    n_cat = cat_counts.get(cat, 0)
    print(f"    {cat:<12}: {n_cat} buses")

sub("Buses by category (sorted by trickle%)")
print(f"  {'Bus':<12} {'MW':>6} {'%On':>7} {'%Trickle':>9} {'CheapCov%':>10} "
      f"{'%Congested':>11} {'Category':<14}")
print(f"  {DASH}")
for _, r in bc_df.iterrows():
    marker = " ←" if r.category in ("STRUCTURAL","LP-SPREAD") else ""
    print(f"  {str(r.bus):<12} {r.p_nom_mw:>6.0f} {r.pct_on:>6.1f}% "
          f"{r.pct_trickle:>8.1f}% {r.cheap_cover_pct:>9.1f}% "
          f"{r.pct_congested:>10.1f}% {r.category:<14}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 4 — Residual demand structure: why is trickle persistent?
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 4 — RESIDUAL DEMAND: STRUCTURAL NEED VS LP ARTEFACT")
print("  For each trickle-heavy bus: does local demand exceed cheap gen + net imports?")
print("  If local_residual > 0 → CCGT structurally needed (not just LP spreading).")
print("  If local_residual ≤ 0 → LP artefact; MIP would have kept this unit off.\n")

trickle_buses = bc_df[bc_df["pct_trickle"] > 20]["bus"].tolist()

for bus in trickle_buses[:15]:  # cap at 15 to keep output readable
    local_ccgt_cols = [g for g in es_ccgt_gens[es_ccgt_gens["bus"] == bus].index
                       if g in n.generators_t.p.columns]
    p_ccgt = n.generators_t.p[local_ccgt_cols].sum(axis=1) if local_ccgt_cols else pd.Series(0.0, index=snaps)
    p_nom_bus = es_ccgt_gens[es_ccgt_gens["bus"] == bus]["p_nom"].sum()

    # Trickle hours at this bus
    trickle_at_bus = (p_ccgt > 0.5) & ((p_ccgt / p_nom_bus) < 0.05)

    # Cheap gen
    cheap_at = [g for g in n.generators.index
                if n.generators.at[g, "bus"] == bus
                and n.generators.at[g, "carrier"] in
                    ("solar","onwind","nuclear","ror","biomass","hydro")
                and g in n.generators_t.p.columns]
    cheap_gen = n.generators_t.p[cheap_at].sum(axis=1) if cheap_at else pd.Series(0.0, index=snaps)
    cheap_su = [u for u in n.storage_units.index
                if n.storage_units.at[u, "bus"] == bus
                and n.storage_units.at[u, "carrier"] in ("hydro","PHS")
                and u in n.storage_units_t.p_dispatch.columns]
    if cheap_su:
        cheap_gen = cheap_gen + n.storage_units_t.p_dispatch[cheap_su].sum(axis=1)

    # Local load
    load_at = [l for l in n.loads.index
               if n.loads.at[l, "bus"] == bus and l in n.loads_t.p_set.columns]
    load_ts = n.loads_t.p_set[load_at].sum(axis=1) if load_at else pd.Series(0.0, index=snaps)

    # Net imports to bus (lines)
    net_import = pd.Series(0.0, index=snaps)
    for l in n.lines[(n.lines["bus0"] == bus) | (n.lines["bus1"] == bus)].index:
        if l not in n.lines_t.p0.columns:
            continue
        p0 = n.lines_t.p0[l]
        net_import += (-p0 if n.lines.at[l, "bus0"] == bus else p0)
    # DC links
    for lk in n.links[(n.links["bus0"] == bus) | (n.links["bus1"] == bus)].index:
        if n.links.at[lk, "carrier"] != "DC":
            continue
        if lk not in n.links_t.p0.columns:
            continue
        p0 = n.links_t.p0[lk]
        net_import += (-p0 if n.links.at[lk, "bus0"] == bus else p0)

    local_residual = load_ts - cheap_gen - net_import

    # In trickle hours: how often is residual > 0?
    tr_hrs = trickle_at_bus.sum()
    if tr_hrs == 0:
        continue
    pct_struct = (local_residual[trickle_at_bus] > 0).mean() * 100
    pct_artefact = 100 - pct_struct
    mean_res = local_residual[trickle_at_bus].mean()
    cat = bc_df[bc_df["bus"] == bus]["category"].values[0] if bus in bc_df["bus"].values else "?"

    print(f"  {bus:<12} [{cat:<12}]  trickle hrs: {tr_hrs:4d}  "
          f"residual>0 (structural): {pct_struct:5.1f}%  "
          f"residual≤0 (LP artefact): {pct_artefact:5.1f}%  "
          f"mean residual: {mean_res:+.0f} MW")

structural_buses  = [r["bus"] for _, r in bc_df.iterrows()
                     if r["category"] == "STRUCTURAL"]
lp_spread_buses   = [r["bus"] for _, r in bc_df.iterrows()
                     if r["category"] == "LP-SPREAD"]
print(f"\n  Summary: {len(structural_buses)} structural pockets, {len(lp_spread_buses)} LP-spread pockets")
if lp_spread_buses:
    print(f"  LP-spread buses (MIP would eliminate trickle here): {lp_spread_buses}")
if structural_buses:
    print(f"  Structural buses (MIP alone won't fix these): {structural_buses}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 5 — PDC impact
# ─────────────────────────────────────────────────────────────────────────────
hdr("PART 5 — PDC IMPACT OF TRICKLE DISPATCH")
print("  How much of the price duration curve is trickle-driven?")
print("  Counterfactual: trickle hours → price floor (near-zero) to show maximum shift.\n")

pdc_actual      = np.sort(es_price_ts.values)[::-1]
pct_hours       = np.linspace(0, 100, n_hours)

# Counterfactual: trickle hours replaced by the median price from non-trickle hours
# with similar total ES dispatch (±500 MW) — simplified: use median of non-trickle pool
non_trickle_median = es_price_ts[~is_trickle_hour].median() if (~is_trickle_hour).any() else 0.0
cf_prices = es_price_ts.copy()
cf_prices[is_trickle_hour] = non_trickle_median
pdc_cf_median = np.sort(cf_prices.values)[::-1]

# Floor counterfactual: trickle hours → €0 (absolute lower bound)
cf_floor = es_price_ts.copy()
cf_floor[is_trickle_hour] = 0.0
pdc_cf_floor = np.sort(cf_floor.values)[::-1]

print(f"  PDC percentile comparison (actual vs counterfactual with trickle → non-trickle median)")
print(f"  Non-trickle median price used as replacement: €{non_trickle_median:.1f}/MWh\n")
print(f"  {'Percentile':>12} {'Actual €':>10} {'CF-median €':>12} {'CF-floor €':>11} "
      f"{'Δ median':>10} {'Δ floor':>9}")
print(f"  {DASH}")
for pct in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
    idx = int((100 - pct) / 100 * (n_hours - 1))
    actual_p = pdc_actual[idx]
    cf_med_p = pdc_cf_median[idx]
    cf_fl_p  = pdc_cf_floor[idx]
    print(f"  P{pct:>2} (top {pct:>2}%):  "
          f"{actual_p:>8.1f}  {cf_med_p:>10.1f}  {cf_fl_p:>9.1f}  "
          f"{cf_med_p - actual_p:>+9.1f}  {cf_fl_p - actual_p:>+8.1f}")

# How many hours shift by >€5?
shift_gt5  = ((pdc_actual - pdc_cf_median) > 5).sum()
shift_gt10 = ((pdc_actual - pdc_cf_median) > 10).sum()
print(f"\n  Hours where trickle raises price by >€5 : {shift_gt5} ({shift_gt5/n_hours*100:.1f}%)")
print(f"  Hours where trickle raises price by >€10: {shift_gt10} ({shift_gt10/n_hours*100:.1f}%)")
print(f"  Mean price: actual €{es_price_ts.mean():.1f}  CF-median €{cf_prices.mean():.1f}  "
      f"delta {cf_prices.mean()-es_price_ts.mean():+.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE CSV
# ─────────────────────────────────────────────────────────────────────────────
hdr("SAVING OUTPUTS")

hourly_df = pd.DataFrame({
    "snapshot": snaps,
    "trickle_mw": trickle_mw_ts.values,
    "is_trickle_hour": is_trickle_hour.values,
    "es_price": es_price_ts.values,
    "ccgt_mc_fleet": es_ccgt_mc.values,
    "price_gap_vs_mc": (es_price_ts - es_ccgt_mc).values,
}).set_index("snapshot")

csv_path = OUT_DIR / "diag_trickle_price.csv"
hourly_df.to_csv(csv_path)
print(f"  Saved: {csv_path.relative_to(REPO)}")

td_df.to_csv(OUT_DIR / "diag_trickle_generators.csv", index=False)
bc_df.to_csv(OUT_DIR / "diag_trickle_buses.csv", index=False)
print(f"  Saved: Analysis/validation_output/diag_trickle_generators.csv")
print(f"  Saved: Analysis/validation_output/diag_trickle_buses.csv")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 13))
fig.suptitle(f"CCGT Trickle Dispatch & Price-Setting Diagnostic\n{NC_PATH.name}",
             fontsize=13, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# ── Panel 1: Top-20 buses by trickle% ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
top_buses = bc_df.head(20).copy()
colors = {"STRUCTURAL": "#d62728", "LP-SPREAD": "#ff7f0e",
          "BORDERLINE": "#bcbd22", "OK": "#2ca02c"}
bar_colors = [colors.get(c, "grey") for c in top_buses["category"]]
bars = ax1.barh(range(len(top_buses)), top_buses["pct_trickle"].values,
                color=bar_colors, edgecolor="white", linewidth=0.5)
ax1.set_yticks(range(len(top_buses)))
ax1.set_yticklabels(top_buses["bus"].values, fontsize=8)
ax1.invert_yaxis()
ax1.set_xlabel("% hours in trickle (<5% p_nom)", fontsize=9)
ax1.set_title("Trickle Dispatch by Bus (Top 20)", fontsize=10, fontweight="bold")
ax1.axvline(20, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
# Legend
from matplotlib.patches import Patch
legend_elems = [Patch(facecolor=v, label=k) for k, v in colors.items()]
ax1.legend(handles=legend_elems, fontsize=7, loc="lower right")

# ── Panel 2: Price distribution trickle vs non-trickle ────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bins = np.arange(0, 200, 5)
if is_trickle_hour.sum() > 0:
    ax2.hist(es_price_ts[is_trickle_hour].values, bins=bins,
             alpha=0.6, color="#d62728", label=f"Trickle hours ({is_trickle_hour.sum()}h)",
             density=True)
if (~is_trickle_hour).sum() > 0:
    ax2.hist(es_price_ts[~is_trickle_hour].values, bins=bins,
             alpha=0.6, color="#1f77b4", label=f"Non-trickle ({(~is_trickle_hour).sum()}h)",
             density=True)
ax2.axvline(es_price_ts[is_trickle_hour].mean() if is_trickle_hour.sum() > 0 else 0,
            color="#d62728", linestyle="--", linewidth=1.2,
            label=f"Trickle mean €{es_price_ts[is_trickle_hour].mean():.0f}")
ax2.axvline(es_price_ts[~is_trickle_hour].mean() if (~is_trickle_hour).sum() > 0 else 0,
            color="#1f77b4", linestyle="--", linewidth=1.2,
            label=f"Non-trickle mean €{es_price_ts[~is_trickle_hour].mean():.0f}")
ax2.set_xlabel("ES mean price (€/MWh)", fontsize=9)
ax2.set_ylabel("Density", fontsize=9)
ax2.set_title("Price Distribution: Trickle vs Non-Trickle Hours", fontsize=10, fontweight="bold")
ax2.legend(fontsize=8)
ax2.set_xlim(0, 200)

# ── Panel 3: Bus category summary (horizontal bar by category) ────────────
ax3 = fig.add_subplot(gs[1, 0])
cat_order = ["STRUCTURAL", "LP-SPREAD", "BORDERLINE", "OK"]
cat_vals  = [cat_counts.get(c, 0) for c in cat_order]
cat_cols  = [colors[c] for c in cat_order]
ax3.barh(cat_order, cat_vals, color=cat_cols, edgecolor="white")
for i, (v, c) in enumerate(zip(cat_vals, cat_order)):
    if v > 0:
        ax3.text(v + 0.1, i, str(v), va="center", fontsize=10, fontweight="bold")
ax3.set_xlabel("Number of ES CCGT buses", fontsize=9)
ax3.set_title("Bus Pocket Classification", fontsize=10, fontweight="bold")
ax3.set_xlim(0, max(cat_vals) * 1.25 + 1)
# Add annotation about LP-spread implication
ax3.text(0.97, 0.05,
         "LP-SPREAD → MIP would eliminate\nSTRUCTURAL → congestion fix needed",
         transform=ax3.transAxes, ha="right", va="bottom", fontsize=7.5,
         color="grey", style="italic")

# ── Panel 4: Price duration curve actual vs counterfactual ────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(pct_hours, pdc_actual,    color="#1f77b4", linewidth=1.8,
         label=f"Actual (mean €{es_price_ts.mean():.0f})")
ax4.plot(pct_hours, pdc_cf_median, color="#ff7f0e", linewidth=1.5, linestyle="--",
         label=f"CF: trickle→non-trickle median (€{cf_prices.mean():.0f})")
ax4.plot(pct_hours, pdc_cf_floor,  color="#d62728", linewidth=1.2, linestyle=":",
         label=f"CF: trickle→€0 floor (€{cf_floor.mean():.0f})")
ax4.fill_between(pct_hours, pdc_cf_median, pdc_actual,
                 where=(pdc_actual > pdc_cf_median),
                 alpha=0.15, color="#d62728", label="Trickle premium")
ax4.set_xlabel("% of hours (0 = highest price)", fontsize=9)
ax4.set_ylabel("ES mean price (€/MWh)", fontsize=9)
ax4.set_title("Price Duration Curve: Actual vs Counterfactual", fontsize=10, fontweight="bold")
ax4.legend(fontsize=8)
ax4.set_xlim(0, 100)
ax4.set_ylim(bottom=0)
ax4.axhline(0, color="grey", linewidth=0.5)

png_path = OUT_DIR / "diag_trickle_price.png"
fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {png_path.relative_to(REPO)}")
print()
