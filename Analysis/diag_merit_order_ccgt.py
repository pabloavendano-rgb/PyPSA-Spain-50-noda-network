#!/usr/bin/env python3
"""
CCGT Merit Order & Overdispatch Diagnosis
==========================================
Two-panel figure showing why CCGT overdispatch drives high model prices:
  Left:  Non-linear supply curve (step function) at three seasonal MIBGAS levels,
         annotated with residual demand percentiles.
  Right: Seasonal MC variation by tier vs real OMIE monthly means.

Usage:
    python3 Analysis/diag_merit_order_ccgt.py
Output:
    Analysis/validation_output/ccgt_merit_order.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "Analysis"))
from config import MODEL_CONFIG as cfg

OUT = REPO / "Analysis/validation_output/ccgt_merit_order.png"

# ── Config parameters ─────────────────────────────────────────────────────────
co2    = float(cfg["co2_price"])               # 60 €/t
co2th  = float(cfg["gas_co2_intensity_th"])    # 0.202 tCO₂/MWh_th
vom_c  = float(cfg["gas_vom"]["CCGT"])         # 3 €/MWh_e
vom_f  = float(cfg["gas_vom"].get("CCGT_flex", 1.0))
vom_o  = float(cfg["gas_vom"]["OCGT"])         # 7 €/MWh_e
mult   = float(cfg.get("ccgt_mc_multiplier", 1.0))
tiers  = cfg["ccgt_efficiency_tiers"]["ES"]    # 4 tiers after T5/T6 removal
flex_t = cfg["ccgt_flex"].get("efficiency_tiers") or [cfg["ccgt_flex"]["efficiency_range"]]
ocgt_eta = float(cfg["peakers"]["OCGT_pk"]["eta"])
flex_frac = float(cfg["ccgt_flex"]["capacity_fraction"])

# Fleet sizes (MW)
TOTAL_CCGT_MW  = 19_600.0
FLEX_MW        = TOTAL_CCGT_MW * flex_frac          # 3,920 MW
CCGT_MW        = TOTAL_CCGT_MW * (1 - flex_frac)    # 15,680 MW
PER_TIER_MW    = CCGT_MW / len(tiers)               # 3,920 MW each
OCGT_MW        = float(cfg["peakers"]["OCGT_pk"]["total_mw"])  # 2,500 MW
MUST_RUN_MW    = float(cfg["ccgt_must_run"]["target_mw"])      # 1,400 MW (MC=0, not shown here)

# Residual demand percentiles from diagnostic (Part 2 histogram)
# Full-year solve: residual = ES_load - cheap_gen - net_IC
RESIDUAL_MEAN  = 5_270   # MW  — T2 territory
RESIDUAL_P80   = 10_000  # MW  — T3 territory (estimated from histogram shape)
RESIDUAL_P95   = 15_000  # MW  — T4/flex boundary

# ── Load MIBGAS ───────────────────────────────────────────────────────────────
gas_path = REPO / cfg["gas_prices_csv"]
gas_raw  = pd.read_csv(gas_path)
gas_raw.columns = ["date", "price"]
gas_raw["date"]  = pd.to_datetime(gas_raw["date"], dayfirst=True)
gas_raw = gas_raw.sort_values("date").set_index("date")
monthly_mibgas = gas_raw["price"].resample("ME").mean()
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# Three representative MIBGAS values
MIBGAS_WINTER = float(gas_raw.loc["2024-01":"2024-02", "price"].mean())  # ~27.5
MIBGAS_MEAN   = float(gas_raw["price"].mean())                            # ~34.5
MIBGAS_SUMMER = float(gas_raw.loc["2024-08":"2024-09", "price"].mean())  # ~37–39

# ── Load OMIE prices ──────────────────────────────────────────────────────────
omie_path = REPO / cfg["validation"]["omie_csv"]
omie_monthly_mean = None
try:
    omie_raw = pd.read_csv(omie_path)
    dt_col   = omie_raw.columns[0]
    pr_col   = omie_raw.columns[1]
    omie_raw[dt_col] = pd.to_datetime(omie_raw[dt_col], utc=True, dayfirst=True)
    omie_ts  = pd.Series(omie_raw[pr_col].values,
                         index=omie_raw[dt_col]).sort_index()
    omie_ts  = omie_ts[omie_ts.index.year == 2024]
    omie_monthly_mean = omie_ts.resample("ME").mean()
except Exception:
    pass

# ── Helper: build supply curve at a given MIBGAS ─────────────────────────────
def supply_curve(mibgas):
    """Returns (capacity_mw_steps, mc_steps) for step-function plotting."""
    fc = mibgas + co2th * co2  # fuel+CO₂ in €/MWh_th
    steps_mw = []
    steps_mc = []
    cum = 0.0
    for eta_lo, eta_hi in tiers:
        eta = (eta_lo + eta_hi) / 2
        mc  = fc / eta + vom_c        # raw formula (no multiplier shown separately)
        steps_mw += [cum, cum + PER_TIER_MW]
        steps_mc  += [mc, mc]
        cum += PER_TIER_MW
    # CCGT_flex
    for eta_lo, eta_hi in flex_t:
        eta = (eta_lo + eta_hi) / 2
        mc  = fc / eta + vom_f
        steps_mw += [cum, cum + FLEX_MW]
        steps_mc  += [mc, mc]
        cum += FLEX_MW
    # OCGT
    mc_ocgt = fc / ocgt_eta + vom_o
    steps_mw += [cum, cum + OCGT_MW]
    steps_mc  += [mc_ocgt, mc_ocgt]
    return np.array(steps_mw), np.array(steps_mc)


# ── Tier colours ─────────────────────────────────────────────────────────────
TIER_COLOURS = ["#93C6E0", "#4A9CC4", "#1E5F8C", "#0A2E4A"]  # T1 (light) → T4 (dark)
FLEX_COLOUR  = "#E74C3C"
OCGT_COLOUR  = "#E67E22"

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle(
    "ES CCGT Non-Linear Merit Order — Why CCGT Overdispatch Drives High Prices",
    fontsize=13, fontweight="bold", y=0.99,
)

# ═══════════════════════════════════════════════════════════════
# LEFT PANEL — Supply curve
# ═══════════════════════════════════════════════════════════════
scenarios = [
    (MIBGAS_WINTER, "Winter (Jan, MIBGAS≈{:.0f})".format(MIBGAS_WINTER), "#6BAED6", "--"),
    (MIBGAS_MEAN,   "Annual mean (MIBGAS≈{:.0f})".format(MIBGAS_MEAN),   "#2171B5", "-"),
    (MIBGAS_SUMMER, "Summer (Aug, MIBGAS≈{:.0f})".format(MIBGAS_SUMMER), "#08306B", ":"),
]

for mibgas, label, colour, ls in scenarios:
    xv, yv = supply_curve(mibgas)
    ax1.step(xv, yv, where="post", color=colour, linewidth=2, linestyle=ls, label=label)

# Colour tier bands (vertical shading) at mean MIBGAS
x_left = 0
fc_mean = MIBGAS_MEAN + co2th * co2
band_colours = TIER_COLOURS + [FLEX_COLOUR] + [OCGT_COLOUR]
band_labels  = [f"T{i+1}" for i in range(len(tiers))] + ["Flex", "OCGT"]
cum = 0
for i, (eta_lo, eta_hi) in enumerate(tiers):
    ax1.axvspan(cum, cum + PER_TIER_MW, alpha=0.08, color=TIER_COLOURS[i])
    mc = (fc_mean / ((eta_lo+eta_hi)/2) + vom_c)
    ax1.text(cum + PER_TIER_MW/2, mc + 2.5, f"T{i+1}\n€{mc:.0f}", ha="center",
             fontsize=7.5, color=TIER_COLOURS[i], fontweight="bold")
    cum += PER_TIER_MW
ax1.axvspan(cum, cum + FLEX_MW, alpha=0.08, color=FLEX_COLOUR)
mc_flex = (fc_mean / ((flex_t[0][0]+flex_t[0][1])/2) + vom_f)
ax1.text(cum + FLEX_MW/2, mc_flex + 2.5, f"Flex\n€{mc_flex:.0f}", ha="center",
         fontsize=7.5, color=FLEX_COLOUR, fontweight="bold")
cum += FLEX_MW

# Vertical demand lines
vline_specs = [
    (RESIDUAL_MEAN, "Mean residual\n(5,270 MW)", "#2CA02C", "-"),
    (RESIDUAL_P80,  "p80 residual\n(~10,000 MW)", "#FF7F0E", "--"),
    (RESIDUAL_P95,  "p95 residual\n(~15,000 MW)", "#D62728", ":"),
]
for xv_d, lbl, col, ls in vline_specs:
    ax1.axvline(xv_d, color=col, linewidth=1.5, linestyle=ls, alpha=0.85)
    ax1.text(xv_d + 150, 150, lbl, color=col, fontsize=7.5, va="top")

# Horizontal price references
ax1.axhline(52,  color="green",      linewidth=1.2, alpha=0.7, linestyle="-")
ax1.axhline(80,  color="green",      linewidth=1.0, alpha=0.5, linestyle="--")
ax1.axhline(95,  color="red",        linewidth=1.2, alpha=0.7, linestyle="-")
ax1.text(500, 53, "Real OMIE 2024 mean (€52)", color="green", fontsize=7.5)
ax1.text(500, 81, "Real OMIE p75 (~€80)",       color="green", fontsize=7.5)
ax1.text(500, 96, "Model mean clearing (~€95)", color="red",   fontsize=7.5)

# T4→Flex cliff annotation
xc = CCGT_MW
ax1.annotate(
    "T4→Flex cliff\n≈€37 jump\nat mean MIBGAS",
    xy=(xc + FLEX_MW*0.1, mc_flex),
    xytext=(xc - 2500, mc_flex + 10),
    arrowprops=dict(arrowstyle="->", color=FLEX_COLOUR),
    fontsize=8, color=FLEX_COLOUR, fontweight="bold",
)

ax1.set_xlabel("Cumulative ES CCGT capacity dispatched (MW)", fontsize=10)
ax1.set_ylabel("Marginal cost (€/MWh_e)", fontsize=10)
ax1.set_title("Non-Linear Supply Curve — Seasonal Variation", fontsize=11)
ax1.set_xlim(0, CCGT_MW + FLEX_MW + OCGT_MW + 1000)
ax1.set_ylim(45, 165)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(axis="y", alpha=0.3)

# Annotation box
info_txt = (
    "At mean residual (5,270 MW) → T2 marginal → €64–71\n"
    "Peak hours push into T3/T4 (€79–98)\n"
    "Flex-marginal hours (€125) pull mean to €95\n"
    "Real OMIE mean is €52 → gap ≈ +€43"
)
ax1.text(0.01, 0.98, info_txt, transform=ax1.transAxes,
         fontsize=7.5, va="top", ha="left",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85))

# ═══════════════════════════════════════════════════════════════
# RIGHT PANEL — Seasonal MC bands
# ═══════════════════════════════════════════════════════════════
month_nums = list(range(1, 13))
mibgas_monthly = [monthly_mibgas[monthly_mibgas.index.month == m].mean()
                  if m in monthly_mibgas.index.month else np.nan
                  for m in month_nums]
mibgas_monthly = np.array(mibgas_monthly, dtype=float)
# Fill any missing months with annual mean
mibgas_monthly = np.where(np.isnan(mibgas_monthly), MIBGAS_MEAN, mibgas_monthly)

for i, (eta_lo, eta_hi) in enumerate(tiers):
    eta = (eta_lo + eta_hi) / 2
    mc_arr = (mibgas_monthly + co2th * co2) / eta + vom_c
    ax2.fill_between(month_nums, mc_arr - 2, mc_arr + 2,
                     alpha=0.35, color=TIER_COLOURS[i], label=f"T{i+1} (η={eta_lo:.2f}–{eta_hi:.2f})")
    ax2.plot(month_nums, mc_arr, color=TIER_COLOURS[i], linewidth=1.5)

# CCGT_flex band
for eta_lo, eta_hi in flex_t:
    eta = (eta_lo + eta_hi) / 2
    mc_flex_arr = (mibgas_monthly + co2th * co2) / eta + vom_f
    ax2.fill_between(month_nums, mc_flex_arr - 3, mc_flex_arr + 3,
                     alpha=0.25, color=FLEX_COLOUR, label=f"CCGT_flex (η={eta_lo:.2f}–{eta_hi:.2f})")
    ax2.plot(month_nums, mc_flex_arr, color=FLEX_COLOUR, linewidth=1.5, linestyle="--")

# OMIE monthly overlay
if omie_monthly_mean is not None:
    omie_vals = [omie_monthly_mean[omie_monthly_mean.index.month == m].mean()
                 for m in month_nums]
    ax2.plot(month_nums, omie_vals, color="green", linewidth=2.5,
             marker="o", markersize=5, label="Real OMIE 2024 monthly mean", zorder=5)
    ax2.fill_between(month_nums, 0, omie_vals, alpha=0.07, color="green")

# Model mean reference
ax2.axhline(95, color="red", linewidth=1.5, linestyle="-", alpha=0.7, label="Model mean clearing (~€95)")

# Annotation: where OMIE sits relative to tiers
ax2.text(1.3, 55, "OMIE sits between T1 and T2\nmost months → CCGT should be\nT1–T2 marginal, not T3–T4",
         fontsize=7.5, color="green",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", alpha=0.85))

ax2.set_xlabel("Month (2024)", fontsize=10)
ax2.set_ylabel("Marginal cost / Price (€/MWh)", fontsize=10)
ax2.set_title("Seasonal MC Variation vs Real OMIE Prices", fontsize=11)
ax2.set_xticks(month_nums)
ax2.set_xticklabels(MONTHS, fontsize=8)
ax2.set_ylim(40, 160)
ax2.legend(fontsize=7.5, loc="upper right", ncol=1)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}")

# ── Text summary ──────────────────────────────────────────────────────────────
print()
print("=== CCGT MERIT ORDER SUMMARY ===")
print(f"Config: {len(tiers)} tiers, flex_frac={flex_frac:.0%}, mult=×{mult:.2f}")
print()
print(f"{'Tier':<14} {'η range':<12} {'Cap MW':>8} {'MC Winter':>11} {'MC Mean':>10} {'MC Summer':>11}")
print("-" * 70)
cum = 0
for i, (lo, hi) in enumerate(tiers):
    eta = (lo+hi)/2
    mc_w = (MIBGAS_WINTER + co2th*co2)/eta + vom_c
    mc_m = (MIBGAS_MEAN   + co2th*co2)/eta + vom_c
    mc_s = (MIBGAS_SUMMER + co2th*co2)/eta + vom_c
    cum += PER_TIER_MW
    print(f"T{i+1:<13} {lo:.2f}–{hi:.2f}   {PER_TIER_MW:>8.0f} {mc_w:>10.1f} {mc_m:>10.1f} {mc_s:>11.1f}")

for lo, hi in flex_t:
    eta=(lo+hi)/2
    mc_w=(MIBGAS_WINTER+co2th*co2)/eta+vom_f
    mc_m=(MIBGAS_MEAN  +co2th*co2)/eta+vom_f
    mc_s=(MIBGAS_SUMMER+co2th*co2)/eta+vom_f
    print(f"{'CCGT_flex':<14} {lo:.2f}–{hi:.2f}   {FLEX_MW:>8.0f} {mc_w:>10.1f} {mc_m:>10.1f} {mc_s:>11.1f}  ← no mult applied")

mc_w=(MIBGAS_WINTER+co2th*co2)/ocgt_eta+vom_o
mc_m=(MIBGAS_MEAN  +co2th*co2)/ocgt_eta+vom_o
mc_s=(MIBGAS_SUMMER+co2th*co2)/ocgt_eta+vom_o
print(f"{'OCGT':<14} {ocgt_eta:.2f}       {OCGT_MW:>8.0f} {mc_w:>10.1f} {mc_m:>10.1f} {mc_s:>11.1f}")

print()
print(f"T1–T4 total: {CCGT_MW:,.0f} MW  |  Flex: {FLEX_MW:,.0f} MW  |  OCGT: {OCGT_MW:,.0f} MW")
print()
print("RESIDUAL DEMAND vs SUPPLY CURVE (at mean MIBGAS):")
for label, rdmw in [("Mean (5,270 MW)",  RESIDUAL_MEAN),
                    ("p80 (~10,000 MW)", RESIDUAL_P80),
                    ("p95 (~15,000 MW)", RESIDUAL_P95)]:
    cum_cap = 0
    marginal = "CCGT_flex"
    mc_marginal = (MIBGAS_MEAN+co2th*co2)/((flex_t[0][0]+flex_t[0][1])/2)+vom_f
    for ti, (lo, hi) in enumerate(tiers):
        cum_cap += PER_TIER_MW
        if rdmw <= cum_cap:
            marginal = f"T{ti+1}"
            mc_marginal = (MIBGAS_MEAN+co2th*co2)/((lo+hi)/2)+vom_c
            break
    print(f"  {label}: marginal={marginal}  MC≈€{mc_marginal:.0f}/MWh")

print()
print("REAL OMIE 2024 mean: €52/MWh  →  model clearing mean: €95/MWh  →  gap: +€43")
print("Root cause: demand tail (p80–p95) pushes into T3/T4/flex territory,")
print("pulling the mean far above the T2 level that typical hours would suggest.")
