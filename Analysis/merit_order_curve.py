"""
merit_order_curve.py — Merit order supply stack for ES, FR, PT.

Shows the sorted marginal cost curve for each country after all refinements,
overlaid with actual dispatch from a solved network.

Usage:
    pixi run python Analysis/merit_order_curve.py

Output:
    Analysis/validation_output/MO_ES.png
    Analysis/validation_output/MO_FR.png
    Analysis/validation_output/MO_PT.png
    Analysis/validation_output/MO_combined.png
"""

import logging
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "Analysis"))

from config import MODEL_CONFIG
from refinery import apply_non_linear_refinements

# ─── Style (mirrors run_validation.py) ────────────────────────────────────────

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          10,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.28,
    "grid.linewidth":     0.6,
    "grid.color":         "#cccccc",
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "legend.framealpha":  0.93,
    "legend.fontsize":    8,
    "legend.edgecolor":   "#dddddd",
})

_SAVE_DPI = 200

TECH_COLORS = {
    "nuclear":        "#3B4CC0",
    "hydro":          "#1E8BC3",
    "ror":            "#0097A7",
    "PHS":            "#7EC8E3",
    "onwind":         "#2ECC71",
    "offwind":        "#27AE60",
    "offwind-float":  "#1E8449",
    "solar":          "#F1C40F",
    "CCGT":           "#E67E22",
    "CCGT_flex":      "#E74C3C",
    "coal":           "#7F8C8D",
    "biomass":        "#795548",
    "OCGT":           "#C0392B",
    "diesel":         "#8B0000",
    "oil":            "#6D4C41",
    "load_shedding":  "#FF00FF",
    "other":          "#BBBBBB",
}

_THRESHOLD_LINES = [
    (80.0,  "CCGT ≈ €80"),
    (120.0, "CCGT flex ≈ €120"),
    (170.0, "Peaker ≈ €170"),
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _build_merit_order(n, country_prefix):
    buses = n.buses.index[n.buses.index.str.startswith(country_prefix)]
    gens = n.generators.loc[
        n.generators["bus"].isin(buses),
        ["p_nom", "marginal_cost", "carrier"]
    ].copy()
    gens["name"] = gens.index
    gens = gens.sort_values("marginal_cost", ascending=True)
    gens["cum_capacity"] = gens["p_nom"].cumsum()
    return gens


def _get_dispatch(n, country_prefix):
    buses = n.buses.index[n.buses.index.str.startswith(country_prefix)]
    gen_names = n.generators.index[n.generators["bus"].isin(buses)]
    return {
        g: float(n.generators_t.p[g].mean()) if g in n.generators_t.p.columns else 0.0
        for g in gen_names
    }


def _draw_staircase(ax, mo, alpha=1.0, lw_h=3.0, lw_v=1.5):
    """Draw the merit order staircase. Returns the rightmost x position."""
    prev_cum = 0.0
    prev_mc  = 0.0
    for _, row in mo.iterrows():
        mc     = row["marginal_cost"]
        p      = row["p_nom"]
        color  = TECH_COLORS.get(row["carrier"], "#BBBBBB")
        ax.plot([prev_cum, prev_cum],     [prev_mc, mc],      color=color, lw=lw_v, alpha=alpha)
        ax.plot([prev_cum, prev_cum + p], [mc, mc],           color=color, lw=lw_h,
                alpha=alpha, solid_capstyle="butt")
        prev_cum += p
        prev_mc   = mc
    return prev_cum


def _draw_dispatch_overlay(ax, mo, dispatch):
    """Overlay actual average dispatch as a dashed staircase."""
    mo_d = mo.copy()
    mo_d["dispatch_mw"] = mo_d["name"].map(dispatch).fillna(0.0)
    prev_cum = 0.0
    prev_mc  = 0.0
    for _, row in mo_d.iterrows():
        mc  = row["marginal_cost"]
        p_d = row["dispatch_mw"]
        if p_d < 0.1:
            continue
        color = TECH_COLORS.get(row["carrier"], "#BBBBBB")
        ax.plot([prev_cum, prev_cum],       [prev_mc, mc],        color=color, lw=0.8, alpha=0.55)
        ax.plot([prev_cum, prev_cum + p_d], [mc, mc],             color=color, lw=1.6,
                alpha=0.70, ls="--")
        prev_cum += p_d
        prev_mc   = mc


def _add_threshold_lines(ax, x_max):
    for price, label in _THRESHOLD_LINES:
        ax.axhline(price, color="#999999", ls=":", lw=1.0, alpha=0.75)
        ax.text(x_max * 0.97, price + 2, label,
                fontsize=7.5, color="#555555", ha="right", va="bottom")


def _carrier_legend(ax, carriers_in_plot, ncol=2):
    handles = [
        plt.Line2D([0], [0], color=TECH_COLORS.get(c, "#BBB"), lw=3, label=c)
        for c in carriers_in_plot
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.93, ncol=ncol)


# ─── Single-country merit order ────────────────────────────────────────────────

def _plot_merit_order(n, country_prefix, country_label, out_dir, dispatch=None):
    mo = _build_merit_order(n, country_prefix)
    if mo.empty:
        log.warning("No generators for %s — skipping", country_label)
        return None

    fig, ax = plt.subplots(figsize=(14, 7))

    x_max = _draw_staircase(ax, mo)
    if dispatch is not None:
        _draw_dispatch_overlay(ax, mo, dispatch)

    _add_threshold_lines(ax, x_max)

    # Shaded zones
    for lo, hi, color in [(0, 30, "#2ECC71"), (30, 80, "#1E8BC3"),
                          (80, 130, "#E67E22"), (130, 220, "#C0392B")]:
        ax.axhspan(lo, hi, alpha=0.04, color=color, zorder=0)

    _carrier_legend(ax, mo["carrier"].unique())

    # Installed vs dispatched annotation
    total_gw = mo["p_nom"].sum() / 1000
    if dispatch is not None:
        disp_gw = sum(v for v in dispatch.values() if v > 0) / 1000
        ax.text(0.98, 0.96,
                f"Installed: {total_gw:.1f} GW\nAvg dispatch: {disp_gw:.1f} GW",
                transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
    else:
        ax.text(0.98, 0.96, f"Installed: {total_gw:.1f} GW",
                transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))

    # Dispatch legend item
    if dispatch is not None:
        disp_line = plt.Line2D([0], [0], color="grey", lw=1.5, ls="--",
                               label="Avg dispatch (dashed)")
        ax.legend(handles=list(ax.get_legend().legend_handles) + [disp_line],
                  loc="upper left", fontsize=8, framealpha=0.93, ncol=2)

    ax.set_xlabel("Cumulative Installed Capacity (MW)")
    ax.set_ylabel("Marginal Cost (EUR/MWh)")
    ax.set_title(f"{country_label} — Merit Order Supply Stack")
    ax.set_xlim(0, x_max * 1.02)
    ax.set_ylim(0, max(float(mo["marginal_cost"].max()) * 1.08, 220.0))

    path = out_dir / f"MO_{country_prefix}.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)
    return mo


# ─── Combined 3-country figure ────────────────────────────────────────────────

def _plot_combined_merit_order(n, out_dir):
    countries = [("ES", "Spain"), ("FR", "France"), ("PT", "Portugal")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    all_carriers = set()
    for ax, (pfx, label) in zip(axes, countries):
        mo = _build_merit_order(n, pfx)
        if mo.empty:
            ax.set_title(f"{label} — no data")
            continue
        all_carriers.update(mo["carrier"].unique())

        x_max = _draw_staircase(ax, mo, lw_h=2.5, lw_v=1.0)
        _add_threshold_lines(ax, x_max)

        for lo, hi, color in [(0, 30, "#2ECC71"), (30, 80, "#1E8BC3"),
                              (80, 130, "#E67E22"), (130, 220, "#C0392B")]:
            ax.axhspan(lo, hi, alpha=0.04, color=color, zorder=0)

        ax.set_xlabel("Cumulative MW")
        ax.set_title(label, fontsize=11)
        ax.set_xlim(0, x_max * 1.02)

        total_gw = mo["p_nom"].sum() / 1000
        ax.text(0.97, 0.97, f"{total_gw:.1f} GW total",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc"))

    axes[0].set_ylabel("Marginal Cost (EUR/MWh)")
    axes[0].set_ylim(0, 230)

    # Shared legend at bottom
    handles = [
        plt.Line2D([0], [0], color=TECH_COLORS.get(c, "#BBB"), lw=3, label=c)
        for c in sorted(all_carriers)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=8,
               framealpha=0.93, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Merit Order Comparison — ES / FR / PT", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14)

    path = out_dir / "MO_combined.png"
    fig.savefig(path, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


# ─── Summary table ────────────────────────────────────────────────────────────

def _print_merit_table(mo, country_label):
    print(f"\n  ── {country_label} Merit Order Summary ──")
    print(f"  {'Carrier':<16} {'Units':>5} {'Capacity (MW)':>14} {'MC Range':>14}")
    print(f"  {'-'*54}")
    for carrier, grp in mo.groupby("carrier"):
        n_units  = len(grp)
        total_mw = grp["p_nom"].sum()
        mc_lo, mc_hi = grp["marginal_cost"].min(), grp["marginal_cost"].max()
        mc_str = f"€{mc_lo:.0f}" if mc_lo == mc_hi else f"€{mc_lo:.0f}–{mc_hi:.0f}"
        print(f"  {carrier:<16} {n_units:>5} {total_mw:>10,.0f}  {mc_str:>14}")
    print(f"  {'-'*54}")
    print(f"  {'TOTAL':<16} {len(mo):>5} {mo['p_nom'].sum():>10,.0f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import pypsa

    cfg = MODEL_CONFIG
    val = cfg["validation"]

    out_dir = ROOT / val["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    net_path = ROOT / val["network_path"]
    log.info("Loading %s", net_path.name)
    n = pypsa.Network(str(net_path))

    log.info("Applying refinements...")
    n = apply_non_linear_refinements(n, cfg)

    start  = pd.Timestamp(val["start_date"])
    n_days = int(val["n_days"])
    end    = start + pd.Timedelta(hours=n_days * 24 - 1)
    snap   = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
    n.set_snapshots(snap)
    log.info("Sliced to %d snapshots (%s → %s)", len(snap), snap[0], snap[-1])

    log.info("Solving with %s...", val["solver"])
    try:
        n.optimize(solver_name=val["solver"])
    except Exception as exc:
        log.error("Solve failed: %s", exc)
        sys.exit(1)
    if not hasattr(n, "objective") or n.objective is None:
        log.error("No objective after solve")
        sys.exit(1)
    log.info("Solve complete  objective = %.2f M€", n.objective / 1e6)

    dispatch_es = _get_dispatch(n, "ES")
    dispatch_fr = _get_dispatch(n, "FR")
    dispatch_pt = _get_dispatch(n, "PT")

    mo_es = _plot_merit_order(n, "ES", "Spain",    out_dir, dispatch_es)
    mo_fr = _plot_merit_order(n, "FR", "France",   out_dir, dispatch_fr)
    mo_pt = _plot_merit_order(n, "PT", "Portugal", out_dir, dispatch_pt)
    _plot_combined_merit_order(n, out_dir)

    if mo_es is not None:
        _print_merit_table(mo_es, "Spain")
    if mo_fr is not None:
        _print_merit_table(mo_fr, "France")
    if mo_pt is not None:
        _print_merit_table(mo_pt, "Portugal")

    print(f"\n  Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
