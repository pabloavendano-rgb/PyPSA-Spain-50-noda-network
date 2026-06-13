"""
Analysis/dispatch_resolution_comparison.py

Four-panel dispatch resolution comparison — Spain model vs ENTSO-E hourly actuals.

  Panel 1 — 48-hour:  hourly stacked area, model left | ENTSO-E right
  Panel 2 — Day:      daily GWh for a representative week, paired bars
  Panel 3 — Weekly:   weekly GWh for the full model period, paired bars
  Panel 4 — Monthly:  monthly GWh for the full model period, paired bars

This cascade reveals at which temporal resolution the model dispatch is accurate.

Usage (from project root):
    python Analysis/dispatch_resolution_comparison.py

Configure NETWORK_PATH and TWO_DAY_START below.
The ENTSO-E hourly CSV must exist at data/validation/spain_actual_generation_2024.csv;
run fetch_spain_2024_actuals.py first if it doesn't.
"""

import os
import sys
from pathlib import Path

import matplotlib

if sys.platform in ("darwin", "win32") or os.environ.get("DISPLAY"):
    pass
else:
    matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# ─── Configuration ────────────────────────────────────────────────────────────

# Path to a solved PyPSA network (.nc). Must have dispatch results (n.generators_t.p).
NETWORK_PATH = ROOT / "solved_networks/04_solve_diagnostic/solved_2024_fullyear_co2.nc"

# ENTSO-E hourly actuals produced by fetch_spain_2024_actuals.py
ENTSOE_PATH = ROOT / "data/validation/spain_actual_generation_2024.csv"

# Where to save the figure
OUT_PATH = ROOT / "Analysis/validation_output/dispatch_resolution_comparison.png"

# Start of the 48-hour window. Set to e.g. "2024-07-01" or None for auto (first 2 days).
TWO_DAY_START = None

SAVE_DPI = 200

# ─── Technology groups ────────────────────────────────────────────────────────

# Model carrier → display group
GROUPS = {
    "Nuclear": ["nuclear"],
    "Hydro":   ["hydro", "PHS", "ror"],
    "Wind":    ["onwind", "offwind", "offwind-float"],
    "Solar":   ["solar"],
    "Thermal": ["CCGT", "CCGT_flex", "coal", "OCGT", "diesel",
                "biomass", "oil", "other", "load_shedding"],
}

GROUP_COLORS = {
    "Nuclear": "#3B4CC0",
    "Hydro":   "#1E8BC3",
    "Wind":    "#2ECC71",
    "Solar":   "#F1C40F",
    "Thermal": "#E67E22",
}

# ENTSO-E column → display group (mirrors fetch_spain_2024_actuals.py mapping)
ENTSOE_TO_GROUP = {
    "Nuclear":         "Nuclear",
    "Hydro_Reservoir": "Hydro",
    "Hydro_River":     "Hydro",
    "Wind":            "Wind",
    "Solar_PV":        "Solar",
    "CCGT":            "Thermal",
    "Coal":            "Thermal",
    "Cogeneration":    "Thermal",
    "Other":           "Thermal",
}

# ─── Matplotlib style ─────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linewidth":    0.55,
    "grid.color":        "#cccccc",
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "legend.framealpha": 0.93,
    "legend.fontsize":   7.5,
    "legend.edgecolor":  "#dddddd",
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
})

# ─── Data helpers ─────────────────────────────────────────────────────────────

def _es_buses(n):
    return n.buses.index[
        n.buses.index.str.startswith("ES")
        & ~n.buses.index.str.contains("H2")
        & ~n.buses.index.str.contains("battery")
    ]


def _model_dispatch_es(n):
    """Return hourly Spain dispatch by carrier (MW). Includes storage net dispatch."""
    buses = set(_es_buses(n))

    gen_mask = n.generators["bus"].isin(buses)
    avail = [g for g in n.generators.index[gen_mask] if g in n.generators_t.p.columns]
    if avail:
        carriers = n.generators.loc[avail, "carrier"]
        gen_d = n.generators_t.p[avail].T.groupby(carriers).sum().T
    else:
        gen_d = pd.DataFrame(index=n.snapshots)

    su_mask = n.storage_units["bus"].isin(buses)
    p_dis = getattr(n.storage_units_t, "p_dispatch", pd.DataFrame())
    p_str = getattr(n.storage_units_t, "p_store", pd.DataFrame())
    dis_avail = [s for s in n.storage_units.index[su_mask] if s in p_dis.columns]
    if dis_avail:
        su_car = n.storage_units.loc[dis_avail, "carrier"]
        net = p_dis[dis_avail].copy()
        str_avail = [s for s in dis_avail if s in p_str.columns]
        if str_avail:
            net[str_avail] -= p_str[str_avail]
        su_d = net.T.groupby(su_car).sum().T
        gen_d = pd.concat([gen_d, su_d], axis=1)

    return gen_d.clip(lower=0)


def _apply_groups(carrier_df):
    """Aggregate carrier-level DataFrame into group-level DataFrame."""
    out = {}
    for grp, carriers in GROUPS.items():
        cols = [c for c in carriers if c in carrier_df.columns]
        out[grp] = carrier_df[cols].sum(axis=1) if cols else pd.Series(0.0, index=carrier_df.index)
    return pd.DataFrame(out)


def _load_entsoe(path, snapshots):
    """
    Load ENTSO-E hourly CSV (MW), strip UTC tz-label, reindex to model snapshots.
    Values represent MW of generation per hour.
    """
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    snap = snapshots
    if snap.tz is not None:
        snap = snap.tz_localize(None)

    df = df.reindex(snap, method="nearest", tolerance=pd.Timedelta("90min")).fillna(0.0)
    df.index = snapshots  # restore original tz if any
    return df


def _entsoe_by_group(df):
    out = {g: pd.Series(0.0, index=df.index) for g in GROUP_COLORS}
    for col, grp in ENTSOE_TO_GROUP.items():
        if col in df.columns:
            out[grp] = out[grp] + df[col]
    return pd.DataFrame(out)


def _first_full_week(snapshots):
    """First Monday where the full Mon–Sun week fits inside the model period."""
    snap_end = snapshots[-1]
    for ts in snapshots:
        if ts.dayofweek == 0:
            if ts + pd.Timedelta(days=6, hours=23) <= snap_end:
                return ts
    return snapshots[0]  # fallback: use start


def _tz_naive(idx):
    """Return tz-naive copy of a DatetimeIndex."""
    return idx.tz_localize(None) if idx.tz is not None else idx


# ─── Plot helpers ─────────────────────────────────────────────────────────────

def _stacked_area(ax, groups_df, title, ylabel, x_loc=None, x_fmt=None):
    """Draw stacked fill_between on ax. Returns legend handles."""
    order = [g for g in GROUP_COLORS if g in groups_df.columns]
    bottom = np.zeros(len(groups_df))
    handles = []
    for grp in order:
        vals = groups_df[grp].clip(lower=0).values
        if vals.sum() < 0.01:
            continue
        ax.fill_between(groups_df.index, bottom, bottom + vals,
                        step="post", color=GROUP_COLORS[grp], alpha=0.88)
        handles.append(mpatches.Patch(color=GROUP_COLORS[grp], label=grp))
        bottom = bottom + vals
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if x_loc:
        ax.xaxis.set_major_locator(x_loc)
    if x_fmt:
        ax.xaxis.set_major_formatter(x_fmt)
    return handles


def _paired_bars(ax, model_grp, entsoe_grp, labels, ylabel, title):
    """
    Per-period paired stacked bars: model (solid) + ENTSO-E (hatched).
    Both stacks are colored by group; solid vs hatch distinguishes source.
    """
    n_periods = len(model_grp)
    bar_w   = 0.38
    x_model = np.arange(n_periods, dtype=float)
    x_ent   = x_model + bar_w + 0.04

    order = [g for g in GROUP_COLORS if g in model_grp.columns]

    bot_m = np.zeros(n_periods)
    bot_e = np.zeros(n_periods)

    for grp in order:
        c = GROUP_COLORS[grp]

        if grp in model_grp.columns:
            vals_m = model_grp[grp].fillna(0).values
            ax.bar(x_model, vals_m, bar_w, bottom=bot_m, color=c, alpha=0.92, zorder=3)
            bot_m += vals_m

        if entsoe_grp is not None and grp in entsoe_grp.columns:
            vals_e = entsoe_grp[grp].reindex(model_grp.index, fill_value=0).values
            ax.bar(x_ent, vals_e, bar_w, bottom=bot_e,
                   color=c, alpha=0.55, hatch="///",
                   edgecolor="white", linewidth=0.3, zorder=3)
            bot_e += vals_e

    # x-ticks at mid-point of each pair
    ax.set_xticks((x_model + x_ent) / 2)
    ax.set_xticklabels(labels, rotation=28, ha="right")

    # Faint separators between period groups
    for xi in x_model[1:]:
        ax.axvline(xi - 0.1, color="#e0e0e0", lw=0.7, zorder=1)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(bottom=0)

    # Legend: group colours + model/ENTSO-E proxy
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g)
               for g in order if g in GROUP_COLORS]
    handles += [
        mpatches.Patch(color="#888", alpha=0.92, label="Model  (solid)"),
        mpatches.Patch(color="#888", alpha=0.55, hatch="///", label="ENTSO-E  (hatched)"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=7)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Load network ──────────────────────────────────────────────────────────
    print(f"Loading: {NETWORK_PATH.name}")
    if not NETWORK_PATH.exists():
        raise FileNotFoundError(f"Network not found: {NETWORK_PATH}")
    n = pypsa.Network(str(NETWORK_PATH))

    if n.generators_t.p.empty:
        raise RuntimeError(
            "Network has no dispatch results (n.generators_t.p is empty). "
            "Solve the network before running this script."
        )

    snapshots = n.snapshots
    snap_start, snap_end = snapshots[0], snapshots[-1]
    n_days = len(snapshots) / 24
    print(f"Model period : {snap_start} → {snap_end}  ({len(snapshots)} h, {n_days:.0f} days)")

    # ── Model dispatch ────────────────────────────────────────────────────────
    print("Extracting Spain dispatch by carrier...")
    model_hourly   = _model_dispatch_es(n)
    model_by_group = _apply_groups(model_hourly)

    # ── ENTSO-E actuals ───────────────────────────────────────────────────────
    if not ENTSOE_PATH.exists():
        raise FileNotFoundError(
            f"ENTSO-E hourly file not found:\n  {ENTSOE_PATH}\n"
            "Run fetch_spain_2024_actuals.py first."
        )
    print("Loading ENTSO-E hourly actuals...")
    entsoe_raw     = _load_entsoe(ENTSOE_PATH, snapshots)
    entsoe_by_group = _entsoe_by_group(entsoe_raw)

    # ── 2-day window ──────────────────────────────────────────────────────────
    if TWO_DAY_START:
        t2_s = pd.Timestamp(TWO_DAY_START)
    else:
        t2_s = _tz_naive(pd.DatetimeIndex([snap_start]))[0].normalize()

    t2_e = t2_s + pd.Timedelta(hours=47)
    idx_naive = _tz_naive(model_by_group.index)
    mask_2d   = (idx_naive >= t2_s) & (idx_naive <= t2_e)

    if mask_2d.sum() < 2:
        # fallback: first 48 rows
        mask_2d = pd.Series(False, index=model_by_group.index)
        mask_2d.iloc[:48] = True

    m2d = model_by_group.loc[mask_2d] / 1000   # → GW
    e2d = entsoe_by_group.loc[mask_2d] / 1000

    # ── Temporal aggregations (GWh) ───────────────────────────────────────────
    model_daily  = model_by_group.resample("D").sum() / 1000
    entsoe_daily = entsoe_by_group.resample("D").sum() / 1000
    entsoe_daily = entsoe_daily.reindex(model_daily.index, fill_value=0.0)

    # Representative week
    wk_start   = _first_full_week(snapshots)
    wk_s_naive = pd.Timestamp(wk_start.date() if hasattr(wk_start, "date") else wk_start)
    wk_e_naive = wk_s_naive + pd.Timedelta(days=7)
    d_idx_naive = _tz_naive(model_daily.index)
    wk_mask     = (d_idx_naive >= wk_s_naive) & (d_idx_naive < wk_e_naive)
    model_week  = model_daily.loc[wk_mask]
    entsoe_week = entsoe_daily.loc[wk_mask]
    week_labels = [
        pd.Timestamp(d).strftime("%a\n%d %b")
        for d in _tz_naive(model_week.index)
    ]

    model_weekly  = model_by_group.resample("W-MON", label="left", closed="left").sum() / 1000
    entsoe_weekly = entsoe_by_group.resample("W-MON", label="left", closed="left").sum() / 1000
    entsoe_weekly = entsoe_weekly.reindex(model_weekly.index, fill_value=0.0)
    wk_labels = [
        pd.Timestamp(d).strftime("w/c %d %b")
        for d in _tz_naive(model_weekly.index)
    ]

    model_monthly  = model_by_group.resample("ME").sum() / 1000
    entsoe_monthly = entsoe_by_group.resample("ME").sum() / 1000
    entsoe_monthly = entsoe_monthly.reindex(model_monthly.index, fill_value=0.0)
    mo_labels = [
        pd.Timestamp(d).strftime("%b\n%Y")
        for d in _tz_naive(model_monthly.index)
    ]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 22), constrained_layout=True)
    gs  = fig.add_gridspec(
        4, 2,
        height_ratios=[1.6, 1.0, 1.0, 1.0],
        hspace=0.44, wspace=0.06,
    )
    ax_m2d = fig.add_subplot(gs[0, 0])
    ax_e2d = fig.add_subplot(gs[0, 1], sharey=ax_m2d)
    ax_day = fig.add_subplot(gs[1, :])
    ax_wk  = fig.add_subplot(gs[2, :])
    ax_mo  = fig.add_subplot(gs[3, :])

    period_str = (
        f"{pd.Timestamp(snap_start).strftime('%d %b')} – "
        f"{pd.Timestamp(snap_end).strftime('%d %b %Y')}"
    )
    fig.suptitle(
        f"Dispatch Resolution Cascade — Spain    ({period_str})",
        fontsize=13, fontweight="bold", y=0.975,
    )

    # ── Panel 1: 48-hour stacked area ─────────────────────────────────────────
    t2_label = (
        f"{t2_s.strftime('%d %b')} – "
        f"{t2_e.strftime('%d %b %Y')}"
    )
    loc_6h = mdates.HourLocator(byhour=[0, 6, 12, 18])
    fmt_6h = mdates.DateFormatter("%H:%M\n%d %b")

    handles = _stacked_area(
        ax_m2d, m2d,
        title=f"Model  —  48-hour Hourly  ({t2_label})",
        ylabel="Generation (GW)",
        x_loc=loc_6h, x_fmt=fmt_6h,
    )
    _stacked_area(
        ax_e2d, e2d,
        title=f"ENTSO-E  —  48-hour Hourly  ({t2_label})",
        ylabel="",
        x_loc=loc_6h, x_fmt=fmt_6h,
    )
    ax_e2d.tick_params(labelleft=False)

    for ax in (ax_m2d, ax_e2d):
        ax.set_xlim(m2d.index[0], m2d.index[-1])
        for ts in m2d.index:
            if pd.Timestamp(ts).hour == 0 and ts > m2d.index[0]:
                ax.axvline(ts, color="#aaaaaa", lw=0.8, zorder=0)

    ax_m2d.legend(
        handles=handles[::-1],
        loc="upper left", fontsize=7.5,
        title="Technology group", title_fontsize=7.5, ncol=1,
    )
    ax_e2d.annotate(
        "← same y-scale as Model",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=7, color="#666", va="top",
    )

    # ── Panel 2: daily totals for representative week ─────────────────────────
    _paired_bars(
        ax_day, model_week, entsoe_week, week_labels,
        "Generation (GWh)",
        f"Daily Totals — representative week  (w/c {wk_start.strftime('%d %b %Y')})",
    )

    # ── Panel 3: weekly totals ────────────────────────────────────────────────
    _paired_bars(
        ax_wk, model_weekly, entsoe_weekly, wk_labels,
        "Generation (GWh)",
        "Weekly Totals",
    )

    # ── Panel 4: monthly totals ───────────────────────────────────────────────
    _paired_bars(
        ax_mo, model_monthly, entsoe_monthly, mo_labels,
        "Generation (GWh)",
        "Monthly Totals",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    fig.savefig(OUT_PATH, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"\nSaved → {OUT_PATH}")

    # ── Console summary table ─────────────────────────────────────────────────
    total_m = model_daily.sum()
    total_e = entsoe_daily.sum()
    sep = "─" * 58
    print(f"\n{sep}")
    print("  PERIOD TOTAL GWh — Spain  (Model vs ENTSO-E)")
    print(f"  {'Group':<12}  {'Model':>8}  {'ENTSO-E':>8}  {'Δ GWh':>8}  {'Δ%':>7}")
    print(f"  {sep}")
    for grp in GROUP_COLORS:
        m = float(total_m.get(grp, 0.0))
        e = float(total_e.get(grp, 0.0))
        diff = m - e
        pct  = diff / e * 100 if e else float("nan")
        print(f"  {grp:<12}  {m:>8.1f}  {e:>8.1f}  {diff:>+8.1f}  {pct:>+6.1f}%")
    total_m_sum = sum(float(total_m.get(g, 0)) for g in GROUP_COLORS)
    total_e_sum = sum(float(total_e.get(g, 0)) for g in GROUP_COLORS)
    diff_sum = total_m_sum - total_e_sum
    pct_sum  = diff_sum / total_e_sum * 100 if total_e_sum else float("nan")
    print(f"  {sep}")
    print(f"  {'TOTAL':<12}  {total_m_sum:>8.1f}  {total_e_sum:>8.1f}  {diff_sum:>+8.1f}  {pct_sum:>+6.1f}%")
    print(f"\n  Figure → {OUT_PATH}")


if __name__ == "__main__":
    main()
