"""
Diagnostic runner — short-window solve with a looping week-by-week summary.

Usage (from repo root):
    pixi run python Analysis/run_diagnostic.py

Reads start_date / n_days from config.py → MODEL_CONFIG["validation"].
Set n_days=14 for a fast check run, n_days=365 for full-year.

Output after solve: week-by-week table with price stats, spike hours,
CCGT commitment counts (MIP), IC flows, and dispatch mix — everything
needed to interpret whether price formation is realistic.
"""

import logging
import sys
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed

import numpy as np
import pandas as pd
import pypsa

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL_CONFIG
from refinery import apply_non_linear_refinements
from run_validation import (
    _add_fr_missing_demand,
    _apply_fr_demand_scaler,
    _add_bess_fleet,
    _dispatch_by_carrier,
    _load_omie,
    _load_real_dispatch,
    _mean_es_price,
    _to_daily_gwh,
    _print_stats,
    _print_cost_and_price_setter_table,
    _add_su_ramp_constraints,
    _add_hydro_min_dispatch,
    _add_hydro_terminal_soc,
    _get_price_setter_series,
    _net_import_topo,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_W = 74   # table width
_SEP  = "═" * _W
_SEP2 = "─" * _W


# ── Price spike thresholds (EUR/MWh) ─────────────────────────────────────────

_SPIKE_THRESHOLDS = [100, 150, 200, 300]


# ── Terminal colour helpers ───────────────────────────────────────────────────

def _c(text, code):
    """Wrap text in ANSI colour code (stripped on non-TTY)."""
    if not sys.stdout.isatty():
        return str(text)
    return f"\033[{code}m{text}\033[0m"

def _red(t):    return _c(t, "31")
def _green(t):  return _c(t, "32")
def _yellow(t): return _c(t, "33")
def _cyan(t):   return _c(t, "36")
def _bold(t):   return _c(t, "1")


# ── Key metric extractors ─────────────────────────────────────────────────────

def _es_price_series(n, snaps=None):
    """Load-weighted mean ES price for each snapshot."""
    mp = _mean_es_price(n)
    if snaps is not None:
        mp = mp.reindex(snaps)
    return mp.dropna()


def _ic_flows(n, snaps=None):
    """
    Return dict with FR→ES and PT→ES hourly MW (positive = import into ES).
    Aggregates across all FR/PT border links and lines.
    """
    if snaps is None:
        snaps = n.snapshots

    result = {"FR_to_ES": pd.Series(0.0, index=snaps),
              "PT_to_ES": pd.Series(0.0, index=snaps)}

    # Lines (AC interconnectors): p0 flows from bus0 → bus1
    if not n.lines_t.p0.empty:
        for line, row in n.lines.iterrows():
            b0, b1 = row["bus0"], row["bus1"]
            if line not in n.lines_t.p0.columns:
                continue
            flow = n.lines_t.p0[line].reindex(snaps).fillna(0.0)
            # Determine direction: positive = b0→b1
            if b0.startswith("FR") and b1.startswith("ES"):
                result["FR_to_ES"] += flow
            elif b0.startswith("ES") and b1.startswith("FR"):
                result["FR_to_ES"] -= flow
            elif b0.startswith("PT") and b1.startswith("ES"):
                result["PT_to_ES"] += flow
            elif b0.startswith("ES") and b1.startswith("PT"):
                result["PT_to_ES"] -= flow

    # Links (HVDC / DC links)
    if not n.links_t.p0.empty:
        for link, row in n.links.iterrows():
            b0, b1 = row["bus0"], row["bus1"]
            if link not in n.links_t.p0.columns:
                continue
            flow = n.links_t.p0[link].reindex(snaps).fillna(0.0)
            if b0.startswith("FR") and b1.startswith("ES"):
                result["FR_to_ES"] += flow
            elif b0.startswith("ES") and b1.startswith("FR"):
                result["FR_to_ES"] -= flow
            elif b0.startswith("PT") and b1.startswith("ES"):
                result["PT_to_ES"] += flow
            elif b0.startswith("ES") and b1.startswith("PT"):
                result["PT_to_ES"] -= flow

    return result


def _ccgt_commit_hours(n, snaps):
    """Count hours where at least one ES CCGT is committed (MIP status=1)."""
    if "status" not in n.generators_t:
        return None
    status = n.generators_t["status"]
    ccgt_cols = [
        g for g in status.columns
        if n.generators.at[g, "carrier"] in ("CCGT", "CCGT_flex")
        and n.generators.at[g, "bus"].startswith("ES")
    ]
    if not ccgt_cols:
        return None
    s = status.loc[snaps, ccgt_cols].reindex(snaps)
    committed_hours = int((s.sum(axis=1) > 0).sum())
    n_committed_per_hour = s.sum(axis=1)
    return {
        "committed_hours": committed_hours,
        "mean_units_on": float(n_committed_per_hour.mean()),
        "max_units_on": int(n_committed_per_hour.max()),
    }


def _dispatch_mix(n, snaps):
    """Return dispatch fraction by carrier group for ES, clipped to snapshots."""
    disp = _dispatch_by_carrier(n, "ES").reindex(snaps).fillna(0.0)
    total = disp.sum().sum()
    if total <= 0:
        return {}

    groups = {
        "nuclear": ["nuclear"],
        "VRE":     ["onwind", "offwind", "solar"],
        "hydro":   ["hydro", "ror", "PHS", "PHS_new", "hydro_storage"],
        "CCGT":    ["CCGT", "CCGT_flex", "CCGT_must_run"],
        "OCGT":    ["OCGT", "OCGT_pk"],
        "other":   [],   # catch-all below
    }
    used = set()
    out = {}
    for grp, carriers in groups.items():
        cols = [c for c in disp.columns if c in carriers]
        used.update(cols)
        out[grp] = float(disp[cols].sum().sum()) / total * 100 if cols else 0.0

    leftover = [c for c in disp.columns if c not in used]
    out["other"] = float(disp[leftover].sum().sum()) / total * 100 if leftover else 0.0
    return out


# ── Weekly loop printer ───────────────────────────────────────────────────────

def print_weekly_loop(n, omie_series, cfg):
    """
    Loop through weeks in the solved network and print a formatted diagnostic
    table for each. Covers: price stats, spike hours, CCGT commitment (MIP),
    IC flows, dispatch mix, OMIE comparison.
    """
    snaps = n.snapshots
    start_ts = snaps[0]
    end_ts   = snaps[-1]

    weeks = pd.date_range(start_ts, end_ts, freq="W-MON", tz=start_ts.tzinfo)
    if len(weeks) == 0 or weeks[0] > start_ts:
        weeks = pd.DatetimeIndex([start_ts]) .append(weeks)
    week_bounds = [(weeks[i], weeks[i + 1] if i + 1 < len(weeks) else end_ts + pd.Timedelta(hours=1))
                   for i in range(len(weeks))]

    # Pre-compute IC flows for full period
    ic_full = _ic_flows(n)

    print(f"\n{_SEP}")
    print(_bold(f"  WEEK-BY-WEEK DIAGNOSTIC LOOP"))
    print(f"  Period: {start_ts.date()} → {end_ts.date()}  ({len(snaps)} snapshots)")
    print(_SEP)

    # Header row
    hdr = (f"  {'Week':<6}  {'Date range':<12}  "
           f"{'Mdl':>6}  {'OMIE':>6}  {'Bias':>6}  {'P95m':>5}  {'P95o':>5}  {'Max':>6}  "
           f"{'≥100h':>5}  {'≥200h':>5}  "
           f"{'FR→ES':>6}  {'PT→ES':>6}  "
           f"{'CCGT%':>5}  {'VRE%':>5}  {'NUC%':>5}  {'nCmt':>5}")
    print(hdr)
    print("  " + "─" * (_W - 2))

    for w_idx, (w_start, w_end) in enumerate(week_bounds, 1):
        w_snaps = snaps[(snaps >= w_start) & (snaps < w_end)]
        if len(w_snaps) == 0:
            continue

        mp = _es_price_series(n, w_snaps)
        omie_w = omie_series.reindex(w_snaps).dropna()

        mean_m  = mp.mean()
        mean_o  = omie_w.mean()
        bias    = mean_m - mean_o if not (np.isnan(mean_m) or np.isnan(mean_o)) else float("nan")
        p95_m   = float(np.percentile(mp,      95)) if len(mp)     > 0 else float("nan")
        p95_o   = float(np.percentile(omie_w,  95)) if len(omie_w) > 0 else float("nan")
        max_m   = float(mp.max()) if len(mp) > 0 else float("nan")

        sp100   = int((mp > 100).sum())
        sp200   = int((mp > 200).sum())

        fr_flow = ic_full["FR_to_ES"].reindex(w_snaps).fillna(0.0)
        pt_flow = ic_full["PT_to_ES"].reindex(w_snaps).fillna(0.0)
        mean_fr = float(fr_flow.mean())
        mean_pt = float(pt_flow.mean())

        mix     = _dispatch_mix(n, w_snaps)
        ccgt_pct = mix.get("CCGT", 0.0)
        vre_pct  = mix.get("VRE",  0.0)
        nuc_pct  = mix.get("nuclear", 0.0)

        commit = _ccgt_commit_hours(n, w_snaps)
        n_cmt  = commit["committed_hours"] if commit else -1

        # Colour price bias
        bias_str  = f"{bias:+.1f}" if not np.isnan(bias) else "  n/a"
        if not np.isnan(bias):
            bias_str = _green(bias_str) if abs(bias) < 5 else _yellow(bias_str) if abs(bias) < 15 else _red(bias_str)

        max_str   = _red(f"{max_m:>6.0f}") if max_m > 200 else f"{max_m:>6.0f}"
        sp200_str = _red(f"{sp200:>5d}") if sp200 > 0 else f"{sp200:>5d}"
        fr_str    = _red(f"{mean_fr:>6.0f}") if mean_fr < 0 else f"{mean_fr:>6.0f}"

        date_range = f"{w_start.strftime('%m/%d')}–{(w_end - pd.Timedelta(hours=1)).strftime('%m/%d')}"
        cmt_str    = f"{n_cmt:>5d}" if n_cmt >= 0 else "  MIP-"

        line = (
            f"  {w_idx:<6d}  {date_range:<12}  "
            f"{mean_m:>6.1f}  {mean_o:>6.1f}  {bias_str}  {p95_m:>5.0f}  {p95_o:>5.0f}  {max_str}  "
            f"{sp100:>5d}  {sp200_str}  "
            f"{fr_str}  {mean_pt:>6.0f}  "
            f"{ccgt_pct:>5.1f}  {vre_pct:>5.1f}  {nuc_pct:>5.1f}  {cmt_str}"
        )
        print(line)

    print(_SEP)
    print(f"  Columns: Mdl/OMIE=mean price, Bias=model−OMIE, P95m/P95o=95th pct,")
    print(f"           Max=model peak, ≥100h/≥200h=spike hours, FR→ES/PT→ES=mean net MW,")
    print(f"           CCGT%/VRE%/NUC%=dispatch share, nCmt=hours any CCGT committed (MIP)")
    print(f"  Colours: {_green('green bias')}<€5  {_yellow('yellow')}<€15  {_red('red')}≥€15 or spike/reverse-flow")
    print(_SEP)


def print_price_spike_detail(n, omie_series, threshold=150):
    """Print a table of individual hours where price exceeded threshold."""
    mp = _es_price_series(n)
    spike_hours = mp[mp > threshold].sort_values(ascending=False)
    if spike_hours.empty:
        print(f"\n  No model hours with price > €{threshold}/MWh")
        return

    print(f"\n{_SEP2}")
    print(f"  TOP PRICE SPIKE HOURS  (model > €{threshold}/MWh)  — showing up to 30")
    print(_SEP2)
    print(f"  {'Timestamp':<22}  {'Model':>7}  {'OMIE':>7}  {'Δ':>7}  {'Notes'}")
    print(f"  {'─'*62}")
    for ts, m_pr in spike_hours.head(30).items():
        o_pr  = omie_series.get(ts, float("nan"))
        delta = m_pr - o_pr if not np.isnan(o_pr) else float("nan")
        delta_str = f"{delta:>+7.1f}" if not np.isnan(delta) else "    n/a"
        flag = ""
        if m_pr > 500:
            flag = " ← VOLL?"
        elif m_pr > 200:
            flag = " ← scarcity"
        print(f"  {str(ts):<22}  {m_pr:>7.1f}  {o_pr:>7.1f}  {delta_str}  {flag}")
    if len(spike_hours) > 30:
        print(f"  ... and {len(spike_hours) - 30} more hours")
    print(_SEP2)


def print_ic_balance_summary(n, cfg):
    """Print interconnector flow summary vs actual ENTSOE data."""
    ic = _ic_flows(n)
    fr_model = ic["FR_to_ES"]
    pt_model = ic["PT_to_ES"]

    print(f"\n{_SEP2}")
    print("  INTERCONNECTOR FLOWS  (positive = import INTO Spain)")
    print(_SEP2)

    real_fr_path = ROOT / cfg["validation"].get("real_flows_fr_csv", "")
    real_pt_path = ROOT / cfg["validation"].get("real_flows_pt_csv", "")

    for label, model_ts, real_path in [
        ("FR↔ES", fr_model, real_fr_path),
        ("PT↔ES", pt_model, real_pt_path),
    ]:
        print(f"\n  {label}")
        print(f"    Model mean:   {model_ts.mean():>+8.0f} MW")
        print(f"    Model →ES h:  {(model_ts > 0).sum():>5d} ({(model_ts > 0).mean()*100:.1f}%)")
        print(f"    Model ←ES h:  {(model_ts < 0).sum():>5d} ({(model_ts < 0).mean()*100:.1f}%)")

        if real_path.exists():
            try:
                rdf = pd.read_csv(real_path, parse_dates=[0], index_col=0)
                snaps = n.snapshots
                if label.startswith("FR"):
                    net_real = (rdf.get("FR_to_ES", 0) - rdf.get("ES_to_FR", 0))
                else:
                    net_real = (rdf.get("PT_to_ES", 0) - rdf.get("ES_to_PT", 0))
                net_real = net_real.reindex(snaps, method="nearest").fillna(0.0)
                print(f"    Actual mean:  {net_real.mean():>+8.0f} MW")
                print(f"    Actual →ES h: {(net_real > 0).sum():>5d} ({(net_real > 0).mean()*100:.1f}%)")
                print(f"    Actual ←ES h: {(net_real < 0).sum():>5d} ({(net_real < 0).mean()*100:.1f}%)")
            except Exception as e:
                print(f"    [actual data unavailable: {e}]")
    print(_SEP2)


# ── Extended diagnostics (D1–D5 + supply/inflow plots) ───────────────────────

def _out_dir():
    return ROOT / "Analysis" / "validation_output"


def _es_load_series(n, snaps):
    """Return total ES load MW time series for the given snapshots."""
    es_loads = n.loads.index[n.loads["bus"].str.startswith("ES")]
    load_t = pd.Series(0.0, index=snaps)
    ts = n.loads_t.p_set
    if not ts.empty:
        ts_cols = [c for c in es_loads if c in ts.columns]
        if ts_cols:
            load_t += ts[ts_cols].sum(axis=1).reindex(snaps).fillna(0.0)
    static_cols = [c for c in es_loads if c not in (ts.columns if not ts.empty else [])]
    if static_cols:
        load_t += n.loads.loc[static_cols, "p_set"].sum()
    return load_t


def _p_nom(n, component="generators"):
    """Return p_nom_opt if present, else p_nom."""
    df = getattr(n, component)
    return df["p_nom_opt"] if "p_nom_opt" in df.columns else df["p_nom"]


def print_mcq_uplift_audit(n, omie_series):
    """D1: Fleet-weighted MCQ marginal cost uplift for ES CCGTs."""
    mcq_col = "marginal_cost_quadratic"
    if mcq_col not in n.generators.columns:
        print(f"\n  [D1 MCQ audit] Column '{mcq_col}' absent — MCQ not applied.")
        return

    es_ccgt = n.generators.index[
        n.generators["carrier"].isin(["CCGT", "CCGT_flex"]) &
        n.generators["bus"].str.startswith("ES")
    ]
    with_mcq = es_ccgt[n.generators.loc[es_ccgt, mcq_col] > 0]
    if with_mcq.empty:
        print(f"\n  [D1 MCQ audit] All ES CCGT marginal_cost_quadratic = 0 — MCQ inactive.")
        return

    snaps = n.snapshots
    p_nom_s = _p_nom(n, "generators")

    weighted_num = pd.Series(0.0, index=snaps)
    weighted_den = pd.Series(0.0, index=snaps)
    for g in with_mcq:
        alpha = float(n.generators.at[g, mcq_col])
        pn    = float(p_nom_s.at[g])
        if pn <= 0:
            continue
        p_t = (n.generators_t.p[g] if g in n.generators_t.p.columns
               else pd.Series(0.0, index=snaps)).reindex(snaps).fillna(0.0)
        uplift = 2.0 * alpha * p_t / pn
        weighted_num += uplift * p_t
        weighted_den += p_t

    active = weighted_den > 0.1
    fleet_uplift = weighted_num[active] / weighted_den[active]

    mp        = _es_price_series(n, snaps)
    price_err = (mp - omie_series.reindex(snaps)).dropna()
    common    = fleet_uplift.index.intersection(price_err.index)

    print(f"\n{_SEP2}")
    print("  D1 · MCQ UPLIFT AUDIT  (ES CCGT fleet-weighted Δ from quadratic term)")
    print(_SEP2)
    print(f"  Generators with MCQ active : {len(with_mcq)} / {len(es_ccgt)}")
    print(f"  CCGT dispatch hours        : {active.sum()} / {len(snaps)}")
    if len(fleet_uplift) > 0:
        q = fleet_uplift.quantile([0.10, 0.50, 0.90])
        print(f"  Uplift (dispatched hrs)    : mean {fleet_uplift.mean():.1f}  "
              f"P10 {q[0.10]:.1f}  P50 {q[0.50]:.1f}  P90 {q[0.90]:.1f}  €/MWh")
        if len(common) > 10:
            corr = fleet_uplift.reindex(common).corr(price_err.reindex(common))
            print(f"  Corr(MCQ uplift, Δprice)   : {corr:+.3f}"
                  + ("  ← MCQ is primary inflation driver" if corr > 0.5 else ""))
    print(f"  Reference: CCGT-hours price gap = +17.1 €/MWh")
    print(_SEP2)


def print_monthly_stack(n, omie_series):
    """D2: Monthly supply-demand balance table — prints and returns DataFrame."""
    snaps     = n.snapshots
    dispatch  = _dispatch_by_carrier(n, "ES").reindex(snaps).fillna(0.0)
    ic        = _ic_flows(n)
    fr_imp    = ic["FR_to_ES"].reindex(snaps).fillna(0.0).clip(lower=0)
    pt_imp    = ic["PT_to_ES"].reindex(snaps).fillna(0.0).clip(lower=0)
    load_t    = _es_load_series(n, snaps)

    must_run_t = dispatch[[c for c in dispatch.columns if c in ("nuclear", "CCGT_must_run")]].sum(axis=1)
    vre_t      = dispatch[[c for c in dispatch.columns if c in ("onwind", "offwind", "solar")]].sum(axis=1)
    hydro_t    = dispatch[[c for c in dispatch.columns if c in ("hydro", "ror")]].sum(axis=1)
    ccgt_t     = dispatch[[c for c in dispatch.columns
                            if "CCGT" in c and c != "CCGT_must_run"]].sum(axis=1)
    net_imp    = fr_imp + pt_imp

    mp     = _es_price_series(n, snaps)
    err_t  = (mp - omie_series.reindex(snaps))
    months = snaps.to_period("M").unique()

    rows = []
    for p in months:
        mask = snaps.to_period("M") == p
        sm   = snaps[mask]
        rows.append({
            "month":    str(p),
            "load":     load_t.reindex(sm).mean(),
            "must_run": must_run_t.reindex(sm).mean(),
            "VRE":      vre_t.reindex(sm).mean(),
            "hydro":    hydro_t.reindex(sm).mean(),
            "imports":  net_imp.reindex(sm).mean(),
            "CCGT":     ccgt_t.reindex(sm).mean(),
            "err":      err_t.reindex(sm).mean(),
        })
    df = pd.DataFrame(rows).set_index("month")
    df["ccgt_need"] = df["load"] - df["must_run"] - df["VRE"] - df["hydro"] - df["imports"]

    print(f"\n{_SEP2}")
    print("  D2 · MONTHLY SUPPLY-DEMAND STACK  (mean MW)")
    print(_SEP2)
    hdr = (f"  {'Month':<8}  {'Load':>7}  {'MustRun':>8}  {'VRE':>7}  "
           f"{'Hydro':>7}  {'Imports':>8}  {'CCGT':>7}  {'CCGTneed':>9}  {'PriceErr':>9}")
    print(hdr)
    print("  " + "─" * 80)
    for month, row in df.iterrows():
        es = f"{row['err']:>+9.1f}"
        if abs(row['err']) > 20: es = _red(es)
        elif abs(row['err']) > 10: es = _yellow(es)
        print(f"  {month:<8}  {row['load']:>7.0f}  {row['must_run']:>8.0f}  {row['VRE']:>7.0f}  "
              f"{row['hydro']:>7.0f}  {row['imports']:>8.0f}  {row['CCGT']:>7.0f}  "
              f"{row['ccgt_need']:>9.0f}  {es}")
    print(_SEP2)
    return df


def plot_monthly_supply_demand(n, omie_series, stack_df, out_dir):
    """User request 1: Stacked monthly supply bar chart + price comparison panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    snaps    = n.snapshots
    dispatch = _dispatch_by_carrier(n, "ES").reindex(snaps).fillna(0.0)
    ic       = _ic_flows(n)
    fr_imp   = ic["FR_to_ES"].reindex(snaps).fillna(0.0).clip(lower=0)
    pt_imp   = ic["PT_to_ES"].reindex(snaps).fillna(0.0).clip(lower=0)
    load_t   = _es_load_series(n, snaps)
    mp       = _es_price_series(n, snaps)
    omie_r   = omie_series.reindex(snaps)
    months   = snaps.to_period("M").unique()
    x        = np.arange(len(months))

    groups  = [
        ("Nuclear",    ["nuclear"],                          "#4e79a7"),
        ("VRE",        ["onwind", "offwind", "solar"],       "#f28e2b"),
        ("Hydro",      ["hydro", "ror"],                     "#59a14f"),
        ("PHS",        ["PHS", "PHS_new"],                   "#76b7b2"),
        ("CCGT",       ["CCGT", "CCGT_flex", "CCGT_must_run"], "#e15759"),
        ("OCGT",       ["OCGT", "OCGT_pk"],                  "#ff9da7"),
        ("Coal/other", ["coal", "lignite", "oil"],           "#9c755f"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11))

    def _monthly_mean(series):
        return np.array([series.reindex(snaps[snaps.to_period("M") == p]).mean()
                         for p in months])

    bottom = np.zeros(len(months))
    for label, cars, color in groups:
        cols = [c for c in dispatch.columns if c in cars]
        if not cols:
            continue
        vals = _monthly_mean(dispatch[cols].sum(axis=1))
        ax1.bar(x, vals, bottom=bottom, label=label, color=color, width=0.55)
        bottom += vals

    for label, flow, color in [("FR import", fr_imp, "#b07aa1"), ("PT import", pt_imp, "#d4a6c8")]:
        vals = _monthly_mean(flow)
        ax1.bar(x, vals, bottom=bottom, label=label, color=color, width=0.55)
        bottom += vals

    ax1.plot(x, _monthly_mean(load_t), "k-o", lw=2, ms=5, zorder=10, label="ES Load")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(p) for p in months], rotation=45, ha="right")
    ax1.set_ylabel("Mean MW")
    ax1.set_title("Monthly Supply Stack vs ES Load (2024 model)")
    ax1.legend(loc="upper right", fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f} GW"))

    w = 0.35
    model_v = _monthly_mean(mp)
    omie_v  = _monthly_mean(omie_r)
    ax2.bar(x - w/2, model_v, width=w, label="Model", color="#e15759", alpha=0.85)
    ax2.bar(x + w/2, omie_v,  width=w, label="OMIE",  color="#4e79a7", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.5)
    for i, (xi, ev) in enumerate(zip(x, model_v - omie_v)):
        ax2.text(xi, max(model_v[i], omie_v[i]) + 1.5, f"{ev:+.0f}",
                 ha="center", va="bottom", fontsize=7,
                 color="red" if abs(ev) > 15 else "#555")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(p) for p in months], rotation=45, ha="right")
    ax2.set_ylabel("Mean price (€/MWh)")
    ax2.set_title("Monthly Mean Price — Model vs OMIE (error annotated)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / "monthly_supply_demand.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved: {out_path.name}]")


def plot_hydro_soc_trajectory(n, omie_series, out_dir):
    """D3: Weekly ES hydro SOC % vs price error (with inflow overlay if available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    soc_t = n.storage_units_t.get("state_of_charge", pd.DataFrame())
    if soc_t.empty:
        print("\n  [D3 hydro SOC] No SOC data — skipping plot.")
        return

    snaps   = n.snapshots
    es_mask = (n.storage_units["carrier"] == "hydro") & n.storage_units["bus"].str.startswith("ES")
    es_su   = n.storage_units.index[es_mask]
    soc_cols = [u for u in es_su if u in soc_t.columns]
    if not soc_cols:
        print("\n  [D3 hydro SOC] No ES hydro units with SOC data.")
        return

    p_nom_s = _p_nom(n, "storage_units")
    e_cap   = (p_nom_s.loc[soc_cols] * n.storage_units.loc[soc_cols, "max_hours"]).sum()
    soc_pct = soc_t[soc_cols].sum(axis=1).reindex(snaps).fillna(0.0) / e_cap * 100
    soc_w   = soc_pct.resample("W").mean()

    inflow_t  = n.storage_units_t.get("inflow", pd.DataFrame())
    es_inf    = [u for u in es_su if not inflow_t.empty and u in inflow_t.columns]
    inf_w     = (inflow_t[es_inf].sum(axis=1).resample("W").mean() / 1000
                 if es_inf else None)

    mp      = _es_price_series(n, snaps)
    err_w   = (mp - omie_series.reindex(snaps)).resample("W").mean()
    err_aln = err_w.reindex(soc_w.index, method="nearest")

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.fill_between(soc_w.index, soc_w.values, alpha=0.30, color="#4e79a7")
    ax1.plot(soc_w.index, soc_w.values, color="#4e79a7", lw=2, label="ES Hydro SOC (%)")
    ax1.set_ylabel("SOC (%)", color="#4e79a7")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="y", labelcolor="#4e79a7")

    ax2 = ax1.twinx()
    bar_c = ["#e15759" if v > 0 else "#4e79a7" for v in err_aln.fillna(0)]
    ax2.bar(soc_w.index, err_aln.values, width=5, color=bar_c, alpha=0.45, label="Price error")
    ax2.axhline(0, color="black", lw=0.5, ls="--")
    ax2.set_ylabel("Price error (€/MWh)", color="#e15759")
    ax2.tick_params(axis="y", labelcolor="#e15759")

    if inf_w is not None:
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.10))
        ax3.plot(inf_w.index, inf_w.values, color="#59a14f", lw=1.5, ls="--", label="ES Inflow (GW avg)")
        ax3.set_ylabel("Inflow (GW avg)", color="#59a14f")
        ax3.tick_params(axis="y", labelcolor="#59a14f")

    ax1.set_title("ES Hydro — SOC trajectory vs price error (weekly)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = out_dir / "hydro_soc_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved: {out_path.name}]")


def plot_hydro_inflow_fr_pt(n, out_dir):
    """User request 2: Monthly FR and PT hydro inflow vs dispatch from solved network."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inflow_t = n.storage_units_t.get("inflow", pd.DataFrame())
    if inflow_t.empty:
        print("\n  [FR/PT inflow plot] No inflow data in network — skipping.")
        return

    snaps  = n.snapshots
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    any_plotted = False

    for ax, ctry, col_in, col_d in [
        (axes[0], "FR", "#4e79a7", "#aec7e8"),
        (axes[1], "PT", "#59a14f", "#b5cf8e"),
    ]:
        ctry_mask = (
            n.storage_units["carrier"].isin(["hydro", "ror", "PHS"]) &
            n.storage_units["bus"].str.startswith(ctry)
        )
        units     = n.storage_units.index[ctry_mask]
        inf_cols  = [u for u in units if u in inflow_t.columns]
        disp_cols = [u for u in units if u in n.storage_units_t.p.columns]

        if not inf_cols:
            ax.text(0.5, 0.5, f"No {ctry} hydro inflow in solved network",
                    transform=ax.transAxes, ha="center", va="center", fontsize=11)
            ax.set_title(f"{ctry} Hydro Inflow vs Dispatch")
            continue

        total_inf  = inflow_t[inf_cols].sum(axis=1).reindex(snaps).fillna(0.0)
        total_disp = (n.storage_units_t.p[disp_cols].sum(axis=1).reindex(snaps).fillna(0.0)
                      if disp_cols else pd.Series(0.0, index=snaps))

        m_inf  = total_inf.resample("ME").sum() / 1000
        m_disp = total_disp.resample("ME").sum() / 1000

        x = np.arange(len(m_inf))
        w = 0.38
        ax.bar(x - w/2, m_inf.values,  width=w, label="Inflow",   color=col_in, alpha=0.85)
        ax.bar(x + w/2, m_disp.values, width=w, label="Dispatch", color=col_d,  alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([str(d)[:7] for d in m_inf.index], rotation=45, ha="right")
        ax.set_ylabel("GWh / month")
        ax.set_title(f"{ctry} Hydro — Monthly Inflow vs Dispatch (from solved network)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return

    fig.tight_layout()
    out_path = out_dir / "hydro_inflow_fr_pt.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved: {out_path.name}]")


def print_fr_nuclear_ic_monthly(n, omie_series):
    """D4: FR nuclear utilisation and IC direction by month."""
    snaps = n.snapshots

    fr_nuc_mask = (n.generators["carrier"] == "nuclear") & n.generators["bus"].str.startswith("FR")
    fr_nuc_gens = n.generators.index[fr_nuc_mask]
    if len(fr_nuc_gens) == 0:
        print("\n  [D4] No FR nuclear generators found.")
        return

    p_nom_s      = _p_nom(n, "generators")
    fr_nuc_p_nom = p_nom_s.loc[fr_nuc_gens].sum()
    nuc_cols     = [g for g in fr_nuc_gens if g in n.generators_t.p.columns]
    fr_nuc_t     = n.generators_t.p.reindex(columns=nuc_cols).sum(axis=1).reindex(snaps).fillna(0.0)

    mp_t    = n.buses_t.get("marginal_price", pd.DataFrame())
    fr_cols = [b for b in n.buses.index if b.startswith("FR") and not mp_t.empty and b in mp_t.columns]
    fr_prices = (mp_t[fr_cols].mean(axis=1).reindex(snaps)
                 if fr_cols else pd.Series(float("nan"), index=snaps))

    ic     = _ic_flows(n)
    fr_flow = ic["FR_to_ES"].reindex(snaps).fillna(0.0)
    mp      = _es_price_series(n, snaps)
    omie_r  = omie_series.reindex(snaps)
    months  = snaps.to_period("M").unique()

    print(f"\n{_SEP2}")
    print(f"  D4 · FR NUCLEAR & IC DIRECTION BY MONTH   (FR p_nom: {fr_nuc_p_nom:.0f} MW)")
    print(_SEP2)
    print(f"  {'Month':<8}  {'FR nuc MW':>10}  {'CF':>6}  {'FR→ES MW':>9}  "
          f"{'ES→FR h':>8}  {'FR→ES h':>8}  {'FR price':>9}  {'ES err':>8}")
    print("  " + "─" * 82)
    for period in months:
        mask   = snaps.to_period("M") == period
        snap_m = snaps[mask]
        nm     = fr_nuc_t.reindex(snap_m).mean()
        cf     = nm / fr_nuc_p_nom if fr_nuc_p_nom > 0 else 0.0
        fm     = fr_flow.reindex(snap_m).mean()
        es_fr  = int((fr_flow.reindex(snap_m) < -50).sum())
        fr_es  = int((fr_flow.reindex(snap_m) > 50).sum())
        frp    = fr_prices.reindex(snap_m).mean()
        err    = (mp.reindex(snap_m) - omie_r.reindex(snap_m)).mean()
        es_s   = f"{err:>+8.1f}"
        if abs(err) > 20: es_s = _red(es_s)
        elif abs(err) > 10: es_s = _yellow(es_s)
        print(f"  {str(period):<8}  {nm:>10.0f}  {cf:>6.3f}  {fm:>9.0f}  "
              f"{es_fr:>8d}  {fr_es:>8d}  {frp:>9.1f}  {es_s}")
    print(_SEP2)


def print_vre_cf_monthly(n):
    """D5: Monthly VRE capacity factors for ES and FR."""
    snaps     = n.snapshots
    months    = snaps.to_period("M").unique()
    p_nom_s   = _p_nom(n, "generators")
    months_s  = [str(m) for m in months]

    rows = []
    for ctry in ["ES", "FR"]:
        for carrier in ["onwind", "offwind", "solar"]:
            mask = (n.generators["carrier"] == carrier) & n.generators["bus"].str.startswith(ctry)
            gens = n.generators.index[mask]
            if len(gens) == 0:
                continue
            pn_sum = p_nom_s.loc[gens].sum()
            if pn_sum <= 0:
                continue
            disp_cols = [g for g in gens if g in n.generators_t.p.columns]
            disp_t    = (n.generators_t.p.reindex(columns=disp_cols)
                         .sum(axis=1).reindex(snaps).fillna(0.0))
            row = {"country": ctry, "carrier": carrier}
            for p in months:
                mask_m = snaps.to_period("M") == p
                row[str(p)] = float(disp_t.reindex(snaps[mask_m]).mean() / pn_sum)
            rows.append(row)

    if not rows:
        print("\n  [D5] No VRE data found.")
        return

    df = pd.DataFrame(rows)
    print(f"\n{_SEP2}")
    print("  D5 · VRE CAPACITY FACTORS BY MONTH  (mean dispatch / p_nom)")
    print(_SEP2)
    hdr = f"  {'Ctry':<5}  {'Carrier':<8}" + "".join(f"  {m:>7}" for m in months_s)
    print(hdr)
    print("  " + "─" * (19 + len(months_s) * 9))
    for _, row in df.iterrows():
        vals = "".join(f"  {row.get(m, 0):>7.3f}" for m in months_s)
        print(f"  {row['country']:<5}  {row['carrier']:<8}{vals}")
    print(_SEP2)


def print_ccgt_formation(n, omie_series):
    """CCGT price formation: monthly residual demand decomposition.

    Shows why gas still sets ~80% of prices by decomposing the supply stack
    and measuring how much CCGT is structurally needed each month.
    Also shows the IC distortion (model FR flow vs actual 344 MW mean).
    """
    from run_validation import _get_price_setter_series

    snaps    = n.snapshots
    months   = snaps.to_period("M").unique()
    gp       = n.generators_t.p if not n.generators_t.p.empty else pd.DataFrame(index=snaps)
    p_nom_s  = _p_nom(n, "generators")

    # ── ES load ────────────────────────────────────────────────────────────────
    load_t   = _es_load_series(n, snaps)

    # ── Must-run (nuclear + biomass + CCGT_must_run) ───────────────────────────
    must_carriers = {"nuclear", "biomass", "CCGT_must_run"}
    mr_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(must_carriers)
    ]
    mr_cols = [g for g in mr_gens if g in gp.columns]
    must_t  = gp[mr_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if mr_cols else pd.Series(0.0, index=snaps)

    # ── VRE dispatch (ES) ──────────────────────────────────────────────────────
    vre_carriers = {"onwind", "offwind-ac", "offwind-dc", "offwind-float", "solar"}
    vre_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(vre_carriers)
    ]
    vre_cols = [g for g in vre_gens if g in gp.columns]
    vre_t    = gp[vre_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if vre_cols else pd.Series(0.0, index=snaps)

    # ── ES hydro dispatch (reservoir + ror, not CCGT) ─────────────────────────
    hyd_carriers = {"hydro", "ror"}
    hyd_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(hyd_carriers)
    ]
    hyd_cols = [g for g in hyd_gens if g in gp.columns]
    hyd_t    = gp[hyd_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if hyd_cols else pd.Series(0.0, index=snaps)
    # Add storage-unit hydro dispatch
    su_p = getattr(n.storage_units_t, "p", pd.DataFrame())
    es_hsu = [g for g in n.storage_units.index
              if n.storage_units.at[g, "carrier"] == "hydro"
              and n.storage_units.at[g, "bus"].startswith("ES")
              and g in su_p.columns]
    if es_hsu:
        hyd_t = hyd_t + su_p[es_hsu].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0)

    # ── FR / PT net import series ──────────────────────────────────────────────
    ic    = _ic_flows(n, snaps)
    fr_t  = ic["FR_to_ES"]
    pt_t  = ic["PT_to_ES"]

    # ── ES CCGT dispatch ───────────────────────────────────────────────────────
    gas_carriers = {"CCGT", "CCGT_flex", "OCGT"}
    gas_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(gas_carriers)
    ]
    gas_cols = [g for g in gas_gens if g in gp.columns]
    gas_t    = gp[gas_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if gas_cols else pd.Series(0.0, index=snaps)

    # ── Price setter ───────────────────────────────────────────────────────────
    _, setter_t = _get_price_setter_series(n, "ES")
    gas_setter_mask = setter_t.isin({"CCGT", "CCGT_flex", "OCGT"})

    # ── Residual = load − must_run − VRE − hydro − FR_net − PT_net ────────────
    residual_t = load_t - must_t - vre_t - hyd_t - fr_t - pt_t

    # ── Actual FR mean for counterfactual ─────────────────────────────────────
    FR_ACTUAL_MEAN = 344.0   # MW, 2024 ENTSOE mean

    print(f"\n{_SEP2}")
    print("  CCGT PRICE FORMATION — WHY GAS IS MARGINAL")
    print(_SEP2)
    print(f"  Residual = load − must_run − VRE − ES_hydro − FR_net − PT_net")
    print(f"  When Residual > 0, CCGT must dispatch to balance the system.\n")

    hdr = (f"  {'Month':<8}  {'Load':>6}  {'MustRun':>7}  {'VRE':>6}  "
           f"{'Hydro':>6}  {'FRnet':>6}  {'Resid':>6}  {'GasHrs%':>7}  {'GasMW':>6}  {'Error':>6}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    total_gas_hrs = 0
    total_hrs = 0
    for p in months:
        mask_m  = snaps.to_period("M") == p
        sl      = snaps[mask_m]
        n_hrs   = int(mask_m.sum())
        if n_hrs == 0:
            continue

        load_m  = float(load_t.reindex(sl).mean())
        mr_m    = float(must_t.reindex(sl).mean())
        vre_m   = float(vre_t.reindex(sl).mean())
        hyd_m   = float(hyd_t.reindex(sl).mean())
        fr_m    = float(fr_t.reindex(sl).mean())
        res_m   = float(residual_t.reindex(sl).mean())
        gas_hrs = int((residual_t.reindex(sl) > 100.0).sum())
        gas_pct = gas_hrs / n_hrs * 100
        gas_m   = float(gas_t.reindex(sl).mean())
        total_gas_hrs += gas_hrs
        total_hrs     += n_hrs

        err_m   = float((omie_series.reindex(sl)).mean()) if omie_series is not None else float("nan")
        mp_m    = float(n.buses_t.marginal_price[[b for b in n.buses_t.marginal_price.columns
                                                   if b.startswith("ES")]].mean(axis=1).reindex(sl).mean())
        err_str = f"{mp_m - err_m:+.0f}" if not pd.isna(err_m) else "  n/a"

        print(f"  {str(p):<8}  {load_m/1e3:>5.1f}k  {mr_m/1e3:>6.1f}k  {vre_m/1e3:>5.1f}k  "
              f"{hyd_m/1e3:>5.1f}k  {fr_m/1e3:>+5.1f}k  {res_m/1e3:>5.1f}k  "
              f"{gas_pct:>6.0f}%  {gas_m/1e3:>5.1f}k  {err_str:>6}")

    print("  " + "─" * (len(hdr) - 2))
    full_pct = total_gas_hrs / max(1, total_hrs) * 100
    print(f"  {'Full yr':<8}  {'':>6}  {'':>7}  {'':>6}  {'':>6}  {'':>6}  {'':>6}  "
          f"{full_pct:>6.0f}%")

    # ── Counterfactual: if FR imports were at actual (344 MW) ─────────────────
    fr_model_mean = float(fr_t.mean())
    fr_delta      = fr_model_mean - FR_ACTUAL_MEAN
    res_counterfactual = residual_t + fr_delta   # less supply if FR reduced
    cf_gas_hrs    = int((res_counterfactual > 100.0).sum())
    cf_gas_pct    = cf_gas_hrs / max(1, len(snaps)) * 100

    print(f"\n  IC COUNTERFACTUAL")
    print(f"  FR model mean flow : {fr_model_mean:+.0f} MW  (8× actual {FR_ACTUAL_MEAN:.0f} MW)")
    print(f"  If FR at actual   → residual rises by {fr_delta:.0f} MW mean")
    print(f"                    → CCGT needed {cf_gas_pct:.0f}% of hours (vs {full_pct:.0f}% now)")

    # ── VRE threshold to displace last CCGT ───────────────────────────────────
    # Hours where VRE alone (ignoring hydro + FR) > load - must_run
    flex_residual  = load_t - must_t - hyd_t - fr_t - pt_t   # what VRE needs to cover
    vre_surplus_t  = vre_t - flex_residual
    vre_sets_theory = int((vre_surplus_t > 0).sum())
    vre_sets_actual = int(setter_t.isin({"onwind", "offwind", "offwind-ac", "offwind-dc",
                                          "offwind-float", "solar"}).sum())
    print(f"\n  VRE PRICE-SETTING")
    print(f"  Hours VRE theoretically beats CCGT (VRE > flex residual): "
          f"{vre_sets_theory} / {len(snaps)}  ({vre_sets_theory/max(1,len(snaps))*100:.1f}%)")
    print(f"  Hours VRE actually sets price:                             "
          f"{vre_sets_actual} / {len(snaps)}  ({vre_sets_actual/max(1,len(snaps))*100:.1f}%)")
    if vre_sets_theory > vre_sets_actual:
        blocked = vre_sets_theory - vre_sets_actual
        print(f"  Blocked hours (theory > actual):  {blocked}  "
              f"— must-run floor, IC congestion or ramp constraints preventing VRE clearing")
    print(_SEP2)


# ── D6: Export-vs-Domestic CCGT dispatch decomposition ────────────────────────

def print_ccgt_export_decomposition(n, omie_series):
    """Decompose CCGT dispatch into domestic-serving vs export-driven components.

    For each hour:
      domestic_supply_before_gas = must_run + VRE + hydro + FR_import + PT_import
      domestic_shortfall = load - domestic_supply_before_gas

      If domestic_shortfall > 0:
        domestic_ccgt = min(shortfall, total_ccgt)
        export_ccgt   = total_ccgt - domestic_ccgt
      Else:
        domestic_ccgt = 0
        export_ccgt   = total_ccgt

    Also computes counterfactual: if FR/PT exports capped at historical means.
    """
    snaps  = n.snapshots
    months = snaps.to_period("M").unique()
    gp     = n.generators_t.p if not n.generators_t.p.empty else pd.DataFrame(index=snaps)
    p_nom_s = _p_nom(n, "generators")

    # ── ES load ────────────────────────────────────────────────────────────────
    load_t = _es_load_series(n, snaps)

    # ── Must-run (nuclear + biomass + CCGT_must_run) ───────────────────────────
    must_carriers = {"nuclear", "biomass", "CCGT_must_run"}
    mr_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(must_carriers)
    ]
    mr_cols = [g for g in mr_gens if g in gp.columns]
    must_t  = gp[mr_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if mr_cols else pd.Series(0.0, index=snaps)

    # ── VRE dispatch (ES) ──────────────────────────────────────────────────────
    vre_carriers = {"onwind", "offwind-ac", "offwind-dc", "offwind-float", "solar"}
    vre_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(vre_carriers)
    ]
    vre_cols = [g for g in vre_gens if g in gp.columns]
    vre_t    = gp[vre_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if vre_cols else pd.Series(0.0, index=snaps)

    # ── ES hydro dispatch (reservoir + ror) ────────────────────────────────────
    hyd_carriers = {"hydro", "ror"}
    hyd_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(hyd_carriers)
    ]
    hyd_cols = [g for g in hyd_gens if g in gp.columns]
    hyd_t    = gp[hyd_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if hyd_cols else pd.Series(0.0, index=snaps)
    # Add storage-unit hydro dispatch
    su_p = getattr(n.storage_units_t, "p", pd.DataFrame())
    es_hsu = [g for g in n.storage_units.index
              if n.storage_units.at[g, "carrier"] == "hydro"
              and n.storage_units.at[g, "bus"].startswith("ES")
              and g in su_p.columns]
    if es_hsu:
        hyd_t = hyd_t + su_p[es_hsu].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0)

    # ── FR / PT net import series ──────────────────────────────────────────────
    ic    = _ic_flows(n, snaps)
    fr_t  = ic["FR_to_ES"]   # positive = import into ES
    pt_t  = ic["PT_to_ES"]

    # ── ES CCGT dispatch ───────────────────────────────────────────────────────
    gas_carriers = {"CCGT", "CCGT_flex", "OCGT"}
    gas_gens = n.generators.index[
        n.generators["bus"].str.startswith("ES") &
        n.generators["carrier"].isin(gas_carriers)
    ]
    gas_cols = [g for g in gas_gens if g in gp.columns]
    gas_t    = gp[gas_cols].clip(lower=0).sum(axis=1).reindex(snaps, fill_value=0.0) if gas_cols else pd.Series(0.0, index=snaps)

    # ── Price setter ───────────────────────────────────────────────────────────
    _, setter_t = _get_price_setter_series(n, "ES")
    gas_setter_mask = setter_t.isin({"CCGT", "CCGT_flex", "OCGT"})

    # ── Decompose: domestic vs export-driven CCGT ──────────────────────────────
    domestic_supply_before_gas = must_t + vre_t + hyd_t + fr_t + pt_t
    domestic_shortfall = load_t - domestic_supply_before_gas

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

    # Hours where CCGT runs solely for exports (domestic shortfall ≤ 0)
    export_only_mask = (domestic_shortfall <= 0) & (gas_t > 0)
    export_only_hrs  = int(export_only_mask.sum())
    export_only_gwh  = float(gas_t[export_only_mask].sum()) / 1e3

    # ── Counterfactual: cap FR/PT exports at historical means ──────────────────
    FR_ACTUAL_MEAN = 344.0    # MW, 2024 ENTSOE mean (positive = import to ES)
    PT_ACTUAL_MEAN = -200.0   # MW, approximate: Spain typically exports ~200 MW to PT
    # Current export flows (negative = ES exporting)
    fr_export_t = (-fr_t).clip(lower=0)   # ES→FR export MW
    pt_export_t = (-pt_t).clip(lower=0)   # ES→PT export MW
    # Capped: reduce export to historical mean if above it
    fr_export_capped = fr_export_t.clip(upper=abs(FR_ACTUAL_MEAN))
    pt_export_capped = pt_export_t.clip(upper=abs(PT_ACTUAL_MEAN))
    # Reduction in export = additional supply available domestically
    fr_export_reduction = fr_export_t - fr_export_capped
    pt_export_reduction = pt_export_t - pt_export_capped
    total_export_reduction = fr_export_reduction + pt_export_reduction

    # Counterfactual: if exports were capped, how much less CCGT would dispatch?
    # The reduction in exports means more supply stays in Spain → less CCGT needed
    cf_ccgt_t = (gas_t - total_export_reduction).clip(lower=0.0)
    cf_ccgt_gwh = float(cf_ccgt_t.sum()) / 1e3
    actual_ccgt_gwh = float(gas_t.sum()) / 1e3

    # Counterfactual price setter (approximate: if CCGT dispatch drops, who sets price?)
    cf_gas_setter_hrs = int((cf_ccgt_t > 100.0).sum())  # rough: CCGT sets price if >100 MW
    actual_gas_setter_hrs = int(gas_setter_mask.sum())

    # ── Print ──────────────────────────────────────────────────────────────────
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
    if export_only_hrs > 0:
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
        fr_exp  = float((-fr_t.reindex(sl)).clip(lower=0).mean())
        pt_exp  = float((-pt_t.reindex(sl)).clip(lower=0).mean())
        fr_imp  = float(fr_t.reindex(sl).clip(lower=0).mean())
        pt_imp  = float(pt_t.reindex(sl).clip(lower=0).mean())
        print(f"  {str(p):<8}  {ccgt_m:>7.0f}  {dom_m:>9.0f}  {exp_m:>7.0f}  {exp_pct:>4.0f}%  "
              f"{fr_exp:>+9.0f}  {pt_exp:>+9.0f}  {fr_imp:>+9.0f}  {pt_imp:>+9.0f}")

    # Table 4: Counterfactual
    print(f"\n  COUNTERFACTUAL — if FR/PT exports capped at historical means")
    print(f"  FR export cap: {abs(FR_ACTUAL_MEAN):.0f} MW  |  PT export cap: {abs(PT_ACTUAL_MEAN):.0f} MW")
    print(f"  {'─'*57}")
    print(f"  {'Metric':<35}  {'Current':>10}  {'Capped':>10}  {'Δ':>8}")
    print(f"  {'─'*65}")
    print(f"  {'CCGT dispatch (GWh)':<35}  {actual_ccgt_gwh:>10.0f}  {cf_ccgt_gwh:>10.0f}  {cf_ccgt_gwh - actual_ccgt_gwh:>+8.0f}")
    print(f"  {'CCGT price-setting hours':<35}  {actual_gas_setter_hrs:>5d} / {len(snaps):<3d}  {cf_gas_setter_hrs:>5d} / {len(snaps):<3d}  {cf_gas_setter_hrs - actual_gas_setter_hrs:>+8d}")
    print(f"  {'Export reduction (GWh)':<35}  {'':>10}  {float(total_export_reduction.sum())/1e3:>10.0f}  {'':>8}")

    # Export-driven CCGT as fraction of total
    exp_frac = exp_gwh / max(total_gwh, 1)
    print(f"\n  INTERPRETATION:")
    if exp_frac > 0.30:
        print(f"  ⚠ {exp_frac*100:.0f}% of CCGT dispatch is export-driven — FR/PT modelling artifact is")
        print(f"     a major driver of CCGT marginality. Fix: increase FR demand scaler or")
        print(f"     enable exogenous borders with historical ENTSOE flows.")
    elif exp_frac > 0.10:
        print(f"  ⚡ {exp_frac*100:.0f}% of CCGT dispatch is export-driven — meaningful but not dominant.")
        print(f"     Both FR/PT modelling and domestic must_run floor contribute to CCGT marginality.")
    else:
        print(f"  ✓ Only {exp_frac*100:.0f}% of CCGT dispatch is export-driven — the problem is")
        print(f"    primarily domestic (must_run floor + LP trickle-dispatch).")
        print(f"    Fix: MIP ON + MCQ (already designed).")
    print(_SEP2)


# ── D7: Seasonal curtailment analysis ─────────────────────────────────────────

def print_curtailment_seasonal(n):
    """Break down VRE curtailment by season, node region, and carrier.

    Explains the observed pattern: northern Spain wind curtailment in spring/autumn
    vs low solar curtailment in summer.
    """
    snaps  = n.snapshots
    gen    = n.generators
    gen_t  = n.generators_t.p
    tv_pmax = getattr(n.generators_t, "p_max_pu", pd.DataFrame())
    es_set = set(_es_buses(n))

    # ── Define seasons ─────────────────────────────────────────────────────────
    def _season(dt):
        m = dt.month
        if m in (3, 4, 5):    return "Spring"
        if m in (6, 7, 8):    return "Summer"
        if m in (9, 10, 11):  return "Autumn"
        return "Winter"

    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    snap_season = pd.Series({s: _season(s) for s in snaps}, index=snaps)

    # ── Node regions ───────────────────────────────────────────────────────────
    # ES0 24 = FR_WEST border (Basque Country / Navarre — north)
    # ES0 43 = FR_EAST border (Catalonia — north-east)
    # ES0 27 = PT_NORTH border (Galicia — north-west)
    # ES0 10 = PT_CENTRE border (Extremadura — centre-west)
    # ES0 23 = PT_SOUTH border (Andalusia — south)
    # Other nodes classified by bus y-coordinate
    def _region(bus):
        if bus not in n.buses.index:
            return "other"
        y = float(n.buses.at[bus, "y"])
        if y > 43.0:   return "north_coast"     # Cantabrian coast
        if y > 42.0:   return "north"            # Pyrenees / Ebro valley
        if y > 40.0:   return "centre"           # Madrid / Castilla y León
        if y > 38.5:   return "south_centre"     # Extremadura / La Mancha
        return "south"                            # Andalusia / Murcia

    # ── Compute curtailment by (carrier, season, region) ───────────────────────
    vre_carriers = {"solar", "onwind"}
    rows = []
    for carrier in vre_carriers:
        for season in seasons:
            season_mask = snap_season == season
            season_snaps = snaps[season_mask.values]
            if len(season_snaps) == 0:
                continue

            # Per-node curtailment
            node_curt_gwh = {}
            node_pot_gwh  = {}
            for name, row in gen.iterrows():
                if row.get("carrier") != carrier or row.get("bus") not in es_set:
                    continue
                if name not in gen_t.columns:
                    continue
                bus = row["bus"]
                p_nom = float(row["p_nom"])
                if name in tv_pmax.columns:
                    potential = (tv_pmax[name].reindex(season_snaps, fill_value=0.0) * p_nom).sum()
                else:
                    potential = float(row.get("p_max_pu", 1.0)) * p_nom * len(season_snaps)
                actual = float(gen_t[name].reindex(season_snaps).sum())
                curt = max(potential - actual, 0.0) / 1e6  # MWh → GWh
                pot  = potential / 1e6
                if pot > 0:
                    node_curt_gwh[bus] = node_curt_gwh.get(bus, 0.0) + curt
                    node_pot_gwh[bus]  = node_pot_gwh.get(bus, 0.0) + pot

            # Aggregate by region
            region_curt = {}
            region_pot  = {}
            for bus, curt in node_curt_gwh.items():
                reg = _region(bus)
                region_curt[reg] = region_curt.get(reg, 0.0) + curt
                region_pot[reg]  = region_pot.get(reg, 0.0) + node_pot_gwh.get(bus, 0.0)

            for reg in sorted(region_curt.keys()):
                pct = 100 * region_curt[reg] / max(region_pot[reg], 0.001)
                rows.append({
                    "carrier": carrier,
                    "season":  season,
                    "region":  reg,
                    "curt_gwh": round(region_curt[reg], 1),
                    "pot_gwh":  round(region_pot[reg], 1),
                    "curt_pct": round(pct, 1),
                })

    if not rows:
        print(f"\n  [D7] No VRE curtailment data found.")
        return

    df = pd.DataFrame(rows)

    # ── Print table ────────────────────────────────────────────────────────────
    print(f"\n{_SEP2}")
    print("  D7 · SEASONAL VRE CURTAILMENT BY REGION")
    print(_SEP2)
    print(f"  {'Carrier':<8}  {'Season':<8}  {'Region':<14}  {'Curt (GWh)':>11}  "
          f"{'Pot (GWh)':>9}  {'Curt%':>6}")
    print(f"  {'─'*60}")
    for _, r in df.iterrows():
        print(f"  {r['carrier']:<8}  {r['season']:<8}  {r['region']:<14}  "
              f"{r['curt_gwh']:>10.1f}  {r['pot_gwh']:>8.1f}  {r['curt_pct']:>5.1f}%")

    # ── Summary by season ──────────────────────────────────────────────────────
    print(f"\n  SEASONAL SUMMARY (all regions, all VRE)")
    print(f"  {'Season':<8}  {'Curt (GWh)':>11}  {'Pot (GWh)':>9}  {'Curt%':>6}  "
          f"{'Wind curt%':>11}  {'Solar curt%':>12}")
    print(f"  {'─'*62}")
    for season in seasons:
        sdf = df[df["season"] == season]
        total_curt = float(sdf["curt_gwh"].sum())
        total_pot  = float(sdf["pot_gwh"].sum())
        pct = 100 * total_curt / max(total_pot, 0.001)
        wind_sdf = sdf[sdf["carrier"] == "onwind"]
        solar_sdf = sdf[sdf["carrier"] == "solar"]
        wind_pct = 100 * float(wind_sdf["curt_gwh"].sum()) / max(float(wind_sdf["pot_gwh"].sum()), 0.001) if len(wind_sdf) > 0 else 0.0
        solar_pct = 100 * float(solar_sdf["curt_gwh"].sum()) / max(float(solar_sdf["pot_gwh"].sum()), 0.001) if len(solar_sdf) > 0 else 0.0
        print(f"  {season:<8}  {total_curt:>10.1f}  {total_pot:>8.1f}  {pct:>5.1f}%  "
              f"{wind_pct:>10.1f}%  {solar_pct:>11.1f}%")

    # ── Northern Spain detail ──────────────────────────────────────────────────
    north_regions = {"north_coast", "north"}
    north_df = df[df["region"].isin(north_regions)]
    if len(north_df) > 0:
        print(f"\n  NORTHERN SPAIN DETAIL (north_coast + north regions)")
        print(f"  {'Carrier':<8}  {'Season':<8}  {'Curt (GWh)':>11}  {'Pot (GWh)':>9}  {'Curt%':>6}")
        print(f"  {'─'*46}")
        for _, r in north_df.iterrows():
            print(f"  {r['carrier']:<8}  {r['season']:<8}  {r['curt_gwh']:>10.1f}  "
                  f"{r['pot_gwh']:>8.1f}  {r['curt_pct']:>5.1f}%")

    # ── IC congestion overlay ──────────────────────────────────────────────────
    # Show FR border line congestion by season
    ic = _ic_flows(n, snaps)
    fr_export_t = (-ic["FR_to_ES"]).clip(lower=0)
    fr_import_t = ic["FR_to_ES"].clip(lower=0)
    print(f"\n  FR BORDER CONGESTION BY SEASON")
    print(f"  {'Season':<8}  {'FR_export mean MW':>17}  {'FR_import mean MW':>17}  "
          f"{'FR_export hrs':>13}  {'FR_import hrs':>13}")
    print(f"  {'─'*64}")
    for season in seasons:
        sm = snap_season == season
        ss = snaps[sm.values]
        if len(ss) == 0:
            continue
        fr_exp_m = float(fr_export_t.reindex(ss).mean())
        fr_imp_m = float(fr_import_t.reindex(ss).mean())
        fr_exp_h = int((fr_export_t.reindex(ss) > 50).sum())
        fr_imp_h = int((fr_import_t.reindex(ss) > 50).sum())
        print(f"  {season:<8}  {fr_exp_m:>16.0f}  {fr_imp_m:>16.0f}  "
              f"{fr_exp_h:>5d}/{len(ss):<5d}  {fr_imp_h:>5d}/{len(ss):<5d}")

    print(_SEP2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cfg = MODEL_CONFIG
    val = cfg["validation"]

    start  = pd.Timestamp(val["start_date"])
    n_days = int(val["n_days"])
    end    = start + pd.Timedelta(hours=n_days * 24 - 1)

    print(f"\n{_SEP}")
    print(_bold(f"  PyPSA-Spain DIAGNOSTIC RUN"))
    print(f"  Window : {start.date()} → {end.date()}  ({n_days} days)")
    print(f"  MIP    : {'ON  (pass-1 MILP → pass-2 LP for duals)' if cfg.get('mip', {}).get('enabled') else 'OFF (pure LP)'}")
    print(f"  MCQ    : {'ON  (α_CCGT={})'.format(cfg.get('gas_mcq_alpha',{}).get('CCGT',0)) if cfg.get('use_mcq') else 'OFF'}")
    print(f"  Solver : {val['solver']}  Crossover={val['solver_options'].get('Crossover',0)}")
    print(_SEP)

    # Load
    net_path = ROOT / val["network_path"]
    log.info("Loading %s ...", net_path.name)
    n = pypsa.Network(str(net_path))

    n = _add_fr_missing_demand(n, cfg)
    n = _apply_fr_demand_scaler(n, cfg)

    log.info("Applying refinements ...")
    n = apply_non_linear_refinements(n, cfg)
    _add_bess_fleet(n, cfg)

    # Slice
    snap = n.snapshots[(n.snapshots >= start) & (n.snapshots <= end)]
    n.set_snapshots(snap)
    log.info("Sliced to %d snapshots", len(snap))

    # Solve
    solver_opts = val.get("solver_options", {})
    mip_enabled = cfg.get("mip", {}).get("enabled", False)

    def extra_functionality(n, snapshots):
        _add_su_ramp_constraints(n, snapshots, cfg)
        _add_hydro_min_dispatch(n, snapshots, cfg)
        _add_hydro_terminal_soc(n, snapshots, cfg)

    _mcq_col   = "marginal_cost_quadratic"
    _mcq_saved = None
    if mip_enabled and _mcq_col in n.generators.columns:
        nz = (n.generators[_mcq_col] != 0).sum()
        if nz > 0:
            _mcq_saved = n.generators[_mcq_col].copy()
            n.generators[_mcq_col] = 0.0
            log.info("MCQ zeroed for pass-1 MILP (%d generators)", nz)

    committable_idx = n.generators.index[
        n.generators.get("committable", pd.Series(dtype=bool)).astype(bool)
    ].tolist()

    log.info("Pass-1 solve (%s) ...", "MILP" if mip_enabled else "LP")
    import time
    t0 = time.time()
    status, cond = n.optimize(
        snapshots=n.snapshots,
        solver_name=val["solver"],
        solver_options=solver_opts,
        extra_functionality=extra_functionality,
    )
    elapsed = time.time() - t0
    log.info("Pass-1 done in %.0fs  status=%s  cond=%s  obj=%.2fM€",
             elapsed, status, cond, n.objective / 1e6)

    if status != "ok":
        log.error("Solve failed: %s / %s — check IIS", status, cond)
        sys.exit(1)

    # Pass-2: fix commitment → LP for dual recovery
    if mip_enabled and committable_idx and "status" in n.generators_t:
        st = n.generators_t["status"]
        cmt = [g for g in committable_idx if g in st.columns]
        if cmt:
            if "p_max_pu" not in n.generators_t or n.generators_t["p_max_pu"].empty:
                n.generators_t["p_max_pu"] = pd.DataFrame(index=n.snapshots)
            for g in cmt:
                n.generators_t["p_max_pu"][g] = st[g].astype(float)
            if "p_min_pu" in n.generators_t:
                drop = [g for g in cmt if g in n.generators_t["p_min_pu"].columns]
                if drop:
                    n.generators_t["p_min_pu"].drop(columns=drop, inplace=True)
            n.generators.loc[cmt, "committable"]   = False
            n.generators.loc[cmt, "start_up_cost"] = 0.0
            n.generators.loc[cmt, "p_min_pu"]      = 0.0
            if _mcq_saved is not None:
                n.generators[_mcq_col] = _mcq_saved
            lp_opts = val.get("pass2_solver_options") or {
                k: v for k, v in solver_opts.items() if k != "MIPGap"
            }
            log.info("Pass-2 LP for dual recovery (Crossover=1) ...")
            t1 = time.time()
            n.optimize(snapshots=n.snapshots, solver_name=val["solver"],
                       solver_options=lp_opts,
                       extra_functionality=extra_functionality)
            log.info("Pass-2 done in %.0fs", time.time() - t1)

    # ── Diagnostics ──────────────────────────────────────────────────────────
    omie = _load_omie(cfg, n.snapshots)

    # 1. Standard stats table
    real_daily     = _load_real_dispatch(cfg, n.snapshots)
    model_daily_es = _to_daily_gwh(_dispatch_by_carrier(n, "ES"))
    _print_stats(n, omie, model_daily_es, real_daily, start, n_days, cfg)

    # 2. MC + price-setter table
    _print_cost_and_price_setter_table(n)

    # 3. Week-by-week loop
    print_weekly_loop(n, omie, cfg)

    # 4. Price spike detail
    print_price_spike_detail(n, omie, threshold=150)

    # 5. IC balance
    print_ic_balance_summary(n, cfg)

    # ── Extended diagnostics ─────────────────────────────────────────────────
    out = _out_dir()
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{_SEP}")
    print(_bold("  EXTENDED DIAGNOSTICS"))
    print(_SEP)

    print_mcq_uplift_audit(n, omie)                       # D1
    stack_df = print_monthly_stack(n, omie)               # D2
    plot_monthly_supply_demand(n, omie, stack_df, out)    # user request 1
    plot_hydro_soc_trajectory(n, omie, out)               # D3
    print_fr_nuclear_ic_monthly(n, omie)                  # D4
    print_vre_cf_monthly(n)                               # D5
    plot_hydro_inflow_fr_pt(n, out)                       # user request 2
    print_ccgt_formation(n, omie)                         # CCGT residual decomposition
    print_ccgt_export_decomposition(n, omie)               # D6: export vs domestic CCGT
    print_curtailment_seasonal(n)                          # D7: seasonal curtailment

    print(f"\n{_SEP}")
    print(_bold("  DIAGNOSTIC COMPLETE"))
    print(_SEP)


if __name__ == "__main__":
    main()
