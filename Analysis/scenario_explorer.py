"""
PyPSA-Spain scenario explorer dashboard

Cross-scenario comparison: {battery, no_battery} × gas shock {Baseline, Low, Mid, High}

Run:
    python Analysis/scenario_explorer.py

Opens at http://localhost:8051

Requires: dash plotly pandas numpy
Optional: pip install kaleido   (for ZIP-download of all charts)
"""

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "final_outputs"

# ── Palette (Cyber-Industrial) ─────────────────────────────────────────────────
_SLATE     = "#1E252B"
_BG        = "#F4F6F8"
_WHITE     = "#FFFFFF"
_GRID      = "#DDE2E6"
_PLOT_TEXT = "#374151"
_MUTED     = "#8896A7"
_TEAL      = "#00A896"
_AMBER     = "#F4A261"
_CORAL     = "#FF6B6B"
_GRID_SB   = "#2e3e4e"

# Battery scenarios → teal family (light→dark as shock intensifies)
# No-battery scenarios → coral family
SCEN_COLORS = {
    "battery_x1.0":    "#8FD8CC",
    "battery_x1.5":    "#00A896",
    "battery_x2.0":    "#00726A",
    "battery_x3.0":    "#003D38",
    "no_battery_x1.0": "#FFD0C7",
    "no_battery_x1.5": "#FF6B6B",
    "no_battery_x2.0": "#C43A3A",
    "no_battery_x3.0": "#7A1E1E",
}

SCEN_DASH = {"1.0": "solid", "1.5": "dash", "2.0": "dot", "3.0": "dashdot"}

# Professional gas-shock labels — used on all axes and legend entries
GAS_SHOCK_LABELS = {1.0: "Baseline", 1.5: "Low shock", 2.0: "Mid shock", 3.0: "High shock"}
GAS_SHOCK_ORDER  = ["Baseline", "Low shock", "Mid shock", "High shock"]

MONTH_SHORT = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
               7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
ALL_MONTHS  = [MONTH_SHORT[m] for m in range(1, 13)]

_PLOT_BASE = dict(
    paper_bgcolor=_WHITE,
    plot_bgcolor=_WHITE,
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
              color=_PLOT_TEXT, size=11),
    xaxis=dict(gridcolor=_GRID, gridwidth=0.5, griddash="dot",
               linecolor=_GRID, tickcolor=_MUTED, showgrid=True),
    yaxis=dict(gridcolor=_GRID, gridwidth=0.5, griddash="dot",
               linecolor=_GRID, tickcolor=_MUTED, showgrid=True),
    margin=dict(l=58, r=24, t=48, b=88),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                font=dict(size=10, color=_PLOT_TEXT),
                orientation="h", x=0, y=-0.34, xanchor="left"),
)

# Technology colours for dispatch charts
TECH_COLORS = {
    "ES_solar":               _AMBER,
    "ES_onwind":              "#4DD9C0",
    "ES_nuclear":             "#4A90D9",
    "ES_hydro_dispatch":      "#2196F3",
    "ES_PHS_dispatch":        "#5B8DB8",
    "ES_PHS_new_dispatch":    "#7BA8CC",
    "ES_battery_dispatch":    "#9C27B0",
    "ES_csp_dispatch":        "#FF9800",
    "ES_thermal_storage_dispatch": "#FFC107",
    "ES_CCGT":                _CORAL,
    "ES_CCGT_flex":           "#E04040",
    "ES_CCGT_must_run":       "#9E2A2A",
    "ES_biomass":             "#66BB6A",
    "ES_ror":                 "#80DEEA",
    "ES_OCGT":                "#FF7043",
}

TECH_LABELS = {
    "ES_solar":               "Solar",
    "ES_onwind":              "Onshore wind",
    "ES_nuclear":             "Nuclear",
    "ES_hydro_dispatch":      "Hydro",
    "ES_PHS_dispatch":        "PHS",
    "ES_PHS_new_dispatch":    "PHS (new)",
    "ES_battery_dispatch":    "Battery",
    "ES_csp_dispatch":        "CSP",
    "ES_thermal_storage_dispatch": "Thermal storage",
    "ES_CCGT":                "CCGT",
    "ES_CCGT_flex":           "CCGT flex",
    "ES_CCGT_must_run":       "CCGT must-run",
    "ES_biomass":             "Biomass",
    "ES_ror":                 "Run-of-river",
    "ES_OCGT":                "OCGT",
}

ALL_SCENARIOS = [
    "no_battery_x1.0", "no_battery_x1.5", "no_battery_x2.0", "no_battery_x3.0",
    "battery_x1.0",    "battery_x1.5",    "battery_x2.0",    "battery_x3.0",
]

# CCGT physics parameters (from config.py / refinery.py)
_MIBGAS_MEAN_2024 = 34.54   # €/MWh_th  (annual mean 2024 MIBGAS PVB)
_CO2_PRICE        = 65.0    # €/tCO₂  (EU ETS 2024)
_CO2_TH_INTENSITY = 0.202   # tCO₂/MWh_th  (natural gas, IPCC AR5)
_CO2_TH_COST      = _CO2_TH_INTENSITY * _CO2_PRICE   # €/MWh_th
_CCGT_VOM         = 5.0     # €/MWh_e  (variable O&M from config)
_CCGT_TIERS = [
    ("T1 — efficient (η 0.76)", 0.76),
    ("T2 — mid-fleet (η 0.68)", 0.68),
    ("T3 — older (η 0.62)",     0.62),
]
# Installed CCGT capacities (MW) — constant across scenarios
_CCGT_CAP    = 19000.8
_CCGT_FLEX_CAP  = 5250.2
_CCGT_MUST_CAP  = 2000.0


# ── Data loading ───────────────────────────────────────────────────────────────
def _load_scenarios() -> dict:
    data = {}
    for s in ALL_SCENARIOS:
        folder = DATA_DIR / s
        meta = json.loads((folder / "scenario_meta.json").read_text())

        prices   = pd.read_csv(folder / "hourly_national_prices.csv",
                               index_col=0, parse_dates=True)
        curtail  = pd.read_csv(folder / "vre_curtailment_by_carrier.csv",
                               index_col=0, parse_dates=True)
        dispatch = pd.read_csv(folder / "hourly_dispatch_national.csv",
                               index_col=0, parse_dates=True)

        for df in (prices, curtail, dispatch):
            df["month"] = df.index.month
            df["hour"]  = df.index.hour

        # Available VRE = actual + curtailed
        dispatch["solar_available"]  = dispatch["ES_solar"]  + curtail["solar"]
        dispatch["onwind_available"] = dispatch["ES_onwind"] + curtail["onwind"]

        bat_agg      = None
        bat_unit_rev = None
        if meta["battery"]:
            bat_raw = pd.read_csv(folder / "battery_arbitrage.csv",
                                  index_col=0, parse_dates=True)
            d_cols = [c for c in bat_raw.columns if c.endswith("_dispatch_MW")]
            s_cols = [c for c in bat_raw.columns if c.endswith("_store_MW")]
            r_cols = [c for c in bat_raw.columns if c.endswith("_revenue_eur")]
            bat_agg = pd.DataFrame({
                "agg_dispatch_MW": bat_raw[d_cols].sum(axis=1),
                "agg_store_MW":    bat_raw[s_cols].sum(axis=1),
                "agg_revenue_eur": bat_raw[r_cols].sum(axis=1),
                "month": bat_raw.index.month,
                "hour":  bat_raw.index.hour,
            }, index=bat_raw.index)
            bat_unit_rev = bat_raw[r_cols].sum() / 1e6  # M€ per unit per year

        data[s] = dict(meta=meta, prices=prices, curtail=curtail,
                       dispatch=dispatch, bat_agg=bat_agg,
                       bat_unit_rev=bat_unit_rev)
    return data


print("Loading scenario data…")
SCENARIOS = _load_scenarios()
print(f"Loaded {len(SCENARIOS)} scenarios.")


# ── Network topology + nodal data ──────────────────────────────────────────────
_NODAL_CACHE: dict = {}
_BASELINE_MAP_SCENARIO = "no_battery_x1.0"


def _load_topology() -> dict | None:
    """Extract bus coords, lines and interconnectors from the presolved network."""
    try:
        import pypsa, logging as _lg
        _lg.disable(_lg.CRITICAL)
        nc = DATA_DIR / "battery_x1.0" / "presolved_20240101_365d_20260604.nc"
        if not nc.exists():
            return None
        n = pypsa.Network(str(nc))

        buses = (n.buses[n.buses.carrier == "AC"][["x", "y", "country"]]
                  .rename(columns={"x": "lon", "y": "lat"}))

        lines = n.lines[["bus0", "bus1", "s_nom"]].copy()
        ic_mask = (lines.bus0.str.startswith(("FR", "PT")) |
                   lines.bus1.str.startswith(("FR", "PT")))
        # Separate: internal ES lines / cross-border AC / internal FR-PT lines
        es_int   = lines[~ic_mask & ~lines.bus0.str.startswith(("FR","PT")) & ~lines.bus1.str.startswith(("FR","PT"))].copy()
        ic_ac    = lines[ic_mask & (lines.bus0.str.startswith("ES") | lines.bus1.str.startswith("ES"))].copy()
        fr_pt_int= lines[ic_mask & ~lines.bus0.str.startswith("ES") & ~lines.bus1.str.startswith("ES")].copy()

        dc_links = n.links[n.links.carrier == "DC"][["bus0", "bus1", "p_nom"]].copy()

        gen_buses = n.generators[["bus", "carrier", "p_nom"]].copy()
        return dict(buses=buses, es_int=es_int, ic_ac=ic_ac,
                    fr_pt_int=fr_pt_int, dc_links=dc_links,
                    gen_buses=gen_buses)
    except Exception as e:
        print(f"Warning: topology not loaded — {e}")
        return None


def _load_nodal_data(scenario: str) -> dict | None:
    """Per-bus annual dispatch + VRE curtailment from solved network. Cached."""
    if scenario in _NODAL_CACHE:
        return _NODAL_CACHE[scenario]
    nc = DATA_DIR / scenario / f"solved_{scenario}.nc"
    if not nc.exists():
        return None
    try:
        import pypsa, logging as _lg
        _lg.disable(_lg.CRITICAL)
        n = pypsa.Network(str(nc))
        gens = n.generators[["bus", "carrier", "p_nom"]].copy()
        gens["annual_gwh"] = n.generators_t.p.sum().reindex(gens.index, fill_value=0) / 1000

        vre = gens[gens.carrier.isin(["solar", "onwind", "offwind"])].copy()
        p_max = n.generators_t.p_max_pu
        avail_cols = vre.index.intersection(p_max.columns)
        avail = (p_max[avail_cols] * gens.loc[avail_cols, "p_nom"]).sum() / 1000
        vre["avail_gwh"]   = avail.reindex(vre.index, fill_value=0)
        vre["curtail_gwh"] = (vre["avail_gwh"] - vre["annual_gwh"]).clip(lower=0)

        result = {
            "dispatch":   gens.groupby(["bus", "carrier"])["annual_gwh"].sum().reset_index(),
            "curtailment": vre.groupby(["bus", "carrier"])["curtail_gwh"].sum().reset_index(),
        }
        _NODAL_CACHE[scenario] = result
        return result
    except Exception as e:
        print(f"Warning: nodal data for {scenario} failed — {e}")
        return None


print("Loading network topology…")
TOPO = _load_topology()
n_buses = len(TOPO["buses"]) if TOPO else 0
print(f"Topology: {n_buses} buses")

print("Pre-loading nodal data for all 8 scenarios (~8 s)…")
for _s in ALL_SCENARIOS:
    _load_nodal_data(_s)
print("Nodal data ready.")


# ── Helpers ────────────────────────────────────────────────────────────────────
def label(s: str) -> str:
    m = SCENARIOS[s]["meta"]
    batt  = "Battery" if m["battery"] else "No battery"
    shock = GAS_SHOCK_LABELS[m["gas_mult"]]
    return f"{batt} · {shock}"

def shock_label(g: float) -> str:
    return GAS_SHOCK_LABELS.get(g, f"×{g}")

def col(s: str) -> str:
    return SCEN_COLORS[s]

def lnstyle(s: str) -> str:
    g = str(SCENARIOS[s]["meta"]["gas_mult"])
    return SCEN_DASH.get(g, "solid")

def _fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_PLOT_BASE,
        title=dict(text=title, font=dict(size=13, color=_PLOT_TEXT),
                   x=0.5, xanchor="center"),
    )
    return fig

def _image_cfg(filename: str) -> dict:
    """Plotly toolbar: show only the camera (PNG download) button."""
    return {
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d",
            "autoScale2d", "resetScale2d", "hoverClosestCartesian",
            "hoverCompareCartesian", "toggleSpikelines",
        ],
        "toImageButtonOptions": {
            "format": "png", "filename": filename,
            "height": 700, "width": 1400, "scale": 2,
        },
    }

def _card(*children) -> html.Div:
    return html.Div(list(children),
        style={"background": _WHITE, "border": f"1px solid {_GRID}",
               "borderRadius": "8px", "padding": "6px 8px",
               "marginBottom": "16px", "boxShadow": "0 1px 4px rgba(0,0,0,0.05)"})

def _row(*children) -> html.Div:
    return html.Div(
        [html.Div(c, style={"flex": "1", "minWidth": "380px"}) for c in children],
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
               "marginBottom": "16px"},
    )

def _graph(fig_id: str, filename: str) -> dcc.Graph:
    return dcc.Graph(id=fig_id, config=_image_cfg(filename),
                     style={"height": "380px"})

_HOUR_TICKS = dict(tickvals=list(range(0, 24, 3)),
                   ticktext=[f"{h:02d}:00" for h in range(0, 24, 3)])


# ── Figure builders — Tab 1: Price overview ────────────────────────────────────

def fig_summary_cards(sel: list) -> html.Div:
    cards = []
    for s in sel:
        p = SCENARIOS[s]["prices"]
        mean_p = p["model_ES"].mean()
        p90    = p["model_ES"].quantile(0.9)
        low_h  = int((p["model_ES"] < 5).sum())
        meta   = SCENARIOS[s]["meta"]
        pair   = f"no_battery_x{meta['gas_mult']}"
        saving_str = ""
        if meta["battery"] and pair in SCENARIOS:
            delta = SCENARIOS[pair]["prices"]["model_ES"].mean() - mean_p
            saving_str = f"−{delta:.1f} €/MWh vs no-batt"
        cards.append(html.Div([
            html.Div(label(s), style={"fontSize": "10px", "color": col(s),
                                       "fontWeight": "700", "textTransform": "uppercase",
                                       "letterSpacing": "0.5px", "marginBottom": "4px"}),
            html.Div(f"€{mean_p:.1f}", style={"fontSize": "24px", "fontWeight": "700",
                                               "color": _PLOT_TEXT, "lineHeight": "1.1"}),
            html.Div(f"Mean · P90: €{p90:.0f}", style={"fontSize": "10px", "color": _MUTED}),
            html.Div(f"{low_h} hrs < €5", style={"fontSize": "10px", "color": _MUTED}),
            html.Div(saving_str, style={"fontSize": "10px", "color": _TEAL,
                                        "marginTop": "3px", "fontWeight": "600"}),
        ], style={"background": _WHITE, "border": f"1px solid {_GRID}",
                  "borderRadius": "8px", "padding": "10px 14px", "flex": "1",
                  "minWidth": "130px", "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"}))
    return html.Div(cards, style={"display": "flex", "gap": "10px",
                                   "flexWrap": "wrap", "marginBottom": "16px"})


def fig_annual_mean_price(sel: list) -> go.Figure:
    fig = _fig("Annual mean price by gas scenario")
    for s in sel:
        m = SCENARIOS[s]["meta"]
        mean = SCENARIOS[s]["prices"]["model_ES"].mean()
        fig.add_bar(x=[shock_label(m["gas_mult"])], y=[mean], name=label(s),
                    marker_color=col(s))
    fig.update_layout(barmode="group", yaxis_title="€/MWh",
                      xaxis=dict(categoryorder="array", categoryarray=GAS_SHOCK_ORDER))
    return fig


def fig_pdc(sel: list) -> go.Figure:
    fig = _fig("Price duration curve — all scenarios")
    # OMIE actuals as a reference benchmark (same data regardless of scenario)
    omie_raw = SCENARIOS["battery_x1.0"]["prices"].get("omie_ES")
    if omie_raw is not None:
        omie_sorted = np.sort(omie_raw.dropna().values)[::-1]
        x_omie = np.linspace(0, 100, len(omie_sorted))
        fig.add_scatter(x=x_omie, y=omie_sorted, mode="lines",
                        name="OMIE actuals 2024",
                        line=dict(color=_MUTED, width=2, dash="longdash"))
    for s in sel:
        p = np.sort(SCENARIOS[s]["prices"]["model_ES"].values)[::-1]
        x = np.linspace(0, 100, len(p))
        fig.add_scatter(x=x, y=p, mode="lines", name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=1.8))
    fig.update_layout(xaxis_title="Cumulative hours (%)", yaxis_title="€/MWh")
    return fig


def fig_price_violin(sel: list) -> go.Figure:
    fig = _fig("Hourly price distribution")
    # Single OMIE reference distribution
    p_data = SCENARIOS["battery_x1.0"]["prices"]
    if "omie_ES" in p_data.columns:
        omie = p_data["omie_ES"].dropna()
        fig.add_violin(y=omie, name="OMIE actuals 2024",
                       box_visible=True, meanline_visible=True,
                       line=dict(color=_MUTED, width=1.2),
                       fillcolor="rgba(136,150,167,0.08)", opacity=0.9)
    # Model distributions
    for s in sel:
        p = SCENARIOS[s]["prices"]["model_ES"]
        fig.add_violin(y=p, name=label(s), box_visible=True,
                       meanline_visible=True, fillcolor=col(s),
                       line_color=col(s), opacity=0.65, showlegend=True)
    fig.update_layout(yaxis_title="€/MWh", xaxis_title="Scenario",
                      violingap=0.05, violinmode="overlay")
    return fig


def fig_battery_price_reduction(sel: list) -> go.Figure:
    fig = _fig("Battery price reduction vs gas shock level")
    xs, ys = [], []
    for g in [1.0, 1.5, 2.0, 3.0]:
        sb, sn = f"battery_x{g}", f"no_battery_x{g}"
        if sb in sel and sn in sel:
            delta = (SCENARIOS[sn]["prices"]["model_ES"].mean()
                     - SCENARIOS[sb]["prices"]["model_ES"].mean())
            xs.append(shock_label(g))
            ys.append(round(delta, 2))
    if xs:
        fig.add_bar(x=xs, y=ys, marker_color=_TEAL, showlegend=False,
                    text=[f"{y:.1f} €/MWh" for y in ys],
                    textposition="outside", textfont=dict(size=10))
    fig.update_layout(yaxis_title="Mean price reduction (€/MWh)",
                      xaxis=dict(categoryorder="array", categoryarray=GAS_SHOCK_ORDER),
                      yaxis=dict(rangemode="tozero"))
    return fig


# ── Figure builders — Tab 2: Representative day ───────────────────────────────

def fig_rep_day_price(sel: list, month: int) -> go.Figure:
    mn = MONTH_SHORT.get(month, "")
    fig = _fig(f"Representative hourly price profile — {mn}")
    for s in sel:
        p = SCENARIOS[s]["prices"]
        hourly = p[p["month"] == month].groupby("hour")["model_ES"].mean()
        fig.add_scatter(x=list(range(24)), y=hourly.reindex(range(24)).values,
                        mode="lines+markers", name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=2),
                        marker=dict(size=4))
    fig.update_layout(xaxis=dict(**_HOUR_TICKS, **{k: v for k, v in _PLOT_BASE["xaxis"].items()}),
                      yaxis_title="€/MWh", xaxis_title="Hour of day")
    return fig


def fig_price_heatmap(sel: list) -> go.Figure:
    if not sel:
        return _fig("Price heatmap by month and hour")
    s = sel[0]
    p = SCENARIOS[s]["prices"]
    pivot = (p.groupby(["month", "hour"])["model_ES"].mean()
              .unstack(level="hour")
              .reindex(index=range(1, 13), columns=range(24)))
    fig = _fig(f"Price heatmap by month and hour — {label(s)}")
    fig.add_heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in range(24)],
        y=ALL_MONTHS,
        colorscale="RdYlGn_r",
        colorbar=dict(title=dict(text="€/MWh", font=dict(size=10)), thickness=12),
        hovertemplate="Hour: %{x}<br>Month: %{y}<br>Price: %{z:.1f} €/MWh<extra></extra>",
    )
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="Month")
    return fig


def fig_monthly_volatility(sel: list) -> go.Figure:
    fig = _fig("Intra-month price volatility")
    for s in sel:
        p = SCENARIOS[s]["prices"]
        std_m = p.groupby("month")["model_ES"].std()
        fig.add_scatter(x=ALL_MONTHS, y=std_m.reindex(range(1, 13)).values,
                        mode="lines+markers", name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=2),
                        marker=dict(size=5))
    fig.update_layout(yaxis_title="Price std dev (€/MWh)", xaxis_title="Month")
    return fig


def fig_low_price_hours(sel: list) -> go.Figure:
    fig = _fig("Low-price hours by month (<5 €/MWh)")
    for s in sel:
        p = SCENARIOS[s]["prices"]
        low = (p[p["model_ES"] < 5].groupby("month").size()
                                    .reindex(range(1, 13), fill_value=0))
        fig.add_bar(x=ALL_MONTHS, y=low.values, name=label(s),
                    marker_color=col(s), opacity=0.85)
    fig.update_layout(barmode="group", yaxis_title="Hours", xaxis_title="Month")
    return fig


# ── Figure builders — Tab 3: Curtailment ──────────────────────────────────────

def fig_curtailment_by_scenario(sel: list) -> go.Figure:
    fig = _fig("Annual VRE curtailment by scenario")
    first = True
    for s in sel:
        c = SCENARIOS[s]["curtail"]
        solar_gwh = c["solar"].sum() / 1000
        wind_gwh  = c["onwind"].sum() / 1000
        fig.add_bar(x=[label(s)], y=[solar_gwh], name="Solar",
                    marker_color=_AMBER, legendgroup="solar", showlegend=first)
        fig.add_bar(x=[label(s)], y=[wind_gwh], name="Wind",
                    marker_color=_TEAL, legendgroup="wind", showlegend=first)
        first = False
    fig.update_layout(barmode="stack", yaxis_title="GWh curtailed",
                      xaxis_title="Scenario")
    return fig


def fig_curtailment_reduction(sel: list) -> go.Figure:
    fig = _fig("Curtailment reduction from batteries")
    xs, ys = [], []
    for g in [1.0, 1.5, 2.0, 3.0]:
        sb, sn = f"battery_x{g}", f"no_battery_x{g}"
        if sb in sel and sn in sel:
            Cb = SCENARIOS[sb]["curtail"]["total_curtailed_MW"].sum() / 1000
            Cn = SCENARIOS[sn]["curtail"]["total_curtailed_MW"].sum() / 1000
            xs.append(shock_label(g))
            ys.append(round(Cn - Cb, 1))
    if xs:
        fig.add_bar(x=xs, y=ys, marker_color=_TEAL, showlegend=False,
                    text=[f"{y:.0f} GWh" for y in ys],
                    textposition="outside", textfont=dict(size=10))
    fig.update_layout(yaxis_title="Curtailment reduction (GWh)",
                      xaxis=dict(categoryorder="array", categoryarray=GAS_SHOCK_ORDER),
                      yaxis=dict(rangemode="tozero"))
    return fig


def fig_monthly_curtailment(sel: list) -> go.Figure:
    fig = _fig("Monthly VRE curtailment — solar and wind")
    for s in sel:
        c = SCENARIOS[s]["curtail"]
        total_m = (c.groupby("month")["total_curtailed_MW"].sum() / 1000).reindex(range(1, 13), fill_value=0)
        fig.add_scatter(x=ALL_MONTHS, y=total_m.values,
                        mode="lines+markers", name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=2),
                        marker=dict(size=5))
    fig.update_layout(yaxis_title="GWh curtailed", xaxis_title="Month")
    return fig


def fig_rep_curtailment_day(sel: list, month: int) -> go.Figure:
    mn = MONTH_SHORT.get(month, "")
    fig = _fig(f"Representative curtailment profile — {mn}")
    for s in sel:
        c = SCENARIOS[s]["curtail"]
        hourly = (c[c["month"] == month]
                   .groupby("hour")["total_curtailed_MW"].mean()
                   .reindex(range(24), fill_value=0))
        fig.add_scatter(x=list(range(24)), y=hourly.values, mode="lines",
                        name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=2))
    fig.update_layout(xaxis=dict(**_HOUR_TICKS, **{k: v for k, v in _PLOT_BASE["xaxis"].items()}),
                      yaxis_title="MW curtailed", xaxis_title="Hour of day")
    return fig


def fig_vre_utilisation(sel: list) -> go.Figure:
    fig = _fig("Monthly VRE utilisation rate")
    for s in sel:
        d = SCENARIOS[s]["dispatch"]
        gen   = d.groupby("month")[["ES_solar", "ES_onwind"]].sum()
        avail = d.groupby("month")[["solar_available", "onwind_available"]].sum()
        total_gen   = gen["ES_solar"] + gen["ES_onwind"]
        total_avail = avail["solar_available"] + avail["onwind_available"]
        util = (total_gen / total_avail.replace(0, np.nan) * 100).reindex(range(1, 13))
        fig.add_scatter(x=ALL_MONTHS, y=util.values, mode="lines+markers",
                        name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=2),
                        marker=dict(size=5))
    fig.update_layout(yaxis_title="VRE utilisation (%)", xaxis_title="Month",
                      yaxis=dict(range=[50, 102]))
    return fig


# ── Figure builders — Tab 4: Battery usage ────────────────────────────────────

def fig_bat_rep_day(sel: list, month: int) -> go.Figure:
    mn = MONTH_SHORT.get(month, "")
    fig = _fig(f"Battery aggregate dispatch profile — {mn}")
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]
    for s in bat_sel:
        b = SCENARIOS[s]["bat_agg"]
        sub = b[b["month"] == month]
        h_d = sub.groupby("hour")["agg_dispatch_MW"].mean().reindex(range(24), fill_value=0)
        h_s = sub.groupby("hour")["agg_store_MW"].mean().reindex(range(24), fill_value=0)
        fig.add_scatter(x=list(range(24)), y=h_d.values,
                        mode="lines", name=f"{label(s)} ↑ dispatch",
                        fill="tozeroy",
                        line=dict(color=col(s), width=1.5))
        fig.add_scatter(x=list(range(24)), y=(-h_s.values),
                        mode="lines", name=f"{label(s)} ↓ charge",
                        fill="tozeroy",
                        line=dict(color=col(s), width=1.5, dash="dot"))
    fig.update_layout(
        xaxis=dict(**_HOUR_TICKS, **{k: v for k, v in _PLOT_BASE["xaxis"].items()}),
        yaxis_title="MW  (positive = dispatch, negative = charging)",
        xaxis_title="Hour of day",
        shapes=[dict(type="line", y0=0, y1=0, x0=0, x1=23,
                     line=dict(color=_GRID, width=1))],
    )
    return fig


def fig_bess_utilisation(sel: list) -> go.Figure:
    fig = _fig("Monthly battery operating mode distribution")
    THRESH = 100
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]
    for s in bat_sel:
        b = SCENARIOS[s]["bat_agg"]
        discharge_h = (b.assign(_d=b["agg_dispatch_MW"] > THRESH)
                        .groupby("month")["_d"].sum()
                        .reindex(range(1, 13), fill_value=0))
        charge_h = (b.assign(_c=b["agg_store_MW"] > THRESH)
                     .groupby("month")["_c"].sum()
                     .reindex(range(1, 13), fill_value=0))
        fig.add_bar(x=ALL_MONTHS, y=discharge_h.values,
                    name=f"{label(s)} discharging",
                    marker_color=col(s), opacity=0.9)
        fig.add_bar(x=ALL_MONTHS, y=charge_h.values,
                    name=f"{label(s)} charging",
                    marker_color=col(s), opacity=0.45)
    fig.update_layout(barmode="group", yaxis_title="Hours per month",
                      xaxis_title="Month")
    return fig


def fig_bat_revenue_dist(sel: list) -> go.Figure:
    fig = _fig("Battery unit revenue distribution by gas scenario")
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_unit_rev"] is not None]
    for s in bat_sel:
        rev = SCENARIOS[s]["bat_unit_rev"]
        fig.add_histogram(x=rev.values, name=label(s), marker_color=col(s),
                          opacity=0.7, nbinsx=30)
    fig.update_layout(barmode="overlay", xaxis_title="Annual revenue per unit (M€)",
                      yaxis_title="Count of battery units")
    return fig


def fig_bat_vs_solar(sel: list) -> go.Figure:
    fig = _fig("Battery charging vs solar output")
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]
    for s in bat_sel:
        b = SCENARIOS[s]["bat_agg"]
        d = SCENARIOS[s]["dispatch"]
        h_store = b.groupby(["month", "hour"])["agg_store_MW"].mean()
        h_solar = d.groupby(["month", "hour"])["ES_solar"].mean()
        idx = h_store.index.intersection(h_solar.index)
        fig.add_scatter(x=h_solar.loc[idx].values, y=h_store.loc[idx].values,
                        mode="markers", name=label(s),
                        marker=dict(color=col(s), size=4, opacity=0.5))
    fig.update_layout(xaxis_title="Solar output (MW)",
                      yaxis_title="Battery charging (MW)")
    return fig


def fig_arbitrage_spread(sel: list) -> go.Figure:
    fig = _fig("Charge and discharge price distributions")
    THRESH = 100
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]
    for s in bat_sel:
        b = SCENARIOS[s]["bat_agg"]
        p = SCENARIOS[s]["prices"]
        d_mask = b["agg_dispatch_MW"] > THRESH
        c_mask = b["agg_store_MW"]    > THRESH
        d_px = p.loc[b.index[d_mask], "model_ES"].values
        c_px = p.loc[b.index[c_mask], "model_ES"].values
        if len(d_px):
            fig.add_histogram(x=d_px, name=f"{label(s)} discharge",
                              marker_color=col(s), opacity=0.8, nbinsx=40)
        if len(c_px):
            fig.add_histogram(x=c_px, name=f"{label(s)} charge",
                              marker_color=col(s), opacity=0.35, nbinsx=40)
    fig.update_layout(barmode="overlay", xaxis_title="Electricity price (€/MWh)",
                      yaxis_title="Hours")
    return fig


def fig_soc_proxy(sel: list) -> go.Figure:
    fig = _fig("Battery net dispatch trajectory (annual)")
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]
    for s in bat_sel:
        b = SCENARIOS[s]["bat_agg"]
        net = (b["agg_dispatch_MW"] - b["agg_store_MW"]).cumsum()
        fig.add_scatter(x=b.index, y=net.values, mode="lines",
                        name=label(s),
                        line=dict(color=col(s), width=1.5, dash=lnstyle(s)))
    fig.update_layout(xaxis_title="Date",
                      yaxis_title="Cumulative net dispatch (MWh equiv.)")
    return fig


# ── Figure builders — Tab 5: Dispatch mix ─────────────────────────────────────

_ES_TECH_ORDER = [
    "ES_solar", "ES_onwind", "ES_nuclear", "ES_ror",
    "ES_hydro_dispatch", "ES_PHS_dispatch", "ES_PHS_new_dispatch",
    "ES_csp_dispatch", "ES_thermal_storage_dispatch", "ES_battery_dispatch",
    "ES_biomass", "ES_CCGT_must_run", "ES_CCGT", "ES_CCGT_flex", "ES_OCGT",
]


def fig_gen_mix(sel: list) -> go.Figure:
    fig = _fig("Annual generation mix by scenario")
    for tech in _ES_TECH_ORDER:
        xs, ys = [], []
        for s in sel:
            d = SCENARIOS[s]["dispatch"]
            if tech not in d.columns:
                continue
            gwh = d[tech].sum() / 1e6
            if gwh > 0.05:
                xs.append(label(s))
                ys.append(gwh)
        if xs:
            fig.add_bar(y=xs, x=ys, orientation="h",
                        name=TECH_LABELS.get(tech, tech),
                        marker_color=TECH_COLORS.get(tech, "#aaa"))
    fig.update_layout(barmode="stack", xaxis_title="TWh", yaxis_title="Scenario",
                      margin=dict(l=140, r=24, t=48, b=60),
                      legend=dict(y=-0.3, orientation="h", x=0))
    return fig


def fig_tech_dispatch(sel: list, tech: str = "ES_CCGT") -> go.Figure:
    tech_lbl = TECH_LABELS.get(tech, tech)
    fig = _fig(f"Annual dispatch by technology — {tech_lbl}")
    xs, ys, cs = [], [], []
    for s in sel:
        d = SCENARIOS[s]["dispatch"]
        if tech in d.columns:
            xs.append(label(s))
            ys.append(round(d[tech].sum() / 1e6, 2))
            cs.append(col(s))
    if xs:
        fig.add_bar(x=xs, y=ys, marker_color=cs, showlegend=False,
                    text=[f"{y:.1f}" for y in ys],
                    textposition="outside", textfont=dict(size=10))
    fig.update_layout(yaxis_title="TWh", xaxis_title="Scenario",
                      yaxis=dict(rangemode="tozero"))
    return fig


def fig_rep_dispatch_day(sel: list, month: int) -> go.Figure:
    mn = MONTH_SHORT.get(month, "")
    if not sel:
        return _fig(f"Representative dispatch stack — {mn}")
    s = sel[0]
    d = SCENARIOS[s]["dispatch"]
    sub = d[d["month"] == month]
    available_techs = [t for t in _ES_TECH_ORDER if t in d.columns]
    hourly = sub.groupby("hour")[available_techs].mean()
    fig = _fig(f"Representative dispatch stack — {mn}, {label(s)}")
    for tech in available_techs:
        vals = hourly[tech].reindex(range(24), fill_value=0).values
        if vals.sum() < 1:
            continue
        tc = TECH_COLORS.get(tech, "#aaa")
        fig.add_scatter(x=list(range(24)), y=vals, mode="lines",
                        name=TECH_LABELS.get(tech, tech),
                        stackgroup="one",
                        line=dict(width=0, color=tc),
                        fillcolor=tc,
                        hovertemplate=f"{TECH_LABELS.get(tech, tech)}: %{{y:.0f}} MW<extra></extra>")
    fig.update_layout(
        xaxis=dict(**_HOUR_TICKS, **{k: v for k, v in _PLOT_BASE["xaxis"].items()}),
        yaxis_title="MW", xaxis_title="Hour of day",
        legend=dict(y=-0.35, orientation="h", x=0),
        hovermode="x unified",
    )
    return fig


def fig_ccgt_displacement(sel: list) -> go.Figure:
    fig = _fig("CCGT generation displaced by batteries")
    nobat_xs, nobat_ys, bat_xs, bat_ys = [], [], [], []
    for g in [1.0, 1.5, 2.0, 3.0]:
        sb, sn = f"battery_x{g}", f"no_battery_x{g}"
        if sn in sel:
            nobat_xs.append(shock_label(g))
            nobat_ys.append(round(SCENARIOS[sn]["dispatch"]["ES_CCGT"].sum() / 1e6, 2))
        if sb in sel:
            bat_xs.append(shock_label(g))
            bat_ys.append(round(SCENARIOS[sb]["dispatch"]["ES_CCGT"].sum() / 1e6, 2))
    if nobat_xs:
        fig.add_bar(x=nobat_xs, y=nobat_ys, name="Without batteries",
                    marker_color=_CORAL)
    if bat_xs:
        fig.add_bar(x=bat_xs, y=bat_ys, name="With batteries",
                    marker_color=_TEAL)
    fig.update_layout(barmode="group", yaxis_title="CCGT generation (TWh)",
                      xaxis=dict(categoryorder="array", categoryarray=GAS_SHOCK_ORDER))
    return fig


# ── Figure builders — Tab 6: Gas shock impact ─────────────────────────────────

def _shock_tick_layout() -> dict:
    return dict(tickvals=[1.0, 1.5, 2.0, 3.0],
                ticktext=["Baseline", "Low shock", "Mid shock", "High shock"],
                title_text="Gas shock scenario")

def fig_price_elasticity(sel: list) -> go.Figure:
    fig = _fig("Electricity price response to gas shock")
    bat_pts, nobat_pts = [], []
    for s in sel:
        m    = SCENARIOS[s]["meta"]
        mean = SCENARIOS[s]["prices"]["model_ES"].mean()
        if m["battery"]:
            bat_pts.append((m["gas_mult"], mean))
        else:
            nobat_pts.append((m["gas_mult"], mean))
    for pts, name, c in [(bat_pts, "With batteries", _TEAL),
                         (nobat_pts, "Without batteries", _CORAL)]:
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            fig.add_scatter(x=list(xs), y=list(ys), mode="lines+markers",
                            name=name, line=dict(color=c, width=2.5),
                            marker=dict(size=9, symbol="circle" if "With" in name else "diamond"))
    fig.update_layout(yaxis_title="Mean electricity price (€/MWh)",
                      xaxis=dict(**_shock_tick_layout()))
    return fig


def fig_seasonal_elasticity(sel: list) -> go.Figure:
    SEASONS = [("Q1 (Jan–Mar)", [1, 2, 3]), ("Q2 (Apr–Jun)", [4, 5, 6]),
               ("Q3 (Jul–Sep)", [7, 8, 9]),  ("Q4 (Oct–Dec)", [10, 11, 12])]
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[s[0] for s in SEASONS],
                        shared_yaxes=False)
    fig.update_layout(**{k: v for k, v in _PLOT_BASE.items() if k not in ("xaxis", "yaxis")},
                      title=dict(text="Seasonal price response to gas shock",
                                 font=dict(size=13, color=_PLOT_TEXT),
                                 x=0.5, xanchor="center"))
    for idx, (season, months) in enumerate(SEASONS):
        r, c2 = divmod(idx, 2)
        row, col2 = r + 1, c2 + 1
        bat_pts, nobat_pts = [], []
        for s in sel:
            meta = SCENARIOS[s]["meta"]
            p    = SCENARIOS[s]["prices"]
            mean = p[p["month"].isin(months)]["model_ES"].mean()
            if meta["battery"]:
                bat_pts.append((meta["gas_mult"], mean))
            else:
                nobat_pts.append((meta["gas_mult"], mean))
        for pts, name, c in [(bat_pts, "With batteries", _TEAL),
                             (nobat_pts, "Without batteries", _CORAL)]:
            if pts:
                pts.sort()
                xs, ys = zip(*pts)
                fig.add_scatter(x=list(xs), y=list(ys), mode="lines+markers",
                                name=name, line=dict(color=c, width=1.8),
                                marker=dict(size=6),
                                showlegend=(idx == 0),
                                row=row, col=col2)
    fig.update_xaxes(tickvals=[1.0, 1.5, 2.0, 3.0],
                     ticktext=["Baseline", "Low", "Mid", "High"])
    fig.update_yaxes(title_text="Mean price (€/MWh)", col=1)
    return fig


def fig_ic_price_diff(sel: list) -> go.Figure:
    fig = _fig("Interconnector price differentials under gas shock")
    fr_pts_bat, fr_pts_nobat = [], []
    pt_pts_bat, pt_pts_nobat = [], []
    for s in sel:
        p    = SCENARIOS[s]["prices"]
        m    = SCENARIOS[s]["meta"]
        g    = m["gas_mult"]
        diff_fr = (p["model_ES"] - p["model_FR"]).mean()
        diff_pt = (p["model_ES"] - p["model_PT"]).mean()
        if m["battery"]:
            fr_pts_bat.append((g, diff_fr))
            pt_pts_bat.append((g, diff_pt))
        else:
            fr_pts_nobat.append((g, diff_fr))
            pt_pts_nobat.append((g, diff_pt))
    combos = [
        (fr_pts_bat,   "ES−FR (battery)",    _TEAL,  "circle"),
        (fr_pts_nobat, "ES−FR (no battery)", _CORAL, "circle-open"),
        (pt_pts_bat,   "ES−PT (battery)",    _AMBER, "diamond"),
        (pt_pts_nobat, "ES−PT (no battery)", "#C97A20", "diamond-open"),
    ]
    for pts, name, c, sym in combos:
        if pts:
            pts.sort()
            xs, ys = zip(*pts)
            fig.add_scatter(x=list(xs), y=list(ys), mode="lines+markers",
                            name=name, line=dict(color=c, width=1.8),
                            marker=dict(size=8, symbol=sym, color=c))
    fig.update_layout(
        yaxis_title="Mean price spread (€/MWh)",
        xaxis=dict(**_shock_tick_layout()),
        shapes=[dict(type="line", y0=0, y1=0, x0=0.9, x1=3.1,
                     line=dict(color=_MUTED, dash="dot", width=1))],
    )
    return fig


def fig_merit_order_shift(sel: list) -> go.Figure:
    fig = _fig("CCGT dispatch volume by gas scenario")
    nobat_xs, nobat_ys, bat_xs, bat_ys = [], [], [], []
    for g in [1.0, 1.5, 2.0, 3.0]:
        sb, sn = f"battery_x{g}", f"no_battery_x{g}"
        if sn in sel:
            d   = SCENARIOS[sn]["dispatch"]
            gwh = (d[["ES_CCGT", "ES_CCGT_flex", "ES_CCGT_must_run"]]
                    .sum().sum() / 1e6)
            nobat_xs.append(shock_label(g))
            nobat_ys.append(round(gwh, 2))
        if sb in sel:
            d   = SCENARIOS[sb]["dispatch"]
            gwh = (d[["ES_CCGT", "ES_CCGT_flex", "ES_CCGT_must_run"]]
                    .sum().sum() / 1e6)
            bat_xs.append(shock_label(g))
            bat_ys.append(round(gwh, 2))
    if nobat_xs:
        fig.add_bar(x=nobat_xs, y=nobat_ys, name="Without batteries",
                    marker_color=_CORAL)
    if bat_xs:
        fig.add_bar(x=bat_xs, y=bat_ys, name="With batteries",
                    marker_color=_TEAL)
    fig.update_layout(barmode="group", yaxis_title="Total CCGT generation (TWh)",
                      xaxis=dict(categoryorder="array", categoryarray=GAS_SHOCK_ORDER))
    return fig


# ── Figure builders — Tab 7: CCGT fleet analysis ──────────────────────────────

def fig_ccgt_tranche_dispatch(sel: list) -> go.Figure:
    fig = _fig("CCGT fleet dispatch by tranche across scenarios")
    first = True
    tranche_cfg = [
        ("ES_CCGT",           "CCGT (main fleet)",  _CORAL),
        ("ES_CCGT_flex",      "CCGT flex",          "#E04040"),
        ("ES_CCGT_must_run",  "CCGT must-run",      "#9E2A2A"),
    ]
    for col_t, tranche_lbl, c in tranche_cfg:
        xs, ys = [], []
        for s in sel:
            d = SCENARIOS[s]["dispatch"]
            if col_t in d.columns:
                xs.append(label(s))
                ys.append(round(d[col_t].sum() / 1e6, 2))
        if xs:
            fig.add_bar(x=xs, y=ys, name=tranche_lbl,
                        marker_color=c, showlegend=first or True)
        first = False
    fig.update_layout(barmode="stack", yaxis_title="TWh",
                      xaxis_title="Scenario")
    return fig


def fig_ccgt_mc_decomposition() -> go.Figure:
    """Average CCGT MC breakdown (fuel / CO₂ / VOM) by tranche and gas shock.

    Efficiencies (η) derived from config.py calibrated tier ranges:
      CCGT main fleet: capacity-weighted average of T1/T2/T3 η → 0.69
      CCGT flex:       mid of 0.38–0.44 partial-load range     → 0.41  VOM=8
      CCGT must-run:   mid of T3 range, MC set to 0 in market  → 0.62  (cost basis only)
    """
    shock_vals = [1.0, 1.5, 2.0, 3.0]
    shock_lbls = [GAS_SHOCK_LABELS[g] for g in shock_vals]

    # Tranche: (panel_title, η, VOM, MC_note)
    TRANCHES = [
        ("CCGT — main fleet\n(avg η 0.69)", 0.69, _CCGT_VOM,  None),
        ("CCGT flex\n(η 0.41, partial load)", 0.41, 8.0,       None),
        ("CCGT must-run\n(η 0.62, cost basis*)", 0.62, _CCGT_VOM, "* market MC = 0; bars show physical cost basis"),
    ]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=[t[0].replace("\n", "<br>") for t in TRANCHES],
                        shared_yaxes=True)
    base = {k: v for k, v in _PLOT_BASE.items() if k not in ("xaxis", "yaxis", "margin", "legend")}
    fig.update_layout(
        **base,
        title=dict(text="CCGT marginal cost decomposition by tranche",
                   font=dict(size=13, color=_PLOT_TEXT), x=0.5, xanchor="center"),
        barmode="stack",
        margin=dict(l=58, r=24, t=70, b=100),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                    borderwidth=1, font=dict(size=10, color=_PLOT_TEXT),
                    orientation="h", x=0.5, xanchor="center", y=-0.18),
    )

    COMPONENT_COLORS = [
        ("Gas fuel cost", _CORAL),
        ("CO₂ cost",      _AMBER),
        ("VOM",           _MUTED),
    ]

    for col_idx, (panel_title, eta, vom, note) in enumerate(TRANCHES):
        first_col = (col_idx == 0)
        fuels, co2s, voms = [], [], []
        for g in shock_vals:
            fuels.append(round((_MIBGAS_MEAN_2024 * g) / eta, 1))
            co2s.append(round(_CO2_TH_COST / eta, 1))
            voms.append(vom)

        for (comp_name, comp_color), values in zip(COMPONENT_COLORS,
                                                   [fuels, co2s, voms]):
            fig.add_bar(
                x=shock_lbls, y=values,
                name=comp_name,
                marker_color=comp_color,
                legendgroup=comp_name,
                showlegend=first_col,
                row=1, col=col_idx + 1,
                text=[f"{v:.0f}" for v in values],
                textposition="inside",
                textfont=dict(size=8, color=_WHITE),
            )

        if note:
            fig.add_annotation(
                text=note, xref=f"x{col_idx+1}", yref="paper",
                x=1.5, y=-0.14, showarrow=False,
                font=dict(size=8, color=_MUTED), xanchor="center",
            )

    fig.update_xaxes(categoryorder="array", categoryarray=GAS_SHOCK_ORDER,
                     tickfont=dict(size=9))
    fig.update_yaxes(title_text="€/MWh_e", col=1)
    return fig


def fig_ccgt_co2_share() -> go.Figure:
    """CO₂ fraction of total CCGT marginal cost across shock scenarios."""
    fig = _fig("CO₂ cost share of CCGT marginal cost")
    shock_vals = [1.0, 1.5, 2.0, 3.0]
    shock_lbls = [GAS_SHOCK_LABELS[g] for g in shock_vals]
    tier_dash  = {"T1 — efficient (η 0.76)": "solid",
                  "T2 — mid-fleet (η 0.68)": "dash",
                  "T3 — older (η 0.62)":     "dot"}
    tier_cols  = {"T1 — efficient (η 0.76)": "#4A90D9",
                  "T2 — mid-fleet (η 0.68)": _TEAL,
                  "T3 — older (η 0.62)":     "#374151"}
    for tier_lbl, eta in _CCGT_TIERS:
        shares = []
        for g in shock_vals:
            fuel_e = (_MIBGAS_MEAN_2024 * g) / eta
            co2_e  = _CO2_TH_COST / eta
            total  = fuel_e + co2_e + _CCGT_VOM
            shares.append(round(co2_e / total * 100, 1))
        short = tier_lbl.split("—")[0].strip()
        fig.add_scatter(x=shock_lbls, y=shares, mode="lines+markers",
                        name=short,
                        line=dict(color=tier_cols[tier_lbl],
                                  dash=tier_dash[tier_lbl], width=2),
                        marker=dict(size=8),
                        text=[f"{v:.0f}%" for v in shares],
                        textposition="top center")
    fig.update_layout(yaxis_title="CO₂ share of marginal cost (%)",
                      xaxis=dict(categoryorder="array", categoryarray=GAS_SHOCK_ORDER),
                      yaxis=dict(range=[0, 35]))
    return fig


def fig_ccgt_rep_day(sel: list, month: int) -> go.Figure:
    mn = MONTH_SHORT.get(month, "")
    fig = _fig(f"CCGT hourly dispatch profile — {mn}")
    for s in sel:
        d = SCENARIOS[s]["dispatch"]
        sub = d[d["month"] == month]
        hourly = (sub.groupby("hour")[["ES_CCGT", "ES_CCGT_flex", "ES_CCGT_must_run"]]
                     .mean().sum(axis=1).reindex(range(24), fill_value=0))
        fig.add_scatter(x=list(range(24)), y=hourly.values, mode="lines+markers",
                        name=label(s),
                        line=dict(color=col(s), dash=lnstyle(s), width=2),
                        marker=dict(size=4))
    fig.update_layout(
        xaxis=dict(**_HOUR_TICKS, **{k: v for k, v in _PLOT_BASE["xaxis"].items()}),
        yaxis_title="CCGT fleet dispatch (MW)", xaxis_title="Hour of day",
    )
    return fig


def fig_ccgt_capacity_factor(sel: list) -> go.Figure:
    fig = _fig("CCGT fleet capacity factor")
    hours = 8736
    tranche_cfg = [
        ("ES_CCGT",          "CCGT",          _CCGT_CAP,       _CORAL),
        ("ES_CCGT_flex",     "CCGT flex",     _CCGT_FLEX_CAP,  "#E04040"),
        ("ES_CCGT_must_run", "CCGT must-run", _CCGT_MUST_CAP,  "#9E2A2A"),
    ]
    first = True
    for col_t, tranche_lbl, cap_mw, c in tranche_cfg:
        xs, ys = [], []
        for s in sel:
            d = SCENARIOS[s]["dispatch"]
            if col_t in d.columns:
                cf = d[col_t].sum() / (cap_mw * hours) * 100
                xs.append(label(s))
                ys.append(round(cf, 1))
        if xs:
            fig.add_bar(x=xs, y=ys, name=tranche_lbl, marker_color=c,
                        showlegend=True,
                        text=[f"{y:.0f}%" for y in ys],
                        textposition="outside", textfont=dict(size=9))
        first = False
    fig.update_layout(barmode="group", yaxis_title="Capacity factor (%)",
                      xaxis_title="Scenario", yaxis=dict(range=[0, 85]))
    return fig


# ── Figure builders — Tab 8: Maps ─────────────────────────────────────────────

_MAP_CENTER  = dict(lat=39.8, lon=-3.5)
_MAP_ZOOM    = 5.0
_PIE_FIG_W   = 1100   # px — used for Mercator→paper conversion
_PIE_FIG_H   = 580    # px
_PIE_MAP_ZOOM= 5.1

_MAP_FONT = dict(family="Helvetica Neue, Arial, sans-serif", color=_PLOT_TEXT, size=11)
_MAP_LEG  = dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                 font=dict(size=10, color=_PLOT_TEXT), x=0.01, y=0.99,
                 xanchor="left", yanchor="top")


def _map_layout(title="", zoom=_MAP_ZOOM, center=_MAP_CENTER,
                w=None, h=None, margin=None) -> dict:
    d = dict(
        map=dict(style="carto-positron", center=center, zoom=zoom),
        paper_bgcolor=_WHITE,
        font=_MAP_FONT,
        margin=margin or dict(l=0, r=0, t=44, b=0),
        title=dict(text=title, font=dict(size=13, color=_PLOT_TEXT),
                   x=0.5, xanchor="center"),
        legend=_MAP_LEG,
    )
    if w: d["width"]  = w
    if h: d["height"] = h
    return d


def _lntraces(lines_df, buses, color, name, width=1.2, showlegend=True):
    """Scattermap line trace from a lines DataFrame with bus0/bus1 columns."""
    lats, lons = [], []
    for _, r in lines_df.iterrows():
        b0 = buses.loc[r.bus0] if r.bus0 in buses.index else None
        b1 = buses.loc[r.bus1] if r.bus1 in buses.index else None
        if b0 is None or b1 is None:
            continue
        lats += [b0.lat, b1.lat, None]
        lons += [b0.lon, b1.lon, None]
    return go.Scattermap(lat=lats, lon=lons, mode="lines",
                          line=dict(color=color, width=width),
                          name=name, showlegend=showlegend, hoverinfo="skip")


# ── Mercator helpers for pie chart positioning ─────────────────────────────────

def _ll_to_paper(lat, lon, c_lat=39.8, c_lon=-3.5, zoom=_PIE_MAP_ZOOM,
                  w=_PIE_FIG_W, h=_PIE_FIG_H) -> tuple[float, float]:
    """Convert lat/lon to figure paper coords [0,1]×[0,1] for a mapbox projection."""
    scale = 256.0 * (2 ** zoom)
    def _mx(ln): return scale * (ln + 180.0) / 360.0
    def _my(la):
        lr = np.radians(la)
        return scale / (2.0 * np.pi) * (np.pi - np.log(np.tan(np.pi / 4.0 + lr / 2.0)))
    cx, cy = _mx(c_lon), _my(c_lat)
    return 0.5 + (_mx(lon) - cx) / w, 0.5 - (_my(lat) - cy) / h


def _pie_r(total_gwh: float, lo=400, hi=50000,
           rmin=0.013, rmax=0.038) -> float:
    """Log-scaled pie radius in paper units. Returns 0 if below threshold."""
    if total_gwh < lo:
        return 0.0
    t = (np.log(max(total_gwh, lo)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return rmin + min(t, 1.0) * (rmax - rmin)


# ── Map figure builders ────────────────────────────────────────────────────────

def fig_network_basemap() -> go.Figure:
    if TOPO is None:
        return _fig("Base network — topology not available")
    buses = TOPO["buses"]
    fig = go.Figure()
    fig.add_trace(_lntraces(TOPO["es_int"],    buses, "#C8D0DB", "AC lines (ES)",       width=0.9))
    fig.add_trace(_lntraces(TOPO["fr_pt_int"], buses, "#A8BACC", "AC lines (FR / PT)", width=1.0))
    fig.add_trace(_lntraces(TOPO["ic_ac"],     buses, "#2C5F8A", "AC interconnectors",  width=2.2))
    fig.add_trace(_lntraces(TOPO["dc_links"],  buses, "#FF9800", "DC links (Mallorca · INELFE)", width=2.8))

    es_b  = buses[buses.country == "ES"]
    fr_pt = buses[buses.country.isin(["FR", "PT"])]
    fig.add_trace(go.Scattermap(lat=es_b.lat, lon=es_b.lon, mode="markers",
                                 marker=dict(size=5, color="#374151"),
                                 name="ES nodes", hovertext=es_b.index, hoverinfo="text"))
    fig.add_trace(go.Scattermap(lat=fr_pt.lat, lon=fr_pt.lon, mode="markers+text",
                                 marker=dict(size=9, color="#2C5F8A"),
                                 text=fr_pt.index, textposition="top right",
                                 textfont=dict(size=9, color="#2C5F8A"),
                                 name="FR / PT nodes"))
    if "ES1 0" in buses.index:
        m = buses.loc["ES1 0"]
        fig.add_trace(go.Scattermap(lat=[m.lat], lon=[m.lon], mode="markers+text",
                                     marker=dict(size=9, color="#FF9800"),
                                     text=["Mallorca (ES1 0)"], textposition="top left",
                                     textfont=dict(size=9, color="#FF9800"),
                                     name="Mallorca (DC)"))
    fig.update_layout(**_map_layout("Base network map — Spain + interconnectors"))
    return fig


def _curtailment_panel(nd, buses, carrier_cols):
    if nd is None:
        return []
    curt   = nd["curtailment"]
    traces = []
    for carrier, c in carrier_cols.items():
        sub = curt[curt.carrier == carrier]
        if sub.empty: continue
        lats, lons, sizes, texts = [], [], [], []
        for _, row in sub.iterrows():
            if row.bus not in buses.index or row.curtail_gwh < 0.5: continue
            b = buses.loc[row.bus]
            lats.append(b.lat); lons.append(b.lon)
            sizes.append(max(5, np.sqrt(row.curtail_gwh) * 3.5))
            texts.append(f"{row.bus}<br>{carrier}: {row.curtail_gwh:.0f} GWh curtailed")
        traces.append(go.Scattermap(lat=lats, lon=lons, mode="markers",
                                     marker=dict(size=sizes, color=c, opacity=0.75),
                                     text=texts, hoverinfo="text",
                                     name=f"{carrier.capitalize()} curtailment",
                                     legendgroup=carrier, showlegend=True))
    return traces


def fig_curtailment_map(comparison_scenario: str) -> go.Figure:
    if TOPO is None:
        return _fig("Curtailment map — topology not available")
    buses  = TOPO["buses"]
    nd_b   = _NODAL_CACHE.get(_BASELINE_MAP_SCENARIO)
    nd_c   = _NODAL_CACHE.get(comparison_scenario)
    c_cols = {"solar": _AMBER, "onwind": _TEAL, "offwind": "#4DD9C0"}
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "map"}, {"type": "map"}]],
                        subplot_titles=["No battery · Baseline", label(comparison_scenario)],
                        horizontal_spacing=0.02)
    for nd, ci in [(nd_b, 1), (nd_c, 2)]:
        for t in _curtailment_panel(nd, buses, c_cols):
            t.showlegend = (ci == 1)
            fig.add_trace(t, row=1, col=ci)
    mc = dict(style="carto-positron", center=_MAP_CENTER, zoom=4.8)
    fig.update_layout(
        map=mc, map2=mc,
        title=dict(text="Annual VRE curtailment by bus — baseline vs scenario",
                   font=dict(size=13, color=_PLOT_TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=_WHITE, font=_MAP_FONT,
        margin=dict(l=0, r=0, t=48, b=0), legend=_MAP_LEG,
    )
    return fig


def _line_loading_panel(csv_path, buses, showlegend=True):
    """Lines coloured green→red by days/year at >80% loading. Grey = zero days."""
    ll = pd.read_csv(csv_path)
    ll["days"] = ll["pct_gt80pct"] / 100 * 365

    # Green → yellow → orange → red; most lines will be grey (0 days)
    BUCKETS = [
        (0,    0.5,  "#D4D8DE", "0 days (unconstrained)",  0.7),
        (0.5,  5,    "#74C476", "0–5 days/yr",              1.2),
        (5,    15,   "#FDD835", "5–15 days/yr",             1.8),
        (15,   30,   "#F4A261", "15–30 days/yr",            2.4),
        (30,   999,  "#CC3300", ">30 days/yr",              3.4),
    ]
    traces = []
    for lo, hi, color, name, width in BUCKETS:
        bkt = ll[(ll.days >= lo) & (ll.days < hi)]
        lats, lons, hovers = [], [], []
        for _, r in bkt.iterrows():
            b0 = buses.loc[r.bus0] if r.bus0 in buses.index else None
            b1 = buses.loc[r.bus1] if r.bus1 in buses.index else None
            if b0 is None or b1 is None:
                continue
            lats  += [b0.lat, b1.lat, None]
            lons  += [b0.lon, b1.lon, None]
        if lats:
            traces.append(go.Scattermap(
                lat=lats, lon=lons, mode="lines",
                line=dict(color=color, width=width),
                name=name, legendgroup=name,
                showlegend=showlegend, hoverinfo="skip",
            ))
    return traces


def fig_line_loading_map(comparison_scenario: str) -> go.Figure:
    if TOPO is None:
        return _fig("Transmission constraints — topology not available")
    buses = TOPO["buses"]
    bp = DATA_DIR / _BASELINE_MAP_SCENARIO / "line_loading_stats.csv"
    cp = DATA_DIR / comparison_scenario    / "line_loading_stats.csv"
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "map"}, {"type": "map"}]],
                        subplot_titles=["No battery · Baseline", label(comparison_scenario)],
                        horizontal_spacing=0.02)
    for path, ci, sl in [(bp, 1, True), (cp, 2, False)]:
        for t in _line_loading_panel(path, buses, showlegend=sl):
            fig.add_trace(t, row=1, col=ci)
    mc = dict(style="carto-positron", center=_MAP_CENTER, zoom=4.8)
    fig.update_layout(
        map=mc, map2=mc,
        title=dict(text="Transmission constraints — days per year at >80% capacity",
                   font=dict(size=13, color=_PLOT_TEXT), x=0.5, xanchor="center"),
        paper_bgcolor=_WHITE, font=_MAP_FONT,
        margin=dict(l=0, r=0, t=48, b=0), legend=_MAP_LEG,
    )
    return fig


# Carriers shown on dispatch pies — order, colour, label
_DISPATCH_CARRIERS = [
    ("solar",        _AMBER,    "Solar"),
    ("onwind",       "#4DD9C0", "Onshore wind"),
    ("offwind",      "#00A896", "Offshore wind"),
    ("nuclear",      "#4A90D9", "Nuclear"),
    ("hydro",        "#2196F3", "Hydro"),
    ("ror",          "#80DEEA", "Run-of-river"),
    ("PHS",          "#5B8DB8", "PHS"),
    ("CCGT",         _CORAL,    "CCGT"),
    ("CCGT_flex",    "#E04040", "CCGT flex"),
    ("CCGT_must_run","#9E2A2A", "CCGT must-run"),
    ("biomass",      "#66BB6A", "Biomass"),
    ("csp",          "#FF9800", "CSP"),
    ("battery",      "#9C27B0", "Battery"),
]
_DC_COLOR = {c: col for c, col, _ in _DISPATCH_CARRIERS}
_DC_LABEL = {c: lbl for c, _, lbl in _DISPATCH_CARRIERS}


def fig_nodal_dispatch_pies(scenario: str) -> go.Figure:
    """
    go.Pie traces at each ES bus, positioned via Mercator→paper conversion.
    Uses the legacy Scattermapbox / layout.mapbox API — go.Pie overlay only
    works with the old Mapbox stack, not the new Scattermap/MapLibre stack.
    FR / PT buses are excluded.
    """
    nd = _NODAL_CACHE.get(scenario)
    if nd is None or TOPO is None:
        return _fig("Nodal dispatch — data not available")

    all_buses = TOPO["buses"]
    # ES0 buses only
    buses = all_buses[all_buses.index.str.startswith("ES0")].copy()

    disp        = nd["dispatch"]
    es_disp     = disp[disp.bus.str.startswith("ES0")]
    bus_total   = es_disp.groupby("bus")["annual_gwh"].sum()
    bus_carrier = es_disp.set_index(["bus", "carrier"])["annual_gwh"].to_dict()

    fig = go.Figure()

    # ── Background lines — must use Scattermapbox (old API) ─────────────────
    def _smb_line(lines_df, color, width):
        lats, lons = [], []
        for _, r in lines_df.iterrows():
            b0 = all_buses.loc[r.bus0] if r.bus0 in all_buses.index else None
            b1 = all_buses.loc[r.bus1] if r.bus1 in all_buses.index else None
            if b0 is None or b1 is None:
                continue
            lats += [b0.lat, b1.lat, None]
            lons += [b0.lon, b1.lon, None]
        return go.Scattermapbox(lat=lats, lon=lons, mode="lines",
                                 line=dict(color=color, width=width),
                                 showlegend=False, hoverinfo="skip")

    fig.add_trace(_smb_line(TOPO["es_int"],   "#DDE2E6", 0.7))
    fig.add_trace(_smb_line(TOPO["ic_ac"],    "#B0BDCC", 1.4))
    fig.add_trace(_smb_line(TOPO["dc_links"], "#FFD0A0", 1.8))

    # ── Node dots ─────────────────────────────────────────────────────────────
    dot_lat, dot_lon, dot_txt = [], [], []
    for bn in bus_total.index:
        if bn not in buses.index:
            continue
        b = buses.loc[bn]
        dot_lat.append(b.lat); dot_lon.append(b.lon)
        dot_txt.append(f"{bn}: {bus_total[bn]:.0f} GWh total")
    fig.add_trace(go.Scattermapbox(
        lat=dot_lat, lon=dot_lon, mode="markers",
        marker=dict(size=3, color="#374151", opacity=0.55),
        text=dot_txt, hoverinfo="text", showlegend=False))

    # ── go.Pie traces at each bus (Mercator → paper coords) ──────────────────
    aspect = _PIE_FIG_H / _PIE_FIG_W   # y-radius correction for non-square figure

    for bn in bus_total.index:
        total = float(bus_total[bn])
        if bn not in buses.index:
            continue
        r = _pie_r(total)
        if r == 0.0:
            continue

        b = buses.loc[bn]
        xp, yp = _ll_to_paper(b.lat, b.lon)
        if not (0.02 < xp < 0.98 and 0.02 < yp < 0.98):
            continue

        labels, values, colors = [], [], []
        for carrier, c, lbl in _DISPATCH_CARRIERS:
            v = bus_carrier.get((bn, carrier), 0)
            if v > 0.5:
                labels.append(lbl); values.append(v); colors.append(c)
        if not labels:
            continue

        ry = r / aspect
        fig.add_trace(go.Pie(
            labels=labels, values=values,
            domain=dict(
                x=[max(0.0, xp - r),  min(1.0, xp + r)],
                y=[max(0.0, yp - ry), min(1.0, yp + ry)],
            ),
            marker=dict(colors=colors, line=dict(color=_WHITE, width=0.8)),
            textinfo="none",
            hovertemplate=(
                "<b>%{label}</b><br>%{value:.0f} GWh (%{percent})"
                f"<extra>{bn} — {total:.0f} GWh total</extra>"
            ),
            showlegend=False, sort=False,
        ))

    # ── Legend swatches (invisible Scattermapbox markers) ────────────────────
    used = es_disp[es_disp.annual_gwh > 0.5].carrier.unique()
    for carrier, c, lbl in _DISPATCH_CARRIERS:
        if carrier in used:
            fig.add_trace(go.Scattermapbox(
                lat=[None], lon=[None], mode="markers",
                marker=dict(size=10, color=c),
                name=lbl, showlegend=True))

    # ── Layout — old mapbox API required for go.Pie overlay to render ─────────
    fig.update_layout(
        mapbox=dict(style="carto-positron", center=_MAP_CENTER, zoom=_PIE_MAP_ZOOM),
        paper_bgcolor=_WHITE,
        font=_MAP_FONT,
        width=_PIE_FIG_W, height=_PIE_FIG_H,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=_MAP_LEG,
    )
    fig.add_annotation(
        text=f"Nodal annual dispatch by technology — {label(scenario)}",
        xref="paper", yref="paper", x=0.5, y=0.99,
        showarrow=False, font=dict(size=13, color=_PLOT_TEXT),
        bgcolor="rgba(255,255,255,0.85)", borderpad=4,
    )
    return fig


# ── App layout ─────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="PyPSA-Spain scenario explorer",
                suppress_callback_exceptions=True)
server = app.server

_SIDEBAR_OPEN = {
    "width": "260px", "minWidth": "260px", "background": _SLATE,
    "padding": "20px 16px", "overflowY": "auto", "height": "100vh",
    "position": "sticky", "top": "0", "flexShrink": "0",
    "transition": "width 0.2s ease, min-width 0.2s ease, padding 0.2s ease",
}
_SIDEBAR_CLOSED = {
    **_SIDEBAR_OPEN,
    "width": "0", "minWidth": "0", "padding": "0",
    "overflow": "hidden",
}
_MAIN = {
    "flex": "1", "padding": "20px 24px", "background": _BG,
    "overflowY": "auto", "minHeight": "100vh",
}
_TAB  = {"padding": "7px 16px", "fontSize": "12px", "color": _MUTED,
          "background": "none", "borderBottom": "2px solid transparent",
          "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
          "letterSpacing": "0.2px"}
_TAB_SEL = {**_TAB, "color": _PLOT_TEXT,
            "borderBottom": f"2px solid {_TEAL}", "fontWeight": "600"}


def _label_sb(text: str) -> html.Div:
    return html.Div(text, style={"fontSize": "10px", "color": _MUTED,
                                  "fontWeight": "600", "textTransform": "uppercase",
                                  "letterSpacing": "0.5px", "marginBottom": "6px",
                                  "marginTop": "14px"})


sidebar = html.Div([
    # Header row: title + collapse button
    html.Div([
        html.Div([
            html.Div("Scenario explorer",
                     style={"color": _WHITE, "fontWeight": "700", "fontSize": "15px",
                            "letterSpacing": "0.3px",
                            "fontFamily": "Helvetica Neue, Arial, sans-serif"}),
            html.Div("PyPSA-Spain 2024  ·  8 scenarios",
                     style={"color": _MUTED, "fontSize": "10px", "marginTop": "2px"}),
        ], style={"flex": "1"}),
        html.Button("◀", id="btn-collapse", n_clicks=0,
                    title="Collapse sidebar",
                    style={"background": "none", "border": "none",
                           "color": _MUTED, "fontSize": "16px", "cursor": "pointer",
                           "padding": "0 0 0 8px", "lineHeight": "1",
                           "alignSelf": "flex-start"}),
    ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "14px"}),

    html.Hr(style={"borderColor": _GRID_SB, "margin": "0 0 8px 0"}),

    _label_sb("Scenarios"),
    dcc.Checklist(
        id="sel-scenarios",
        options=[{"label": html.Span(label(s), style={"color": col(s), "fontSize": "11px",
                                                        "paddingLeft": "4px"}),
                  "value": s}
                 for s in ALL_SCENARIOS],
        value=ALL_SCENARIOS,
        style={"display": "flex", "flexDirection": "column", "gap": "5px"},
        inputStyle={"accentColor": _TEAL},
    ),

    _label_sb("Month (representative day)"),
    dcc.Dropdown(
        id="sel-month",
        options=[{"label": MONTH_SHORT[m], "value": m} for m in range(1, 13)],
        value=7, clearable=False,
        style={"fontSize": "11px"},
    ),

    _label_sb("Technology (dispatch tab)"),
    dcc.Dropdown(
        id="sel-tech",
        options=[{"label": TECH_LABELS.get(k, k), "value": k} for k in _ES_TECH_ORDER
                 if k in TECH_LABELS],
        value="ES_CCGT", clearable=False,
        style={"fontSize": "11px"},
    ),

    _label_sb("Map comparison scenario"),
    dcc.Dropdown(
        id="sel-map-scenario",
        options=[{"label": label(s), "value": s,
                  "style": {"color": col(s)}}
                 for s in ALL_SCENARIOS],
        value="battery_x1.0", clearable=False,
        style={"fontSize": "11px"},
    ),

    html.Hr(style={"borderColor": _GRID_SB, "margin": "18px 0 10px 0"}),
    html.Div("Each chart has a camera icon (↗) in its toolbar for PNG download.",
             style={"color": _MUTED, "fontSize": "10px", "lineHeight": "1.5",
                    "marginBottom": "10px"}),
    html.Button("Download all charts (ZIP)", id="btn-zip", n_clicks=0,
                style={"width": "100%", "padding": "8px 0", "background": "none",
                       "border": f"1px solid {_GRID_SB}", "borderRadius": "6px",
                       "color": _MUTED, "fontSize": "11px", "cursor": "pointer"}),
    dcc.Download(id="dl-zip"),
    html.Div(id="zip-msg",
             style={"color": _MUTED, "fontSize": "10px", "marginTop": "6px"}),
], id="sidebar", style=_SIDEBAR_OPEN)


def _tab(value: str, label_text: str, *children) -> dcc.Tab:
    return dcc.Tab(
        label=label_text, value=value,
        style=_TAB, selected_style=_TAB_SEL,
        children=[html.Div(list(children), style={"paddingTop": "16px"})],
    )


# Expand button — visible only when sidebar is collapsed
_EXPAND_BTN = html.Button("▶", id="btn-expand", n_clicks=0,
    title="Expand sidebar",
    style={"background": _SLATE, "border": "none", "color": _MUTED,
           "fontSize": "14px", "cursor": "pointer", "padding": "8px 10px",
           "borderRadius": "0 6px 6px 0", "alignSelf": "flex-start",
           "marginTop": "8px", "display": "none",   # hidden by default
           "boxShadow": "2px 0 4px rgba(0,0,0,0.12)"})

app.layout = html.Div([
    dcc.Store(id="sidebar-open", data=True),
    sidebar,
    html.Div([
        _EXPAND_BTN,
        dcc.Tabs(id="tabs", value="tab-price", children=[

            _tab("tab-price", "Price overview",
                html.Div(id="price-cards"),
                _row(_card(_graph("fig-annual-price", "annual_mean_price")),
                     _card(_graph("fig-pdc",          "price_duration_curve"))),
                _row(_card(_graph("fig-violin",       "hourly_price_distribution")),
                     _card(_graph("fig-batt-reduc",   "battery_price_reduction"))),
            ),

            _tab("tab-repday", "Representative day",
                _card(_graph("fig-repday-price", "rep_day_price")),
                _card(_graph("fig-heatmap",      "price_heatmap")),
                _row(_card(_graph("fig-volatility", "monthly_volatility")),
                     _card(_graph("fig-lowprice",   "low_price_hours"))),
            ),

            _tab("tab-curtail", "Curtailment",
                _row(_card(_graph("fig-curtail-scen",  "curtailment_by_scenario")),
                     _card(_graph("fig-curtail-reduc", "curtailment_reduction"))),
                _card(_graph("fig-monthly-curtail", "monthly_curtailment")),
                _row(_card(_graph("fig-rep-curtail", "rep_curtailment_day")),
                     _card(_graph("fig-vre-util",    "vre_utilisation"))),
            ),

            _tab("tab-battery", "Battery usage",
                _card(_graph("fig-bat-repday",  "bat_rep_day")),
                _card(_graph("fig-bess-util",   "bess_utilisation")),
                _row(_card(_graph("fig-bat-revenue", "bat_revenue_dist")),
                     _card(_graph("fig-bat-solar",   "bat_vs_solar"))),
                _row(_card(_graph("fig-arb-spread",  "arbitrage_spread")),
                     _card(_graph("fig-soc-proxy",   "soc_proxy"))),
            ),

            _tab("tab-dispatch", "Dispatch mix",
                _card(_graph("fig-gen-mix",      "annual_gen_mix")),
                _row(_card(_graph("fig-tech-disp",    "tech_dispatch")),
                     _card(_graph("fig-ccgt-displ",   "ccgt_displacement"))),
                _card(_graph("fig-rep-dispatch", "rep_dispatch_day")),
            ),

            _tab("tab-gas", "Gas shock impact",
                _row(_card(_graph("fig-elasticity",    "price_elasticity")),
                     _card(_graph("fig-seasonal-elast","seasonal_elasticity"))),
                _row(_card(_graph("fig-ic-spread",     "ic_price_diff")),
                     _card(_graph("fig-merit-shift",   "merit_order_shift"))),
            ),

            _tab("tab-ccgt", "CCGT fleet",
                _row(_card(_graph("fig-ccgt-tranche",  "ccgt_tranche_dispatch")),
                     _card(_graph("fig-ccgt-capfac",   "ccgt_capacity_factor"))),
                _card(_graph("fig-ccgt-mc",        "ccgt_mc_decomposition")),
                _row(_card(_graph("fig-ccgt-co2",      "ccgt_co2_share")),
                     _card(_graph("fig-ccgt-repday",   "ccgt_rep_day"))),
            ),

            _tab("tab-maps", "Maps",
                _card(dcc.Graph(id="fig-basemap", config=_image_cfg("network_basemap"),
                                style={"height": "560px"})),
                _row(
                    _card(dcc.Graph(id="fig-curt-map",  config=_image_cfg("curtailment_map"),
                                    style={"height": "480px"})),
                    _card(dcc.Graph(id="fig-load-map",  config=_image_cfg("line_loading_map"),
                                    style={"height": "480px"})),
                ),
                _card(dcc.Graph(id="fig-disp-pies", config=_image_cfg("nodal_dispatch"),
                                style={"height": "560px"})),
            ),

        ], colors={"border": _GRID, "primary": _TEAL, "background": _BG}),
    ], id="main-content", style=_MAIN),
], style={"display": "flex", "height": "100vh", "overflow": "hidden",
          "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif"})


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("sidebar",      "style"),
    Output("btn-expand",   "style"),
    Output("sidebar-open", "data"),
    Input("btn-collapse",  "n_clicks"),
    Input("btn-expand",    "n_clicks"),
    State("sidebar-open",  "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(n_collapse, n_expand, is_open):
    from dash import ctx
    if ctx.triggered_id == "btn-collapse":
        is_open = False
    elif ctx.triggered_id == "btn-expand":
        is_open = True
    expand_style = {
        "background": _SLATE, "border": "none", "color": _MUTED,
        "fontSize": "14px", "cursor": "pointer", "padding": "8px 10px",
        "borderRadius": "0 6px 6px 0", "alignSelf": "flex-start",
        "marginTop": "8px", "display": "none" if is_open else "block",
        "boxShadow": "2px 0 4px rgba(0,0,0,0.12)",
    }
    return (_SIDEBAR_OPEN if is_open else _SIDEBAR_CLOSED), expand_style, is_open


@app.callback(
    Output("price-cards",    "children"),
    Output("fig-annual-price","figure"),
    Output("fig-pdc",         "figure"),
    Output("fig-violin",      "figure"),
    Output("fig-batt-reduc",  "figure"),
    Input("sel-scenarios",   "value"),
)
def cb_price(sel):
    sel = sel or []
    return (
        fig_summary_cards(sel),
        fig_annual_mean_price(sel),
        fig_pdc(sel),
        fig_price_violin(sel),
        fig_battery_price_reduction(sel),
    )


@app.callback(
    Output("fig-repday-price", "figure"),
    Output("fig-heatmap",      "figure"),
    Output("fig-volatility",   "figure"),
    Output("fig-lowprice",     "figure"),
    Input("sel-scenarios", "value"),
    Input("sel-month",     "value"),
)
def cb_repday(sel, month):
    sel   = sel   or []
    month = month or 7
    return (
        fig_rep_day_price(sel, month),
        fig_price_heatmap(sel),
        fig_monthly_volatility(sel),
        fig_low_price_hours(sel),
    )


@app.callback(
    Output("fig-curtail-scen",  "figure"),
    Output("fig-curtail-reduc", "figure"),
    Output("fig-monthly-curtail","figure"),
    Output("fig-rep-curtail",   "figure"),
    Output("fig-vre-util",      "figure"),
    Input("sel-scenarios", "value"),
    Input("sel-month",     "value"),
)
def cb_curtail(sel, month):
    sel   = sel   or []
    month = month or 7
    return (
        fig_curtailment_by_scenario(sel),
        fig_curtailment_reduction(sel),
        fig_monthly_curtailment(sel),
        fig_rep_curtailment_day(sel, month),
        fig_vre_utilisation(sel),
    )


@app.callback(
    Output("fig-bat-repday",  "figure"),
    Output("fig-bess-util",   "figure"),
    Output("fig-bat-revenue", "figure"),
    Output("fig-bat-solar",   "figure"),
    Output("fig-arb-spread",  "figure"),
    Output("fig-soc-proxy",   "figure"),
    Input("sel-scenarios", "value"),
    Input("sel-month",     "value"),
)
def cb_battery(sel, month):
    sel     = sel   or []
    month   = month or 7
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]
    return (
        fig_bat_rep_day(bat_sel, month),
        fig_bess_utilisation(bat_sel),
        fig_bat_revenue_dist(bat_sel),
        fig_bat_vs_solar(bat_sel),
        fig_arbitrage_spread(bat_sel),
        fig_soc_proxy(bat_sel),
    )


@app.callback(
    Output("fig-gen-mix",     "figure"),
    Output("fig-tech-disp",   "figure"),
    Output("fig-rep-dispatch","figure"),
    Output("fig-ccgt-displ",  "figure"),
    Input("sel-scenarios", "value"),
    Input("sel-month",     "value"),
    Input("sel-tech",      "value"),
)
def cb_dispatch(sel, month, tech):
    sel   = sel   or []
    month = month or 7
    tech  = tech  or "ES_CCGT"
    return (
        fig_gen_mix(sel),
        fig_tech_dispatch(sel, tech),
        fig_rep_dispatch_day(sel, month),
        fig_ccgt_displacement(sel),
    )


@app.callback(
    Output("fig-elasticity",     "figure"),
    Output("fig-seasonal-elast", "figure"),
    Output("fig-ic-spread",      "figure"),
    Output("fig-merit-shift",    "figure"),
    Input("sel-scenarios", "value"),
)
def cb_gas(sel):
    sel = sel or []
    return (
        fig_price_elasticity(sel),
        fig_seasonal_elasticity(sel),
        fig_ic_price_diff(sel),
        fig_merit_order_shift(sel),
    )


@app.callback(
    Output("fig-ccgt-tranche", "figure"),
    Output("fig-ccgt-capfac",  "figure"),
    Output("fig-ccgt-mc",      "figure"),
    Output("fig-ccgt-co2",     "figure"),
    Output("fig-ccgt-repday",  "figure"),
    Input("sel-scenarios", "value"),
    Input("sel-month",     "value"),
)
def cb_ccgt(sel, month):
    sel   = sel   or []
    month = month or 7
    return (
        fig_ccgt_tranche_dispatch(sel),
        fig_ccgt_capacity_factor(sel),
        fig_ccgt_mc_decomposition(),
        fig_ccgt_co2_share(),
        fig_ccgt_rep_day(sel, month),
    )


# ── ZIP download (server-side kaleido) ────────────────────────────────────────

@app.callback(
    Output("fig-basemap",   "figure"),
    Output("fig-curt-map",  "figure"),
    Output("fig-load-map",  "figure"),
    Output("fig-disp-pies", "figure"),
    Input("sel-map-scenario", "value"),
)
def cb_maps(map_scenario):
    map_scenario = map_scenario or _BASELINE_MAP_SCENARIO
    return (
        fig_network_basemap(),
        fig_curtailment_map(map_scenario),
        fig_line_loading_map(map_scenario),
        fig_nodal_dispatch_pies(map_scenario),
    )


@app.callback(
    Output("dl-zip",  "data"),
    Output("zip-msg", "children"),
    Input("btn-zip", "n_clicks"),
    State("sel-scenarios", "value"),
    State("sel-month",     "value"),
    State("sel-tech",      "value"),
    prevent_initial_call=True,
)
def cb_zip(n_clicks, sel, month, tech):
    if not n_clicks:
        raise PreventUpdate
    try:
        import kaleido  # noqa
    except ImportError:
        return dash.no_update, "Run: pip install kaleido  to enable ZIP export"

    sel     = sel   or []
    month   = month or 7
    tech    = tech  or "ES_CCGT"
    bat_sel = [s for s in sel if SCENARIOS[s]["bat_agg"] is not None]

    charts = [
        ("01_annual_mean_price",       fig_annual_mean_price(sel)),
        ("01_price_duration_curve",    fig_pdc(sel)),
        ("01_price_distribution",      fig_price_violin(sel)),
        ("01_battery_price_reduction", fig_battery_price_reduction(sel)),
        ("02_rep_day_price",           fig_rep_day_price(sel, month)),
        ("02_price_heatmap",           fig_price_heatmap(sel)),
        ("02_monthly_volatility",      fig_monthly_volatility(sel)),
        ("02_low_price_hours",         fig_low_price_hours(sel)),
        ("03_curtailment_by_scenario", fig_curtailment_by_scenario(sel)),
        ("03_curtailment_reduction",   fig_curtailment_reduction(sel)),
        ("03_monthly_curtailment",     fig_monthly_curtailment(sel)),
        ("03_rep_curtailment_day",     fig_rep_curtailment_day(sel, month)),
        ("03_vre_utilisation",         fig_vre_utilisation(sel)),
        ("04_bat_rep_day",             fig_bat_rep_day(bat_sel, month)),
        ("04_bess_utilisation",        fig_bess_utilisation(bat_sel)),
        ("04_bat_revenue_dist",        fig_bat_revenue_dist(bat_sel)),
        ("04_bat_vs_solar",            fig_bat_vs_solar(bat_sel)),
        ("04_arbitrage_spread",        fig_arbitrage_spread(bat_sel)),
        ("04_soc_proxy",               fig_soc_proxy(bat_sel)),
        ("05_gen_mix",                 fig_gen_mix(sel)),
        ("05_tech_dispatch",           fig_tech_dispatch(sel, tech)),
        ("05_rep_dispatch_day",        fig_rep_dispatch_day(sel, month)),
        ("05_ccgt_displacement",       fig_ccgt_displacement(sel)),
        ("06_price_elasticity",        fig_price_elasticity(sel)),
        ("06_seasonal_elasticity",     fig_seasonal_elasticity(sel)),
        ("06_ic_price_diff",           fig_ic_price_diff(sel)),
        ("06_merit_order_shift",       fig_merit_order_shift(sel)),
        ("07_ccgt_tranche_dispatch",   fig_ccgt_tranche_dispatch(sel)),
        ("07_ccgt_capacity_factor",    fig_ccgt_capacity_factor(sel)),
        ("07_ccgt_mc_decomposition",   fig_ccgt_mc_decomposition()),
        ("07_ccgt_co2_share",          fig_ccgt_co2_share()),
        ("07_ccgt_rep_day",            fig_ccgt_rep_day(sel, month)),
    ]

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, fig in charts:
            img = io.BytesIO()
            pio.write_image(fig, img, format="png", width=1400, height=700, scale=2)
            zf.writestr(f"{fname}.png", img.getvalue())

    zip_buf.seek(0)
    return dcc.send_bytes(zip_buf.read(), "pypsa_spain_scenarios.zip"), "ZIP ready ✓"


if __name__ == "__main__":
    app.run(debug=False, port=8051)
