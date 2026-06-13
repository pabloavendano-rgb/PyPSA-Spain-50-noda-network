"""
PyPSA-Spain Energy Market Dashboard

Run from repo root:
    python Analysis/dashboard.py

Opens at http://localhost:8050
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json

import dash
from dash import dcc, html, dash_table, Input, Output, State
from dash.exceptions import PreventUpdate

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from run_validation import COLORS as _BASE_COLORS, CARRIER_ORDER as _BASE_ORDER
from dashboard_utils import (
    list_solved_networks,
    load_and_extract,
    deserialise,
    start_solve,
    poll_solve_result,
    clear_solve_state,
    MODEL_CONFIG,
)

log = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
_SLATE     = "#1E252B"   # sidebar / chrome (stays dark)
_PLOT_TEXT = "#374151"   # softer text inside charts
_WHITE     = "#FFFFFF"   # plot background
_GRID      = "#E8ECF0"   # lighter grid lines
_GRID_SB   = "#2e3e4e"   # separator in sidebar
_CORAL     = "#E85D5D"   # re-solve button (slightly muted)
_TEAL      = "#00A896"
_AMBER     = "#F4A261"
_MUTED     = "#8896A7"   # secondary labels

COLORS = {
    **_BASE_COLORS,
    "CCGT_must_run": "#2C3E50",
    "cogen":         "#7E57C2",   # industrial gas CHP (ENTSO-E "Cogeneration") — purple
    "other":         "#B0B7C3",
    # CCGT tranches — three shades of the base CCGT coral
    "CCGT_lo":  "#FFAA85",   # cheapest tier (65–84 €/MWh) — light coral
    "CCGT_mid": "#FF6B6B",   # mid tier (84–93 €/MWh) — base coral
    "CCGT_hi":  "#B03030",   # expensive tier (>93 €/MWh) — dark red
}
# Replace the bare "CCGT" slot with three ordered tranches
CARRIER_ORDER = [c for c in _BASE_ORDER if c not in {"CCGT", "CCGT_must_run"}] + \
                ["CCGT_lo", "CCGT_mid", "CCGT_hi", "CCGT_must_run"]

_PLOT_BASE = dict(
    paper_bgcolor=_WHITE,
    plot_bgcolor=_WHITE,
    font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
              color=_PLOT_TEXT, size=11),
    xaxis=dict(gridcolor=_GRID, gridwidth=0.5, griddash="dash",
               linecolor=_GRID, tickcolor=_MUTED, showgrid=True),
    yaxis=dict(gridcolor=_GRID, gridwidth=0.5, griddash="dash",
               linecolor=_GRID, tickcolor=_MUTED, showgrid=True),
    margin=dict(l=58, r=60, t=32, b=38),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                borderwidth=1, font=dict(size=10, color=_PLOT_TEXT),
                orientation="h", x=0, y=-0.22, xanchor="left"),
)

# Mark style for sliders on the dark sidebar
def _mk(*vals, fmt=str):
    return {v: {"label": fmt(v), "style": {"color": "#A8B5C0", "fontSize": "10px"}} for v in vals}


# ── Data helpers ──────────────────────────────────────────────────────────────

def _slice(d: dict, i0: int, i1: int) -> dict:
    """Integer-index slice of all time-series in a deserialised dict."""
    i1 = max(i0 + 1, i1)
    out = {"timestamps": d["timestamps"][i0:i1]}
    _df_keys = {"dispatch_es", "dispatch_fr", "dispatch_pt", "ree_actual"}
    for k in ("dispatch_es", "dispatch_fr", "dispatch_pt", "ree_actual",
              "price_es", "price_tw_t", "setter_es", "fr_net", "pt_net", "omie", "es_load",
              "fr_load_t", "pt_load_t", "fr_price_t", "pt_price_t",
              "fr_wind_t", "fr_solar_t", "fr_surplus_t",
              "actual_fr_t", "actual_pt_t", "omie_fr", "omie_pt"):
        v = d.get(k)
        if v is None:
            out[k] = pd.DataFrame() if k in _df_keys else pd.Series(dtype=float)
            continue
        if isinstance(v, (pd.DataFrame, pd.Series)):
            out[k] = v.iloc[i0:i1]
        else:
            out[k] = v
    bp = d.get("bus_prices", pd.DataFrame())
    out["bus_prices"] = bp.iloc[i0:i1] if isinstance(bp, pd.DataFrame) and not bp.empty else bp
    return out


# ── Figure builders ───────────────────────────────────────────────────────────

def _make_summary_cards(d: dict) -> list:
    """Stat cards for top of dispatch tab: MAE, bias, IC hours model vs actual."""
    price   = d.get("price_es", pd.Series(dtype=float))
    omie    = d.get("omie")
    fr_net  = d.get("fr_net",      pd.Series(dtype=float))
    pt_net  = d.get("pt_net",      pd.Series(dtype=float))
    afr     = d.get("actual_fr_t", pd.Series(dtype=float))
    apt     = d.get("actual_pt_t", pd.Series(dtype=float))

    def _card(label, value_str, sub="", value_color=_PLOT_TEXT, bg=_WHITE):
        return html.Div([
            html.Div(label, style={"fontSize": "10px", "color": _MUTED,
                                   "fontWeight": "600", "marginBottom": "3px",
                                   "textTransform": "uppercase", "letterSpacing": "0.5px"}),
            html.Div(value_str, style={"fontSize": "18px", "fontWeight": "700",
                                       "color": value_color, "lineHeight": "1.1"}),
            html.Div(sub, style={"fontSize": "10px", "color": _MUTED,
                                 "marginTop": "2px"}),
        ], style={
            "backgroundColor": bg,
            "border": f"1px solid {_GRID}",
            "borderRadius": "10px",
            "padding": "10px 14px",
            "minWidth": "130px",
            "flex": "1",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
        })

    cards = []

    # ── MAE and bias ─────────────────────────────────────────────────────────
    if omie is not None and not price.empty and len(omie) == len(price):
        err  = (price - omie).dropna()
        mae  = float(err.abs().mean())
        bias = float(err.mean())
        rmse = float(np.sqrt((err**2).mean()))
        bias_col = _CORAL if bias > 5 else (_TEAL if bias < -5 else "#2196F3")
        cards.append(_card("MAE vs OMIE",   f"€{mae:.1f}",
                           f"RMSE €{rmse:.1f}  ({len(err)} hrs)"))
        cards.append(_card("Mean error",    f"{bias:+.1f} €/MWh",
                           "model − OMIE", value_color=bias_col))
    else:
        cards.append(_card("MAE vs OMIE",   "—", "no OMIE data"))
        cards.append(_card("Mean error",    "—"))

    # ── FR import hours ───────────────────────────────────────────────────────
    IC_THRESH = 200.0
    if not fr_net.empty:
        m_fr = int((fr_net > IC_THRESH).sum())
        a_fr = int((afr   > IC_THRESH).sum()) if not afr.empty else None
        a_str = f"actual {a_fr} h" if a_fr is not None else "actual n/a"
        pct   = m_fr / max(1, len(fr_net)) * 100
        sub   = f"{a_str}  |  >{IC_THRESH:.0f} MW threshold"
        col   = _TEAL if a_fr is None or abs(m_fr - a_fr) < a_fr * 0.15 else _CORAL
        cards.append(_card("FR import hrs", f"{m_fr}", sub, value_color=col))
    else:
        cards.append(_card("FR import hrs", "—"))

    # ── PT import hours ───────────────────────────────────────────────────────
    if not pt_net.empty:
        m_pt = int((pt_net > IC_THRESH).sum())
        a_pt = int((apt    > IC_THRESH).sum()) if not apt.empty else None
        a_str = f"actual {a_pt} h" if a_pt is not None else "actual n/a"
        sub   = f"{a_str}  |  >{IC_THRESH:.0f} MW threshold"
        col   = _TEAL if a_pt is None or abs(m_pt - a_pt) < a_pt * 0.15 else _CORAL
        cards.append(_card("PT import hrs", f"{m_pt}", sub, value_color=col))
    else:
        cards.append(_card("PT import hrs", "—"))

    # ── IC coverage ──────────────────────────────────────────────────────────
    n_total = len(d.get("timestamps", []))
    a_cov   = int(afr.notna().sum()) if not afr.empty else 0
    pct_cov = a_cov / max(1, n_total) * 100
    cards.append(_card("IC data coverage", f"{pct_cov:.0f}%",
                       f"{a_cov}/{n_total} actual hrs"))

    # ── TW vs LW price bias ───────────────────────────────────────────────────
    price_tw = d.get("price_tw_t")
    if (isinstance(price_tw, pd.Series) and not price_tw.empty
            and not price.empty and len(price_tw) == len(price)):
        tw_mean  = float(price_tw.mean())
        lw_mean  = float(price.mean())
        bias_eur = lw_mean - tw_mean          # positive: LW > TW (uplift)
        bias_col = _AMBER if abs(bias_eur) > 2 else _TEAL
        cards.append(_card(
            "LW vs TW bias",
            f"{bias_eur:+.1f} €/MWh",
            f"TW €{tw_mean:.1f}  |  LW €{lw_mean:.1f}",
            value_color=bias_col,
        ))
    else:
        cards.append(_card("LW vs TW bias", "—", "load network to compute"))

    return [html.Div(cards, style={
        "display": "flex", "gap": "10px", "flexWrap": "wrap",
        "marginBottom": "10px", "marginTop": "4px",
    })]


def make_price_error_figure(d: dict, i0: int, i1: int) -> go.Figure:
    """2-panel: price error timeseries (left) + tech correlation bars (right)."""
    s       = _slice(d, i0, i1)
    price   = s.get("price_es", pd.Series(dtype=float))
    omie    = s.get("omie")
    ts      = list(s["timestamps"])
    dispatch = s.get("dispatch_es", pd.DataFrame())
    afr     = s.get("actual_fr_t", pd.Series(dtype=float))
    apt     = s.get("actual_pt_t", pd.Series(dtype=float))
    fr_net  = s.get("fr_net",      pd.Series(dtype=float))
    pt_net  = s.get("pt_net",      pd.Series(dtype=float))

    if omie is None or price.empty or len(omie) != len(price):
        fig = go.Figure()
        fig.update_layout(**{**_PLOT_BASE, "height": 300,
                             "title": dict(text="Price Error vs OMIE — no OMIE data loaded",
                                           font=dict(size=11, color=_MUTED))})
        return fig

    err  = (price - omie).fillna(0.0)
    roll = err.rolling(24, min_periods=4, center=True).mean()
    mae  = float(err.abs().mean())
    bias = float(err.mean())
    rmse = float(np.sqrt((err**2).mean()))

    # ── Positive / negative error fill ───────────────────────────────────────
    pos_y = [max(0.0, v) for v in err.tolist()]
    neg_y = [min(0.0, v) for v in err.tolist()]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.68, 0.32],
        subplot_titles=["Model − OMIE price error (€/MWh)",
                        "Correlation with price error"],
        horizontal_spacing=0.06,
    )

    # Positive fill (model > OMIE)
    fig.add_trace(go.Scatter(
        x=ts, y=pos_y, name="Error > 0 (over-priced)",
        mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
        fill="tozeroy", fillcolor="rgba(232,93,93,0.25)",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Negative fill (model < OMIE)
    fig.add_trace(go.Scatter(
        x=ts, y=neg_y, name="Error < 0 (under-priced)",
        mode="lines", line=dict(width=0, color="rgba(0,0,0,0)"),
        fill="tozeroy", fillcolor="rgba(0,168,150,0.20)",
        showlegend=True,
        hoverinfo="skip",
    ), row=1, col=1)

    # Hourly error line
    fig.add_trace(go.Scatter(
        x=ts, y=err.tolist(), name="Hourly error",
        mode="lines", line=dict(color="#374151", width=0.7),
        hovertemplate="<b>Error</b>: %{y:+.1f} €/MWh<extra></extra>",
    ), row=1, col=1)

    # 24h rolling mean
    fig.add_trace(go.Scatter(
        x=ts, y=roll.tolist(), name="24h rolling mean",
        mode="lines", line=dict(color=_CORAL, width=2.2),
        hovertemplate="<b>24h mean error</b>: %{y:+.1f} €/MWh<extra></extra>",
    ), row=1, col=1)

    # Zero reference line
    fig.add_hline(y=0, line=dict(color=_GRID, width=1.0), row=1, col=1)

    # Annotation: stats
    fig.add_annotation(
        xref="x domain", yref="y domain",
        x=0.02, y=0.97, xanchor="left", yanchor="top",
        text=f"MAE {mae:.1f}  |  RMSE {rmse:.1f}  |  Bias {bias:+.1f}  €/MWh",
        showarrow=False,
        font=dict(size=10, color=_PLOT_TEXT),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor=_GRID, borderwidth=1,
        row=1, col=1,
    )

    # ── Correlation bars ──────────────────────────────────────────────────────
    err_vals = err.values
    corr_items = []

    # ES dispatch carriers
    if isinstance(dispatch, pd.DataFrame) and not dispatch.empty:
        for col in dispatch.columns:
            v = dispatch[col].values
            if len(v) == len(err_vals) and v.std() > 0.01:
                r = float(np.corrcoef(err_vals, v)[0, 1])
                if not np.isnan(r):
                    corr_items.append((f"ES {col}", r))

    # IC flows
    for label, series in [("FR model flow", fr_net), ("PT model flow", pt_net),
                           ("FR actual flow", afr), ("PT actual flow", apt)]:
        if isinstance(series, pd.Series) and not series.empty and len(series) == len(err_vals):
            v = series.values
            if v.std() > 0.01:
                r = float(np.corrcoef(err_vals, v)[0, 1])
                if not np.isnan(r):
                    corr_items.append((label, r))

    if corr_items:
        corr_items.sort(key=lambda x: abs(x[1]), reverse=True)
        corr_items = corr_items[:16]  # top 16 by |r|
        labels = [x[0] for x in corr_items]
        rs     = [x[1] for x in corr_items]
        colors = [_CORAL if r > 0 else _TEAL for r in rs]
        fig.add_trace(go.Bar(
            x=rs, y=labels, orientation="h",
            marker_color=colors,
            text=[f"{r:+.2f}" for r in rs],
            textposition="outside",
            textfont=dict(size=9, color=_PLOT_TEXT),
            name="Pearson r",
            hovertemplate="<b>%{y}</b>: r=%{x:+.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
        fig.add_vline(x=0, line=dict(color=_GRID, width=1), row=1, col=2)
        fig.update_xaxes(range=[-1.05, 1.05], row=1, col=2)

    layout = dict(**_PLOT_BASE)
    layout["height"] = 320
    layout["title"]  = dict(text="Price Error Analysis  (model vs OMIE)",
                             font=dict(size=12, color=_PLOT_TEXT))
    layout["hovermode"] = "x unified"
    layout["yaxis"]  = dict(_PLOT_BASE["yaxis"],
                             title_text="Price error (€/MWh)",
                             title_font=dict(size=10, color=_MUTED),
                             zeroline=True, zerolinecolor=_GRID)
    layout["yaxis2"] = dict(
        gridcolor="rgba(0,0,0,0)", showgrid=False,
        tickfont=dict(size=9, color=_PLOT_TEXT),
    )
    layout["xaxis2"] = dict(
        title_text="Pearson r  (correlation with price error)",
        title_font=dict(size=9, color=_MUTED),
        gridcolor=_GRID, gridwidth=0.5, tickcolor=_MUTED, showgrid=True,
    )
    layout["legend"] = dict(
        bgcolor="rgba(255,255,255,0.90)", bordercolor=_GRID, borderwidth=1,
        font=dict(size=10), orientation="h", x=0, y=-0.26, xanchor="left",
    )
    layout["margin"] = dict(l=58, r=100, t=50, b=50)
    fig.update_layout(**layout)
    return fig


def _stack_traces(fig, dispatch: pd.DataFrame, ts: list,
                  stackgroup: str, showlegend: bool = True, prefix: str = ""):
    """Add stacked area traces for carrier dispatch to fig (yaxis='y').

    Storage charging shows as negative in raw dispatch — clip to 0 so the
    generation stack stays clean.  Charging is a load, not supply, and would
    otherwise punch a large hole through whatever carrier sits above it in the
    stacked area.
    """
    present = [c for c in CARRIER_ORDER if c in dispatch.columns]
    present += [c for c in dispatch.columns if c not in CARRIER_ORDER]
    for carrier in present:
        raw  = dispatch[carrier].tolist()
        vals = [max(0.0, v) for v in raw]   # clip charging (negative) to 0
        if max(vals) < 1.0:
            continue
        label = f"{prefix}{carrier}" if prefix else carrier
        fig.add_trace(go.Scatter(
            x=ts, y=vals,
            name=label,
            yaxis="y",
            mode="lines",
            line=dict(width=0.3, color=COLORS.get(carrier, "#BBBBBB")),
            fillcolor=COLORS.get(carrier, "#BBBBBB"),
            stackgroup=stackgroup,
            showlegend=showlegend,
            hovertemplate=f"<b>{label}</b>: %{{y:.0f}} MW<extra></extra>",
        ))


def make_dispatch_figure(d: dict, i0: int, i1: int) -> go.Figure:
    """Stacked area dispatch (yaxis) + price overlay (yaxis2).

    Uses go.Figure() directly — make_subplots(secondary_y=True) silently drops
    secondary trace data in Plotly 6.x.
    """
    s        = _slice(d, i0, i1)
    dispatch = s["dispatch_es"]
    price    = s["price_es"]
    setter   = s["setter_es"]
    fr_net   = s["fr_net"]
    omie     = s["omie"]
    ts       = list(s["timestamps"])

    fig = go.Figure()

    if isinstance(dispatch, pd.DataFrame) and not dispatch.empty:
        _stack_traces(fig, dispatch, ts, "one")

    load = s.get("es_load", pd.Series(dtype=float))
    if not load.empty:
        fig.add_trace(go.Scatter(
            x=ts, y=load.tolist(),
            name="ES Demand",
            yaxis="y",
            mode="lines",
            line=dict(color=_SLATE, width=2.2, dash="dash"),
            hovertemplate="<b>ES Demand</b>: %{y:.0f} MW<extra></extra>",
        ))

    if len(price) > 0:
        fig.add_trace(go.Scatter(
            x=ts, y=price.tolist(),
            name="Model price",
            yaxis="y2",
            mode="lines",
            line=dict(color=_SLATE, width=1.8),
            hovertemplate="<b>Model price</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

    if omie is not None and len(omie) > 0:
        fig.add_trace(go.Scatter(
            x=ts, y=omie.tolist(),
            name="OMIE actual",
            yaxis="y2",
            mode="lines",
            line=dict(color=_AMBER, width=1.4, dash="dot"),
            hovertemplate="<b>OMIE</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

    # Price setter — one trace per unique carrier (all hours, no subsampling)
    # Separate traces let the user toggle individual carriers in the legend.
    if len(price) > 0 and len(setter) > 0:
        unique_setters = [s for s in setter.unique() if str(s) not in ("nan", "None", "")]
        for carrier in unique_setters:
            mask     = setter == carrier
            carr_ts  = [ts[i] for i, m in enumerate(mask.tolist()) if m]
            carr_pr  = [float(price.iloc[i]) for i, m in enumerate(mask.tolist()) if m]
            if not carr_ts:
                continue
            col = COLORS.get(carrier, "#888888")
            fig.add_trace(go.Scatter(
                x=carr_ts, y=carr_pr,
                yaxis="y2",
                mode="markers",
                marker=dict(
                    color=col, size=8, symbol="circle",
                    line=dict(width=1.5, color="white"),
                    opacity=0.92,
                ),
                name=f"Setter: {carrier}",
                hovertemplate=f"<b>Setter: {carrier}</b><br>€%{{y:.1f}}/MWh<extra></extra>",
                legendgroup="setter",
                legendgrouptitle_text="Price setter",
            ))

    if len(fr_net) > 0:
        fig.add_trace(go.Scatter(
            x=ts, y=fr_net.tolist(),
            name="FR import (MW)",
            yaxis="y2",
            mode="lines",
            line=dict(color="#457B9D", width=0.8, dash="dot"),
            visible="legendonly",
            hovertemplate="<b>FR→ES</b>: %{y:.0f} MW<extra></extra>",
        ))

    layout = dict(**_PLOT_BASE)
    layout["height"]   = 400
    layout["title"]    = dict(text="Spain — Dispatch & Price",
                               font=dict(size=12, color=_PLOT_TEXT))
    layout["dragmode"] = "pan"
    layout["yaxis"]    = dict(_PLOT_BASE["yaxis"],
                               title_text="Dispatch (MW)",
                               title_font=dict(size=10, color=_MUTED),
                               rangemode="tozero")
    layout["yaxis2"]   = dict(
        title_text="Price (€/MWh)",
        title_font=dict(size=10, color=_MUTED),
        overlaying="y", side="right",
        tickcolor=_MUTED,
        gridcolor="rgba(0,0,0,0)",
        showgrid=False,
        zeroline=False,
    )
    fig.update_layout(**layout)
    return fig


def _make_country_flow_figure(ts: list, dispatch: pd.DataFrame,
                               net: pd.Series, country: str, height: int,
                               load: pd.Series | None = None,
                               actual: pd.Series | None = None) -> go.Figure:
    """
    Country (FR or PT) generation stack + net flow to Spain.

    Stacked area = what that country is generating by technology (carrier colours).
    Dashed line + fill = net MW flowing into Spain (positive = importing to ES).
    """
    fig = go.Figure()

    flow_color = "#457B9D" if country == "FR" else _TEAL
    flow_rgba  = "rgba(69,123,157,0.25)" if country == "FR" else "rgba(0,168,150,0.25)"

    # Dispatch stack (left y — full generation scale)
    if isinstance(dispatch, pd.DataFrame) and not dispatch.empty:
        _stack_traces(fig, dispatch, ts, f"disp_{country}", showlegend=True, prefix=f"{country} ")

    # Country demand line (on left axis so it sits against the gen stack)
    if load is not None and not load.empty:
        fig.add_trace(go.Scatter(
            x=ts, y=load.tolist(),
            name=f"{country} Demand",
            yaxis="y",
            mode="lines",
            line=dict(color=_SLATE, width=2.0, dash="dash"),
            hovertemplate=f"<b>{country} Demand</b>: %{{y:.0f}} MW<extra></extra>",
        ))

    # Net flow to ES (right y — interconnector scale)
    if len(net) > 0:
        net_vals = net.tolist()
        # Import area (positive)
        import_vals = [max(0.0, v) for v in net_vals]
        fig.add_trace(go.Scatter(
            x=ts, y=import_vals,
            name=f"{country}→ES import",
            yaxis="y2",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tozeroy",
            fillcolor=flow_rgba,
            showlegend=False,
            hoverinfo="skip",
        ))
        # Export area (negative)
        export_vals = [min(0.0, v) for v in net_vals]
        fig.add_trace(go.Scatter(
            x=ts, y=export_vals,
            name=f"ES→{country} export",
            yaxis="y2",
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=0),
            fill="tozeroy",
            fillcolor="rgba(232,93,93,0.18)",
            showlegend=False,
            hoverinfo="skip",
        ))
        # Net flow line (model)
        fig.add_trace(go.Scatter(
            x=ts, y=net_vals,
            name=f"ES↔{country} model flow",
            yaxis="y2",
            mode="lines",
            line=dict(color=flow_color, width=1.8),
            hovertemplate=f"<b>{country}↔ES model</b>: %{{y:.0f}} MW<extra></extra>",
        ))

        # Actual ENTSOE flow overlay
        if actual is not None and not actual.empty and actual.abs().sum() > 1.0:
            fig.add_trace(go.Scatter(
                x=ts, y=actual.tolist(),
                name=f"ES↔{country} actual (ENTSOE)",
                yaxis="y2",
                mode="lines",
                line=dict(color="#555555", width=1.2, dash="dot"),
                hovertemplate=f"<b>{country}↔ES actual</b>: %{{y:.0f}} MW<extra></extra>",
            ))

    fig.add_hline(y=0, line=dict(color=_GRID, width=0.8))

    layout = dict(**_PLOT_BASE)
    layout["height"] = height
    layout["title"]  = dict(
        text=f"{country} Generation Mix & Flow to Spain",
        font=dict(size=11, color=_PLOT_TEXT))
    layout["yaxis"]  = dict(_PLOT_BASE["yaxis"],
                             title_text=f"{country} generation (MW)",
                             title_font=dict(size=9, color=_MUTED),
                             rangemode="tozero")
    layout["yaxis2"] = dict(
        title_text="Flow to ES (MW, +import)",
        title_font=dict(size=9, color=_MUTED),
        overlaying="y", side="right",
        tickcolor=_MUTED,
        showgrid=False,
        zeroline=True,
        zerolinecolor=_GRID,
        zerolinewidth=1,
    )
    layout["margin"]  = dict(l=58, r=70, t=30, b=38)
    layout["legend"]  = dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=_GRID,
                              borderwidth=1, font=dict(size=9),
                              orientation="h", x=0, y=-0.28, xanchor="left")
    fig.update_layout(**layout)
    return fig


def make_flow_fr_figure(d: dict, i0: int, i1: int) -> go.Figure:
    s = _slice(d, i0, i1)
    return _make_country_flow_figure(
        list(s["timestamps"]), s.get("dispatch_fr", pd.DataFrame()),
        s["fr_net"], "FR", 260,
        load=s.get("fr_load_t", pd.Series(dtype=float)),
        actual=s.get("actual_fr_t", pd.Series(dtype=float)))


def make_flow_pt_figure(d: dict, i0: int, i1: int) -> go.Figure:
    s = _slice(d, i0, i1)
    return _make_country_flow_figure(
        list(s["timestamps"]), s.get("dispatch_pt", pd.DataFrame()),
        s["pt_net"], "PT", 240,
        load=s.get("pt_load_t", pd.Series(dtype=float)),
        actual=s.get("actual_pt_t", pd.Series(dtype=float)))


def make_ree_figure(d: dict, i0: int, i1: int) -> go.Figure:
    """REE actual dispatch stacked area (ENTSO-E 2024 data, carrier-coloured)."""
    s   = _slice(d, i0, i1)
    ree = s.get("ree_actual", pd.DataFrame())
    ts  = list(s["timestamps"])

    fig = go.Figure()
    if isinstance(ree, pd.DataFrame) and not ree.empty:
        _stack_traces(fig, ree, ts, "ree_stack")

    layout = dict(**_PLOT_BASE)
    layout["height"] = 340
    layout["title"]  = dict(text="REE Actual Dispatch — ENTSO-E 2024",
                             font=dict(size=12, color=_PLOT_TEXT))
    layout["yaxis"]  = dict(_PLOT_BASE["yaxis"],
                             title_text="Actual generation (MW)",
                             title_font=dict(size=10, color=_MUTED),
                             rangemode="tozero")
    fig.update_layout(**layout)
    return fig


# ── Carrier aggregation for model-vs-actual bar comparison ───────────────────
_MODEL_AGG = {
    "CCGT": "CCGT", "CCGT_lo": "CCGT", "CCGT_mid": "CCGT", "CCGT_hi": "CCGT",
    "CCGT_flex": "CCGT", "CCGT_must_run": "CCGT", "cogen": "CCGT",
    "OCGT": "OCGT",
    "nuclear": "Nuclear",
    "hydro": "Hydro", "ror": "Hydro",
    "phs": "PHS",
    "wind": "Wind", "onwind": "Wind", "offwind-ac": "Wind", "offwind-dc": "Wind",
    "solar": "Solar", "solar_rooftop": "Solar",
    "biomass": "Biomass",
    "coal": "Coal", "lignite": "Coal",
    "oil": "Oil",
}
_AGG_COLORS = {
    "CCGT": COLORS.get("CCGT_mid", "#FF6B6B"),
    "OCGT": COLORS.get("OCGT", "#E67E22"),
    "Nuclear": COLORS.get("nuclear", "#9B59B6"),
    "Hydro": COLORS.get("hydro", "#2980B9"),
    "PHS": COLORS.get("phs", "#1ABC9C"),
    "Wind": COLORS.get("wind", "#27AE60"),
    "Solar": COLORS.get("solar", "#F1C40F"),
    "Biomass": COLORS.get("biomass", "#795548"),
    "Coal": COLORS.get("coal", "#607D8B"),
    "Oil": COLORS.get("oil", "#BDC3C7"),
}
_AGG_ORDER = ["Nuclear", "Hydro", "PHS", "Wind", "Solar", "Biomass",
              "CCGT", "OCGT", "Coal", "Oil"]


def _agg_ree_col(col: str) -> str:
    """Map an ENTSO-E/REE column name to a display category."""
    c = col.lower()
    if any(x in c for x in ("nuclear",)):                return "Nuclear"
    if any(x in c for x in ("hydro",)) and "pump" not in c: return "Hydro"
    if any(x in c for x in ("pump", "phs")):             return "PHS"
    if any(x in c for x in ("wind",)):                   return "Wind"
    if any(x in c for x in ("solar", "photo", "pv")):    return "Solar"
    if any(x in c for x in ("biomass", "waste", "biogas")): return "Biomass"
    if any(x in c for x in ("gas", "ccgt", "thermal", "combined")): return "CCGT"
    if any(x in c for x in ("ocgt", "open", "turbine")): return "OCGT"
    if any(x in c for x in ("coal", "lignite", "hard")):  return "Coal"
    if any(x in c for x in ("oil", "fuel", "diesel")):   return "Oil"
    return col


def make_total_dispatch_bar(d: dict, i0: int, i1: int) -> go.Figure:
    """Grouped bar: model total GWh vs REE actual GWh per technology for the selected window.

    A dashed horizontal line marks total Spanish demand so you can see whether local
    generation covers load or whether net imports are filling the gap.
    """
    s        = _slice(d, i0, i1)
    dispatch = s.get("dispatch_es", pd.DataFrame())
    ree      = s.get("ree_actual",  pd.DataFrame())
    load     = s.get("es_load",     pd.Series(dtype=float))

    # ── Aggregate model dispatch ───────────────────────────────────────────────
    model_gwh: dict = {}
    if isinstance(dispatch, pd.DataFrame) and not dispatch.empty:
        for col in dispatch.columns:
            cat = _MODEL_AGG.get(col, col)
            gwh = float(dispatch[col].clip(lower=0).sum() / 1000)
            model_gwh[cat] = model_gwh.get(cat, 0.0) + gwh

    # ── Aggregate REE actual ───────────────────────────────────────────────────
    ree_gwh: dict = {}
    if isinstance(ree, pd.DataFrame) and not ree.empty:
        for col in ree.columns:
            cat = _agg_ree_col(col)
            gwh = float(ree[col].clip(lower=0).sum() / 1000)
            ree_gwh[cat] = ree_gwh.get(cat, 0.0) + gwh

    if not model_gwh and not ree_gwh:
        return go.Figure(layout=dict(height=320, paper_bgcolor=_WHITE,
                                     annotations=[dict(text="Load a network to view.",
                                     x=0.5, y=0.5, xref="paper", yref="paper",
                                     showarrow=False, font=dict(color=_MUTED))]))

    # ── Build ordered carrier list ─────────────────────────────────────────────
    all_cats = set(model_gwh) | set(ree_gwh)
    ordered  = [c for c in _AGG_ORDER if c in all_cats] + \
               sorted(c for c in all_cats if c not in _AGG_ORDER)

    m_vals = [model_gwh.get(c, 0.0) for c in ordered]
    r_vals = [ree_gwh.get(c,  0.0) for c in ordered]
    colors = [_AGG_COLORS.get(c, "#888") for c in ordered]

    fig = go.Figure()
    fig.add_bar(
        name="Model",
        x=ordered, y=m_vals,
        marker=dict(color=colors, opacity=0.92),
        hovertemplate="<b>%{x}</b> Model: %{y:.1f} GWh<extra></extra>",
    )
    fig.add_bar(
        name="REE Actual",
        x=ordered, y=r_vals,
        marker=dict(color=colors, opacity=0.45,
                    pattern=dict(shape="/", solidity=0.4)),
        hovertemplate="<b>%{x}</b> REE Actual: %{y:.1f} GWh<extra></extra>",
    )

    # ── Total demand dashed reference line ─────────────────────────────────────
    demand_gwh = None
    if isinstance(load, pd.Series) and not load.empty:
        demand_gwh = float(load.sum() / 1000)
        fig.add_hline(
            y=demand_gwh,
            line=dict(color="#C0392B", width=1.8, dash="dash"),
            annotation_text=f"Total demand: {demand_gwh:.0f} GWh",
            annotation_position="top right",
            annotation_font=dict(size=9, color="#C0392B"),
        )
        # Invisible scatter trace so the demand line appears in the legend
        fig.add_scatter(
            x=[None], y=[None], mode="lines",
            name=f"ES Demand ({demand_gwh:.0f} GWh)",
            line=dict(color="#C0392B", width=2, dash="dash"),
            showlegend=True,
        )

    n_hrs  = max(i1 - i0, 1)
    window = f"{n_hrs}h window"
    layout = dict(**_PLOT_BASE)
    layout["height"]   = 320
    layout["barmode"]  = "group"
    layout["bargap"]   = 0.22
    layout["bargroupgap"] = 0.08
    layout["title"]    = dict(
        text=f"Spain Total Dispatch — Model vs REE Actual  ·  {window}",
        font=dict(size=12, color=_PLOT_TEXT),
    )
    layout["yaxis"]    = dict(_PLOT_BASE["yaxis"],
                              title_text="GWh", rangemode="tozero")
    layout["xaxis"]    = dict(_PLOT_BASE["xaxis"], title_text="Technology")
    layout["legend"]   = dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                               borderwidth=1, font=dict(size=10, color=_PLOT_TEXT),
                               orientation="h", x=0, y=-0.22, xanchor="left")
    fig.update_layout(**layout)
    return fig


def make_fr_pt_weekly_dispatch(d: dict, i0: int, i1: int) -> go.Figure:
    """FR and PT weekly model dispatch (stacked bars) + model vs actual IC export to ES (lines).

    Actual data is net cross-border flow into ES (actual_fr_t / actual_pt_t).
    Model IC export is fr_net / pt_net.  Both converted to GWh/week.
    Helps diagnose whether FR/PT over-dispatch is spilling into Spain.
    """
    from plotly.subplots import make_subplots as _ms
    s = _slice(d, i0, i1)
    ts = list(s["timestamps"])
    if not ts:
        return go.Figure(layout=dict(height=420, paper_bgcolor=_WHITE))

    idx = pd.to_datetime(ts)

    def _weekly_dispatch_gwh(disp: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(disp, pd.DataFrame) or disp.empty:
            return pd.DataFrame()
        agg: dict = {}
        for col in disp.columns:
            cat = _MODEL_AGG.get(col, col)
            agg[cat] = agg.get(cat, 0) + disp[col].clip(lower=0)
        out = pd.DataFrame(agg, index=disp.index)
        out.index = idx
        return out.resample("W-MON").sum() / 1000  # MWh → GWh

    def _weekly_flow_gwh(series: pd.Series) -> pd.Series:
        if not isinstance(series, pd.Series) or series.empty:
            return pd.Series(dtype=float)
        s2 = series.copy()
        s2.index = idx
        return s2.resample("W-MON").sum() / 1000  # MW·h → GWh

    _PANEL = [
        ("FR", s.get("dispatch_fr", pd.DataFrame()),
                s.get("fr_net",      pd.Series(dtype=float)),
                s.get("actual_fr_t", pd.Series(dtype=float))),
        ("PT", s.get("dispatch_pt", pd.DataFrame()),
                s.get("pt_net",      pd.Series(dtype=float)),
                s.get("actual_pt_t", pd.Series(dtype=float))),
    ]

    fig = _ms(rows=2, cols=1, shared_xaxes=True,
              vertical_spacing=0.10,
              subplot_titles=["FR Weekly Dispatch + Export to ES (GWh/week)",
                              "PT Weekly Dispatch + Export to ES (GWh/week)"])

    for row, (ctry, disp_df, model_ic, actual_ic) in enumerate(_PANEL, start=1):
        weekly = _weekly_dispatch_gwh(disp_df)
        model_weekly  = _weekly_flow_gwh(model_ic)
        actual_weekly = _weekly_flow_gwh(actual_ic)

        # Stacked bars — model dispatch by aggregated carrier
        shown = [c for c in _AGG_ORDER if c in weekly.columns]
        shown += sorted(c for c in weekly.columns if c not in _AGG_ORDER)
        for carrier in shown:
            fig.add_bar(
                x=weekly.index, y=weekly[carrier].values,
                name=carrier, legendgroup=carrier,
                showlegend=(row == 1),
                marker_color=_AGG_COLORS.get(carrier, "#888"),
                hovertemplate=f"<b>{carrier}</b> %{{x|W%W}}: %{{y:.0f}} GWh<extra></extra>",
                row=row, col=1,
            )

        # Model IC export line (positive = export to ES)
        if not model_weekly.empty:
            fig.add_scatter(
                x=model_weekly.index, y=model_weekly.values,
                mode="lines+markers", name=f"{ctry} model export→ES",
                legendgroup=f"{ctry}_model_ic", showlegend=(row == 1),
                line=dict(color=_PLOT_TEXT, width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{ctry} model</b> %{{x|W%W}}: %{{y:.0f}} GWh<extra></extra>",
                row=row, col=1,
            )

        # Actual IC flow line (dashed)
        if not actual_weekly.empty and actual_weekly.abs().sum() > 0:
            fig.add_scatter(
                x=actual_weekly.index, y=actual_weekly.values,
                mode="lines+markers", name=f"{ctry} actual export→ES",
                legendgroup=f"{ctry}_actual_ic", showlegend=(row == 1),
                line=dict(color=_CORAL, width=1.5, dash="dash"),
                marker=dict(size=4),
                hovertemplate=f"<b>{ctry} actual</b> %{{x|W%W}}: %{{y:.0f}} GWh<extra></extra>",
                row=row, col=1,
            )

    n_wks = max(len(pd.to_datetime(ts).to_series().resample("W-MON").count()), 1)
    _base = {k: v for k, v in _PLOT_BASE.items()
             if k not in ("xaxis", "yaxis", "hovermode", "legend", "margin")}
    fig.update_layout(
        **_base,
        height=520, barmode="stack",
        margin=dict(l=58, r=60, t=44, b=38),
        hovermode="x unified",
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                    borderwidth=1, font=dict(size=9, color=_PLOT_TEXT),
                    orientation="h", x=0, y=-0.10, xanchor="left"),
        title=dict(text=f"FR / PT Weekly Dispatch + Export to ES  ·  {n_wks} weeks",
                   font=dict(size=12, color=_PLOT_TEXT)),
    )
    fig.update_yaxes(gridcolor=_GRID, gridwidth=0.5, griddash="dash",
                     title_text="GWh/week", title_font=dict(size=9, color=_MUTED))
    return fig


def make_ic_tech_figure(d: dict) -> go.Figure:
    """FR/PT generation mix split by whether they are net-exporting or net-importing.

    Three flow-direction groups per country (>500 MW export, balanced, >500 MW import).
    Shows mean hourly dispatch by carrier — proxy for what technology drives IC flows.
    """
    THRESH = 500.0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["FR ↔ ES  (model dispatch by flow direction)",
                        "PT ↔ ES  (model dispatch by flow direction)"],
        shared_yaxes=False,
    )

    for col_idx, (country, dispatch_key, net_key) in enumerate([
        ("FR", "dispatch_fr", "fr_net"),
        ("PT", "dispatch_pt", "pt_net"),
    ], start=1):
        dispatch = d.get(dispatch_key)
        net      = d.get(net_key)
        if dispatch is None or net is None:
            continue
        if not isinstance(dispatch, pd.DataFrame):
            dispatch = pd.DataFrame(dispatch)
        if not isinstance(net, pd.Series):
            net = pd.Series(net)
        if dispatch.empty or net.empty:
            continue

        masks = {
            f"{country}→ES": net > THRESH,
            "Balanced":      net.abs() <= THRESH,
            f"ES→{country}": net < -THRESH,
        }
        group_labels = list(masks.keys())
        counts       = [int(m.sum()) for m in masks.values()]
        x_labels     = [f"{lbl}<br><sub>n={cnt}h</sub>" for lbl, cnt in zip(group_labels, counts)]

        carriers_ordered = [c for c in _AGG_ORDER if c in dispatch.columns]
        carriers_ordered += sorted(c for c in dispatch.columns if c not in _AGG_ORDER)
        shown_first = col_idx == 1

        for carrier in carriers_ordered:
            vals = [
                float(dispatch[carrier][m].mean()) if m.any() else 0.0
                for m in masks.values()
            ]
            agg_name = _MODEL_AGG.get(carrier, carrier)
            color    = _AGG_COLORS.get(agg_name, COLORS.get(carrier, "#999"))
            fig.add_trace(go.Bar(
                x=x_labels,
                y=vals,
                name=agg_name,
                showlegend=shown_first,
                legendgroup=agg_name,
                marker_color=color,
                hovertemplate=f"<b>{carrier}</b>: %{{y:.0f}} MW<extra></extra>",
            ), row=1, col=col_idx)
            shown_first = False   # only show each agg_name once in legend

    layout = dict(**{k: v for k, v in _PLOT_BASE.items()
                     if k not in ("xaxis", "yaxis", "margin", "legend")})
    layout["barmode"] = "stack"
    layout["height"]  = 380
    layout["title"]   = dict(
        text="IC Export/Import Technology Composition — mean dispatch by flow direction",
        font=dict(size=11, color=_PLOT_TEXT),
    )
    layout["margin"] = dict(l=55, r=140, t=50, b=60)
    layout["legend"] = dict(
        bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
        borderwidth=1, font=dict(size=10),
        x=1.02, y=1, xanchor="left",
    )
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="Mean hourly dispatch (MW)",
                     gridcolor=_GRID, gridwidth=0.5)
    return fig


def make_pdc_figure(d: dict, i0: int, i1: int) -> go.Figure:
    """Price duration curve coloured by price-setter, import hours flagged."""
    s      = _slice(d, i0, i1)
    price  = s["price_es"]
    setter = s["setter_es"]
    fr_net = s["fr_net"]
    pt_net = s["pt_net"]
    omie   = s["omie"]

    if not isinstance(price, pd.Series) or price.empty:
        return go.Figure()

    p_vals  = price.tolist()
    st_vals = setter.tolist() if isinstance(setter, pd.Series) and len(setter) == len(p_vals) else ["—"] * len(p_vals)
    fr_v    = fr_net.tolist() if isinstance(fr_net, pd.Series) and len(fr_net) == len(p_vals) else [0.0] * len(p_vals)
    pt_v    = pt_net.tolist() if isinstance(pt_net, pd.Series) and len(pt_net) == len(p_vals) else [0.0] * len(p_vals)
    net_imp = [fr_v[i] + pt_v[i] for i in range(len(p_vals))]

    # Sort ascending by price
    order   = sorted(range(len(p_vals)), key=lambda i: p_vals[i])
    sp      = [p_vals[i] for i in order]
    ss      = [st_vals[i] for i in order]
    simp    = [net_imp[i] for i in order]
    pct_x   = [100.0 * k / max(1, len(sp) - 1) for k in range(len(sp))]

    fig = go.Figure()

    # Scatter coloured by setter carrier — one trace per carrier for legend
    seen = {}
    for k, carrier in enumerate(ss):
        seen.setdefault(carrier, {"x": [], "y": []})
        seen[carrier]["x"].append(pct_x[k])
        seen[carrier]["y"].append(sp[k])

    # CARRIER_ORDER determines legend order; unknown carriers appended at end
    legend_order = [c for c in CARRIER_ORDER if c in seen] + [c for c in seen if c not in CARRIER_ORDER]
    for carrier in legend_order:
        pts = seen[carrier]
        fig.add_trace(go.Scatter(
            x=pts["x"], y=pts["y"],
            mode="markers",
            name=carrier,
            marker=dict(color=COLORS.get(carrier, "#999"), size=4, opacity=0.85),
            hovertemplate=f"<b>{carrier}</b>: €%{{y:.1f}}/MWh<extra></extra>",
        ))

    # OMIE actual as a faint step line for comparison
    if omie is not None and len(omie) == len(p_vals):
        omie_sorted = sorted(omie.tolist())
        fig.add_trace(go.Scatter(
            x=pct_x, y=omie_sorted,
            mode="lines",
            name="OMIE ES actual",
            line=dict(color=_AMBER, width=1.2, dash="dot"),
            opacity=0.7,
            hovertemplate="<b>OMIE ES</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

    # FR actual (EPEX) PDC
    omie_fr = s.get("omie_fr")
    if omie_fr is not None and hasattr(omie_fr, "__len__") and len(omie_fr) > 0:
        fr_sorted = sorted(omie_fr.dropna().tolist())
        fr_pct    = [100.0 * k / max(1, len(fr_sorted) - 1) for k in range(len(fr_sorted))]
        fig.add_trace(go.Scatter(
            x=fr_pct, y=fr_sorted,
            mode="lines",
            name="EPEX FR actual",
            line=dict(color="#2E86AB", width=1.2, dash="dash"),
            opacity=0.75,
            hovertemplate="<b>EPEX FR</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

    # PT actual PDC
    omie_pt = s.get("omie_pt")
    if omie_pt is not None and hasattr(omie_pt, "__len__") and len(omie_pt) > 0:
        pt_sorted = sorted(omie_pt.dropna().tolist())
        pt_pct    = [100.0 * k / max(1, len(pt_sorted) - 1) for k in range(len(pt_sorted))]
        fig.add_trace(go.Scatter(
            x=pt_pct, y=pt_sorted,
            mode="lines",
            name="OMIE PT actual",
            line=dict(color="#27AE60", width=1.2, dash="longdash"),
            opacity=0.75,
            hovertemplate="<b>OMIE PT</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

    # Model ES/PT joint PDC — model PT sorted line + gap fill for MIBEL coupling view
    pt_price_raw = s.get("pt_price_t", pd.Series(dtype=float))
    CONG_THRESH  = 1.0   # €/MWh — gap below this is rounding noise
    n_cong       = 0
    pct_cong     = 0.0
    mean_gap     = 0.0
    cong_rent_meur = float("nan")

    if isinstance(pt_price_raw, pd.Series) and not pt_price_raw.empty and len(pt_price_raw) == len(p_vals):
        gap       = (price - pt_price_raw).fillna(0.0)
        cong_mask = gap.abs() > CONG_THRESH
        n_cong    = int(cong_mask.sum())
        pct_cong  = n_cong / max(len(gap), 1) * 100
        mean_gap  = float(gap[cong_mask].mean()) if n_cong > 0 else 0.0
        pt_net_s  = pt_net if isinstance(pt_net, pd.Series) and len(pt_net) == len(gap) \
                    else pd.Series(0.0, index=price.index)
        cong_rent_meur = float((gap.abs() * pt_net_s.abs() / 1e6).sum())

        fig.add_trace(go.Scatter(
            x=pct_x, y=sp,
            mode="lines", name="Model ES (line)",
            line=dict(color=_CORAL, width=1.5),
            hovertemplate="<b>Model ES</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

        pt_sorted_vals = sorted(pt_price_raw.dropna().tolist())
        pt_pct_x = [100.0 * k / max(1, len(pt_sorted_vals) - 1) for k in range(len(pt_sorted_vals))]
        fig.add_trace(go.Scatter(
            x=pt_pct_x, y=pt_sorted_vals,
            mode="lines", name="Model PT (line)",
            line=dict(color="#27AE60", width=1.5),
            fill="tonexty",
            fillcolor="rgba(39,174,96,0.12)",
            hovertemplate="<b>Model PT</b>: €%{y:.1f}/MWh<extra></extra>",
        ))

    if n_cong > 0 and not pd.isna(cong_rent_meur):
        ann_text = (
            f"Congested: {pct_cong:.1f}% of hours ({n_cong}h)<br>"
            f"Mean |gap|: {abs(mean_gap):.1f} €/MWh<br>"
            f"Est. cong. rent: €{cong_rent_meur:.2f}M"
        )
        fig.add_annotation(
            text=ann_text, align="left",
            x=0.02, y=0.03, xref="paper", yref="paper",
            showarrow=False,
            font=dict(size=9, color=_PLOT_TEXT),
            bgcolor="rgba(255,255,255,0.88)", bordercolor=_GRID, borderwidth=1,
        )

    # Flag hours with significant net imports (>500 MW)
    IMPORT_THRESH = 500.0
    imp_x = [pct_x[k] for k, v in enumerate(simp) if v > IMPORT_THRESH]
    imp_y = [sp[k]    for k, v in enumerate(simp) if v > IMPORT_THRESH]
    imp_t = [f"{simp[k]:.0f} MW" for k, v in enumerate(simp) if v > IMPORT_THRESH]
    if imp_x:
        fig.add_trace(go.Scatter(
            x=imp_x, y=imp_y,
            mode="markers",
            name=">500 MW import",
            marker=dict(symbol="diamond-open", size=9,
                        color="rgba(0,0,0,0)",
                        line=dict(color=_SLATE, width=1.5)),
            text=imp_t,
            hovertemplate="Net import %{text}<br>€%{y:.1f}/MWh<extra></extra>",
        ))

    layout = dict(**_PLOT_BASE)
    layout["height"] = 300
    layout["title"]  = dict(text="Price Duration Curve — ES & PT model + actuals  ◇ = large import hours",
                             font=dict(size=11, color=_PLOT_TEXT))
    layout["xaxis"]  = dict(_PLOT_BASE["xaxis"],
                             title_text="% of hours (low → high price)",
                             ticksuffix="%")
    layout["yaxis"]  = dict(_PLOT_BASE["yaxis"],
                             title_text="Price (€/MWh)")
    layout["margin"] = dict(l=58, r=130, t=32, b=45)
    layout["legend"] = dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                             borderwidth=1, font=dict(size=10),
                             orientation="v", x=1.02, y=1, xanchor="left")
    fig.update_layout(**layout)
    return fig


def make_capacity_figure(d: dict) -> go.Figure:
    """Horizontal bar chart of installed capacity (p_nom) by carrier, ES only."""
    cap = d.get("capacity", {})
    if not cap:
        return go.Figure()

    # Order: follow CARRIER_ORDER; append anything not in it at the end
    order = [c for c in CARRIER_ORDER if c in cap] + \
            [c for c in cap if c not in CARRIER_ORDER]
    # Reverse so largest is at top of horizontal chart
    order = [c for c in order if cap.get(c, 0) > 0]
    order_rev = list(reversed(order))

    labels = order_rev
    values = [cap[c] for c in order_rev]
    colors = [COLORS.get(c, "#BBBBBB") for c in order_rev]

    # Add MW label annotations
    text_vals = [f"{v/1000:.1f} GW" if v >= 1000 else f"{v:.0f} MW" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=text_vals,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b>: %{x:.0f} MW<extra></extra>",
    ))

    layout = dict(**_PLOT_BASE)
    layout["height"]  = max(300, 32 * len(labels) + 60)
    layout["title"]   = dict(text="Installed Capacity — Spain (model p_nom)",
                              font=dict(size=12, color=_PLOT_TEXT))
    layout["xaxis"]   = dict(_PLOT_BASE["xaxis"],
                              title_text="Installed capacity (MW)",
                              title_font=dict(size=10, color=_MUTED))
    layout["yaxis"]   = dict(gridcolor="rgba(0,0,0,0)", showgrid=False,
                              tickfont=dict(size=11, color=_PLOT_TEXT))
    layout["margin"]  = dict(l=130, r=80, t=40, b=40)
    layout["legend"]  = dict(visible=False)
    layout["hovermode"] = "y unified"
    # Extend x-axis slightly so text labels don't get clipped
    max_val = max(values) if values else 1
    layout["xaxis"]["range"] = [0, max_val * 1.22]
    fig.update_layout(**layout)
    return fig


def make_hour_detail(d: dict, ts_str: str) -> list:
    """Build the right-panel content for the hovered timestamp."""
    if not ts_str or not d or not d.get("timestamps"):
        return [html.P("Hover over the chart to see hour detail.",
                       style={"color": _MUTED, "fontSize": "12px", "marginTop": "8px"})]

    ts_list = d["timestamps"]
    idx = None
    for i, t in enumerate(ts_list):
        if ts_str[:16] in t:
            idx = i
            break
    if idx is None:
        return [html.P(f"No data for {ts_str[:16]}",
                       style={"color": _CORAL, "fontSize": "11px"})]

    dispatch   = d["dispatch_es"]
    price_s    = d["price_es"]
    setter_s   = d["setter_es"]
    fr_net     = d["fr_net"]
    pt_net     = d["pt_net"]
    bus_prices = d["bus_prices"]
    omie       = d["omie"]

    def _get(s, i):
        try:
            return s.iloc[i] if isinstance(s, (pd.Series, pd.DataFrame)) else None
        except Exception:
            return None

    price_val  = _get(price_s, idx)
    setter_val = str(_get(setter_s, idx) or "—")
    fr_val     = _get(fr_net, idx)
    pt_val     = _get(pt_net, idx)
    omie_val   = _get(omie, idx) if omie is not None else None

    setter_color = COLORS.get(setter_val, "#888888")
    elems = []

    elems.append(html.Div(ts_str[:16],
        style={"fontSize": "11px", "color": _MUTED, "fontFamily": "monospace",
               "marginBottom": "10px"}))

    elems.append(html.Div([
        html.Div("Price setter", style={"fontSize": "10px", "color": _MUTED, "marginBottom": "3px"}),
        html.Div([
            html.Span("■ ", style={"color": setter_color, "fontSize": "16px"}),
            html.Span(setter_val, style={"fontSize": "13px", "fontWeight": "bold",
                                         "color": _PLOT_TEXT}),
        ]),
    ], style={"marginBottom": "10px"}))

    if price_val is not None:
        omie_str = f"  OMIE: €{float(omie_val):.1f}" if omie_val is not None else ""
        elems.append(html.Div([
            html.Div("Clearing price", style={"fontSize": "10px", "color": _MUTED, "marginBottom": "3px"}),
            html.Span(f"€{float(price_val):.1f}/MWh",
                      style={"fontSize": "14px", "fontWeight": "bold", "color": _CORAL}),
            html.Span(omie_str, style={"fontSize": "11px", "color": _AMBER}),
        ], style={"marginBottom": "12px"}))

    if isinstance(dispatch, pd.DataFrame) and not dispatch.empty and len(dispatch) > idx:
        row = dispatch.iloc[idx]
        bar_data = [(c, float(v)) for c, v in row.items() if abs(float(v)) > 1.0]
        bar_data.sort(key=lambda x: abs(x[1]), reverse=True)
        if bar_data:
            elems.append(html.Div("Generation mix",
                style={"fontSize": "10px", "color": _MUTED, "marginBottom": "5px",
                       "fontWeight": "600"}))
            max_mw = max(abs(v) for _, v in bar_data)
            for carrier, mw in bar_data[:12]:
                pct   = min(100.0, abs(mw) / max_mw * 100)
                color = COLORS.get(carrier, "#BBBBBB")
                sign  = "▼ " if mw < 0 else ""
                elems.append(html.Div([
                    html.Div(carrier,
                        style={"width": "90px", "fontSize": "10px", "color": _PLOT_TEXT,
                               "display": "inline-block", "overflow": "hidden",
                               "textOverflow": "ellipsis", "whiteSpace": "nowrap",
                               "verticalAlign": "middle"}),
                    html.Div(style={
                        "display": "inline-block",
                        "width": f"{max(pct, 2):.0f}px",
                        "maxWidth": "80px",
                        "height": "7px",
                        "backgroundColor": color,
                        "marginLeft": "4px",
                        "verticalAlign": "middle",
                        "borderRadius": "2px",
                    }),
                    html.Span(f" {sign}{abs(mw):.0f}",
                        style={"fontSize": "10px", "color": _PLOT_TEXT, "marginLeft": "3px"}),
                ], style={"marginBottom": "3px", "display": "flex", "alignItems": "center"}))

    if isinstance(bus_prices, pd.DataFrame) and not bus_prices.empty and len(bus_prices) > idx:
        bp_row = bus_prices.iloc[idx].dropna()
        if not bp_row.empty:
            bp_min, bp_max = float(bp_row.min()), float(bp_row.max())
            spread = bp_max - bp_min
            elems.append(html.Div([
                html.Div("Nodal price spread",
                    style={"fontSize": "10px", "color": _MUTED,
                           "marginTop": "10px", "marginBottom": "4px", "fontWeight": "600"}),
                html.Div(f"Max: €{bp_max:.1f}  ({bp_row.idxmax()[:8]})",
                    style={"fontSize": "10px", "color": _PLOT_TEXT}),
                html.Div(f"Min: €{bp_min:.1f}  ({bp_row.idxmin()[:8]})",
                    style={"fontSize": "10px", "color": _PLOT_TEXT}),
                html.Div(f"Spread: €{spread:.1f}/MWh",
                    style={"fontSize": "11px", "fontWeight": "bold",
                           "color": _CORAL if spread > 15 else _TEAL,
                           "marginTop": "2px"}),
            ]))

    elems.append(html.Div("Interconnector flows",
        style={"fontSize": "10px", "color": _MUTED, "marginTop": "10px",
               "marginBottom": "4px", "fontWeight": "600"}))
    for label, val_mw in [("FR↔ES", fr_val), ("PT↔ES", pt_val)]:
        if val_mw is not None:
            mw   = float(val_mw)
            dirn = "import" if mw > 0 else "export"
            color = "#457B9D" if label == "FR↔ES" else _TEAL
            elems.append(html.Div(
                f"{label}: {mw:+.0f} MW ({dirn})",
                style={"fontSize": "10px", "color": color}))

    return elems


# ── Network map ───────────────────────────────────────────────────────────────

# Line loading colour bands: (lower, upper, hex, legend label)
_LINE_BINS = [
    (0.00, 0.30, "#4CAF50", "0–30%"),
    (0.30, 0.60, "#8BC34A", "30–60%"),
    (0.60, 0.80, "#FFC107", "60–80%"),
    (0.80, 0.95, "#FF5722", "80–95%"),
    (0.95, 1.01, "#B71C1C", ">95% congested"),
]

_PRICE_SCALE = [
    [0.00, "#2166AC"],
    [0.35, "#92C5DE"],
    [0.50, "#FFFFBF"],
    [0.65, "#FDAE61"],
    [0.85, _CORAL],
    [1.00, "#7B0000"],
]


def make_map_figure(d: dict, hour_idx: int) -> go.Figure:
    """
    Spain nodal price map rendered on Carto-Positron tile basemap.

    Buses:  circles coloured by marginal price (blue→yellow→red), sized by bus.
    Lines:  coloured by loading band (green → dark-red for congestion).
    Links:  thick coloured lines; blue = import to ES, coral = export.
    """
    map_meta      = d.get("map_meta", {})
    bus_prices    = d.get("bus_prices", pd.DataFrame())
    line_loadings = d.get("line_loadings", {})
    link_flows    = d.get("link_flows", {})
    bus_gen       = d.get("bus_gen", {})
    ts_list       = d.get("timestamps", [])

    fig = go.Figure()

    if not map_meta or not map_meta.get("buses"):
        fig.update_layout(
            annotations=[dict(
                text="Load a solved network to view the spatial map.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=13, color=_MUTED),
            )],
            height=520, paper_bgcolor=_WHITE,
        )
        return fig

    buses    = map_meta["buses"]
    tx_lines = map_meta.get("lines", {})
    ic_links = map_meta.get("links", {})

    n_ts     = len(ts_list)
    hour_idx = max(0, min(int(hour_idx), n_ts - 1)) if n_ts > 0 else 0
    ts_label = ts_list[hour_idx][:16] if ts_list else "—"

    # ── 1. Transmission lines coloured by loading band ─────────────────────────
    bin_lat = [[] for _ in _LINE_BINS]
    bin_lon = [[] for _ in _LINE_BINS]
    for lid, ld in tx_lines.items():
        ts_vals = line_loadings.get(lid, [])
        loading = ts_vals[hour_idx] if hour_idx < len(ts_vals) else 0.0
        for bi, (lo, hi, _col, _lbl) in enumerate(_LINE_BINS):
            if lo <= loading < hi:
                bin_lat[bi] += [ld["y0"], ld["y1"], None]
                bin_lon[bi] += [ld["x0"], ld["x1"], None]
                break

    for bi, (_, _, col, lbl) in enumerate(_LINE_BINS):
        if not bin_lat[bi]:
            continue
        fig.add_trace(go.Scattermapbox(
            lat=bin_lat[bi], lon=bin_lon[bi],
            mode="lines",
            name=f"Line {lbl}",
            line=dict(color=col, width=2.5 + bi * 0.5),
            hoverinfo="skip",
            legendrank=200 + bi,
            legendgroup="tx",
            legendgrouptitle_text="Transmission",
        ))

    # ── 2. Interconnector lines — colour by utilisation, width by |flow| ─────
    def _util_color(frac: float) -> str:
        if frac < 0.33:  return "#27AE60"   # green   — lightly loaded
        if frac < 0.60:  return "#F0B429"   # amber   — moderate
        if frac < 0.85:  return "#E67E22"   # orange  — high
        return "#E74C3C"                     # red     — near saturation

    _util_seen: set = set()   # track which colour bands appear (for legend)

    for lid, lk in ic_links.items():
        flow_ts  = link_flows.get(lid, [])
        flow     = float(flow_ts[hour_idx]) if hour_idx < len(flow_ts) else 0.0
        p_nom    = lk.get("p_nom", 1.0) or 1.0
        tech     = lk.get("tech", "DC")   # "AC" or "DC"
        util     = min(abs(flow) / p_nom, 1.05)
        color    = _util_color(util)
        width    = max(1.8, min(9.0, abs(flow) / 280 + 1.5))

        # Direction label (positive p0 = bus0→bus1)
        b0, b1   = lk.get("bus0", "?"), lk.get("bus1", "?")
        if flow >= 0:
            direc = f"{b0} → {b1}"
        else:
            direc = f"{b1} → {b0}"

        util_band = (
            "<33%" if util < 0.33 else
            "33–60%" if util < 0.60 else
            "60–85%" if util < 0.85 else
            ">85%"
        )
        _util_seen.add(util_band)

        htxt = (
            f"<b>{lid}</b>  [{tech}]<br>"
            f"Flow: {flow:+.0f} MW  ({direc})<br>"
            f"Capacity: {p_nom:.0f} MW<br>"
            f"Utilisation: {util*100:.1f}%"
        )

        fig.add_trace(go.Scattermapbox(
            lat=[lk["y0"], lk["y1"]],
            lon=[lk["x0"], lk["x1"]],
            mode="lines",
            line=dict(color=color, width=width),
            hovertemplate=htxt + "<extra></extra>",
            showlegend=False,
        ))

        # Midpoint label: utilisation % + flow arrow
        mid_lat = (lk["y0"] + lk["y1"]) / 2
        mid_lon = (lk["x0"] + lk["x1"]) / 2
        arrow   = "↗" if flow >= 0 else "↙"
        fig.add_trace(go.Scattermapbox(
            lat=[mid_lat], lon=[mid_lon],
            mode="markers+text",
            marker=dict(color=color, size=7, opacity=0.9),
            text=[f"{arrow}{abs(flow):.0f}MW ({util*100:.0f}%)"],
            textfont=dict(size=8, color=color),
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ))

    # Legend: one entry per utilisation band seen
    _ic_legend = [
        ("<33% loaded",  "#27AE60", 90),
        ("33–60% loaded","#F0B429", 91),
        ("60–85% loaded","#E67E22", 92),
        (">85% loaded",  "#E74C3C", 93),
    ]
    for lbl, col, rank in _ic_legend:
        fig.add_trace(go.Scattermapbox(
            lat=[None], lon=[None], mode="lines",
            name=f"IC {lbl}",
            line=dict(color=col, width=3),
            legendrank=rank, legendgroup="ic",
            legendgrouptitle_text="Interconnectors (utilisation)",
        ))

    # ── 3. Bus price scatter with colorbar ─────────────────────────────────────
    bus_ids  = sorted(buses.keys())
    bus_lats = [buses[b]["lat"] for b in bus_ids]
    bus_lons = [buses[b]["lon"] for b in bus_ids]

    if isinstance(bus_prices, pd.DataFrame) and not bus_prices.empty and hour_idx < len(bus_prices):
        price_row = bus_prices.iloc[hour_idx]
        bus_pvals = [float(price_row.get(b, float("nan"))) for b in bus_ids]
    else:
        bus_pvals = [float("nan")] * len(bus_ids)

    valid_p = [v for v in bus_pvals if v == v]
    p_min   = min(valid_p) if valid_p else 0.0
    p_max   = max(valid_p) if valid_p else 120.0
    if p_max <= p_min:
        p_max = p_min + 1.0

    # Hover text (per bus)
    hover_texts = []
    for b, v in zip(bus_ids, bus_pvals):
        htxt = [f"<b>{b}</b>", f"€{v:.1f}/MWh" if v == v else "no price"]
        gen_at_bus = bus_gen.get(b, {})
        carrier_mw = sorted(
            [(c, gen_ts[hour_idx] if hour_idx < len(gen_ts) else 0.0)
             for c, gen_ts in gen_at_bus.items()],
            key=lambda x: x[1], reverse=True,
        )
        if any(mw > 0.5 for _, mw in carrier_mw):
            htxt.append("──────────")
            for c, mw in carrier_mw:
                if mw > 0.5:
                    htxt.append(f"{c}: {mw:.0f} MW")
        hover_texts.append("<br>".join(htxt))

    # Ensure colorbar spans at least 20 €/MWh so ticks are distinct
    _actual_span = p_max - p_min
    if _actual_span < 20.0:
        _cb_mid = (p_min + p_max) / 2.0
        cb_min  = max(0.0, _cb_mid - 10.0)
        cb_max  = cb_min + 20.0
    else:
        cb_min, cb_max = p_min, p_max
    _cspan = cb_max - cb_min
    _tvals = [round(cb_min + _cspan * f, 1) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    _tlbls = [f"{tv:.0f} €" for tv in _tvals]

    fig.add_trace(go.Scattermapbox(
        lat=bus_lats, lon=bus_lons,
        mode="markers",
        name="Bus price (€/MWh)",
        showlegend=False,
        marker=dict(
            color=bus_pvals,
            colorscale=_PRICE_SCALE,
            cmin=cb_min, cmax=cb_max,
            size=16,
            opacity=0.92,
            sizemode="diameter",
            colorbar=dict(
                title=dict(text="€/MWh", font=dict(size=12, color=_PLOT_TEXT)),
                thickness=20,
                len=0.72,
                x=1.01,
                xpad=6,
                y=0.5,
                yanchor="middle",
                tickfont=dict(size=11, color=_PLOT_TEXT),
                tickvals=_tvals,
                ticktext=_tlbls,
                outlinewidth=0,
                bordercolor=_GRID,
                borderwidth=1,
                bgcolor="rgba(255,255,255,0.9)",
            ),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        customdata=[[b] for b in bus_ids],
    ))

    fig.update_layout(
        title=dict(
            text=f"Spain Nodal Prices & Grid Loading  ·  {ts_label}",
            font=dict(size=12, color=_PLOT_TEXT, family="Helvetica Neue, Helvetica, Arial, sans-serif"),
            x=0.01, pad=dict(t=6),
        ),
        height=580,
        paper_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        mapbox=dict(
            style="carto-positron",
            center=dict(lon=-3.7, lat=40.2),
            zoom=5.5,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.94)",
            bordercolor=_GRID, borderwidth=1,
            font=dict(size=10, color=_PLOT_TEXT),
            orientation="v",
            x=0.01, y=0.98,
            xanchor="left", yanchor="top",
            tracegroupgap=6,
            itemsizing="constant",
        ),
        margin=dict(l=0, r=86, t=44, b=0),
    )
    return fig


def _select_price(d: dict, method: str) -> pd.Series:
    """Return price series for the chosen construction method (lw or tw)."""
    if method == "tw":
        return d.get("price_tw_t", d.get("price_es", pd.Series(dtype=float)))
    return d.get("price_es", pd.Series(dtype=float))


def make_load_map(d: dict) -> go.Figure:
    """Annual mean load per node — bubble map sized and coloured by demand (MW)."""
    map_meta     = d.get("map_meta", {})
    bus_load_ann = d.get("bus_load_annual", {})

    fig = go.Figure()
    if not map_meta or not map_meta.get("buses"):
        fig.update_layout(
            annotations=[dict(text="Load a solved network to view the load map.",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=13, color=_MUTED))],
            height=520, paper_bgcolor=_WHITE,
        )
        return fig

    buses   = map_meta["buses"]
    bus_ids = [b for b in sorted(buses.keys()) if b in bus_load_ann]
    if not bus_ids:
        fig.update_layout(
            annotations=[dict(text="No load data available.", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False,
                              font=dict(size=13, color=_MUTED))],
            height=520, paper_bgcolor=_WHITE,
        )
        return fig

    bus_lats  = [buses[b]["lat"] for b in bus_ids]
    bus_lons  = [buses[b]["lon"] for b in bus_ids]
    load_mw   = [bus_load_ann[b] for b in bus_ids]
    max_load  = max(load_mw) or 1.0
    sizes     = [max(10, int(44 * (v / max_load) ** 0.5)) for v in load_mw]
    hover_txts = [f"<b>{b}</b><br>Mean load: {v / 1000:.2f} GW"
                  for b, v in zip(bus_ids, load_mw)]

    fig.add_trace(go.Scattermapbox(
        lat=bus_lats, lon=bus_lons,
        mode="markers",
        name="Annual mean load",
        marker=dict(
            color=load_mw,
            colorscale=[[0.0, "#D0E8FF"], [0.4, "#4A90D9"], [1.0, "#003A80"]],
            cmin=0, cmax=max_load,
            size=sizes, opacity=0.85, sizemode="diameter",
            colorbar=dict(
                title=dict(text="MW", font=dict(size=12, color=_PLOT_TEXT)),
                thickness=18, len=0.68, x=1.01, y=0.5, yanchor="middle",
                tickfont=dict(size=11, color=_PLOT_TEXT),
                outlinewidth=0, bgcolor="rgba(255,255,255,0.9)",
            ),
        ),
        text=hover_txts,
        hovertemplate="%{text}<extra></extra>",
    ))
    # Bus number labels
    fig.add_trace(go.Scattermapbox(
        lat=bus_lats, lon=bus_lons,
        mode="text",
        text=[b[-4:] if len(b) > 4 else b for b in bus_ids],
        textfont=dict(size=7, color="#003A80"),
        textposition="middle center",
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Spain Node Load Map  ·  Annual Mean Demand",
                   font=dict(size=12, color=_PLOT_TEXT,
                             family="Helvetica Neue, Helvetica, Arial, sans-serif"),
                   x=0.01, pad=dict(t=6)),
        height=520, paper_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        mapbox=dict(style="carto-positron", center=dict(lon=-3.7, lat=40.2), zoom=5.5),
        margin=dict(l=0, r=86, t=44, b=0),
    )
    return fig


def make_capacity_map(d: dict) -> go.Figure:
    """Installed capacity per node — bubble map sized and coloured by total MW."""
    map_meta = d.get("map_meta", {})
    bus_cap  = d.get("bus_cap", {})

    fig = go.Figure()
    if not map_meta or not map_meta.get("buses"):
        fig.update_layout(
            annotations=[dict(text="Load a solved network to view the capacity map.",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=13, color=_MUTED))],
            height=520, paper_bgcolor=_WHITE,
        )
        return fig

    buses     = map_meta["buses"]
    total_cap = {b: sum(carriers.values())
                 for b, carriers in bus_cap.items()
                 if carriers and b in buses}
    bus_ids   = [b for b in sorted(buses.keys())
                 if b in total_cap and total_cap[b] > 0]
    if not bus_ids:
        fig.update_layout(
            annotations=[dict(text="No capacity data available.", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False,
                              font=dict(size=13, color=_MUTED))],
            height=520, paper_bgcolor=_WHITE,
        )
        return fig

    bus_lats = [buses[b]["lat"] for b in bus_ids]
    bus_lons = [buses[b]["lon"] for b in bus_ids]
    cap_mw   = [total_cap[b] for b in bus_ids]
    max_cap  = max(cap_mw) or 1.0
    sizes    = [max(10, int(44 * (v / max_cap) ** 0.5)) for v in cap_mw]

    hover_txts = []
    for b, tot in zip(bus_ids, cap_mw):
        carriers  = sorted(bus_cap[b].items(), key=lambda x: -x[1])
        breakdown = "<br>".join(f"  {c}: {mw:.0f} MW"
                                for c, mw in carriers if mw > 0)
        hover_txts.append(f"<b>{b}</b><br>Total: {tot / 1000:.2f} GW<br>{breakdown}")

    fig.add_trace(go.Scattermapbox(
        lat=bus_lats, lon=bus_lons,
        mode="markers",
        name="Installed capacity",
        marker=dict(
            color=cap_mw,
            colorscale=[[0.0, "#D5F5E3"], [0.4, "#27AE60"], [1.0, "#0B4B20"]],
            cmin=0, cmax=max_cap,
            size=sizes, opacity=0.85, sizemode="diameter",
            colorbar=dict(
                title=dict(text="MW", font=dict(size=12, color=_PLOT_TEXT)),
                thickness=18, len=0.68, x=1.01, y=0.5, yanchor="middle",
                tickfont=dict(size=11, color=_PLOT_TEXT),
                outlinewidth=0, bgcolor="rgba(255,255,255,0.9)",
            ),
        ),
        text=hover_txts,
        hovertemplate="%{text}<extra></extra>",
    ))
    # Bus number labels
    fig.add_trace(go.Scattermapbox(
        lat=bus_lats, lon=bus_lons,
        mode="text",
        text=[b[-4:] if len(b) > 4 else b for b in bus_ids],
        textfont=dict(size=7, color="#0B4B20"),
        textposition="middle center",
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        title=dict(text="Spain Node Capacity Map  ·  Total Installed",
                   font=dict(size=12, color=_PLOT_TEXT,
                             family="Helvetica Neue, Helvetica, Arial, sans-serif"),
                   x=0.01, pad=dict(t=6)),
        height=520, paper_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        mapbox=dict(style="carto-positron", center=dict(lon=-3.7, lat=40.2), zoom=5.5),
        margin=dict(l=0, r=86, t=44, b=0),
    )
    return fig


# ── FR overnight analysis figures ─────────────────────────────────────────────

def _make_overnight_profile(d: dict) -> go.Figure:
    """4-panel subplot: hour-of-day averages for price, FR imports, CCGT, and FR tech."""
    from plotly.subplots import make_subplots

    price    = d.get("price_es",    pd.Series(dtype=float))
    omie     = d.get("omie")
    omie_fr  = d.get("omie_fr")
    omie_pt  = d.get("omie_pt")
    fr_model = d.get("fr_net",      pd.Series(dtype=float))
    fr_act   = d.get("actual_fr_t", pd.Series(dtype=float))
    fr_nuc   = d.get("fr_nuclear_t", pd.Series(dtype=float))
    fr_hyd   = d.get("fr_hydro_t",   pd.Series(dtype=float))
    disp_es  = d.get("dispatch_es",  pd.DataFrame())
    ree      = d.get("ree_actual",   pd.DataFrame())
    ts       = d.get("timestamps",   [])

    if not ts:
        return go.Figure()

    idx  = pd.to_datetime(ts)
    hour = idx.hour
    hours = list(range(24))
    ticks = list(range(0, 24, 3))
    tlbls = [f"{h:02d}:00" for h in ticks]

    # Aggregate by hour of day
    mp_h   = price.groupby(hour).agg(["mean", "std"]) if not price.empty else pd.DataFrame()
    omie_h = omie.groupby(hour).mean() if omie is not None else pd.Series(dtype=float)
    omie_fr_h = (omie_fr.groupby(hour).mean()
                 if isinstance(omie_fr, pd.Series) and not omie_fr.empty
                 else pd.Series(dtype=float))
    omie_pt_h = (omie_pt.groupby(hour).mean()
                 if isinstance(omie_pt, pd.Series) and not omie_pt.empty
                 else pd.Series(dtype=float))

    fr_mod_h = fr_model.groupby(hour).mean() if not fr_model.empty else pd.Series(dtype=float)
    fr_act_h = (fr_act.groupby(hour).mean()
                if not fr_act.empty and fr_act.abs().sum() > 10 else pd.Series(dtype=float))

    ccgt_cols = [c for c in (disp_es.columns if isinstance(disp_es, pd.DataFrame) else [])
                 if "CCGT" in c and "must_run" not in c.lower()]
    ccgt_mod_h = (disp_es[ccgt_cols].clip(lower=0).sum(axis=1).groupby(hour).mean()
                  if ccgt_cols else pd.Series(dtype=float))
    ccgt_ree_h = (ree["CCGT"].groupby(hour).mean()
                  if isinstance(ree, pd.DataFrame) and "CCGT" in ree.columns else pd.Series(dtype=float))

    fr_nuc_h = fr_nuc.groupby(hour).mean() if not fr_nuc.empty else pd.Series(dtype=float)
    fr_hyd_h = fr_hyd.groupby(hour).mean() if not fr_hyd.empty else pd.Series(dtype=float)

    # Compute model FR CCGT dispatch by hour (carriers from dispatch_fr)
    disp_fr = d.get("dispatch_fr", pd.DataFrame())
    fr_ccgt_h = pd.Series(dtype=float)
    if isinstance(disp_fr, pd.DataFrame) and "CCGT" in disp_fr.columns:
        fr_ccgt_h = disp_fr["CCGT"].clip(lower=0).groupby(hour).mean()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=[
            "① Price by hour of day (€/MWh)",
            "② FR→ES net import by hour (MW)  ·  positive = Spain imports",
            "③ ES flex CCGT dispatch by hour (MW)",
            "④ Model FR generation mix feeding exports (MW)",
        ],
    )

    # ① Price
    if not mp_h.empty:
        lo = (mp_h["mean"] - mp_h["std"]).reindex(hours, fill_value=float("nan"))
        hi = (mp_h["mean"] + mp_h["std"]).reindex(hours, fill_value=float("nan"))
        fig.add_trace(go.Scatter(x=hours, y=hi.tolist(), mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=lo.tolist(), mode="lines",
                                 fill="tonexty", fillcolor="rgba(70,130,180,0.12)",
                                 line=dict(width=0), name="Model ±1σ"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mp_h["mean"].reindex(hours).tolist(),
                                 mode="lines+markers", name="Model price",
                                 line=dict(color="#457B9D", width=2.5),
                                 marker=dict(size=5)), row=1, col=1)
    if not omie_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=omie_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="OMIE ES actual",
                                 line=dict(color=_CORAL, width=2.5, dash="dot"),
                                 marker=dict(size=5)), row=1, col=1)
    if not omie_fr_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=omie_fr_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="EPEX FR actual",
                                 line=dict(color="#2E86AB", width=2.0, dash="dash"),
                                 marker=dict(size=4, symbol="diamond")), row=1, col=1)
    if not omie_pt_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=omie_pt_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="OMIE PT actual",
                                 line=dict(color="#27AE60", width=2.0, dash="longdash"),
                                 marker=dict(size=4, symbol="square")), row=1, col=1)

    # ② FR imports
    if not fr_mod_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=fr_mod_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="Model FR import",
                                 line=dict(color="#457B9D", width=2.5),
                                 marker=dict(size=5)), row=2, col=1)
    if not fr_act_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=fr_act_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="Actual FR import (ENTSOE)",
                                 line=dict(color=_AMBER, width=2.5, dash="dot"),
                                 marker=dict(size=5)), row=2, col=1)
    # Zero line on import panel
    fig.add_hline(y=0, line_width=0.8, line_dash="dot", line_color=_MUTED, row=2, col=1)

    # ③ CCGT
    if not ccgt_mod_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=ccgt_mod_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="Model ES CCGT",
                                 line=dict(color=_CORAL, width=2.5),
                                 marker=dict(size=5)), row=3, col=1)
    if not ccgt_ree_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=ccgt_ree_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="Actual ES CCGT (REE)",
                                 line=dict(color=_AMBER, width=2.5, dash="dot"),
                                 marker=dict(size=5)), row=3, col=1)

    # ④ FR tech mix
    if not fr_nuc_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=fr_nuc_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="FR nuclear (model)",
                                 line=dict(color="#6C5B7B", width=2.5),
                                 marker=dict(size=5)), row=4, col=1)
    if not fr_hyd_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=fr_hyd_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="FR hydro (model)",
                                 line=dict(color="#355C7D", width=2.5),
                                 marker=dict(size=5)), row=4, col=1)
    if not fr_ccgt_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=fr_ccgt_h.reindex(hours).tolist(),
                                 mode="lines+markers", name="FR CCGT (model)",
                                 line=dict(color=_CORAL, width=2, dash="dash"),
                                 marker=dict(size=4)), row=4, col=1)
    if not fr_mod_h.empty:
        fig.add_trace(go.Scatter(x=hours, y=fr_mod_h.reindex(hours).tolist(),
                                 mode="lines", name="FR→ES export (model)",
                                 line=dict(color="#F8B400", width=1.5, dash="dot"),
                                 showlegend=True), row=4, col=1)

    # Overnight shading on all panels
    for row in [1, 2, 3, 4]:
        fig.add_vrect(x0=-0.5, x1=5.5, fillcolor="rgba(30,37,43,0.055)",
                      layer="below", line_width=0, row=row, col=1)

    for row in range(1, 5):
        fig.update_xaxes(tickvals=ticks, ticktext=tlbls, row=row, col=1)
    fig.update_xaxes(title_text="Hour of day", row=4, col=1)

    fig.update_layout(
        height=820,
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=10), orientation="h", x=0, y=-0.06, xanchor="left"),
        margin=dict(l=60, r=30, t=55, b=80),
        **{f"xaxis{'' if i == 1 else i}_gridcolor": _GRID for i in range(1, 5)},
        **{f"yaxis{'' if i == 1 else i}_gridcolor": _GRID for i in range(1, 5)},
    )
    return fig


def _make_fr_price_drivers(d: dict) -> go.Figure:
    """3-panel deep-dive: what drives French electricity prices and ES exports.

    Panel ①  ES price vs FR model price — how tightly coupled the markets are.
    Panel ②  FR generation surplus (nuclear+hydro+VRE − FR load) vs FR→ES flow —
             the surplus is the economic pressure pushing exports.
    Panel ③  FR price vs ES price scatter by hour-of-day — identifies when
             the two markets decouple (congestion, nuclear baseload patterns).
    """
    import numpy as np
    from plotly.subplots import make_subplots

    price_es  = d.get("price_es",    pd.Series(dtype=float))
    fr_price  = d.get("fr_price_t",  pd.Series(dtype=float))
    fr_net    = d.get("fr_net",      pd.Series(dtype=float))
    fr_load   = d.get("fr_load_t",   pd.Series(dtype=float))
    fr_nuc    = d.get("fr_nuclear_t", pd.Series(dtype=float))
    fr_hyd    = d.get("fr_hydro_t",   pd.Series(dtype=float))
    fr_wind   = d.get("fr_wind_t",    pd.Series(dtype=float))
    fr_solar  = d.get("fr_solar_t",   pd.Series(dtype=float))
    fr_surplus= d.get("fr_surplus_t", pd.Series(dtype=float))
    omie      = d.get("omie")
    omie_fr   = d.get("omie_fr")   # actual FR day-ahead market price
    ts        = d.get("timestamps",   [])

    if not ts or price_es.empty:
        return go.Figure(layout=dict(height=600, paper_bgcolor=_WHITE,
            annotations=[dict(text="Load a solved network to view FR price drivers.",
                x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                font=dict(size=12, color=_MUTED))]))

    idx  = pd.to_datetime(ts)
    hour = idx.hour.values

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=False, vertical_spacing=0.09,
        subplot_titles=[
            "① ES price vs FR price (model shadow prices, €/MWh)",
            "② FR generation surplus vs FR→ES net flow (MW)  ·  surplus = nuc+hyd+VRE − FR load",
            "③ FR price vs ES price scatter — market coupling by hour of day",
        ],
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}],
               [{"secondary_y": False}]],
    )

    # ── ① Price comparison timeseries ────────────────────────────────────────
    ts_list = list(ts)
    fig.add_trace(go.Scatter(x=ts_list, y=price_es.tolist(),
                             name="ES price (model)", mode="lines",
                             line=dict(color="#457B9D", width=1.8)),
                  row=1, col=1)
    if not fr_price.empty and fr_price.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=ts_list, y=fr_price.tolist(),
                                 name="FR price (model shadow)", mode="lines",
                                 line=dict(color="#6C5B7B", width=1.8, dash="dot")),
                      row=1, col=1)
    if omie is not None:
        fig.add_trace(go.Scatter(x=ts_list, y=omie.tolist(),
                                 name="ES actual (OMIE)", mode="lines",
                                 line=dict(color=_CORAL, width=1.2, dash="dash")),
                      row=1, col=1)
    if isinstance(omie_fr, pd.Series) and not omie_fr.empty and omie_fr.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=ts_list, y=omie_fr.tolist(),
                                 name="FR actual (EPEX)", mode="lines",
                                 line=dict(color="#A855F7", width=1.4, dash="dash")),
                      row=1, col=1)
        # Update panel ③ scatter to also show actual FR price vs ES actual
        if omie is not None and omie_fr.notna().sum() > 5:
            fr_act_v = omie_fr.values
            es_act_v = omie.values
            valid2   = np.isfinite(fr_act_v) & np.isfinite(es_act_v)
            if valid2.sum() > 5:
                r_act = float(np.corrcoef(fr_act_v[valid2], es_act_v[valid2])[0, 1])
                fig.add_trace(go.Scatter(
                    x=fr_act_v[valid2].tolist(), y=es_act_v[valid2].tolist(),
                    mode="markers",
                    marker=dict(color="#A855F7", size=3, opacity=0.35, symbol="x"),
                    name=f"Actual FR vs ES  (r={r_act:.2f})",
                    hovertemplate="FR actual: €%{x:.1f}<br>ES actual: €%{y:.1f}<extra></extra>",
                ), row=3, col=1)

    # ── ② Surplus vs flow ────────────────────────────────────────────────────
    if not fr_surplus.empty:
        # Shade surplus area
        surplus_vals = fr_surplus.tolist()
        pos_s = [max(0.0, v) for v in surplus_vals]
        neg_s = [min(0.0, v) for v in surplus_vals]
        fig.add_trace(go.Scatter(x=ts_list, y=pos_s, name="FR surplus (gen>load)",
                                 mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor="rgba(70,130,180,0.20)",
                                 showlegend=True, hoverinfo="skip"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=ts_list, y=neg_s, name="FR deficit (load>gen)",
                                 mode="lines", line=dict(width=0),
                                 fill="tozeroy", fillcolor="rgba(232,93,93,0.20)",
                                 showlegend=True, hoverinfo="skip"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=ts_list, y=surplus_vals,
                                 name="FR surplus (net)", mode="lines",
                                 line=dict(color="#457B9D", width=2.0),
                                 hovertemplate="FR surplus: %{y:.0f} MW<extra></extra>"),
                      row=2, col=1)

    # Nuclear, hydro, wind, solar breakdown as reference lines
    for label, series, color in [
        ("FR nuclear", fr_nuc,   "#6C5B7B"),
        ("FR hydro",   fr_hyd,   "#355C7D"),
        ("FR wind",    fr_wind,  "#52B788"),
        ("FR solar",   fr_solar, _AMBER),
        ("FR load",    fr_load,  _SLATE),
    ]:
        if not series.empty and series.sum() > 0:
            lw   = 1.5 if label != "FR load" else 2.0
            dash = "solid" if label != "FR load" else "dash"
            fig.add_trace(go.Scatter(x=ts_list, y=series.tolist(),
                                     name=label, mode="lines",
                                     line=dict(color=color, width=lw, dash=dash),
                                     visible="legendonly"),
                          row=2, col=1)

    # FR→ES net flow on secondary y of panel ②
    if not fr_net.empty:
        fig.add_trace(go.Scatter(x=ts_list, y=fr_net.tolist(),
                                 name="FR→ES net flow (MW)",
                                 mode="lines",
                                 line=dict(color=_CORAL, width=1.5),
                                 yaxis="y4",
                                 hovertemplate="FR→ES: %{y:.0f} MW<extra></extra>"),
                      row=2, col=1)

    # ── ③ FR vs ES price scatter ──────────────────────────────────────────────
    if not fr_price.empty and fr_price.notna().sum() > 5:
        es_v = price_es.values
        fr_v = fr_price.values
        valid = np.isfinite(es_v) & np.isfinite(fr_v)
        es_v, fr_v, hr_v = es_v[valid], fr_v[valid], hour[valid]

        hour_colors = [
            "#1D3557" if h < 6 else "#AEC6CF" if h < 12
            else "#FFDAB9" if h < 18 else "#C8A4A5"
            for h in hr_v
        ]
        fig.add_trace(go.Scatter(x=fr_v.tolist(), y=es_v.tolist(),
                                 mode="markers",
                                 marker=dict(color=hour_colors, size=3.5, opacity=0.6),
                                 showlegend=False,
                                 hovertemplate="FR: €%{x:.1f}<br>ES: €%{y:.1f}<extra></extra>"),
                      row=3, col=1)
        # 45° perfect-coupling line
        p_lo = min(fr_v.min(), es_v.min())
        p_hi = max(fr_v.max(), es_v.max())
        fig.add_trace(go.Scatter(x=[p_lo, p_hi], y=[p_lo, p_hi],
                                 mode="lines", name="Perfect coupling (45°)",
                                 line=dict(color=_MUTED, width=1, dash="dot"),
                                 showlegend=True),
                      row=3, col=1)
        if len(es_v) > 5:
            r = float(np.corrcoef(fr_v, es_v)[0, 1])
            fig.add_annotation(
                x=0.02, y=0.96, xref="x3 domain", yref="y3 domain",
                text=f"Pearson r = {r:.3f}", showarrow=False,
                font=dict(size=11, color=_PLOT_TEXT),
                bgcolor="rgba(255,255,255,0.85)", bordercolor=_GRID,
            )

        # Hour-of-day colour legend
        for label, color in [("00–05 overnight", "#1D3557"), ("06–11 morning", "#AEC6CF"),
                              ("12–17 afternoon", "#FFDAB9"), ("18–23 evening", "#C8A4A5")]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(color=color, size=8),
                                     name=label, showlegend=True),
                          row=3, col=1)

    # Axis labels
    fig.update_yaxes(title_text="€/MWh", row=1, col=1, gridcolor=_GRID)
    fig.update_yaxes(title_text="MW (surplus / deficit)", row=2, col=1, gridcolor=_GRID)
    fig.update_yaxes(title_text="FR→ES flow (MW)", row=2, col=1,
                     secondary_y=True, showgrid=False, zeroline=True,
                     zerolinecolor=_GRID, zerolinewidth=1)
    fig.update_xaxes(title_text="FR price (€/MWh)", row=3, col=1)
    fig.update_yaxes(title_text="ES price (€/MWh)", row=3, col=1, gridcolor=_GRID)
    for i in range(1, 4):
        fig.update_xaxes(gridcolor=_GRID, row=i, col=1)

    fig.update_layout(
        height=860,
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=10), orientation="h", x=0, y=-0.04, xanchor="left"),
        margin=dict(l=65, r=75, t=55, b=80),
        hovermode="x unified",
    )
    return fig


def _make_fr_tech_scatter(d: dict) -> go.Figure:
    """Scatter showing what FR technology drives exports and its effect on ES price.

    X: actual FR→ES export (MW, positive = Spain imports)
    Y: model FR nuclear dispatch (MW) — the dominant cheap driver
    Marker colour: price error (model − OMIE, €/MWh)
    Helps answer: when France exports a lot, is the model reflecting enough FR nuclear,
    and does more FR nuclear export push ES prices down?
    """
    import numpy as np

    price    = d.get("price_es",     pd.Series(dtype=float))
    omie     = d.get("omie")
    fr_act   = d.get("actual_fr_t",  pd.Series(dtype=float))
    fr_model = d.get("fr_net",       pd.Series(dtype=float))
    fr_nuc   = d.get("fr_nuclear_t", pd.Series(dtype=float))
    ts       = d.get("timestamps",   [])

    if not ts or fr_nuc.empty:
        return go.Figure(layout=dict(height=360, paper_bgcolor=_WHITE,
            annotations=[dict(text="FR nuclear data unavailable",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=12, color=_MUTED))]))

    idx   = pd.to_datetime(ts)
    hour  = idx.hour.values
    err   = (price - omie).values if omie is not None else np.zeros(len(idx))
    has_actual = fr_act.abs().sum() > 10.0

    x_vals = fr_act.values if has_actual else fr_model.values
    y_vals = fr_nuc.values
    valid  = np.isfinite(x_vals) & np.isfinite(y_vals) & np.isfinite(err)
    x_v, y_v, e_v, h_v = x_vals[valid], y_vals[valid], err[valid], hour[valid]

    # Compute implied price impact: regression of FR nuclear on price error
    if len(x_v) > 10:
        # Partial correlation: does more FR nuclear export → lower price error?
        fr_exp_mask = x_v > 200  # hours when France is actually exporting
        if fr_exp_mask.sum() > 5:
            r_exp = float(np.corrcoef(y_v[fr_exp_mask], e_v[fr_exp_mask])[0, 1])
        else:
            r_exp = float("nan")
    else:
        r_exp = float("nan")

    clim = float(min(np.nanpercentile(np.abs(e_v), 95), 50))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_v.tolist(), y=y_v.tolist(),
        mode="markers",
        marker=dict(
            color=e_v.tolist(),
            colorscale=[[0.0, "#2166AC"], [0.5, "#F7F7F7"], [1.0, "#B2182B"]],
            cmin=-clim, cmax=clim,
            cmid=0, size=4, opacity=0.6,
            colorbar=dict(
                title=dict(text="Price error<br>(€/MWh)", font=dict(size=10)),
                thickness=12, len=0.7, x=1.01,
                tickfont=dict(size=9), tickformat=".0f", nticks=6,
            ),
        ),
        customdata=list(zip(e_v.tolist(), h_v.tolist())),
        hovertemplate=(
            "FR export: %{x:.0f} MW<br>"
            "FR nuclear: %{y:.0f} MW<br>"
            "Price error: %{customdata[0]:.1f} €/MWh<br>"
            "Hour: %{customdata[1]:02d}:00<extra></extra>"
        ),
        showlegend=False,
    ))

    # Regression line for export hours only
    if has_actual and len(x_v) > 10:
        x_line = np.linspace(x_v.min(), x_v.max(), 80)
        m, b = np.polyfit(x_v, y_v, 1)
        fig.add_trace(go.Scatter(
            x=x_line.tolist(), y=(m * x_line + b).tolist(),
            mode="lines", name="OLS fit",
            line=dict(color="#374151", width=1.2, dash="dash"),
        ))

    r_str = f"r = {r_exp:.3f}" if np.isfinite(r_exp) else "r = n/a"
    x_label = "Actual FR→ES export (MW)" if has_actual else "Model FR→ES export (MW)"
    fig.update_layout(
        title=dict(
            text=(f"FR nuclear dispatch vs FR export size  ·  "
                  f"colour = price error  ·  FR nuclear–price error corr {r_str}"),
            font=dict(size=12, color=_PLOT_TEXT), x=0.01,
        ),
        height=400,
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        xaxis=dict(title=x_label, gridcolor=_GRID, zeroline=True,
                   zerolinecolor=_GRID, zerolinewidth=1.2),
        yaxis=dict(title="Model FR nuclear dispatch (MW)", gridcolor=_GRID),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                    font=dict(size=10), x=0.01, y=0.99),
        margin=dict(l=70, r=80, t=50, b=50),
    )
    return fig


def _make_price_error_heatmap(d: dict) -> go.Figure:
    """Calendar heatmap: model−OMIE price error by hour of day × date."""
    price = d.get("price_es", pd.Series(dtype=float))
    omie  = d.get("omie")
    ts    = d.get("timestamps", [])

    if not ts or omie is None or price.empty:
        return go.Figure(layout=dict(
            annotations=[dict(text="Load OMIE data to view heatmap",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=320, paper_bgcolor=_WHITE))

    idx   = pd.to_datetime(ts)
    error = (price - omie).values

    df = pd.DataFrame({
        "date": idx.date,
        "hour": idx.hour,
        "error": error,
    })
    pivot = df.pivot_table(index="hour", columns="date", values="error", aggfunc="mean")
    # rows = hours (0–23, y-axis bottom=0), cols = dates (x-axis)
    dates_str = [str(c) for c in pivot.columns]

    abs_max = float(max(abs(pivot.values[pivot.notna().values].min()),
                        abs(pivot.values[pivot.notna().values].max()), 1))
    clim = min(abs_max, 60)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=dates_str,
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale=[
            [0.0, "#2166AC"],
            [0.5, "#F7F7F7"],
            [1.0, "#B2182B"],
        ],
        zmid=0, zmin=-clim, zmax=clim,
        colorbar=dict(
            title=dict(text="Error (€/MWh)", font=dict(size=10, color=_PLOT_TEXT)),
            thickness=12, len=0.8, x=1.01,
            tickfont=dict(size=9, color=_PLOT_TEXT),
            tickformat=".0f", nticks=7,
        ),
        hovertemplate="Date: %{x}<br>Hour: %{y}<br>Error: %{z:.1f} €/MWh<extra></extra>",
    ))

    # Overnight band annotation
    fig.add_hrect(y0="-0.5", y1="05:00", fillcolor="rgba(30,37,43,0.08)",
                  layer="below", line_width=0)

    fig.update_layout(
        title=dict(text="Price Error Heatmap  (model − OMIE, €/MWh)  ·  grey band = 00:00–05:00",
                   font=dict(size=12, color=_PLOT_TEXT), x=0.01),
        height=420,
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=10),
        xaxis=dict(title="Date", tickangle=-45, tickfont=dict(size=9),
                   gridcolor=_GRID),
        yaxis=dict(title="Hour", tickfont=dict(size=9), autorange="reversed",
                   gridcolor=_GRID),
        margin=dict(l=58, r=72, t=44, b=60),
    )
    return fig


def _make_import_scatter(d: dict) -> go.Figure:
    """Scatter: price error (model−OMIE) vs FR import deviation (actual−model).

    Each point is one hour. Coloured by hour-of-day (overnight in dark blue).
    Includes a regression line and Pearson r annotation.
    """
    import numpy as np

    price    = d.get("price_es",    pd.Series(dtype=float))
    omie     = d.get("omie")
    fr_model = d.get("fr_net",      pd.Series(dtype=float))
    fr_act   = d.get("actual_fr_t", pd.Series(dtype=float))
    ts       = d.get("timestamps",  [])

    if not ts or omie is None or price.empty or fr_model.empty:
        return go.Figure(layout=dict(
            annotations=[dict(text="Insufficient data for scatter",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=380, paper_bgcolor=_WHITE))

    idx        = pd.to_datetime(ts)
    price_err  = (price - omie).values          # model − OMIE (positive = model too high)
    has_actual = fr_act.abs().sum() > 10.0
    if has_actual:
        import_err = (fr_act - fr_model).values  # actual − model (positive = model underestimates)
    else:
        # Fall back to model FR imports vs mean (illustrative)
        import_err = (fr_model - fr_model.mean()).values
    hour = idx.hour.values

    # Remove NaN rows
    valid = np.isfinite(price_err) & np.isfinite(import_err)
    pe, ie, hr = price_err[valid], import_err[valid], hour[valid]

    r = float(np.corrcoef(ie, pe)[0, 1]) if len(pe) > 2 else float("nan")

    # Colour by hour: overnight (0-5) dark blue, other hours lighter
    colors = ["#1D3557" if h < 6 else "#AEC6CF" if h < 12 else "#FFDAB9" if h < 18 else "#C8A4A5"
              for h in hr]

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=ie, y=pe,
        mode="markers",
        marker=dict(color=colors, size=4, opacity=0.65,
                    line=dict(width=0)),
        hovertemplate="Import dev: %{x:.0f} MW<br>Price error: %{y:.1f} €/MWh<extra></extra>",
        showlegend=False,
    ))

    # OLS regression line
    if len(pe) > 5:
        m, b = np.polyfit(ie, pe, 1)
        x_line = np.array([ie.min(), ie.max()])
        fig.add_trace(go.Scatter(
            x=x_line.tolist(), y=(m * x_line + b).tolist(),
            mode="lines", name=f"OLS fit  (r = {r:.3f})",
            line=dict(color=_CORAL, width=1.5, dash="dash"),
        ))

    # Colour legend by hour group
    for label, color in [("00–05 (overnight)", "#1D3557"),
                          ("06–11 (morning)",  "#AEC6CF"),
                          ("12–17 (afternoon)","#FFDAB9"),
                          ("18–23 (evening)",  "#C8A4A5")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=8),
            name=label, showlegend=True,
        ))

    x_label = "FR import deviation: actual − model (MW)" if has_actual else "FR import deviation vs period mean (MW)"
    fig.update_layout(
        title=dict(text=f"Price Error vs FR Import Deviation  ·  Pearson r = {r:.3f}",
                   font=dict(size=12, color=_PLOT_TEXT), x=0.01),
        height=380,
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        xaxis=dict(title=x_label, gridcolor=_GRID, zeroline=True,
                   zerolinecolor=_GRID, zerolinewidth=1.5),
        yaxis=dict(title="Price error: model − OMIE (€/MWh)", gridcolor=_GRID,
                   zeroline=True, zerolinecolor=_GRID, zerolinewidth=1.5),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=10), x=0.01, y=0.99, xanchor="left", yanchor="top"),
        margin=dict(l=70, r=30, t=44, b=50),
    )
    return fig


def _compute_overnight_summary(d: dict) -> list:
    """Return Dash children: a styled summary stats box for overnight diagnostics."""
    import numpy as np

    price    = d.get("price_es",    pd.Series(dtype=float))
    omie     = d.get("omie")
    fr_model = d.get("fr_net",      pd.Series(dtype=float))
    fr_act   = d.get("actual_fr_t", pd.Series(dtype=float))
    disp_es  = d.get("dispatch_es", pd.DataFrame())
    ree      = d.get("ree_actual",  pd.DataFrame())
    ts       = d.get("timestamps",  [])

    if not ts or omie is None or price.empty:
        return [html.P("Load a solved network with OMIE data to view overnight stats.",
                       style={"color": _MUTED, "fontSize": "12px"})]

    idx   = pd.to_datetime(ts)
    hour  = idx.hour
    err   = price - omie

    overnight = hour < 6
    daytime   = (hour >= 9) & (hour <= 19)

    def _fmt(s, mask=None):
        v = s[mask] if mask is not None else s
        v = v.dropna()
        return f"{v.mean():+.1f}" if len(v) > 0 else "—"

    def _n(mask):
        return int(mask.sum())

    # FR import error
    has_actual_fr = fr_act.abs().sum() > 10.0
    if has_actual_fr:
        fr_err = fr_act - fr_model
        fr_corr_val = float(np.corrcoef(
            fr_err.dropna().values, err.loc[fr_err.dropna().index].values
        )[0, 1]) if fr_err.dropna().shape[0] > 5 else float("nan")
        fr_err_night = _fmt(fr_err, overnight)
    else:
        fr_err_night = "no actual data"
        fr_corr_val  = float("nan")

    # CCGT over-dispatch
    ccgt_model_cols = [c for c in (disp_es.columns if isinstance(disp_es, pd.DataFrame) else [])
                       if "CCGT" in c and "must_run" not in c.lower()]
    has_ree_ccgt = isinstance(ree, pd.DataFrame) and "CCGT" in ree.columns
    if ccgt_model_cols and has_ree_ccgt:
        ccgt_m = disp_es[ccgt_model_cols].clip(lower=0).sum(axis=1)
        ccgt_a = ree["CCGT"]
        ccgt_dev = (ccgt_m - ccgt_a).reindex(idx)
        ccgt_dev_night = _fmt(ccgt_dev, overnight)
    else:
        ccgt_dev_night = "—"

    def _stat_row(label, val, hint=""):
        color = _CORAL if val.startswith("+") and val != "+0.0" else _TEAL if val.startswith("-") else _PLOT_TEXT
        return html.Div([
            html.Span(label, style={"fontSize": "11px", "color": _MUTED,
                                    "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
                                    "minWidth": "220px", "display": "inline-block"}),
            html.Span(val,   style={"fontSize": "12px", "fontWeight": "700",
                                    "color": color, "fontFamily": "monospace",
                                    "marginRight": "8px"}),
            html.Span(hint,  style={"fontSize": "10px", "color": _MUTED,
                                    "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif"}),
        ], style={"marginBottom": "4px"})

    corr_str = f"r = {fr_corr_val:.3f}" if np.isfinite(fr_corr_val) else "—"

    return [
        html.Div("Overnight Diagnostic Summary  (00:00–05:00)", style={
            "fontSize": "12px", "fontWeight": "700", "color": _PLOT_TEXT,
            "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
            "marginBottom": "8px", "marginTop": "4px",
            "borderBottom": f"1px solid {_GRID}", "paddingBottom": "4px",
        }),
        _stat_row("Mean price error — overnight",     _fmt(err, overnight),  "€/MWh"),
        _stat_row("Mean price error — daytime",       _fmt(err, daytime),    "€/MWh"),
        _stat_row("Overnight hours (00–05)",          str(_n(overnight)),    f"/ {len(idx)} total"),
        _stat_row("Mean FR import error — overnight", fr_err_night,          "MW  (actual − model)"),
        _stat_row("FR import–price error correlation", corr_str,             "(full period)"),
        _stat_row("Mean CCGT over-dispatch — overnight", ccgt_dev_night,     "MW  (model − REE)"),
    ]


# ── Diagnostic / correlation helpers ─────────────────────────────────────────

def _compute_price_error_correlations(d: dict) -> list[tuple[str, float]]:
    """Return [(label, pearson_r), ...] sorted by |r| desc.

    Correlates hourly price error (model − OMIE) against every ES carrier,
    FR/PT model flows, and actual ENTSOE IC flows where available.
    """
    price   = d.get("price_es", pd.Series(dtype=float))
    omie    = d.get("omie")
    dispatch = d.get("dispatch_es", pd.DataFrame())
    fr_net  = d.get("fr_net",      pd.Series(dtype=float))
    pt_net  = d.get("pt_net",      pd.Series(dtype=float))
    afr     = d.get("actual_fr_t", pd.Series(dtype=float))
    apt     = d.get("actual_pt_t", pd.Series(dtype=float))

    if omie is None or price.empty or len(omie) != len(price):
        return []

    err = (price - omie).values
    items: list[tuple[str, float]] = []

    if isinstance(dispatch, pd.DataFrame) and not dispatch.empty:
        for col in dispatch.columns:
            v = dispatch[col].values
            if len(v) == len(err) and v.std() > 0.01:
                r = float(np.corrcoef(err, v)[0, 1])
                if np.isfinite(r):
                    items.append((f"ES {col}", r))

    for label, series in [
        ("FR model flow", fr_net), ("PT model flow", pt_net),
        ("FR actual flow", afr),   ("PT actual flow", apt),
    ]:
        if isinstance(series, pd.Series) and not series.empty and len(series) == len(err):
            v = series.values
            if v.std() > 0.01:
                r = float(np.corrcoef(err, v)[0, 1])
                if np.isfinite(r):
                    items.append((label, r))

    items.sort(key=lambda x: abs(x[1]), reverse=True)
    return items


def _make_pt_import_scatter(d: dict) -> go.Figure:
    """2-panel: PT price comparison (top) + price error vs PT import deviation (bottom)."""
    price    = d.get("price_es",    pd.Series(dtype=float))
    omie     = d.get("omie")
    omie_pt  = d.get("omie_pt")    # actual PT day-ahead market price
    pt_model = d.get("pt_net",      pd.Series(dtype=float))
    pt_act   = d.get("actual_pt_t", pd.Series(dtype=float))
    pt_price = d.get("pt_price_t",  pd.Series(dtype=float))
    ts       = d.get("timestamps",  [])

    if not ts or pt_model.empty:
        return go.Figure(layout=dict(
            annotations=[dict(text="Load a network to view PT analysis",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=520, paper_bgcolor=_WHITE))

    ts_list    = list(ts)
    idx        = pd.to_datetime(ts)
    has_omie   = omie is not None and not price.empty and len(omie) == len(price)
    has_omie_pt = isinstance(omie_pt, pd.Series) and not omie_pt.empty and omie_pt.notna().sum() > 0

    fig = make_subplots(
        rows=2, cols=1, vertical_spacing=0.10,
        subplot_titles=[
            "① PT price — model shadow vs actual market vs ES OMIE (€/MWh)",
            "② Price error vs PT import deviation",
        ],
    )

    # ── ① Price comparison ────────────────────────────────────────────────────
    if not price.empty:
        fig.add_trace(go.Scatter(x=ts_list, y=price.tolist(),
                                 name="ES price (model)", mode="lines",
                                 line=dict(color="#457B9D", width=1.5)),
                      row=1, col=1)
    if has_omie:
        fig.add_trace(go.Scatter(x=ts_list, y=omie.tolist(),
                                 name="ES actual (OMIE)", mode="lines",
                                 line=dict(color=_CORAL, width=1.2, dash="dash")),
                      row=1, col=1)
    if not pt_price.empty and pt_price.notna().sum() > 0:
        fig.add_trace(go.Scatter(x=ts_list, y=pt_price.tolist(),
                                 name="PT price (model shadow)", mode="lines",
                                 line=dict(color=_TEAL, width=1.5, dash="dot")),
                      row=1, col=1)
    if has_omie_pt:
        fig.add_trace(go.Scatter(x=ts_list, y=omie_pt.tolist(),
                                 name="PT actual (OMIP)", mode="lines",
                                 line=dict(color="#10B981", width=1.8)),
                      row=1, col=1)

    # ── ② Scatter: price error vs PT import deviation ─────────────────────────
    if has_omie and not pt_model.empty:
        price_err  = (price - omie).values
        has_actual = not pt_act.empty and pt_act.abs().sum() > 10.0
        if has_actual:
            import_err = (pt_act - pt_model).values
            x_label    = "PT import deviation: actual − model (MW)"
        else:
            import_err = (pt_model - pt_model.mean()).values
            x_label    = "PT import vs period mean (MW)"

        hour  = idx.hour.values
        valid = np.isfinite(price_err) & np.isfinite(import_err)
        pe, ie, hr = price_err[valid], import_err[valid], hour[valid]
        r = float(np.corrcoef(ie, pe)[0, 1]) if len(pe) > 2 else float("nan")

        colors = ["#1D3557" if h < 6 else "#AEC6CF" if h < 12 else "#FFDAB9" if h < 18 else "#C8A4A5"
                  for h in hr]
        fig.add_trace(go.Scatter(
            x=ie, y=pe, mode="markers",
            marker=dict(color=colors, size=4, opacity=0.65, line=dict(width=0)),
            hovertemplate="PT dev: %{x:.0f} MW<br>Error: %{y:.1f} €/MWh<extra></extra>",
            showlegend=False,
        ), row=2, col=1)

        if len(pe) > 5:
            m, b = np.polyfit(ie, pe, 1)
            x_line = np.array([ie.min(), ie.max()])
            r_str = f"{r:.3f}" if np.isfinite(r) else "n/a"
            fig.add_trace(go.Scatter(
                x=x_line.tolist(), y=(m * x_line + b).tolist(),
                mode="lines", name=f"OLS fit (r={r_str})",
                line=dict(color=_TEAL, width=1.5, dash="dash"),
            ), row=2, col=1)

        for label, color in [("00–05 overnight", "#1D3557"), ("06–11 morning", "#AEC6CF"),
                              ("12–17 afternoon","#FFDAB9"), ("18–23 evening", "#C8A4A5")]:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                     marker=dict(color=color, size=8),
                                     name=label), row=2, col=1)

        fig.update_xaxes(title_text=x_label, gridcolor=_GRID,
                         zeroline=True, zerolinecolor=_GRID, row=2, col=1)
        fig.update_yaxes(title_text="Price error: model − OMIE (€/MWh)",
                         gridcolor=_GRID, zeroline=True, zerolinecolor=_GRID,
                         row=2, col=1)

    fig.update_yaxes(title_text="€/MWh", gridcolor=_GRID, row=1, col=1)
    fig.update_xaxes(gridcolor=_GRID, row=1, col=1)
    fig.update_layout(
        height=560,
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif",
                  color=_PLOT_TEXT, size=11),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=10), orientation="h", x=0, y=-0.08, xanchor="left"),
        margin=dict(l=65, r=30, t=50, b=60),
        hovermode="x unified",
    )
    return fig


# ── FR / PT generation breakdown comparison ──────────────────────────────────

_IC_DIR = ROOT / "Analysis" / "interconnector_analysis"

def _load_actual_gen_breakdown(timestamps: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load monthly actual generation for FR and PT, filtered to the solve period.

    Returns (fr_df, pt_df) each with columns: carrier, actual_mw (mean over month).
    """
    import warnings
    ts = pd.to_datetime(timestamps)
    months_in_period = ts.to_period("M").unique()

    # ── Portugal (Our World in Data monthly TWh format) ────────────────────────
    pt_df = pd.DataFrame()
    try:
        raw = pd.read_csv(_IC_DIR / "PTGAL_monthy_gen_breakdown.csv")
        raw["full_date"] = pd.to_datetime(raw["full_date"])
        raw["period"] = raw["full_date"].dt.to_period("M")
        raw = raw[raw["period"].isin(months_in_period) & ~raw["is_aggregate_series"]]
        if not raw.empty:
            # hours in each month → mean MW = TWh × 1e6 / hours
            raw["hours"] = raw["full_date"].dt.daysinmonth * 24
            raw["mean_mw"] = raw["generation_twh"] * 1e6 / raw["hours"]
            _PT_MAP = {
                "Hydro":     "hydro",
                "Wind":      "onwind",
                "Gas":       "CCGT",
                "Solar":     "solar",
                "Bioenergy": "biomass",
                "Coal":      "coal",
            }
            raw["carrier"] = raw["series"].map(_PT_MAP)
            pt_df = (raw.dropna(subset=["carrier"])
                       .groupby("carrier")["mean_mw"]
                       .mean()
                       .reset_index())
    except Exception:
        pass

    # ── France (RTE monthly TWh, tab-as-decimal format) ────────────────────────
    fr_df = pd.DataFrame()
    try:
        raw = pd.read_csv(_IC_DIR / "FR_monthy_gen_breakdown.csv",
                          names=["Date", "Filiere", "Valeur_TWh", "Nature"],
                          skiprows=1)
        raw["Valeur_TWh"] = (raw["Valeur_TWh"].astype(str)
                              .str.replace("\t", ".").astype(float))
        raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)
        raw["period"] = raw["Date"].dt.to_period("M")
        raw = raw[raw["period"].isin(months_in_period) &
                  ~raw["Filiere"].str.strip().str.startswith("Total")]
        if not raw.empty:
            raw["hours"]   = raw["Date"].dt.daysinmonth * 24
            raw["mean_mw"] = raw["Valeur_TWh"] * 1e6 / raw["hours"]
            _FR_MAP = {
                "Nuclear":                    "nuclear",
                "Hydropower":                 "hydro",
                "Wind":                       "onwind",
                "Solar":                      "solar",
                "Fossil-fired thermal":       "CCGT",
                "Renewable thermal and waste":"biomass",
            }
            raw["carrier"] = raw["Filiere"].str.strip().map(_FR_MAP)
            fr_df = (raw.dropna(subset=["carrier"])
                       .groupby("carrier")["mean_mw"]
                       .mean()
                       .reset_index())
    except Exception:
        pass

    return fr_df, pt_df


def _make_gen_breakdown_figure(d: dict) -> go.Figure:
    """Grouped bar chart: actual vs model mean generation by carrier for FR and PT.

    Shows each technology as two horizontal bars (actual=grey, model=coloured),
    making model over/under-production instantly visible.
    """
    ts        = d.get("timestamps", [])
    disp_fr   = d.get("dispatch_fr", pd.DataFrame())
    disp_pt   = d.get("dispatch_pt", pd.DataFrame())

    if not ts:
        return go.Figure().update_layout(title="No data loaded")

    actual_fr, actual_pt = _load_actual_gen_breakdown(ts)

    # ── Carrier colours (reuse dashboard palette) ─────────────────────────────
    _CARR_COLOR = {
        "nuclear": "#8C6C9F", "hydro": "#4A90D9", "onwind": "#5BA85E",
        "solar":   "#F5C542", "CCGT":  "#E07B39", "biomass": "#A0785A",
        "coal":    "#888888", "OCGT":  "#CC4444",
    }
    _CARR_LABEL = {
        "nuclear": "Nuclear", "hydro": "Hydro", "onwind": "Wind",
        "solar": "Solar", "CCGT": "CCGT/Fossil", "biomass": "Biomass/RES-T",
        "coal": "Coal", "OCGT": "OCGT",
    }

    def _model_means(disp: pd.DataFrame) -> pd.Series:
        if disp.empty:
            return pd.Series(dtype=float)
        return disp.clip(lower=0).mean()

    fr_model = _model_means(disp_fr)
    pt_model = _model_means(disp_pt)

    def _build_panel(actual_df: pd.DataFrame, model_s: pd.Series,
                     country: str) -> tuple[list, list, list]:
        """Return (carriers, actual_mw_list, model_mw_list) sorted by actual desc."""
        rows = []
        all_carriers = set()
        if not actual_df.empty:
            all_carriers.update(actual_df["carrier"].tolist())
        all_carriers.update(model_s.index.tolist())

        for c in all_carriers:
            act = float(actual_df.loc[actual_df.carrier == c, "mean_mw"].values[0]) \
                  if not actual_df.empty and (actual_df.carrier == c).any() else 0.0
            mod = float(model_s.get(c, 0.0))
            if act > 5 or mod > 5:
                rows.append((c, act, mod))

        rows.sort(key=lambda x: x[1], reverse=True)
        return rows

    fr_rows = _build_panel(actual_fr, fr_model, "FR")
    pt_rows = _build_panel(actual_pt, pt_model, "PT")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "France — Actual vs Model Generation (mean MW, Jan 2024)",
            "Portugal — Actual vs Model Generation (mean MW, Jan 2024)",
        ],
        vertical_spacing=0.14,
    )

    def _add_bars(rows, row_idx):
        carriers = [_CARR_LABEL.get(r[0], r[0]) for r in rows]
        actuals  = [r[1] for r in rows]
        models   = [r[2] for r in rows]
        colors   = [_CARR_COLOR.get(r[0], "#999") for r in rows]
        errors   = [m - a for m, a in zip(models, actuals)]

        # Actual (grey)
        fig.add_trace(go.Bar(
            name="Actual (RTE/REN)", y=carriers, x=actuals,
            orientation="h", marker_color="#888888",
            opacity=0.65, legendgroup="actual",
            showlegend=(row_idx == 1),
            hovertemplate="%{y}<br>Actual: %{x:.0f} MW<extra></extra>",
        ), row=row_idx, col=1)

        # Model (coloured)
        fig.add_trace(go.Bar(
            name="Model (PyPSA)", y=carriers, x=models,
            orientation="h", marker_color=colors,
            opacity=0.85, legendgroup="model",
            showlegend=(row_idx == 1),
            hovertemplate="%{y}<br>Model: %{x:.0f} MW<extra></extra>",
            customdata=[[f"{e:+.0f} MW"] for e in errors],
            text=[f"{e:+.0f}" for e in errors],
            textposition="outside", textfont_size=9,
        ), row=row_idx, col=1)

    _add_bars(fr_rows, 1)
    _add_bars(pt_rows, 2)

    period_str = f"{ts[0][:10]} → {ts[-1][:10]}" if ts else ""
    fig.update_layout(
        title=dict(text=f"FR / PT Generation: Model vs Actual  |  {period_str}",
                   font_size=13),
        barmode="group",
        height=580,
        legend=dict(orientation="h", y=1.04, x=0),
        margin=dict(l=10, r=20, t=60, b=10),
        font_family="Helvetica Neue, Helvetica, Arial, sans-serif",
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
    )
    fig.update_xaxes(title_text="Mean MW", row=1, col=1)
    fig.update_xaxes(title_text="Mean MW", row=2, col=1)
    return fig


# ── Diagnostic export helpers ─────────────────────────────────────────────────

def _build_diagnostic_df(d: dict) -> pd.DataFrame:
    """
    Build a comprehensive per-hour diagnostic DataFrame.

    Columns capture everything an AI agent needs to diagnose systematic pricing
    errors: dispatch by tech/tranche, model vs OMIE price, net imports, load,
    nodal spread, price-setter identification, VRE curtailment, must-run stack,
    FR/PT generation breakdown, CCGT MC decomposition, and price gap.
    """
    price      = d.get("price_es",        pd.Series(dtype=float))
    setter     = d.get("setter_es",       pd.Series(dtype=str))
    fr_net     = d.get("fr_net",          pd.Series(dtype=float))
    pt_net     = d.get("pt_net",          pd.Series(dtype=float))
    omie       = d.get("omie")
    load       = d.get("es_load",         pd.Series(dtype=float))
    disp       = d.get("dispatch_es",     pd.DataFrame())
    disp_fr    = d.get("dispatch_fr",     pd.DataFrame())
    disp_pt    = d.get("dispatch_pt",     pd.DataFrame())
    bp         = d.get("bus_prices",      pd.DataFrame())
    ts         = d.get("timestamps",      [])
    vre_pot    = d.get("vre_potential_es",  pd.Series(dtype=float))
    vre_act    = d.get("vre_actual_es",     pd.Series(dtype=float))
    must_run   = d.get("must_run_es",       pd.Series(dtype=float))
    ccgt_mc    = d.get("ccgt_mc_t",         pd.Series(dtype=float))
    setter_mc  = d.get("setter_mc_t",       pd.Series(dtype=float))
    fr_nuc     = d.get("fr_nuclear_t",      pd.Series(dtype=float))
    fr_hyd     = d.get("fr_hydro_t",        pd.Series(dtype=float))
    pt_hyd     = d.get("pt_hydro_t",        pd.Series(dtype=float))
    fr_price   = d.get("fr_price_t",        pd.Series(dtype=float))
    pt_price   = d.get("pt_price_t",        pd.Series(dtype=float))
    fr_load    = d.get("fr_load_t",         pd.Series(dtype=float))
    pt_load    = d.get("pt_load_t",         pd.Series(dtype=float))
    fr_wind    = d.get("fr_wind_t",         pd.Series(dtype=float))
    fr_solar   = d.get("fr_solar_t",        pd.Series(dtype=float))
    fr_surplus = d.get("fr_surplus_t",      pd.Series(dtype=float))
    actual_fr  = d.get("actual_fr_t",       pd.Series(dtype=float))
    actual_pt  = d.get("actual_pt_t",       pd.Series(dtype=float))
    fr_ic_sat  = d.get("fr_ic_sat_t",       pd.Series(dtype=int))
    pt_ic_sat  = d.get("pt_ic_sat_t",       pd.Series(dtype=int))
    fr_rent    = d.get("fr_rent_t",         pd.Series(dtype=float))
    pt_rent    = d.get("pt_rent_t",         pd.Series(dtype=float))
    int_cong   = d.get("internal_cong_t",   pd.Series(dtype=int))
    flex_up    = d.get("flex_up_t",         pd.Series(dtype=float))
    flex_dn    = d.get("flex_dn_t",         pd.Series(dtype=float))
    startups   = d.get("startups_t",        pd.Series(dtype=float))
    su_eur_mwh = d.get("startup_eur_mwh_t", pd.Series(dtype=float))

    idx = pd.to_datetime(ts)
    rows: dict = {"timestamp": ts}

    # Prices
    rows["model_price"]   = price.reindex(idx).tolist() if not price.empty else [None]*len(idx)
    rows["omie_price"]    = omie.reindex(idx).tolist()  if omie is not None else [None]*len(idx)
    if omie is not None and not price.empty:
        rows["price_error"] = (price.reindex(idx) - omie.reindex(idx)).tolist()
    else:
        rows["price_error"] = [None]*len(idx)
    rows["price_setter"]     = setter.reindex(idx).tolist()    if not setter.empty    else [None]*len(idx)
    rows["setter_mc_eur_mwh"]= setter_mc.reindex(idx).tolist() if not setter_mc.empty else [None]*len(idx)
    # price gap: bus price − setter MC (should be ~0 in LP; large gap = mis-attribution)
    if not price.empty and not setter_mc.empty:
        rows["price_gap_eur"] = (price.reindex(idx) - setter_mc.reindex(idx)).tolist()
    else:
        rows["price_gap_eur"] = [None]*len(idx)

    # Load & imports
    rows["es_load_MW"]    = load.reindex(idx).tolist()    if not load.empty else [None]*len(idx)
    rows["fr_import_MW"]  = fr_net.reindex(idx).tolist()  if not fr_net.empty else [None]*len(idx)
    rows["pt_import_MW"]  = pt_net.reindex(idx).tolist()  if not pt_net.empty else [None]*len(idx)
    if not fr_net.empty and not pt_net.empty:
        rows["net_import_MW"] = (fr_net.reindex(idx) + pt_net.reindex(idx)).tolist()
    else:
        rows["net_import_MW"] = [None]*len(idx)

    # VRE (wind + solar at ES)
    rows["vre_potential_MW"]    = vre_pot.reindex(idx).tolist()  if not vre_pot.empty   else [None]*len(idx)
    rows["vre_actual_MW"]       = vre_act.reindex(idx).tolist()  if not vre_act.empty   else [None]*len(idx)
    if not vre_pot.empty and not vre_act.empty:
        rows["vre_curtailment_MW"] = (vre_pot.reindex(idx) - vre_act.reindex(idx)).clip(lower=0).tolist()
    else:
        rows["vre_curtailment_MW"] = [None]*len(idx)

    # Must-run & residual demand
    rows["must_run_MW"] = must_run.reindex(idx).tolist() if not must_run.empty else [None]*len(idx)
    if not load.empty and not vre_act.empty and not must_run.empty:
        rows["residual_demand_MW"] = (
            load.reindex(idx) - vre_act.reindex(idx) - must_run.reindex(idx)
        ).tolist()
    else:
        rows["residual_demand_MW"] = [None]*len(idx)

    # CCGT time-varying MC
    rows["ccgt_mc_eur_mwh"] = ccgt_mc.reindex(idx).tolist() if not ccgt_mc.empty else [None]*len(idx)

    # FR / PT generation breakdown
    rows["fr_nuclear_MW"]  = fr_nuc.reindex(idx).tolist()  if not fr_nuc.empty  else [None]*len(idx)
    rows["fr_hydro_MW"]    = fr_hyd.reindex(idx).tolist()  if not fr_hyd.empty  else [None]*len(idx)
    rows["fr_wind_MW"]     = fr_wind.reindex(idx).tolist() if not fr_wind.empty  else [None]*len(idx)
    rows["fr_solar_MW"]    = fr_solar.reindex(idx).tolist() if not fr_solar.empty else [None]*len(idx)
    rows["pt_hydro_MW"]    = pt_hyd.reindex(idx).tolist()  if not pt_hyd.empty  else [None]*len(idx)

    # FR / PT market prices (shadow prices at model buses)
    rows["fr_price_EUR"]   = fr_price.reindex(idx).tolist()  if not fr_price.empty  else [None]*len(idx)
    rows["pt_price_EUR"]   = pt_price.reindex(idx).tolist()  if not pt_price.empty  else [None]*len(idx)
    if not fr_price.empty and not price.empty:
        rows["fr_es_price_spread"] = (price.reindex(idx) - fr_price.reindex(idx)).tolist()

    # FR / PT demand
    rows["fr_load_MW"]     = fr_load.reindex(idx).tolist()   if not fr_load.empty   else [None]*len(idx)
    rows["pt_load_MW"]     = pt_load.reindex(idx).tolist()   if not pt_load.empty   else [None]*len(idx)

    # FR generation surplus vs load (positive = France has excess capacity → exports)
    rows["fr_surplus_MW"]  = fr_surplus.reindex(idx).tolist() if not fr_surplus.empty else [None]*len(idx)

    # Actual ENTSOE interconnector flows (vs model)
    rows["actual_fr_import_MW"] = actual_fr.reindex(idx).tolist() if not actual_fr.empty else [None]*len(idx)
    rows["actual_pt_import_MW"] = actual_pt.reindex(idx).tolist() if not actual_pt.empty else [None]*len(idx)
    if not actual_fr.empty and not fr_net.empty:
        rows["fr_import_error_MW"] = (actual_fr.reindex(idx) - fr_net.reindex(idx)).tolist()
    if not actual_pt.empty and not pt_net.empty:
        rows["pt_import_error_MW"] = (actual_pt.reindex(idx) - pt_net.reindex(idx)).tolist()

    # ES dispatch by carrier (clip negatives — charging is not generation)
    def _safe_key(col):
        return (col.replace(" ", "_").replace("(", "").replace(")", "")
                   .replace("€", "EUR").replace("–", "-").replace("≤", "le")
                   .replace(">", "gt").replace("/", "_"))

    if isinstance(disp, pd.DataFrame) and not disp.empty:
        for col in disp.columns:
            rows[f"es_{_safe_key(col)}_MW"] = disp[col].clip(lower=0).reindex(idx).tolist()
        if "PHS" in disp.columns:
            rows["es_PHS_charge_MW"] = (-disp["PHS"].clip(upper=0)).reindex(idx).tolist()

    # FR dispatch by carrier
    if isinstance(disp_fr, pd.DataFrame) and not disp_fr.empty:
        for col in disp_fr.columns:
            rows[f"fr_{_safe_key(col)}_MW"] = disp_fr[col].clip(lower=0).reindex(idx).tolist()

    # PT dispatch by carrier
    if isinstance(disp_pt, pd.DataFrame) and not disp_pt.empty:
        for col in disp_pt.columns:
            rows[f"pt_{_safe_key(col)}_MW"] = disp_pt[col].clip(lower=0).reindex(idx).tolist()

    # Calendar features (useful for regression / groupby analysis)
    rows["hour"]        = idx.hour.tolist()
    rows["day_of_week"] = idx.dayofweek.tolist()   # 0=Mon … 6=Sun
    rows["month"]       = idx.month.tolist()
    rows["is_weekend"]  = (idx.dayofweek >= 5).tolist()
    rows["is_overnight"]= (idx.hour < 6).tolist()

    # Nodal spread
    if isinstance(bp, pd.DataFrame) and not bp.empty:
        spread = (bp.max(axis=1) - bp.min(axis=1)).reindex(idx)
        rows["nodal_spread_EUR"] = spread.tolist()
        rows["nodal_max_EUR"]    = bp.max(axis=1).reindex(idx).tolist()
        rows["nodal_min_EUR"]    = bp.min(axis=1).reindex(idx).tolist()
        rows["nodal_max_bus"]    = bp.idxmax(axis=1).reindex(idx).tolist()
        rows["nodal_min_bus"]    = bp.idxmin(axis=1).reindex(idx).tolist()

    # ── LP constraint mechanics ───────────────────────────────────────────────
    def _ser(s): return s.reindex(idx).tolist() if not s.empty else [None]*len(idx)
    rows["fr_ic_sat_count"]      = _ser(fr_ic_sat)
    rows["pt_ic_sat_count"]      = _ser(pt_ic_sat)
    rows["fr_congestion_rent_EUR"] = _ser(fr_rent)
    rows["pt_congestion_rent_EUR"] = _ser(pt_rent)
    rows["internal_congestion_count"] = _ser(int_cong)
    rows["flex_headroom_MW"]     = _ser(flex_up)
    rows["down_headroom_MW"]     = _ser(flex_dn)
    rows["startups_count"]       = _ser(startups)
    rows["startup_cost_EUR_MWh"] = _ser(su_eur_mwh)

    # ── CCGT tranche breakdown (price-formation detail) ───────────────────────
    disp_e = d.get("dispatch_es", pd.DataFrame())
    bounds = d.get("ccgt_bounds", {})
    cap    = d.get("capacity", {})
    for tranche in ("CCGT_lo", "CCGT_mid", "CCGT_hi"):
        if tranche in disp_e.columns:
            rows[f"{tranche}_dispatch_MW"] = disp_e[tranche].clip(lower=0).reindex(idx).tolist()
    ccgt_tot = pd.Series(0.0, index=idx)
    ccgt_cap = sum(cap.get(t, 0) for t in ("CCGT_lo", "CCGT_mid", "CCGT_hi", "CCGT"))
    for tranche in ("CCGT_lo", "CCGT_mid", "CCGT_hi", "CCGT"):
        if tranche in disp_e.columns:
            ccgt_tot = ccgt_tot + disp_e[tranche].clip(lower=0).reindex(idx, fill_value=0.0)
    rows["ccgt_total_dispatch_MW"] = ccgt_tot.tolist()
    if ccgt_cap > 0:
        rows["ccgt_util_pct"] = (ccgt_tot / ccgt_cap * 100).round(1).tolist()
    if bounds:
        rows["ccgt_lo_mc_threshold"] = [bounds.get("lo", None)] * len(idx)
        rows["ccgt_hi_mc_threshold"] = [bounds.get("hi", None)] * len(idx)

    # ── VRE price-formation accounting columns ────────────────────────────────
    pfm_keys = {
        "pfm_inflex_floor":  "pfm_inflex_floor_MW",
        "pfm_net_import":    "pfm_net_import_MW",
        "pfm_residual":      "pfm_residual_MW",
        "pfm_res_margin":    "pfm_residual_margin_MW",
        "pfm_theory_vre":    "pfm_theory_vre_hour",
        "pfm_trapped_vre":   "pfm_trapped_vre_hour",
    }
    for src, dst in pfm_keys.items():
        s = d.get(src, pd.Series(dtype=float))
        rows[dst] = _ser(s) if not s.empty else [None]*len(idx)

    return pd.DataFrame(rows)


def _build_ai_prompt(d: dict) -> str:
    """
    Generate a structured text summary for pasting into an AI agent.

    Covers: model period, overall price accuracy, per-setter calibration errors,
    CCGT tranche context, import/export patterns, VRE utilisation, must-run stack,
    FR/PT generation breakdown, CCGT MC distribution, per-setter percentiles,
    price-gap diagnostics, hydro dispatch deep-dive, FR interconnector deep-dive,
    FR generation problem, PT interconnector diagnostics, MIP/startup cost impact,
    monthly SOC trajectory, CCGT tier dispatch breakdown, VRE curtailment by tech,
    price-formation bottleneck summary, cross-border congestion rent analysis,
    FR/PT actual vs model generation comparison, and overnight vs daytime error.
    """
    price      = d.get("price_es",       pd.Series(dtype=float))
    setter     = d.get("setter_es",      pd.Series(dtype=str))
    fr_net     = d.get("fr_net",         pd.Series(dtype=float))
    pt_net     = d.get("pt_net",         pd.Series(dtype=float))
    omie       = d.get("omie")
    load       = d.get("es_load",        pd.Series(dtype=float))
    bounds     = d.get("ccgt_bounds",    {})
    ts         = d.get("timestamps",     [])
    vre_pot    = d.get("vre_potential_es", pd.Series(dtype=float))
    vre_act    = d.get("vre_actual_es",    pd.Series(dtype=float))
    must_run   = d.get("must_run_es",      pd.Series(dtype=float))
    ccgt_mc    = d.get("ccgt_mc_t",        pd.Series(dtype=float))
    setter_mc  = d.get("setter_mc_t",      pd.Series(dtype=float))
    fr_nuc     = d.get("fr_nuclear_t",     pd.Series(dtype=float))
    fr_hyd     = d.get("fr_hydro_t",       pd.Series(dtype=float))
    pt_hyd     = d.get("pt_hydro_t",       pd.Series(dtype=float))

    ccgt_tiers = d.get("ccgt_tier_mc",      {})
    mibgas_t   = d.get("mibgas_t",         pd.Series(dtype=float))
    es_wind_t  = d.get("es_wind_t",        pd.Series(dtype=float))
    es_solar_t = d.get("es_solar_t",       pd.Series(dtype=float))

    fr_price   = d.get("fr_price_t",       pd.Series(dtype=float))
    pt_price   = d.get("pt_price_t",       pd.Series(dtype=float))
    fr_load    = d.get("fr_load_t",        pd.Series(dtype=float))
    pt_load    = d.get("pt_load_t",        pd.Series(dtype=float))
    fr_wind    = d.get("fr_wind_t",        pd.Series(dtype=float))
    fr_solar   = d.get("fr_solar_t",       pd.Series(dtype=float))
    fr_surplus = d.get("fr_surplus_t",     pd.Series(dtype=float))
    actual_fr  = d.get("actual_fr_t",      pd.Series(dtype=float))
    actual_pt  = d.get("actual_pt_t",      pd.Series(dtype=float))
    fr_ic_sat  = d.get("fr_ic_sat_t",      pd.Series(dtype=int))
    pt_ic_sat  = d.get("pt_ic_sat_t",      pd.Series(dtype=int))
    fr_rent    = d.get("fr_rent_t",        pd.Series(dtype=float))
    pt_rent    = d.get("pt_rent_t",        pd.Series(dtype=float))
    int_cong   = d.get("internal_cong_t",  pd.Series(dtype=int))
    flex_up    = d.get("flex_up_t",        pd.Series(dtype=float))
    flex_dn    = d.get("flex_dn_t",        pd.Series(dtype=float))
    startups   = d.get("startups_t",       pd.Series(dtype=float))
    su_eur_mwh = d.get("startup_eur_mwh_t",pd.Series(dtype=float))

    # Hydro diagnostics
    hydro_soc_gwh    = d.get("hydro_soc_gwh",    pd.Series(dtype=float))
    hydro_inflow_gwh = d.get("hydro_inflow_gwh", pd.Series(dtype=float))
    fr_soc_gwh       = d.get("fr_soc_gwh",       pd.Series(dtype=float))
    pt_soc_gwh       = d.get("pt_soc_gwh",       pd.Series(dtype=float))
    fr_infl_gwh      = d.get("fr_infl_gwh",      pd.Series(dtype=float))
    pt_infl_gwh      = d.get("pt_infl_gwh",      pd.Series(dtype=float))
    fr_hydro_mw      = d.get("fr_hydro_mw",      pd.Series(dtype=float))
    pt_hydro_mw      = d.get("pt_hydro_mw",      pd.Series(dtype=float))
    fr_su_hydro_gw   = d.get("fr_su_hydro_gw",   pd.Series(dtype=float))
    pt_su_hydro_gw   = d.get("pt_su_hydro_gw",   pd.Series(dtype=float))

    # VRE per-technology curtailment
    vre_tech_curtail = d.get("vre_tech_curtail_mw", {})

    # Price-formation bottleneck
    pfm_inflex_floor = d.get("pfm_inflex_floor", pd.Series(dtype=float))
    pfm_net_import   = d.get("pfm_net_import",   pd.Series(dtype=float))
    pfm_residual     = d.get("pfm_residual",     pd.Series(dtype=float))
    pfm_res_margin   = d.get("pfm_res_margin",   pd.Series(dtype=float))
    pfm_theory_vre   = d.get("pfm_theory_vre",   pd.Series(dtype=int))
    pfm_trapped_vre  = d.get("pfm_trapped_vre",  pd.Series(dtype=int))
    pfm_cong_top     = d.get("pfm_cong_top",      [])

    # Actual market prices
    omie_fr = d.get("omie_fr")
    omie_pt = d.get("omie_pt")

    # FR/PT actual generation breakdown
    actual_fr_gen = d.get("actual_fr_gen", {})
    actual_pt_gen = d.get("actual_pt_gen", {})

    # Dispatch dataframes
    dispatch_es = d.get("dispatch_es", pd.DataFrame())
    dispatch_fr = d.get("dispatch_fr", pd.DataFrame())
    dispatch_pt = d.get("dispatch_pt", pd.DataFrame())

    has_omie = omie is not None and not omie.empty and len(omie) == len(price)
    net_imp  = (fr_net + pt_net) if (not fr_net.empty and not pt_net.empty) else None

    lines = ["# PyPSA-Spain Model Diagnostic Summary",
             f"Period: {ts[0][:10] if ts else '?'} -> {ts[-1][:10] if ts else '?'}  "
             f"({len(ts)} hours)",
             ""]

    # ===== 1. OVERALL PRICE ACCURACY =====
    if has_omie and not price.empty:
        err_s = price - omie
        lines += [
            "## Overall Price Accuracy",
            f"  Model mean:  {price.mean():.1f} EUR/MWh",
            f"  OMIE mean:   {omie.mean():.1f} EUR/MWh",
            f"  Mean error:  {err_s.mean():+.1f} EUR/MWh  (std {err_s.std():.1f})",
            f"  MAE:         {err_s.abs().mean():.1f} EUR/MWh",
            f"  Correlation: {price.corr(omie):.3f}",
            f"  RMSE:        {(err_s**2).mean()**0.5:.1f} EUR/MWh",
            "",
        ]

    # ===== 2. PER-SETTER CALIBRATION BREAKDOWN =====
    lines += ["## Calibration by Price-Setting Technology",
              f"{'Tech':<22} {'Hrs':>4}  {'%':>5}  {'Model':>7}  "
              f"{'OMIE':>7}  {'Error':>7}  {'ErrSD':>6}  {'p10':>6}  {'p90':>6}  {'NetImp':>8}"]
    lines.append("-" * 92)
    for carrier in setter.value_counts().index:
        mask  = setter == carrier
        n_hrs = int(mask.sum())
        mp    = price[mask]
        op    = omie[mask] if has_omie else None
        ni    = net_imp[mask] if net_imp is not None else None
        err   = f"{(mp - op).mean():+.1f}" if op is not None else "  n/a"
        esd   = f"{(mp - op).std():.1f}"   if op is not None else "  n/a"
        om    = f"{op.mean():.1f}"          if op is not None else "  n/a"
        p10s  = f"{mp.quantile(0.10):.1f}"
        p90s  = f"{mp.quantile(0.90):.1f}"
        nistr = f"{ni.mean():+.0f}"         if ni is not None else "  n/a"
        lines.append(f"{carrier:<22} {n_hrs:>4}  {100*mask.mean():>4.1f}%  "
                     f"{mp.mean():>7.1f}  {om:>7}  {err:>7}  {esd:>6}  "
                     f"{p10s:>6}  {p90s:>6}  {nistr:>8}")
    lines.append("")

    # ===== 3. PRICE GAP (bus price - setter MC) =====
    if not price.empty and not setter_mc.empty:
        gap = price - setter_mc
        gap_valid = gap.dropna()
        if len(gap_valid) > 0:
            lines += [
                "## Price Gap (bus price - setter MC)",
                "  [~0 expected in LP; large gap = setter mis-attribution or constraint shadow]",
                f"  Mean:  {gap_valid.mean():+.2f} EUR/MWh",
                f"  P90:   {gap_valid.quantile(0.90):+.2f} EUR/MWh",
                f"  P99:   {gap_valid.quantile(0.99):+.2f} EUR/MWh",
                f"  Hours with |gap| > 5EUR: {(gap_valid.abs() > 5).sum()} / {len(gap_valid)}",
                f"  Hours with |gap| > 20EUR: {(gap_valid.abs() > 20).sum()} / {len(gap_valid)}",
                "",
            ]

    # ===== 4. ES CCGT MC DISTRIBUTION =====
    if not ccgt_mc.empty:
        cmc = ccgt_mc.dropna()
        if len(cmc) > 0:
            lines += [
                "## ES CCGT MC Distribution (time-varying, MIBGAS-derived)",
                f"  P10: {cmc.quantile(0.10):.1f} EUR/MWh",
                f"  P25: {cmc.quantile(0.25):.1f} EUR/MWh",
                f"  Mean:{cmc.mean():.1f} EUR/MWh",
                f"  P75: {cmc.quantile(0.75):.1f} EUR/MWh",
                f"  P90: {cmc.quantile(0.90):.1f} EUR/MWh",
                f"  Min: {cmc.min():.1f}  Max: {cmc.max():.1f} EUR/MWh",
                "",
            ]

    # ===== 5. VRE UTILISATION =====
    if not vre_pot.empty and not vre_act.empty:
        curtail = (vre_pot - vre_act).clip(lower=0)
        pot_mean = vre_pot.mean()
        act_mean = vre_act.mean()
        curt_pct = 100.0 * curtail.mean() / pot_mean if pot_mean > 0 else 0.0
        lines += [
            "## VRE Utilisation (ES Wind + Solar)",
            f"  Mean available:  {pot_mean:.0f} MW",
            f"  Mean dispatched: {act_mean:.0f} MW",
            f"  Curtailment:     {curt_pct:.1f}%  (mean {curtail.mean():.0f} MW curtailed)",
            f"  Curtailment hours (>100 MW): {(curtail > 100).sum()} / {len(curtail)}",
            f"  Curtailment hours (>500 MW): {(curtail > 500).sum()} / {len(curtail)}",
            "",
        ]

    # ===== 6. VRE CURTAILMENT BY TECHNOLOGY =====
    if vre_tech_curtail:
        lines += ["## VRE Curtailment by Technology (ES, mean MW curtailed)"]
        for carrier in sorted(vre_tech_curtail.keys()):
            vals = pd.Series(vre_tech_curtail[carrier])
            if vals.mean() > 0.1:
                lines.append(f"  {carrier:<20}  mean {vals.mean():.0f} MW  "
                             f"(max {vals.max():.0f} MW,  hours >100 MW: {(vals > 100).sum()})")
        lines.append("")

    # ===== 7. MUST-RUN STACK =====
    if not must_run.empty:
        mr = must_run
        load_mean = load.mean() if not load.empty else float("nan")
        mr_pct = 100.0 * mr.mean() / load_mean if load_mean > 0 else float("nan")
        lines += [
            "## Must-Run Stack (ES Nuclear + Biomass + CCGT_must_run)",
            f"  Mean: {mr.mean():.0f} MW  (min {mr.min():.0f}, max {mr.max():.0f})",
            f"  As % of mean ES load: {mr_pct:.1f}%",
            "",
        ]

    # ===== 8. HYDRO DISPATCH DEEP-DIVE =====
    have_hydro_soc = not hydro_soc_gwh.empty and hydro_soc_gwh.sum() > 0
    have_hydro_inf = not hydro_inflow_gwh.empty and hydro_inflow_gwh.sum() > 0
    if have_hydro_soc or have_hydro_inf:
        lines += ["## Hydro Dispatch Deep-Dive (ES reservoir storage units)"]
        if have_hydro_soc:
            soc_start = hydro_soc_gwh.iloc[0] if len(hydro_soc_gwh) > 0 else 0
            soc_end   = hydro_soc_gwh.iloc[-1] if len(hydro_soc_gwh) > 0 else 0
            soc_min   = hydro_soc_gwh.min()
            soc_max   = hydro_soc_gwh.max()
            soc_mean  = hydro_soc_gwh.mean()
            lines += [
                f"  SOC trajectory:  {soc_start:.0f} -> {soc_end:.0f} GWh  "
                f"(min {soc_min:.0f}, max {soc_max:.0f}, mean {soc_mean:.0f})",
                f"  SOC change:      {soc_end - soc_start:+.0f} GWh over period",
            ]
            n_days = max(len(hydro_soc_gwh) / 24, 1)
            daily_depletion = (soc_start - soc_end) / n_days
            lines.append(f"  Mean daily depletion: {daily_depletion:+.0f} GWh/day")
            lines.append(f"  SOC p10: {hydro_soc_gwh.quantile(0.10):.0f} GWh  "
                         f"p90: {hydro_soc_gwh.quantile(0.90):.0f} GWh")
        if have_hydro_inf:
            inflow_mean = hydro_inflow_gwh.mean()
            inflow_total = hydro_inflow_gwh.sum()
            lines += [
                f"  Mean inflow:       {inflow_mean:.0f} GWh/h  "
                f"(total {inflow_total:.0f} GWh over period)",
                f"  Inflow p10: {hydro_inflow_gwh.quantile(0.10):.0f} GWh/h  "
                f"p90: {hydro_inflow_gwh.quantile(0.90):.0f} GWh/h",
            ]
        if not fr_soc_gwh.empty and fr_soc_gwh.sum() > 0:
            lines.append(f"  FR hydro SOC: {fr_soc_gwh.iloc[0]:.0f} -> {fr_soc_gwh.iloc[-1]:.0f} GWh  "
                         f"(mean {fr_soc_gwh.mean():.0f})")
        if not pt_soc_gwh.empty and pt_soc_gwh.sum() > 0:
            lines.append(f"  PT hydro SOC: {pt_soc_gwh.iloc[0]:.0f} -> {pt_soc_gwh.iloc[-1]:.0f} GWh  "
                         f"(mean {pt_soc_gwh.mean():.0f})")
        if not fr_infl_gwh.empty and fr_infl_gwh.sum() > 0:
            lines.append(f"  FR hydro inflow: mean {fr_infl_gwh.mean():.0f} GWh/h  "
                         f"(total {fr_infl_gwh.sum():.0f} GWh)")
        if not pt_infl_gwh.empty and pt_infl_gwh.sum() > 0:
            lines.append(f"  PT hydro inflow: mean {pt_infl_gwh.mean():.0f} GWh/h  "
                         f"(total {pt_infl_gwh.sum():.0f} GWh)")
        lines.append("")

    # ===== 9. MONTHLY SOC TRAJECTORY SUMMARY =====
    if have_hydro_soc and ts:
        idx_ = pd.to_datetime(ts)
        months = idx_.to_period("M").unique()
        if len(months) > 1:
            soc_series = hydro_soc_gwh
            lines += ["## Monthly SOC Trajectory (ES hydro, GWh)"]
            lines.append(f"  {'Month':<10}  {'Start':>7}  {'End':>7}  {'DeltaSOC':>7}  "
                         f"{'Mean':>7}  {'Min':>7}  {'Max':>7}")
            for mo in months:
                mo_mask = idx_.to_period("M") == mo
                if mo_mask.any():
                    mo_soc = soc_series[mo_mask]
                    mo_start = mo_soc.iloc[0]
                    mo_end   = mo_soc.iloc[-1]
                    lines.append(f"  {str(mo):<10}  {mo_start:>7.0f}  {mo_end:>7.0f}  "
                                 f"{mo_end - mo_start:>+7.0f}  {mo_soc.mean():>7.0f}  "
                                 f"{mo_soc.min():>7.0f}  {mo_soc.max():>7.0f}")
            lines.append("")

    # ===== 10. FR/PT GENERATION CONTEXT =====
    have_fr_nuc = not fr_nuc.empty and fr_nuc.sum() > 0
    have_fr_hyd = not fr_hyd.empty and fr_hyd.sum() > 0
    have_pt_hyd = not pt_hyd.empty and pt_hyd.sum() > 0
    if have_fr_nuc or have_fr_hyd or have_pt_hyd:
        lines += ["## FR/PT Generation Context"]
        if have_fr_nuc:
            lines.append(f"  FR nuclear: mean {fr_nuc.mean():.0f} MW  "
                         f"(min {fr_nuc.min():.0f}, max {fr_nuc.max():.0f})")
        if have_fr_hyd:
            lines.append(f"  FR hydro:   mean {fr_hyd.mean():.0f} MW  "
                         f"(min {fr_hyd.min():.0f}, max {fr_hyd.max():.0f})")
        if have_pt_hyd:
            lines.append(f"  PT hydro:   mean {pt_hyd.mean():.0f} MW  "
                         f"(min {pt_hyd.min():.0f}, max {pt_hyd.max():.0f})")
        if not fr_su_hydro_gw.empty and fr_su_hydro_gw.sum() > 0:
            lines.append(f"  FR SU hydro: mean {fr_su_hydro_gw.mean():.2f} GW  "
                         f"(min {fr_su_hydro_gw.min():.2f}, max {fr_su_hydro_gw.max():.2f})")
        if not pt_su_hydro_gw.empty and pt_su_hydro_gw.sum() > 0:
            lines.append(f"  PT SU hydro: mean {pt_su_hydro_gw.mean():.2f} GW  "
                         f"(min {pt_su_hydro_gw.min():.2f}, max {pt_su_hydro_gw.max():.2f})")
        lines.append("")

    # ===== 11. CCGT TRANCHE MC BOUNDARIES =====
    if bounds:
        lines += [
            "## CCGT Tranche MC Boundaries (Spanish fleet)",
            f"  Low  (CCGT_lo) :  MC <= {bounds.get('lo', '?'):.1f} EUR/MWh,  "
            f"capacity {bounds.get('lo_cap', '?'):.0f} MW",
            f"  Mid  (CCGT_mid):  MC {bounds.get('lo', '?'):.1f}-{bounds.get('hi', '?'):.1f} EUR/MWh,  "
            f"capacity {bounds.get('mid_cap', '?'):.0f} MW",
            f"  High (CCGT_hi) :  MC > {bounds.get('hi', '?'):.1f} EUR/MWh,  "
            f"capacity {bounds.get('hi_cap', '?'):.0f} MW",
            "",
        ]

    # ===== 12. CCGT TIER DISPATCH BREAKDOWN =====
    if isinstance(dispatch_es, pd.DataFrame) and not dispatch_es.empty:
        ccgt_tier_cols = [c for c in dispatch_es.columns if c.startswith("CCGT_")]
        if ccgt_tier_cols:
            lines += ["## CCGT Tier Dispatch Breakdown (ES, mean MW)"]
            for col in ccgt_tier_cols:
                vals = dispatch_es[col]
                cap_key = col.replace("CCGT_", "") + "_cap"
                tier_cap = bounds.get(cap_key, 1)
                lines.append(f"  {col:<20}  mean {vals.mean():.0f} MW  "
                             f"(max {vals.max():.0f} MW,  CF {100*vals.mean()/max(tier_cap,1):.0f}%)")
            lines.append("")

    # ===== 13. IMPORT/EXPORT CONTEXT =====
    if net_imp is not None:
        lines += [
            "## Import/Export Context",
            f"  Mean net import (FR+PT): {net_imp.mean():+.0f} MW",
            f"  Import hours (>200MW):   {(net_imp > 200).sum()} / {len(net_imp)}",
            f"  Export hours (< -200MW): {(net_imp < -200).sum()} / {len(net_imp)}",
            f"  FR mean flow: {fr_net.mean():+.0f} MW  (p10 {fr_net.quantile(0.10):+.0f}, p90 {fr_net.quantile(0.90):+.0f})",
            f"  PT mean flow: {pt_net.mean():+.0f} MW  (p10 {pt_net.quantile(0.10):+.0f}, p90 {pt_net.quantile(0.90):+.0f})",
            "",
        ]

    # ===== 14. ES DEMAND CONTEXT =====
    if not load.empty:
        idx_ = pd.to_datetime(ts)
        hour_ = idx_.hour
        load_overnight = load[hour_ < 6].mean() if (hour_ < 6).any() else float("nan")
        load_peak      = load[(hour_ >= 9) & (hour_ <= 19)].mean()
        lines += [
            "## ES Demand Context",
            f"  Mean ES load:      {load.mean():.0f} MW  (min {load.min():.0f}, max {load.max():.0f})",
            f"  Mean overnight:    {load_overnight:.0f} MW  (00-05)",
            f"  Mean daytime peak: {load_peak:.0f} MW  (09-19)",
            "",
        ]
        if not fr_load.empty and fr_load.sum() > 0:
            lines.append(f"  Mean FR load:      {fr_load.mean():.0f} MW")
        if not pt_load.empty and pt_load.sum() > 0:
            lines.append(f"  Mean PT load:      {pt_load.mean():.0f} MW")
        if not must_run.empty:
            mr_frac = 100 * must_run.mean() / load.mean() if load.mean() > 0 else float("nan")
            lines.append(f"  Must-run / load:   {mr_frac:.1f}%  (ES nuclear+biomass+must_run CCGT)")
        lines.append("")


    # ===== 15. FR MARKET & EXPORT DRIVER ANALYSIS =====
    have_fr_price = not fr_price.empty and fr_price.notna().sum() > 5
    if have_fr_price or not fr_surplus.empty:
        lines += ["## FR Market & Export Driver Analysis"]
        if have_fr_price and not price.empty:
            spread = (price - fr_price).dropna()
            corr   = price.corr(fr_price) if fr_price.notna().sum() > 5 else float("nan")
            lines += [
                f"  FR mean price (model shadow): {fr_price.mean():.1f} EUR/MWh",
                f"  ES mean price (model):        {price.mean():.1f} EUR/MWh",
                f"  Mean ES-FR spread:            {spread.mean():+.1f} EUR/MWh  (positive = ES premium)",
                f"  ES/FR price correlation:      {corr:.3f}  (1.0 = fully coupled)",
                f"  Hours |spread| > 10 EUR:        {(spread.abs() > 10).sum()} / {len(spread)} "
                f"  [= congestion or price floor biting]",
            ]
            if isinstance(omie_fr, pd.Series) and not omie_fr.empty and omie_fr.notna().sum() > 5:
                fr_actual_mean = omie_fr.mean()
                fr_model_mean  = fr_price.mean()
                fr_price_err   = fr_model_mean - fr_actual_mean
                lines += [
                    f"  FR actual market price (EPEX): {fr_actual_mean:.1f} EUR/MWh",
                    f"  FR model vs actual bias:       {fr_price_err:+.1f} EUR/MWh",
                ]
        if not fr_surplus.empty and fr_surplus.sum() != 0:
            surp = fr_surplus.dropna()
            surplus_hrs = (surp > 0).sum()
            deficit_hrs = (surp < 0).sum()
            surp_export_corr = float("nan")
            if not fr_net.empty and len(surp) == len(fr_net):
                surp_export_corr = fr_surplus.corr(fr_net)
            lines += [
                f"  FR surplus hours (gen>load):  {surplus_hrs} / {len(surp)}",
                f"  FR deficit hours (load>gen):  {deficit_hrs} / {len(surp)}",
                f"  Mean FR surplus:              {surp.mean():+.0f} MW",
                f"  Surplus-export correlation:   {surp_export_corr:.3f}  "
                f"[expect ~0.5-0.8 when IC not congested]",
            ]
        if not fr_wind.empty and not fr_solar.empty:
            lines.append(f"  FR wind mean:  {fr_wind.mean():.0f} MW  |  FR solar mean: {fr_solar.mean():.0f} MW")
        lines.append("")

    # ===== 16. FR INTERCONNECTOR DEEP-DIVE =====
    if have_fr_price and not fr_net.empty:
        lines += ["## FR Interconnector Deep-Dive"]
        _fr_cong_hrs = int((fr_ic_sat > 0).sum()) if not fr_ic_sat.empty else 0
        _n_ts = len(ts)
        lines += [
            f"  FR IC congested hours:  {_fr_cong_hrs} / {_n_ts}  "
            f"({100*_fr_cong_hrs/max(_n_ts,1):.1f}%)",
        ]
        if not fr_ic_sat.empty:
            _fr_full_hrs = int((fr_ic_sat == fr_ic_sat.max()).sum()) if fr_ic_sat.max() > 0 else 0
            lines.append(f"  All FR lines/INELFE saturated: {_fr_full_hrs} hrs")
        if not fr_ic_sat.empty and not price.empty and not fr_price.empty:
            cong_mask = fr_ic_sat > 0
            if cong_mask.any() and (~cong_mask).any():
                spread_cong = (price[cong_mask] - fr_price[cong_mask]).dropna()
                spread_free = (price[~cong_mask] - fr_price[~cong_mask]).dropna()
                lines += [
                    f"  Mean ES-FR spread when congested:   {spread_cong.mean():+.1f} EUR/MWh",
                    f"  Mean ES-FR spread when uncongested: {spread_free.mean():+.1f} EUR/MWh",
                ]
        if have_fr_nuc:
            fr_nuc_pnom = float(d.get("fr_nuclear_pnom", 0))
            if fr_nuc_pnom > 0:
                fr_nuc_cf = 100 * fr_nuc.mean() / fr_nuc_pnom
                lines += [
                    f"  FR nuclear capacity factor: {fr_nuc_cf:.1f}%  "
                    f"(p_nom {fr_nuc_pnom:.0f} MW, mean dispatch {fr_nuc.mean():.0f} MW)",
                ]
        # Actual vs model IC flow comparison
        if not actual_fr.empty and actual_fr.abs().sum() > 10:
            fr_err = actual_fr - fr_net
            idx_ = pd.to_datetime(ts)
            ov_mask = idx_.hour < 6
            lines += [
                f"  Actual vs model FR flow:",
                f"    Model mean: {fr_net.mean():+.0f} MW  |  Actual mean: {actual_fr.mean():+.0f} MW",
                f"    Error (actual-model): mean {fr_err.mean():+.0f} MW  "
                f"(overnight: {fr_err[ov_mask].mean():+.0f} MW)",
            ]
            if has_omie:
                fr_err_corr = fr_err.corr(price - omie)
                lines.append(f"    Corr(FR error, price error): {fr_err_corr:.3f}")
        lines.append("")

    # ===== 17. FR GENERATION PROBLEM =====
    if have_fr_nuc:
        lines += ["## FR Generation Problem"]
        # Nuclear must-run floor
        fr_nuc_pmin_pu = float(d.get("fr_nuclear_pmin_pu", 0.50))
        fr_nuc_pmax_pu = float(d.get("fr_nuclear_pmax_pu", 0.62))
        fr_nuc_pnom = float(d.get("fr_nuclear_pnom", 0))
        fr_nuc_floor = fr_nuc_pmin_pu * fr_nuc_pnom
        fr_nuc_ceil  = fr_nuc_pmax_pu * fr_nuc_pnom
        lines += [
            f"  Nuclear must-run floor: {fr_nuc_floor:.0f} MW  "
            f"(p_min_pu={fr_nuc_pmin_pu:.2f} x {fr_nuc_pnom:.0f} MW)",
            f"  Nuclear max ceiling:    {fr_nuc_ceil:.0f} MW  "
            f"(p_max_pu={fr_nuc_pmax_pu:.2f})",
            f"  Mean dispatch vs floor: {fr_nuc.mean():.0f} MW vs {fr_nuc_floor:.0f} MW floor  "
            f"({fr_nuc.mean() - fr_nuc_floor:+.0f} MW headroom)",
        ]
        # FR generation mix summary
        if not fr_wind.empty and not fr_solar.empty:
            fr_total_gen = fr_nuc + fr_hyd + fr_wind + fr_solar
            lines += [
                f"  FR total gen: mean {fr_total_gen.mean():.0f} MW  "
                f"(nuc {100*fr_nuc.mean()/max(fr_total_gen.mean(),1):.0f}%, "
                f"hyd {100*fr_hyd.mean()/max(fr_total_gen.mean(),1):.0f}%, "
                f"wind {100*fr_wind.mean()/max(fr_total_gen.mean(),1):.0f}%, "
                f"solar {100*fr_solar.mean()/max(fr_total_gen.mean(),1):.0f}%)",
            ]
        # FR load vs generation balance
        if not fr_load.empty and fr_load.sum() > 0:
            fr_balance = fr_total_gen - fr_load
            lines += [
                f"  FR load: mean {fr_load.mean():.0f} MW",
                f"  FR gen-load balance: {fr_balance.mean():+.0f} MW  "
                f"(positive = surplus for export)",
            ]
        # FR price vs FR nuclear dispatch correlation
        if have_fr_price and not fr_price.empty:
            fr_nuc_price_corr = fr_nuc.corr(fr_price) if fr_nuc.std() > 0.1 else float("nan")
            lines.append(f"  Corr(FR nuclear dispatch, FR price): {fr_nuc_price_corr:.3f}")
        lines.append("")

    # ===== 18. PT INTERCONNECTOR & GENERATION DIAGNOSTICS =====
    if not pt_net.empty:
        lines += ["## PT Interconnector & Generation Diagnostics"]
        _pt_cong_hrs = int((pt_ic_sat > 0).sum()) if not pt_ic_sat.empty else 0
        lines += [
            f"  PT IC congested hours:  {_pt_cong_hrs} / {_n_ts}  "
            f"({100*_pt_cong_hrs/max(_n_ts,1):.1f}%)",
        ]
        if not pt_price.empty and pt_price.notna().sum() > 5 and not price.empty:
            pt_spread = (price - pt_price).dropna()
            pt_corr = price.corr(pt_price)
            lines += [
                f"  PT mean price (model shadow): {pt_price.mean():.1f} EUR/MWh",
                f"  Mean ES-PT spread:            {pt_spread.mean():+.1f} EUR/MWh",
                f"  ES/PT price correlation:      {pt_corr:.3f}",
            ]
            if isinstance(omie_pt, pd.Series) and not omie_pt.empty and omie_pt.notna().sum() > 5:
                lines.append(f"  PT actual market price (OMIE): {omie_pt.mean():.1f} EUR/MWh")
        if not actual_pt.empty and actual_pt.abs().sum() > 10:
            pt_err = actual_pt - pt_net
            lines += [
                f"  Actual vs model PT flow:",
                f"    Model mean: {pt_net.mean():+.0f} MW  |  Actual mean: {actual_pt.mean():+.0f} MW",
                f"    Error (actual-model): mean {pt_err.mean():+.0f} MW",
            ]
        if have_pt_hyd:
            lines.append(f"  PT hydro dispatch: mean {pt_hyd.mean():.0f} MW")
        lines.append("")

    # ===== 19. ACTUAL VS MODEL INTERCONNECTOR FLOWS (ENTSOE) =====
    has_actual_fr = not actual_fr.empty and actual_fr.abs().sum() > 10
    has_actual_pt = not actual_pt.empty and actual_pt.abs().sum() > 10
    if (has_actual_fr or has_actual_pt) and has_omie:
        lines += ["## Actual vs Model Interconnector Flows (ENTSOE)"]
        if has_actual_fr and not fr_net.empty:
            fr_err = actual_fr - fr_net
            idx_   = pd.to_datetime(ts)
            ov_mask = idx_.hour < 6
            lines += [
                f"  FR model mean:  {fr_net.mean():+.0f} MW  |  Actual mean: {actual_fr.mean():+.0f} MW",
                f"  FR import error (actual-model): mean {fr_err.mean():+.0f} MW  "
                f"(overnight: {fr_err[ov_mask].mean():+.0f} MW)",
                f"  Corr(FR import error, price error): "
                f"{fr_err.corr(price - omie):.3f}" if has_omie else "",
            ]
        if has_actual_pt and not pt_net.empty:
            pt_err = actual_pt - pt_net
            lines.append(f"  PT model mean:  {pt_net.mean():+.0f} MW  |  Actual mean: {actual_pt.mean():+.0f} MW")
            lines.append(f"  PT import error: mean {pt_err.mean():+.0f} MW")
        lines.append("")

    # ===== 20. LP CONSTRAINT MECHANICS (Price Formation) =====
    _n_ts = len(ts)
    _has_cong = not fr_ic_sat.empty or not int_cong.empty
    if _has_cong:
        lines += ["## LP Constraint Mechanics (Price Formation)"]
        if not fr_ic_sat.empty and not pt_ic_sat.empty:
            _fr_cong_hrs = int((fr_ic_sat > 0).sum())
            _pt_cong_hrs = int((pt_ic_sat > 0).sum())
            _fr_full_hrs = int((fr_ic_sat == fr_ic_sat.max()).sum()) if fr_ic_sat.max() > 0 else 0
            lines += [
                f"  -- Interconnector Saturation (>=98% capacity = congested) --",
                f"  FR IC congested:  {_fr_cong_hrs} / {_n_ts} hrs  "
                f"(all FR lines/INELFE saturated: {_fr_full_hrs} hrs)",
                f"  PT IC congested:  {_pt_cong_hrs} / {_n_ts} hrs",
            ]
        if not fr_rent.empty and not pt_rent.empty:
            lines += [
                f"  FR congestion rent:  mean {fr_rent.mean():.1f} EUR/MWh  "
                f"(p90: {np.nanpercentile(fr_rent.dropna(), 90):.1f})",
                f"  PT congestion rent:  mean {pt_rent.mean():.1f} EUR/MWh  "
                f"(p90: {np.nanpercentile(pt_rent.dropna(), 90):.1f})",
                f"  [rent = |country shadow price - ES price|; non-zero -> ICs price-decoupled]",
            ]
        if not int_cong.empty:
            _int_hrs = int((int_cong > 0).sum())
            lines += [
                f"  Internal ES congestion:  {_int_hrs} / {_n_ts} hrs with >=1 binding line  "
                f"(mean {int_cong.mean():.1f} lines/hr)",
            ]
        if not flex_up.empty and not flex_dn.empty:
            lines += [
                f"",
                f"  -- Supply Stack Headroom (CCGT + OCGT + hydro, online units only) --",
                f"  Upward headroom:   mean {flex_up.mean():.0f} MW  "
                f"(min: {flex_up.min():.0f} MW,  p10: {np.nanpercentile(flex_up.dropna(), 10):.0f} MW)",
                f"  Downward headroom: mean {flex_dn.mean():.0f} MW  "
                f"(min: {flex_dn.min():.0f} MW,  p10: {np.nanpercentile(flex_dn.dropna(), 10):.0f} MW)",
                f"  [<500 MW upward headroom -> model vulnerable to OCGT/VOLL dispatch]",
            ]
        if not startups.empty:
            _su_tot  = int(startups.sum())
            _su_hrs  = int((startups > 0).sum())
            _su_cost = float(su_eur_mwh.mean()) if not su_eur_mwh.empty else 0.0
            lines += [
                f"",
                f"  -- MIP Startup Events (ES CCGT + OCGT) --",
                f"  Total startups: {_su_tot}  over period  ({_su_hrs} hours with >=1 startup)",
                f"  Mean startups/day: {_su_tot / max(_n_ts / 24, 1):.1f}",
                f"  Startup cost impact: {_su_cost:.3f} EUR/MWh mean  "
                f"({'unconfigured - start_up_cost = 0' if _su_cost == 0 else 'significant uplift present'})",
            ]
        lines.append("")

    # ===== 21. CROSS-BORDER CONGESTION RENT ANALYSIS =====
    if not fr_rent.empty and not pt_rent.empty:
        lines += ["## Cross-Border Congestion Rent Analysis"]
        lines += [
            f"  FR rent > 5 EUR/MWh: {(fr_rent > 5).sum()} / {_n_ts} hrs  "
            f"({100*(fr_rent > 5).sum()/max(_n_ts,1):.1f}%)",
            f"  PT rent > 5 EUR/MWh: {(pt_rent > 5).sum()} / {_n_ts} hrs  "
            f"({100*(pt_rent > 5).sum()/max(_n_ts,1):.1f}%)",
        ]
        # Correlation between FR rent and ES price error
        if has_omie and not price.empty:
            err_s = price - omie
            fr_rent_err_corr = fr_rent.corr(err_s) if fr_rent.std() > 0.01 else float("nan")
            pt_rent_err_corr = pt_rent.corr(err_s) if pt_rent.std() > 0.01 else float("nan")
            lines += [
                f"  Corr(FR rent, price error): {fr_rent_err_corr:.3f}  "
                f"[positive = congestion inflates ES price]",
                f"  Corr(PT rent, price error): {pt_rent_err_corr:.3f}",
            ]
        lines.append("")

    # ===== 22. PRICE-FORMATION BOTTLENECK SUMMARY =====
    if not pfm_theory_vre.empty:
        theory_vre_hrs = int(pfm_theory_vre.sum())
        trapped_vre_hrs = int(pfm_trapped_vre.sum()) if not pfm_trapped_vre.empty else 0
        lines += ["## Price-Formation Bottleneck Summary"]
        lines += [
            f"  Hours where VRE could theoretically set price: {theory_vre_hrs} / {_n_ts}",
            f"  Hours where VRE is 'trapped' (gas sets price despite VRE potential): {trapped_vre_hrs} / {_n_ts}",
        ]
        if theory_vre_hrs > 0:
            trap_pct = 100 * trapped_vre_hrs / theory_vre_hrs
            lines.append(f"  Trapped VRE ratio: {trap_pct:.1f}%  "
                         f"[>50% = must-run floor blocks VRE from clearing]")
        if not pfm_res_margin.empty:
            mean_margin = pfm_res_margin.mean()
            neg_margin_hrs = int((pfm_res_margin < 0).sum())
            lines += [
                f"  Mean residual margin (load - VRE - IC - inflex): {mean_margin:.0f} MW",
                f"  Hours with negative residual margin: {neg_margin_hrs} / {_n_ts}  "
                f"[negative = VRE alone could meet residual demand]",
            ]
        if pfm_cong_top:
            lines.append(f"  Top congested lines during trapped-VRE hours:")
            for entry in pfm_cong_top[:5]:
                lines.append(f"    {entry['line']:<25}  {entry['count']} hrs  "
                             f"({entry['bus0']} - {entry['bus1']})")
        lines.append("")

    # ===== 23. CCGT TIER MC BREAKDOWN (Step 3) =====
    tier_mc  = d.get("ccgt_tier_mc", {})
    mibgas_t = d.get("mibgas_t", pd.Series(dtype=float))
    if tier_mc:
        co2_price  = float(MODEL_CONFIG.get("co2_price", 60.0))
        co2_inten  = float(MODEL_CONFIG.get("gas_co2_intensity_th", 0.202))
        co2_per_mwh_gas = co2_price * co2_inten
        lines += ["## CCGT Tier MC Breakdown (Step 3)"]
        lines.append(f"  CO2 at {co2_price:.0f} EUR/t x {co2_inten:.3f} tCO2/MWh_th"
                     f" = {co2_per_mwh_gas:.2f} EUR/MWh_th CO2 adder")
        for tier, series in sorted(tier_mc.items()):
            vals = series.dropna() if hasattr(series, "dropna") else pd.Series(series)
            if len(vals) > 0:
                lines.append(f"  {tier}: mean {vals.mean():.1f}  p10 {vals.quantile(0.10):.1f}"
                             f"  p90 {vals.quantile(0.90):.1f}  min {vals.min():.1f}"
                             f"  max {vals.max():.1f} EUR/MWh")
        mb_valid = mibgas_t.dropna() if hasattr(mibgas_t, "dropna") else pd.Series(dtype=float)
        if len(mb_valid) > 0:
            lines += [
                f"  Implied MIBGAS (back-calc from T1 MC):",
                f"    mean {mb_valid.mean():.1f}  min {mb_valid.min():.1f}"
                f"  max {mb_valid.max():.1f} EUR/MWh_th",
                f"  [target: Jan 2024 MIBGAS PVB approx 28-32 EUR/MWh_th]",
            ]
        lines.append("")

    # ===== 24. HIGH-VRE HOUR CROSS-ANALYSIS (Step 4) =====
    es_wind  = d.get("es_wind_t",  pd.Series(dtype=float))
    es_solar = d.get("es_solar_t", pd.Series(dtype=float))
    if not es_wind.empty and not price.empty and setter is not None and not setter.empty:
        vre_tot  = es_wind + es_solar if not es_solar.empty else es_wind
        vre_pot2 = d.get("vre_potential_es", pd.Series(dtype=float))
        curtail2 = (vre_pot2 - vre_tot).clip(lower=0) if not vre_pot2.empty else pd.Series(0.0, index=vre_tot.index)
        gas_set  = {"CCGT", "CCGT_flex", "OCGT"}
        net_imp2 = (fr_net + pt_net) if (not fr_net.empty and not pt_net.empty) else None
        high_vre_q = vre_tot.quantile(0.75)
        high_mask  = vre_tot > high_vre_q
        lines += [f"## High-VRE Hour Cross-Analysis (Step 4) - top-quartile VRE > {high_vre_q:.0f} MW"]
        lines.append(f"  Total high-VRE hours:  {high_mask.sum()}")
        curt_high  = (curtail2[high_mask] > 100).sum()
        gas_high   = setter[high_mask].isin(gas_set).sum()
        nuc_high   = (setter[high_mask] == "nuclear").sum()
        hyd_high   = (setter[high_mask] == "hydro").sum()
        exp_high   = (net_imp2[high_mask] < -200).sum() if net_imp2 is not None else "n/a"
        lines += [
            f"  Curtailing (>100 MW) in high-VRE hrs:    {curt_high}",
            f"  Gas sets price in high-VRE hrs:           {gas_high}  ({100*gas_high/max(high_mask.sum(),1):.0f}%)",
            f"  Nuclear sets price in high-VRE hrs:       {nuc_high}",
            f"  Hydro sets price in high-VRE hrs:         {hyd_high}",
            f"  Exporting (net imp < -200 MW) in high-VRE: {exp_high}",
            f"  [If gas % high in high-VRE hours -> must-run too high or IC too constrained]",
        ]
        lines.append("")

    # ===== 25. MONTHLY ERROR BREAKDOWN (Step 6) =====
    if has_omie and not price.empty and ts:
        idx_  = pd.to_datetime(ts)
        err_s = price - omie
        months = idx_.to_period("M").unique()
        if len(months) > 1:
            lines += ["## Monthly Error Breakdown (Step 6)"]
            lines.append(f"  {'Month':<10}  {'Hrs':>4}  {'MeanErr':>8}  {'MAE':>6}  {'Corr':>6}")
            for mo in months:
                mo_mask = idx_.to_period("M") == mo
                e = err_s[mo_mask]
                p = price[mo_mask]
                o = omie[mo_mask]
                corr = p.corr(o) if len(p) > 2 else float("nan")
                lines.append(f"  {str(mo):<10}  {mo_mask.sum():>4}  "
                             f"{e.mean():>+8.1f}  {e.abs().mean():>6.1f}  {corr:>6.3f}")
            lines.append("")

    # ===== 26. ERROR BY TIME-OF-DAY AND SEASON =====
    if has_omie and not price.empty:
        idx_   = pd.to_datetime(ts)
        hour_  = idx_.hour
        err_s  = price - omie
        lines += ["## Error by Time-of-Day"]
        for label, mask in [
            ("Overnight (00-05)", hour_ < 6),
            ("Morning  (06-11)", (hour_ >= 6) & (hour_ < 12)),
            ("Afternoon (12-17)", (hour_ >= 12) & (hour_ < 18)),
            ("Evening  (18-23)", hour_ >= 18),
        ]:
            if mask.any():
                e = err_s[mask]
                lines.append(f"  {label:<22}  n={mask.sum():>4}  "
                             f"mean {e.mean():+.1f}  MAE {e.abs().mean():.1f}  "
                             f"std {e.std():.1f}  EUR/MWh")
        # Weekend vs weekday breakdown
        weekday_mask = idx_.weekday < 5
        if weekday_mask.any() and (~weekday_mask).any():
            wd_err = err_s[weekday_mask]
            we_err = err_s[~weekday_mask]
            lines += [
                f"  Weekday (Mon-Fri):       n={weekday_mask.sum():>4}  "
                f"mean {wd_err.mean():+.1f}  MAE {wd_err.abs().mean():.1f}",
                f"  Weekend (Sat-Sun):       n={(~weekday_mask).sum():>4}  "
                f"mean {we_err.mean():+.1f}  MAE {we_err.abs().mean():.1f}",
            ]
        lines.append("")

    # ===== 27. PRICE ERROR CORRELATIONS =====
    corr_items = _compute_price_error_correlations(d)
    if corr_items:
        lines += ["## Price Error Correlations (sorted by |r|)",
                  "  [Pearson r between hourly price error (model-OMIE) and each variable]",
                  f"  {'Variable':<28}  {'r':>7}  {'Direction'}"]
        lines.append("  " + "-" * 58)
        for label, r in corr_items[:20]:
            direction = "error up when var up" if r > 0 else "error down when var up"
            lines.append(f"  {label:<28}  {r:>+.3f}  {direction}")
        lines.append("")

    # ===== 28. LIKELY DIAGNOSIS TARGETS =====
    lines += [
        "## Likely Diagnosis Targets",
        "  1. If CCGT error is large (+) and import is high: IC factor may be too low,",
        "     limiting cheap FR nuclear from setting a lower clearing price.",
        "  2. If VRE curtailment is near-zero but VRE never sets the price: nuclear p_min",
        "     floor prevents VRE from being the marginal unit (must-run % of load > VRE %)",
        "  3. If must-run > 40% of load: CCGT_must_run or nuclear p_min is too high -",
        "     LP fills remaining demand with flex CCGT regardless of VRE generation.",
        "  4. If price gap (bus price - setter MC) > 5 EUR/MWh for many hours: the detected",
        "     price-setter is not the actual shadow-price generator - check transmission",
        "     constraints or storage shadow prices.",
        "  5. If FR nuclear dispatch is consistently near its p_min floor and FR->ES flow",
        "     is low: raise FR missing demand scalar or lower FR nuclear p_min.",
        "  6. CO2 price multiplies directly into all thermal MC values - check calibration.",
        "  7. Large nodal spread hours suggest internal transmission constraints are binding.",
        "  8. If hydro SOC depletes rapidly in early months: monthly MC profile may be too",
        "     low for winter months, causing over-dispatch in Jan-Feb.",
        "  9. If FR IC is 100% congested: check FR nuclear p_max_pu ceiling and FR missing",
        "     demand - the model may be dispatching FR nuclear below its true capability.",
        " 10. If trapped VRE ratio > 50%: must-run floor (nuclear p_min + biomass + CCGT_mr)",
        "     is too high relative to minimum residual demand.",
    ]

    return "\n".join(lines)

def make_calibration_table(d: dict) -> list:
    """
    Price-setter error table: for each technology that sets the marginal price,
    show mean model price, mean OMIE price, mean error (model−OMIE), std of
    error, and mean net import during those hours.

    This reveals systematic over/under-pricing per technology and whether
    errors correlate with import/export behaviour.
    """
    price  = d.get("price_es")
    setter = d.get("setter_es")
    fr_net = d.get("fr_net")
    pt_net = d.get("pt_net")
    omie   = d.get("omie")

    if not isinstance(price, pd.Series) or price.empty:
        return [html.P("No price data loaded.", style={"color": _MUTED, "fontSize": "12px"})]

    if not isinstance(setter, pd.Series) or setter.empty:
        return [html.P("No price-setter data.", style={"color": _MUTED, "fontSize": "12px"})]

    has_omie = isinstance(omie, pd.Series) and not omie.empty and len(omie) == len(price)
    net_imp  = None
    if isinstance(fr_net, pd.Series) and isinstance(pt_net, pd.Series):
        if len(fr_net) == len(price) and len(pt_net) == len(price):
            net_imp = fr_net + pt_net

    rows = []
    # Order carriers by frequency (most common setter first)
    carrier_counts = setter.value_counts()
    for carrier in carrier_counts.index:
        mask = setter == carrier
        n_hrs = int(mask.sum())
        if n_hrs == 0:
            continue

        mp     = price[mask]
        m_mean = mp.mean()
        m_std  = mp.std()
        pct    = 100.0 * n_hrs / len(setter)

        omie_mean = omie_diff_mean = omie_diff_std = None
        if has_omie:
            op            = omie[mask]
            omie_mean     = op.mean()
            diff          = mp - op
            omie_diff_mean = diff.mean()
            omie_diff_std  = diff.std()

        imp_mean = float(net_imp[mask].mean()) if net_imp is not None else None

        row = {
            "Technology":      carrier,
            "Hours":           n_hrs,
            "% Time":          f"{pct:.1f}%",
            "Model (€/MWh)":   f"{m_mean:.1f}",
            "Model σ":         f"{m_std:.1f}",
            "OMIE (€/MWh)":    f"{omie_mean:.1f}" if omie_mean is not None else "—",
            "Error (M−O)":     f"{omie_diff_mean:+.1f}" if omie_diff_mean is not None else "—",
            "Error σ":         f"{omie_diff_std:.1f}" if omie_diff_std is not None else "—",
            "Net Import (MW)": f"{imp_mean:+.0f}" if imp_mean is not None else "—",
            "_color":          COLORS.get(str(carrier), "#BBBBBB"),
            "_err":            omie_diff_mean if omie_diff_mean is not None else 0.0,
        }
        rows.append(row)

    if not rows:
        return [html.P("No data.", style={"color": _MUTED})]

    columns = [
        "Technology", "Hours", "% Time",
        "Model (€/MWh)", "Model σ",
        "OMIE (€/MWh)", "Error (M−O)", "Error σ",
        "Net Import (MW)",
    ]

    # Build conditional style: colour "Error (M−O)" cell by sign/magnitude
    style_data_conditional = []
    for i, row in enumerate(rows):
        err = row["_err"]
        if abs(err) < 2:
            bg = "rgba(220,230,240,0.25)"
        elif err > 0:
            # overpriced — red gradient
            intensity = min(1.0, abs(err) / 30.0)
            r = int(255)
            g = int(235 - intensity * 90)
            b = int(235 - intensity * 90)
            bg = f"rgba({r},{g},{b},0.55)"
        else:
            # underpriced — teal/green gradient
            intensity = min(1.0, abs(err) / 30.0)
            r = int(235 - intensity * 80)
            g = int(255)
            b = int(235 - intensity * 60)
            bg = f"rgba({r},{g},{b},0.55)"
        style_data_conditional.append({
            "if": {"row_index": i, "column_id": "Error (M−O)"},
            "backgroundColor": bg,
            "fontWeight": "600",
        })
        # Colour tech name cell with carrier colour swatch (left border)
        style_data_conditional.append({
            "if": {"row_index": i, "column_id": "Technology"},
            "borderLeft": f"4px solid {row['_color']}",
        })

    table_rows = [{c: r[c] for c in columns} for r in rows]

    tbl = dash_table.DataTable(
        data=table_rows,
        columns=[{"name": c, "id": c} for c in columns],
        style_table={"overflowX": "auto", "borderRadius": "6px",
                     "border": f"1px solid {_GRID}"},
        style_header={
            "backgroundColor": _SLATE,
            "color": "#C8D4DC",
            "fontWeight": "600",
            "fontSize": "11px",
            "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
            "borderBottom": f"2px solid {_CORAL}",
            "padding": "8px 10px",
        },
        style_cell={
            "backgroundColor": _WHITE,
            "color": _PLOT_TEXT,
            "fontSize": "11px",
            "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
            "padding": "7px 10px",
            "border": f"1px solid {_GRID}",
            "textAlign": "right",
            "minWidth": "70px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Technology"}, "textAlign": "left",
             "fontWeight": "600", "minWidth": "120px"},
        ],
        style_data_conditional=style_data_conditional,
        style_as_list_view=False,
        sort_action="native",
        page_size=20,
    )

    note = html.Div([
        html.Span("Error (M−O) = model price minus OMIE actual, averaged over hours when that "
                  "technology sets the marginal price.  ",
                  style={"fontSize": "10px", "color": _MUTED}),
        html.Span("Red = model overprices.  Green = model underprices.  "
                  "Net Import > 0 means Spain was importing when that tech was marginal.",
                  style={"fontSize": "10px", "color": _MUTED}),
    ], style={"marginTop": "8px"})

    return [tbl, note]


# ── Price Formation (VRE Bottleneck) figures ─────────────────────────────────

def _make_pfm_ts(d: dict) -> go.Figure:
    """Time series: VRE potential vs actual dispatch + residual demand + price,
    with trapped-VRE hours shaded in light red."""
    ts  = pd.to_datetime(d.get("timestamps", []))
    if ts.empty:
        return go.Figure().update_layout(title="No data loaded")

    vre_pot  = d.get("vre_potential_es", pd.Series(dtype=float))
    vre_act  = d.get("vre_actual_es",    pd.Series(dtype=float))
    residual = d.get("pfm_residual",     pd.Series(dtype=float))
    floor    = d.get("pfm_inflex_floor", pd.Series(dtype=float))
    price    = d.get("price_es",         pd.Series(dtype=float))
    trapped  = d.get("pfm_trapped_vre",  pd.Series(dtype=int))

    if residual.empty:
        fig = go.Figure()
        fig.update_layout(title="Reload the network to compute price-formation data",
                          paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
                          font=dict(color=_MUTED, size=11))
        return fig

    # REE actual VRE
    ree = d.get("ree_actual", pd.DataFrame())
    ree_vre = pd.Series(0.0, index=ts)
    for col in ["solar", "onwind"]:
        if isinstance(ree, pd.DataFrame) and col in ree.columns:
            ree_vre = ree_vre + ree[col].reindex(ts, fill_value=0.0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.06)

    # Shading: trapped VRE hours (add as shapes, not vrect — avoids subplot axis confusion)
    _trap_bool = trapped.astype(bool) if not trapped.empty else pd.Series(False, index=ts)
    _prev = False; _x0 = None
    for i, (t, v) in enumerate(zip(ts, _trap_bool)):
        if v and not _prev:
            _x0 = t
        if _x0 is not None and (not v and _prev):
            fig.add_vrect(x0=_x0, x1=t, fillcolor="rgba(232,93,93,0.12)",
                          layer="below", line_width=0)
            _x0 = None
        _prev = v

    fig.add_trace(go.Scatter(x=ts, y=vre_pot.values, name="VRE Potential",
                             line=dict(color="#5BA85E", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=vre_act.values, name="VRE Dispatch",
                             line=dict(color="#5BA85E", width=1.8),
                             fill="tonexty", fillcolor="rgba(91,168,94,0.18)"), row=1, col=1)
    if not ree_vre.empty and ree_vre.sum() > 0:
        fig.add_trace(go.Scatter(x=ts, y=ree_vre.values, name="VRE Actual (REE)",
                                 line=dict(color="#2E7D32", width=1.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=residual.values, name="Residual Demand",
                             line=dict(color="#374151", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=floor.values, name="Inflexible Floor",
                             line=dict(color="#CC4444", width=1.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts, y=price.values, name="ES Price (€/MWh)",
                             line=dict(color=_AMBER, width=1.5)), row=2, col=1)
    if d.get("omie") is not None:
        omie = d["omie"]
        if not omie.empty:
            fig.add_trace(go.Scatter(x=ts, y=omie.values, name="OMIE Actual",
                                     line=dict(color="#8896A7", width=1.2, dash="dash")),
                          row=2, col=1)

    fig.update_layout(
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Helvetica, Arial, sans-serif", color=_PLOT_TEXT, size=11),
        height=520, hovermode="x unified",
        title=dict(text="VRE Price Formation — Potential, Dispatch & Residual", font=dict(size=12)),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=9), orientation="h", x=0, y=-0.14),
        margin=dict(l=58, r=60, t=36, b=38),
    )
    fig.update_yaxes(title_text="MW",      gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text="€/MWh",   gridcolor=_GRID, row=2, col=1)
    fig.update_xaxes(gridcolor=_GRID, row=1, col=1)
    fig.update_xaxes(gridcolor=_GRID, row=2, col=1)
    return fig


def _make_pfm_residual(d: dict) -> go.Figure:
    """Scatter: residual margin vs ES price, coloured by price-setter carrier."""
    ts     = pd.to_datetime(d.get("timestamps", []))
    margin = d.get("pfm_res_margin", pd.Series(dtype=float))
    price  = d.get("price_es",       pd.Series(dtype=float))
    setter = d.get("setter_es",      pd.Series(dtype=str))

    if ts.empty or margin.empty:
        return go.Figure().update_layout(title="Reload network to compute",
                                         paper_bgcolor=_WHITE, plot_bgcolor=_WHITE)

    fig = go.Figure()
    for c in sorted(setter.dropna().unique()):
        mask = setter == c
        fig.add_trace(go.Scatter(
            x=margin[mask].values, y=price[mask].values,
            mode="markers",
            marker=dict(color=COLORS.get(c, _MUTED), size=4, opacity=0.65),
            name=c,
            hovertemplate=f"<b>{c}</b><br>Margin: %{{x:.0f}} MW<br>Price: %{{y:.1f}} €/MWh<extra></extra>",
        ))
    fig.add_vline(x=0, line_color="#CC4444", line_dash="dash", line_width=1)
    fig.add_annotation(x=0, y=1, xref="x", yref="paper",
                       text="VRE should dominate ←", showarrow=False,
                       font=dict(size=9, color="#CC4444"), xanchor="right")

    fig.update_layout(**_PLOT_BASE)
    fig.update_layout(height=360, hovermode="closest",
                      title=dict(text="Residual Margin vs Price (by setter)", font=dict(size=12)),
                      legend=dict(orientation="h", x=0, y=-0.22, font=dict(size=9)))
    fig.update_xaxes(title_text="Residual Margin MW  (negative → VRE could clear)")
    fig.update_yaxes(title_text="ES Shadow Price  €/MWh")
    return fig


def _make_pfm_actual_compare(d: dict) -> go.Figure:
    """Grouped bar: model vs REE actual mean dispatch for key technologies."""
    ts  = pd.to_datetime(d.get("timestamps", []))
    if ts.empty:
        return go.Figure().update_layout(title="No data loaded",
                                         paper_bgcolor=_WHITE, plot_bgcolor=_WHITE)

    ree  = d.get("ree_actual", pd.DataFrame())
    d_es = d.get("dispatch_es", pd.DataFrame())

    _show = [("solar", "Solar"), ("onwind", "Wind"), ("nuclear", "Nuclear"), ("CCGT", "CCGT")]
    model_means  = {}
    actual_means = {}
    for carrier, label in _show:
        mc = [c for c in d_es.columns if c == carrier or c.startswith(carrier + "_")]
        model_means[label]  = float(d_es[mc].clip(lower=0).sum(axis=1).mean()) if mc else 0.0
        actual_means[label] = float(ree[carrier].mean()) if (isinstance(ree, pd.DataFrame)
                                                              and not ree.empty
                                                              and carrier in ree.columns) else 0.0

    labels = [l for _, l in _show]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[actual_means[l] for l in labels],
                         name="Actual (REE)", marker_color=_MUTED, opacity=0.75))
    fig.add_trace(go.Bar(x=labels, y=[model_means[l] for l in labels],
                         name="Model", marker_color=_TEAL, opacity=0.85))

    fig.update_layout(**_PLOT_BASE)
    fig.update_layout(height=300, barmode="group",
                      title=dict(text="Mean Dispatch: Model vs REE Actual", font=dict(size=12)),
                      legend=dict(orientation="h", x=0, y=-0.22, font=dict(size=9)))
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="Mean MW")
    return fig


def _make_pfm_congestion(d: dict) -> go.Figure:
    """Horizontal bar: congestion frequency during trapped-VRE hours."""
    cong      = d.get("pfm_cong_top", [])
    trapped   = d.get("pfm_trapped_vre", pd.Series(dtype=int))
    n_trapped = int(trapped.sum()) if not trapped.empty else 0

    if not cong:
        fig = go.Figure()
        fig.update_layout(**_PLOT_BASE, height=180,
                          title=dict(text=f"No trapped-VRE hours found  (n_trapped={n_trapped})",
                                     font=dict(size=12)))
        return fig

    lines  = [r["line"] for r in cong]
    counts = [r["count"] for r in cong]
    pct    = [c / max(n_trapped, 1) * 100 for c in counts]
    labels = [f"{l}  ({p:.0f}%)" for l, p in zip(lines, pct)]

    fig = go.Figure(go.Bar(x=counts, y=labels, orientation="h",
                           marker_color=_CORAL,
                           hovertemplate="%{y}<br>%{x} congested hours<extra></extra>"))
    fig.update_layout(**_PLOT_BASE,
                      height=max(200, 60 + 38 * len(cong)),
                      title=dict(text=f"Congested Lines in Trapped-VRE Hours  (n={n_trapped})",
                                 font=dict(size=12)),
                      margin=dict(l=180, r=40, t=36, b=38))
    fig.update_xaxes(title_text="Hours at ≥95% capacity")
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig


# ── Diagnostic supplement charts ──────────────────────────────────────────────

def _make_setter_bar(d: dict) -> go.Figure:
    """Grouped bar: model vs OMIE mean price per setter technology, sorted by |error|."""
    price  = d.get("price_es",  pd.Series(dtype=float))
    omie   = d.get("omie")
    setter = d.get("setter_es")
    if price.empty or omie is None or setter is None or setter.empty:
        return go.Figure(layout=dict(
            annotations=[dict(text="Load OMIE data to view calibration chart",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=280, paper_bgcolor=_WHITE))
    rows = []
    for tech in setter.dropna().unique():
        if str(tech) in ("nan", "None", ""):
            continue
        mask  = setter == tech
        n_hrs = int(mask.sum())
        if n_hrs < 2:
            continue
        m_mean = float(price[mask].mean())
        o_mean = float(omie[mask].mean())
        rows.append({"tech": str(tech), "model": m_mean, "omie": o_mean,
                     "error": m_mean - o_mean, "hrs": n_hrs})
    if not rows:
        return go.Figure(layout=dict(height=280, paper_bgcolor=_WHITE))
    df = pd.DataFrame(rows).sort_values("error", key=abs, ascending=False)
    fig = go.Figure()
    fig.add_bar(name="OMIE actual", x=df["tech"], y=df["omie"],
                marker_color=_MUTED, opacity=0.75,
                hovertemplate="%{x} — OMIE: %{y:.1f} €/MWh<extra></extra>")
    fig.add_bar(name="Model", x=df["tech"], y=df["model"],
                marker_color=[COLORS.get(t, "#888888") for t in df["tech"]],
                hovertemplate="%{x} — Model: %{y:.1f} €/MWh<extra></extra>")
    for _, row in df.iterrows():
        col = "#B2182B" if row["error"] > 8 else "#2166AC" if row["error"] < -5 else _MUTED
        fig.add_annotation(x=row["tech"],
                           y=max(row["model"], row["omie"]) + max(2, abs(row["error"]) * 0.05),
                           text=f"{row['error']:+.0f}",
                           font=dict(size=11, color=col), showarrow=False)
    fig.update_layout(**_PLOT_BASE, barmode="group", height=300)
    fig.update_layout(title=dict(
        text="Setter calibration — model vs OMIE (annotated error, €/MWh)",
        font=dict(size=12, color=_PLOT_TEXT), x=0.01))
    fig.update_yaxes(title_text="€/MWh")
    return fig


def _make_price_ts(d: dict) -> go.Figure:
    """
    LW price (setter-coloured dots) + TW price (teal line) + OMIE actual (black).
    Lower subplot shows hourly LW−TW divergence — spikes reveal internal congestion.
    """
    from plotly.subplots import make_subplots as _msp

    ts     = d.get("timestamps", [])
    lw     = d.get("price_lw",  d.get("price_es", pd.Series(dtype=float)))
    tw     = d.get("price_tw_t", pd.Series(dtype=float))
    omie   = d.get("omie")
    setter = d.get("setter_es")

    if not ts or lw.empty:
        return go.Figure(layout=dict(height=340, paper_bgcolor=_WHITE))

    idx  = pd.to_datetime(ts)
    have_tw   = isinstance(tw, pd.Series) and not tw.empty and len(tw) == len(lw)
    divergence = (lw - tw) if have_tw else pd.Series(dtype=float)
    mean_div   = float(divergence.abs().mean()) if have_tw else 0.0

    fig = _msp(
        rows=2, cols=1,
        row_heights=[0.72, 0.28],
        shared_xaxes=True,
        vertical_spacing=0.04,
    )

    # ── Row 1: LW solid line (coral) ─────────────────────────────────────────
    fig.add_scatter(
        row=1, col=1,
        x=idx, y=lw.values, mode="lines",
        name="LW (load-weighted)",
        line=dict(color=_CORAL, width=1.5),
        hovertemplate="LW %{x|%d %b %H:%M}: %{y:.1f} €/MWh<extra></extra>",
    )

    # ── Row 1: LW setter-coloured dots (technology overlay) ───────────────────
    if setter is not None and not setter.empty:
        for tech in [s for s in setter.unique() if str(s) not in ("nan", "None", "")]:
            mask = (setter == tech).values
            fig.add_scatter(
                row=1, col=1,
                x=idx[mask], y=lw.values[mask], mode="markers",
                name=tech, marker=dict(color=COLORS.get(tech, "#888"), size=3, opacity=0.55),
                hovertemplate=f"{tech} %{{x|%d %b %H:%M}}: %{{y:.1f}} €/MWh<extra></extra>",
                legendgroup="setters", legendgrouptitle_text="Price setter",
            )

    # ── Row 1: TW solid line (teal) ───────────────────────────────────────────
    if have_tw:
        fig.add_scatter(
            row=1, col=1,
            x=idx, y=tw.values, mode="lines",
            name="TW (time-weighted)",
            line=dict(color=_TEAL, width=1.5),
            hovertemplate="TW %{x|%d %b %H:%M}: %{y:.1f} €/MWh<extra></extra>",
        )

    # ── Row 1: OMIE actual (dark line) ───────────────────────────────────────
    if omie is not None:
        fig.add_scatter(
            row=1, col=1,
            x=idx, y=omie.values, mode="lines",
            name="OMIE actual",
            line=dict(color="#1A1A2E", width=1.5, dash="dot"),
            hovertemplate="OMIE %{x|%d %b %H:%M}: %{y:.1f} €/MWh<extra></extra>",
        )

    # ── Row 2: LW − TW divergence (grey fill) ────────────────────────────────
    if have_tw:
        div_vals = divergence.values
        fig.add_scatter(
            row=2, col=1,
            x=idx, y=div_vals, mode="lines",
            name="LW−TW spread",
            line=dict(color=_MUTED, width=0.8),
            fill="tozeroy",
            fillcolor="rgba(136,150,167,0.22)",
            hovertemplate="LW−TW %{x|%d %b %H:%M}: %{y:+.2f} €/MWh<extra></extra>",
        )
        fig.add_hline(y=0, row=2, col=1, line=dict(color=_GRID, width=1))

    # ── Layout ────────────────────────────────────────────────────────────────
    bias_note = f"  ·  mean |LW−TW| = {mean_div:.2f} €/MWh" if have_tw else ""
    # Exclude keys we set explicitly below so there are no duplicate kwargs
    _base = {k: v for k, v in _PLOT_BASE.items()
             if k not in ("xaxis", "yaxis", "hovermode", "legend", "margin")}
    fig.update_layout(
        **_base,
        height=340,
        hovermode="x unified",
        margin=dict(l=58, r=60, t=38, b=30),
        title=dict(
            text=f"LW (coral)  vs  TW (teal)  vs  OMIE (black dotted){bias_note}",
            font=dict(size=11, color=_PLOT_TEXT), x=0.01,
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=10, color=_PLOT_TEXT),
                    orientation="h", x=0, y=-0.14, xanchor="left"),
    )
    fig.update_yaxes(title_text="€/MWh", row=1, col=1,
                     gridcolor=_GRID, gridwidth=0.5)
    fig.update_yaxes(title_text="LW−TW (€)", row=2, col=1,
                     gridcolor=_GRID, gridwidth=0.5, zeroline=True, zerolinecolor=_GRID)
    fig.update_xaxes(gridcolor=_GRID, gridwidth=0.5, row=2, col=1)
    return fig


def _make_headroom_ts(d: dict) -> go.Figure:
    """Upward supply headroom over time with VOLL / diesel / OCGT event markers."""
    ts      = d.get("timestamps", [])
    flex_up = d.get("flex_up_t", pd.Series(dtype=float))
    price   = d.get("price_es",  pd.Series(dtype=float))
    setter  = d.get("setter_es")
    if not ts or flex_up.empty:
        return go.Figure(layout=dict(height=220, paper_bgcolor=_WHITE))
    idx = pd.to_datetime(ts)
    fig = go.Figure()
    fig.add_scatter(x=idx, y=flex_up.values, mode="lines", name="Upward headroom",
                    line=dict(color="#2166AC", width=1.2),
                    fill="tozeroy", fillcolor="rgba(33,102,172,0.09)",
                    hovertemplate="Headroom %{x|%d %b %H:%M}: %{y:.0f} MW<extra></extra>")
    if setter is not None and not setter.empty:
        extreme = setter.isin({"load_shedding", "diesel", "OCGT"})
        if extreme.any():
            fig.add_scatter(x=idx[extreme.values], y=price.values[extreme.values],
                            mode="markers", name="VOLL / diesel / OCGT sets price",
                            marker=dict(color="#B2182B", size=9, symbol="x-open", line_width=2),
                            yaxis="y2",
                            hovertemplate="Event %{x|%d %b %H:%M}: %{y:.1f} €/MWh<extra></extra>")
    fig.update_layout(**_PLOT_BASE, height=250)
    fig.update_layout(
        title=dict(text="Supply headroom (MW, blue) + VOLL/diesel/OCGT price spikes (€/MWh, right)",
                   font=dict(size=12, color=_PLOT_TEXT), x=0.01),
        yaxis2=dict(overlaying="y", side="right", showgrid=False,
                    tickfont=dict(size=9, color="#B2182B"), title_text="Spike price (€/MWh)"),
    )
    fig.update_yaxes(title_text="Headroom (MW)")
    return fig


def _make_hydro_soc(d: dict) -> go.Figure:
    """ES / FR / PT reservoir SoC (GWh) + price in hydro-setting hours.

    SOC already includes inflows via PyPSA's mass balance (SOC[t] = SOC[t-1] + inflow[t] - dispatch[t]).
    Inflow is not plotted separately to avoid implying double-counting.
    """
    from plotly.subplots import make_subplots as _ms
    ts     = d.get("timestamps", [])
    if not ts:
        return go.Figure(layout=dict(
            annotations=[dict(text="Hydro SoC data unavailable — re-solve to populate",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=320, paper_bgcolor=_WHITE))
    idx = pd.to_datetime(ts)

    _countries = [
        ("ES", d.get("hydro_soc_gwh",   pd.Series(dtype=float)), "#4393C3", "rgba(67,147,195,0.12)"),
        ("FR", d.get("fr_soc_gwh",       pd.Series(dtype=float)), "#E67E22", "rgba(230,126,34,0.12)"),
        ("PT", d.get("pt_soc_gwh",       pd.Series(dtype=float)), "#27AE60", "rgba(39,174,96,0.12)"),
    ]
    price  = d.get("price_es", pd.Series(dtype=float))
    setter = d.get("setter_es")

    fig = _ms(rows=4, cols=1, shared_xaxes=True,
              vertical_spacing=0.06,
              subplot_titles=[
                  "ES Hydro SOC (GWh)",
                  "FR Hydro SOC (GWh)",
                  "PT Hydro SOC (GWh)",
                  "Price in hydro-setting hours (€/MWh)",
              ],
              row_heights=[0.27, 0.27, 0.27, 0.19])

    for row_idx, (ctry, soc, line_col, fill_col) in enumerate(_countries, start=1):
        has_soc = isinstance(soc, pd.Series) and not soc.empty and soc.abs().sum() > 0
        if has_soc:
            fig.add_scatter(
                x=idx, y=soc.values, mode="lines",
                name=f"{ctry} SOC",
                line=dict(color=line_col, width=1.8),
                fill="tozeroy", fillcolor=fill_col,
                row=row_idx, col=1,
                hovertemplate=f"<b>{ctry}</b> SOC %{{x|%d %b}}: %{{y:.0f}} GWh<extra></extra>",
            )
        else:
            fig.add_annotation(
                text=f"{ctry}: no reservoir data",
                x=0.5, y=0.5, xref=f"x{row_idx} domain", yref=f"y{row_idx} domain",
                showarrow=False, font=dict(size=10, color=_MUTED),
                row=row_idx, col=1,
            )

    if setter is not None and isinstance(setter, pd.Series) and not setter.empty:
        h_mask = (setter == "hydro").values
        if h_mask.any() and isinstance(price, pd.Series) and not price.empty:
            fig.add_scatter(
                x=idx[h_mask], y=price.values[h_mask],
                mode="markers", name="Price (hydro sets)",
                marker=dict(color="#4393C3", size=4, opacity=0.8),
                row=4, col=1,
                hovertemplate="Hydro-set %{x|%d %b %H:%M}: %{y:.1f} €/MWh<extra></extra>",
            )

    fig.update_layout(
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=_PLOT_TEXT, size=10),
        height=620,
        margin=dict(l=58, r=60, t=44, b=38),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                    borderwidth=1, font=dict(size=9),
                    orientation="h", x=0, y=-0.08),
    )
    fig.update_yaxes(gridcolor=_GRID, gridwidth=0.5, griddash="dash")
    return fig


def _make_ccgt_tier_figure(d: dict) -> go.Figure:
    """Step 3 — CCGT MC by tier: box plots per tier with OMIE mean reference."""
    tier_mc  = d.get("ccgt_tier_mc", {})
    omie     = d.get("omie",         pd.Series(dtype=float))
    mibgas_t = d.get("mibgas_t",     pd.Series(dtype=float))

    if not tier_mc:
        return go.Figure(layout=dict(
            annotations=[dict(text="CCGT tier MC data unavailable — re-solve to populate",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=360, paper_bgcolor=_WHITE))

    tier_colors = {
        "T1": "#1A6496", "T2": "#2E86AB", "T3": "#5BA4CF",
        "T4": "#E07B39", "T5": "#C0392B", "T6": "#922B21",
    }

    fig = go.Figure()
    for tier, series in sorted(tier_mc.items()):
        vals = series.dropna().values if hasattr(series, "dropna") else series
        fig.add_trace(go.Box(
            y=vals, name=tier,
            marker_color=tier_colors.get(tier, "#888"),
            boxpoints="outliers",
            hovertemplate=f"{tier}: %{{y:.1f}} €/MWh<extra></extra>",
        ))

    if not omie.empty:
        omie_mean = omie.mean()
        fig.add_hline(y=omie_mean, line_dash="dash", line_color="#00A896", line_width=1.5,
                      annotation_text=f"OMIE mean {omie_mean:.1f} €",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#00A896"))

    mibgas_valid = mibgas_t.dropna() if hasattr(mibgas_t, "dropna") else pd.Series(dtype=float)
    if len(mibgas_valid) > 0:
        mb_mean = mibgas_valid.mean()
        mb_min  = mibgas_valid.min()
        mb_max  = mibgas_valid.max()
        fig.add_annotation(
            text=f"Implied MIBGAS (from T1 MC):<br>"
                 f"mean {mb_mean:.1f}  min {mb_min:.1f}  max {mb_max:.1f} €/MWh_th",
            xref="paper", yref="paper", x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            showarrow=False, bgcolor="rgba(255,255,255,0.88)",
            bordercolor=_GRID, borderwidth=1,
            font=dict(size=9, color=_PLOT_TEXT),
        )

    fig.update_layout(
        title=dict(text="ES CCGT Marginal Cost by Tier (T1=best η, T6=worst η)",
                   font=dict(size=12)),
        yaxis_title="MC (€/MWh)", xaxis_title="Tier",
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=_PLOT_TEXT, size=10),
        height=380, margin=dict(l=58, r=60, t=44, b=38),
        showlegend=False,
    )
    fig.update_yaxes(gridcolor=_GRID, gridwidth=0.5, griddash="dash")
    return fig


def _make_curtailment_price_scatter(d: dict) -> go.Figure:
    """Step 4 — VRE dispatch vs ES price scatter, colored by price-setter carrier."""
    vre_act  = d.get("vre_actual_es",  pd.Series(dtype=float))
    vre_pot  = d.get("vre_potential_es", pd.Series(dtype=float))
    price    = d.get("price_es",       pd.Series(dtype=float))
    setter   = d.get("setter_es",      pd.Series(dtype=str))
    omie     = d.get("omie",           pd.Series(dtype=float))
    ts       = d.get("timestamps",     [])

    if vre_act.empty or price.empty:
        return go.Figure(layout=dict(
            annotations=[dict(text="VRE / price data unavailable — re-solve to populate",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=380, paper_bgcolor=_WHITE))

    curtail = (vre_pot - vre_act).clip(lower=0) if not vre_pot.empty else pd.Series(0.0, index=vre_act.index)
    idx     = pd.to_datetime(ts) if ts else vre_act.index
    hover   = [f"{t.strftime('%d %b %H:%M')}<br>VRE {v:.0f} MW<br>Price {p:.1f} €/MWh<br>Setter: {s}"
               for t, v, p, s in zip(idx, vre_act.values, price.values,
                                     setter.values if setter is not None else ["?"]*len(price))]

    carrier_color = {
        "CCGT":        "#E07B39",
        "CCGT_flex":   "#C0392B",
        "OCGT":        "#922B21",
        "nuclear":     "#5BA4CF",
        "hydro":       "#4393C3",
        "solar":       "#F4D03F",
        "onwind":      "#27AE60",
        "offwind":     "#1E8449",
        "biomass":     "#8E44AD",
        "VOLL":        "#1A1A1A",
        "ror":         "#2471A3",
    }

    fig = go.Figure()

    # Curtailment indicator (background shading — vertical band is tricky, use color on points)
    curtail_mask = curtail > 100
    gas_setters  = {"CCGT", "CCGT_flex", "OCGT"}

    carriers = setter.unique() if setter is not None and hasattr(setter, "unique") else []
    for carrier in sorted(carriers):
        mask = (setter == carrier)
        if not mask.any():
            continue
        color = carrier_color.get(carrier, "#888888")
        symbol = "circle-open" if curtail_mask[mask].any() else "circle"
        fig.add_trace(go.Scatter(
            x=vre_act[mask].values,
            y=price[mask].values,
            mode="markers",
            name=carrier,
            marker=dict(color=color, size=5, opacity=0.65,
                        symbol=["diamond" if c else "circle" for c in curtail_mask[mask].values]),
            text=[hover[i] for i, m in enumerate(mask.values) if m],
            hoverinfo="text",
        ))

    # OMIE reference line (slope = 0, just horizontal at OMIE mean for context)
    if not omie.empty:
        fig.add_hline(y=omie.mean(), line_dash="dot", line_color="#00A896", line_width=1,
                      annotation_text=f"OMIE mean {omie.mean():.1f} €",
                      annotation_position="top right",
                      annotation_font=dict(size=9, color="#00A896"))

    # Annotations: curtailment rate, gas-sets-during-high-VRE hours
    high_vre_mask  = vre_act > vre_act.quantile(0.75)
    gas_set_highvre = (high_vre_mask & setter.isin(gas_setters)).sum() if setter is not None else 0
    total_highvre   = high_vre_mask.sum()
    curt_hrs        = curtail_mask.sum()

    fig.add_annotation(
        text=(f"Curtailment >100 MW: {curt_hrs} hrs (diamond markers)<br>"
              f"Gas sets price in top-quartile VRE hrs: "
              f"{gas_set_highvre}/{total_highvre} ({100*gas_set_highvre/max(total_highvre,1):.0f}%)"),
        xref="paper", yref="paper", x=0.01, y=0.99,
        xanchor="left", yanchor="top", showarrow=False,
        bgcolor="rgba(255,255,255,0.88)", bordercolor=_GRID, borderwidth=1,
        font=dict(size=9, color=_PLOT_TEXT),
    )

    fig.update_layout(
        title=dict(text="VRE Dispatch vs ES Price — colored by price-setter, ◇ = curtailment >100 MW",
                   font=dict(size=12)),
        xaxis_title="ES VRE dispatch (MW)",
        yaxis_title="ES price (€/MWh)",
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=_PLOT_TEXT, size=10),
        height=420, margin=dict(l=58, r=60, t=44, b=38),
        legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID, borderwidth=1,
                    font=dict(size=9), orientation="v", x=1.01, y=1, xanchor="left"),
    )
    fig.update_xaxes(gridcolor=_GRID, gridwidth=0.5, griddash="dash")
    fig.update_yaxes(gridcolor=_GRID, gridwidth=0.5, griddash="dash")
    return fig


def _make_curtailment_monthly(d: dict) -> go.Figure:
    """Heatmap: curtailed hours per technology × month.

    Cell value = number of hours where curtailment > 10 MW.
    Hover reveals: total curtailed GWh, mean curtailment depth (MW),
    and max curtailment event for that tech×month.
    """
    vre_curt = d.get("vre_tech_curtail_mw", {})
    ts       = d.get("timestamps", [])
    if not vre_curt or not ts:
        return go.Figure(layout=dict(
            annotations=[dict(text="Curtailment data unavailable — re-solve to populate",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=420, paper_bgcolor=_WHITE))

    idx = pd.to_datetime(ts)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    carrier_meta = {
        "solar":       {"label": "Solar",         "y": 0},
        "onwind":      {"label": "Onshore Wind",  "y": 1},
        "offwind-ac":  {"label": "Offshore Wind (AC)",  "y": 2},
        "offwind-dc":  {"label": "Offshore Wind (DC)",  "y": 3},
        "offwind-float": {"label": "Offshore Wind (Float)", "y": 4},
    }

    carriers_present = sorted(k for k in vre_curt if k in carrier_meta)
    if not carriers_present:
        return go.Figure(layout=dict(
            annotations=[dict(text="No VRE curtailment recorded",
                              x=0.5, y=0.5, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=12, color=_MUTED))],
            height=420, paper_bgcolor=_WHITE))

    # Build 2D arrays: [tech, month]
    n_tech = len(carriers_present)
    curt_hours = np.zeros((n_tech, 12), dtype=float)   # hours with >10 MW curtailment
    mean_mw    = np.zeros((n_tech, 12), dtype=float)    # mean curtailment depth (MW)
    total_gwh  = np.zeros((n_tech, 12), dtype=float)    # total curtailed GWh
    max_mw     = np.zeros((n_tech, 12), dtype=float)    # peak curtailment event

    for ti, carrier in enumerate(carriers_present):
        s = pd.Series(vre_curt[carrier], index=idx, dtype=float)
        for mi in range(12):
            month_num = mi + 1
            mask = idx.month == month_num
            vals = s[mask]
            curt = vals[vals > 10.0]  # only count hours with meaningful curtailment
            curt_hours[ti, mi] = len(curt)
            mean_mw[ti, mi]    = curt.mean() if len(curt) > 0 else 0.0
            total_gwh[ti, mi]  = vals.sum() / 1e3
            max_mw[ti, mi]     = curt.max() if len(curt) > 0 else 0.0

    tech_labels = [carrier_meta[c]["label"] for c in carriers_present]

    # Custom hover text
    hover_texts = []
    for ti in range(n_tech):
        row = []
        for mi in range(12):
            h = curt_hours[ti, mi]
            g = total_gwh[ti, mi]
            m = mean_mw[ti, mi]
            x = max_mw[ti, mi]
            if h > 0:
                row.append(
                    f"<b>{tech_labels[ti]}</b><br>"
                    f"{month_labels[mi]}<br>"
                    f"Curtailed hours: <b>{h:.0f}</b><br>"
                    f"Total curtailed: {g:.1f} GWh<br>"
                    f"Mean depth: {m:.0f} MW<br>"
                    f"Peak event: {x:.0f} MW"
                )
            else:
                row.append(
                    f"<b>{tech_labels[ti]}</b><br>"
                    f"{month_labels[mi]}<br>"
                    f"No curtailment"
                )
        hover_texts.append(row)

    colorscale = [
        [0.0,  "#F8F9F9"],   # near-white for zero
        [0.05, "#E8F8F5"],
        [0.15, "#A3E4D7"],
        [0.30, "#48C9B0"],
        [0.50, "#1ABC9C"],
        [0.70, "#17A589"],
        [0.85, "#0E6655"],
        [1.0,  "#0B5345"],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=curt_hours,
        x=month_labels,
        y=tech_labels,
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text="Curtailed<br>hours", side="right"),
            tickfont=dict(size=9),
            thickness=15,
            len=0.85,
        ),
        xgap=2,
        ygap=2,
    ))

    # Annotate each cell with the curtailed-hour count
    annotations = []
    for ti in range(n_tech):
        for mi in range(12):
            h = curt_hours[ti, mi]
            g = total_gwh[ti, mi]
            if h > 0:
                text = f"<b>{h:.0f}</b><br><span style='font-size:9px'>{g:.1f} GWh</span>"
            else:
                text = ""
            annotations.append(dict(
                x=mi, y=ti,
                text=text,
                showarrow=False,
                font=dict(size=10 if h > 0 else 1, color="white" if h > 100 else "#444"),
                xanchor="center", yanchor="middle",
            ))

    fig.update_layout(
        title=dict(text="VRE Curtailment Hours by Technology and Month — Spain",
                   font=dict(size=13)),
        paper_bgcolor=_WHITE, plot_bgcolor=_WHITE,
        font=dict(family="Helvetica Neue, Arial, sans-serif", color=_PLOT_TEXT, size=10),
        height=180 + 40 * n_tech,
        margin=dict(l=10, r=60, t=44, b=50),
        annotations=annotations,
        xaxis=dict(side="top", dtick=1),
        yaxis=dict(dtick=1),
    )

    return fig


# ── Monthly Analysis tab figures ─────────────────────────────────────────────

def _monthly_agg(series, func="mean"):
    """Group a time-indexed Series by calendar month. Returns a Period-indexed Series."""
    if series is None or (hasattr(series, "empty") and series.empty):
        return pd.Series(dtype=float)
    s = series if isinstance(series, pd.Series) else pd.Series(series)
    if not hasattr(s.index, "to_period"):
        return pd.Series(dtype=float)
    try:
        idx = s.index.tz_localize(None) if getattr(s.index, "tz", None) else s.index
        grp = s.groupby(idx.to_period("M"))
        return getattr(grp, func)().dropna()
    except Exception:
        return pd.Series(dtype=float)


def _monthly_sum_gw_to_gwh(series):
    """Sum a GW time-series by month to produce GWh/month (1 snapshot = 1 h)."""
    return _monthly_agg(series, func="sum")


def _make_monthly_stack(d: dict) -> go.Figure:
    """Stacked monthly supply bars (ES carriers + imports) with load line."""
    ts = d.get("timestamps", [])
    if not ts:
        fig = go.Figure()
        fig.update_layout(**{**_PLOT_BASE, "height": 390,
                             "title": dict(text="Load a solved network first")})
        return fig

    dispatch = d.get("dispatch_es", pd.DataFrame())
    es_load  = d.get("es_load",  pd.Series(dtype=float))
    fr_net   = d.get("fr_net",   pd.Series(dtype=float))
    pt_net   = d.get("pt_net",   pd.Series(dtype=float))

    carrier_groups = [
        ("Nuclear",    ["nuclear"],                                    COLORS.get("nuclear", "#4e79a7")),
        ("VRE",        ["onwind", "offwind", "solar"],                 COLORS.get("solar",   "#f28e2b")),
        ("Hydro",      ["hydro", "ror"],                               COLORS.get("hydro",   "#59a14f")),
        ("PHS",        ["PHS", "PHS_new"],                             COLORS.get("PHS",     "#76b7b2")),
        ("CCGT",       ["CCGT", "CCGT_flex", "CCGT_must_run"],         COLORS.get("CCGT",    "#e15759")),
        ("OCGT",       ["OCGT", "OCGT_pk"],                            COLORS.get("OCGT",    "#ff9da7")),
        ("Coal/other", ["coal", "lignite", "oil"],                     COLORS.get("coal",    "#9c755f")),
    ]

    fig = go.Figure()
    for label, carriers, color in carrier_groups:
        if not isinstance(dispatch, pd.DataFrame) or dispatch.empty:
            continue
        cols = [c for c in carriers if c in dispatch.columns]
        if not cols:
            continue
        agg = _monthly_agg(dispatch[cols].sum(axis=1))
        if agg.empty or agg.max() < 1:
            continue
        fig.add_trace(go.Bar(
            name=label, x=[str(p) for p in agg.index], y=agg.values,
            marker_color=color, opacity=0.88,
        ))

    for label, flow, color in [
        ("FR import", fr_net.clip(lower=0) if isinstance(fr_net, pd.Series) else pd.Series(dtype=float), "#b07aa1"),
        ("PT import", pt_net.clip(lower=0) if isinstance(pt_net, pd.Series) else pd.Series(dtype=float), "#d4a6c8"),
    ]:
        agg = _monthly_agg(flow)
        if not agg.empty and agg.max() > 1:
            fig.add_trace(go.Bar(
                name=label, x=[str(p) for p in agg.index], y=agg.values,
                marker_color=color, opacity=0.88,
            ))

    load_agg = _monthly_agg(es_load)
    if not load_agg.empty:
        fig.add_trace(go.Scatter(
            name="ES Load", x=[str(p) for p in load_agg.index], y=load_agg.values,
            mode="lines+markers",
            line=dict(color="black", width=2, dash="dot"),
            marker=dict(size=6, color="black"),
        ))

    layout = {**_PLOT_BASE}
    layout.update(
        height=390, barmode="stack",
        title=dict(text="Monthly Supply Stack vs ES Load  (mean MW)", font=dict(size=12)),
        yaxis=dict(_PLOT_BASE["yaxis"],
                   title_text="Mean MW",
                   title_font=dict(size=10, color=_MUTED)),
        xaxis=dict(_PLOT_BASE["xaxis"], title_text=""),
        legend=dict(orientation="h", x=0, y=-0.22, xanchor="left",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=_GRID,
                    borderwidth=1, font=dict(size=9)),
        margin=dict(l=58, r=20, t=42, b=100),
    )
    fig.update_layout(**layout)
    return fig


def _make_monthly_price(d: dict) -> go.Figure:
    """Grouped monthly bar chart: model vs OMIE price with error annotations."""
    price = d.get("price_es", pd.Series(dtype=float))
    omie  = d.get("omie")

    model_m = _monthly_agg(price)
    omie_m  = _monthly_agg(omie) if omie is not None else pd.Series(dtype=float)

    # Align on common months
    all_months = sorted(set(list(model_m.index) + list(omie_m.index)), key=str)
    months_str = [str(p) for p in all_months]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Model", x=months_str,
        y=[float(model_m.get(p, float("nan"))) for p in all_months],
        marker_color=_CORAL, opacity=0.85, width=0.35, offset=-0.18,
    ))
    if not omie_m.empty:
        fig.add_trace(go.Bar(
            name="OMIE", x=months_str,
            y=[float(omie_m.get(p, float("nan"))) for p in all_months],
            marker_color="#4e79a7", opacity=0.85, width=0.35, offset=0.18,
        ))
        for p in all_months:
            mv = float(model_m.get(p, float("nan")))
            ov = float(omie_m.get(p,  float("nan")))
            if not (pd.isna(mv) or pd.isna(ov)):
                err = mv - ov
                col = _CORAL if abs(err) > 15 else (_AMBER if abs(err) > 8 else _MUTED)
                fig.add_annotation(
                    x=str(p), y=max(mv, ov) + 2.5,
                    text=f"{err:+.0f}", showarrow=False,
                    font=dict(size=9, color=col), yanchor="bottom",
                )

    layout = {**_PLOT_BASE}
    layout.update(
        height=280, barmode="overlay",
        title=dict(text="Monthly Mean Price — Model vs OMIE  (€/MWh)", font=dict(size=12)),
        yaxis=dict(_PLOT_BASE["yaxis"],
                   title_text="Mean price (€/MWh)",
                   title_font=dict(size=10, color=_MUTED),
                   rangemode="tozero"),
        xaxis=dict(_PLOT_BASE["xaxis"], title_text=""),
        margin=dict(l=58, r=20, t=42, b=38),
    )
    fig.update_layout(**layout)
    return fig


def _make_fr_pt_inflow(d: dict) -> go.Figure:
    """2-row subplot: FR and PT monthly hydro inflow vs dispatch (GWh/month)."""
    fr_infl = d.get("fr_infl_gwh",    pd.Series(dtype=float))
    pt_infl = d.get("pt_infl_gwh",    pd.Series(dtype=float))
    fr_disp = d.get("fr_su_hydro_gw", pd.Series(dtype=float))
    pt_disp = d.get("pt_su_hydro_gw", pd.Series(dtype=float))

    fr_infl_m  = _monthly_sum_gw_to_gwh(fr_infl)
    pt_infl_m  = _monthly_sum_gw_to_gwh(pt_infl)
    fr_disp_m  = _monthly_sum_gw_to_gwh(fr_disp)
    pt_disp_m  = _monthly_sum_gw_to_gwh(pt_disp)

    all_months = sorted(
        set(list(fr_infl_m.index) + list(pt_infl_m.index)
            + list(fr_disp_m.index) + list(pt_disp_m.index)),
        key=str,
    )
    months_str = [str(p) for p in all_months]

    def _vals(agg):
        return [float(agg.get(p, 0.0)) for p in all_months]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["France — Hydro Inflow vs Dispatch (GWh/month)",
                        "Portugal — Hydro Inflow vs Dispatch (GWh/month)"],
        vertical_spacing=0.14,
    )
    fig.add_trace(go.Bar(name="FR Inflow",    x=months_str, y=_vals(fr_infl_m),
                         marker_color="#4e79a7", opacity=0.85, offsetgroup=0), row=1, col=1)
    fig.add_trace(go.Bar(name="FR Dispatch",  x=months_str, y=_vals(fr_disp_m),
                         marker_color="#aec7e8", opacity=0.85, offsetgroup=1), row=1, col=1)
    fig.add_trace(go.Bar(name="PT Inflow",    x=months_str, y=_vals(pt_infl_m),
                         marker_color="#59a14f", opacity=0.85, offsetgroup=0), row=2, col=1)
    fig.add_trace(go.Bar(name="PT Dispatch",  x=months_str, y=_vals(pt_disp_m),
                         marker_color="#b5cf8e", opacity=0.85, offsetgroup=1), row=2, col=1)

    layout = {**_PLOT_BASE}
    layout.update(
        height=520, barmode="group",
        title=dict(text="FR & PT Hydro — Monthly Inflow vs Dispatch", font=dict(size=12)),
        margin=dict(l=58, r=20, t=42, b=60),
    )
    fig.update_layout(**layout)
    fig.update_yaxes(title_text="GWh / month", title_font=dict(size=10, color=_MUTED))
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────

_SIDEBAR_EXPANDED = {
    "width": "218px", "minWidth": "218px",
    "backgroundColor": _SLATE,
    "padding": "12px 11px",
    "overflowY": "auto",
    "overflowX": "hidden",
    "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
    "boxSizing": "border-box",
    "transition": "width 0.18s ease",
    "flexShrink": "0",
}
_SIDEBAR_COLLAPSED = {
    **_SIDEBAR_EXPANDED,
    "width": "38px", "minWidth": "38px",
    "padding": "12px 4px",
    "overflowY": "hidden",
}


def _slider_block(id_, label, min_val, max_val, default, step, marks):
    return html.Div([
        html.Div(label,
            style={"fontSize": "10px", "color": "#A8B5C0",
                   "marginTop": "10px", "marginBottom": "2px"}),
        dcc.Slider(id=id_, min=min_val, max=max_val, value=default,
                   step=step, marks=marks, updatemode="mouseup",
                   tooltip={"placement": "bottom", "always_visible": False}),
    ])


def _build_layout() -> html.Div:
    net_opts   = list_solved_networks()
    default_nc = net_opts[0]["value"] if net_opts else None
    cfg        = MODEL_CONFIG

    # Sidebar content (hidden when collapsed)
    sidebar_content = html.Div(id="sidebar-content", children=[
        html.Div(style={"display": "flex", "alignItems": "center",
                         "justifyContent": "space-between", "marginBottom": "5px"},
                 children=[
            html.Div("NETWORK", style={"fontSize": "10px", "color": "#6a7a8a",
                                        "letterSpacing": "1px"}),
            html.Button("↻", id="btn-refresh-nets", n_clicks=0,
                        title="Refresh network list (picks up files created after dashboard start)",
                        style={"fontSize": "12px", "padding": "1px 7px",
                               "backgroundColor": "transparent", "color": "#6a7a8a",
                               "border": f"1px solid {_GRID_SB}", "borderRadius": "3px",
                               "cursor": "pointer"}),
        ]),
        dcc.Dropdown(id="net-dd", options=net_opts, value=default_nc,
                     clearable=False,
                     style={"fontSize": "11px", "color": _SLATE,
                            "backgroundColor": _WHITE}),

        html.Hr(style={"borderColor": _GRID_SB, "margin": "12px 0"}),
        html.Div("DISPLAY", style={"fontSize": "10px", "color": "#6a7a8a",
                                    "letterSpacing": "1px"}),
        html.Div("Price construction method",
                 style={"fontSize": "10px", "color": _MUTED, "marginBottom": "3px",
                        "marginTop": "8px"}),
        dcc.Dropdown(
            id="price-method-dd",
            options=[
                {"label": "Load-Weighted (current)", "value": "lw"},
                {"label": "Time-Weighted Average",   "value": "tw"},
            ],
            value="lw",
            clearable=False,
            style={"fontSize": "11px", "color": _SLATE, "backgroundColor": _WHITE},
        ),

        html.Hr(style={"borderColor": _GRID_SB, "margin": "12px 0"}),
        html.Div("PARAMETERS", style={"fontSize": "10px", "color": "#6a7a8a",
                                       "letterSpacing": "1px"}),

        _slider_block("s-co2",   "CO₂ price (€/t)",       20,   120,
                      cfg["co2_price"], 5,
                      _mk(20, 65, 120)),
        _slider_block("s-ic",    "Interconnector factor",  0.05, 1.0,
                      cfg["borders"]["ic_factor"], 0.05,
                      _mk(0.05, 0.25, 0.5, 1.0)),
        _slider_block("s-fnuc",  "FR nuclear p_min",       0.10, 0.60,
                      cfg["nuclear"]["per_country"]["FR"]["p_min_pu"], 0.05,
                      _mk(0.10, 0.25, 0.40, 0.60)),
        _slider_block("s-fhyd",  "FR hydro p_max",         0.20, 1.0,
                      cfg["hydro"]["per_country"]["FR"].get("p_max_pu", 0.45), 0.05,
                      _mk(0.20, 0.45, 0.70, 1.0)),
        _slider_block("s-mr",    "CCGT must-run (MW)",     0,    5000,
                      cfg["ccgt_must_run"]["target_mw"], 200,
                      _mk(0, 2000, 5000)),
        _slider_block("s-phs",   "PHS p_max_pu",           0.10, 1.0,
                      cfg["phs"]["p_max_pu"], 0.05,
                      _mk(0.10, 0.50, 1.0)),
        _slider_block("s-trans", "Trans factor",           0.30, 1.0,
                      cfg["transmission"]["trans_factor"], 0.05,
                      _mk(0.30, 0.70, 1.0)),
        _slider_block("s-days",  "Window (days)",          7,    60,
                      cfg["validation"]["n_days"], 1,
                      _mk(7, 14, 30, 60)),

        dcc.Checklist(id="mip-toggle",
                      options=[{"label": " MIP / Unit Commitment", "value": "mip"}],
                      value=["mip"] if cfg["mip"]["enabled"] else [],
                      labelStyle={"color": "#A8B5C0", "fontSize": "11px"},
                      style={"marginTop": "10px"}),

        html.Div(id="change-badge",
                 style={"fontSize": "10px", "color": _AMBER, "marginTop": "8px",
                        "minHeight": "16px"}),

        html.Button("⟳  Re-solve", id="solve-btn", n_clicks=0,
            style={"width": "100%", "marginTop": "10px", "padding": "9px 0",
                   "backgroundColor": _CORAL, "color": "white",
                   "border": "none", "borderRadius": "4px",
                   "fontSize": "13px", "fontWeight": "bold",
                   "cursor": "pointer",
                   "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif"}),

        html.Div(id="solve-status",
                 style={"fontSize": "11px", "color": _TEAL,
                        "marginTop": "8px", "minHeight": "18px"}),
    ])

    sidebar = html.Div(id="sidebar", style=_SIDEBAR_EXPANDED, children=[
        html.Button("◀", id="sidebar-toggle", n_clicks=0,
            title="Collapse sidebar",
            style={"width": "100%", "background": "none", "border": "none",
                   "color": "#6a7a8a", "fontSize": "14px", "cursor": "pointer",
                   "textAlign": "right", "padding": "0 0 8px 0",
                   "fontFamily": "monospace"}),
        sidebar_content,
    ])

    _tab_style = {
        "padding": "6px 16px",
        "fontSize": "11px",
        "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
        "color": _MUTED,
        "backgroundColor": "#F5F6F8",
        "border": "none",
        "borderBottom": f"2px solid {_GRID}",
    }
    _tab_selected = {
        **_tab_style,
        "color": _SLATE,
        "fontWeight": "600",
        "borderBottom": f"2px solid {_CORAL}",
        "backgroundColor": _WHITE,
    }

    main = html.Div([
        html.Div([
            html.Div("Time window", style={"fontSize": "10px", "color": _MUTED,
                                            "marginBottom": "2px"}),
            dcc.RangeSlider(id="range-sl", min=0, max=100, value=[0, 100],
                            step=1,
                            marks={0: {"label": "start", "style": {"color": _MUTED}},
                                   100: {"label": "end",  "style": {"color": _MUTED}}},
                            updatemode="mouseup",
                            tooltip={"placement": "bottom", "always_visible": False}),
        ], style={"marginBottom": "4px", "paddingRight": "8px"}),

        dcc.Tabs(id="main-tabs", value="tab-dispatch",
                 style={"marginBottom": "0px"},
                 children=[
            dcc.Tab(label="Model Dispatch", value="tab-dispatch",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[
                        html.Div(id="summary-cards"),
                        dcc.Graph(id="g-dispatch",
                                  config={"displayModeBar": True, "scrollZoom": True,
                                          "modeBarButtonsToRemove": ["zoom2d", "autoScale2d"],
                                          "modeBarButtonsToAdd":    ["pan2d"]}),
                        dcc.Graph(id="g-price-error",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-flow-fr",
                                  config={"displayModeBar": False, "scrollZoom": True}),
                        dcc.Graph(id="g-flow-pt",
                                  config={"displayModeBar": False, "scrollZoom": True}),
                    ]),
            dcc.Tab(label="REE Actual (ENTSO-E)", value="tab-ree",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[
                        dcc.Graph(id="g-ree",
                                  config={"displayModeBar": False, "scrollZoom": True}),
                        dcc.Graph(id="g-dispatch-comp",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-fr-pt-weekly",
                                  config={"displayModeBar": False}),
                    ]),
            dcc.Tab(label="Price Duration Curve", value="tab-pdc",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[
                        dcc.Graph(id="g-pdc",
                                  config={"displayModeBar": False, "scrollZoom": False}),
                    ]),
            dcc.Tab(label="Price Calibration", value="tab-calib",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[
                        html.Div([
                            html.Div("Price error by marginal technology — model vs OMIE actual",
                                style={"fontSize": "12px", "fontWeight": "600",
                                       "color": _PLOT_TEXT, "marginBottom": "10px",
                                       "marginTop": "8px"}),
                            html.Div(id="calib-table"),
                            dcc.Loading(type="circle", color=_CORAL, children=[
                                dcc.Graph(id="g-setter-bar",
                                          config={"displayModeBar": False}),
                            ]),
                            dcc.Loading(type="circle", color=_CORAL, children=[
                                dcc.Graph(id="g-price-ts",
                                          config={"displayModeBar": True, "scrollZoom": True}),
                            ]),
                            dcc.Loading(type="circle", color=_CORAL, children=[
                                dcc.Graph(id="g-headroom-ts",
                                          config={"displayModeBar": True, "scrollZoom": True}),
                            ]),
                        ], style={"padding": "4px 2px"}),
                    ]),

            dcc.Tab(label="Capacity Mix", value="tab-cap",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[
                        dcc.Graph(id="g-cap",
                                  config={"displayModeBar": False}),
                    ]),

            dcc.Tab(label="Network Map", value="tab-map",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([
                        # Hour selector
                        html.Div([
                            html.Div("Snapshot", style={"fontSize": "10px", "color": _MUTED,
                                                         "marginBottom": "2px"}),
                            dcc.Slider(
                                id="map-hour-sl", min=0, max=100, value=0, step=1,
                                marks={0: {"label": "—", "style": {"color": _MUTED, "fontSize": "9px"}}},
                                updatemode="mouseup",
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ], style={"marginBottom": "4px", "paddingRight": "8px"}),
                        dcc.Graph(id="g-map",
                                  config={"displayModeBar": True, "scrollZoom": False}),

                        # ── Static map legend ─────────────────────────────────
                        html.Div([
                            # Price colour scale
                            html.Div([
                                html.Span("Bus colour  =  electricity price",
                                          style={"fontWeight": "600", "marginRight": "10px",
                                                 "fontSize": "11px"}),
                                html.Div(style={
                                    "background": (
                                        "linear-gradient(to right,"
                                        " #2166AC, #92C5DE, #FFFFBF, #FDAE61,"
                                        f" {_CORAL}, #7B0000)"
                                    ),
                                    "height": "14px", "width": "200px",
                                    "borderRadius": "3px", "display": "inline-block",
                                    "verticalAlign": "middle", "margin": "0 8px",
                                    "border": f"1px solid {_GRID}",
                                }),
                                html.Span("Low →", style={"color": "#2166AC", "fontSize": "11px",
                                                           "marginRight": "4px"}),
                                html.Span("→ High  (€/MWh, adjusts per snapshot)",
                                          style={"color": "#7B0000", "fontSize": "11px"}),
                            ], style={"display": "flex", "alignItems": "center",
                                      "marginBottom": "7px", "flexWrap": "wrap"}),

                            # Transmission loading bands
                            html.Div([
                                html.Span("Line loading:",
                                          style={"fontWeight": "600", "marginRight": "8px",
                                                 "fontSize": "11px"}),
                                *[html.Span(f" {label} ", style={
                                    "backgroundColor": col,
                                    "color": "white" if bi >= 2 else "#333",
                                    "fontSize": "10px",
                                    "padding": "2px 8px",
                                    "borderRadius": "3px",
                                    "marginRight": "4px",
                                    "fontWeight": "600",
                                }) for bi, (_, _, col, label) in enumerate(_LINE_BINS)],
                            ], style={"display": "flex", "alignItems": "center",
                                      "marginBottom": "7px", "flexWrap": "wrap"}),

                            # Interconnector arrows
                            html.Div([
                                html.Span("Interconnectors:",
                                          style={"fontWeight": "600", "marginRight": "8px",
                                                 "fontSize": "11px"}),
                                html.Span("━━━ FR/PT → ES  (import)",
                                          style={"color": "#457B9D", "fontSize": "11px",
                                                 "marginRight": "16px"}),
                                html.Span("━━━ ES → FR/PT  (export)",
                                          style={"color": _CORAL, "fontSize": "11px",
                                                 "marginRight": "12px"}),
                                html.Span("(line width ∝ flow MW)",
                                          style={"color": _MUTED, "fontSize": "10px",
                                                 "fontStyle": "italic"}),
                            ], style={"display": "flex", "alignItems": "center",
                                      "flexWrap": "wrap"}),
                        ], style={
                            "backgroundColor": "#F8F9FA",
                            "border": f"1px solid {_GRID}",
                            "borderRadius": "5px",
                            "padding": "10px 14px",
                            "marginTop": "6px",
                            "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
                            "color": _PLOT_TEXT,
                        }),

                        # ── Dynamic hover detail panel ────────────────────────
                        html.Div(id="map-hover-info",
                                 children=[
                                     html.Span(
                                         "Hover over a bus node to see price and generation breakdown.",
                                         style={"color": _MUTED, "fontSize": "11px",
                                                "fontStyle": "italic"},
                                     ),
                                 ],
                                 style={
                                     "backgroundColor": _WHITE,
                                     "border": f"1px solid {_GRID}",
                                     "borderRadius": "5px",
                                     "padding": "10px 14px",
                                     "marginTop": "6px",
                                     "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
                                     "minHeight": "64px",
                                 }),
                    ], style={"padding": "4px 2px"})]),

            dcc.Tab(label="FR/PT Import Analysis", value="tab-fr",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([
                        html.Div(id="fr-overnight-stats",
                                 style={"padding": "8px 4px 4px 4px",
                                        "backgroundColor": "#F8F9FA",
                                        "borderRadius": "4px",
                                        "border": f"1px solid {_GRID}",
                                        "marginBottom": "8px"}),
                        dcc.Graph(id="g-ic-tech",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-gen-breakdown",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-fr-price-drivers",
                                  config={"displayModeBar": True, "scrollZoom": True}),
                        dcc.Graph(id="g-fr-profile",
                                  config={"displayModeBar": True, "scrollZoom": True}),
                        dcc.Graph(id="g-fr-tech-scatter",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-fr-heatmap",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-fr-scatter",
                                  config={"displayModeBar": False}),
                        # ── Portugal section ───────────────────────────────────
                        html.Div("Portugal — Actual vs Model Import Analysis",
                                 style={"fontSize": "12px", "fontWeight": "700",
                                        "color": _PLOT_TEXT, "marginTop": "18px",
                                        "marginBottom": "6px",
                                        "borderTop": f"2px solid {_GRID}",
                                        "paddingTop": "10px"}),
                        dcc.Graph(id="g-pt-scatter",
                                  config={"displayModeBar": False}),
                    ], style={"padding": "4px 2px"})]),
            dcc.Tab(label="Price Formation", value="tab-pfm",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([
                        html.Div("VRE Bottleneck Analysis — when should renewables clear the market but gas does instead?",
                                 style={"fontSize": "11px", "color": _MUTED,
                                        "padding": "6px 4px 8px 4px"}),
                        dcc.Graph(id="g-pfm-ts",
                                  config={"displayModeBar": True, "scrollZoom": True}),
                        dcc.Graph(id="g-pfm-residual",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-pfm-actual",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-pfm-cong",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-hydro-soc",
                                  config={"displayModeBar": True, "scrollZoom": True}),
                        dcc.Graph(id="g-ccgt-tiers",
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="g-curtail-scatter",
                                  config={"displayModeBar": False}),
                    ], style={"padding": "4px 2px"})]),
            dcc.Tab(label="Curtailment", value="tab-curtail",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([
                        html.Div("VRE Curtailment — monthly breakdown by technology",
                                 style={"fontSize": "11px", "color": _MUTED,
                                        "padding": "6px 4px 8px 4px"}),
                        dcc.Graph(id="g-curtail-monthly",
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={"padding": "4px 2px"})]),
            dcc.Tab(label="Monthly Analysis", value="tab-monthly",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([
                        html.Div(
                            "Monthly supply stack, price comparison, and FR/PT hydro inflow "
                            "— full loaded period, independent of the time slider.",
                            style={"fontSize": "11px", "color": _MUTED,
                                   "padding": "6px 4px 8px 4px"},
                        ),
                        dcc.Graph(id="g-monthly-stack",
                                  config={"displayModeBar": True, "scrollZoom": False}),
                        dcc.Graph(id="g-monthly-price",
                                  config={"displayModeBar": True, "scrollZoom": False}),
                        dcc.Graph(id="g-fr-pt-inflow",
                                  config={"displayModeBar": True, "scrollZoom": False}),
                    ], style={"padding": "4px 2px"})]),
            dcc.Tab(label="Diagnostic Export", value="tab-diag",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([

                        # Action bar
                        html.Div([
                            html.Button("⬇ Download hourly CSV", id="btn-csv",
                                n_clicks=0,
                                style={"padding": "7px 16px", "backgroundColor": _TEAL,
                                       "color": "white", "border": "none",
                                       "borderRadius": "4px", "fontSize": "12px",
                                       "fontWeight": "600", "cursor": "pointer",
                                       "marginRight": "10px",
                                       "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif"}),
                            html.Button("⬇ Download AI prompt (.txt)", id="btn-ai-txt",
                                n_clicks=0,
                                style={"padding": "7px 16px", "backgroundColor": _AMBER,
                                       "color": "white", "border": "none",
                                       "borderRadius": "4px", "fontSize": "12px",
                                       "fontWeight": "600", "cursor": "pointer",
                                       "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif"}),
                        ], style={"marginBottom": "12px", "marginTop": "8px"}),

                        # AI summary preview
                        html.Div("AI Diagnostic Summary — copy & paste into Claude",
                            style={"fontSize": "11px", "fontWeight": "600",
                                   "color": _MUTED, "marginBottom": "6px"}),
                        html.Pre(id="ai-summary",
                            style={"backgroundColor": "#F8F9FA",
                                   "border": f"1px solid {_GRID}",
                                   "borderRadius": "4px",
                                   "padding": "12px", "fontSize": "10px",
                                   "fontFamily": "monospace",
                                   "color": _PLOT_TEXT,
                                   "overflowY": "auto", "maxHeight": "420px",
                                   "whiteSpace": "pre-wrap"}),

                        # Hourly data preview table (first 48 rows)
                        html.Div("Hourly data preview (first 48 rows) — full dataset via CSV",
                            style={"fontSize": "11px", "fontWeight": "600",
                                   "color": _MUTED, "marginTop": "14px",
                                   "marginBottom": "6px"}),
                        html.Div(id="diag-preview"),

                    ], style={"padding": "4px 2px"})]),

            dcc.Tab(label="Spatial Maps", value="tab-spatial",
                    style=_tab_style, selected_style=_tab_selected,
                    children=[html.Div([
                        html.Div(
                            "Annual mean demand and installed capacity per node "
                            "— bubble size and colour scale with value.",
                            style={"fontSize": "11px", "color": _MUTED,
                                   "padding": "6px 4px 8px 4px"},
                        ),
                        html.Div([
                            html.Div("Load Map", style={"fontSize": "12px", "fontWeight": "600",
                                                         "color": _PLOT_TEXT, "marginBottom": "4px"}),
                            dcc.Graph(id="g-load-map",
                                      config={"displayModeBar": True, "scrollZoom": False}),
                        ], style={"marginBottom": "16px"}),
                        html.Div([
                            html.Div("Capacity Map", style={"fontSize": "12px", "fontWeight": "600",
                                                             "color": _PLOT_TEXT, "marginBottom": "4px"}),
                            dcc.Graph(id="g-cap-map",
                                      config={"displayModeBar": True, "scrollZoom": False}),
                        ]),
                    ], style={"padding": "4px 2px"})]),
        ]),
    ], style={
        "flex": "1",
        "backgroundColor": "#F5F6F8",
        "padding": "8px 12px",
        "overflowY": "auto",
        "minWidth": "0",
    })

    detail = html.Div([
        html.Div("HOUR DETAIL", style={"fontSize": "10px", "color": "#7a8a9a",
                                        "letterSpacing": "1px", "marginBottom": "10px"}),
        html.Div(id="hour-detail", children=[
            html.P("Hover over the dispatch chart to see hour detail.",
                   style={"fontSize": "11px", "color": _MUTED}),
        ]),
    ], style={
        "width": "220px", "minWidth": "220px",
        "backgroundColor": _WHITE,
        "padding": "14px 12px",
        "borderLeft": f"1px solid {_GRID}",
        "overflowY": "auto",
        "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
        "boxSizing": "border-box",
    })

    return html.Div([
        dcc.Store(id="store-data"),
        dcc.Interval(id="poll",        interval=2500,  n_intervals=0, disabled=True),
        dcc.Interval(id="net-refresh", interval=15000, n_intervals=0),
        dcc.Download(id="diag-download"),

        html.Div([
            html.Span("PyPSA-Spain  ·  Energy Market Dashboard",
                style={"fontSize": "17px", "fontWeight": "bold", "color": _SLATE}),
            html.Span("  50-node Spain + FR / PT  ·  Gurobi LP/MIP",
                style={"fontSize": "11px", "color": "#7a8a9a"}),
            html.Span("  v1.3",
                style={"fontSize": "10px", "color": "#aab5bf",
                       "marginLeft": "12px", "fontFamily": "monospace"}),
        ], style={
            "backgroundColor": _WHITE,
            "padding": "10px 18px",
            "borderBottom": f"2px solid {_CORAL}",
            "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
        }),

        html.Div([sidebar, main, detail],
                 style={"display": "flex",
                        "height": "calc(100vh - 50px)",
                        "overflow": "hidden"}),
    ])


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="PyPSA-Spain Dashboard",
                suppress_callback_exceptions=True)
app.layout = _build_layout()


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("store-data", "data"),
    Output("range-sl",   "max"),
    Output("range-sl",   "value"),
    Output("range-sl",   "marks"),
    Input("net-dd", "value"),
    prevent_initial_call=False,
)
def cb_load_network(nc_path):
    if not nc_path:
        raise PreventUpdate
    log.info("Loading %s…", Path(nc_path).name)
    data = load_and_extract(nc_path)
    n    = len(data["timestamps"])
    half = min(84, n // 2)
    mid  = n // 2
    i0, i1 = max(0, mid - half), min(n, mid + half)
    step  = max(1, n // 8)
    marks = {i: {"label": data["timestamps"][i][5:10],
                 "style": {"color": _MUTED, "fontSize": "10px"}}
             for i in range(0, n, step)}
    return data, n - 1, [i0, i1], marks


@app.callback(
    Output("g-dispatch",      "figure"),
    Output("g-ree",           "figure"),
    Output("g-dispatch-comp", "figure"),
    Output("g-fr-pt-weekly",  "figure"),
    Output("g-pdc",           "figure"),
    Output("g-flow-fr",       "figure"),
    Output("g-flow-pt",       "figure"),
    Output("g-price-error",   "figure"),
    Output("summary-cards",   "children"),
    Input("store-data",       "data"),
    Input("range-sl",         "value"),
    prevent_initial_call=True,
)
def cb_update_figures(data, rng):
    if not data:
        raise PreventUpdate
    d  = deserialise(data)
    i0 = int(rng[0]) if rng else 0
    i1 = int(rng[1]) + 1 if rng else len(d["timestamps"])
    return (make_dispatch_figure(d, i0, i1),
            make_ree_figure(d, i0, i1),
            make_total_dispatch_bar(d, i0, i1),
            make_fr_pt_weekly_dispatch(d, i0, i1),
            make_pdc_figure(d, i0, i1),
            make_flow_fr_figure(d, i0, i1),
            make_flow_pt_figure(d, i0, i1),
            make_price_error_figure(d, i0, i1),
            _make_summary_cards(d))


@app.callback(
    Output("hour-detail", "children"),
    Input("g-dispatch",   "hoverData"),
    State("store-data",   "data"),
    prevent_initial_call=True,
)
def cb_hover_detail(hover_data, data):
    if not hover_data or not data:
        raise PreventUpdate
    pts = hover_data.get("points", [{}])
    x   = str(pts[0].get("x", "")) if pts else ""
    return make_hour_detail(deserialise(data), x)


@app.callback(
    Output("poll",         "disabled"),
    Output("solve-status", "children"),
    Output("solve-btn",    "disabled"),
    Input("solve-btn",     "n_clicks"),
    State("s-co2",    "value"),
    State("s-ic",     "value"),
    State("s-fnuc",   "value"),
    State("s-fhyd",   "value"),
    State("s-mr",     "value"),
    State("s-phs",    "value"),
    State("s-trans",  "value"),
    State("s-days",   "value"),
    State("mip-toggle", "value"),
    prevent_initial_call=True,
)
def cb_trigger_solve(n_clicks, co2, ic, fnuc, fhyd, mr, phs, trans, days, mip_vals):
    status, _, _ = poll_solve_result()
    if status == "running":
        return False, "⚙ Already solving…", True
    overrides = {
        "co2_price":        co2,
        "ic_factor":        ic,
        "fr_nuclear_pmin":  fnuc,
        "fr_hydro_pmax":    fhyd,
        "ccgt_must_run_mw": mr,
        "phs_pmax":         phs,
        "trans_factor":     trans,
        "n_days":           days,
        "mip_enabled":      "mip" in (mip_vals or []),
    }
    start_solve(overrides)
    mip_note = " (MIP ~90s)" if "mip" in (mip_vals or []) else " (~30s LP)"
    return False, f"⚙ Solving{mip_note}…", True


@app.callback(
    Output("store-data",   "data",     allow_duplicate=True),
    Output("poll",         "disabled", allow_duplicate=True),
    Output("solve-status", "children", allow_duplicate=True),
    Output("solve-btn",    "disabled", allow_duplicate=True),
    Input("poll", "n_intervals"),
    prevent_initial_call=True,
)
def cb_poll(n_intervals):
    status, result_data, error = poll_solve_result()
    if status == "running":
        raise PreventUpdate
    if status == "done" and result_data is not None:
        clear_solve_state()
        return result_data, True, "✓ Solve complete", False
    if status == "error":
        clear_solve_state()
        msg = f"✗ {error[:70]}" if error else "✗ Solve failed"
        return dash.no_update, True, msg, False
    raise PreventUpdate


@app.callback(
    Output("net-dd",       "options"),
    Output("net-dd",       "value"),
    Input("net-refresh",     "n_intervals"),
    Input("btn-refresh-nets","n_clicks"),
    State("net-dd",          "value"),
    State("net-dd",          "options"),
    prevent_initial_call=True,
)
def cb_refresh_networks(_, __, current_value, current_options):
    """Poll solved_networks/ every 15 s or on manual refresh button."""
    new_opts = list_solved_networks()
    if new_opts == current_options:
        raise PreventUpdate
    # Keep current selection if it still exists, otherwise default to newest
    values = [o["value"] for o in new_opts]
    new_val = current_value if current_value in values else (values[0] if values else None)
    return new_opts, new_val


@app.callback(
    Output("calib-table", "children"),
    Input("store-data",   "data"),
    prevent_initial_call=True,
)
def cb_calib_table(data):
    if not data:
        raise PreventUpdate
    return make_calibration_table(deserialise(data))


@app.callback(
    Output("change-badge", "children"),
    Input("s-co2",    "value"),
    Input("s-ic",     "value"),
    Input("s-fnuc",   "value"),
    Input("s-fhyd",   "value"),
    Input("s-mr",     "value"),
    Input("s-phs",    "value"),
    Input("s-trans",  "value"),
    Input("s-days",   "value"),
    Input("mip-toggle", "value"),
    prevent_initial_call=True,
)
def cb_mark_changes(*_):
    return "⚠ Unsaved changes — click Re-solve"


@app.callback(
    Output("g-cap",     "figure"),
    Input("store-data", "data"),
    prevent_initial_call=True,
)
def cb_capacity(data):
    if not data:
        raise PreventUpdate
    return make_capacity_figure(deserialise(data))


@app.callback(
    Output("ai-summary",  "children"),
    Output("diag-preview","children"),
    Input("store-data",   "data"),
    prevent_initial_call=True,
)
def cb_diag_panel(data):
    """Populate the AI summary text and the hourly preview table."""
    if not data:
        raise PreventUpdate
    d   = deserialise(data)
    txt = _build_ai_prompt(d)

    # Preview table — first 48 rows of full diagnostic df
    df  = _build_diagnostic_df(d).head(48)
    cols = [{"name": c, "id": c} for c in df.columns]
    preview = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=cols,
        style_table={"overflowX": "auto", "fontSize": "10px"},
        style_header={"backgroundColor": _SLATE, "color": "#C8D4DC",
                      "fontWeight": "600", "fontSize": "10px",
                      "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
                      "padding": "5px 8px"},
        style_cell={"backgroundColor": _WHITE, "color": _PLOT_TEXT,
                    "fontSize": "10px", "padding": "4px 8px",
                    "fontFamily": "Helvetica Neue, Helvetica, Arial, sans-serif",
                    "border": f"1px solid {_GRID}"},
        page_size=48,
        style_as_list_view=False,
    )
    return txt, preview


@app.callback(
    Output("diag-download", "data"),
    Input("btn-csv",        "n_clicks"),
    Input("btn-ai-txt",     "n_clicks"),
    State("store-data",     "data"),
    prevent_initial_call=True,
)
def cb_download(_n_csv, _n_txt, data):
    if not data:
        raise PreventUpdate
    ctx     = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    d       = deserialise(data)
    ts0     = d["timestamps"][0][:10].replace("-", "") if d["timestamps"] else "unknown"

    if trigger == "btn-csv":
        df  = _build_diagnostic_df(d)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return dcc.send_string(buf.getvalue(), filename=f"pypsa_diag_{ts0}.csv")

    if trigger == "btn-ai-txt":
        txt = _build_ai_prompt(d)
        return dcc.send_string(txt, filename=f"pypsa_ai_summary_{ts0}.txt")

    raise PreventUpdate


@app.callback(
    Output("sidebar",         "style"),
    Output("sidebar-content", "style"),
    Output("sidebar-toggle",  "children"),
    Output("sidebar-toggle",  "title"),
    Input("sidebar-toggle",   "n_clicks"),
    prevent_initial_call=True,
)
def cb_toggle_sidebar(n):
    if n and n % 2 == 1:
        return _SIDEBAR_COLLAPSED, {"display": "none"}, "▶", "Expand sidebar"
    return _SIDEBAR_EXPANDED, {}, "◀", "Collapse sidebar"


@app.callback(
    Output("map-hour-sl", "max"),
    Output("map-hour-sl", "value"),
    Output("map-hour-sl", "marks"),
    Input("store-data",   "data"),
    prevent_initial_call=True,
)
def cb_init_map_slider(data):
    """Sync the map hour slider to the loaded network's snapshot count."""
    if not data:
        raise PreventUpdate
    ts = data.get("timestamps", [])
    n  = len(ts)
    if n == 0:
        raise PreventUpdate
    step  = max(1, n // 10)
    marks = {
        i: {"label": ts[i][5:13], "style": {"color": _MUTED, "fontSize": "9px"}}
        for i in range(0, n, step)
    }
    return n - 1, 0, marks


@app.callback(
    Output("g-map",       "figure"),
    Input("store-data",   "data"),
    Input("map-hour-sl",  "value"),
    prevent_initial_call=True,
)
def cb_update_map(data, hour_idx):
    if not data:
        raise PreventUpdate
    d   = deserialise(data)
    idx = int(hour_idx) if hour_idx is not None else 0
    return make_map_figure(d, idx)


def _loading_badge(frac: float) -> html.Span:
    """Coloured loading % badge for connected-line table."""
    pct = frac * 100
    if frac < 0.30:
        bg, fg = "#4CAF50", "white"
    elif frac < 0.60:
        bg, fg = "#8BC34A", "#333"
    elif frac < 0.80:
        bg, fg = "#FFC107", "#333"
    elif frac < 0.95:
        bg, fg = "#FF5722", "white"
    else:
        bg, fg = "#B71C1C", "white"
    return html.Span(f"{pct:.0f}%", style={
        "backgroundColor": bg, "color": fg,
        "padding": "1px 7px", "borderRadius": "3px",
        "fontSize": "10px", "fontWeight": "700",
    })


@app.callback(
    Output("map-hover-info", "children"),
    Input("g-map", "hoverData"),
    State("store-data", "data"),
    State("map-hour-sl", "value"),
    prevent_initial_call=True,
)
def cb_map_hover(hover_data, store_data, hour_sl):
    if not hover_data or not store_data:
        raise PreventUpdate
    pts = hover_data.get("points", [])
    if not pts:
        raise PreventUpdate
    cd = pts[0].get("customdata")
    if not cd:
        raise PreventUpdate

    bus_id  = cd[0]
    idx     = int(hour_sl) if hour_sl is not None else 0
    ts_list = store_data.get("timestamps", [])
    ts_lbl  = ts_list[idx][:16] if idx < len(ts_list) else "—"

    # Raw lookups (cheaper than full deserialise)
    price_es_list  = store_data.get("price_es", [])
    bus_prices_raw = store_data.get("bus_prices", {})
    bus_gen_raw    = store_data.get("bus_gen", {})
    bus_cap_raw    = store_data.get("bus_cap", {})
    bus_load_raw   = store_data.get("bus_load", {})
    tx_lines_raw   = store_data.get("map_meta", {}).get("lines", {})
    line_ldgs_raw  = store_data.get("line_loadings", {})

    # ── Prices ──────────────────────────────────────────────────────────────
    es_mean   = price_es_list[idx] if idx < len(price_es_list) else None
    bus_pseries = bus_prices_raw.get(bus_id, [])
    bus_price   = bus_pseries[idx] if idx < len(bus_pseries) else None

    if bus_price is not None and es_mean is not None:
        diff      = bus_price - es_mean
        arrow     = "▲" if diff >= 0 else "▼"
        diff_col  = _CORAL if diff > 2 else (_TEAL if diff < -2 else _MUTED)
        price_blk = html.Div([
            html.Span(f"€{bus_price:.1f}/MWh", style={
                "backgroundColor": "#EEF2FF", "border": f"1px solid {_GRID}",
                "borderRadius": "4px", "padding": "3px 12px",
                "fontSize": "15px", "fontWeight": "800", "color": _SLATE,
                "marginRight": "10px",
            }),
            html.Span(f"{arrow} {abs(diff):.1f} vs ES avg (€{es_mean:.1f})",
                      style={"fontSize": "12px", "color": diff_col, "fontWeight": "600"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})
    else:
        price_blk = html.Div("No price data", style={"color": _MUTED, "fontSize": "12px",
                                                       "marginBottom": "10px"})

    # ── Generation & capacity table ──────────────────────────────────────────
    gen_hour = bus_gen_raw.get(bus_id, {})
    cap_bus  = bus_cap_raw.get(bus_id, {})
    all_carriers = sorted(set(list(gen_hour.keys()) + list(cap_bus.keys())))

    gen_rows = []
    total_gen = 0.0
    for carrier in all_carriers:
        gen_ts  = gen_hour.get(carrier, [])
        mw_now  = gen_ts[idx] if idx < len(gen_ts) else 0.0
        cap_mw  = cap_bus.get(carrier, 0.0)
        util    = mw_now / cap_mw * 100 if cap_mw > 0.5 else 0.0
        total_gen += mw_now
        col     = COLORS.get(carrier, _MUTED)
        # Utilisation bar (out of 60px wide)
        bar_w   = max(1, int(util / 100 * 60))
        gen_rows.append(html.Tr([
            html.Td(html.Span("█", style={"color": col, "fontSize": "12px"}),
                    style={"width": "12px", "paddingRight": "4px"}),
            html.Td(carrier, style={"minWidth": "100px", "fontSize": "11px",
                                    "color": _PLOT_TEXT}),
            html.Td(f"{cap_mw:.0f} MW", style={"textAlign": "right", "fontSize": "11px",
                                                 "color": _MUTED, "paddingRight": "10px"}),
            html.Td(f"{mw_now:.0f} MW", style={"textAlign": "right", "fontSize": "11px",
                                                 "fontWeight": "700", "paddingRight": "10px"}),
            html.Td([
                html.Div(style={
                    "width": f"{bar_w}px", "height": "8px",
                    "backgroundColor": col, "borderRadius": "2px",
                    "display": "inline-block",
                }),
                html.Span(f"  {util:.0f}%", style={"fontSize": "10px", "color": _MUTED,
                                                     "marginLeft": "4px"}),
            ], style={"whiteSpace": "nowrap"}),
        ]))

    gen_table = html.Table([
        html.Thead(html.Tr([
            html.Th("", style={"width": "12px"}),
            html.Th("Carrier", style={"fontSize": "10px", "color": _MUTED,
                                       "fontWeight": "400", "textAlign": "left"}),
            html.Th("Capacity", style={"fontSize": "10px", "color": _MUTED,
                                        "fontWeight": "400", "textAlign": "right",
                                        "paddingRight": "10px"}),
            html.Th("Now", style={"fontSize": "10px", "color": _MUTED,
                                   "fontWeight": "400", "textAlign": "right",
                                   "paddingRight": "10px"}),
            html.Th("Utilisation", style={"fontSize": "10px", "color": _MUTED,
                                           "fontWeight": "400"}),
        ])),
        html.Tbody(gen_rows if gen_rows else [
            html.Tr([html.Td("No generators at this bus.",
                             colSpan=5, style={"color": _MUTED, "fontSize": "11px"})])
        ]),
    ], style={"borderCollapse": "collapse", "width": "100%", "marginBottom": "8px"})

    # ── Load & net injection ─────────────────────────────────────────────────
    load_ts  = bus_load_raw.get(bus_id, [])
    load_mw  = load_ts[idx] if idx < len(load_ts) else None
    net_inj  = total_gen - load_mw if load_mw is not None else None
    load_row = html.Div([
        html.Span("Load: ", style={"color": _MUTED, "fontSize": "11px"}),
        html.Span(f"{load_mw:.0f} MW" if load_mw is not None else "n/a",
                  style={"fontWeight": "700", "fontSize": "11px", "marginRight": "18px"}),
        html.Span("Net injection: ", style={"color": _MUTED, "fontSize": "11px"}),
        html.Span(
            f"{net_inj:+.0f} MW" if net_inj is not None else "n/a",
            style={"fontWeight": "700", "fontSize": "11px",
                   "color": _TEAL if (net_inj or 0) >= 0 else _CORAL},
        ),
        html.Span(f"  (total gen: {total_gen:.0f} MW)",
                  style={"color": _MUTED, "fontSize": "10px", "marginLeft": "8px"}),
    ], style={"marginBottom": "10px"})

    # ── Connected transmission lines ─────────────────────────────────────────
    connected = [
        (lid, ldata, line_ldgs_raw.get(lid, []))
        for lid, ldata in tx_lines_raw.items()
        if ldata.get("bus0") == bus_id or ldata.get("bus1") == bus_id
    ]
    if connected:
        line_rows = []
        for _, ldata, ldg_ts in sorted(connected):
            frac   = ldg_ts[idx] if idx < len(ldg_ts) else 0.0
            s_nom  = ldata.get("s_nom", 0)
            other  = ldata["bus1"] if ldata.get("bus0") == bus_id else ldata["bus0"]
            line_rows.append(html.Tr([
                html.Td(f"{bus_id} ↔ {other}",
                        style={"fontSize": "10px", "color": _PLOT_TEXT,
                               "paddingRight": "12px"}),
                html.Td(f"{s_nom:.0f} MW cap",
                        style={"fontSize": "10px", "color": _MUTED,
                               "paddingRight": "12px"}),
                html.Td(_loading_badge(frac)),
            ]))
        lines_blk = html.Div([
            html.Div("Connected lines", style={"fontSize": "11px", "fontWeight": "700",
                                                "color": _PLOT_TEXT, "marginBottom": "4px"}),
            html.Table([html.Tbody(line_rows)],
                       style={"borderCollapse": "collapse"}),
        ])
    else:
        lines_blk = html.Div()

    # ── Assemble panel ───────────────────────────────────────────────────────
    return html.Div([
        html.Div([
            html.Strong(bus_id, style={"fontSize": "15px", "color": _SLATE,
                                        "marginRight": "10px"}),
            html.Span(ts_lbl, style={"fontSize": "11px", "color": _MUTED}),
        ], style={"marginBottom": "8px"}),
        price_blk,
        gen_table,
        load_row,
        lines_blk,
    ])


@app.callback(
    Output("g-ic-tech",          "figure"),
    Output("g-gen-breakdown",    "figure"),
    Output("g-fr-price-drivers", "figure"),
    Output("g-fr-profile",       "figure"),
    Output("g-fr-tech-scatter",  "figure"),
    Output("g-fr-heatmap",       "figure"),
    Output("g-fr-scatter",       "figure"),
    Output("g-pt-scatter",       "figure"),
    Output("fr-overnight-stats", "children"),
    Input("store-data",          "data"),
    prevent_initial_call=True,
)
def cb_fr_analysis(data):
    if not data:
        raise PreventUpdate
    d = deserialise(data)
    return (
        make_ic_tech_figure(d),
        _make_gen_breakdown_figure(d),
        _make_fr_price_drivers(d),
        _make_overnight_profile(d),
        _make_fr_tech_scatter(d),
        _make_price_error_heatmap(d),
        _make_import_scatter(d),
        _make_pt_import_scatter(d),
        _compute_overnight_summary(d),
    )


@app.callback(
    Output("g-pfm-ts",       "figure"),
    Output("g-pfm-residual", "figure"),
    Output("g-pfm-actual",   "figure"),
    Output("g-pfm-cong",     "figure"),
    Input("store-data",      "data"),
    prevent_initial_call=True,
)
def cb_price_formation(data):
    if not data:
        raise PreventUpdate
    d = deserialise(data)
    return (
        _make_pfm_ts(d),
        _make_pfm_residual(d),
        _make_pfm_actual_compare(d),
        _make_pfm_congestion(d),
    )


@app.callback(
    Output("g-setter-bar",   "figure"),
    Output("g-price-ts",     "figure"),
    Output("g-headroom-ts",  "figure"),
    Input("store-data",      "data"),
    Input("price-method-dd", "value"),
    prevent_initial_call=True,
)
def cb_calib_charts(data, method):
    if not data:
        raise PreventUpdate
    d = deserialise(data)
    # Preserve original LW for _make_price_ts (needs both series to plot comparison)
    d["price_lw"] = d.get("price_es", pd.Series(dtype=float))
    # Separate dict for setter-bar + headroom (statistics use the selected method)
    d_sel = dict(d)
    d_sel["price_es"] = _select_price(d, method or "lw")
    return (_make_setter_bar(d_sel), _make_price_ts(d), _make_headroom_ts(d_sel))


@app.callback(
    Output("g-load-map", "figure"),
    Output("g-cap-map",  "figure"),
    Input("store-data",  "data"),
    prevent_initial_call=True,
)
def cb_spatial_maps(data):
    if not data:
        raise PreventUpdate
    d = deserialise(data)
    return make_load_map(d), make_capacity_map(d)


@app.callback(
    Output("g-hydro-soc", "figure"),
    Input("store-data",   "data"),
    prevent_initial_call=True,
)
def cb_hydro_soc(data):
    if not data:
        raise PreventUpdate
    return _make_hydro_soc(deserialise(data))


@app.callback(
    Output("g-ccgt-tiers",      "figure"),
    Output("g-curtail-scatter", "figure"),
    Input("store-data",         "data"),
    prevent_initial_call=True,
)
def cb_calibration_panels(data):
    if not data:
        raise PreventUpdate
    d = deserialise(data)
    return _make_ccgt_tier_figure(d), _make_curtailment_price_scatter(d)


@app.callback(
    Output("g-curtail-monthly", "figure"),
    Input("store-data",         "data"),
    prevent_initial_call=True,
)
def cb_curtailment(data):
    if not data:
        raise PreventUpdate
    return _make_curtailment_monthly(deserialise(data))


@app.callback(
    Output("g-monthly-stack", "figure"),
    Output("g-monthly-price", "figure"),
    Output("g-fr-pt-inflow",  "figure"),
    Input("store-data", "data"),
    prevent_initial_call=True,
)
def cb_monthly_analysis(data):
    if not data:
        raise PreventUpdate
    d = deserialise(data)
    return (
        _make_monthly_stack(d),
        _make_monthly_price(d),
        _make_fr_pt_inflow(d),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    print("\n  PyPSA-Spain Dashboard  →  http://localhost:8050\n")
    app.run(debug=False, port=8050)
