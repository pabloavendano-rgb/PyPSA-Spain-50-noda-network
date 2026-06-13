"""
Throwaway comparison script — burn after reading.

Loads:
  - Analysis/data/daily_gen_spain.csv   (REE daily generation, technologies as rows)
  - data/validation/spain_actual_generation_2024.csv  (ENTSO-E hourly actuals)

Outputs saved to Analysis/data/:
  - comparison_annual_totals.csv   (side-by-side annual TWh per technology)
  - comparison_daily_gas.csv       (daily gas GWh: REE vs ENTSO-E, with diff)

Run: python Analysis/compare_gas_daily.py  (from project root)
     OR: python compare_gas_daily.py       (from Analysis/)
"""

import os
import sys

import numpy as np
import pandas as pd

# Resolve paths whether run from project root or Analysis/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DAILY_CSV = os.path.join(SCRIPT_DIR, "data", "daily_gen_spain.csv")
HOURLY_CSV = os.path.join(PROJECT_ROOT, "data", "validation", "spain_actual_generation_2024.csv")
OUT_ANNUAL = os.path.join(SCRIPT_DIR, "data", "comparison_annual_totals.csv")
OUT_DAILY_GAS = os.path.join(SCRIPT_DIR, "data", "comparison_daily_gas.csv")

# Spanish month abbreviations → English (for pd.to_datetime)
ES_MONTHS = {
    "ene": "Jan", "feb": "Feb", "mar": "Mar", "abr": "Apr",
    "may": "May", "jun": "Jun", "jul": "Jul", "ago": "Aug",
    "sep": "Sep", "oct": "Oct", "nov": "Nov", "dic": "Dec",
}


def parse_spanish_date(s):
    """Convert '01/ene/24' → pd.Timestamp('2024-01-01')."""
    day, mon, yr = s.strip().split("/")
    return pd.to_datetime(f"{day}/{ES_MONTHS[mon]}/{yr}", format="%d/%b/%y")


def load_daily_ree(path):
    """
    Load REE daily CSV (technologies as rows, Spanish dates as columns).
    Returns a DataFrame indexed by date (daily), columns = technology names.
    Values are GWh/day.
    """
    raw = pd.read_csv(path, index_col=0)
    # Replace '-' strings with NaN, cast to float
    raw = raw.replace("-", np.nan).apply(pd.to_numeric, errors="coerce")

    # Parse column date strings
    dates = [parse_spanish_date(c) for c in raw.columns]

    # Transpose: rows = dates, columns = technologies
    df = raw.T.copy()
    df.index = pd.DatetimeIndex(dates)
    df.index.name = "date"
    df = df.sort_index()
    return df


def load_hourly_entsoe(path):
    """
    Load ENTSO-E hourly CSV.
    Values are MW; aggregate to daily GWh (MW × 1h / 1000).
    Returns DataFrame indexed by date (daily), columns = technology names.
    """
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    df.index = df.index.tz_convert("UTC")

    # Sum hourly MW values per day → daily MWh, then divide by 1000 → GWh
    daily = df.groupby(df.index.date).sum() / 1000.0
    daily.index = pd.DatetimeIndex(daily.index)
    daily.index.name = "date"
    return daily


def build_comparison(ree, entsoe):
    """
    Map both datasets to comparable gas/hydro/nuclear/coal buckets.
    Returns a dict of {technology: (ree_daily_series, entsoe_daily_series)}.
    """
    # Gas in REE CSV
    ree_gas_cols = [c for c in ["Ciclo combinado", "Turbina de gas", "Turbina de vapor"] if c in ree.columns]
    ree_gas = ree[ree_gas_cols].sum(axis=1)

    # Gas in ENTSO-E (Fossil Gas → CCGT bucket)
    entsoe_gas = entsoe.get("CCGT", pd.Series(0, index=entsoe.index))

    # Hydro
    ree_hydro_cols = [c for c in ["Hidráulica", "Hidroeólica"] if c in ree.columns]
    ree_hydro = ree[ree_hydro_cols].sum(axis=1)
    entsoe_hydro = (
        entsoe.get("Hydro_Reservoir", pd.Series(0, index=entsoe.index))
        + entsoe.get("Hydro_River", pd.Series(0, index=entsoe.index))
    )

    # Nuclear (direct match)
    ree_nuclear = ree.get("Nuclear", pd.Series(np.nan, index=ree.index))
    entsoe_nuclear = entsoe.get("Nuclear", pd.Series(np.nan, index=entsoe.index))

    # Coal
    ree_coal = ree.get("Carbón", pd.Series(np.nan, index=ree.index))
    entsoe_coal = entsoe.get("Coal", pd.Series(np.nan, index=entsoe.index))

    # Wind
    ree_wind = ree.get("Eólica", pd.Series(np.nan, index=ree.index))
    entsoe_wind = entsoe.get("Wind", pd.Series(np.nan, index=entsoe.index))

    # Solar
    ree_solar_cols = [c for c in ["Solar fotovoltaica", "Solar térmica"] if c in ree.columns]
    ree_solar = ree[ree_solar_cols].sum(axis=1) if ree_solar_cols else pd.Series(np.nan, index=ree.index)
    entsoe_solar = entsoe.get("Solar_PV", pd.Series(np.nan, index=entsoe.index))

    return {
        "Gas":     (ree_gas,     entsoe_gas),
        "Hydro":   (ree_hydro,   entsoe_hydro),
        "Nuclear": (ree_nuclear, entsoe_nuclear),
        "Coal":    (ree_coal,    entsoe_coal),
        "Wind":    (ree_wind,    entsoe_wind),
        "Solar":   (ree_solar,   entsoe_solar),
    }


def main():
    # --- Check inputs exist ---
    if not os.path.exists(HOURLY_CSV):
        sys.exit(
            f"ERROR: {HOURLY_CSV} not found.\n"
            "Run fetch_spain_2024_actuals.py first to download ENTSO-E data."
        )

    print("Loading REE daily CSV...")
    ree = load_daily_ree(DAILY_CSV)
    print(f"  {len(ree)} days, technologies: {list(ree.columns)}")

    print("Loading ENTSO-E hourly CSV and aggregating to daily...")
    entsoe = load_hourly_entsoe(HOURLY_CSV)
    print(f"  {len(entsoe)} days, technologies: {list(entsoe.columns)}")

    buckets = build_comparison(ree, entsoe)

    # --- Annual totals ---
    print("\n" + "=" * 60)
    print("ANNUAL TOTALS (TWh)")
    print("=" * 60)
    rows = []
    for tech, (s_ree, s_entsoe) in buckets.items():
        ree_twh = s_ree.sum() / 1000.0
        entsoe_twh = s_entsoe.sum() / 1000.0
        diff = entsoe_twh - ree_twh
        pct = (diff / ree_twh * 100) if ree_twh != 0 else np.nan
        rows.append({
            "Technology": tech,
            "REE_daily_TWh": round(ree_twh, 2),
            "ENTSOE_TWh": round(entsoe_twh, 2),
            "Diff_TWh (ENTSO-REE)": round(diff, 2),
            "Diff_%": round(pct, 1),
        })
        print(f"  {tech:<10} REE={ree_twh:7.2f}  ENTSO-E={entsoe_twh:7.2f}  "
              f"Δ={diff:+7.2f} TWh  ({pct:+.1f}%)")

    annual_df = pd.DataFrame(rows).set_index("Technology")
    annual_df.to_csv(OUT_ANNUAL)
    print(f"\n  Saved → {OUT_ANNUAL}")

    # --- Daily gas comparison ---
    ree_gas, entsoe_gas = buckets["Gas"]
    common_idx = ree_gas.index.intersection(entsoe_gas.index)
    daily_gas = pd.DataFrame({
        "REE_gas_GWh": ree_gas.reindex(common_idx),
        "ENTSOE_gas_GWh": entsoe_gas.reindex(common_idx),
    })
    daily_gas["diff_GWh"] = daily_gas["ENTSOE_gas_GWh"] - daily_gas["REE_gas_GWh"]
    daily_gas["diff_%"] = (daily_gas["diff_GWh"] / daily_gas["REE_gas_GWh"] * 100).round(1)
    daily_gas.index.name = "date"
    daily_gas.to_csv(OUT_DAILY_GAS)
    print(f"  Saved → {OUT_DAILY_GAS}")

    # --- Printed summary: gas divergence ---
    print("\n" + "=" * 60)
    print("TOP 5 DAYS WITH LARGEST GAS DIVERGENCE (absolute GWh)")
    print("=" * 60)
    top5 = daily_gas["diff_GWh"].abs().nlargest(5).index
    print(daily_gas.loc[top5].to_string())

    print("\n" + "=" * 60)
    print("MONTHLY AVERAGE GAS DIVERGENCE (ENTSO-E minus REE, GWh/day)")
    print("=" * 60)
    monthly = daily_gas["diff_GWh"].resample("ME").mean().rename("avg_daily_diff_GWh")
    monthly.index = monthly.index.strftime("%Y-%m")
    print(monthly.to_string())


if __name__ == "__main__":
    main()
