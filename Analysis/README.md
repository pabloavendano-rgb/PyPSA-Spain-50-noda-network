# Analysis Pipeline — PyPSA-Spain Validation Suite

This directory contains the validation and refinement pipeline for the 50-node Spain non-linear merit order model. It is the **Analysis counterpart** to the Snakemake workflow in `scripts/`.

## Quick Start

```bash
# Run full validation (6 months by default)
pixi run python Analysis/run_validation.py

# Run diagnostics only (no solve)
pixi run python Analysis/diagnose_network.py

# Plot merit order curves from a solved network
pixi run python Analysis/merit_order_curve.py
```

## Snakemake Pipeline Context

This Analysis pipeline operates on a network built by the **PyPSA-Spain Snakemake workflow** — a fork of [PyPSA-Eur](https://github.com/pypsa/pypsa-eur) extended for high-resolution Spanish modelling.

### Workflow Overview

The Snakemake pipeline is configured via [`config/config_ES.yaml`](config/config_ES.yaml) and executed as:

```bash
snakemake all --configfile config/config_ES.yaml --cores 4
```

Key stages (defined in [`Snakefile`](Snakefile) and `rules/*.smk`):

| Stage | Rule | Description |
|-------|------|-------------|
| **Retrieve** | `rules/retrieve.smk` | Downloads OSM network, cutouts, demand data |
| **Build electricity** | `rules/build_electricity.smk` | Clusters OSM to NUTS2/3 resolution, attaches renewable profiles (Q2Q), demand, hydro |
| **Prepare sector** | `rules/build_sector.smk` | Adds gas/LNG links, H₂ infrastructure, conventional generators |
| **Solve** | `rules/solve_overnight.smk` | Solves the LP with Gurobi (barrier, no crossover) |

### Configuration Basis ([`config/config_ES.yaml`](config/config_ES.yaml))

| Setting | Value | Purpose |
|---------|-------|---------|
| `countries` | `[ES]` | Single-country focus — only Spanish territory meshed |
| `snapshots` | `2023-01-01` → `2024-01-01` | Full 2024 calendar year (leap day dropped) |
| `clusters` | `[8]` | NUTS2-level clustering → ~50 electrical nodes |
| `planning_horizons` | `[2030]` | Forward-looking capacity expansion target |
| `foresight` | `overnight` | Greenfield optimisation (no myopic/perfect foresight) |
| `electricity.base_network` | `osm` | OpenStreetMap topology (not ENTSO-E) |
| `atlite.default_cutout` | `europe-2023-sarah3-era5` | ERA5 + SARAH3 solar irradiance |
| `energy.energy_totals_year` | `2023` | Proxy year for energy statistics (closest available to 2024) |
| `costs.year` | `2030` | Forward-looking CAPEX/OPEX for 2030 technology costs |

#### Q2Q Renewable Transform

Enabled via `pypsa_spain.q2q_transform` — replaces ERA5-derived wind/solar profiles with higher-resolution (4 km × 1 h) Spanish-specific profiles from the [Q2Q repository](https://github.com/cristobal-GC/Q2Q_repository):

```yaml
pypsa_spain:
  q2q_transform:
    enable: true
    onwind: data_ES/q2q/q2q_onwind_REF_v2.pkl
    solar: data_ES/q2q/q2q_solar_REF_v1.pkl
    solar-hsat: data_ES/q2q/q2q_solar-hsat_REF_v1.pkl
```

#### ESIOS Capacity Data (2024 Calibration)

The `pypsa_spain.update_elec_capacities` section reads REE/ESIOS CSV files to override PyPSA-Eur's default capacity estimates with official Spanish installed capacity data for 2024. Carriers updated: onwind, solar, hydro, PHS, coal, OCGT.

#### ISA Environmental Constraints

`pypsa_spain.ISA_class` applies the Spanish *Índice de Sensibilidad Ambiental* (Environmental Sensitivity Index) to restrict renewable deployment to grid codes 3 and 4 (lowest sensitivity), replacing the default CORINE/Natura 2000 constraints.

### Proxy Network Creation (France & Portugal)

Since PyPSA-Spain only meshes Spanish territory, neighbouring countries are represented as **simplified proxy networks** — not full PyPSA-Eur solves. These are created by [`Analysis/interconnector_analysis/create_proxy_networks.py`](Analysis/interconnector_analysis/create_proxy_networks.py).

#### France — 2-Bus Proxy

| Bus | Location | Coordinates | ES Corridor | AC Capacity | DC Capacity |
|-----|----------|-------------|-------------|-------------|-------------|
| `FR_WEST` | Bordeaux / Aquitaine | -0.60°E, 44.80°N | ES-FR WEST | 2,793 MW (3 AC) | 0 MW |
| `FR_EAST` | Montpellier / Languedoc | 3.80°E, 43.70°N | ES-FR EAST | 1,787 MW (1 AC) | 2,000 MW (2 DC) |

Internal line: `FR_WE` — southern French 400 kV arc (Aquitaine–Languedoc), 3 circuits × 1,787 MW = 5,362 MW, calibrated from 7 cross-lines in base_s.nc totalling 8,658 MW.

#### Portugal — 3-Bus Proxy

| Bus | Location | Coordinates | ES Corridor | ES Capacity |
|-----|----------|-------------|-------------|-------------|
| `PT_NORTH` | Porto / Recarei | -8.40°E, 41.10°N | ES-PT NORTH | 3,500 MW (3 AC) |
| `PT_CENTRE` | Setúbal / Lisbon | -9.00°E, 38.70°N | ES-PT CENTRE | 3,575 MW (2 AC) |
| `PT_SOUTH` | Ferreira do Alentejo / Beja | -8.00°E, 37.60°N | ES-PT SOUTH | 1,787 MW (1 AC) |

Internal lines: `PT_NC` (Porto–Lisbon, 3,575 MW) and `PT_CS` (Lisbon–Alentejo, 3,575 MW) — calibrated from 11 + 3 lines in base_s.nc.

#### Merging Process

The proxy networks are saved as `resources/networks/pt_proxy.nc` and `resources/networks/fr_proxy.nc`, then merged with the high-resolution 50-node ES network during the `prepare_sector_networks` Snakemake rule. ES interconnectors are added manually as `Link` components (not `Line`) to allow asymmetric NTC values (export ≠ import capacity). The interconnector market prices are pre-computed from historical 2023 data (`mp_FR_2030_cutout2023.csv`, `mp_PT_2030_cutout2023.csv`) and attached as `p_max_pu` / `p_min_pu` time series on each interconnector link.

### 2024 Harmonisation

The model is configured for a **2024 baseline** with 2030 costs:

- **Snapshots**: 2024-01-01 → 2025-01-01 (inclusive left, leap day dropped)
- **Energy totals**: 2023 as proxy (`energy_totals_year: 2023`) — the closest year with complete Eurostat data
- **Hydro**: Historic average inflows from ERA5 cutout, calibrated to Spanish reservoir statistics (23.0 TWh usable storage, 1,200 max_hours)
- **Demand**: 344 TWh annual (PNIEC 2030 target), NUTS3 profiles from economic sector disaggregation
- **Interconnector prices**: Historical 2023 day-ahead prices from OMIE (ES), EPEX SPOT (FR), and OMIE (PT) — **not** a nested PyPSA-Eur equilibrium

## Pipeline Architecture

### 1. Configuration — [`config.py`](Analysis/config.py)

Single source of truth (`MODEL_CONFIG` dict) for all refinements. Key sections:

| Section | Purpose |
|---------|---------|
| `co2_price` / `co2_intensity` | CO₂ adder per carrier (€65/t) |
| `ccgt_tiers` | Per-country CCGT fuel MC ranges, sorted by p_nom descending |
| `ccgt_flex` | 20% of each ES CCGT split to flexible units (MC €125–140) |
| `peakers` | OCGT (1,149 MW @ €160) + Diesel (769 MW @ €210) distributed by load/CCGT weight |
| `nuclear` | p_min_pu=0.50 (ES) / 0.30 (FR), ramp=0.20 (ES) / 0.10 (FR), MC jitter €12–18 |
| `hydro` | max_hours=1200, ramp_limit=0.20 (ES/FR) / 0.25 (PT), inflow×1.0 |
| `borders` | Realistic NTC values for FR→ES (4,000 MW) and PT→ES (3,600 MW) + `ic_factor` |
| `transmission` | trans_factor=0.50, s_max_pu=0.70 |
| `validation` | Start date, n_days, network path, solver, output dir |

### 2. Refinery — [`refinery.py`](Analysis/refinery.py)

Idempotent in-memory refinements applied in order:

1. **CCGT tiering + CO₂ adder** — Assigns fuel MCs by p_nom rank (larger = more efficient), adds CO₂ cost. Overrides `p_nom_min` from 100% to 0% so CCGTs can dispatch flexibly.
2. **CCGT_Flex split** — Splits 20% of each ES CCGT into a separate `CCGT_flex` generator with higher MC (€125–140) and faster ramping.
3. **Peaker fleet** — Distributes OCGT (1,149 MW) and Diesel (769 MW) peakers across ES nodes weighted 70% by load, 30% by CCGT capacity.
4. **Nuclear constraints** — Sets p_min_pu, ramp limits, and per-reactor MC jitter.
5. **Hydro parameters** — Caps reservoir max_hours at 1,200h, sets ramp limits on hydro generators, disables cyclic SOC, sets initial SOC to 50%.
6. **Border restoration** — Restores interconnector NTC values from config, then applies `ic_factor` to shrink all interconnectors proportionally.
7. **Transmission limits** — Scales ES internal line s_nom by `trans_factor`, sets `s_max_pu`.
8. **VOLL** — Optionally adds load-shedding generators at each ES bus.
9. **Lock capacities** — Disables all `p_nom_extendable` flags → dispatch-only solve.

### 3. Validation — [`run_validation.py`](Analysis/run_validation.py)

Loads the base network, applies refinements, slices to the analysis window, solves with Gurobi, and produces:

**Console output:**
- Price statistics (mean, median, p5, p95, max, bias, MAE, RMSE, correlation vs OMIE)
- Price frequency distribution (7 bands from near-zero to scarcity)
- VRE curtailment stats (total GWh, %, worst node)
- Dispatch comparison (Nuclear, Hydro, Wind, Solar, Thermal, Shed vs REE real data)
- Capacity comparison (Model p_nom vs 2024 REE installed capacity)

**Plots (saved to `Analysis/validation_output/`):**

| # | File | Description |
|---|------|-------------|
| 1 | `01_price_comparison.png` | Hourly price: model avg ES nodal vs OMIE |
| 2 | `02_price_duration_curve.png` | Sorted price duration curve with CCGT/peaker thresholds |
| 3 | `03_hourly_dispatch.png` | Hourly dispatch stack for ES, FR, PT (3 panels) |
| 4 | `04_daily/weekly/monthly_comparison.png` | Model vs real dispatch at multiple time resolutions |
| 5 | `05_congestion_map.png` | Nodal price map (mean + peak evening) with line loading overlay |
| 6 | `06_curtailment_map.png` | VRE curtailment rate + absolute GWh by node |
| 7 | `07_dispatch_pie_map.png` | Dispatch-by-node pie charts (size = total dispatch, segments = carrier mix) |
| 8 | `08_network_map.png` | Full ES+FR+PT network map, colour-coded by carrier/voltage |

### 4. Merit Order — [`merit_order_curve.py`](Analysis/merit_order_curve.py)

Plots the supply stack (sorted marginal cost curve) for each country, overlaying actual dispatch from a solved network. Outputs to `Analysis/validation_output/06_merit_order_*.png`.

### 5. Diagnostics — [`diagnose_network.py`](Analysis/diagnose_network.py)

Pre-solve sanity checks: load time series, generator p_max_pu, storage inflow, interconnector capacities, CCGT/nuclear/hydro ramp limits, transmission s_max_pu, VOLL presence, capacity lock.

## Key Model Parameters

### Generation Fleet (Spain, post-refinery)

| Carrier | Capacity (MW) | MC Range (€/MWh) | Notes |
|---------|--------------|-------------------|-------|
| Nuclear | 7,408 | 12–18 | p_min=0.50, ramp=0.20 |
| Hydro (reservoir) | 14,900 | 22–35 | max_hours=1,200, ramp=0.20 |
| PHS | 3,332 | — | Pumped storage, 1,200h |
| Run-of-river | 2,200 | — | Must-run when water |
| Onwind | 32,103 | 0.01 | p_max_pu from cutout |
| Solar | 32,350 | 0.01 | p_max_pu from cutout |
| CCGT | 26,251 | 75–105 | 3 tiers + CO₂ adder €22.6 |
| CCGT_flex | ~5,250 | 125–140 | 20% split from ES CCGTs |
| Coal | 2,061 | 113–116 | ETS-inclusive |
| OCGT peakers | 1,149 | 160 | Distributed by load/CCGT |
| Diesel peakers | 769 | 210 | Distributed by load/CCGT |

### Interconnectors (post-ic_factor)

| Route | Direction | NTC (MW) | With ic_factor=0.25 |
|-------|-----------|----------|---------------------|
| FR_WEST | ES↔FR | 2,200 | 550 |
| FR_EAST | ES↔FR | 1,800 | 450 |
| PT_NORTH | ES→PT / PT→ES | 1,422 / 1,137 | 356 / 284 |
| PT_CENTRE | ES→PT / PT→ES | 1,452 / 1,162 | 363 / 291 |
| PT_SOUTH | ES→PT / PT→ES | 726 / 581 | 182 / 145 |

## Recent Changes

### 2026-04-27: ic_factor + Output Pipeline

- **Added `ic_factor`** to [`config.py`](Analysis/config.py:134) — shrinks all interconnector p_nom by a configurable proportion after restoring NTC values. Set to 0.25 to simulate missing FR export routes (Italy/Germany).
- **Updated `_restore_borders()`** in [`refinery.py`](Analysis/refinery.py:382) — applies ic_factor after NTC restoration, with logging of total MW change.
- **Added dispatch-by-node pie chart map** — [`_plot_dispatch_pie_map()`](Analysis/run_validation.py) shows 50 Spanish nodes with pie charts sized by total dispatch, segments by carrier mix.
- **Added network topology map** — [`_plot_network_map()`](Analysis/run_validation.py) shows full ES+FR+PT network with colour-coded buses (by country) and lines (by voltage), with interconnector labels.

### 2026-04-26: Hydro Energy Fix

- Scaled hydro reservoir `max_hours` from 55.4 TWh → 23.0 TWh (×0.415) to match real Spanish reservoir storage capacity.
- Set `state_of_charge_initial` to 50% of new capacity.
- Disabled `cyclic_state_of_charge` so initial SOC is actually enforced.
- See [`fix_hydro_max_hours.py`](Analysis/04_solve_diagnostic/fix_hydro_max_hours.py).

### 2026-04-25: Non-Linear Refinery Pipeline

- Migrated from Jupyter notebooks to Python modules ([`refinery.py`](Analysis/refinery.py), [`config.py`](Analysis/config.py)).
- Added CCGT tiering (3 tiers by p_nom), CCGT_flex split (20%), peaker distribution.
- Added nuclear p_min_pu/ramp constraints, hydro ramp limits, border restoration.
- Added transmission scaling, VOLL, capacity locking.
- Created [`run_validation.py`](Analysis/run_validation.py) — full validation pipeline with solve + plots.
- Created [`diagnose_network.py`](Analysis/diagnose_network.py) — pre-solve sanity checks.

### 2026-04-24: Ramping Adjustments

- Applied ramp rate limits to all thermal generators (nuclear 0.20, CCGT 0.50–0.80, coal 0.40).
- See [`07_ramping_adjustments.ipynb`](Analysis/07_marginal_costs/07_ramping_adjustments.ipynb).

### 2026-04-23: Non-Linear Marginal Costs

- Applied technology-tiered merit order: nuclear €12–18, hydro €22–35, CCGT €52–88 (3 tiers), coal €112–118, peakers €160–210.
- See [`non_linear_MCs.ipynb`](Analysis/01_capacity/non_linear_MCs.ipynb).

### 2026-04-22: REE Capacity Fix

- Added ESIOS capacity files for hydro (17.1 GW), PHS (3.3 GW), coal (2.1 GW), OCGT (5.6 GW).
- Updated `fun_update_elec_capacities()` to handle `StorageUnit` components (hydro, PHS).
- Updated `config.yaml` with new carriers and ESIOS file references.

## Output Files

All validation outputs go to [`Analysis/validation_output/`](Analysis/validation_output/):

```
01_price_comparison.png        — Hourly price trace
02_price_duration_curve.png    — Sorted price duration
03_hourly_dispatch.png         — Dispatch stack (ES/FR/PT)
04_daily_comparison.png        — Daily bars model vs real
04_weekly_comparison.png       — Weekly bars model vs real
04_monthly_comparison.png      — Monthly bars model vs real
05_congestion_map.png          — Nodal prices + line loading
06_curtailment_map.png         — VRE curtailment by node
06_merit_order_ES.png          — ES supply stack
06_merit_order_FR.png          — FR supply stack
06_merit_order_PT.png          — PT supply stack
06_merit_order_all.png         — Combined supply stack
07_dispatch_pie_map.png        — Dispatch-by-node pie charts
08_network_map.png             — Full network topology
```

## Data Sources

- **OMIE prices**: [`Analysis/data/Spain_prices.csv`](Analysis/data/Spain_prices.csv) — hourly day-ahead prices
- **Real dispatch**: [`Analysis/data/daily_gen_spain.csv`](Analysis/data/daily_gen_spain.csv) — daily generation by technology from REE
- **Base network**: `resources/networks/base_s_50_elec_2704_fixed.nc` — 50-node clustered network
- **Interconnector prices**: `data_ES/interconnections/ic_market_prices/mp_FR_*.csv`, `mp_PT_*.csv`
