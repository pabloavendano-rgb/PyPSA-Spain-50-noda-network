# Non-Linear Marginal Costs — Base Case Rules

> **Source:** [`Analysis/01_capacity/non_linear_MCs.ipynb`](../Analysis/01_capacity/non_linear_MCs.ipynb)
> **Last run:** 2026-04-26
> **Network modified:** `resources/networks/50n_ES_FR_PT.nc` (backup created as `50n_ES_FR_PT_backup_20260426_122612.nc`)
> **Status:** This is the **base case** — applied before any peaker fleet additions, CCGT_Flex split, or CO₂ price overlay.

---

## 1. Parameters

All values in EUR/MWh unless stated.

| Parameter | Value | Scope |
|-----------|-------|-------|
| `RANDOM_SEED` | `42` | Reproducible jitter for all random draws |
| `SOLAR_MC` | `0.10` | All solar generators |
| `WIND_MC` | `0.50` | onwind, offwind, offwind-float |
| `NUC_MC_LO` | `12.0` | Nuclear lower bound |
| `NUC_MC_HI` | `18.0` | Nuclear upper bound |
| `NUC_PMIN` | `0.50` | Nuclear must-run fraction |
| `ES_HYDRO_OC` | `28.0` | ES reservoirs (storage_units) |
| `FR_HYDRO_OC` | `22.0` | FR hydro (generators) |
| `PT_HYDRO_OC` | `35.0` | PT hydro (generators) |
| `BIOMASS_MC` | `40.0` | Biomass (flat) |
| `COGEN_MC` | `45.0` | Cogeneration (PT industrial CHP) |
| `COGEN_PMIN` | `0.70` | Cogeneration must-run fraction |
| `CCGT_IBERIA_T1` | `(52.0, 62.0)` | ES/PT Tier 1 — high efficiency |
| `CCGT_IBERIA_T2` | `(62.0, 72.0)` | ES/PT Tier 2 — standard fleet |
| `CCGT_IBERIA_T3` | `(72.0, 82.0)` | ES/PT Tier 3 — aged fleet |
| `CCGT_FR_T1` | `(68.0, 78.0)` | FR Tier 1 |
| `CCGT_FR_T2` | `(78.0, 88.0)` | FR Tier 2 |
| `COAL_MC_LO` | `112.0` | Coal lower bound (ETS-inclusive) |
| `COAL_MC_HI` | `118.0` | Coal upper bound |
| `OCGT_MC` | `125.0` | PT OCGT peakers |
| `OIL_MC` | `180.0` | FR Fioul turbines |

---

## 2. Technology Rules

### 2.1 VRE Floor

| Carrier | MC (EUR/MWh) | Rationale |
|---------|-------------|-----------|
| `solar` | `0.10` | Near-zero marginal cost; small floor prevents division-by-zero in merit order plots |
| `onwind` | `0.50` | Slightly higher O&M floor than solar |
| `offwind` | `0.50` | Same as onwind |
| `offwind-float` | `0.50` | Same as onwind |
| `ror` | `0.00` | Run-of-river — truly must-dispatch, no fuel cost |

**Code:**
```python
for car in SOLAR_CARRIERS:
    mask = n.generators['carrier'] == car
    n.generators.loc[mask, 'marginal_cost'] = SOLAR_MC

for car in WIND_CARRIERS:
    mask = n.generators['carrier'] == car
    n.generators.loc[mask, 'marginal_cost'] = WIND_MC
```

### 2.2 Nuclear — Per-Reactor Jitter + Must-Run

| Parameter | Value |
|-----------|-------|
| MC range | U[12.0, 18.0] per reactor |
| `p_min_pu` | `0.50` (50% must-run) |

Each reactor gets an independent uniform draw from U[12, 18] to represent different refuelling cycles and maintenance states. `p_min_pu = 0.50` forces baseload commitment and creates negative-price risk during solar peaks.

**Code:**
```python
nuc_mask = n.generators['carrier'] == 'nuclear'
n_nuc    = nuc_mask.sum()
jitter   = rng.uniform(NUC_MC_LO, NUC_MC_HI, size=n_nuc)
n.generators.loc[nuc_mask, 'marginal_cost'] = jitter
n.generators.loc[nuc_mask, 'p_min_pu']      = NUC_PMIN
```

### 2.3 Biomass & Cogeneration

| Carrier | MC (EUR/MWh) | p_min_pu | Scope |
|---------|-------------|----------|-------|
| `biomass` | `40.0` | — | ES / PT |
| `cogen` | `45.0` | `0.70` | PT only |

Biomass at €40 reflects wood pellet / agricultural waste sourcing. Cogeneration at €45 with `p_min_pu = 0.70` represents industrial units subsidised by heat demand.

**Code:**
```python
bio_mask = n.generators['carrier'] == 'biomass'
n.generators.loc[bio_mask, 'marginal_cost'] = BIOMASS_MC

cog_mask = n.generators['carrier'] == 'cogen'
n.generators.loc[cog_mask, 'marginal_cost'] = COGEN_MC
n.generators.loc[cog_mask, 'p_min_pu']      = COGEN_PMIN
```

### 2.4 Hydro — Opportunity Cost

| Country | Type | MC (EUR/MWh) | Rationale |
|---------|------|-------------|-----------|
| ES | Reservoirs (storage_units) | `28.0` | Iberian reservoirs — moderate drought risk |
| ES | PHS (storage_units) | `0.0` | Pure price-arbitrage device; cost is round-trip efficiency |
| FR | Generators | `22.0` | Large Alpine reservoirs — lower scarcity |
| PT | Generators | `35.0` | Iberian watershed — higher drought exposure |

**Key distinction:** ES reservoirs are `storage_units` in PyPSA. Setting `marginal_cost` on a storage unit is the cost the optimizer charges per MWh *dispatched* — this is the correct way to express opportunity cost for a reservoir (the value of water withheld for a drier hour).

FR / PT hydro are modelled as generators; OC is applied directly as `marginal_cost`.

ES PHS stays at €0 — its effective "cost" is the electricity consumed during pumping, which the optimizer already accounts for through the storage round-trip efficiency.

**Code:**
```python
# ES hydro reservoirs (storage units)
es_hyd = (su_cc == 'ES') & (n.storage_units['carrier'] == 'hydro')
n.storage_units.loc[es_hyd, 'marginal_cost'] = ES_HYDRO_OC

# ES PHS: leave at 0
es_phs = (su_cc == 'ES') & (n.storage_units['carrier'] == 'PHS')
# (left at 0.0)

# FR / PT hydro (generators)
for cc, oc in [('FR', FR_HYDRO_OC), ('PT', PT_HYDRO_OC)]:
    m = (gen_cc == cc) & (n.generators['carrier'] == 'hydro')
    n.generators.loc[m, 'marginal_cost'] = oc
```

### 2.5 CCGT — Three-Tier Efficiency Ladder

#### Tier Structure

| Tier | Efficiency Analogue | ES / PT Range | FR Range |
|------|-------------------|---------------|----------|
| 1 | 62% CCGT — high eff., cheap Iberian LNG | €52 – €62 | €68 – €78 |
| 2 | 55% CCGT — standard fleet | €62 – €72 | €78 – €88 |
| 3 | 48% CCGT — aged fleet | €72 – €82 | — |

#### Assignment Logic

Generators are **ranked by `p_nom` descending** — larger plants are treated as more modern and efficient and receive lower marginal costs. The sorted list is split into equal-ish tiers using `np.array_split`. Each unit draws a uniform random MC from within its tier's range.

FR has only 2 units so a 2-tier split is used.

**Code:**
```python
def assign_ccgt_tiers(country_code, tier_ranges):
    mask = (gen_cc == country_code) & (n.generators['carrier'] == 'CCGT')
    idx  = n.generators.loc[mask].sort_values('p_nom', ascending=False).index
    n_units = len(idx)
    if n_units == 0:
        return

    n_tiers = len(tier_ranges)
    splits  = np.array_split(idx, n_tiers)

    for tier_idx, (split_idx, (lo, hi)) in enumerate(zip(splits, tier_ranges), start=1):
        mcs = rng.uniform(lo, hi, size=len(split_idx))
        n.generators.loc[split_idx, 'marginal_cost'] = mcs

# Iberia: 3-tier
assign_ccgt_tiers('ES', [CCGT_IBERIA_T1, CCGT_IBERIA_T2, CCGT_IBERIA_T3])
assign_ccgt_tiers('PT', [CCGT_IBERIA_T1, CCGT_IBERIA_T2, CCGT_IBERIA_T3])

# France: 2-tier
assign_ccgt_tiers('FR', [CCGT_FR_T1, CCGT_FR_T2])
```

### 2.6 Coal, OCGT, Oil — Price Ceiling

| Carrier | MC (EUR/MWh) | Scope | Notes |
|---------|-------------|-------|-------|
| `coal` | U[112.0, 118.0] | ES + FR | ETS-inclusive (~€65/tCO₂ baked in) |
| `OCGT` | `125.0` | PT only | Cheaper gas ceiling |
| `oil` | `180.0` | FR only | Emergency Fioul turbines |

**Code:**
```python
# Coal: per-unit jitter
coal_mask = n.generators['carrier'] == 'coal'
n_coal    = coal_mask.sum()
coal_mcs  = rng.uniform(COAL_MC_LO, COAL_MC_HI, size=n_coal)
n.generators.loc[coal_mask, 'marginal_cost'] = coal_mcs

# OCGT: flat
ocgt_mask = n.generators['carrier'] == 'OCGT'
n.generators.loc[ocgt_mask, 'marginal_cost'] = OCGT_MC

# Oil: flat
oil_mask  = n.generators['carrier'] == 'oil'
n.generators.loc[oil_mask, 'marginal_cost'] = OIL_MC
```

---

## 3. Read-Back Verification (from last run)

After saving, the notebook performs a sanity read-back check. Results from the last run:

```
Generators:
  ES nuclear   : MC 14.6–17.2  p_min=0.50
  ES biomass   : MC 40.0–40.0  p_min=0.00
  ES CCGT      : MC 53.3–81.7  p_min=0.00
  ES coal      : MC 113.9–116.2  p_min=0.00
  FR nuclear   : MC 12.6–17.9  p_min=0.50
  FR hydro     : MC 22.0–22.0  p_min=0.00
  FR CCGT      : MC 72.4–86.3  p_min=0.00
  PT hydro     : MC 35.0–35.0  p_min=0.00
  PT CCGT      : MC 56.8–78.7  p_min=0.00

Storage units:
  ES hydro     : MC 28.0–28.0  (n=29)
  ES PHS       : MC 0.0–0.0    (n=13)
```

---

## 4. Save Procedure

```python
import shutil
from pathlib import Path
from datetime import datetime

net_path = Path(NET_PATH)

# Backup
timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_path = net_path.with_name(f'{net_path.stem}_backup_{timestamp}.nc')
shutil.copy2(net_path, backup_path)

# Save (overwrites original)
n.export_to_netcdf(str(net_path))

# Read-back sanity check
n_check = pypsa.Network(str(net_path))
# ... (verification loop as above)
```

---

## 5. Relationship to Other Modifications

This notebook is the **base case** for marginal costs. It is applied **before**:

1. **Peaker fleet additions** (CCGT_Flex split, OCGT/Diesel peakers) — these add new generators with higher MCs on top of this base
2. **CO₂ price overlay** (€65/tCO₂) — adds a per-technology CO₂ cost adder on top of the base MCs
3. **Hydro energy constraint** (23 TWh annual limit) — constrains total hydro dispatch independently of MC

The current network (`resources/networks/50n_ES_FR_PT.nc`) already has these rules applied. The CCGT MCs range from €53.3 to €86.3/MWh across 36 generators (49,937 MW total), confirming the tiered structure is in place.
