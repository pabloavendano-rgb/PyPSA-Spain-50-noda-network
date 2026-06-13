"""
MODEL_CONFIG — Single Source of Truth for the 50-node Spain validation.

═══════════════════════════════════════════════════════════════════════════
MARGINAL COST PIPELINE — how each generator's MC is built
═══════════════════════════════════════════════════════════════════════════

GAS-FIRED GENERATORS  (CCGT, CCGT_flex, OCGT)
──────────────────────────────────────────────
  Fuel input:   MIBGAS PVB day-ahead index, daily 2024 (gas_prices_csv)
                Annual mean 2024: €34.5/MWh_th; range €22.7–49.4/MWh_th.
                Source: MIBGAS operator data (mibgas.es), 2024.

  CO₂ physics:  0.202 tCO₂/MWh_thermal (natural gas combustion factor)
                Source: IPCC AR5 WGIII, Annex II Table A.II.4 (2014).

  ETS price:    co2_price = 65 €/tCO₂
                Source: ICE ECX EUA 2024 annual average ≈ €62/t; rounded up
                to €65 for conservatism. Update per validation month if needed.

  CO₂ adder (thermal): 0.202 × 65 = €13.1/MWh_th

  Formula:  MC_e(t)  = (MIBGAS(t) + 13.1) / η  +  VOM    [€/MWh_e]
            - (MIBGAS(t) + 13.1) converts gas from thermal to electrical using η
            - Less efficient units burn more gas → higher MC per MWh_e produced
            - VOM covers variable O&M above fuel + carbon costs

  Efficiency η by unit type (refinery assigns per-generator from tier or fixed):
    CCGT tier 1 (largest/newest, ≥350 MW):  η 0.72–0.80  [calibrated; physical design η 0.57–0.60]
    CCGT tier 2 (mid-fleet, 200–350 MW):    η 0.65–0.72  [calibrated; physical η 0.52–0.57]
    CCGT tier 3 (older/<200 MW):            η 0.58–0.65  [calibrated; physical η 0.46–0.52]
      NOTE: Effective η values are 15–20 pp above physical fleet design efficiency.
      This is a calibration choice to align model clearing prices with OMIE 2024
      observed prices. The physical η is absorbed into the calibrated parameter.
    CCGT_flex (partial-load / open-cycle mode):  η 0.38–0.44
      Source: combined-cycle at <60% load loses ~5–8 pp of efficiency vs design
              (EPRI TR-107395, 1997); proxy for units dispatched outside optimal load.
    OCGT (simple-cycle turbogas):                η = 0.38
      Source: IEA Technology Perspectives 2023; Spain REE turbogas fleet (GE LM6000).

  VOM by carrier (gas_vom):
    CCGT: €3/MWh_e  (source: IRENA Renewable Power Generation Costs 2023, gas O&M)
    CCGT_flex: €3/MWh_e  (same maintenance basis as CCGT)
    OCGT: €7/MWh_e  (higher: simple-cycle has ~2× starts per year, greater turbine
                     wear; source: NREL ATB 2024 gas peaker O&M, mid-case)

NON-GAS FOSSIL GENERATORS  (coal, diesel)
──────────────────────────────────────────
  Coal:   MC baked into base network [113.9–116.2 €/MWh_e].
          Includes ETS at prevailing price during base-network construction.
          co2_intensity = 0.895 tCO₂/MWh_e (hard coal, η≈0.38, 0.340/0.38).
          co2_baked_in = ["coal"] → refinery skips CO₂ re-addition.
          Source: EEA emission factor hard coal = 94.6 tCO₂/TJ = 0.340 tCO₂/MWh_th
                  (EEA/EMEP Guidebook 2023, Table 1–1).

  Diesel: MC computed as  base_mc + co2_intensity × co2_price + VOM.
          base_mc = 160 €/MWh_e (gasoil ~€56/MWh_th ÷ η 0.35).
          co2_intensity = 0.763 tCO₂/MWh_e (gas-oil, η=0.35, 0.267/0.35).
          co2_adder = 0.763 × 65 = €49.6/MWh_e.
          Source: IPCC AR5 Table A.II.4, gas-oil EF = 74.1 tCO₂/TJ = 0.267 tCO₂/MWh_th.

ZERO-CARBON GENERATORS  (nuclear, hydro, VRE)
──────────────────────────────────────────────
  MCs are operational only (fuel cycle + O&M for nuclear; water value for hydro).
  See per-carrier config sections below.

═══════════════════════════════════════════════════════════════════════════
"""

MODEL_CONFIG = {
    # ─── CO₂ pricing ──────────────────────────────────────────────────────────
    "co2_price": 65.0,  # EUR/tCO₂  (EU ETS — ICE ECX EUA 2024 annual average ≈ €62;
                         # rounded to €65 for conservatism per original calibration)

    # Thermal CO₂ intensity of natural gas (fixed physics, IPCC AR5).
    # All gas-fired generator MCs are derived from this single constant + per-unit η.
    # Formula: MC_e(t) = (MIBGAS(t) + 0.202 × co2_price) / η  +  VOM
    "gas_co2_intensity_th": 0.202,   # tCO₂/MWh_thermal, natural gas

    # Non-gas fossil CO₂ intensities (tCO₂/MWh_e, used directly since η is not
    # modelled per-unit for coal/diesel):
    "co2_intensity": {
        "coal":   0.895,   # hard coal η≈0.38 → adder €58.2 at €65/t
        "diesel": 0.763,   # gas-oil  η≈0.35 → adder €49.6 at €65/t
    },

    # Coal MC is already ETS-inclusive in the base network [113.9–116.2 €/MWh_e].
    "co2_baked_in": ["coal"],

    # ─── Coal capacity override ───────────────────────────────────────────────
    # All Spanish coal plants were closed by June 2023 (As Pontes last, 1,400 MW).
    # Zero capacity removes coal from 2024 dispatch; reverts the 5.8% coal-set hours
    # at €114.5/MWh that were artificially elevating the model mean price.
    "coal": {
        "disable_es": True,   # set False to restore base-network coal capacity
    },

    # ─── Gas fuel prices (MIBGAS / PVB day-ahead) ────────────────────────────
    # Daily MIBGAS PVB index for 2024 — the primary fuel cost input for all
    # gas-fired generators (CCGT, CCGT_flex, OCGT).
    # Refinery broadcasts daily price to hourly snapshots and computes per-unit
    # MC using each generator's efficiency η.
    "gas_prices_csv": "Analysis/data/gas_prices_daily.csv",

    # Variable O&M by gas carrier (€/MWh_e) — cost above fuel + CO₂.
    # Baseline 2024 values for dispatch-only model matching OMIE prices.
    "gas_vom": {
        "CCGT":      3.0,    # IRENA Renewable Power Generation Costs 2023, gas O&M
        "CCGT_flex": 3.0,    # same maintenance basis as CCGT
        "OCGT":      7.0,    # NREL ATB 2024 gas peaker O&M, mid-case
    },

    # MIBGAS fuel price multiplier — scales the gas price BEFORE computing MCs for ALL
    # gas-fired generators (CCGT, CCGT_flex, OCGT) via their η-based MC formulas.
    # MC(t) = (mibgas_mult × MIBGAS(t) + 0.202 × co2_price) / η + VOM
    #
    # 1.0 = historical MIBGAS (2024 baseline, mean ~€34.5/MWh_th).
    # 2.0 = 2× gas price shock — propagates through every gas generator's MC formula.
    #       At mean MIBGAS=34.5: effective fuel price = €69 → T1 MC €110, T4 MC €150.
    #
    # Replaces the old ccgt_mc_multiplier (post-hoc scale on final MCs) which was
    # η-agnostic and didn't propagate to CCGT_flex/OCGT. This is physically correct:
    # fuel price scales before η conversion, so less-efficient units get proportionally
    # larger absolute €/MWh increases.
    "mibgas_multiplier": 1.0,
    "fr_gas_multiplier": 0.85,    # FR TTF ~€29/MWh_th vs MIBGAS ~€34.5; scales FR gas before MC

    # ─── CCGT efficiency tiers ─────────────────────────────────────────────────
    # Generators sorted by p_nom descending → largest = most modern = best η.
    # Equal-MW splits across tiers. Gap in η space = price cliff (hockey-stick).
    #
    # MC at MIBGAS=36, CO₂=60: (36 + 0.202×60) / η + VOM = 48.12 / η + 3
    #
    # ES — 6 tiers, calibrated to match OMIE 2024 price distribution.
    # NOTE: η values here deliberately exceed physical CCGT design efficiency (max ~0.62
    # for SGT5-4000F at ISO conditions). They are calibrated parameters, not measured
    # efficiencies — the physical η is absorbed into the effective value to match observed
    # clearing prices after accounting for unit commitment, dispatch patterns, and
    # bid-cost differences between OMIE marginal bids and fuel+CO₂ cost estimates.
    # Effective η is ~15–20 pp above real design η → treat as a MC calibration knob.
    #
    # At MIBGAS=28, CO₂=60: fuel_adder = 28 + 0.202×60 = 40.12 €/MWh_th
    #   T1  η 0.72–0.80 → MC  53.2– 58.7  ← large/modern; price floor in CCGT hours
    #   T2  η 0.65–0.72 → MC  58.7– 64.7
    #   T3  η 0.58–0.65 → MC  64.7– 72.2
    #   T4  η 0.52–0.58 → MC  72.2– 80.2  ← end cheap cluster
    #   [cliff ~14 €/MWh]
    #   T5  η 0.38–0.45 → MC  92.2–108.6  ← expensive
    #   T6  η 0.28–0.38 → MC 108.6–146.4  ← worst/peaker
    # At mean MIBGAS=34.5: fuel_adder=46.62 → T1 MC 61–68, T4 MC 83–93
    # T5 (η=0.38-0.45) and T6 (η=0.28-0.38) removed — OCGT-equivalent efficiency.
    # Diagnostic confirmed these created a €30 cliff at T4→T5 and pulled mean clearing
    # from ~€88 to ~€108. CCGT_flex (η=0.33-0.42) already covers this efficiency range.
    # 4-tier fleet: T1-T4 ≈ 12.5 GW of realistic CCGT. When peak residual exceeds this,
    # CCGT_flex and OCGT take over — which is physically correct.
    # fuel_adder: mean MIBGAS=34.5 → 46.62  |  min MIBGAS=22.7 → 34.82  |  max MIBGAS=49.4 → 61.52
    # T1/T2 unchanged — set the correct middle PDC band.
    # T3+ aggressive ramp: each tier jumps sharply; MIBGAS amplification grows with lower η
    # (T5 η=0.20: +€1 MIBGAS → +€5 MC vs T1 η=0.76: +€1 MIBGAS → +€1.3 MC).
    "ccgt_efficiency_tiers": {
        # Calibrated η values are 15–20 pp ABOVE physical design efficiency.
        # This is intentional: aligns modelled clearing prices with OMIE 2024 observations.
        # Physical design η for modern CCGT ≈ 0.57–0.62; calibrated T1 → 0.72–0.80.
        #
        # RAISED MCs (2026-06-03): all tiers shifted down ~4 pp in η space, plus VOM
        # raised from 3→5. At MIBGAS=34.5, CO₂=65: fuel_adder = 47.6 €/MWh_th
        #   T1 η=0.72 mid → MC = 47.6/0.72 + 5 ≈ €71   ← was €66 (+€5)
        #   T2 η=0.62 mid → MC = 47.6/0.62 + 5 ≈ €82   ← was €73 (+€9)
        #   T3 η=0.52 mid → MC = 47.6/0.52 + 5 ≈ €97   ← was €82 (+€15)
        #   T4 η=0.42 mid → MC = 47.6/0.42 + 5 ≈ €118  ← was €95 (+€23)
        # The η compression (T1-T4 spread: 0.32 vs 0.24) means each tier jump is
        # larger in €/MWh — steeper merit-order staircase, fewer CCGT price-setter hours.
        "ES": [
            (0.68, 0.76),   # T1 — MC €68–75  (modern CCGT, SGT5-4000F class)
            (0.58, 0.66),   # T2 — MC €77–87  (mid-fleet 200–350 MW)
            (0.48, 0.56),   # T3 — MC €90–104 (older <200 MW)
            (0.38, 0.46),   # T4 — MC €108–130 (oldest vintage, near CCGT_flex territory)
        ],
        "PT": [
            (0.46, 0.54),   # T1 — MC €93–108  (modern PT CCGT — was 0.50-0.58)
            (0.40, 0.46),   # T2 — MC €108–124 (older PT CCGT — was 0.44-0.50)
        ],
        "FR": [
            # With fr_gas_multiplier=0.85: fuel_adder_FR = MIBGAS×0.85 + CO₂ = 29.3+13.1 = 42.4
            # T1 η=0.60 mid → 42.4/0.60+3 = €73.7  (target €70–80)
            (0.56, 0.64),   # T1 — MC €69–79  (best FR CCGTs — TTF-adjusted gas)
            (0.48, 0.56),   # T2 — MC €79–91  (mid fleet)
            (0.42, 0.48),   # T3 — MC €91–104 (oldest; price-setter in peak demand only)
        ],
    },

    # ─── Merit-order splits for FR/PT aggregated fleets ───────────────────────
    # FR and PT generators are highly aggregated in the base network (1-3 blocks
    # per carrier). Splitting them into capacity-weighted tiers with different MCs
    # gives realistic merit-order staircase resolution and better import/export
    # dynamics at the Pyrenees and Iberian borders.
    #
    # Each tier: fraction of original p_nom + MC range (uniform random draw).
    # Fractions must sum to 1.0.
    #
    # FR nuclear — based on EDF fleet vintage:
    #   30% N4 + best 1300 MW units:  €12–18 variable (fuel cycle + variable O&M)
    #   45% main 1300 MW (P4/P'4):   €18–25
    #   25% oldest 900 MW (CP0/CP1): €25–35 (higher variable O&M)
    #   Nuclear has zero CO₂ cost — no adder applied.
    #   Calibration (2026-05-26): raised from (8-12, 12-17, 17-23) to make FR
    #   more expensive across the board — FR was too cheap, driving wrong IC flows.
    #
    # FR hydro — based on French reservoir typology:
    #   35% alpine seasonal storage (Rhône-Alpes basins): baseload, MC €5–12
    #   35% flexible large reservoir (Dordogne, Lot, Garonne): €14–24
    #   30% pondage / daily-cycle peakers: €28–45
    #   Calibration (2026-05-26): flexible_mc raised from 20→30 (see hydro.per_country.FR).
    #
    # FR CCGT — CO₂ INCLUDED in MC ranges. Calibrated to 2024 TTF (~€28/MWh_th) + CO₂ €60/t.
    #   Physical FR fleet (Landivisiau 843 MW, Bouchain 605 MW, older EDF units) η ≈ 0.56–0.62.
    #   MC_physics_T1 = (28+12.12)/0.59 + 3 ≈ 71 €/MWh.
    #   Ranges here use physical η (NOT boosted like ES) — FR CCGT must stay MORE expensive
    #   than the calibrated ES T1 floor (53–68 €/MWh at MIBGAS=28–34.5) to preserve
    #   the correct border flow direction (FR exports nuclear/hydro, not CCGT).
    #   T1 floor raised to 75 from 65 to keep clear daylight above ES T3 (65–72 €/MWh).
    #   Actual 2024 FR↔ES flows: 56.5% FR→ES, 43.5% ES→FR (bidirectional target).
    #   ccgt_mc_multiplier=1.0 applies directly to these ranges (no scaling).
    #
    # PT hydro — based on REN cascade typology:
    #   45% large Douro/Tejo cascade reservoirs: MC €8–20
    #   35% flexible pondage (Alqueva, Aguieira): €22–35
    #   20% small peaking / run-of-river type:   €38–55
    "merit_splits": {
        "nuclear": {
            "FR": [
                {"fraction": 0.30, "mc_lo": 12.0, "mc_hi": 18.0},
                {"fraction": 0.45, "mc_lo": 18.0, "mc_hi": 25.0},
                {"fraction": 0.25, "mc_lo": 25.0, "mc_hi": 35.0},
            ],
        },
        # hydro: no merit-split tiers — flat low MC applied in _apply_hydro for all
        # hydro (storage_units + generators). LP water shadow price (storage dual)
        # determines the effective clearing price; generators bid at floor only.
        "CCGT": {
            # FR removed — now uses MIBGAS-based time-varying MCs via ccgt_efficiency_tiers.
            # Static merit_splits MCs were causing FR CCGTs to not respond to gas price
            # fluctuations, making them artificially cheap during high-gas periods and
            # driving wrong-direction FR→ES imports.
        },
    },

    # ─── CCGT operational parameters ──────────────────────────────────────────
    # Base network ships p_nom_min = p_nom (must-run at full capacity) for ES
    # CCGTs. We override to zero so CCGTs can dispatch flexibly at partial load.
    # FR/PT CCGTs already have p_nom_min=0.0 and ramp limits in the base network;
    # the per-country overrides below let us tune each market independently.
    # Ramp limits: 1.0 = full output change in one hour (unconstrained).
    "ccgt": {
        "p_min_pu":       0.0,    # minimum stable generation (fraction of p_nom)
        "ramp_limit_up":  0.8,    # fraction of p_nom per hour
        "ramp_limit_down": 0.8,
        # Per-country overrides (None = fall back to top-level values)
        "per_country": {
            "ES": {"p_min_pu": 0.0, "ramp_limit_up": 0.8, "ramp_limit_down": 0.8},
            "PT": {"p_min_pu": 0.0, "ramp_limit_up": 0.8, "ramp_limit_down": 0.8},
            "FR": {"p_min_pu": 0.0, "ramp_limit_up": 0.8, "ramp_limit_down": 0.8},
        },
    },

    # ─── CCGT efficiency tranches ────────────────────────────────────────────
    # Splits each CCGT into multiple tranches with decreasing η (increasing MC).
    # Creates a piecewise-linear cost ramp that prevents trickle-dispatch:
    # the LP fills the cheapest tranche first, committing _mid/_peak only when
    # demand justifies the higher cost.
    # Fractions must sum to 1.0. Set enabled=False to keep current flat-MC.
    "ccgt_tranches": {
        "enabled": True,
        "countries": ["ES"],
        "tranches": [
            {"suffix": "_base", "eta_multiplier": 1.00, "capacity_fraction": 0.40},  # was 0.50
            {"suffix": "_mid",  "eta_multiplier": 0.75, "capacity_fraction": 0.30},  # was 0.80/0.25
            {"suffix": "_peak", "eta_multiplier": 0.50, "capacity_fraction": 0.30},  # was 0.60/0.25
        ],
    },

    # ─── CCGT_Flex split ──────────────────────────────────────────────────────
    # Represents the least-efficient fraction of each ES CCGT used in peaking
    # mode (open-cycle fallback, partial-load, older vintage units).
    # η 0.33–0.42 → always more expensive than regular CCGT T4 (η 0.43–0.47).
    # MC is computed from MIBGAS + CO₂ + VOM, same formula as regular CCGT.
    # Single tier (reduced from 3) — MCQ provides within-unit cost gradation.
    # Reduces flex generator count from ~93 (31×3) to ~31 (31×1).
    "ccgt_flex": {
        "capacity_fraction": 0.20,   # 20% of each ES CCGT p_nom carved out as flex
        "efficiency_tiers": [
            (0.25, 0.32),   # lowered from (0.28,0.35) — even more expensive peaking.
                            # At MIBGAS=34.5, CO₂=60: η=0.32 → MC €154, η=0.25 → MC €195
                            # (was €138-171). Combined with VOM 5→8: +€19-24/MWh.
        ],
        "ramp_limit_pu":     1.0,
    },

    # ─── CCGT must-run (industrial CHP proxy) ────────────────────────────────
    # Spanish cogeneración (~5.6 GW) has a heat obligation making it partially
    # must-run regardless of the electricity price. MC=0 forces LP to always
    # dispatch it first, replicating must-run behaviour without binary commitment.
    # Carved proportionally from ES CCGT fleet; net-neutral (total MW unchanged).
    "ccgt_must_run": {
        "enabled":       True,
        "target_mw":     2000.0,    # Increased to 2 GW to push CCGTs more marginal.
                               # MC=€2 sits below nuclear (€4-8), above VRE (€0.01).
                               # Doesn't set price over VRE — just compresses the
                               # dispatchable fleet, making regular CCGTs run fewer
                               # hours with higher MCQ adder when they do run.
        "marginal_cost": 2.0,      # MC=€2 → above VRE (€0.01), below nuclear (€4-8); VRE sets price
        "p_min_pu":      0.0,
        "ramp_limit_pu": 1.0,
        "color":         "#1E252B",
    },

    # ─── Biomass / cogeneration ────────────────────────────────────────────────
    # Cogeneration (gas-fired industrial CHP + dedicated biomass) has a heat
    # obligation and green-certificate revenue that makes it effectively must-run
    # at near-constant output throughout the year.
    # REE ENTSO-E "Cogeneration" category, 2024: mean dispatch ≈ 601 MW.
    # Source: data/validation/spain_actual_generation_2024.csv.
    #
    # Treatment: treat as a constant 600 MW injection with MC=0 (like CCGT_must_run).
    # p_min_pu = p_max_pu = 1.0 forces the LP to dispatch exactly p_nom = 600 MW
    # every hour — equivalent to removing cogen from the optimization and adding
    # 600 MW to the load offset. No MIP needed.
    "biomass": {
        "enabled":       True,
        "target_mw":     600.0,    # fixed injection, MW — calibrated to 2024 mean dispatch
        "p_min_pu":      1.0,      # must-run at 100% (p_min = p_max = p_nom)
        "marginal_cost": 0.0,      # MC=€0 — priority dispatch (green certificate + heat obligation).
                               # Below VRE (€0.01) so biomass never appears as price setter.
                               # Fixed at p_nom regardless of MC, so this doesn't affect dispatch.
    },

    # ─── Peaker proxy fleet ───────────────────────────────────────────────────
    # Distributed across ES nodes: 70% load-proportional, 30% CCGT-proportional.
    #
    # OCGT: gas-fired simple-cycle units (GE LM6000 / Siemens SGT-800 class).
    #   Capacity: 2 500 MW (REE 2024 ESIOS, "turbinas de gas" category).
    #   η = 0.38 — representative of Spanish OCGT fleet (simple-cycle, no heat recovery).
    #   Source: IEA Technology Perspectives 2023, OCGT electrical efficiency 36–40%.
    #   MC(t) = (MIBGAS(t) + 0.202 × co2_price) / 0.38 + VOM
    #   VOM = €7/MWh_e (higher than CCGT: simple-cycle has ~2× more starts/year,
    #   greater turbine wear; source: NREL ATB 2024 gas peaker O&M).
    #
    # Diesel: gasoil reciprocating engines (island / back-up power).
    #   Capacity: 769 MW (REE 2024 ESIOS, "grupos diesel").
    #   η = 0.35 — gas-oil engine efficiency (MTU/Wärtsilä 50DF class, Wärtsilä 2023).
    #   Fuel cost: diesel/gasoil ~2× MIBGAS on energy basis. Rather than a separate
    #   fuel price CSV, we use a fixed base_mc = fuel_price/η = 160 €/MWh_e
    #   (calibrated to 2024 Spanish gasoil spot ~€56/MWh_th ÷ 0.35).
    #   CO₂ adder applied via co2_intensity["diesel"] = 0.763 tCO₂/MWh_e.
    "peakers": {
        "OCGT_pk": {
            "total_mw": 2500.0,   # REE 2024 ESIOS turbogas fleet
            "eta":        0.30,   # lowered from 0.33 — η penalty ensures OCGT is always
                                  # above CCGT T4. At MIBGAS=34.5, CO₂=60:
                                  # MC = 46.62/0.30 + 15 = €170 (was €151).
            "vom":       15.0,    # raised from 10 — simple-cycle, more maintenance per kWh
            "carrier":  "OCGT",
        },
        "Diesel_pk": {
            "total_mw": 769.0,
            "base_mc":  200.0,   # raised from 180 — fixed fuel €/MWh_e, ensures diesel is
                                  # always the most expensive option (above CCGT_flex/OCGT)
            "vom":        0.0,
            "carrier":  "diesel",
        },
        "load_weight": 0.70,
        "ccgt_weight":  0.30,
    },

    # ─── Nuclear ──────────────────────────────────────────────────────────────
    # Per-country overrides let us tune must-run and ramping independently.
    #   ES: p_min_pu=0.50 (must-run at 50%), ramp=0.10 (20%/h)
    #   FR: p_min_pu=0.30 (less must-run), ramp=0.10 (slower — 10%/h)
    #   PT: no nuclear
    "nuclear": {
        "p_min_pu":       0.30,
        "ramp_limit_pu":  0.05,
        "mc_range":       (8.0, 12.0),  # per-reactor jitter
        "random_seed":    42,
        "per_country": {
            "ES": {"p_min_pu": 0.52, "ramp_limit_pu": 0.03, "p_max_pu": 0.80,
                   "mc_range": (4.0, 8.0)},
            # ES p_max_pu=0.65: Spanish nuclear 2024 actual CF = 64.7% (11.5 TWh / 7408 MW / 8760h).
            # Spring maintenance outages (Mar–Jun) reduce availability below annual average.
            # 0.80 gave 79% CF in Mar–Jun diagnostic → nuclear overdispatch +22% vs actual.
            # 0.65 → max 4,815 MW (7408 × 0.65), floor 2,222 MW (× 0.30). Matches annual CF.
            # ES mc_range: OMIE diagnostic shows nuclear-marginal hours clear at ~€6.9.
            # Fuel-cycle cost (enrichment + waste disposal) ≈ €4–8/MWh_e.
            # Source: NEA/OECD "Projected Costs of Generating Electricity 2020", nuclear
            # variable O&M + fuel cycle €4–9/MWh.
            "FR": {"p_min_pu": 0.33, "ramp_limit_pu": 0.03, "p_max_pu": 0.75,   # raised from 0.62: ceiling 46,050 MW lets nuclear fill baseload without forcing CCGT
                   "mc_range": (25.0, 35.0)},  # FR nuclear MC: fuel-cycle cost + variable O&M.
            # p_max_pu progression: 0.80 → 0.70 → 0.65.
            # At 0.70: FR price rose 38.4 → 48 €/MWh but still −10 vs actual ~58; IC 88% congested.
            # 0.65: ceiling 61,400 × 0.65 = 39,910 MW. FR peaks ~42 GW → CCGT fills 2+ GW peaks
            # → FR prices rise toward €54–58. If overshoots (FR > €65): revert to 0.67.
            # FR p_min_pu = 0.50: 61,400 MW × 0.50 = 30,700 MW baseload floor.
            # Safe margin vs FR min demand floor ~38,280 MW (from infeasibility analysis at p_min=0.60).
            # 0.50 = 30,700 MW nuclear floor + 5 GW RoR/wind = 35,700 MW < 38,280 MW min demand. Safe.
            #
            # p_max_pu=0.62: ceiling 61,400 × 0.62 = 38,068 MW.
            # Lowered from 0.72 (44,208 MW): at 0.72 FR nuclear covered demand too cheaply,
            # pushing FR prices to €25 vs real €54 and inverting IC direction to FR→ES when
            # real Apr-Jul flows are ES→FR (Spain cheaper). At 0.62 FR needs CCGT more often.
        },
    },

    # ─── VRE marginal cost ────────────────────────────────────────────────────
    # Base network ships solar/wind with MC=0. Setting a small positive MC (0.25
    # €/MWh) makes VRE a genuine price-setter in high-penetration hours and gives
    # the LP a tiebreaker that favours local generation over remote generation
    # when line capacity is the binding constraint (avoids phantom long-distance
    # dispatch across congested corridors).
    # Applied to ES generators only — FR/PT VRE retain base-network MC.
    "vre": {
        "marginal_cost": {
            "solar":          0.01,   # €/MWh — near-zero: lets VRE set price in high-penetration hours
            "onwind":         0.01,   # €/MWh — same basis; was 0.25, lowered to increase zero-price hours
            "offwind-ac":     0.01,
            "offwind-dc":     0.01,
            "offwind-float":  0.01,
        },
        # Scale ES solar p_nom to match REE end-2024 installed capacity.
        # ESIOS CSV: 32,350 MW → REE 2024: 39,321 MW → factor = 39,321 / 32,350 ≈ 1.22×
        "solar_capacity_scaler": 1.22,
    },

    # ─── Solar Thermal (CSP) ──────────────────────────────────────────────────
    # 53 CSP plants totalling 2,303 MW (REE "Solar térmica" end-2024).
    # Profiles are pre-computed by Analysis/build_csp_profiles.py via atlite
    # convert_csp() from ERA5 DNI data (parabolic trough, lossless installation).
    #
    # Each bus gets a single StorageUnit (not Generator) with:
    #   - inflow = solar heat collected (MW_electrical-equivalent, post-turbine)
    #   - max_hours = 6.0 h thermal storage (reduced from 7.5 to force same-day cycling)
    #   - efficiency_dispatch = 1.0 (turbine conversion baked into inflow scaling)
    #   - efficiency_store = 0.99 (near-lossless heat-to-tank)
    #   - standing_loss = 0.03 (3%/h — strong thermal decay forces same-day use)
    #   - cyclic_state_of_charge = False (rolling horizon handles SOC)
    #   - initial_soc = 0.50
    #
    # The inflow is scaled so annual sum = target_gwh (5,000 GWh matches
    # Spain 2024 actual "Solar térmica" generation).
    "solar_thermal": {
        "enabled": True,
        "csv_path": "data/CSP_Spain.csv",
        "profile_path": "data/csp_profiles.nc",
        "target_gwh": 5000.0,           # annual solar heat collected (GWh_e)
        "max_hours": 6.0,               # hours of thermal storage at p_nom
        "efficiency_store": 0.99,       # heat-to-tank efficiency
        "efficiency_dispatch": 1.0,     # post-turbine eq. — already in inflow
        "standing_loss": 0.03,          # 3%/h — forces same-day cycling
        "initial_soc": 0.50,            # initial state of charge (fraction)
        "marginal_cost": 0.5,           # €/MWh — tiny, lets LP dispatch freely
        "marginal_cost_storage": 0.0,   # €/MWh — no cost to store
        "p_min_pu": 0.0,                # no must-run on the turbine
    },

    # ─── Hydro ────────────────────────────────────────────────────────────────
    # Reservoir hydro (carrier='hydro') is cheap (MC €22–35) and flexible.
    # Without ramp limits, FR/PT hydro (26.8 GW + 8.3 GW at MC €22–35) can
    # freely undercut Spanish CCGTs. We add ramp limits to constrain them.
    # Run-of-river (carrier='ror') is left unconstrained (must-run when water).
    "hydro": {
        "max_hours":          2000,
        "initial_soc":        0.68,  # Wet 2024 start: Spanish reservoirs were above-average in Jan 2024.
                                      # Back-calculated from real 2024 dispatch + 1.20× inflow (wet year).
        "inflow_multiplier":  1.00,  # set to 1.0 until ESIOS inflow verified; 1.40 caused +77.9% hydro vs real
                                      # raised inflows ~20% above ERA5/ENTSO-E historical baseline.
                                      # Back-calculated from real REE reservoir dispatch vs model inflow deficit:
                                      # model annual inflow 25.6 TWh vs real dispatch 30.9 TWh; 1.20× brings
                                      # annual inflow to ~30.7 TWh, consistent with 2024 wet-year hydrology.
        # Path to ENTSO-E monthly inflow CSV (MWh/month per node, columns = "{bus}_hydro").
        # If the file exists, it overrides per-country inflow_pu for FR and PT.
        # Run fetch_hydro_entsoe.py to regenerate.  Set to null to force inflow_pu fallback.
        "inflow_csv": "data_ES/hydro/inflow_reservoir_monthly_2024.csv",
        "ramp_limit_pu":      0.15,   # default ramp limit for all hydro generators (20%/h — ecological flow + turbine inertia)
        "efficiency_dispatch": 1.0,  # override base-network 0.9 — avoids phantom 11% reservoir drain
        "spill_cost":         10.0,  # re-activated: makes spilling more expensive than dispatching at spring MC (8-20 €/MWh)
                                      # → LP prefers to dispatch rather than spill when reservoir is near-full
        "capacity_scaler":    1.0,  # 19.2% reduction: 17,100 → 13,824 MW (REE 2024)
        # Separate scaler for ES run-of-river generators.
        # ERA5 p_max_pu profiles for ES ror are too conservative: model CF ≈ 24% vs real 44%
        # (model 2.2 TWh vs real 4.0 TWh Apr-Sep 2024). p_nom is scaled to compensate since
        # dispatch = p_nom × p_max_pu and we cannot easily recalibrate the ERA5 CF profiles.
        # Target p_nom: 2,090 MW (base) × 1.75 ≈ 3,658 MW → mean dispatch ≈ 875 MW ≈ real 923 MW.
        "ror_capacity_scaler": 1.65,

        # ── Daisy-chain inflow redistribution ─────────────────────────────────
        # ERA5 distributes runoff proportionally by p_nom within each country
        # (add_electricity.py:777).  Small pondage units get far more inflow-per-MW
        # than large reservoirs — ES0 36 (42 MW) gets 1,564 GWh/yr inflow while
        # ES0 37 (1,354 MW) gets only 16 GWh/yr.  The LP spills the excess because
        # the turbine is 36× undersized.
        #
        # This redirects surplus inflow (above target_cf × p_nom) from source to
        # target, keeping enough for the source to run at ~90% CF and sending the
        # rest to a geographically-close reservoir with spare turbine capacity.
        #
        # Pairs (from diag_hydro_geo.py analysis):
        #   ES0 36 (42 MW, 1,564 GWh/yr) → ES0 37 (1,354 MW, 2,709 GWh e_cap)
        #     keeps 180 GWh/yr (98.6% CF), redirects 1,384 GWh/yr
        #     target new inflow=1,400 GWh/yr, CF=23.9%, fill=51.7% ✅
        #   ES0 20 (14 MW, 135 GWh/yr) → ES0 12 (607 MW, 1,214 GWh e_cap)
        #     keeps 52 GWh/yr (89.8% CF), redirects 83 GWh/yr
        #     target new inflow=190 GWh/yr, CF=7.2%, fill=15.6% ✅
        #   ES0 44 (28 MW, 128 GWh/yr) → ES0 12 (607 MW, 1,214 GWh e_cap)
        #     keeps 98 GWh/yr (82.6% CF), redirects 29 GWh/yr
        #     target new inflow=136 GWh/yr, CF=5.2%, fill=11.2% ✅
        # Total redirected: 1,496 GWh/yr = 1.50 TWh (closes 28% of 5.4 TWh deficit)
        "inflow_redistribution": {
            "enabled": True,
            "target_cf": 0.90,  # source keeps enough inflow for 90% turbine CF
            "pairs": [
                {"source": "ES0 36 hydro", "target": "ES0 37 hydro"},
                {"source": "ES0 20 hydro", "target": "ES0 12 hydro"},
                {"source": "ES0 44 hydro", "target": "ES0 12 hydro"},
            ],
        },

        # ── Inflow-based water values ─────────────────────────────────────────
        # Log-space normalization: MC rises steeply as inflow drops below median,
        # mimicking real operator behaviour (bid up aggressively before near-empty).
        #
        # ES: uses ERA5 hourly inflow from n.storage_units_t.inflow (29 reservoirs).
        # FR/PT: use river discharge proxies (Loire/Tagus) from FR_PT_monthly_flows.csv,
        #        interpolated to hourly.  Absolute scale doesn't matter — log normalisation
        #        is scale-invariant; only seasonal shape is used.
        #
        # Bands calibrated against new CCGT merit order (fr_gas_mult applied):
        #   ES (15,80): dry-season hydro between CCGT T1 (€69-79) and T2 (€79-91)
        #   FR (35,75): Loire overstates winter flow vs Alpine; mc_min=35 → Jan ~€58, below CCGT T1
        #               Aug-Sep ~€72, just above CCGT T1 floor — defensive in drought
        #   PT (45,90): Tagus wet Jan-Apr (960 m³/s) → mc_min=45 gives €45 winter MC
        #               Sep drought hits mc_max=90; PT CCGT T1 ~€95 so hydro always cheaper
        "inflow_mc": {
            "enabled": True,
            "river_proxy_csv": "data_ES/hydro/FR_PT_monthly_flows.csv",
            "countries": {
                "ES": {"window_days":  7, "mc_min": 15.0, "mc_max": 80.0},
                "FR": {"window_days": 28, "mc_min": 35.0, "mc_max": 75.0, "river_proxy_col": 2},  # Loire col
                "PT": {"window_days": 14, "mc_min": 45.0, "mc_max": 90.0, "river_proxy_col": 1},  # Tagus col
            },
        },

        "marginal_cost_gen":  15.6,   # RoR MC — doubled then +30% (×2.6)
        "per_country": {
            "ES": {
                # Initial SOC fractions from ESIOS 2024 monthly reservoir fill data.
                # Used to set state_of_charge_initial when starting a mid-year solve.
                "initial_soc_monthly": {
                    1: 0.44, 2: 0.42, 3: 0.47, 4: 0.53, 5: 0.50,
                    6: 0.45, 7: 0.45, 8: 0.45, 9: 0.35, 10: 0.31,
                    11: 0.41, 12: 0.46,
                },
                # Minimum dispatch (ecological flow / caudal ecológico) derived from
                # real REE 2024 hourly data (spain_actual_generation_2024.csv).
                # Values are P3 of hourly Hydro_Reservoir + Hydro_River dispatch (MW)
                # divided by p_nom (14,900 MW), converted to p_min_pu fraction.
                # P3 chosen over P5 to avoid draining reservoirs — the 3rd percentile
                # captures genuine ecological flow without being distorted by a single
                # anomalous low hour (the absolute min).
                # Source: Analysis/data/spain_actual_generation_2024.csv
                #   Hydro_Reservoir + Hydro_River columns, hourly resolution.
                # Verified 2026-06-03: hourly P3, not daily-average P3.
                "min_dispatch": {
                    "p_min_pu": {
                        1: 0.155, 2: 0.143, 3: 0.283, 4: 0.161,
                        5: 0.141, 6: 0.095, 7: 0.073, 8: 0.060,
                        9: 0.052, 10: 0.106, 11: 0.112, 12: 0.109,
                    },
                    "drought_soc": 0.10,
                },
            },
            "FR": {
                "convert_to_storage": True,
                # Initial SOC from RTE Eco2mix 2024 reservoir fill data (% of usable capacity).
                "initial_soc_monthly": {
                    1: 0.41, 2: 0.49, 3: 0.54, 4: 0.53, 5: 0.46,
                    6: 0.53, 7: 0.61, 8: 0.70, 9: 0.79, 10: 0.93,
                    11: 0.98, 12: 0.95,
                },
                "max_hours":  500,   # reduced from 1700 → was giving 45,638 GWh (3-4× real French usable storage).
                                     # Base network lumps all 26,846 MW of French hydro (reservoir + pondage + RoR
                                     # + pumped storage) into 2 aggregate 'hydro' units. At 1700h the LP effectively
                                     # had unlimited seasonal water — model dispatched 71 TWh vs real 46 TWh.
                                     # At 500h: 26,846 MW × 500h = 13,423 GWh ≈ France's actual alpine + Atlantic
                                     # seasonal reservoir capacity (EDF annual report 2024: ~14,000 GWh usable).
                                     # FR RoR (1,985 MW mean) now added separately via _apply_fr_pt_ror.
                "inflow_pu":  0.192,
                "p_max_pu":   0.7,
                # FR alpine hydro: lower ecological flow obligation than ES.
                # Snowmelt regime means higher minimums in summer (snowmelt),
                # lower in winter. Flat 0.03 as conservative estimate.
                "min_dispatch": {
                    "p_min_pu": 0.03,
                    "drought_soc": 0.10,
                },
            },
            "PT": {
                "convert_to_storage": True,
                # Initial SOC from REN 2024 reservoir fill data (% of usable capacity).
                "initial_soc_monthly": {
                    1: 0.43, 2: 0.44, 3: 0.50, 4: 0.61, 5: 0.64,
                    6: 0.47, 7: 0.36, 8: 0.28, 9: 0.26, 10: 0.32,
                    11: 0.39, 12: 0.42,
                },
                "max_hours":  1200,
                "inflow_pu":  0.105,
                "p_max_pu":   0.60,
                # PT Atlantic hydro: similar ecological flow regime to ES but
                # smaller reservoirs. Flat 0.04 as conservative estimate.
                "min_dispatch": {
                    "p_min_pu": 0.04,
                    "drought_soc": 0.10,
                },
            },
        },
    },

    # ─── FR/PT Run-of-River generators ───────────────────────────────────────────
    # The base PyPSA-Eur 50-node network has NO ror generators for France or Portugal.
    # All French/Portuguese hydro was lumped into 2 aggregate 'hydro' StorageUnits per
    # country, dropping the ~2,000 MW FR RoR and ~413 MW PT RoR entirely.
    #
    # Without FR RoR: FR generation is ~17 TWh/yr short → more CCGT dispatches → higher
    # FR prices → wrong ES↔FR price differential → incorrect interconnector flow direction.
    #
    # Data source: data_ES/hydro/generation_ror_hourly_2024.csv
    # Columns match FR/PT aggregation buses (strip '_hydro' suffix for bus name).
    # p_nom = max(hourly series) per corridor; p_max_pu = actual/p_nom (time-varying).
    # marginal_cost = hydro.marginal_cost_gen (default 6 €/MWh — same as ES ror).
    #
    # Corridor capacities (from 2024 actual data):
    #   FR_WEST: p_nom=1,824 MW  mean=695 MW   total=6.1 TWh/yr
    #   FR_EAST: p_nom=3,388 MW  mean=1,290 MW total=11.3 TWh/yr
    #   PT_NORTH: p_nom=957 MW   mean=247 MW   total=2.2 TWh/yr
    #   PT_CENTRE: p_nom=558 MW  mean=144 MW   total=1.3 TWh/yr
    #   PT_SOUTH:  p_nom=80 MW   mean=21 MW    total=0.2 TWh/yr
    "fr_pt_ror": {
        "enabled":  True,
        "csv_path": "data_ES/hydro/generation_ror_hourly_2024.csv",
    },

    # ─── Wind availability scaler ─────────────────────────────────────────────
    # ERA5 2023 cutout overestimates FR/PT wind speeds for Feb-Apr 2024.
    # Diagnostic: FR onwind CF = 39.2% model vs 30.2% actual (RTE) — +2,079 MW
    # mean over-dispatch. This scaler multiplies onwind p_nom for generators in
    # the specified countries to bring CF closer to reality (capacity reduction).
    # Factor = target_CF / model_CF. For FR: 30.2 / 39.2 ≈ 0.77.
    "wind_availability": {
        "per_country": {
            "ES": 1.1, # 1.20 (ERA5 CF correction: model 20.9% → real ~25%) ×
                        # 1.10 (+10% capacity expansion test across ES fleet).
                        # 32.1 GW installed × 1.32 = 42.4 GW effective p_nom.
                        # Expected mean dispatch: 42.4 GW × 20.9% CF ≈ 8,860 MW
                        # (+~800 MW vs ERA5-only 1.20 baseline of ~8,060 MW).
            "FR": 1.0,  # FR onwind at 1.0 for now (ERA5 FR bias fixed earlier)
            # PT: no adjustment needed yet (PT wind CF was closer to actual)
        },
    },

    # ─── BESS fleet (2030 scenario) ───────────────────────────────────────────
    # Set enabled=True for the 2030 PNIEC scenario; keep False for 2024 baseline.
    # Each project is mapped to its closest ES network bus via haversine distance.
    # Canary Islands projects (lat ≤ canary_lat_max) are automatically excluded.
    #
    # Technology mapping:
    #   Li-ion / Stand-alone → carrier='battery'
    #   PHS                  → carrier='PHS_new'   (new greenfield; separate from existing)
    #   Termico              → carrier='thermal_storage'
    "bess_fleet": {
        "enabled":              False,   # ← flip to True for 2030 scenario
        "csv_path":             "Analysis/data/2024_batteries.csv",
        "canary_lat_max":       29.5,    # exclude projects at lat ≤ 29.5 °N
        "soc_initial_fraction": 0.50,    # start all units at 50% SOC
        "technology_map": {
            "Li-ion":      "battery",
            "Stand-alone": "battery",
            "PHS":         "PHS_new",
            "Termico":     "thermal_storage",
        },
    },

    # ─── PHS operational friction ─────────────────────────────────────────────
    # Base network has PHS MC=0, making it a perfect price-smoother. Adding
    # friction prevents it arbitraging tiny price spreads and washing out peaks.
    # Efficiency is already correctly set at 0.866 (75% round-trip) in base network.
    "phs": {
        "marginal_cost":         10.00,  # €/MWh dispatch friction (wear & tear)
        "marginal_cost_storage": 4.00,  # €/MWh pumping friction (asymmetric — pumping is cheaper)
        "p_max_pu":              1.0,  # cap max dispatch AND store to 60% of p_nom per hour
        # ramp_limit_dispatch removed — PHS ramp constraints were inflating LP size significantly
    },

    # ─── Borders ──────────────────────────────────────────────────────────────────
    # Cross-border connections are now modelled as proper AC Lines (PT, FR_WEST,
    # FR_EAST Vic–Baixas) + one bidirectional DC Link (INELFE HVDC).
    # The original DC_ic Link pairs are removed by _apply_border_ac_dc() in refinery.
    #
    # ic_factor: applied to ALL border connections (AC Line s_nom + INELFE p_nom).
    #   1.0 = full physical capacity  |  <1.0 = simulate security margins / outages
    "borders": {
        "ic_factor": 0.95,
        # No DC_ic link entries remain — PT and FR are now AC Lines (see border_ac_lines).
        # Balearic HVDC cable (carrier="DC") is untouched by this block.
    },

    # ─── Cross-border AC Lines ─────────────────────────────────────────────────
    # All physically AC overhead 400 kV connections between ES and PT/FR.
    # Replaces the old DC_ic transport-model Link pairs.
    #
    # x (Ω) calibrated to match the internal ES line convention from PyPSA-Eur:
    #   x = 0.35 rad × V_nom² / s_nom  →  at s_nom capacity the angle spread is ~20°
    #   This is consistent with other ES lines (e.g. 60.8 Ω / 1251 MW / 260 km).
    # s_nom: symmetric thermal rating; ic_factor applied multiplicatively.
    # r (Ω): ≈ 0.12 × x (typical r/x ratio for 380 kV bundled conductors).
    #
    # Portugal–Spain (3 corridors, all 400 kV AC overhead):
    #   PT_NORTH  — Alto Lindoso–Cartelle + Lagoaça–Aldeadávila (dominant corridor)
    #               2 parallel 400 kV circuits, ~135 km → s_nom 2,000 MW
    #   PT_CENTRE — Falagueira–Cedillo 400 kV (single circuit, ~200 km)
    #               s_nom 490 MW — the smallest of the 3 corrii adors
    #   PT_SOUTH  — Alqueva–Brovales + Tavira–Puebla de Guzmán (2 circuits, ~240 km)
    #               s_nom 940 MW
    #   Total 2024 observed: ES→PT 3,712 MW, PT→ES 3,090 MW (AC Lines are symmetric;
    #   asymmetry is approximated by the LP routing given price signals)
    #
    # Spain–France western (all AC, Atlantic/Central Pyrenees):
    #   FR_WEST   — Hernani–Argia + Arkale–Argia + Biescas–Pragnères
    #               Multiple parallel short paths (30–120 km), 400 kV + 220 kV
    #               Aggregate s_nom 1,400 MW; lower x reflects parallel redundancy.
    #
    # Spain–France eastern AC component (Vic–Baixas only):
    #   FR_EAST_AC — Vic–Baixas 400 kV AC, single circuit, ~175 km, 290 MW
    #               The dominant 2,000 MW INELFE HVDC is in the "inelfe" block below.
    "border_ac_lines": {
        "PT_NORTH":   {"bus0": "ES0 27", "bus1": "PT_NORTH",  "s_nom": 2000.0, "x":  25.3, "r":  3.0},
        "PT_CENTRE":  {"bus0": "ES0 10", "bus1": "PT_CENTRE", "s_nom":  490.0, "x": 103.1, "r": 12.4},
        "PT_SOUTH":   {"bus0": "ES0 23", "bus1": "PT_SOUTH",  "s_nom":  940.0, "x":  53.8, "r":  6.5},
        # FR AC lines: marginal_cost adds light friction to discourage cheap FR power flooding ES
        # without penalising legitimate ES→FR exports. Set to 1.0 (was 5.0) — just enough to bias
        # the LP against routing FR surplus through AC lines when DC is cheaper, but not enough to
        # block ES exports when ES is genuinely cheaper than FR.
        "FR_WEST":    {"bus0": "ES0 24", "bus1": "FR_WEST",   "s_nom": 1400.0, "x":  36.1, "r":  4.3, "marginal_cost": 1.0},
        "FR_EAST_AC": {"bus0": "ES0 43", "bus1": "FR_EAST",   "s_nom":  290.0, "x": 174.3, "r": 20.9, "marginal_cost": 1.0},
    },

    # ─── INELFE HVDC Link (ES0 43 ↔ FR_EAST) ─────────────────────────────────
    # Santa Llogaia (ES) – Baixas (FR): two ±320 kV VSC cables, 1,000 MW each.
    # Bidirectional (p_min_pu=-1.0): operates heavily in both directions year-round.
    # Efficiency=0.97: ~3% VSC converter station loss per direction.
    # 2024 data shows max observed FR↔ES total flow of 3,686 MW (Jan 4) and
    # 3,626 MW (Jan 7) — the full ~3,690 MW physical capacity was reached in both
    # directions during the same cold week, confirming the 2,000 MW DC component.
    "inelfe": {
        "enabled":    True,
        "bus0":       "ES0 43",
        "bus1":       "FR_EAST",
        "p_nom":      2000.0,    # 2 × 1,000 MW ±320 kV VSC cables
        "efficiency": 0.97,      # ~3% VSC converter loss
        "p_min_pu":   -1.0,      # fully bidirectional (FR→ES at full p_nom)
        "marginal_cost": 1.0,    # €/MWh light friction — reduced from 5.0 to avoid penalising ES→FR exports
    },

    # ─── French missing demand (non-Spain export routes)
    # France exports to Italy, Germany, Belgium, UK, Switzerland — not just Spain.
    # This CSV contains the hourly net export to those missing routes (MW).
    # The demand is weighted by each FR node's annual total and added to its p_set.
    "fr_missing_demand": {
        "csv_path": "Analysis/data/FR_missing_demand.csv",
        "column":   "FR_net_export",
        "enabled":  True,   # set False to skip without deleting code
    },

    # ─── VOLL ─────────────────────────────────────────────────────────────────
    # Set to None to disable load-shedding generators entirely.
    # When enabled, refinery adds a VOLL generator at each ES bus with MC = voll.
    "voll": 300,   # EUR/MWh — reduced from 3000 to prevent mean-price contamination.
                   # Maths: V × (3000 − 46) / 2136 = 5.7 → just 4 VOLL hours at 3000 explains
                   # the entire +5.7 €/MWh mean error. 2024 OMIE actual max ≈ 193 €/MWh;
                   # 300 provides a realistic scarcity ceiling (20 VOLL hours × 254/2136 ≈ 2 €/MWh
                   # mean contamination — negligible). Prevents tight-headroom overnight hours
                   # from spiking to 3000 when the IC tries to reverse direction.

    # ─── Transmission ─────────────────────────────────────────────────────────
    "transmission": {
        "trans_factor": 0.9,   # scale internal ES line s_nom by this factor
                                # (applied in refinery — reduces thermal capacity)
                                # 0.40 × 355 GVA → 142 GVA total ES capacity 
        "s_max_pu":     0.9,    # copper-plate approximation: no thermal congestion on internal lines.
                                # 0.80 created solar-node pockets of near-zero price that couldn't
                                # propagate to demand centers, suppressing the national near-zero count.
                                # With 1.0, VRE surplus equalises across all ES nodes → system price
                                # drops to near-zero whenever national VRE > demand+nuclear+hydro.
        # HVDC cable loss: applied to links where carrier=="DC" (Balearic cable only).
        # FR/PT interconnectors use carrier "DC_ic export/import" and are LP modeling
        # artifacts for directional capacity bounds — those are AC overhead lines and
        # are NOT touched by this parameter.
        # 0.97 = 3% loss on the ~230 km ES0 5 → ES1 0 (Balearic Islands) HVDC cable.
        "dc_loss_efficiency": 0.94,
    },

    # ─── Validation run ───────────────────────────────────────────────────────
    # Set start_date (YYYY-MM-DD) and n_days to define the analysis window.
    # The script slices the full-year network to exactly this range before solving.
    "validation": {
        "network_path": "resources/networks/base_s_50_elec_2704_fixed.nc",
        "start_date":   "2024-11-01",   # ← full-year start
        "n_days":       3,
        "solver":       "gurobi",
        # Pure LP — barrier is fastest for 8760h LP.
        # Method:2 = barrier. Crossover:0 = interior-point duals returned directly
        # (no simplex post-processing). Interior-point duals are well-defined for LP
        # and avoid the phantom €2000/MWh prices from degenerate simplex crossover.
        # NumericFocus:1 + DualReductions:0 keeps the barrier math clean while
        # preserving degenerate duals for correct nodal prices.
        # ScaleFlag:2 = aggressive scaling — large LP with MW/€/GWh mix is poorly conditioned.
        # BarConvTol:1e-7 = slightly relaxed convergence (default 1e-8) saves a few barrier iters.
        "solver_options": {
            "Threads":        8,
            "Method":          2,     # barrier — fastest for large LP
            "Presolve":        2,     # aggressive presolve
            "Crossover":       0,     # interior-point duals directly (no simplex crossover)
            "ScaleFlag":       2,     # aggressive scaling — helps LP conditioning
            "BarConvTol":      1e-7,  # slightly relaxed (default 1e-8) — saves barrier iters
            "DualReductions":  0,     # preserve degenerate duals for nodal prices
            "NumericFocus":    1,
        },
        "omie_csv":            "Analysis/data/Spain_prices.csv",
        "france_prices_csv":   "Analysis/data/France_prices.csv",
        "portugal_prices_csv": "Analysis/data/Portugal_prices.csv",
        "real_dispatch_csv": "Analysis/data/spain_actual_generation_2024.csv",
        "real_flows_fr_csv": "Interconnector_ENTSOE_pull/data/FR_ES_cross_border_flows_2024.csv",
        "real_flows_pt_csv": "Analysis/interconnector_analysis/2024_PT_ES_balance_hourly.csv",
        "output_dir":   "Analysis/validation_output",
    },
}
