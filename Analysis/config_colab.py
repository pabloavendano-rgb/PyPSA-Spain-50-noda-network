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
    CCGT tier 1 (largest/newest, ≥350 MW):  η 0.57–0.60
    CCGT tier 2 (mid-fleet, 200–350 MW):    η 0.52–0.57
    CCGT tier 3 (older/<200 MW):            η 0.46–0.52
      Source: IEA Technology Perspectives 2023; Spanish CCGT fleet built 2000–2010
              (Siemens SGT5-4000F / GE 9F.04 series), typical η 0.55–0.60 design,
              degraded ~5% at age 15+ years.
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
    "co2_price": 60.0,  # EUR/tCO₂  (EU ETS — annual avg 2024 ≈ €62; 60 used here
                         # to remove conservative rounding-up buffer)

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

    # ─── Gas fuel prices (MIBGAS / PVB day-ahead) ────────────────────────────
    # Daily MIBGAS PVB index for 2024 — the primary fuel cost input for all
    # gas-fired generators (CCGT, CCGT_flex, OCGT).
    # Refinery broadcasts daily price to hourly snapshots and computes per-unit
    # MC using each generator's efficiency η.
    "gas_prices_csv": "Analysis/data/gas_prices_daily.csv",

    # Variable O&M by gas carrier (€/MWh_e) — cost above fuel + CO₂.
    "gas_vom": {
        "CCGT":      3.0,
        "CCGT_flex": 1.0,   # same physical plant as base CCGT; η penalty already captures part-load cost
        "OCGT":      7.0,   # higher: simple-cycle, more maintenance per kWh
    },

    # Toggle: True = quadratic dispatch cost (QP, slower but within-unit ramp shape).
    #         False = pure linear tiers (LP, fast, fleet supply curve only).
    # When False, gas_mcq_alpha values are ignored entirely.
    "use_mcq": True,

    # Quadratic dispatch cost — only applied when use_mcq=True.
    # MCQ = alpha / p_nom → ∂Cost/∂p (€/MWh) = MC_linear + 2 × alpha × (p / p_nom)
    # Uplift at p_nom = 2 × alpha €/MWh.
    #
    # Calibration note (2026-05-26): MC × 0.80 makes the linear base cheaper
    # ("less aggressive at lower end"), while MCQ alpha=12.0 gives a steeper
    # ramp ("more aggressive with MCQ"). Combined effect:
    #   At p_min=0.22: uplift = 2×12×0.22 = €5.3/MWh (small — cheap at low dispatch)
    #   At p_nom=1.0:  uplift = 2×12×1.0  = €24/MWh (significant — expensive at full output)
    "gas_mcq_alpha": {
        "CCGT":       12.0,  # +24 €/MWh uplift at p_nom (was 6.0 → +12, too flat)
        "CCGT_flex":  3.0,
        "OCGT":       3.0,
    },

    # Global CCGT marginal cost multiplier — corrects the +26 €/MWh bias.
    # Applied to ALL CCGTs (ES MIBGAS-based + FR/PT merit_splits) after
    # all other MC computations are complete. 0.80 = 20% reduction.
    # Source: changes_log.md recommendation: "n.generators.loc[ccgt_mask, 'marginal_cost'] *= 0.80"
    "ccgt_mc_multiplier": 0.80,

    # ─── CCGT efficiency tiers ─────────────────────────────────────────────────
    # Generators sorted by p_nom descending → largest = most modern = best η.
    # Equal-MW splits across tiers. Gap in η space = price cliff (hockey-stick).
    #
    # MC at MIBGAS=36, CO₂=60: (36 + 0.202×60) / η + VOM = 48.12 / η + 3
    #
    # ES — 6 tiers, calibrated to Spanish F-class CCGT fleet (built 2000–2010).
    # Source: IEA TP 2023; Siemens SGT5-4000F / GE 9F.04 design η 0.57–0.60,
    #   degraded ~3–5 pp at age 15+. Matches config docstring (η 0.57–0.60 T1).
    # T1–T4 cheap cluster; ~15 €/MWh gap to T5–T6.
    # At MIBGAS=28, CO₂=60: fuel_adder = 28 + 0.202×60 = 40.12 €/MWh_th
    #   T1  η 0.57–0.60 → MC  71.9– 73.5  ← large/modern; T2 is marginal at mean OMIE
    #   T2  η 0.52–0.57 → MC  73.5– 80.2
    #   T3  η 0.47–0.52 → MC  80.2– 88.4
    #   T4  η 0.43–0.47 → MC  88.4– 96.3  ← end cheap cluster
    #   [gap η 0.40–0.43: ~8 €/MWh cliff]
    #   T5  η 0.34–0.40 → MC 103.3–120.9  ← expensive
    #   T6  η 0.27–0.34 → MC 120.9–151.6  ← worst/peaker
    "ccgt_efficiency_tiers": {
        "ES": [
            (0.57, 0.60),   # T1 — large/modern (SGT5-4000F, GE 9F.04)
            (0.52, 0.57),   # T2 — mid fleet
            (0.47, 0.52),   # T3 — smaller/older combined-cycle
            (0.43, 0.47),   # T4  ← end cheap cluster
            (0.34, 0.40),   # T5  ← cliff jump
            (0.27, 0.34),   # T6 — worst/peaker
        ],
        # PT — 3 tiers. Portuguese CCGT fleet (EDP Pego, Tapada do Outeiro class)
        #   similar vintage to Spanish fleet; slightly older on average.
        #   T1 η 0.54–0.58 → MC  72.6– 77.6  (inline with ES T1–T2)
        #   T2 η 0.47–0.54 → MC  77.6– 88.3  ← end cheap cluster
        #   T3 η 0.30–0.40 → MC 103.3–136.9  ← expensive
        "PT": [
            (0.54, 0.58),   # T1
            (0.47, 0.54),   # T2  ← end cheap cluster
            (0.30, 0.40),   # T3 — expensive
        ],
    },

    # CCGT detection threshold — if post-refinement mean MC exceeds this,
    # CO₂ has already been applied (used to skip FR which is handled via merit_splits).
    "ccgt_co2_threshold": 82.0,

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
    #   30% N4 + best 1300 MW units:  €8–12 variable (fuel cycle + variable O&M)
    #   45% main 1300 MW (P4/P'4):   €12–17
    #   25% oldest 900 MW (CP0/CP1): €17–23 (higher variable O&M)
    #   Nuclear has zero CO₂ cost — no adder applied.
    #
    # FR hydro — based on French reservoir typology:
    #   35% alpine seasonal storage (Rhône-Alpes basins): baseload, MC €5–12
    #   35% flexible large reservoir (Dordogne, Lot, Garonne): €14–24
    #   30% pondage / daily-cycle peakers: €28–45
    #
    # FR CCGT — split with CO₂ ALREADY INCLUDED in MC ranges. RAISED relative to
    #   ES CCGT because French gas supply is less flexible (no Algeria pipeline,
    #   less LNG terminal capacity) and the French CCGT fleet is smaller/less
    #   efficient on average. Spanish CCGTs benefit from MIBGAS hub liquidity.
    #   Without this premium, the LP sees FR CCGT as cheaper than ES CCGT and
    #   dispatches FR first → ES imports from FR (reversing the real flow).
    #   40% modern efficient (Landivisiau, Bouchain): base €62–75 + CO₂ €22.6 → €85–98
    #   40% mid-efficiency fleet:                    base €78–90 + CO₂       → €101–113
    #   20% older / peaking units:                   base €95–110 + CO₂      → €118–133
    #
    # PT hydro — based on REN cascade typology:
    #   45% large Douro/Tejo cascade reservoirs: MC €8–20
    #   35% flexible pondage (Alqueva, Aguieira): €22–35
    #   20% small peaking / run-of-river type:   €38–55
    "merit_splits": {
        "nuclear": {
            "FR": [
                {"fraction": 0.30, "mc_lo": 12.0, "mc_hi": 18.0},   # was 8-12
                {"fraction": 0.45, "mc_lo": 18.0, "mc_hi": 25.0},   # was 12-17
                {"fraction": 0.25, "mc_lo": 25.0, "mc_hi": 35.0},   # was 17-23
            ],
        },
        # hydro: no merit-split tiers — flat low MC applied in _apply_hydro for all
        # hydro (storage_units + generators). LP water shadow price (storage dual)
        # determines the effective clearing price; generators bid at floor only.
        "CCGT": {
            "FR": [
                # Raw values PRE ×0.80 multiplier. After ×0.80: €80–95, €95–110, €110–130.
                {"fraction": 0.40, "mc_lo": 100.0, "mc_hi": 119.0},   # modern — was 70-82
                {"fraction": 0.40, "mc_lo": 119.0, "mc_hi": 138.0},   # mid fleet — was 82-96
                {"fraction": 0.20, "mc_lo": 138.0, "mc_hi": 163.0},   # older/peaking — was 96-115
            ],
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
        "ramp_limit_up":  0.55,    # fraction of p_nom per hour
        "ramp_limit_down": 0.55,
        # Per-country overrides (None = fall back to top-level values)
        "per_country": {
            "ES": {"p_min_pu": 0.0, "ramp_limit_up": 0.55, "ramp_limit_down": 0.55},
            "PT": {"p_min_pu": 0.0, "ramp_limit_up": 0.55, "ramp_limit_down": 0.55},
            "FR": {"p_min_pu": 0.0, "ramp_limit_up": 0.55, "ramp_limit_down": 0.55},
        },
    },

    # ─── CCGT_Flex split ──────────────────────────────────────────────────────
    # Represents the least-efficient fraction of each ES CCGT used in peaking
    # mode (open-cycle fallback, partial-load, older vintage units).
    # η 0.38–0.44 → always more expensive than regular CCGT tier 3 (η 0.46–0.52).
    # MC is computed from MIBGAS + CO₂ + VOM, same formula as regular CCGT.
    "ccgt_flex": {
        "capacity_fraction": 0.20,   # 20% of each ES CCGT p_nom carved out as flex
        # 3 tiers for flex units — part-load / open-cycle regime, slot above T4 cliff.
        # Each CCGT spawns 3 flex sub-units (capacity_fraction / 3 each).
        # MC at MIBGAS=36, CO₂=60 (VOM=1):
        #   TF1 η 0.44–0.49 → MC  99.3–110.5  (just above T4 cliff)
        #   TF2 η 0.39–0.44 → MC 110.5–124.4  (interleaves with T5)
        #   Single tier (reduced from 3) — MCQ provides within-unit cost gradation.
        #   Reduces flex generator count from ~93 (31×3) to ~31 (31×1).
        "efficiency_tiers": [
            (0.33, 0.42),   # single tier — above T4 cliff (η<0.43); MCQ provides gradation
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
        "target_mw":     800.0,    # Step 2 calibration: trimmed from 1200 to isolate floor effect.
                               # Spanish cogeneración ~1.2 GW total, but not all is MC=0 must-run;
                               # 800 MW is a conservative estimate of the firmly obligation-bound share.
        "marginal_cost": 0.0,      # MC=0 → always dispatched (must-run proxy)
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
        "marginal_cost": 0.0,      # zero MC → always dispatched first; no price impact
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
            "eta":        0.38,   # simple-cycle electrical efficiency (IEA 2023)
            "vom":        7.0,    # variable O&M €/MWh_e (NREL ATB 2024)
            "carrier":  "OCGT",
        },
        "Diesel_pk": {
            "total_mw": 769.0,
            "base_mc":  100.0,   # fixed fuel €/MWh_e (gasoil ~€56/MWh_th ÷ η 0.35)
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
        "mc_range":       (12.0, 18.0),  # per-reactor jitter
        "random_seed":    42,
        "per_country": {
            "ES": {"p_min_pu": 0.30, "ramp_limit_pu": 0.03, "p_max_pu": 0.8,
                   "mc_range": (4.0, 8.0)},   # p_min_pu 0.40→0.30: Step 1 calibration
            # ES mc_range: OMIE diagnostic shows nuclear-marginal hours clear at ~€6.9.
            # Fuel-cycle cost (enrichment + waste disposal) ≈ €4–8/MWh_e.
            # Source: NEA/OECD "Projected Costs of Generating Electricity 2020", nuclear
            # variable O&M + fuel cycle €4–9/MWh.
            "FR": {"p_min_pu": 0.25, "ramp_limit_pu": 0.03, "p_max_pu": 0.85},
            # FR p_max_pu=0.80: 0.90 made FR too cheap (FR shadow 54.6 < ES 60.0),
            # reversing FR→ES flow direction vs actual (model +206 MW, actual -720 MW ES→FR).
            # 0.80 = 49,096 MW cap → FR needs ES imports in more hours → correct flow direction.
            # after ASN inspection backlog cleared. Jan 2024 peak CF = 78.6% (48,287 MW).
            # Old 0.60 (36,840 MW cap) was severely under-serving France, causing the model
            # to import expensive Spanish gas in many hours → FR mean price was €128/MWh
            # vs real EPEX France 2024 ~€55-65/MWh. 0.75 = 46,050 MW ≈ 2024 annual average.
            # Effect: France becomes self-sufficient in more hours, stops pulling Spanish
            # CCGT online, allows more VRE to set prices in Spain.
        },
    },

    # ─── Hydro ────────────────────────────────────────────────────────────────
    # Reservoir hydro (carrier='hydro') is cheap (MC €22–35) and flexible.
    # Without ramp limits, FR/PT hydro (26.8 GW + 8.3 GW at MC €22–35) can
    # freely undercut Spanish CCGTs. We add ramp limits to constrain them.
    # Run-of-river (carrier='ror') is left unconstrained (must-run when water).
    "hydro": {
        "max_hours":          1700,
        "initial_soc":        0.50,   # fraction of energy capacity
        "inflow_multiplier":  1.0,   # 2024 wet-year adjustment
        "ramp_limit_pu":      0.90,   # default ramp limit for all hydro generators
        "capacity_scaler":    0.95,  # 19.2% reduction: 17,100 → 13,824 MW (REE 2024)

        # Soft terminal SOC penalty: penalise ES reservoirs for finishing below
        # their initial SOC. Adds penalty €/MWh × shortfall to the objective so
        # the LP avoids end-of-horizon liquidation without risking infeasibility.
        # Set to 0 to disable.
        #
        # Calibration (2026-05-26): reduced from 50→20→10 so the LP is even less
        # afraid of draining reservoirs in a wet-spring window. With inflows
        # corrected, reservoirs should be filling naturally — a high penalty just
        # withholds water that could displace expensive thermal.
        "terminal_soc_penalty": 10.0,

        # ── SOC-tiered MC for reservoir storage_units ─────────────────────────
        # Water opportunity cost rises as reservoirs deplete. The LP sees a higher
        # MC when SOC is low, encouraging conservation; a lower MC when full,
        # allowing cheap dispatch. Applied per-unit based on state_of_charge_initial
        # (re-applied each rolling-horizon window as SOC evolves).
        #
        # Tiers are evaluated highest-soc_min first; first match wins.
        # Calibration (2026-05-26): lowered from (5, 15, 30) → (3, 8, 18) →
        # (2, 6, 15). In a wet period real water value is near zero across most
        # SOC; even €3/8/18 was steep. Combined with terminal_soc_penalty=10
        # (was 20), hydro can displace expensive thermal more freely.
        "soc_mc_tiers": [
            {"soc_min": 0.60, "mc":  2.0},   # SOC ≥ 60%: full — dispatch aggressively (was 3)
            {"soc_min": 0.30, "mc":  6.0},   # SOC 30–60%: medium — selective dispatch (was 8)
            {"soc_min": 0.00, "mc": 15.0},   # SOC < 30%: low — conserve water (was 18)
        ],

        "marginal_cost_gen":  6.0,   # flat MC for hydro generators (FR/PT reservoir gen,
                                      # ES ror). Generators have no SOC so tiers don't apply;
                                      # this floor prevents trivial undercutting.
        "per_country": {
            "ES": {
                "ramp_limit_pu": 0.35,   # tightened from 0.7 — less flexible, prevents hydro spiking
            },
            "FR": {
                "ramp_limit_pu": 0.7,
                "p_max_pu": 0.5,
                # FR p_max_pu=0.65: allows realistic peak dispatch (RTE 2024 +8% vs average).
                "use_flexible_mc": True,   # True → use flexible_mc; False → global marginal_cost_gen
                "flexible_mc": 15.0,       # €/MWh — lowered from 30 (2026-05-26): FR hydro was setting a fixed price floor
                # marginal_cost_gen (6.0) is the fallback when use_flexible_mc=False.
            },
            "PT": {
                "ramp_limit_pu": 0.4,
                "p_max_pu": 0.20,   # tightened from 0.35 — caps PT hydro, forces PT to import from ES
                # PT hydro is a generator (no SOC tracking); uses global marginal_cost_gen.
            },
        },
    },

    # ─── Wind availability scaler ─────────────────────────────────────────────
    # ERA5 2023 cutout overestimates FR/PT wind speeds for Feb-Apr 2024.
    # Diagnostic: FR onwind CF = 39.2% model vs 30.2% actual (RTE) — +2,079 MW
    # mean over-dispatch. This scaler multiplies onwind p_nom for generators in
    # the specified countries to bring CF closer to reality (capacity reduction).
    # Factor = target_CF / model_CF. For FR: 30.2 / 39.2 ≈ 0.77.
    "wind_availability": {
        "per_country": {
            "FR": 0.9,  # reduce FR onwind p_nom by 23% to match RTE Feb-Apr 2024 CF
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
        "enabled":              True,   # ← flip to True for 2030 scenario
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

    # ─── Rolling horizon ──────────────────────────────────────────────────────
    # DISABLED: rolling horizon caused persistent window-2 infeasibility due to
    # SoC chaining artifacts. Reverted to single full-horizon dispatch.
    # The hydro inflow parameters (max_hours, capacity_scaler, initial_soc,
    # inflow_multiplier, per_country ES/FR/PT) are still applied by the refinery.
    "rolling_horizon": {
        "enabled":     False,
        "window_days": 14,
        "step_days":   14,
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
        "FR_WEST":    {"bus0": "ES0 24", "bus1": "FR_WEST",   "s_nom": 1400.0, "x":  36.1, "r":  4.3},
        "FR_EAST_AC": {"bus0": "ES0 43", "bus1": "FR_EAST",   "s_nom":  290.0, "x": 174.3, "r": 20.9},
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
    },

    # ─── French missing demand (non-Spain export routes) ───────────────────────
    # France exports to Italy, Germany, Belgium, UK, Switzerland — not just Spain.
    # This CSV contains the hourly net export to those missing routes (MW).
    # The demand is weighted by each FR node's annual total and added to its p_set.
    "fr_missing_demand": {
        "csv_path": "Analysis/data/FR_missing_demand.csv",
        "column":   "FR_net_export",
        "enabled":  True,   # set False to skip without deleting code
    },

    # ─── French demand scaler ──────────────────────────────────────────────────
    # Applied AFTER fr_missing_demand is added to FR loads (so it scales the
    # combined domestic + export demand). Factor > 1.0 increases total FR demand
    # (France imports more / exports less). Factor < 1.0 has the opposite effect.
    # Set to 1.0 (or remove the key) for no scaling.
    # Diagnostic use: if FR exports to ES are too high, try e.g. 1.05 to add 5%
    # demand — the LP will push more ES→FR flow to satisfy the gap.
    "fr_demand_scaler": 1.1,

    # ─── VOLL ─────────────────────────────────────────────────────────────────
    # Set to None to disable load-shedding generators entirely.
    # When enabled, refinery adds a VOLL generator at each ES bus with MC = voll.
    "voll": 3000,  # EUR/MWh — reduced from 10000; tightens objective range for barrier convergence

    # ─── Transmission ─────────────────────────────────────────────────────────
    "transmission": {
        "trans_factor": 0.9,   # scale internal ES line s_nom by this factor
                                # (applied in refinery — reduces thermal capacity)
                                # 0.40 × 355 GVA → 142 GVA total ES capacity
        "s_max_pu":     0.80,   # thermal limit applied to all internal ES lines
                                # 0.70 → no additional restriction beyond s_nom scaling
        # HVDC cable loss: applied to links where carrier=="DC" (Balearic cable only).
        # FR/PT interconnectors use carrier "DC_ic export/import" and are LP modeling
        # artifacts for directional capacity bounds — those are AC overhead lines and
        # are NOT touched by this parameter.
        # 0.97 = 3% loss on the ~230 km ES0 5 → ES1 0 (Balearic Islands) HVDC cable.
        "dc_loss_efficiency": 0.97,
    },

    # ─── Unit Commitment (MIP) ────────────────────────────────────────────────
    # Set enabled=True to activate binary unit commitment for thermal generators.
    # Gurobi handles a 1-week MIP in seconds; a 90-day MIP takes a few minutes.
    # Toggle enabled: False to revert to pure LP dispatch with no code changes.
    "mip": {
        "enabled": False,       # disabled — MCQ provides within-unit shaping; MIP adds B&B overhead
        "countries": ["ES"],      # country prefixes to apply UC to (add "FR" later if needed)
        "carriers":  ["CCGT"],    # biomass excluded — floor/ceiling via p_min_pu/p_max_pu is sufficient
        "CCGT": {
            "n_top_mip":       4,      # commit only the 4 largest ES CCGTs (T1 by p_nom);
                                       # remaining 27 units LP-dispatchable with fixed linear MC
            "p_min_pu":        0.22,
            "min_up_time":     1,
            "min_down_time":   1,
            "start_up_cost":   10000,
            "up_time_before":  0,
            # Ramp 0.60: a committed 400 MW CCGT can change output by 240 MW/hr —
            # consistent with ~10-15 MW/min physical ramp rate (combined-cycle).
            # This prevents the LP from instantly zeroing out CCGTs during short VRE
            # peaks (solar noon), keeping some gas online for load-following realism.
            # Safe with p_min_pu=0.30: ramp_up=0.60 > p_min so startup is feasible in 1h.
            # CCGT_flex and OCGT are NOT in mip.carriers so their ramps are unaffected.
            "ramp_limit_up":   0.60,
            "ramp_limit_down": 0.60,
        },
        "OCGT": {
            "p_min_pu":        0.20,
            "min_up_time":     1,
            "min_down_time":   1,
            "start_up_cost":   15000,
            "up_time_before":  0,      # peakers start cold
            "ramp_limit_up":   1.0,
            "ramp_limit_down": 1.0,
        },
        # Biomass is handled by LP floor/ceiling (p_min_pu in biomass config block),
        # not by MIP unit commitment — removed from carriers list above.
    },

    # ─── Validation run ───────────────────────────────────────────────────────
    # Set start_date (YYYY-MM-DD) and n_days to define the analysis window.
    # The script slices the full-year network to exactly this range before solving.
    "validation": {
        "network_path": "resources/networks/base_s_50_elec_2704_fixed.nc",
        "start_date":   "2024-01-01",   # ← start date
        "n_days":       90,               # Q1 2024 — meaningful validation, manageable on Colab
        "solver":       "gurobi",
        "solver_options": {
            "Threads":   6,
            "Method":    2,       # barrier-only — no concurrent optimizer, Crossover:0 respected
            "Crossover": 0,       # skip crossover — interior-point duals returned directly
            "Presolve":  2,       # aggressive presolve — removes redundant constraints before solve
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
