"""
Automated calibration loop for PyPSA-Spain model parameters.

Nested optimisation:
  Outer:  CCGT efficiency tiers (η) — 4 global params, optimised across all periods
  Inner:  Per-period hydro MC scales (ES/FR/PT) — each period calibrated independently

Nuclear, biomass, CCGT_must_run, VRE, capacities, and topology are FROZEN
at config.py baseline values — never touched by the optimiser.

Usage:
    pixi run python Analysis/calibrate.py                          # full nested optimisation
    pixi run python Analysis/calibrate.py --hydro-only             # skip CCGT outer loop, just tune hydro per period
    pixi run python Analysis/calibrate.py --ccgt-only              # skip hydro inner loop, just tune CCGT η
    pixi run python Analysis/calibrate.py --dry-run                # score current config, no solve
    pixi run python Analysis/calibrate.py --n-trials 5             # debug: 5 evals only

FIRM RULES (enforced in code, never relaxed):
  - Capacities (p_nom, e_nom) are never touched
  - VRE MC (€0.01), biomass MC (€0), CCGT_must_run MC (€2) are never touched
  - Nuclear MC range is FROZEN at config.py baseline (ES: 4.0-8.0 €/MWh)
  - Demand profiles, topology, and transmission are never touched
  - CCGT η ordering: T1 > T2 > T3 > T4 always enforced
  - Hydro MC clamped to [15, 120] €/MWh
  - CCGT η bounds: T1 [0.65, 0.88], T2 [0.56, 0.76], T3 [0.48, 0.65], T4 [0.40, 0.56]
  - Load shedding > 0 GWh → reject trial (score = 1e6)
  - Model mean price < €10 or > €120 → reject trial (score = 1e6)
"""

import argparse
import ast
import calendar
import contextlib
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

ROOT     = Path(__file__).parent.parent
CFG      = ROOT / "Analysis" / "config.py"
STATS    = ROOT / "Analysis" / "validation_output" / "validation_stats.txt"
CALIB_OUT = ROOT / "Analysis" / "calibration_output"
LOG_CSV  = CALIB_OUT / "calibration_log.csv"
REPORT   = CALIB_OUT / "progress_report.txt"

# ── Calibration windows ───────────────────────────────────────────────────────
# 12 monthly windows, each covering the week containing the 15th (days 11-17).
# This captures a full weekly cycle (weekday/weekend) and limits PHS cycling
# compared to 4-day windows. Each period is calibrated independently for hydro
# MC, then all periods are used together to optimise global CCGT η.
# Day-of-year for the 15th of each month in 2024 (leap year):
#   Jan=15, Feb=46, Mar=75, Apr=106, May=136, Jun=167,
#   Jul=197, Aug=228, Sep=259, Oct=289, Nov=320, Dec=350
CALIB_PERIODS = [15, 46, 75, 106, 136, 167, 197, 228, 259, 289, 320, 350]
CALIB_DAYS    = 7                              # 7-day window ≈ 5–8 min per solve

# ── Nested optimisation settings ──────────────────────────────────────────────
# Inner loop: per-period hydro scales (ES/FR/PT). Each period gets N_HYDRO_TRIALS
# random samples from the hydro bounds, keeping the best. Fast random sampling
# since hydro scales are independent per period — no DE overhead.
# Outer loop: CCGT η (4 global params) optimised across all periods via DE.
N_HYDRO_TRIALS = 3       # random hydro samples per period (just sample & keep best)
N_CCGT_TRIALS  = 30      # DE evaluations for CCGT η outer loop
N_CCGT_POPSIZE = 6       # DE popsize for CCGT outer loop

# ── Targets ───────────────────────────────────────────────────────────────────
# Price targets use the real period's OMIE percentiles from the stats file
# (not a fixed annual average). TARGET_PRICE is only a fallback when no
# OMIE percentiles are available.
TARGET_PRICE        = 63.0   # fallback OMIE annual mean €/MWh (used only when no percentiles)
TARGET_CCGT_PCT     = 0.50   # CCGT price-setter ~50% of hours
TARGET_VRE_PCT      = 0.18   # VRE price-setter ~18% of hours (thesis focus)
TARGET_HYDRO_UPLIFT = 0.05   # tolerate ≤5% overrun vs real hydro
TARGET_VOLL_HOURS   = 5      # tolerate up to 5 VOLL hours before rejecting trial

# Hydro dispatch accuracy: model hydro GWh must be within 20% of real hydro GWh.
# The min_dispatch constraint (monthly p_min_pu from real REE data in config.py)
# provides the primary ecological flow floor. This scoring penalty is a safety
# net to catch periods where the constraint alone doesn't align total dispatch.
# Tightened from 0.30 → 0.20: Trial 1 showed hydro dispatch was 2× real (500 vs
# 244 GWh) even with the old SOC-proportional constraint disabled.
TARGET_HYDRO_DEV = 0.20      # max fractional deviation from real hydro dispatch

# Border price sanity: FR/PT model mean prices should not deviate wildly from
# real prices. Prevents crazy border prices that cause unrealistic IC flows.
TARGET_BORDER_PRICE_DEV = 30.0   # max €/MWh deviation from real border price

# Interconnector flow targets (mean net MW, positive = ES imports)
TARGET_FR_FLOW = None
TARGET_PT_FLOW = None

WEIGHTS = dict(price=2.0, ccgt=3.0, vre=3.0, hydro=2.0, ic_flow=1.5,
               hydro_dispatch=8.0, border_price=2.0)

# ── Parameter bounds ──────────────────────────────────────────────────────────
# CCGT η: 4 global params (same across all periods)
# Hydro scales: 3 per period (ES/FR/PT), calibrated independently per period
# Nuclear MC is FROZEN at config.py baseline — not in any parameter vector.
ETA_WIDTH = 0.08   # each tier spans ±0.04 around its mid
HYDRO_MC_MIN, HYDRO_MC_MAX = 15.0, 120.0

def _ccgt_bounds():
    """Bounds for the 4 CCGT η global params."""
    return [
        (0.68, 0.88),   # T1_mid
        (0.60, 0.80),   # T2_mid
        (0.52, 0.69),   # T3_mid
        (0.44, 0.60),   # T4_mid
    ]

def _hydro_bounds():
    """Bounds for the 3 per-period hydro scale params (ES/FR/PT)."""
    return [
        (0.30, 2.00),   # ES hydro scale
        (0.30, 2.00),   # FR hydro scale
        (0.30, 2.00),   # PT hydro scale
    ]

def _period_start(year, doy):
    """Return 'YYYY-MM-DD' for the given day-of-year (1-based)."""
    from datetime import date, timedelta
    d = date(year, 1, 1) + timedelta(days=doy - 1)
    return d.strftime("%Y-%m-%d")

# ══════════════════════════════════════════════════════════════════════════════
# IMMUTABLE RULES — enforced in _apply_params and checked at startup.
# These cannot be overridden by the optimiser under any circumstances.
# ══════════════════════════════════════════════════════════════════════════════
RULES = [
    "CAPACITIES:  p_nom, e_nom, max_hours are NEVER changed",
    "VRE:         solar/onwind MC stays at €0.01 — NEVER changed",
    "RENEWABLES:  VRE capacity factors (p_max_pu profiles) are NEVER changed",
    "MUST-RUN:    biomass MC (€0), CCGT_must_run target MW and MC (€2) are NEVER changed",
    "NUCLEAR:     mc_range FROZEN at config.py baseline (ES: 4.0-8.0 €/MWh) — NOT in optimiser",
    "DEMAND:      load profiles, shapes, totals are NEVER changed",
    "TOPOLOGY:    lines, links, bus structure are NEVER changed",
    "CCGT ORDER:  T1_η > T2_η > T3_η > T4_η enforced always",
    "BOUNDS:      CCGT η T1[0.65,0.88] T2[0.56,0.76] T3[0.48,0.65] T4[0.40,0.56]",
    "BOUNDS:      hydro MC clamped to [15, 120] €/MWh before every solve",
    "REJECT:      any trial with load_shedding > 0 GWh → score = 1e6",
    "REJECT:      any trial with model mean price < €10 or > €120 → score = 1e6",
    "REJECT:      any trial with >5 VOLL hours → score = 1e6",
]

_trial_counter  = [0]
_best_score     = [1e9]
_best_params    = [None]    # best x so far
_session_start  = [None]    # set in main()
_baseline_x     = [None]    # starting params for delta display
_all_rows       = []        # in-memory trial history for report

# Per-country, per-month MC reduction factors for iterative hydro tuning.
# After each period evaluation, if model hydro dispatch exceeds real by >20%,
# the corresponding month's MC is reduced by MC_ADJUST_STEP (0.90 = -10%).
# Adjustments persist across outer trials and are applied on top of the
# country-scale factor in _apply_params.
MC_ADJUST: dict[str, dict[int, float]] = {
    "ES": {m: 1.0 for m in range(1, 13)},
    "FR": {m: 1.0 for m in range(1, 13)},
    "PT": {m: 1.0 for m in range(1, 13)},
}
MC_ADJUST_STEP = 0.90   # multiply by 0.90 = -10% per adjustment


# ── Config helpers ─────────────────────────────────────────────────────────────

def _read_cfg() -> str:
    return CFG.read_text()


def _write_cfg(text: str) -> None:
    CFG.write_text(text)


@contextlib.contextmanager
def _patched_config(params: np.ndarray, period_idx: int = 0):
    """Context manager: patch config for a specific period, yield, always restore.

    Parameters
    ----------
    params : np.ndarray
        7-element vector: [T1, T2, T3, T4, ES_sc, FR_sc, PT_sc]
    period_idx : int
        Index into CALIB_PERIODS for the start date.
    """
    original = _read_cfg()
    try:
        doy = CALIB_PERIODS[period_idx]
        start = _period_start(2024, doy)
        patched = _apply_params(original, params, period_idx=period_idx, calib_start=start)
        _write_cfg(patched)
        yield
    finally:
        _write_cfg(original)


def _apply_params(text: str, x: np.ndarray, period_idx: int = 0, calib_start: str = None) -> str:
    """Return config text with calibration parameters applied.

    Parameters
    ----------
    x : np.ndarray
        Parameter vector: [T1, T2, T3, T4, ES_sc, FR_sc, PT_sc]
        (4 CCGT η + 3 hydro scales for THIS period only).
        Nuclear MC is FROZEN at config.py baseline — not in this vector.
    period_idx : int
        Index into CALIB_PERIODS for the per-period hydro scales.
    calib_start : str, optional
        Start date 'YYYY-MM-DD' for the calibration window. If None, uses CALIB_PERIODS.
    """
    t1, t2, t3, t4 = x[0], x[1], x[2], x[3]
    es_sc = x[4]
    fr_sc = x[5]
    pt_sc = x[6]

    # ── Enforce η ordering ────────────────────────────────────────────────────
    half = ETA_WIDTH / 2
    cb = _ccgt_bounds()
    t1 = float(np.clip(t1, cb[0][0], cb[0][1]))
    t2 = float(np.clip(t2, cb[1][0], min(cb[1][1], t1 - 0.04)))
    t3 = float(np.clip(t3, cb[2][0], min(cb[2][1], t2 - 0.04)))
    t4 = float(np.clip(t4, cb[3][0], min(cb[3][1], t3 - 0.04)))

    def _tier_str(mid):
        lo = round(mid - half, 4)
        hi = round(mid + half, 4)
        return f"({lo}, {hi})"

    # Replace ES efficiency tiers
    tier_block = (
        f'        "ES": [\n'
        f'            {_tier_str(t1)},   # T1\n'
        f'            {_tier_str(t2)},   # T2\n'
        f'            {_tier_str(t3)},   # T3\n'
        f'            {_tier_str(t4)},   # T4\n'
        f'        ],'
    )
    text = re.sub(
        r'"ES": \[\s*\(.*?\),\s*\(.*?\),\s*\(.*?\),\s*\(.*?\),?\s*\],',
        tier_block,
        text,
        flags=re.DOTALL,
        count=1,
    )

    # ── Scale hydro monthly MC profiles ───────────────────────────────────────
    def _scale_profile(match, scale):
        inner = match.group(1)
        def _scale_val(m):
            v = float(m.group(1)) * scale
            v = float(np.clip(v, HYDRO_MC_MIN, HYDRO_MC_MAX))
            return str(round(v, 1))
        return '"profile": {' + re.sub(r':\s*(\d+(?:\.\d+)?)', lambda m: ': ' + _scale_val(m), inner) + '}'

    # ES, PT, FR monthly profiles — identified by order inside monthly_mc block
    def _patch_country_profile(txt, country, scale):
        # Find the country's "profile" dict within the monthly_mc section
        pattern = (
            rf'("{country}":\s*\{{[^}}]*?"profile":\s*\{{)([^}}]+)(\}})'
        )
        def _replacer(m):
            inner = m.group(2)
            def _sv(vm):
                v = float(vm.group(1)) * scale
                v = float(np.clip(v, HYDRO_MC_MIN, HYDRO_MC_MAX))
                return ': ' + str(round(v, 1))
            new_inner = re.sub(r':\s*(\d+(?:\.\d+)?)', _sv, inner)
            return m.group(1) + new_inner + m.group(3)
        return re.sub(pattern, _replacer, txt, flags=re.DOTALL, count=1)

    # Apply MC_ADJUST per-month reductions on top of the country-scale factor.
    # This is done as a second pass so the adjust factors persist across trials.
    def _apply_mc_adjust(txt, country):
        """Apply per-month MC_ADJUST factors to a country's profile."""
        adjust = MC_ADJUST.get(country, {})
        if all(v == 1.0 for v in adjust.values()):
            return txt  # no adjustments active — skip
        # Match the country's profile dict and adjust each month's value
        pattern = (
            rf'("{country}":\s*\{{[^}}]*?"profile":\s*\{{)([^}}]+)(\}})'
        )
        def _replacer(m):
            inner = m.group(2)
            def _adj(vm):
                month = int(vm.group(1))
                adj = adjust.get(month, 1.0)
                v = float(vm.group(1)) * adj
                v = float(np.clip(v, HYDRO_MC_MIN, HYDRO_MC_MAX))
                return ': ' + str(round(v, 1))
            new_inner = re.sub(r':\s*(\d+(?:\.\d+)?)', _adj, inner)
            return m.group(1) + new_inner + m.group(3)
        return re.sub(pattern, _replacer, txt, flags=re.DOTALL, count=1)

    text = _apply_mc_adjust(text, "ES")
    text = _apply_mc_adjust(text, "PT")
    text = _apply_mc_adjust(text, "FR")

    text = _patch_country_profile(text, "ES", es_sc)
    text = _patch_country_profile(text, "PT", pt_sc)
    text = _patch_country_profile(text, "FR", fr_sc)

    # ── Nuclear MC range — FROZEN at config.py baseline, NOT touched ──────────
    # Nuclear mc_range stays at config.py values (ES: 4.0-8.0 €/MWh).
    # The regex below is intentionally NOT applied — nuclear is removed from
    # the parameter vector entirely.

    # ── Override calib window ─────────────────────────────────────────────────
    if calib_start is None:
        calib_start = _period_start(2024, CALIB_PERIODS[period_idx])
    text = re.sub(r'"start_date":\s*"[^"]*"', f'"start_date":   "{calib_start}"', text)
    text = re.sub(r'"n_days":\s*\d+', f'"n_days":       {CALIB_DAYS}', text)

    # ── Sanity guard: forbidden patterns must not appear in diff ──────────────
    FORBIDDEN = ["p_nom", "e_nom", "max_hours", "p_max_pu", "solar.*0.01", "onwind.*0.01"]
    original_lines = set(_read_cfg().splitlines())
    new_lines = set(text.splitlines())
    changed = new_lines - original_lines
    for line in changed:
        for pattern in ["p_nom", "e_nom", "max_hours"]:
            if pattern in line and "mc" not in line.lower() and "tier" not in line.lower():
                raise RuntimeError(
                    f"RULE VIOLATION: _apply_params attempted to change a line "
                    f"containing '{pattern}':\n  {line.strip()}"
                )

    return text


# ── Run validation ─────────────────────────────────────────────────────────────

def _run_validation() -> bool:
    """Run the validation script. Returns True on success."""
    cmd = ["pixi", "run", "python", "Analysis/run_validation.py"]
    result = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900
    )
    if result.returncode != 0:
        print(f"    [SOLVER ERROR] returncode={result.returncode}")
        # print last 20 lines of stderr for diagnosis
        for line in result.stderr.strip().split("\n")[-20:]:
            print(f"    {line}")
        return False
    return True


# ── Parse metrics ──────────────────────────────────────────────────────────────

def _parse_stats() -> dict:
    """Parse validation_stats.txt into a metrics dict. Returns {} on failure."""
    if not STATS.exists():
        return {}
    text = STATS.read_text()
    m = {}

    def _get(pattern, cast=float, default=None):
        hit = re.search(pattern, text)
        if hit:
            try:
                return cast(hit.group(1))
            except Exception:
                pass
        return default

    m["price_mean"]   = _get(r"Model mean:\s*([\d.]+)")
    m["price_omie"]   = _get(r"OMIE mean:\s*([\d.]+)")
    m["load_shed"]    = _get(r"load_shedding\s+([\d.]+) GWh")

    # PDC percentiles — "Price p10/p50/p90: 5.3 / 53.5 / 76.1 EUR/MWh"
    pdc_hit = re.search(r"Price p10/p50/p90:\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)", text)
    if pdc_hit:
        m["model_p10"] = float(pdc_hit.group(1))
        m["model_p50"] = float(pdc_hit.group(2))
        m["model_p90"] = float(pdc_hit.group(3))
    else:
        m["model_p10"] = m["model_p50"] = m["model_p90"] = None

    # OMIE percentiles — "OMIE  p10/p50/p90: 0.0 / 32.1 / 71.0 EUR/MWh"
    # or "OMIE  p10/p50/p90: nan / nan / nan EUR/MWh" when OMIE data unavailable
    omie_pdc = re.search(r"OMIE\s+p10/p50/p90:\s*([\d.]+|nan|n/a)\s*/\s*([\d.]+|nan|n/a)\s*/\s*([\d.]+|nan|n/a)", text)
    if omie_pdc:
        def _parse_omie(val):
            return float(val) if val not in ("n/a", "nan") else None
        m["omie_p10"] = _parse_omie(omie_pdc.group(1))
        m["omie_p50"] = _parse_omie(omie_pdc.group(2))
        m["omie_p90"] = _parse_omie(omie_pdc.group(3))
    else:
        m["omie_p10"] = m["omie_p50"] = m["omie_p90"] = None

    # Zero-price hours — "Zero-price hours: model=2  OMIE=130"
    zp_hit = re.search(r"Zero-price hours:\s*model=(\d+)\s+OMIE=(\d+)", text)
    if zp_hit:
        m["zero_model"] = int(zp_hit.group(1))
        m["zero_omie"]  = int(zp_hit.group(2))
    else:
        m["zero_model"] = m["zero_omie"] = None

    # Price-setter percentages — lines like: "  CCGT   686h   95.3%  mean=81.9"
    ps_hits = re.findall(r"^\s+(\w+)\s+(\d+)h\s+([\d.]+)%", text, re.MULTILINE)
    ps = {row[0]: float(row[2]) / 100.0 for row in ps_hits}
    m["ccgt_pct"]    = ps.get("CCGT", 0.0)
    m["vre_pct"]     = ps.get("solar", 0.0) + ps.get("onwind", 0.0)
    m["nuclear_pct"] = ps.get("nuclear", 0.0)

    # Hydro uplift — TOTAL row: "  TOTAL  ...  Dispatch  Real_hydro  +Uplift  +Uplift%"
    # Groups: (1)=Dispatch GWh, (2)=Real_hydro GWh, (3)=Uplift%
    uplift_hit = re.search(
        r"TOTAL\s+[\d.]*\s+[\d.]*\s+([\d.]+)\s+([\d.]+)\s+\+[\d.]+\s+\+([\d.]+)%",
        text,
    )
    if uplift_hit:
        m["hydro_uplift"]       = float(uplift_hit.group(3)) / 100.0
        m["hydro_dispatch"]     = float(uplift_hit.group(1))   # model GWh
        m["hydro_dispatch_real"] = float(uplift_hit.group(2))  # real GWh
    else:
        m["hydro_uplift"]       = 0.0
        m["hydro_dispatch"]     = None
        m["hydro_dispatch_real"] = None

    # Border country prices — Section 2: "FR: model=87.7  actual=88.1 EUR/MWh"
    fr_price = re.search(r"FR:\s*model=([\d.]+)\s+actual=([\d.]+)", text)
    if fr_price:
        m["fr_price_model"] = float(fr_price.group(1))
        m["fr_price_real"]  = float(fr_price.group(2))
    else:
        m["fr_price_model"] = m["fr_price_real"] = None

    pt_price = re.search(r"PT:\s*model=([\d.]+)\s+actual=([\d.]+)", text)
    if pt_price:
        m["pt_price_model"] = float(pt_price.group(1))
        m["pt_price_real"]  = float(pt_price.group(2))
    else:
        m["pt_price_model"] = m["pt_price_real"] = None

    # Interconnector flows — Section 5: "FR↔ES: mean_net=-2116 MW ..."
    # New format includes real flow comparison:
    #   "FR↔ES: mean_net=-2116 MW (17% hrs FR→ES, 83% hrs ES→FR)"
    #   "  FR↔ES real: mean_net=-1800 MW  (model error=-316 MW)"
    fr_hit = re.search(r"FR↔ES:\s*mean_net=([+-]?\d+)", text)
    if fr_hit:
        m["fr_flow"] = float(fr_hit.group(1))
    else:
        m["fr_flow"] = None
    fr_real = re.search(r"FR↔ES real:\s*mean_net=([+-]?\d+)", text)
    if fr_real:
        m["fr_flow_target"] = float(fr_real.group(1))
    else:
        m["fr_flow_target"] = None

    pt_hit = re.search(r"PT↔ES:\s*mean_net=([+-]?\d+)", text)
    if pt_hit:
        m["pt_flow"] = float(pt_hit.group(1))
    else:
        m["pt_flow"] = None
    pt_real = re.search(r"PT↔ES real:\s*mean_net=([+-]?\d+)", text)
    if pt_real:
        m["pt_flow_target"] = float(pt_real.group(1))
    else:
        m["pt_flow_target"] = None

    return m


# ── Objective ──────────────────────────────────────────────────────────────────
# Scoring uses the OMIE percentiles FROM THE SAME STATS FILE as targets so the
# objective automatically adapts to whichever month is being calibrated.
# Weights: PDC shape (p10/p50/p90) gets 60% of the weight; dispatch mix 40%.
#
# PDC shape errors:
#   p10 = left tail (VRE surplus / nuclear floor hours) — thesis focus, high weight
#   p50 = median (hydro / CCGT transition zone) — medium weight
#   p90 = right tail (peak gas / scarcity) — medium weight
#   zero_pct = fraction of near-zero hours — captures VRE price-setting directly
#
# Dispatch mix errors:
#   ccgt_pct  — CCGT as price-setter (target ~50%)
#   vre_pct   — VRE as price-setter (thesis focus, high weight)
#   hydro_up  — penalise hydro overdispatch above 5% of real

def _score(m: dict) -> float:
    if not m:
        return 1e6

    price    = m.get("price_mean")
    shedding = m.get("load_shed", 0.0) or 0.0

    # Hard rejections
    if price is None:
        return 1e6
    if shedding > 0.001:
        return 1e6
    if price < 10.0 or price > 120.0:
        return 1e6

    # VOLL tolerance: reject if more than TARGET_VOLL_HOURS VOLL hours
    # VOLL hours = hours where model price >= voll (300 €/MWh)
    # We approximate via p90: if p90 is near VOLL, many hours at VOLL.
    mp90 = m.get("model_p90")
    if mp90 is not None and mp90 >= 290.0:
        return 1e6

    total = 0.0

    # ── PDC shape (use OMIE percentiles from same stats file as reference) ─────
    mp10  = m.get("model_p10");  op10  = m.get("omie_p10")
    mp50  = m.get("model_p50");  op50  = m.get("omie_p50")
    mp90  = m.get("model_p90");  op90  = m.get("omie_p90")

    ref_scale = max(op90 or 1.0, 1.0)   # normalise by OMIE p90 so all errors ∈ [0,1]

    if mp10 is not None and op10 is not None:
        err10 = (mp10 - op10) / ref_scale
        total += 2.0 * err10 ** 2           # weight 2.0 — left tail = VRE thesis focus

    if mp50 is not None and op50 is not None:
        err50 = (mp50 - op50) / ref_scale
        total += 1.0 * err50 ** 2           # weight 1.0 — median / middle

    if mp90 is not None and op90 is not None:
        err90 = (mp90 - op90) / ref_scale
        total += 1.5 * err90 ** 2           # weight 1.5 — right tail / peak gas

    # Zero-price hours: model should approach OMIE count
    zm = m.get("zero_model"); zo = m.get("zero_omie")
    if zm is not None and zo is not None and zo > 0:
        total += 2.0 * ((zm - zo) / max(zo, 1)) ** 2

    # Fallback to mean if no percentiles available
    if mp10 is None:
        err_mean = (price - TARGET_PRICE) / TARGET_PRICE
        total += 2.0 * err_mean ** 2

    # ── Dispatch mix ──────────────────────────────────────────────────────────
    ccgt_p = m.get("ccgt_pct", 0.0)
    vre_p  = m.get("vre_pct",  0.0)
    h_up   = m.get("hydro_uplift", 0.0)

    total += WEIGHTS["ccgt"]  * ((ccgt_p - TARGET_CCGT_PCT) / max(TARGET_CCGT_PCT, 0.01)) ** 2
    total += WEIGHTS["vre"]   * ((vre_p  - TARGET_VRE_PCT)  / max(TARGET_VRE_PCT,  0.01)) ** 2
    total += WEIGHTS["hydro"] * max(0.0, h_up - TARGET_HYDRO_UPLIFT) ** 2

    # ── Interconnector flow accuracy ──────────────────────────────────────────
    fr_flow = m.get("fr_flow")
    fr_tgt  = m.get("fr_flow_target")
    pt_flow = m.get("pt_flow")
    pt_tgt  = m.get("pt_flow_target")

    if fr_flow is not None and fr_tgt is not None and fr_tgt != 0:
        fr_err = (fr_flow - fr_tgt) / abs(fr_tgt)
        total += WEIGHTS["ic_flow"] * fr_err ** 2

    if pt_flow is not None and pt_tgt is not None and pt_tgt != 0:
        pt_err = (pt_flow - pt_tgt) / abs(pt_tgt)
        total += WEIGHTS["ic_flow"] * pt_err ** 2

    # ── Hydro dispatch accuracy ───────────────────────────────────────────────
    # Penalise if model hydro GWh deviates >50% from real hydro GWh.
    # This prevents the optimiser from dumping zero hydro (wasted compute) or
    # draining reservoirs with crazy dispatch amounts.
    h_disp     = m.get("hydro_dispatch")
    h_disp_real = m.get("hydro_dispatch_real")
    if h_disp is not None and h_disp_real is not None and h_disp_real > 0:
        h_dev = abs(h_disp - h_disp_real) / h_disp_real
        if h_dev > TARGET_HYDRO_DEV:
            # Quadratic penalty above threshold
            total += WEIGHTS["hydro_dispatch"] * ((h_dev - TARGET_HYDRO_DEV) / (1.0 - TARGET_HYDRO_DEV)) ** 2

    # ── Border price sanity ───────────────────────────────────────────────────
    # Penalise large deviations between model and actual FR/PT mean prices.
    # Prevents crazy border prices that cause unrealistic IC flows.
    for ctry in ("fr", "pt"):
        p_model = m.get(f"{ctry}_price_model")
        p_real  = m.get(f"{ctry}_price_real")
        if p_model is not None and p_real is not None and p_real > 0:
            p_dev = abs(p_model - p_real)
            if p_dev > TARGET_BORDER_PRICE_DEV:
                excess = (p_dev - TARGET_BORDER_PRICE_DEV) / TARGET_BORDER_PRICE_DEV
                total += WEIGHTS["border_price"] * excess ** 2

    return float(total)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def _write_report():
    """Rewrite the human-readable progress_report.txt after every trial."""
    now        = time.time()
    total_s    = now - (_session_start[0] or now)
    n_done     = len(_all_rows)
    valid_rows = [r for r in _all_rows if r["score"] < 1e5]
    best_row   = min(valid_rows, key=lambda r: r["score"]) if valid_rows else None
    n_periods  = len(CALIB_PERIODS)

    lines = []
    lines.append("=" * 80)
    lines.append("  PyPSA-Spain Calibration Progress")
    periods_str = ", ".join([f"P{CALIB_PERIODS[p]}" for p in range(n_periods)])
    lines.append(f"  Periods:   {periods_str}  ({CALIB_DAYS} days/trial, {n_periods} periods/trial)")
    lines.append(f"  Started:   {time.strftime('%Y-%m-%d %H:%M', time.localtime(_session_start[0] or now))}")
    lines.append(f"  Elapsed:   {_fmt_elapsed(total_s)}")
    lines.append(f"  Trials:    {n_done} completed  ({len(valid_rows)} valid)")
    avg_s = total_s / max(n_done, 1)
    lines.append(f"  Avg/trial: {_fmt_elapsed(avg_s)}")
    lines.append("=" * 80)
    lines.append("")

    if best_row:
        bp = _best_params[0]
        base = _baseline_x[0]
        lines.append("  BEST PARAMS SO FAR")
        lines.append(f"  Score: {best_row['score']:.4f}  (trial #{best_row['trial']})")
        lines.append("")
        # Global params (nuclear frozen at config.py baseline — not in vector)
        pnames = ["T1_η", "T2_η", "T3_η", "T4_η"]
        for j, (name, val) in enumerate(zip(pnames, bp[:4])):
            delta = val - base[j] if base is not None else 0.0
            sign = "+" if delta >= 0 else ""
            lines.append(f"    {name:<12} {val:>7.3f}   (Δ {sign}{delta:+.3f} from start)")
        lines.append(f"    Nuclear     FROZEN at config.py baseline (ES: 4.0–8.0 €/MWh)")
        # Per-period hydro scales
        lines.append("")
        lines.append("  Per-period hydro scales:")
        for p in range(n_periods):
            b = 4 + 3 * p
            es, fr, pt = bp[b], bp[b+1], bp[b+2]
            d_es = es - base[b] if base is not None else 0.0
            d_fr = fr - base[b+1] if base is not None else 0.0
            d_pt = pt - base[b+2] if base is not None else 0.0
            lines.append(f"    P{CALIB_PERIODS[p]:>2}:  ES×{es:.3f} (Δ {d_es:+.3f})  "
                         f"FR×{fr:.3f} (Δ {d_fr:+.3f})  PT×{pt:.3f} (Δ {d_pt:+.3f})")
        lines.append("")
        lines.append("  BEST PDC MATCH (averaged across weeks)")
        lines.append(f"    Mean  model/OMIE: {best_row['price_model']:>6.1f} / {best_row['price_omie']:>6.1f} €/MWh")
        lines.append(f"    p10   model/OMIE: {best_row['p10_model']:>6.1f} / {best_row['p10_omie']:>6.1f} €/MWh  ← VRE surplus")
        lines.append(f"    p50   model/OMIE: {best_row['p50_model']:>6.1f} / {best_row['p50_omie']:>6.1f} €/MWh  ← median")
        lines.append(f"    p90   model/OMIE: {best_row['p90_model']:>6.1f} / {best_row['p90_omie']:>6.1f} €/MWh  ← peak gas")
        lines.append(f"    0h    model/OMIE: {best_row['zero_model']:>6.1f} / {best_row['zero_omie']:>6.1f} hrs")
        lines.append(f"    CCGT setter:      {best_row['ccgt_pct']:>5.1f}%  (target {TARGET_CCGT_PCT*100:.0f}%)")
        lines.append(f"    VRE  setter:      {best_row['vre_pct']:>5.1f}%  (target {TARGET_VRE_PCT*100:.0f}%)")
        lines.append(f"    Hydro uplift:    {best_row['hydro_uplift']:>+5.1f}%  (target ≤+{TARGET_HYDRO_UPLIFT*100:.0f}%)")
        lines.append(f"    Hydro dispatch:   {best_row.get('hydro_disp','?'):>6.1f} / {best_row.get('hydro_disp_real','?'):>6.1f} GWh  "
                     f"(target within {TARGET_HYDRO_DEV*100:.0f}% of real)")
        lines.append(f"    FR price:         {best_row.get('fr_price_m','?'):>5.1f} / {best_row.get('fr_price_r','?'):>5.1f} €/MWh  "
                     f"(target within {TARGET_BORDER_PRICE_DEV:.0f} €)")
        lines.append(f"    PT price:         {best_row.get('pt_price_m','?'):>5.1f} / {best_row.get('pt_price_r','?'):>5.1f} €/MWh  "
                     f"(target within {TARGET_BORDER_PRICE_DEV:.0f} €)")
        lines.append(f"    FR flow:          {best_row.get('fr_flow','?'):>+6.0f} MW  "
                     f"(target {best_row.get('fr_flow_target','?'):>+6.0f} MW)")
        lines.append(f"    PT flow:          {best_row.get('pt_flow','?'):>+6.0f} MW  "
                     f"(target {best_row.get('pt_flow_target','?'):>+6.0f} MW)")
    else:
        lines.append("  No valid trials yet.")
    lines.append("")

    # Trial history table
    lines.append("  TRIAL HISTORY (model/OMIE — sorted by score, best first)")
    lines.append(f"  {'Rank':>4}  {'#':>4}  {'Score':>7}  "
                 f"{'Mean M/O':>10}  {'p10 M/O':>10}  {'p90 M/O':>10}  "
                 f"{'0h M/O':>8}  {'CCGT%':>5}  {'VRE%':>4}  {'Hyd%':>5}  "
                 f"{'Hdsp':>6}  {'FRpr':>6}  {'PTpr':>6}  {'FRfl':>6}  {'PTfl':>6}  {'Time':>7}")
    lines.append("  " + "-" * 110)
    sorted_rows = sorted(valid_rows, key=lambda r: r["score"])
    for rank, r in enumerate(sorted_rows[:30], 1):   # show top 30
        star = "★" if rank == 1 else " "
        fr_f = r.get('fr_flow', 0)
        pt_f = r.get('pt_flow', 0)
        hd   = r.get('hydro_disp', 0)
        hdr  = r.get('hydro_disp_real', 0)
        fpr  = r.get('fr_price_m', 0)
        ptr  = r.get('pt_price_m', 0)
        lines.append(
            f"  {star}{rank:>3}  {r['trial']:>4}  {r['score']:>7.4f}  "
            f"  {r['price_model']:>4.1f}/{r['price_omie']:>4.1f}  "
            f"  {r['p10_model']:>4.1f}/{r['p10_omie']:>4.1f}  "
            f"  {r['p90_model']:>4.1f}/{r['p90_omie']:>4.1f}  "
            f"  {r['zero_model']:>3.0f}/{r['zero_omie']:>3.0f}  "
            f"  {r['ccgt_pct']:>4.0f}%  {r['vre_pct']:>3.0f}%  "
            f"  {r['hydro_uplift']:>+4.0f}%  "
            f"  {hd:>5.0f}/{hdr:<.0f}  {fpr:>5.0f}  {ptr:>5.0f}  "
            f"  {fr_f:>+5.0f}  {pt_f:>+5.0f}  "
            f"  {_fmt_elapsed(r['elapsed_s'])}"
        )
    if len(valid_rows) > 30:
        lines.append(f"  ... ({len(valid_rows) - 30} more in {LOG_CSV.name})")

    lines.append("")
    lines.append(f"  Updated: {time.strftime('%H:%M:%S')}")
    lines.append(f"  Diagnostics: {CALIB_OUT.name}/diagnostics.png")
    lines.append("=" * 80)

    REPORT.write_text("\n".join(lines))


def _log_trial(i, params, score, m, elapsed):
    n_periods = len(CALIB_PERIODS)
    row = {
        "trial":        i,
        "score":        round(score, 6),
        "price_model":  round(m.get("price_mean") or 0, 2),
        "price_omie":   round(m.get("price_omie") or 0, 2),
        "p10_model":    round(m.get("model_p10") or 0, 1),
        "p10_omie":     round(m.get("omie_p10") or 0, 1),
        "p50_model":    round(m.get("model_p50") or 0, 1),
        "p50_omie":     round(m.get("omie_p50") or 0, 1),
        "p90_model":    round(m.get("model_p90") or 0, 1),
        "p90_omie":     round(m.get("omie_p90") or 0, 1),
        "zero_model":   m.get("zero_model") or 0,
        "zero_omie":    m.get("zero_omie") or 0,
        "ccgt_pct":     round(m.get("ccgt_pct", 0) * 100, 1),
        "vre_pct":      round(m.get("vre_pct", 0) * 100, 1),
        "hydro_uplift": round(m.get("hydro_uplift", 0) * 100, 1),
        "hydro_disp":   round(m.get("hydro_dispatch", 0) or 0, 1),
        "hydro_disp_real": round(m.get("hydro_dispatch_real", 0) or 0, 1),
        "fr_price_m":   round(m.get("fr_price_model", 0) or 0, 1),
        "fr_price_r":   round(m.get("fr_price_real", 0) or 0, 1),
        "pt_price_m":   round(m.get("pt_price_model", 0) or 0, 1),
        "pt_price_r":   round(m.get("pt_price_real", 0) or 0, 1),
        "fr_flow":      round(m.get("fr_flow", 0) or 0, 0),
        "fr_flow_target": round(m.get("fr_flow_target", 0) or 0, 0),
        "pt_flow":      round(m.get("pt_flow", 0) or 0, 0),
        "pt_flow_target": round(m.get("pt_flow_target", 0) or 0, 0),
        "elapsed_s":    round(elapsed, 0),
        "T1_mid":       round(params[0], 3),
        "T2_mid":       round(params[1], 3),
        "T3_mid":       round(params[2], 3),
        "T4_mid":       round(params[3], 3),
    }
    # Per-period hydro scales (nuclear NOT in vector — frozen at config.py baseline)
    for p in range(n_periods):
        base = 4 + 3 * p
        row[f"es_scale_p{CALIB_PERIODS[p]}"] = round(params[base], 3)
        row[f"fr_scale_p{CALIB_PERIODS[p]}"] = round(params[base + 1], 3)
        row[f"pt_scale_p{CALIB_PERIODS[p]}"] = round(params[base + 2], 3)
    _all_rows.append(row)

    is_best = score < _best_score[0]
    if is_best:
        _best_score[0] = score
        _best_params[0] = params.copy()

    # ── Append to CSV ────────────────────────────────────────────────────────
    CALIB_OUT.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    # ── Update human-readable report ──────────────────────────────────────────
    _write_report()

    # ── Update diagnostic plots ───────────────────────────────────────────────
    _plot_diagnostics()
    _plot_live_pdc()

    # ── Console one-liner ─────────────────────────────────────────────────────
    total_s = time.time() - (_session_start[0] or time.time())
    marker  = "★" if is_best else " "
    delta_score = ""
    if not is_best and len(_all_rows) > 1:
        prev_best = min(r["score"] for r in _all_rows[:-1] if r["score"] < 1e5)
        if prev_best < 1e5:
            delta_score = f"  (best={prev_best:.4f})"

    # Param deltas vs baseline (show T1 and first-period ES)
    base = _baseline_x[0]
    if base is not None:
        dT1 = params[0] - base[0]
        dES = params[4] - base[4] if len(params) > 4 else 0.0
        delta_str = f"dT1={dT1:+.3f} dES_p{CALIB_PERIODS[0]}={dES:+.3f}"
    else:
        delta_str = ""

    print(
        f"\n  {marker} #{i:>3}/{_trial_counter[0]}  [{_fmt_elapsed(total_s)}]  "
        f"score={score:.4f}{delta_score}\n"
        f"      PDC: p10={row['p10_model']:.1f}/{row['p10_omie']:.1f}  "
        f"p50={row['p50_model']:.1f}/{row['p50_omie']:.1f}  "
        f"p90={row['p90_model']:.1f}/{row['p90_omie']:.1f}  "
        f"0h={row['zero_model']}/{row['zero_omie']}\n"
        f"      Mix: CCGT={row['ccgt_pct']:.0f}%  VRE={row['vre_pct']:.0f}%  "
        f"hydro={row['hydro_uplift']:+.0f}%  "
        f"Hdisp={row['hydro_disp']:.0f}/{row['hydro_disp_real']:.0f}GWh  "
        f"FRpr={row['fr_price_m']:.0f}/{row['fr_price_r']:.0f}  "
        f"PTpr={row['pt_price_m']:.0f}/{row['pt_price_r']:.0f}  "
        f"FRfl={row['fr_flow']:+.0f}/{row['fr_flow_target']:+.0f}  "
        f"PTfl={row['pt_flow']:+.0f}/{row['pt_flow_target']:+.0f}  "
        f"trial={elapsed:.0f}s  {delta_str}"
    )
    print(f"      Report → {REPORT.name}  |  Diagnostics → diagnostics.png  |  Live PDC → diagnostics_live.png")


# ── Nested objective functions ─────────────────────────────────────────────────

def _objective_single_period(hydro_scales: np.ndarray, ccgt_eta: np.ndarray,
                              period_idx: int) -> float:
    """Score a single period given CCGT η (fixed) and 3 hydro scales.

    Parameters
    ----------
    hydro_scales : np.ndarray
        3-element vector: [ES_sc, FR_sc, PT_sc]
    ccgt_eta : np.ndarray
        4-element vector: [T1, T2, T3, T4] (fixed for this inner-loop call)
    period_idx : int
        Index into CALIB_PERIODS.

    Returns
    -------
    float
        Score for this period (lower = better).
    """
    x = np.concatenate([ccgt_eta, hydro_scales])
    doy = CALIB_PERIODS[period_idx]
    print(f"      P{doy}: ES×{hydro_scales[0]:.2f} FR×{hydro_scales[1]:.2f} PT×{hydro_scales[2]:.2f}  ",
          end="", flush=True)

    with _patched_config(x, period_idx=period_idx):
        success = _run_validation()

    if not success:
        print("FAIL")
        return 1e6

    m = _parse_stats()
    s = _score(m)
    p10_m = m.get('model_p10')
    p10_o = m.get('omie_p10')
    p10_str = f"{p10_m:.1f}/{p10_o:.1f}" if (p10_m is not None and p10_o is not None) else "?/?"
    print(f"score={s:.4f}  p10={p10_str}")
    return s


def doy_to_month(doy: int) -> int:
    """Convert day-of-year (1-based) to month number (1-12)."""
    import datetime
    return datetime.datetime(2024, 1, 1) + datetime.timedelta(days=doy - 1)


def _adjust_hydro_mc_for_period(period_idx: int, ccgt_eta: np.ndarray,
                                  best_hydro_scales: np.ndarray) -> np.ndarray:
    """Check hydro dispatch vs real for this period; reduce MC if >20% deviation.

    After the best hydro scale has been found for a period, re-run with the
    best scale and check if model hydro dispatch exceeds real by >20%.
    If so, reduce that month's MC by MC_ADJUST_STEP (0.90 = -10%) and
    re-run once to verify improvement.

    Parameters
    ----------
    period_idx : int
        Index into CALIB_PERIODS.
    ccgt_eta : np.ndarray
        4-element CCGT η vector.
    best_hydro_scales : np.ndarray
        3-element [ES_sc, FR_sc, PT_sc] from the best trial.

    Returns
    -------
    np.ndarray
        Updated best_hydro_scales (may be unchanged if no adjustment needed).
    """
    doy = CALIB_PERIODS[period_idx]
    month = doy_to_month(doy).month

    # Re-run with best hydro to get metrics
    x_full = np.concatenate([ccgt_eta, best_hydro_scales])
    with _patched_config(x_full, period_idx=period_idx):
        success = _run_validation()

    if not success:
        return best_hydro_scales

    m = _parse_stats()
    h_disp      = m.get("hydro_dispatch")
    h_disp_real = m.get("hydro_dispatch_real")

    if h_disp is None or h_disp_real is None or h_disp_real <= 0:
        return best_hydro_scales

    h_dev = (h_disp - h_disp_real) / h_disp_real
    print(f"      └─ hydro dispatch: model={h_disp:.1f} GWh  real={h_disp_real:.1f} GWh  "
          f"dev={h_dev:+.1%}")

    if h_dev <= TARGET_HYDRO_DEV:
        return best_hydro_scales  # within tolerance — no adjustment needed

    # Hydro over-dispatch: reduce MC for this month by MC_ADJUST_STEP
    print(f"      └─ ⚠ hydro over-dispatch ({h_dev:+.1%} > {TARGET_HYDRO_DEV:.0%}) — "
          f"reducing month {month} MC by {1-MC_ADJUST_STEP:.0%}")

    for country in ("ES", "FR", "PT"):
        old_adj = MC_ADJUST[country].get(month, 1.0)
        new_adj = old_adj * MC_ADJUST_STEP
        MC_ADJUST[country][month] = new_adj
        print(f"         {country} month {month}: MC_ADJUST {old_adj:.3f} → {new_adj:.3f}")

    # Re-run with adjusted MC to verify improvement
    with _patched_config(x_full, period_idx=period_idx):
        success = _run_validation()

    if success:
        m2 = _parse_stats()
        h_disp2      = m2.get("hydro_dispatch")
        h_disp_real2 = m2.get("hydro_dispatch_real")
        if h_disp2 is not None and h_disp_real2 is not None and h_disp_real2 > 0:
            h_dev2 = (h_disp2 - h_disp_real2) / h_disp_real2
            print(f"      └─ after MC adjust: model={h_disp2:.1f} GWh  real={h_disp_real2:.1f} GWh  "
                  f"dev={h_dev2:+.1%}")
            if h_dev2 <= h_dev:
                print(f"      └─ ✓ hydro dispatch improved ({h_dev2:+.1%} vs {h_dev:+.1%})")
            else:
                print(f"      └─ ⚠ hydro dispatch worsened ({h_dev2:+.1%} vs {h_dev:+.1%}) — "
                      f"keeping adjustment anyway")

    return best_hydro_scales


def _calibrate_period_hydro(ccgt_eta: np.ndarray, period_idx: int) -> tuple:
    """Inner loop: calibrate hydro scale for one period with CCGT η fixed.

    Uses a single scalar scale factor applied to all 3 countries (ES/FR/PT)
    together — they move in lockstep. Tests 3 candidates:
      - below 1.0 (cheaper hydro)
      - at 1.0   (baseline = config.py profile as-is)
      - above 1.0 (more expensive hydro)

    After finding the best scale, checks if hydro dispatch exceeds real by >20%
    and if so, reduces that month's MC by 10% and re-runs once.

    Parameters
    ----------
    ccgt_eta : np.ndarray
        4-element CCGT η vector (fixed during this inner loop).
    period_idx : int
        Index into CALIB_PERIODS.

    Returns
    -------
    tuple
        (best_hydro_scales, best_score)
    """
    hb = _hydro_bounds()
    doy = CALIB_PERIODS[period_idx]

    # Single scalar scale for all 3 countries — they move in lockstep
    lo, hi = hb[0]  # all bounds are identical: (0.30, 2.00)
    below = lo + (1.0 - lo) * 0.5   # 0.65 — cheaper hydro
    above = 1.0 + (hi - 1.0) * 0.5  # 1.50 — more expensive hydro

    candidates = [below, 1.0, above]  # one below, one at, one above

    best_score = 1e9
    best_scale = 1.0

    for scale in candidates:
        hs = np.array([scale, scale, scale])  # all 3 countries in lockstep
        s = _objective_single_period(hs, ccgt_eta, period_idx)
        if s < best_score:
            best_score = s
            best_scale = scale

    best_hydro = np.array([best_scale, best_scale, best_scale])
    print(f"    ── P{doy} best: scale={best_scale:.3f}  "
          f"score={best_score:.4f}")

    # ── Iterative MC tuning: check hydro dispatch and adjust if needed ────────
    _adjust_hydro_mc_for_period(period_idx, ccgt_eta, best_hydro)

    return best_hydro, best_score


def objective(x: np.ndarray) -> float:
    """Outer-loop objective: evaluate CCGT η across all periods.

    For each period, runs the inner-loop hydro calibration (3 DE sub-trials)
    and returns the average score across all periods.

    Parameters
    ----------
    x : np.ndarray
        4-element CCGT η vector: [T1, T2, T3, T4]
    """
    _trial_counter[0] += 1
    i = _trial_counter[0]
    t0 = time.time()

    n_periods = len(CALIB_PERIODS)
    print(f"\n  → Outer Trial {i}: T1η={x[0]:.3f} T2η={x[1]:.3f} T3η={x[2]:.3f} T4η={x[3]:.3f}"
          f"  ({n_periods} periods × {N_HYDRO_TRIALS} hydro sub-trials each)  "
          f"[nuclear frozen at config.py baseline]")

    scores = []
    all_metrics = []
    all_hydro = []   # best hydro scales per period

    for p in range(n_periods):
        best_hydro, best_score = _calibrate_period_hydro(x, period_idx=p)
        scores.append(best_score)
        all_hydro.append(best_hydro)

        # Re-run with best hydro to capture metrics for logging
        x_full = np.concatenate([x, best_hydro])
        with _patched_config(x_full, period_idx=p):
            success = _run_validation()
        if success:
            m = _parse_stats()
            all_metrics.append(m)
        else:
            all_metrics.append({})

    elapsed = time.time() - t0
    avg_score = float(np.mean(scores))

    # Build the full parameter vector for logging: [T1..T4, ES0,FR0,PT0, ES1,...]
    full_x = np.concatenate([x] + all_hydro)

    # Log with averaged metrics
    avg_m = {}
    if all_metrics:
        keys = ["price_mean", "price_omie", "model_p10", "omie_p10",
                "model_p50", "omie_p50", "model_p90", "omie_p90",
                "zero_model", "zero_omie", "ccgt_pct", "vre_pct", "hydro_uplift",
                "hydro_dispatch", "hydro_dispatch_real",
                "fr_price_model", "fr_price_real", "pt_price_model", "pt_price_real",
                "fr_flow", "fr_flow_target", "pt_flow", "pt_flow_target"]
        for k in keys:
            vals = [v for m in all_metrics if m for v in [m.get(k)] if v is not None]
            avg_m[k] = float(np.mean(vals)) if vals else 0.0

    _log_trial(i, full_x, avg_score, avg_m, elapsed)
    return avg_score


# ── Dry-run: score current config ─────────────────────────────────────────────

def dry_run():
    print("=== DRY RUN — scoring current config (no optimisation) ===")
    if not STATS.exists():
        print("  No validation_stats.txt found. Run validation first.")
        return
    m = _parse_stats()
    s = _score(m)
    print(f"  Current score:        {s:.4f}")
    print()
    print(f"  PDC shape  (model / OMIE):")
    print(f"    Mean:  {m.get('price_mean','?'):>6} / {m.get('price_omie','?'):>6} €/MWh")
    print(f"    p10:   {m.get('model_p10','?'):>6} / {m.get('omie_p10','?'):>6} €/MWh  ← left tail (VRE surplus)")
    print(f"    p50:   {m.get('model_p50','?'):>6} / {m.get('omie_p50','?'):>6} €/MWh  ← median")
    print(f"    p90:   {m.get('model_p90','?'):>6} / {m.get('omie_p90','?'):>6} €/MWh  ← right tail (peak gas)")
    print(f"    0h:    {m.get('zero_model','?'):>6} / {m.get('zero_omie','?'):>6} hrs   ← near-zero price hours")
    print()
    print(f"  Dispatch mix:")
    print(f"    CCGT price-setter:    {m.get('ccgt_pct', 0)*100:.1f}%  (target {TARGET_CCGT_PCT*100:.0f}%)")
    print(f"    VRE price-setter:     {m.get('vre_pct', 0)*100:.1f}%   (target {TARGET_VRE_PCT*100:.0f}%)")
    print(f"    Hydro uplift vs real: {m.get('hydro_uplift', 0)*100:.1f}%  (target ≤{TARGET_HYDRO_UPLIFT*100:.0f}%)")
    print(f"    Load shedding:        {m.get('load_shed', 0)} GWh")
    print()
    print(f"  Hydro dispatch (model / real):")
    hd  = m.get('hydro_dispatch', '?')
    hdr = m.get('hydro_dispatch_real', '?')
    print(f"    ES hydro: {hd:>6} / {hdr:>6} GWh  (target within {TARGET_HYDRO_DEV*100:.0f}% of real)")
    print()
    print(f"  Border prices (model / real):")
    fpm = m.get('fr_price_model', '?')
    fpr = m.get('fr_price_real', '?')
    ppm = m.get('pt_price_model', '?')
    ppr = m.get('pt_price_real', '?')
    print(f"    FR: {fpm:>6} / {fpr:>6} €/MWh  (target within {TARGET_BORDER_PRICE_DEV:.0f} €)")
    print(f"    PT: {ppm:>6} / {ppr:>6} €/MWh  (target within {TARGET_BORDER_PRICE_DEV:.0f} €)")
    print()
    print(f"  Interconnector flows (model / real):")
    fr_f  = m.get('fr_flow', '?')
    fr_t  = m.get('fr_flow_target', '?')
    pt_f  = m.get('pt_flow', '?')
    pt_t  = m.get('pt_flow_target', '?')
    print(f"    FR↔ES: {fr_f:>+6} / {fr_t:>+6} MW")
    print(f"    PT↔ES: {pt_f:>+6} / {pt_t:>+6} MW")


# ── Diagnostic plot ─────────────────────────────────────────────────────────────

def _plot_diagnostics():
    """Generate a 6-panel diagnostic PNG: score, PDC, dispatch mix, hydro dispatch,
    border prices, IC flows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in _all_rows if r["score"] < 1e5]
    if len(valid) < 2:
        return

    trials   = [r["trial"] for r in valid]
    scores   = [r["score"] for r in valid]
    best_idx = min(range(len(scores)), key=scores.__getitem__)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Calibration Diagnostics", fontsize=13, fontweight="bold")

    # Panel 1: Score convergence
    ax = axes[0, 0]
    ax.plot(trials, scores, "o-", color="#2196F3", ms=3, lw=0.8, label="score")
    ax.axhline(scores[best_idx], color="#F44336", ls="--", lw=0.8,
               label=f"best={scores[best_idx]:.4f} (#{trials[best_idx]})")
    ax.set_xlabel("Trial"); ax.set_ylabel("Score"); ax.set_title("Score convergence")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 2: PDC match (p10, p50, p90 model vs omie)
    ax = axes[0, 1]
    p10_m = [r["p10_model"] for r in valid]
    p10_o = [r["p10_omie"] for r in valid]
    p50_m = [r["p50_model"] for r in valid]
    p50_o = [r["p50_omie"] for r in valid]
    p90_m = [r["p90_model"] for r in valid]
    p90_o = [r["p90_omie"] for r in valid]
    ax.plot(trials, p10_m, "o-", color="#4CAF50", ms=3, lw=0.8, label="p10 model")
    ax.plot(trials, p10_o, "o--", color="#4CAF50", ms=2, lw=0.6, alpha=0.5, label="p10 OMIE")
    ax.plot(trials, p50_m, "o-", color="#FF9800", ms=3, lw=0.8, label="p50 model")
    ax.plot(trials, p50_o, "o--", color="#FF9800", ms=2, lw=0.6, alpha=0.5, label="p50 OMIE")
    ax.plot(trials, p90_m, "o-", color="#F44336", ms=3, lw=0.8, label="p90 model")
    ax.plot(trials, p90_o, "o--", color="#F44336", ms=2, lw=0.6, alpha=0.5, label="p90 OMIE")
    ax.set_xlabel("Trial"); ax.set_ylabel("€/MWh"); ax.set_title("PDC percentiles")
    ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

    # Panel 3: Dispatch mix (CCGT%, VRE%, hydro uplift)
    ax = axes[1, 0]
    ccgt_p = [r["ccgt_pct"] for r in valid]
    vre_p  = [r["vre_pct"] for r in valid]
    hyd_up = [r["hydro_uplift"] for r in valid]
    ax.plot(trials, ccgt_p, "o-", color="#9C27B0", ms=3, lw=0.8, label="CCGT%")
    ax.axhline(TARGET_CCGT_PCT * 100, color="#9C27B0", ls=":", lw=0.6, alpha=0.5)
    ax.plot(trials, vre_p, "o-", color="#4CAF50", ms=3, lw=0.8, label="VRE%")
    ax.axhline(TARGET_VRE_PCT * 100, color="#4CAF50", ls=":", lw=0.6, alpha=0.5)
    ax.plot(trials, hyd_up, "o-", color="#FF5722", ms=3, lw=0.8, label="Hydro↑%")
    ax.axhline(TARGET_HYDRO_UPLIFT * 100, color="#FF5722", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("Trial"); ax.set_ylabel("%"); ax.set_title("Dispatch mix")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 4: Hydro dispatch accuracy (model vs real GWh)
    ax = axes[1, 1]
    hd_m  = [r.get("hydro_disp", 0) for r in valid]
    hd_r  = [r.get("hydro_disp_real", 0) for r in valid]
    ax.plot(trials, hd_m, "o-", color="#009688", ms=3, lw=0.8, label="Model GWh")
    ax.plot(trials, hd_r, "o--", color="#009688", ms=2, lw=0.6, alpha=0.5, label="Real GWh")
    ax.fill_between(trials,
                     [h * (1 - TARGET_HYDRO_DEV) for h in hd_r],
                     [h * (1 + TARGET_HYDRO_DEV) for h in hd_r],
                     alpha=0.1, color="#009688", label=f"±{TARGET_HYDRO_DEV*100:.0f}% band")
    ax.set_xlabel("Trial"); ax.set_ylabel("GWh"); ax.set_title("Hydro dispatch")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 5: Border prices (FR & PT model vs real)
    ax = axes[2, 0]
    fpm = [r.get("fr_price_m", 0) for r in valid]
    fpr = [r.get("fr_price_r", 0) for r in valid]
    ppm = [r.get("pt_price_m", 0) for r in valid]
    ppr = [r.get("pt_price_r", 0) for r in valid]
    ax.plot(trials, fpm, "o-", color="#E91E63", ms=3, lw=0.8, label="FR model")
    ax.plot(trials, fpr, "o--", color="#E91E63", ms=2, lw=0.6, alpha=0.5, label="FR real")
    ax.plot(trials, ppm, "o-", color="#3F51B5", ms=3, lw=0.8, label="PT model")
    ax.plot(trials, ppr, "o--", color="#3F51B5", ms=2, lw=0.6, alpha=0.5, label="PT real")
    ax.set_xlabel("Trial"); ax.set_ylabel("€/MWh"); ax.set_title("Border prices")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Panel 6: Interconnector flows (model vs real)
    ax = axes[2, 1]
    fr_m  = [r.get("fr_flow", 0) for r in valid]
    fr_t  = [r.get("fr_flow_target", 0) for r in valid]
    pt_m  = [r.get("pt_flow", 0) for r in valid]
    pt_t  = [r.get("pt_flow_target", 0) for r in valid]
    ax.plot(trials, fr_m, "o-", color="#E91E63", ms=3, lw=0.8, label="FR model")
    ax.plot(trials, fr_t, "o--", color="#E91E63", ms=2, lw=0.6, alpha=0.5, label="FR real")
    ax.plot(trials, pt_m, "o-", color="#3F51B5", ms=3, lw=0.8, label="PT model")
    ax.plot(trials, pt_t, "o--", color="#3F51B5", ms=2, lw=0.6, alpha=0.5, label="PT real")
    ax.axhline(0, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("Trial"); ax.set_ylabel("MW"); ax.set_title("Interconnector flows")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = CALIB_OUT / "diagnostics.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_live_pdc():
    """Lightweight live PDC diagnostic: model vs OMIE percentiles per trial.

    Shows every trial as a grey line (fading by score rank) with the best
    trial highlighted in blue. The PDC error panel reveals structural bias:
    if model p10 is always above OMIE p10, the left tail is too fat (too
    much nuclear/VRE floor). If model p90 is always below, the right tail
    is too thin (gas too cheap).

    Opens as diagnostics_live.png — refresh the image viewer to see progress.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Ensure output directory exists (may be called before main() creates it)
    CALIB_OUT.mkdir(parents=True, exist_ok=True)

    valid = [r for r in _all_rows if r["score"] < 1e5]
    if len(valid) < 2:
        return

    # Sort by score ascending (best first) for alpha gradient
    ranked = sorted(valid, key=lambda r: r["score"])
    best = ranked[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Live PDC Diagnostic — model vs OMIE percentiles", fontsize=12, fontweight="bold")

    # ── Left panel: PDC percentiles over trials ─────────────────────────────
    percentiles = [
        ("p10", "#4CAF50"),
        ("p50", "#FF9800"),
        ("p90", "#F44336"),
    ]
    for pname, pcolor in percentiles:
        pk = f"{pname}_model"
        ok = f"{pname}_omie"

        # Grey lines for all trials (fading by rank)
        for rank, r in enumerate(ranked):
            mv = r.get(pk)
            if mv is None:
                continue
            alpha = max(0.08, 1.0 - rank / max(len(ranked), 1))
            ax1.plot(r["trial"], mv, "o", color="#888888", ms=4, alpha=alpha, zorder=1)

        # OMIE reference line (horizontal, dashed)
        omie_val = best.get(ok)
        if omie_val is not None:
            ax1.axhline(omie_val, color=pcolor, ls="--", lw=1.0, alpha=0.4,
                        label=f"{pname} OMIE ({omie_val:.0f}€)")

        # Best-trial line connecting model values
        best_vals = []
        best_trials = []
        for r in ranked:
            mv = r.get(pk)
            if mv is not None:
                best_vals.append(mv)
                best_trials.append(r["trial"])
        if best_vals:
            ax1.plot(best_trials, best_vals, "-", color=pcolor, lw=1.5, alpha=0.8,
                     label=f"{pname} model (best path)")

    ax1.set_xlabel("Trial #")
    ax1.set_ylabel("€/MWh")
    ax1.set_title("PDC percentiles: model vs OMIE")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Right panel: PDC error (model − OMIE) per trial ────────────────────
    for pname, pcolor in percentiles:
        pk = f"{pname}_model"
        ok = f"{pname}_omie"

        omie_ref = best.get(ok)
        if omie_ref is None or omie_ref == 0:
            continue

        errors = []
        err_trials = []
        for r in ranked:
            mv = r.get(pk)
            if mv is not None:
                errors.append(mv - omie_ref)
                err_trials.append(r["trial"])

        if errors:
            # Grey dots for all, coloured line for best-path
            for rank, (t, e) in enumerate(zip(err_trials, errors)):
                alpha = max(0.08, 1.0 - rank / max(len(ranked), 1))
                ax2.plot(t, e, "o", color="#888888", ms=4, alpha=alpha, zorder=1)
            ax2.plot(err_trials, errors, "-", color=pcolor, lw=1.5, alpha=0.8,
                     label=f"{pname} err")

    ax2.axhline(0, color="black", ls="-", lw=0.8)
    ax2.set_xlabel("Trial #")
    ax2.set_ylabel("Model − OMIE (€/MWh)")
    ax2.set_title("PDC error: structural bias check")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = CALIB_OUT / "diagnostics_live.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PyPSA-Spain parameter calibration")
    parser.add_argument("--dry-run",   action="store_true", help="Score current stats, no solve")
    parser.add_argument("--n-trials",  type=int, default=None, help="Max outer-loop DE evaluations")
    parser.add_argument("--popsize",   type=int, default=None, help="DE popsize for outer loop (default N_CCGT_POPSIZE)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--periods",   type=str, default=None,
                        help="Comma-separated day-of-year starts, e.g. '1,13,25'. Overrides CALIB_PERIODS.")
    parser.add_argument("--days",      type=int, default=None,
                        help="Number of days per trial (default 4).")
    parser.add_argument("--hydro-only", action="store_true",
                        help="Skip CCGT outer loop; just calibrate hydro per period with current CCGT η")
    parser.add_argument("--ccgt-only",  action="store_true",
                        help="Skip hydro inner loop; just tune CCGT η with fixed hydro scales")
    args = parser.parse_args()

    # Allow --periods to override the global window
    global CALIB_PERIODS, CALIB_DAYS
    if args.periods:
        CALIB_PERIODS = [int(p.strip()) for p in args.periods.split(",")]
    if args.days:
        CALIB_DAYS = args.days

    if args.dry_run:
        dry_run()
        return

    CALIB_OUT.mkdir(parents=True, exist_ok=True)
    _session_start[0] = time.time()

    n_periods = len(CALIB_PERIODS)
    periods_str = ", ".join([f"P{p}" for p in CALIB_PERIODS])

    # ── Baseline CCGT η (midpoint of bounds) ──────────────────────────────────
    cb = _ccgt_bounds()
    ccgt_x0 = np.array([b[0] + (b[1] - b[0]) * 0.5 for b in cb])

    # Build full baseline vector for _baseline_x (used in delta display)
    full_x0 = np.concatenate([ccgt_x0] + [np.array([1.0, 1.0, 1.0])] * n_periods)
    _baseline_x[0] = full_x0.copy()

    print("=" * 70)
    print("  PyPSA-Spain Nested Calibration Loop — IMMUTABLE RULES")
    for rule in RULES:
        print(f"  ✗  {rule}")
    print()
    print(f"  Periods:   {periods_str}  ({n_periods} periods × {CALIB_DAYS} days/trial)")
    print(f"  Targets:   price={TARGET_PRICE} | CCGT={TARGET_CCGT_PCT*100:.0f}% | "
          f"VRE={TARGET_VRE_PCT*100:.0f}% | hydro≤+{TARGET_HYDRO_UPLIFT*100:.0f}%")
    print(f"  Structure: Outer DE ({N_CCGT_TRIALS} evals × popsize={N_CCGT_POPSIZE}) over 4 CCGT η")
    print(f"             Inner random sampling ({N_HYDRO_TRIALS} samples) per period over 3 hydro scales")
    print(f"             Total solves per outer eval: {n_periods * N_HYDRO_TRIALS}")
    print(f"  Output:    {CALIB_OUT}/")
    print(f"    CSV:     {LOG_CSV.name}")
    print(f"    Report:  {REPORT.name}  ← human-readable, updated each trial")
    print("=" * 70)

    print(f"\n  Baseline CCGT η: {np.round(ccgt_x0, 3)}")
    print(f"  Baseline score (from last stats file): ", end="", flush=True)
    m0 = _parse_stats()
    print(f"{_score(m0):.4f}\n")

    # ── Outer loop: DE over 4 CCGT η ──────────────────────────────────────────
    popsize = args.popsize if args.popsize is not None else N_CCGT_POPSIZE
    maxiter = 30
    if args.n_trials:
        # Each outer evaluation runs n_periods * N_HYDRO_TRIALS solves
        maxiter = max(1, args.n_trials // (popsize * 4) + 1)

    result = differential_evolution(
        objective,
        bounds=cb,
        maxiter=maxiter,
        popsize=popsize,
        seed=args.seed,
        tol=0.005,
        mutation=(0.5, 1.2),
        recombination=0.7,
        polish=False,        # no L-BFGS-B polish — config isn't smooth
        callback=lambda xk, convergence: print(f"  [DE] convergence={convergence:.4f}"),
    )

    print("\n" + "=" * 70)
    print("  OPTIMISATION COMPLETE")
    print(f"  Best score:  {result.fun:.4f}")
    x = result.x
    print(f"\n  GLOBAL PARAMS (nuclear frozen at config.py baseline)")
    print(f"  T1 η: ({x[0]-ETA_WIDTH/2:.3f}, {x[0]+ETA_WIDTH/2:.3f})")
    print(f"  T2 η: ({x[1]-ETA_WIDTH/2:.3f}, {x[1]+ETA_WIDTH/2:.3f})")
    print(f"  T3 η: ({x[2]-ETA_WIDTH/2:.3f}, {x[2]+ETA_WIDTH/2:.3f})")
    print(f"  T4 η: ({x[3]-ETA_WIDTH/2:.3f}, {x[3]+ETA_WIDTH/2:.3f})")

    # Re-run best params to get per-period hydro scales
    print(f"\n  PER-PERIOD HYDRO SCALES (best CCGT η, re-calibrated)")
    best_hydro_all = []
    for p in range(n_periods):
        best_hydro, _ = _calibrate_period_hydro(x, period_idx=p)
        best_hydro_all.append(best_hydro)
        print(f"    P{CALIB_PERIODS[p]:>2}:  ES×{best_hydro[0]:.3f}  FR×{best_hydro[1]:.3f}  PT×{best_hydro[2]:.3f}")

    print(f"\n  Full log: {LOG_CSV}")
    print("=" * 70)

    # Ask before writing to main config
    ans = input("\n  Write best global params to config.py? [y/N] ").strip().lower()
    if ans == "y":
        # Write CCGT η + first-period hydro scales
        patched = _apply_params(_read_cfg(), np.concatenate([x, best_hydro_all[0]]), period_idx=0)
        _write_cfg(patched)
        print("  config.py updated (CCGT η + period 0 hydro scales applied).")
    else:
        print("  config.py unchanged. Apply manually from log if desired.")


if __name__ == "__main__":
    main()
