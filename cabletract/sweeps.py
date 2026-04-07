"""Legacy v1 sensitivity sweeps over the deterministic ``run_single`` model.

These were previously inlined as ``analyze_setup_time``, ``analyze_draft_load``,
``analyze_field_length``, ``analyze_shape_efficiency``, ``offgrid_feasibility_grid``
and ``compare_operation_types`` in ``cabletract_analysis.py:199-303``. They are
preserved here unchanged so that the legacy figures still reproduce, and so
that Phase 5 (Monte Carlo + Sobol) has a clean comparison baseline.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .params import CableTractParams
from .simulate import run_single


def _clone(base: CableTractParams) -> CableTractParams:
    """Return a fresh CableTractParams populated from ``base``."""
    return CableTractParams(**asdict(base))


def analyze_setup_time(
    base: CableTractParams, setup_range_s: Iterable[float]
) -> pd.DataFrame:
    rows = []
    for t in setup_range_s:
        p = _clone(base)
        p.setup_time_s = float(t)
        res = run_single(p)
        row = asdict(res)
        row["setup_time_s"] = t
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_draft_load(
    base: CableTractParams, draft_range_N: Iterable[float]
) -> pd.DataFrame:
    rows = []
    for d in draft_range_N:
        p = _clone(base)
        p.draft_load_N = float(d)
        res = run_single(p)
        row = asdict(res)
        row["draft_load_N"] = d
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_field_length(
    base: CableTractParams, length_range_m: Iterable[float]
) -> pd.DataFrame:
    rows = []
    for L in length_range_m:
        p = _clone(base)
        p.span_m = float(L)
        res = run_single(p)
        row = asdict(res)
        row["span_m"] = L
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_shape_efficiency(
    base: CableTractParams, shape_factors: Iterable[float]
) -> pd.DataFrame:
    rows = []
    for sf in shape_factors:
        p = _clone(base)
        p.shape_efficiency = float(sf)
        res = run_single(p)
        row = asdict(res)
        row["shape_efficiency"] = sf
        rows.append(row)
    return pd.DataFrame(rows)


def offgrid_feasibility_grid(
    base: CableTractParams,
    solar_area_values: Iterable[float],
    battery_values_Wh: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for area in solar_area_values:
        for batt in battery_values_Wh:
            p = _clone(base)
            p.solar_area_m2 = float(area)
            p.battery_Wh = float(batt)
            res = run_single(p)
            rows.append(
                {
                    "solar_area_m2": area,
                    "battery_Wh": batt,
                    "decares_per_day_offgrid": res.decares_per_day_offgrid,
                    "grid_charge_needed_Wh": res.grid_charge_needed_Wh,
                    "battery_sufficient": res.battery_sufficient,
                }
            )
    return pd.DataFrame(rows)


def compare_operation_types(base: CableTractParams) -> pd.DataFrame:
    """Five hand-picked operation scenarios from the v1 manuscript scaffold.

    Phase 2 will replace these with the ASABE D497 implement library and
    stochastic draft sampling, but the deterministic v1 version is kept
    here so that the legacy figure (``operation_type_comparison.png``) can
    still be reproduced.
    """

    scenarios = [
        {
            "operation": "Heavy tillage",
            "draft_load_N": 3000.0,
            "electric_tractor_Wh_per_decare": 4000.0,
            "fuel_l_per_decare": 3.0,
        },
        {
            "operation": "Medium cultivation",
            "draft_load_N": 1800.0,
            "electric_tractor_Wh_per_decare": 2500.0,
            "fuel_l_per_decare": 1.6,
        },
        {
            "operation": "Seeding",
            "draft_load_N": 1000.0,
            "electric_tractor_Wh_per_decare": 1800.0,
            "fuel_l_per_decare": 1.1,
        },
        {
            "operation": "Spraying",
            "draft_load_N": 500.0,
            "electric_tractor_Wh_per_decare": 1200.0,
            "fuel_l_per_decare": 0.8,
        },
        {
            "operation": "Weeding",
            "draft_load_N": 700.0,
            "electric_tractor_Wh_per_decare": 1400.0,
            "fuel_l_per_decare": 0.9,
        },
    ]

    rows = []
    for sc in scenarios:
        p = _clone(base)
        p.draft_load_N = sc["draft_load_N"]
        p.electric_tractor_Wh_per_decare = sc["electric_tractor_Wh_per_decare"]
        p.fuel_l_per_decare = sc["fuel_l_per_decare"]
        res = run_single(p)
        cabletract_wh = res.energy_per_decare_Wh
        electric_wh = p.electric_tractor_Wh_per_decare
        diesel_wh_equiv = np.nan  # legacy: leave blank unless an LHV conversion is added
        rows.append(
            {
                "operation": sc["operation"],
                "draft_load_N": p.draft_load_N,
                "cabletract_Wh_per_decare": cabletract_wh,
                "electric_tractor_Wh_per_decare": electric_wh,
                "relative_vs_electric": cabletract_wh / electric_wh,
                "fuel_l_per_decare": p.fuel_l_per_decare,
                "fuel_savings_usd_year": res.fuel_savings_usd_year,
                "electricity_savings_usd_year": res.electricity_savings_usd_year,
            }
        )
    return pd.DataFrame(rows)
