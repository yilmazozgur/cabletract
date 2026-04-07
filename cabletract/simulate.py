"""Atomic CableTract evaluation: ``run_single(params) -> Results``.

This is the v1 first-principles screening model, lifted unchanged from
``cabletract_analysis.py:compute_reference_case`` (lines 96-188 of the legacy
script). It is intentionally identical so that ``tests/test_simulate.py`` can
regression-lock the existing ``reference_case.csv`` numbers to 6 significant
figures.

Phases 1-3 (physics, soil, energy) will replace pieces of this function in
place; the function signature ``run_single(params) -> Results`` is the stable
contract that the rest of the package (sweeps, uncertainty, ml) depends on.
"""

from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .params import CableTractParams, CableTractResults


def run_single(p: CableTractParams) -> CableTractResults:
    """Compute one CableTract scenario from a parameter set.

    This is the unmodified work-energy / time-budget / harvested-energy
    screening model from the v1 manuscript scaffold. The naming and ordering
    of the result fields are preserved so that the legacy regression CSV is
    reproduced bit-for-bit.
    """

    # Adjust field productivity for non-ideal shape
    effective_width = p.width_m * p.shape_efficiency
    rounds_per_decare = 1000.0 / (p.span_m * effective_width)

    work_per_round_J = (
        p.draft_load_N * p.span_m
        + p.implement_weight_N * p.span_m
        + p.system_weight_N * p.system_travel_per_round_m
    )
    energy_per_round_Wh = work_per_round_J / 3600.0 / p.winch_efficiency
    energy_per_decare_Wh = energy_per_round_Wh * rounds_per_decare

    harvested_energy_per_day_Wh = (
        p.solar_power_W_m2 * p.solar_area_m2 * p.solar_hours_per_day
        + p.wind_power_W * p.wind_hours_per_day
    )
    decares_per_day_energy_limited = harvested_energy_per_day_Wh / energy_per_decare_Wh
    rounds_per_day = decares_per_day_energy_limited * rounds_per_decare

    total_time_s = p.operating_hours_per_day * 3600.0
    usable_time_s = total_time_s - rounds_per_day * p.setup_time_s
    usable_time_s = max(usable_time_s, 1.0)

    operation_time_s = usable_time_s * p.operation_time_ratio
    travel_time_s = usable_time_s - operation_time_s

    operation_distance_m_day = rounds_per_day * p.span_m
    # Same formulation as the uploaded workbook: unloaded return + lateral body travel
    travel_distance_m_day = (
        rounds_per_day * p.span_m
        + decares_per_day_energy_limited * p.width_m * rounds_per_decare
    )

    operation_speed_m_s = operation_distance_m_day / max(operation_time_s, 1.0)
    travel_speed_m_s = travel_distance_m_day / max(travel_time_s, 1.0)

    # Output power on winch based on active loaded segment only
    active_round_time_s = p.span_m / max(operation_speed_m_s, 1e-9)
    winch_output_power_W = (p.draft_load_N * p.span_m) / active_round_time_s
    winch_input_power_W = winch_output_power_W / p.winch_efficiency

    harvested_power_W = p.solar_power_W_m2 * p.solar_area_m2 + p.wind_power_W
    surplus_power_W = harvested_power_W - winch_input_power_W
    battery_chargeable_Wh = max(surplus_power_W, 0.0) * p.solar_hours_per_day

    battery_hours_needed = max(p.operating_hours_per_day - p.solar_hours_per_day, 0.0)
    battery_operated_decares = (
        battery_hours_needed / p.operating_hours_per_day * decares_per_day_energy_limited
    )
    battery_required_Wh = battery_operated_decares * energy_per_decare_Wh
    grid_charge_needed_Wh = max(battery_required_Wh - battery_chargeable_Wh, 0.0)
    battery_sufficient = p.battery_Wh > battery_required_Wh

    decares_per_day_offgrid = (
        decares_per_day_energy_limited - grid_charge_needed_Wh / energy_per_decare_Wh
    )
    decares_per_day_offgrid = max(decares_per_day_offgrid, 0.0)

    decares_per_year = decares_per_day_energy_limited * p.operating_days_per_year
    fuel_savings_usd_year = (
        decares_per_year * p.fuel_l_per_decare * p.fuel_price_usd_per_l
    )
    electricity_savings_usd_year = (
        decares_per_year * p.electric_tractor_Wh_per_decare * p.electricity_price_usd_per_Wh
    )
    sales_price_usd = p.cost_cabletract_usd * (1.0 + p.sales_margin_pct / 100.0)

    payback_months_vs_fuel = sales_price_usd / max(fuel_savings_usd_year, 1e-9) * 12.0
    payback_months_vs_electric = (
        sales_price_usd / max(electricity_savings_usd_year, 1e-9) * 12.0
    )

    return CableTractResults(
        rounds_per_decare=rounds_per_decare,
        work_per_round_J=work_per_round_J,
        energy_per_round_Wh=energy_per_round_Wh,
        energy_per_decare_Wh=energy_per_decare_Wh,
        harvested_energy_per_day_Wh=harvested_energy_per_day_Wh,
        decares_per_day_energy_limited=decares_per_day_energy_limited,
        rounds_per_day=rounds_per_day,
        usable_time_s=usable_time_s,
        operation_time_s=operation_time_s,
        travel_time_s=travel_time_s,
        operation_distance_m_day=operation_distance_m_day,
        travel_distance_m_day=travel_distance_m_day,
        operation_speed_m_s=operation_speed_m_s,
        travel_speed_m_s=travel_speed_m_s,
        winch_output_power_W=winch_output_power_W,
        winch_input_power_W=winch_input_power_W,
        harvested_power_W=harvested_power_W,
        surplus_power_W=surplus_power_W,
        battery_chargeable_Wh=battery_chargeable_Wh,
        battery_required_Wh=battery_required_Wh,
        grid_charge_needed_Wh=grid_charge_needed_Wh,
        battery_sufficient=battery_sufficient,
        decares_per_day_offgrid=decares_per_day_offgrid,
        decares_per_year=decares_per_year,
        fuel_savings_usd_year=fuel_savings_usd_year,
        electricity_savings_usd_year=electricity_savings_usd_year,
        sales_price_usd=sales_price_usd,
        payback_months_vs_fuel=payback_months_vs_fuel,
        payback_months_vs_electric=payback_months_vs_electric,
    )


def results_to_series(r: CableTractResults) -> pd.Series:
    """Convert a Results object to a pandas Series for CSV output."""
    return pd.Series(asdict(r))
