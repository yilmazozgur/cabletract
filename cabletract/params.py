"""Parameters and result containers for the CableTract feasibility model.

This module is the single source of truth for the CableTract parameter space.
It defines:

- ``CableTractParams``: the complete input parameter set with units. The
  default constructor returns the *generic-implement* baseline (50 m span,
  2700 N draft, 0.5 lumped efficiency, 20 m² PV, 15 kWh battery, 170 d/yr)
  used as the reference for backward compatibility with the legacy
  ``reference_case.csv``. The :meth:`CableTractParams.codesigned` classmethod
  returns the **co-designed reference**: a CableTract sized for the
  CableTract-native implement library defined in
  ``cabletract/data/asabe_d497_cabletract.csv`` (1.5 m strip, 1800 N median
  draft, ~3.5 kN P90 primary tillage, 1.5 kW continuous motor, smaller PV
  and battery).
- ``CableTractResults``: the bag of computed scalar outputs returned by
  ``cabletract.simulate.run_single``.
- ``salib_problem``: a SALib-style problem dict listing the parameter ranges
  used by Phase 5 global sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CableTractParams:
    """Complete CableTract parameter set.

    Units:
        forces in newtons (N), distances in metres (m), times in seconds (s),
        powers in watts (W), energies in watt-hours (Wh), prices in USD,
        areas in square metres (m^2), efficiencies and ratios are dimensionless.

    The ``shape_efficiency`` field is a placeholder scalar in the v1 model
    (``cabletract_analysis.py``); it is replaced in Phase 4 by the polygon
    coverage planner in ``cabletract.layout``.
    """

    span_m: float = 50.0                    # Distance between Main Unit and Anchor
    width_m: float = 2.0                    # Effective strip width per round
    implement_weight_N: float = 1000.0      # Implement + cable equivalent load
    draft_load_N: float = 2700.0            # Draft load during active operation
    system_weight_N: float = 3000.0         # System total equivalent travel load
    system_travel_per_round_m: float = 2.0  # Lateral reposition distance per round
    winch_efficiency: float = 0.5           # Overall winch/motor efficiency (lumped)
    solar_power_W_m2: float = 150.0         # Solar specific power
    solar_area_m2: float = 20.0             # Unfolded solar area
    solar_hours_per_day: float = 6.0        # Solar production window
    wind_power_W: float = 100.0             # Wind generator power
    wind_hours_per_day: float = 12.0        # Wind operation hours
    setup_time_s: float = 60.0              # Time lost between rounds
    operation_time_ratio: float = 0.8       # Fraction of usable time for loaded operation
    battery_Wh: float = 15000.0             # Battery size
    operating_hours_per_day: float = 10.0   # Daily total operating time window
    operating_days_per_year: float = 170.0  # Annual operating days
    fuel_l_per_decare: float = 2.0          # Diesel tractor fuel use baseline
    fuel_price_usd_per_l: float = 1.0       # Fuel price
    electric_tractor_Wh_per_decare: float = 4000.0
    electricity_price_usd_per_Wh: float = 0.00016
    cost_cabletract_usd: float = 15000.0
    sales_margin_pct: float = 33.0
    shape_efficiency: float = 1.0           # 1.0 means ideal rectangular field

    @classmethod
    def codesigned(cls) -> "CableTractParams":
        """Co-designed reference: CableTract sized for the CableTract-native
        implement library, not for off-the-shelf tractor implements.

        Justification for each changed parameter (vs the generic baseline):

        - ``draft_load_N = 1800``: median P50 across the 10-implement
          co-designed library (Phase 2, F4 panel b). The original 2700 N
          number was a hand-pick aimed at heavy-tillage edge cases; the
          co-designed library shifts the *typical* operation down to ~1.5 kN
          (cultivator, planter, hoe, sprayer, mower) with the heaviest
          primary-tillage operation (narrow ripper / chisel) sitting at
          3–4 kN P50 — still within a comfortable anchor envelope.
        - ``width_m = 1.5``: matches the co-designed implement strip widths
          (1.5 m disk, 1.5 m sweep, 0.6–0.9 m chisel/cultivator carriages).
        - ``implement_weight_N = 600``: ~40 % lighter implement assembly
          because the bodies no longer need to survive a 6 t tractor's
          hitch loads. Frame steel mass drops from ~120 kg to ~70 kg per
          implement.
        - ``system_weight_N = 2200``: smaller motor (1.5 kW vs 3 kW),
          smaller battery (9 vs 15 kWh), narrower carriage frame.
        - ``solar_area_m2 = 15``: lower energy demand allows a smaller PV
          array — reduces capex and visual footprint.
        - ``battery_Wh = 9000``: peak winch input drops from ~3 kW to
          ~1.5 kW, so the battery just needs to bridge the off-PV hours
          rather than carry a heavy load surge.
        - ``cost_cabletract_usd = 35570``: itemised BOM total from the
          Phase 5 economics chain (``EconParams.codesigned``):
          €17,500 main unit + €7,500 anchor + €3,420 battery (9 kWh) +
          €1,650 PV (15 m²) + €1,500 wind + €4,000 install. The legacy
          generic baseline used €11,500 as a placeholder lump sum; the
          codesigned reference now mirrors §5.8 so the variant table and
          the NPV chain agree on the same number. Currency is nominal
          EUR — the field name still says ``usd`` for backwards
          compatibility with the legacy CSVs.
        - ``span_m`` and ``setup_time_s``: unchanged. Span is set by the
          field, setup is set by the operator.
        - ``operating_days_per_year``: unchanged at 170. The co-design
          does not change the agronomic calendar.
        - ``winch_efficiency``: unchanged at 0.5 (lumped). Phase 1's
          decomposed drivetrain chain (motor × controller × gearbox × drum
          × pulley × cable) is the per-component model; this scalar is the
          screening-model fallback used by ``run_single``.
        """
        return cls(
            span_m=50.0,
            width_m=1.5,
            implement_weight_N=600.0,
            draft_load_N=1800.0,
            system_weight_N=2200.0,
            system_travel_per_round_m=2.0,
            winch_efficiency=0.5,
            solar_power_W_m2=150.0,
            solar_area_m2=15.0,
            solar_hours_per_day=6.0,
            wind_power_W=100.0,
            wind_hours_per_day=12.0,
            setup_time_s=60.0,
            operation_time_ratio=0.8,
            battery_Wh=9000.0,
            operating_hours_per_day=10.0,
            operating_days_per_year=170.0,
            fuel_l_per_decare=2.0,
            fuel_price_usd_per_l=1.40,
            electric_tractor_Wh_per_decare=4000.0,
            electricity_price_usd_per_Wh=0.00016,
            cost_cabletract_usd=35570.0,
            sales_margin_pct=33.0,
            shape_efficiency=1.0,
        )


@dataclass
class CableTractResults:
    """Scalar outputs of one ``run_single`` evaluation.

    Field order matches ``cabletract_analysis.py:64-93`` exactly so that the
    legacy ``reference_case.csv`` regression test continues to pass.
    """

    rounds_per_decare: float
    work_per_round_J: float
    energy_per_round_Wh: float
    energy_per_decare_Wh: float
    harvested_energy_per_day_Wh: float
    decares_per_day_energy_limited: float
    rounds_per_day: float
    usable_time_s: float
    operation_time_s: float
    travel_time_s: float
    operation_distance_m_day: float
    travel_distance_m_day: float
    operation_speed_m_s: float
    travel_speed_m_s: float
    winch_output_power_W: float
    winch_input_power_W: float
    harvested_power_W: float
    surplus_power_W: float
    battery_chargeable_Wh: float
    battery_required_Wh: float
    grid_charge_needed_Wh: float
    battery_sufficient: bool
    decares_per_day_offgrid: float
    decares_per_year: float
    fuel_savings_usd_year: float
    electricity_savings_usd_year: float
    sales_price_usd: float
    payback_months_vs_fuel: float
    payback_months_vs_electric: float


def salib_problem() -> Dict[str, Any]:
    """Return a SALib-compatible problem dictionary for Phase 5 Sobol analysis.

    The bounds are set to physically defensible ranges around the reference
    case. They will be tightened or widened in Phase 5 once Phase 1 has
    decomposed ``winch_efficiency`` into its component efficiencies.
    """

    names = [
        "span_m",
        "width_m",
        "implement_weight_N",
        "draft_load_N",
        "system_weight_N",
        "winch_efficiency",
        "solar_power_W_m2",
        "solar_area_m2",
        "solar_hours_per_day",
        "wind_power_W",
        "wind_hours_per_day",
        "setup_time_s",
        "operation_time_ratio",
        "battery_Wh",
        "operating_hours_per_day",
        "operating_days_per_year",
        "fuel_l_per_decare",
        "fuel_price_usd_per_l",
        "electric_tractor_Wh_per_decare",
        "electricity_price_usd_per_Wh",
        "cost_cabletract_usd",
        "shape_efficiency",
    ]
    bounds = [
        [20.0, 120.0],
        [1.0, 4.0],
        [400.0, 2000.0],
        [200.0, 4000.0],
        [1500.0, 6000.0],
        [0.30, 0.85],
        [80.0, 220.0],
        [5.0, 50.0],
        [3.0, 9.0],
        [0.0, 500.0],
        [0.0, 18.0],
        [10.0, 180.0],
        [0.5, 0.95],
        [2000.0, 40000.0],
        [6.0, 14.0],
        [120.0, 220.0],
        [0.5, 4.0],
        [0.5, 2.0],
        [1500.0, 6000.0],
        [0.00008, 0.00030],
        [8000.0, 30000.0],
        [0.5, 1.0],
    ]
    return {
        "num_vars": len(names),
        "names": names,
        "bounds": bounds,
    }
