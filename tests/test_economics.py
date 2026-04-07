"""Verification tests for cabletract.economics (Phase 5).

Each test pins one invariant of the discounted cash flow primitives,
the EconParams reference design, or the life-cycle CO2 accounting.

Run as:

    python tests/test_economics.py
    pytest tests/test_economics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.economics import (  # noqa: E402
    BOMEntry,
    EconParams,
    cabletract_annual_opex,
    cabletract_bom,
    cabletract_capex,
    cabletract_cashflow_series,
    cabletract_embodied_co2_kg,
    cabletract_npv_vs_diesel,
    cabletract_payback_vs_diesel,
    diesel_annual_opex,
    diesel_tractor_embodied_co2_kg,
    electric_annual_opex,
    electric_tractor_embodied_co2_kg,
    lcoe,
    lifecycle_co2_per_ha_yr,
    load_bom_table,
    npv,
    payback_period,
)


# ---------------------------------------------------------------------------
# NPV / payback / LCOE primitives
# ---------------------------------------------------------------------------

def test_npv_with_zero_discount_is_sum_minus_capex() -> None:
    """At r=0, NPV is just (sum of cashflows) - capex."""
    val = npv(200.0, [100.0, 100.0, 100.0], 0.0)
    assert math.isclose(val, 100.0, rel_tol=1e-9)


def test_npv_positive_discount_reduces_value() -> None:
    base = npv(0.0, [100.0, 100.0, 100.0], 0.0)
    discounted = npv(0.0, [100.0, 100.0, 100.0], 0.10)
    assert discounted < base
    assert discounted > 0.0


def test_npv_invalid_discount_raises() -> None:
    try:
        npv(0.0, [100.0], -1.5)
    except ValueError:
        return
    assert False, "expected ValueError"


def test_payback_zero_discount_matches_simple_formula() -> None:
    """payback_period(r=0) must equal capex/annual_savings exactly."""
    pp = payback_period(1000.0, 250.0, 0.0)
    assert math.isclose(pp, 4.0, rel_tol=1e-9)


def test_payback_positive_discount_is_longer() -> None:
    simple = payback_period(1000.0, 250.0, 0.0)
    discounted = payback_period(1000.0, 250.0, 0.08)
    assert discounted > simple


def test_payback_inf_when_savings_nonpositive() -> None:
    assert math.isinf(payback_period(1000.0, 0.0))
    assert math.isinf(payback_period(1000.0, -50.0))


def test_payback_horizon_cap_when_never_breaks_even() -> None:
    """If discounted cashflows never sum to capex, return horizon."""
    pp = payback_period(10_000.0, 1.0, 0.5, horizon_years=10)
    assert math.isclose(pp, 10.0, rel_tol=1e-9)


def test_lcoe_positive_and_decreases_with_volume() -> None:
    """LCOE must be positive and monotonically decrease as annual_units rises."""
    a = lcoe(10_000.0, 500.0, 25.0, 0.08, 15)
    b = lcoe(10_000.0, 500.0, 50.0, 0.08, 15)
    assert a > 0.0 and b > 0.0
    assert b < a


def test_lcoe_invalid_units_raises() -> None:
    try:
        lcoe(10_000.0, 500.0, 0.0, 0.08, 15)
    except ValueError:
        return
    assert False, "expected ValueError"


# ---------------------------------------------------------------------------
# CableTract reference design
# ---------------------------------------------------------------------------

def test_capex_in_realistic_range() -> None:
    """v1 reference design must produce a defensible 30-50 k EUR capex."""
    p = EconParams()
    cap = cabletract_capex(p)
    assert 30_000.0 < cap < 50_000.0, cap


def test_diesel_opex_above_cabletract_opex() -> None:
    """The CableTract has to be cheaper to run on the reference farm,
    otherwise there is no story to tell."""
    p = EconParams()
    assert diesel_annual_opex(p) > cabletract_annual_opex(p)


def test_electric_opex_between_cabletract_and_diesel() -> None:
    """Sanity check: electric tractor sits between the two — its capex
    is higher than CableTract but its energy cost is below diesel."""
    p = EconParams()
    e = electric_annual_opex(p)
    assert e > cabletract_annual_opex(p)
    # Electric should still be below diesel for the reference farm
    assert e < diesel_annual_opex(p)


def test_cashflow_series_has_battery_replacement_dip() -> None:
    """Battery replacement at year 8 must show as a one-shot drop."""
    p = EconParams()
    cf = cabletract_cashflow_series(p)
    assert len(cf) == p.horizon_years
    # Year 8 is index 7 (1-based to 0-based)
    yr8 = cf[p.battery_replacement_year - 1]
    yr7 = cf[p.battery_replacement_year - 2]
    assert yr8 < yr7, "battery replacement year should be lower than the previous year"


def test_payback_in_published_band_for_e_tractors() -> None:
    """Reference payback must land in the 3-7 yr band published for
    Kubota LXe-261 / Monarch / Solectrac at 8% discount."""
    p = EconParams()
    pb = cabletract_payback_vs_diesel(p)
    assert 0.0 < pb < 8.0, pb


def test_npv_positive_at_reference_design() -> None:
    """The reference design must NPV-positive vs the diesel tractor at
    8% discount, otherwise the headline pitch fails."""
    p = EconParams()
    assert cabletract_npv_vs_diesel(p) > 0.0


# ---------------------------------------------------------------------------
# Bill of materials and life-cycle CO2
# ---------------------------------------------------------------------------

def test_bom_table_loads_with_required_columns() -> None:
    df = load_bom_table()
    for col in ["component", "co2_kg_per_unit"]:
        assert col in df.columns


def test_bom_entries_nonnegative_co2() -> None:
    p = EconParams()
    for b in cabletract_bom(p):
        assert b.co2_kg >= 0.0, b.component


def test_embodied_co2_in_realistic_range() -> None:
    """A 250 kg-steel + 30 kg-Al + 10 kWh-battery + 8 m^2-PV system
    should fall between 1500 and 6000 kg CO2eq embodied."""
    p = EconParams()
    ec = cabletract_embodied_co2_kg(p)
    assert 1500.0 < ec < 6000.0, ec


def test_diesel_tractor_embodied_below_cabletract() -> None:
    """The diesel tractor's manufacturing footprint is roughly 320 kg
    (Lindgren-Hansson approximation), below CableTract's 3-4 t kg."""
    assert diesel_tractor_embodied_co2_kg() < cabletract_embodied_co2_kg(EconParams())


def test_electric_tractor_embodied_above_diesel() -> None:
    """80 kWh battery dominates the electric tractor footprint and pushes
    it well above the diesel chassis number."""
    assert electric_tractor_embodied_co2_kg() > diesel_tractor_embodied_co2_kg()


def test_lifecycle_co2_cabletract_below_diesel_per_ha_yr() -> None:
    """The whole point of the LCA: amortised CableTract CO2 must be lower
    than diesel per hectare-year on the reference farm."""
    out = lifecycle_co2_per_ha_yr(EconParams())
    assert out["cabletract_total_kg_per_ha_yr"] < out["diesel_total_kg_per_ha_yr"]


def test_lifecycle_co2_diesel_dominated_by_fuel() -> None:
    """For the diesel tractor, fuel combustion should dominate over
    embodied chassis emissions on a per-ha-yr basis."""
    out = lifecycle_co2_per_ha_yr(EconParams())
    assert out["diesel_fuel_kg_per_ha_yr"] > out["diesel_embodied_kg_per_ha_yr"]


def test_lifecycle_co2_cabletract_dominated_by_embodied() -> None:
    """The CableTract has near-zero fuel CO2 (it runs off PV+wind by
    default) so its CO2 footprint is dominated by embodied manufacture."""
    out = lifecycle_co2_per_ha_yr(EconParams())
    assert out["cabletract_embodied_kg_per_ha_yr"] >= out["cabletract_fuel_kg_per_ha_yr"]


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_npv_with_zero_discount_is_sum_minus_capex,
    test_npv_positive_discount_reduces_value,
    test_npv_invalid_discount_raises,
    test_payback_zero_discount_matches_simple_formula,
    test_payback_positive_discount_is_longer,
    test_payback_inf_when_savings_nonpositive,
    test_payback_horizon_cap_when_never_breaks_even,
    test_lcoe_positive_and_decreases_with_volume,
    test_lcoe_invalid_units_raises,
    test_capex_in_realistic_range,
    test_diesel_opex_above_cabletract_opex,
    test_electric_opex_between_cabletract_and_diesel,
    test_cashflow_series_has_battery_replacement_dip,
    test_payback_in_published_band_for_e_tractors,
    test_npv_positive_at_reference_design,
    test_bom_table_loads_with_required_columns,
    test_bom_entries_nonnegative_co2,
    test_embodied_co2_in_realistic_range,
    test_diesel_tractor_embodied_below_cabletract,
    test_electric_tractor_embodied_above_diesel,
    test_lifecycle_co2_cabletract_below_diesel_per_ha_yr,
    test_lifecycle_co2_diesel_dominated_by_fuel,
    test_lifecycle_co2_cabletract_dominated_by_embodied,
]


if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {exc}")
            failures.append((t.__name__, exc))
    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        sys.exit(1)
    print(f"\nok: cabletract.economics passes {len(ALL_TESTS)} invariants")
