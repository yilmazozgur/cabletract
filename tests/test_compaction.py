"""Verification tests for cabletract.compaction (Phase 4).

Each test pins down one invariant of the contact-pressure model, the
per-vehicle compaction summary, or the side-by-side comparison.

Run as:

    python tests/test_compaction.py
    pytest tests/test_compaction.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapely.geometry import Polygon  # noqa: E402

from cabletract.compaction import (  # noqa: E402
    CABLETRACT_CARRIAGE,
    TRACTOR_REFERENCE,
    VehicleSpec,
    WheelSpec,
    compacted_path_polygon,
    compaction_summary_for_vehicle,
    compare_vehicles_on_field,
    contact_pressure,
)
from cabletract.layout import load_field_corpus  # noqa: E402


# ---------------------------------------------------------------------------
# Contact pressure
# ---------------------------------------------------------------------------

def test_contact_pressure_simple_value() -> None:
    """1000 N over 0.05 m² → 20000 Pa = 20 kPa."""
    p = contact_pressure(1000.0, 0.05)
    assert math.isclose(p, 20000.0, rel_tol=1e-9)


def test_contact_pressure_invalid_area_raises() -> None:
    try:
        contact_pressure(1000.0, 0.0)
    except ValueError:
        return
    assert False, "expected ValueError"


def test_tractor_pressures_in_typical_band() -> None:
    """A reference 80 hp tractor should sit between 100 and 250 kPa, the
    band typically reported by Keller & Lamandé (2010)."""
    for w in TRACTOR_REFERENCE.wheels:
        p_kPa = contact_pressure(w.load_N, w.contact_area_m2) / 1000.0
        assert 100.0 < p_kPa < 250.0, (w.name, p_kPa)


def test_carriage_pressures_below_50_kpa() -> None:
    """The CableTract carriage rollers must be at least 3× lighter than
    a tractor tyre — that is the entire compaction-reduction story."""
    for w in CABLETRACT_CARRIAGE.wheels:
        p_kPa = contact_pressure(w.load_N, w.contact_area_m2) / 1000.0
        assert p_kPa < 50.0, (w.name, p_kPa)


def test_carriage_max_pressure_at_least_3x_lower_than_tractor() -> None:
    p_tractor_kPa = max(contact_pressure(w.load_N, w.contact_area_m2) for w in TRACTOR_REFERENCE.wheels) / 1000.0
    p_carriage_kPa = max(contact_pressure(w.load_N, w.contact_area_m2) for w in CABLETRACT_CARRIAGE.wheels) / 1000.0
    assert p_tractor_kPa > 3.0 * p_carriage_kPa


# ---------------------------------------------------------------------------
# compaction_summary_for_vehicle
# ---------------------------------------------------------------------------

SQUARE = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


def test_summary_for_tractor_returns_field_area() -> None:
    summ = compaction_summary_for_vehicle(SQUARE, TRACTOR_REFERENCE)
    assert math.isclose(summ.field_area_m2, 10000.0, rel_tol=1e-9)
    assert summ.compacted_area_m2 > 0.0


def test_summary_for_empty_polygon_returns_zero() -> None:
    summ = compaction_summary_for_vehicle(Polygon(), TRACTOR_REFERENCE)
    assert summ.field_area_m2 == 0.0
    assert summ.compacted_area_m2 == 0.0


def test_carriage_compacted_area_strictly_smaller_than_tractor() -> None:
    """The defining claim of CableTract: less ground gets driven on."""
    fields = load_field_corpus()
    for f in fields:
        tractor, carriage = compare_vehicles_on_field(f.polygon, span=50.0)
        assert carriage.compacted_area_m2 < tractor.compacted_area_m2, f.id


def test_carriage_energy_index_at_least_10x_lower() -> None:
    """The compaction-energy index reduction must be at least 10× because
    the pressure ratio enters squared in the integrand."""
    summ_t = compaction_summary_for_vehicle(SQUARE, TRACTOR_REFERENCE)
    summ_c = compaction_summary_for_vehicle(SQUARE, CABLETRACT_CARRIAGE)
    assert summ_t.compaction_energy_index > 10.0 * summ_c.compaction_energy_index


def test_summary_pressures_within_wheel_extremes() -> None:
    summ = compaction_summary_for_vehicle(SQUARE, TRACTOR_REFERENCE)
    pressures_kPa = [contact_pressure(w.load_N, w.contact_area_m2) / 1000.0 for w in TRACTOR_REFERENCE.wheels]
    assert min(pressures_kPa) <= summ.mean_pressure_kPa <= max(pressures_kPa)
    assert math.isclose(summ.max_pressure_kPa, max(pressures_kPa), rel_tol=1e-9)


def test_carriage_compacted_fraction_below_5pct_on_square() -> None:
    """On a clean rectangle the carriage should compact <5% of the field."""
    summ = compaction_summary_for_vehicle(SQUARE, CABLETRACT_CARRIAGE, span=50.0)
    assert summ.compacted_area_frac < 0.05


def test_pass_count_scales_compacted_area_linearly() -> None:
    """Doubling pass_count must double the compacted area for a tractor."""
    one_pass = VehicleSpec(
        name="t1",
        wheels=TRACTOR_REFERENCE.wheels,
        track_gauge_m=TRACTOR_REFERENCE.track_gauge_m,
        pass_count=1,
    )
    two_pass = VehicleSpec(
        name="t2",
        wheels=TRACTOR_REFERENCE.wheels,
        track_gauge_m=TRACTOR_REFERENCE.track_gauge_m,
        pass_count=2,
    )
    s1 = compaction_summary_for_vehicle(SQUARE, one_pass)
    s2 = compaction_summary_for_vehicle(SQUARE, two_pass)
    assert math.isclose(s2.compacted_area_m2, 2.0 * s1.compacted_area_m2, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Path polygon visualisation helper
# ---------------------------------------------------------------------------

def test_compacted_path_polygon_for_tractor_is_full_field() -> None:
    pp = compacted_path_polygon(SQUARE, TRACTOR_REFERENCE)
    assert math.isclose(pp.area, SQUARE.area, rel_tol=1e-6)


def test_compacted_path_polygon_for_carriage_is_proper_subset() -> None:
    pp = compacted_path_polygon(SQUARE, CABLETRACT_CARRIAGE, span=50.0)
    assert pp.area < SQUARE.area
    assert pp.within(SQUARE.buffer(1e-6))  # allow tiny boundary slack


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_contact_pressure_simple_value,
    test_contact_pressure_invalid_area_raises,
    test_tractor_pressures_in_typical_band,
    test_carriage_pressures_below_50_kpa,
    test_carriage_max_pressure_at_least_3x_lower_than_tractor,
    test_summary_for_tractor_returns_field_area,
    test_summary_for_empty_polygon_returns_zero,
    test_carriage_compacted_area_strictly_smaller_than_tractor,
    test_carriage_energy_index_at_least_10x_lower,
    test_summary_pressures_within_wheel_extremes,
    test_carriage_compacted_fraction_below_5pct_on_square,
    test_pass_count_scales_compacted_area_linearly,
    test_compacted_path_polygon_for_tractor_is_full_field,
    test_compacted_path_polygon_for_carriage_is_proper_subset,
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
    print(f"\nok: cabletract.compaction passes {len(ALL_TESTS)} invariants")
