"""Verification tests for cabletract.layout (Phase 4).

Each test pins down one invariant of the strip decomposition, the
shape-efficiency definition, the corpus loader, or the farm-tour helper.

Run as:

    python tests/test_layout.py
    pytest tests/test_layout.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shapely.geometry import Polygon  # noqa: E402

from cabletract.layout import (  # noqa: E402
    Field,
    best_orientation_efficiency,
    corpus_shape_efficiency_summary,
    effective_shape_efficiency,
    farm_tour,
    load_field_corpus,
    strip_decomposition,
    strip_plan_segments_xy,
)


# ---------------------------------------------------------------------------
# Strip decomposition basics
# ---------------------------------------------------------------------------

def test_strip_decomposition_unit_square_one_strip() -> None:
    sq = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    segs = strip_decomposition(sq, span=100.0, swath=2.0, orientation_deg=0.0)
    assert len(segs) == 1
    assert math.isclose(segs[0].length_m, 100.0, rel_tol=1e-9)
    assert math.isclose(segs[0].area_m2, 10000.0, rel_tol=1e-9)


def test_strip_decomposition_unit_square_two_strips_at_half_span() -> None:
    sq = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    segs = strip_decomposition(sq, span=50.0, swath=2.0, orientation_deg=0.0)
    assert len(segs) == 2
    assert all(math.isclose(s.length_m, 100.0, rel_tol=1e-9) for s in segs)


def test_strip_decomposition_l_shape_two_strips_with_unequal_area() -> None:
    """L-shape: 100x100 with 40x40 bite from top-right (area 8400 m²)."""
    L = Polygon([(0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100)])
    segs = strip_decomposition(L, span=50.0, swath=2.0, orientation_deg=0.0)
    assert len(segs) == 2
    total_area = sum(s.area_m2 for s in segs)
    assert math.isclose(total_area, L.area, rel_tol=1e-6)


def test_strip_decomposition_total_area_equals_polygon_area() -> None:
    poly = Polygon([(0, 0), (200, 0), (180, 90), (40, 110)])
    segs = strip_decomposition(poly, span=40.0, swath=2.0, orientation_deg=0.0)
    total = sum(s.area_m2 for s in segs)
    assert math.isclose(total, poly.area, rel_tol=1e-6)


def test_strip_decomposition_invalid_inputs_raise() -> None:
    sq = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    try:
        strip_decomposition(sq, span=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        strip_decomposition(sq, span=10.0, swath=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Shape efficiency definition
# ---------------------------------------------------------------------------

def test_shape_efficiency_unit_square_equals_one() -> None:
    sq = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    assert math.isclose(effective_shape_efficiency(sq, span=100.0), 1.0, abs_tol=1e-9)


def test_shape_efficiency_l_shape_below_one() -> None:
    """L: area 8400, 2 strips × 50 × 100 = 10000 → η = 0.84."""
    L = Polygon([(0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100)])
    eta = effective_shape_efficiency(L, span=50.0)
    assert math.isclose(eta, 0.84, rel_tol=1e-6)


def test_shape_efficiency_hollow_square_below_one() -> None:
    """100x100 with 20x20 hole: area 9600, 2 strips × 50 × 100 = 10000 → 0.96."""
    h = Polygon(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [[(40, 40), (60, 40), (60, 60), (40, 60)]],
    )
    eta = effective_shape_efficiency(h, span=50.0)
    assert math.isclose(eta, 0.96, rel_tol=1e-6)


def test_shape_efficiency_in_unit_interval() -> None:
    poly = Polygon([(0, 0), (200, 10), (190, 100), (10, 90)])
    for span in (20.0, 40.0, 80.0, 200.0):
        eta = effective_shape_efficiency(poly, span=span)
        assert 0.0 <= eta <= 1.0 + 1e-9, (span, eta)


def test_shape_efficiency_zero_for_empty_polygon() -> None:
    empty = Polygon()
    assert effective_shape_efficiency(empty, span=10.0) == 0.0


def test_best_orientation_at_least_as_good_as_x_axis() -> None:
    poly = Polygon([(0, 0), (200, 0), (180, 80), (10, 60)])
    eta_x = effective_shape_efficiency(poly, span=40.0, orientation_deg=0.0)
    eta_best, _ = best_orientation_efficiency(poly, span=40.0, n_orient=12)
    assert eta_best >= eta_x - 1e-9


def test_strip_plan_segments_count_matches_decomposition() -> None:
    L = Polygon([(0, 0), (100, 0), (100, 60), (60, 60), (60, 100), (0, 100)])
    segs = strip_decomposition(L, span=50.0)
    plan = strip_plan_segments_xy(L, span=50.0)
    assert len(plan) == len(segs)


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------

def test_corpus_has_50_fields() -> None:
    fields = load_field_corpus()
    assert len(fields) == 50


def test_corpus_classes_complete() -> None:
    fields = load_field_corpus()
    classes = {f.shape_class for f in fields}
    expected = {"rectangle", "L_shape", "irregular_convex", "irregular_concave", "real_shape"}
    assert expected.issubset(classes), expected - classes


def test_corpus_polygons_are_valid() -> None:
    fields = load_field_corpus()
    for f in fields:
        assert f.polygon.is_valid, f.id
        assert f.polygon.area > 0.0, f.id


def test_corpus_summary_returns_one_row_per_field() -> None:
    fields = load_field_corpus()
    rows = corpus_shape_efficiency_summary(fields, span=50.0, swath=2.0, n_orient=4)
    assert len(rows) == len(fields)
    assert all(0.0 <= r["eta_best"] <= 1.0 + 1e-9 for r in rows)
    # Rectangles should achieve eta=1 in best orientation
    rects = [r for r in rows if r["shape_class"] == "rectangle"]
    assert all(math.isclose(r["eta_best"], 1.0, abs_tol=1e-6) for r in rects)


# ---------------------------------------------------------------------------
# Farm tour
# ---------------------------------------------------------------------------

def test_farm_tour_empty_returns_zero() -> None:
    d, order = farm_tour([], depot=(0.0, 0.0))
    assert d == 0.0
    assert order == []


def test_farm_tour_single_field_round_trip() -> None:
    """One field 100 m east of depot → out 100 + back 100 = 200 m."""
    d, order = farm_tour([(100.0, 0.0)], depot=(0.0, 0.0))
    assert math.isclose(d, 200.0, rel_tol=1e-9)
    assert order == [0]


def test_farm_tour_visits_each_field_exactly_once() -> None:
    fields_xy = [(50.0, 0.0), (60.0, 80.0), (200.0, 30.0), (10.0, 200.0)]
    d, order = farm_tour(fields_xy, depot=(0.0, 0.0))
    assert sorted(order) == list(range(len(fields_xy)))
    assert d > 0.0


def test_farm_tour_nearest_neighbour_picks_closest_first() -> None:
    """From depot (0,0), nearest is the (10,0) point, not (1000,0)."""
    _, order = farm_tour([(1000.0, 0.0), (10.0, 0.0)], depot=(0.0, 0.0))
    assert order[0] == 1  # the (10,0) field is index 1


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_strip_decomposition_unit_square_one_strip,
    test_strip_decomposition_unit_square_two_strips_at_half_span,
    test_strip_decomposition_l_shape_two_strips_with_unequal_area,
    test_strip_decomposition_total_area_equals_polygon_area,
    test_strip_decomposition_invalid_inputs_raise,
    test_shape_efficiency_unit_square_equals_one,
    test_shape_efficiency_l_shape_below_one,
    test_shape_efficiency_hollow_square_below_one,
    test_shape_efficiency_in_unit_interval,
    test_shape_efficiency_zero_for_empty_polygon,
    test_best_orientation_at_least_as_good_as_x_axis,
    test_strip_plan_segments_count_matches_decomposition,
    test_corpus_has_50_fields,
    test_corpus_classes_complete,
    test_corpus_polygons_are_valid,
    test_corpus_summary_returns_one_row_per_field,
    test_farm_tour_empty_returns_zero,
    test_farm_tour_single_field_round_trip,
    test_farm_tour_visits_each_field_exactly_once,
    test_farm_tour_nearest_neighbour_picks_closest_first,
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
    print(f"\nok: cabletract.layout passes {len(ALL_TESTS)} invariants")
