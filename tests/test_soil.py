"""Verification tests for cabletract.soil (Phase 2).

Each test pins down one invariant of the ASABE D497.7 implementation against
either the standard's textbook reference values or a physical sanity bound.

Run as:

    python tests/test_soil.py
    pytest tests/test_soil.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cabletract.soil import (  # noqa: E402
    MOISTURE_OPTIMUM,
    Implement,
    draft_distribution,
    implement_by_name,
    library_draft_summary,
    load_implement_library,
    moisture_factor,
    percentiles,
    texture_factor,
)


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def test_library_has_twelve_implements() -> None:
    lib = load_implement_library()
    assert len(lib) == 12, f"expected 12 implements, got {len(lib)}"


def test_library_categories_complete() -> None:
    """Every D497 category planned for Phase 2 must be represented."""
    lib = load_implement_library()
    cats = {imp.category for imp in lib}
    expected = {
        "primary_tillage",
        "secondary_tillage",
        "seeding",
        "spraying",
        "mowing",
        "weeding",
    }
    assert expected.issubset(cats), expected - cats


# ---------------------------------------------------------------------------
# Texture and moisture factors
# ---------------------------------------------------------------------------

def test_texture_factor_fine_is_unity_for_all_categories() -> None:
    for cat in (
        "primary_tillage",
        "secondary_tillage",
        "seeding",
        "spraying",
        "mowing",
        "weeding",
    ):
        assert texture_factor("fine", cat) == 1.0


def test_texture_factor_strictly_decreasing_with_coarsness_for_tillage() -> None:
    """Coarser soils have lower draft for tillage (sand crumbles)."""
    for cat in ("primary_tillage", "secondary_tillage"):
        assert (
            texture_factor("fine", cat)
            > texture_factor("medium", cat)
            > texture_factor("coarse", cat)
        )


def test_moisture_factor_minimum_at_optimum() -> None:
    f_opt = moisture_factor(MOISTURE_OPTIMUM)
    f_dry = moisture_factor(MOISTURE_OPTIMUM - 0.10)
    f_wet = moisture_factor(MOISTURE_OPTIMUM + 0.10)
    assert math.isclose(f_opt, 1.0, rel_tol=1e-12)
    assert f_dry > f_opt
    assert f_wet > f_opt


def test_moisture_factor_symmetric_around_optimum() -> None:
    """The U-shape is symmetric: ±delta gives the same penalty."""
    for delta in (0.02, 0.05, 0.10):
        a = moisture_factor(MOISTURE_OPTIMUM - delta)
        b = moisture_factor(MOISTURE_OPTIMUM + delta)
        assert math.isclose(a, b, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Implement.draft_N — analytic ASABE values
# ---------------------------------------------------------------------------

def test_moldboard_plow_recovers_asabe_textbook_value() -> None:
    """ASABE D497.7 example: moldboard plow at S=8 km/h, T=15 cm, fine soil,
    optimum moisture. The ASABE published value for a 1-m share is

        D = 1.00 * (652 + 5.1 * 64) * 1.0 * 15 ≈ 14,675 N

    Our implementation must reproduce this within 1% (no Monte Carlo)."""
    imp = Implement(
        name="moldboard_test",
        category="primary_tillage",
        A=652.0,
        B=0.0,
        C=5.1,
        width_basis="meter",
        width_count=1.0,
        depth_dependent=True,
        typical_depth_cm=15.0,
        A_units_note="",
        source="",
    )
    D = imp.draft_N(speed_km_h=8.0, depth_cm=15.0, soil_texture="fine", moisture=MOISTURE_OPTIMUM)
    expected = (652.0 + 5.1 * 64.0) * 1.0 * 15.0
    assert math.isclose(D, expected, rel_tol=0.01), (D, expected)


def test_chisel_plow_speed_linearity() -> None:
    """Chisel plow draft is linear in speed (ASABE D497 — B>0, C=0)."""
    imp = implement_by_name("chisel_plow_twisted")
    D5 = imp.base_draft_N(speed_km_h=5.0, depth_cm=15.0)
    D10 = imp.base_draft_N(speed_km_h=10.0, depth_cm=15.0)
    D15 = imp.base_draft_N(speed_km_h=15.0, depth_cm=15.0)
    # Equal increments must give equal draft increases (linear)
    assert math.isclose(D10 - D5, D15 - D10, rel_tol=1e-12)


def test_planter_speed_independent() -> None:
    """A row-crop planter has B=C=0 → draft independent of speed."""
    imp = implement_by_name("row_planter")
    D2 = imp.base_draft_N(speed_km_h=2.0, depth_cm=1.0)
    D8 = imp.base_draft_N(speed_km_h=8.0, depth_cm=1.0)
    assert math.isclose(D2, D8, rel_tol=1e-12)


def test_moldboard_speed_quadratic() -> None:
    """A moldboard plow has C>0 → draft grows faster than linearly with speed."""
    imp = implement_by_name("moldboard_plow")
    D4 = imp.base_draft_N(speed_km_h=4.0, depth_cm=15.0)
    D8 = imp.base_draft_N(speed_km_h=8.0, depth_cm=15.0)
    D12 = imp.base_draft_N(speed_km_h=12.0, depth_cm=15.0)
    # Increment from 8→12 must exceed increment from 4→8 (convex)
    assert (D12 - D8) > (D8 - D4)


# ---------------------------------------------------------------------------
# draft_distribution sampler
# ---------------------------------------------------------------------------

def test_draft_distribution_shape_and_sign() -> None:
    imp = implement_by_name("moldboard_plow")
    samples = draft_distribution(
        imp,
        soil_texture="medium",
        moisture_range=(0.15, 0.25),
        speed_range_km_h=(1.0, 4.0),
        depth_range_cm=(10.0, 20.0),
        n_samples=200,
    )
    assert samples.shape == (200,)
    assert np.all(samples > 0.0)


def test_draft_distribution_p10_lt_p50_lt_p90() -> None:
    imp = implement_by_name("disk_harrow_tandem")
    samples = draft_distribution(
        imp,
        soil_texture="medium",
        moisture_range=(0.10, 0.30),
        speed_range_km_h=(1.0, 6.0),
        depth_range_cm=(5.0, 15.0),
        n_samples=2000,
    )
    ps = percentiles(samples)
    assert ps["P10"] < ps["P50"] < ps["P90"]


def test_draft_distribution_reproducible_with_seed() -> None:
    imp = implement_by_name("chisel_plow_twisted")
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    a = draft_distribution(
        imp, "fine", (0.18, 0.22), (1.0, 3.0), (15.0, 25.0), n_samples=500, rng=rng1
    )
    b = draft_distribution(
        imp, "fine", (0.18, 0.22), (1.0, 3.0), (15.0, 25.0), n_samples=500, rng=rng2
    )
    assert np.array_equal(a, b)


def test_draft_distribution_higher_speed_increases_mean_for_chisel() -> None:
    imp = implement_by_name("chisel_plow_twisted")
    common = dict(
        soil_texture="medium",
        moisture_range=(0.18, 0.22),
        depth_range_cm=(15.0, 20.0),
        n_samples=2000,
    )
    slow = draft_distribution(imp, speed_range_km_h=(1.0, 2.0), **common)
    fast = draft_distribution(imp, speed_range_km_h=(8.0, 10.0), **common)
    assert np.mean(fast) > np.mean(slow)


def test_draft_distribution_depth_independent_for_planter() -> None:
    """A planter's draft must not change with depth_range_cm (B=C=0, depth_dependent=False)."""
    imp = implement_by_name("row_planter")
    common = dict(
        soil_texture="medium",
        moisture_range=(0.20, 0.20),
        speed_range_km_h=(2.0, 2.0),
        n_samples=200,
    )
    a = draft_distribution(imp, depth_range_cm=(1.0, 1.0), **common)
    b = draft_distribution(imp, depth_range_cm=(10.0, 20.0), **common)
    # Both arrays should be identical because depth is ignored.
    assert math.isclose(float(np.mean(a)), float(np.mean(b)), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# library_draft_summary headline check
# ---------------------------------------------------------------------------

def test_library_summary_returns_one_row_per_implement() -> None:
    df = library_draft_summary()
    lib = load_implement_library()
    assert len(df) == len(lib)


def test_library_summary_moldboard_p50_in_kn_range() -> None:
    """At CableTract's slow operating speeds (1-4 km/h) and a 40 cm share,
    a moldboard plow should sit somewhere between 1 and 6 kN P50. This is
    the band the v1 'Heavy tillage' 3 kN scenario claims."""
    df = library_draft_summary()
    p50 = df.loc[df["implement"] == "moldboard_plow", "P50_N"].iloc[0]
    assert 1000.0 < p50 < 6000.0, p50


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_library_has_twelve_implements,
    test_library_categories_complete,
    test_texture_factor_fine_is_unity_for_all_categories,
    test_texture_factor_strictly_decreasing_with_coarsness_for_tillage,
    test_moisture_factor_minimum_at_optimum,
    test_moisture_factor_symmetric_around_optimum,
    test_moldboard_plow_recovers_asabe_textbook_value,
    test_chisel_plow_speed_linearity,
    test_planter_speed_independent,
    test_moldboard_speed_quadratic,
    test_draft_distribution_shape_and_sign,
    test_draft_distribution_p10_lt_p50_lt_p90,
    test_draft_distribution_reproducible_with_seed,
    test_draft_distribution_higher_speed_increases_mean_for_chisel,
    test_draft_distribution_depth_independent_for_planter,
    test_library_summary_returns_one_row_per_implement,
    test_library_summary_moldboard_p50_in_kn_range,
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
    print(f"\nok: cabletract.soil passes {len(ALL_TESTS)} invariants")
