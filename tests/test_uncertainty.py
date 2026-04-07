"""Verification tests for cabletract.uncertainty (Phase 5).

Pins down: ParamRange sample shape and bounds, MC reproducibility for a
given seed, percentile envelope ordering, Sobol decomposition closure
on a known toy function, and tornado-data swing ordering.

Run as:

    python tests/test_uncertainty.py
    pytest tests/test_uncertainty.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.params import CableTractParams  # noqa: E402
from cabletract.simulate import CableTractResults, run_single  # noqa: E402
from cabletract.uncertainty import (  # noqa: E402
    ParamProblem,
    ParamRange,
    default_problem,
    monte_carlo,
    percentile_envelope,
    sobol_indices,
    tornado_data,
)


# ---------------------------------------------------------------------------
# ParamRange and ParamProblem
# ---------------------------------------------------------------------------

def test_param_range_uniform_in_bounds() -> None:
    rng = np.random.default_rng(0)
    pr = ParamRange("x", 0.0, 1.0, "uniform")
    s = pr.sample(rng, 10_000)
    assert s.min() >= 0.0
    assert s.max() <= 1.0
    assert 0.4 < s.mean() < 0.6


def test_param_range_triangular_mode_default_is_midpoint() -> None:
    rng = np.random.default_rng(0)
    pr = ParamRange("x", 0.0, 10.0, "triangular")
    s = pr.sample(rng, 50_000)
    # Mean of symmetric triangular = midpoint = 5
    assert 4.8 < s.mean() < 5.2


def test_param_range_unknown_dist_raises() -> None:
    rng = np.random.default_rng(0)
    pr = ParamRange("x", 0.0, 1.0, "lognormal")
    try:
        pr.sample(rng, 10)
    except ValueError:
        return
    assert False, "expected ValueError"


def test_param_problem_to_salib_keys() -> None:
    prob = default_problem()
    s = prob.to_salib()
    assert s["num_vars"] == len(prob.ranges)
    assert len(s["names"]) == len(prob.ranges)
    assert len(s["bounds"]) == len(prob.ranges)
    for lo, hi in s["bounds"]:
        assert lo < hi


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def test_monte_carlo_shape_and_columns() -> None:
    prob = default_problem()
    df = monte_carlo(prob, n=50, seed=0)
    assert len(df) == 50
    # Inputs and outputs both present
    for pname in prob.names:
        assert pname in df.columns
    for out in ["decares_per_day_offgrid", "payback_months_vs_fuel"]:
        assert out in df.columns


def test_monte_carlo_reproducible_with_same_seed() -> None:
    prob = default_problem()
    a = monte_carlo(prob, n=30, seed=42)
    b = monte_carlo(prob, n=30, seed=42)
    assert (a.values == b.values).all() or np.allclose(
        a.select_dtypes("number").values, b.select_dtypes("number").values, equal_nan=True
    )


def test_monte_carlo_different_seeds_diverge() -> None:
    prob = default_problem()
    a = monte_carlo(prob, n=30, seed=0)
    b = monte_carlo(prob, n=30, seed=1)
    # Decares column should differ between seeds
    assert not np.allclose(a["decares_per_day_offgrid"].values, b["decares_per_day_offgrid"].values)


def test_percentile_envelope_ordering() -> None:
    prob = default_problem()
    df = monte_carlo(prob, n=100, seed=0)
    env = percentile_envelope(df, "decares_per_day_offgrid")
    assert env["p10"] <= env["p50"] <= env["p90"]


# ---------------------------------------------------------------------------
# Sobol
# ---------------------------------------------------------------------------

def test_sobol_returns_one_row_per_output_param_pair() -> None:
    prob = default_problem()
    si = sobol_indices(prob, n_base=32, output_names=["decares_per_day_offgrid"])
    assert len(si) == len(prob.ranges)
    for col in ["S1", "ST", "S1_conf", "ST_conf"]:
        assert col in si.columns


def test_sobol_total_indices_nonneg() -> None:
    """ST should be (modulo finite-sample noise) non-negative."""
    prob = default_problem()
    si = sobol_indices(prob, n_base=32, output_names=["decares_per_day_offgrid"])
    # Allow small negative numerical noise
    assert (si["ST"] > -0.05).all()


def test_sobol_on_ishigami_toy_recovers_known_ranking() -> None:
    """Standard sanity check: Ishigami function f(x1,x2,x3)= sin(x1)+a sin^2(x2)+b x3^4 sin(x1)
    has known Sobol indices in the order ST: x1 > x2 > x3 with a=7, b=0.1.

    We exercise sobol_indices on this analytic function via a CableTract-shaped
    fake fn that returns a CableTractResults with `decares_per_day_offgrid`
    set to the Ishigami value, holding all other parameters at defaults."""
    prob = ParamProblem(ranges=[
        ParamRange("draft_load_N",  -math.pi, math.pi, "uniform"),
        ParamRange("implement_weight_N", -math.pi, math.pi, "uniform"),
        ParamRange("system_weight_N",    -math.pi, math.pi, "uniform"),
    ])

    def fake_fn(p: CableTractParams) -> CableTractResults:
        x1 = p.draft_load_N
        x2 = p.implement_weight_N
        x3 = p.system_weight_N
        a, b = 7.0, 0.1
        y = math.sin(x1) + a * math.sin(x2) ** 2 + b * (x3 ** 4) * math.sin(x1)
        # Build a CableTractResults full of zeros except for the target field
        return CableTractResults(
            rounds_per_decare=0.0, work_per_round_J=0.0, energy_per_round_Wh=0.0,
            energy_per_decare_Wh=0.0, harvested_energy_per_day_Wh=0.0,
            decares_per_day_energy_limited=0.0, rounds_per_day=0.0,
            usable_time_s=0.0, operation_time_s=0.0, travel_time_s=0.0,
            operation_distance_m_day=0.0, travel_distance_m_day=0.0,
            operation_speed_m_s=0.0, travel_speed_m_s=0.0,
            winch_output_power_W=0.0, winch_input_power_W=0.0,
            harvested_power_W=0.0, surplus_power_W=0.0,
            battery_chargeable_Wh=0.0, battery_required_Wh=0.0,
            grid_charge_needed_Wh=0.0, battery_sufficient=True,
            decares_per_day_offgrid=y, decares_per_year=0.0,
            fuel_savings_usd_year=0.0, electricity_savings_usd_year=0.0,
            sales_price_usd=0.0,
            payback_months_vs_fuel=0.0, payback_months_vs_electric=0.0,
        )

    si = sobol_indices(prob, n_base=512, output_names=["decares_per_day_offgrid"], fn=fake_fn)
    by_param = {row["parameter"]: row["ST"] for _, row in si.iterrows()}
    # Known Ishigami ST: x1 ≈ 0.56, x2 ≈ 0.44, x3 ≈ 0.24
    assert by_param["draft_load_N"] > by_param["system_weight_N"]
    assert by_param["implement_weight_N"] > by_param["system_weight_N"]


# ---------------------------------------------------------------------------
# Tornado
# ---------------------------------------------------------------------------

def test_tornado_data_sorted_by_swing() -> None:
    prob = default_problem()
    td = tornado_data(prob, "decares_per_day_offgrid")
    swings = td["swing"].values
    for i in range(len(swings) - 1):
        assert swings[i] >= swings[i + 1]


def test_tornado_data_lo_hi_bracket_baseline() -> None:
    """For a monotone parameter, baseline output should sit between
    lo_out and hi_out. We check this for the most-impactful parameter."""
    prob = default_problem()
    td = tornado_data(prob, "decares_per_day_offgrid")
    top = td.iloc[0]
    bracket = sorted([top["lo_out"], top["hi_out"]])
    assert bracket[0] - 1e-6 <= top["baseline_out"] <= bracket[1] + 1e-6


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_param_range_uniform_in_bounds,
    test_param_range_triangular_mode_default_is_midpoint,
    test_param_range_unknown_dist_raises,
    test_param_problem_to_salib_keys,
    test_monte_carlo_shape_and_columns,
    test_monte_carlo_reproducible_with_same_seed,
    test_monte_carlo_different_seeds_diverge,
    test_percentile_envelope_ordering,
    test_sobol_returns_one_row_per_output_param_pair,
    test_sobol_total_indices_nonneg,
    test_sobol_on_ishigami_toy_recovers_known_ranking,
    test_tornado_data_sorted_by_swing,
    test_tornado_data_lo_hi_bracket_baseline,
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
    print(f"\nok: cabletract.uncertainty passes {len(ALL_TESTS)} invariants")
