"""Validation for S7 -- nonlinear p--y helical-auger lateral capacity.

Checks (per SIM_IMPLEMENTATION_PLAN.md):
  1. The FD beam-column solver reproduces the closed-form semi-infinite
     beam-on-elastic-foundation head deflection y0 = H/(2 beta^3 EI) in the
     constant-modulus linear limit.
  2. API ultimate-resistance coefficients match the published API curve at a
     reference angle.
  3. The derived per-auger nominal brackets the cited Khand/Magnum range and
     collapses the dual reference to a single auger count.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.anchor_py import (
    LOOSE_SAND, MEDIUM_DENSE_SAND, PileSection, api_sand_coefficients,
    group_capacity, lateral_capacity, required_augers_for, solve_pile,
    winkler_head_deflection_closed_form,
)


def test_solver_matches_winkler_closed_form() -> None:
    """Constant-modulus, long pile -> Hetenyi semi-infinite-beam closed form."""
    sec = PileSection(length_m=12.0, n_nodes=601)  # L >> 1/beta
    kf = 4.0e6  # N/m^2 constant foundation modulus
    H = 2000.0
    sol = solve_pile(sec, H, k_const_Npm2=kf, head_fixity="free")
    closed = winkler_head_deflection_closed_form(H, sec.EI, kf)
    assert sol.converged
    assert abs(sol.head_deflection_m - closed) / closed < 0.01  # <1%


def test_solver_is_linear_in_load_for_constant_modulus() -> None:
    sec = PileSection(length_m=12.0, n_nodes=601)
    kf = 4.0e6
    d1 = solve_pile(sec, 1000.0, k_const_Npm2=kf).head_deflection_m
    d2 = solve_pile(sec, 2000.0, k_const_Npm2=kf).head_deflection_m
    assert abs(d2 / d1 - 2.0) < 1e-3  # doubling load doubles deflection


def test_api_coefficients_match_published_curve() -> None:
    # API/Reese chart at phi=30 deg: C2 ~ 2.7, C3 ~ 29 (within a few %).
    c1, c2, c3 = api_sand_coefficients(30.0)
    assert abs(c2 - 2.67) < 0.2
    assert abs(c3 - 29.0) < 2.0
    # coefficients increase with friction angle
    c1b, c2b, c3b = api_sand_coefficients(36.0)
    assert c1b > c1 and c2b > c2 and c3b > c3


def test_no_overflow_in_capacity_search() -> None:
    """The clipped tanh tangent must not raise overflow during a full search."""
    sec = PileSection()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cap = lateral_capacity(sec, MEDIUM_DENSE_SAND, head_fixity="fixed")
    assert np.isfinite(cap) and cap > 0


def test_capacities_ordered_physically() -> None:
    sec = PileSection()
    loose_free = lateral_capacity(sec, LOOSE_SAND, head_fixity="free")
    loose_fixed = lateral_capacity(sec, LOOSE_SAND, head_fixity="fixed")
    md_free = lateral_capacity(sec, MEDIUM_DENSE_SAND, head_fixity="free")
    # fixed head is stiffer -> higher capacity; denser soil -> higher capacity
    assert loose_fixed > loose_free
    assert md_free > loose_free
    # nonlinear capacity is well below the naive elastic value (~13 kN) for loose
    assert 2.0e3 < loose_free < 9.0e3


def test_derived_nominal_brackets_literature_and_sets_auger_count() -> None:
    sec = PileSection()
    g_loose = group_capacity(sec, LOOSE_SAND, spacing_over_d=3.0, safety_factor=1.5)
    g_md = group_capacity(sec, MEDIUM_DENSE_SAND, spacing_over_d=3.0, safety_factor=1.5)
    # 3x3 mean p-multiplier (0.8,0.4,0.3) = 0.5
    assert abs(g_loose.group_efficiency - 0.5) < 1e-9
    # derived nominal band brackets the Magnum 2 kN datasheet value and lies
    # above Khand's conservative 400 N floor
    lo, hi = g_loose.nominal_working_per_auger_N, g_md.nominal_working_per_auger_N
    assert 400.0 < lo < 2000.0 < hi
    # single-pile fixed-head capacities reach the Magnum fixed-head range (kN)
    assert g_md.single_fixed_N > 10.0e3
    # collapses the dual reference: at codesigned P90 3 kN the auger count is a
    # single small number (between the old 9 @400 N and 2 @2 kN extremes)
    n = required_augers_for(3000.0, lo)
    assert 2 <= n <= 9
