"""Verification tests for cabletract.physics (Phase 1).

Each test checks one of the physical invariants the module is supposed to
respect. Together they pin down the catenary, anchor, motor and regen
sub-models against simple analytic limits, so a future refactor cannot
silently break the manuscript figures F1-F3.

Run as:

    python tests/test_physics.py
    pytest tests/test_physics.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.physics import (  # noqa: E402
    G,
    DrivetrainEfficiency,
    anchor_reaction_envelope,
    cable_elastic_stretch,
    catenary_sag,
    default_drivetrain,
    parabolic_sag,
    peak_motor_power,
    regen_energy,
    tension_balance,
)


# ---------------------------------------------------------------------------
# Catenary
# ---------------------------------------------------------------------------

def test_catenary_sag_monotone_in_span() -> None:
    """For fixed tension and weight, sag must grow with span."""
    T = 5000.0
    w = 0.5
    sags = [catenary_sag(T, w, L)[0] for L in (20.0, 40.0, 60.0, 80.0, 100.0)]
    assert all(sags[i] < sags[i + 1] for i in range(len(sags) - 1)), sags


def test_catenary_sag_decreases_with_tension() -> None:
    """For fixed span and weight, sag must decrease as tension grows."""
    sags = [catenary_sag(T, 0.5, 50.0)[0] for T in (500.0, 1000.0, 5000.0, 50000.0)]
    assert all(sags[i] > sags[i + 1] for i in range(len(sags) - 1)), sags


def test_catenary_sag_to_zero_as_tension_to_infty() -> None:
    """In the high-tension limit, sag approaches zero."""
    sag, _arc, _T = catenary_sag(1.0e9, 0.5, 50.0)
    assert sag < 1.0e-6, sag


def test_catenary_matches_parabolic_in_small_sag_limit() -> None:
    """The exact catenary and the parabolic approximation must agree to <1%
    when w*L/(2T) is well below 0.3."""
    T, w, L = 50_000.0, 0.5, 50.0
    sag_exact, _arc, _T_end = catenary_sag(T, w, L)
    sag_para = parabolic_sag(T, w, L)
    rel_err = abs(sag_exact - sag_para) / sag_para
    assert rel_err < 0.01, (sag_exact, sag_para, rel_err)


def test_catenary_arc_length_strictly_above_span() -> None:
    """Any sagging cable is strictly longer than the chord."""
    _sag, arc, _T_end = catenary_sag(5000.0, 0.5, 50.0)
    assert arc > 50.0


def test_catenary_end_tension_at_least_horizontal() -> None:
    """T_end = T_h cosh(L/2a) >= T_h."""
    _sag, _arc, T_end = catenary_sag(5000.0, 0.5, 50.0)
    assert T_end >= 5000.0


# ---------------------------------------------------------------------------
# Cable elastic stretch
# ---------------------------------------------------------------------------

def test_elastic_stretch_linear_in_force() -> None:
    """Hooke's law: dL is linear in F at fixed E, A, L."""
    s1 = cable_elastic_stretch(1000.0, 1.10e11, 5.0e-5, 50.0)
    s2 = cable_elastic_stretch(2000.0, 1.10e11, 5.0e-5, 50.0)
    assert math.isclose(s2, 2.0 * s1, rel_tol=1e-12)


def test_elastic_stretch_steel_50m_at_5kn_in_realistic_range() -> None:
    """Sanity: a 50 m wire-rope at 5 kN with E=110 GPa, A=50 mm^2 stretches
    about F·L/(E·A) ≈ 45 mm. Allow a 10–100 mm window."""
    dL = cable_elastic_stretch(5000.0, 1.10e11, 5.0e-5, 50.0)
    assert 0.01 < dL < 0.10, dL


# ---------------------------------------------------------------------------
# Tension balance (working geometry)
# ---------------------------------------------------------------------------

def test_tension_balance_draft_bound_at_2700N_dyneema() -> None:
    """At 2700 N draft on a Dyneema 8 mm cable spanning 50 m, the operation
    is draft-bound: the cable's own weight cannot pull it down to the soil
    against the working tension, so T_anchor ≈ F_draft."""
    state = tension_balance(
        F_draft=2700.0,
        F_implement_weight=200.0,
        w_cable=0.046 * 9.80665,
        span=50.0,
        pulley_height=1.5,
    )
    assert state.regime == "draft-bound"
    assert math.isclose(state.T_horiz_anchor, 2700.0, rel_tol=1e-9)
    assert state.T_winch >= state.T_anchor


def test_tension_balance_sag_bound_for_low_draft_steel() -> None:
    """At very low draft on a heavy steel cable, sag would exceed the
    clearance budget so the winch must over-tension. We must end up in the
    sag-bound regime with T_anchor > F_draft."""
    state = tension_balance(
        F_draft=50.0,
        F_implement_weight=200.0,
        w_cable=0.255 * 9.80665,
        span=80.0,
        pulley_height=1.0,
        min_ground_clearance=0.10,
    )
    assert state.regime == "sag-bound"
    assert state.T_horiz_anchor > 50.0
    # Winch must additionally carry the draft force on top of the sag tension
    assert state.T_horiz > state.T_horiz_anchor


def test_tension_balance_winch_geq_anchor() -> None:
    """The winch always sees at least as much load as the anchor."""
    state = tension_balance(
        F_draft=2700.0,
        F_implement_weight=200.0,
        w_cable=0.255 * 9.80665,
        span=50.0,
        pulley_height=1.5,
    )
    assert state.T_winch >= state.T_anchor


def test_tension_balance_higher_draft_needs_more_tension() -> None:
    """Heavier draft must monotonically push the required winch tension up."""
    common = dict(
        F_implement_weight=200.0,
        w_cable=0.046 * 9.80665,  # Dyneema
        span=50.0,
        pulley_height=1.5,
    )
    Ts = [tension_balance(F_draft=fd, **common).T_winch for fd in (500, 1500, 2700, 4000)]
    assert all(Ts[i] < Ts[i + 1] for i in range(len(Ts) - 1)), Ts


def test_tension_balance_clearance_respected_in_sag_bound() -> None:
    """In the sag-bound regime the resulting ground clearance must equal
    the prescribed minimum (within tolerance)."""
    state = tension_balance(
        F_draft=20.0,
        F_implement_weight=200.0,
        w_cable=0.255 * 9.80665,
        span=80.0,
        pulley_height=1.0,
        min_ground_clearance=0.10,
    )
    assert state.regime == "sag-bound"
    assert abs(state.ground_clearance - 0.10) < 0.01


# ---------------------------------------------------------------------------
# Anchor envelope
# ---------------------------------------------------------------------------

def test_anchor_envelope_required_count_grows_with_load() -> None:
    """Doubling the cable tension at least doubles the working auger count."""
    e1 = anchor_reaction_envelope(T_anchor=1000.0, n_augers=4)
    e2 = anchor_reaction_envelope(T_anchor=2000.0, n_augers=4)
    assert e2.n_augers_required_working >= 2 * e1.n_augers_required_working - 1


def test_anchor_envelope_safety_factor_inversely_proportional() -> None:
    """The safety factor falls as 1/T_anchor for fixed n_augers."""
    e1 = anchor_reaction_envelope(T_anchor=400.0, n_augers=4)
    e2 = anchor_reaction_envelope(T_anchor=800.0, n_augers=4)
    assert math.isclose(e1.safety_factor_working, 2.0 * e2.safety_factor_working, rel_tol=1e-9)


def test_anchor_envelope_khand_2024_single_auger_holds_400N() -> None:
    """Khand 2024 4-pile raft → ~400 N per pile in the conservative loose-sand
    interpretation: that should give a working safety factor of exactly 1.0
    at 400 N load on a single auger."""
    env = anchor_reaction_envelope(T_anchor=400.0, n_augers=1)
    assert math.isclose(env.safety_factor_working, 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Drivetrain decomposition
# ---------------------------------------------------------------------------

def test_drivetrain_decomposition_recovers_v1_lump() -> None:
    """The product of the per-component default efficiencies must be ~0.5,
    matching the v1 lumped winch_efficiency."""
    eta = default_drivetrain().total
    assert 0.45 < eta < 0.55, eta


def test_motor_power_peak_geq_continuous() -> None:
    env = peak_motor_power(
        F_continuous=3000.0, v_continuous=0.5, F_peak=8000.0, v_peak=0.3
    )
    assert env.peak_W >= env.continuous_W
    assert env.peak_to_continuous > 1.0


def test_motor_power_uses_decomposed_chain() -> None:
    """Custom drivetrain must be honoured (output power scales 1/eta)."""
    custom = DrivetrainEfficiency(
        motor=0.99,
        controller=0.99,
        gearbox=0.99,
        drum=0.99,
        pulley=0.99,
        cable_elongation=0.99,
    )
    env_default = peak_motor_power(3000.0, 0.5, 5000.0, 0.4)
    env_high = peak_motor_power(3000.0, 0.5, 5000.0, 0.4, drivetrain=custom)
    assert env_high.continuous_W < env_default.continuous_W


# ---------------------------------------------------------------------------
# Regen
# ---------------------------------------------------------------------------

def test_regen_zero_on_flat_with_rolling_loss() -> None:
    """On flat ground rolling resistance is non-zero, so no recoverable energy."""
    assert regen_energy(slope_rad=0.0, mass_kg=200.0, distance_m=50.0) == 0.0


def test_regen_bounded_by_potential_energy() -> None:
    """Recovered energy must not exceed eta_regen * m g h, even before subtracting
    rolling losses."""
    slope = math.radians(8.0)
    distance = 50.0
    mass = 200.0
    eta = 0.55
    h = distance * math.sin(slope)
    pe_Wh = mass * G * h / 3600.0
    e_regen = regen_energy(slope_rad=slope, mass_kg=mass, distance_m=distance, eta_regen=eta)
    assert e_regen <= eta * pe_Wh + 1e-9, (e_regen, eta * pe_Wh)


def test_regen_grows_with_slope() -> None:
    e_low = regen_energy(slope_rad=math.radians(4.0), mass_kg=200.0, distance_m=50.0)
    e_high = regen_energy(slope_rad=math.radians(10.0), mass_kg=200.0, distance_m=50.0)
    assert e_high > e_low


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_catenary_sag_monotone_in_span,
    test_catenary_sag_decreases_with_tension,
    test_catenary_sag_to_zero_as_tension_to_infty,
    test_catenary_matches_parabolic_in_small_sag_limit,
    test_catenary_arc_length_strictly_above_span,
    test_catenary_end_tension_at_least_horizontal,
    test_elastic_stretch_linear_in_force,
    test_elastic_stretch_steel_50m_at_5kn_in_realistic_range,
    test_tension_balance_draft_bound_at_2700N_dyneema,
    test_tension_balance_sag_bound_for_low_draft_steel,
    test_tension_balance_winch_geq_anchor,
    test_tension_balance_higher_draft_needs_more_tension,
    test_tension_balance_clearance_respected_in_sag_bound,
    test_anchor_envelope_required_count_grows_with_load,
    test_anchor_envelope_safety_factor_inversely_proportional,
    test_anchor_envelope_khand_2024_single_auger_holds_400N,
    test_drivetrain_decomposition_recovers_v1_lump,
    test_motor_power_peak_geq_continuous,
    test_motor_power_uses_decomposed_chain,
    test_regen_zero_on_flat_with_rolling_loss,
    test_regen_bounded_by_potential_energy,
    test_regen_grows_with_slope,
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
    print(f"\nok: cabletract.physics passes {len(ALL_TESTS)} invariants")
