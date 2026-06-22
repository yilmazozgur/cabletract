"""Validation for S1 -- depth-control co-simulation and the lumped cable.

Cable checks (per SIM_IMPLEMENTATION_PLAN.md):
  1. static equilibrium reproduces physics.catenary_sag;
  2. first transverse mode matches the taut-string f1 = (1/2L) sqrt(T/mu);
  3. an undamped free run conserves total energy.
Control checks:
  4. the cable vertical point stiffness is orders of magnitude too soft to set
     depth (so the gauge wheel does, as the manuscript claims);
  5. the regulator holds the setpoint with no disturbance;
  6. the heavy tractor rejects a stone strike far better than the light
     carriage and never lifts off, while the carriage stays within the +/-2 cm
     agronomic tolerance;
  7. the offset-free MPC rejects a sustained (hardpan) disturbance;
  8. raising the down-pressure reserve removes gauge-wheel lift-off.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.physics import catenary_sag
from cabletract.dynamics import (
    DepthPlant, LumpedCable, MPC, hardpan_step, load_cable_props,
    moisture_ramp, simulate_depth, stability_envelope, stone_impulse,
    tuned_pid,
)


def _cable(T=1800.0):
    props = load_cable_props()
    c = LumpedCable(span=50.0, props=props, T_target=T, n_seg=40)
    c.static_equilibrium()
    return c, props


def test_static_sag_matches_catenary() -> None:
    c, props = _cable(1800.0)
    sag_analytic, _, _ = catenary_sag(1800.0, props.w_per_m_N, 50.0)
    assert abs(c.midspan_sag() - sag_analytic) / sag_analytic < 0.02


def test_transverse_first_mode_matches_taut_string() -> None:
    c, _ = _cable(1800.0)
    f1 = c.transverse_frequencies(1)[0]
    assert abs(f1 - c.taut_string_f1()) / c.taut_string_f1() < 0.05


def test_undamped_energy_is_conserved() -> None:
    c, _ = _cable(1800.0)
    yp = 0.02 * np.sin(np.pi * np.arange(c.N) / (c.N - 1))
    E = c.integrate_free(yp, t_end=2.0, dt=2.0e-4)
    drift = (E.max() - E.min()) / abs(np.mean(E))
    assert drift < 0.01


def test_cable_too_soft_to_set_depth() -> None:
    c, _ = _cable(1800.0)
    k_cable = c.vertical_point_stiffness()
    plant = DepthPlant()
    # cable vertical stiffness ~150 N/m is >1000x softer than the gauge wheel
    assert k_cable < 300.0
    assert plant.k_w / k_cable > 1.0e3


def test_regulator_holds_setpoint_without_disturbance() -> None:
    plant = DepthPlant()
    res = simulate_depth(plant, tuned_pid(plant), lambda t: 0.0, t_end=2.0)
    assert res.peak_e_mm < 0.5  # stays essentially on setpoint


def test_tractor_beats_carriage_and_does_not_lift_off() -> None:
    carriage, tractor = DepthPlant(), DepthPlant.tractor()
    rc = simulate_depth(carriage, tuned_pid(carriage), stone_impulse, t_end=4.0)
    rt = simulate_depth(tractor, tuned_pid(tractor), stone_impulse, t_end=4.0)
    assert rt.peak_e_mm < rc.peak_e_mm           # inertia + reserve wins
    assert rt.liftoff_s == 0.0 and rt.min_N > 0  # tractor never unloads
    assert rc.peak_e_mm < 20.0                   # carriage still within 2 cm


def test_carriage_within_tolerance_all_disturbances() -> None:
    carriage = DepthPlant()
    for dist in (stone_impulse, hardpan_step, moisture_ramp):
        res = simulate_depth(carriage, tuned_pid(carriage), dist, t_end=4.0)
        assert res.in_tol  # peak <= 2 cm


def test_mpc_is_offset_free_on_sustained_disturbance() -> None:
    carriage = DepthPlant()
    mpc = MPC(carriage, dt=0.01)
    res = simulate_depth(carriage, mpc, hardpan_step, t_end=6.0)
    # offset-free: depth error has decayed back into a tight band by the end
    assert abs(res.e[-1]) < 1.0e-3


def test_downpressure_reserve_removes_liftoff() -> None:
    base = DepthPlant()
    N0_grid = np.array([500.0, 4000.0])
    bw_grid = np.array([5.0])
    env = stability_envelope(stone_impulse, N0_grid, bw_grid, t_end=2.5, dt=1.0e-3)
    lift = env["liftoff_s"][0]
    assert lift[0] > 0.0      # low reserve -> lift-off
    assert lift[-1] == 0.0    # high reserve -> no lift-off
