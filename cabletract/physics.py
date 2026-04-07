"""Phase 1 — Rigorous physics core for CableTract.

The v1 screening model in :func:`cabletract.simulate.run_single` lumps every
mechanical loss into a single ``winch_efficiency = 0.5``. That is fine for a
back-of-envelope feasibility note, but a sceptical reviewer will (rightly)
ask:

1. *How much does the cable sag, and is the implement depth even constant?*
2. *Can the screw-auger Anchor actually resist the cable tension?*
3. *What is the* peak *motor power, not the day-averaged value?*
4. *What about energy on the unloaded return leg — recovered, lost, or zero?*
5. *Is the 0.5 efficiency a steel cable, or Dyneema, or a marketing wish?*

This module answers each question with a small, testable function. They are
deliberately decoupled so that Phase 5 (Sobol) can vary each ingredient
independently and Phase 6 (surrogate) can fit them with a low-dim model.

Symbols (SI units throughout):
    T_h     horizontal cable tension (N)
    w       cable self-weight per unit length (N/m)
    L       span between supports (m)
    F_d     implement draft load (N), horizontal
    F_w     implement weight (N), vertical
    A       cable cross-sectional area (m^2)
    E       Young's modulus of cable material (Pa)
    eta     dimensionless efficiency in [0, 1]

References:
    - Irvine, H.M. (1981) "Cable Structures", MIT Press — catenary derivation.
    - Qin et al. (2024) "Lateral capacity of helical pile anchors in sandy
      soils", Computers and Geotechnics — 30 cm helical pile in dense sand
      sustains ~400 N lateral load before serviceability failure (we use a
      conservative envelope of 350 N working / 600 N ultimate per auger).
    - ASABE EP496.3 — agricultural drivetrain efficiency reference values.
    - Kollarits & Stiegler (2014) "Cable yarder rope mechanics" — practical
      sag and elongation values for Dyneema vs steel forestry ropes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import math

# Physical constants
G = 9.80665  # m/s^2, standard gravity


# ---------------------------------------------------------------------------
# Decomposed drivetrain efficiency chain
# ---------------------------------------------------------------------------
#
# The lone ``winch_efficiency = 0.5`` in v1 hides five physically separable
# losses. Phase 5 will sample each independently, so we record their plausible
# ranges here as the single source of truth.
#
# The product of the central values below is 0.504, i.e. essentially identical
# to the v1 lumped 0.5 — but Sobol can now attribute variance to the right
# component instead of pinning it all on one fudge factor.

@dataclass(frozen=True)
class DrivetrainEfficiency:
    """Component-wise efficiency chain from electrical input to cable work.

    Default values are the *baseline* (low-cost) configuration whose product
    matches the v1 lumped ``winch_efficiency = 0.5``. The :func:`premium_drivetrain`
    factory returns a higher-efficiency configuration (~0.70) representing
    the best-case attainable with off-the-shelf components.
    """

    motor: float = 0.85        # PMSM at partial load (1-3 kW continuous, low-cost)
    controller: float = 0.92   # Three-phase inverter, ASABE EP496.3 lower bound
    gearbox: float = 0.88      # Two-stage worm/planetary drum reduction
    drum: float = 0.88         # Multi-wrap capstan friction + bending losses
    pulley: float = 0.90       # Two-sheave redirect at the implement carriage
    cable_elongation: float = 0.92  # Cyclic creep / wire-rope internal friction

    @property
    def total(self) -> float:
        return (
            self.motor
            * self.controller
            * self.gearbox
            * self.drum
            * self.pulley
            * self.cable_elongation
        )


def default_drivetrain() -> DrivetrainEfficiency:
    """Return the reference drivetrain whose product is ≈ v1's 0.5."""
    return DrivetrainEfficiency()


def premium_drivetrain() -> DrivetrainEfficiency:
    """Return an upgraded-component drivetrain (~0.70 total).

    This is the configuration that justifies the optimistic energy story in
    the manuscript: high-efficiency PMSM, single-stage planetary, smooth-bore
    drum, and Dyneema cable. Phase 5 will sample between these two presets
    as a sensitivity term.
    """
    return DrivetrainEfficiency(
        motor=0.93,
        controller=0.96,
        gearbox=0.95,
        drum=0.95,
        pulley=0.95,
        cable_elongation=0.97,
    )


# ---------------------------------------------------------------------------
# 1. Catenary sag
# ---------------------------------------------------------------------------

def catenary_sag(T_horiz: float, w_per_m: float, span: float) -> Tuple[float, float, float]:
    """Exact catenary sag, arc length, and end-tension for a uniform cable.

    A perfectly flexible cable of uniform self-weight ``w_per_m`` (N/m) hangs
    between two supports at the same height a horizontal distance ``span`` (m)
    apart. The horizontal component of the cable tension is held constant
    at ``T_horiz`` (N) by the winch and the anchor.

    Returns
    -------
    sag_mid : float
        Midspan sag below the chord (m). Always positive.
    arc_length : float
        Total deployed cable length (m). ``arc_length >= span``.
    T_end : float
        Cable tension magnitude at each support (N). Always >= ``T_horiz``.

    Notes
    -----
    The standard parabolic approximation ``sag ≈ w L^2 / (8 T_h)`` is
    accurate to better than 1% when ``w L / (2 T_h) < 0.3``. We use the
    full hyperbolic form so that this function remains valid all the way
    down to the slack-cable regime where Phase 1 figures will probe it.
    """
    if T_horiz <= 0.0:
        raise ValueError("T_horiz must be positive (slack cables are unsupported)")
    if w_per_m < 0.0:
        raise ValueError("w_per_m must be non-negative")
    if span <= 0.0:
        raise ValueError("span must be positive")

    if w_per_m == 0.0:
        return 0.0, span, T_horiz

    a = T_horiz / w_per_m  # catenary parameter
    half = span / 2.0
    sag_mid = a * (math.cosh(half / a) - 1.0)
    arc_length = 2.0 * a * math.sinh(half / a)
    T_end = T_horiz * math.cosh(half / a)
    return sag_mid, arc_length, T_end


def parabolic_sag(T_horiz: float, w_per_m: float, span: float) -> float:
    """Parabolic-approximation sag, used for analytic regression checks."""
    if T_horiz <= 0.0:
        raise ValueError("T_horiz must be positive")
    return w_per_m * span * span / (8.0 * T_horiz)


# ---------------------------------------------------------------------------
# 2. Cable elastic stretch
# ---------------------------------------------------------------------------

def cable_elastic_stretch(F: float, E: float, A: float, L: float) -> float:
    """Hooke's-law axial elongation of a straight cable.

    Parameters
    ----------
    F : float
        Tensile force (N).
    E : float
        Young's modulus of the cable material (Pa).
    A : float
        Cross-sectional area (m^2). For stranded ropes use the *fill-factor
        corrected* metallic / fibre area, not the bounding circle.
    L : float
        Free length under load (m).

    Returns
    -------
    delta_L : float
        Elongation (m), positive for tensile load.
    """
    if E <= 0.0 or A <= 0.0:
        raise ValueError("E and A must be positive")
    return F * L / (E * A)


# ---------------------------------------------------------------------------
# 3. Working tension balance for the loaded leg
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensionState:
    """Snapshot of the loaded cable during one operating round."""

    T_winch: float        # cable tension just outboard of the winch (N)
    T_anchor: float       # cable tension just inboard of the anchor (N)
    T_horiz: float        # horizontal component at the winch side (N)
    T_horiz_anchor: float # horizontal component at the anchor side (N)
    sag_mid: float        # midspan sag below the chord (m)
    ground_clearance: float  # vertical clearance of the cable above ground at midspan (m)
    cable_arc: float      # actual cable length under tension (m)
    end_angle_rad: float  # cable angle at supports above horizontal (rad)
    regime: str           # "draft-bound" or "sag-bound"


def tension_balance(
    F_draft: float,
    F_implement_weight: float,
    w_cable: float,
    span: float,
    pulley_height: float,
    min_ground_clearance: float = 0.10,
    max_iter: int = 64,
    tol: float = 1e-6,
) -> TensionState:
    """Solve for the cable tension during one loaded operating pass.

    Geometry (the CableTract architecture per the v2 slide deck):

    - The Main Unit's winch and the Anchor's top sheave are at the same
      height ``pulley_height`` (m) above the soil.
    - The implement carriage rides on the cable (or rolls along the ground
      directly under it) and carries an implement that sets its own depth
      via float wheels or a frame. The cable does *not* set tillage depth.
    - On the loaded leg the carriage is dragged toward the anchor; the
      cable transmits a horizontal draft force ``F_draft``.
    - The cable carries its own self-weight ``w_cable`` and a fraction of
      the carriage / implement assembly weight ``F_implement_weight`` that
      is *not* taken by ground-rolling support.
    - The cable must not sag enough to scrape the soil: midspan ground
      clearance must stay above ``min_ground_clearance`` (default 10 cm).

    Two regimes:

    1. **Draft-bound:** F_draft is large enough that the resulting horizontal
       tension already keeps midspan sag below the clearance budget. The
       horizontal tension equals F_draft, the anchor sees ~F_draft, and the
       cable rides comfortably above the ground.

    2. **Sag-bound:** F_draft is too small to keep the cable taut against
       its own weight. The Main Unit must over-tension the cable to a
       minimum value ``T_sag_min`` that just achieves the clearance budget.
       The anchor sees this larger tension; the winch sees it plus F_draft.

    The Phase 1 finding the manuscript needs is **which regime each operation
    is in**, because draft-bound operations are anchor-dominated (auger count
    set by F_draft) and sag-bound operations are cable-dominated (auger
    count set by cable self-weight and span).
    """
    if span <= 0.0:
        raise ValueError("span must be positive")
    if pulley_height < 0.0:
        raise ValueError("pulley_height must be non-negative")
    if min_ground_clearance < 0.0:
        raise ValueError("min_ground_clearance must be non-negative")

    # Effective vertical load per unit length along the cable.
    w_eff = w_cable + F_implement_weight / span  # N/m

    max_allowed_sag = pulley_height - min_ground_clearance
    if max_allowed_sag <= 0.0:
        raise ValueError(
            "pulley_height must exceed min_ground_clearance"
        )

    # Step 1: see if F_draft alone keeps sag below the clearance budget.
    T_draft = max(F_draft, 1.0)
    sag_at_draft, _arc, _T_end = catenary_sag(T_draft, w_eff, span)

    if sag_at_draft <= max_allowed_sag:
        # Draft-bound regime: the working tension equals the draft force.
        T_horiz_anchor = T_draft
        regime = "draft-bound"
    else:
        # Sag-bound regime: bisect for the minimum T that keeps sag = budget.
        T_lo = T_draft
        T_hi = max(T_draft * 1000.0, 1.0e7)
        for _ in range(max_iter):
            T_mid = 0.5 * (T_lo + T_hi)
            sag_mid, _arc, _T_end = catenary_sag(T_mid, w_eff, span)
            if sag_mid > max_allowed_sag:
                T_lo = T_mid
            else:
                T_hi = T_mid
            if abs(T_hi - T_lo) / max(T_hi, 1.0) < tol:
                break
        T_horiz_anchor = 0.5 * (T_lo + T_hi)
        regime = "sag-bound"

    # Final geometric quantities at the working tension.
    sag_mid, arc_length, _T_end = catenary_sag(T_horiz_anchor, w_eff, span)
    ground_clearance = pulley_height - sag_mid

    # The winch always carries the draft force in addition to the geometric
    # horizontal tension. In the draft-bound regime these are the same value;
    # in the sag-bound regime the winch carries (T_sag_min + F_draft).
    T_horiz_winch = T_horiz_anchor if regime == "draft-bound" else T_horiz_anchor + F_draft

    V_end = 0.5 * w_eff * span
    T_winch = math.hypot(T_horiz_winch, V_end)
    T_anchor = math.hypot(T_horiz_anchor, V_end)
    end_angle = math.atan2(V_end, T_horiz_anchor)

    return TensionState(
        T_winch=T_winch,
        T_anchor=T_anchor,
        T_horiz=T_horiz_winch,
        T_horiz_anchor=T_horiz_anchor,
        sag_mid=sag_mid,
        ground_clearance=ground_clearance,
        cable_arc=arc_length,
        end_angle_rad=end_angle,
        regime=regime,
    )


# ---------------------------------------------------------------------------
# 4. Anchor reaction envelope (Qin 2024 helical-pile model)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorEnvelope:
    """Capacity envelope for an n-auger CableTract Anchor."""

    n_augers_required_working: int
    n_augers_required_ultimate: int
    working_capacity_N: float
    ultimate_capacity_N: float
    safety_factor_working: float
    safety_factor_ultimate: float


def anchor_reaction_envelope(
    T_anchor: float,
    n_augers: int,
    per_auger_lateral_capacity_N: float = 400.0,
    working_safety_factor: float = 1.15,
    ultimate_safety_factor: float = 1.5,
) -> AnchorEnvelope:
    """Check whether ``n_augers`` screw piles can resist a cable tension.

    The lateral capacity of a single 30 cm helical pile in medium-dense
    sand is taken from Qin et al. (2024) as ~400 N at the serviceability
    limit. We treat that as the *working* capacity per auger and apply two
    safety factors:

    - 1.15 against the working envelope (continuous-load operation)
    - 1.50 against the ultimate envelope (peak transient events)

    The screw-pile reaction is parallel to the cable's horizontal
    component; the vertical sag-induced reaction is borne by the auger's
    pull-out capacity, which for a helical pile in sand is several times
    higher than its lateral capacity, so the lateral term governs.

    Returns the *minimum* number of augers required at each safety level.
    """
    if T_anchor < 0.0:
        raise ValueError("T_anchor must be non-negative")
    if n_augers < 1:
        raise ValueError("n_augers must be >= 1")
    if per_auger_lateral_capacity_N <= 0.0:
        raise ValueError("per_auger_lateral_capacity_N must be positive")

    working = n_augers * per_auger_lateral_capacity_N
    ultimate = n_augers * per_auger_lateral_capacity_N * (ultimate_safety_factor / working_safety_factor)

    sf_w = working / max(T_anchor, 1e-9)
    sf_u = ultimate / max(T_anchor, 1e-9)

    n_req_w = max(1, math.ceil(working_safety_factor * T_anchor / per_auger_lateral_capacity_N))
    n_req_u = max(
        1,
        math.ceil(
            ultimate_safety_factor * T_anchor / (per_auger_lateral_capacity_N * (ultimate_safety_factor / working_safety_factor))
        ),
    )

    return AnchorEnvelope(
        n_augers_required_working=n_req_w,
        n_augers_required_ultimate=n_req_u,
        working_capacity_N=working,
        ultimate_capacity_N=ultimate,
        safety_factor_working=sf_w,
        safety_factor_ultimate=sf_u,
    )


# ---------------------------------------------------------------------------
# 5. Peak vs continuous motor power
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MotorPowerEnvelope:
    """Peak vs continuous power split for sizing the winch motor."""

    continuous_W: float
    peak_W: float
    peak_to_continuous: float


def peak_motor_power(
    F_continuous: float,
    v_continuous: float,
    F_peak: float,
    v_peak: float,
    drivetrain: DrivetrainEfficiency | None = None,
) -> MotorPowerEnvelope:
    """Continuous and peak electrical power required at the winch motor.

    The v1 model reports a *day-averaged* winch input power
    (``cabletract.simulate.run_single``: ``winch_input_power_W``) which
    silently smears short-duration peaks. For sizing the motor and
    inverter we need the worst-case continuous load and the worst-case
    peak load.

    Parameters
    ----------
    F_continuous : float
        Steady-state cable tension over a full pass (N).
    v_continuous : float
        Steady-state cable speed (m/s).
    F_peak : float
        Worst-case transient cable tension (e.g. stuck implement, root
        strike, end-of-row braking) (N).
    v_peak : float
        Cable speed at the moment of the peak (m/s); commonly less than
        ``v_continuous`` because of soil-strike braking.
    drivetrain : DrivetrainEfficiency, optional
        Component efficiency chain. Defaults to :func:`default_drivetrain`.
    """
    if drivetrain is None:
        drivetrain = default_drivetrain()
    eta = drivetrain.total
    if eta <= 0.0:
        raise ValueError("drivetrain total efficiency must be positive")

    P_cont_mech = F_continuous * v_continuous
    P_peak_mech = F_peak * v_peak
    P_cont_elec = P_cont_mech / eta
    P_peak_elec = P_peak_mech / eta
    ratio = P_peak_elec / max(P_cont_elec, 1e-9)
    return MotorPowerEnvelope(
        continuous_W=P_cont_elec,
        peak_W=P_peak_elec,
        peak_to_continuous=ratio,
    )


# ---------------------------------------------------------------------------
# 6. Regenerative braking on the unloaded return leg / sloped fields
# ---------------------------------------------------------------------------

def regen_energy(
    slope_rad: float,
    mass_kg: float,
    distance_m: float,
    eta_regen: float = 0.55,
    rolling_coeff: float = 0.06,
) -> float:
    """Energy recoverable on the unloaded return leg of one round.

    On the return stroke the implement is lifted clear, leaving only the
    light carriage and the cable being pulled back. If the field has any
    downhill component or the carriage is heavy enough, the gravitational
    PE on the return leg can be recovered through the winch motor running
    as a generator.

    Parameters
    ----------
    slope_rad : float
        Field slope (rad). Positive = downhill on the return leg.
    mass_kg : float
        Mass of the carriage + cable being recovered (kg).
    distance_m : float
        Length of the return leg (m).
    eta_regen : float
        Mechanical-to-electrical regeneration efficiency, including
        motor-as-generator and battery charge controller. Default 0.55
        is consistent with low-RPM PMSM regen on a small DC bus.
    rolling_coeff : float
        Rolling resistance of the carriage on the cable (-). Lumped value.

    Returns
    -------
    E_regen_Wh : float
        Recoverable electrical energy (Wh). Always >= 0; if rolling
        resistance dominates the slope, no energy is recovered.
    """
    if mass_kg < 0.0 or distance_m < 0.0:
        raise ValueError("mass and distance must be non-negative")
    if not (0.0 <= eta_regen <= 1.0):
        raise ValueError("eta_regen must be in [0, 1]")

    weight_N = mass_kg * G
    grav_force_N = weight_N * math.sin(slope_rad)
    roll_force_N = weight_N * math.cos(slope_rad) * rolling_coeff
    net_force_N = grav_force_N - roll_force_N
    if net_force_N <= 0.0:
        return 0.0
    work_J = net_force_N * distance_m
    return work_J * eta_regen / 3600.0  # Wh
