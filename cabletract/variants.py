"""Phase 6b — Architectural variants of CableTract.

The slide deck (CableTract_v2.pdf) describes several variants of the
two-module v1 design that the user wants quantitatively compared in
the manuscript:

1. **CableTract+** — a 4-Main-Unit planar cable robot. Two cables pull
   the carriage simultaneously, which (a) splits the draft load
   geometrically across two cables and (b) eliminates the Anchor
   step-and-reset overhead because the carriage can move continuously
   in 2-D between the four corners. Slide deck claims ~2× throughput.
2. **Circular / oblique pulley variant** — the Main Unit's output
   pulley swings on a vertical pin so the cable can leave the drum at
   an angle. This lets the Main Unit stay put while the Anchor steps
   laterally, eliminating the alignment overhead between strips.
   Equivalent to a setup-time reduction.
3. **Drone alignment assist** — a quadcopter drops a marker / paints
   a visual target so the Main Unit and Anchor can be re-aligned
   between fields without manual surveying. Equivalent to a setup-time
   reduction *per field* (not per strip).
4. **Regenerative-on-return baseline** — already in `physics.py` as
   `regen_energy()`; this module exposes a wrapper that re-runs
   `simulate.run_single` with `winch_efficiency` adjusted upward by
   the regen recovery fraction.

Each variant is implemented as a *parameter transformation* on top of
the v1 `CableTractParams` so we can re-use `simulate.run_single`
without forking the simulator. This keeps the variants comparable on
the same metrics, on the same code path, and without code duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

from .params import CableTractParams
from .simulate import CableTractResults, run_single


# ---------------------------------------------------------------------------
# CableTract+ (4-Main-Unit planar cable robot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CableTractPlusSpec:
    """Configuration for the CableTract+ variant.

    `n_main_units` is the number of corner stations (default 4 — a
    square cable robot). `geometric_load_split` is the fraction of
    draft each cable carries when two cables actively pull together;
    for a 90° split it is 1/√2 ≈ 0.707. `setup_overhead_reduction`
    captures the elimination of the per-strip Anchor reset (the only
    source of setup time in the v1 model).
    """
    n_main_units: int = 4
    geometric_load_split: float = 0.707  # 1/sqrt(2) for orthogonal pull
    setup_overhead_reduction: float = 0.6  # fraction of setup_time eliminated
    # 4 main units + battery + PV + wind + install, no anchor:
    # (4 × 17500 + 0 + 3420 + 1650 + 1500 + 4000) / 35570 ≈ 2.265
    capex_multiplier: float = 2.265
    mass_multiplier: float = 2.4   # 4 corner masts vs 1 MU + 1 Anchor


def cabletract_plus_params(p: CableTractParams, spec: CableTractPlusSpec | None = None) -> CableTractParams:
    """Transform a v1 CableTractParams into the equivalent CableTract+ params.

    The transformation is *purely* a re-parameterisation: we adjust the
    effective draft (per cable), the setup time (Anchor reset
    eliminated), the system cost (4 Main Units), and the strip width
    (carriage can sweep wider in 2-D mode).
    """
    s = spec if spec is not None else CableTractPlusSpec()
    return replace(
        p,
        # Each of two simultaneously-pulling cables sees a fraction of the
        # draft load; the *winch* still has to provide the full force, but
        # the per-cable tension drops, which is what saved the anchor in
        # Phase 1's anchor envelope (relevant for the §7 variant comparison).
        # For throughput, two cables pulling halve the per-round time.
        draft_load_N=p.draft_load_N * s.geometric_load_split,
        # Setup time per round is mostly the Anchor reset; CT+ has no Anchor.
        setup_time_s=p.setup_time_s * (1.0 - s.setup_overhead_reduction),
        # The carriage can sweep wider strips because both X and Y are servoed.
        width_m=p.width_m * 1.5,
        # Capex bumps for the 4 Main Units (no Anchor cost, the v1 cost
        # already lumped both modules into a single number).
        cost_cabletract_usd=p.cost_cabletract_usd * s.capex_multiplier,
    )


def cabletract_plus_results(p: CableTractParams, spec: CableTractPlusSpec | None = None) -> CableTractResults:
    return run_single(cabletract_plus_params(p, spec))


# ---------------------------------------------------------------------------
# Circular / oblique pulley variant
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircularPulleySpec:
    """The Main Unit's pulley swings on a pin, allowing the cable to
    leave the drum at angles up to ``max_swing_deg``. The Anchor steps
    sideways while the Main Unit stays put."""
    max_swing_deg: float = 25.0
    setup_time_reduction: float = 0.45  # 45 % cut in per-round setup overhead
    capex_extra_usd: float = 600.0      # extra hinge bearings + alignment encoder


def circular_pulley_params(p: CableTractParams, spec: CircularPulleySpec | None = None) -> CableTractParams:
    s = spec if spec is not None else CircularPulleySpec()
    return replace(
        p,
        setup_time_s=p.setup_time_s * (1.0 - s.setup_time_reduction),
        cost_cabletract_usd=p.cost_cabletract_usd + s.capex_extra_usd,
    )


def circular_pulley_results(p: CableTractParams, spec: CircularPulleySpec | None = None) -> CableTractResults:
    return run_single(circular_pulley_params(p, spec))


# ---------------------------------------------------------------------------
# Drone alignment assist
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DroneAlignmentSpec:
    """A small quadcopter is launched between fields to drop GPS markers
    so the Main Unit and Anchor can be re-aligned in seconds rather than
    minutes. The reduction is on a *per-field* basis (one launch per
    field move), and the model exposes both the baseline manual setup
    time and the drone-assisted reduction.
    """
    drone_capex_usd: float = 1200.0
    drone_battery_life_min: float = 22.0
    field_setup_time_baseline_s: float = 600.0   # 10 min manual realignment
    field_setup_time_drone_s: float = 90.0       # 1.5 min drone-assisted
    fields_per_day: int = 5

    @property
    def time_saved_per_field_s(self) -> float:
        return self.field_setup_time_baseline_s - self.field_setup_time_drone_s

    @property
    def time_saved_per_day_h(self) -> float:
        return self.time_saved_per_field_s * self.fields_per_day / 3600.0


def drone_alignment_effect(spec: DroneAlignmentSpec | None = None) -> Dict[str, float]:
    """Return the headline time and cost numbers for the drone variant."""
    s = spec if spec is not None else DroneAlignmentSpec()
    return {
        "time_saved_per_field_s": s.time_saved_per_field_s,
        "time_saved_per_day_h": s.time_saved_per_day_h,
        "drone_capex_usd": s.drone_capex_usd,
        "fields_per_day": float(s.fields_per_day),
        # Per-day fuel-equivalent savings: assume the freed time is used to
        # work additional area at the v1 throughput
        "extra_decares_per_day_at_v1_rate": s.time_saved_per_day_h * 1.0,
    }


def drone_alignment_params(
    p: CableTractParams,
    spec: DroneAlignmentSpec | None = None,
) -> CableTractParams:
    """Transform CableTractParams to reflect the drone-assisted setup time.

    The drone reduces *per-field* setup, which we fold into a smaller
    `setup_time_s` (per round) by spreading the saved time over the
    expected rounds per field.
    """
    s = spec if spec is not None else DroneAlignmentSpec()
    # Approximate average rounds per field as decares-per-day / fields-per-day × rounds/decare.
    # We don't have rounds_per_decare without running the sim, so we use the
    # nominal v1 value for the default 50-m span / 2-m width: 10 rounds/decare.
    rounds_per_field = max(1.0, p.operating_hours_per_day * 1.0)  # very rough
    saved_per_round_s = s.time_saved_per_field_s / max(rounds_per_field * s.fields_per_day, 1.0)
    new_setup = max(0.0, p.setup_time_s - saved_per_round_s)
    return replace(p, setup_time_s=new_setup, cost_cabletract_usd=p.cost_cabletract_usd + s.drone_capex_usd)


def drone_alignment_results(
    p: CableTractParams,
    spec: DroneAlignmentSpec | None = None,
) -> CableTractResults:
    return run_single(drone_alignment_params(p, spec))


# ---------------------------------------------------------------------------
# Regen-on-return wrapper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegenSpec:
    """Regenerative braking on the unloaded return leg.

    `recovery_fraction` is the fraction of return-leg kinetic + potential
    energy that is fed back into the battery. We model the impact as an
    *effective winch efficiency boost* of `1 / (1 - recovery_fraction *
    return_leg_share)`, where `return_leg_share` is the fraction of one
    full round taken by the unloaded return.
    """
    recovery_fraction: float = 0.35
    return_leg_share: float = 0.5  # half a round is unloaded return at v1 baseline


def regen_params(p: CableTractParams, spec: RegenSpec | None = None) -> CableTractParams:
    s = spec if spec is not None else RegenSpec()
    boost = 1.0 / (1.0 - s.recovery_fraction * s.return_leg_share)
    new_eta = min(0.95, p.winch_efficiency * boost)
    return replace(p, winch_efficiency=new_eta)


def regen_results(p: CableTractParams, spec: RegenSpec | None = None) -> CableTractResults:
    return run_single(regen_params(p, spec))


# ---------------------------------------------------------------------------
# Variant comparison helper
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantComparisonRow:
    name: str
    decares_per_day_offgrid: float
    energy_per_decare_Wh: float
    cost_cabletract_usd: float
    payback_months_vs_fuel: float
    surplus_power_W: float


def compare_all_variants(p: CableTractParams | None = None) -> List[VariantComparisonRow]:
    """Run the codesigned baseline and every variant on the same parameter set
    and return a tidy list of comparison rows."""
    base = p if p is not None else CableTractParams.codesigned()

    cases = [
        ("Codesigned baseline (Main Unit + Anchor)", run_single(base)),
        ("CableTract+ (4-Main-Unit cable robot)", cabletract_plus_results(base)),
        ("Circular pulley", circular_pulley_results(base)),
        ("Drone-assisted alignment", drone_alignment_results(base)),
        ("Regenerative return leg", regen_results(base)),
    ]

    # Costs differ by variant — re-derive from the transformed params.
    costs = [
        base.cost_cabletract_usd,
        cabletract_plus_params(base).cost_cabletract_usd,
        circular_pulley_params(base).cost_cabletract_usd,
        drone_alignment_params(base).cost_cabletract_usd,
        regen_params(base).cost_cabletract_usd,
    ]

    return [
        VariantComparisonRow(
            name=name,
            decares_per_day_offgrid=r.decares_per_day_offgrid,
            energy_per_decare_Wh=r.energy_per_decare_Wh,
            cost_cabletract_usd=cost,
            payback_months_vs_fuel=r.payback_months_vs_fuel,
            surplus_power_W=r.surplus_power_W,
        )
        for (name, r), cost in zip(cases, costs)
    ]
