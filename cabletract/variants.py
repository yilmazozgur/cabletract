"""Phase 6b — Architectural variants of CableTract.

This module compares several architectural variants of the two-module
CableTract design against the codesigned baseline:

1. **CableTract+** — a 4-Main-Unit planar cable robot. Two cables pull
   the carriage simultaneously, which (a) splits the PER-CABLE tension
   geometrically across two cables (an anchor-envelope benefit, not an
   energy saving) and (b) eliminates the per-round anchoring cycle
   because the four corner stations are set once per field.
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
the baseline `CableTractParams` so we can re-use `simulate.run_single`
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
    draft each of two simultaneously-pulling cables carries (1/sqrt(2)
    for an orthogonal pull); it relaxes the PER-CABLE tension and the
    per-corner anchor reaction, but it is NOT an energy saving — the
    winches jointly still deliver the full draft x velocity, so it is
    deliberately not applied to `draft_load_N` (v2 wrongly did, which
    booked the tension split as a 29% cut in soil-draft energy).
    `setup_overhead_reduction` captures the elimination of the per-strip
    anchoring cycle: the four corner stations are set once per field, so
    both the per-round re-anchoring energy and most of the per-round
    setup time disappear; a small carriage-turnaround overhead remains.
    """
    n_main_units: int = 4
    geometric_load_split: float = 0.707  # per-cable tension relief only (anchor envelope)
    setup_overhead_reduction: float = 0.9  # per-round anchoring cycle eliminated; turnaround remains
    # 4 main units (each incl. regen drive) + battery + PV + wind + install, no anchor:
    # (4 × 17800 + 0 + 2800 + 3420 + 1650 + 1500 + 4000) / 38670 ≈ 2.198
    capex_multiplier: float = 2.198
    mass_multiplier: float = 2.4   # 4 corner masts vs 1 MU + 1 Anchor


def cabletract_plus_params(p: CableTractParams, spec: CableTractPlusSpec | None = None) -> CableTractParams:
    """Transform a CableTractParams into the equivalent CableTract+ params.

    Honest accounting (v3): the draft load and strip width are UNCHANGED
    — decomposing the pull onto two cables does not reduce the work done
    against the soil, and servoing two axes does not widen the
    implement. What CT+ genuinely changes: (a) the per-round anchoring
    cycle disappears (corner stations are set once per field), which
    removes the per-round anchoring energy and most of the setup time;
    (b) per-cable tension and per-corner reaction drop by ~0.707 (an
    anchor-envelope benefit, reported in text, not an energy term);
    (c) capex multiplies for the four Main Units.
    """
    s = spec if spec is not None else CableTractPlusSpec()
    return replace(
        p,
        # Anchoring cycle eliminated: corner stations set once per field.
        anchoring_energy_Wh_per_round=0.0,
        setup_time_s=p.setup_time_s * (1.0 - s.setup_overhead_reduction),
        cost_cabletract_usd=p.cost_cabletract_usd * s.capex_multiplier,
    )


def cabletract_plus_results(p: CableTractParams, spec: CableTractPlusSpec | None = None) -> CableTractResults:
    return run_single(cabletract_plus_params(p, spec))


# ---------------------------------------------------------------------------
# Multi-strip anchoring (beam + travelling sheave) — v4 candidate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MultiStripAnchorSpec:
    """Variant 4: anchor each module once per ``k_strips`` strips.

    The Anchor gains a rigid transverse beam (folded for transport) with a
    travelling sheave trolley; the MU holds by braked wheels + its screws
    set once per block. Between strips only the trolleys index laterally
    by one strip width (~``trolley_shift_s``); the full 13-screw anchoring
    cycle (energy ``anchoring_energy_Wh_per_round`` of the baseline, time
    ``block_cycle_s``) is paid once per block of ``k_strips``.

    The load-path price: with the trolley at the beam end, the cable pull
    acts at offset ((k-1)/2 x strip width) from the screw-cluster centre,
    producing a yaw moment the cluster must resist as differential
    lateral screw loads (see cabletract.anchoring.beam_yaw_per_screw_N).
    That check, not the mechanism, is what limits k.
    """
    k_strips: int = 4
    trolley_shift_s: float = 15.0    # index trolleys + re-tension, per strip
    block_cycle_s: float = 75.0      # full screw cycle + k-strip roll + survey
    beam_capex_eur: float = 1500.0   # folding beam + trolley + end fittings


def multi_strip_anchor_params(p: CableTractParams, spec: MultiStripAnchorSpec | None = None) -> CableTractParams:
    """Transform the baseline into the multi-strip-anchoring variant.

    Honest accounting: the anchoring ENERGY and the block cycle TIME are
    divided over k strips; nothing else changes (draft, width, drivetrain
    identical). The yaw-moment feasibility of k is checked separately and
    reported in the manuscript text, not assumed here.
    """
    s = spec if spec is not None else MultiStripAnchorSpec()
    k = max(int(s.k_strips), 1)
    mean_setup_s = ((k - 1) * s.trolley_shift_s + s.block_cycle_s) / k
    return replace(
        p,
        anchoring_energy_Wh_per_round=p.anchoring_energy_Wh_per_round / k,
        setup_time_s=mean_setup_s,
        cost_cabletract_usd=p.cost_cabletract_usd + s.beam_capex_eur,
    )


def multi_strip_anchor_results(p: CableTractParams, spec: MultiStripAnchorSpec | None = None) -> CableTractResults:
    return run_single(multi_strip_anchor_params(p, spec))


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

    # The codesigned baseline includes regenerative braking by default
    # (winch_efficiency = 0.518 = 0.50 one-way chain + a slope-averaged
    # ~3.5% four-quadrant recovery, ~0 on flat ground, plus a €300
    # four-quadrant drive). The "unidirectional" variant strips both back
    # out to show what regen buys.
    no_regen = replace(base, winch_efficiency=0.5,
                       cost_cabletract_usd=base.cost_cabletract_usd - 300.0)

    cases = [
        ("Codesigned baseline (regen default)", run_single(base)),
        ("Multi-strip anchoring (beam, k=4)", multi_strip_anchor_results(base)),
        ("CableTract+ (4-Main-Unit cable robot)", cabletract_plus_results(base)),
        ("Circular pulley", circular_pulley_results(base)),
        ("Drone-assisted alignment", drone_alignment_results(base)),
        ("Unidirectional drivetrain (no regen)", run_single(no_regen)),
    ]

    # Costs differ by variant — re-derive from the transformed params.
    costs = [
        base.cost_cabletract_usd,
        multi_strip_anchor_params(base).cost_cabletract_usd,
        cabletract_plus_params(base).cost_cabletract_usd,
        circular_pulley_params(base).cost_cabletract_usd,
        drone_alignment_params(base).cost_cabletract_usd,
        no_regen.cost_cabletract_usd,
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
