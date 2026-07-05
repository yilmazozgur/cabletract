"""Anchoring (auger installation) energy, power, and time model.

v2 of the manuscript budgeted the per-round setup as *time only* (60 s)
and carried no energy or power term for driving and retracting the
helical ground screws, while the S7 capacity study assumed 2.0-m piles
whose installation torque would be incompatible with the small per-auger
BLDC drives and the 30-s parallel insert/retract budget. v3 resolves the
inconsistency by (a) respecifying the anchors as short ground screws
(1.0 m embedment, ~150 mm single helix, 75 mm pitch) sized to the
*required* per-auger working load (~0.3-0.5 kN with the 9-auger group at
the codesigned P90 of 3.0 kN) rather than to construction-pile capacity,
and (b) charging an explicit installation-energy term per anchoring
cycle in the energy budget.

Torque model
------------
Installation torque is taken to ramp linearly with depth from ~0 at the
surface to ``torque_final_Nm`` at full embedment, so the mean over the
insertion is ``torque_final_Nm / 2``. The final torque for a ~150-mm
single-helix screw at 1.0 m in medium agricultural soil is of order
60-120 N.m (hand-held two-operator earth-auger territory); we default to
90 N.m. Retraction is charged at ``retract_frac`` of installation work
(the helix unloads most of the bearing resistance on the way out).
The electrical draw divides mechanical work by ``drive_eta``.

The torque-to-capacity sanity check follows the AC358-style Kt
correlation (Kt ~ 25-33 1/m for small shafts): Q_ult ~ Kt x T_final
= 26 x 90 ~ 2.3 kN ultimate axial per screw; lateral working capacity
after group and safety factors lands in the 0.3-1 kN band consistent
with the manuscript's loose-sand (0.4 kN) envelope floor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AugerSpec:
    """One ground-screw anchor and its drive."""

    depth_m: float = 1.0            # embedment depth
    pitch_m: float = 0.075          # helix pitch (advance per revolution)
    torque_final_Nm: float = 90.0   # installation torque at full depth
    retract_frac: float = 0.30      # retraction work as fraction of insertion
    drive_eta: float = 0.70         # BLDC + gearbox + soil-side losses
    insert_time_s: float = 20.0     # per-auger insert time (parallel drives)

    @property
    def revolutions(self) -> float:
        return self.depth_m / self.pitch_m

    @property
    def mech_energy_insert_J(self) -> float:
        """Mechanical work for one insertion: mean torque x total angle."""
        mean_torque = 0.5 * self.torque_final_Nm
        return mean_torque * 2.0 * math.pi * self.revolutions

    @property
    def mech_energy_cycle_J(self) -> float:
        """Insert + retract mechanical work for one full cycle."""
        return self.mech_energy_insert_J * (1.0 + self.retract_frac)

    @property
    def elec_energy_cycle_J(self) -> float:
        return self.mech_energy_cycle_J / self.drive_eta

    @property
    def peak_drive_power_W(self) -> float:
        """Average electrical power per auger drive during insertion."""
        return (self.mech_energy_insert_J / self.drive_eta) / self.insert_time_s


def anchoring_energy_per_round_Wh(
    n_anchor_augers: int = 9,
    n_mu_augers: int = 4,
    spec: AugerSpec | None = None,
) -> float:
    """Electrical energy (Wh) for one full re-anchoring cycle of both
    modules: all Anchor augers plus all Main-Unit self-anchoring augers,
    inserted and retracted once."""
    s = spec if spec is not None else AugerSpec()
    n = n_anchor_augers + n_mu_augers
    return n * s.elec_energy_cycle_J / 3600.0


def anchoring_transient_power_W(
    n_anchor_augers: int = 9,
    spec: AugerSpec | None = None,
) -> float:
    """Aux-pack electrical transient while the Anchor's augers drive in
    concurrently (the Main Unit's four augers run from the main pack)."""
    s = spec if spec is not None else AugerSpec()
    return n_anchor_augers * s.peak_drive_power_W


def beam_yaw_per_screw_N(
    tension_N: float = 3000.0,
    trolley_offset_m: float = 2.25,
    cluster_row_spacing_m: float = 1.2,
    screws_per_row: int = 3,
) -> float:
    """Differential lateral load per outer-row screw when the cable pull
    acts at ``trolley_offset_m`` from the anchored cluster centre (the
    multi-strip beam variant). The yaw moment T x offset is resisted as a
    push--pull couple across the cluster's outer rows.

    For k strips at 1.5 m the worst offset is (k-1)/2 x 1.5 m:
    k=4 -> 2.25 m, k=6 -> 3.75 m.
    """
    moment = tension_N * trolley_offset_m
    row_pair_force = moment / cluster_row_spacing_m
    return row_pair_force / screws_per_row


if __name__ == "__main__":
    s = AugerSpec()
    print(f"revolutions per insert:        {s.revolutions:.1f}")
    print(f"mech energy per cycle:         {s.mech_energy_cycle_J/1000.0:.2f} kJ")
    print(f"elec energy per cycle:         {s.elec_energy_cycle_J/1000.0:.2f} kJ")
    print(f"per-drive power during insert: {s.peak_drive_power_W:.0f} W")
    print(f"13-auger round energy:         {anchoring_energy_per_round_Wh():.2f} Wh")
    print(f"9-auger Anchor transient:      {anchoring_transient_power_W():.0f} W")
    for k in (2, 4, 6):
        off = (k - 1) / 2.0 * 1.5
        y50 = beam_yaw_per_screw_N(1800.0, off)
        y90 = beam_yaw_per_screw_N(3000.0, off)
        print(f"beam k={k}: offset {off:.2f} m -> per-screw yaw load P50 {y50:.0f} N / P90 {y90:.0f} N")