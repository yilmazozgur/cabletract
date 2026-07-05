"""Verification tests for cabletract.variants (Phase 6).

Each test pins one invariant of the variant transformations or the
combined comparison helper.

Run as:

    python tests/test_variants.py
    pytest tests/test_variants.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.params import CableTractParams  # noqa: E402
from cabletract.simulate import run_single  # noqa: E402
from cabletract.variants import (  # noqa: E402
    CableTractPlusSpec,
    MultiStripAnchorSpec,
    multi_strip_anchor_params,
    multi_strip_anchor_results,
    CircularPulleySpec,
    DroneAlignmentSpec,
    RegenSpec,
    cabletract_plus_params,
    cabletract_plus_results,
    circular_pulley_params,
    circular_pulley_results,
    compare_all_variants,
    drone_alignment_effect,
    drone_alignment_params,
    drone_alignment_results,
    regen_params,
    regen_results,
)


# ---------------------------------------------------------------------------
# CableTract+
# ---------------------------------------------------------------------------

def test_ctplus_capex_higher_than_baseline() -> None:
    base = CableTractParams()
    ct_plus = cabletract_plus_params(base)
    assert ct_plus.cost_cabletract_usd > base.cost_cabletract_usd


def test_ctplus_setup_time_lower_than_baseline() -> None:
    base = CableTractParams()
    ct_plus = cabletract_plus_params(base)
    assert ct_plus.setup_time_s < base.setup_time_s


def test_ctplus_throughput_above_baseline() -> None:
    # v3 honest accounting: CT+ gains throughput only from eliminating
    # the per-round anchoring cycle (setup + anchoring energy), not from
    # the per-cable tension split, so the gain is modest.
    base = CableTractParams.codesigned()
    base_r = run_single(base)
    plus_r = cabletract_plus_results(base)
    assert plus_r.decares_per_day_offgrid > base_r.decares_per_day_offgrid


def test_ctplus_energy_accounting_honest() -> None:
    # The tension split relaxes PER-CABLE tension (anchor envelope), but
    # must NOT be booked as a reduction in soil-draft energy, and the
    # implement width must not silently widen. The anchoring cycle is
    # eliminated (corner stations set once per field).
    base = CableTractParams.codesigned()
    plus = cabletract_plus_params(base)
    assert plus.draft_load_N == base.draft_load_N
    assert plus.width_m == base.width_m
    assert plus.anchoring_energy_Wh_per_round == 0.0



def test_multistrip_energy_and_setup_reduced() -> None:
    # k=4 divides the anchoring energy by 4 and cuts the mean setup time;
    # draft, width and drivetrain are untouched (honest accounting).
    base = CableTractParams.codesigned()
    v4 = multi_strip_anchor_params(base)
    assert abs(v4.anchoring_energy_Wh_per_round - base.anchoring_energy_Wh_per_round / 4) < 1e-9
    assert v4.setup_time_s < base.setup_time_s
    assert v4.draft_load_N == base.draft_load_N
    assert v4.width_m == base.width_m
    assert v4.winch_efficiency == base.winch_efficiency
    assert v4.cost_cabletract_usd > base.cost_cabletract_usd


def test_multistrip_throughput_between_baseline_and_ctplus() -> None:
    base_r = run_single(CableTractParams.codesigned())
    v4_r = multi_strip_anchor_results(CableTractParams.codesigned())
    plus_r = cabletract_plus_results(CableTractParams.codesigned())
    assert base_r.decares_per_day_offgrid < v4_r.decares_per_day_offgrid < plus_r.decares_per_day_offgrid


def test_beam_yaw_check_scales_with_k() -> None:
    from cabletract.anchoring import beam_yaw_per_screw_N
    y2 = beam_yaw_per_screw_N(3000.0, 0.75)
    y4 = beam_yaw_per_screw_N(3000.0, 2.25)
    y6 = beam_yaw_per_screw_N(3000.0, 3.75)
    assert y2 < y4 < y6
    assert abs(y4 - 1875.0) < 1.0  # k=4 at P90: within the frame-coupled nominal

# ---------------------------------------------------------------------------
# Circular pulley
# ---------------------------------------------------------------------------

def test_circular_pulley_setup_time_reduced() -> None:
    base = CableTractParams()
    cp = circular_pulley_params(base)
    assert cp.setup_time_s < base.setup_time_s


def test_circular_pulley_capex_slightly_higher() -> None:
    base = CableTractParams()
    cp = circular_pulley_params(base)
    assert cp.cost_cabletract_usd > base.cost_cabletract_usd
    # But not dramatically — only a small mechanical add-on
    assert cp.cost_cabletract_usd < base.cost_cabletract_usd * 1.1


def test_circular_pulley_throughput_above_baseline() -> None:
    base = CableTractParams()
    base_r = run_single(base)
    cp_r = circular_pulley_results(base)
    assert cp_r.decares_per_day_offgrid >= base_r.decares_per_day_offgrid


# ---------------------------------------------------------------------------
# Drone alignment
# ---------------------------------------------------------------------------

def test_drone_alignment_time_saved_positive() -> None:
    out = drone_alignment_effect()
    assert out["time_saved_per_field_s"] > 0.0
    assert out["time_saved_per_day_h"] > 0.0


def test_drone_alignment_time_saved_scales_with_fields() -> None:
    a = drone_alignment_effect(DroneAlignmentSpec(fields_per_day=2))
    b = drone_alignment_effect(DroneAlignmentSpec(fields_per_day=8))
    assert b["time_saved_per_day_h"] > a["time_saved_per_day_h"]


def test_drone_alignment_capex_passed_through() -> None:
    base = CableTractParams()
    da = drone_alignment_params(base)
    assert da.cost_cabletract_usd > base.cost_cabletract_usd


# ---------------------------------------------------------------------------
# Regen on return leg
# ---------------------------------------------------------------------------

def test_regen_increases_effective_winch_efficiency() -> None:
    base = CableTractParams()
    rg = regen_params(base)
    assert rg.winch_efficiency > base.winch_efficiency
    # Should not blow past unity
    assert rg.winch_efficiency < 1.0


def test_regen_reduces_energy_per_decare() -> None:
    base = CableTractParams()
    base_r = run_single(base)
    rg_r = regen_results(base)
    assert rg_r.energy_per_decare_Wh < base_r.energy_per_decare_Wh


def test_regen_recovery_zero_is_identity() -> None:
    base = CableTractParams()
    rg = regen_params(base, RegenSpec(recovery_fraction=0.0))
    assert math.isclose(rg.winch_efficiency, base.winch_efficiency, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# compare_all_variants
# ---------------------------------------------------------------------------

def test_compare_all_variants_returns_6_rows() -> None:
    rows = compare_all_variants()
    assert len(rows) == 6


def test_compare_all_variants_unique_names() -> None:
    rows = compare_all_variants()
    names = [r.name for r in rows]
    assert len(set(names)) == len(names)


def test_compare_all_variants_baseline_first() -> None:
    rows = compare_all_variants()
    assert "baseline" in rows[0].name.lower()


def test_compare_all_variants_ctplus_highest_throughput() -> None:
    """The 4-Main-Unit cable robot must beat every other variant on
    raw decares/day, by construction."""
    rows = compare_all_variants()
    by_name = {r.name: r for r in rows}
    ctplus = [v for k, v in by_name.items() if "CableTract+" in k][0]
    others = [v for k, v in by_name.items() if "CableTract+" not in k]
    for o in others:
        assert ctplus.decares_per_day_offgrid > o.decares_per_day_offgrid, o.name


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_ctplus_capex_higher_than_baseline,
    test_ctplus_setup_time_lower_than_baseline,
    test_ctplus_throughput_above_baseline,
    test_ctplus_energy_accounting_honest,
    test_multistrip_energy_and_setup_reduced,
    test_multistrip_throughput_between_baseline_and_ctplus,
    test_beam_yaw_check_scales_with_k,
    test_circular_pulley_setup_time_reduced,
    test_circular_pulley_capex_slightly_higher,
    test_circular_pulley_throughput_above_baseline,
    test_drone_alignment_time_saved_positive,
    test_drone_alignment_time_saved_scales_with_fields,
    test_drone_alignment_capex_passed_through,
    test_regen_increases_effective_winch_efficiency,
    test_regen_reduces_energy_per_decare,
    test_regen_recovery_zero_is_identity,
    test_compare_all_variants_returns_6_rows,
    test_compare_all_variants_unique_names,
    test_compare_all_variants_baseline_first,
    test_compare_all_variants_ctplus_highest_throughput,
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
    print(f"\nok: cabletract.variants passes {len(ALL_TESTS)} invariants")
