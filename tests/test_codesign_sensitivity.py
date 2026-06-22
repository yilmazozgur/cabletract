"""Validation for S4 -- co-design lever sensitivity sweep."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.codesign_sensitivity import sweep_codesign, required_augers


def test_reference_row_reproduces_manuscript() -> None:
    df = sweep_codesign(n=21)
    ref = df.iloc[0]  # first row is the codesigned reference draft
    assert abs(ref["draft_P50_N"] - 1800.0) < 1e-6
    assert abs(ref["energy_per_decare_Wh"] - 889.5) < 1.0      # 889 Wh/decare headline
    assert 11.0 < ref["decares_per_day_offgrid"] < 12.5        # ~11.6 dec/day
    assert int(ref["augers_req_P90_cap400N"]) == 9             # P90 3.0 kN -> 9 augers (400 N)
    assert int(ref["augers_req_P90_cap2000N"]) == 2            # P90 3.0 kN -> 2 augers (2 kN)
    assert abs(ref["npv_replacement_eur"] - 3575.0) < 5.0      # NPV +EUR3575


def test_weaker_codesign_degrades_energy_and_anchor_but_not_npv() -> None:
    df = sweep_codesign(n=21)
    first, last = df.iloc[0], df.iloc[-1]
    # weaker co-design (higher draft) raises energy and anchor demand...
    assert last["energy_per_decare_Wh"] > first["energy_per_decare_Wh"]
    assert last["augers_req_P90_cap400N"] > first["augers_req_P90_cap400N"]
    assert last["decares_per_day_offgrid"] < first["decares_per_day_offgrid"]
    # ...but the off-grid replacement-frame NPV is essentially invariant
    assert abs(last["npv_replacement_eur"] - first["npv_replacement_eur"]) < 1.0


def test_required_augers_formula() -> None:
    assert required_augers(3000.0, 400.0) == 9   # ceil(1.15*3000/400)=ceil(8.625)
    assert required_augers(3000.0, 2000.0) == 2   # ceil(1.15*3000/2000)=ceil(1.725)
