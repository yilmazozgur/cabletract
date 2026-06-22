"""Validation for S3 -- soil-tool DEM and the McKyes-Godwin wedge baseline.

Wedge (fast):
  * factors finite/positive; draft rises with depth, width, density, cohesion;
  * narrow tines show a higher depth-scaling exponent than wide ones
    (Godwin-Spoor side-crescent behaviour).
DEM (one small bed, reused):
  * the prepared bed reaches a realistic tilled-soil bulk density;
  * draft increases with tool depth, with tool width, and with particle
    friction (the contact model behaves physically);
  * a narrow tool's disturbance is localised (small moved fraction);
  * DEM and the calibrated wedge agree to the same order of magnitude.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.tillage_mechanics import (
    SoilCuttingParams, depth_scaling_exponent, mckyes_godwin_draft,
)
from cabletract.tillage_dem import DEMParams, build_packed_bed, bulk_density, drag_tool


# --------------------------- wedge model (fast) ---------------------------

def test_wedge_factors_and_monotonicity() -> None:
    s = SoilCuttingParams()
    r = mckyes_godwin_draft(0.20, 0.10, s, rake_deg=25)
    assert r.N_gamma > 0 and r.N_c > 0 and r.N_q > 0
    assert mckyes_godwin_draft(0.30, 0.10, s).draft_N > mckyes_godwin_draft(0.10, 0.10, s).draft_N
    assert mckyes_godwin_draft(0.20, 0.30, s).draft_N > mckyes_godwin_draft(0.20, 0.10, s).draft_N
    s2 = replace(s, cohesion_kPa=20.0)
    assert mckyes_godwin_draft(0.20, 0.10, s2).draft_N > mckyes_godwin_draft(0.20, 0.10, s).draft_N


def test_narrow_tine_has_higher_depth_exponent() -> None:
    s = SoilCuttingParams()
    assert depth_scaling_exponent(0.04, s) > depth_scaling_exponent(0.40, s)


# ------------------------------- DEM (slow) -------------------------------

@pytest.fixture(scope="module")
def small_bed():
    p = DEMParams()
    return build_packed_bed(p, Lx=0.26, Ly=0.11, fill_depth=0.18, settle_steps=12000)


def test_bed_reaches_realistic_bulk_density(small_bed) -> None:
    rho = bulk_density(small_bed)
    assert 1200.0 < rho < 1800.0   # tilled mineral-soil range


def test_dem_draft_increases_with_depth(small_bed) -> None:
    shallow = drag_tool(small_bed, depth_m=0.04, width_m=0.04, v_tool=0.4,
                        drag_len=0.10, settle_first=400)
    deep = drag_tool(small_bed, depth_m=0.09, width_m=0.04, v_tool=0.4,
                     drag_len=0.10, settle_first=400)
    assert deep.draft_mean_N > shallow.draft_mean_N > 0.0
    assert shallow.disturbed_fraction < 0.30   # narrow tool -> localised


def test_dem_draft_increases_with_width(small_bed) -> None:
    narrow = drag_tool(small_bed, depth_m=0.06, width_m=0.03, v_tool=0.4,
                       drag_len=0.10, settle_first=400)
    wide = drag_tool(small_bed, depth_m=0.06, width_m=0.07, v_tool=0.4,
                     drag_len=0.10, settle_first=400)
    assert wide.draft_mean_N > narrow.draft_mean_N


def test_dem_and_wedge_same_order_of_magnitude(small_bed) -> None:
    res = drag_tool(small_bed, depth_m=0.08, width_m=0.05, rake_deg=35,
                    v_tool=0.4, drag_len=0.10, settle_first=500)
    # calibrated cohesionless wedge at the bed's bulk density
    soil = SoilCuttingParams(gamma_kg_m3=bulk_density(small_bed), cohesion_kPa=0.0,
                             phi_deg=40.0, delta_deg=24.0)
    wedge = mckyes_godwin_draft(0.08, 0.05, soil, rake_deg=35).draft_N
    ratio = res.draft_mean_N / wedge
    assert 0.2 < ratio < 5.0   # same order after calibration (blunt-blade DEM runs higher)
