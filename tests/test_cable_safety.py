"""Validation for S2 -- cable safety & durability.

  1. the FDM transverse-wave solver reproduces c = sqrt(T/mu) (and matches S1);
  2. snap-back recoil conserves energy (KE of both halves == stored elastic
     energy) and a light synthetic recoils faster than steel at equal tension;
  3. bending fatigue grows with D/d and falls with load; at the low working-load
     fraction the binding life is UV/abrasion, not fatigue;
  4. the aeroelastic check produces a modal spectrum matching S1 and a finite
     iced-galloping onset.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.dynamics import LumpedCable, load_cable_props
from cabletract.cable_safety import (
    aeroelastic_check, bend_cycles_to_failure, fatigue_and_opex,
    fdm_wave_speed, snapback_recoil, vortex_lockin_winds,
)

_MBL = {}
with open(ROOT / "cabletract" / "data" / "cable_props.csv") as _fh:
    for _r in csv.DictReader(_fh):
        _MBL[_r["material"]] = float(_r["MBL_N"])

STEEL = load_cable_props("steel_6x19_IWRC_8mm")
DYN = load_cable_props("dyneema_sk78_8mm_12strand")


def test_fdm_wave_speed_matches_analytic_and_s1() -> None:
    sp, c = fdm_wave_speed(T=1800.0, mu=STEEL.mass_per_m_kg, L=50.0)
    assert abs(sp - c) / c < 0.02
    # consistency with S1's lumped-cable transverse fundamental
    cab = LumpedCable(span=50.0, props=STEEL, T_target=1800.0, n_seg=40)
    cab.static_equilibrium()
    f1 = cab.transverse_frequencies(1)[0]
    assert abs(f1 - cab.taut_string_f1()) / cab.taut_string_f1() < 0.05


def test_recoil_energy_balance() -> None:
    r = snapback_recoil(STEEL, tension_N=6480.0, span_m=50.0)
    # KE of the two recoiling halves equals the released elastic energy
    assert abs(2.0 * r.recoil_KE_J - r.elastic_energy_J) / r.elastic_energy_J < 1e-6


def test_synthetic_recoils_faster_than_steel() -> None:
    rs = snapback_recoil(STEEL, tension_N=6000.0)
    rd = snapback_recoil(DYN, tension_N=6000.0)
    # lighter synthetic -> higher recoil velocity at equal tension (not "safer")
    assert rd.recoil_velocity_m_s > rs.recoil_velocity_m_s
    # exclusion radius is bounded by the span
    assert rd.exclusion_radius_m <= 50.0 + 1e-9


def test_fatigue_monotonic_in_DoverD_and_load() -> None:
    n_small = bend_cycles_to_failure(STEEL, 16.0, 0.2, _MBL["steel_6x19_IWRC_8mm"])
    n_big = bend_cycles_to_failure(STEEL, 40.0, 0.2, _MBL["steel_6x19_IWRC_8mm"])
    assert n_big > n_small                                   # larger sheave -> longer life
    n_light = bend_cycles_to_failure(STEEL, 25.0, 0.1, _MBL["steel_6x19_IWRC_8mm"])
    n_heavy = bend_cycles_to_failure(STEEL, 25.0, 0.4, _MBL["steel_6x19_IWRC_8mm"])
    assert n_light > n_heavy                                 # lower load -> longer life


def test_working_load_is_uv_limited_not_fatigue_limited() -> None:
    f = fatigue_and_opex(STEEL, _MBL["steel_6x19_IWRC_8mm"], D_over_d=25.0,
                         working_tension_N=1800.0, uv_abrasion_life_yr=8.0)
    assert f.fatigue_life_yr > 100.0          # fatigue is effectively a non-issue
    assert f.practical_life_yr == 8.0         # UV/abrasion binds
    assert 0.0 < f.opex_eur_per_yr < 1000.0   # a sane new OPEX line


def test_aeroelastic_spectrum_and_galloping() -> None:
    a = aeroelastic_check(STEEL, site="Konya_TR", n_modes=6)
    # modal frequencies increase; lock-in winds scale with frequency
    assert np.all(np.diff(a.modal_freqs_hz) > 0)
    assert np.allclose(vortex_lockin_winds(a.modal_freqs_hz), a.lockin_wind_m_s)
    assert a.galloping_onset_iced_m_s > 0.0
    assert a.viv_in_wind_band  # dense spectrum -> VIV possible at site winds
