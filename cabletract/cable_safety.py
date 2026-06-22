"""S2 -- Cable safety & durability.

Section 7.2 of the manuscript flags the cable as an unquantified safety surface
and the BOM/LCA carries no cable-replacement schedule. This module closes both
with three high-fidelity-in-Python sub-models:

1. **Snap-back recoil** -- a taut cable at working tension stores elastic strain
   energy U = T^2 L / (2 EA). On a mid-span break each end recoils at the axial
   release velocity v = T / sqrt(EA * m'); the freed end is then a projectile of
   that speed, giving a ballistic **exclusion-zone radius** R ~ v^2 / g. A 1-D
   transient wave solver (numba, with a pure-numpy fallback) on the *transverse*
   string equation mu y_tt = T y_xx validates the wave speed c = sqrt(T/mu).
   Steel vs Dyneema vs UHMWPE are compared from ``cable_props.csv``.

2. **Bending-over-sheave fatigue** -- a Feyrer-type bending-cycle model
   N_bend(D/d, S) for steel wire rope and a cyclic-bend-over-sheave (CBOS)
   endurance curve for HMPE, anchored to published reference points, converted
   through the reeling duty (bends per year) into a **replacement interval** and
   the resulting OPEX (EUR/yr) and embodied-CO2 (kg/yr) lines for the LCA.

3. **Aeroelastic stability** -- vortex-shedding lock-in (f_vs = St U / d against
   the taut-cable modal spectrum from S1's ``LumpedCable``) and the Den Hartog
   galloping onset for an iced section, compared to the bundled monthly mean
   winds (``energy.load_site_meta``) to give a **safe-operation wind band**.

References
----------
- Feyrer, K. (2015) "Wire Ropes: Tension, Endurance, Reliability", 2nd ed.,
  Springer -- bending-fatigue regression and D/d sensitivity.
- Den Hartog, J.P. (1956) "Mechanical Vibrations", 4th ed. -- galloping criterion
  dC_L/d(alpha) + C_D < 0 and the onset wind speed.
- Blevins, R.D. (1990) "Flow-Induced Vibration", 2nd ed. -- Strouhal lock-in.
- Ridge, I.M.L. et al. / HMPE rope CBOS endurance (Bridon, Marlow technical data).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dynamics import CableProps, LumpedCable, load_cable_props

G = 9.80665
RHO_AIR = 1.20      # kg/m^3
STROUHAL = 0.20     # circular cylinder, subcritical Re

CABLE_DIAMETER_M = 0.008  # all three ropes in cable_props.csv are 8 mm


# ===========================================================================
# 1. Snap-back recoil
# ===========================================================================

@dataclass
class RecoilResult:
    material: str
    tension_N: float
    recoil_velocity_m_s: float
    elastic_energy_J: float
    recoil_KE_J: float
    exclusion_radius_m: float        # bounded whip reach (min of ballistic & span)
    ballistic_range_m: float         # unbounded v^2/g (reported for context)
    hazardous: bool                  # recoil KE above a projectile-lethality threshold


# A ~79 J kinetic-energy threshold is a common projectile-lethality guideline.
KE_LETHAL_J = 79.0


def snapback_recoil(props: CableProps, tension_N: float, span_m: float = 50.0,
                    half_mass_fraction: float = 0.5) -> RecoilResult:
    """Recoil velocity, stored/kinetic energy and a bounded exclusion radius.

    The axial release velocity of a suddenly-freed tensioned cable end is
    v = strain * c = (T/EA) * sqrt(EA/m') = T / sqrt(EA * m'). A 45-degree
    ballistic range v^2/g indicates severity, but the end stays tethered to the
    cable, so the *reachable* exclusion radius is capped at the span (the
    geometric whip reach). Severity is flagged by the recoil kinetic energy.
    """
    EA = props.EA_N
    mprime = props.mass_per_m_kg
    v = tension_N / math.sqrt(EA * mprime)
    U = tension_N ** 2 * span_m / (2.0 * EA)             # total stored energy
    half_len = span_m * half_mass_fraction
    ke = 0.5 * (mprime * half_len) * v ** 2              # KE of one recoiling half
    ballistic = v ** 2 / G                               # 45-deg ballistic range
    R = min(ballistic, span_m)                           # bounded by cable geometry
    return RecoilResult(material=props.name, tension_N=tension_N,
                        recoil_velocity_m_s=v, elastic_energy_J=U,
                        recoil_KE_J=ke, exclusion_radius_m=R,
                        ballistic_range_m=ballistic, hazardous=ke > KE_LETHAL_J)


def _fdm_wave_speed_numpy(T, mu, L, n, t_end):
    """Explicit FDM of mu y_tt = T y_xx; returns measured front speed (m/s)."""
    dx = L / (n - 1)
    c = math.sqrt(T / mu)
    dt = 0.4 * dx / c                       # CFL < 1
    r2 = (c * dt / dx) ** 2
    y = np.zeros(n)
    # smooth initial pulse near the left end
    x = np.linspace(0, L, n)
    x0 = 0.12 * L
    y = np.exp(-((x - x0) / (0.02 * L)) ** 2)
    y_prev = y.copy()
    y_new = np.zeros(n)
    nsteps = int(t_end / dt)
    # track the right-moving front (peak position) after it separates
    peak_t0 = x0
    front_x, front_t = [], []
    for k in range(nsteps):
        y_new[1:-1] = (2 * y[1:-1] - y_prev[1:-1]
                       + r2 * (y[2:] - 2 * y[1:-1] + y[:-2]))
        y_new[0] = 0.0
        y_new[-1] = 0.0
        y_prev, y, y_new = y, y_new, y_prev
        if k > nsteps // 6:
            ip = int(np.argmax(y))
            front_x.append(x[ip]); front_t.append((k + 1) * dt)
    # linear fit of front position vs time on the clean propagation window
    fx = np.array(front_x); ft = np.array(front_t)
    sel = (fx > x0 + 0.05 * L) & (fx < 0.85 * L)
    if sel.sum() < 5:
        return float("nan"), c
    speed = np.polyfit(ft[sel], fx[sel], 1)[0]
    return float(speed), c


def fdm_wave_speed(T: float, mu: float, L: float = 50.0, n: int = 2001,
                   t_end: float | None = None) -> tuple[float, float]:
    """Measured vs analytic transverse wave speed (validation of the solver)."""
    c = math.sqrt(T / mu)
    if t_end is None:
        t_end = 0.6 * L / c
    try:
        from numba import njit
        fn = njit(cache=False)(_fdm_wave_step)
        return _run_fdm(fn, T, mu, L, n, t_end), c
    except Exception:
        return _fdm_wave_speed_numpy(T, mu, L, n, t_end)


def _fdm_wave_step(y, y_prev, y_new, r2, n):  # numba kernel
    for i in range(1, n - 1):
        y_new[i] = (2.0 * y[i] - y_prev[i]
                    + r2 * (y[i + 1] - 2.0 * y[i] + y[i - 1]))
    y_new[0] = 0.0
    y_new[n - 1] = 0.0


def _run_fdm(step_fn, T, mu, L, n, t_end):
    dx = L / (n - 1)
    c = math.sqrt(T / mu)
    dt = 0.4 * dx / c
    r2 = (c * dt / dx) ** 2
    x = np.linspace(0, L, n)
    x0 = 0.12 * L
    y = np.exp(-((x - x0) / (0.02 * L)) ** 2)
    y_prev = y.copy()
    y_new = np.zeros(n)
    nsteps = int(t_end / dt)
    fx, ft = [], []
    for k in range(nsteps):
        step_fn(y, y_prev, y_new, r2, n)
        y_prev, y, y_new = y, y_new, y_prev
        if k > nsteps // 6:
            ip = int(np.argmax(y))
            fx.append(x[ip]); ft.append((k + 1) * dt)
    fx = np.array(fx); ft = np.array(ft)
    sel = (fx > x0 + 0.05 * L) & (fx < 0.85 * L)
    if sel.sum() < 5:
        return float("nan")
    return float(np.polyfit(ft[sel], fx[sel], 1)[0])


# ===========================================================================
# 2. Bending-over-sheave fatigue
# ===========================================================================

@dataclass
class FatigueResult:
    material: str
    D_over_d: float
    load_fraction_MBL: float
    bend_cycles_to_failure: float
    bends_per_year: float
    fatigue_life_yr: float
    practical_life_yr: float        # min(fatigue, UV/abrasion limit)
    opex_eur_per_yr: float
    co2_kg_per_yr: float


# Feyrer-type anchor points (bending cycles to failure):
#   steel 6x19 at D/d=25, S=0.20*MBL  ->  ~5e5 bends (Feyrer charts, EIPS)
#   HMPE CBOS  at D/d=20, S=0.20*MBL  ->  ~3e4 cycles (synthetic CBOS data)
_FATIGUE_ANCHOR = {
    "steel": dict(N_ref=5.0e5, DoverD_ref=25.0, S_ref=0.20, p_DoverD=3.0, q_stress=4.8),
    "hmpe": dict(N_ref=3.0e4, DoverD_ref=20.0, S_ref=0.20, p_DoverD=2.0, q_stress=5.0),
}


def _family(material: str) -> str:
    return "steel" if material.startswith("steel") else "hmpe"


def bend_cycles_to_failure(props: CableProps, D_over_d: float,
                           load_fraction_MBL: float, MBL_N: float) -> float:
    """Feyrer-type bending-fatigue life: power law in D/d and a Basquin stress term.

    N = N_ref (D/d / (D/d)_ref)^p (S_ref / S)^q, anchored to published points.
    """
    a = _FATIGUE_ANCHOR[_family(props.name)]
    S = max(load_fraction_MBL, 1e-3)
    return (a["N_ref"] * (D_over_d / a["DoverD_ref"]) ** a["p_DoverD"]
            * (a["S_ref"] / S) ** a["q_stress"])


def fatigue_and_opex(props: CableProps, MBL_N: float, *, D_over_d: float = 25.0,
                     working_tension_N: float = 1800.0,
                     bends_per_year: float = 7000.0,
                     cable_length_m: float = 120.0,
                     cost_eur_per_m: float | None = None,
                     co2_kg_per_kg: float | None = None,
                     uv_abrasion_life_yr: float = 8.0) -> FatigueResult:
    """Replacement interval + OPEX/CO2 line from bending fatigue and UV limits."""
    fam = _family(props.name)
    if cost_eur_per_m is None:
        cost_eur_per_m = 4.0 if fam == "steel" else 12.0      # rope catalogues
    if co2_kg_per_kg is None:
        co2_kg_per_kg = 2.0 if fam == "steel" else 6.0        # steel vs HMPE fibre
    S = working_tension_N / MBL_N
    N = bend_cycles_to_failure(props, D_over_d, S, MBL_N)
    life_fatigue = N / bends_per_year
    life = min(life_fatigue, uv_abrasion_life_yr)
    mass_kg = props.mass_per_m_kg * cable_length_m
    cost = cost_eur_per_m * cable_length_m
    return FatigueResult(material=props.name, D_over_d=D_over_d,
                         load_fraction_MBL=S, bend_cycles_to_failure=N,
                         bends_per_year=bends_per_year, fatigue_life_yr=life_fatigue,
                         practical_life_yr=life, opex_eur_per_yr=cost / life,
                         co2_kg_per_yr=mass_kg * co2_kg_per_kg / life)


# ===========================================================================
# 3. Aeroelastic stability
# ===========================================================================

@dataclass
class AeroResult:
    material: str
    modal_freqs_hz: np.ndarray
    lockin_wind_m_s: np.ndarray       # vortex lock-in wind per mode
    galloping_onset_iced_m_s: float
    site: str
    wind_min_max_m_s: tuple[float, float]
    viv_in_wind_band: bool
    galloping_in_wind_band: bool


def vortex_lockin_winds(modal_freqs_hz: np.ndarray,
                        d: float = CABLE_DIAMETER_M) -> np.ndarray:
    """Wind speed at which vortex shedding locks onto each mode: U = f d / St."""
    return modal_freqs_hz * d / STROUHAL


def den_hartog_galloping_onset(props: CableProps, f1_hz: float, *,
                               d: float = CABLE_DIAMETER_M,
                               zeta: float = 0.005, a_coeff: float = 3.0) -> float:
    """Galloping onset wind for an iced/non-circular section (Den Hartog).

    U_g = 4 m' zeta omega_n / (rho_air d a),  a = -(dCl/dalpha + Cd) > 0.
    A bare circular cable is stable (a<=0 -> no galloping); ``a_coeff`` models a
    representative iced D-section.
    """
    omega_n = 2.0 * math.pi * f1_hz
    return 4.0 * props.mass_per_m_kg * zeta * omega_n / (RHO_AIR * d * a_coeff)


def aeroelastic_check(props: CableProps, T_target: float = 1800.0,
                      span_m: float = 50.0, site: str = "Konya_TR",
                      n_modes: int = 8) -> AeroResult:
    from .energy import load_site_meta
    cab = LumpedCable(span=span_m, props=props, T_target=T_target, n_seg=60)
    cab.static_equilibrium()
    freqs = cab.transverse_frequencies(n_modes)
    lockin = vortex_lockin_winds(freqs)
    f1 = freqs[0]
    U_g = den_hartog_galloping_onset(props, f1)
    meta = load_site_meta(site)
    wmin, wmax = float(meta.wind_mean_m_s.min()), float(meta.wind_mean_m_s.max())
    # A long taut cable has a dense modal spectrum; VIV is possible once the
    # lowest lock-in wind falls at or below the site wind (some mode is always
    # near the shedding frequency above that).
    viv = bool(lockin.min() <= wmax)
    return AeroResult(material=props.name, modal_freqs_hz=freqs,
                      lockin_wind_m_s=lockin, galloping_onset_iced_m_s=U_g,
                      site=site, wind_min_max_m_s=(wmin, wmax),
                      viv_in_wind_band=viv,
                      galloping_in_wind_band=bool(U_g <= wmax))
