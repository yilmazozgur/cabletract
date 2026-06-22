"""S7 -- Helical-auger lateral capacity from a nonlinear p--y Winkler model.

Section 5.4 of the manuscript currently quotes *two* per-auger lateral
capacities from the literature and lets the reader pick: ~400 N (Khand 2024,
conservative loose-sand interpretation of a 4-pile raft test) and ~2 kN
(Magnum Piering datasheet, medium-dense fixed-head). That "have-it-both-ways"
gap is a genuine weak spot. This module replaces it with *one* internally
derived per-auger nominal, computed from the geotechnical standard for a
laterally loaded pile -- a **nonlinear p--y (Winkler) beam-column** -- plus a
3x3 group-efficiency factor, with a stated safety factor and a sensitivity
band that should *bracket* (not cherry-pick) the cited literature range.

Physics
-------
Euler-Bernoulli beam-column on a nonlinear elastic foundation,

    EI y''''(z) = -p(y, z),                                   (1)

with z measured downward from the ground line, y the lateral deflection, and
p the distributed soil reaction (N/m) given by the **API RP 2A / Reese sand
p--y curve**,

    p(y, z) = A * p_u(z) * tanh( k * z * y / (A * p_u(z)) ),   (2)

where p_u(z) = min(p_us, p_ud) is the ultimate soil resistance per unit length
(shallow wedge vs deep flow), A = max(3 - 0.8 z/b, 0.9) is the static
depth-reduction factor, k is the initial modulus of subgrade reaction
(N/m^3, density dependent), and b the shaft diameter. The initial tangent of
(2) at y=0 is dp/dy = k*z, i.e. a subgrade modulus that grows linearly with
depth -- the standard "n_h z" sand model.

The ultimate-resistance coefficients C1, C2, C3 follow Murchison & O'Neill
(1984) as adopted by API RP 2A-WSD:

    p_us = (C1 z + C2 b) gamma' z       (shallow wedge)
    p_ud =  C3 b gamma' z               (deep flow-around)

We solve the nonlinear BVP (1)-(2) by finite differences + Newton iteration
(dense; the pile is short, ~10^2 nodes), for both *free-head* and *fixed-head*
conditions, and define the **serviceability lateral capacity** as the head
load H at which the ground-line deflection reaches the IBC 1-inch
(25.4 mm) limit. Group action (3x3 cluster) is handled with row
**p-multipliers** (Mokwa 1999; Reese & Van Impe 2011); the group efficiency
is the mean multiplier, and the derived per-auger *working* nominal applies a
stated safety factor to the group-average serviceability capacity.

Verification (see ``tests/test_anchor_py.py``)
----------------------------------------------
With a *constant* foundation modulus k_f (N/m^2) and a long pile, the numeric
free-end head deflection reproduces the closed-form semi-infinite
beam-on-elastic-foundation result

    y0 = H / (2 beta^3 EI),   beta = (k_f / (4 EI))^{1/4},

which is the exact solver check requested in the plan; and the API ultimate
capacities bracket the cited Khand/Magnum literature range.

References
----------
- Murchison, J.M. & O'Neill, M.W. (1984) "Evaluation of p--y relationships in
  cohesionless soils", ASCE Analysis and Design of Pile Foundations, 174-191.
- API RP 2A-WSD (2014) Recommended Practice for Planning, Designing and
  Constructing Fixed Offshore Platforms, sec. 6.8.
- Reese, L.C. & Van Impe, W.F. (2011) "Single Piles and Pile Groups Under
  Lateral Loading", 2nd ed., CRC Press -- p-multipliers, characteristic length.
- Mokwa, R.L. (1999) "Investigation of the Resistance of Pile Caps to Lateral
  Loading", PhD thesis, Virginia Tech -- group p-multipliers.
- Hetenyi, M. (1946) "Beams on Elastic Foundation", Univ. Michigan Press --
  closed-form semi-infinite-beam solution used for verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

G = 9.80665  # m/s^2

IBC_DEFLECTION_LIMIT_M = 0.0254  # 1 inch serviceability head deflection (IBC 2021)


# ---------------------------------------------------------------------------
# 1. Soil and pile descriptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SandProfile:
    """A cohesionless soil profile for the API/Reese sand p--y curve.

    ``k_subgrade_Npm3`` is the initial modulus of subgrade reaction (the API
    "n_h", N/m^3) for sand *above* the water table, taken from the API RP 2A
    recommended values keyed to relative density / friction angle.
    """

    name: str
    phi_deg: float          # effective friction angle (deg)
    gamma_eff_Npm3: float   # effective unit weight (N/m^3)
    k_subgrade_Npm3: float  # initial subgrade modulus n_h (N/m^3)


# API RP 2A recommended n_h for sand above the water table:
#   loose  (Dr~35%, phi~30 deg):  ~5.4 MN/m^3  (20 lb/in^3)
#   medium-dense (Dr~65%, phi~36 deg): ~24.4 MN/m^3 (90 lb/in^3)
LOOSE_SAND = SandProfile("loose", phi_deg=30.0, gamma_eff_Npm3=16.0e3,
                         k_subgrade_Npm3=5.4e6)
MEDIUM_DENSE_SAND = SandProfile("medium-dense", phi_deg=36.0, gamma_eff_Npm3=18.0e3,
                                k_subgrade_Npm3=24.4e6)


@dataclass(frozen=True)
class PileSection:
    """A single helical-auger shaft modelled as a laterally loaded pile.

    Lateral resistance is carried by the *shaft* (the helix plates contribute
    mainly to axial pull-out/bearing, so ignoring them is conservative for the
    lateral check). Default: a 73 mm OD steel tube (2-7/8"), 5.5 mm wall, the
    common small helical-pier shaft, embedded 2.0 m.
    """

    diameter_m: float = 0.073
    wall_m: float = 0.0055
    length_m: float = 2.0
    E_Pa: float = 200.0e9   # structural steel
    n_nodes: int = 201

    @property
    def I_m4(self) -> float:
        od = self.diameter_m
        idia = max(self.diameter_m - 2.0 * self.wall_m, 0.0)
        return math.pi / 64.0 * (od ** 4 - idia ** 4)

    @property
    def EI(self) -> float:
        return self.E_Pa * self.I_m4


# ---------------------------------------------------------------------------
# 2. API / Reese sand p--y curve
# ---------------------------------------------------------------------------

def api_sand_coefficients(phi_deg: float) -> tuple[float, float, float]:
    """Ultimate-resistance coefficients (C1, C2, C3), Murchison & O'Neill (1984).

    Verified against the published API curves: at phi=30 deg this returns
    C2~2.7, C3~29 (both within a few % of the API chart); C1 governs only the
    very shallow term.
    """
    phi = math.radians(phi_deg)
    alpha = phi / 2.0
    beta = math.pi / 4.0 + phi / 2.0
    K0 = 0.4
    Ka = (1.0 - math.sin(phi)) / (1.0 + math.sin(phi))

    tan_b = math.tan(beta)
    tan_a = math.tan(alpha)
    tan_bmphi = math.tan(beta - phi)

    C1 = (tan_b * tan_a
          + K0 * (math.tan(phi) * math.sin(beta) / (math.cos(alpha) * tan_bmphi)
                  + tan_b * (math.tan(phi) * math.sin(beta) - tan_a)))
    C2 = tan_b / tan_bmphi - Ka
    C3 = Ka * (tan_b ** 8 - 1.0) + K0 * math.tan(phi) * tan_b ** 4
    return C1, C2, C3


def p_ultimate(z: np.ndarray, sand: SandProfile, b: float) -> np.ndarray:
    """Ultimate soil resistance per unit length p_u(z) (N/m), API sand."""
    C1, C2, C3 = api_sand_coefficients(sand.phi_deg)
    g = sand.gamma_eff_Npm3
    p_us = (C1 * z + C2 * b) * g * z       # shallow wedge
    p_ud = C3 * b * g * z                  # deep flow-around
    return np.minimum(p_us, p_ud)


def A_static(z: np.ndarray, b: float) -> np.ndarray:
    """Static depth-reduction factor A = max(3 - 0.8 z/b, 0.9)."""
    return np.maximum(3.0 - 0.8 * z / b, 0.9)


def py_resistance(y: np.ndarray, z: np.ndarray, sand: SandProfile,
                  b: float) -> np.ndarray:
    """API sand p--y soil reaction p(y, z) (N/m). Odd in y."""
    pu = p_ultimate(z, sand, b)
    A = A_static(z, b)
    k = sand.k_subgrade_Npm3
    out = np.zeros_like(y, dtype=float)
    m = pu > 1e-9
    arg = np.zeros_like(y)
    arg[m] = (k * z[m] / (A[m] * pu[m])) * y[m]
    out[m] = A[m] * pu[m] * np.tanh(arg[m])
    return out


_TANH_ARG_CLIP = 30.0  # tanh/sech^2 are saturated well before cosh overflows


def py_tangent(y: np.ndarray, z: np.ndarray, sand: SandProfile,
               b: float) -> np.ndarray:
    """d p / d y of the API sand curve (N/m per m). Initial value is k*z."""
    pu = p_ultimate(z, sand, b)
    A = A_static(z, b)
    k = sand.k_subgrade_Npm3
    out = np.zeros_like(y, dtype=float)
    m = pu > 1e-9
    arg = np.zeros_like(y)
    arg[m] = np.clip((k * z[m] / (A[m] * pu[m])) * y[m], -_TANH_ARG_CLIP, _TANH_ARG_CLIP)
    # d/dy [A pu tanh(c y)] = A pu * c * sech^2 = (k z) * sech^2,  c = k z/(A pu)
    out[m] = (k * z[m]) / np.cosh(arg[m]) ** 2
    return out


# ---------------------------------------------------------------------------
# 3. Finite-difference beam-column-on-Winkler-foundation solver
# ---------------------------------------------------------------------------

@dataclass
class PileSolution:
    z: np.ndarray
    y: np.ndarray
    head_deflection_m: float
    H_N: float
    M_head_Nm: float
    head_fixity: str
    converged: bool
    iters: int


def _assemble_linear(section: PileSection, k_tangent: np.ndarray, h: float):
    """Banded EI*D4 operator + soil tangent on the augmented (N+4) system.

    Unknowns are y[-2..N+1] (two ghost nodes each end). Rows 0..N-1 are the
    governing equation at each node; the last four rows impose the boundary
    conditions (filled by the caller).
    """
    N = section.n_nodes
    EI = section.EI
    M = N + 4  # total unknowns incl. ghosts; index map: u[j] = y[j-2]
    A = np.zeros((M, M))
    rhs = np.zeros(M)
    c = EI / h ** 4
    # node i (physical) -> equation row i; unknown index of y[i] is i+2
    for i in range(N):
        r = i
        A[r, i + 0] += c * 1.0     # y[i-2]
        A[r, i + 1] += c * (-4.0)  # y[i-1]
        A[r, i + 2] += c * 6.0     # y[i]
        A[r, i + 3] += c * (-4.0)  # y[i+1]
        A[r, i + 4] += c * 1.0     # y[i+2]
        A[r, i + 2] += k_tangent[i]  # soil tangent on the diagonal (linearized)
    return A, rhs


def _apply_bcs(A, rhs, section: PileSection, h: float, H: float, M_head: float,
               head_fixity: str):
    """Fill the four BC rows (indices N..N+3) of the augmented system."""
    N = section.n_nodes
    EI = section.EI
    # convenience: unknown index of physical node n is n+2
    def ui(n):
        return n + 2

    rN = N      # top moment / top slope
    rN1 = N + 1  # top shear
    rN2 = N + 2  # bottom moment
    rN3 = N + 3  # bottom shear

    # --- Top (z=0) ---
    if head_fixity == "free":
        # moment M(0) = M_head:  EI*(y[-1]-2y[0]+y[1])/h^2 = M_head
        A[rN, ui(-1)] += EI / h ** 2
        A[rN, ui(0)] += -2.0 * EI / h ** 2
        A[rN, ui(1)] += EI / h ** 2
        rhs[rN] = M_head
    elif head_fixity == "fixed":
        # slope y'(0) = 0:  (y[1]-y[-1])/(2h) = 0
        A[rN, ui(1)] += 1.0 / (2.0 * h)
        A[rN, ui(-1)] += -1.0 / (2.0 * h)
        rhs[rN] = 0.0
    else:
        raise ValueError("head_fixity must be 'free' or 'fixed'")

    # top shear V(0) = EI*y'''(0) = H (applied lateral load at the head)
    # central 3rd derivative at node 0
    A[rN1, ui(-2)] += -EI / (2.0 * h ** 3)
    A[rN1, ui(-1)] += 2.0 * EI / (2.0 * h ** 3)
    A[rN1, ui(1)] += -2.0 * EI / (2.0 * h ** 3)
    A[rN1, ui(2)] += EI / (2.0 * h ** 3)
    rhs[rN1] = H

    # --- Bottom (z=L), free end: M=0, V=0 ---
    A[rN2, ui(N - 2)] += EI / h ** 2
    A[rN2, ui(N - 1)] += -2.0 * EI / h ** 2
    A[rN2, ui(N)] += EI / h ** 2
    rhs[rN2] = 0.0

    A[rN3, ui(N - 3)] += -EI / (2.0 * h ** 3)
    A[rN3, ui(N - 2)] += 2.0 * EI / (2.0 * h ** 3)
    A[rN3, ui(N)] += -2.0 * EI / (2.0 * h ** 3)
    A[rN3, ui(N + 1)] += EI / (2.0 * h ** 3)
    rhs[rN3] = 0.0
    return A, rhs


def solve_pile(section: PileSection, H: float, sand: SandProfile | None = None,
               *, k_const_Npm2: float | None = None, M_head: float = 0.0,
               head_fixity: str = "free", tol: float = 1e-10,
               max_iter: int = 60) -> PileSolution:
    """Solve the laterally loaded pile for head load ``H`` (N).

    Supply either ``sand`` (nonlinear API p--y) or ``k_const_Npm2`` (a constant
    foundation modulus in N/m^2, used for the closed-form verification). Newton
    iteration on the FD residual; converges in a handful of steps.
    """
    if (sand is None) == (k_const_Npm2 is None):
        raise ValueError("supply exactly one of sand= or k_const_Npm2=")
    N = section.n_nodes
    L = section.length_m
    h = L / (N - 1)
    z = np.linspace(0.0, L, N)

    def soil_p(yv):
        if sand is not None:
            return py_resistance(yv, z, sand, section.diameter_m)
        return k_const_Npm2 * yv

    def soil_k(yv):
        if sand is not None:
            return py_tangent(yv, z, sand, section.diameter_m)
        return np.full_like(yv, k_const_Npm2)

    # Newton iteration on the augmented system.
    u = np.zeros(N + 4)  # includes ghosts; physical y = u[2:N+2]
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        y = u[2:N + 2]
        kt = soil_k(y)
        A, _rhs = _assemble_linear(section, kt, h)
        # Residual rows 0..N-1: EI*D4*y + p(y) - 0 ; build with true p(y) (not tangent)
        EI = section.EI
        c = EI / h ** 4
        res = np.zeros(N + 4)
        D4y = (c * (u[0:N] - 4.0 * u[1:N + 1] + 6.0 * u[2:N + 2]
                    - 4.0 * u[3:N + 3] + u[4:N + 4]))
        res[0:N] = D4y + soil_p(y)
        # BC residual: assemble BC rows then compute (A_bc @ u - rhs_bc)
        Abc, rhsbc = _apply_bcs(np.zeros((N + 4, N + 4)), np.zeros(N + 4),
                                section, h, H, M_head, head_fixity)
        res[N:] = Abc[N:] @ u - rhsbc[N:]
        # Tangent matrix: interior rows from _assemble_linear (EI*D4 + diag(kt)),
        # BC rows from Abc.
        J = A.copy()
        J[N:] = Abc[N:]
        try:
            delta = np.linalg.solve(J, -res)
        except np.linalg.LinAlgError:
            break
        u = u + delta
        if not np.all(np.isfinite(u)):
            converged = False
            break
        if np.linalg.norm(delta[2:N + 2], ord=np.inf) < tol * max(1.0, np.linalg.norm(u[2:N + 2], ord=np.inf)):
            converged = True
            break

    y = u[2:N + 2]
    if not np.all(np.isfinite(y)):
        converged = False
    return PileSolution(z=z, y=y, head_deflection_m=float(y[0]), H_N=float(H),
                        M_head_Nm=float(M_head), head_fixity=head_fixity,
                        converged=converged, iters=it)


def lateral_capacity(section: PileSection, sand: SandProfile, *,
                     deflection_limit_m: float = IBC_DEFLECTION_LIMIT_M,
                     head_fixity: str = "free",
                     H_bounds: tuple[float, float] = (1.0, 1.0e5),
                     tol_N: float = 1.0) -> float:
    """Head load (N) that drives the ground-line deflection to the limit.

    Bisection on H. The serviceability limit is reached *below* the ultimate
    (soil-plasticisation) load, so the controlling solves converge cleanly; a
    non-converged or non-finite solve means the pile has run past its ultimate
    capacity and is treated as "deflection exceeds the limit" (move ``hi``
    down). The response is monotone-hardening, so this bisection is sound.
    """
    lo, hi = H_bounds

    def exceeds_limit(H: float) -> bool:
        s = solve_pile(section, H, sand=sand, head_fixity=head_fixity)
        if not s.converged or not math.isfinite(s.head_deflection_m):
            return True  # past ultimate -> effectively infinite deflection
        return s.head_deflection_m >= deflection_limit_m

    if not exceeds_limit(hi):
        return hi  # carries more than the search ceiling
    while hi - lo > tol_N:
        mid = 0.5 * (lo + hi)
        if exceeds_limit(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# 4. 3x3 group p-multipliers and the derived per-auger nominal
# ---------------------------------------------------------------------------

def p_multipliers_3x3(spacing_over_d: float) -> list[float]:
    """Row p-multipliers for a 3x3 cluster (leading, middle, trailing rows).

    Values follow Mokwa (1999) / Reese & Van Impe (2011) for the in-line
    spacing s/d; linearly interpolated between the standard 3d and 5d+ anchors
    and clamped to 1.0 (fully efficient at wide spacing).
    """
    # Standard anchors: at s/d=3, fm=(0.8, 0.4, 0.3); at s/d>=5, fm->1.0.
    lead3, mid3, trail3 = 0.8, 0.4, 0.3
    if spacing_over_d <= 3.0:
        return [lead3, mid3, trail3]
    if spacing_over_d >= 5.0:
        return [1.0, 1.0, 1.0]
    t = (spacing_over_d - 3.0) / (5.0 - 3.0)
    return [lead3 + t * (1.0 - lead3),
            mid3 + t * (1.0 - mid3),
            trail3 + t * (1.0 - trail3)]


@dataclass
class GroupResult:
    single_free_N: float
    single_fixed_N: float
    p_multipliers: list[float]
    group_efficiency: float
    per_auger_group_free_N: float
    per_auger_group_fixed_N: float
    nominal_working_per_auger_N: float
    safety_factor: float
    soil_name: str
    spacing_over_d: float


def group_capacity(section: PileSection, sand: SandProfile, *,
                   spacing_over_d: float = 3.0, safety_factor: float = 1.5,
                   deflection_limit_m: float = IBC_DEFLECTION_LIMIT_M) -> GroupResult:
    """Derive one per-auger working nominal from the single-pile capacities.

    The group efficiency is the mean row p-multiplier; the per-auger *group*
    serviceability capacity is the single-pile value scaled by that mean; the
    *working* nominal divides by the stated safety factor. We report both
    free- and fixed-head single-pile capacities and adopt the conservative
    free-head value for the nominal.
    """
    single_free = lateral_capacity(section, sand,
                                   deflection_limit_m=deflection_limit_m,
                                   head_fixity="free")
    single_fixed = lateral_capacity(section, sand,
                                    deflection_limit_m=deflection_limit_m,
                                    head_fixity="fixed")
    fm = p_multipliers_3x3(spacing_over_d)
    eff = float(np.mean(fm))
    per_free = single_free * eff
    per_fixed = single_fixed * eff
    nominal = per_free / safety_factor
    return GroupResult(
        single_free_N=single_free,
        single_fixed_N=single_fixed,
        p_multipliers=fm,
        group_efficiency=eff,
        per_auger_group_free_N=per_free,
        per_auger_group_fixed_N=per_fixed,
        nominal_working_per_auger_N=nominal,
        safety_factor=safety_factor,
        soil_name=sand.name,
        spacing_over_d=spacing_over_d,
    )


# ---------------------------------------------------------------------------
# 5. Closed-form verification (Hetenyi semi-infinite beam on elastic foundation)
# ---------------------------------------------------------------------------

def winkler_head_deflection_closed_form(H: float, EI: float,
                                        k_const_Npm2: float) -> float:
    """y0 = H / (2 beta^3 EI), beta = (k_f / 4EI)^{1/4}  (free end, long pile)."""
    beta = (k_const_Npm2 / (4.0 * EI)) ** 0.25
    return H / (2.0 * beta ** 3 * EI)


def required_augers_for(draft_P90_N: float, nominal_per_auger_N: float,
                        working_safety_factor: float = 1.15) -> int:
    """Auger count at the working envelope: ceil(SF * T / nominal)."""
    return max(1, math.ceil(working_safety_factor * draft_P90_N / nominal_per_auger_N))
