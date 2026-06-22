"""S3 baseline -- McKyes-Godwin / Reece narrow-tine soil-cutting draft.

This is the analytic sanity bound the DEM (``tillage_dem``) is checked against,
and an independent cross-check on the ASABE D497 draft the manuscript uses.

Model. The Reece (1965) fundamental earthmoving equation gives the 2-D
plane-strain soil-cutting force per unit tine width for a planar failure wedge
at angle ``beta`` to the horizontal:

    H_2D(beta) / w = gamma g d^2 N_gamma + c d N_c + q d N_q

with the dimensionless factors (rake angle alpha, soil friction phi, soil-metal
friction delta):

    den       = cos(alpha+delta) + sin(alpha+delta) cot(beta+phi)
    N_gamma   = (cot alpha + cot beta) / (2 den)
    N_c       = (1 + cot beta cot(beta+phi)) / den
    N_q       = (cot alpha + cot beta) / den

The operative ``beta`` is the one that *minimises* the cutting force (the soil
fails on the easiest surface); we minimise numerically.

Narrow tines fail soil sideways as well -- the Godwin & Spoor (1977) "side
crescents". We add them by extending the effective width by the forward rupture
reach r = d cot(beta), scaled by a crescent factor ``kappa``:

    H_total = (H_2D / w) * (w + kappa * d * cot beta)

This reproduces the central Godwin-Spoor result: a *wide* tine (w >> d) gives
draft proportional to d^2 with constant draft-per-width, while a *narrow* tine
(w << d) approaches draft proportional to d^3 -- the super-linear depth
penalty that makes shallow-narrow passes attractive.

References
----------
- Reece, A.R. (1965) "The fundamental equation of earthmoving mechanics",
  Proc. IMechE 179(6).
- Godwin, R.J. & Spoor, G. (1977) "Soil failure with narrow tines", J. Agric.
  Eng. Res. 22(3):213-228 -- side-crescent / narrow-tine behaviour.
- McKyes, E. (1985) "Soil Cutting and Tillage", Elsevier -- consolidated factors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.80665


@dataclass(frozen=True)
class SoilCuttingParams:
    gamma_kg_m3: float = 1500.0   # bulk density
    cohesion_kPa: float = 5.0     # soil cohesion
    phi_deg: float = 30.0         # internal friction angle
    delta_deg: float = 20.0       # soil-metal friction angle
    surcharge_kPa: float = 0.0    # vertical surface surcharge q
    kappa_side: float = 0.6       # Godwin-Spoor crescent factor


def _factors(alpha: float, phi: float, delta: float, beta: float):
    cot = lambda x: 1.0 / math.tan(x)
    den = math.cos(alpha + delta) + math.sin(alpha + delta) * cot(beta + phi)
    if den <= 1e-9:
        return None
    N_gamma = (cot(alpha) + cot(beta)) / (2.0 * den)
    N_c = (1.0 + cot(beta) * cot(beta + phi)) / den
    N_q = (cot(alpha) + cot(beta)) / den
    return N_gamma, N_c, N_q


def _draft_per_width(beta, alpha, phi, delta, gamma, c, q, d):
    f = _factors(alpha, phi, delta, beta)
    if f is None:
        return math.inf
    N_gamma, N_c, N_q = f
    if min(N_gamma, N_c, N_q) < 0:
        return math.inf
    return gamma * G * d * d * N_gamma + c * d * N_c + q * d * N_q


@dataclass
class CuttingResult:
    draft_N: float
    draft_per_width_N_m: float
    beta_deg: float
    forward_reach_m: float
    N_gamma: float
    N_c: float
    N_q: float


def mckyes_godwin_draft(depth_m: float, width_m: float,
                        soil: SoilCuttingParams | None = None,
                        rake_deg: float = 25.0) -> CuttingResult:
    """Total horizontal draft (N) for a narrow tine, with side crescents."""
    s = soil or SoilCuttingParams()
    alpha = math.radians(rake_deg)
    phi = math.radians(s.phi_deg)
    delta = math.radians(s.delta_deg)
    c = s.cohesion_kPa * 1e3
    q = s.surcharge_kPa * 1e3
    gamma = s.gamma_kg_m3

    # minimise draft-per-width over the failure angle beta
    betas = np.radians(np.linspace(5.0, 80.0, 400))
    vals = [_draft_per_width(b, alpha, phi, delta, gamma, c, q, depth_m) for b in betas]
    j = int(np.argmin(vals))
    beta = float(betas[j])
    per_w = float(vals[j])
    N_gamma, N_c, N_q = _factors(alpha, phi, delta, beta)

    reach = depth_m / math.tan(beta)                      # forward rupture reach
    eff_width = width_m + s.kappa_side * reach            # + side crescents
    draft = per_w * eff_width
    return CuttingResult(draft_N=draft, draft_per_width_N_m=per_w,
                         beta_deg=math.degrees(beta), forward_reach_m=reach,
                         N_gamma=N_gamma, N_c=N_c, N_q=N_q)


def depth_scaling_exponent(width_m: float, soil: SoilCuttingParams | None = None,
                           rake_deg: float = 25.0,
                           depths=(0.10, 0.40)) -> float:
    """Local exponent n in draft ~ depth^n between two depths (2 -> wide, ->3 narrow)."""
    d0, d1 = depths
    h0 = mckyes_godwin_draft(d0, width_m, soil, rake_deg).draft_N
    h1 = mckyes_godwin_draft(d1, width_m, soil, rake_deg).draft_N
    return math.log(h1 / h0) / math.log(d1 / d0)
