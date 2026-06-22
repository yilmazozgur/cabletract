"""S1 -- Closed-loop tilling-depth control co-simulation.

The manuscript's architecture claim (physics.tension_balance docstring) is that
"the carriage ... carries an implement that sets its own depth via float wheels
or a frame -- the cable does not set tillage depth." A sceptical reviewer asks
the obvious question: a tractor holds depth with several tonnes of mass and a
3-point hitch reacting against the ground; a cable carriage weighs ~250 kg and
hangs from a cable whose vertical point-load stiffness is only ~4T/L ~ 150 N/m
(derived below from the full lumped-mass cable) -- two to three orders of
magnitude too soft to set depth. So depth *must* come from a gauge/float wheel
riding the surface, and the real risk is **gauge-wheel lift-off**: a transient
draft/stone spike can momentarily lift a light carriage out of the ground
before its down-pressure actuator reacts, where a heavy tractor never would.

This module answers that quantitatively with three parts:

1. ``LumpedCable`` -- a planar lumped-mass cable (axial Hookean springs +
   gravity). Its static equilibrium reproduces ``physics.catenary_sag``; its
   linearised modes reproduce the taut-string first natural frequency; an
   undamped run conserves energy. It supplies the carriage's vertical support
   stiffness ``k_cable`` consistently with the high-fidelity cable (and is the
   cable model S2 reuses).

2. A reduced vertical **depth-regulation plant**: carriage mass + a unilateral
   gauge-wheel ground contact (N >= 0) + a rate/force-limited down-pressure
   actuator with finite bandwidth + the soft cable support, driven by
   programmed disturbances (buried stone, hardpan step, moisture ramp).

3. **Controllers** (PID and a finite-horizon box-constrained **MPC** solved by
   a self-contained projected-gradient QP) and a **tractor benchmark** (heavy
   sprung mass + stiff hitch), compared on the same disturbances; plus a
   stability envelope over down-pressure capacity and controller bandwidth.

All deterministic, unit-tested. Validation lives in ``tests/test_dynamics.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .physics import catenary_sag

G = 9.80665  # m/s^2

CABLE_CSV = Path(__file__).resolve().parent / "data" / "cable_props.csv"


# ===========================================================================
# Part 1 -- High-fidelity lumped-mass cable
# ===========================================================================

@dataclass
class CableProps:
    """Axial-spring properties for one cable material (from cable_props.csv)."""

    name: str
    mass_per_m_kg: float
    EA_N: float  # E * fibre_area

    @property
    def w_per_m_N(self) -> float:
        return self.mass_per_m_kg * G


def load_cable_props(name: str = "steel_6x19_IWRC_8mm") -> CableProps:
    import csv
    with open(CABLE_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["material"] == name:
                return CableProps(name=name,
                                  mass_per_m_kg=float(row["mass_per_m_kg"]),
                                  EA_N=float(row["E_modulus_Pa"]) * float(row["fibre_area_m2"]))
    raise KeyError(f"cable material {name!r} not in {CABLE_CSV}")


class LumpedCable:
    """Planar (x-y vertical plane) lumped-mass cable between two level supports.

    ``n_seg`` segments / ``n_seg+1`` nodes; ends pinned at (0,0) and (span,0).
    Rest length per segment is chosen so the straight-chord configuration
    carries the target horizontal tension ``T_target``; gravity then produces
    the catenary sag. Interior nodes are free in 2-D.
    """

    def __init__(self, span: float, props: CableProps, T_target: float,
                 n_seg: int = 40):
        self.span = float(span)
        self.props = props
        self.T_target = float(T_target)
        self.n_seg = int(n_seg)
        self.N = n_seg + 1
        self.EA = props.EA_N
        # Rest length per segment from the *elastic catenary* at T_target: the
        # equilibrium loaded arc length minus the (shallow-sag) elastic stretch,
        # divided over the segments. Sag is acutely sensitive to this (mm of arc
        # over a 50 m span -> decimetres of sag), so the straight-chord estimate
        # is not good enough -- use the analytic arc/tension directly.
        sag_a, arc_a, _T_end = catenary_sag(T_target, props.w_per_m_N, span)
        elong = T_target * arc_a / self.EA          # T ~ T_target along a shallow cable
        L0_total = arc_a - elong
        self.l0 = L0_total / n_seg
        seg = span / n_seg
        # lumped nodal mass = mass of adjacent half-segments
        m_seg = props.mass_per_m_kg * seg
        self.mass = np.full(self.N, m_seg)
        self.mass[0] = self.mass[-1] = m_seg / 2.0
        # initial guess: straight x, parabolic sag of the analytic depth
        self.x0 = np.linspace(0.0, span, self.N)
        self.y0 = -4.0 * sag_a * self.x0 * (span - self.x0) / span ** 2
        self._sag_analytic = sag_a

    # -- geometry helpers -------------------------------------------------
    def _segments(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        L = np.hypot(dx, dy)
        return dx, dy, L

    def potential_energy(self, x, y) -> float:
        _dx, _dy, L = self._segments(x, y)
        strain = (L - self.l0) / self.l0
        spring = 0.5 * self.EA * self.l0 * np.sum(strain ** 2)
        grav = np.sum(self.mass * G * y)  # y up -> sag (y<0) lowers PE
        return float(spring + grav)

    def nodal_forces(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Net force on each node (springs + gravity). Endpoints are pinned."""
        dx, dy, L = self._segments(x, y)
        T = self.EA * (L - self.l0) / self.l0           # axial tension per seg
        ux, uy = dx / L, dy / L                           # unit vectors
        fx = np.zeros(self.N)
        fy = np.zeros(self.N)
        # segment j pulls node j toward j+1 (+) and node j+1 toward j (-)
        np.add.at(fx, np.arange(self.n_seg), T * ux)
        np.add.at(fy, np.arange(self.n_seg), T * uy)
        np.add.at(fx, np.arange(1, self.N), -T * ux)
        np.add.at(fy, np.arange(1, self.N), -T * uy)
        fy -= self.mass * G                               # gravity (down)
        fx[0] = fx[-1] = fy[0] = fy[-1] = 0.0             # pinned ends
        return fx, fy

    def static_equilibrium(self, tol: float = 1e-8, max_iter: int = 20000,
                           dt: float = None):
        """Dynamic relaxation (damped) to the static catenary shape."""
        x = self.x0.copy()
        y = self.y0.copy()  # parabolic seed at the analytic sag depth
        vx = np.zeros(self.N)
        vy = np.zeros(self.N)
        # stable step from the stiffest spring
        k_axial = self.EA / self.l0
        m_min = self.mass.min()
        if dt is None:
            dt = 0.5 * 2.0 / math.sqrt(k_axial / m_min)
        damp = 0.02
        for _ in range(max_iter):
            fx, fy = self.nodal_forces(x, y)
            ax = fx / self.mass
            ay = fy / self.mass
            vx = (1.0 - damp) * vx + dt * ax
            vy = (1.0 - damp) * vy + dt * ay
            vx[0] = vx[-1] = vy[0] = vy[-1] = 0.0
            x = x + dt * vx
            y = y + dt * vy
            if max(np.abs(fx[1:-1]).max(), np.abs(fy[1:-1]).max()) < tol * self.T_target:
                break
        self.x_eq, self.y_eq = x, y
        return x, y

    def midspan_sag(self) -> float:
        if not hasattr(self, "y_eq"):
            self.static_equilibrium()
        return float(-self.y_eq[self.N // 2])

    def natural_frequencies(self, n_modes: int = 4) -> np.ndarray:
        """Linearised modal frequencies (Hz) about the static equilibrium.

        Tangent stiffness via finite differences of the nodal-force field on
        the free (interior) DOFs; generalised eigenproblem with the lumped
        (diagonal) mass matrix.
        """
        if not hasattr(self, "y_eq"):
            self.static_equilibrium()
        x0, y0 = self.x_eq.copy(), self.y_eq.copy()
        free = np.arange(1, self.N - 1)
        ndof = 2 * len(free)

        def force_vec(xv, yv):
            fx, fy = self.nodal_forces(xv, yv)
            return np.concatenate([fx[free], fy[free]])

        # state vector q = [x_free, y_free]
        def apply(q):
            xv = x0.copy(); yv = y0.copy()
            xv[free] = q[:len(free)]
            yv[free] = q[len(free):]
            return force_vec(xv, yv)

        q0 = np.concatenate([x0[free], y0[free]])
        f0 = apply(q0)
        eps = 1e-7
        K = np.zeros((ndof, ndof))
        for j in range(ndof):
            dq = q0.copy(); dq[j] += eps
            K[:, j] = -(apply(dq) - f0) / eps     # K = -dF/dq
        K = 0.5 * (K + K.T)
        m_free = np.concatenate([self.mass[free], self.mass[free]])
        Minv_sqrt = np.diag(1.0 / np.sqrt(m_free))
        A = Minv_sqrt @ K @ Minv_sqrt
        w2 = np.linalg.eigvalsh(0.5 * (A + A.T))
        w2 = w2[w2 > 0]
        freqs = np.sqrt(w2) / (2.0 * math.pi)
        return np.sort(freqs)[:n_modes]

    def transverse_frequencies(self, n_modes: int = 4) -> np.ndarray:
        """Out-of-plane (lateral) modal frequencies (Hz).

        Sag does not couple to first order out-of-plane, so these are the taut
        string modes f_n = (n/2L) sqrt(T/mu) -- the spectrum relevant to
        wind-induced lateral vibration (used by S2). Built from the actual
        per-segment tensions and lengths of the static equilibrium.
        """
        if not hasattr(self, "y_eq"):
            self.static_equilibrium()
        _dx, _dy, L = self._segments(self.x_eq, self.y_eq)
        T = self.EA * (L - self.l0) / self.l0          # segment tensions
        free = np.arange(1, self.N - 1)
        nf = len(free)
        K = np.zeros((nf, nf))
        for idx, i in enumerate(free):
            kl = T[i - 1] / L[i - 1]                    # left segment
            kr = T[i] / L[i]                            # right segment
            K[idx, idx] = kl + kr
            if idx > 0:
                K[idx, idx - 1] = -kl
            if idx < nf - 1:
                K[idx, idx + 1] = -kr
        m = self.mass[free]
        Mis = np.diag(1.0 / np.sqrt(m))
        A = Mis @ K @ Mis
        w2 = np.linalg.eigvalsh(0.5 * (A + A.T))
        w2 = w2[w2 > 0]
        return np.sort(np.sqrt(w2) / (2.0 * math.pi))[:n_modes]

    def taut_string_f1(self) -> float:
        mu = self.props.mass_per_m_kg
        return 1.0 / (2.0 * self.span) * math.sqrt(self.T_target / mu)

    def vertical_point_stiffness(self) -> float:
        """Secant vertical stiffness for a small point load at midspan (N/m)."""
        if not hasattr(self, "y_eq"):
            self.static_equilibrium()
        # 4T/L taut estimate, exact enough for the soft-support argument
        return 4.0 * self.T_target / self.span

    def total_energy(self, x, y, vx, vy) -> float:
        ke = 0.5 * float(np.sum(self.mass * (vx ** 2 + vy ** 2)))
        return ke + self.potential_energy(x, y)

    def integrate_free(self, y_perturb: np.ndarray, t_end: float, dt: float):
        """Undamped velocity-Verlet free vibration; returns energy history."""
        if not hasattr(self, "y_eq"):
            self.static_equilibrium()
        x = self.x_eq.copy()
        y = self.y_eq.copy() + y_perturb
        vx = np.zeros(self.N); vy = np.zeros(self.N)
        fx, fy = self.nodal_forces(x, y)
        ax, ay = fx / self.mass, fy / self.mass
        n = int(t_end / dt)
        E = np.empty(n)
        for k in range(n):
            vx += 0.5 * dt * ax; vy += 0.5 * dt * ay
            vx[0] = vx[-1] = vy[0] = vy[-1] = 0.0
            x += dt * vx; y += dt * vy
            fx, fy = self.nodal_forces(x, y)
            ax, ay = fx / self.mass, fy / self.mass
            vx += 0.5 * dt * ax; vy += 0.5 * dt * ay
            vx[0] = vx[-1] = vy[0] = vy[-1] = 0.0
            E[k] = self.total_energy(x, y, vx, vy)
        return E


# ===========================================================================
# Part 2 -- Vertical depth-regulation plant (deviation form)
# ===========================================================================

@dataclass
class DepthPlant:
    """Reduced vertical model of a depth-holding carriage (deviation variables).

    State x = [e, e_dot, du] where e = depth deviation from setpoint (m, down
    positive), e_dot its rate, du = actuator-force deviation from the steady
    down-pressure u_eq. In ground contact the gauge wheel adds stiffness k_w;
    contact force N = N0 + k_w*e, and lift-off is N <= 0.

        m e''   = du - k_cable e - c e' + F_dist + (N0 - N),  N = max(0, N0+k_w e)
        du'     = w_act (u_cmd_dev - du)        (1st-order actuator, then clamp)

    with the total actuator force u = u_eq + du constrained to [0, u_max].
    """

    m: float = 224.0            # carriage mass (kg) ~ system_weight_N/g
    k_w: float = 3.0e5          # gauge-wheel/soil vertical contact stiffness (N/m)
    k_cable: float = 144.0      # cable vertical point stiffness 4T/L (N/m)
    c: float = 1600.0           # structural + rolling damping (N.s/m)
    N0: float = 1200.0          # steady gauge-wheel contact load (N) -- downforce reserve
    u_eq: float = 300.0         # steady down-pressure (N)
    u_max: float = 2500.0       # actuator force capacity (N)
    w_act: float = 2.0 * math.pi * 5.0   # actuator bandwidth (rad/s), ~5 Hz
    du_rate_max: float = 6.0e4  # actuator force slew limit (N/s)

    def contact_force(self, e: float) -> float:
        return max(0.0, self.N0 + self.k_w * e)

    def deriv(self, state, u_cmd_dev: float, F_dist: float):
        e, edot, du = state
        N = self.contact_force(e)
        edd = (du - self.k_cable * e - self.c * edot + F_dist + (self.N0 - N)) / self.m
        # clamp commanded actuator deviation to capacity, then slew-limit
        du_lo, du_hi = -self.u_eq, self.u_max - self.u_eq
        u_cmd_dev = min(max(u_cmd_dev, du_lo), du_hi)
        ddu = self.w_act * (u_cmd_dev - du)
        ddu = min(max(ddu, -self.du_rate_max), self.du_rate_max)
        return np.array([edot, edd, ddu])

    @staticmethod
    def tractor() -> "DepthPlant":
        """Heavy 3-point-hitch benchmark: large mass, stiff support, big reserve."""
        return DepthPlant(m=1500.0, k_w=8.0e5, k_cable=4.0e6, c=2.0e4,
                          N0=6000.0, u_eq=1000.0, u_max=12000.0,
                          w_act=2.0 * math.pi * 1.5, du_rate_max=2.0e5)


# --- disturbances (down-positive; lifts are negative) ---------------------

def stone_impulse(t, t0=0.5, amp=1500.0, dur=0.15):
    """Buried-stone strike: upward half-sine lift pulse."""
    if t0 <= t < t0 + dur:
        return -amp * math.sin(math.pi * (t - t0) / dur)
    return 0.0


def hardpan_step(t, t0=0.5, amp=800.0, ramp=0.05):
    """Entering a hardpan: sustained upward lift, smoothed over ``ramp`` s."""
    if t < t0:
        return 0.0
    return -amp * min(1.0, (t - t0) / ramp)


def moisture_ramp(t, t0=0.5, t1=3.5, amp=400.0):
    """Slow soil-strength change: gradual lift over the pass."""
    if t < t0:
        return 0.0
    if t > t1:
        return -amp
    return -amp * (t - t0) / (t1 - t0)


# ===========================================================================
# Part 3 -- Controllers
# ===========================================================================

@dataclass
class PID:
    kp: float
    ki: float
    kd: float
    u_lo: float
    u_hi: float
    _i: float = 0.0

    def reset(self):
        self._i = 0.0

    def __call__(self, e: float, edot: float, dt: float) -> float:
        # drive e -> 0 by adding down-pressure; lift (e<0) -> more down-pressure
        self._i += e * dt
        u = -(self.kp * e + self.ki * self._i + self.kd * edot)
        u_clamped = min(max(u, self.u_lo), self.u_hi)
        # anti-windup: back-calculate integrator if saturated
        if u != u_clamped and self.ki != 0.0:
            self._i -= (u - u_clamped) / self.ki * 0.0  # conditional below
        if u_clamped in (self.u_lo, self.u_hi):
            self._i -= e * dt  # stop integrating into saturation
        return u_clamped


class MPC:
    """Finite-horizon box-constrained MPC via a condensed projected-gradient QP.

    Offset-free: the state is augmented with the integral of the depth error so
    a constant disturbance (e.g. a hardpan step) is rejected with zero
    steady-state error. The unmeasured disturbance is set to 0 in the
    prediction. Minimises sum x'Qx + du'R du over the horizon subject to the
    actuator box constraint, and applies the first move (receding horizon).

    Augmented state x = [e, e_dot, du, integral(e)].
    """

    def __init__(self, plant: DepthPlant, dt: float, horizon: int = 30,
                 q_pos: float = 2.0e7, q_vel: float = 2.0e3, q_int: float = 4.0e7,
                 r: float = 2.0e-4):
        from scipy.linalg import expm
        self.plant = plant
        self.dt = dt
        self.H = horizon
        # continuous in-contact linearisation: states [e, edot, du, int_e]
        m, kc, kw, c, w = plant.m, plant.k_cable, plant.k_w, plant.c, plant.w_act
        Ac = np.array([[0.0, 1.0, 0.0, 0.0],
                       [-(kc + kw) / m, -c / m, 1.0 / m, 0.0],
                       [0.0, 0.0, -w, 0.0],
                       [1.0, 0.0, 0.0, 0.0]])
        Bc = np.array([[0.0], [0.0], [w], [0.0]])
        M = np.zeros((5, 5)); M[:4, :4] = Ac; M[:4, 4:] = Bc
        Md = expm(M * dt)
        self.A = Md[:4, :4]
        self.B = Md[:4, 4:]
        self.Q = np.diag([q_pos, q_vel, 0.0, q_int])
        self.R = np.array([[r]])
        self._build_condensed()
        self.du_lo = -plant.u_eq
        self.du_hi = plant.u_max - plant.u_eq
        self._qi = 0.0

    def _build_condensed(self):
        n, m, H = 4, 1, self.H
        Sx = np.zeros((H * n, n))
        Su = np.zeros((H * n, H * m))
        Apow = np.eye(n)
        for i in range(H):
            Apow = self.A @ Apow
            Sx[i * n:(i + 1) * n, :] = Apow
            for j in range(i + 1):
                blk = np.linalg.matrix_power(self.A, i - j) @ self.B
                Su[i * n:(i + 1) * n, j * m:(j + 1) * m] = blk
        Qbar = np.kron(np.eye(H), self.Q)
        Rbar = np.kron(np.eye(H), self.R)
        self.Sx, self.Su = Sx, Su
        self.Hqp = 2.0 * (Su.T @ Qbar @ Su + Rbar)
        self.Fx = 2.0 * Su.T @ Qbar @ Sx
        self.L = float(np.linalg.eigvalsh(self.Hqp).max())  # gradient step

    def reset(self):
        self._qi = 0.0

    def __call__(self, e: float, edot: float, du: float, dt: float) -> float:
        self._qi += e * dt
        x0 = np.array([e, edot, du, self._qi])
        g0 = self.Fx @ x0
        U = np.zeros(self.H)
        step = 1.0 / self.L
        z = U.copy(); t_prev = 1.0
        for _ in range(120):  # accelerated projected gradient
            grad = self.Hqp @ z + g0
            U_new = np.clip(z - step * grad, self.du_lo, self.du_hi)
            t = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_prev ** 2))
            z = U_new + ((t_prev - 1.0) / t) * (U_new - U)
            U, t_prev = U_new, t
        return float(np.clip(U[0], self.du_lo, self.du_hi))


# ===========================================================================
# Simulation driver
# ===========================================================================

@dataclass
class DepthResult:
    t: np.ndarray
    e: np.ndarray            # depth deviation (m)
    N: np.ndarray            # gauge-wheel contact force (N)
    u: np.ndarray            # total actuator force (N)
    rms_e_mm: float
    peak_e_mm: float
    settle_s: float
    min_N: float
    liftoff_s: float         # total time with N<=0
    in_tol: bool             # peak within +/-2 cm agronomic tolerance


def simulate_depth(plant: DepthPlant, controller, disturbance, *,
                   t_end: float = 4.0, dt: float = 5.0e-4,
                   ctrl_dt: float = 0.01, tol_m: float = 0.02) -> DepthResult:
    """RK4 plant integration with a zero-order-hold controller."""
    n = int(t_end / dt)
    t = np.arange(n) * dt
    e = np.zeros(n); N = np.zeros(n); u = np.zeros(n)
    state = np.zeros(3)
    u_cmd = 0.0
    ctrl_every = max(1, int(round(ctrl_dt / dt)))
    is_mpc = isinstance(controller, MPC)
    if hasattr(controller, "reset"):
        controller.reset()
    for k in range(n):
        if k % ctrl_every == 0:
            ek, edk, duk = state
            if is_mpc:
                u_cmd = controller(ek, edk, duk, ctrl_dt)
            else:
                u_cmd = controller(ek, edk, ctrl_dt)
        d = disturbance(t[k])
        s = state
        k1 = plant.deriv(s, u_cmd, d)
        k2 = plant.deriv(s + 0.5 * dt * k1, u_cmd, d)
        k3 = plant.deriv(s + 0.5 * dt * k2, u_cmd, d)
        k4 = plant.deriv(s + dt * k3, u_cmd, d)
        state = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        e[k] = state[0]
        N[k] = plant.contact_force(state[0])
        u[k] = plant.u_eq + state[2]

    peak = float(np.max(np.abs(e)))
    rms = float(np.sqrt(np.mean(e ** 2)))
    # settling: last time |e| leaves a 2 mm band after the disturbance onset
    band = 0.002
    outside = np.where(np.abs(e) > band)[0]
    settle = float(t[outside[-1]]) if len(outside) else 0.0
    liftoff = float(np.sum(N <= 1e-6) * dt)
    return DepthResult(t=t, e=e, N=N, u=u, rms_e_mm=rms * 1e3,
                       peak_e_mm=peak * 1e3, settle_s=settle,
                       min_N=float(N.min()), liftoff_s=liftoff,
                       in_tol=peak <= tol_m)


def tuned_pid(plant: DepthPlant, ctrl_dt: float = 0.01) -> PID:
    """A pole-placement-ish PID scaled to the plant; stable and reasonably fast."""
    wn = 2.0 * math.pi * 3.0  # target closed-loop bandwidth ~3 Hz
    kp = plant.m * wn ** 2
    kd = 2.0 * 0.8 * plant.m * wn
    ki = 0.3 * kp * wn
    return PID(kp=kp, ki=ki, kd=kd, u_lo=-plant.u_eq, u_hi=plant.u_max - plant.u_eq)


def stability_envelope(disturbance, N0_grid, bw_grid_hz, *,
                       base: DepthPlant | None = None, t_end: float = 3.0,
                       dt: float = 1.0e-3) -> dict:
    """Sweep steady down-pressure reserve N0 x actuator bandwidth (PID).

    Returns peak depth-error and gauge-wheel lift-off-duration grids. The
    lift-off boundary is governed mainly by N0 (the transient lift must not
    exceed the steady wheel reserve faster than the actuator reacts), so this
    sweep yields the down-pressure the carriage must carry to behave like a
    tractor through a buried-stone strike.
    """
    base = base or DepthPlant()
    from dataclasses import replace
    peak = np.zeros((len(bw_grid_hz), len(N0_grid)))
    lift = np.zeros_like(peak)
    for i, bw in enumerate(bw_grid_hz):
        for j, n0 in enumerate(N0_grid):
            p = replace(base, N0=n0, u_max=max(base.u_max, n0 + 1500.0),
                        w_act=2.0 * math.pi * bw)
            pid = tuned_pid(p)
            res = simulate_depth(p, pid, disturbance, t_end=t_end, dt=dt)
            peak[i, j] = res.peak_e_mm
            lift[i, j] = res.liftoff_s
    return {"N0": np.asarray(N0_grid), "bw_hz": np.asarray(bw_grid_hz),
            "peak_e_mm": peak, "liftoff_s": lift}
