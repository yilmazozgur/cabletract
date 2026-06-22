"""S3 -- Soil-tool soft-sphere DEM (Hertz-Mindlin contacts, numba).

Genuine discrete-element soil: polydisperse spheres packed under gravity into a
bin, then a rigid blade dragged through them. Contact forces (Hertzian normal
with restitution damping + regularised-Coulomb tangential friction) are summed
on the blade to give the **draft**; particle displacements give the **disturbed
cross-section** (the tilth proxy). This answers two manuscript questions:

  * do the narrow co-designed tools reach the D497 / McKyes-Godwin draft, and
  * is a shallow narrow pass agronomically equivalent (disturbed area) to a deep
    wide one?

It is a real DEM (not a wedge surrogate) but at feasibility scale (10^3-10^4
particles, soft-particle stiffness), not LIGGGHTS research scale. The analytic
McKyes-Godwin wedge (``tillage_mechanics``) is the cross-check.

Contact model (per pair / particle-wall, with effective R*, m*):
    Hertz normal   F_n = (4/3) E* sqrt(R*) delta^1.5  + gamma_n v_n   (>=0, no adhesion)
    Coulomb tang.  F_t = -mu F_n tanh(|v_t|/v_ref) * v_t/|v_t|
Integration: semi-implicit (symplectic) Euler, which is robust with damping.
numba njit kernels with a cell linked-list neighbour search.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if (a and callable(a[0])) is False else a[0]

G = 9.80665


@dataclass
class DEMParams:
    r_min: float = 0.005
    r_max: float = 0.008
    rho_p: float = 2600.0
    E: float = 5.0e6
    nu: float = 0.3
    restitution: float = 0.4
    mu_pp: float = 0.5      # particle-particle friction
    mu_wall: float = 0.4    # particle-wall friction
    mu_tool: float = 0.35   # particle-tool friction
    v_ref: float = 0.01     # friction regularisation velocity (m/s)
    dt: float = 5.0e-5


# ---------------------------------------------------------------------------
# numba kernels
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _build_cells(pos, n, Lx, Ly, Lz, cs, ncx, ncy, ncz, head, nxt):
    for c in range(head.shape[0]):
        head[c] = -1
    for i in range(n):
        ix = int(pos[i, 0] / cs)
        iy = int(pos[i, 1] / cs)
        iz = int(pos[i, 2] / cs)
        if ix < 0: ix = 0
        if iy < 0: iy = 0
        if iz < 0: iz = 0
        if ix >= ncx: ix = ncx - 1
        if iy >= ncy: iy = ncy - 1
        if iz >= ncz: iz = ncz - 1
        c = ix + iy * ncx + iz * ncx * ncy
        nxt[i] = head[c]
        head[c] = i


@njit(cache=True, fastmath=True)
def _pair_force(i, j, pos, vel, rad, mass, force, Estar, mu, v_ref, gamma_coef):
    dx = pos[i, 0] - pos[j, 0]
    dy = pos[i, 1] - pos[j, 1]
    dz = pos[i, 2] - pos[j, 2]
    dist2 = dx * dx + dy * dy + dz * dz
    rsum = rad[i] + rad[j]
    if dist2 >= rsum * rsum or dist2 < 1e-18:
        return
    dist = math.sqrt(dist2)
    delta = rsum - dist
    nx, ny, nz = dx / dist, dy / dist, dz / dist
    Rstar = rad[i] * rad[j] / rsum
    mstar = mass[i] * mass[j] / (mass[i] + mass[j])
    Sn = 2.0 * Estar * math.sqrt(Rstar * delta)
    Fn_el = (4.0 / 3.0) * Estar * math.sqrt(Rstar) * delta ** 1.5
    # relative velocity, normal component (positive = approaching)
    rvx = vel[i, 0] - vel[j, 0]
    rvy = vel[i, 1] - vel[j, 1]
    rvz = vel[i, 2] - vel[j, 2]
    vn = rvx * nx + rvy * ny + rvz * nz
    Fn = Fn_el - gamma_coef * math.sqrt(Sn * mstar) * vn
    if Fn < 0.0:
        Fn = 0.0
    # tangential velocity
    vtx = rvx - vn * nx
    vty = rvy - vn * ny
    vtz = rvz - vn * nz
    vt = math.sqrt(vtx * vtx + vty * vty + vtz * vtz)
    ftx = fty = ftz = 0.0
    if vt > 1e-12:
        ft = mu * Fn * math.tanh(vt / v_ref)
        ftx = -ft * vtx / vt
        fty = -ft * vty / vt
        ftz = -ft * vtz / vt
    fx = Fn * nx + ftx
    fy = Fn * ny + fty
    fz = Fn * nz + ftz
    force[i, 0] += fx; force[i, 1] += fy; force[i, 2] += fz
    force[j, 0] -= fx; force[j, 1] -= fy; force[j, 2] -= fz


@njit(cache=True, fastmath=True)
def _wall_force(i, pos, vel, rad, mass, force, Estar, mu, v_ref, gamma_coef,
                Lx, Ly, Lz, use_side_walls):
    # six axis-aligned walls; outward normal points into the domain
    for axis in range(3):
        for side in range(2):
            if axis == 0 and use_side_walls == 0:
                continue
            if axis == 1 and use_side_walls == 0:
                continue
            if axis == 2 and side == 1:
                continue  # open top
            if axis == 0:
                wallpos = 0.0 if side == 0 else Lx
                coord = pos[i, 0]
            elif axis == 1:
                wallpos = 0.0 if side == 0 else Ly
                coord = pos[i, 1]
            else:
                wallpos = 0.0 if side == 0 else Lz
                coord = pos[i, 2]
            if side == 0:
                pen = rad[i] - (coord - wallpos)
                ndir = 1.0
            else:
                pen = rad[i] - (wallpos - coord)
                ndir = -1.0
            if pen <= 0.0:
                continue
            nx = ny = nz = 0.0
            if axis == 0:
                nx = ndir
            elif axis == 1:
                ny = ndir
            else:
                nz = ndir
            Rstar = rad[i]
            mstar = mass[i]
            Sn = 2.0 * Estar * math.sqrt(Rstar * pen)
            Fn_el = (4.0 / 3.0) * Estar * math.sqrt(Rstar) * pen ** 1.5
            vn = vel[i, 0] * nx + vel[i, 1] * ny + vel[i, 2] * nz
            Fn = Fn_el - gamma_coef * math.sqrt(Sn * mstar) * vn
            if Fn < 0.0:
                Fn = 0.0
            vtx = vel[i, 0] - vn * nx
            vty = vel[i, 1] - vn * ny
            vtz = vel[i, 2] - vn * nz
            vt = math.sqrt(vtx * vtx + vty * vty + vtz * vtz)
            ftx = fty = ftz = 0.0
            if vt > 1e-12:
                ft = mu * Fn * math.tanh(vt / v_ref)
                ftx = -ft * vtx / vt; fty = -ft * vty / vt; ftz = -ft * vtz / vt
            force[i, 0] += Fn * nx + ftx
            force[i, 1] += Fn * ny + fty
            force[i, 2] += Fn * nz + ftz


@njit(cache=True, fastmath=True)
def _tool_force(i, pos, vel, rad, mass, force, Estar, mu, v_ref, gamma_coef,
                Xtool, yc, w, depth, sin_a, cos_a, v_tool, tool_acc):
    # inclined flat blade: cutting edge at (Xtool, *, 0), face normal n=(sin_a,0,cos_a)
    if pos[i, 1] < yc - 0.5 * w or pos[i, 1] > yc + 0.5 * w:
        return
    if pos[i, 2] < 0.0 or pos[i, 2] > depth:
        return
    s = (pos[i, 0] - Xtool) * sin_a + pos[i, 2] * cos_a
    if s <= 0.0 or s >= rad[i]:
        return
    pen = rad[i] - s
    nx, nz = sin_a, cos_a
    Rstar = rad[i]; mstar = mass[i]
    Sn = 2.0 * Estar * math.sqrt(Rstar * pen)
    Fn_el = (4.0 / 3.0) * Estar * math.sqrt(Rstar) * pen ** 1.5
    # relative velocity (particle - tool); tool moves +x at v_tool
    rvx = vel[i, 0] - v_tool
    rvy = vel[i, 1]
    rvz = vel[i, 2]
    vn = rvx * nx + rvz * nz
    Fn = Fn_el - gamma_coef * math.sqrt(Sn * mstar) * vn
    if Fn < 0.0:
        Fn = 0.0
    vtx = rvx - vn * nx
    vty = rvy
    vtz = rvz - vn * nz
    vt = math.sqrt(vtx * vtx + vty * vty + vtz * vtz)
    ftx = fty = ftz = 0.0
    if vt > 1e-12:
        ft = mu * Fn * math.tanh(vt / v_ref)
        ftx = -ft * vtx / vt; fty = -ft * vty / vt; ftz = -ft * vtz / vt
    fx = Fn * nx + ftx
    fy = fty
    fz = Fn * nz + ftz
    force[i, 0] += fx; force[i, 1] += fy; force[i, 2] += fz
    # reaction on tool = -f ; accumulate tool force components
    tool_acc[0] -= fx
    tool_acc[1] -= fy
    tool_acc[2] -= fz


@njit(cache=True, fastmath=True)
def _compute_all_forces(pos, vel, rad, mass, force, n, Lx, Ly, Lz, cs,
                        ncx, ncy, ncz, head, nxt, Estar, mu_pp, mu_wall,
                        v_ref, gamma_coef, g, use_side_walls,
                        tool_on, Xtool, yc, w, depth, sin_a, cos_a, v_tool,
                        mu_tool, tool_acc):
    for i in range(n):
        force[i, 0] = 0.0
        force[i, 1] = 0.0
        force[i, 2] = -mass[i] * g
    tool_acc[0] = 0.0; tool_acc[1] = 0.0; tool_acc[2] = 0.0
    _build_cells(pos, n, Lx, Ly, Lz, cs, ncx, ncy, ncz, head, nxt)
    for cz in range(ncz):
        for cy in range(ncy):
            for cx in range(ncx):
                c = cx + cy * ncx + cz * ncx * ncy
                i = head[c]
                while i != -1:
                    # neighbour cells
                    for dz in range(-1, 2):
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                nxc = cx + dx; nyc = cy + dy; nzc = cz + dz
                                if nxc < 0 or nyc < 0 or nzc < 0:
                                    continue
                                if nxc >= ncx or nyc >= ncy or nzc >= ncz:
                                    continue
                                cc = nxc + nyc * ncx + nzc * ncx * ncy
                                j = head[cc]
                                while j != -1:
                                    if j > i:
                                        _pair_force(i, j, pos, vel, rad, mass,
                                                    force, Estar, mu_pp, v_ref,
                                                    gamma_coef)
                                    j = nxt[j]
                    _wall_force(i, pos, vel, rad, mass, force, Estar, mu_wall,
                                v_ref, gamma_coef, Lx, Ly, Lz, use_side_walls)
                    if tool_on == 1:
                        _tool_force(i, pos, vel, rad, mass, force, Estar,
                                    mu_tool, v_ref, gamma_coef, Xtool, yc, w,
                                    depth, sin_a, cos_a, v_tool, tool_acc)
                    i = nxt[i]


@njit(cache=True, fastmath=True)
def _integrate(pos, vel, rad, mass, force, n, dt, Lx, Ly, Lz, cs, ncx, ncy, ncz,
               head, nxt, Estar, mu_pp, mu_wall, v_ref, gamma_coef, g,
               use_side_walls, nsteps, tool_on, X0, yc, w, depth, sin_a, cos_a,
               v_tool, mu_tool, draft_out, tool_acc, vel_damp):
    keep = 1.0 - vel_damp
    for step in range(nsteps):
        Xtool = X0 + v_tool * dt * step
        _compute_all_forces(pos, vel, rad, mass, force, n, Lx, Ly, Lz, cs,
                            ncx, ncy, ncz, head, nxt, Estar, mu_pp, mu_wall,
                            v_ref, gamma_coef, g, use_side_walls, tool_on,
                            Xtool, yc, w, depth, sin_a, cos_a, v_tool, mu_tool,
                            tool_acc)
        for i in range(n):
            vel[i, 0] = (vel[i, 0] + dt * force[i, 0] / mass[i]) * keep
            vel[i, 1] = (vel[i, 1] + dt * force[i, 1] / mass[i]) * keep
            vel[i, 2] = (vel[i, 2] + dt * force[i, 2] / mass[i]) * keep
            pos[i, 0] += dt * vel[i, 0]
            pos[i, 1] += dt * vel[i, 1]
            pos[i, 2] += dt * vel[i, 2]
        if tool_on == 1:
            draft_out[step] = -tool_acc[0]   # x-resistance on the tool


# ---------------------------------------------------------------------------
# Python-side helpers
# ---------------------------------------------------------------------------

def _gamma_coef(restitution: float) -> float:
    e = max(min(restitution, 0.999), 1e-3)
    beta = math.log(e) / math.sqrt(math.log(e) ** 2 + math.pi ** 2)
    return -2.0 * math.sqrt(5.0 / 6.0) * beta


def _Estar(E: float, nu: float) -> float:
    return E / (2.0 * (1.0 - nu * nu))


@dataclass
class Bed:
    pos: np.ndarray
    vel: np.ndarray
    rad: np.ndarray
    mass: np.ndarray
    Lx: float
    Ly: float
    Lz: float
    params: DEMParams
    z_surface: float = 0.0


def build_packed_bed(p: DEMParams, Lx=0.40, Ly=0.16, fill_depth=0.22,
                     seed=0, settle_steps=14000) -> Bed:
    """Rain polydisperse spheres onto a loose lattice and settle under gravity."""
    rng = np.random.default_rng(seed)
    rmean = 0.5 * (p.r_min + p.r_max)
    spacing = 2.2 * p.r_max   # > 2 r_max: no initial overlaps -> no ejection
    nx = int(Lx / spacing); ny = int(Ly / spacing)
    # loose lattice packs down under gravity; start ~2.2x the target depth so
    # the settled bed lands near ``fill_depth``.
    nz = int((fill_depth * 2.2) / spacing)
    xs = (np.arange(nx) + 0.5) * spacing
    ys = (np.arange(ny) + 0.5) * spacing
    zs = (np.arange(nz) + 0.5) * spacing + rmean
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pos = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    pos += rng.uniform(-0.2 * rmean, 0.2 * rmean, pos.shape)
    n = pos.shape[0]
    rad = rng.uniform(p.r_min, p.r_max, n)
    mass = p.rho_p * (4.0 / 3.0) * math.pi * rad ** 3
    vel = np.zeros((n, 3))
    Lz = zs[-1] + 4 * rmean
    bed = Bed(pos=pos, vel=vel, rad=rad, mass=mass, Lx=Lx, Ly=Ly, Lz=Lz, params=p)
    # Densify frictionless (sliding spheres pack to ~0.6), then restore the real
    # friction for the drag -- a standard DEM bed-preparation trick.
    from dataclasses import replace
    bed.params = replace(p, mu_pp=0.0, mu_wall=0.05)
    _settle(bed, settle_steps)
    bed.params = p
    bed.z_surface = float(np.percentile(bed.pos[:, 2] + bed.rad, 97))
    return bed


def _grid(bed: Bed):
    cs = 2.0 * bed.params.r_max * 1.05
    ncx = max(1, int(bed.Lx / cs)); ncy = max(1, int(bed.Ly / cs))
    ncz = max(1, int(bed.Lz / cs))
    head = np.full(ncx * ncy * ncz, -1, dtype=np.int64)
    nxt = np.full(bed.pos.shape[0], -1, dtype=np.int64)
    return cs, ncx, ncy, ncz, head, nxt


def _settle(bed: Bed, nsteps: int):
    """Two-phase: gravity-pack with light damping, then quench to quasi-static."""
    p = bed.params
    cs, ncx, ncy, ncz, head, nxt = _grid(bed)
    force = np.zeros_like(bed.pos)
    draft = np.zeros(1)
    tool_acc = np.zeros(3)
    args = (_Estar(p.E, p.nu), p.mu_pp, p.mu_wall, p.v_ref,
            _gamma_coef(p.restitution), G, 1)
    n = bed.pos.shape[0]
    # phase 1: pack (very light damping so particles actually fall & rearrange)
    n_pack = int(0.8 * nsteps)
    _integrate(bed.pos, bed.vel, bed.rad, bed.mass, force, n, p.dt, bed.Lx,
               bed.Ly, bed.Lz, cs, ncx, ncy, ncz, head, nxt, *args, n_pack, 0,
               0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, p.mu_tool, draft, tool_acc,
               0.0006)
    # phase 2: quench residual kinetic energy
    _integrate(bed.pos, bed.vel, bed.rad, bed.mass, force, n, p.dt, bed.Lx,
               bed.Ly, bed.Lz, cs, ncx, ncy, ncz, head, nxt, *args,
               nsteps - n_pack, 0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
               p.mu_tool, draft, tool_acc, 0.06)
    bed.vel *= 0.0


@dataclass
class DragResult:
    draft_mean_N: float
    draft_series_N: np.ndarray
    x_tool: np.ndarray
    disturbed_area_m2: float
    disturbed_fraction: float
    depth_m: float
    width_m: float


def drag_tool(bed: Bed, depth_m: float, width_m: float, rake_deg: float = 35.0,
              v_tool: float = 0.3, drag_len: float = 0.22, x_start: float = 0.06,
              settle_first: int = 800, disturb_thresh: float = 0.02) -> DragResult:
    """Drag an inclined blade through a copy of the bed; return draft + tilth."""
    p = bed.params
    pos = bed.pos.copy(); vel = np.zeros_like(pos)
    rad = bed.rad.copy(); mass = bed.mass.copy()
    n = pos.shape[0]
    if settle_first:
        b2 = Bed(pos=pos, vel=vel, rad=rad, mass=mass, Lx=bed.Lx, Ly=bed.Ly,
                 Lz=bed.Lz, params=p)
        _settle(b2, settle_first)
    pos0 = pos.copy()
    cs, ncx, ncy, ncz, head, nxt = _grid(bed)
    force = np.zeros_like(pos)
    yc = bed.Ly * 0.5
    alpha = math.radians(rake_deg)
    sin_a, cos_a = math.sin(alpha), math.cos(alpha)
    nsteps = int(drag_len / (v_tool * p.dt))
    draft = np.zeros(nsteps)
    tool_acc = np.zeros(3)
    _integrate(pos, vel, rad, mass, force, n, p.dt, bed.Lx, bed.Ly, bed.Lz, cs,
               ncx, ncy, ncz, head, nxt, _Estar(p.E, p.nu), p.mu_pp, p.mu_wall,
               p.v_ref, _gamma_coef(p.restitution), G, 1, nsteps, 1, x_start,
               yc, width_m, depth_m, sin_a, cos_a, v_tool, p.mu_tool, draft,
               tool_acc, 0.0)  # no global damping during the dynamic drag
    x_tool = x_start + v_tool * p.dt * np.arange(nsteps)
    # steady-state window: tool between 30% and 90% of its travel
    lo = int(0.35 * nsteps); hi = int(0.9 * nsteps)
    draft_mean = float(np.mean(draft[lo:hi]))
    # Tilth metric: disturbed-soil cross-sectional area = disturbed bulk volume
    # per unit travel. Robust to projection saturation, and scales with depth.
    disp = np.linalg.norm(pos - pos0, axis=1)
    moved = disp > disturb_thresh
    pvol = (4.0 / 3.0) * math.pi * bed.rad ** 3
    packing = 0.59
    disturbed_vol = float(pvol[moved].sum()) / packing
    area = disturbed_vol / drag_len
    return DragResult(draft_mean_N=draft_mean, draft_series_N=draft, x_tool=x_tool,
                      disturbed_area_m2=area,
                      disturbed_fraction=float(moved.mean()),
                      depth_m=depth_m, width_m=width_m)


def bulk_density(bed: Bed) -> float:
    """Settled bulk density (kg/m^3) of the prepared bed."""
    vol_solid = (4.0 / 3.0) * math.pi * float(np.sum(bed.rad ** 3))
    return bed.params.rho_p * vol_solid / (bed.Lx * bed.Ly * bed.z_surface)
