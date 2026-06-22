# CableTract — High-Fidelity Simulation Implementation Plan (S1, S2, S3, S4, S7)

Build plan for the five selected studies from `REGEN_VERSION_SIM_PLAN.md`, at the
**highest fidelity runnable in this environment**.

## Environment ceiling (probed 2026-06-22)

`python 3.12.7 · numpy 1.26.4 · scipy 1.14.1 · matplotlib 3.9.2 · numba 0.59.1 · sympy 1.13.1`.
**No** external solvers (sfepy, FEniCS/dolfin, skfem, YADE, pysph, control, cvxpy) and they are
not practically installable here. Consequences for "highest fidelity":

- **S1, S2, S4** — pure numpy/scipy *is* the high-fidelity route (stiff ODE integration, FDM wave
  propagation, full parameter sweeps). numba used to accelerate inner loops.
- **S3 (tillage)** — highest runnable option is a **numba soft-sphere DEM** (Hertz–Mindlin contacts,
  $10^4$–$10^5$ particles). Genuine DEM physics, not a wedge surrogate; not LIGGGHTS research scale.
  An analytic McKyes–Godwin wedge model is built alongside as the verification baseline.
- **S7 (anchor)** — highest *appropriate* method for a laterally loaded pile is a **nonlinear
  p–y Winkler beam-column BVP** (API/Reese sand p–y curves) with **group p-multipliers** (Mokwa,
  CHANCE), solved with scipy. This is the geotechnical standard and is more defensible for piles
  than a generic continuum FE; a 3-D continuum FE would need an absent solver.

All modules follow the repo convention: deterministic, parameterized dataclasses, unit-tested,
each emitting figures to `figures/` and CSVs to `tables/`, driven by a `scripts/run_phaseN_*.py`.

---

## S4 — Co-design lever sensitivity sweep  ★★★  (pure; build first, smallest)

**Question.** If co-design delivers only 0.6× or 0.8× the draft reduction instead of the claimed
0.37×, how much of the energy / off-grid / anchor / economics story survives?

**Method (high fidelity = full pipeline, no surrogate).** Parameterize a *co-design effectiveness*
that scales the reference draft between the codesigned point (1.8 kN) and the conventional library
median (≈6.15 kN). For each draft level: (i) `run_single` → energy/decare, off-grid throughput;
(ii) `soil.draft_distribution` P90 → required augers via `physics.anchor_reaction_envelope`
(both 2 kN nominal and 400 N worst-case bounds); (iii) `economics.cabletract_npv_vs_diesel` → NPV,
payback. Report all four headline outputs vs achieved draft ratio, with the reference marked.

**Outputs.** `figures/F22_codesign_sensitivity.png` (4 panels), `tables/F22_codesign_sensitivity.csv`.
**Files.** `cabletract/codesign_sensitivity.py`, `scripts/run_phase8_codesign_sensitivity.py`.
**Validation.** At the reference draft the four outputs reproduce the manuscript values
(889 Wh/decare, ~11.6 dec/day, 9 augers @400 N P90, NPV +€3575). **Effort: S.**

## S1 — Closed-loop tilling-depth control co-simulation  ★★★  (highest value)

**Question.** Can a cable carriage with no vehicle inertia hold tilling depth through a transient
draft spike the way a tractor's 3-point hitch does?

**Method (high fidelity = multibody cable + soil + closed loop).**
- **Cable:** lumped-mass discretization, $N\!\sim\!50$–100 nodes, axial Hookean springs (EA from
  `cable_props.csv`), structural + aerodynamic damping, gravity → catenary; optional bending
  stiffness. State $(\mathbf{x}_i,\dot{\mathbf{x}}_i)$ integrated with `scipy.integrate.solve_ivp`
  (Radau/BDF, stiff) or a numba semi-implicit (Newmark-$\beta$) integrator for speed.
- **Carriage + depth actuator:** mass + a depth-control degree of freedom with actuator bandwidth
  and rate/force limits; tool depth $d(t)$ sets soil reaction.
- **Soil reaction:** D497 draft as a function of instantaneous depth/speed (from `soil.py`) **plus**
  programmed transients — buried stone (force impulse), hardpan (depth-dependent step), moisture
  ramp; a tyre/penetration stiffness $k_\text{soil}$ couples depth error to vertical force.
- **Controllers:** PID and a finite-horizon **MPC** (QP via a small dense solver in numpy/scipy
  since cvxpy is absent) on winch torque + depth actuator.
- **Benchmark:** identical disturbance applied to a tractor model (3-point hitch + several-tonne
  inertia) for a side-by-side depth-error comparison.

**Outputs.** depth-error time series, RMS/overshoot/settling **stability envelope** vs cable
stiffness, pretension, span, controller bandwidth (`figures/F23_depth_control.png`,
`F24_stability_envelope.png`, CSVs). Side benefit: a SIM estimate of the regen `f_rec`.
**Files.** `cabletract/dynamics.py`, `scripts/run_phase9_depth_control.py`.
**Validation.** static equilibrium reproduces `physics.catenary_sag`; energy conservation on an
undamped free-vibration test; first natural frequency matches the taut-string analytic
$f_1=\tfrac{1}{2L}\sqrt{T/\mu}$. **Effort: M–L.**

## S2 — Cable safety & durability  ★★  (pure)

Three sub-models, all high-fidelity-in-Python:
1. **Snap-back recoil** — 1-D transient wave equation on the tensioned cable, FDM (numba),
   sudden mid-span release → recoil velocity field, peak KE, fragment throw, **exclusion-zone
   radius**; Dyneema vs steel from `cable_props.csv`.
2. **Bending-over-sheave fatigue** — Feyrer wire-rope bending-fatigue model (steel) and a Dyneema
   cyclic-creep/endurance curve → cycles-to-failure at the drum/sheave $D/d$ ratio and per-round
   tension → **cable replacement interval** → OPEX + LCA line.
3. **Aeroelastic stability** — modal analysis of the taut cable + Den Hartog galloping criterion
   and vortex-shedding lock-in vs the bundled monthly mean wind speeds → safe-operation band.

**Outputs.** `figures/F25_cable_safety.png` (3 panels), CSVs; a safety/cable-replacement cost line
for the economics. **Files.** `cabletract/cable_safety.py`, `scripts/run_phase10_cable_safety.py`.
**Validation.** wave speed $c=\sqrt{T/\mu}$ matches FDM dispersion; $f_1$ matches S1. **Effort: M.**

## S3 — Soil–tool DEM of the co-designed implements  ★★★  (numba DEM)

**Question.** Do the narrow co-designed tools achieve the D497-predicted draft, and is a shallow
narrow pass agronomically equivalent (soil fragmentation/disturbance) to a deep wide one?

**Method (high fidelity = soft-sphere DEM).**
- **Soil:** polydisperse sphere packing in a 3-D bin; **Hertz–Mindlin** normal + tangential
  contacts, Coulomb friction, optional cohesion (bonded-particle) for cohesive soils; gravity
  compaction to a target bulk density. numba-jit neighbour lists (cell linked-list) for $10^4$–$10^5$
  particles.
- **Tool:** rigid triangulated narrow ripper / chisel / sweep dragged at the codesigned depth/speed;
  integrate contact forces → **draft** and **disturbed-soil cross-section / fragmentation**.
- **Runs:** the depth-dependent tillage implements (ripper 15 vs 40 cm; chisel; sweep) at codesigned
  vs conventional operating points; compare draft to D497 and tilth (disturbance area, fragment-size
  distribution) for the shallow-vs-deep equivalence question.
- **Baseline check:** analytic **McKyes–Godwin** soil-failure-wedge draft model implemented in
  parallel as a sanity bound (`cabletract/tillage_mechanics.py`).

**Outputs.** DEM-vs-D497 draft per implement; shallow-vs-deep tilth-equivalence verdict
(`figures/F26_dem_draft.png`, `F27_tilth_equivalence.png`, CSVs).
**Files.** `cabletract/tillage_dem.py` (numba), `cabletract/tillage_mechanics.py`,
`scripts/run_phase11_tillage.py`.
**Validation.** quasi-static draft converges with particle count + time-step; wedge model and DEM
agree within ~30% on a calibrated case; angle of repose of the packing matches the input friction.
**Effort: L** (the most demanding; DEM calibration + runtime).

## S7 — 3×3 helical-auger group lateral capacity  ★★  (nonlinear p–y BVP)

**Question.** Replace the "Khand 400 N vs Magnum 2 kN have-it-both-ways" literature gap with one
internally-derived per-auger capacity + a group-efficiency factor.

**Method (high fidelity for piles = nonlinear Winkler beam-column).**
- **Single pile:** Euler–Bernoulli beam-column, $EI\,y'''' = -p(y,z)$, with **API/Reese sand
  p–y curves** $p = A\,p_u \tanh(k z y / (A p_u))$ (loose vs medium-dense $\phi$, $\gamma$, $k$);
  solve the nonlinear BVP (scipy `solve_bvp` or Newton on an FD discretization) for the
  lateral capacity at the IBC 1-inch (25.4 mm) head-deflection serviceability limit, free-head
  and fixed-head.
- **Group (3×3):** **p-multipliers** by row (leading/trailing) per Mokwa/Reese for the cluster
  spacing → group efficiency and per-auger allowable capacity with a stated safety factor.
- **Output:** one derived nominal per-auger capacity (with assumptions) + sensitivity band, to
  replace §5.4's dual reference; recomputed auger counts for the full library.

**Outputs.** p–y capacity vs soil/fixity, group efficiency, derived nominal
(`figures/F28_anchor_py.png`, CSV). **Files.** `cabletract/anchor_py.py`,
`scripts/run_phase12_anchor_py.py`.
**Validation.** linear small-deflection limit matches the Winkler closed form
$y_0 = H/(2\beta^3 EI)$, $\beta=(k/4EI)^{1/4}$; ultimate capacity brackets the cited
Khand/Magnum range. **Effort: M.**

---

## Build order & integration

1. **S4** (smallest, validates the pipeline wiring) → `run_phase8`.
2. **S7** (self-contained, replaces a known weak spot) → `run_phase12`.
3. **S1** (highest value; reuses the cable model S2 also needs) → `run_phase9`.
4. **S2** (shares the cable model with S1) → `run_phase10`.
5. **S3** (most expensive; DEM + wedge) → `run_phase11`.

Each is a standalone module + driver; results feed back into the manuscript (S1→depth-control risk,
S2→cable-replacement OPEX + safety, S3→co-design draft + agronomic-equivalence table, S4→robustness
panel, S7→single anchor nominal). Tests live in `tests/test_<module>.py` with the validation checks
above. numba functions guarded so the suite runs without JIT if needed.
