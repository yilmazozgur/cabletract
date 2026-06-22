# CableTract (regen-default version) — Simulation Plan

A prioritized catalogue of **simulation-only** studies that would strengthen and de-risk the
regen-default CableTract design, without any bench rig or field campaign. Every item is
designed to run inside the existing `cabletract/` Python stack (numpy / scipy / pandas /
SALib / pymoo / pvlib / shapely / sklearn) and to extend the single-`run_single` code path
rather than fork it.

This plan targets the **regen-default reference** (winch round-trip η = 0.606, 760 Wh/decare,
€35,870 capex / €870 incremental, 1.30 yr discounted payback @ 25 ha, 12–17 decares/day
off-grid, 88 % off-grid-feasible cells, lifecycle CO₂ 14.6 kg/ha·yr).

---

## Why these (and not others)

The manuscript's own Limitations rank the binding near-term risks as **mechanical and
operational, not financial**. The studies below are ordered to attack, in priority:

1. the dominant *unresolved mechanical risk* — closed-loop tilling-depth control;
2. the two *load-bearing assumptions* — the co-design draft reduction and the compaction story;
3. the *unmeasured anchor* (cyclic wear / self-fatigue);
4. the places the pipeline currently uses its *crudest approximation* (the linear PV-harvest
   fit in the operating envelope, the typical-year TMY, the static battery);
5. the *strongest reviewer attack* — the 19th-century steam-cable-ploughing precedent.

### Honest caveat (applies to every item)

A simulation **cannot validate** the design — it can only (a) quantify a risk, (b) test
whether a claim is internally consistent under a richer model, or (c) bound a failure mode.
SIM-only results keep the paper's correct framing: *an internally consistent analytical
envelope, not a guarantee*. Two design choices below mitigate this: each study reports
**sensitivity bands**, not point estimates, and **S17 (Bayesian calibration scaffold)** is
built so that the moment any bench/field datum arrives, the headline numbers can be
re-anchored with posterior credible intervals.

### Legend

- **Effort:** S (days) · M (1–2 weeks) · L (3+ weeks)
- **Priority:** ★★★ retire a binding risk / validate a load-bearing claim · ★★ materially
  strengthen · ★ nice-to-have
- **Stack:** `pure` = runnable today in the current deps · `+lib` = one new pure-Python
  dependency · `+ext` = heavy external solver (a reduced-order pure-Python alternative is
  always given so the study stays SIM-doable now).

---

## Group 1 — Dominant mechanical risk: dynamics & control

### S1 ★★★ — Closed-loop tilling-depth control co-simulation
- **Addresses:** the manuscript's self-named *single largest unresolved technical risk*
  (Open Risks §7.2): can a cable carriage, with no vehicle inertia and only cable elasticity
  + motor torque bandwidth as a reference frame, hold tilling depth through a transient draft
  spike the way a tractor's 3-point hitch does?
- **Method (reduced-order, `pure`):** a small ODE model integrated with `scipy.integrate`:
  cable as a tensioned spring-damper (use the real 8 mm Dyneema EA from `cable_props.csv`),
  carriage mass + depth-control actuator, and a soil-reaction force = D497 nominal (from
  `soil.py`) **plus** programmed transients (buried stone → force impulse; hardpan → step;
  moisture gradient → ramp). Close a PID and an MPC loop on winch torque + depth wheel.
- **Method (high-fidelity, `+ext`):** lumped-element or FE cable (multibody) — optional;
  the ODE version captures the first-order stability question already.
- **Inputs:** `cable_props.csv` (EA, mass/m), codesigned draft P50/P90, span 50 m,
  carriage 250 kg.
- **Outputs:** depth-tracking RMS error, overshoot, settling time vs cable stiffness,
  pretension, span, and controller bandwidth; a **stability envelope** figure; a one-line
  agronomic-acceptance verdict (depth error vs tilth tolerance).
- **Repo:** new `cabletract/dynamics.py`; driver `scripts/run_phase8_depth_control.py`.
- **Effort:** M. **Bonus:** the same model yields a **SIM estimate of the regen recovery
  fraction** `f_rec` (currently the assumed 0.35 that the *whole regen-default* now rests on)
  from the integrated motor four-quadrant power on the return leg.

### S2 ★★ — Cable safety & durability simulations
- **Addresses:** the unquantified cable safety surface (§7.2) and the absence of a cable
  replacement schedule in the BOM/LCA.
- **Sub-studies (`pure`):**
  - **Snap-back recoil:** energy stored in a taut 50 m Dyneema/steel cable at working tension,
    released on mid-span failure → recoil velocity, kinetic energy, and a required
    **exclusion-zone radius**. Compare Dyneema vs steel (steel stores ~5× the elastic energy
    per the modulus/mass in `cable_props.csv`).
  - **Bending-over-sheave fatigue:** Feyrer wire-rope bending-fatigue model (or a
    Dyneema cyclic-creep curve) for the drum/sheave radius and per-round tension cycles →
    **cycles-to-failure → cable replacement interval (yr)**, which becomes a new OPEX + LCA line.
  - **Aeroelastic check:** galloping / vortex-shedding of a long taut cable vs the bundled
    monthly mean wind speeds — flag any resonance with the carriage traverse frequency.
- **Outputs:** exclusion-zone vs tension table; cable-life vs sheave-radius curve; a wind-speed
  safe-operation band.
- **Repo:** extend `cabletract/physics.py`; driver folds into `run_phase1`.
- **Effort:** S–M.

---

## Group 2 — Validate the two load-bearing assumptions

### S3 ★★★ — Soil–tool simulation of the co-designed implement library
- **Addresses:** the central ~0.37× draft lever currently rests on **D497 coefficient scaling
  for narrow tools that were never built**, and the agronomic-equivalence question
  (does a 15 cm narrow-ripper pass — or two — replace a 40 cm subsoiler pass?).
- **Method (reduced-order, `pure`):** an analytic soil-failure-wedge model
  (McKyes–Godwin / universal earthmoving equation) for the narrow ripper, chisel, sweep —
  predicts draft *and* the disturbed-soil cross-section independently of D497, so it
  **cross-checks** the library and produces a fragmentation/tilth proxy.
- **Method (high-fidelity, `+ext`):** true DEM (YADE/LIGGGHTS) tillage — optional; the
  analytic wedge model answers the first-order draft + tilth-equivalence question now.
- **Outputs:** DEM/analytic-vs-D497 draft comparison per implement; a **shallow-narrow vs
  deep-wide tilth-equivalence** verdict (disturbed area + fragmentation), which directly
  answers the referee objection that the two are not interchangeable.
- **Repo:** new `cabletract/tillage_mechanics.py`; folds into `run_phase2`.
- **Effort:** M (analytic) / L (DEM).

### S4 ★★★ — Co-design lever sensitivity sweep
- **Addresses:** "if co-design delivers only 0.6× or 0.8× instead of 0.37×, how much of the
  paper survives?" — a reviewer-requested robustness check, trivial but high-value.
- **Method (`pure`):** parameterize the achieved draft ratio and propagate it through
  `run_single` → energy/decare, off-grid throughput, auger count, NPV. Sweep 0.37 → 0.8.
- **Outputs:** a single panel showing the four headline claims (energy, off-grid, anchor,
  NPV) as a function of the *achieved* co-design reduction — turns the load-bearing assumption
  into a transparent dial.
- **Repo:** extend `cabletract/uncertainty.py`; folds into `run_phase5`.
- **Effort:** S.

### S5 ★★★ — Depth-resolved compaction model + repeated-strip trafficking + yield coupling
- **Addresses:** the 73× index is a per-contact-patch surface metric (a referee flagged that
  "richer models only help us" is not defensible — repeated trafficking of the *same* narrow
  carriage strip cuts the other way), and the paper disclaims any soil→yield link.
- **Method (`pure`):**
  - Replace the surface `Σp²·A` index with a **Söhne/Boussinesq stress-propagation kernel**
    (closed-form, no FE needed): compute bulk-density change vs depth under the carriage roller
    vs the tractor wheel.
  - **Accumulate over the carriage's repeated passes on the same strip** (the carriage hits
    its strip-midlines every pass, unlike the tractor's once-per-area) — test whether the
    narrow strip becomes a compacted rut.
  - **Couple to a crop response:** a penetration-resistance → root-growth → relative-yield
    function (Whitfield/da Silva-style, or a thin APSIM/DSSAT hook) to convert the index into a
    **yield delta (%)** — closing the link the paper explicitly disclaims.
- **Outputs:** depth-resolved bulk-density profiles (carriage vs tractor); a corrected,
  honestly-bounded compaction-energy ratio; an estimated yield delta with uncertainty.
- **Repo:** new `cabletract/compaction_depth.py` (extends `compaction.py`); folds into
  `run_phase4`.
- **Effort:** M.

---

## Group 3 — The unmeasured anchor (numerical geotechnics)

### S6 ★★★ — Season-long "soil-state memory" Monte Carlo (anchor self-fatigue)
- **Addresses:** the §7.2 risk that each Anchor station progressively loosens its own soil,
  eroding the 9-auger lateral-capacity margin *from the inside* over a season.
- **Method (`pure`):** extend `run_single` (or wrap it) with a per-station soil-state variable:
  each placement reduces local relative density / cone index by Δ, degrading per-auger lateral
  capacity along an assumed curve; replay the full operating calendar and check whether heavy
  primary tillage stays inside the 9-auger envelope by mid/late season.
- **Outputs:** capacity-margin vs operating-day curve per soil class; the operating-day at
  which heavy tillage drops out; sensitivity to the (assumed) loosening rate Δ.
- **Repo:** new `cabletract/anchor_state.py`; folds into `run_phase1`/`run_phase5`.
- **Effort:** M.

### S7 ★★ — 3×3 helical-auger group lateral-capacity model
- **Addresses:** the "Khand 400 N vs Magnum 2 kN, have-it-both-ways" literature gap — replace
  it with one internally-derived nominal + a group-efficiency factor.
- **Method (reduced-order, `pure`):** a **p–y curve / Winkler-spring** beam-on-foundation model
  (ODE, `scipy`) for a single helical pile under lateral load in loose vs medium-dense sand,
  then a group-reduction (shadowing) factor for the 3×3 cluster.
- **Method (`+ext`):** 3-D FE (sfepy/FEniCS) — optional confirmation.
- **Outputs:** a single derived per-auger working lateral capacity with its assumptions
  (density, embedment, deflection limit, group efficiency) and a sensitivity band, replacing
  the dual-reference oscillation in §5.4.
- **Repo:** extend `cabletract/physics.py` (anchor section); folds into `run_phase1`.
- **Effort:** M.

### S8 ★★ — Cyclic insertion/extraction wear model (SIM proxy for the bench test)
- **Addresses:** the totally-unmeasured helix-wear / retraction-fatigue regime; gives the
  missing **auger-replacement interval** for the economics/LCA.
- **Method (`pure`/`+ext`):** an abrasive-wear model (Archard wear law on the helix leading
  edge driven by insertion torque × cycles) — or a reduced DEM of repeated insertion — to
  estimate torque growth (blunting) and a **mean-time-between-auger-replacement**.
- **Outputs:** auger-life vs soil-abrasivity curve → an explicit OPEX line (feeds S14) and an
  embodied-CO₂ term (feeds the LCA).
- **Repo:** new `cabletract/auger_wear.py`.
- **Effort:** M. **Note:** inherently uncertain without bench data; report as a bounded
  scenario, not a point estimate.

---

## Group 4 — Extend the pipeline where it is currently crudest (all `pure`)

### S9 ★★★ — Full hourly-TMY operating envelope (replace the linear α = 0.169 fit)
- **Addresses:** the headline operating-envelope result (§5.10) currently rests on the
  *crudest* model in the paper — a one-parameter linear PV-harvest fit — while the validated
  hourly simulator sits unused for the sweep.
- **Method (`pure`):** run the 3,600-cell (GHI × farm-size) envelope on the **actual hourly
  TMY + battery SoC simulator** (`energy.py`), not the α-fit. With regen now default
  (760 Wh/decare) the off-grid-feasible region should *expand* beyond the current 88 %.
- **Outputs:** a re-derived F21 with the true off-grid contour and payback distribution; the
  error the α-fit introduced near the breakeven contour (the binding constraint).
- **Repo:** rewrite the sweep in `scripts/run_phase7_envelope.py` to call the hourly model.
- **Effort:** M (mostly compute time for 3,600 hourly runs — cache aggressively).

### S10 ★★ — Multi-year / worst-year TMY resilience
- **Addresses:** the off-grid claim is a "worst-week-of-worst-month" assertion built on a
  *single synthesized typical year*.
- **Method (`pure`):** drive the hourly simulator with multiple real TMY/actual-year series
  (or a stochastic ensemble of cloudiness), including a deliberately bad year (El Niño /
  aerosol-dimmed). Report off-grid feasibility at the P10 *year*, not the median year.
- **Outputs:** grid-hours distribution across years per site; a worst-year off-grid verdict.
- **Repo:** extend `cabletract/energy.py` (multi-year driver); folds into `run_phase3`.
- **Effort:** M.

### S11 ★★ — Battery state-of-health degradation over the 15-yr horizon
- **Addresses:** the static battery assumption; couples to the year-8 replacement.
- **Method (`pure`):** a calendar+cycle aging model (capacity fade vs throughput and time);
  re-run off-grid feasibility in year 1 vs year 7 (pre-replacement) vs year 9.
- **Outputs:** off-grid feasibility vs battery SoH; whether the year-8 replacement timing is
  right; sensitivity of the 88 % envelope to fade.
- **Repo:** extend `cabletract/energy.py` (`BatterySpec` + SoH); folds into `run_phase3`.
- **Effort:** S–M.

### S12 ★★ — Smart dispatch (MPC / dynamic programming) vs the naive 09:00–15:00 duty cycle
- **Addresses:** a real, free design lever — scheduling operations into sunny windows could
  push Mediterranean/temperate sites into off-grid feasibility.
- **Method (`+lib`, e.g. `cvxpy` or DP in numpy):** optimize the daily operating schedule
  against the hourly harvest + battery SoC to minimize grid import, vs the fixed-window
  baseline.
- **Outputs:** grid-hours reduction from smart dispatch per site; the off-grid envelope
  expansion; how much of Beauce's deficit dispatch alone can close.
- **Repo:** new `cabletract/dispatch.py`; folds into `run_phase3`.
- **Effort:** M.

---

## Group 5 — Economics, geometry & uncertainty (mostly `pure`)

### S13 ★★ — Stochastic diesel-price (real-options) + carbon-price economics
- **Addresses:** the tornado showed the result is **exogenous-led** (diesel price/volume);
  a stochastic treatment is the honest next step, and a carbon price monetizes the 2.2× CO₂ edge.
- **Method (`pure`):** model diesel price as a stochastic process (GBM with historical
  drift/volatility) → Monte Carlo NPV *distribution* and the **option value** of fuel-price
  exposure; add a carbon-price scenario as a revenue line.
- **Outputs:** NPV distribution (not point estimate) per farm size; probability NPV > 0 in
  the additive frame; carbon-price breakeven.
- **Repo:** extend `cabletract/economics.py`; folds into `run_phase5`.
- **Effort:** S–M.

### S14 ★ — Auger-replacement OPEX integration
- **Addresses:** the missing OPEX/LCA line a reviewer flagged (depends on **S8**).
- **Method (`pure`):** fold the S8 mean-time-between-replacement into the DCF cashflow series
  and the embodied-CO₂ accounting; re-report NPV/payback with the new line.
- **Outputs:** NPV/payback sensitivity to auger life; a defensible maintenance schedule.
- **Repo:** extend `cabletract/economics.py`.
- **Effort:** S.

### S15 ★★ — Real-farmland corpus + fleet routing at scale
- **Addresses:** the 50 hand-tuned adversarial polygons are not representative; and the
  additive economics only turn positive at large area (~240 ha), which needs a multi-machine view.
- **Method (`pure`, `+lib` for parcel I/O):** ingest a real parcel corpus (EuroCrops / LPIS /
  USDA-CDL) for a target region (Konya, Beauce), re-run `layout.py` shape efficiency at scale;
  add a fleet-routing sim (how many units, headland logistics) for large farms.
- **Outputs:** shape-efficiency distribution on representative farmland vs the adversarial
  corpus; units-required and routing cost at 240+ ha.
- **Repo:** extend `cabletract/layout.py`; new corpus loader; folds into `run_phase4`/`run_phase6`.
- **Effort:** M.

---

## Group 6 — Creative / integrative

### S16 ★★★ — Counterfactual steam-cable-ploughing techno-economic model
- **Addresses:** the strongest reviewer attack — "the 19th century already ran this experiment
  (Fowler/Howard cable ploughing) and it failed; you admit you haven't engaged why."
- **Method (`pure`):** build a techno-economic model of steam-era cable ploughing, then apply
  the modern deltas **one at a time** — autonomy removes per-station crew; parallel augering +
  drone alignment removes repositioning cost; co-design removes the draft penalty;
  electrification removes the boiler — and show *quantitatively* which of the four documented
  historical failure modes each delta closes.
- **Outputs:** a waterfall chart from "1880s economics" to "regen-default CableTract"; a
  point-by-point rebuttal that converts an admitted weakness into a contribution.
- **Repo:** new `cabletract/historical_baseline.py`; new appendix figure.
- **Effort:** M.

### S17 ★★ — Full digital-twin "virtual season" + Bayesian calibration scaffold
- **Addresses:** the integrating vision, and the path from "analytical envelope" to
  "data-anchored" the moment any future datum arrives.
- **Method (`pure`/`+lib`):** a co-simulation that runs a full virtual cropping season on a
  virtual field, coupling S1 (depth control), S5 (compaction), S6 (anchor state), and the
  hourly energy/economics — emitting throughput, depth-quality, anchor margin, compaction,
  energy, and NPV *together*. Wrap the headline parameters in a Bayesian layer
  (`+lib`: `emcee`/`pymc`) so that any future bench/field measurement re-anchors the model
  with **posterior credible intervals** instead of a deterministic point.
- **Outputs:** one end-to-end virtual-season report; a prior-predictive envelope today, ready
  to become a posterior the day data exists.
- **Repo:** new `cabletract/digital_twin.py` orchestrating the new modules.
- **Effort:** L (build last, after S1/S5/S6 exist).

### S18 ★ — Agrivoltaic dual-use techno-economics
- **Addresses:** the Main Unit is *stationary* with deployable PV — could it earn revenue as a
  distributed grid-feeding / agrivoltaic PV asset during the off-season and idle hours?
- **Method (`pure`):** extend the energy + economics chain to credit exported PV (feed-in or
  self-consumption) when the machine is not operating; re-evaluate NPV with this second revenue
  stream.
- **Outputs:** NPV uplift from dual-use; how much it offsets the regen premium / shortens
  small-farm payback.
- **Repo:** extend `cabletract/economics.py` + `energy.py`.
- **Effort:** S–M.

---

## Suggested sequencing

**Phase A — retire the binding risk & validate the levers (do first):**
S1 (depth control) · S4 (co-design sensitivity, trivial) · S9 (full-hourly envelope) ·
S3 (soil–tool draft/tilth) · S5 (compaction→yield).

**Phase B — close the geotech & energy gaps:**
S6 (anchor self-fatigue) · S7 (auger group capacity) · S10 (multi-year) · S11 (battery SoH) ·
S12 (smart dispatch).

**Phase C — economics, geometry, narrative:**
S13 (stochastic/carbon) · S15 (real corpus + fleet) · S16 (steam-plough rebuttal) ·
S8 + S14 (auger wear → OPEX) · S2 (cable safety/fatigue) · S18 (agrivoltaic).

**Phase D — integrate:**
S17 (digital twin + Bayesian calibration scaffold).

### If you can only do three (all `pure`, all this week)
1. **S4** — co-design lever sensitivity (hours of work, directly answers a referee).
2. **S9** — full-hourly operating envelope (replaces the crudest model; likely *expands* the
   regen-default off-grid region).
3. **S1** — depth-control co-simulation (the one study that touches the #1 mechanical risk;
   also yields a SIM estimate of the regen `f_rec` the whole default now depends on).

---

## New repo structure these imply

```
cabletract/
  dynamics.py            # S1  depth-control ODE + regen f_rec estimate
  tillage_mechanics.py   # S3  analytic soil-failure-wedge draft + tilth
  compaction_depth.py    # S5  Söhne/Boussinesq depth-resolved + yield
  anchor_state.py        # S6  season-long soil-state memory
  auger_wear.py          # S8  cyclic wear -> replacement interval
  dispatch.py            # S12 MPC/DP battery-aware scheduling
  historical_baseline.py # S16 steam-cable-ploughing counterfactual
  digital_twin.py        # S17 virtual-season co-simulation + Bayesian layer
  (extend) physics.py    # S2 cable safety/fatigue, S7 p-y auger group
  (extend) energy.py     # S10 multi-year, S11 battery SoH
  (extend) economics.py  # S13 stochastic/carbon, S14 auger OPEX, S18 agrivoltaic
  (extend) uncertainty.py# S4 co-design lever sweep
  (extend) layout.py     # S15 real corpus + fleet routing
scripts/
  run_phase8_depth_control.py   # S1
  run_phase9_compaction_depth.py# S5
  run_phase10_anchor_state.py   # S6
  run_phase11_dispatch.py       # S12
  run_phase12_historical.py     # S16
  run_phase13_digital_twin.py   # S17
  (rewrite) run_phase7_envelope.py  # S9 full-hourly envelope
```

All new modules follow the existing convention: deterministic, parameterized, unit-tested,
each emitting figures into `figures/` and per-figure CSVs into `tables/`.
