# CableTract manuscript v3 — change log (2026-07-05)

v3 implements the full multi-layer review of 2026-07-05 (`manuscript/review/cabletract_v2_review_2026-07-05.md`).
Every model change below was made in the pipeline first; every manuscript number was then regenerated from one
canonical run (`manuscript/review/v3_canonical_numbers.md`). All 191 tests pass (two pre-existing environment
failures excluded: `test_tillage.py` collection, `test_simulate.py` missing legacy CSV).

## A. Model corrections (pipeline code)

1. **Compaction accounting (`compaction.py`)** — v2 billed the carriage one roller track per 50-m span band
   (~33× undercount). v3 bills one traverse per 1.5-m swath. Headline: ~98% area reduction → **46.7%**
   (26.7% vs 50% per pass); new field-integrated p²·area metric: **39.4×**; per-gear index unchanged (73.4×).
2. **Anchoring energy (`anchoring.py`, new; `params.py`, `simulate.py`)** — v2 charged the per-round anchoring
   cycle as time only. v3 respecifies the anchors as **1.0-m × 60-mm ground screws** (90 N·m final torque,
   269 W/drive, 2.4 kW Anchor transient — consistent with the 30-s parallel insert/retract and "small BLDC"
   claims; the v2 S7 2-m construction pile was operationally impossible) and charges **25.3 Wh/round**
   (27% of the electrical budget). Energy headline: 889 → **1226.8 Wh/decare** (12.27 kWh/ha).
3. **S7 re-run at the installed screw geometry (`run_phase12`)** — derived working nominals now: free-head
   loose 0.37 kN (reproduces the 0.4-kN literature bound), fixed-head 1.9–2.6 kN (reproduces the 2-kN
   nominal). The 9-screw cluster relies on frame coupling (fixed-head) in loose sand — stated explicitly.
4. **CableTract+ physics (`variants.py`)** — v2 booked the 1/√2 tension split as an energy saving and widened
   the implement for free. v3: draft and width unchanged; CT+ eliminates the anchoring cycle (energy + 90% of
   setup). New row: 16.5 daa/day (1.37×), 889 Wh/daa (0.73×), €84,997 (2.20×). The 2.61×/0.53× multipliers
   are withdrawn in the text.
5. **Reproducibility (`energy.py`)** — salted `hash(site)` seed → `zlib.crc32` (bit-for-bit re-runs verified).
6. **Weather realism (`energy.py`)** — i.i.d. hourly clouds → AR(1) daily cloud state (ρ=0.65, Aguiar-style)
   × mean-1 hourly jitter, so multi-day overcast spells exist.
7. **Seasonal duty cycle (`run_phase3`)** — the 2-kW work draw now applies only on 170 operating-season days
   (hemisphere-aware), not 365 d/yr. Beauce's "911 grid-h/yr, infeasible" was an artefact of phantom winter
   demand; new per-site grid hours at reference hardware: **Konya 58, Des Moines 150, Palencia 162,
   São Paulo 164, Ludhiana 211, Beauce 282** (new `tables/F8_site_summary.csv` incl. worst-week imports).
8. **Economics (`economics.py`)** — carriage costed (€2,800; machine capex €38,670); implement set (€9,500)
   carried symmetrically on both sides (netting assumption disclosed, tornado axis); cable-replacement line
   (€180/yr) folded into the reference cash flow; **additive frame made frame-consistent** (diesel-maintenance
   credit removed — v2 understated the additive loss by ~€15k); hour-based maintenance sensitivity helper added.
   New chain: NPV@8% = −5,175 (1 ha) / **−1,724 (25 ha)** / +1,871 (50) / +9,061 (100); breakeven ≈ 40 ha/yr;
   additive −61,203 @25 ha. Envelope: **48.3% of cells NPV-positive** (was "100%"); additive 11.6%; energy+ 80.8%.
9. **LCA (`economics.py`)** — replacement battery pack, second cable, carriage steel added to embodied;
   e-tractor grid draw 22 → 40 kWh/ha (consistent with its own drawbar work); diesel factor relabelled
   tank-to-wheel. New: **17.0 vs 32.5 vs 27.9** kg CO₂/ha-yr (1.9× vs diesel, 1.6× vs e-tractor).
10. **F5 regenerated with codesigned implements** (v2 plotted the conventional library against a codesigned
    caption); ripper speed saving corrected 41% → **31%**. F3 legend η=0.70 → 0.74. F11 re-run at codesigned
    parameters with per-round setup: work 68.3 h / setup 52.6 h (**43% setup share** — the v2 "operation-bound,
    9% setup" conclusion is withdrawn and the design implications reversed). F12 regenerated per-swath.
    F20 axis USD → EUR. F21 retitled "annual energy balance". phase-7 sweep now also records additive NPV.

## B. Manuscript rewrites (v3 tex, 68 pp, compiles clean, 0 undefined refs)

- **Abstract, key results, C1–C4, Results, envelope, Discussion, Conclusion** rewritten to the corrected,
  scale-conditional story: ~half trafficked area at 4.6× lower pressure (39× field-integrated), 2.9×/10×
  energy ratios (basis stated), off-grid climate-conditional but nowhere categorical, economics cost-neutral
  at 25 ha/yr and positive ≥40 ha/yr. Where a v2 claim was wrong, v3 says so explicitly ("an earlier revision…
  is withdrawn") rather than silently changing numbers.
- **New Methods subsections:** *Anchoring energy, power, and time* (§ anchoring-energy) and *Carriage auxiliary
  power* (mower PTO ~6–7.5 kW, sprayer pump, payloads — scoped out of headline claims and into limitations).
- **Related Work rebuilt:** steam cable ploughing (Tyler & Haining; Lane) moved up from Limitations and cited;
  winch-assist forestry; CDPR literature (Pott, RoboCrane, ETH FIP); **new CTF/gantry subsection**
  (Tullberg, Chamen, McPhee) — compaction claim now positioned as "extreme CTF", not two orders of magnitude.
- **Anchor chapter:** Qureshi et al. (2024) authorship corrected everywhere (was "Khand"); Magnum deflection
  basis corrected (0.5-inch); MU-side check and sheave-height overturning couple added; head-fixity condition
  stated as load-bearing.
- **Mechanics honesty:** carriage described as soil-supported and cable-towed throughout (not "rolls along the
  cable"); cable topology specified (continuous loop over the Anchor sheave — makes the return leg and regen
  physically consistent); eq. regen gets max(0,·) and defined symbols; regen story unified (slope-averaged
  ~3.5%, ≈0 flat); block repositioning + module traffic lanes acknowledged and counted (compaction caveat).
- **Draft model disclosures:** category-lumped texture factors, the non-standard moisture multiplier (formula
  given), actual sampler windows (1–2.5 / 5–9 km/h, clamped depths), sub-calibration-speed extrapolation caveat,
  Kheiralla comparison scoped to the conventional library.
- **New content:** per-site off-grid table (tab:site-summary); timeliness constraint (≈21 days per pass at
  25 ha vs seeding windows) in Discussion + Limitations; **lateral guidance/row registration added to the risk
  register** (ranked alongside depth control); anchoring-model limitations (sand-only, static, ad hoc SFs)
  + torque-verification QA recommendation; labour asymmetry discussed in the tornado reading.
- **Verification studies:** S7 rewritten for the 1.0-m screw; S4 restated (structural NPV invariance, 0.34×
  vs 0.37× statistics disambiguated, new sweep numbers); S2 recoil quoted at operating tension (6.3 m/s, 22 J,
  sub-threshold) with the 25 m/s figure re-scoped to fault conditions; S3 tilth corrected to pipeline values
  (10× less cross-section at 0.65× draft) and "settled" bed density.
- **Front/back matter:** affiliation updated (Adana Alparslan Türkeş STU); single email; AI-render disclosure
  on the hero figure; F0d annotations corrected (50%@100–250 kPa vs 27%@31 kPa); glossary cleaned (ICE collision
  resolved, LCO defined, DEM/MPC/PID added, dead entries removed); "authors" → "author's"; 14 new references
  (51 total); competitor table corrected (ecoRobotix AVO class, Naïo Oz, 3-kW rating, scale-dependent payback).

## C. Decisions the author should review (defensible defaults chosen under full autonomy)

1. **Implement-set netting** (€9,500 on both sides). Alternative framings (used implements retained, or
   codesigned implements dearer) shift the reference NPV by roughly ∓€1k per €1k of set delta.
2. **Ground-screw respec** (1.0 m / 60 mm / 90 N·m). Chosen because it uniquely reconciles capacity, time,
   power, and energy budgets *and* internally derives both v2 envelope bounds. A heavier anchor is retained
   as an option in the text.
3. **Flat 5%/4% maintenance basis kept as default** (justified as calendar-dominated for an aged used machine),
   with the hour-based EP496 sensitivity disclosed — it would push the 25-ha reference further negative.
4. **Seasonal duty window** (Apr 1–Sep 17 N / Oct 1–Mar 19 S, contiguous 170 days). A split-season calendar
   would change per-site grid hours modestly.
5. **Mower and rotary hoe scoped out of the headline envelope** (carriage power / efficacy at cable speeds)
   rather than re-costed — reinstating them requires a powered-carriage variant design.

## D. Remaining known work (not blocking)

- Cosmetic figure pass: strip remaining embedded titles/F-numbers from PNGs not touched in this run
  (F13–F17, F23–F28 backgrounds), colour-blind-safe palettes, snake_case label mapping.
- The journal-length restructure (~10k words, strict IMRaD, supplementary split) per the review's §10 —
  v3 is the corrected full-length (arXiv-lineage) version from which that condensation should be made.
- Zenodo DOI reservation; competitor-row primary-source URLs with access dates.

## E. Addendum (2026-07-06): Variant 2 — Multi-strip anchoring (beam + travelling sheave)

Added the quantified "v4 concept" as a variant, keeping the fully-anchored baseline as the reference design:

- **Model** (`variants.py: MultiStripAnchorSpec/multi_strip_anchor_params`): anchor once per k-strip block;
  per-strip trolley index ~15 s / ~0 Wh; full 13-screw cycle (75 s, 25.3 Wh) once per block; +€1,500 beam capex.
  MU holds by braked wheels between blocks (5.5–7 kN friction ≥ 3.0 kN P90 on firm ground).
- **Load-path check** (`anchoring.py: beam_yaw_per_screw_N`): trolley-end yaw moment resisted as differential
  lateral screw loads. At P90 3.0 kN: k=2 → 0.63 kN/screw, k=4 → 1.9 kN (bottom of the frame-coupled working
  range), k=6 → 3.1 kN (exceeds it). Honest envelope: k=4 medium-dense frame-coupled, k=2 loose sand.
- **Pipeline row (k=4)**: 14.5 daa/day (1.21×), 974 Wh/daa (0.79×), €40,170 (1.04×), simple payback 148.7 mo
  (0.82× — best in table), 891 W surplus. F20 regenerated with 4 bars; tests extended (20 pass).
- **Manuscript**: new §"Variant 2 — Multi-strip anchoring" (mechanism, two-tier cycle, yaw limit, verification
  gate); variants renumbered (CT+ → Variant 3, unidirectional → Variant 4); intro/table/takeaways/Results 7.6/
  layout-findings/anchoring-energy/Conclusion/appendix table updated. The naive swivelling-sheave realisation is
  explicitly documented as geometrically unsound (fan overlap near the pivot) and superseded by the beam.
- **Status framing**: baseline remains the reference (its budgets are fully verified); the beam variant is the
  quantified upside, gated on yaw-moment/frame-coupling verification and braked-wheel holding on wet/sloped
  headlands — named as prototype-campaign test items.
