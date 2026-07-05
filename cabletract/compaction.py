"""Phase 4 — Soil-compaction model.

The slide deck claims CableTract reduces soil compaction because the
heavy Main Unit and Anchor stay at the field edge while only a light
implement carriage rolls across the working area. The v1 model never
quantifies this. This module replaces hand-waving with a defensible
two-component metric:

1. **Compacted area** — the total ground-plan area subjected to a
   wheel/track contact pressure above a soil-specific threshold. For a
   conventional tractor this is the entire pass plan; for CableTract it
   is only the implement-carriage path width × strip count.
2. **Compacted-energy index** — a Söhne-inspired stress integrand that
   weights every compacted m² by its contact pressure squared, capturing
   the fact that a 200 kPa tractor tyre damages the soil far more than
   a 30 kPa CableTract carriage roller, even on the same area.

We deliberately do *not* attempt a calibrated SoilFlex / APSoil
simulation. The point of the figure is the order-of-magnitude
comparison, not a soil-rheology paper.

v3 accounting note: v2 billed the carriage one roller track per 50-m
*span band* (one traverse per anchor placement), which under-counted
carriage trafficking by a factor of span/swath (~33x) — the carriage
must traverse every swath to work the field. v3 bills one traverse per
swath: trafficked fraction = total track width / swath width
(0.4 m / 1.5 m ~ 27 % per pass vs the tractor's ~50 %). The defensible
findings are therefore:

- CableTract reduces trafficked *area* by roughly half vs a tractor on
  the same field (~27 % vs ~50 % of field area per pass);
- reduces mean contact pressure ~4.6x (31 vs 143 kPa, below the ~50 kPa
  subsoil-stress criterion);
- and reduces the field-integrated p^2-weighted exposure
  (mean pressure squared x trafficked area x passes) by ~40x.

References:
    Söhne, W. (1953). "Druckverteilung im Boden und Bodenverformung
        unter Schlepperreifen." Grundlagen der Landtechnik 5, 49–63.
    Keller, T. & Lamandé, M. (2010). "Challenges in the development of
        analytical soil compaction models." Soil & Tillage Research 111.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union

from cabletract.layout import strip_decomposition, strip_plan_segments_xy


# ---------------------------------------------------------------------------
# Vehicle / carriage parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WheelSpec:
    """One contact patch on a vehicle. Used for both tractor tyres and
    CableTract carriage rollers."""

    name: str
    load_N: float           # static load on this contact patch
    contact_area_m2: float  # tyre/roller footprint area
    track_width_m: float    # rolling width of the patch (visible footprint)


@dataclass(frozen=True)
class VehicleSpec:
    """A complete ground-contacting vehicle: list of wheels + lateral
    spacing between left and right tracks."""

    name: str
    wheels: Tuple[WheelSpec, ...]
    track_gauge_m: float    # distance between left and right wheel centerlines
    pass_count: int = 1     # number of passes per cropping season


def contact_pressure(load_N: float, contact_area_m2: float) -> float:
    """Static ground contact pressure (Pa). Trivial but exposed for tests."""
    if contact_area_m2 <= 0.0:
        raise ValueError("contact_area_m2 must be positive")
    return load_N / contact_area_m2


# ---------------------------------------------------------------------------
# Reference vehicles
# ---------------------------------------------------------------------------

# Conventional ~80 hp 4WD utility tractor (≈ 4 t total mass with implement)
# 4 wheels, rear duals carrying ~60% of weight at ~150-220 kPa.
TRACTOR_REFERENCE = VehicleSpec(
    name="reference_tractor_80hp",
    wheels=(
        WheelSpec("front_left",  load_N=8000.0,  contact_area_m2=0.060, track_width_m=0.30),
        WheelSpec("front_right", load_N=8000.0,  contact_area_m2=0.060, track_width_m=0.30),
        WheelSpec("rear_left",   load_N=12000.0, contact_area_m2=0.080, track_width_m=0.45),
        WheelSpec("rear_right",  load_N=12000.0, contact_area_m2=0.080, track_width_m=0.45),
    ),
    track_gauge_m=1.80,
    pass_count=4,  # plough, secondary cultivation, drill, sprayer
)

# CableTract Main Unit + Anchor never enter the field; they sit on
# headland strips. Only the implement carriage rolls across the soil:
# small lightweight steel cylinder pulling a tool. ~250 kg total at low
# pressure.
CABLETRACT_CARRIAGE = VehicleSpec(
    name="cabletract_carriage",
    wheels=(
        # Two wide rollers in tandem
        WheelSpec("roller_front", load_N=1250.0, contact_area_m2=0.040, track_width_m=0.20),
        WheelSpec("roller_rear",  load_N=1250.0, contact_area_m2=0.040, track_width_m=0.20),
    ),
    track_gauge_m=0.60,
    pass_count=4,
)


# ---------------------------------------------------------------------------
# Compaction summary for one (field, vehicle) pair
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompactionSummary:
    """Per-vehicle, per-field compaction metric set."""

    vehicle: str
    field_area_m2: float
    compacted_area_m2: float
    compacted_area_frac: float
    mean_pressure_kPa: float
    max_pressure_kPa: float
    compaction_energy_index: float  # ∫ p² dA over the contact patches (kPa² m²), per-running-gear
    field_p2_index: float           # mean_p² × trafficked area × passes (kPa² m²), field-integrated
    pass_count: int


def _vehicle_track_total_width(vehicle: VehicleSpec) -> float:
    """Total ground-contact track width per pass for one vehicle.

    For a 4-wheel tractor the rear (wider) tyres overrun the front
    tyres, so the ground footprint per pass is 2 × max(track_widths).
    For a single-axle carriage it's 2 × roller width."""
    if not vehicle.wheels:
        return 0.0
    return 2.0 * max(w.track_width_m for w in vehicle.wheels)


def compaction_summary_for_vehicle(
    polygon: Polygon,
    vehicle: VehicleSpec,
    span: float = 50.0,
    swath: float = 1.5,
) -> CompactionSummary:
    """Compute compaction metrics for one vehicle running over one field.

    The model:

    - For a CableTract-style carriage, the swath plan from
      :func:`cabletract.layout.strip_decomposition` defines exactly which
      strips are traversed; total compacted area = sum_strip(piece_length
      × track_total_width).
    - For a conventional tractor, every square metre of the polygon is
      contacted at least once per pass (worst case for compaction). We
      bound it at the polygon area itself, multiplied by the per-pass
      ground-coverage fraction (track_total_width / track_gauge), then
      capped at the polygon area.
    - Mean pressure is the load-weighted mean over all wheels.
    - Compacted-energy index sums p² × patch area over every wheel,
      times the number of passes, normalised to (kPa² m²).
    """
    if polygon.is_empty or polygon.area <= 0.0:
        return CompactionSummary(vehicle.name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vehicle.pass_count)

    track_total_w = _vehicle_track_total_width(vehicle)

    if vehicle.name.startswith("cabletract"):
        # The carriage traverses EVERY swath of each covered strip piece
        # (one 50-m traverse per `swath` of lateral step), so the rolled
        # path per pass is piece_area / swath and the trafficked area is
        # piece_area × (track_total_width / swath). v2 wrongly billed one
        # traverse per 50-m span band (see module docstring).
        segs = strip_decomposition(polygon, span=span, swath=swath, orientation_deg=0.0)
        per_pass_frac = min(track_total_w / max(swath, 1e-9), 1.0)
        compacted = sum(s.area_m2 for s in segs) * per_pass_frac * vehicle.pass_count
    else:
        # Tractor: per-pass coverage = track_total_width / track_gauge
        per_pass_frac = min(track_total_w / max(vehicle.track_gauge_m, 1e-9), 1.0)
        compacted = polygon.area * per_pass_frac * vehicle.pass_count

    compacted = min(compacted, polygon.area * vehicle.pass_count)
    pressures_Pa = [contact_pressure(w.load_N, w.contact_area_m2) for w in vehicle.wheels]
    loads = np.array([w.load_N for w in vehicle.wheels])
    mean_p_Pa = float(np.average(pressures_Pa, weights=loads))
    max_p_Pa = float(max(pressures_Pa))

    # Per-running-gear index Σ p² · A_patch · pass_count, in (kPa² m²)
    energy = 0.0
    for w, p in zip(vehicle.wheels, pressures_Pa):
        energy += (p / 1000.0) ** 2 * w.contact_area_m2
    energy *= vehicle.pass_count

    # Field-integrated p²-weighted exposure: mean pressure squared times
    # total trafficked area (already includes pass_count), in kPa² m².
    field_p2 = (mean_p_Pa / 1000.0) ** 2 * float(compacted)

    return CompactionSummary(
        vehicle=vehicle.name,
        field_area_m2=float(polygon.area),
        compacted_area_m2=float(compacted),
        compacted_area_frac=float(compacted / (polygon.area * vehicle.pass_count)),
        mean_pressure_kPa=mean_p_Pa / 1000.0,
        max_pressure_kPa=max_p_Pa / 1000.0,
        compaction_energy_index=float(energy),
        field_p2_index=field_p2,
        pass_count=vehicle.pass_count,
    )


def compare_vehicles_on_field(
    polygon: Polygon,
    span: float = 50.0,
    swath: float = 1.5,
) -> List[CompactionSummary]:
    """Convenience: run both reference vehicles against one field."""
    return [
        compaction_summary_for_vehicle(polygon, TRACTOR_REFERENCE, span=span, swath=swath),
        compaction_summary_for_vehicle(polygon, CABLETRACT_CARRIAGE, span=span, swath=swath),
    ]


def compacted_path_polygon(
    polygon: Polygon,
    vehicle: VehicleSpec,
    span: float = 50.0,
    swath: float = 1.5,
) -> Polygon:
    """Return a buffered union of the carriage roller tracks, intersected
    with the field. Used for the F12 visualisation.

    For a tractor, returns the entire field polygon (whole-field contact
    over a season). For the CableTract carriage, returns one
    ``track_total_width``-wide band per swath traverse — i.e. a track
    every ``swath`` metres across the field, matching the corrected
    per-swath trafficking accounting.
    """
    if not vehicle.name.startswith("cabletract"):
        return polygon
    track_total_w = _vehicle_track_total_width(vehicle)
    minx, miny, maxx, maxy = polygon.bounds
    bands = []
    y = miny + swath / 2.0
    while y < maxy:
        line = LineString([(minx - 1.0, y), (maxx + 1.0, y)])
        bands.append(line.buffer(track_total_w / 2.0, cap_style=2))
        y += swath
    if not bands:
        return Polygon()
    union = unary_union(bands)
    return union.intersection(polygon)
