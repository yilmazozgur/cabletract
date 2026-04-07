"""Phase 4 — Field-geometry coverage planner.

The v1 model treats field geometry with a single scalar
``shape_efficiency`` (default 0.85). A square and an L-field with the
same scalar look identical to the model, even though the L-field
forces the cable robot to anchor twice (once per arm of the L) and
loses headland to the inside corner.

This module replaces the scalar with a real strip decomposition over
arbitrary polygons:

1. ``load_field_corpus`` reads the bundled GeoJSON FeatureCollection at
   ``cabletract/data/fields/fields.geojson`` and returns a list of
   shapely Polygons + properties.
2. ``strip_decomposition(polygon, span, swath, orientation)`` rotates
   the polygon, sweeps a series of swath-wide strips parallel to the
   chosen axis, clips each strip to the polygon (including holes), and
   returns the resulting list of working segments. Each segment carries
   the strip index, working length (m), and whether it required a fresh
   anchor setup (a strip whose clip yields more than one disconnected
   piece counts each piece as a separate setup).
3. ``effective_shape_efficiency(polygon, span, swath, orientation)``
   collapses a strip plan into the v1-compatible scalar
   ``working_length / (n_setups * span)``.
4. ``best_orientation_efficiency(polygon, span, swath, n_orient=12)``
   sweeps 12 candidate sweep directions and returns the maximum
   shape-efficiency together with the chosen orientation.
5. ``farm_tour(field_centroids, depot)`` solves a small inter-field
   travelling-salesman tour with a nearest-neighbour heuristic, used by
   F11's daily-time-budget pie chart.

The intent is *not* a fully optimal coverage-path planner; it is a
defensible decomposition that gives a representative shape-efficiency
distribution over a corpus of 50 fields. Phase 6 (the random-forest
polygon predictor) will then learn this map directly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.ops import unary_union

FIELDS_GEOJSON = Path(__file__).resolve().parent / "data" / "fields" / "fields.geojson"


# ---------------------------------------------------------------------------
# Field corpus loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Field:
    """One field record from the bundled GeoJSON corpus."""

    id: str
    shape_class: str
    nominal_area_ha: float
    description: str
    polygon: Polygon


def load_field_corpus(geojson_path: Path = FIELDS_GEOJSON) -> List[Field]:
    """Load every Polygon feature from the bundled GeoJSON FeatureCollection.

    Geometries are buffered by 0 to repair any self-intersections introduced
    by the procedural corpus generator. Non-polygon results after the buffer
    fix-up (e.g. degenerate input) are dropped silently."""
    data = json.loads(geojson_path.read_text())
    out: List[Field] = []
    for feat in data["features"]:
        geom = shape(feat["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not isinstance(geom, Polygon) or geom.is_empty:
            continue
        props = feat.get("properties", {})
        out.append(
            Field(
                id=str(props.get("id", "")),
                shape_class=str(props.get("shape_class", "")),
                nominal_area_ha=float(props.get("nominal_area_ha", geom.area / 10000.0)),
                description=str(props.get("description", "")),
                polygon=geom,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Strip decomposition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StripSegment:
    """One working segment produced by clipping a sweep strip to a polygon."""

    strip_index: int        # 0..n_strips-1
    piece_index: int        # 0..k-1 within this strip if the clip is disconnected
    length_m: float         # bounding-box axial extent of this piece (m)
    area_m2: float          # actual covered area of this piece (m²)
    needs_setup: bool       # True for piece_index==0 of every strip and any extra piece


def _polygon_pieces(geom) -> List[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if hasattr(geom, "geoms"):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_polygon_pieces(g))
        return out
    return []


def strip_decomposition(
    polygon: Polygon,
    span: float,
    swath: float = 2.0,
    orientation_deg: float = 0.0,
) -> List[StripSegment]:
    """Decompose ``polygon`` into a sequence of swath-wide working strips.

    The CableTract Main Unit and Anchor are placed on opposite sides of a
    band of width ``span`` (the cable length); the implement carriage works
    inside that band along strips that are ``swath`` wide. We approximate
    a single anchor placement as covering one ``span``-wide vertical strip
    of the rotated polygon, and we tile non-overlapping strips of width
    ``span`` to cover the bounding box. Within each strip we clip a swath
    of full strip-width to the polygon and sum the resulting axial extent.

    Parameters
    ----------
    polygon : shapely.Polygon
        Field outline (with optional interior holes).
    span : float
        Cable length / Anchor-to-Main-Unit distance, m.
    swath : float
        Working width per traverse, m. Mostly cosmetic for the working-length
        calculation; needed for area accounting.
    orientation_deg : float
        Sweep direction. 0 deg = strips run parallel to the X axis.

    Returns
    -------
    list[StripSegment]
        One entry per disconnected piece of every strip.
    """
    if span <= 0.0:
        raise ValueError("span must be positive")
    if swath <= 0.0:
        raise ValueError("swath must be positive")

    if polygon.is_empty or polygon.area <= 0.0:
        return []
    # Rotate the polygon so the sweep direction aligns with X
    rot = rotate(polygon, -orientation_deg, origin="centroid", use_radians=False)
    if not rot.is_valid:
        rot = rot.buffer(0)
    minx, miny, maxx, maxy = rot.bounds
    width_y = maxy - miny
    n_strips = max(1, int(math.ceil(width_y / span)))

    segments: List[StripSegment] = []
    for i in range(n_strips):
        y0 = miny + i * span
        y1 = min(y0 + span, maxy)
        strip = Polygon([(minx, y0), (maxx, y0), (maxx, y1), (minx, y1)])
        try:
            clipped = rot.intersection(strip)
        except Exception:
            clipped = rot.buffer(0).intersection(strip)
        pieces = _polygon_pieces(clipped)
        for j, piece in enumerate(pieces):
            pminx, pminy, pmaxx, pmaxy = piece.bounds
            length = pmaxx - pminx
            area = piece.area
            if length <= 0.0 or area <= 0.0:
                continue
            segments.append(
                StripSegment(
                    strip_index=i,
                    piece_index=j,
                    length_m=float(length),
                    area_m2=float(area),
                    needs_setup=True,  # every disconnected piece needs an anchor reset
                )
            )
    return segments


def effective_shape_efficiency(
    polygon: Polygon,
    span: float,
    swath: float = 2.0,
    orientation_deg: float = 0.0,
) -> float:
    """Return the v1-compatible scalar shape efficiency in [0, 1].

    Defined as

        η = polygon.area / Σ_i (span × bounding-box-length_i)

    where the sum runs over every disconnected strip-clip piece. The
    denominator is the *cable-bound bounding box* the Anchor must fence
    off to cover that piece (one full ``span`` per setup, times the axial
    extent the cable must traverse). The numerator is the *useful*
    polygon area inside that envelope. A unit square with span exactly
    equal to the side gives 1.0. An L-shape, irregular concave field, or
    field with interior holes all give < 1.0 because each disconnected
    piece bills a full ``span × bounds_length`` rectangle but covers
    less than that area in actual ground.
    """
    segs = strip_decomposition(polygon, span, swath, orientation_deg)
    if not segs:
        return 0.0
    bound_total = sum(span * s.length_m for s in segs)
    if bound_total <= 0.0:
        return 0.0
    return float(min(polygon.area / bound_total, 1.0))


def best_orientation_efficiency(
    polygon: Polygon,
    span: float,
    swath: float = 2.0,
    n_orient: int = 12,
) -> Tuple[float, float]:
    """Return ``(best_eta, best_orientation_deg)`` over a sweep of n_orient angles."""
    best_eta = 0.0
    best_deg = 0.0
    for k in range(n_orient):
        deg = 180.0 * k / n_orient
        eta = effective_shape_efficiency(polygon, span, swath, deg)
        if eta > best_eta:
            best_eta = eta
            best_deg = deg
    return best_eta, best_deg


# ---------------------------------------------------------------------------
# Strip plan visualisation helper
# ---------------------------------------------------------------------------

def strip_plan_segments_xy(
    polygon: Polygon,
    span: float,
    swath: float = 2.0,
    orientation_deg: float = 0.0,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Return ((x0,y0), (x1,y1)) midline segments for every strip piece, in
    the *original* (un-rotated) polygon coordinate frame."""
    rot = rotate(polygon, -orientation_deg, origin="centroid", use_radians=False)
    if not rot.is_valid:
        rot = rot.buffer(0)
    minx, miny, maxx, maxy = rot.bounds
    width_y = maxy - miny
    n_strips = max(1, int(math.ceil(width_y / span)))

    midlines: List[LineString] = []
    for i in range(n_strips):
        y0 = miny + i * span
        y1 = min(y0 + span, maxy)
        ymid = 0.5 * (y0 + y1)
        strip = Polygon([(minx, y0), (maxx, y0), (maxx, y1), (minx, y1)])
        try:
            clipped = rot.intersection(strip)
        except Exception:
            clipped = rot.buffer(0).intersection(strip)
        for piece in _polygon_pieces(clipped):
            pminx, _, pmaxx, _ = piece.bounds
            line = LineString([(pminx, ymid), (pmaxx, ymid)])
            midlines.append(line)

    # Rotate the midlines back to the original frame
    out: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    centroid = polygon.centroid
    for ln in midlines:
        rotated_back = rotate(ln, orientation_deg, origin=centroid, use_radians=False)
        coords = list(rotated_back.coords)
        out.append((tuple(coords[0]), tuple(coords[1])))
    return out


# ---------------------------------------------------------------------------
# Farm tour (inter-field travel)
# ---------------------------------------------------------------------------

def farm_tour(
    centroids: Sequence[Tuple[float, float]],
    depot: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, List[int]]:
    """Nearest-neighbour TSP tour starting and ending at ``depot``.

    Returns ``(total_distance_m, visit_order)`` where ``visit_order`` is the
    sequence of field indices in the order visited (depot is implicit at
    both ends).
    """
    n = len(centroids)
    if n == 0:
        return 0.0, []
    remaining = set(range(n))
    order: List[int] = []
    pos = depot
    total = 0.0
    while remaining:
        nxt = min(remaining, key=lambda i: math.hypot(centroids[i][0] - pos[0], centroids[i][1] - pos[1]))
        d = math.hypot(centroids[nxt][0] - pos[0], centroids[nxt][1] - pos[1])
        total += d
        pos = centroids[nxt]
        order.append(nxt)
        remaining.discard(nxt)
    # Return-to-depot leg
    total += math.hypot(pos[0] - depot[0], pos[1] - depot[1])
    return total, order


# ---------------------------------------------------------------------------
# Convenience: corpus-wide shape efficiency summary
# ---------------------------------------------------------------------------

def corpus_shape_efficiency_summary(
    fields: Iterable[Field],
    span: float,
    swath: float = 2.0,
    n_orient: int = 12,
) -> List[dict]:
    """Compute (best, fixed-X) shape efficiency for every field in the corpus."""
    rows: List[dict] = []
    for f in fields:
        eta_x = effective_shape_efficiency(f.polygon, span, swath, orientation_deg=0.0)
        eta_best, deg_best = best_orientation_efficiency(f.polygon, span, swath, n_orient=n_orient)
        rows.append(
            {
                "id": f.id,
                "shape_class": f.shape_class,
                "area_ha": f.nominal_area_ha,
                "span_m": span,
                "eta_x": eta_x,
                "eta_best": eta_best,
                "best_orientation_deg": deg_best,
            }
        )
    return rows
