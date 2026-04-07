"""Procedural generator for the bundled Phase-4 field corpus.

Generates ~50 polygons covering the four shape classes the manuscript
needs (rectangle, L-shape, irregular convex, irregular concave/obstacle).
The output is a single GeoJSON FeatureCollection at
``cabletract/data/fields/fields.geojson`` with ``properties`` containing
``id``, ``shape_class``, ``nominal_area_ha``, ``description``.

All coordinates are in metres in a local Cartesian frame; CableTract's
shape-efficiency calculation does not depend on a geographic projection,
and shipping a local-metres corpus avoids a hard pyproj dependency.

Run as:

    python cabletract/data/fields/build_corpus.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

OUT = Path(__file__).resolve().parent / "fields.geojson"

Coord = Tuple[float, float]
Ring = List[Coord]


def _close(ring: Ring) -> Ring:
    if ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring


def _feature(idx: int, shape_class: str, exterior: Ring, holes: List[Ring], desc: str) -> dict:
    geom_coords = [_close(list(map(tuple, exterior)))] + [_close(list(map(tuple, h))) for h in holes]
    # Compute area via the shoelace formula minus holes
    def _area(ring: Ring) -> float:
        n = len(ring) - 1
        s = 0.0
        for i in range(n):
            x0, y0 = ring[i]
            x1, y1 = ring[i + 1]
            s += x0 * y1 - x1 * y0
        return abs(s) / 2.0
    a = _area(geom_coords[0]) - sum(_area(h) for h in geom_coords[1:])
    return {
        "type": "Feature",
        "properties": {
            "id": f"field_{idx:03d}",
            "shape_class": shape_class,
            "nominal_area_ha": round(a / 10000.0, 4),
            "description": desc,
        },
        "geometry": {"type": "Polygon", "coordinates": geom_coords},
    }


def _rectangle(w: float, h: float) -> Ring:
    return [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]


def _l_shape(w: float, h: float, cut_w: float, cut_h: float) -> Ring:
    """L-shape with a rectangular bite taken out of the top-right corner."""
    return [
        (0.0, 0.0),
        (w, 0.0),
        (w, h - cut_h),
        (w - cut_w, h - cut_h),
        (w - cut_w, h),
        (0.0, h),
    ]


def _irregular_convex(rng: np.random.Generator, n: int, radius: float) -> Ring:
    """Random convex polygon: pick angles, sort, push outward by random radii.

    Guaranteed convex because we sweep monotonically around the centre."""
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, n))
    radii = rng.uniform(0.7 * radius, 1.0 * radius, n)
    return [(float(r * math.cos(a)), float(r * math.sin(a))) for a, r in zip(angles, radii)]


def _irregular_concave(rng: np.random.Generator, n: int, radius: float, concavity: float) -> Ring:
    """Star-shaped polygon: alternating long/short radii. concavity in (0,1)."""
    angles = np.sort(rng.uniform(0.0, 2.0 * math.pi, n))
    radii = []
    for i in range(n):
        base = rng.uniform(0.85 * radius, 1.0 * radius)
        if i % 2 == 1:
            base *= 1.0 - concavity * rng.uniform(0.4, 0.9)
        radii.append(base)
    return [(float(r * math.cos(a)), float(r * math.sin(a))) for a, r in zip(angles, radii)]


def _circle_hole(cx: float, cy: float, r: float, n: int = 16) -> Ring:
    """Approximate a circular obstacle (well, pond, pylon) by an n-gon."""
    return [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def build() -> None:
    rng = np.random.default_rng(0xF1E1D)
    features: List[dict] = []
    idx = 0

    # --- Rectangles (10 of them, 0.5 to 8 ha) ---
    for w, h in [(50, 100), (75, 100), (100, 100), (50, 200), (60, 150), (80, 200), (100, 200), (120, 200), (150, 200), (200, 400)]:
        idx += 1
        features.append(_feature(idx, "rectangle", _rectangle(w, h), [], f"axis-aligned {w}x{h} m"))

    # --- L-shapes (10) ---
    for w, h, cw, ch in [
        (100, 100, 30, 30),
        (120, 100, 40, 30),
        (150, 120, 50, 40),
        (100, 80, 25, 35),
        (200, 150, 60, 50),
        (140, 100, 30, 50),
        (180, 140, 80, 60),
        (160, 110, 40, 30),
        (220, 180, 100, 70),
        (250, 200, 120, 80),
    ]:
        idx += 1
        features.append(_feature(idx, "L_shape", _l_shape(w, h, cw, ch), [], f"L {w}x{h} cut {cw}x{ch}"))

    # --- Irregular convex (10) ---
    for k in range(10):
        idx += 1
        n = int(rng.integers(6, 12))
        radius = float(rng.uniform(40.0, 110.0))
        ring = _irregular_convex(rng, n, radius)
        features.append(_feature(idx, "irregular_convex", ring, [], f"convex n={n} r~{radius:.0f}"))

    # --- Irregular concave with optional obstacles (15) ---
    for k in range(15):
        idx += 1
        n = int(rng.integers(8, 16))
        radius = float(rng.uniform(50.0, 130.0))
        concavity = float(rng.uniform(0.20, 0.55))
        ring = _irregular_concave(rng, n, radius, concavity)

        holes: List[Ring] = []
        if k >= 5:  # last 10 fields get one or two interior obstacles
            n_holes = 1 if k < 10 else 2
            for _ in range(n_holes):
                # Place obstacle near the centroid; small radius vs field
                hr = float(rng.uniform(5.0, 12.0))
                hx = float(rng.uniform(-radius * 0.3, radius * 0.3))
                hy = float(rng.uniform(-radius * 0.3, radius * 0.3))
                holes.append(_circle_hole(hx, hy, hr))
        features.append(
            _feature(
                idx, "irregular_concave", ring, holes,
                f"concave n={n} c={concavity:.2f} holes={len(holes)}",
            )
        )

    # --- 5 "real-shape" hand-tuned polygons (synthetic but plausible) ---
    # 1. Long narrow strip (river-edge field)
    idx += 1
    features.append(_feature(idx, "real_shape", [(0, 0), (320, -10), (340, 30), (15, 35)], [], "river-edge strip"))
    # 2. Trapezoidal field (typical road-frontage)
    idx += 1
    features.append(_feature(idx, "real_shape", [(0, 0), (180, 0), (160, 110), (20, 110)], [], "road-frontage trapezoid"))
    # 3. Pentagon (5-side property line)
    idx += 1
    features.append(_feature(idx, "real_shape", [(0, 0), (150, 0), (180, 80), (90, 140), (-20, 80)], [], "pentagon plot"))
    # 4. Wedge (curve approximated)
    idx += 1
    wedge = [(0, 0)] + [(120 * math.cos(t), 120 * math.sin(t)) for t in np.linspace(0, math.pi / 2, 12)]
    features.append(_feature(idx, "real_shape", wedge, [], "quarter-circle wedge"))
    # 5. Two-lobe peanut (concave waist)
    idx += 1
    peanut: Ring = []
    for t in np.linspace(0, 2 * math.pi, 32, endpoint=False):
        r = 80.0 + 30.0 * math.cos(2 * t)  # two-lobe envelope
        peanut.append((float(r * math.cos(t)), float(r * math.sin(t))))
    features.append(_feature(idx, "real_shape", peanut, [], "two-lobe peanut"))

    fc = {"type": "FeatureCollection", "features": features}
    OUT.write_text(json.dumps(fc, indent=2))
    print(f"Wrote {len(features)} features to {OUT}")
    classes = {}
    for f in features:
        c = f["properties"]["shape_class"]
        classes[c] = classes.get(c, 0) + 1
    for c, n in sorted(classes.items()):
        print(f"  {c}: {n}")
    total_ha = sum(f["properties"]["nominal_area_ha"] for f in features)
    print(f"Total area: {total_ha:.2f} ha")


if __name__ == "__main__":
    build()
