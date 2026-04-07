"""Phase 2 — ASABE D497.7 implement library and stochastic draft sampler.

The v1 model (``cabletract_analysis.py:compare_operation_types``) used five
hand-picked single-point draft loads. A reviewer will (rightly) ask: where
do those five numbers come from, and what is the operating-condition
distribution that produced them? This module replaces them with the
ASABE D497.7 implement-draft library and a stochastic sampler that
propagates speed, depth, soil texture, and moisture variability into a
draft distribution.

Reference equation (ASABE D497.7, 2011, R2015):

    D = F_i * [A + B * S + C * S^2] * W * T

where
    D     = implement draft load (N)
    F_i   = soil-texture multiplier (dimensionless)
    A     = constant draft term (N per width unit per depth unit)
    B     = linear speed term (N s / km per width unit per depth unit)
    C     = quadratic speed term (N s^2 / km^2 per width unit per depth unit)
    S     = field operating speed (km/h)
    W     = width count in the implement's natural unit (m, tools, rows, ...)
    T     = working depth (cm) for depth-dependent tools, else 1.

A multiplicative moisture factor ``F_m`` is applied on top to capture the
U-shaped sensitivity of soil draft to soil moisture content (dry/hard and
wet/sticky both increase draft). This is an empirical layer that ASABE D497
itself does not include explicitly.

Soil texture categories follow ASABE D497 Table 3:
    'fine'   — clay, silty clay, clay loam       (F_i = 1.00 for primary tillage)
    'medium' — loam, silt loam, sandy clay loam  (F_i ≈ 0.85)
    'coarse' — sandy loam, sand, loamy sand      (F_i ≈ 0.55)

Secondary tillage tools have a slightly steeper texture sensitivity. Seeding
and spraying are essentially texture-independent.

The library is loaded from ``cabletract/data/asabe_d497.csv``; coefficients
are taken from D497.7 Table 4 plus the Grisso et al. (2007) Virginia Tech
extension fact sheet that republishes the same numbers in metric units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ASABE_CSV = Path(__file__).resolve().parent / "data" / "asabe_d497.csv"
CODESIGN_CSV = Path(__file__).resolve().parent / "data" / "asabe_d497_cabletract.csv"


# ---------------------------------------------------------------------------
# Soil texture and moisture multipliers
# ---------------------------------------------------------------------------

# ASABE D497 F_i values, broken out by implement category. Values within each
# category are the central estimates used for the texture×category lookup;
# Phase 5 will sample around them.
TEXTURE_FACTORS: Dict[str, Dict[str, float]] = {
    "primary_tillage": {"fine": 1.00, "medium": 0.85, "coarse": 0.55},
    "secondary_tillage": {"fine": 1.00, "medium": 0.78, "coarse": 0.45},
    "seeding": {"fine": 1.00, "medium": 0.95, "coarse": 0.85},
    "spraying": {"fine": 1.00, "medium": 1.00, "coarse": 1.00},
    "mowing": {"fine": 1.00, "medium": 0.95, "coarse": 0.90},
    "weeding": {"fine": 1.00, "medium": 0.92, "coarse": 0.80},
}

MOISTURE_OPTIMUM = 0.20  # gravimetric water content where draft is minimum
MOISTURE_SCALE = 0.20    # half-width of the U-shape (dimensionless)


def texture_factor(soil_texture: str, category: str) -> float:
    """Return the dimensionless ASABE F_i multiplier for a (category, texture)."""
    cat = TEXTURE_FACTORS.get(category)
    if cat is None:
        raise ValueError(f"unknown implement category: {category}")
    if soil_texture not in cat:
        raise ValueError(
            f"unknown soil_texture {soil_texture!r}; must be one of {list(cat)}"
        )
    return cat[soil_texture]


def moisture_factor(moisture_gravimetric: float) -> float:
    """Return the dimensionless empirical moisture multiplier.

    Modeled as a U-shaped quadratic centred on ``MOISTURE_OPTIMUM`` with the
    moisture penalty saturating at ±50% of the dry baseline at the dry/wet
    extremes. This is *not* in ASABE D497; it is a simple empirical layer
    motivated by the depth-resistance literature.
    """
    delta = (moisture_gravimetric - MOISTURE_OPTIMUM) / MOISTURE_SCALE
    return float(1.0 + 0.5 * delta * delta)


# ---------------------------------------------------------------------------
# Implement library
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Implement:
    """ASABE D497.7 implement record.

    The optional ``replaces`` and ``co_design_note`` fields are populated for
    the CableTract co-designed library (`asabe_d497_cabletract.csv`) and left
    empty for the conventional ASABE library. They are not used by the
    physical model — they are metadata for the F4 comparison plot and the
    manuscript table in §3.5.
    """

    name: str
    category: str
    A: float
    B: float
    C: float
    width_basis: str
    width_count: float
    depth_dependent: bool
    typical_depth_cm: float
    A_units_note: str
    source: str
    replaces: str = ""
    co_design_note: str = ""

    def base_draft_N(self, speed_km_h: float, depth_cm: float) -> float:
        """Mean draft at a single (S, T) without texture/moisture multipliers."""
        T = depth_cm if self.depth_dependent else 1.0
        return (self.A + self.B * speed_km_h + self.C * speed_km_h * speed_km_h) * self.width_count * T

    def draft_N(
        self,
        speed_km_h: float,
        depth_cm: float,
        soil_texture: str = "medium",
        moisture: float = MOISTURE_OPTIMUM,
    ) -> float:
        """Deterministic draft including texture and moisture multipliers."""
        f_t = texture_factor(soil_texture, self.category)
        f_m = moisture_factor(moisture)
        return f_t * f_m * self.base_draft_N(speed_km_h, depth_cm)


def load_implement_library(csv_path: Path = ASABE_CSV) -> List[Implement]:
    """Load implements from a CSV table (default = ASABE conventional library).

    Pass ``CODESIGN_CSV`` (or use :func:`load_cabletract_implement_library`)
    to load the CableTract co-designed library instead.
    """
    df = pd.read_csv(csv_path)
    has_replaces = "replaces" in df.columns
    has_note = "co_design_note" in df.columns
    out: List[Implement] = []
    for _, row in df.iterrows():
        out.append(
            Implement(
                name=str(row["name"]),
                category=str(row["category"]),
                A=float(row["A"]),
                B=float(row["B"]),
                C=float(row["C"]),
                width_basis=str(row["width_basis"]),
                width_count=float(row["width_count"]),
                depth_dependent=bool(row["depth_dependent"]),
                typical_depth_cm=float(row["typical_depth_cm"]),
                A_units_note=str(row["A_units_note"]),
                source=str(row["source"]),
                replaces=str(row["replaces"]) if has_replaces else "",
                co_design_note=str(row["co_design_note"]) if has_note else "",
            )
        )
    return out


def load_cabletract_implement_library() -> List[Implement]:
    """Load the CableTract co-designed implement library.

    These are 10 implements right-sized for CableTract's force budget
    (≤ 3 kN typical draft) and slow operating regime (1–2.5 km/h). Each
    record names the conventional ASABE implement it replaces and gives a
    short rationale in ``co_design_note``. The ASABE D497.7 coefficient
    framework is unchanged — only the width, depth, and (for the lightest-
    weight tools) the body-rolling-resistance contribution to ``A`` are
    altered, and only where the change is mechanically defensible.
    """
    return load_implement_library(CODESIGN_CSV)


def implement_by_name(name: str, library: Sequence[Implement] | None = None) -> Implement:
    """Look up one implement by name from the library."""
    lib = list(library) if library is not None else load_implement_library()
    for imp in lib:
        if imp.name == name:
            return imp
    raise KeyError(f"implement not found: {name}")


# ---------------------------------------------------------------------------
# Stochastic draft sampler
# ---------------------------------------------------------------------------

def draft_distribution(
    implement: Implement,
    soil_texture: str,
    moisture_range: Tuple[float, float],
    speed_range_km_h: Tuple[float, float],
    depth_range_cm: Tuple[float, float],
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Monte-Carlo draft samples for one implement under one (texture) regime.

    Parameters
    ----------
    implement : Implement
        ASABE D497 implement record.
    soil_texture : {'fine', 'medium', 'coarse'}
        Soil texture class. The implement's category-specific F_i multiplier
        is applied (see :data:`TEXTURE_FACTORS`).
    moisture_range : (lo, hi)
        Uniform sampling range for gravimetric soil moisture content (-).
    speed_range_km_h : (lo, hi)
        Uniform sampling range for field operating speed (km/h).
    depth_range_cm : (lo, hi)
        Uniform sampling range for working depth (cm). Ignored if the
        implement is not depth-dependent.
    n_samples : int
        Number of Monte Carlo draws. Default 5000 gives stable P10/P50/P90.
    rng : np.random.Generator, optional
        Random source. Defaults to a fixed seed for reproducibility.

    Returns
    -------
    drafts : np.ndarray, shape (n_samples,)
        Draft samples in newtons.
    """
    if rng is None:
        rng = np.random.default_rng(0xCAB1E)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    s_lo, s_hi = speed_range_km_h
    if s_lo <= 0.0 or s_hi < s_lo:
        raise ValueError("invalid speed_range_km_h")
    d_lo, d_hi = depth_range_cm
    if d_lo < 0.0 or d_hi < d_lo:
        raise ValueError("invalid depth_range_cm")
    m_lo, m_hi = moisture_range
    if m_lo < 0.0 or m_hi < m_lo:
        raise ValueError("invalid moisture_range")

    speeds = rng.uniform(s_lo, s_hi, n_samples)
    if implement.depth_dependent:
        depths = rng.uniform(d_lo, d_hi, n_samples)
    else:
        depths = np.ones(n_samples)
    moistures = rng.uniform(m_lo, m_hi, n_samples)

    f_texture = texture_factor(soil_texture, implement.category)
    f_moisture = 1.0 + 0.5 * ((moistures - MOISTURE_OPTIMUM) / MOISTURE_SCALE) ** 2

    base = implement.A + implement.B * speeds + implement.C * speeds * speeds
    drafts = f_texture * f_moisture * base * implement.width_count * depths
    return drafts


def percentiles(samples: np.ndarray, ps: Sequence[float] = (10, 50, 90)) -> Dict[str, float]:
    """Return P10/P50/P90 (or any custom set) of a sample array as a dict."""
    qs = np.percentile(samples, ps)
    return {f"P{int(p)}": float(q) for p, q in zip(ps, qs)}


# ---------------------------------------------------------------------------
# Convenience: draft summary across the whole library
# ---------------------------------------------------------------------------

def library_draft_summary(
    soil_texture: str = "medium",
    moisture_range: Tuple[float, float] = (0.12, 0.28),
    speed_range_km_h: Tuple[float, float] = (1.0, 4.0),
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
    library: Sequence[Implement] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with P10/P50/P90 draft per implement in the library.

    Defaults reflect a generic field operation regime (1-4 km/h) on a
    medium-texture soil. Pass ``library=load_cabletract_implement_library()``
    and ``speed_range_km_h=(1.0, 2.5)`` to summarise the co-designed library
    in the CableTract operating regime.
    """
    rng = rng if rng is not None else np.random.default_rng(0xCAB1E)
    lib = list(library) if library is not None else load_implement_library()
    rows = []
    for imp in lib:
        depth_lo = max(imp.typical_depth_cm - 5.0, 1.0)
        depth_hi = imp.typical_depth_cm + 5.0
        samples = draft_distribution(
            imp,
            soil_texture=soil_texture,
            moisture_range=moisture_range,
            speed_range_km_h=speed_range_km_h,
            depth_range_cm=(depth_lo, depth_hi),
            n_samples=n_samples,
            rng=rng,
        )
        ps = percentiles(samples)
        rows.append(
            {
                "implement": imp.name,
                "category": imp.category,
                "P10_N": ps["P10"],
                "P50_N": ps["P50"],
                "P90_N": ps["P90"],
                "mean_N": float(np.mean(samples)),
                "std_N": float(np.std(samples)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Side-by-side comparison: conventional vs CableTract co-designed
# ---------------------------------------------------------------------------

def compare_conventional_vs_codesign(
    soil_texture: str = "medium",
    moisture_range: Tuple[float, float] = (0.12, 0.28),
    conventional_speed_range_km_h: Tuple[float, float] = (5.0, 9.0),
    codesign_speed_range_km_h: Tuple[float, float] = (1.0, 2.5),
    n_samples: int = 5000,
    rng_seed: int = 0xCAB1E,
) -> pd.DataFrame:
    """Side-by-side P50 draft comparison: conventional ASABE vs CableTract co-designed.

    Each row of the returned DataFrame is one CableTract co-designed implement
    paired with its conventional counterpart (via the ``replaces`` field). The
    conventional implement is sampled at conventional tractor field speeds
    (5–9 km/h) and the co-designed implement is sampled at CableTract speeds
    (1–2.5 km/h), so the comparison includes the speed-quadratic effect.

    The ratio column ``co_designed_over_conventional_P50`` tells you, for each
    operation, what fraction of the conventional draft the co-designed
    CableTract version actually requires. Values < 1.0 mean co-design wins;
    > 1.0 mean it loses (and would push us out of the anchor envelope).
    """
    rng = np.random.default_rng(rng_seed)
    conventional = load_implement_library()
    codesign = load_cabletract_implement_library()

    conv_by_name: Dict[str, Implement] = {imp.name: imp for imp in conventional}

    rows = []
    for cd in codesign:
        conv_name = cd.replaces.split("+")[0].strip().split(" at ")[0].strip()
        # The "replaces" field is human-readable ("subsoiler 40 cm + 7-tool ...")
        # — we resolve to a canonical conventional implement name by keyword.
        conv_canon = _canonicalise_replaces(cd.replaces, conv_by_name)
        conv = conv_by_name.get(conv_canon)
        if conv is None:
            continue

        cd_depth_lo = max(cd.typical_depth_cm - 3.0, 1.0)
        cd_depth_hi = cd.typical_depth_cm + 3.0
        conv_depth_lo = max(conv.typical_depth_cm - 5.0, 1.0)
        conv_depth_hi = conv.typical_depth_cm + 5.0

        cd_samples = draft_distribution(
            cd,
            soil_texture=soil_texture,
            moisture_range=moisture_range,
            speed_range_km_h=codesign_speed_range_km_h,
            depth_range_cm=(cd_depth_lo, cd_depth_hi),
            n_samples=n_samples,
            rng=rng,
        )
        conv_samples = draft_distribution(
            conv,
            soil_texture=soil_texture,
            moisture_range=moisture_range,
            speed_range_km_h=conventional_speed_range_km_h,
            depth_range_cm=(conv_depth_lo, conv_depth_hi),
            n_samples=n_samples,
            rng=rng,
        )
        cd_p50 = float(np.percentile(cd_samples, 50))
        conv_p50 = float(np.percentile(conv_samples, 50))
        rows.append({
            "category": cd.category,
            "conventional_implement": conv.name,
            "conventional_P50_N": conv_p50,
            "codesigned_implement": cd.name,
            "codesigned_P50_N": cd_p50,
            "codesigned_over_conventional_P50": cd_p50 / max(conv_p50, 1e-9),
            "codesigned_width_count": cd.width_count,
            "conventional_width_count": conv.width_count,
            "codesigned_depth_cm": cd.typical_depth_cm if cd.depth_dependent else 0.0,
            "conventional_depth_cm": conv.typical_depth_cm if conv.depth_dependent else 0.0,
        })
    return pd.DataFrame(rows)


def _canonicalise_replaces(replaces_field: str, conv_by_name: Dict[str, Implement]) -> str:
    """Map the human-readable ``replaces`` text to a conventional library key.

    The co-design CSV's ``replaces`` column contains free-form text like
    "7-tool chisel plow at 18 cm" or "subsoiler 40 cm + 7-tool chisel plow".
    This helper picks the *first* canonical conventional implement key that
    appears as a keyword in the text, falling back to a category-based guess.
    """
    text = replaces_field.lower()
    keyword_to_name = {
        "subsoil": "subsoiler_single_shank",
        "chisel": "chisel_plow_twisted",
        "moldboard": "moldboard_plow",
        "tandem disk": "disk_harrow_tandem",
        "disk": "disk_harrow_tandem",
        "secondary cultivator": "field_cultivator_secondary",
        "field cultivator": "field_cultivator_secondary",
        "sweep": "sweep_plow",
        "rotary hoe": "rotary_hoe",
        "planter": "row_planter",
        "grain drill": "grain_drill",
        "drill": "grain_drill",
        "sprayer": "boom_sprayer",
        "rotary disc mower": "rotary_mower_disc",
        "disc mower": "rotary_mower_disc",
        "mower": "rotary_mower_disc",
    }
    for kw, canon in keyword_to_name.items():
        if kw in text and canon in conv_by_name:
            return canon
    return ""
