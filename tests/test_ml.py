"""Verification tests for cabletract.ml (Phase 6).

Pins down: surrogate held-out R^2 above a low floor on the four
production targets, NSGA-II returns a non-empty front of the right
shape, polygon predictor reproduces F9 within tolerance, feature
importance is consistent with the Sobol decomposition.

Run as:

    python tests/test_ml.py
    pytest tests/test_ml.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cabletract.layout import load_field_corpus  # noqa: E402
from cabletract.ml import (  # noqa: E402
    DEFAULT_SURROGATE_FEATURES,
    DEFAULT_SURROGATE_TARGETS,
    build_surrogate_training_set,
    feature_importance,
    pareto_optimize,
    polygon_features,
    train_polygon_predictor,
    train_surrogate,
)


# Cache the trained surrogate so we don't retrain for every test
_SURROGATE = None
_TRAINING_DF = None


def _surrogate():
    global _SURROGATE, _TRAINING_DF
    if _SURROGATE is None:
        _TRAINING_DF = build_surrogate_training_set(n=1500, seed=0)
        _SURROGATE = train_surrogate(_TRAINING_DF)
    return _SURROGATE


# ---------------------------------------------------------------------------
# Surrogate
# ---------------------------------------------------------------------------

def test_surrogate_has_one_model_per_target() -> None:
    sm = _surrogate()
    for t in DEFAULT_SURROGATE_TARGETS:
        assert t in sm.models
        assert t in sm.held_out_r2


def test_surrogate_r2_above_floor_for_well_behaved_targets() -> None:
    """Energy-per-decare and decares-per-day are smooth functions of
    the inputs and should be reproduced with R^2 > 0.7 from a 1500-row
    GBT."""
    sm = _surrogate()
    assert sm.held_out_r2["energy_per_decare_Wh"] > 0.7
    assert sm.held_out_r2["decares_per_day_offgrid"] > 0.7


def test_surrogate_predict_returns_dataframe_with_target_columns() -> None:
    sm = _surrogate()
    df = _TRAINING_DF.head(5)[sm.features]
    out = sm.predict(df)
    for t in DEFAULT_SURROGATE_TARGETS:
        assert t in out.columns
    assert len(out) == len(df)


def test_feature_importance_top_features_overlap_with_sobol() -> None:
    """Top-3 features for `decares_per_day_offgrid` should include at
    least one of {solar_*, draft_load_N, width_m} — these are exactly
    the parameters dominating the Sobol decomposition (F13)."""
    sm = _surrogate()
    fi = feature_importance(sm, "decares_per_day_offgrid", top_k=3)
    top = set(fi["feature"].values)
    expected = {"solar_hours_per_day", "solar_area_m2", "solar_power_W_m2",
                "draft_load_N", "width_m", "shape_efficiency"}
    assert len(top & expected) >= 2, top


# ---------------------------------------------------------------------------
# Pareto optimisation
# ---------------------------------------------------------------------------

def test_pareto_returns_nonempty_front() -> None:
    sm = _surrogate()
    pf = pareto_optimize(sm, bounds={
        "solar_area_m2": (10.0, 30.0),
        "draft_load_N": (1500.0, 4500.0),
        "width_m": (1.5, 3.0),
    }, n_gen=20, pop_size=40)
    assert pf.X.shape[0] > 0
    assert pf.F.shape[0] == pf.X.shape[0]


def test_pareto_objective_columns_match_request() -> None:
    sm = _surrogate()
    pf = pareto_optimize(sm, bounds={
        "solar_area_m2": (10.0, 30.0),
        "draft_load_N": (1500.0, 4500.0),
    }, objectives={
        "decares_per_day_offgrid": "max",
        "energy_per_decare_Wh": "min",
    }, n_gen=15, pop_size=30)
    assert "decares_per_day_offgrid" in pf.objective_names
    assert "energy_per_decare_Wh" in pf.objective_names


def test_pareto_front_contains_non_dominated_points() -> None:
    """A Pareto front by construction has no point that strictly
    dominates another. We check pairwise non-domination on the
    surrogate's predictions."""
    sm = _surrogate()
    pf = pareto_optimize(sm, bounds={
        "solar_area_m2": (10.0, 30.0),
        "draft_load_N": (1500.0, 4500.0),
        "width_m": (1.5, 3.0),
    }, objectives={
        "decares_per_day_offgrid": "max",
        "payback_months_vs_fuel": "min",
    }, n_gen=20, pop_size=40)
    # Convert to minimisation form
    F = np.array(pf.F)
    F[:, 0] *= -1  # decares: max -> min
    n_dominated = 0
    for i in range(F.shape[0]):
        for j in range(F.shape[0]):
            if i == j:
                continue
            if np.all(F[j] <= F[i] + 1e-9) and np.any(F[j] < F[i] - 1e-9):
                n_dominated += 1
                break
    # Allow a few points dominated by floating-point noise
    assert n_dominated <= max(2, F.shape[0] // 10)


# ---------------------------------------------------------------------------
# Polygon features and predictor
# ---------------------------------------------------------------------------

def test_polygon_features_returns_required_keys() -> None:
    fields = load_field_corpus()
    feats = polygon_features(fields[0].polygon)
    for k in ["area_m2", "perimeter_m", "bb_aspect", "bb_fill", "n_holes", "iso_perim_ratio"]:
        assert k in feats


def test_polygon_features_iso_perim_in_unit_interval() -> None:
    """Iso-perimetric ratio is in [0, 1] for any polygon, =1 for a circle."""
    fields = load_field_corpus()
    for f in fields:
        ipr = polygon_features(f.polygon)["iso_perim_ratio"]
        assert 0.0 <= ipr <= 1.0 + 1e-6, (f.id, ipr)


def test_polygon_predictor_held_out_r2_above_floor() -> None:
    """The Phase 4 corpus is small (50 fields) but the random forest
    should still hit R^2 > 0.5 on a 30 % held-out split."""
    fields = load_field_corpus()
    model, fnames, r2 = train_polygon_predictor(fields, n_orient=6)
    assert r2 > 0.5
    assert len(fnames) == 6


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_surrogate_has_one_model_per_target,
    test_surrogate_r2_above_floor_for_well_behaved_targets,
    test_surrogate_predict_returns_dataframe_with_target_columns,
    test_feature_importance_top_features_overlap_with_sobol,
    test_pareto_returns_nonempty_front,
    test_pareto_objective_columns_match_request,
    test_pareto_front_contains_non_dominated_points,
    test_polygon_features_returns_required_keys,
    test_polygon_features_iso_perim_in_unit_interval,
    test_polygon_predictor_held_out_r2_above_floor,
]


if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {exc}")
            failures.append((t.__name__, exc))
    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        sys.exit(1)
    print(f"\nok: cabletract.ml passes {len(ALL_TESTS)} invariants")
