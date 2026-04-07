"""Phase 6 — ML surrogate, multi-objective optimisation, polygon predictor.

Three pieces, all designed to support the design-explorer story in §7
without forcing a reviewer to re-run the full simulator.

1. **Surrogate model** — `train_surrogate(df, targets)` fits a
   `GradientBoostingRegressor` per output. Trained on a few thousand
   `run_single` evaluations from the Phase 5 Monte Carlo loop. We use
   GBT (not GP) because it scales better past 1k samples and gives
   per-feature importance for free, which we report as a sanity-check
   against the Sobol indices.
2. **Multi-objective Pareto** — `pareto_optimize(...)` runs NSGA-II
   over (capex, throughput, off-grid surplus, mass) on the surrogate
   and returns the Pareto front. The surrogate is necessary because
   NSGA-II would otherwise need ~50k simulator calls per generation.
3. **Polygon throughput predictor** — `polygon_throughput_predictor(...)`
   trains a `RandomForestRegressor` on (polygon_features → effective
   shape efficiency) for the Phase 4 corpus, so unseen field shapes
   can be scored in milliseconds rather than re-running the strip
   decomposition planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .params import CableTractParams
from .simulate import CableTractResults, run_single


# ---------------------------------------------------------------------------
# Surrogate model
# ---------------------------------------------------------------------------

DEFAULT_SURROGATE_FEATURES: List[str] = [
    "span_m", "width_m", "implement_weight_N", "draft_load_N",
    "system_weight_N", "winch_efficiency", "solar_power_W_m2",
    "solar_area_m2", "solar_hours_per_day", "wind_power_W",
    "wind_hours_per_day", "setup_time_s", "operation_time_ratio",
    "battery_Wh", "operating_hours_per_day", "operating_days_per_year",
    "fuel_l_per_decare", "fuel_price_usd_per_l", "cost_cabletract_usd",
    "shape_efficiency",
]

DEFAULT_SURROGATE_TARGETS: List[str] = [
    "decares_per_day_offgrid",
    "energy_per_decare_Wh",
    "payback_months_vs_fuel",
    "surplus_power_W",
]


@dataclass(frozen=True)
class SurrogateModel:
    """One-target-per-model wrapper. Holds the fitted regressors and
    the held-out R² for each target."""
    features: List[str]
    targets: List[str]
    models: Dict[str, object]
    held_out_r2: Dict[str, float]

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for t in self.targets:
            out[t] = self.models[t].predict(X[self.features].values)
        return pd.DataFrame(out, index=X.index)


def train_surrogate(
    df: pd.DataFrame,
    features: Sequence[str] = tuple(DEFAULT_SURROGATE_FEATURES),
    targets: Sequence[str] = tuple(DEFAULT_SURROGATE_TARGETS),
    test_frac: float = 0.2,
    random_state: int = 0,
) -> SurrogateModel:
    """Train one GBT regressor per output column. Returns a
    `SurrogateModel` with the fitted regressors and the held-out R²
    measured on the (random, fixed-seed) test split."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    feats = list(features)
    targs = list(targets)
    X = df[feats].values
    rng = np.random.default_rng(random_state)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(len(df) * test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    models: Dict[str, object] = {}
    r2s: Dict[str, float] = {}
    for t in targs:
        y = df[t].values
        # Drop rows that are non-finite for this target
        finite = np.isfinite(y)
        X_train = X[np.intersect1d(train_idx, np.where(finite)[0])]
        y_train = y[np.intersect1d(train_idx, np.where(finite)[0])]
        X_test = X[np.intersect1d(test_idx, np.where(finite)[0])]
        y_test = y[np.intersect1d(test_idx, np.where(finite)[0])]
        m = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=random_state)
        m.fit(X_train, y_train)
        y_hat = m.predict(X_test)
        r2s[t] = float(r2_score(y_test, y_hat))
        models[t] = m
    return SurrogateModel(features=feats, targets=targs, models=models, held_out_r2=r2s)


def feature_importance(model: SurrogateModel, target: str, top_k: int = 10) -> pd.DataFrame:
    """Per-feature importance for one target, ranked descending."""
    m = model.models[target]
    fi = getattr(m, "feature_importances_", None)
    if fi is None:
        raise ValueError(f"model for {target} has no feature_importances_")
    df = pd.DataFrame({"feature": model.features, "importance": fi})
    return df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-objective Pareto via NSGA-II
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParetoFront:
    X: np.ndarray              # decision variables (n_points, n_vars)
    F: np.ndarray              # objectives (n_points, n_objectives)
    feature_names: List[str]
    objective_names: List[str]

    def to_df(self) -> pd.DataFrame:
        df_x = pd.DataFrame(self.X, columns=self.feature_names)
        df_f = pd.DataFrame(self.F, columns=self.objective_names)
        return pd.concat([df_x, df_f], axis=1)


def pareto_optimize(
    surrogate: SurrogateModel,
    bounds: Dict[str, Tuple[float, float]],
    objectives: Dict[str, str] = None,  # {target_name: "min"|"max"}
    n_gen: int = 40,
    pop_size: int = 80,
    seed: int = 0,
) -> ParetoFront:
    """Run NSGA-II on the surrogate over the given parameter bounds.

    `objectives` maps surrogate target names to ``"min"`` or ``"max"``.
    Defaults to {capex/cost: min, decares_per_day_offgrid: max,
    payback_months_vs_fuel: min, surplus_power_W: max}.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.sampling.lhs import LHS

    if objectives is None:
        objectives = {
            "decares_per_day_offgrid": "max",
            "payback_months_vs_fuel": "min",
            "energy_per_decare_Wh": "min",
        }

    # Determine which features to vary (those in bounds) and which to fix
    vary_feats = list(bounds.keys())
    fixed_feats = [f for f in surrogate.features if f not in bounds]
    fixed_defaults = {f: getattr(CableTractParams(), f) for f in fixed_feats}

    xl = np.array([bounds[f][0] for f in vary_feats])
    xu = np.array([bounds[f][1] for f in vary_feats])

    target_names = list(objectives.keys())
    sense = np.array([1.0 if objectives[t] == "min" else -1.0 for t in target_names])

    class _SurrogateProblem(Problem):
        def __init__(self) -> None:
            super().__init__(n_var=len(vary_feats), n_obj=len(target_names), xl=xl, xu=xu)

        def _evaluate(self, X, out, *args, **kwargs) -> None:  # noqa: ARG002
            df = pd.DataFrame(X, columns=vary_feats)
            for f, val in fixed_defaults.items():
                df[f] = val
            df = df[surrogate.features]
            yhat = surrogate.predict(df)
            F = np.zeros((X.shape[0], len(target_names)))
            for j, t in enumerate(target_names):
                F[:, j] = sense[j] * yhat[t].values
            out["F"] = F

    algo = NSGA2(pop_size=pop_size, sampling=LHS())
    res = pymoo_minimize(_SurrogateProblem(), algo, ("n_gen", n_gen), seed=seed, verbose=False)

    # Restore the natural sign of the objectives in the returned front
    F_signed = res.F * sense
    return ParetoFront(
        X=res.X,
        F=F_signed,
        feature_names=vary_feats,
        objective_names=target_names,
    )


# ---------------------------------------------------------------------------
# Polygon throughput predictor
# ---------------------------------------------------------------------------

def polygon_features(polygon) -> Dict[str, float]:
    """Extract scalar features from a Shapely polygon for the random
    forest predictor."""
    minx, miny, maxx, maxy = polygon.bounds
    bb_w = max(maxx - minx, 1e-9)
    bb_h = max(maxy - miny, 1e-9)
    bb_area = bb_w * bb_h
    return {
        "area_m2": float(polygon.area),
        "perimeter_m": float(polygon.length),
        "bb_aspect": float(bb_w / bb_h),
        "bb_fill": float(polygon.area / bb_area),
        "n_holes": float(len(polygon.interiors)),
        "iso_perim_ratio": float(4.0 * np.pi * polygon.area / max(polygon.length ** 2, 1e-9)),
    }


def train_polygon_predictor(
    fields,  # list of FieldRecord
    span: float = 50.0,
    swath: float = 2.0,
    n_orient: int = 8,
    random_state: int = 0,
) -> Tuple[object, List[str], float]:
    """Train a random forest mapping polygon features → best-orientation
    shape efficiency. Returns ``(model, feature_names, held_out_r2)``.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    from .layout import best_orientation_efficiency

    rows = []
    ys = []
    for f in fields:
        feats = polygon_features(f.polygon)
        eta, _ = best_orientation_efficiency(f.polygon, span=span, swath=swath, n_orient=n_orient)
        rows.append(feats)
        ys.append(eta)
    df = pd.DataFrame(rows)
    feature_names = list(df.columns)
    X = df.values
    y = np.array(ys)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=random_state)
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    r2 = float(r2_score(y_test, y_hat))
    return model, feature_names, r2


# ---------------------------------------------------------------------------
# Convenience: build a training set on the fly
# ---------------------------------------------------------------------------

def build_surrogate_training_set(n: int, seed: int = 0) -> pd.DataFrame:
    """Run `monte_carlo` from the uncertainty module with the default
    parameter problem to produce an n-row surrogate training set."""
    from .uncertainty import default_problem, monte_carlo
    return monte_carlo(default_problem(), n=n, seed=seed)
