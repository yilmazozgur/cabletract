"""Phase 5 — Monte Carlo, Sobol global sensitivity, tornado helpers.

The v1 model in `cabletract.simulate.run_single` is fully deterministic
in 24 input parameters. Reviewer-grade sensitivity analysis means we
must (a) propagate the joint uncertainty in those parameters through
`run_single` to obtain P10/P50/P90 envelopes for the headline outputs,
and (b) decompose the variance of each output into per-parameter and
per-pair contributions (Sobol S1 / ST indices).

This module provides:

1. `ParamRange` and `ParamProblem` — a thin abstraction over SALib's
   `problem` dictionary that lets us specify {name, lo, hi, dist} for
   each parameter and generate samples from either uniform or
   triangular marginals.
2. `monte_carlo(problem, n, fn)` — draws n samples from the joint and
   evaluates `fn` (typically a wrapper around `run_single`) on each,
   returning a tidy DataFrame with one row per draw.
3. `sobol_indices(problem, n_base, fn, output_names)` — wraps SALib's
   Saltelli sampler + `sobol.analyze` to return a tidy DataFrame of
   S1, ST and confidence intervals per (output, parameter).
4. `tornado_data(df, baseline_row, output_col, top_k=10)` — computes
   one-at-a-time elasticities for tornado plots.
5. `default_problem()` — the canonical Phase 5 parameter problem
   matching the codesigned reference design with realistic ranges
   defended in the manuscript text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .params import CableTractParams
from .simulate import CableTractResults, run_single


# ---------------------------------------------------------------------------
# Parameter problem abstraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamRange:
    """One uncertain parameter with a marginal distribution.

    `dist` may be ``uniform`` (lo, hi) or ``triangular`` (lo, mode, hi).
    For triangular distributions ``mode`` defaults to the midpoint
    if not given.
    """
    name: str
    lo: float
    hi: float
    dist: str = "uniform"
    mode: float | None = None

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.dist == "uniform":
            return rng.uniform(self.lo, self.hi, size=n)
        if self.dist == "triangular":
            m = self.mode if self.mode is not None else 0.5 * (self.lo + self.hi)
            return rng.triangular(self.lo, m, self.hi, size=n)
        raise ValueError(f"unknown dist: {self.dist}")


@dataclass(frozen=True)
class ParamProblem:
    """Collection of `ParamRange` records — convertible to SALib's
    `problem` dictionary."""
    ranges: Sequence[ParamRange]

    @property
    def names(self) -> List[str]:
        return [r.name for r in self.ranges]

    def to_salib(self) -> Dict:
        return {
            "num_vars": len(self.ranges),
            "names": self.names,
            "bounds": [[r.lo, r.hi] for r in self.ranges],
        }


# ---------------------------------------------------------------------------
# Default problem
# ---------------------------------------------------------------------------

def default_problem() -> ParamProblem:
    """Canonical Phase 5 parameter ranges for the codesigned reference.

    Ranges are defended in the manuscript text §5.10 — they correspond
    to ±20–30 % around each codesigned default unless a published
    reference forces a wider band (e.g. winch_efficiency 0.35–0.65,
    draft_load 1500–4500 N spans the ASABE D497 P10–P90 envelope from
    Phase 2 across both the conventional and the codesigned implement
    libraries).
    """
    return ParamProblem(ranges=[
        ParamRange("span_m",                    30.0,    100.0),
        ParamRange("width_m",                    1.5,      3.0),
        ParamRange("implement_weight_N",       500.0,   2000.0),
        ParamRange("draft_load_N",            1500.0,   4500.0),
        ParamRange("system_weight_N",         2000.0,   4500.0),
        ParamRange("winch_efficiency",           0.35,     0.65),
        ParamRange("solar_power_W_m2",         100.0,    220.0),
        ParamRange("solar_area_m2",             10.0,     30.0),
        ParamRange("solar_hours_per_day",        4.0,      8.0),
        ParamRange("wind_power_W",              50.0,    200.0),
        ParamRange("wind_hours_per_day",         6.0,     16.0),
        ParamRange("setup_time_s",              30.0,    180.0),
        ParamRange("operation_time_ratio",       0.6,      0.95),
        ParamRange("battery_Wh",              5000.0,  25000.0),
        ParamRange("operating_hours_per_day",    8.0,     12.0),
        ParamRange("operating_days_per_year",  120.0,    220.0),
        ParamRange("fuel_l_per_decare",          1.0,      3.0),
        ParamRange("fuel_price_usd_per_l",       1.0,      1.8),
        ParamRange("cost_cabletract_usd",    25000.0,  50000.0),
        ParamRange("shape_efficiency",           0.55,     1.00),
    ])


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def _params_from_row(row: Dict[str, float]) -> CableTractParams:
    """Build a CableTractParams from a parameter dict, leaving any
    unspecified field at the dataclass default."""
    return CableTractParams(**row)


def _result_to_dict(r: CableTractResults) -> Dict[str, float]:
    return {
        "decares_per_day_offgrid": r.decares_per_day_offgrid,
        "decares_per_day_energy_limited": r.decares_per_day_energy_limited,
        "decares_per_year": r.decares_per_year,
        "energy_per_decare_Wh": r.energy_per_decare_Wh,
        "winch_input_power_W": r.winch_input_power_W,
        "harvested_power_W": r.harvested_power_W,
        "surplus_power_W": r.surplus_power_W,
        "grid_charge_needed_Wh": r.grid_charge_needed_Wh,
        "fuel_savings_usd_year": r.fuel_savings_usd_year,
        "payback_months_vs_fuel": r.payback_months_vs_fuel,
        "payback_months_vs_electric": r.payback_months_vs_electric,
    }


def monte_carlo(
    problem: ParamProblem,
    n: int,
    seed: int = 0,
    fn: Callable[[CableTractParams], CableTractResults] = run_single,
) -> pd.DataFrame:
    """Draw `n` samples from the joint and evaluate `fn` on each.

    Returns a tidy DataFrame with one row per draw — columns are the
    union of input parameter names and the output names from
    `_result_to_dict`. Deterministic for a given `seed`.
    """
    rng = np.random.default_rng(seed)
    samples = {pr.name: pr.sample(rng, n) for pr in problem.ranges}
    inputs = pd.DataFrame(samples)

    out_rows: List[Dict[str, float]] = []
    for _, row in inputs.iterrows():
        p = _params_from_row(row.to_dict())
        try:
            r = fn(p)
            out_rows.append(_result_to_dict(r))
        except Exception:  # noqa: BLE001
            out_rows.append({k: float("nan") for k in _result_to_dict(run_single(CableTractParams())).keys()})
    outputs = pd.DataFrame(out_rows)
    return pd.concat([inputs.reset_index(drop=True), outputs.reset_index(drop=True)], axis=1)


def percentile_envelope(df: pd.DataFrame, output_col: str, q: Sequence[float] = (0.1, 0.5, 0.9)) -> Dict[str, float]:
    """Return P10/P50/P90 (or arbitrary quantiles) for one output column."""
    vals = df[output_col].dropna()
    return {f"p{int(qi*100)}": float(vals.quantile(qi)) for qi in q}


# ---------------------------------------------------------------------------
# Sobol via SALib
# ---------------------------------------------------------------------------

def sobol_indices(
    problem: ParamProblem,
    n_base: int,
    output_names: Sequence[str],
    fn: Callable[[CableTractParams], CableTractResults] = run_single,
) -> pd.DataFrame:
    """Compute Sobol S1 / ST indices for each (output, parameter).

    `n_base` is the Saltelli base sample size; the total number of
    `fn` evaluations is `n_base * (2 * num_vars + 2)`. Returns a tidy
    DataFrame with columns: output, parameter, S1, S1_conf, ST, ST_conf.
    """
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze

    salib_problem = problem.to_salib()
    X = sobol_sample.sample(salib_problem, n_base, calc_second_order=False)

    # Evaluate fn on every row of X
    n_evals = X.shape[0]
    Ys: Dict[str, np.ndarray] = {name: np.empty(n_evals) for name in output_names}
    fallback = _result_to_dict(run_single(CableTractParams()))
    for i in range(n_evals):
        row = {pr.name: float(X[i, j]) for j, pr in enumerate(problem.ranges)}
        try:
            d = _result_to_dict(fn(_params_from_row(row)))
        except Exception:  # noqa: BLE001
            d = {k: float("nan") for k in fallback.keys()}
        for name in output_names:
            Ys[name][i] = d.get(name, float("nan"))

    rows = []
    for name in output_names:
        Y = Ys[name]
        if not np.all(np.isfinite(Y)):
            # Replace non-finite with column mean so SALib does not crash
            mu = float(np.nanmean(Y))
            Y = np.where(np.isfinite(Y), Y, mu)
        Si = sobol_analyze.analyze(salib_problem, Y, calc_second_order=False, print_to_console=False)
        for j, pname in enumerate(problem.names):
            rows.append({
                "output": name,
                "parameter": pname,
                "S1": float(Si["S1"][j]),
                "S1_conf": float(Si["S1_conf"][j]),
                "ST": float(Si["ST"][j]),
                "ST_conf": float(Si["ST_conf"][j]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tornado plot data
# ---------------------------------------------------------------------------

def tornado_data(
    problem: ParamProblem,
    output_name: str,
    fn: Callable[[CableTractParams], CableTractResults] = run_single,
    baseline: CableTractParams | None = None,
) -> pd.DataFrame:
    """One-at-a-time elasticity per parameter for tornado plots.

    For each parameter we evaluate `fn` at the lo and hi bound while
    holding every other parameter at its baseline value. Returns a
    DataFrame sorted by absolute swing magnitude descending.
    """
    if baseline is None:
        baseline = CableTractParams()
    base_dict = {pr.name: getattr(baseline, pr.name) for pr in problem.ranges}
    base_out = _result_to_dict(fn(baseline))[output_name]

    rows = []
    for pr in problem.ranges:
        lo_dict = dict(base_dict); lo_dict[pr.name] = pr.lo
        hi_dict = dict(base_dict); hi_dict[pr.name] = pr.hi
        lo_full = {f.name: getattr(baseline, f.name) for f in baseline.__dataclass_fields__.values()}
        hi_full = dict(lo_full)
        lo_full.update(lo_dict)
        hi_full.update(hi_dict)
        try:
            lo_out = _result_to_dict(fn(CableTractParams(**lo_full)))[output_name]
            hi_out = _result_to_dict(fn(CableTractParams(**hi_full)))[output_name]
        except Exception:  # noqa: BLE001
            lo_out = float("nan")
            hi_out = float("nan")
        rows.append({
            "parameter": pr.name,
            "lo_bound": pr.lo,
            "hi_bound": pr.hi,
            "lo_out": lo_out,
            "hi_out": hi_out,
            "baseline_out": base_out,
            "swing": abs(hi_out - lo_out),
        })
    df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    return df
