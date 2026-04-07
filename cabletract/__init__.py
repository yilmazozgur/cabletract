"""CableTract feasibility-analysis package.

Public API:

- ``CableTractParams``     — input parameter dataclass (see ``params.py``)
- ``CableTractResults``    — output dataclass returned by ``run_single``
- ``run_single(params)``   — atomic deterministic evaluation (see ``simulate.py``)
- ``results_to_series``    — convert a Results object to a pandas Series

The legacy v1 sensitivity sweeps live in ``cabletract.sweeps`` and the
matplotlib helpers in ``cabletract.plotting``. Phases 1-6 will add
``physics``, ``soil``, ``energy``, ``layout``, ``compaction``, ``economics``,
``uncertainty``, ``ml`` and ``variants`` modules alongside these.
"""

from .params import CableTractParams, CableTractResults, salib_problem
from .simulate import run_single, results_to_series

__all__ = [
    "CableTractParams",
    "CableTractResults",
    "salib_problem",
    "run_single",
    "results_to_series",
]
