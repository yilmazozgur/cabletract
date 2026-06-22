"""S4 -- Co-design lever sensitivity sweep.

The whole CableTract case rests on the co-designed implement library cutting
median draft to ~0.37x of the conventional library. That reduction is a model
prediction (D497.7 coefficient scaling at a lighter operating point), not a
measurement. This module asks the reviewer's question directly: *if co-design
under-delivers* -- achieving only, say, 0.6x or 0.8x instead of 0.37x -- how
much of the energy / off-grid / anchor / economics story survives?

We sweep the achieved reference draft from the codesigned point up to the
conventional-library median, and propagate each draft level through the full
deterministic pipeline (``run_single``), the anchor envelope, and the
discounted-cash-flow economics. No surrogate: every point is a full model
evaluation. The output is a four-panel robustness picture plus a tidy table.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from .params import CableTractParams
from .simulate import run_single
from .soil import library_draft_summary, load_implement_library
from .economics import EconParams, cabletract_npv_vs_diesel, cabletract_payback_vs_diesel

# Codesigned reference P90/P50 draft ratio (3.0 kN / 1.8 kN), used to map a swept
# P50 draft to the P90 the anchor must hold.
P90_OVER_P50 = 3.0 / 1.8


def conventional_median_draft_N(speed_range_km_h=(5.0, 9.0)) -> float:
    """Median P50 draft of the conventional ASABE D497 library at tractor speeds."""
    df = library_draft_summary(speed_range_km_h=speed_range_km_h,
                               library=load_implement_library())
    col = next(c for c in df.columns if c.lower() in ("p50_n", "p50") or "50" in c)
    return float(np.median(df[col].values))


def required_augers(draft_P90_N: float, per_auger_cap_N: float,
                    safety_factor: float = 1.15) -> int:
    """Allowable-capacity auger count: ceil(SF * T / capacity)."""
    return int(math.ceil(safety_factor * draft_P90_N / per_auger_cap_N))


def sweep_codesign(
    p: CableTractParams | None = None,
    econ: EconParams | None = None,
    n: int = 41,
    per_auger_caps_N=(400.0, 2000.0),
) -> pd.DataFrame:
    """Sweep the reference draft from codesigned to conventional and propagate.

    Returns one row per draft level with energy/decare, off-grid throughput,
    required augers at each per-auger capacity bound, simple payback (which
    varies with throughput), and the replacement-frame NPV (which does not, in
    the off-grid frame -- that flatness is itself the robustness finding)."""
    base = p if p is not None else CableTractParams.codesigned()
    base_econ = econ if econ is not None else EconParams.codesigned()
    conv = conventional_median_draft_N()
    drafts = np.linspace(base.draft_load_N, conv, n)

    rows = []
    for d in drafts:
        r = run_single(replace(base, draft_load_N=float(d)))
        p90 = d * P90_OVER_P50
        # Economics: energy intensity follows the swept draft; in the off-grid
        # frame (grid_share = 0) NPV is independent of it, which we surface.
        e = replace(base_econ, energy_per_ha_kWh=r.energy_per_decare_Wh / 100.0)
        row = {
            "draft_P50_N": float(d),
            "draft_P90_N": float(p90),
            "reduction_ratio_vs_conventional": float(d / conv),
            "energy_per_decare_Wh": float(r.energy_per_decare_Wh),
            "decares_per_day_offgrid": float(r.decares_per_day_offgrid),
            "simple_payback_months": float(r.payback_months_vs_fuel),
            "npv_replacement_eur": float(cabletract_npv_vs_diesel(e)),
            "payback_replacement_yr": float(cabletract_payback_vs_diesel(e)),
        }
        for cap in per_auger_caps_N:
            row[f"augers_req_P90_cap{int(cap)}N"] = required_augers(p90, cap)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["conventional_median_draft_N"] = conv
    df.attrs["reference_draft_N"] = base.draft_load_N
    return df
