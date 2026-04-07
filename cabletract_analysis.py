"""CableTract analytical study — backwards-compatibility shim.

The original 470-line monolithic version of this script has been refactored
into the ``cabletract`` package (see ``cabletract/`` for the new layout).

This shim is preserved so that the existing manuscript draft and reviewers'
muscle memory keep working: running ``python cabletract_analysis.py`` from
this directory still regenerates exactly the same 7 PNG figures and CSV
tables that were checked in alongside the v1 manuscript scaffold.

If you are reading this because you want to add a new analysis: do NOT add
to this file. Add a new module under ``cabletract/`` and a runner script
under ``scripts/``. This shim should only contain the v1 reproduction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cabletract import CableTractParams, run_single, results_to_series
from cabletract.plotting import (
    save_bar_plot,
    save_dual_line_plot,
    save_heatmap,
    save_line_plot,
)
from cabletract.sweeps import (
    analyze_draft_load,
    analyze_field_length,
    analyze_setup_time,
    analyze_shape_efficiency,
    compare_operation_types,
    offgrid_feasibility_grid,
)


def main(output_dir: str = ".") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    base = CableTractParams()
    reference = run_single(base)
    reference_df = results_to_series(reference).to_frame(name="value")
    reference_df.to_csv(out / "reference_case.csv")

    # 1) Setup-time sensitivity
    setup_df = analyze_setup_time(base, np.arange(20, 121, 10))
    setup_df.to_csv(out / "setup_time_sensitivity.csv", index=False)
    save_line_plot(
        setup_df,
        x="setup_time_s",
        y="decares_per_day_energy_limited",
        xlabel="Setup time per round (s)",
        ylabel="Daily area (decare/day)",
        title="Throughput sensitivity to setup time",
        outpath=out / "throughput_vs_setup_time.png",
    )

    # 2) Draft-load sensitivity
    draft_df = analyze_draft_load(base, np.arange(300, 3301, 300))
    draft_df.to_csv(out / "draft_load_sensitivity.csv", index=False)
    save_dual_line_plot(
        draft_df,
        x="draft_load_N",
        y1="energy_per_decare_Wh",
        y2="winch_input_power_W",
        xlabel="Draft load (N)",
        y1label="Energy per decare (Wh/decare)",
        y2label="Winch input power (W)",
        title="Energy and winch power sensitivity to draft load",
        outpath=out / "energy_power_vs_draft.png",
    )

    # 3) Field-length sensitivity
    length_df = analyze_field_length(base, np.arange(20, 121, 10))
    length_df.to_csv(out / "field_length_sensitivity.csv", index=False)
    save_line_plot(
        length_df,
        x="span_m",
        y="decares_per_day_energy_limited",
        xlabel="Field span / pass length (m)",
        ylabel="Daily area (decare/day)",
        title="Throughput sensitivity to field length",
        outpath=out / "throughput_vs_field_length.png",
    )

    # Optional shape-efficiency sensitivity
    shape_df = analyze_shape_efficiency(base, np.linspace(0.5, 1.0, 11))
    shape_df.to_csv(out / "shape_efficiency_sensitivity.csv", index=False)
    save_line_plot(
        shape_df,
        x="shape_efficiency",
        y="decares_per_day_energy_limited",
        xlabel="Shape efficiency factor (-)",
        ylabel="Daily area (decare/day)",
        title="Throughput sensitivity to field shape efficiency",
        outpath=out / "throughput_vs_shape_efficiency.png",
    )

    # 4) Off-grid feasibility map
    offgrid_df = offgrid_feasibility_grid(
        base,
        solar_area_values=np.arange(10, 41, 2),
        battery_values_Wh=np.arange(5000, 30001, 2500),
    )
    offgrid_df.to_csv(out / "offgrid_feasibility_map.csv", index=False)
    save_heatmap(
        offgrid_df,
        x="solar_area_m2",
        y="battery_Wh",
        z="decares_per_day_offgrid",
        xlabel="Solar area (m²)",
        ylabel="Battery capacity (Wh)",
        title="Off-grid feasibility map",
        outpath=out / "offgrid_feasibility_map.png",
    )

    # 5) Baseline comparisons across operation classes
    comp_df = compare_operation_types(base)
    comp_df.to_csv(out / "operation_type_comparison.csv", index=False)
    save_bar_plot(
        comp_df,
        x="operation",
        ycols=["cabletract_Wh_per_decare", "electric_tractor_Wh_per_decare"],
        xlabel="Operation type",
        ylabel="Energy per decare (Wh/decare)",
        title="CableTract vs electric tractor across operation types",
        outpath=out / "operation_type_comparison.png",
    )

    print("Reference case")
    print(reference_df)
    print("\nSaved outputs to:", out.resolve())


if __name__ == "__main__":
    main()
