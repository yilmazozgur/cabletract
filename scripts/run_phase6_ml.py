"""Phase 6 driver — generates F18, F19 + surrogate / Pareto / polygon-predictor outputs.

F18. 2D Pareto frontier (CAPEX vs daily throughput) on the surrogate
     with the codesigned reference point overlaid.
F19. 3D Pareto (CAPEX vs throughput vs payback) coloured by energy
     intensity, with the codesigned reference overlaid.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402  (registers 3D projection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cabletract.layout import load_field_corpus  # noqa: E402
from cabletract.ml import (  # noqa: E402
    build_surrogate_training_set,
    feature_importance,
    pareto_optimize,
    train_polygon_predictor,
    train_surrogate,
)
from cabletract.params import CableTractParams  # noqa: E402
from cabletract.simulate import run_single  # noqa: E402


OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Building surrogate training set (n=4000)...")
    df = build_surrogate_training_set(n=4000, seed=20260407)
    print(f"  built {len(df)} rows")
    print("Training GBT surrogate over 4 targets...")
    sm = train_surrogate(df)
    print("  held-out R^2:")
    for k, v in sm.held_out_r2.items():
        print(f"    {k:30s} {v:.4f}")

    # Feature importances per target
    fi_rows = []
    for t in sm.targets:
        sub = feature_importance(sm, t, top_k=20)
        for _, row in sub.iterrows():
            fi_rows.append({"target": t, **row.to_dict()})
    fi_df = pd.DataFrame(fi_rows)
    fi_df.to_csv(TAB_DIR / "Phase6_surrogate_feature_importance.csv", index=False)

    pd.DataFrame([{"target": t, "held_out_r2": r} for t, r in sm.held_out_r2.items()]).to_csv(
        TAB_DIR / "Phase6_surrogate_r2.csv", index=False)

    print()
    print("Running NSGA-II Pareto over 6 design variables (cost/area/battery/draft/span/width)...")
    bounds = {
        "cost_cabletract_usd": (8000.0, 25000.0),
        "solar_area_m2":       (8.0, 30.0),
        "battery_Wh":          (3000.0, 25000.0),
        "draft_load_N":        (1500.0, 4500.0),
        "span_m":              (30.0, 100.0),
        "width_m":             (1.5, 3.0),
    }
    objectives = {
        "decares_per_day_offgrid": "max",
        "payback_months_vs_fuel":  "min",
        "energy_per_decare_Wh":    "min",
    }
    pf = pareto_optimize(sm, bounds=bounds, objectives=objectives, n_gen=80, pop_size=120, seed=0)
    print(f"  front size: {pf.X.shape[0]} points")
    pf_df = pf.to_df()
    pf_df.to_csv(TAB_DIR / "Phase6_pareto_front.csv", index=False)

    # Codesigned reference point
    cd_params = CableTractParams.codesigned()
    cd = run_single(cd_params)
    cd_capex = cd_params.cost_cabletract_usd
    cd_throughput = cd.decares_per_day_offgrid
    cd_payback = cd.payback_months_vs_fuel
    cd_energy = cd.energy_per_decare_Wh

    # ----------------------------------------------------------------
    # F18 — 2D Pareto: CAPEX vs throughput, with a zoomed inset
    # ----------------------------------------------------------------
    # The Pareto front is bimodal: ~115 points cluster in 8000–10000 USD,
    # plus a handful of dominated extremes at 24000 USD. We use a
    # zoomed inset to make the dense cluster legible without losing the
    # full-range context. Markers are explicit "o", small, edgeless,
    # with alpha so density is visible.
    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(
        pf_df["cost_cabletract_usd"], pf_df["decares_per_day_offgrid"],
        c=pf_df["payback_months_vs_fuel"], cmap="viridis_r",
        s=18, marker="o", alpha=0.75, linewidths=0.0,
        label="Pareto front (NSGA-II on surrogate, 120 points)",
    )
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Payback months (vs diesel)")
    ax.scatter([cd_capex], [cd_throughput], color="#1b5e1b", s=260, marker="*",
               edgecolor="black", lw=0.9, zorder=5,
               label=f"Codesigned reference (USD {cd_capex:.0f}, {cd_throughput:.1f} dec/day)")
    ax.set_xlabel("CableTract CAPEX (USD)")
    ax.set_ylabel("Daily off-grid throughput (decares/day)")
    ax.set_title("F18. 2D Pareto frontier — CAPEX vs throughput\n"
                 "(coloured by payback months; codesigned reference overlaid)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    # Zoomed inset over the dense cluster
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    axins = inset_axes(ax, width="40%", height="40%", loc="center right",
                       borderpad=2.0)
    axins.scatter(pf_df["cost_cabletract_usd"], pf_df["decares_per_day_offgrid"],
                  c=pf_df["payback_months_vs_fuel"], cmap="viridis_r",
                  s=14, marker="o", alpha=0.75, linewidths=0.0)
    axins.set_xlim(7800, 10200)
    axins.set_ylim(32.8, 38.2)
    axins.set_xticks([8000, 9000, 10000])
    axins.tick_params(axis="both", labelsize=8)
    axins.grid(True, alpha=0.3)
    axins.set_title("Dense cluster", fontsize=9)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.4", lw=0.8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "F18_pareto_2d.png", dpi=220)
    plt.close(fig)

    # ----------------------------------------------------------------
    # F19 — REPLACES the 3D scatter with a 3-panel small-multiples
    # showing the three pairwise (objective_i vs objective_j) projections
    # of the same Pareto front. Each panel is colored by the third
    # objective. This is far more readable than a rotated 3D plot.
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    panels = [
        ("cost_cabletract_usd",     "decares_per_day_offgrid", "payback_months_vs_fuel",
         "CAPEX (USD)",             "Throughput (dec/day)",    "Payback (months)",
         (cd_capex, cd_throughput)),
        ("cost_cabletract_usd",     "payback_months_vs_fuel",  "decares_per_day_offgrid",
         "CAPEX (USD)",             "Payback (months)",        "Throughput (dec/day)",
         (cd_capex, cd_payback)),
        ("decares_per_day_offgrid", "payback_months_vs_fuel",  "energy_per_decare_Wh",
         "Throughput (dec/day)",    "Payback (months)",        "Energy/decare (Wh)",
         (cd_throughput, cd_payback)),
    ]
    cmap_for_color = {"payback_months_vs_fuel": "viridis_r",
                      "decares_per_day_offgrid": "viridis",
                      "energy_per_decare_Wh": "plasma"}
    for ax, (xc, yc, cc, xl, yl, cl, (cdx, cdy)) in zip(axes, panels):
        sc = ax.scatter(pf_df[xc], pf_df[yc], c=pf_df[cc],
                        cmap=cmap_for_color[cc], s=18, alpha=0.8, linewidths=0.0)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(cl, fontsize=9)
        cb.ax.tick_params(labelsize=8)
        ax.scatter([cdx], [cdy], color="#1b5e1b", s=220, marker="*",
                   edgecolor="black", lw=0.9, zorder=5,
                   label="codesigned reference")
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.tick_params(axis="x", labelsize=9, rotation=20)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "F19. Pareto frontier projections — three pairwise objective views of the same 120-point NSGA-II front\n"
        "Each panel colours the surface by the third objective. Codesigned reference shown as the green star.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT_DIR / "F19_pareto_3d.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ----------------------------------------------------------------
    # Polygon predictor — sanity-check the F9 corpus
    # ----------------------------------------------------------------
    print()
    print("Training polygon throughput predictor on the Phase 4 corpus...")
    fields = load_field_corpus()
    pmodel, pfeats, pr2 = train_polygon_predictor(fields)
    print(f"  polygon predictor held-out R^2 = {pr2:.3f}")
    pd.DataFrame([{"feature": f, "importance": imp}
                  for f, imp in zip(pfeats, pmodel.feature_importances_)]
                 ).sort_values("importance", ascending=False).to_csv(
        TAB_DIR / "Phase6_polygon_predictor_importance.csv", index=False)
    pd.DataFrame([{"target": "polygon_eta_best", "held_out_r2": pr2}]).to_csv(
        TAB_DIR / "Phase6_polygon_predictor_r2.csv", index=False)

    # ----------------------------------------------------------------
    # Headline numbers — does the codesigned point sit on the Pareto front?
    # ----------------------------------------------------------------
    print()
    print("=== Phase 6 headline numbers ===")
    print()
    # Find Pareto points that beat the codesigned reference on at least
    # one objective without being worse on the others (using surrogate
    # predictions for the comparison).
    obj = pf_df[["decares_per_day_offgrid", "payback_months_vs_fuel", "energy_per_decare_Wh"]].values
    cd_obj = np.array([cd_throughput, cd_payback, cd_energy])
    dominates = ((obj[:, 0] >= cd_obj[0] - 1e-6) &
                 (obj[:, 1] <= cd_obj[1] + 1e-6) &
                 (obj[:, 2] <= cd_obj[2] + 1e-6) &
                 ((obj[:, 0] > cd_obj[0]) | (obj[:, 1] < cd_obj[1]) | (obj[:, 2] < cd_obj[2])))
    n_dom = int(dominates.sum())
    print(f"Pareto points dominating the codesigned reference on the surrogate: {n_dom} / {len(pf_df)}")
    if n_dom > 0:
        best = pf_df[dominates].iloc[0]
        print("Example dominator:")
        print(best.to_string())
    else:
        print("codesigned reference appears to be Pareto-optimal on the surrogate (or extrapolation noise).")

    print()
    print(f"Top 5 features for decares_per_day_offgrid:")
    print(feature_importance(sm, "decares_per_day_offgrid", top_k=5).to_string(index=False))

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
