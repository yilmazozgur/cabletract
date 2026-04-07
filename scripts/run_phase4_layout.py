"""Phase 4 driver — generates F9, F10, F11, F12.

F9. Distribution of effective shape efficiency over the 50-field corpus,
    histogram + CDF, for span values L = 30, 50, 75, 100 m.
F10. Three example field overlays with strip plan visualised: rectangle,
    L-shape, irregular concave with obstacles.
F11. Daily-time-budget pie chart for a 5-field, 80-ha farm: working,
    setup, inter-field travel.
F12. Side-by-side compaction map: tractor whole-field coverage vs
    CableTract carriage strip-only coverage on the same field.
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from cabletract.compaction import (  # noqa: E402
    CABLETRACT_CARRIAGE,
    TRACTOR_REFERENCE,
    compacted_path_polygon,
    compaction_summary_for_vehicle,
)
from cabletract.layout import (  # noqa: E402
    best_orientation_efficiency,
    corpus_shape_efficiency_summary,
    farm_tour,
    load_field_corpus,
    strip_plan_segments_xy,
)


OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


# ---------------------------------------------------------------------------
# F9 — Shape-efficiency distribution over the corpus
# ---------------------------------------------------------------------------

def figure_9_shape_efficiency_distribution(out_png: Path, out_csv: Path) -> pd.DataFrame:
    fields = load_field_corpus()
    spans = [30.0, 50.0, 75.0, 100.0]
    rows = []
    for span in spans:
        for r in corpus_shape_efficiency_summary(fields, span=span, swath=2.0, n_orient=12):
            r = dict(r)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for span, color in zip(spans, colors):
        sub = df[df["span_m"] == span]
        axes[0].hist(sub["eta_best"], bins=np.linspace(0, 1, 21), alpha=0.5, color=color, label=f"L={span:.0f} m")
    axes[0].set_xlabel("Effective shape efficiency η")
    axes[0].set_ylabel("Number of fields")
    axes[0].set_title("Histogram (50 fields × 4 spans)")
    axes[0].axvline(0.85, color="black", linestyle="--", lw=1, label="v1 default 0.85")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # CDF
    for span, color in zip(spans, colors):
        sub = df[df["span_m"] == span]
        sorted_eta = np.sort(sub["eta_best"].values)
        cdf = np.arange(1, len(sorted_eta) + 1) / len(sorted_eta)
        axes[1].plot(sorted_eta, cdf, color=color, label=f"L={span:.0f} m", lw=2)
    axes[1].axvline(0.85, color="black", linestyle="--", lw=1, label="v1 default 0.85")
    axes[1].set_xlabel("Effective shape efficiency η")
    axes[1].set_ylabel("CDF (fraction of corpus ≤ η)")
    axes[1].set_title("CDF")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("F9. Shape-efficiency distribution across the 50-field corpus", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F10 — Strip-plan visualisation on three example fields
# ---------------------------------------------------------------------------

def _plot_polygon(ax, poly, face="#a8e6a8", edge="#2a7a2a", lw=1.2) -> None:
    """Plot a shapely Polygon (with optional holes) on a matplotlib Axes.

    Explicitly updates the axes data limits and triggers an autoscale, since
    ``add_patch`` alone does not always rescale the visible window when no
    other artist (e.g. ``ax.plot``) is added afterwards.
    """
    ext = list(poly.exterior.coords)
    ax.add_patch(MplPolygon(ext, closed=True, facecolor=face, edgecolor=edge, lw=lw))
    for hole in poly.interiors:
        ax.add_patch(MplPolygon(list(hole.coords), closed=True, facecolor="white", edgecolor=edge, lw=lw))
    minx, miny, maxx, maxy = poly.bounds
    ax.update_datalim([(minx, miny), (maxx, maxy)])
    ax.autoscale_view()


def figure_10_strip_plans(out_png: Path) -> None:
    fields = load_field_corpus()
    # Pick one rectangle, one L, one irregular concave with holes
    rect = next(f for f in fields if f.shape_class == "rectangle")
    lshape = next(f for f in fields if f.shape_class == "L_shape")
    concave_with_holes = next(
        f for f in fields if f.shape_class == "irregular_concave" and len(f.polygon.interiors) > 0
    )
    selected = [(rect, "Rectangle"), (lshape, "L-shape"), (concave_with_holes, "Irregular concave + obstacle")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (field, label) in zip(axes, selected):
        eta_best, deg_best = best_orientation_efficiency(field.polygon, span=50.0, n_orient=12)
        _plot_polygon(ax, field.polygon)
        midlines = strip_plan_segments_xy(field.polygon, span=50.0, swath=2.0, orientation_deg=deg_best)
        for (a, b) in midlines:
            ax.plot([a[0], b[0]], [a[1], b[1]], color="#d62728", lw=1.5)
        ax.set_aspect("equal")
        ax.set_title(f"{label} ({field.id})\nη={eta_best:.2f}, sweep={deg_best:.0f}°", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("F10. CableTract strip plan on three representative fields (span L=50 m)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# F11 — Daily time budget pie chart for a 5-field 80-ha farm
# ---------------------------------------------------------------------------

def figure_11_time_budget(out_png: Path, out_csv: Path) -> pd.DataFrame:
    fields = load_field_corpus()
    # Pick the 5 largest fields totalling ~80 ha
    by_size = sorted(fields, key=lambda f: f.nominal_area_ha, reverse=True)[:5]
    total_ha = sum(f.nominal_area_ha for f in by_size)

    # Compute working time per field
    work_speed_km_h = 3.0  # CableTract operating speed
    swath_m = 2.0
    setup_h_per_field = 0.5     # Anchor placement + alignment per field
    setup_h_per_strip = 0.05    # 3 minutes per strip transition

    rows = []
    total_work_h = 0.0
    total_setup_h = 0.0
    for f in by_size:
        eta_best, deg_best = best_orientation_efficiency(f.polygon, span=50.0, n_orient=12)
        # Approximate working length: polygon area / swath, scaled by 1/eta
        work_length_m = (f.polygon.area / swath_m) / max(eta_best, 0.05)
        work_h = work_length_m / (work_speed_km_h * 1000.0)
        # Strip count from the corpus summary
        from cabletract.layout import strip_decomposition
        n_strips = len(strip_decomposition(f.polygon, span=50.0, orientation_deg=deg_best))
        setup_h = setup_h_per_field + setup_h_per_strip * n_strips
        total_work_h += work_h
        total_setup_h += setup_h
        rows.append(
            {
                "field_id": f.id,
                "area_ha": f.nominal_area_ha,
                "shape_class": f.shape_class,
                "eta_best": eta_best,
                "n_strips": n_strips,
                "work_h": work_h,
                "setup_h": setup_h,
            }
        )

    # Inter-field travel: TSP tour over centroids
    centroids = [(float(f.polygon.centroid.x), float(f.polygon.centroid.y)) for f in by_size]
    tour_m, _ = farm_tour(centroids, depot=(0.0, 0.0))
    travel_speed_km_h = 6.0  # CableTract on dirt road
    travel_h = tour_m / (travel_speed_km_h * 1000.0)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Time budget pie
    fig, ax = plt.subplots(figsize=(8, 6))
    sizes = [total_work_h, total_setup_h, travel_h]
    labels = [
        f"Operating ({total_work_h:.1f} h)",
        f"Setup / anchor reset ({total_setup_h:.1f} h)",
        f"Inter-field travel ({travel_h:.1f} h)",
    ]
    colors_pie = ["#2a7a2a", "#d4a017", "#5a8fc8"]
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.0f%%", startangle=90)
    for t in autotexts:
        t.set_color("white")
        t.set_weight("bold")
    ax.set_title(
        f"F11. Time budget for one full pass over a {len(by_size)}-field, {total_ha:.1f} ha farm\n"
        f"(largest 5 fields in the corpus, {total_work_h + total_setup_h + travel_h:.1f} h "
        f"at 3 km/h working / 6 km/h transit)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# F12 — Compaction map: tractor vs CableTract carriage
# ---------------------------------------------------------------------------

def figure_12_compaction_map(out_png: Path, out_csv: Path) -> pd.DataFrame:
    fields = load_field_corpus()

    # One representative field per class for the visualisation
    pick = {}
    for f in fields:
        if f.shape_class not in pick:
            pick[f.shape_class] = f
    selected = [pick[c] for c in ["rectangle", "L_shape", "irregular_concave"] if c in pick]

    # Side-by-side panels per field
    fig, axes = plt.subplots(len(selected), 2, figsize=(11, 4.0 * len(selected)))
    if len(selected) == 1:
        axes = np.array([axes])
    rows = []
    for i, field in enumerate(selected):
        # Tractor — whole field
        _plot_polygon(axes[i, 0], field.polygon, face="#cd5c5c", edge="black")
        axes[i, 0].set_aspect("equal")
        axes[i, 0].set_title(f"Tractor (4 passes)\n{field.id} {field.shape_class}", fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)

        # CableTract carriage — strip bands only
        _plot_polygon(axes[i, 1], field.polygon, face="#e8f5e8", edge="#2a7a2a")
        carriage_path = compacted_path_polygon(field.polygon, CABLETRACT_CARRIAGE, span=50.0)
        if hasattr(carriage_path, "geoms"):
            for g in carriage_path.geoms:
                if g.geom_type == "Polygon":
                    _plot_polygon(axes[i, 1], g, face="#d62728", edge="#8b0000", lw=0.6)
        elif carriage_path.geom_type == "Polygon" and not carriage_path.is_empty:
            _plot_polygon(axes[i, 1], carriage_path, face="#d62728", edge="#8b0000", lw=0.6)
        axes[i, 1].set_aspect("equal")
        axes[i, 1].set_title(f"CableTract carriage (4 passes)\n{field.id} {field.shape_class}", fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)

        # Recompute summaries for the table
        st = compaction_summary_for_vehicle(field.polygon, TRACTOR_REFERENCE, span=50.0)
        sc = compaction_summary_for_vehicle(field.polygon, CABLETRACT_CARRIAGE, span=50.0)
        rows.append(
            {
                "field_id": field.id,
                "shape_class": field.shape_class,
                "field_area_m2": st.field_area_m2,
                "tractor_compacted_m2": st.compacted_area_m2,
                "tractor_compacted_frac": st.compacted_area_frac,
                "carriage_compacted_m2": sc.compacted_area_m2,
                "carriage_compacted_frac": sc.compacted_area_frac,
                "area_reduction_pct": 100.0 * (1.0 - sc.compacted_area_m2 / st.compacted_area_m2),
                "tractor_mean_p_kPa": st.mean_pressure_kPa,
                "carriage_mean_p_kPa": sc.mean_pressure_kPa,
                "tractor_energy_idx": st.compaction_energy_index,
                "carriage_energy_idx": sc.compaction_energy_index,
                "energy_reduction_factor": st.compaction_energy_index / max(sc.compaction_energy_index, 1e-9),
            }
        )

    fig.suptitle(
        "F12. Compaction footprint — conventional tractor vs CableTract carriage\n"
        "Left: tractor traffic covers the whole field (red). Right: only the carriage's narrow strip rolls inside the field (red strips on green polygon). 4 passes / cropping season.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating F9  — shape-efficiency distribution (50 fields × 4 spans)...")
    f9 = figure_9_shape_efficiency_distribution(
        OUT_DIR / "F9_shape_efficiency_distribution.png",
        TAB_DIR / "F9_shape_efficiency_distribution.csv",
    )
    print("Generating F10 — strip-plan visualisation on 3 example fields...")
    figure_10_strip_plans(OUT_DIR / "F10_strip_plans.png")
    print("Generating F11 — daily time budget for a 5-field farm...")
    f11 = figure_11_time_budget(OUT_DIR / "F11_time_budget.png", TAB_DIR / "F11_time_budget.csv")
    print("Generating F12 — compaction map (tractor vs carriage)...")
    f12 = figure_12_compaction_map(OUT_DIR / "F12_compaction_map.png", TAB_DIR / "F12_compaction_map.csv")

    print()
    print("=== Phase 4 headline numbers ===")
    print()
    print("F9. Median best-orientation η per shape class (span=50 m):")
    sub50 = f9[f9["span_m"] == 50.0]
    for cls in sorted(sub50["shape_class"].unique()):
        vals = sub50[sub50["shape_class"] == cls]["eta_best"]
        print(f"  {cls:20s} n={len(vals):2d}  median={vals.median():.3f}  P10={vals.quantile(0.1):.3f}  P90={vals.quantile(0.9):.3f}")
    print()
    print("F9. Corpus median η across all 50 fields, by span:")
    for span in sorted(f9["span_m"].unique()):
        vals = f9[f9["span_m"] == span]["eta_best"]
        print(f"  L={span:.0f} m  median={vals.median():.3f}  fraction below v1 default 0.85: {(vals < 0.85).mean():.0%}")

    print()
    print("F11. Daily time budget summary:")
    print(f11.to_string(index=False, float_format="%.2f"))

    print()
    print("F12. Compaction reduction summary:")
    cols = ["field_id", "shape_class", "tractor_compacted_frac", "carriage_compacted_frac", "area_reduction_pct", "energy_reduction_factor"]
    print(f12[cols].to_string(index=False, float_format="%.2f"))

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
