"""Phase 6b driver — F20 architecture comparison bar chart.

F20. Architecture comparison: codesigned baseline / CableTract+ /
     regen on return leg, on five metrics (energy/decare, decares/day,
     CAPEX, payback, surplus power). Each variant is run on the same
     codesigned parameter set so the bars are directly comparable.
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

from cabletract.params import CableTractParams  # noqa: E402
from cabletract.variants import compare_all_variants  # noqa: E402


OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Comparing 3 architectural variants on the codesigned reference parameter set...")
    all_rows = compare_all_variants(CableTractParams.codesigned())
    # Keep only the three variants the manuscript discusses: codesigned
    # baseline, CableTract+ (4-MU planar cable robot), and regenerative
    # return leg. Circular pulley and drone alignment are dropped.
    keep = {
        "Codesigned baseline (Main Unit + Anchor)",
        "CableTract+ (4-Main-Unit cable robot)",
        "Regenerative return leg",
    }
    rows = [r for r in all_rows if r.name in keep]
    df = pd.DataFrame([{
        "variant": r.name,
        "decares_per_day_offgrid": r.decares_per_day_offgrid,
        "energy_per_decare_Wh": r.energy_per_decare_Wh,
        "cost_cabletract_usd": r.cost_cabletract_usd,
        "payback_months_vs_fuel": r.payback_months_vs_fuel,
        "surplus_power_W": r.surplus_power_W,
    } for r in rows])
    df.to_csv(TAB_DIR / "F20_variant_comparison.csv", index=False)

    # F20 — 5-panel bar chart (one panel per metric, x-axis = variant)
    metrics = [
        ("decares_per_day_offgrid", "Throughput (decares/day, off-grid)", "#2a7a2a", False),
        ("energy_per_decare_Wh",    "Energy intensity (Wh/decare)",        "#1f77b4", False),
        ("cost_cabletract_usd",     "CAPEX (USD)",                         "#d4a017", False),
        ("payback_months_vs_fuel",  "Payback vs diesel (months)",          "#d62728", False),
        ("surplus_power_W",         "Surplus power (W)",                   "#9467bd", False),
    ]

    # Three variants fit cleanly in a 2-row, 3-col grid: top row = the
    # three "scalar" metrics (throughput, energy, capex), bottom row =
    # the two derived metrics (payback, surplus). The 6th slot stays
    # empty so each panel has plenty of horizontal room for labels.
    fig, axes2d = plt.subplots(2, 3, figsize=(15, 9.0))
    axes = axes2d.flatten()
    axes[5].axis("off")  # leave the 6th slot empty for breathing room

    short_labels = [
        "Codesigned\nbaseline",
        "CableTract+\n(4-MU)",
        "Regen\nreturn",
    ]
    x = np.arange(len(rows))
    for ax, (col, ylabel, color, _) in zip(axes[:5], metrics):
        vals = df[col].values
        bars = ax.bar(x, vals, color=color, edgecolor="black", lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        # Make room above the tallest bar for the value label
        ymax = float(np.max(vals))
        ymin = float(min(0.0, np.min(vals)))
        ax.set_ylim(ymin, ymax * 1.22 if ymax > 0 else 1.0)
        # Annotate bars
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.1f}" if abs(v) < 1000 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
        # Highlight codesigned baseline with a thick black border
        bars[0].set_edgecolor("black")
        bars[0].set_linewidth(2.4)

    fig.suptitle(
        "F20. Architectural variant comparison on the codesigned reference parameter set\n"
        "Codesigned baseline outlined in black. Lower is better for energy, CAPEX, and payback;\n"
        "higher is better for throughput and surplus.",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "F20_variant_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print()
    print("=== Phase 6b headline numbers ===")
    print()
    print(df.to_string(index=False, float_format="%.2f"))

    # Per-metric Δ vs baseline
    print()
    base = df.iloc[0]
    print("Δ vs codesigned baseline:")
    for _, row in df.iloc[1:].iterrows():
        print(f"  {row['variant']:42s}  "
              f"throughput {row['decares_per_day_offgrid'] / base['decares_per_day_offgrid']:.2f}×  "
              f"energy {row['energy_per_decare_Wh'] / base['energy_per_decare_Wh']:.2f}×  "
              f"capex {row['cost_cabletract_usd'] / base['cost_cabletract_usd']:.2f}×")

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
