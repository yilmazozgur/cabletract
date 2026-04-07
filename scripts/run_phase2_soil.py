"""Phase 2 driver — soil/draft figures (F4, F5).

F4. Two-panel ASABE D497.7 draft comparison.
    (a) Violin plot of P10/P50/P90 draft per implement for the conventional
        ASABE library at conventional tractor field speeds (5–9 km/h).
    (b) The CableTract co-designed library at CableTract operating speeds
        (1–2.5 km/h). Each co-designed implement is paired with the
        conventional implement it replaces; the bar above each pair gives
        the codesigned/conventional draft ratio.

F5. Speed-dependence of draft for three implements that span the ASABE
    coefficient regime: row planter (depth-independent), chisel plow
    (linear in speed), moldboard plow (quadratic). Highlights both the
    CableTract operating window (1–2.5 km/h) and the conventional tractor
    operating window (5–9 km/h), and reports the per-implement draft
    growth from one regime to the other.
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

from cabletract.soil import (  # noqa: E402
    compare_conventional_vs_codesign,
    draft_distribution,
    implement_by_name,
    library_draft_summary,
    load_cabletract_implement_library,
    load_implement_library,
)


OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


CAT_COLORS = {
    "primary_tillage": "#8B4513",
    "secondary_tillage": "#CD853F",
    "seeding": "#228B22",
    "spraying": "#1E90FF",
    "mowing": "#FFD700",
    "weeding": "#9370DB",
}


def _draft_samples_for_library(library, speed_range_km_h, rng):
    """Return a dict {implement_name: numpy.ndarray of draft samples (N)}."""
    out = {}
    for imp in library:
        depth_lo = max(imp.typical_depth_cm - 5.0, 1.0)
        depth_hi = imp.typical_depth_cm + 5.0
        samples = draft_distribution(
            imp,
            soil_texture="medium",
            moisture_range=(0.12, 0.28),
            speed_range_km_h=speed_range_km_h,
            depth_range_cm=(depth_lo, depth_hi),
            n_samples=5000,
            rng=rng,
        )
        out[imp.name] = samples
    return out


# ---------------------------------------------------------------------------
# F4 — CableTract co-designed library draft distribution with anchor envelopes
# ---------------------------------------------------------------------------

def figure_4_draft_comparison(out_png: Path, out_csv: Path) -> pd.DataFrame:
    rng = np.random.default_rng(0xCAB1E)

    codesign = load_cabletract_implement_library()
    cd_samples = _draft_samples_for_library(codesign, (1.0, 2.5), rng)

    # Persist per-implement P10/P50/P90 for the codesigned library.
    rows = []
    for imp in codesign:
        s = cd_samples[imp.name]
        rows.append({
            "library": "co-designed",
            "implement": imp.name,
            "category": imp.category,
            "P10_N": float(np.percentile(s, 10)),
            "P50_N": float(np.percentile(s, 50)),
            "P90_N": float(np.percentile(s, 90)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(11, 7))

    # x-axis upper bound: leave room for the 9-auger Magnum line at 18 kN.
    cd_max = max(np.max(cd_samples[imp.name]) for imp in codesign) / 1000.0
    x_max = max(cd_max * 1.05, 20.0)

    pos = np.arange(len(codesign))
    data = [cd_samples[imp.name] / 1000.0 for imp in codesign]
    parts = ax.violinplot(data, positions=pos, vert=False,
                          showmeans=False, showmedians=True, widths=0.8)
    for body, imp in zip(parts["bodies"], codesign):
        body.set_facecolor(CAT_COLORS[imp.category])
        body.set_alpha(0.7)
        body.set_edgecolor("black")
    ax.set_yticks(pos)
    ax.set_yticklabels([imp.name for imp in codesign], fontsize=10)
    ax.set_xlabel("Draft load (kN)", fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, x_max)

    # Two anchor reference envelopes (matching F2 — plotted as raw 9 ×
    # per_auger capacity, the same convention as the F2 horizontal lines).
    khand_9_kN = 9 * 0.400  # = 3.6 kN
    magnum_9_kN = 9 * 2.000  # = 18 kN

    ax.axvline(khand_9_kN, color="#cc0000", lw=2.2, ls="--", alpha=0.95,
               label=f"9-auger envelope · Khand 2024 (loose sand) = {khand_9_kN:.1f} kN")
    ax.axvline(magnum_9_kN, color="#003c99", lw=2.2, ls="-", alpha=0.95,
               label=f"9-auger envelope · Magnum 2024 (medium-dense) = {magnum_9_kN:.1f} kN")

    # Shade the "loose-sand restricted" zone between the two envelopes,
    # so the reader can see at a glance which implements only run in
    # medium-dense soil.
    ax.axvspan(khand_9_kN, magnum_9_kN, color="#ffcc66", alpha=0.18,
               label="Loose-sand restricted (Magnum-only) zone")

    # Inline label for the Khand line (under the curve, where space is free).
    # The Magnum line is named in the upper-right legend.
    ax.text(khand_9_kN + 0.12, 0.3,
            "9 augers · Khand", color="#cc0000",
            fontsize=9, ha="left", va="bottom", fontweight="bold",
            rotation=90)

    # Category legend (left) + envelope legend (right).
    from matplotlib.patches import Patch
    cat_handles = [
        Patch(facecolor=col, edgecolor="black", alpha=0.7, label=cat.replace("_", " "))
        for cat, col in CAT_COLORS.items()
    ]
    leg1 = ax.legend(handles=cat_handles, loc="lower right",
                     fontsize=8, framealpha=0.92, title="implement category",
                     title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    fig.suptitle(
        "F4. CableTract co-designed implement library — ASABE D497.7 draft distributions\n"
        "Sampled at CableTract operating speeds (1–2.5 km/h). 9-auger Anchor envelopes from F2 overlaid.",
        fontsize=11, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F5 — Speed dependence
# ---------------------------------------------------------------------------

def figure_5_speed_dependence(out_png: Path, out_csv: Path) -> pd.DataFrame:
    selected = ["row_planter", "chisel_plow_twisted", "moldboard_plow"]
    speeds = np.linspace(0.5, 12.0, 60)

    rows = []
    for name in selected:
        imp = implement_by_name(name)
        depth = imp.typical_depth_cm
        for s in speeds:
            D = imp.draft_N(speed_km_h=s, depth_cm=depth, soil_texture="medium", moisture=0.20)
            rows.append({"implement": name, "speed_km_h": s, "draft_N": D, "depth_cm": depth})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"row_planter": "#228B22", "chisel_plow_twisted": "#8B4513", "moldboard_plow": "#CD5C5C"}
    label_map = {
        "row_planter": "Row planter (B=C=0, speed-independent)",
        "chisel_plow_twisted": "Chisel plow (B>0, linear in speed)",
        "moldboard_plow": "Moldboard plow (C>0, quadratic in speed)",
    }
    for name in selected:
        sub = df[df["implement"] == name]
        ax.plot(sub["speed_km_h"], sub["draft_N"] / 1000.0, label=label_map[name], color=colors[name], lw=2)

    # CableTract operating window (1–2.5 km/h).
    ax.axvspan(1.0, 2.5, alpha=0.15, color="green")
    ax.text(1.75, ax.get_ylim()[1] * 0.05, "CableTract\noperating window\n(1–2.5 km/h)",
            ha="center", fontsize=8, color="darkgreen")

    # Conventional tractor operating window (5–9 km/h).
    ax.axvspan(5.0, 9.0, alpha=0.15, color="grey")
    ax.text(7.0, ax.get_ylim()[1] * 0.05, "Conventional tractor\noperating window\n(5–9 km/h)",
            ha="center", fontsize=8, color="dimgrey")

    ax.set_xlabel("Field speed (km/h)")
    ax.set_ylabel("Draft load (kN)")
    ax.set_title("F5. Speed-dependence of draft — slow operation suppresses the v² term")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    f4 = figure_4_draft_comparison(OUT_DIR / "F4_draft_violin.png", TAB_DIR / "F4_draft_violin.csv")
    f5 = figure_5_speed_dependence(OUT_DIR / "F5_speed_dependence.png", TAB_DIR / "F5_speed_dependence.csv")

    # Co-design comparison summary table — used by §3.5 of the manuscript.
    cmp = compare_conventional_vs_codesign()
    cmp.to_csv(TAB_DIR / "F4_codesign_comparison.csv", index=False)

    print("=== Phase 2 headline numbers ===")
    print()
    print("F4. Conventional ASABE library at 5–9 km/h, P10/P50/P90 (medium soil):")
    conv = f4[f4["library"] == "conventional"]
    print(conv[["implement", "P10_N", "P50_N", "P90_N"]].to_string(index=False, float_format="%.0f"))

    print()
    print("F4. CableTract co-designed library at 1–2.5 km/h, P10/P50/P90 (medium soil):")
    cd = f4[f4["library"] == "co-designed"]
    print(cd[["implement", "P10_N", "P50_N", "P90_N"]].to_string(index=False, float_format="%.0f"))

    print()
    print("F4. Co-designed/conventional draft ratio per operation (P50):")
    for _, r in cmp.iterrows():
        ratio = r["codesigned_over_conventional_P50"]
        marker = "✔ within 3 kN" if r["codesigned_P50_N"] < 3000 else (
            "△ borderline (3–5 kN)" if r["codesigned_P50_N"] < 5000 else "✗ exceeds 5 kN"
        )
        print(
            f"  {r['conventional_implement']:30s}  {r['conventional_P50_N']:7.0f} N → "
            f"{r['codesigned_implement']:25s} {r['codesigned_P50_N']:6.0f} N "
            f"(ratio {ratio:.2f})  {marker}"
        )

    median_ratio = float(cmp["codesigned_over_conventional_P50"].median())
    n_within = int((cmp["codesigned_P50_N"] < 3000).sum())
    print()
    print(
        f"  → median co-designed/conventional draft ratio: {median_ratio:.2f}  "
        f"({n_within} of {len(cmp)} co-designed implements sit at P50 < 3 kN)"
    )

    print()
    print("F5. Speed dependence — moldboard draft growth from 2 to 8 km/h:")
    mb_2 = f5[(f5["implement"] == "moldboard_plow") & (np.isclose(f5["speed_km_h"], 2.0, atol=0.15))]["draft_N"].iloc[0]
    mb_8 = f5[(f5["implement"] == "moldboard_plow") & (np.isclose(f5["speed_km_h"], 8.0, atol=0.15))]["draft_N"].iloc[0]
    print(f"  moldboard@2 km/h = {mb_2:.0f} N")
    print(f"  moldboard@8 km/h = {mb_8:.0f} N  ({mb_8/mb_2:.2f}x)")

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
