"""Phase 7 driver — operating envelope figure F21.

F21. CableTract operating envelope on (annual GHI × farm size). The
     codesigned reference is NPV-positive at every point on the swept
     plane (see Phase 5 / §5.8 for the underlying economics chain), so
     the only quantity that varies meaningfully across the envelope is
     the *off-grid energy balance* — does the PV harvest cover the
     annual operating demand without grid backup?

     This is the single-question framing the figure is built around:

       (a) Heatmap of annual off-grid surplus (kWh/yr) on the
           (annual GHI × farm size) plane, diverging colormap centred
           at 0, with the off-grid breakeven contour (surplus = 0)
           overlaid in black. The 6 Phase 3 reference sites are
           scattered at 25 ha so the reader can read each site's
           verdict at a glance.

       (b) Histogram of discounted payback values across every cell
           in the swept envelope. This panel is the "all cells are
           NPV-positive and pay back in under 2 years" finding shown
           visually instead of asserted in the caption — the
           distribution sits entirely below 2 years and never crosses
           the financial-feasibility threshold a small-farm investor
           would care about.

The PV harvest model is a one-parameter linear fit calibrated to
Phase 3:

    annual_harvest_kWh = α × pv_area_m² × annual_GHI_kWh_per_m²_yr

with α fitted as the median across the 6 bundled sites at the
codesigned 15 m² PV.

The previous version of this script also ran a "subsidised diesel @
0.70 €/L" stress-test panel and overlaid four families of contour
lines (NPV = 0, payback = 1/3/5/7 yr). Both were dropped because in
the realistic swept range NPV stays positive in 100 % of cells and
discounted payback stays under 2 years in 100 % of cells, so those
overlays were always invisible in practice and confused the figure.
The simplified design here keeps only the one constraint that
actually varies across the envelope.
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

from cabletract.economics import (  # noqa: E402
    EconParams,
    cabletract_npv_vs_diesel,
    cabletract_payback_vs_diesel,
)
from cabletract.energy import load_site_meta, synthesize_tmy_year  # noqa: E402
from cabletract.params import CableTractParams  # noqa: E402
from cabletract.simulate import run_single  # noqa: E402

OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


# ---------------------------------------------------------------------------
# Calibration of the linear PV harvest model α from Phase 3
# ---------------------------------------------------------------------------

def calibrate_alpha(pv_area_m2: float = 15.0) -> tuple[float, pd.DataFrame]:
    """Fit α in annual_harvest_kWh = α × pv_area × annual_GHI to the 6
    bundled sites' Phase 3 outputs."""
    sites = load_site_meta()
    rows = []
    for name, meta in sites.items():
        tmy = synthesize_tmy_year(name, year=2024, seed=0)
        annual_ghi_kWh_per_m2 = float(tmy["ghi_W_m2"].sum() / 1000.0)
        from cabletract.energy import (
            PanelSpec,
            WindSpec,
            daily_harvested_energy,
        )
        panel = PanelSpec(area_m2=pv_area_m2)
        wind = WindSpec(swept_area_m2=2.0)
        daily = daily_harvested_energy(tmy, panel, wind)
        annual_solar_kWh = float(daily["solar_Wh"].sum() / 1000.0)
        rows.append({
            "site": name,
            "annual_GHI_kWh_per_m2_yr": annual_ghi_kWh_per_m2,
            "annual_harvest_kWh": annual_solar_kWh,
            "alpha_per_site": annual_solar_kWh / (pv_area_m2 * annual_ghi_kWh_per_m2),
        })
    df = pd.DataFrame(rows)
    return float(df["alpha_per_site"].median()), df


# ---------------------------------------------------------------------------
# Single-scenario envelope sweep
# ---------------------------------------------------------------------------

def _sweep_envelope(
    base: EconParams,
    annual_ghi_axis: np.ndarray,
    farm_size_axis: np.ndarray,
    alpha: float,
    decares_per_day: float,
):
    """Sweep one (GHI × farm size) grid and return surplus / NPV / payback arrays.

    Only the baseline diesel scenario (€1.40/L from EconParams.codesigned)
    is evaluated; the previous subsidised-diesel stress test has been
    dropped because both scenarios produced NPV-positive payback < 2 yr
    in 100 % of cells, so the second panel added complexity without
    new information.
    """
    operating_hours_per_day = 6.0
    idle_W = 50.0

    surplus = np.zeros((len(farm_size_axis), len(annual_ghi_axis)))
    npv = np.zeros_like(surplus)
    payback = np.zeros_like(surplus)

    for i, farm_ha in enumerate(farm_size_axis):
        days_to_cover = farm_ha / max(decares_per_day / 10.0, 1e-6)
        operating_hours_yr = min(operating_hours_per_day * days_to_cover, 8760.0)
        idle_hours_yr = max(8760.0 - operating_hours_yr, 0.0)
        operating_kWh_yr = base.energy_per_ha_kWh * farm_ha
        idle_kWh_yr = idle_W * idle_hours_yr / 1000.0
        demand_kWh_yr = operating_kWh_yr + idle_kWh_yr

        for j, ghi in enumerate(annual_ghi_axis):
            harvest_kWh_yr = alpha * base.pv_area_m2 * ghi
            grid_kWh_yr = max(0.0, demand_kWh_yr - harvest_kWh_yr)
            grid_share_per_ha = grid_kWh_yr / max(farm_ha, 1e-9)

            kwargs = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
            kwargs["annual_hectares"] = float(farm_ha)
            kwargs["grid_share_kWh_per_ha"] = float(grid_share_per_ha)
            p = EconParams(**kwargs)

            surplus[i, j] = harvest_kWh_yr - demand_kWh_yr
            npv[i, j] = cabletract_npv_vs_diesel(p)
            payback[i, j] = cabletract_payback_vs_diesel(p)

    return surplus, npv, payback


# ---------------------------------------------------------------------------
# Figure F21
# ---------------------------------------------------------------------------

def figure_21_envelope(out_png: Path, out_csv: Path) -> pd.DataFrame:
    base = EconParams.codesigned()
    cd_sim = run_single(CableTractParams.codesigned())

    alpha, calib_df = calibrate_alpha(pv_area_m2=base.pv_area_m2)
    print(f"  PV harvest model: harvest = {alpha:.3f} × PV_area × annual_GHI")
    print("  Calibration sites:")
    print(calib_df.to_string(index=False, float_format="%.3f"))

    decares_per_day = float(cd_sim.decares_per_day_offgrid)

    annual_ghi_axis = np.linspace(800.0, 2300.0, 60)
    # Farm size sweep extended to 1000 ha so the off-grid breakeven
    # contour cuts diagonally across the plot rather than sitting in a
    # single corner. The realistic single-machine ceiling is closer to
    # 100–200 ha; the upper range is a stress test that exposes where
    # PV harvest stops covering operating demand.
    farm_size_axis = np.geomspace(1.0, 1000.0, 60)

    print()
    print("  Sweeping baseline scenario (diesel @ 1.40 €/L)...")
    surplus, npv, payback = _sweep_envelope(
        base, annual_ghi_axis, farm_size_axis, alpha, decares_per_day,
    )

    # ----- Persist long-form CSV -----
    rows = []
    for i, farm_ha in enumerate(farm_size_axis):
        for j, ghi in enumerate(annual_ghi_axis):
            rows.append({
                "annual_GHI_kWh_per_m2_yr": float(ghi),
                "farm_size_ha": float(farm_ha),
                "surplus_kWh_yr": float(surplus[i, j]),
                "npv_eur": float(npv[i, j]),
                "payback_yr": float(payback[i, j]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # ----- Figure: 2 panels (heatmap + payback histogram) -----
    fig = plt.figure(figsize=(15.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.28)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    extent = (
        annual_ghi_axis[0], annual_ghi_axis[-1],
        farm_size_axis[0], farm_size_axis[-1],
    )
    GHI, FARM = np.meshgrid(annual_ghi_axis, farm_size_axis)

    surp_max = float(np.abs(surplus).max())
    vmin, vmax = -surp_max, +surp_max

    im = ax_map.imshow(
        surplus,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
    )
    ax_map.set_yscale("log")
    ax_map.set_xlabel("Annual global horizontal irradiance, GHI (kWh m$^{-2}$ yr$^{-1}$)", fontsize=11)
    ax_map.set_ylabel("Annual operating area (ha, log scale)", fontsize=11)
    ax_map.set_title("(a) Off-grid energy balance over (GHI × farm size)", fontsize=11)

    # Off-grid breakeven contour (the only one that exists in the
    # swept range — see module docstring).
    cs = ax_map.contour(GHI, FARM, surplus, levels=[0.0],
                        colors="black", linewidths=2.4, linestyles="-")
    if cs.collections:
        ax_map.clabel(cs, fmt={0.0: "off-grid breakeven"},
                      fontsize=10, inline=True, inline_spacing=6)

    # Site scatter at 25 ha annual area, with staggered labels.
    sites = load_site_meta()
    site_items = sorted(sites.items(),
                        key=lambda kv: kv[1].published_GHI_kWh_m2_yr)
    offsets_y = [+38, -28, +38, -28, +38, -28]
    for (name, meta), dy in zip(site_items, offsets_y):
        ghi_site = meta.published_GHI_kWh_m2_yr
        ax_map.scatter([ghi_site], [25.0], marker="o", s=110, color="white",
                       edgecolor="black", linewidths=1.6, zorder=5)
        label = name.replace("_", " ")
        ax_map.annotate(label, (ghi_site, 25.0),
                        xytext=(0, dy), textcoords="offset points",
                        fontsize=8, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc="white", ec="black", lw=0.6, alpha=0.85),
                        arrowprops=dict(arrowstyle="-",
                                        color="black", lw=0.6))

    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    cbar.set_label("Annual off-grid surplus (kWh/yr)\ngreen = energy positive · red = grid needed",
                   fontsize=9)

    # ----- Right panel: discounted payback histogram across all cells -----
    pb_flat = payback.flatten()
    npv_flat = npv.flatten()
    pb_min = float(pb_flat.min())
    pb_max = float(pb_flat.max())
    pb_med = float(np.median(pb_flat))
    npv_min = float(npv_flat.min())
    npv_pos_pct = 100.0 * float((npv_flat > 0).mean())

    ax_hist.hist(pb_flat, bins=30, color="#2a7a2a", edgecolor="black", lw=0.4)
    ax_hist.axvline(pb_med, color="black", lw=2.0, label=f"median = {pb_med:.2f} yr")
    ax_hist.axvline(2.0, color="#cc0000", lw=1.5, ls="--",
                    label="2 yr investor threshold")
    ax_hist.set_xlabel("Discounted payback vs diesel (years)", fontsize=11)
    ax_hist.set_ylabel("Number of (GHI × farm size) cells", fontsize=11)
    ax_hist.set_title("(b) Financial story across the same envelope", fontsize=11)
    ax_hist.set_xlim(0.0, max(pb_max * 1.15, 2.4))
    ax_hist.grid(True, alpha=0.3, axis="y")
    ax_hist.legend(loc="upper right", fontsize=8, framealpha=0.92)

    # Annotate the headline finding inside the histogram axes.
    ann_text = (
        f"All {len(pb_flat):,d} cells:\n"
        f"  NPV-positive: {npv_pos_pct:.0f} %\n"
        f"  payback range: {pb_min:.2f}–{pb_max:.2f} yr\n"
        f"  min NPV: €{npv_min:,.0f}"
    )
    ax_hist.text(0.04, 0.96, ann_text, transform=ax_hist.transAxes,
                 fontsize=9, va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec="black", lw=0.7, alpha=0.92))

    fig.suptitle(
        "F21. CableTract operating envelope — codesigned reference, diesel @ 1.40 €/L\n"
        "(a) Off-grid energy balance: PV harvest minus annual demand on (GHI × farm size). "
        "Black contour: off-grid breakeven. (b) Discounted payback distribution across every cell of (a) — "
        "the financial story is uniformly favourable, so the binding constraint is climate, not finance.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating F21 — operating envelope (simplified, single scenario)...")
    df = figure_21_envelope(OUT_DIR / "F21_operating_envelope.png",
                            TAB_DIR / "F21_operating_envelope.csv")

    print()
    print("=== Phase 7 headline numbers ===")
    print()
    npv_pos = (df["npv_eur"] > 0.0).mean()
    pb_med = df["payback_yr"].median()
    pb_max = df["payback_yr"].max()
    off_grid = (df["surplus_kWh_yr"] > 0.0).mean()
    print(f"Cells NPV-positive (diesel @ 1.40 €/L):  {npv_pos*100:.1f} %")
    print(f"Discounted payback median / max:         {pb_med:.2f} yr / {pb_max:.2f} yr")
    print(f"Cells off-grid feasible (surplus > 0):   {off_grid*100:.1f} %")

    # Site-by-site verdict at farm ≈ 25 ha (closest grid row).
    print()
    print("Site verdicts at ~25 ha annual operating area:")
    sites = load_site_meta()
    farm_grid = df["farm_size_ha"].unique()
    farm_25 = float(farm_grid[np.abs(farm_grid - 25.0).argmin()])
    sub_25ha = df[df["farm_size_ha"] == farm_25]
    for name, meta in sites.items():
        if sub_25ha.empty:
            continue
        idx = (sub_25ha["annual_GHI_kWh_per_m2_yr"] - meta.published_GHI_kWh_m2_yr).abs().idxmin()
        r = sub_25ha.loc[idx]
        verdict = []
        if r["npv_eur"] > 0:
            verdict.append("NPV+")
        if r["payback_yr"] < 2.0:
            verdict.append("PB<2yr")
        if r["surplus_kWh_yr"] > 0:
            verdict.append("off-grid")
        tag = ",".join(verdict) if verdict else "none"
        print(f"  {name:15s} GHI={meta.published_GHI_kWh_m2_yr:5.0f}  "
              f"NPV={r['npv_eur']/1000:6.1f}k€  PB={r['payback_yr']:4.2f}yr  "
              f"surplus={r['surplus_kWh_yr']:6.0f} kWh/yr  [{tag}]")

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
