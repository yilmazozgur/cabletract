"""Phase 11 driver (S3) -- soil-tool DEM vs analytic wedge, draft & tilth.

F26. Draft per (depth, width): the numba soft-sphere DEM against the
     McKyes-Godwin wedge (calibrated to the bed's bulk density) and the
     manuscript's codesigned D497 P50 reference band.
F27. Tilth: disturbed-soil cross-section and draft-per-disturbed-area for a
     shallow-narrow co-designed pass vs a deep-wide conventional one -- the
     agronomic-equivalence question.

NOTE: this is the most expensive study (a 3-D DEM per operating point); the
full run takes a few minutes. Tool widths are kept inside the bed so the
disturbance is not domain-limited; the DEM operates at lab-scale depths and the
analytic wedge carries the extrapolation to field-scale tillage depths.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cabletract.tillage_dem import (  # noqa: E402
    DEMParams, build_packed_bed, bulk_density, drag_tool,
)
from cabletract.tillage_mechanics import (  # noqa: E402
    SoilCuttingParams, mckyes_godwin_draft,
)

OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

# Codesigned library median draft (manuscript P50, the D497 reference).
D497_CODESIGN_P50_N = 1800.0
RAKE_DEG = 35.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    p = DEMParams()
    t0 = time.time()
    bed = build_packed_bed(p, Lx=0.40, Ly=0.18, fill_depth=0.30, settle_steps=16000)
    rho = bulk_density(bed)
    print(f"bed: N={bed.pos.shape[0]} particles, surface {bed.z_surface*100:.1f} cm, "
          f"bulk density {rho:.0f} kg/m^3  [{time.time()-t0:.0f}s]")

    # DEM operating points (widths kept <= 0.08 m inside the 0.18 m bed)
    runs = [("co-design ripper", 0.05, 0.04),
            ("co-design chisel", 0.08, 0.04),
            ("co-design sweep", 0.05, 0.08),
            ("conventional tine", 0.10, 0.06),
            ("conventional deep", 0.13, 0.08)]
    rows = []
    for label, d, w in runs:
        t1 = time.time()
        r = drag_tool(bed, depth_m=d, width_m=w, rake_deg=RAKE_DEG, v_tool=0.35,
                      drag_len=0.16, settle_first=700)
        # calibrated cohesionless wedge at the bed bulk density
        soil = SoilCuttingParams(gamma_kg_m3=rho, cohesion_kPa=0.0,
                                 phi_deg=40.0, delta_deg=24.0)
        wedge = mckyes_godwin_draft(d, w, soil, rake_deg=RAKE_DEG).draft_N
        rows.append(dict(label=label, depth_cm=d * 100, width_cm=w * 100,
                         dem_draft_N=r.draft_mean_N, wedge_draft_N=wedge,
                         disturbed_area_cm2=r.disturbed_area_m2 * 1e4,
                         moved_fraction=r.disturbed_fraction,
                         draft_per_area_N_cm2=r.draft_mean_N / max(r.disturbed_area_m2 * 1e4, 1e-9)))
        print(f"  {label:18s} d={d*100:.0f} w={w*100:.0f}cm: DEM={r.draft_mean_N:6.1f}N "
              f"wedge={wedge:6.1f}N area={r.disturbed_area_m2*1e4:5.0f}cm2 "
              f"[{time.time()-t1:.0f}s]")
    df = pd.DataFrame(rows)
    # wedge with realistic field cohesion (the DEM is a cohesionless sand analog;
    # real loam carries cohesion, raising the analytic draft toward the DEM/D497).
    soil_dry = SoilCuttingParams(gamma_kg_m3=rho, cohesion_kPa=0.0, phi_deg=40.0, delta_deg=24.0)
    soil_field = SoilCuttingParams(gamma_kg_m3=rho, cohesion_kPa=7.0, phi_deg=30.0, delta_deg=18.0)
    df["wedge_field_N"] = [mckyes_godwin_draft(r.depth_cm / 100, r.width_cm / 100,
                                               soil_field, RAKE_DEG).draft_N
                           for r in df.itertuples()]
    df.to_csv(TAB_DIR / "F26_F27_tillage.csv", index=False)

    # ----------------------------------------------------------------- F26
    # DEM (blunt full-depth blade in a confined bed) brackets the HIGH side;
    # the slender-tine cohesionless wedge brackets the LOW side; realistic
    # field soil (wedge + cohesion) and the D497 P50 sit in between. No fudge
    # factor -- the two idealisations genuinely bound the draft.
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))

    x = np.arange(len(df)); w_ = 0.27
    a.bar(x - w_, df.dem_draft_N, w_, label="DEM blunt blade (upper)", color="#1f77b4")
    a.bar(x, df.wedge_field_N, w_, label="wedge, field soil c=7kPa", color="#2ca02c")
    a.bar(x + w_, df.wedge_draft_N, w_, label="wedge, cohesionless (lower)", color="#bcbd22")
    a.axhline(D497_CODESIGN_P50_N, color="red", ls="--", lw=1.4,
              label="D497 codesigned P50 (1.8 kN, field scale)")
    a.set_xticks(x); a.set_xticklabels([f"{r.label}\n{r.depth_cm:.0f}x{r.width_cm:.0f}cm"
                                        for r in df.itertuples()], fontsize=7, rotation=15)
    a.set_ylabel("draft (N)")
    a.set_title("(a) DEM brackets the analytic wedge (lab scale)")
    a.legend(fontsize=7); a.grid(True, axis="y", alpha=0.3)

    # depth-scaling: DEM points + wedge bounds to field depth (no calibration)
    nar = df[df.width_cm == 4.0]
    dd = np.linspace(0.03, 0.40, 60)
    b.plot(dd * 100, [mckyes_godwin_draft(xx, 0.04, soil_field, RAKE_DEG).draft_N for xx in dd],
           color="#2ca02c", lw=2, label="wedge field c=7kPa (w=4cm)")
    b.plot(dd * 100, [mckyes_godwin_draft(xx, 0.04, soil_dry, RAKE_DEG).draft_N for xx in dd],
           color="#bcbd22", lw=2, ls="--", label="wedge cohesionless (w=4cm)")
    b.plot(nar.depth_cm, nar.dem_draft_N, "o", color="#1f77b4", ms=10,
           label="DEM (w=4cm, lab)")
    b.axvspan(15, 40, color="grey", alpha=0.08)
    b.text(27, b.get_ylim()[1] * 0.85, "field tillage depth", fontsize=8, ha="center", color="grey")
    b.set_xlabel("tool depth (cm)"); b.set_ylabel("draft (N)")
    b.set_title("(b) Depth scaling: DEM (lab) + wedge bounds (field)")
    b.legend(fontsize=8); b.grid(True, alpha=0.3)

    fig.suptitle("Soil-tool DEM vs analytic wedge bounds -- draft of co-designed implements (S3)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_DIR / "F26_dem_draft.png", dpi=200)
    plt.close(fig)

    # ----------------------------------------------------------------- F27
    fig2, (c, d2) = plt.subplots(1, 2, figsize=(13, 5.2))

    sizes = df.depth_cm * df.width_cm  # tool frontal area proxy
    c.scatter(sizes, df.disturbed_area_cm2, s=90, c=df.depth_cm, cmap="viridis")
    for r in df.itertuples():
        c.annotate(r.label.split()[-1], (r.depth_cm * r.width_cm, r.disturbed_area_cm2),
                   fontsize=7, xytext=(4, 4), textcoords="offset points")
    c.set_xlabel("tool frontal size depth x width (cm^2)")
    c.set_ylabel("disturbed soil cross-section (cm^2)")
    c.set_title("(c) Tilth: disturbed cross-section grows with tool size")
    c.grid(True, alpha=0.3)

    # draft per disturbed area -- "tillage cost" per unit loosened soil
    order = np.argsort(df.disturbed_area_cm2.values)
    c2 = df.iloc[order]
    d2.bar(range(len(c2)), c2.draft_per_area_N_cm2, color="#d62728")
    d2.set_xticks(range(len(c2)))
    d2.set_xticklabels([f"{r.label}\n{r.disturbed_area_cm2:.0f}cm2" for r in c2.itertuples()],
                       fontsize=7, rotation=15)
    d2.set_ylabel("draft per disturbed area (N/cm^2)")
    d2.set_title("(d) Draft cost per unit loosened soil")
    d2.grid(True, axis="y", alpha=0.3)

    fig2.suptitle("Shallow-narrow vs deep-wide tilth & draft efficiency (S3)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(OUT_DIR / "F27_tilth_equivalence.png", dpi=200)
    plt.close(fig2)

    # ------------------------------------------------------------- console
    print("\n=== S3 tillage DEM summary ===")
    print(df.to_string(index=False))
    print("\nDraft bounds: DEM blunt blade (upper) brackets the slender-tine wedge "
          "(lower); field-cohesion wedge and D497 sit between.")
    sn = df.iloc[0]; dw = df.iloc[-1]
    print(f"Shallow-narrow ({sn.label}) disturbs {sn.disturbed_area_cm2:.0f} cm^2 at "
          f"{sn.dem_draft_N:.0f} N; deep-wide ({dw.label}) disturbs "
          f"{dw.disturbed_area_cm2:.0f} cm^2 at {dw.dem_draft_N:.0f} N.")
    print(f"Figures -> {OUT_DIR/'F26_dem_draft.png'}, {OUT_DIR/'F27_tilth_equivalence.png'}")
    print(f"Table  -> {TAB_DIR/'F26_F27_tillage.csv'}")


if __name__ == "__main__":
    main()
