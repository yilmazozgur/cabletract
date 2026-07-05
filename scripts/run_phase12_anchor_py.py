"""Phase 12 driver (S7) -- helical-auger lateral capacity from p--y analysis.

F28. Replaces section 5.4's dual literature reference (Khand ~400 N vs Magnum
     ~2 kN) with one internally derived per-auger nominal:
       (a) API/Reese sand p--y curves at sample depths;
       (b) pile-head pushover (load vs ground-line deflection), free/fixed head,
           with the IBC 1-inch serviceability limit and capacity markers;
       (c) derived per-auger capacities vs the cited literature values;
       (d) recomputed auger count vs P90 draft using the derived nominal,
           against the old 400 N / 2 kN bounds.
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

from cabletract.anchor_py import (  # noqa: E402
    IBC_DEFLECTION_LIMIT_M, LOOSE_SAND, MEDIUM_DENSE_SAND, PileSection,
    SandProfile, group_capacity, lateral_capacity, p_ultimate, py_resistance,
    required_augers_for, solve_pile,
)

OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

KHAND_N = 400.0      # conservative loose-sand 4-pile-raft per-pile (Khand 2024)
MAGNUM_N = 2000.0    # medium-dense fixed-head datasheet nominal (Magnum 2024)


def pushover(section: PileSection, sand: SandProfile, head_fixity: str,
             H_max: float = 4.0e4, n: int = 80):
    """Load-deflection curve up to soil plasticisation (non-convergence)."""
    Hs, ys = [], []
    for H in np.linspace(H_max / n, H_max, n):
        s = solve_pile(section, float(H), sand=sand, head_fixity=head_fixity)
        if not s.converged or s.head_deflection_m < 0 or s.head_deflection_m > 0.2:
            break
        Hs.append(H)
        ys.append(s.head_deflection_m)
    return np.array(ys), np.array(Hs)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    # v3: the installed anchors are 60 mm OD x 1.0 m ground screws (the
    # geometry the 60-s parallel insert/retract budget and the 25.3
    # Wh/round anchoring-energy term assume — cabletract.anchoring), not
    # the 73 mm x 2.0 m construction pile v2 analysed. The deep pile is
    # retained in the manuscript text as a heavy-anchor option.
    sec = PileSection(diameter_m=0.060, wall_m=0.004, length_m=1.0)
    soils = [LOOSE_SAND, MEDIUM_DENSE_SAND]

    # --- capacities + group nominal table ---
    rows = []
    groups = {}
    for sand in soils:
        g = group_capacity(sec, sand, spacing_over_d=3.0, safety_factor=1.5)
        groups[sand.name] = g
        rows.append({
            "soil": sand.name,
            "phi_deg": sand.phi_deg,
            "k_subgrade_MNpm3": sand.k_subgrade_Npm3 / 1e6,
            "single_free_N": round(g.single_free_N, 1),
            "single_fixed_N": round(g.single_fixed_N, 1),
            "group_efficiency": round(g.group_efficiency, 3),
            "per_auger_group_free_N": round(g.per_auger_group_free_N, 1),
            "nominal_working_per_auger_N": round(g.nominal_working_per_auger_N, 1),
            "safety_factor": g.safety_factor,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB_DIR / "F28_anchor_py.csv", index=False)

    # adopted (conservative) nominal = loose-sand group-derived working value
    nominal = groups["loose"].nominal_working_per_auger_N

    # ---------------------------------------------------------------- figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    a, b, c, d = axes.flatten()

    # (a) p--y curves at sample depths
    yy = np.linspace(0, 0.05, 200)
    for sand, ls in zip(soils, ("-", "--")):
        for zc, col in zip((0.3, 0.6, 1.0), ("#9ecae1", "#4292c6", "#08519c")):
            z = np.full_like(yy, zc)
            p = py_resistance(yy, z, sand, sec.diameter_m)
            a.plot(yy * 1e3, p / 1e3, ls=ls, color=col, lw=1.6,
                   label=f"{sand.name}, z={zc:.1f} m")
    a.set_xlabel("Lateral deflection y (mm)")
    a.set_ylabel("Soil reaction p (kN/m)")
    a.set_title("(a) API/Reese sand p--y curves")
    a.legend(fontsize=7, ncol=2); a.grid(True, alpha=0.3)

    # (b) pushover load-deflection
    styles = {("loose", "free"): ("#cd5c5c", "-"),
              ("loose", "fixed"): ("#cd5c5c", "--"),
              ("medium-dense", "free"): ("#1f77b4", "-"),
              ("medium-dense", "fixed"): ("#1f77b4", "--")}
    for sand in soils:
        for fix in ("free", "fixed"):
            ys, Hs = pushover(sec, sand, fix)
            col, ls = styles[(sand.name, fix)]
            b.plot(ys * 1e3, Hs / 1e3, color=col, ls=ls, lw=1.8,
                   label=f"{sand.name}, {fix}-head")
            cap = lateral_capacity(sec, sand, head_fixity=fix)
            b.plot(IBC_DEFLECTION_LIMIT_M * 1e3, cap / 1e3, "o", color=col, ms=5)
    b.axvline(IBC_DEFLECTION_LIMIT_M * 1e3, color="black", ls=":", lw=1.2,
              label="IBC 1-inch limit")
    b.set_xlabel("Ground-line deflection (mm)")
    b.set_ylabel("Pile-head load H (kN)")
    b.set_title("(b) Pushover & serviceability capacity")
    b.legend(fontsize=7); b.grid(True, alpha=0.3); b.set_xlim(0, 40)

    # (c) derived capacities vs cited literature
    labels = ["loose\nfree", "loose\nfixed", "med-dense\nfree", "med-dense\nfixed",
              "loose group\nnominal", "med-dense group\nnominal"]
    vals = [groups["loose"].single_free_N, groups["loose"].single_fixed_N,
            groups["medium-dense"].single_free_N, groups["medium-dense"].single_fixed_N,
            groups["loose"].nominal_working_per_auger_N,
            groups["medium-dense"].nominal_working_per_auger_N]
    cols = ["#9ecae1", "#4292c6", "#9ecae1", "#4292c6", "#2a7a2a", "#2a7a2a"]
    c.bar(range(len(vals)), np.array(vals) / 1e3, color=cols)
    c.axhline(KHAND_N / 1e3, color="#d62728", ls="--", lw=1.4,
              label=f"Khand 2024 (~{KHAND_N:.0f} N)")
    c.axhline(MAGNUM_N / 1e3, color="#ff7f0e", ls="--", lw=1.4,
              label=f"Magnum 2024 (~{MAGNUM_N/1e3:.0f} kN)")
    c.set_xticks(range(len(labels)))
    c.set_xticklabels(labels, fontsize=7)
    c.set_ylabel("Per-auger lateral capacity (kN)")
    c.set_title("(c) Derived capacity vs cited literature")
    c.legend(fontsize=8); c.grid(True, axis="y", alpha=0.3)

    # (d) auger count vs P90 draft, derived nominal vs old bounds
    p90 = np.linspace(500, 8000, 200)
    n_nom = [required_augers_for(t, nominal) for t in p90]
    n_400 = [required_augers_for(t, 400.0) for t in p90]
    n_2k = [required_augers_for(t, 2000.0) for t in p90]
    d.plot(p90 / 1e3, n_400, color="#cd5c5c", lw=2, label="old worst case (400 N)")
    d.plot(p90 / 1e3, n_2k, color="#1f77b4", lw=2, label="old nominal (2 kN)")
    d.plot(p90 / 1e3, n_nom, color="#2a7a2a", lw=2.4,
           label=f"derived nominal ({nominal:.0f} N)")
    d.axvline(3.0, color="black", ls=":", lw=1.2, label="codesigned P90 (3.0 kN)")
    d.axhline(9, color="grey", ls="-.", lw=1.0, alpha=0.7, label="installed 9 augers")
    d.set_xlabel("P90 draft the anchor must hold (kN)")
    d.set_ylabel("Augers required (SF 1.15)")
    d.set_title("(d) Auger count: derived nominal vs old dual reference")
    d.legend(fontsize=7); d.grid(True, alpha=0.3); d.set_ylim(0, 25)

    fig.suptitle("Helical-auger lateral capacity from a nonlinear p--y Winkler model (S7)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "F28_anchor_py.png", dpi=200)
    plt.close(fig)

    # ----------------------------------------------------------- console summary
    print("=== S7 anchor p--y lateral capacity ===")
    print(f"pile: {sec.diameter_m*1e3:.0f} mm OD x {sec.wall_m*1e3:.1f} mm wall, "
          f"L={sec.length_m:.1f} m, EI={sec.EI:.3e} N.m^2")
    print(df.to_string(index=False))
    print(f"\nAdopted conservative nominal (loose, group, SF 1.5) = {nominal:.0f} N")
    print(f"At codesigned P90 = 3.0 kN: augers required = "
          f"{required_augers_for(3000.0, nominal)} "
          f"(was 9 @400 N worst-case / 2 @2 kN nominal)")
    print(f"Figure -> {OUT_DIR/'F28_anchor_py.png'}")
    print(f"Table  -> {TAB_DIR/'F28_anchor_py.csv'}")


if __name__ == "__main__":
    main()
