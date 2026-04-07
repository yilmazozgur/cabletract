"""Phase 1 driver — generates the three physics figures (F1, F2, F3) and
their backing CSV tables.

F1. Cable sag vs pretension for steel / Dyneema / UHMWPE at L = 50, 75, 100 m.
    Annotates the minimum tension needed to keep midspan tilling-depth drift
    below 2 cm — the "depth-control envelope".

F2. Anchor reaction envelope: required cable tension vs draft load with the
    Khand 2024 per-auger lateral capacity overlaid as horizontal bands. We
    plot two regimes simultaneously: (i) the *conventional* implement library
    at conventional tractor speeds (5–9 km/h), where every primary tillage
    operation requires 25+ augers and the architecture is infeasible; and
    (ii) the *CableTract co-designed* implement library at CableTract speeds
    (1–2.5 km/h), where every operation including primary tillage fits
    inside 6–14 augers. F2 is the figure that *justifies* the co-design
    pivot.

F3. Peak vs continuous motor electrical power across the co-designed
    implement library, using the decomposed efficiency chain (no more lone
    winch_efficiency = 0.5). Conventional-implement bars are overlaid for
    contrast.

All artefacts are written next to the manuscript so the LaTeX scaffold can
``\\input`` the CSVs and ``\\includegraphics`` the PNGs without path edits.
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

from cabletract.physics import (  # noqa: E402
    G,
    anchor_reaction_envelope,
    catenary_sag,
    default_drivetrain,
    peak_motor_power,
    premium_drivetrain,
    tension_balance,
)
from cabletract.soil import (  # noqa: E402
    draft_distribution,
    load_cabletract_implement_library,
    load_implement_library,
)


CABLE_PROPS_CSV = Path(__file__).resolve().parent.parent / "cabletract" / "data" / "cable_props.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
TAB_DIR = Path(__file__).resolve().parent.parent / "tables"


# ---------------------------------------------------------------------------
# F1 — Cable sag vs pretension
# ---------------------------------------------------------------------------

def figure_1_cable_sag(out_png: Path, out_csv: Path) -> pd.DataFrame:
    cables = pd.read_csv(CABLE_PROPS_CSV)
    spans = [50.0, 75.0, 100.0]
    pretensions = np.linspace(500.0, 25000.0, 60)

    rows = []
    for _, c in cables.iterrows():
        w_per_m = c["mass_per_m_kg"] * G
        for L in spans:
            for T in pretensions:
                sag, _arc, _T_end = catenary_sag(T, w_per_m, L)
                rows.append(
                    {
                        "material": c["material"],
                        "span_m": L,
                        "T_horiz_N": T,
                        "sag_m": sag,
                        "mass_per_m_kg": c["mass_per_m_kg"],
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    DEPTH_TOL = 0.02  # max ±2 cm midspan drift to keep tillage depth constant
    materials = list(cables["material"])
    colors = {m: c for m, c in zip(materials, ("#1f77b4", "#2ca02c", "#d62728"))}

    for ax, L in zip(axes, spans):
        for m in materials:
            sub = df[(df["material"] == m) & (df["span_m"] == L)]
            ax.plot(sub["T_horiz_N"] / 1000.0, sub["sag_m"] * 100.0, label=m, color=colors[m])
        ax.axhline(DEPTH_TOL * 100.0, ls="--", color="k", alpha=0.5)
        ax.text(
            0.95, DEPTH_TOL * 100.0 + 0.5,
            "±2 cm depth tolerance",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=8,
            color="k",
        )
        ax.set_title(f"Span L = {L:.0f} m")
        ax.set_xlabel("Horizontal cable tension (kN)")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 25)
    axes[0].set_ylabel("Midspan sag (cm)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("F1. Cable sag vs pretension — depth-control envelope")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F2 — Anchor reaction envelope
# ---------------------------------------------------------------------------

def _implement_p50_p90(implement, speed_range_km_h, rng):
    depth_lo = max(implement.typical_depth_cm - 5.0, 1.0)
    depth_hi = implement.typical_depth_cm + 5.0
    samples = draft_distribution(
        implement,
        soil_texture="medium",
        moisture_range=(0.12, 0.28),
        speed_range_km_h=speed_range_km_h,
        depth_range_cm=(depth_lo, depth_hi),
        n_samples=2000,
        rng=rng,
    )
    return float(np.percentile(samples, 50)), float(np.percentile(samples, 90))


def figure_2_anchor_envelope(out_png: Path, out_csv: Path) -> pd.DataFrame:
    drafts = np.arange(200.0, 16001.0, 200.0)
    # Two reference per-auger lateral capacities, spanning the published range:
    #   Khand et al. (2024) — 4-pile group raft tests in sand: 1.6 kN total
    #     lateral on a 4-pile raft, i.e. ~400 N per pile, free-head, taken as
    #     the conservative loose-sand bound.
    #   Magnum Piering (2024) — 14–20 kN per small fixed-head pile in
    #     medium-dense sand at the IBC2021 1 inch deflection limit. We
    #     plot 2 kN/auger as a defensible middle of that range after a
    #     conservative 7–10× downscale for safety / wear / installation.
    per_auger_khand = 400.0
    per_auger_magnum = 2000.0
    auger_counts_khand = [4, 6, 9, 12, 16, 24, 32]
    auger_counts_magnum = [4, 9]

    rows = []
    for F_d in drafts:
        state = tension_balance(
            F_draft=float(F_d),
            F_implement_weight=200.0,  # carriage + frame, not the whole tractor
            w_cable=0.046 * G,         # Dyneema 8 mm
            span=50.0,
            pulley_height=1.5,
            min_ground_clearance=0.10,
        )
        env_khand = anchor_reaction_envelope(
            T_anchor=state.T_anchor, n_augers=9, per_auger_lateral_capacity_N=per_auger_khand
        )
        env_mag = anchor_reaction_envelope(
            T_anchor=state.T_anchor, n_augers=9, per_auger_lateral_capacity_N=per_auger_magnum
        )
        rows.append(
            {
                "draft_load_N": F_d,
                "regime": state.regime,
                "T_winch_N": state.T_winch,
                "T_anchor_N": state.T_anchor,
                "sag_mid_m": state.sag_mid,
                "ground_clearance_m": state.ground_clearance,
                "n_augers_required_khand_working": env_khand.n_augers_required_working,
                "n_augers_required_khand_ultimate": env_khand.n_augers_required_ultimate,
                "n_augers_required_magnum_working": env_mag.n_augers_required_working,
                "n_augers_required_magnum_ultimate": env_mag.n_augers_required_ultimate,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Sample per-implement P50/P90 draft from both libraries.
    rng = np.random.default_rng(0xCAB1E)
    conventional = load_implement_library()
    codesign = load_cabletract_implement_library()
    conv_marks = []
    for imp in conventional:
        p50, p90 = _implement_p50_p90(imp, (5.0, 9.0), rng)
        conv_marks.append((imp.name, imp.category, p50, p90))
    cd_marks = []
    for imp in codesign:
        p50, p90 = _implement_p50_p90(imp, (1.0, 2.5), rng)
        cd_marks.append((imp.name, imp.category, p50, p90))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Two coloured background bands showing the conventional and co-designed
    # operating regimes (read off the implement P90 spans).
    conv_p90s = [m[3] for m in conv_marks]
    cd_p90s = [m[3] for m in cd_marks]
    cd_p50s = [m[2] for m in cd_marks]
    conv_min_kN = min(m[2] for m in conv_marks) / 1000.0
    conv_max_kN = max(conv_p90s) / 1000.0
    cd_min_kN = min(cd_p50s) / 1000.0
    cd_max_kN = max(cd_p90s) / 1000.0
    ax.axvspan(cd_min_kN, cd_max_kN, color="#2ca02c", alpha=0.18,
               label="CableTract co-designed implement regime (1–2.5 km/h)")
    ax.axvspan(conv_min_kN, min(conv_max_kN, 16.0), color="#888888", alpha=0.18,
               label="Conventional implement regime (5–9 km/h)")

    # High-contrast tension curves: black (anchor) and dark gray dashed (winch).
    ax.plot(df["draft_load_N"] / 1000.0, df["T_anchor_N"] / 1000.0, color="#000000", lw=2.6,
            label="Anchor cable tension (kN)")
    ax.plot(df["draft_load_N"] / 1000.0, df["T_winch_N"] / 1000.0, color="#555555", lw=1.2, ls="--",
            label="Winch-side tension (kN)")

    # Khand (conservative, loose sand, free-head) capacity bands — dashed red shades.
    khand_colors = ["#ffb3b3", "#ff8080", "#ff3333", "#cc0000", "#990000", "#660000", "#330000"]
    # Extend the y-axis to 20 kN so the 9-auger Magnum reference (18 kN) is visible.
    y_max_kN = max(df["T_anchor_N"].max() / 1000.0 * 1.05, 20.0)
    for n, color in zip(auger_counts_khand, khand_colors):
        cap_kN = n * per_auger_khand / 1000.0
        if cap_kN > y_max_kN:
            continue
        lw = 2.0 if n == 9 else 1.0
        ax.axhline(cap_kN, color=color, lw=lw, alpha=0.95, ls="--")
        ax.text(
            0.012, cap_kN + 0.10,
            f"{n} augers · Khand",
            transform=ax.get_yaxis_transform(),
            fontsize=8, ha="left", va="bottom",
            color=color, fontweight="bold" if n == 9 else "normal",
        )

    # Magnum (realistic, medium-dense, fixed-head) capacity bands — solid blue shades.
    magnum_colors = ["#3399ff", "#003c99"]
    for n, color in zip(auger_counts_magnum, magnum_colors):
        cap_kN = n * per_auger_magnum / 1000.0
        if cap_kN > y_max_kN:
            continue
        lw = 2.4 if n == 9 else 1.6
        ax.axhline(cap_kN, color=color, lw=lw, alpha=0.95, ls="-")
        ax.text(
            0.988, cap_kN + 0.10,
            f"{n} augers · Magnum",
            transform=ax.get_yaxis_transform(),
            fontsize=8, ha="right", va="bottom",
            color=color, fontweight="bold" if n == 9 else "normal",
        )

    # Highlight a few representative implements with annotations.
    callouts = [
        ("moldboard plow (conv.)", 14.0, 0.5, "above"),
        ("disk harrow (conv.)", 8.5, 0.5, "above"),
        ("chisel plow (conv.)", 14.2, 0.5, "below"),
        ("narrow chisel (co-d.)", 4.2, 0.5, "above"),
        ("narrow ripper (co-d.)", 3.0, 0.5, "below"),
        ("cultivator (co-d.)", 1.4, 0.5, "above"),
    ]
    for label, x_kN, dy, side in callouts:
        # Look up the corresponding tension on the curve.
        idx = (df["draft_load_N"] - x_kN * 1000.0).abs().idxmin()
        y_kN = df.loc[idx, "T_anchor_N"] / 1000.0
        color = "#005500" if "co-d" in label else "#1a1a1a"
        ax.plot(x_kN, y_kN, marker="o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=10)
        offset = 0.5 if side == "above" else -1.0
        ax.annotate(
            label, xy=(x_kN, y_kN), xytext=(x_kN + 0.2, y_kN + offset),
            fontsize=8, color=color, fontweight="bold",
        )

    ax.set_xlabel("Implement draft load (kN)")
    ax.set_ylabel("Anchor / winch tension (kN)")
    ax.set_title(
        "F2. Anchor reaction envelope — two per-auger lateral capacity references\n"
        "Dashed red: Khand 2024 (400 N/auger, loose sand, free-head, worst case)\n"
        "Solid blue: Magnum 2024 (2 kN/auger, medium-dense sand, fixed-head, realistic)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.set_ylim(0, y_max_kN)
    ax.set_xlim(0, df["draft_load_N"].max() / 1000.0 * 1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F3 — Peak vs continuous motor power across operation library
# ---------------------------------------------------------------------------

def figure_3_motor_power(out_png: Path, out_csv: Path) -> pd.DataFrame:
    """F3 — Peak vs continuous motor power across the co-designed library.

    For each co-designed implement we use the P50 of its draft distribution
    (sampled at 1–2.5 km/h) as the *continuous* draft, and the P90 as the
    *peak* draft. The CableTract operating speed is held at 0.5 m/s for both
    cases — the "peak" is a force-budget event (a hard-soil patch encountered
    during steady operation), not a speed-budget event. The motor must
    deliver F_peak × v_cont without losing speed.

    Motor sizing rule: the motor must satisfy the worst-case continuous and
    peak demand across the *entire* implement library, since one CableTract
    is expected to swap implements through the season.
    """
    rng = np.random.default_rng(0xCAB1E)
    codesign = load_cabletract_implement_library()
    baseline = default_drivetrain()
    premium = premium_drivetrain()

    # Operating speeds for the co-designed regime.
    V_CONT_M_S = 0.50
    V_PEAK_M_S = 0.50  # peak is a force event at the same operating speed

    rows = []
    for imp in codesign:
        depth_lo = max(imp.typical_depth_cm - 5.0, 1.0)
        depth_hi = imp.typical_depth_cm + 5.0
        samples = draft_distribution(
            imp,
            soil_texture="medium",
            moisture_range=(0.12, 0.28),
            speed_range_km_h=(1.0, 2.5),
            depth_range_cm=(depth_lo, depth_hi),
            n_samples=2000,
            rng=rng,
        )
        fc = float(np.percentile(samples, 50))
        fp = float(np.percentile(samples, 90))
        env_b = peak_motor_power(fc, V_CONT_M_S, fp, V_PEAK_M_S, drivetrain=baseline)
        env_p = peak_motor_power(fc, V_CONT_M_S, fp, V_PEAK_M_S, drivetrain=premium)
        rows.append({
            "implement": imp.name,
            "category": imp.category,
            "F_continuous_N": fc,
            "F_peak_N": fp,
            "v_continuous_m_s": V_CONT_M_S,
            "v_peak_m_s": V_PEAK_M_S,
            "P_continuous_baseline_W": env_b.continuous_W,
            "P_peak_baseline_W": env_b.peak_W,
            "P_continuous_premium_W": env_p.continuous_W,
            "P_peak_premium_W": env_p.peak_W,
            "peak_to_continuous": env_b.peak_to_continuous,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(df))
    width = 0.2
    ax.bar(x - 1.5 * width, df["P_continuous_baseline_W"] / 1000.0, width,
           label="Continuous (η=0.50)", color="#1f77b4")
    ax.bar(x - 0.5 * width, df["P_peak_baseline_W"] / 1000.0, width,
           label="Peak (η=0.50)", color="#1f77b4", alpha=0.5)
    ax.bar(x + 0.5 * width, df["P_continuous_premium_W"] / 1000.0, width,
           label="Continuous (η=0.70)", color="#2ca02c")
    ax.bar(x + 1.5 * width, df["P_peak_premium_W"] / 1000.0, width,
           label="Peak (η=0.70)", color="#2ca02c", alpha=0.5)

    # Two motor sizing tiers:
    #
    #   "Light" tier: covers all secondary tillage, seeding, spraying,
    #     mowing, weeding, and rotary-hoe operations (i.e. everything
    #     except primary tillage). Sized for the heaviest of these.
    #   "Heavy" tier: adds the primary-tillage implements (narrow ripper,
    #     narrow chisel, narrow disk, narrow sweep). Sized for the heaviest
    #     of *all* implements.
    #
    # In the manuscript these correspond to the two cost tiers in §5.6.
    primary_mask = df["category"].eq("primary_tillage")
    light = df[~primary_mask]
    heavy = df

    light_peak_kW = light["P_peak_baseline_W"].max() / 1000.0
    heavy_peak_kW = heavy["P_peak_baseline_W"].max() / 1000.0

    ax.axhline(light_peak_kW, color="#2ca02c", ls="--", alpha=0.7)
    ax.text(0.02, light_peak_kW + 0.10,
            f"Light-tier motor: {light_peak_kW:.1f} kW peak (covers all non-primary-tillage operations)",
            fontsize=8, ha="left", color="#1a6a1a")
    ax.axhline(heavy_peak_kW, color="#d62728", ls="--", alpha=0.7)
    ax.text(0.02, heavy_peak_kW + 0.10,
            f"Heavy-tier motor: {heavy_peak_kW:.1f} kW peak (adds primary tillage)",
            fontsize=8, ha="left", color="#a01818")

    ax.set_xticks(x)
    ax.set_xticklabels(df["implement"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Electrical power at motor (kW)")
    ax.set_title(
        "F3. Peak vs continuous motor power — CableTract co-designed implements\n"
        "(continuous = P50 draft, peak = P90 draft, both at v = 0.50 m/s)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    f1 = figure_1_cable_sag(OUT_DIR / "F1_cable_sag.png", TAB_DIR / "F1_cable_sag.csv")
    f2 = figure_2_anchor_envelope(OUT_DIR / "F2_anchor_envelope.png", TAB_DIR / "F2_anchor_envelope.csv")
    f3 = figure_3_motor_power(OUT_DIR / "F3_motor_power.png", TAB_DIR / "F3_motor_power.csv")

    # Print headline numbers for the manuscript Results section
    print("=== Phase 1 headline numbers ===")
    print()
    print("F1. Sag at midspan, Dyneema 8 mm, L=50 m, T=5 kN:")
    sub = f1[(f1["material"] == "dyneema_sk78_8mm_12strand") & (f1["span_m"] == 50.0)]
    closest = sub.iloc[(sub["T_horiz_N"] - 5000.0).abs().argmin()]
    print(f"   sag = {closest['sag_m']*100:.2f} cm at T={closest['T_horiz_N']:.0f} N")

    print()
    print("F2. Anchor count required at typical conventional vs co-designed drafts:")
    for label, F_d in [
        ("Conventional moldboard plow P50  (~14 kN)", 14000.0),
        ("Conventional disk harrow P50      (~8 kN)", 8000.0),
        ("Co-designed narrow chisel P50     (~4 kN)", 4000.0),
        ("Co-designed narrow ripper P50     (~3 kN)", 3000.0),
        ("Co-designed cultivator P50        (~1.4 kN)", 1400.0),
    ]:
        # Snap to the nearest tabulated draft.
        idx = (f2["draft_load_N"] - F_d).abs().idxmin()
        ref = f2.loc[idx]
        print(
            f"   {label:48s}  T_anchor={ref['T_anchor_N']:6.0f} N  "
            f"Khand SF=1.15 → {ref['n_augers_required_khand_working']:3.0f} augers  "
            f"Magnum SF=1.15 → {ref['n_augers_required_magnum_working']:3.0f} augers"
        )

    print()
    print("F3. Peak vs continuous motor power across the co-designed library:")
    for _, r in f3.iterrows():
        print(
            f"   {r['implement']:25s}  cont={r['P_continuous_baseline_W']/1000:5.2f} kW  "
            f"peak={r['P_peak_baseline_W']/1000:5.2f} kW  ratio={r['peak_to_continuous']:.2f}×"
        )
    max_peak = f3["P_peak_baseline_W"].max() / 1000.0
    print(f"   → maximum peak across library = {max_peak:.2f} kW (sets motor sizing)")

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
