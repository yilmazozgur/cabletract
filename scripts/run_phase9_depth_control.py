"""Phase 9 driver (S1) -- closed-loop tilling-depth control co-simulation.

F23. Depth-holding under three disturbances (buried stone, hardpan step,
     moisture ramp) for the light cable carriage (PID and offset-free MPC)
     vs a heavy tractor benchmark, plus the gauge-wheel contact force showing
     transient lift-off.
F24. Stability/feasibility envelope over the steady down-pressure reserve and
     the actuator bandwidth: where does the carriage hold depth within the
     +/-2 cm agronomic tolerance with no gauge-wheel lift-off?
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

from cabletract.dynamics import (  # noqa: E402
    DepthPlant, LumpedCable, MPC, hardpan_step, load_cable_props,
    moisture_ramp, simulate_depth, stability_envelope, stone_impulse,
    tuned_pid,
)

OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

SCEN = {"stone": stone_impulse, "hardpan": hardpan_step, "moisture": moisture_ramp}
TITLE = {"stone": "buried stone (1.5 kN, 0.15 s)",
         "hardpan": "hardpan step (0.8 kN)",
         "moisture": "moisture ramp (0.4 kN over 3 s)"}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    carriage = DepthPlant()
    tractor = DepthPlant.tractor()

    # cable vertical stiffness (consistency with the high-fidelity cable)
    props = load_cable_props()
    cab = LumpedCable(span=50.0, props=props, T_target=1800.0, n_seg=40)
    cab.static_equilibrium()
    k_cable = cab.vertical_point_stiffness()

    # ---- run all scenarios x controllers, collect time series + metrics ----
    runs = {}
    rows = []
    for name, dist in SCEN.items():
        runs[name] = {
            "carriage PID": simulate_depth(carriage, tuned_pid(carriage), dist, t_end=4.0),
            "carriage MPC": simulate_depth(carriage, MPC(carriage, dt=0.01), dist, t_end=4.0),
            "tractor": simulate_depth(tractor, tuned_pid(tractor), dist, t_end=4.0),
        }
        for label, r in runs[name].items():
            rows.append({"scenario": name, "controller": label,
                         "peak_mm": round(r.peak_e_mm, 2), "rms_mm": round(r.rms_e_mm, 2),
                         "settle_s": round(r.settle_s, 2), "min_N": round(r.min_N, 1),
                         "liftoff_ms": round(r.liftoff_s * 1e3, 1), "in_2cm": r.in_tol})
    pd.DataFrame(rows).to_csv(TAB_DIR / "F23_depth_control.csv", index=False)

    colors = {"carriage PID": "#1f77b4", "carriage MPC": "#2ca02c", "tractor": "#d62728"}

    # ---------------------------------------------------------------- F23
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    a, b, c, d = axes.flatten()

    # (a) depth error -- stone (headline)
    for label, r in runs["stone"].items():
        a.plot(r.t, r.e * 1e3, color=colors[label], lw=1.8, label=label)
    a.axhspan(-20, 20, color="grey", alpha=0.08)
    a.axhline(20, color="grey", ls=":", lw=1); a.axhline(-20, color="grey", ls=":", lw=1)
    a.text(3.0, 20, "+/-2 cm agronomic band", fontsize=7, va="bottom", color="grey")
    a.set_xlim(0.4, 2.0); a.set_xlabel("time (s)"); a.set_ylabel("depth error (mm)")
    a.set_title("(a) Depth error -- buried-stone strike"); a.legend(fontsize=8); a.grid(True, alpha=0.3)

    # (b) gauge-wheel contact force -- stone (lift-off)
    for label, r in runs["stone"].items():
        b.plot(r.t, r.N / 1e3, color=colors[label], lw=1.8, label=label)
    b.axhline(0, color="black", lw=1.0)
    b.text(1.5, 0.05, "N=0 -> gauge wheel lifts off", fontsize=7, color="black")
    b.set_xlim(0.4, 2.0); b.set_xlabel("time (s)"); b.set_ylabel("gauge-wheel force N (kN)")
    b.set_title("(b) Wheel contact -- carriage briefly lifts, tractor never does")
    b.legend(fontsize=8); b.grid(True, alpha=0.3)

    # (c) depth error -- hardpan (offset-free MPC vs PID)
    for label, r in runs["hardpan"].items():
        c.plot(r.t, r.e * 1e3, color=colors[label], lw=1.8, label=label)
    c.set_xlim(0.4, 4.0); c.set_xlabel("time (s)"); c.set_ylabel("depth error (mm)")
    c.set_title("(c) Depth error -- sustained hardpan step"); c.legend(fontsize=8); c.grid(True, alpha=0.3)

    # (d) summary: peak depth error by scenario x controller
    labels = list(colors.keys())
    x = np.arange(len(SCEN)); width = 0.26
    for k, label in enumerate(labels):
        vals = [runs[s][label].peak_e_mm for s in SCEN]
        d.bar(x + (k - 1) * width, vals, width, color=colors[label], label=label)
    d.axhline(20, color="grey", ls=":", lw=1, label="2 cm tolerance")
    d.set_xticks(x); d.set_xticklabels(list(SCEN.keys()))
    d.set_ylabel("peak depth error (mm)"); d.set_title("(d) Peak depth error summary")
    d.legend(fontsize=8); d.grid(True, axis="y", alpha=0.3)

    fig.suptitle(f"Closed-loop tilling-depth control (S1)  --  cable vertical stiffness "
                 f"{k_cable:.0f} N/m sets nothing; the gauge wheel does", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "F23_depth_control.png", dpi=200)
    plt.close(fig)

    # ---------------------------------------------------------------- F24
    N0_grid = np.linspace(500, 4000, 15)
    bw_grid = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0])
    env = stability_envelope(stone_impulse, N0_grid, bw_grid, t_end=2.5, dt=1.0e-3)
    pd.DataFrame(env["liftoff_s"] * 1e3, index=[f"{b:.0f}Hz" for b in bw_grid],
                 columns=[f"{n:.0f}N" for n in N0_grid]).to_csv(
        TAB_DIR / "F24_liftoff_ms.csv")
    pd.DataFrame(env["peak_e_mm"], index=[f"{b:.0f}Hz" for b in bw_grid],
                 columns=[f"{n:.0f}N" for n in N0_grid]).to_csv(
        TAB_DIR / "F24_peak_mm.csv")

    fig2, (e, f) = plt.subplots(1, 2, figsize=(13, 5))
    ext = [N0_grid[0] / 1e3, N0_grid[-1] / 1e3, bw_grid[0], bw_grid[-1]]

    im1 = e.imshow(env["liftoff_s"] * 1e3, origin="lower", aspect="auto",
                   extent=ext, cmap="inferno")
    cs = e.contour(N0_grid / 1e3, bw_grid, env["liftoff_s"] * 1e3, levels=[0.1],
                   colors="cyan", linewidths=2)
    e.clabel(cs, fmt="no lift-off", fontsize=8)
    e.plot(DepthPlant().N0 / 1e3, 5.0, "w*", ms=14, label="nominal design")
    e.set_xlabel("steady down-pressure reserve N0 (kN)")
    e.set_ylabel("actuator bandwidth (Hz)")
    e.set_title("(a) Gauge-wheel lift-off duration (ms)")
    e.legend(fontsize=8, loc="upper right")
    fig2.colorbar(im1, ax=e, label="lift-off (ms)")

    im2 = f.imshow(env["peak_e_mm"], origin="lower", aspect="auto", extent=ext,
                   cmap="viridis")
    cs2 = f.contour(N0_grid / 1e3, bw_grid, env["peak_e_mm"], levels=[5, 10, 15],
                    colors="white", linewidths=1)
    f.clabel(cs2, fmt="%.0f mm", fontsize=8)
    f.plot(DepthPlant().N0 / 1e3, 5.0, "r*", ms=14, label="nominal design")
    f.set_xlabel("steady down-pressure reserve N0 (kN)")
    f.set_ylabel("actuator bandwidth (Hz)")
    f.set_title("(b) Peak depth error (mm) -- all < 2 cm tolerance")
    f.legend(fontsize=8, loc="upper right")
    fig2.colorbar(im2, ax=f, label="peak error (mm)")

    fig2.suptitle("Depth-control feasibility envelope: reserve x bandwidth to behave "
                  "like a tractor through a stone strike (S1)", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.95))
    fig2.savefig(OUT_DIR / "F24_stability_envelope.png", dpi=200)
    plt.close(fig2)

    # ------------------------------------------------------------- console
    # required N0 for zero lift-off at 5 Hz
    i5 = int(np.argmin(np.abs(bw_grid - 5.0)))
    ok = N0_grid[env["liftoff_s"][i5] <= 1e-9]
    req = ok.min() if len(ok) else float("nan")
    print("=== S1 depth-control co-simulation ===")
    print(f"cable vertical point stiffness = {k_cable:.0f} N/m "
          f"(gauge-wheel k_w = {DepthPlant().k_w:.0e} N/m -> cable sets no depth)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nDesign requirement (5 Hz actuator): down-pressure reserve >= {req:.0f} N "
          f"to eliminate gauge-wheel lift-off in a 1.5 kN stone strike.")
    print(f"Figures -> {OUT_DIR/'F23_depth_control.png'}, {OUT_DIR/'F24_stability_envelope.png'}")
    print(f"Tables  -> {TAB_DIR/'F23_depth_control.csv'} (+ F24_*.csv)")


if __name__ == "__main__":
    main()
