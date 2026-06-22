"""Phase 8 driver (S4) -- co-design lever sensitivity.

F22. Four-panel robustness picture: as the achieved reference draft rises from
     the codesigned point (1.8 kN, the claimed ~0.37x reduction) toward the
     conventional-library median (co-design fails), how do energy/decare,
     off-grid throughput, required augers, and economics respond?
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

from cabletract.codesign_sensitivity import sweep_codesign  # noqa: E402
from cabletract.params import CableTractParams  # noqa: E402

OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    df = sweep_codesign()
    df.to_csv(TAB_DIR / "F22_codesign_sensitivity.csv", index=False)
    ref = df.attrs["reference_draft_N"]
    conv = df.attrs["conventional_median_draft_N"]
    x = df["draft_P50_N"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    a, b, c, d = axes.flatten()

    def mark_ref(ax):
        ax.axvline(ref, color="#1b5e1b", ls="--", lw=1.4,
                   label=f"codesigned ref ({ref:.0f} N, {ref/conv:.2f}$\\times$)")
        ax.set_xlabel("Reference P50 draft (N)  --  weaker co-design $\\rightarrow$")
        ax.grid(True, alpha=0.3)

    a.plot(x, df["energy_per_decare_Wh"], color="#1f77b4", lw=2)
    a.set_ylabel("Energy per decare (Wh)"); a.set_title("(a) Energy intensity"); mark_ref(a); a.legend(fontsize=8)

    b.plot(x, df["decares_per_day_offgrid"], color="#2a7a2a", lw=2)
    b.set_ylabel("Off-grid throughput (decares/day)"); b.set_title("(b) Daily throughput"); mark_ref(b)

    c.plot(x, df["augers_req_P90_cap2000N"], color="#1f77b4", lw=2, label="medium-dense nominal (2 kN/auger)")
    c.plot(x, df["augers_req_P90_cap400N"], color="#cd5c5c", lw=2, label="loose-sand worst case (400 N/auger)")
    c.axhline(9, color="black", ls=":", lw=1.2, label="installed 9-auger Anchor")
    c.set_ylabel("Augers required at P90 (SF 1.15)"); c.set_title("(c) Anchor demand"); mark_ref(c); c.legend(fontsize=8)

    d.plot(x, df["simple_payback_months"], color="#d62728", lw=2, label="simple payback (additive frame)")
    npv0 = df["npv_replacement_eur"].iloc[0]
    d.set_ylabel("Simple payback (months)"); d.set_title("(d) Economics"); mark_ref(d)
    d.text(0.04, 0.06,
           f"Replacement-frame NPV is essentially draft-invariant\n"
           f"off-grid: {df['npv_replacement_eur'].min():,.0f} to "
           f"{df['npv_replacement_eur'].max():,.0f} EUR (no fuel cost to save).",
           transform=d.transAxes, fontsize=8, va="bottom",
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.6, alpha=0.9))
    d.legend(fontsize=8, loc="upper left")

    fig.suptitle("Co-design lever sensitivity: headline outputs vs achieved draft reduction (S4)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "F22_codesign_sensitivity.png", dpi=200)
    plt.close(fig)

    # Console summary at the reference and at a degraded (0.6x) point.
    print("=== S4 co-design sensitivity ===")
    print(f"conventional median draft = {conv:.0f} N; codesigned reference = {ref:.0f} N "
          f"(ratio {ref/conv:.2f})")
    refrow = df.iloc[0]
    print(f"  at reference: {refrow['energy_per_decare_Wh']:.0f} Wh/decare, "
          f"{refrow['decares_per_day_offgrid']:.1f} dec/day, "
          f"augers P90 = {int(refrow['augers_req_P90_cap400N'])} (400N) / "
          f"{int(refrow['augers_req_P90_cap2000N'])} (2kN), "
          f"NPV {refrow['npv_replacement_eur']:,.0f} EUR")
    # nearest row to ratio 0.6
    import numpy as np
    i06 = int(np.argmin(np.abs(df["reduction_ratio_vs_conventional"].values - 0.6)))
    r06 = df.iloc[i06]
    print(f"  if co-design only reaches 0.60x: {r06['energy_per_decare_Wh']:.0f} Wh/decare, "
          f"{r06['decares_per_day_offgrid']:.1f} dec/day, "
          f"augers P90 = {int(r06['augers_req_P90_cap400N'])} (400N) / "
          f"{int(r06['augers_req_P90_cap2000N'])} (2kN), "
          f"NPV {r06['npv_replacement_eur']:,.0f} EUR")
    print(f"Figure -> {OUT_DIR/'F22_codesign_sensitivity.png'}")
    print(f"Table  -> {TAB_DIR/'F22_codesign_sensitivity.csv'}")


if __name__ == "__main__":
    main()
