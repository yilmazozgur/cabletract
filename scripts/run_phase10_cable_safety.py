"""Phase 10 driver (S2) -- cable safety & durability.

F25. Three panels:
  (a) snap-back recoil velocity + exclusion radius vs tension (steel/Dyneema/UHMWPE);
  (b) bending-fatigue life vs sheave ratio D/d, with the working-load and UV/abrasion
      bounds -> the new cable-replacement OPEX/LCA line;
  (c) aeroelastic safe-operation band: vortex lock-in winds and the iced-galloping
      onset against the bundled monthly mean winds.
"""

from __future__ import annotations

import csv
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

from cabletract.dynamics import load_cable_props  # noqa: E402
from cabletract.cable_safety import (  # noqa: E402
    aeroelastic_check, bend_cycles_to_failure, fatigue_and_opex,
    fdm_wave_speed, snapback_recoil,
)
from cabletract.energy import load_site_meta  # noqa: E402

OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"

MATERIALS = ["steel_6x19_IWRC_8mm", "dyneema_sk78_8mm_12strand", "uhmwpe_spectra_8mm"]
SHORT = {"steel_6x19_IWRC_8mm": "steel 6x19",
         "dyneema_sk78_8mm_12strand": "Dyneema SK78",
         "uhmwpe_spectra_8mm": "UHMWPE Spectra"}
COLORS = {"steel_6x19_IWRC_8mm": "#555555",
          "dyneema_sk78_8mm_12strand": "#1f77b4",
          "uhmwpe_spectra_8mm": "#2ca02c"}


def load_mbl_wll():
    mbl, wll = {}, {}
    with open(ROOT / "cabletract" / "data" / "cable_props.csv") as fh:
        for r in csv.DictReader(fh):
            mbl[r["material"]] = float(r["MBL_N"])
            wll[r["material"]] = float(r["working_load_N"])
    return mbl, wll


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    mbl, wll = load_mbl_wll()
    props = {m: load_cable_props(m) for m in MATERIALS}

    # FDM validation up front
    sp, c = fdm_wave_speed(T=1800.0, mu=props[MATERIALS[0]].mass_per_m_kg, L=50.0)

    # ---- gather data ----
    recoil_rows, fatigue_rows, aero_rows = [], [], []
    for m in MATERIALS:
        for frac in np.linspace(0.05, 1.0, 20):
            r = snapback_recoil(props[m], frac * mbl[m])
            recoil_rows.append(dict(material=m, frac_MBL=frac, tension_N=frac * mbl[m],
                                    v_m_s=r.recoil_velocity_m_s, KE_J=r.recoil_KE_J,
                                    excl_m=r.exclusion_radius_m, hazardous=r.hazardous))
        f = fatigue_and_opex(props[m], mbl[m], D_over_d=25.0, working_tension_N=1800.0)
        fatigue_rows.append(dict(material=m, **{k: getattr(f, k) for k in
                            ("load_fraction_MBL", "fatigue_life_yr", "practical_life_yr",
                             "opex_eur_per_yr", "co2_kg_per_yr")}))
        a = aeroelastic_check(props[m], site="Konya_TR", n_modes=10)
        aero_rows.append(dict(material=m, f1_hz=a.modal_freqs_hz[0],
                              gallop_iced_m_s=a.galloping_onset_iced_m_s,
                              lockin_mode1_m_s=a.lockin_wind_m_s[0],
                              viv=a.viv_in_wind_band, gallop=a.galloping_in_wind_band))
    pd.DataFrame(recoil_rows).to_csv(TAB_DIR / "F25_recoil.csv", index=False)
    pd.DataFrame(fatigue_rows).to_csv(TAB_DIR / "F25_fatigue_opex.csv", index=False)
    pd.DataFrame(aero_rows).to_csv(TAB_DIR / "F25_aeroelastic.csv", index=False)

    # ---------------------------------------------------------------- figure
    fig, (a, b, c2) = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) recoil velocity (left axis) + exclusion radius (right) vs tension
    df = pd.DataFrame(recoil_rows)
    a2 = a.twinx()
    for m in MATERIALS:
        d = df[df.material == m]
        a.plot(d.tension_N / 1e3, d.v_m_s, color=COLORS[m], lw=2, label=SHORT[m])
        a2.plot(d.tension_N / 1e3, d.excl_m, color=COLORS[m], lw=1.2, ls="--")
        a.plot(wll[m] / 1e3, snapback_recoil(props[m], wll[m]).recoil_velocity_m_s,
               "o", color=COLORS[m], ms=7)
    a.set_xlabel("cable tension (kN)")
    a.set_ylabel("recoil velocity (m/s)  [solid]")
    a2.set_ylabel("exclusion radius (m, capped at span)  [dashed]")
    a.set_title("(a) Snap-back recoil\n(o = working load; synthetic recoils faster)")
    a.legend(fontsize=8, loc="upper left"); a.grid(True, alpha=0.3)

    # (b) bending-fatigue life vs D/d
    dd = np.linspace(10, 60, 60)
    for m in MATERIALS:
        S = 1800.0 / mbl[m]
        life = [bend_cycles_to_failure(props[m], x, S, mbl[m]) / 7000.0 for x in dd]
        b.semilogy(dd, life, color=COLORS[m], lw=2, label=SHORT[m])
    b.axhline(8.0, color="red", ls=":", lw=1.5, label="UV/abrasion limit (~8 yr)")
    b.axhspan(0, 8.0, color="red", alpha=0.06)
    b.axvline(25.0, color="black", ls="--", lw=1, label="design D/d=25")
    b.set_xlabel("sheave ratio D/d"); b.set_ylabel("bending-fatigue life (yr, 7000 bends/yr)")
    b.set_title("(b) Bending fatigue is NOT binding at 5% MBL\n-> UV/abrasion sets ~8 yr life")
    b.legend(fontsize=8, loc="lower right"); b.grid(True, which="both", alpha=0.3)

    # (c) aeroelastic safe-operation band
    meta = load_site_meta("Konya_TR")
    months = np.arange(1, 13)
    c2.bar(months, meta.wind_mean_m_s, color="#cfe8ff", edgecolor="#1f77b4",
           label="monthly mean wind (Konya)")
    for m in MATERIALS:
        a_res = aeroelastic_check(props[m], site="Konya_TR")
        c2.axhline(a_res.galloping_onset_iced_m_s, color=COLORS[m], ls="--", lw=1.6,
                   label=f"{SHORT[m]} iced-galloping onset")
    c2.set_xlabel("month"); c2.set_ylabel("wind speed (m/s)")
    c2.set_title("(c) Aeroelastic band: iced-galloping onset\nbelow site winds -> de-tension in icing")
    c2.set_xticks(months)
    c2.legend(fontsize=7, loc="upper right"); c2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cable safety & durability (S2) -- snap-back, bending fatigue, aeroelastic band "
                 f"[FDM wave speed {sp:.1f} vs sqrt(T/mu) {c:.1f} m/s]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT_DIR / "F25_cable_safety.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------- console
    print("=== S2 cable safety & durability ===")
    print(f"FDM transverse wave speed = {sp:.2f} m/s vs analytic sqrt(T/mu) = {c:.2f} m/s "
          f"({abs(sp-c)/c*100:.2f}%)")
    print("\nSnap-back at working load:")
    for m in MATERIALS:
        r = snapback_recoil(props[m], wll[m])
        print(f"  {SHORT[m]:15s}: v={r.recoil_velocity_m_s:5.1f} m/s  KE={r.recoil_KE_J:5.0f} J  "
              f"excl={r.exclusion_radius_m:4.0f} m  hazardous={r.hazardous}")
    print("\nReplacement interval / new OPEX+LCA line (D/d=25, 1800 N):")
    for row in fatigue_rows:
        print(f"  {SHORT[row['material']]:15s}: fatigue {row['fatigue_life_yr']:.0f} yr -> "
              f"practical {row['practical_life_yr']:.1f} yr; "
              f"OPEX {row['opex_eur_per_yr']:.0f} EUR/yr, CO2 {row['co2_kg_per_yr']:.1f} kg/yr")
    print("\nAeroelastic (Konya winds 3.3-3.9 m/s):")
    for row in aero_rows:
        print(f"  {SHORT[row['material']]:15s}: f1={row['f1_hz']:.2f} Hz  "
              f"iced-galloping onset={row['gallop_iced_m_s']:.2f} m/s  "
              f"VIV={row['viv']}  galloping(iced)={row['gallop']}")
    print(f"\nFigure -> {OUT_DIR/'F25_cable_safety.png'}")
    print(f"Tables -> {TAB_DIR}/F25_recoil.csv, F25_fatigue_opex.csv, F25_aeroelastic.csv")


if __name__ == "__main__":
    main()
