"""Phase 5 driver — generates F13, F14, F15, F16, F17.

F13. Sobol S1 / ST bars for the top-12 parameters across the 4 main
     outputs (decares/day off-grid, payback months, energy/decare,
     surplus power).
F14. Tornado plot of the top-10 swing contributors to NPV vs diesel.
F15. Monte Carlo P10/P50/P90 envelopes for daily throughput vs draft
     load — replaces the deterministic curve from the v1 sweep.
F16. NPV vs farm size (1 / 5 / 25 / 100 ha) at three discount rates
     (5/8/12 %).
F17. Life-cycle CO2 per hectare-year for CableTract / diesel /
     electric tractor as stacked bars (embodied + fuel).
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
    cabletract_capex,
    cabletract_npv_vs_diesel,
    cabletract_payback_vs_diesel,
    lifecycle_co2_per_ha_yr,
)
from cabletract.params import CableTractParams  # noqa: E402
from cabletract.simulate import run_single  # noqa: E402
from cabletract.uncertainty import (  # noqa: E402
    default_problem,
    monte_carlo,
    percentile_envelope,
    sobol_indices,
    tornado_data,
)


OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


# ---------------------------------------------------------------------------
# F13 — Sobol bars
# ---------------------------------------------------------------------------

def figure_13_sobol(out_png: Path, out_csv: Path) -> pd.DataFrame:
    prob = default_problem()
    outputs = [
        "decares_per_day_offgrid",
        "payback_months_vs_fuel",
        "energy_per_decare_Wh",
        "surplus_power_W",
    ]
    print(f"  running Sobol with n_base=256 over {len(prob.ranges)} params, {len(outputs)} outputs...")
    df = sobol_indices(prob, n_base=256, output_names=outputs)
    df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()
    for ax, out in zip(axes, outputs):
        sub = df[df["output"] == out].sort_values("ST", ascending=False).head(12)
        y = np.arange(len(sub))
        ax.barh(y - 0.18, sub["S1"].values, 0.36, label="S1 (first-order)", color="#1f77b4")
        ax.barh(y + 0.18, sub["ST"].values, 0.36, label="ST (total-order)", color="#d62728")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["parameter"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Sobol index")
        ax.set_title(out, fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(
        "F13. Sobol global sensitivity (S1 vs ST) — codesigned reference, 20 parameters",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F14 — Tornado plot for NPV vs diesel
# ---------------------------------------------------------------------------

def _econ_npv_from_params(sim: CableTractParams) -> float:
    """Run the simulator on a perturbed CableTractParams, then map the
    resulting fuel/electricity/payback into the EconParams NPV pipeline.

    The economic baseline is :meth:`EconParams.codesigned`; the function
    only overrides the fields that the simulator parameter set actually
    drives (annual hectares, diesel intensity / price, capex split,
    battery and PV sizes). Everything else — discount rate, horizon,
    grid price, electric-tractor reference — comes from the codesigned
    economics record so the tornado plot perturbs *one* parameter at a
    time around a single coherent reference.
    """
    r = run_single(sim)
    base = EconParams.codesigned()
    # Capex split mirrors EconParams.codesigned() exactly so the override
    # sums back to the §5.8 BOM (€35,570 = 17,500 main + 7,500 anchor +
    # 4,000 install + 3,420 battery + 1,650 PV + 1,500 wind):
    #   main_unit / cost_cabletract_usd = 17500/35570 = 0.4920
    #   anchor    / cost_cabletract_usd =  7500/35570 = 0.2109
    #   install   / cost_cabletract_usd =  4000/35570 = 0.1124
    # The remaining 18.5 % (battery + PV + wind) is taken from the base
    # EconParams record below.
    overrides = {
        "annual_hectares": r.decares_per_year / 10.0,
        "diesel_litres_per_ha": sim.fuel_l_per_decare * 10.0,
        "diesel_price_eur_per_litre": sim.fuel_price_usd_per_l,
        "capex_main_unit_eur": sim.cost_cabletract_usd * 0.4920,
        "capex_anchor_eur": sim.cost_cabletract_usd * 0.2109,
        "capex_install_overhead_eur": sim.cost_cabletract_usd * 0.1124,
        "battery_capacity_kWh": sim.battery_Wh / 1000.0,
        "pv_area_m2": sim.solar_area_m2,
    }
    kwargs = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
    kwargs.update(overrides)
    return cabletract_npv_vs_diesel(EconParams(**kwargs))


def figure_14_tornado(out_png: Path, out_csv: Path) -> pd.DataFrame:
    prob = default_problem()

    # Tornado around the codesigned simulator baseline. We perturb one
    # CableTractParams field at a time and re-evaluate NPV.
    base_params = CableTractParams.codesigned()
    base_npv = _econ_npv_from_params(base_params)

    rows = []
    for pr in prob.ranges:
        rows_one = {}
        for tag, val in [("lo", pr.lo), ("hi", pr.hi)]:
            params_kwargs = {f.name: getattr(base_params, f.name) for f in base_params.__dataclass_fields__.values()}
            params_kwargs[pr.name] = val
            try:
                npv_val = _econ_npv_from_params(CableTractParams(**params_kwargs))
            except Exception:  # noqa: BLE001
                npv_val = float("nan")
            rows_one[tag] = npv_val
        rows.append({
            "parameter": pr.name,
            "lo_bound": pr.lo,
            "hi_bound": pr.hi,
            "lo_npv": rows_one["lo"],
            "hi_npv": rows_one["hi"],
            "baseline_npv": base_npv,
            "swing": abs(rows_one["hi"] - rows_one["lo"]),
        })
    df = pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)
    top = df.head(10)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(top))
    centers = top["baseline_npv"].values
    los = np.minimum(top["lo_npv"].values, top["hi_npv"].values)
    his = np.maximum(top["lo_npv"].values, top["hi_npv"].values)
    ax.barh(y, his - centers, left=centers, color="#2a7a2a", label="parameter at hi bound")
    ax.barh(y, los - centers, left=centers, color="#cd5c5c", label="parameter at lo bound")
    ax.axvline(base_npv, color="black", linestyle="--", lw=1.0, label=f"baseline NPV ({base_npv:,.0f} €)")
    ax.set_yticks(y)
    ax.set_yticklabels(top["parameter"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("NPV vs diesel reference (€)")
    ax.set_title("F14. Tornado plot — top-10 NPV swing contributors (codesigned baseline)", fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F15 — Monte Carlo throughput envelopes vs draft load
# ---------------------------------------------------------------------------

def figure_15_mc_envelopes(out_png: Path, out_csv: Path) -> pd.DataFrame:
    """For each draft load bin, draw 500 MC samples (other params perturbed)
    and report P10/P50/P90 of decares-per-day off-grid."""
    prob = default_problem()
    base_n = 1000
    df_full = monte_carlo(prob, n=base_n, seed=20260407)

    # Bin the MC sample by draft_load_N and report percentiles per bin
    bins = np.linspace(1500.0, 4500.0, 11)  # 10 bins of 300 N each
    centers = 0.5 * (bins[:-1] + bins[1:])
    df_full["bin"] = pd.cut(df_full["draft_load_N"], bins=bins, include_lowest=True)
    rows = []
    for c, (_, sub) in zip(centers, df_full.groupby("bin", observed=True)):
        if len(sub) < 5:
            continue
        env = percentile_envelope(sub, "decares_per_day_offgrid")
        rows.append({
            "draft_N_center": c,
            "n": int(len(sub)),
            **env,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(df["draft_N_center"], df["p10"], df["p90"], alpha=0.30, color="#2a7a2a", label="P10–P90 envelope")
    ax.plot(df["draft_N_center"], df["p50"], color="#1b5e1b", lw=2.0, label="median (P50)")

    # Codesigned deterministic reference point (1800 N P50 of the
    # CableTract-native implement library at 1.5 m strip width).
    cd_params = CableTractParams.codesigned()
    cd_ref = run_single(cd_params)
    ax.scatter([cd_params.draft_load_N], [cd_ref.decares_per_day_offgrid],
               color="#1b5e1b", s=110, marker="*", zorder=5,
               label=f"codesigned reference ({cd_ref.decares_per_day_offgrid:.1f} decares/day)")

    ax.set_xlabel("Draft load (N)")
    ax.set_ylabel("Daily off-grid throughput (decares/day)")
    ax.set_title(f"F15. Monte Carlo throughput envelope (n={base_n} draws over 20 codesigned parameters)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# F16 — NPV vs farm size at three discount rates
# ---------------------------------------------------------------------------

def figure_16_npv_vs_farm_size(out_png: Path, out_csv: Path) -> pd.DataFrame:
    """F16 — NPV / payback / LCOE vs farm size for the co-designed reference.

    Three panels:
        (a) NPV vs annual area at 5/8/12 % discount rates.
        (b) Discounted payback vs annual area at the same three rates.
        (c) LCOE per ha-year for CableTract vs diesel at 8 % discount,
            showing the absolute per-hectare cost picture.
    """
    farm_sizes_ha = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    discount_rates = [0.05, 0.08, 0.12]

    def _params_for(ha: float, r: float) -> EconParams:
        # Build a co-designed EconParams with the per-call overrides applied.
        base = EconParams.codesigned()
        kwargs = {f.name: getattr(base, f.name) for f in base.__dataclass_fields__.values()}
        kwargs["annual_hectares"] = ha
        kwargs["discount_rate"] = r
        return EconParams(**kwargs)

    rows = []
    for ha in farm_sizes_ha:
        for r in discount_rates:
            p = _params_for(ha, r)
            rows.append({
                "annual_hectares": ha,
                "discount_rate": r,
                "capex_eur": cabletract_capex(p),
                "npv_eur": cabletract_npv_vs_diesel(p),
                "payback_yr": cabletract_payback_vs_diesel(p),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Panel (c): LCOE per ha-yr at 8% discount, CableTract vs diesel.
    from cabletract.economics import (
        cabletract_annual_opex,
        diesel_annual_opex,
        lcoe,
    )
    lcoe_rows = []
    for ha in farm_sizes_ha:
        p = _params_for(ha, 0.08)
        ct_lcoe = lcoe(
            cabletract_capex(p), cabletract_annual_opex(p),
            ha, 0.08, p.horizon_years,
        )
        d_lcoe = lcoe(
            p.diesel_capex_eur, diesel_annual_opex(p),
            ha, 0.08, p.horizon_years,
        )
        lcoe_rows.append({
            "annual_hectares": ha,
            "ct_lcoe_eur_per_ha": ct_lcoe,
            "diesel_lcoe_eur_per_ha": d_lcoe,
        })
    lcoe_df = pd.DataFrame(lcoe_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    colors = {0.05: "#2a7a2a", 0.08: "#1f77b4", 0.12: "#d62728"}
    for r in discount_rates:
        sub = df[df["discount_rate"] == r]
        axes[0].plot(sub["annual_hectares"], sub["npv_eur"] / 1000.0, "-o",
                     color=colors[r], lw=2, label=f"r={int(r*100)}%")
        axes[1].plot(sub["annual_hectares"], sub["payback_yr"], "-o",
                     color=colors[r], lw=2, label=f"r={int(r*100)}%")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xlabel("Annual area worked (ha)")
    axes[0].set_ylabel("Incremental NPV vs diesel (k€)")
    axes[0].set_title("(a) NPV at three discount rates", fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_xscale("log")

    axes[1].set_xlabel("Annual area worked (ha)")
    axes[1].set_ylabel("Discounted incremental payback (years)")
    axes[1].set_title("(b) Incremental payback vs diesel", fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_xscale("log")
    axes[1].set_ylim(0, 16)

    axes[2].plot(lcoe_df["annual_hectares"], lcoe_df["ct_lcoe_eur_per_ha"],
                 "-o", color="#1f77b4", lw=2, label="CableTract (codesigned)")
    axes[2].plot(lcoe_df["annual_hectares"], lcoe_df["diesel_lcoe_eur_per_ha"],
                 "-s", color="#d62728", lw=2, label="Diesel tractor")
    axes[2].set_xlabel("Annual area worked (ha)")
    axes[2].set_ylabel("LCOE per ha-year (€)")
    axes[2].set_title("(c) Levelised cost of operation @ 8 %", fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    axes[2].set_xscale("log")

    fig.suptitle("F16. NPV, payback, and LCOE vs farm size — codesigned CableTract reference",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Persist LCOE alongside the NPV/payback CSV.
    lcoe_df.to_csv(out_csv.parent / "F16_lcoe_vs_farm_size.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# F17 — Life-cycle CO2 stacked bars
# ---------------------------------------------------------------------------

def figure_17_lca(out_png: Path, out_csv: Path) -> pd.DataFrame:
    p = EconParams.codesigned()
    out = lifecycle_co2_per_ha_yr(p)
    rows = [
        {
            "vehicle": "CableTract",
            "embodied_kg_per_ha_yr": out["cabletract_embodied_kg_per_ha_yr"],
            "fuel_kg_per_ha_yr": out["cabletract_fuel_kg_per_ha_yr"],
            "total_kg_per_ha_yr": out["cabletract_total_kg_per_ha_yr"],
        },
        {
            "vehicle": "Diesel tractor",
            "embodied_kg_per_ha_yr": out["diesel_embodied_kg_per_ha_yr"],
            "fuel_kg_per_ha_yr": out["diesel_fuel_kg_per_ha_yr"],
            "total_kg_per_ha_yr": out["diesel_total_kg_per_ha_yr"],
        },
        {
            "vehicle": "Electric tractor",
            "embodied_kg_per_ha_yr": out["electric_embodied_kg_per_ha_yr"],
            "fuel_kg_per_ha_yr": out["electric_fuel_kg_per_ha_yr"],
            "total_kg_per_ha_yr": out["electric_total_kg_per_ha_yr"],
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(df))
    embodied = df["embodied_kg_per_ha_yr"].values
    fuel = df["fuel_kg_per_ha_yr"].values
    ax.bar(x, embodied, color="#5a8fc8", label="embodied (manufacture, amortised over horizon)")
    ax.bar(x, fuel, bottom=embodied, color="#d4a017", label="operating energy (fuel / grid)")
    for i, (e, f, t) in enumerate(zip(embodied, fuel, df["total_kg_per_ha_yr"].values)):
        ax.text(i, t + 0.5, f"{t:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["vehicle"].values)
    ax.set_ylabel("Life-cycle CO₂ (kg CO₂eq / ha-year)")
    ax.set_title(f"F17. Life-cycle CO₂ per hectare-year — codesigned reference "
                 f"({p.annual_hectares:.0f} ha/yr, {p.horizon_years} yr horizon)",
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating F13 — Sobol global sensitivity bars...")
    f13 = figure_13_sobol(OUT_DIR / "F13_sobol_bars.png", TAB_DIR / "F13_sobol_indices.csv")
    print("Generating F14 — Tornado plot of NPV swing contributors...")
    f14 = figure_14_tornado(OUT_DIR / "F14_tornado_npv.png", TAB_DIR / "F14_tornado_npv.csv")
    print("Generating F15 — Monte Carlo throughput envelopes...")
    f15 = figure_15_mc_envelopes(OUT_DIR / "F15_mc_throughput_envelope.png",
                                 TAB_DIR / "F15_mc_throughput_envelope.csv")
    print("Generating F16 — NPV vs farm size...")
    f16 = figure_16_npv_vs_farm_size(OUT_DIR / "F16_npv_vs_farm_size.png",
                                     TAB_DIR / "F16_npv_vs_farm_size.csv")
    print("Generating F17 — Life-cycle CO2 stacked bars...")
    f17 = figure_17_lca(OUT_DIR / "F17_lca_co2.png", TAB_DIR / "F17_lca_co2.csv")

    print()
    print("=== Phase 5 headline numbers ===")
    print()
    print("F13. Top 5 ST per output (Sobol n_base=256):")
    for out in sorted(f13["output"].unique()):
        sub = f13[f13["output"] == out].sort_values("ST", ascending=False).head(5)
        print(f"  {out}:")
        for _, row in sub.iterrows():
            print(f"    {row['parameter']:25s}  S1={row['S1']:.3f}  ST={row['ST']:.3f}")

    print()
    print("F14. Top 10 NPV swing contributors:")
    print(f14.head(10).to_string(index=False, float_format="%.0f"))

    print()
    print("F15. P10/P50/P90 throughput by draft bin:")
    print(f15.to_string(index=False, float_format="%.2f"))

    print()
    print("F16. NPV / payback vs farm size:")
    print(f16.to_string(index=False, float_format="%.2f"))

    print()
    print("F17. Life-cycle CO2 per ha-yr:")
    print(f17.to_string(index=False, float_format="%.2f"))

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
