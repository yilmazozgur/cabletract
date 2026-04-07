"""Phase 3 driver — generates the three energy figures (F6, F7, F8).

F6. Calendar heatmap (12 months x 31 days) of daily decares-covered under
    harvested-only operation, for each of the 6 bundled sites. Uses the
    codesigned CableTract energy-per-decare (~0.92 kWh/decare).

F7. Battery SoC time-series for one representative summer week per site
    plus a stacked-area chart of solar / wind / battery / grid contributions
    around the codesigned PV (15 m²) and 9 kWh battery.

F8. Off-grid feasibility map: for the best, median, and worst site, a 2D
    heatmap of (solar area, battery capacity) -> annual grid hours needed
    under the codesigned duty load while the CableTract operates. Replaces
    the single-irradiance feasibility map of the original scaffold.
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

from cabletract.energy import (  # noqa: E402
    BatterySpec,
    PanelSpec,
    WindSpec,
    battery_soc_simulation,
    daily_harvested_energy,
    load_site_meta,
    panel_dc_power,
    synthesize_tmy_year,
    wind_turbine_power,
)


OUT_DIR = ROOT / "figures"
TAB_DIR = ROOT / "tables"


# CableTract codesigned-reference duty cycle. Numbers are computed from
# run_single(CableTractParams.codesigned()) so this driver and Phase 5 share
# one canonical reference: 921 Wh/decare, ~2 kW average operating draw,
# 15 m² PV, 9 kWh battery, 6 h daytime working window.
ENERGY_PER_DECARE_WH = 921.5     # run_single(codesigned()).energy_per_decare_Wh
OPERATING_POWER_W = 2000.0       # average winch input power during operation
IDLE_POWER_W = 50.0              # housekeeping (controllers, comms) when parked
OPERATING_HOURS = (9, 15)        # 09:00-15:00 = 6 h operating window per day

PV_AREA_M2_REF = 15.0            # codesigned PV area
BATTERY_WH_REF = 9000.0          # codesigned battery
PANEL = PanelSpec(area_m2=PV_AREA_M2_REF)
WIND = WindSpec(swept_area_m2=2.0)     # small helix turbine, 2 m² swept


def operating_load_profile(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Return a load profile (W) that matches the codesigned reference duty
    cycle: OPERATING_POWER_W during the OPERATING_HOURS window each day,
    IDLE_POWER_W otherwise. Yields ~12 kWh/day average draw — the
    codesigned CableTract working ~13 decares per day at 0.92 kWh/decare."""
    hour = timestamps.hour
    operating = (hour >= OPERATING_HOURS[0]) & (hour < OPERATING_HOURS[1])
    return np.where(operating, OPERATING_POWER_W, IDLE_POWER_W).astype(float)


# ---------------------------------------------------------------------------
# F6 — Calendar heatmap of daily decares-covered per site
# ---------------------------------------------------------------------------

def figure_6_calendar_heatmaps(out_png: Path, out_csv: Path) -> pd.DataFrame:
    sites = load_site_meta()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    axes = axes.flatten()

    summary_rows = []
    for ax, (name, meta) in zip(axes, sites.items()):
        tmy = synthesize_tmy_year(name, year=2024, seed=0)
        daily = daily_harvested_energy(tmy, PANEL, WIND)
        decares = daily["total_Wh"] / ENERGY_PER_DECARE_WH

        # Reshape into 12 months x 31 days
        grid = np.full((12, 31), np.nan)
        for ts, val in zip(daily.index, decares.values):
            grid[ts.month - 1, ts.day - 1] = val

        im = ax.imshow(grid, aspect="auto", cmap="YlOrRd", vmin=0, vmax=25)
        ax.set_title(name, fontsize=10)
        ax.set_yticks(range(12))
        ax.set_yticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"], fontsize=8)
        ax.set_xticks([0, 9, 19, 29])
        ax.set_xticklabels(["1", "10", "20", "30"], fontsize=8)
        ax.set_xlabel("Day of month", fontsize=8)

        summary_rows.append(
            {
                "site": name,
                "annual_solar_kWh": float(daily["solar_Wh"].sum() / 1000.0),
                "annual_wind_kWh": float(daily["wind_Wh"].sum() / 1000.0),
                "annual_total_kWh": float(daily["total_Wh"].sum() / 1000.0),
                "annual_decares": float(decares.sum()),
                "best_month": int(daily.resample("1ME")["total_Wh"].sum().idxmax().month),
                "worst_month": int(daily.resample("1ME")["total_Wh"].sum().idxmin().month),
                "p10_decares_per_day": float(np.percentile(decares, 10)),
                "p50_decares_per_day": float(np.percentile(decares, 50)),
                "p90_decares_per_day": float(np.percentile(decares, 90)),
            }
        )

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="decares/day under harvested energy")
    fig.suptitle(
        f"F6. Daily decares-covered from harvested solar+wind — codesigned reference\n"
        f"({PV_AREA_M2_REF:.0f} m² PV, 2 m² turbine, 0.92 kWh/decare)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    df = pd.DataFrame(summary_rows)
    df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# F7 — Battery SoC time-series for a representative summer week
# ---------------------------------------------------------------------------

def figure_7_battery_week(out_png: Path, out_csv: Path) -> pd.DataFrame:
    sites = load_site_meta()
    battery = BatterySpec(capacity_Wh=BATTERY_WH_REF)  # 9 kWh codesigned pack

    # For each site we pick that site's *own* brightest 7-day window — the
    # week with the maximum daily-mean GHI sum over a sliding 7-day window.
    # This is hemisphere-agnostic: the northern sites land in late June, the
    # southern São Paulo lands in its January summer, and every panel shows
    # a like-for-like "best representative summer week" comparison.
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes_iter = iter(axes.flatten())

    def _pick_brightest_week(tmy: pd.DataFrame) -> int:
        """Return start hour index of the 7-day window with maximum total GHI."""
        ghi_hourly = tmy["ghi_W_m2"].to_numpy()
        # Aggregate to daily totals (365 days × 24 h = 8760 hours).
        n_days = len(ghi_hourly) // 24
        ghi_daily = ghi_hourly[: n_days * 24].reshape(n_days, 24).sum(axis=1)
        # 7-day rolling sum (length n_days - 6).
        window = 7
        kernel = np.ones(window)
        rolling = np.convolve(ghi_daily, kernel, mode="valid")
        start_day = int(np.argmax(rolling))
        return start_day  # day-of-year index (0-based)

    _MONTHS = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    def _doy_to_label(doy_zero_based: int) -> str:
        ts = pd.Timestamp("2024-01-01") + pd.Timedelta(days=doy_zero_based)
        return f"{_MONTHS[ts.month - 1]} {ts.day}"

    rows = []
    for name in sites:
        tmy = synthesize_tmy_year(name, year=2024, seed=0)

        start_day = _pick_brightest_week(tmy)
        start_hour = start_day * 24
        wk = tmy.iloc[start_hour : start_hour + 7 * 24]

        p_solar = np.array([panel_dc_power(g, t, PANEL) for g, t in zip(wk["ghi_W_m2"], wk["temp_C"])])
        p_wind = np.array([wind_turbine_power(w, WIND) for w in wk["wind_m_s"]])
        p_in = p_solar + p_wind
        p_out = operating_load_profile(wk.index)

        soc_df = battery_soc_simulation(p_in, p_out, battery)

        ax = next(axes_iter)
        hours = np.arange(len(wk))
        ax.fill_between(hours, 0, p_solar, color="#FFD700", alpha=0.6, label="Solar")
        ax.fill_between(hours, p_solar, p_solar + p_wind, color="#87CEEB", alpha=0.6, label="Wind")
        ax.fill_between(hours, 0, -soc_df["p_discharge_W"], color="#90EE90", alpha=0.6, label="Battery out")
        ax.fill_between(hours, -soc_df["p_discharge_W"], -(soc_df["p_discharge_W"] + soc_df["p_grid_W"]), color="#CD5C5C", alpha=0.6, label="Grid")
        ax2 = ax.twinx()
        ax2.plot(hours, soc_df["soc_frac"] * 100, color="black", lw=1.5, label="SoC (%)")
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("Battery SoC (%)", fontsize=8)
        week_label = _doy_to_label(start_day)
        ax.set_title(f"{name}  —  brightest week starts {week_label}", fontsize=10)
        ax.set_ylabel("Power (W)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", lw=0.5)

        rows.append(
            {
                "site": name,
                "brightest_week_start_doy": start_day + 1,  # store as 1-based DoY
                "brightest_week_start_label": week_label,
                "week_solar_kWh": float(p_solar.sum() / 1000.0),
                "week_wind_kWh": float(p_wind.sum() / 1000.0),
                "week_load_kWh": float(p_out.sum() / 1000.0),
                "week_grid_kWh": float(soc_df["p_grid_W"].sum() / 1000.0),
                "week_curtailed_kWh": float(soc_df["p_curtailed_W"].sum() / 1000.0),
                "soc_min_pct": float(soc_df["soc_frac"].min() * 100.0),
                "soc_max_pct": float(soc_df["soc_frac"].max() * 100.0),
            }
        )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=4, fontsize=9)
    fig.suptitle(
        "F7. Battery state-of-charge over each site's brightest 7-day window — codesigned reference\n"
        "Hemisphere-symmetric: every panel shows that site's own peak-irradiance week, so all sites\n"
        "are compared on a like-for-like 'best representative summer week' basis.\n"
        f"({PV_AREA_M2_REF:.0f} m² PV, 2 m² turbine, {BATTERY_WH_REF/1000:.0f} kWh pack, "
        f"{OPERATING_POWER_W/1000:.1f} kW operating draw 09:00–15:00)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


# ---------------------------------------------------------------------------
# F8 — Off-grid feasibility map for best / median / worst site
# ---------------------------------------------------------------------------

def figure_8_feasibility_map(
    out_png: Path,
    out_csv: Path,
    panel_areas_m2: np.ndarray | None = None,
    battery_capacities_Wh: np.ndarray | None = None,
) -> pd.DataFrame:
    if panel_areas_m2 is None:
        # Sweep brackets the codesigned 15 m² reference and extends both
        # ways for sensitivity. We hold the small wind turbine fixed.
        panel_areas_m2 = np.linspace(4.0, 30.0, 10)
    if battery_capacities_Wh is None:
        # Sweep brackets the codesigned 9 kWh reference.
        battery_capacities_Wh = np.linspace(2000.0, 24000.0, 10)

    sites = load_site_meta()
    annual_kWh = {}
    for name in sites:
        tmy = synthesize_tmy_year(name, year=2024, seed=0)
        annual_kWh[name] = float(tmy["ghi_W_m2"].sum() / 1000.0)

    sorted_sites = sorted(annual_kWh.items(), key=lambda kv: kv[1], reverse=True)
    best = sorted_sites[0][0]
    median = sorted_sites[len(sorted_sites) // 2][0]
    worst = sorted_sites[-1][0]
    selected = [best, median, worst]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    rows = []
    for ax, name in zip(axes, selected):
        tmy = synthesize_tmy_year(name, year=2024, seed=0)
        # Compute the wind contribution once (independent of panel area)
        p_wind = np.array([wind_turbine_power(w, WIND) for w in tmy["wind_m_s"]])

        grid = np.zeros((len(battery_capacities_Wh), len(panel_areas_m2)))
        for i, area in enumerate(panel_areas_m2):
            panel_i = PanelSpec(area_m2=float(area))
            p_solar = np.array([panel_dc_power(g, t, panel_i) for g, t in zip(tmy["ghi_W_m2"], tmy["temp_C"])])
            p_in = p_solar + p_wind
            p_out = operating_load_profile(tmy.index)
            for j, cap in enumerate(battery_capacities_Wh):
                bat = BatterySpec(capacity_Wh=float(cap))
                soc_df = battery_soc_simulation(p_in, p_out, bat)
                grid_hours = float((soc_df["p_grid_W"] > 0.0).sum())
                grid[j, i] = grid_hours
                rows.append(
                    {
                        "site": name,
                        "panel_area_m2": float(area),
                        "battery_capacity_Wh": float(cap),
                        "annual_grid_hours": grid_hours,
                        "annual_grid_kWh": float(soc_df["p_grid_W"].sum() / 1000.0),
                        "annual_curtailed_kWh": float(soc_df["p_curtailed_W"].sum() / 1000.0),
                    }
                )

        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=(panel_areas_m2[0], panel_areas_m2[-1], battery_capacities_Wh[0] / 1000.0, battery_capacities_Wh[-1] / 1000.0),
            cmap="YlOrRd",
            vmin=0,
            vmax=4000,
        )
        # Contour at 100 grid hours = "essentially off-grid"
        cs = ax.contour(
            panel_areas_m2,
            battery_capacities_Wh / 1000.0,
            grid,
            levels=[100, 500, 1500],
            colors="black",
            linewidths=1.0,
        )
        ax.clabel(cs, fmt="%.0f h")
        # Codesigned reference marker at (15 m² PV, 9 kWh battery).
        ax.scatter(
            [PV_AREA_M2_REF], [BATTERY_WH_REF / 1000.0],
            marker="*", s=180, color="white", edgecolor="black", linewidth=1.2,
            zorder=6, label="codesigned reference",
        )
        ax.set_xlabel("Panel area (m²)")
        ax.set_title(f"{name}\n({annual_kWh[name]:.0f} kWh/m²/yr)", fontsize=10)

    axes[0].set_ylabel("Battery capacity (kWh)")
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Annual grid hours")
    fig.suptitle(
        f"F8. Off-grid feasibility — annual grid hours, codesigned CableTract duty cycle\n"
        f"({OPERATING_POWER_W/1000:.1f} kW draw 09:00–15:00 daily, "
        f"{IDLE_POWER_W:.0f} W idle; codesigned reference at "
        f"{PV_AREA_M2_REF:.0f} m² PV / {BATTERY_WH_REF/1000:.0f} kWh)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating F6 — calendar heatmaps (6 sites x 8760 hours each)...")
    f6 = figure_6_calendar_heatmaps(OUT_DIR / "F6_calendar_heatmap.png", TAB_DIR / "F6_calendar_heatmap.csv")
    print("Generating F7 — battery SoC week per site...")
    f7 = figure_7_battery_week(OUT_DIR / "F7_battery_week.png", TAB_DIR / "F7_battery_week.csv")
    print("Generating F8 — off-grid feasibility maps (3 sites x 100 design points)...")
    f8 = figure_8_feasibility_map(OUT_DIR / "F8_feasibility_map.png", TAB_DIR / "F8_feasibility_map.csv")

    print()
    print("=== Phase 3 headline numbers ===")
    print()
    print("F6. Annual harvested energy and decares-covered per site:")
    print(f6[["site", "annual_solar_kWh", "annual_wind_kWh", "annual_decares", "p10_decares_per_day", "p50_decares_per_day", "p90_decares_per_day"]].to_string(index=False, float_format="%.1f"))

    print()
    print(f"F7. Summer-week energy balance (load = {OPERATING_POWER_W/1000:.1f} kW operating):")
    print(f7.to_string(index=False, float_format="%.2f"))

    print()
    print("F8. Off-grid feasibility per site:")
    for site in f8["site"].unique():
        sub = f8[f8["site"] == site]
        # Try the strict <100 h target first
        strict = sub[sub["annual_grid_hours"] < 100.0]
        if not strict.empty:
            strict = strict.assign(cost_proxy=strict["panel_area_m2"] * strict["battery_capacity_Wh"])
            row = strict.loc[strict["cost_proxy"].idxmin()]
            print(f"  {site:15s} <100 h target met: panel={row['panel_area_m2']:.1f} m²  battery={row['battery_capacity_Wh']/1000:.1f} kWh  grid={row['annual_grid_hours']:.0f} h")
        else:
            # Fall back to the lowest grid-hours design in the swept space
            row = sub.loc[sub["annual_grid_hours"].idxmin()]
            print(f"  {site:15s} <100 h NOT achievable; best: panel={row['panel_area_m2']:.1f} m²  battery={row['battery_capacity_Wh']/1000:.1f} kWh  grid={row['annual_grid_hours']:.0f} h ({row['annual_grid_kWh']:.0f} kWh)")

    print()
    print(f"Figures written to {OUT_DIR.resolve()}")
    print(f"Tables  written to {TAB_DIR.resolve()}")


if __name__ == "__main__":
    main()
