"""Verification tests for cabletract.energy (Phase 3).

Each test pins down one invariant of the TMY synthesizer, the panel/wind
power converters, or the battery state-of-charge simulator.

Run as:

    python tests/test_energy.py
    pytest tests/test_energy.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from cabletract.energy import (  # noqa: E402
    BatterySpec,
    PanelSpec,
    SiteMeta,
    WindSpec,
    battery_soc_simulation,
    clear_sky_ghi,
    daily_harvested_energy,
    extraterrestrial_ghi,
    load_site_meta,
    panel_dc_power,
    solar_declination_rad,
    solar_zenith_cosine,
    synthesize_tmy_year,
    wind_turbine_power,
)


# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------

def test_declination_summer_solstice() -> None:
    """Day 172 (≈ 21 June) → declination ≈ +23.45°."""
    delta = solar_declination_rad(172)
    assert math.isclose(math.degrees(delta), 23.45, abs_tol=0.5)


def test_declination_winter_solstice() -> None:
    """Day 355 (≈ 21 December) → declination ≈ -23.45°."""
    delta = solar_declination_rad(355)
    assert math.isclose(math.degrees(delta), -23.45, abs_tol=0.5)


def test_zenith_cosine_solar_noon_equinox_equator() -> None:
    """At lat=0, doy=80 (≈ vernal equinox), solar noon → cos(z) ≈ 1."""
    cos_z = solar_zenith_cosine(0.0, 80, 12.0)
    assert math.isclose(cos_z, 1.0, abs_tol=0.05)


def test_zenith_cosine_negative_at_night() -> None:
    """Midnight → cos(z) < 0 anywhere on Earth."""
    cos_z = solar_zenith_cosine(40.0, 172, 0.0)
    assert cos_z < 0.0


def test_clear_sky_ghi_zero_at_night() -> None:
    assert clear_sky_ghi(40.0, 172, 0.0) == 0.0
    assert clear_sky_ghi(40.0, 172, 23.5) == 0.0


def test_clear_sky_ghi_peak_under_solar_constant() -> None:
    """Haurwitz noon GHI must be below the solar constant (1361 W/m²)."""
    g = clear_sky_ghi(0.0, 80, 12.0)
    assert 0.0 < g < 1361.0


def test_extraterrestrial_ghi_above_clear_sky() -> None:
    """ET-GHI must exceed surface clear-sky GHI (atmosphere absorbs ~20%)."""
    et = extraterrestrial_ghi(40.0, 172, 12.0)
    cs = clear_sky_ghi(40.0, 172, 12.0)
    assert et > cs > 0.0


def test_extraterrestrial_ghi_zero_at_night() -> None:
    assert extraterrestrial_ghi(40.0, 172, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Site metadata loader
# ---------------------------------------------------------------------------

def test_site_meta_has_six_sites() -> None:
    sites = load_site_meta()
    assert isinstance(sites, dict)
    assert len(sites) == 6


def test_site_meta_kt_arrays_length_12() -> None:
    sites = load_site_meta()
    for meta in sites.values():
        assert len(meta.Kt) == 12
        assert len(meta.wind_mean_m_s) == 12
        assert len(meta.T_mean_C) == 12


def test_site_meta_lookup_konya() -> None:
    meta = load_site_meta("Konya_TR")
    assert isinstance(meta, SiteMeta)
    assert math.isclose(meta.latitude, 37.87, abs_tol=0.01)


# ---------------------------------------------------------------------------
# TMY synthesizer
# ---------------------------------------------------------------------------

def test_synthesize_tmy_shape() -> None:
    tmy = synthesize_tmy_year("Konya_TR", year=2023, seed=0)
    assert tmy.shape == (8760, 5)
    assert set(tmy.columns) == {"ghi_W_m2", "dni_W_m2", "dhi_W_m2", "wind_m_s", "temp_C"}


def test_synthesize_tmy_ghi_nonnegative_and_bounded() -> None:
    tmy = synthesize_tmy_year("Konya_TR", year=2023, seed=0)
    assert (tmy["ghi_W_m2"] >= 0.0).all()
    # No hour can exceed the Haurwitz physical clear-sky maximum ~1100 W/m².
    assert tmy["ghi_W_m2"].max() < 1200.0


def test_synthesize_tmy_annual_total_within_15pct_of_published() -> None:
    """The synthesizer must reproduce published annual GHI within 15%
    for every bundled site. This is the Phase 3 manuscript-claim test."""
    sites = load_site_meta()
    for name, meta in sites.items():
        tmy = synthesize_tmy_year(name, year=2024, seed=0)
        annual_kWh = float(tmy["ghi_W_m2"].sum() / 1000.0)
        ratio = annual_kWh / meta.published_GHI_kWh_m2_yr
        assert 0.85 <= ratio <= 1.15, f"{name}: ratio={ratio:.2f}"


def test_synthesize_tmy_reproducible_with_seed() -> None:
    a = synthesize_tmy_year("Palencia_ES", year=2023, seed=42)
    b = synthesize_tmy_year("Palencia_ES", year=2023, seed=42)
    assert np.allclose(a["ghi_W_m2"], b["ghi_W_m2"])
    assert np.allclose(a["wind_m_s"], b["wind_m_s"])


def test_synthesize_tmy_different_seeds_differ() -> None:
    a = synthesize_tmy_year("Palencia_ES", year=2023, seed=1)
    b = synthesize_tmy_year("Palencia_ES", year=2023, seed=2)
    assert not np.allclose(a["ghi_W_m2"], b["ghi_W_m2"])


def test_synthesize_tmy_sunnier_site_has_higher_annual_ghi() -> None:
    """Konya (sunny) must out-irradiate Beauce (cloudy)."""
    konya = synthesize_tmy_year("Konya_TR", year=2024, seed=0)
    beauce = synthesize_tmy_year("Beauce_FR", year=2024, seed=0)
    assert konya["ghi_W_m2"].sum() > beauce["ghi_W_m2"].sum()


# ---------------------------------------------------------------------------
# PV and wind power converters
# ---------------------------------------------------------------------------

def test_panel_zero_at_night() -> None:
    p = PanelSpec(area_m2=10.0)
    assert panel_dc_power(0.0, 10.0, p) == 0.0


def test_panel_linear_in_ghi_at_fixed_temp() -> None:
    p = PanelSpec(area_m2=10.0)
    g1 = panel_dc_power(200.0, 25.0, p)
    g2 = panel_dc_power(800.0, 25.0, p)
    # 4× irradiance → 4× power (no temperature drift since temp is fixed)
    assert math.isclose(g2, 4.0 * g1, rel_tol=1e-9)


def test_panel_lower_at_high_temp() -> None:
    p = PanelSpec(area_m2=10.0)
    cool = panel_dc_power(800.0, 5.0, p)
    hot = panel_dc_power(800.0, 35.0, p)
    assert cool > hot


def test_panel_stc_consistency() -> None:
    """At GHI=1000, ambient T=0 C → cell T = 25 C, the temp deration is 1.0,
    so power ≈ 1000 * area * eta * soiling * inverter."""
    p = PanelSpec(area_m2=10.0, eta_stc=0.20, soiling=0.95, inverter_eta=0.96)
    expected = 1000.0 * 10.0 * 0.20 * 0.95 * 0.96
    actual = panel_dc_power(1000.0, 0.0, p)
    assert math.isclose(actual, expected, rel_tol=1e-6)


def test_wind_zero_below_cut_in_and_above_cut_out() -> None:
    w = WindSpec(swept_area_m2=2.0)
    assert wind_turbine_power(1.0, w) == 0.0
    assert wind_turbine_power(30.0, w) == 0.0


def test_wind_clamped_at_rated() -> None:
    w = WindSpec(swept_area_m2=2.0, rated_power_W=600.0)
    assert wind_turbine_power(20.0, w) == 600.0


def test_wind_cubic_below_rated() -> None:
    """Below rated, power ∝ v³."""
    w = WindSpec(swept_area_m2=2.0, rated_power_W=10000.0, cp=0.30)
    p1 = wind_turbine_power(4.0, w)
    p2 = wind_turbine_power(8.0, w)
    assert math.isclose(p2, 8.0 * p1, rel_tol=1e-9)  # (8/4)^3 = 8


# ---------------------------------------------------------------------------
# Daily harvested energy
# ---------------------------------------------------------------------------

def test_daily_harvested_energy_columns_and_length() -> None:
    tmy = synthesize_tmy_year("Konya_TR", year=2023, seed=0)
    df = daily_harvested_energy(tmy, PanelSpec(area_m2=8.0), WindSpec(swept_area_m2=2.0))
    assert set(df.columns) == {"solar_Wh", "wind_Wh", "total_Wh"}
    assert len(df) == 365
    assert (df["solar_Wh"] >= 0.0).all()
    assert (df["wind_Wh"] >= 0.0).all()


def test_daily_harvested_energy_summer_higher_than_winter_in_konya() -> None:
    tmy = synthesize_tmy_year("Konya_TR", year=2023, seed=0)
    df = daily_harvested_energy(tmy, PanelSpec(area_m2=8.0), WindSpec(swept_area_m2=2.0))
    # June-August solar > December-February solar
    summer = df["solar_Wh"].iloc[151:243].sum()
    winter = df["solar_Wh"].iloc[0:59].sum() + df["solar_Wh"].iloc[334:365].sum()
    assert summer > winter


# ---------------------------------------------------------------------------
# Battery state-of-charge simulator
# ---------------------------------------------------------------------------

def test_battery_constant_surplus_charges_to_max() -> None:
    """A constant 1 kW surplus eventually fills a 5 kWh battery."""
    n = 24
    p_in = np.full(n, 1000.0)
    p_out = np.zeros(n)
    bat = BatterySpec(capacity_Wh=5000.0, soc_init_frac=0.1)
    df = battery_soc_simulation(p_in, p_out, bat)
    assert math.isclose(df["soc_frac"].iloc[-1], bat.soc_max_frac, abs_tol=1e-9)
    # Surplus after saturation must show up as curtailment
    assert df["p_curtailed_W"].iloc[-1] > 0.0


def test_battery_constant_deficit_drains_to_min_then_grid() -> None:
    n = 24
    p_in = np.zeros(n)
    p_out = np.full(n, 1000.0)
    bat = BatterySpec(capacity_Wh=2000.0, soc_init_frac=0.5)
    df = battery_soc_simulation(p_in, p_out, bat)
    assert math.isclose(df["soc_frac"].iloc[-1], bat.soc_min_frac, abs_tol=1e-9)
    # Once empty, the deficit must be served by grid
    assert df["p_grid_W"].iloc[-1] > 0.0


def test_battery_charge_conservation_no_grid() -> None:
    """Energy in (after charge eta) - energy out (after discharge eta)
    = ΔSoC, when neither curtailment nor grid is involved."""
    rng = np.random.default_rng(0)
    n = 100
    p_in = rng.uniform(0.0, 800.0, n)
    p_out = rng.uniform(0.0, 800.0, n)
    bat = BatterySpec(capacity_Wh=10000.0, soc_init_frac=0.5)
    df = battery_soc_simulation(p_in, p_out, bat)
    # Filter to clean steps (no curtailment, no grid)
    mask = (df["p_curtailed_W"] == 0.0) & (df["p_grid_W"] == 0.0)
    soc = df["soc_Wh"].values
    soc_prev = np.concatenate([[bat.capacity_Wh * bat.soc_init_frac], soc[:-1]])
    delta = soc - soc_prev
    expected = (
        df["p_charge_W"].values * bat.eta_charge
        - df["p_discharge_W"].values / bat.eta_discharge
    )
    assert np.allclose(delta[mask], expected[mask], atol=1e-6)


def test_battery_soc_within_bounds() -> None:
    rng = np.random.default_rng(7)
    n = 200
    p_in = rng.uniform(0.0, 1500.0, n)
    p_out = rng.uniform(0.0, 1500.0, n)
    bat = BatterySpec(capacity_Wh=4000.0)
    df = battery_soc_simulation(p_in, p_out, bat)
    assert (df["soc_frac"] >= bat.soc_min_frac - 1e-9).all()
    assert (df["soc_frac"] <= bat.soc_max_frac + 1e-9).all()


def test_battery_zero_in_zero_out_keeps_soc_constant() -> None:
    n = 24
    bat = BatterySpec(capacity_Wh=3000.0, soc_init_frac=0.5)
    df = battery_soc_simulation(np.zeros(n), np.zeros(n), bat)
    assert math.isclose(df["soc_frac"].iloc[-1], 0.5, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Standalone driver
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_declination_summer_solstice,
    test_declination_winter_solstice,
    test_zenith_cosine_solar_noon_equinox_equator,
    test_zenith_cosine_negative_at_night,
    test_clear_sky_ghi_zero_at_night,
    test_clear_sky_ghi_peak_under_solar_constant,
    test_extraterrestrial_ghi_above_clear_sky,
    test_extraterrestrial_ghi_zero_at_night,
    test_site_meta_has_six_sites,
    test_site_meta_kt_arrays_length_12,
    test_site_meta_lookup_konya,
    test_synthesize_tmy_shape,
    test_synthesize_tmy_ghi_nonnegative_and_bounded,
    test_synthesize_tmy_annual_total_within_15pct_of_published,
    test_synthesize_tmy_reproducible_with_seed,
    test_synthesize_tmy_different_seeds_differ,
    test_synthesize_tmy_sunnier_site_has_higher_annual_ghi,
    test_panel_zero_at_night,
    test_panel_linear_in_ghi_at_fixed_temp,
    test_panel_lower_at_high_temp,
    test_panel_stc_consistency,
    test_wind_zero_below_cut_in_and_above_cut_out,
    test_wind_clamped_at_rated,
    test_wind_cubic_below_rated,
    test_daily_harvested_energy_columns_and_length,
    test_daily_harvested_energy_summer_higher_than_winter_in_konya,
    test_battery_constant_surplus_charges_to_max,
    test_battery_constant_deficit_drains_to_min_then_grid,
    test_battery_charge_conservation_no_grid,
    test_battery_soc_within_bounds,
    test_battery_zero_in_zero_out_keeps_soc_constant,
]


if __name__ == "__main__":
    failures = []
    for t in ALL_TESTS:
        try:
            t()
            print(f"  ok  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {t.__name__}: {exc}")
            failures.append((t.__name__, exc))
    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        sys.exit(1)
    print(f"\nok: cabletract.energy passes {len(ALL_TESTS)} invariants")
