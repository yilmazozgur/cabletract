"""Phase 3 — Real solar/wind harvested energy and battery state-of-charge.

The v1 model collapses the entire renewable-energy story into

    harvested_per_day = solar_W_m2 * solar_area * solar_hours + wind_W * wind_hours

with `solar_W_m2 = 150` and `solar_hours = 6` (see
`cabletract.simulate.run_single`). This is a single-site, single-day,
single-irradiance approximation that hides the seasonal/diurnal/cloudy
reality every reviewer will probe.

This module replaces it with:

1. A small Haurwitz clear-sky model (`clear_sky_ghi`) so we can compute
   irradiance from latitude, day-of-year and hour-of-day with no
   external dependencies.
2. A deterministic TMY synthesizer (`synthesize_tmy_year`) that takes
   the monthly clearness index `Kt`, mean wind speed and mean temperature
   from `cabletract/data/tmy/site_meta.csv` and returns an 8760-row hourly
   DataFrame for any of six bundled sites: Konya (TR), Palencia (ES),
   Beauce (FR), Des Moines (IA, US), Ludhiana (IN), and São Paulo (BR).
3. A panel/wind power converter (`panel_dc_power`, `wind_turbine_power`)
   with explicit datasheet-style efficiency parameters.
4. An hourly battery state-of-charge simulator
   (`battery_soc_simulation`) with charge/discharge limits and round-trip
   loss accounting, returning the full SoC trajectory plus the grid-charge
   energy needed to keep the system running 24/7.

The synthesizer is *not* a replacement for measured TMY3 data — it is
honest about that and is documented in the manuscript as
"synthesized from monthly statistics". For a feasibility paper this is
sufficient: a sceptical reviewer can swap in real TMY3 by replacing
`synthesize_tmy_year` with `pvlib.iotools.read_tmy3`.

References for the synthesis model:
    - Haurwitz, B. (1945) "Insolation in relation to cloudiness and
      cloud density", Journal of Meteorology — clear-sky GHI form.
    - Cooper, P.I. (1969) "The absorption of radiation in solar stills",
      Solar Energy — solar declination formula.
    - Justus, C.G. (1978) "Winds and wind system performance", Franklin
      Institute Press — Weibull-shaped wind frequency distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

SITE_META_CSV = Path(__file__).resolve().parent / "data" / "tmy" / "site_meta.csv"

# Physical constants
SOLAR_CONSTANT = 1361.0  # W/m^2
RHO_AIR = 1.225          # kg/m^3 (sea level, 15 C)


# ---------------------------------------------------------------------------
# Solar geometry
# ---------------------------------------------------------------------------

def solar_declination_rad(day_of_year: int) -> float:
    """Cooper (1969) solar declination in radians."""
    return math.radians(23.45) * math.sin(2.0 * math.pi * (284 + day_of_year) / 365.0)


def solar_zenith_cosine(latitude_deg: float, day_of_year: int, solar_hour: float) -> float:
    """Cosine of solar zenith angle (1 = sun overhead, 0 = horizon, <0 = night).

    Uses local solar time (no equation-of-time correction). The hour angle is
    measured from solar noon at 15 deg / hour.
    """
    phi = math.radians(latitude_deg)
    delta = solar_declination_rad(day_of_year)
    omega = math.radians(15.0 * (solar_hour - 12.0))
    cos_z = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)
    return cos_z


def extraterrestrial_ghi(latitude_deg: float, day_of_year: int, solar_hour: float) -> float:
    """Top-of-atmosphere global horizontal irradiance in W/m^2.

    This is the standard reference irradiance against which the clearness
    index ``Kt = GHI_surface / GHI_extraterrestrial`` is defined in NSRDB,
    PVGIS, and Iqbal (1983). Includes the eccentricity correction
    ``1 + 0.033 cos(2π·doy/365)``.

    Returns 0 below the horizon.
    """
    cos_z = solar_zenith_cosine(latitude_deg, day_of_year, solar_hour)
    if cos_z <= 0.0:
        return 0.0
    eccentricity = 1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    return SOLAR_CONSTANT * eccentricity * cos_z


def clear_sky_ghi(latitude_deg: float, day_of_year: int, solar_hour: float) -> float:
    """Haurwitz (1945) clear-sky global horizontal irradiance in W/m^2.

    Returns 0 when the sun is below the horizon. The Haurwitz form is

        GHI_clear = 1098 * cos(z) * exp(-0.057 / cos(z))         (W/m^2)

    Used as the *upper bound* for synthesised hourly irradiance: a stochastic
    cloud factor cannot push the surface GHI above this value.
    """
    cos_z = solar_zenith_cosine(latitude_deg, day_of_year, solar_hour)
    if cos_z <= 0.0:
        return 0.0
    return 1098.0 * cos_z * math.exp(-0.057 / cos_z)


# ---------------------------------------------------------------------------
# Site metadata loader
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SiteMeta:
    """Monthly statistics for one TMY synthesis site."""

    name: str
    latitude: float
    longitude: float
    elevation_m: float
    Kt: np.ndarray            # length-12 monthly clearness index
    wind_mean_m_s: np.ndarray # length-12 monthly mean wind speed
    T_mean_C: np.ndarray      # length-12 monthly mean air temperature
    published_GHI_kWh_m2_yr: float
    source: str


def load_site_meta(site: str | None = None) -> Dict[str, SiteMeta] | SiteMeta:
    """Load all site metadata, or one site by name."""
    df = pd.read_csv(SITE_META_CSV)
    out: Dict[str, SiteMeta] = {}
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    for _, row in df.iterrows():
        meta = SiteMeta(
            name=str(row["site"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            elevation_m=float(row["elevation_m"]),
            Kt=np.array([float(row[f"Kt_{m}"]) for m in months]),
            wind_mean_m_s=np.array([float(row[f"wind_{m}"]) for m in months]),
            T_mean_C=np.array([float(row[f"T_{m}"]) for m in months]),
            published_GHI_kWh_m2_yr=float(row["published_GHI_kWh_m2_yr"]),
            source=str(row["source"]),
        )
        out[meta.name] = meta
    if site is not None:
        if site not in out:
            raise KeyError(f"site {site!r} not in site_meta.csv (have {list(out)})")
        return out[site]
    return out


# ---------------------------------------------------------------------------
# TMY synthesis
# ---------------------------------------------------------------------------

def _month_of_doy(doy: int) -> int:
    """Return 1-12 month of a (1-indexed) day-of-year for a non-leap year."""
    cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]
    for m in range(12):
        if doy <= cum[m + 1]:
            return m + 1
    return 12


def synthesize_tmy_year(
    site: str,
    year: int = 2024,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate one full year of hourly weather for the named site.

    The synthesis model:

    1. For each hour we compute the *extraterrestrial* GHI (top-of-atmosphere)
       using `extraterrestrial_ghi`. This is the reference irradiance against
       which the published clearness index `Kt` is defined in NSRDB / PVGIS.
    2. We multiply by the monthly `Kt` to obtain the mean surface irradiance
       for that month.
    3. We modulate hour-to-hour by a stochastic cloud factor sampled from a
       Beta(2, 2) distribution scaled to mean 1.0 (so a sunny month with
       Kt=0.7 still has cloudy hours, and a cloudy month with Kt=0.3 still
       has bright hours), and clamp the result by `clear_sky_ghi` so the
       synthesised hourly value never exceeds the Haurwitz physical maximum.
    4. Wind speed is drawn from a Weibull(k=2.0, scale=monthly_mean*1.128)
       distribution every hour. The 1.128 ≈ 1/Γ(1.5) makes the Weibull
       mean equal the published monthly mean.
    5. Temperature is a sinusoidal diurnal swing of ±5 C around the
       monthly mean.

    Returns a DataFrame indexed by hourly timestamps with columns
    ``ghi_W_m2, dni_W_m2, dhi_W_m2, wind_m_s, temp_C``. DNI/DHI are
    derived with the simple Erbs split (DHI = 1.0 * GHI for K_t < 0.22
    is replaced here with a flat 0.3 GHI for simplicity, since the
    Phase 3 figures depend on GHI not the split).
    """
    meta_obj = load_site_meta(site)
    if not isinstance(meta_obj, SiteMeta):
        raise TypeError("internal error: load_site_meta returned dict")
    meta = meta_obj
    rng = np.random.default_rng(seed + hash(site) % 2**32)

    timestamps = pd.date_range(start=f"{year}-01-01 00:00", periods=8760, freq="1h")
    rows = []
    for ts in timestamps:
        doy = int(ts.dayofyear)
        if doy > 365:
            doy = 365
        month_idx = _month_of_doy(doy) - 1
        hour = ts.hour + 0.5  # midpoint of the hour
        ghi_et = extraterrestrial_ghi(meta.latitude, doy, hour)
        ghi_max = clear_sky_ghi(meta.latitude, doy, hour)
        kt = float(meta.Kt[month_idx])
        # Beta(4,4) has mean 0.5, std 0.158 → scale to mean 1.0. The narrower
        # distribution (vs Beta(2,2)) reduces high-cloud-factor outliers that
        # would otherwise get clamped by the Haurwitz ceiling at low latitudes.
        cloud = 2.0 * float(rng.beta(4.0, 4.0))
        ghi = max(0.0, min(kt * ghi_et * cloud, ghi_max))

        # Weibull wind, k=2, scale chosen so the mean equals the monthly mean
        mean_wind = float(meta.wind_mean_m_s[month_idx])
        scale = mean_wind / 0.886227  # 1 / Γ(1.5)
        wind = float(rng.weibull(2.0)) * scale

        # Temperature: monthly mean + ±5 C diurnal sinusoid (peak at 14:00)
        diurnal = 5.0 * math.sin(2.0 * math.pi * (hour - 8.0) / 24.0)
        temp_C = float(meta.T_mean_C[month_idx]) + diurnal

        # Erbs-style simple split into DNI/DHI
        if ghi <= 0.0:
            dni = 0.0
            dhi = 0.0
        else:
            dhi = 0.30 * ghi
            cos_z = solar_zenith_cosine(meta.latitude, doy, hour)
            dni = max(0.0, (ghi - dhi) / max(cos_z, 0.05))
        rows.append((ts, ghi, dni, dhi, wind, temp_C))

    return pd.DataFrame(rows, columns=["timestamp", "ghi_W_m2", "dni_W_m2", "dhi_W_m2", "wind_m_s", "temp_C"]).set_index("timestamp")


# ---------------------------------------------------------------------------
# Solar PV and wind turbine models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelSpec:
    """Mono-c-Si flat-plate PV array spec."""

    area_m2: float
    eta_stc: float = 0.20         # 20% module efficiency at STC
    temp_coef_per_C: float = -0.004  # -0.4 %/C above 25 C
    inverter_eta: float = 0.96
    soiling: float = 0.95            # 5% soiling/dirt loss


def panel_dc_power(ghi_W_m2: float, temp_C: float, panel: PanelSpec) -> float:
    """Instantaneous AC output (W) of a flat-plate PV array under given GHI."""
    if ghi_W_m2 <= 0.0:
        return 0.0
    cell_T = temp_C + 25.0  # NOCT-style 25 C above ambient under load
    eta_T = panel.eta_stc * (1.0 + panel.temp_coef_per_C * (cell_T - 25.0))
    eta_T = max(eta_T, 0.0)
    p_dc = ghi_W_m2 * panel.area_m2 * eta_T * panel.soiling
    return p_dc * panel.inverter_eta


@dataclass(frozen=True)
class WindSpec:
    """Small helix / vertical-axis wind turbine spec."""

    swept_area_m2: float
    cp: float = 0.30           # power coefficient (Betz limit 0.59)
    rated_power_W: float = 600.0
    cut_in_m_s: float = 2.5
    cut_out_m_s: float = 25.0


def wind_turbine_power(wind_m_s: float, wind: WindSpec) -> float:
    """Instantaneous output (W) of a small wind turbine."""
    if wind_m_s < wind.cut_in_m_s or wind_m_s > wind.cut_out_m_s:
        return 0.0
    p_aero = 0.5 * RHO_AIR * wind.swept_area_m2 * (wind_m_s ** 3) * wind.cp
    return min(p_aero, wind.rated_power_W)


# ---------------------------------------------------------------------------
# Daily harvested energy summary
# ---------------------------------------------------------------------------

def daily_harvested_energy(
    tmy: pd.DataFrame,
    panel: PanelSpec,
    wind: WindSpec,
) -> pd.DataFrame:
    """Daily totals (Wh) of solar, wind, and combined harvested energy."""
    p_solar = np.array([panel_dc_power(g, t, panel) for g, t in zip(tmy["ghi_W_m2"], tmy["temp_C"])])
    p_wind = np.array([wind_turbine_power(w, wind) for w in tmy["wind_m_s"]])
    df = pd.DataFrame(
        {
            "p_solar_W": p_solar,
            "p_wind_W": p_wind,
            "p_total_W": p_solar + p_wind,
        },
        index=tmy.index,
    )
    daily = df.resample("1D").sum()  # Wh per day (each hour contributes W·1h = Wh)
    daily.columns = ["solar_Wh", "wind_Wh", "total_Wh"]
    return daily


# ---------------------------------------------------------------------------
# Battery state-of-charge simulation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BatterySpec:
    """Lithium battery for the CableTract Main Unit."""

    capacity_Wh: float
    eta_charge: float = 0.96
    eta_discharge: float = 0.96
    max_charge_W: float = 5000.0
    max_discharge_W: float = 5000.0
    soc_init_frac: float = 0.5
    soc_min_frac: float = 0.10
    soc_max_frac: float = 0.95


def battery_soc_simulation(
    p_in_series: np.ndarray,
    p_out_series: np.ndarray,
    battery: BatterySpec,
    dt_h: float = 1.0,
) -> pd.DataFrame:
    """Simulate hourly battery SoC under a (p_in, p_out) profile.

    Parameters
    ----------
    p_in_series : np.ndarray
        Power available from harvesting (W) at each timestep.
    p_out_series : np.ndarray
        Power demanded by the load (W) at each timestep.
    battery : BatterySpec
        Capacity, charge/discharge limits and round-trip efficiency.
    dt_h : float
        Timestep in hours (default 1.0).

    Returns
    -------
    DataFrame with columns:
        soc_Wh           battery state-of-charge (Wh)
        soc_frac         SoC as a fraction of capacity
        p_charge_W       actual power into the battery (W, ≥0)
        p_discharge_W    actual power out of the battery (W, ≥0)
        p_curtailed_W    excess power that overflowed full battery (W)
        p_grid_W         grid import to keep the load served (W)
    """
    if p_in_series.shape != p_out_series.shape:
        raise ValueError("p_in_series and p_out_series must have the same shape")
    n = len(p_in_series)
    cap = battery.capacity_Wh
    soc_min = cap * battery.soc_min_frac
    soc_max = cap * battery.soc_max_frac
    soc = cap * battery.soc_init_frac

    soc_arr = np.zeros(n)
    p_chg = np.zeros(n)
    p_dis = np.zeros(n)
    p_cur = np.zeros(n)
    p_grid = np.zeros(n)

    for i in range(n):
        p_in = float(p_in_series[i])
        p_out = float(p_out_series[i])
        net = p_in - p_out  # >0 surplus, <0 deficit

        if net >= 0.0:
            # Surplus → try to charge
            p_charge_request = min(net, battery.max_charge_W)
            energy_in = p_charge_request * dt_h * battery.eta_charge
            room = soc_max - soc
            if energy_in > room:
                # Battery saturates → curtail
                p_charge_actual = (room / battery.eta_charge) / dt_h
                p_curtailed = net - p_charge_actual
                soc = soc_max
            else:
                p_charge_actual = p_charge_request
                p_curtailed = net - p_charge_actual
                soc += energy_in
            p_chg[i] = p_charge_actual
            p_cur[i] = max(p_curtailed, 0.0)
        else:
            # Deficit → discharge battery, fall back to grid if needed
            p_deficit = -net
            p_dis_request = min(p_deficit, battery.max_discharge_W)
            energy_out = p_dis_request * dt_h / battery.eta_discharge
            available = soc - soc_min
            if energy_out > available:
                p_dis_actual = (available * battery.eta_discharge) / dt_h
                p_grid_needed = p_deficit - p_dis_actual
                soc = soc_min
            else:
                p_dis_actual = p_dis_request
                p_grid_needed = p_deficit - p_dis_actual
                soc -= energy_out
            p_dis[i] = p_dis_actual
            p_grid[i] = max(p_grid_needed, 0.0)

        soc_arr[i] = soc

    return pd.DataFrame(
        {
            "soc_Wh": soc_arr,
            "soc_frac": soc_arr / cap,
            "p_charge_W": p_chg,
            "p_discharge_W": p_dis,
            "p_curtailed_W": p_cur,
            "p_grid_W": p_grid,
        }
    )
