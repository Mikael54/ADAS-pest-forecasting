"""Feature construction for the ADAS pest forecast.

Information rule (enforced everywhere)
--------------------------------------
To predict harvest year Y we may use:
  * weather observed from Sept (Y-1) through JUNE (Y).  The ADAS survey scores
    L1/L2 around GS71-75 (late June / July) and Met Office monthly areal series
    for June are published in the first days of July, so a forecast issued in
    July of year Y legitimately has these.  Capping at June also makes the
    backtest features identical in availability to what we hold for 2026 today.
  * disease observations up to Y-1.
  * agronomic / fungicide / land-use covariates up to Y-1 (the repo has no 2026
    rows for these, so the pipeline must not depend on year-Y values).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data_external"

MO_VARS = ["Tmax", "Tmin", "Tmean", "Sunshine", "Rainfall", "Raindays1mm", "AirFrost"]
# season month -> (calendar month, year offset). Capped at June of harvest year.
SEASON_LABEL = {
    "s09": (9, -1), "s10": (10, -1), "s11": (11, -1), "s12": (12, -1),
    "s01": (1, 0), "s02": (2, 0), "s03": (3, 0), "s04": (4, 0),
    "s05": (5, 0), "s06": (6, 0),
}


def season_monthly():
    """Met Office monthly weather on the (Region, harvest Year) grid."""
    mo = pd.read_csv(EXT / "metoffice_regions_monthly.csv")
    long = mo.melt(id_vars=["Region", "Year"], var_name="col", value_name="value")
    long = long[long.col.str.startswith("mo_")].copy()
    long["var"] = long.col.str.split("_").str[1]
    long["month"] = long.col.str.split("_m").str[1].astype(int)
    out = []
    for lab, (m, off) in SEASON_LABEL.items():
        d = long[long.month == m].copy()
        d["HarvestYear"] = d.Year - off
        d["feat"] = "w_" + d["var"] + "_" + lab
        out.append(d[["Region", "HarvestYear", "feat", "value"]])
    d = pd.concat(out, ignore_index=True)
    w = d.pivot_table(index=["Region", "HarvestYear"], columns="feat", values="value")
    return w.reset_index().rename(columns={"HarvestYear": "Year"})


def _sum(df, var, labs):
    cols = [f"w_{var}_{l}" for l in labs]
    return df[cols].sum(axis=1, min_count=len(cols))


def _mean(df, var, labs):
    cols = [f"w_{var}_{l}" for l in labs]
    return df[cols].mean(axis=1)


def epi_features(w):
    """Epidemiologically motivated aggregates over the monthly weather.

    Septoria tritici: splash-dispersed, needs rain + leaf wetness during stem
    extension (Apr-Jun) to climb the canopy onto L2/L1; mild winters build the
    autumn/winter inoculum base.
    Yellow rust: overwinters as mycelium, so mild frost-free winters raise
    inoculum; epidemics run in cool moist springs and are shut down by June heat.
    """
    f = pd.DataFrame(index=w.index)
    f["Region"], f["Year"] = w.Region, w.Year

    AUT, WIN, SPR, LSPR = ["s09", "s10", "s11"], ["s12", "s01", "s02"], \
                          ["s03", "s04", "s05"], ["s04", "s05", "s06"]

    # --- moisture / wetness (Septoria driver) ---
    f["e_rain_spring"] = _sum(w, "Rainfall", SPR)
    f["e_rain_latespring"] = _sum(w, "Rainfall", LSPR)
    f["e_rain_apr_may"] = _sum(w, "Rainfall", ["s04", "s05"])
    f["e_rain_autumn"] = _sum(w, "Rainfall", AUT)
    f["e_rain_winter"] = _sum(w, "Rainfall", WIN)
    f["e_rain_season"] = _sum(w, "Rainfall", list(SEASON_LABEL))
    f["e_rdays_spring"] = _sum(w, "Raindays1mm", SPR)
    f["e_rdays_latespring"] = _sum(w, "Raindays1mm", LSPR)
    f["e_rdays_apr_may"] = _sum(w, "Raindays1mm", ["s04", "s05"])
    f["e_rdays_autumn"] = _sum(w, "Raindays1mm", AUT)
    f["e_sun_spring"] = _sum(w, "Sunshine", SPR)
    f["e_sun_latespring"] = _sum(w, "Sunshine", LSPR)
    f["e_sun_apr_may"] = _sum(w, "Sunshine", ["s04", "s05"])
    # rain days per sunshine hour: a compact "how wet was the canopy" index
    f["e_wetness_idx"] = f["e_rdays_latespring"] / (f["e_sun_latespring"] / 100 + 1)

    # --- temperature ---
    f["e_tmean_winter"] = _mean(w, "Tmean", WIN)
    f["e_tmin_winter"] = _mean(w, "Tmin", WIN)
    f["e_tmean_spring"] = _mean(w, "Tmean", SPR)
    f["e_tmin_spring"] = _mean(w, "Tmin", SPR)
    f["e_tmean_autumn"] = _mean(w, "Tmean", AUT)
    f["e_tmax_jun"] = w["w_Tmax_s06"]
    f["e_tmin_jun"] = w["w_Tmin_s06"]
    f["e_tmean_jun"] = w["w_Tmean_s06"]
    f["e_frost_winter"] = _sum(w, "AirFrost", WIN)
    f["e_frost_season"] = _sum(w, "AirFrost", list(SEASON_LABEL))
    f["e_frost_spring"] = _sum(w, "AirFrost", SPR)

    # --- interactions the epidemiology implies ---
    # Septoria needs wet AND mild together during stem extension
    f["e_warm_wet_spring"] = f["e_rain_spring"] * f["e_tmean_spring"] / 100
    f["e_warmwet_latespring"] = f["e_rdays_latespring"] * f["e_tmean_spring"] / 10
    # Yellow rust: mild winter inoculum survival, then cool spring, no June heat
    f["e_mild_winter"] = f["e_tmin_winter"] - f["e_frost_winter"] / 10
    f["e_rust_window"] = f["e_mild_winter"] * (20 - f["e_tmax_jun"])
    f["e_dry_bright_spring"] = f["e_sun_apr_may"] / (f["e_rain_apr_may"] + 20)
    return f


def add_climatology_anomalies(f, ref_start=1961, ref_end=2000):
    """Express each epi feature as an anomaly vs its own region's climatology.

    Regional climate differs hugely (Wales is far wetter than East Anglia), so
    the absolute value confounds "which region" with "how unusual was this year".

    The reference window ends in 2000, before the first backtest year (2005), so
    the standardising constants never see data from or after any year being
    predicted. (An earlier 1961-2010 window leaked mildly into the 2005-2010
    evaluations.)
    """
    epi = [c for c in f.columns if c.startswith("e_")]
    ref = f[(f.Year >= ref_start) & (f.Year <= ref_end)].groupby("Region")[epi]
    mu, sd = ref.mean(), ref.std()
    out = f.copy()
    for c in epi:
        out[c + "_anom"] = (f[c] - f.Region.map(mu[c])) / f.Region.map(sd[c]).replace(0, np.nan)
    return out


def build_weather_features():
    w = season_monthly()
    f = epi_features(w)
    f = add_climatology_anomalies(f)
    return w.merge(f, on=["Region", "Year"], how="left")


# --------------------------------------------------------------------------
def national_year_index(pest, targets):
    """National (cross-region) mean of each target per year, for lag features."""
    return pest.groupby("Year")[targets].mean()


def _last_prior(obs_years, obs_vals, query_years):
    """For each query year, the value at the most recent strictly-earlier obs."""
    idx = np.searchsorted(obs_years, query_years, side="left") - 1
    out = np.where(idx >= 0, obs_vals[np.clip(idx, 0, None)], np.nan)
    return out


def add_lag_features(df, pest, targets, max_year):
    """Lagged disease state. Only pest rows with Year <= max_year may be read.

    The survey has real gaps (no 2020 at all; Wales absent 2003-2019), so "lag 1"
    means the most recent *observed* prior year, not literally Year-1.
    """
    hist = pest[pest.Year <= max_year]
    nat = national_year_index(hist, targets)
    out = df.copy()
    for t in targets:
        s = hist[["Year", "Region", t]].dropna(subset=[t]).sort_values("Year")
        vals = np.full(len(out), np.nan)
        for reg, g in s.groupby("Region"):
            m = (out.Region == reg).to_numpy()
            if m.sum():
                vals[m] = _last_prior(g.Year.to_numpy(), g[t].to_numpy(),
                                      out.Year.to_numpy()[m])
        out[f"lag1_{t}"] = vals
        # national previous-year level: the shared epidemic regime
        ns = nat[t].dropna().sort_index()
        out[f"natlag1_{t}"] = _last_prior(ns.index.to_numpy(), ns.to_numpy(),
                                          out.Year.to_numpy())
    return out
