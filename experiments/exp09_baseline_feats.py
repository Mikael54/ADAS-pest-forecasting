"""Experiment 09: add "as-of" disease-baseline features to the weather model.

exp08's per-year diagnosis showed the weather model's biggest errors are a LEVEL
problem, not a shape problem: septoria incidence in the 2020s sits far above
what the 1971-2019 relationship implies (decade means 51.9 -> 75.6), and yellow
rust jumped from ~1 to ~26 in 2021/2025. That is pathogen-population and
varietal change, which weather cannot see.

Fix: give the model an explicit, strictly-backward-looking baseline level
(region and national exponentially-weighted means of the target over previously
*observed* years) and let ridge learn how much to trust it vs the weather
anomaly, instead of using a hand-set blend weight.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from features import build_weather_features

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

WF = build_weather_features()
PEST = load_pest()
OBS = PEST[PEST.Year <= 2025]
REGIONS = sorted(PEST.Region.unique())
EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]

SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]


def as_of_baselines(pest, targets, halflives=(4, 10)):
    """For every (Region, Year) row, EW means of the target over STRICTLY earlier
    observed years -- at region level and at national level. Backward-looking by
    construction, so it is valid at any forecast origin."""
    out = pest[["Year", "Region"]].copy()
    nat = pest.groupby("Year")[targets].mean()
    for t in targets:
        for hl in halflives:
            lam = 0.5 ** (1.0 / hl)
            # region level
            vals = np.full(len(pest), np.nan)
            for reg, g in pest.groupby("Region"):
                g = g.sort_values("Year")
                obs_y = g.Year[g[t].notna()].to_numpy()
                obs_v = g[t].dropna().to_numpy()
                for i in g.index:
                    y = pest.at[i, "Year"]
                    m = obs_y < y
                    if m.sum():
                        w = lam ** (y - obs_y[m])
                        vals[pest.index.get_loc(i)] = np.average(obs_v[m], weights=w)
            out[f"bl_reg{hl}_{t}"] = vals
            # national level
            ns = nat[t].dropna()
            nv = np.full(len(pest), np.nan)
            for j, y in enumerate(pest.Year.to_numpy()):
                m = ns.index.to_numpy() < y
                if m.sum():
                    w = lam ** (y - ns.index.to_numpy()[m])
                    nv[j] = np.average(ns.to_numpy()[m], weights=w)
            out[f"bl_nat{hl}_{t}"] = nv
    return out


print("building as-of baselines ...")
BL = as_of_baselines(OBS.reset_index(drop=True), TARGETS)
DF = OBS.reset_index(drop=True).merge(WF, on=["Year", "Region"], how="left") \
        .merge(BL, on=["Year", "Region"], how="left")

EVAL = [y for y in range(2005, 2026) if y != 2020]


def predict(target, T, cfg):
    fs = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if cfg.get("feats") == "all":
        fs = list(EPI_ANOM)
    if cfg.get("no_weather"):
        fs = []
    bl_cols = []
    if cfg.get("use_bl", True):
        bl_cols = [f"bl_reg4_{target}", f"bl_reg10_{target}",
                   f"bl_nat4_{target}", f"bl_nat10_{target}"]
    cols = fs + bl_cols

    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"]) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(te) == 0 or len(tr) < 30 or not cols:
        return None
    cy = sorted(tr.Year.unique())[-cfg["clim_k"]:]
    clim = tr[tr.Year.isin(cy)]
    nat_clim = clim[target].mean()
    reg_clim = clim.groupby("Region")[target].mean()
    base = (cfg["reg_w"] * te.Region.map(reg_clim).fillna(nat_clim).to_numpy()
            + (1 - cfg["reg_w"]) * nat_clim)

    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    keep = Xtr.notna().all(axis=1) & np.isfinite(Xtr).all(axis=1)
    Xtr, tr2 = Xtr[keep], tr[keep]
    Xte = Xte.fillna(0.0)
    if len(tr2) < 30:
        return None
    if cfg.get("use_region", True):
        for r in REGIONS[1:]:
            Xtr[f"R_{r}"] = (tr2.Region == r).astype(float)
            Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    y = tr2[target].to_numpy(float)
    w = 0.5 ** ((T - tr2.Year.to_numpy()) / cfg["halflife"]) if cfg["halflife"] else None
    m = Ridge(alpha=cfg["alpha"]).fit(sc.transform(Xtr), y, sample_weight=w)
    pv = m.predict(sc.transform(Xte))
    val = (1 - cfg["blend"]) * pv + cfg["blend"] * base
    return te[["Year", "Region"]].assign(
        target=target, value=np.clip(val, 0, 100 if target in INCIDENCE else None))


def run(cfg, years=EVAL, targets=TARGETS):
    out = [p for t in targets for T in years if (p := predict(t, T, cfg)) is not None]
    return pd.concat(out, ignore_index=True) if out else None


C = lambda **kw: dict(dict(alpha=10.0, halflife=None, min_year=1971, blend=0.0,
                           clim_k=12, reg_w=0.5, use_bl=True), **kw)

CONFIGS = {
    "D1 weather+baseline":            C(),
    "D2 weather+baseline hl=15":      C(halflife=15),
    "D3 weather+baseline 1990+":      C(min_year=1990),
    "D4 weather+baseline a=30":       C(alpha=30.0),
    "D5 weather+baseline a=3":        C(alpha=3.0),
    "D6 baseline ONLY (no weather)":  C(no_weather=True),
    "D7 weather only (no baseline)":  C(use_bl=False, blend=0.3),
    "D8 w+bl allfeat":                C(feats="all"),
    "D9 w+bl hl=15 blend.25":         C(halflife=15, blend=0.25),
    "D10 w+bl 1990+ a=30":            C(min_year=1990, alpha=30.0),
    "D11 w+bl 1990+ hl=15":           C(min_year=1990, halflife=15),
    "D12 w+bl noregion":              C(use_region=False),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    rows = []
    for name, cfg in CONFIGS.items():
        p = run(cfg)
        if p is None:
            continue
        _, a = score(p, truth, EVAL)
        _, m = score(p, truth, [y for y in range(2005, 2020)])
        _, r = score(p, truth, [2021, 2022, 2023, 2024, 2025])
        rows.append({"config": name,
                     "sept_05_25": a["septoria_pooled"], "sept_05_19": m["septoria_pooled"],
                     "sept_21_25": r["septoria_pooled"],
                     "rust_05_25": a["rust_pooled"], "rust_05_19": m["rust_pooled"],
                     "rust_21_25": r["rust_pooled"]})
    print("=" * 120)
    print("WITH AS-OF BASELINE FEATURES   (exp08 best: sept 14.60/13.36/17.55  |  rust clim 5.82/2.34/10.56)")
    print("=" * 120)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
