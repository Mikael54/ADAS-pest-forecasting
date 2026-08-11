"""Experiment 07: per-target tuned model with recency weighting.

Motivated by exp06:
  * septoria has real weather skill (corr 0.60-0.73 on the national year mean),
    yellow rust has NEGATIVE out-of-sample weather skill -> rust wants
    climatology, not a weather model;
  * the series are strongly NON-STATIONARY (rust L1 incidence: 22.0 in the
    1970s -> 0.8 in the 2000s -> 9.4 in the 2020s, driven by varietal
    resistance breakdown, not weather), so old years must be down-weighted.

Model selection is NESTED: hyper-parameters are chosen on 2011-2019 only, then
scored on 2021-2025 which the tuner never saw. That keeps the headline number
an honest estimate of the *procedure*, not of a lucky config.
"""
import itertools
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from features import build_weather_features

warnings.filterwarnings("ignore")
pd.set_option("display.width", 240)

WF = build_weather_features()
PEST = load_pest()
OBS = PEST[PEST.Year <= 2025]
REGIONS = sorted(PEST.Region.unique())
DF = OBS.merge(WF, on=["Year", "Region"], how="left")

EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]
SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
SEPT_WIDE = SEPT_FEATS + ["e_rain_autumn_anom", "e_rdays_spring_anom",
                          "e_tmin_winter_anom", "e_tmean_autumn_anom",
                          "e_warmwet_latespring_anom", "e_sun_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]

FEATSETS = {"sept": SEPT_FEATS, "sept_wide": SEPT_WIDE, "rust": RUST_FEATS,
            "all": EPI_ANOM, "none": []}

TUNE_YEARS = [y for y in range(2011, 2020)]
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def _fwd(t, y):
    if t in SEVERITY:
        return np.log1p(np.clip(y, 0, None))
    p = np.clip(np.asarray(y, float) / 100, .005, .995)
    return np.log(p / (1 - p))


def _inv(t, z):
    if t in SEVERITY:
        return np.clip(np.expm1(np.clip(z, -20, 20)), 0, None)
    return 100 / (1 + np.exp(-np.clip(z, -12, 12)))


def predict_target(target, T, cfg):
    """Rolling-origin prediction of `target` for every region in year T."""
    fs = FEATSETS[cfg["feats"]]
    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"])].copy()
    tr = tr[tr[target].notna()]
    te = DF[DF.Year == T].copy()
    if len(te) == 0 or len(tr) < 20:
        return None

    # climatology reference (recent years, per region + national)
    cy = sorted(tr.Year.unique())[-cfg["clim_k"]:]
    clim = tr[tr.Year.isin(cy)]
    nat_clim = clim[target].mean()
    reg_clim = clim.groupby("Region")[target].mean()
    base = (cfg["reg_w"] * te.Region.map(reg_clim).fillna(nat_clim).to_numpy()
            + (1 - cfg["reg_w"]) * nat_clim)

    if not fs or cfg["blend"] >= 1.0:
        return te[["Year", "Region"]].assign(target=target, value=base)

    cols = list(fs)
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    if cfg["use_region"]:
        for r in REGIONS[1:]:
            Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
            Xte[f"R_{r}"] = (te.Region == r).astype(float)
    if cfg["use_trend"]:
        Xtr["yr"] = (tr.Year - 2000) / 25.0
        Xte["yr"] = (te.Year - 2000) / 25.0
    ok = Xtr.notna().all(axis=1)
    Xtr, ytr, yr_tr = Xtr[ok], tr.loc[ok, target], tr.loc[ok, "Year"]
    if len(ytr) < 20:
        return te[["Year", "Region"]].assign(target=target, value=base)

    sc = StandardScaler().fit(Xtr)
    z = _fwd(target, ytr.to_numpy()) if cfg["transform"] else ytr.to_numpy(float)
    if cfg["halflife"]:
        w = 0.5 ** ((T - yr_tr.to_numpy()) / cfg["halflife"])
    else:
        w = None
    m = Ridge(alpha=cfg["alpha"]).fit(sc.transform(Xtr), z, sample_weight=w)
    pz = m.predict(sc.transform(Xte))
    pv = _inv(target, pz) if cfg["transform"] else pz
    val = (1 - cfg["blend"]) * pv + cfg["blend"] * base
    val = np.clip(val, 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=val)


def run_target(target, cfg, years):
    out = [p for T in years if (p := predict_target(target, T, cfg)) is not None]
    if not out:
        return None
    return pd.concat(out, ignore_index=True)


def eval_target(target, cfg, years):
    p = run_target(target, cfg, years)
    if p is None:
        return np.inf
    t = to_long(OBS[OBS.Year.isin(years)], [target])
    m = t.merge(p, on=["Year", "Region", "target"], suffixes=("_t", "_p")).dropna(subset=["value_t"])
    return rmse(m.value_t, m.value_p)


DEFAULT = dict(feats="sept", alpha=10.0, transform=False, halflife=None,
               min_year=1971, blend=0.3, clim_k=12, reg_w=0.5,
               use_region=True, use_trend=False)

GRID = dict(
    feats=["sept", "sept_wide", "rust", "all", "none"],
    alpha=[3.0, 10.0, 30.0, 100.0],
    transform=[False, True],
    halflife=[None, 25, 12, 6],
    min_year=[1971, 1990, 2000],
    blend=[0.0, 0.25, 0.5, 0.75, 1.0],
    clim_k=[6, 12, 20],
    reg_w=[0.0, 0.5, 1.0],
)


def coordinate_search(target, years, rounds=3):
    """Greedy coordinate descent over the grid -- a full product is too big."""
    cfg = dict(DEFAULT)
    cfg["feats"] = "rust" if target in RUST else "sept"
    best = eval_target(target, cfg, years)
    for _ in range(rounds):
        improved = False
        for k, opts in GRID.items():
            for v in opts:
                if cfg[k] == v:
                    continue
                trial = dict(cfg); trial[k] = v
                s = eval_target(target, trial, years)
                if s < best - 1e-9:
                    best, cfg, improved = s, trial, True
        if not improved:
            break
    return cfg, best


if __name__ == "__main__":
    print("=" * 108)
    print("NESTED TUNING: configs chosen on 2011-2019, scored on 2021-2025 (unseen by tuner)")
    print("=" * 108)
    chosen, rows = {}, []
    for t in TARGETS:
        cfg, tune_rmse = coordinate_search(t, TUNE_YEARS)
        test_rmse = eval_target(t, cfg, TEST_YEARS)
        base_cfg = dict(DEFAULT, feats="none", blend=1.0)
        base_tune = eval_target(t, base_cfg, TUNE_YEARS)
        base_test = eval_target(t, base_cfg, TEST_YEARS)
        chosen[t] = cfg
        rows.append({"target": t, "tune": tune_rmse, "test": test_rmse,
                     "clim_tune": base_tune, "clim_test": base_test,
                     "skill%": 100 * (1 - test_rmse / base_test)})
        print(f"\n{t}")
        print(f"  cfg: {cfg}")
        print(f"  tune(11-19) {tune_rmse:8.4f} vs clim {base_tune:8.4f}   |"
              f"  TEST(21-25) {test_rmse:8.4f} vs clim {base_test:8.4f}"
              f"   skill {100*(1-test_rmse/base_test):+5.1f}%")

    print("\n" + "=" * 108)
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # pooled scores on the untouched test window
    allp = pd.concat([run_target(t, chosen[t], TEST_YEARS) for t in TARGETS],
                     ignore_index=True)
    pt, sm = score(allp, to_long(OBS), TEST_YEARS)
    climp = pd.concat([run_target(t, dict(DEFAULT, feats="none", blend=1.0), TEST_YEARS)
                       for t in TARGETS], ignore_index=True)
    ptc, smc = score(climp, to_long(OBS), TEST_YEARS)
    print(f"\nPOOLED on 2021-2025 (honest, nested):")
    print(f"  septoria  tuned {sm['septoria_pooled']:7.3f}   climatology {smc['septoria_pooled']:7.3f}")
    print(f"  rust      tuned {sm['rust_pooled']:7.3f}   climatology {smc['rust_pooled']:7.3f}")
    import json
    with open("chosen_cfg.json", "w") as f:
        json.dump({k: v for k, v in chosen.items()}, f, indent=2)
    print("\nwrote chosen_cfg.json")
