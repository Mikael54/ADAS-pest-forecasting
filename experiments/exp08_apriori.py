"""Experiment 08: a-priori configs + per-year error diagnosis.

exp07 showed that coordinate-descent tuning on 2011-2019 does NOT transfer to
2021-2025 -- the selection overfits a 9-year window. So here we compare a small
number of configs fixed in advance on epidemiological grounds, evaluate them
leave-one-year-out over a long window, and print per-year errors so failures are
visible rather than averaged away.
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
DF = OBS.merge(WF, on=["Year", "Region"], how="left")
EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]

SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]

EVAL = [y for y in range(2005, 2026) if y != 2020]


def _fwd(t, y):
    if t in SEVERITY:
        return np.log1p(np.clip(y, 0, None))
    p = np.clip(np.asarray(y, float) / 100, .005, .995)
    return np.log(p / (1 - p))


def _inv(t, z):
    if t in SEVERITY:
        return np.clip(np.expm1(np.clip(z, -20, 20)), 0, None)
    return 100 / (1 + np.exp(-np.clip(z, -12, 12)))


def predict(target, T, cfg):
    fs = SEPT_FEATS if target in SEPTORIA else RUST_FEATS
    if cfg.get("feats") == "all":
        fs = EPI_ANOM
    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"]) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(te) == 0 or len(tr) < 30:
        return None
    cy = sorted(tr.Year.unique())[-cfg["clim_k"]:]
    clim = tr[tr.Year.isin(cy)]
    nat_clim = clim[target].mean()
    reg_clim = clim.groupby("Region")[target].mean()
    base = (cfg["reg_w"] * te.Region.map(reg_clim).fillna(nat_clim).to_numpy()
            + (1 - cfg["reg_w"]) * nat_clim)
    if cfg["blend"] >= 1.0:
        return te[["Year", "Region"]].assign(target=target, value=base)

    Xtr, Xte = tr[fs].copy(), te[fs].copy()
    med = Xtr.median(); Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    z = _fwd(target, tr[target].to_numpy()) if cfg["transform"] else tr[target].to_numpy(float)
    w = 0.5 ** ((T - tr.Year.to_numpy()) / cfg["halflife"]) if cfg["halflife"] else None
    m = Ridge(alpha=cfg["alpha"]).fit(sc.transform(Xtr), z, sample_weight=w)
    pz = m.predict(sc.transform(Xte))
    pv = _inv(target, pz) if cfg["transform"] else pz
    val = (1 - cfg["blend"]) * pv + cfg["blend"] * base
    return te[["Year", "Region"]].assign(
        target=target, value=np.clip(val, 0, 100 if target in INCIDENCE else None))


def run(cfg, years=EVAL, targets=TARGETS):
    out = [p for t in targets for T in years if (p := predict(t, T, cfg)) is not None]
    return pd.concat(out, ignore_index=True)


C = lambda **kw: dict(dict(alpha=10.0, transform=False, halflife=None, min_year=1971,
                           blend=0.3, clim_k=12, reg_w=0.5), **kw)

CONFIGS = {
    "C0 climatology only":            C(blend=1.0),
    "C1 ridge blend.3":               C(blend=0.3),
    "C2 ridge blend.5":               C(blend=0.5),
    "C3 ridge blend.3 hl=15":         C(blend=0.3, halflife=15),
    "C4 ridge blend.5 hl=15":         C(blend=0.5, halflife=15),
    "C5 ridge blend.3 1990+":         C(blend=0.3, min_year=1990),
    "C6 ridge blend.5 hl=15 a=30":    C(blend=0.5, halflife=15, alpha=30.0),
    "C7 ridge blend.3 transform":     C(blend=0.3, transform=True),
    "C8 ridge blend.5 hl=25 clim20":  C(blend=0.5, halflife=25, clim_k=20),
    "C9 ridge blend.5 hl=15 allfeat": C(blend=0.5, halflife=15, feats="all"),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    rows = []
    store = {}
    for name, cfg in CONFIGS.items():
        p = run(cfg)
        store[name] = p
        _, s_all = score(p, truth, EVAL)
        _, s_rec = score(p, truth, [2021, 2022, 2023, 2024, 2025])
        _, s_mid = score(p, truth, [y for y in range(2005, 2020) if y != 2020])
        rows.append({"config": name,
                     "sept_05_25": s_all["septoria_pooled"], "sept_05_19": s_mid["septoria_pooled"],
                     "sept_21_25": s_rec["septoria_pooled"],
                     "rust_05_25": s_all["rust_pooled"], "rust_05_19": s_mid["rust_pooled"],
                     "rust_21_25": s_rec["rust_pooled"]})
    res = pd.DataFrame(rows)
    print("=" * 118)
    print("A-PRIORI CONFIGS, leave-one-year-out 2005-2025 (no tuning on these windows)")
    print("=" * 118)
    print(res.round(3).to_string(index=False))

    # --- per-year diagnosis of the national year level -------------------
    print("\n" + "=" * 118)
    print("PER-YEAR national mean: actual vs predicted (config C4), septoria targets")
    print("=" * 118)
    best = store["C4 ridge blend.5 hl=15"]
    for t in ["L1_Zymoseptoria_tritici_Crop_Incidence",
              "L2_Zymoseptoria_tritici_Crop_Incidence",
              "L1_Zymoseptoria_tritici_Disease_Severity",
              "L2_Yellow_rust_Crop_Incidence"]:
        a = OBS[OBS.Year.isin(EVAL)].groupby("Year")[t].mean()
        pp = best[best.target == t].groupby("Year").value.mean()
        cl = store["C0 climatology only"]
        cc = cl[cl.target == t].groupby("Year").value.mean()
        d = pd.DataFrame({"actual": a, "C4_pred": pp, "clim": cc}).dropna()
        d["err_C4"] = d.C4_pred - d.actual
        d["err_clim"] = d.clim - d.actual
        print(f"\n--- {t} ---")
        print(d.round(1).T.to_string())
