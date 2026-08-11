"""Experiment 10: agronomic/fungicide covariates, and model ensembling.

Two questions:
  (a) do the repo's husbandry covariates add anything over weather + baseline?
      NOTE agronomic_data.csv is entirely empty for 2021 and 2025 and has no
      2026 row at all, so even if it helps it cannot be used contemporaneously
      for the 2026 forecast -- only a lag-2 carry-forward.
  (b) does averaging several good configs beat the single best one? With a
      backtest this small, ensembling is usually the safer way to buy accuracy
      than picking a winner.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from features import build_weather_features
from exp09_baseline_feats import as_of_baselines

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

ROOT = ".."
WF = build_weather_features()
PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
REGIONS = sorted(PEST.Region.unique())
EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]

SEPT_FEATS = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
              "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
              "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
RUST_FEATS = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
              "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]


# ---- husbandry covariates -------------------------------------------------
def husbandry():
    a = pd.read_csv(f"{ROOT}/data/agronomic_data.csv")
    a["Region"] = a.Region.str.strip()
    sow = [c for c in a.columns if c.endswith("_ag_1") and "No sow" not in c]
    wk = {c: i for i, c in enumerate(sow)}          # 0 = mid-Sept ... 7 = early Nov
    tot = a[sow].sum(axis=1, min_count=1)
    f = a[["Year", "Region"]].copy()
    f["ag_sow_week"] = sum(a[c].fillna(0) * wk[c] for c in sow) / tot.replace(0, np.nan)
    f["ag_sow_early"] = a[sow[:3]].sum(axis=1, min_count=1) / tot.replace(0, np.nan)

    def pick(sfx, names):
        cols = [c for c in a.columns if c.endswith(sfx)]
        num = [c for c in cols if any(c.startswith(n) for n in names)]
        den = a[cols].sum(axis=1, min_count=1).replace(0, np.nan)
        return a[num].sum(axis=1, min_count=1) / den if num else np.nan

    f["ag_prev_wheat"] = pick("_ag_8_1", ["Wheat", "Cereals"])
    f["ag_plough"] = pick("_ag_5", ["Plough"])
    f["ag_straw_inc"] = pick("_ag_6", ["Chopped"])
    wf = [c for c in a.columns if c.endswith("_ag_11")]
    if wf:
        f["ag_wheat_freq"] = sum(a[c].fillna(0) * (i + 1) for i, c in enumerate(wf)) / \
            a[wf].sum(axis=1, min_count=1).replace(0, np.nan)

    g = pd.read_csv(f"{ROOT}/data/fungicide_data.csv")
    g["Region"] = g.Region.str.strip()
    g["fung_rate"] = pd.to_numeric(g.fungicide_kg, errors="coerce") / \
        pd.to_numeric(g.fungicide_area, errors="coerce")
    f = f.merge(g[["Year", "Region", "fung_rate"]], on=["Year", "Region"], how="outer")
    # carry forward within region (both sources are sparse / biennial)
    f = f.sort_values(["Region", "Year"])
    cols = [c for c in f.columns if c not in ("Year", "Region")]
    f[cols] = f.groupby("Region")[cols].ffill()
    return f


HUS = husbandry()
HUS_COLS = [c for c in HUS.columns if c not in ("Year", "Region")]
print("husbandry coverage by year (non-null cells):")
print(HUS[HUS.Year >= 2018].groupby("Year")[HUS_COLS].apply(
    lambda d: int(d.notna().sum().sum())).to_dict())

BL = as_of_baselines(OBS, TARGETS)
DF = (OBS.merge(WF, on=["Year", "Region"], how="left")
         .merge(BL, on=["Year", "Region"], how="left")
         .merge(HUS, on=["Year", "Region"], how="left"))

EVAL = [y for y in range(2005, 2026) if y != 2020]


def predict(target, T, cfg):
    fs = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if cfg.get("feats") == "all":
        fs = list(EPI_ANOM)
    if cfg.get("no_weather"):
        fs = []
    cols = list(fs)
    if cfg.get("use_bl", True):
        cols += [f"bl_reg4_{target}", f"bl_reg10_{target}",
                 f"bl_nat4_{target}", f"bl_nat10_{target}"]
    if cfg.get("use_hus"):
        cols += HUS_COLS
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
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    if cfg.get("use_region", True):
        for r in REGIONS[1:]:
            Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
            Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    y = tr[target].to_numpy(float)
    w = 0.5 ** ((T - tr.Year.to_numpy()) / cfg["halflife"]) if cfg["halflife"] else None
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

BASE = {
    "D10 w+bl 1990+ a=30":  C(min_year=1990, alpha=30.0),
    "D3  w+bl 1990+":       C(min_year=1990),
    "D2  w+bl hl=15":       C(halflife=15),
    "D6  baseline only":    C(no_weather=True),
    "D11 w+bl 1990+ hl=15": C(min_year=1990, halflife=15),
    "D9  w+bl hl15 bl.25":  C(halflife=15, blend=0.25),
}
HUSC = {
    "H1 D10 + husbandry":   C(min_year=1990, alpha=30.0, use_hus=True),
    "H2 D3  + husbandry":   C(min_year=1990, use_hus=True),
}


def summarize(name, p, truth, rows):
    _, a = score(p, truth, EVAL)
    _, m = score(p, truth, [y for y in range(2005, 2020)])
    _, r = score(p, truth, [2021, 2022, 2023, 2024, 2025])
    rows.append({"config": name,
                 "sept_05_25": a["septoria_pooled"], "sept_05_19": m["septoria_pooled"],
                 "sept_21_25": r["septoria_pooled"], "rust_05_25": a["rust_pooled"],
                 "rust_05_19": m["rust_pooled"], "rust_21_25": r["rust_pooled"]})


if __name__ == "__main__":
    truth = to_long(OBS)
    rows, store = [], {}
    for name, cfg in {**BASE, **HUSC}.items():
        p = run(cfg)
        store[name] = p
        summarize(name, p, truth, rows)

    # --- ensembles ---
    def ens(names, label):
        ps = [store[n] for n in names]
        m = ps[0][["Year", "Region", "target"]].copy()
        m["value"] = np.mean([p.sort_values(["Year", "Region", "target"])
                              .reset_index(drop=True).value.to_numpy() for p in
                              [q.sort_values(["Year", "Region", "target"]).reset_index(drop=True)
                               for q in ps]], axis=0)
        m = ps[0].sort_values(["Year", "Region", "target"]).reset_index(drop=True)[
            ["Year", "Region", "target"]].assign(
            value=np.mean([q.sort_values(["Year", "Region", "target"])
                           .reset_index(drop=True).value.to_numpy() for q in ps], axis=0))
        store[label] = m
        summarize(label, m, truth, rows)

    ens(["D10 w+bl 1990+ a=30", "D3  w+bl 1990+", "D2  w+bl hl=15"], "E1 avg(D10,D3,D2)")
    ens(["D10 w+bl 1990+ a=30", "D3  w+bl 1990+", "D2  w+bl hl=15",
         "D11 w+bl 1990+ hl=15"], "E2 avg(4 weather+bl)")
    ens(["D10 w+bl 1990+ a=30", "D6  baseline only"], "E3 avg(D10, baseline-only)")
    ens(["D10 w+bl 1990+ a=30", "D3  w+bl 1990+", "D2  w+bl hl=15",
         "D11 w+bl 1990+ hl=15", "D9  w+bl hl15 bl.25"], "E4 avg(5)")

    print("\n" + "=" * 120)
    print("COVARIATES + ENSEMBLES   (best so far: sept 14.250/12.953/17.304 | rust 5.669/2.689/10.010)")
    print("=" * 120)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
