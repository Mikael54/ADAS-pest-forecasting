"""Experiment 11: is the husbandry gain real, or just a time-trend proxy?

exp10 found husbandry covariates cut septoria RMSE 14.25 -> 13.46. But those
variables (fungicide rate, % previous-crop wheat, sowing week) drift smoothly
over decades, so they could simply be standing in for "year". That matters a
lot for 2026, because:
  * agronomic_data.csv has NO 2026 row and is entirely empty for 2025, so the
    2026 forecast can only use a 2-year-stale carry-forward;
  * if a plain year trend does the same job, use the trend -- it is honest,
    always available, and cannot go stale.

Tests: (a) explicit year trend instead of husbandry, (b) husbandry lagged by 2
years, which is exactly what 2026 will have to use.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import TARGETS, INCIDENCE, SEPTORIA, load_pest, to_long, score
from features import build_weather_features
from exp09_baseline_feats import as_of_baselines
from exp10_covars_ensemble import husbandry, SEPT_FEATS, RUST_FEATS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

WF = build_weather_features()
PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
REGIONS = sorted(PEST.Region.unique())
EPI_ANOM = [c for c in WF.columns if c.startswith("e_") and c.endswith("_anom")]

HUS = husbandry()
HUS_COLS = [c for c in HUS.columns if c not in ("Year", "Region")]
HUS_L2 = HUS.copy()
HUS_L2["Year"] = HUS_L2["Year"] + 2          # value from Y-2 attached to year Y
HUS_L2 = HUS_L2.rename(columns={c: c + "_L2" for c in HUS_COLS})
HUS_L2_COLS = [c + "_L2" for c in HUS_COLS]

BL = as_of_baselines(OBS, TARGETS)
DF = (OBS.merge(WF, on=["Year", "Region"], how="left")
         .merge(BL, on=["Year", "Region"], how="left")
         .merge(HUS, on=["Year", "Region"], how="left")
         .merge(HUS_L2, on=["Year", "Region"], how="left"))
DF["trend"] = (DF.Year - 2000) / 25.0
DF["trend2"] = DF["trend"] ** 2

EVAL = [y for y in range(2005, 2026) if y != 2020]


def predict(target, T, cfg):
    fs = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if cfg.get("no_weather"):
        fs = []
    cols = list(fs)
    if cfg.get("use_bl", True):
        cols += [f"bl_reg4_{target}", f"bl_reg10_{target}",
                 f"bl_nat4_{target}", f"bl_nat10_{target}"]
    cols += cfg.get("extra", [])
    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"]) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(te) == 0 or len(tr) < 30 or not cols:
        return None
    cy = sorted(tr.Year.unique())[-cfg["clim_k"]:]
    clim = tr[tr.Year.isin(cy)]
    nat_clim, reg_clim = clim[target].mean(), clim.groupby("Region")[target].mean()
    base = (cfg["reg_w"] * te.Region.map(reg_clim).fillna(nat_clim).to_numpy()
            + (1 - cfg["reg_w"]) * nat_clim)
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    w = 0.5 ** ((T - tr.Year.to_numpy()) / cfg["halflife"]) if cfg["halflife"] else None
    m = Ridge(alpha=cfg["alpha"]).fit(sc.transform(Xtr), tr[target].to_numpy(float),
                                      sample_weight=w)
    val = (1 - cfg["blend"]) * m.predict(sc.transform(Xte)) + cfg["blend"] * base
    return te[["Year", "Region"]].assign(
        target=target, value=np.clip(val, 0, 100 if target in INCIDENCE else None))


def run(cfg):
    out = [p for t in TARGETS for T in EVAL if (p := predict(t, T, cfg)) is not None]
    return pd.concat(out, ignore_index=True)


C = lambda **kw: dict(dict(alpha=30.0, halflife=None, min_year=1990, blend=0.0,
                           clim_k=12, reg_w=0.5, use_bl=True, extra=[]), **kw)

CONFIGS = {
    "D10 w+bl (no extras)":        C(),
    "H1  + husbandry (contemp)":   C(extra=HUS_COLS),
    "H3  + husbandry LAG-2":       C(extra=HUS_L2_COLS),
    "R1  + year trend":            C(extra=["trend"]),
    "R2  + trend + trend^2":       C(extra=["trend", "trend2"]),
    "R3  + trend + husbandry":     C(extra=["trend"] + HUS_COLS),
    "R4  + trend + husbandryL2":   C(extra=["trend"] + HUS_L2_COLS),
    "R5  + fung_rate only":        C(extra=["fung_rate"]),
    "R6  + trend, a=100":          C(extra=["trend"], alpha=100.0),
    "R7  + trend, hl=15":          C(extra=["trend"], halflife=15),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    rows = []
    for name, cfg in CONFIGS.items():
        p = run(cfg)
        _, a = score(p, truth, EVAL)
        _, m = score(p, truth, [y for y in range(2005, 2020)])
        _, r = score(p, truth, [2021, 2022, 2023, 2024, 2025])
        rows.append({"config": name,
                     "sept_05_25": a["septoria_pooled"], "sept_05_19": m["septoria_pooled"],
                     "sept_21_25": r["septoria_pooled"], "rust_05_25": a["rust_pooled"],
                     "rust_05_19": m["rust_pooled"], "rust_21_25": r["rust_pooled"]})
    print("=" * 120)
    print("TREND vs HUSBANDRY  -- is the husbandry gain just a time trend?")
    print("=" * 120)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
