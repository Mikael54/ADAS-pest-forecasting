"""Experiment 12: target transforms, non-linear weather terms, rust-specific sweep.

Standing on exp11's honest config (weather anomalies + as-of baselines + year
trend; husbandry excluded because it is unavailable for 2026). Remaining ideas:

  * transforms -- incidence is a bounded percentage that saturates near 100 in
    recent years, so a raw-scale linear fit predicts >100 and gets clipped;
  * non-linearity -- exp06's quintile table showed the septoria response to
    spring sunshine is flat then collapses in the top quintile, i.e. a drought
    THRESHOLD rather than a linear slope;
  * rust wants its own treatment: weather has negative out-of-sample skill for
    it, so the question is only how to shrink.
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
from exp10_covars_ensemble import SEPT_FEATS, RUST_FEATS

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

WF = build_weather_features()
PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
REGIONS = sorted(PEST.Region.unique())
BL = as_of_baselines(OBS, TARGETS)
DF = OBS.merge(WF, on=["Year", "Region"], how="left").merge(BL, on=["Year", "Region"], how="left")
DF["trend"] = (DF.Year - 2000) / 25.0

# --- non-linear weather terms (thresholds, not slopes) ---------------------
DF["nl_drought"] = np.clip(DF["e_sun_apr_may_anom"] - 1.0, 0, None)      # extreme bright/dry spring
DF["nl_verywet"] = np.clip(DF["e_rain_apr_may_anom"] - 0.75, 0, None)    # extreme wet spring
DF["nl_dry2"] = np.clip(-DF["e_rain_apr_may_anom"] - 0.75, 0, None)      # extreme dry spring
DF["nl_rain_sq"] = DF["e_rain_apr_may_anom"] ** 2
DF["nl_sun_sq"] = DF["e_sun_apr_may_anom"] ** 2
DF["nl_junheat"] = np.clip(DF["e_tmean_jun_anom"] - 1.0, 0, None)        # June heat kills rust
DF["nl_mildwin"] = np.clip(DF["e_frost_winter_anom"] * -1 - 0.5, 0, None)
NL_SEPT = ["nl_drought", "nl_verywet", "nl_dry2", "nl_rain_sq", "nl_sun_sq"]
NL_RUST = ["nl_junheat", "nl_mildwin", "nl_drought"]

EVAL = [y for y in range(2005, 2026) if y != 2020]


def _fwd(t, y, kind):
    y = np.asarray(y, float)
    if kind == "none":
        return y
    if t in SEVERITY:
        return np.log1p(np.clip(y, 0, None)) if kind == "log" else np.sqrt(np.clip(y, 0, None))
    if kind == "sqrt":
        return np.sqrt(np.clip(y, 0, None))
    p = np.clip(y / 100, .01, .99)
    return np.log(p / (1 - p))


def _inv(t, z, kind):
    z = np.asarray(z, float)
    if kind == "none":
        return z
    if t in SEVERITY:
        return np.clip(np.expm1(np.clip(z, -20, 20)), 0, None) if kind == "log" \
            else np.clip(z, 0, None) ** 2
    if kind == "sqrt":
        return np.clip(z, 0, None) ** 2
    return 100 / (1 + np.exp(-np.clip(z, -12, 12)))


def predict(target, T, cfg):
    fs = list(SEPT_FEATS if target in SEPTORIA else RUST_FEATS)
    if cfg.get("no_weather"):
        fs = []
    cols = list(fs)
    if cfg.get("nl"):
        cols += (NL_SEPT if target in SEPTORIA else NL_RUST)
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
    if cfg.get("clim_only"):
        return te[["Year", "Region"]].assign(target=target, value=base)

    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    kind = cfg.get("transform", "none")
    z = _fwd(target, tr[target].to_numpy(), kind)
    w = 0.5 ** ((T - tr.Year.to_numpy()) / cfg["halflife"]) if cfg["halflife"] else None
    m = Ridge(alpha=cfg["alpha"]).fit(sc.transform(Xtr), z, sample_weight=w)
    pv = _inv(target, m.predict(sc.transform(Xte)), kind)
    val = (1 - cfg["blend"]) * pv + cfg["blend"] * base
    return te[["Year", "Region"]].assign(
        target=target, value=np.clip(val, 0, 100 if target in INCIDENCE else None))


def run(cfg, targets=TARGETS):
    out = [p for t in targets for T in EVAL if (p := predict(t, T, cfg)) is not None]
    return pd.concat(out, ignore_index=True)


C = lambda **kw: dict(dict(alpha=100.0, halflife=None, min_year=1990, blend=0.0,
                           clim_k=12, reg_w=0.5, use_bl=True, extra=["trend"],
                           transform="none"), **kw)

CONFIGS = {
    "R6 baseline config (a=100)":  C(),
    "R6 a=30":                     C(alpha=30.0),
    "T-sqrt":                      C(transform="sqrt"),
    "T-log/logit":                 C(transform="log"),
    "NL non-linear terms":         C(nl=True),
    "NL + a=30":                   C(nl=True, alpha=30.0),
    "NL + sqrt":                   C(nl=True, transform="sqrt"),
    "NL + sqrt + a=30":            C(nl=True, transform="sqrt", alpha=30.0),
    "NL + reg_w=1":                C(nl=True, reg_w=1.0),
    "NL + hl=20":                  C(nl=True, halflife=20),
    "NL + 1971+":                  C(nl=True, min_year=1971),
    "NL + 2000+":                  C(nl=True, min_year=2000),
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
    print("TRANSFORMS + NON-LINEARITY   (exp11 best honest: sept 13.950/13.094/16.057 | rust 5.719-5.809)")
    print("=" * 120)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # ---- rust-specific: how best to shrink? --------------------------------
    print("\n" + "=" * 120)
    print("RUST-SPECIFIC SWEEP (pooled RMSE over the 4 yellow-rust targets)")
    print("=" * 120)
    rrows = []
    rust_cfgs = {
        "clim k=6  reg_w=.5":   C(clim_only=True, clim_k=6),
        "clim k=12 reg_w=.5":   C(clim_only=True, clim_k=12),
        "clim k=20 reg_w=.5":   C(clim_only=True, clim_k=20),
        "clim k=12 reg_w=0":    C(clim_only=True, clim_k=12, reg_w=0.0),
        "clim k=12 reg_w=1":    C(clim_only=True, clim_k=12, reg_w=1.0),
        "baseline-only ridge":  C(no_weather=True, extra=[]),
        "baseline+trend ridge": C(no_weather=True),
        "bl+trend a=300":       C(no_weather=True, alpha=300.0),
        "bl+trend hl=10":       C(no_weather=True, halflife=10),
        "weather+bl+trend":     C(),
        "weather+bl+NL":        C(nl=True),
        "w+bl blend .5":        C(blend=0.5),
        "w+bl blend .75":       C(blend=0.75),
        "bl+trend sqrt":        C(no_weather=True, transform="sqrt"),
    }
    for name, cfg in rust_cfgs.items():
        p = run(cfg, targets=RUST)
        _, a = score(p, truth, EVAL)
        _, m = score(p, truth, [y for y in range(2005, 2020)])
        _, r = score(p, truth, [2021, 2022, 2023, 2024, 2025])
        rrows.append({"rust config": name, "rust_05_25": a["rust_pooled"],
                      "rust_05_19": m["rust_pooled"], "rust_21_25": r["rust_pooled"]})
    print(pd.DataFrame(rrows).round(3).to_string(index=False))
