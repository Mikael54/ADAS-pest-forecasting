"""Experiment 19: latent-factor (reduced-rank) multi-task model.

A single factor explains 84 % of the variance in each disease's four targets
(L1/L2 x severity/incidence), and two explain 98 %. Fitting eight independent
ridges throws that away: each target's regression sees its own measurement
noise, when they are all noisy readings of one underlying "disease pressure".

Approach
    1. transform the 4 targets of a disease onto comparable scales
       (log1p severity, logit incidence) and standardise;
    2. SVD on the training region x year matrix -> k latent factors;
    3. regress the factor SCORES on weather + baselines -- a much better
       identified problem than 4 separate noisy regressions;
    4. reconstruct the 4 targets through the loadings and invert the transforms.

All of steps 1-3 are refitted inside each rolling-origin fold on training years
only, so the factor basis never sees the year being predicted.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from exp09_baseline_feats import as_of_baselines
from exp18_rolling_mech import build_features, E_ROLL, R_ROLL, M_SEPT, M_RUST

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

FEAT = build_features()
# guard against tiny trailing SDs producing absurd z-scores (June 2023 heat
# scored +18 SD because the trailing window held almost no comparable values)
for c in [c for c in FEAT.columns if c.endswith(("_anom", "_ranom"))]:
    FEAT[c] = FEAT[c].clip(-5, 5)

PEST = load_pest()
OBS = PEST[PEST.Year <= 2025].reset_index(drop=True)
REGIONS = sorted(OBS.Region.unique())
BL = as_of_baselines(PEST.reset_index(drop=True), TARGETS)
DF = (PEST.reset_index(drop=True)
          .merge(FEAT, on=["Year", "Region"], how="left")
          .merge(BL, on=["Year", "Region"], how="left"))
DF["trend"] = (DF.Year - 2000) / 25.0
EVAL = [y for y in range(2005, 2026) if y != 2020]


def fwd(t, y):
    y = np.asarray(y, float)
    if t in SEVERITY:
        return np.log1p(np.clip(y, 0, None))
    p = np.clip(y / 100, .005, .995)
    return np.log(p / (1 - p))


def inv(t, z):
    z = np.asarray(z, float)
    if t in SEVERITY:
        return np.clip(np.expm1(np.clip(z, -20, 20)), 0, None)
    return 100 / (1 + np.exp(-np.clip(z, -12, 12)))


def latent_predict(disease_targets, T, cfg):
    """Fit factor basis + factor regressions on Year<T, predict all regions at T."""
    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"])].copy()
    tr = tr.dropna(subset=disease_targets)
    te = DF[DF.Year == T].copy()
    if len(te) == 0 or len(tr) < 40:
        return None

    # 1. transform + standardise (constants from training only)
    Z = np.column_stack([fwd(t, tr[t].to_numpy()) for t in disease_targets])
    mu, sd = Z.mean(0), Z.std(0)
    sd[sd == 0] = 1.0
    Zs = (Z - mu) / sd

    # 2. factor basis
    U, S, Vt = np.linalg.svd(Zs, full_matrices=False)
    k = cfg["k"]
    V = Vt[:k]                       # k x 4 loadings
    scores = Zs @ V.T                # n x k factor scores

    # 3. regress each factor score on weather + baselines
    sept = disease_targets[0] in SEPTORIA
    fs = list(cfg["sept"] if sept else cfg["rust"])
    bl = [f"bl_nat4_{t}" for t in disease_targets] + \
         [f"bl_reg10_{t}" for t in disease_targets]
    cols = fs + bl + ["trend"]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    pred_scores = np.column_stack([
        Ridge(alpha=cfg["alpha"]).fit(Xtr_s, scores[:, j]).predict(Xte_s)
        for j in range(k)])

    # 4. reconstruct
    Zhat = pred_scores @ V * sd + mu
    out = []
    for j, t in enumerate(disease_targets):
        v = inv(t, Zhat[:, j])
        v = np.clip(v, 0, 100 if t in INCIDENCE else None)
        out.append(te[["Year", "Region"]].assign(target=t, value=v))
    return pd.concat(out, ignore_index=True)


def run_latent(cfg, years=EVAL):
    out = []
    for grp in (SEPTORIA, RUST):
        for T in years:
            p = latent_predict(grp, T, cfg)
            if p is not None:
                out.append(p)
    return pd.concat(out, ignore_index=True)


C = lambda **kw: dict(dict(k=2, alpha=100.0, min_year=1990,
                           sept=M_SEPT + E_ROLL, rust=M_RUST + R_ROLL), **kw)

CONFIGS = {
    "L1 latent k=1":              C(k=1),
    "L2 latent k=2":              C(k=2),
    "L3 latent k=3":              C(k=3),
    "L4 latent k=4 (full rank)":  C(k=4),
    "L5 latent k=2, stat only":   C(k=2, sept=E_ROLL, rust=R_ROLL),
    "L6 latent k=2, mech only":   C(k=2, sept=M_SEPT, rust=M_RUST),
    "L7 latent k=2, a=30":        C(k=2, alpha=30.0),
    "L8 latent k=2, 1971+":       C(k=2, min_year=1971),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    rows = []
    for name, cfg in CONFIGS.items():
        p = run_latent(cfg)
        rec = {"config": name}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(p, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        rows.append(rec)
    print("=" * 112)
    print("LATENT-FACTOR MULTI-TASK   (current best ensemble: sept 13.87 | rust 5.38)")
    print("                           (single-form 'add' reference:  sept 13.94 | rust 5.81)")
    print("=" * 112)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    best = run_latent(C(k=2))
    pt, _ = score(best, truth, EVAL)
    print("\nper-target RMSE, latent k=2 (2005-2025):")
    for t in TARGETS:
        print(f"  {t:<45}{pt[t]:>9.4f}")
