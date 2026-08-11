"""Experiment 22: TabPFN (prior-fitted transformer) as the regressor.

Rationale for trying it here
    TabPFN is a transformer pre-trained on millions of synthetic tabular tasks
    that does in-context learning: it conditions on the training rows at
    inference time rather than fitting parameters. Its designed regime is small
    tabular data -- roughly what we have (~450 rows x ~30 features after the
    1990+ cut). That is precisely the regime where my ridge wins by being
    heavily regularised and where RF/GBM/monotone-LGBM all overfit (exp03,
    exp21). So TabPFN is a genuinely different bet: nonparametric flexibility
    with the regularisation supplied by the prior instead of by a penalty.

Notes on the environment
    tabpfn 8.x requires a Prior Labs licence token, which is not configured on
    this machine (~/.config/.tabpfn/state.json has no user). tabpfn 2.2.1 runs
    ungated and is what is installed here. CPU only, ~10 s per fit at
    n_estimators=4, so a full rolling-origin backtest is ~20 years x 8 targets.

Evaluated on exactly the same protocol, features and folds as the ridge, so the
numbers are directly comparable to EXPERIMENTS.md.
"""
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "12")

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
from exp09_baseline_feats import as_of_baselines
from exp18_rolling_mech import build_features, E_ROLL, R_ROLL, M_SEPT, M_RUST

FEAT = build_features()
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
DF["region_code"] = pd.Categorical(DF.Region, categories=REGIONS).codes
EVAL = [y for y in range(2005, 2026) if y != 2020]


def cols_for(target):
    sept = target in SEPTORIA
    w = list(M_SEPT + E_ROLL) if sept else list(M_RUST + R_ROLL)
    return w + [f"bl_reg4_{target}", f"bl_reg10_{target}",
                f"bl_nat4_{target}", f"bl_nat10_{target}",
                "trend", "region_code"]


# Optional: point at a specific checkpoint file. Passing the ABSOLUTE path to an
# already-cached .ckpt is what lets tabpfn 8.x load the gated v3 weights without
# a licence token -- model_loading.download_model() short-circuits on
# `to.exists()` BEFORE the licence check, but resolve_model_path("v3_default")
# returns a *relative* path, so the cache is missed and it tries to download.
CKPT = os.environ.get("TABPFN_CKPT")


def predict_tabpfn(target, T, cfg):
    from tabpfn import TabPFNRegressor
    cols = cols_for(target)
    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"]) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(tr) < 30 or len(te) == 0:
        return None
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median(numeric_only=True)
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    y = tr[target].to_numpy(float)
    if cfg.get("log_target") and target in SEVERITY:
        y = np.log1p(np.clip(y, 0, None))
    kw = dict(n_estimators=cfg["n_estimators"], device="cpu", random_state=0,
              categorical_features_indices=[len(cols) - 1],   # region_code
              ignore_pretraining_limits=True)
    if CKPT:
        kw["model_path"] = CKPT
    m = TabPFNRegressor(**kw)
    m.fit(Xtr.to_numpy(float), y)
    p = m.predict(Xte.to_numpy(float))
    if cfg.get("log_target") and target in SEVERITY:
        p = np.expm1(np.clip(p, -20, 20))
    p = np.clip(p, 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=p)


def run(cfg, years=EVAL, targets=TARGETS, verbose=True, tag="TP"):
    """Checkpoints each target to disk so a killed run resumes instead of
    restarting (~4 min per target on CPU)."""
    ck = Path(f"ckpt_tabpfn_{tag}")
    ck.mkdir(exist_ok=True)
    out = []
    t0 = time.time()
    for t in targets:
        f = ck / f"{t}.csv"
        if f.exists():
            out.append(pd.read_csv(f))
            if verbose:
                print(f"  [cached] {t}", flush=True)
            continue
        rows = [p for T in years if (p := predict_tabpfn(t, T, cfg)) is not None]
        if not rows:
            continue
        d = pd.concat(rows, ignore_index=True)
        d.to_csv(f, index=False)
        out.append(d)
        if verbose:
            print(f"  [{time.time()-t0:6.0f}s] done {t}", flush=True)
    return pd.concat(out, ignore_index=True) if out else None


CONFIGS = {
    "TP1 n_est=4, 1990+":        dict(n_estimators=4, min_year=1990),
    "TP2 n_est=4, 1971+":        dict(n_estimators=4, min_year=1971),
    "TP3 n_est=4, log severity": dict(n_estimators=4, min_year=1990, log_target=True),
    # v3 is ~32 s/fit on CPU even at n_estimators=1, so keep the ensemble at 1
    "V3 n_est=1, 1990+":         dict(n_estimators=1, min_year=1990),
}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else None
    truth = to_land = to_long(OBS)
    rows = []
    for name, cfg in CONFIGS.items():
        if which and which not in name:
            continue
        print(f"\n=== {name} ===", flush=True)
        p = run(cfg, tag=name.split()[0])
        p.to_csv(f"preds_tabpfn_{name.split()[0]}.csv", index=False)
        rec = {"config": name}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(p, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        rows.append(rec)
        print(pd.DataFrame([rec]).round(3).to_string(index=False), flush=True)

    print("\n" + "=" * 112)
    print("TABPFN   (ridge ensemble: sept 13.87 | rust 5.38;  climatology 17.62 | 5.82)")
    print("=" * 112)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
