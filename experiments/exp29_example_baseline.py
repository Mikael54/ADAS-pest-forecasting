"""Experiment 29: head-to-head against the example pipeline shipped in the repo.

The contest repo ships a worked example (report/example_report.md, by Alex
Rabeau): per-region ElasticNet on lagged disease values plus the four repo
covariate files, with Kalman imputation and skewness correction. Every entry is
implicitly measured against it, so it is worth knowing the actual gap rather
than assuming one.

FAITHFUL REPLICATION
Reproduced as written, including the choices I would not make:
  * a SEPARATE model per region (9 models, ~40 training rows each);
  * target is `_lead1` -- year t features predict year t+1, so no in-season
    information is used at all;
  * predictors are the disease values themselves at t, t-1, t-2, plus
    agronomic / bioclim / fungicide / land-use columns;
  * ElasticNetCV(l1_ratio=0.5) with TimeSeriesSplit(5);
  * Kalman smoothing to impute covariates, log1p/sqrt/square skew correction.

Two deviations, both documented and neither favourable to my model:
  * `pykalman` is unavailable on some installs; the fallback is ffill/bfill,
    which is what the example's own try/except effectively degrades to.
  * the example never maps the -9999 dummies to NaN. Verified inert: -9999
    occurs ONLY in 2026 rows, which never enter a backtest fit.

A REAL DIFFERENCE IN INFORMATION, stated plainly
The example uses nothing from the forecast year. Mine uses observed weather from
September(Y-1) through June(Y), which for 2026 is already published. That is a
genuine advantage and it is most of the point -- but it means the comparison is
"two different framings of the problem", not "same inputs, better regressor". So
a third variant is scored: my model with the weather block removed, leaving only
as-of baselines, trend and region. That isolates how much of the gap is the
extra data and how much is the modelling.

PROTOCOLS
  A. the example's own split: train <= 2020, test 2021-2024.
  B. rolling origin on my eval years, so numbers are directly comparable to
     EXPERIMENTS.md. To predict year T the example trains on rows with
     Year <= T-2 (whose lead1 targets are <= T-1) and applies the fit to the
     Year = T-1 row. No leakage.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, INCIDENCE, load_pest, to_long, score, rmse
import final_model_v2 as FM

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_LAGS = 2
EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]

try:
    from pykalman import KalmanFilter
    HAVE_KF = True
except ImportError:
    HAVE_KF = False


def kalman_impute(series):
    """The example's imputer, verbatim where pykalman is available."""
    s = series.replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() < 3:
        return s
    filled = s.ffill().bfill()
    values = filled.values.astype(float)
    mask = np.isnan(s.values.astype(float))
    if not HAVE_KF:
        return pd.Series(values, index=s.index)
    kf = KalmanFilter(initial_state_mean=values[0])
    try:
        smoothed, _ = kf.em(values, n_iter=5).smooth(values)
        values[mask] = smoothed.flatten()[mask]
    except Exception:
        values[mask] = np.nanmean(values)
    return pd.Series(values, index=s.index)


def build_example_frame():
    """Sections 2-5 of the example report."""
    pest = pd.read_csv(DATA / "pest_data.csv")
    pest.columns = [c.strip().lstrip("﻿") for c in pest.columns]
    lagged = pest.copy()
    lagged["Year"] = lagged["Year"].astype(int)
    lagged = lagged.sort_values(["Region", "Year"])
    for col in TARGETS:
        lagged[f"{col}_lead1"] = lagged.groupby("Region")[col].shift(-1)
        for lag in range(1, N_LAGS + 1):
            lagged[f"{col}_lag{lag}"] = lagged.groupby("Region")[col].shift(lag)

    df = lagged.copy()
    for f in ["agronomic_data.csv", "bioclim_data.csv", "fungicide_data.csv",
              "prop_LUC.csv"]:
        d = pd.read_csv(DATA / f)
        d.columns = [c.strip().lstrip("﻿") for c in d.columns]
        on = ["Year", "Region"] if "Region" in d.columns else ["Year"]
        df = df.merge(d, on=on, how="left")
    for c in df.columns:
        if c != "Region":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    disease_cols = [c for c in df.columns if "L1" in c or "L2" in c]
    for c in df.columns:
        if c not in ["Year", "Region"] and c not in disease_cols:
            df[c] = kalman_impute(df[c])
    df = df.dropna()

    preds = [c for c in df.columns
             if c not in ["Year", "Region"] + [f"{t}_lead1" for t in TARGETS]]
    for c in preds:                                   # skewness correction
        x = df[c]
        if x.nunique() < 3:
            continue
        sk = skew(x, nan_policy="omit")
        if sk > 1:
            df[c] = np.log1p(x) if (x > 0).all() else np.sqrt(x - x.min() + 1)
        elif sk < -1:
            df[c] = x ** 2
    return df, preds


EX, PREDICTORS = build_example_frame()
print(f"example frame: {EX.shape}, years {EX.Year.min()}-{EX.Year.max()}, "
      f"{len(PREDICTORS)} predictors, pykalman={HAVE_KF}", flush=True)


def example_predict(target_years, train_end_fn):
    """Per-region ElasticNetCV, exactly as in the example report."""
    out = []
    for reg, dreg in EX.groupby("Region"):
        for T in target_years:
            tr_end = train_end_fn(T)
            train = dreg[dreg.Year <= tr_end]
            test = dreg[dreg.Year == T - 1]          # its lead1 target IS year T
            if len(train) < 8 or len(test) == 0:
                continue
            sc = StandardScaler()
            Xtr = sc.fit_transform(train[PREDICTORS])
            Xte = sc.transform(test[PREDICTORS])
            tscv = TimeSeriesSplit(n_splits=min(5, max(2, len(train) // 3)))
            for t in TARGETS:
                ytr = train[f"{t}_lead1"]
                ok = ytr.notna().to_numpy()
                if ok.sum() < 6:
                    continue
                try:
                    m = ElasticNetCV(l1_ratio=0.5, cv=tscv, random_state=123)
                    m.fit(Xtr[ok], ytr[ok])
                    p = m.predict(Xte)
                except Exception:
                    continue
                out.append(pd.DataFrame({"Year": T, "Region": reg, "target": t,
                                         "value": np.clip(p, 0, 100 if t in INCIDENCE
                                                          else None)}))
    return pd.concat(out, ignore_index=True) if out else None


def common_cells(*preds):
    """Intersection of (Year, Region, target) cells covered by every model.

    ⚠️ THIS IS NOT COSMETIC. The example's `lead1` framing needs a Year=2020 row
    to forecast 2021, and 2020 does not exist in the survey (no COVID-year
    fieldwork), so the example silently predicts NOTHING for 2021. Since
    `score()` inner-joins, that would quietly grade the example on an easier
    subset -- and 2021 is the single worst year for both models (rust L2
    incidence 25.9 after years near 6; septoria under-predicted by 38.9).
    Scoring on the intersection is the only way to compare like with like.
    """
    idx = None
    for p in preds:
        k = set(map(tuple, p[["Year", "Region", "target"]].to_numpy()))
        idx = k if idx is None else (idx & k)
    return idx


def restrict(p, idx):
    k = list(map(tuple, p[["Year", "Region", "target"]].to_numpy()))
    return p[[t in idx for t in k]].reset_index(drop=True)


def report(name, p, truth, rows):
    rec = {"model": name}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    rows.append(rec)
    return rec


if __name__ == "__main__":
    truth = to_long(FM.OBS)

    # ---- Protocol A: the example's own train/test split -------------------
    print("\n" + "=" * 104)
    print("PROTOCOL A -- the example's OWN split: train <= 2020, test 2021-2024")
    print("=" * 104)
    yrs_a = [2021, 2022, 2023, 2024]
    ex_a = example_predict(yrs_a, lambda T: 2020)
    fm_a = FM.run(yrs_a)
    cl_a = FM.run(yrs_a, lambda t, T: (lambda r: None if r[0] is None else
                  r[1][["Year", "Region"]].assign(target=t, value=r[0]))(
                      FM._fit(t, T, "clim", "fix")))
    idx_a = common_cells(ex_a, cl_a, fm_a)
    print(f"scored on {len(idx_a)} common cells; example covers "
          f"{sorted(ex_a.Year.unique())}, mine covers {sorted(fm_a.Year.unique())}")
    print(f"{'model':<34}{'septoria':>12}{'yellow rust':>14}")
    for nm, p in [("repo example (ElasticNet)", ex_a), ("climatology", cl_a),
                  ("mine (final_model_v2)", fm_a)]:
        _, s = score(restrict(p, idx_a), truth, yrs_a)
        print(f"{nm:<34}{s['septoria_pooled']:>12.3f}{s['rust_pooled']:>14.3f}")

    # ---- Protocol B: rolling origin, comparable to EXPERIMENTS.md ---------
    print("\n" + "=" * 104)
    print("PROTOCOL B -- rolling origin, 2005-2025 (directly comparable to EXPERIMENTS.md)")
    print("=" * 104)
    ex_b = example_predict(EVAL, lambda T: T - 2)
    fm_b = FM.run(EVAL)
    cl_b = FM.run(EVAL, lambda t, T: (lambda r: None if r[0] is None else
                  r[1][["Year", "Region"]].assign(target=t, value=r[0]))(
                      FM._fit(t, T, "clim", "fix")))

    # ablation: my architecture stripped of all weather, to separate data from method
    W0 = FM.weather_cols
    FM.weather_cols = lambda target, basis: []
    try:
        fm_nw = FM.run(EVAL)
    finally:
        FM.weather_cols = W0

    idx = common_cells(ex_b, cl_b, fm_nw, fm_b)
    ex_b, cl_b, fm_nw, fm_b = (restrict(p, idx) for p in (ex_b, cl_b, fm_nw, fm_b))
    miss = sorted({y for y, _, _ in
                   (set(map(tuple, FM.run(EVAL)[["Year", "Region", "target"]]
                            .to_numpy())) - idx)})
    print(f"scored on {len(idx)} cells common to all models; "
          f"years the example cannot cover: {miss}\n")

    rows = []
    for nm, p in [("repo example (ElasticNet)", ex_b), ("climatology", cl_b),
                  ("mine, NO weather (ablation)", fm_nw), ("mine (final_model_v2)", fm_b)]:
        report(nm, p, truth, rows)
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))

    ex_s = df.loc[0, "sept_05_25"]
    ex_r = df.loc[0, "rust_05_25"]
    mn_s = df.loc[3, "sept_05_25"]
    mn_r = df.loc[3, "rust_05_25"]
    print(f"\nvs the repo example:  septoria {ex_s:.2f} -> {mn_s:.2f} "
          f"({100*(1-mn_s/ex_s):+.1f}%)   rust {ex_r:.2f} -> {mn_r:.2f} "
          f"({100*(1-mn_r/ex_r):+.1f}%)")

    print("\nper-target RMSE, 2005-2025:")
    pe, _ = score(ex_b, truth, EVAL)
    pm, _ = score(fm_b, truth, EVAL)
    pc, _ = score(cl_b, truth, EVAL)
    print(f"{'target':<45}{'example':>10}{'clim':>10}{'mine':>10}{'vs example':>12}")
    for t in TARGETS:
        print(f"{t:<45}{pe[t]:>10.3f}{pc[t]:>10.3f}{pm[t]:>10.3f}"
              f"{100*(1-pm[t]/pe[t]):>11.1f}%")
