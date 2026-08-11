"""Shared data loading + rolling-origin backtest harness for the ADAS pest forecasting contest.

Evaluation protocol
-------------------
Rolling origin ("expanding window"): for each evaluation year T, a model may only
use information from years < T (plus, optionally, exogenous weather observed
*during* year T up to the survey date -- see FEATURE_MODES in features.py).
It then predicts all regions present in year T.

This exactly mimics the 2026 task: predict 9 regions for one unseen future year.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

TARGETS = [
    "L1_Zymoseptoria_tritici_Disease_Severity",
    "L1_Yellow_rust_Disease_Severity",
    "L1_Zymoseptoria_tritici_Crop_Incidence",
    "L1_Yellow_rust_Crop_Incidence",
    "L2_Zymoseptoria_tritici_Disease_Severity",
    "L2_Yellow_rust_Disease_Severity",
    "L2_Zymoseptoria_tritici_Crop_Incidence",
    "L2_Yellow_rust_Crop_Incidence",
]
SEPTORIA = [t for t in TARGETS if "Zymoseptoria" in t]
RUST = [t for t in TARGETS if "Yellow_rust" in t]
INCIDENCE = [t for t in TARGETS if t.endswith("Crop_Incidence")]
SEVERITY = [t for t in TARGETS if t.endswith("Disease_Severity")]

# Years used for model selection. 2020 has no survey (COVID).
EVAL_YEARS = [y for y in range(2011, 2026) if y != 2020]
# Wales only re-enters the survey from 2021; these are the years most like 2026.
RECENT_EVAL_YEARS = [2021, 2022, 2023, 2024, 2025]

FORECAST_YEAR = 2026


def load_pest():
    """Targets. -9999 dummy -> NaN. 2026 row exists with all-NaN targets."""
    df = pd.read_csv(DATA / "pest_data.csv")
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df[TARGETS] = df[TARGETS].replace(-9999.0, np.nan)
    df["Year"] = df["Year"].astype(int)
    df["Region"] = df["Region"].str.strip()
    return df.sort_values(["Region", "Year"]).reset_index(drop=True)


def load_repo_covariates():
    """Agronomic / bioclim / fungicide / land-use, on the (Year, Region) grid."""
    agro = pd.read_csv(DATA / "agronomic_data.csv")
    bio = pd.read_csv(DATA / "bioclim_data.csv")
    fung = pd.read_csv(DATA / "fungicide_data.csv")
    luc = pd.read_csv(DATA / "prop_LUC.csv")
    out = {}
    for name, df in [("agro", agro), ("bio", bio), ("fung", fung), ("luc", luc)]:
        df = df.copy()
        df["Year"] = df["Year"].astype(int)
        df["Region"] = df["Region"].astype(str).str.strip()
        # prefix columns so provenance is traceable
        val_cols = [c for c in df.columns if c not in ("Year", "Region")]
        df = df.rename(columns={c: f"{name}__{c}" for c in val_cols})
        out[name] = df
    return out


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    m = ~np.isnan(y_true)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def score(preds: pd.DataFrame, truth: pd.DataFrame, eval_years=None):
    """preds/truth: long frames with columns Year, Region, target, value.

    Returns (per_target_series, summary_dict).
    """
    if eval_years is not None:
        preds = preds[preds.Year.isin(eval_years)]
        truth = truth[truth.Year.isin(eval_years)]
    m = truth.merge(
        preds, on=["Year", "Region", "target"], suffixes=("_true", "_pred"), how="inner"
    ).dropna(subset=["value_true"])
    per_target = m.groupby("target").apply(
        lambda g: rmse(g.value_true, g.value_pred), include_groups=False
    )
    per_target = per_target.reindex(TARGETS)

    def pooled(cols):
        s = m[m.target.isin(cols)]
        return rmse(s.value_true, s.value_pred)

    summary = {
        "septoria_pooled": pooled(SEPTORIA),
        "rust_pooled": pooled(RUST),
        "all_pooled": pooled(TARGETS),
        "sept_sev": pooled([c for c in SEPTORIA if c in SEVERITY]),
        "sept_inc": pooled([c for c in SEPTORIA if c in INCIDENCE]),
        "rust_sev": pooled([c for c in RUST if c in SEVERITY]),
        "rust_inc": pooled([c for c in RUST if c in INCIDENCE]),
        "n_obs": int(len(m)),
    }
    return per_target, summary


def to_long(df, targets=TARGETS):
    return df.melt(
        id_vars=["Year", "Region"], value_vars=targets,
        var_name="target", value_name="value",
    )


def backtest(predict_fn, eval_years=EVAL_YEARS, verbose=False):
    """predict_fn(train_df, test_index) -> long DataFrame of predictions.

    train_df: pest rows with Year < T (targets observed).
    test_index: DataFrame[Year, Region] rows to predict for year T.
    """
    pest = load_pest()
    obs = pest[pest.Year <= 2025]
    all_preds = []
    for T in eval_years:
        train = obs[obs.Year < T].copy()
        test_rows = obs[obs.Year == T][["Year", "Region"]].copy()
        if len(test_rows) == 0:
            continue
        p = predict_fn(train, test_rows)
        all_preds.append(p)
        if verbose:
            pt, sm = score(p, to_long(obs[obs.Year == T]))
            print(f"  {T}: sept={sm['septoria_pooled']:.3f} rust={sm['rust_pooled']:.3f}")
    preds = pd.concat(all_preds, ignore_index=True)
    truth = to_long(obs)
    return preds, truth


def report(name, preds, truth, eval_years=EVAL_YEARS, recent=RECENT_EVAL_YEARS):
    pt, sm = score(preds, truth, eval_years)
    pt_r, sm_r = score(preds, truth, recent)
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"{'target':<45}{'RMSE all':>12}{'RMSE 21-25':>13}")
    for t in TARGETS:
        print(f"{t:<45}{pt.get(t, np.nan):>12.4f}{pt_r.get(t, np.nan):>13.4f}")
    print("-" * 78)
    print(f"{'SEPTORIA pooled':<45}{sm['septoria_pooled']:>12.4f}{sm_r['septoria_pooled']:>13.4f}")
    print(f"{'YELLOW RUST pooled':<45}{sm['rust_pooled']:>12.4f}{sm_r['rust_pooled']:>13.4f}")
    print(f"{'  septoria severity':<45}{sm['sept_sev']:>12.4f}{sm_r['sept_sev']:>13.4f}")
    print(f"{'  septoria incidence':<45}{sm['sept_inc']:>12.4f}{sm_r['sept_inc']:>13.4f}")
    print(f"{'  rust severity':<45}{sm['rust_sev']:>12.4f}{sm_r['rust_sev']:>13.4f}")
    print(f"{'  rust incidence':<45}{sm['rust_inc']:>12.4f}{sm_r['rust_inc']:>13.4f}")
    print(f"{'n obs':<45}{sm['n_obs']:>12d}{sm_r['n_obs']:>13d}")
    return {"name": name, "all": sm, "recent": sm_r, "per_target_all": pt, "per_target_recent": pt_r}
