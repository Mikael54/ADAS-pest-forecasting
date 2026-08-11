"""Experiment 21: monotone-constrained boosting, and severity|incidence decomposition.

ANGLE 1 -- monotone gradient boosting.
exp03 found plain GBM/RF no better than ridge: with ~450 rows they overfit and
cannot extrapolate. But MONOTONE CONSTRAINTS change that. Forcing septoria to
be non-decreasing in spring moisture and non-increasing in spring sunshine is
both a very strong regulariser and a statement of known epidemiology, and
unlike a linear fit it can represent the threshold/saturating shape exp06
found (septoria only collapses in the top sunshine quintile).

ANGLE 2 -- severity | incidence.
Severity is a mean over ALL crops including unaffected ones, so
    severity = (incidence/100) x conditional_severity_among_affected.
Conditional severity should be far more stable year to year than raw severity,
because the big swings come from how WIDELY the disease spread, not how bad it
got where it did. Modelling the two factors separately and multiplying may beat
modelling severity directly.
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

PAIR = {  # severity target -> its matching incidence target
    "L1_Zymoseptoria_tritici_Disease_Severity": "L1_Zymoseptoria_tritici_Crop_Incidence",
    "L2_Zymoseptoria_tritici_Disease_Severity": "L2_Zymoseptoria_tritici_Crop_Incidence",
    "L1_Yellow_rust_Disease_Severity": "L1_Yellow_rust_Crop_Incidence",
    "L2_Yellow_rust_Disease_Severity": "L2_Yellow_rust_Crop_Incidence",
}

# sign of the expected effect on disease: +1 increasing, -1 decreasing, 0 free
MONO_SEPT = {
    "m_liebig_spring_ranom": 1, "m_moist_limited_ranom": 1,
    "m_splash_spring_ranom": 1, "m_splash_L2_ranom": 1,
    "m_sun_per_rainday_ranom": -1, "m_sept_spring_ranom": 1,
    "e_rain_apr_may_ranom": 1, "e_sun_apr_may_ranom": -1,
    "e_rdays_latespring_ranom": 1, "e_frost_winter_ranom": -1,
}
MONO_RUST = {
    "m_frost_win_ranom": -1, "m_heat_kill_ranom": -1,
    "m_survive_x_cyc_ranom": 1, "m_rust_potential_ranom": 1,
    "e_tmin_winter_ranom": 1, "e_tmean_jun_ranom": -1,
}


def _cols(target, kind):
    sept = target in SEPTORIA
    if kind == "mono":
        w = list((MONO_SEPT if sept else MONO_RUST).keys())
    else:
        w = list(M_SEPT + E_ROLL) if sept else list(M_RUST + R_ROLL)
    return w


def fit_ridge(target, T, cols, y_override=None, min_year=1990):
    tr = DF[(DF.Year < T) & (DF.Year >= min_year)].copy()
    te = DF[DF.Year == T].copy()
    y = tr[target] if y_override is None else y_override.reindex(tr.index)
    ok = y.notna()
    tr, y = tr[ok], y[ok]
    if len(tr) < 30 or len(te) == 0:
        return None, None
    full = cols + [f"bl_reg4_{target}", f"bl_reg10_{target}",
                   f"bl_nat4_{target}", f"bl_nat10_{target}", "trend"]
    Xtr, Xte = tr[full].copy(), te[full].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=100.0).fit(sc.transform(Xtr), y.to_numpy(float))
    return m.predict(sc.transform(Xte)), te


def predict_mono(target, T, cfg):
    """LightGBM with monotone constraints encoding known epidemiology."""
    import lightgbm as lgb
    sept = target in SEPTORIA
    mono = MONO_SEPT if sept else MONO_RUST
    wcols = list(mono.keys())
    blc = [f"bl_reg4_{target}", f"bl_reg10_{target}",
           f"bl_nat4_{target}", f"bl_nat10_{target}"]
    cols = wcols + blc + ["trend"]
    constraints = [mono[c] for c in wcols] + [1, 1, 1, 1] + [0]

    tr = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"]) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(tr) < 30 or len(te) == 0:
        return None
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    m = lgb.LGBMRegressor(
        n_estimators=cfg["n"], learning_rate=cfg["lr"], num_leaves=cfg["leaves"],
        min_child_samples=cfg["mcs"], subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=cfg["l2"],
        monotone_constraints=constraints, verbose=-1, random_state=0)
    m.fit(Xtr, tr[target].to_numpy(float))
    v = np.clip(m.predict(Xte), 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=v)


def predict_cond(target, T):
    """severity = (incidence/100) x conditional severity among affected crops."""
    if target not in PAIR:
        return None
    inc_t = PAIR[target]
    # conditional severity, computed on observed rows only
    inc = DF[inc_t]
    cond = DF[target] / (inc / 100).replace(0, np.nan)
    cond = cond.where(inc > 2)          # unstable when almost nothing is infected
    cond = cond.clip(upper=cond.quantile(0.99))
    p_cond, te = fit_ridge(target, T, _cols(target, "full"), y_override=cond)
    p_inc, _ = fit_ridge(inc_t, T, _cols(inc_t, "full"))
    if p_cond is None or p_inc is None:
        return None
    v = np.clip(p_inc, 0, 100) / 100 * np.clip(p_cond, 0, None)
    v = np.clip(v, 0, None)
    return te[["Year", "Region"]].assign(target=target, value=v)


def predict_plain(target, T):
    p, te = fit_ridge(target, T, _cols(target, "full"))
    if p is None:
        return None
    return te[["Year", "Region"]].assign(
        target=target, value=np.clip(p, 0, 100 if target in INCIDENCE else None))


def run(fn, targets=TARGETS, years=EVAL):
    out = [p for t in targets for T in years if (p := fn(t, T)) is not None]
    return pd.concat(out, ignore_index=True) if out else None


if __name__ == "__main__":
    truth = to_long(OBS)

    # --- how stable IS conditional severity? ---
    print("=" * 100)
    print("Is conditional severity more stable than raw severity? (national CV over years)")
    print("=" * 100)
    nat = OBS.groupby("Year")[TARGETS].mean()
    for s, i in PAIR.items():
        c = nat[s] / (nat[i] / 100).replace(0, np.nan)
        c = c[nat[i] > 2]
        print(f"{s:<45} raw CV={nat[s].std()/nat[s].mean():.2f}"
              f"   conditional CV={c.std()/c.mean():.2f}  (n={c.notna().sum()})")

    MONO = dict(n=300, lr=0.03, leaves=7, mcs=20, l2=5.0, min_year=1990)
    runs = {
        "P  plain ridge (reference)": lambda: run(predict_plain),
        "M1 monotone LGBM":           lambda: run(lambda t, T: predict_mono(t, T, MONO)),
        "M2 monotone LGBM shallow":   lambda: run(lambda t, T: predict_mono(t, T, dict(MONO, leaves=3, n=400))),
        "M3 monotone LGBM strong-reg": lambda: run(lambda t, T: predict_mono(t, T, dict(MONO, l2=50.0, mcs=40))),
    }
    rows = []
    store = {}
    for name, f in runs.items():
        p = f()
        store[name] = p
        rec = {"config": name}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(p, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        rows.append(rec)
    print("\n" + "=" * 112)
    print("MONOTONE BOOSTING   (current best ensemble: sept 13.87 | rust 5.38)")
    print("=" * 112)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # --- conditional-severity decomposition, severity targets only ---
    print("\n" + "=" * 112)
    print("SEVERITY = INCIDENCE x CONDITIONAL SEVERITY (severity targets only)")
    print("=" * 112)
    p_cond = run(predict_cond, targets=list(PAIR))
    p_plain = run(predict_plain, targets=list(PAIR))
    for t in PAIR:
        r = []
        for p in (p_plain, p_cond):
            pp = p[p.target == t]
            tt = truth[(truth.target == t) & truth.Year.isin(EVAL)]
            m = tt.merge(pp, on=["Year", "Region", "target"],
                         suffixes=("_t", "_p")).dropna(subset=["value_t"])
            r.append(rmse(m.value_t, m.value_p))
        print(f"  {t:<45} direct={r[0]:8.4f}   decomposed={r[1]:8.4f}"
              f"   {'BETTER' if r[1] < r[0] else ''}")
