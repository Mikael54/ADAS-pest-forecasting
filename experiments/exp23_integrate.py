"""Experiment 23: fold this round's confirmed wins into the ensemble.

Confirmed in exp18-exp21:
  * ROLLING (trailing 30-year) climatology baseline for weather anomalies beats
    a fixed 1961-2000 baseline -- small but consistent, and it removes the
    warming trend that was contaminating the mechanistic degree-day terms.
  * MECHANISTIC features (degree-day cycles x splash events, Liebig limiting
    factor) are better on 2021-2025 and worse on 2005-2019 than statistical
    aggregates -- complementary, so worth ensembling rather than choosing.
  * CONDITIONAL-SEVERITY decomposition, severity = (incidence/100) x
    conditional severity, improved ALL FOUR severity targets. Conditional
    severity is far more stable than raw severity (rust CV 3.35 -> 0.99),
    because the year-to-year swings come from how WIDELY the disease spread,
    not how bad it got where it did.

Rejected: analogue/k-NN forecasting (exp20), monotone-constrained LGBM (exp21),
latent-factor reduced rank (exp19, helps recent years only, hurts rust).

This measures each increment against the current best (sept 13.87 / rust 5.38).
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

E_FIX = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
         "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
         "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
R_FIX = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
         "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]

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
# Rust's threshold terms stay on the FIXED-baseline anomalies. Rebasing them on
# the trailing climatology costs ~0.4 RMSE on rust (5.38 -> 5.78): a rolling
# window keeps re-centring on the recent mild winters, so "frost-free winter"
# stops being flagged as unusual exactly when it matters most.
DF["nl_drought"] = np.clip(DF["e_sun_apr_may_anom"] - 1.0, 0, None)
DF["nl_dry2"] = np.clip(-DF["e_rain_apr_may_anom"] - 0.75, 0, None)
DF["nl_junheat"] = np.clip(DF["e_tmean_jun_anom"] - 1.0, 0, None)
DF["nl_mildwin"] = np.clip(-DF["e_frost_winter_anom"] - 0.5, 0, None)
DF["nl_drought_r"] = np.clip(DF["e_sun_apr_may_ranom"] - 1.0, 0, None)
DF["nl_dry2_r"] = np.clip(-DF["e_rain_apr_may_ranom"] - 0.75, 0, None)
NL_RUST = ["nl_junheat", "nl_mildwin", "nl_drought"]

PAIR = {"L1_Zymoseptoria_tritici_Disease_Severity": "L1_Zymoseptoria_tritici_Crop_Incidence",
        "L2_Zymoseptoria_tritici_Disease_Severity": "L2_Zymoseptoria_tritici_Crop_Incidence",
        "L1_Yellow_rust_Disease_Severity": "L1_Yellow_rust_Crop_Incidence",
        "L2_Yellow_rust_Disease_Severity": "L2_Yellow_rust_Crop_Incidence"}
EVAL = [y for y in range(2005, 2026) if y != 2020]
ALPHA, MIN_YEAR, CLIM_K = 100.0, 1990, 12


def wcols(target, featset):
    sept = target in SEPTORIA
    if featset == "fix":
        base = E_FIX if sept else R_FIX
    elif featset == "roll":
        base = E_ROLL if sept else R_ROLL
    elif featset == "mech":
        base = M_SEPT if sept else M_RUST
    else:                                  # both
        base = (M_SEPT + E_ROLL) if sept else (M_RUST + R_ROLL)
    return list(base) + (NL_RUST if target in RUST else [])


def _fit(target, T, form, featset, y_override=None):
    """Returns (prediction array, test frame). y_override lets us fit a
    transformed response (e.g. conditional severity) with the same design."""
    tr = DF[(DF.Year < T) & (DF.Year >= MIN_YEAR)].copy()
    te = DF[DF.Year == T].copy()
    if len(te) == 0:
        return None, None
    y = tr[target] if y_override is None else y_override.reindex(tr.index)
    ok = y.notna()
    tr, y = tr[ok], y[ok]
    if len(tr) < 30:
        return None, None

    cy = sorted(tr.Year.unique())[-CLIM_K:]
    cl = tr[tr.Year.isin(cy)]
    nc = y[tr.Year.isin(cy)].mean()
    rc = y[tr.Year.isin(cy)].groupby(tr.loc[tr.Year.isin(cy), "Region"]).mean()
    base = 0.5 * te.Region.map(rc).fillna(nc).to_numpy() + 0.5 * nc
    if form == "clim":
        return base, te

    cols = wcols(target, featset) + [
        f"bl_reg4_{target}", f"bl_reg10_{target}",
        f"bl_nat4_{target}", f"bl_nat10_{target}", "trend"]
    if form == "int":
        for d in (tr, te):
            d["ix_dr_bl"] = d["nl_drought"] * d[f"bl_nat4_{target}"]
            d["ix_dry_bl"] = d["nl_dry2"] * d[f"bl_nat4_{target}"]
            d["ix_rain_bl"] = d["e_rain_apr_may_ranom"] * d[f"bl_nat4_{target}"]
        cols = cols + ["ix_dr_bl", "ix_dry_bl", "ix_rain_bl"]

    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    yv = y.to_numpy(float)
    if form == "rel":
        b_tr = tr[f"bl_reg10_{target}"].fillna(tr[f"bl_nat10_{target}"]).to_numpy(float)
        b_te = te[f"bl_reg10_{target}"].fillna(te[f"bl_nat10_{target}"]).to_numpy(float)
        eps = 0.05 * max(np.nanmean(yv), 1e-3)
        z = np.clip(np.log((yv + eps) / (b_tr + eps)), -4, 4)
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), z)
        p = (b_te + eps) * np.exp(np.clip(m.predict(sc.transform(Xte)), -4, 4)) - eps
    else:
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), yv)
        p = m.predict(sc.transform(Xte))
    return p, te


def predict_one(target, T, form, featset, conditional):
    """Severity via decomposition when `conditional`, else direct."""
    if conditional and target in PAIR:
        inc_t = PAIR[target]
        inc = DF[inc_t]
        cond = DF[target] / (inc / 100).replace(0, np.nan)
        cond = cond.where(inc > 2)
        cond = cond.clip(upper=cond.quantile(0.99))
        p_c, te = _fit(target, T, form, featset, y_override=cond)
        p_i, _ = _fit(inc_t, T, form, featset)
        if p_c is None or p_i is None:
            return None
        p = np.clip(p_i, 0, 100) / 100 * np.clip(p_c, 0, None)
    else:
        p, te = _fit(target, T, form, featset)
        if p is None:
            return None
    p = np.clip(p, 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=p)


def run(cfg, years=EVAL):
    out = []
    for t in TARGETS:
        forms = cfg["forms_sept"] if t in SEPTORIA else cfg["forms_rust"]
        for T in years:
            ps = [p for f in forms
                  if (p := predict_one(t, T, f, cfg["featset"], cfg["conditional"])) is not None]
            if not ps:
                continue
            ps = [p.sort_values(["Year", "Region"]).reset_index(drop=True) for p in ps]
            v = np.mean([p.value.to_numpy() for p in ps], axis=0)
            o = ps[0][["Year", "Region", "target"]].copy()
            o["value"] = np.clip(v, 0, 100 if t in INCIDENCE else None)
            out.append(o)
    return pd.concat(out, ignore_index=True)


C = lambda **kw: dict(dict(forms_sept=["add", "int"], forms_rust=["add", "rel", "int"],
                           featset="fix", conditional=False), **kw)

CONFIGS = {
    "G0 current best (fixed anom, direct)": C(),
    "G1 + rolling anomalies":               C(featset="roll"),
    "G2 + mechanistic only":                C(featset="mech"),
    "G3 + mech & rolling":                  C(featset="both"),
    "G4 G0 + conditional severity":         C(conditional=True),
    "G5 rolling + conditional":             C(featset="roll", conditional=True),
    "G6 both + conditional":                C(featset="both", conditional=True),
    "G7 mech + conditional":                C(featset="mech", conditional=True),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    rows, store = [], {}
    for name, cfg in CONFIGS.items():
        p = run(cfg)
        store[name] = p
        rec = {"config": name}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(p, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        rows.append(rec)
    print("=" * 118)
    print("INTEGRATING THIS ROUND'S WINS   (baseline to beat: sept 13.87 | rust 5.38)")
    print("=" * 118)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # cross-feature-set ensemble: mechanistic and statistical fail on different eras
    print("\n" + "=" * 118)
    print("CROSS-FEATURE-SET ENSEMBLE (mech and statistical are complementary)")
    print("=" * 118)
    key = ["Year", "Region", "target"]
    def blend(names, label):
        ps = [store[n].sort_values(key).reset_index(drop=True) for n in names]
        m = ps[0][key].assign(value=np.mean([p.value.to_numpy() for p in ps], axis=0))
        rec = {"config": label}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(m, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        return rec, m
    brows = []
    for names, lab in [
        (["G5 rolling + conditional", "G7 mech + conditional"], "B1 avg(rolling, mech) + cond"),
        (["G4 G0 + conditional severity", "G7 mech + conditional"], "B2 avg(fixed, mech) + cond"),
        (["G4 G0 + conditional severity", "G5 rolling + conditional",
          "G7 mech + conditional"], "B3 avg(fixed, rolling, mech) + cond"),
        (["G0 current best (fixed anom, direct)", "G7 mech + conditional"], "B4 avg(G0, mech+cond)"),
    ]:
        rec, _ = blend(names, lab)
        brows.append(rec)
    print(pd.DataFrame(brows).round(3).to_string(index=False))
