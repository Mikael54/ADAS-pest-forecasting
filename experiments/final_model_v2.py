"""Final forecasting pipeline v2 -- SPHERE-PPL ADAS wheat disease contest.

USAGE:  cd experiments && ../.venv/bin/python final_model_v2.py
Writes submission/pest_forecasts_2026.csv and submission/pest_model_performance.csv.

WHAT CHANGED FROM v1 (see EXPERIMENTS.md sections 6 and 10)
  + mechanistic epidemiology features (degree-day infection cycles x splash
    events, Liebig limiting factor) ensembled alongside the statistical ones --
    they fail on different eras, so averaging beats choosing;
  + trailing 30-year rolling climatology as a second anomaly basis, which also
    removes the warming trend that was contaminating degree-day terms;
  + severity modelled as (incidence/100) x conditional severity, which improved
    all four severity targets;
  = yellow rust deliberately keeps the v1 feature basis: mechanistic features
    and rolling anomalies both make it slightly worse.

WHAT CHANGED IN v3 (round 4; see EXPERIMENTS.md section 17)
  + multi-task coefficient sharing across a disease's four targets -- ~84 % of
    their variance is one latent factor, and full sharing (lam=1) beat
    independent fitting. Septoria INCIDENCE only: it hurt severity, which
    already has its own decomposition, and it hurt rust;
  + averaging over the ridge penalty alpha in {30,100,300} rather than fixing
    100 -- different penalties suit different eras and the forecast year's era
    is unknown, so integrate instead of guessing. RUST only;
  + hurdle decomposition P(present) x E[level | present] for rust incidence,
    46 % of which is exactly zero. Blended at 0.4.
  Every component was required to improve BOTH 2005-2019 and 2021-2025.

  septoria    13.70 -> 13.62   (climatology 17.62,  +22.7 %)
  yellow rust  5.38 ->  5.35   (climatology  5.82,   +8.2 %)

ARCHITECTURE
  For each target, ridge pooled over all 9 regions:
      y ~ weather block + as-of disease baselines + year trend + region dummies
  averaged over
      feature bases : {fixed anomaly, rolling anomaly, mechanistic}  (septoria)
                      {fixed anomaly}                                (rust)
      model forms   : {add, int}                                     (septoria)
                      {add, rel, int}                                (rust)
  where
      add  purely additive
      int  additive + drought x baseline interactions (lets a dry spring cancel
           an elevated recent baseline -- matters because 2026 is a drought year)
      rel  multiplicative: models log(y / baseline)
      off  additive offset on the as-of baseline -- implemented and documented
           but NOT in FORMS: it cut the recent-years bias and raised RMSE
           (exp28), which is why the bias is left in place deliberately.

INFORMATION RULE: weather Sept(Y-1) -> June(Y); disease observations <= Y-1.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    to_long, score, rmse)
from exp09_baseline_feats import as_of_baselines
from exp18_rolling_mech import build_features, E_ROLL, R_ROLL, M_SEPT, M_RUST

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
FORECAST_YEAR = 2026
ALPHA, MIN_YEAR, CLIM_K = 100.0, 1990, 12
# Exponential recency weight on training rows, in years (None = equal weights).
# Knob for exp26; the shipped value is set from that experiment's result.
HALFLIFE = None
# Which as-of baseline half-life the "off" (offset) form centres on. See exp28.
OFF_HL = 4

E_FIX = ["e_rain_spring_anom", "e_rain_apr_may_anom", "e_rdays_latespring_anom",
         "e_sun_apr_may_anom", "e_tmean_spring_anom", "e_frost_winter_anom",
         "e_wetness_idx_anom", "e_dry_bright_spring_anom"]
R_FIX = ["e_tmin_winter_anom", "e_frost_winter_anom", "e_tmean_jun_anom",
         "e_tmean_spring_anom", "e_rain_spring_anom", "e_mild_winter_anom"]
NL_RUST = ["nl_junheat", "nl_mildwin", "nl_drought"]

# septoria averages three feature bases; rust only the fixed one (the others
# each cost it ~0.05 RMSE)
BASES = {"septoria": ["fix", "roll", "mech"], "rust": ["fix"]}
FORMS = {"septoria": ["add", "int"], "rust": ["add", "rel", "int"]}

PAIR = {"L1_Zymoseptoria_tritici_Disease_Severity": "L1_Zymoseptoria_tritici_Crop_Incidence",
        "L2_Zymoseptoria_tritici_Disease_Severity": "L2_Zymoseptoria_tritici_Crop_Incidence",
        "L1_Yellow_rust_Disease_Severity": "L1_Yellow_rust_Crop_Incidence",
        "L2_Yellow_rust_Disease_Severity": "L2_Yellow_rust_Crop_Incidence"}

# --- v3 additions (round 4). Each is applied ONLY where it improved both the
# --- 2005-2019 and 2021-2025 windows; see EXPERIMENTS.md section 17.
SEPT_INC = [t for t in SEPTORIA if t in INCIDENCE]
RUST_INC = [t for t in RUST if t in INCIDENCE]
ALPHAS_RUST = [30.0, 100.0, 300.0]   # exp31: average over the penalty, rust only
MT_BLEND = 0.4                       # exp35: multi-task share, septoria incidence
HURDLE_BLEND, HURDLE_THRESH = 0.4, 1.0   # exp33: zero-inflation, rust incidence


def build_frame():
    feat = build_features()
    for c in [c for c in feat.columns if c.endswith(("_anom", "_ranom"))]:
        feat[c] = feat[c].clip(-5, 5)      # tiny trailing SDs can blow up z-scores
    pest = load_pest()                     # includes the empty 2026 block
    bl = as_of_baselines(pest.reset_index(drop=True), TARGETS)
    df = (pest.reset_index(drop=True)
              .merge(feat, on=["Year", "Region"], how="left")
              .merge(bl, on=["Year", "Region"], how="left"))
    df["trend"] = (df.Year - 2000) / 25.0
    df["nl_drought"] = np.clip(df["e_sun_apr_may_anom"] - 1.0, 0, None)
    df["nl_dry2"] = np.clip(-df["e_rain_apr_may_anom"] - 0.75, 0, None)
    df["nl_junheat"] = np.clip(df["e_tmean_jun_anom"] - 1.0, 0, None)
    df["nl_mildwin"] = np.clip(-df["e_frost_winter_anom"] - 0.5, 0, None)
    return df, pest[pest.Year <= 2025].reset_index(drop=True)


DF, OBS = build_frame()
REGIONS = sorted(OBS.Region.unique())


def weather_cols(target, basis):
    sept = target in SEPTORIA
    block = {"fix": E_FIX if sept else R_FIX,
             "roll": E_ROLL if sept else R_ROLL,
             "mech": M_SEPT if sept else M_RUST}[basis]
    return list(block) + (NL_RUST if target in RUST else [])


def _fit(target, T, form, basis, y_override=None):
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
    inw = tr.Year.isin(cy)
    nc = y[inw].mean()
    rc = y[inw].groupby(tr.loc[inw, "Region"]).mean()
    base = 0.5 * te.Region.map(rc).fillna(nc).to_numpy() + 0.5 * nc
    if form == "clim":
        return base, te

    cols = weather_cols(target, basis) + [
        f"bl_reg4_{target}", f"bl_reg10_{target}",
        f"bl_nat4_{target}", f"bl_nat10_{target}", "trend"]
    if form == "int":
        for d in (tr, te):
            d["ix_dr_bl"] = d["nl_drought"] * d[f"bl_nat4_{target}"]
            d["ix_dry_bl"] = d["nl_dry2"] * d[f"bl_nat4_{target}"]
            d["ix_rain_bl"] = d["e_rain_apr_may_anom"] * d[f"bl_nat4_{target}"]
        cols = cols + ["ix_dr_bl", "ix_dry_bl", "ix_rain_bl"]

    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    yv = y.to_numpy(float)
    sw = (0.5 ** ((T - tr.Year.to_numpy()) / HALFLIFE)) if HALFLIFE else None
    if form == "rel":
        b_tr = tr[f"bl_reg10_{target}"].fillna(tr[f"bl_nat10_{target}"]).to_numpy(float)
        b_te = te[f"bl_reg10_{target}"].fillna(te[f"bl_nat10_{target}"]).to_numpy(float)
        eps = 0.05 * max(np.nanmean(yv), 1e-3)
        z = np.clip(np.log((yv + eps) / (b_tr + eps)), -4, 4)
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), z, sample_weight=sw)
        p = (b_te + eps) * np.exp(np.clip(m.predict(sc.transform(Xte)), -4, 4)) - eps
    elif form == "off":
        # Offset form: fit the RESIDUAL above the as-of baseline, then add the
        # baseline back. Ridge shrinks its coefficients toward zero, so the
        # shrinkage target becomes "this region's current level" rather than
        # "the mean of the 1990-onward training window". That matters because
        # the model under-predicts the elevated 2020s by ~16 points on septoria
        # L1 incidence (exp27). The baseline columns stay in the design matrix,
        # so the fit can still walk the offset back if it is unhelpful.
        b_tr = tr[f"bl_reg{OFF_HL}_{target}"].fillna(tr[f"bl_nat{OFF_HL}_{target}"])
        b_te = te[f"bl_reg{OFF_HL}_{target}"].fillna(te[f"bl_nat{OFF_HL}_{target}"])
        b_tr = b_tr.fillna(np.nanmean(yv)).to_numpy(float)
        b_te = b_te.fillna(np.nanmean(yv)).to_numpy(float)
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), yv - b_tr, sample_weight=sw)
        p = b_te + m.predict(sc.transform(Xte))
    else:
        m = Ridge(alpha=ALPHA).fit(sc.transform(Xtr), yv, sample_weight=sw)
        p = m.predict(sc.transform(Xte))
    return p, te


def _member(target, T, form, basis):
    """One ensemble member. Severity goes through the incidence x conditional
    decomposition, which beat direct fitting on all four severity targets."""
    if target in PAIR:
        inc_t = PAIR[target]
        inc = DF[inc_t]
        cond = DF[target] / (inc / 100).replace(0, np.nan)
        cond = cond.where(inc > 2)                       # unstable near zero
        cond = cond.clip(upper=cond.quantile(0.99))
        p_c, te = _fit(target, T, form, basis, y_override=cond)
        p_i, _ = _fit(inc_t, T, form, basis)
        if p_c is None or p_i is None:
            return None
        p = np.clip(p_i, 0, 100) / 100 * np.clip(p_c, 0, None)
    else:
        p, te = _fit(target, T, form, basis)
        if p is None:
            return None
    return te[["Year", "Region"]].assign(
        target=target, value=np.clip(p, 0, 100 if target in INCIDENCE else None))


def _ensemble(target, T):
    """The v2 ensemble: mean over feature bases x model forms."""
    grp = "septoria" if target in SEPTORIA else "rust"
    ps = [p for b in BASES[grp] for f in FORMS[grp]
          if (p := _member(target, T, f, b)) is not None]
    if not ps:
        return None
    ps = [p.sort_values(["Year", "Region"]).reset_index(drop=True) for p in ps]
    v = np.mean([p.value.to_numpy() for p in ps], axis=0)
    out = ps[0][["Year", "Region", "target"]].copy()
    out["value"] = np.clip(v, 0, 100 if target in INCIDENCE else None)
    return out


def _blend(a, b, w, target):
    if a is None or b is None:
        return a if b is None else b
    a = a.sort_values(["Year", "Region"]).reset_index(drop=True)
    b = b.sort_values(["Year", "Region"]).reset_index(drop=True)
    out = a[["Year", "Region", "target"]].copy()
    out["value"] = np.clip((1 - w) * a.value.to_numpy() + w * b.value.to_numpy(),
                           0, 100 if target in INCIDENCE else None)
    return out


def _alpha_avg(target, T, fn):
    """Mean over the ridge penalty grid. ALPHA is a point estimate of something
    genuinely uncertain, and exp26 showed different values win in different eras;
    averaging refuses to guess instead of guessing badly."""
    global ALPHA
    keep, ps = ALPHA, []
    for a in ALPHAS_RUST:
        ALPHA = a
        try:
            p = fn(target, T)
        finally:
            ALPHA = keep
        if p is not None:
            ps.append(p.sort_values(["Year", "Region"]).reset_index(drop=True))
    if not ps:
        return None
    out = ps[0][["Year", "Region", "target"]].copy()
    out["value"] = np.clip(np.mean([p.value.to_numpy() for p in ps], axis=0),
                           0, 100 if target in INCIDENCE else None)
    return out


def _design(target, basis, tr, te):
    cols = weather_cols(target, basis) + [
        f"bl_reg4_{target}", f"bl_reg10_{target}",
        f"bl_nat4_{target}", f"bl_nat10_{target}", "trend"]
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    med = Xtr.median()
    Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
    for r in REGIONS[1:]:
        Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
        Xte[f"R_{r}"] = (te.Region == r).astype(float)
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


_MT_CACHE = {}


def _fit_group(group, T, form, basis):
    """Multi-task: one shared weather-coefficient vector for a disease's four
    targets. ~84 % of the variance across them is a single latent factor, so
    fitting them independently is four noisy estimates of nearly one vector.
    Targets are z-scored first so the coefficients are commensurable."""
    targets = SEPTORIA if group == "septoria" else RUST
    tr_all = DF[(DF.Year < T) & (DF.Year >= MIN_YEAR)]
    te = DF[DF.Year == T]
    if len(te) == 0 or len(tr_all) < 40:
        return None
    packs = []
    for t in targets:
        tr = tr_all[tr_all[t].notna()]
        if len(tr) < 30:
            return None
        Xtr, Xte = _design(t, basis, tr, te)
        y = tr[t].to_numpy(float)
        mu, sd = y.mean(), y.std()
        if sd < 1e-9:
            return None
        m = Ridge(alpha=ALPHA).fit(Xtr, (y - mu) / sd)
        packs.append((t, Xte, m, mu, sd))
    shared = np.mean([m.coef_ for _, _, m, _, _ in packs], axis=0)
    out = []
    for t, Xte, m, mu, sd in packs:          # lam = 1: full sharing (exp35)
        v = (Xte @ shared + m.intercept_) * sd + mu
        out.append(te[["Year", "Region"]].assign(
            target=t, value=np.clip(v, 0, 100 if t in INCIDENCE else None)))
    return pd.concat(out, ignore_index=True)


def _multitask(target, T):
    grp = "septoria" if target in SEPTORIA else "rust"
    ps = []
    for b in BASES[grp]:
        for f in FORMS[grp]:
            if f == "rel":                    # no shared-coefficient analogue
                continue
            key = (grp, T, f, b, ALPHA)
            if key not in _MT_CACHE:
                _MT_CACHE[key] = _fit_group(grp, T, f, b)
            g = _MT_CACHE[key]
            if g is not None:
                ps.append(g[g.target == target]
                          .sort_values(["Year", "Region"]).reset_index(drop=True))
    if not ps:
        return None
    out = ps[0][["Year", "Region", "target"]].copy()
    out["value"] = np.mean([p.value.to_numpy() for p in ps], axis=0)
    return out


def _hurdle(target, T, basis="fix"):
    """P(present) x E[level | present]. 46 % of rust incidence observations are
    exactly zero, so one linear fit has to serve both "does an epidemic start"
    (overwintering survival) and "how far does it run" (spring conditions)."""
    tr = DF[(DF.Year < T) & (DF.Year >= MIN_YEAR) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(tr) < 40 or len(te) == 0:
        return None
    y = tr[target].to_numpy(float)
    pos = y > HURDLE_THRESH
    if pos.sum() < 20 or (~pos).sum() < 10:
        return None
    Xtr, Xte = _design(target, basis, tr, te)
    clf = LogisticRegression(C=1.0 / ALPHA, max_iter=2000).fit(Xtr, pos.astype(int))
    lev = Ridge(alpha=ALPHA).fit(Xtr[pos], np.log1p(y[pos]))
    p_level = np.clip(np.expm1(np.clip(lev.predict(Xte), -10, 10)),
                      HURDLE_THRESH, None)
    v = np.clip(clf.predict_proba(Xte)[:, 1] * p_level, 0, 100)
    return te[["Year", "Region"]].assign(target=target, value=v)


def predict(target, T):
    """v3. Each component is applied only where it beat v2 on BOTH the
    2005-2019 and 2021-2025 windows (exp36):
      alpha averaging -> rust only (on septoria it cost 2005-2019);
      multi-task      -> septoria INCIDENCE only (it hurt severity, which
                         already has its own decomposition);
      hurdle          -> rust incidence only (septoria is never near zero).
    """
    if target in RUST:
        core = _alpha_avg(target, T, _ensemble)
    else:
        core = _ensemble(target, T)
    if core is None:
        return None
    if target in SEPT_INC:
        core = _blend(core, _multitask(target, T), MT_BLEND, target)
    if target in RUST_INC:
        core = _blend(core, _hurdle(target, T), HURDLE_BLEND, target)
    return core


# Leaf 2 emerges earlier, sits nearer the splash-dispersed inoculum and is
# exposed longer, so it carries at least as much septoria -- true in 97.8-99.3 %
# of observed rows. NOT imposed for rust, where it holds only 75-78 %.
SEPT_PAIRS = [("L1_Zymoseptoria_tritici_Disease_Severity",
               "L2_Zymoseptoria_tritici_Disease_Severity"),
              ("L1_Zymoseptoria_tritici_Crop_Incidence",
               "L2_Zymoseptoria_tritici_Crop_Incidence")]


def enforce_leaf_order(preds):
    w = preds.pivot_table(index=["Year", "Region"], columns="target",
                          values="value", observed=True)
    for a, b in SEPT_PAIRS:
        if a in w.columns and b in w.columns:
            bad = w[b] < w[a]
            mid = (w.loc[bad, a] + w.loc[bad, b]) / 2
            w.loc[bad, a], w.loc[bad, b] = mid, mid
    return (w.reset_index()
             .melt(id_vars=["Year", "Region"], var_name="target", value_name="value")
             .dropna(subset=["value"]))


def run(years, fn=predict):
    out = [p for t in TARGETS for T in years if (p := fn(t, T)) is not None]
    return enforce_leaf_order(pd.concat(out, ignore_index=True))


if __name__ == "__main__":
    EVAL = [y for y in range(2005, 2026) if y != 2020]
    truth = to_long(OBS)
    bt = run(EVAL)
    cl = run(EVAL, lambda t, T: (lambda r: None if r[0] is None else
             r[1][["Year", "Region"]].assign(target=t, value=r[0]))(_fit(t, T, "clim", "fix")))

    print("=" * 94)
    print("FINAL MODEL v2 -- rolling-origin backtest (train strictly on years < T)")
    print("=" * 94)
    for label, yrs in [("2005-2025 (excl 2020)", EVAL),
                       ("2005-2019", list(range(2005, 2020))),
                       ("2021-2025", [2021, 2022, 2023, 2024, 2025])]:
        _, a = score(bt, truth, yrs)
        _, c = score(cl, truth, yrs)
        print(f"\n{label}")
        for d, k in [("septoria   ", "septoria_pooled"), ("yellow rust", "rust_pooled")]:
            print(f"  {d} model {a[k]:7.3f}   climatology {c[k]:7.3f}   ({100*(1-a[k]/c[k]):+5.1f}%)")

    pt, _ = score(bt, truth, EVAL)
    ptc, _ = score(cl, truth, EVAL)
    print("\nper-target RMSE (2005-2025):")
    print(f"{'target':<45}{'model':>10}{'clim':>10}{'skill':>9}")
    for t in TARGETS:
        print(f"{t:<45}{pt[t]:>10.4f}{ptc[t]:>10.4f}{100*(1-pt[t]/ptc[t]):>8.1f}%")

    fc = run([FORECAST_YEAR]).rename(
        columns={"Year": "year", "Region": "region", "value": "forecast_value"})
    fc["target"] = pd.Categorical(fc.target, categories=TARGETS, ordered=True)
    fc = fc.sort_values(["region", "target"])[
        ["region", "target", "year", "forecast_value"]].reset_index(drop=True)

    perf = []
    m = truth[truth.Year.isin(EVAL)].merge(bt, on=["Year", "Region", "target"],
                                           suffixes=("_t", "_p")).dropna(subset=["value_t"])
    for (reg, t), g in m.groupby(["Region", "target"], observed=True):
        perf.append({"region": reg, "target": t, "rmse": rmse(g.value_t, g.value_p)})
    perf = pd.DataFrame(perf)
    perf["target"] = pd.Categorical(perf.target, categories=TARGETS, ordered=True)
    perf = perf.sort_values(["region", "target"]).reset_index(drop=True)

    sub = ROOT / "submission"
    fc.to_csv(sub / "pest_forecasts_2026.csv", index=False)
    perf.to_csv(sub / "pest_model_performance.csv", index=False)
    print("\n" + "=" * 94)
    print(f"{FORECAST_YEAR} FORECAST")
    print("=" * 94)
    print(fc.pivot_table(index="region", columns="target", values="forecast_value",
                         observed=True).round(2).to_string())
    print(f"\nwrote {len(fc)} forecast rows and {len(perf)} performance rows")
    hist = OBS[OBS.Year >= 2021].groupby("Year")[TARGETS].mean()
    hist.loc["2026 fc"] = fc.groupby("target", observed=True).forecast_value.mean()
    print("\nnational means, recent actuals vs 2026 forecast:")
    print(hist.round(2).to_string())
