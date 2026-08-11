"""Experiment 20: analogue (k-nearest-year) forecasting.

Every model so far is parametric: fit coefficients on weather, extrapolate.
An analogue forecast is the opposite idea, and is how operational meteorology
handled this before regression -- find the historical years whose weather most
resembles the target year, and use what actually happened in those years.

Why it might beat ridge here:
  * it is fully non-parametric, so threshold/saturating responses (which exp06
    showed are real -- septoria only collapses in the TOP sunshine quintile)
    need no functional form;
  * it cannot produce impossible values, because every prediction is a blend of
    things that actually occurred;
  * 2026's spring closely resembles 2025's, and this is exactly the reasoning
    "2026 looks like 2025" formalised.

Level drift is handled by matching on RELATIVE disease: each analogue year
contributes its ratio (or difference) to its own as-of baseline, which is then
applied to 2026's baseline.
"""
import warnings
import numpy as np
import pandas as pd

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
EVAL = [y for y in range(2005, 2026) if y != 2020]

# national weather signature per year
NATF = FEAT.groupby("Year")[[c for c in FEAT.columns
                             if c.endswith(("_anom", "_ranom"))]].mean()

SEPT_SIG = ["m_liebig_spring_ranom", "m_moist_limited_ranom", "m_splash_spring_ranom",
            "m_sun_per_rainday_ranom", "e_rain_apr_may_ranom", "e_sun_apr_may_ranom",
            "e_tmean_spring_ranom"]
RUST_SIG = ["m_rust_potential_ranom", "m_survive_x_cyc_ranom", "m_frost_win_ranom",
            "m_heat_kill_ranom", "e_tmin_winter_ranom", "e_tmean_jun_ranom"]


def analogue_predict(target, T, cfg):
    sept = target in SEPTORIA
    sig = SEPT_SIG if sept else RUST_SIG
    hist = DF[(DF.Year < T) & (DF.Year >= cfg["min_year"]) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(te) == 0 or hist.empty:
        return None
    cand = sorted(hist.Year.unique())
    if len(cand) < cfg["k"] + 2:
        return None

    x_t = NATF.reindex([T])[sig].to_numpy(float)[0]
    X_h = NATF.reindex(cand)[sig].to_numpy(float)
    ok = ~np.isnan(X_h).any(axis=1) & ~np.isnan(x_t).any()
    cand = list(np.array(cand)[ok])
    X_h = X_h[ok]
    if len(cand) < cfg["k"] + 1:
        return None

    d = np.sqrt(((X_h - x_t) ** 2).sum(axis=1))
    order = np.argsort(d)[:cfg["k"]]
    yrs = [cand[i] for i in order]
    w = 1.0 / (d[order] + cfg["eps"])
    w = w / w.sum()

    # baseline at the forecast origin, per region
    b_te = te[f"bl_reg10_{target}"].fillna(te[f"bl_nat10_{target}"]).to_numpy(float)
    vals = np.zeros(len(te))
    tot_w = np.zeros(len(te))
    for wi, ya in zip(w, yrs):
        ha = hist[hist.Year == ya].set_index("Region")
        y_a = te.Region.map(ha[target]).to_numpy(float)
        b_a = te.Region.map(ha[f"bl_reg10_{target}"].fillna(ha[f"bl_nat10_{target}"])
                            ).to_numpy(float)
        if cfg["mode"] == "ratio":
            eps = 0.05 * max(hist[target].mean(), 1e-3)
            contrib = b_te * (y_a + eps) / (b_a + eps) - eps
        elif cfg["mode"] == "diff":
            contrib = b_te + (y_a - b_a)
        else:
            contrib = y_a
        m = ~np.isnan(contrib)
        vals[m] += wi * contrib[m]
        tot_w[m] += wi
    vals = np.where(tot_w > 0, vals / np.where(tot_w == 0, 1, tot_w), np.nan)
    vals = np.where(np.isnan(vals), b_te, vals)
    vals = np.clip(vals, 0, 100 if target in INCIDENCE else None)
    return te[["Year", "Region"]].assign(target=target, value=vals)


def run(cfg, years=EVAL):
    out = [p for t in TARGETS for T in years
           if (p := analogue_predict(t, T, cfg)) is not None]
    return pd.concat(out, ignore_index=True)


C = lambda **kw: dict(dict(k=5, mode="ratio", min_year=1971, eps=0.25), **kw)

CONFIGS = {
    "A1 k=3 ratio":   C(k=3),
    "A2 k=5 ratio":   C(k=5),
    "A3 k=8 ratio":   C(k=8),
    "A4 k=12 ratio":  C(k=12),
    "A5 k=5 diff":    C(k=5, mode="diff"),
    "A6 k=8 diff":    C(k=8, mode="diff"),
    "A7 k=5 raw":     C(k=5, mode="raw"),
    "A8 k=8 ratio 1990+": C(k=8, min_year=1990),
    "A9 k=5 ratio 1990+": C(k=5, min_year=1990),
}

if __name__ == "__main__":
    truth = to_long(OBS)
    rows = []
    for name, cfg in CONFIGS.items():
        p = run(cfg)
        rec = {"config": name}
        for lab, yrs in [("05_25", EVAL), ("05_19", list(range(2005, 2020))),
                         ("21_25", [2021, 2022, 2023, 2024, 2025])]:
            _, s = score(p, truth, yrs)
            rec[f"sept_{lab}"] = s["septoria_pooled"]
            rec[f"rust_{lab}"] = s["rust_pooled"]
        rows.append(rec)
    print("=" * 112)
    print("ANALOGUE FORECASTING   (current best ensemble: sept 13.87 | rust 5.38;  clim 17.62 | 5.82)")
    print("=" * 112)
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # which years does 2026 actually resemble?
    print("\n" + "=" * 112)
    print("Nearest historical analogues for 2026")
    print("=" * 112)
    for lab, sig in [("SEPTORIA signature", SEPT_SIG), ("RUST signature", RUST_SIG)]:
        cand = [y for y in range(1971, 2026) if y != 2020]
        X = NATF.reindex(cand)[sig].to_numpy(float)
        x = NATF.reindex([2026])[sig].to_numpy(float)[0]
        ok = ~np.isnan(X).any(axis=1)
        cand, X = list(np.array(cand)[ok]), X[ok]
        d = np.sqrt(((X - x) ** 2).sum(axis=1))
        o = np.argsort(d)[:8]
        print(f"\n{lab}: " + ", ".join(f"{cand[i]} (d={d[i]:.2f})" for i in o))
        nat = OBS.groupby("Year")[TARGETS].mean()
        show = ["L1_Zymoseptoria_tritici_Crop_Incidence",
                "L1_Zymoseptoria_tritici_Disease_Severity"] if "SEPT" in lab else \
               ["L2_Yellow_rust_Crop_Incidence", "L1_Yellow_rust_Crop_Incidence"]
        print(nat.reindex([cand[i] for i in o])[show].round(2).to_string())
