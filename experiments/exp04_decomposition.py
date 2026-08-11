"""Experiment 04: how much of the error is even reducible?

Decomposes each target into region effect x year effect and computes ORACLE
RMSEs -- what you would score if you knew a given component perfectly. This
bounds what any model can achieve and says where to spend effort.
"""
import numpy as np
import pandas as pd
from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, EVAL_YEARS, RECENT_EVAL_YEARS)

pd.set_option("display.width", 220)
pest = load_pest()
obs = pest[(pest.Year <= 2025) & (pest.Year >= 1971)]

print("=" * 96)
print("VARIANCE DECOMPOSITION (1971-2025): share of variance from year vs region")
print("=" * 96)
print(f"{'target':<45}{'var_year':>10}{'var_region':>12}{'resid':>10}")
rows = []
for t in TARGETS:
    d = obs[["Year", "Region", t]].dropna()
    gm = d[t].mean()
    ye = d.groupby("Year")[t].transform("mean") - gm
    re_ = d.groupby("Region")[t].transform("mean") - gm
    resid = d[t] - gm - ye - re_
    tot = d[t].var()
    rows.append((t, ye.var() / tot, re_.var() / tot, resid.var() / tot))
    print(f"{t:<45}{ye.var()/tot:>10.2f}{re_.var()/tot:>12.2f}{resid.var()/tot:>10.2f}")

print()
print("=" * 96)
print("ORACLE RMSEs on eval years 2011-2025 (excl 2020)  [pooled by disease]")
print("=" * 96)


def oracle_scores(eval_years, label):
    ev = obs[obs.Year.isin(eval_years)]
    res = {}

    def pooled(pred_col, cols):
        d = ev.dropna(subset=cols, how="all")
        vals_t, vals_p = [], []
        for t in cols:
            s = ev[["Year", "Region", t]].dropna()
            vals_t.append(s[t].to_numpy())
            vals_p.append(pred_col[t].loc[s.index].to_numpy())
        return rmse(np.concatenate(vals_t), np.concatenate(vals_p))

    # O1: perfect national year mean, no region info
    p1 = {t: ev.groupby("Year")[t].transform("mean") for t in TARGETS}
    # O2: perfect national year mean x region climatological ratio (multiplicative)
    hist = obs[obs.Year.between(2005, 2025)]
    p2 = {}
    for t in TARGETS:
        ratio = (hist.groupby("Region")[t].mean() / hist[t].mean())
        p2[t] = ev.groupby("Year")[t].transform("mean") * ev.Region.map(ratio)
    # O3: perfect region climatology (last 10y), no year info
    p3 = {}
    for t in TARGETS:
        h = obs[obs.Year.between(2015, 2025)]
        p3[t] = ev.Region.map(h.groupby("Region")[t].mean())
    # O4: perfect year AND region main effects (additive, fitted on eval itself)
    p4 = {}
    for t in TARGETS:
        s = ev[["Year", "Region", t]].dropna()
        gm = s[t].mean()
        ye = s.groupby("Year")[t].transform("mean") - gm
        re_ = s.groupby("Region")[t].transform("mean") - gm
        p4[t] = (gm + ye + re_).reindex(ev.index)

    for nm, p in [("O1 perfect year mean only", p1),
                  ("O2 perfect year x region ratio", p2),
                  ("O3 perfect region clim only", p3),
                  ("O4 perfect year + region (additive)", p4)]:
        res[nm] = {
            "septoria": pooled(p, SEPTORIA), "rust": pooled(p, RUST),
            "sept_inc": pooled(p, [c for c in SEPTORIA if c in INCIDENCE]),
            "rust_inc": pooled(p, [c for c in RUST if c in INCIDENCE]),
            "sept_sev": pooled(p, [c for c in SEPTORIA if c in SEVERITY]),
            "rust_sev": pooled(p, [c for c in RUST if c in SEVERITY]),
        }
    print(f"\n--- {label} ---")
    print(pd.DataFrame(res).T.round(3).to_string())


oracle_scores(EVAL_YEARS, "eval years 2011-2025")
oracle_scores(RECENT_EVAL_YEARS, "recent 2021-2025")

print()
print("=" * 96)
print("Per-target national year series, 2011-2025 (what a year-effect model must track)")
print("=" * 96)
print(obs[obs.Year >= 2011].groupby("Year")[TARGETS].mean().round(2).to_string())

print()
print("=" * 96)
print("Cross-region SD within year (irreducible if you only model the year effect)")
print("=" * 96)
sd = obs[obs.Year >= 2011].groupby("Year")[TARGETS].std()
print(sd.mean().round(3).to_string())
