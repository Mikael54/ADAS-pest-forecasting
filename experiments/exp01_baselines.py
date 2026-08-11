"""Experiment 01: naive baselines. These set the bar every ML model must beat."""
import numpy as np
import pandas as pd
from common import (TARGETS, backtest, report, to_long, EVAL_YEARS, RECENT_EVAL_YEARS)

RESULTS = []


def make_const_predictor(fn):
    def predict(train, test_rows):
        vals = fn(train)  # dict target -> scalar, or DataFrame indexed by Region
        out = []
        for t in TARGETS:
            v = vals[t]
            if isinstance(v, pd.Series):  # per-region
                pv = test_rows.Region.map(v).fillna(np.nanmean(v.values)).values
            else:
                pv = np.full(len(test_rows), v)
            d = test_rows.copy()
            d["target"] = t
            d["value"] = pv
            out.append(d)
        return pd.concat(out, ignore_index=True)
    return predict


# --- B1: global mean of all history ---------------------------------------
b1 = make_const_predictor(lambda tr: {t: tr[t].mean() for t in TARGETS})

# --- B2: global median of all history -------------------------------------
b2 = make_const_predictor(lambda tr: {t: tr[t].median() for t in TARGETS})

# --- B3: mean of last k years (climatology, recent window) ----------------
def make_recent_mean(k, stat="mean"):
    def fn(tr):
        yrs = sorted(tr.Year.unique())[-k:]
        s = tr[tr.Year.isin(yrs)]
        return {t: (s[t].mean() if stat == "mean" else s[t].median()) for t in TARGETS}
    return make_const_predictor(fn)


# --- B4: persistence -- last observed year, per region --------------------
def persistence(train, test_rows):
    last = train.sort_values("Year").groupby("Region").tail(1).set_index("Region")
    out = []
    for t in TARGETS:
        d = test_rows.copy()
        d["target"] = t
        d["value"] = d.Region.map(last[t]).fillna(train[t].mean()).values
        out.append(d)
    return pd.concat(out, ignore_index=True)


# --- B5: per-region mean of last k years ----------------------------------
def make_region_recent_mean(k):
    def predict(train, test_rows):
        yrs = sorted(train.Year.unique())[-k:]
        s = train[train.Year.isin(yrs)]
        gm = s.groupby("Region")[TARGETS].mean()
        out = []
        for t in TARGETS:
            d = test_rows.copy()
            d["target"] = t
            d["value"] = d.Region.map(gm[t]).fillna(s[t].mean()).values
            out.append(d)
        return pd.concat(out, ignore_index=True)
    return predict


# --- B6: shrunk region mean: alpha*region_mean + (1-alpha)*global_mean -----
def make_shrunk(k, alpha):
    def predict(train, test_rows):
        yrs = sorted(train.Year.unique())[-k:]
        s = train[train.Year.isin(yrs)]
        gm = s.groupby("Region")[TARGETS].mean()
        out = []
        for t in TARGETS:
            g = s[t].mean()
            d = test_rows.copy()
            d["target"] = t
            d["value"] = alpha * d.Region.map(gm[t]).fillna(g).values + (1 - alpha) * g
            out.append(d)
        return pd.concat(out, ignore_index=True)
    return predict


if __name__ == "__main__":
    runs = [
        ("B1 global mean (all history)", b1),
        ("B2 global median (all history)", b2),
        ("B3a recent mean k=5", make_recent_mean(5)),
        ("B3b recent mean k=10", make_recent_mean(10)),
        ("B3c recent mean k=15", make_recent_mean(15)),
        ("B3d recent median k=10", make_recent_mean(10, "median")),
        ("B4 persistence (last year, per region)", persistence),
        ("B5a region mean k=5", make_region_recent_mean(5)),
        ("B5b region mean k=10", make_region_recent_mean(10)),
        ("B6 shrunk region mean k=10 a=0.5", make_shrunk(10, 0.5)),
    ]
    rows = []
    for name, fn in runs:
        preds, truth = backtest(fn)
        r = report(name, preds, truth)
        rows.append({
            "model": name,
            "sept_all": r["all"]["septoria_pooled"], "rust_all": r["all"]["rust_pooled"],
            "sept_rec": r["recent"]["septoria_pooled"], "rust_rec": r["recent"]["rust_pooled"],
        })
    print("\n\n" + "=" * 78 + "\nSUMMARY (lower is better)\n" + "=" * 78)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
