"""Experiment 35: multi-task shrinkage across the four targets of a disease.

MOTIVATION
exp04 found that ~84 % of the variance across a disease's four targets (L1/L2 x
incidence/severity) is a single latent factor. They are four views of one
epidemic: if it was a bad septoria year on leaf 2 it was a bad septoria year on
leaf 1, and severity moves with incidence. Yet the model fits all four
INDEPENDENTLY, so each estimates its own weather coefficients from its own ~450
rows and its own noise.

If the underlying weather response is largely shared, that is four noisy
estimates of nearly the same vector, and the obvious fix is to shrink them toward
their common mean -- classic multi-task ridge. The effective sample size for the
shared component becomes ~1800 rows instead of ~450.

IMPLEMENTATION
Per (disease, basis, form):
  1. z-score each target's y using TRAINING mean/sd only, so the four
     coefficient vectors live on a comparable scale (incidence is 0-100,
     severity 0-28 and very skewed -- sharing raw coefficients would be
     meaningless);
  2. fit ridge on each standardised target;
  3. w_shared = mean of the four coefficient vectors;
  4. w_k <- (1 - lam) * w_k + lam * w_shared;
  5. predict, then un-standardise with the training mean/sd.

lam = 0 reproduces independent fitting; lam = 1 forces one common response with
only the per-target mean/sd differing. Sweeping lam traces the whole path, so
this cannot be worse than the current model at lam = 0 by construction -- the
question is whether the interior of the path beats the endpoint.

Blended into the existing ensemble rather than replacing it, since the shipped
severity route (incidence x conditional severity) is a separate win that this
does not attempt to reproduce.

ADOPTION RULE: improve both 2005-2019 and 2021-2025.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, INCIDENCE, to_long, score
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)
DF, REGIONS = FM.DF, FM.REGIONS
GROUP = {"septoria": list(SEPTORIA), "rust": list(RUST)}


def _fit_group(group, T, form, basis, lam):
    """Fit all four targets of a disease jointly with shared-coefficient shrinkage."""
    targets = GROUP[group]
    tr_all = DF[(DF.Year < T) & (DF.Year >= FM.MIN_YEAR)]
    te = DF[DF.Year == T]
    if len(te) == 0 or len(tr_all) < 40:
        return None

    # A single design matrix shared by all four targets. Feature columns are
    # target-specific only through the as-of baselines, so those are dropped
    # here and re-added per target below.
    wcols = FM.weather_cols(targets[0], basis)
    packs = []
    for t in targets:
        cols = wcols + [f"bl_reg4_{t}", f"bl_reg10_{t}",
                        f"bl_nat4_{t}", f"bl_nat10_{t}", "trend"]
        tr = tr_all[tr_all[t].notna()]
        if len(tr) < 30:
            return None
        Xtr, Xte = tr[cols].copy(), te[cols].copy()
        med = Xtr.median()
        Xtr, Xte = Xtr.fillna(med).fillna(0.0), Xte.fillna(med).fillna(0.0)
        for r in REGIONS[1:]:
            Xtr[f"R_{r}"] = (tr.Region == r).astype(float)
            Xte[f"R_{r}"] = (te.Region == r).astype(float)
        sc = StandardScaler().fit(Xtr)
        y = tr[t].to_numpy(float)
        mu, sd = y.mean(), y.std()
        if sd < 1e-9:
            return None
        m = Ridge(alpha=FM.ALPHA).fit(sc.transform(Xtr), (y - mu) / sd)
        packs.append(dict(t=t, sc=sc, Xte=Xte, coef=m.coef_, icpt=m.intercept_,
                          mu=mu, sd=sd))

    shared = np.mean([p["coef"] for p in packs], axis=0)
    out = []
    for p in packs:
        w = (1 - lam) * p["coef"] + lam * shared
        z = p["sc"].transform(p["Xte"]) @ w + p["icpt"]
        v = z * p["sd"] + p["mu"]
        out.append(te[["Year", "Region"]].assign(
            target=p["t"],
            value=np.clip(v, 0, 100 if p["t"] in INCIDENCE else None)))
    return pd.concat(out, ignore_index=True)


def make_predict(lam, blend):
    """Cache the joint fit per (group, year, form, basis); it yields all 4 targets."""
    cache = {}

    def fn(target, T):
        base = FM.predict(target, T)
        if base is None:
            return None
        grp = "septoria" if target in SEPTORIA else "rust"
        ps = []
        for b in FM.BASES[grp]:
            for f in FM.FORMS[grp]:
                if f == "rel":
                    continue                      # multiplicative form has no shared-coef analogue
                key = (grp, T, f, b)
                if key not in cache:
                    cache[key] = _fit_group(grp, T, f, b, lam)
                g = cache[key]
                if g is not None:
                    ps.append(g[g.target == target]
                              .sort_values(["Year", "Region"]).reset_index(drop=True))
        if not ps:
            return base
        mt = np.mean([p.value.to_numpy() for p in ps], axis=0)
        b0 = base.sort_values(["Year", "Region"]).reset_index(drop=True)
        out = b0[["Year", "Region", "target"]].copy()
        out["value"] = np.clip((1 - blend) * b0.value.to_numpy() + blend * mt,
                               0, 100 if target in INCIDENCE else None)
        return out
    return fn


def evaluate(lam, blend, label):
    p = FM.run(EVAL) if blend == 0 else FM.run(EVAL, make_predict(lam, blend))
    rec = {"variant": label}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    return rec


if __name__ == "__main__":
    rows = [evaluate(0, 0, "v2 shipped")]
    print("  done shipped", flush=True)
    for lam in [0.0, 0.5, 1.0]:
        for blend in [0.3, 0.5]:
            rows.append(evaluate(lam, blend, f"multitask lam={lam} blend={blend}"))
            print(f"  done lam={lam} blend={blend}", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    for g in ["sept", "rust"]:
        df[f"d_{g}19"] = df[f"{g}_05_19"] - ref[f"{g}_05_19"]
        df[f"d_{g}25"] = df[f"{g}_21_25"] - ref[f"{g}_21_25"]
        df[f"{g}_both"] = np.where((df[f"d_{g}19"] < 0) & (df[f"d_{g}25"] < 0),
                                   "YES", "")
    print("\n" + "=" * 126)
    print("MULTI-TASK SHRINKAGE  (lam = how far each target's coefficients are pulled "
          "toward the disease-average)")
    print("=" * 126)
    print(df[["variant", "sept_05_25", "sept_05_19", "sept_21_25", "sept_both",
              "rust_05_25", "rust_05_19", "rust_21_25", "rust_both"]]
          .round(3).to_string(index=False))
