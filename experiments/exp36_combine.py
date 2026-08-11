"""Experiment 36: do the three round-4 winners stack?

Three of the five round-4 ideas cleared their pre-registered bar:

  exp31  average over alpha {30, 100, 300} instead of fixing alpha = 100
  exp35  multi-task: share one weather-coefficient vector across a disease's
         four targets (lam = 1), blended 0.4 into the ensemble -- SEPTORIA only
  exp33  hurdle P(present) x E[level | present], blended 0.4 -- RUST incidence only

Individually they are worth roughly -0.02, -0.08 and -0.02 RMSE. They are not
obviously independent: alpha-averaging and multi-task shrinkage are both
variance-reduction devices acting on the same coefficient estimates, so the
second may have little left to remove once the first has run. Gains measured
separately routinely fail to add up, so the combination is measured directly
rather than assumed.

Each is applied only where it earned its keep. Nothing here is applied to a
disease it was not validated on -- exp35 made rust slightly worse in every
configuration, and exp33 is meaningless for septoria, whose incidence is almost
never zero.

The build-up is reported cumulatively so that if the total disappoints it is
visible which step stopped paying.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, INCIDENCE, to_long, score
import final_model_v2 as FM
import exp33_rust_hurdle as HU
import exp35_multitask as MT

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)

ALPHAS = [30.0, 100.0, 300.0]
RUST_INC = [t for t in RUST if t in INCIDENCE]
BASE_ALPHA = FM.ALPHA


def alpha_avg(target, T, inner=None):
    """Mean of `inner` (default FM.predict) over the alpha grid."""
    fn = inner or FM.predict
    ps = []
    for a in ALPHAS:
        FM.ALPHA = a
        try:
            p = fn(target, T)
        finally:
            FM.ALPHA = BASE_ALPHA
        if p is not None:
            ps.append(p.sort_values(["Year", "Region"]).reset_index(drop=True))
    if not ps:
        return None
    out = ps[0][["Year", "Region", "target"]].copy()
    out["value"] = np.clip(np.mean([p.value.to_numpy() for p in ps], axis=0),
                           0, 100 if target in INCIDENCE else None)
    return out


def blend(a, b, w, target):
    if a is None:
        return b
    if b is None:
        return a
    a = a.sort_values(["Year", "Region"]).reset_index(drop=True)
    b = b.sort_values(["Year", "Region"]).reset_index(drop=True)
    out = a[["Year", "Region", "target"]].copy()
    out["value"] = np.clip((1 - w) * a.value.to_numpy() + w * b.value.to_numpy(),
                           0, 100 if target in INCIDENCE else None)
    return out


MT_CACHE = {}


def multitask_member(target, T):
    """Mean over bases/forms of the shared-coefficient fit (lam = 1)."""
    grp = "septoria" if target in SEPTORIA else "rust"
    ps = []
    for b in FM.BASES[grp]:
        for f in FM.FORMS[grp]:
            if f == "rel":
                continue
            key = (grp, T, f, b, FM.ALPHA)
            if key not in MT_CACHE:
                MT_CACHE[key] = MT._fit_group(grp, T, f, b, 1.0)
            g = MT_CACHE[key]
            if g is not None:
                ps.append(g[g.target == target]
                          .sort_values(["Year", "Region"]).reset_index(drop=True))
    if not ps:
        return None
    out = ps[0][["Year", "Region", "target"]].copy()
    out["value"] = np.mean([p.value.to_numpy() for p in ps], axis=0)
    return out


def build(use_alpha, use_mt, use_hurdle):
    def fn(target, T):
        core = alpha_avg(target, T) if use_alpha else FM.predict(target, T)
        if core is None:
            return None
        if use_mt and target in SEPTORIA:
            mt = alpha_avg(target, T, multitask_member) if use_alpha \
                else multitask_member(target, T)
            core = blend(core, mt, 0.4, target)
        if use_hurdle and target in RUST_INC:
            hz = HU.hurdle_member(target, T, "fix", 1.0, True)
            core = blend(core, hz, 0.4, target)
        return core
    return fn


def evaluate(fn, label):
    p = FM.run(EVAL) if fn is None else FM.run(EVAL, fn)
    rec = {"variant": label}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    return rec, p


if __name__ == "__main__":
    steps = [
        (None, "v2 shipped"),
        (build(True, False, False), "+ alpha averaging"),
        (build(True, True, False), "+ multi-task (septoria)"),
        (build(True, True, True), "+ hurdle (rust)  = v3"),
        (build(False, True, True), "[control] no alpha averaging"),
    ]
    rows, preds = [], {}
    for fn, lab in steps:
        rec, p = evaluate(fn, lab)
        rows.append(rec)
        preds[lab] = p
        print(f"  done {lab}", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    for g in ["sept", "rust"]:
        df[f"d_{g}19"] = df[f"{g}_05_19"] - ref[f"{g}_05_19"]
        df[f"d_{g}25"] = df[f"{g}_21_25"] - ref[f"{g}_21_25"]
        df[f"{g}_both"] = np.where((df[f"d_{g}19"] < 0) & (df[f"d_{g}25"] < 0),
                                   "YES", "")
    print("\n" + "=" * 132)
    print("CUMULATIVE BUILD-UP")
    print("=" * 132)
    print(df[["variant", "sept_05_25", "sept_05_19", "sept_21_25", "sept_both",
              "rust_05_25", "rust_05_19", "rust_21_25", "rust_both"]]
          .round(3).to_string(index=False))

    v3 = preds["+ hurdle (rust)  = v3"]
    pt3, _ = score(v3, truth, EVAL)
    pt2, _ = score(preds["v2 shipped"], truth, EVAL)
    print("\nper-target RMSE, 2005-2025:")
    print(f"{'target':<45}{'v2':>10}{'v3':>10}{'change':>10}")
    for t in TARGETS:
        print(f"{t:<45}{pt2[t]:>10.3f}{pt3[t]:>10.3f}"
              f"{100*(1-pt3[t]/pt2[t]):>9.1f}%")
