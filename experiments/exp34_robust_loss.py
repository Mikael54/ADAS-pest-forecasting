"""Experiment 34: robust (Huber) loss instead of squared error for FITTING.

MOTIVATION
exp27's per-year table shows the national year-level miss is dominated by a
handful of years the model gets badly wrong in isolation:

    2016  actual 37.0   model 71.2   (+34.3)
    2021  actual 85.2   model 46.3   (-38.9)
    2023  actual 85.5   model 54.6   (-30.9)

Under squared-error fitting, years like these dominate the training objective
too: a residual of 35 contributes ~50x the gradient of a residual of 5. So a few
anomalous seasons -- which are anomalous precisely because something happened
that the weather does not encode -- get to set the weather coefficients for every
other year.

Huber loss is quadratic near zero and linear in the tail, so outlying seasons
still count but stop dominating. The slope is then estimated from the bulk of
ordinary years, which is where the weather->disease relationship is actually
identifiable.

⚠️ THE OBVIOUS OBJECTION
The contest metric IS squared error, so fitting a different loss deliberately
optimises the wrong objective -- the same argument that killed the log-target
transform in round 1. The counter-argument is that these are different things:
the log transform changed the TARGET (so predictions were biased on the raw
scale), whereas Huber changes only how the coefficients are ESTIMATED. The
prediction remains an estimate of the conditional mean; it is just estimated
more stably. Whether that trade pays is exactly what the backtest decides.

`epsilon` sets where quadratic turns linear, in units of residual SDs; 1.35 is
sklearn's default (95% efficiency under normality). Lower = more robust.

ADOPTION RULE: improve both 2005-2019 and 2021-2025.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, to_long, score
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)

_RIDGE = FM.Ridge


class HuberAsRidge:
    """Drop-in for Ridge(alpha=...) inside final_model_v2._fit.

    HuberRegressor's `alpha` penalises on a different scale to Ridge's, and it
    has no closed form, so it can fail to converge on collinear designs. On any
    failure this falls back to the ridge fit rather than dropping the member --
    a silently missing member would change the ensemble composition and
    confound the comparison.
    """
    EPS = 1.35
    SCALE = 1e-4

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y, sample_weight=None):
        try:
            m = HuberRegressor(epsilon=self.EPS, alpha=self.alpha * self.SCALE,
                               max_iter=400)
            m.fit(X, y, sample_weight=sample_weight)
            if not np.all(np.isfinite(m.coef_)):
                raise ValueError("non-finite coefficients")
            self._m = m
        except Exception:
            self._m = _RIDGE(alpha=self.alpha).fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._m.predict(X)


def evaluate(cls, eps, label):
    HuberAsRidge.EPS = eps
    FM.Ridge = cls
    try:
        p = FM.run(EVAL)
    finally:
        FM.Ridge = _RIDGE
    rec = {"variant": label}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    return rec


if __name__ == "__main__":
    rows = [evaluate(_RIDGE, 1.35, "v2 shipped (ridge)")]
    print("  done ridge", flush=True)
    for eps in [1.1, 1.35, 2.0, 3.0]:
        rows.append(evaluate(HuberAsRidge, eps, f"huber eps={eps}"))
        print(f"  done huber eps={eps}", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    for g in ["sept", "rust"]:
        df[f"d_{g}19"] = df[f"{g}_05_19"] - ref[f"{g}_05_19"]
        df[f"d_{g}25"] = df[f"{g}_21_25"] - ref[f"{g}_21_25"]
        df[f"{g}_both"] = np.where((df[f"d_{g}19"] < 0) & (df[f"d_{g}25"] < 0),
                                   "YES", "")
    print("\n" + "=" * 126)
    print("ROBUST LOSS")
    print("=" * 126)
    print(df[["variant", "sept_05_25", "sept_05_19", "sept_21_25", "sept_both",
              "rust_05_25", "rust_05_19", "rust_21_25", "rust_both"]]
          .round(3).to_string(index=False))
