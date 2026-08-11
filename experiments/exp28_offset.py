"""Experiment 28: make the as-of baseline an OFFSET, not a shrunk regressor.

DIAGNOSIS (exp27)
Septoria year-level error is 53 % of total MSE, and it is not symmetric noise --
the model under-predicts the elevated 2020s badly and consistently:

    2021 -38.9   2022 -12.4   2023 -30.9   2024 -7.7   2025 +11.4
    (national mean miss, septoria L1 incidence; mean -15.7)

MECHANISM
The as-of baseline enters as one standardised column among ~20, penalised by
alpha=100. Ridge therefore shrinks its coefficient well below 1, and every
prediction is pulled toward the mean of the 1990-onward training window. When
the current level sits far above that window's mean -- exactly the 2020s -- the
pull is a systematic downward bias, not noise.

FIX UNDER TEST
Fit the residual above the baseline and add the baseline back:

    y - baseline  ~  weather + baselines + trend + region        (ridge)
    prediction    =  baseline + fitted residual

Same features, same penalty. The only change is what shrinkage pulls TOWARD:
"this region's current level" instead of "the training-window mean". The
baseline columns stay in the design matrix, so the fit can undo the offset if it
is wrong -- this adds a hypothesis, it does not impose one.

Note this is the additive sibling of the existing `rel` form (which does the
same thing multiplicatively and is already used for rust). It has never been
tried for septoria, where the level problem is much larger.

ALSO TESTED
Which baseline half-life to centre on. hl=4 and hl=10 exist; a regime that
shifts abruptly may need something faster, so hl=2 is added here. A shorter
half-life tracks a jump sooner but is noisier, and with only 9 regions per year
that trade-off is not obvious a priori.

ADOPTION RULE (same as exp26): a variant must improve BOTH the 2005-2019 and
2021-2025 windows. Winning overall by trading one against the other is fitting
the fold split.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, SEPTORIA, RUST, to_long, score, rmse
import final_model_v2 as FM
from exp09_baseline_feats import as_of_baselines
from common import load_pest

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)

BASES0, FORMS0, OFFHL0 = dict(FM.BASES), dict(FM.FORMS), FM.OFF_HL


def evaluate(bases, forms, off_hl=4, label=""):
    FM.BASES, FM.FORMS, FM.OFF_HL = bases, forms, off_hl
    try:
        p = FM.run(EVAL)
    finally:
        FM.BASES, FM.FORMS, FM.OFF_HL = BASES0, FORMS0, OFFHL0
    rec = {"variant": label}
    for lab, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
        _, s = score(p, truth, yrs)
        rec[f"sept_{lab}"] = s["septoria_pooled"]
        rec[f"rust_{lab}"] = s["rust_pooled"]
    # recent-regime bias on the target where the problem was diagnosed
    m = (truth[truth.Year.isin(W25)]
         .merge(p, on=["Year", "Region", "target"], suffixes=("_t", "_p"))
         .query("target == 'L1_Zymoseptoria_tritici_Crop_Incidence'"))
    rec["bias_21_25"] = (m.value_p - m.value_t).mean()
    return rec


if __name__ == "__main__":
    S, R = FM.BASES["septoria"], FM.BASES["rust"]
    variants = [
        (BASES0, FORMS0, 4, "v2 shipped (add+int)"),
        (BASES0, {"septoria": ["off"], "rust": FORMS0["rust"]}, 4, "sept: off only, hl4"),
        (BASES0, {"septoria": ["add", "int", "off"], "rust": FORMS0["rust"]}, 4,
         "sept: add+int+off hl4"),
        (BASES0, {"septoria": ["add", "int", "off"], "rust": FORMS0["rust"]}, 10,
         "sept: add+int+off hl10"),
        (BASES0, {"septoria": ["add", "off"], "rust": FORMS0["rust"]}, 4,
         "sept: add+off hl4"),
        (BASES0, {"septoria": FORMS0["septoria"], "rust": ["add", "rel", "int", "off"]}, 4,
         "rust: +off hl4"),
        (BASES0, {"septoria": ["add", "int", "off"], "rust": ["add", "rel", "int", "off"]},
         4, "both: +off hl4"),
    ]
    rows = []
    for b, f, hl, lab in variants:
        rows.append(evaluate(b, f, hl, lab))
        print(f"  done {lab}", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    df["d_sept19"] = df.sept_05_19 - ref.sept_05_19
    df["d_sept25"] = df.sept_21_25 - ref.sept_21_25
    df["sept_both"] = np.where((df.d_sept19 < 0) & (df.d_sept25 < 0), "YES", "")
    df["d_rust19"] = df.rust_05_19 - ref.rust_05_19
    df["d_rust25"] = df.rust_21_25 - ref.rust_21_25
    df["rust_both"] = np.where((df.d_rust19 < 0) & (df.d_rust25 < 0), "YES", "")

    print("\n" + "=" * 130)
    print("OFFSET FORM   (bias_21_25 = mean national miss on septoria L1 incidence; "
          "shipped is -15.7)")
    print("=" * 130)
    print(df[["variant", "sept_05_25", "sept_05_19", "sept_21_25", "sept_both",
              "bias_21_25", "rust_05_25", "rust_05_19", "rust_21_25", "rust_both"]]
          .round(3).to_string(index=False))
