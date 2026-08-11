"""Experiment 33: hurdle (presence x level) decomposition for yellow rust.

MOTIVATION
Yellow rust incidence is not a normally-distributed quantity with a mean -- it is
zero or near-zero in most region-years and occasionally large. Through the 2000s
national L1 incidence sat at 0.8; in 2021 it was 16.9. Fitting a single linear
model to that with squared-error loss forces one relationship to serve two
regimes: "does an epidemic happen at all" and "how big is it once it does".

Those have DIFFERENT drivers. Whether rust establishes is mostly about
overwintering survival (mild frost-free winter, inoculum carry-over). How far it
then runs is about spring conditions and whether June heat shuts it down.

This is exactly the structure that already paid off for severity, where
    severity = (incidence/100) x conditional severity
beat direct fitting on all four severity targets (v2, adopted). The analogous
split has never been tried for rust incidence:

    E[y] = P(y > t) x E[y | y > t]

fitted as a regularised logistic for the presence part and a ridge on the
positive rows for the level part.

WHY IT MIGHT STILL FAIL
The positive-rows fit trains on a shrinking subset (~40-60 % of rows), and rust
is the target with the least weather signal to begin with (exp27: 21 % of its
year-effect gap captured). Splitting scarce data two ways can easily cost more
in variance than the better functional form buys. Tested rather than assumed.

Threshold `t` is swept, since "present" has no natural cut point.

ADOPTION RULE: improve both 2005-2019 and 2021-2025.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import TARGETS, RUST, INCIDENCE, to_long, score
import final_model_v2 as FM

EVAL = [y for y in range(2005, 2026) if y != 2020]
W19 = list(range(2005, 2020))
W25 = [2021, 2022, 2023, 2024, 2025]
truth = to_long(FM.OBS)
RUST_INC = [t for t in RUST if t in INCIDENCE]
DF, REGIONS = FM.DF, FM.REGIONS


def _design(target, basis, tr, te):
    cols = FM.weather_cols(target, basis) + [
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


def hurdle_member(target, T, basis="fix", thresh=1.0, log_level=True):
    """P(y > thresh) x E[y | y > thresh], both fitted on years < T."""
    tr = DF[(DF.Year < T) & (DF.Year >= FM.MIN_YEAR) & DF[target].notna()]
    te = DF[DF.Year == T]
    if len(tr) < 40 or len(te) == 0:
        return None
    y = tr[target].to_numpy(float)
    pos = y > thresh
    # need both classes present and enough positives to fit a level model
    if pos.sum() < 20 or (~pos).sum() < 10:
        return None
    Xtr, Xte = _design(target, basis, tr, te)

    clf = LogisticRegression(C=1.0 / FM.ALPHA, max_iter=2000)
    clf.fit(Xtr, pos.astype(int))
    p_present = clf.predict_proba(Xte)[:, 1]

    ylev = y[pos]
    if log_level:
        ylev = np.log1p(ylev)
    lev = Ridge(alpha=FM.ALPHA).fit(Xtr[pos], ylev)
    p_level = lev.predict(Xte)
    if log_level:
        p_level = np.expm1(np.clip(p_level, -10, 10))
    p_level = np.clip(p_level, thresh, None)

    v = np.clip(p_present * p_level, 0, 100)
    return te[["Year", "Region"]].assign(target=target, value=v)


def make_predict(thresh, log_level, blend):
    """v2 ensemble for everything, plus a hurdle member for rust incidence."""
    def fn(target, T):
        base = FM.predict(target, T)
        if target not in RUST_INC or base is None:
            return base
        h = hurdle_member(target, T, "fix", thresh, log_level)
        if h is None:
            return base
        b = base.sort_values(["Year", "Region"]).reset_index(drop=True)
        h = h.sort_values(["Year", "Region"]).reset_index(drop=True)
        out = b[["Year", "Region", "target"]].copy()
        out["value"] = np.clip((1 - blend) * b.value.to_numpy()
                               + blend * h.value.to_numpy(), 0, 100)
        return out
    return fn


if __name__ == "__main__":
    y = FM.OBS[RUST_INC].to_numpy(float).ravel()
    y = y[~np.isnan(y)]
    print("yellow rust incidence distribution (all region-years, both leaves):")
    for q in [0, 10, 25, 50, 75, 90, 99, 100]:
        print(f"   p{q:<3d} {np.percentile(y, q):7.2f}")
    for t in [0.0, 1.0, 5.0]:
        print(f"   share <= {t:>4.1f}: {100*(y <= t).mean():.1f}%")

    rows = []
    variants = [(None, None, 0.0, "v2 shipped")]
    for th in [1.0, 5.0]:
        for lg in [True, False]:
            variants.append((th, lg, 0.5, f"hurdle t={th:g} log={lg} blend .5"))
    variants += [(1.0, True, 1.0, "hurdle t=1 log=True ALONE"),
                 (5.0, True, 0.33, "hurdle t=5 log=True blend .33")]

    for th, lg, bl, lab in variants:
        p = FM.run(EVAL) if th is None else FM.run(EVAL, make_predict(th, lg, bl))
        rec = {"variant": lab}
        for labw, yrs in [("05_25", EVAL), ("05_19", W19), ("21_25", W25)]:
            _, s = score(p, truth, yrs)
            rec[f"rust_{labw}"] = s["rust_pooled"]
        pt, _ = score(p, truth, EVAL)
        rec["L1_inc"], rec["L2_inc"] = pt[RUST_INC[0]], pt[RUST_INC[1]]
        rows.append(rec)
        print(f"  done {lab}", flush=True)

    df = pd.DataFrame(rows)
    ref = df.iloc[0]
    df["d_19"] = df.rust_05_19 - ref.rust_05_19
    df["d_25"] = df.rust_21_25 - ref.rust_21_25
    df["both"] = np.where((df.d_19 < 0) & (df.d_25 < 0), "YES", "")
    print("\n" + "=" * 118)
    print("RUST HURDLE MODEL   (septoria untouched)")
    print("=" * 118)
    print(df[["variant", "rust_05_25", "rust_05_19", "rust_21_25", "both",
              "L1_inc", "L2_inc"]].round(3).to_string(index=False))
