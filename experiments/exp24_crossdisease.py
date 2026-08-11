"""Experiment 24: cross-disease features, and a 2026 forecast post-mortem.

PART 1 -- cross-disease coupling.
Septoria and yellow rust year-effects are NEGATIVELY correlated (-0.05 to -0.26
across the target pairs). That is not a coincidence: the wet, dull springs that
drive splash-dispersed septoria are not the mild-winter/cool-bright springs that
favour rust, and a canopy already lost to one pathogen offers less to the other.
So each disease's drivers and prevailing level may carry information about the
other. Tested by adding the OTHER disease's weather block and as-of baselines.

PART 2 -- where does the 2026 number actually come from?
2026 is the only year that gets scored, so it is worth auditing rather than
trusting. This decomposes the forecast across ensemble members and compares it
against the analogue years, to see how much of it is weather signal and how much
is the elevated 2020s baseline.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 250)

from common import (TARGETS, SEVERITY, INCIDENCE, SEPTORIA, RUST, load_pest,
                    rmse, to_long, score)
import final_model_v2 as FM

DF, OBS = FM.DF, FM.OBS
EVAL = [y for y in range(2005, 2026) if y != 2020]
truth = to_long(OBS)

# ---------------------------------------------------------------- PART 1
OTHER = {}
for t in TARGETS:
    twin = t.replace("Zymoseptoria_tritici", "TMP").replace("Yellow_rust", "Zymoseptoria_tritici")
    twin = twin.replace("TMP", "Yellow_rust")
    OTHER[t] = twin

_orig_weather_cols = FM.weather_cols


def weather_cols_cross(target, basis):
    """Own block plus the other disease's block and baselines."""
    own = _orig_weather_cols(target, basis)
    other = target.replace("Zymoseptoria_tritici", "TMP") \
                  .replace("Yellow_rust", "Zymoseptoria_tritici").replace("TMP", "Yellow_rust")
    other_block = _orig_weather_cols(other, basis)
    return list(dict.fromkeys(own + other_block +
                              [f"bl_nat4_{other}", f"bl_nat10_{other}"]))


if __name__ == "__main__":
    print("=" * 106)
    print("PART 1 -- does adding the OTHER disease's drivers help?")
    print("=" * 106)
    base = FM.run(EVAL)
    _, b_all = score(base, truth, EVAL)
    _, b_19 = score(base, truth, list(range(2005, 2020)))
    _, b_25 = score(base, truth, [2021, 2022, 2023, 2024, 2025])

    FM.weather_cols = weather_cols_cross
    cross = FM.run(EVAL)
    _, c_all = score(cross, truth, EVAL)
    _, c_19 = score(cross, truth, list(range(2005, 2020)))
    _, c_25 = score(cross, truth, [2021, 2022, 2023, 2024, 2025])
    FM.weather_cols = _orig_weather_cols

    print(f"{'':<16}{'sept 05_25':>12}{'sept 05_19':>12}{'sept 21_25':>12}"
          f"{'rust 05_25':>12}{'rust 05_19':>12}{'rust 21_25':>12}")
    print(f"{'v2 (own only)':<16}{b_all['septoria_pooled']:>12.3f}{b_19['septoria_pooled']:>12.3f}"
          f"{b_25['septoria_pooled']:>12.3f}{b_all['rust_pooled']:>12.3f}"
          f"{b_19['rust_pooled']:>12.3f}{b_25['rust_pooled']:>12.3f}")
    print(f"{'+ cross-disease':<16}{c_all['septoria_pooled']:>12.3f}{c_19['septoria_pooled']:>12.3f}"
          f"{c_25['septoria_pooled']:>12.3f}{c_all['rust_pooled']:>12.3f}"
          f"{c_19['rust_pooled']:>12.3f}{c_25['rust_pooled']:>12.3f}")

    # ------------------------------------------------------------ PART 2
    print("\n" + "=" * 106)
    print("PART 2 -- 2026 forecast post-mortem: member-by-member spread")
    print("=" * 106)
    show = ["L1_Zymoseptoria_tritici_Crop_Incidence",
            "L2_Zymoseptoria_tritici_Crop_Incidence",
            "L1_Zymoseptoria_tritici_Disease_Severity",
            "L2_Yellow_rust_Crop_Incidence"]
    rows = []
    for t in show:
        grp = "septoria" if t in SEPTORIA else "rust"
        rec = {"target": t.replace("_Crop_Incidence", "_inc")
                          .replace("_Disease_Severity", "_sev")}
        for b in FM.BASES[grp]:
            for f in FM.FORMS[grp]:
                p = FM._member(t, 2026, f, b)
                rec[f"{b}/{f}"] = p.value.mean() if p is not None else np.nan
        pc, _ = FM._fit(t, 2026, "clim", "fix")
        rec["climatology"] = np.nanmean(pc)
        rec["ENSEMBLE"] = FM.predict(t, 2026).value.mean()
        rows.append(rec)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nHow the same members scored 2025 (the closest drought analogue):")
    rows = []
    for t in show:
        grp = "septoria" if t in SEPTORIA else "rust"
        rec = {"target": t.replace("_Crop_Incidence", "_inc")
                          .replace("_Disease_Severity", "_sev"),
               "ACTUAL": OBS[OBS.Year == 2025][t].mean()}
        for b in FM.BASES[grp]:
            for f in FM.FORMS[grp]:
                p = FM._member(t, 2025, f, b)
                rec[f"{b}/{f}"] = p.value.mean() if p is not None else np.nan
        rec["ENSEMBLE"] = FM.predict(t, 2025).value.mean()
        rows.append(rec)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nBias of the ensemble by year (national mean, model - actual):")
    bt = FM.run(EVAL)
    for t in show:
        a = OBS[OBS.Year.isin(EVAL)].groupby("Year")[t].mean()
        p = bt[bt.target == t].groupby("Year").value.mean()
        d = (p - a).dropna()
        print(f"  {t.replace('_Crop_Incidence','_inc').replace('_Disease_Severity','_sev'):<42}"
              f"mean bias {d.mean():+7.2f}   |bias| {d.abs().mean():6.2f}   "
              f"last3 {', '.join(f'{y}:{d[y]:+.1f}' for y in [2023,2024,2025] if y in d)}")
