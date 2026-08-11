"""Experiment 17: mechanistic epidemiology instead of statistical weather aggregates.

Everything so far has fed the model monthly means/totals and let ridge find the
relationship. A different angle: build the quantities an epidemiologist would
actually compute, then feed those.

Septoria tritici
  Polycyclic, splash-dispersed. Disease on a given leaf layer is driven by
  (number of infection cycles completed) x (frequency of splash events while
  that leaf is exposed). Latent period is thermal: roughly 250-330 degree-days
  base 0 C. Leaf 2 emerges ~late April, flag leaf (L1) ~mid May, both assessed
  around GS71-75 in late June -- so L1 has a SHORTER exposure window than L2,
  which is exactly why L2 always carries more disease.

Yellow rust
  Much shorter latent period (~130-150 DD base 3 C), overwinters as mycelium so
  survival depends on frost, and is shut down above ~25 C.

From Met Office monthlies we have Raindays1mm (a direct splash-event count),
Tmean (thermal time), and AirFrost (winter kill) -- enough to build these.
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data_external"

DAYS = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31,
        9: 30, 10: 31, 11: 30, 12: 31}
# season label -> (calendar month, harvest-year offset)
SEASON = {"s09": (9, -1), "s10": (10, -1), "s11": (11, -1), "s12": (12, -1),
          "s01": (1, 0), "s02": (2, 0), "s03": (3, 0), "s04": (4, 0),
          "s05": (5, 0), "s06": (6, 0)}

DD_SEPT, BASE_SEPT = 300.0, 0.0     # degree-days per septoria latent cycle
DD_RUST, BASE_RUST = 140.0, 3.0     # degree-days per yellow rust latent cycle

# fraction of each month during which the leaf layer is present and exposed.
# L2 emerges ~25 Apr, L1 ~15 May; assessment ~end June.
EXPOSURE_L2 = {"s04": 0.2, "s05": 1.0, "s06": 1.0}
EXPOSURE_L1 = {"s05": 0.5, "s06": 1.0}


def monthly_panel():
    """Long frame: Region, harvest Year, season label, one column per variable."""
    mo = pd.read_csv(EXT / "metoffice_regions_monthly.csv")
    long = mo.melt(id_vars=["Region", "Year"], var_name="col", value_name="value")
    long = long[long.col.str.startswith("mo_")].copy()
    long["var"] = long.col.str.split("_").str[1]
    long["month"] = long.col.str.split("_m").str[1].astype(int)
    out = []
    for lab, (m, off) in SEASON.items():
        d = long[long.month == m].copy()
        d["Year"] = d.Year - off
        d["slab"] = lab
        out.append(d[["Region", "Year", "slab", "var", "value"]])
    d = pd.concat(out, ignore_index=True)
    return d.pivot_table(index=["Region", "Year", "slab"], columns="var",
                         values="value").reset_index()


def mechanistic_features():
    p = monthly_panel()
    p["ndays"] = p.slab.map({k: DAYS[v[0]] for k, v in SEASON.items()})
    # thermal time and splash frequency per month
    p["dd_sept"] = np.clip(p["Tmean"] - BASE_SEPT, 0, None) * p.ndays
    p["dd_rust"] = np.clip(p["Tmean"] - BASE_RUST, 0, None) * p.ndays
    p["splash_freq"] = p["Raindays1mm"] / p.ndays
    p["cyc_sept"] = p["dd_sept"] / DD_SEPT
    p["cyc_rust"] = p["dd_rust"] / DD_RUST
    # per-month epidemic increment: cycles completed x proportion of days
    # offering a splash dispersal event
    p["inc_sept"] = p["cyc_sept"] * p["splash_freq"]
    p["inc_rust"] = p["cyc_rust"] * p["splash_freq"]

    def agg(col, labs, weights=None):
        d = p[p.slab.isin(labs)].copy()
        if weights:
            d["w"] = d.slab.map(weights)
            d[col] = d[col] * d["w"]
        return d.groupby(["Region", "Year"])[col].sum(min_count=1)

    f = p[["Region", "Year"]].drop_duplicates().set_index(["Region", "Year"])
    SPR = ["s03", "s04", "s05", "s06"]
    AUT = ["s09", "s10", "s11"]
    WIN = ["s12", "s01", "s02"]

    # --- septoria: exposure-weighted epidemic potential per leaf layer ---
    f["m_sept_L1"] = agg("inc_sept", list(EXPOSURE_L1), EXPOSURE_L1)
    f["m_sept_L2"] = agg("inc_sept", list(EXPOSURE_L2), EXPOSURE_L2)
    f["m_sept_spring"] = agg("inc_sept", SPR)
    f["m_sept_autumn"] = agg("inc_sept", AUT)      # autumn inoculum build-up
    f["m_cyc_spring"] = agg("cyc_sept", SPR)
    f["m_splash_spring"] = agg("Raindays1mm", SPR)
    f["m_splash_L1"] = agg("Raindays1mm", list(EXPOSURE_L1), EXPOSURE_L1)
    f["m_splash_L2"] = agg("Raindays1mm", list(EXPOSURE_L2), EXPOSURE_L2)

    # --- rust: winter survival x spring cycles x June heat shutdown ---
    f["m_rust_cyc"] = agg("cyc_rust", SPR)
    f["m_rust_winter_cyc"] = agg("cyc_rust", WIN)     # green-bridge growth
    f["m_frost_win"] = agg("AirFrost", WIN)
    f["m_frost_season"] = agg("AirFrost", AUT + WIN + SPR)
    jun = p[p.slab == "s06"].set_index(["Region", "Year"])
    f["m_jun_tmax"] = jun["Tmax"]
    # heat shutdown: how far June max sits above the ~25 C rust ceiling
    f["m_heat_kill"] = np.clip(jun["Tmax"] - 21.0, 0, None)
    f["m_rust_potential"] = (f["m_rust_winter_cyc"] * f["m_rust_cyc"]
                             / (1 + f["m_frost_win"]) / (1 + f["m_heat_kill"]))
    f["m_survive_x_cyc"] = f["m_rust_cyc"] / (1 + f["m_frost_win"])

    # dry-spell severity: sunshine per rain day during stem extension
    sun = agg("Sunshine", SPR)
    f["m_sun_per_rainday"] = sun / f["m_splash_spring"].replace(0, np.nan)
    return f.reset_index()


def add_anomalies(f, ref_start=1961, ref_end=2000):
    cols = [c for c in f.columns if c.startswith("m_")]
    ref = f[(f.Year >= ref_start) & (f.Year <= ref_end)].groupby("Region")[cols]
    mu, sd = ref.mean(), ref.std()
    out = f.copy()
    for c in cols:
        out[c + "_anom"] = (f[c] - f.Region.map(mu[c])) / \
            f.Region.map(sd[c]).replace(0, np.nan)
    return out


def build():
    return add_anomalies(mechanistic_features())


if __name__ == "__main__":
    from common import load_pest, TARGETS
    pd.set_option("display.width", 250)
    M = build()
    print("mechanistic features:", M.shape, "years", M.Year.min(), "->", M.Year.max())
    anom = [c for c in M.columns if c.endswith("_anom")]

    pest = load_pest()
    pest = pest[pest.Year <= 2025]
    nat = pest.groupby("Year")[TARGETS].mean()
    natM = M.groupby("Year")[anom].mean()

    print("\n=== correlation of national target (log1p) with mechanistic index ===")
    print(f"{'feature':<26}" + "".join(f"{t.replace('_Disease_Severity','_sev').replace('_Crop_Incidence','_inc').replace('Zymoseptoria_tritici','Zt').replace('Yellow_rust','Yr'):>13}" for t in TARGETS))
    for c in anom:
        row = f"{c.replace('_anom',''):<26}"
        for t in TARGETS:
            j = pd.concat([np.log1p(nat[t]), natM[c]], axis=1).dropna()
            row += f"{j.corr().iloc[0,1]:>13.2f}"
        print(row)

    print("\n=== 2026 vs recent years (national mechanistic anomalies) ===")
    print(natM.reindex([2021, 2022, 2023, 2024, 2025, 2026])[
        ["m_sept_L1_anom", "m_sept_L2_anom", "m_splash_spring_anom",
         "m_sun_per_rainday_anom", "m_rust_potential_anom", "m_survive_x_cyc_anom",
         "m_frost_win_anom", "m_heat_kill_anom"]].round(2).to_string())
