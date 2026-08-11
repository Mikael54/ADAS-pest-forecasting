"""Experiment 02: does real observed weather explain the year-to-year disease swings?

Builds growing-season-aligned monthly weather (Sept of Y-1 through Aug of Y, i.e.
the winter-wheat season that is harvested in year Y) and correlates it with the
national mean of each target.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from common import load_pest, TARGETS

pd.set_option("display.width", 250)
ROOT = Path(__file__).resolve().parent.parent
MO = pd.read_csv(ROOT / "data_external" / "metoffice_regions_monthly.csv")

MO_VARS = ["Tmax", "Tmin", "Tmean", "Sunshine", "Rainfall", "Raindays1mm", "AirFrost"]
# Season month offsets: (calendar month, year offset relative to harvest year)
SEASON = ([(m, -1) for m in range(9, 13)] + [(m, 0) for m in range(1, 9)])
SEASON_LABEL = {(9, -1): "s09", (10, -1): "s10", (11, -1): "s11", (12, -1): "s12",
                (1, 0): "s01", (2, 0): "s02", (3, 0): "s03", (4, 0): "s04",
                (5, 0): "s05", (6, 0): "s06", (7, 0): "s07", (8, 0): "s08"}


def build_season_monthly():
    """Wide frame: one row per (Region, harvest Year), cols = var x season month."""
    long = MO.melt(id_vars=["Region", "Year"], var_name="col", value_name="value")
    long = long[long.col.str.startswith("mo_")]
    long["var"] = long.col.str.split("_").str[1]
    long["month"] = long.col.str.split("_m").str[1].astype(int)
    out = []
    for (m, off), lab in SEASON_LABEL.items():
        d = long[long.month == m].copy()
        d["HarvestYear"] = d.Year - off
        d["feat"] = "w_" + d["var"] + "_" + lab
        out.append(d[["Region", "HarvestYear", "feat", "value"]])
    d = pd.concat(out, ignore_index=True)
    w = d.pivot_table(index=["Region", "HarvestYear"], columns="feat", values="value")
    return w.reset_index().rename(columns={"HarvestYear": "Year"})


if __name__ == "__main__":
    W = build_season_monthly()
    print("season weather frame:", W.shape, "years", W.Year.min(), "->", W.Year.max())
    print("2026 non-null feature count per region:")
    w26 = W[W.Year == 2026]
    print(w26.set_index("Region").notna().sum(axis=1).to_dict())
    print("\n2026 available months (Rainfall):")
    print(w26.set_index("Region")[[c for c in W.columns if "Rainfall" in c]].round(1).to_string())

    pest = load_pest()
    df = pest.merge(W, on=["Year", "Region"], how="left")
    feats = [c for c in W.columns if c.startswith("w_")]

    # National-level: average over regions, then correlate year series
    nat = df[df.Year.between(1971, 2025)].groupby("Year")[TARGETS + feats].mean()
    print(f"\n{'='*100}\nCORRELATION of national-mean target with national-mean season weather")
    print(f"(1971-2025, n={nat[TARGETS[0]].notna().sum()} years)\n{'='*100}")
    for t in TARGETS:
        y = np.log1p(nat[t])
        cors = {f: y.corr(nat[f]) for f in feats}
        s = pd.Series(cors).dropna().sort_values()
        top = pd.concat([s.head(6), s.tail(6)])
        print(f"\n--- {t}  (log1p) ---")
        print("  " + "  ".join(f"{k.replace('w_',''):>18s}:{v:+.2f}" for k, v in top.items()))

    # Panel level (region x year), within-year variation retained
    print(f"\n{'='*100}\nPANEL correlation (region x year, 2000-2025)\n{'='*100}")
    p = df[df.Year.between(2000, 2025)].dropna(subset=[TARGETS[0]])
    for t in TARGETS:
        y = np.log1p(p[t])
        s = pd.Series({f: y.corr(p[f]) for f in feats}).dropna().sort_values()
        top = pd.concat([s.head(4), s.tail(4)])
        print(f"{t:<45}" + " ".join(f"{k.replace('w_',''):>16s}:{v:+.2f}" for k, v in top.items()))
