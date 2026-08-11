"""Download real observed weather for the 9 ADAS survey regions.

Two independent sources (both free, no API key):

1. Met Office HadUK-Grid areal series (monthly, 1836-present, updated monthly).
   Authoritative UK observations, but only resolved to Met Office climate
   districts, so some ADAS regions share a district.

2. Open-Meteo ERA5 reanalysis archive (daily, 1950-present, ~5 day lag).
   Coarser physics but true per-region resolution and DAILY, which is what
   epidemiology actually needs (rain-splash days, wet spells, frost events).

Both are pulled to data_external/ and cached; re-running is cheap.
"""
import io
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_external"
OUT.mkdir(exist_ok=True)

REGIONS = [
    "East", "East Midlands", "North East", "North West", "South East",
    "South West", "Wales", "West Midlands", "Yorkshire and The Humber",
]

# ---- 1. Met Office districts ---------------------------------------------
MO_DISTRICT = {
    "East": "East_Anglia",
    "East Midlands": "Midlands",
    "West Midlands": "Midlands",
    "North East": "England_E_and_NE",
    "Yorkshire and The Humber": "England_E_and_NE",
    "North West": "England_NW_and_N_Wales",
    "Wales": "Wales",
    "South East": "England_SE_and_Central_S",
    "South West": "England_SW_and_S_Wales",
}
MO_VARS = ["Tmax", "Tmin", "Tmean", "Sunshine", "Rainfall", "Raindays1mm", "AirFrost"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]


def _get(url, tries=8):
    """GET with exponential backoff; Open-Meteo throttles heavy archive calls."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=180) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 60 * (i + 1)
                print(f"    429 rate limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def parse_metoffice_txt(txt):
    """Parse a Met Office areal-series .txt into year + 12 monthly columns.

    These files MUST be parsed as fixed-width. The current (incomplete) year's
    row is ragged -- whitespace-splitting silently slides its win/spr/sum/aut/ann
    summary values left into the empty month slots. Pandas' own fwf *inference*
    is also unsafe here: it narrows columns to the widths seen in the sampled
    rows and clips the leading digit off 3-digit values (112.8 -> 12.8).

    So: derive field edges from the header token positions (values are
    right-aligned to the same column the header text ends at), then verify
    against whitespace-splitting on the complete rows.
    """
    import re
    lines = txt.splitlines()
    hdr_i = next(i for i, l in enumerate(lines) if l.strip().startswith("year"))
    toks = list(re.finditer(r"\S+", lines[hdr_i]))
    names = [t.group() for t in toks]
    assert names[:13] == ["year"] + MONTHS, f"unexpected header {names[:13]}"
    edges = [t.end() for t in toks[:13]]
    spans = [(0, edges[0])] + [(edges[i], edges[i + 1]) for i in range(12)]

    recs = []
    for line in lines[hdr_i + 1:]:
        if not line.strip() or not line[:4].strip().isdigit():
            continue
        vals = [line[a:b].strip() for a, b in spans]
        rec = [int(vals[0])] + [np.nan if v in ("", "---") else float(v)
                                for v in vals[1:]]
        # cross-check complete rows against naive whitespace splitting
        parts = line.split()
        if len(parts) == len(names):
            ws = [np.nan if p == "---" else float(p) for p in parts[1:13]]
            assert all(
                (np.isnan(a) and np.isnan(b)) or a == b
                for a, b in zip(rec[1:], ws)
            ), f"fixed-width/whitespace mismatch:\n{line}\n{rec[1:]}\n{ws}"
        recs.append(rec)
    return pd.DataFrame(recs, columns=["year"] + MONTHS)


def fetch_metoffice():
    f = OUT / "metoffice_monthly.csv"
    if f.exists():
        print(f"[metoffice] cached {f.name}")
        return pd.read_csv(f)
    frames = []
    for var in MO_VARS:
        for dist in sorted(set(MO_DISTRICT.values())):
            url = (f"https://www.metoffice.gov.uk/pub/data/weather/uk/climate/"
                   f"datasets/{var}/date/{dist}.txt")
            txt = _get(url)
            df = parse_metoffice_txt(txt)
            df = df.melt(id_vars="year", var_name="month", value_name="value")
            df["month"] = df.month.map({m: i + 1 for i, m in enumerate(MONTHS)})
            df["var"] = var
            df["district"] = dist
            frames.append(df)
            print(f"[metoffice] {var:12s} {dist:26s} n={df.value.notna().sum()}")
    out = pd.concat(frames, ignore_index=True).rename(columns={"year": "Year"})
    out.to_csv(f, index=False)
    return out


def metoffice_to_regions(mo):
    """Wide monthly frame keyed on (Region, Year)."""
    rows = []
    for reg, dist in MO_DISTRICT.items():
        d = mo[mo.district == dist].copy()
        d["Region"] = reg
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["col"] = "mo_" + d["var"] + "_m" + d.month.astype(str).str.zfill(2)
    w = d.pivot_table(index=["Region", "Year"], columns="col", values="value")
    return w.reset_index()


# ---- 2. Open-Meteo ERA5 daily --------------------------------------------
# 3 sample points per region, chosen inside the region's wheat-growing area.
REGION_POINTS = {
    "East": [(52.60, 0.70), (52.20, 0.15), (52.85, 0.15)],
    "East Midlands": [(53.20, -0.55), (52.85, -0.90), (52.60, -0.45)],
    "North East": [(55.30, -1.70), (54.70, -1.55), (54.95, -1.85)],
    "North West": [(53.75, -2.75), (53.25, -2.65), (54.10, -2.85)],
    "South East": [(51.25, -1.10), (51.20, 0.60), (51.55, -0.95)],
    "South West": [(51.20, -2.20), (50.85, -2.35), (51.55, -2.55)],
    "Wales": [(51.70, -3.20), (52.45, -3.05), (53.20, -3.30)],
    "West Midlands": [(52.65, -2.55), (52.15, -2.45), (52.80, -2.10)],
    "Yorkshire and The Humber": [(53.95, -1.10), (53.65, -0.60), (54.15, -0.65)],
}
# Open-Meteo's archive endpoint is billed by an "API weight" roughly
# proportional to (points x variables x days), and it throttles hard: the first
# attempt (3 points x 10 vars x 1960-2026) got through only 4 of 9 regions
# before exhausting retries on HTTP 429.
#
# LITE mode keeps only the variables the epidemiology actually needs, one
# sample point per region, and starts at 1970 (the first year of pest data).
# That is ~6x less load, which should complete inside the free daily quota.
# Set ERA5_FULL=1 to restore the original heavier request.
FULL = os.environ.get("ERA5_FULL") == "1"

DAILY_VARS_FULL = [
    "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "relative_humidity_2m_mean",
    "wind_speed_10m_max", "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "dew_point_2m_mean",
]
DAILY_VARS_LITE = [
    "precipitation_sum",            # splash dispersal -- the septoria driver
    "temperature_2m_mean",          # thermal time / latent period
    "temperature_2m_min",           # frost kill (rust overwintering)
    "temperature_2m_max",           # June heat shutdown (rust)
    "relative_humidity_2m_mean",    # leaf wetness proxy
]
DAILY_VARS = DAILY_VARS_FULL if FULL else DAILY_VARS_LITE
START = "1960-01-01" if FULL else "1970-01-01"
N_POINTS = 3 if FULL else 1


def fetch_openmeteo(end_date):
    f = OUT / "era5_daily.csv"
    if f.exists():
        print(f"[open-meteo] cached {f.name}")
        return pd.read_csv(f, parse_dates=["date"])
    # One request per (point, variable-batch): the API rejects 3 points x 10 vars
    # x 60+ years in a single call. Cache per region so a 429 is resumable.
    cache = OUT / "_era5_cache"
    cache.mkdir(exist_ok=True)
    batches = [DAILY_VARS[:5], DAILY_VARS[5:]] if FULL else [DAILY_VARS]
    frames = []
    for reg, pts in REGION_POINTS.items():
        rf = cache / f"{reg.replace(' ', '_')}.csv"
        if rf.exists():
            m = pd.read_csv(rf, parse_dates=["date"])
            print(f"[open-meteo] {reg:26s} cached n={len(m)}")
            frames.append(m)
            continue
        per_pt = []
        for lat, lon in pts[:N_POINTS]:
            parts = []
            for bvars in batches:
                url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}"
                       f"&longitude={lon}&start_date={START}&end_date={end_date}"
                       f"&daily={','.join(bvars)}&timezone=UTC")
                js = json.loads(_get(url))
                d = pd.DataFrame(js["daily"])
                d["date"] = pd.to_datetime(d["time"])
                parts.append(d.drop(columns=["time"]).set_index("date"))
                time.sleep(4.0)
            per_pt.append(pd.concat(parts, axis=1))
        m = pd.concat(per_pt).groupby(level=0).mean().reset_index()
        m["Region"] = reg
        m.to_csv(rf, index=False)
        frames.append(m)
        print(f"[open-meteo] {reg:26s} {m.date.min().date()} -> {m.date.max().date()}"
              f"  n={len(m)}")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(f, index=False)
    return out


if __name__ == "__main__":
    mo = fetch_metoffice()
    reg = metoffice_to_regions(mo)
    reg.to_csv(OUT / "metoffice_regions_monthly.csv", index=False)
    print("\nMet Office regional wide:", reg.shape,
          "years", reg.Year.min(), "->", reg.Year.max())
    latest = mo.dropna(subset=["value"])
    print("latest month with data:",
          latest.groupby("Year").month.max().tail(3).to_dict())

    end = pd.Timestamp.today() - pd.Timedelta(days=7)
    era = fetch_openmeteo(end.strftime("%Y-%m-%d"))
    print("\nERA5 daily:", era.shape, era.date.min(), "->", era.date.max())
