"""Extract a yellow-rust virulence-frequency time series from UKCPVS annual reports.

WHY
exp27 showed yellow rust has captured only 21 % of its available year-effect gap
(septoria: 47 %). The reason is structural: rust's swings are driven by pathogen
race incursions and varietal resistance turnover -- the Warrior race take-over
after 2011 is the textbook case -- and no weather feature can see that. The only
way to close it is a new data source describing the PATHOGEN POPULATION.

SOURCE
The UK Cereal Pathogen Virulence Survey (NIAB/AHDB, running since 1967) screens
~30-100 Puccinia striiformis isolates a year against a differential set of Yr
resistance genes and named varieties. Its annual reports carry a table
"Frequency of detection of isolates carrying virulence to the different yellow
rust resistance genes and varieties over the past five years" -- percentage of
isolates virulent on each gene/variety, five years per report.

WHAT IS AND IS NOT AVAILABLE (checked 2026-08-11)
  * No CSV/JSON/API anywhere. AHDB publishes PDFs plus an interactive app.
  * The only machine-readable files are UKCPVS supplementary data (2022) and
    (2023) .xlsx -- single-year isolate x gene virulence matrices (41 and 26
    isolates). Useless as a time series.
  * So the series has to come out of the PDFs. Each report gives 5 years, with
    overlaps between consecutive reports that serve as a consistency check.

⚠️ COVERAGE LIMIT: this table only starts appearing in the 2014/2015 reports and
its earliest column is 2010. Reports back to 2004 were downloaded and checked --
they contain isolate-level pathotype tables but no frequency summary. So the
series is 2010-2024, about 15 annual values, against a 2005-2025 eval window.
That is thin, and it is the reason exp30 correlates before it models.
"""
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data_external"
PDFS = OUT / "ukcpvs"
PDFS.mkdir(parents=True, exist_ok=True)

BLOB = "https://projectblue.blob.core.windows.net/media/Default"
IMPORTED = (f"{BLOB}/Imported%20Publication%20Docs/AHDB%20Cereals%20&%20Oilseeds"
            f"/Disease/UKCPVS")
# Only reports that actually carry the frequency table are listed. 2004-2013
# were probed and do not have it.
REPORTS = {
    2015: f"{IMPORTED}/UKCPVS%20Annual%20Report%202015.pdf",
    2016: f"{IMPORTED}/UKCPVS%20Annual%20Report%202016.pdf",
    2017: f"{IMPORTED}/UKCPVS%20Annual%20Report%202017.pdf",
    2018: f"{IMPORTED}/UKCPVS%20Annual%20Report%202018.pdf",
    2019: f"{IMPORTED}/UKCPVS%20Annual%20Report%20(2019).pdf",
    2021: f"{BLOB}/Research%20Papers/Cereals%20and%20Oilseed/2021/"
          f"UKCPVS%20Annual%20Report%20(2021).pdf",
    2023: f"{IMPORTED}/UKCPVS%20Annual%20Report%20(2023).pdf",
    2024: f"{BLOB}/Research%20Papers/Cereals%20and%20Oilseed/2025/"
          f"UKCPVS%20Annual%20Report%20(2024).pdf",
}

HDR = re.compile(r"Frequency of detection of isolates carrying virulence to the "
                 r"different\s+(wheat\s+)?yellow\s*rust", re.I)
YEARS = re.compile(r"\b(19|20)\d{2}\b")
# label, then 5 cells: number (maybe decimal), '*', '-' or blank-as-dash
ROW = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z0-9 ()'./&+-]*?)\s{2,}"
                 r"(?P<vals>((\d+(\.\d+)?|\*|-)\s{2,}){4}(\d+(\.\d+)?|\*|-))\s*$")


def fetch():
    for year, url in REPORTS.items():
        pdf = PDFS / f"ukcpvs_{year}.pdf"
        if not pdf.exists():
            subprocess.run(["curl", "-sL", "--max-time", "120", "-o", str(pdf), url],
                           check=True)
        txt = pdf.with_suffix(".txt")
        if not txt.exists():
            subprocess.run(["pdftotext", "-layout", str(pdf), str(txt)], check=True)


def parse_report(year):
    """Return long rows (report, year, item, pct) from one report's table."""
    lines = (PDFS / f"ukcpvs_{year}.txt").read_text(errors="replace").splitlines()
    starts = [i for i, l in enumerate(lines) if HDR.search(l)]
    recs = []
    for s in starts:
        cols = None
        for j in range(s, min(s + 14, len(lines))):
            found = YEARS.findall(lines[j])
            nums = re.findall(r"\b(?:19|20)\d{2}\b", lines[j])
            if len(nums) == 5:
                cols = [int(n) for n in nums]
                body = j + 1
                break
        if cols is None:
            continue
        for k in range(body, min(body + 45, len(lines))):
            line = lines[k]
            if not line.strip():
                continue
            m = ROW.match(line)
            if not m:
                if re.match(r"^\s*(Table|\d+\s*$|-\s*Not tested)", line):
                    break
                continue
            label = m.group("label").strip()
            vals = re.split(r"\s{2,}", m.group("vals").strip())
            if len(vals) != 5:
                continue
            for c, v in zip(cols, vals):
                recs.append({"report": year, "Year": c, "item": label,
                             "pct": np.nan if v in ("*", "-") else float(v)})
        if recs:
            break                      # first matching table is the wheat one
    return recs


def build():
    fetch()
    recs = []
    for y in REPORTS:
        r = parse_report(y)
        yrs = sorted({x["Year"] for x in r})
        print(f"  report {y}: {len(r):4d} cells, years {yrs}, "
              f"{len({x['item'] for x in r})} items")
        recs += r
    d = pd.DataFrame(recs)

    # Overlapping reports restate earlier years, occasionally with small
    # retrospective revisions (2016 Warrior: 33 % in the 2018 report, 37 % in the
    # 2021 one). Prefer the most recent report that covers a year.
    d = (d.sort_values("report")
           .drop_duplicates(subset=["Year", "item"], keep="last"))
    w = d.pivot_table(index="Year", columns="item", values="pct")
    return d, w


if __name__ == "__main__":
    long, wide = build()
    OUT.mkdir(exist_ok=True)
    long.to_csv(OUT / "ukcpvs_virulence_long.csv", index=False)
    wide.to_csv(OUT / "ukcpvs_virulence.csv")
    print(f"\nseries {wide.shape[0]} years x {wide.shape[1]} items, "
          f"{wide.index.min()}-{wide.index.max()}")
    var = wide.std().sort_values(ascending=False)
    print("\nmost variable items (these carry whatever signal exists):")
    print(var.head(14).round(1).to_string())
    print("\nitems pinned at a constant (no usable variance):")
    print(", ".join(var[var < 1e-9].index.tolist()) or "(none)")
    print("\nwide table:")
    print(wide[var.head(10).index].round(0).to_string())
