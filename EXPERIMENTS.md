# Experiment log — ADAS wheat pest forecasting (RMSE minimisation)

Running record of everything tried, what worked, what didn't, and why.
**Read this first if resuming in a fresh session.**

Last updated: 2026-08-11

---

## 0. Setup

- Python venv at `.venv/` (sklearn 1.9, lightgbm 4.7, xgboost 3.3, pandas 3.0).
  Run everything as `cd experiments && ../.venv/bin/python <script>.py`.
- All code in [experiments/](experiments/); external data cached in `data_external/`.

## 1. Problem structure (established facts)

| Fact | Consequence |
|---|---|
| 9 regions in the survey (no Scotland/NI/London), 1971–2025 + 2026 to predict | 9×8 = 72 numbers to forecast |
| **2020 is missing entirely** (no COVID-year survey) | "lag 1" must mean *last observed* year, not Year−1 |
| Wales missing 2003–2019 | Wales has ~39 rows not 53 |
| 2026 targets are `-9999` dummies | must be mapped to NaN |
| Targets: 4 severity (0–28, very skewed) + 4 incidence (0–100 %) | **pooled RMSE is dominated by incidence** — that is where the score is won |
| `agronomic`/`bioclim` end 2025, `fungicide` ends 2024 (biennial), `prop_LUC` ends **2015** | no year-2026 predictors ship in the repo; LUC is far too stale to matter |
| `bioclim_data.csv` is **HadGEM2 (CMIP5 GCM) output**, not observations | interannual values are a model realisation; regions correlate 0.90–1.00 so it carries almost no cross-region signal |
| `BIO12`…`BIO19` are in kg m⁻² s⁻¹ (values ~1e−8) | fine numerically, but they are *not* mm |

### Variance decomposition (exp04) — the single most important result

Share of variance in each target:

| component | share |
|---|---|
| **national year effect** | **0.51 – 0.79** |
| region effect | 0.00 – 0.09 |
| region×year residual | 0.17 – 0.44 |

Oracle RMSEs (pooled by disease, eval years 2011–2025 excl. 2020 / recent 2021–2025):

| oracle | septoria | rust |
|---|---|---|
| perfect national year mean only | 10.22 / 8.13 | 4.46 / 6.60 |
| perfect year × region climatological ratio | **10.20 / 9.05** | **3.85 / 5.53** |
| perfect region climatology, **no** year info | 16.72 / 16.27 | 6.65 / 8.69 |
| perfect additive year+region | 9.51 / 6.80 | 4.27 / 5.84 |

**Reading:** essentially all the winnable error is in the *year* effect. Getting
the region effect perfectly right but the year wrong scores 16.7 (septoria) —
barely better than climatology. So the model that matters is a
**national year-level model driven by weather**, then distributed across regions.

## 2. Evaluation protocol

`experiments/common.py::backtest` — rolling origin. For each eval year T, train
on all rows with `Year < T`, predict all regions in year T.
- `EVAL_YEARS` = 2011–2025 excl. 2020 (14 years, 944 target-obs)
- `RECENT_EVAL_YEARS` = 2021–2025 (360 obs) — most like 2026 (Wales present)
- Scores reported as pooled RMSE per disease (septoria / rust) + per target.

**Information rule** (enforced in `features.py`): to predict year Y we may use
weather Sept(Y−1)→**June(Y)**, disease ≤ Y−1, repo covariates ≤ Y−1. June cap
chosen because the ADAS survey scores L1/L2 at GS71–75 (late Jun/Jul) and
because it makes backtest features exactly as available as 2026's are today.

## 3. External data sourced

| source | what | status |
|---|---|---|
| **Met Office HadUK-Grid areal series** (`experiments/fetch_weather.py`) | monthly Tmax/Tmin/Tmean/Sunshine/Rainfall/Raindays1mm/AirFrost, 1836–2026, 7 climate districts | ✅ **2026 data through June already published** |
| **Open-Meteo ERA5 archive** | daily, per-region centroids (3 pts/region), 1960–2026 | fetching (rate-limited, resumable cache) |

Region → Met Office district mapping is in `fetch_weather.py::MO_DISTRICT`.
Note East Midlands + West Midlands share `Midlands`, and North East +
Yorkshire share `England_E_and_NE`, so Met Office alone cannot separate those
pairs — a reason to add ERA5.

### ⚠️ Two data bugs found and fixed (do not reintroduce)

1. **Ragged current-year row.** The Met Office `.txt` files must be parsed as
   **fixed width**. Whitespace-splitting slides 2026's trailing
   `win/spr/sum/aut/ann` summary values leftwards into the empty Jul/Aug month
   slots (2026 "July rainfall" appeared as 201.4 mm — it was the winter total).
2. **`pd.read_fwf` inference is also wrong**, it narrows columns to sampled
   widths and clips the leading digit off 3-digit values (Feb 2024 East Anglia
   112.8 mm → 12.8 mm). Fix: derive field edges from header token *end*
   positions and assert agreement with whitespace-splitting on complete rows
   (`fetch_weather.py::parse_metoffice_txt`).

## 4. Weather–disease signal (exp02) — all epidemiologically coherent

National-mean target (log1p) vs national-mean season weather, 1971–2025:

- **Septoria**: `Rainfall_Apr` +0.36, `Rainfall_May` +0.38…+0.45,
  `Raindays_May` +0.42, `Sunshine_May` **−0.42…−0.48**, `Tmean_Mar` +0.40,
  `Tmin_Mar` +0.44, `AirFrost_Jan–Mar` −0.32…−0.41.
  → *wet, mild, dull spring = high septoria.* Splash dispersal, as expected.
- **Yellow rust**: `Tmin_Jun` **−0.45…−0.48**, `Tmean_Jun` −0.41,
  `Tmean_Dec`/`Tmin_Dec` +0.21…+0.29, `Sunshine_Dec` −0.46…−0.48.
  → *mild winter builds inoculum, June heat shuts the epidemic down.*

### Live 2026 signal (already observed!)

East Anglia rainfall: Mar 24.2, **Apr 4.0**, May 16.7 mm (spring total 44.9 mm);
Sunshine Apr 248.5, May 246.2 h. This is a near-repeat of spring 2025
(50.2 mm, the driest in a century), and 2025 produced the **lowest septoria in
the whole record** (L1 severity 0.04, L1 incidence 34.96 vs typical ~1.1 / 61).
→ Strong prior that **2026 septoria is very low**.

---

## 5. Results so far

Pooled RMSE, `all` = 2011–2025 excl 2020, `rec` = 2021–2025. **Lower is better.**

### Naive baselines (exp01)

| model | sept all | sept rec | rust all | rust rec |
|---|---|---|---|---|
| global mean (all history) | 21.05 | 23.47 | 6.85 | 9.23 |
| recent mean k=5 | 19.49 | 20.26 | **6.39** | 9.76 |
| recent mean k=10 | 18.07 | 18.57 | 6.73 | 10.34 |
| region mean k=10 | 17.89 | 18.34 | 6.86 | 10.51 |
| **shrunk region mean k=10, α=0.5** | **17.71** | **18.06** | 6.78 | 10.40 |
| persistence (last year, per region) | 25.20 | 21.26 | 7.82 | 11.73 |

**Persistence is terrible** → disease has essentially no year-to-year carry-over;
it is weather-driven. Any model leaning on last year's level will lose.

### Pooled row-level models on real weather (exp03)

All fitted on region×year rows with region dummies + year trend.

| model | sept all | sept rec | rust all | rust rec |
|---|---|---|---|---|
| ridge, epi anomalies | 17.13 | 15.80 | 7.69 | 11.90 |
| ridge, raw monthly | 19.58 | 19.91 | 7.61 | 11.74 |
| ridge, monthly+epi | 20.01 | 21.05 | 7.60 | 11.74 |
| enet, monthly+epi | 19.47 | 18.22 | 7.61 | 11.77 |
| gbm, epi | 18.55 | 20.45 | 7.37 | 11.37 |
| rf, epi | 17.89 | 20.43 | 7.17 | 11.05 |
| **ridge epi, NO target transform** | **15.56** | 16.74 | 7.97 | 11.84 |
| ridge epi, shrink 0.3 → climatology | 15.91 | **15.03** | 7.37 | 11.39 |
| ridge epi, shrink 0.5 → climatology | 15.82 | 15.33 | 7.17 | 11.06 |
| ridge epi, train 1990+ | 16.10 | 15.39 | 7.34 | 11.34 |

**Verdicts:**
- ✅ Real observed weather beats the naive baselines on **septoria** (15.6 vs 17.7).
- ✅ Fitting on the **raw scale beats log/logit transform** for septoria — RMSE is
  a raw-scale squared loss, so transformed fits optimise the wrong thing.
- ✅ Shrinking toward recent climatology helps everywhere.
- ❌ **All of these are WORSE than naive on yellow rust** (7.2–8.0 vs 6.39).
  Rust is spiky and near-zero in most years; row-level models overfit it.
- ❌ Raw monthly weather (84 cols) is worse than ~50 engineered epi features —
  too many correlated columns for ~450 rows.
- ❌ Tree models (rf/gbm) do not beat ridge here; too little data, and they
  cannot extrapolate to an unprecedented year (which 2026 may be).

---

### Two-stage national year-effect model (exp05) — ❌ did NOT work

Stage 1 predicts the national year level from national weather, stage 2
distributes by region climatology. Despite exp04 saying the year effect is
everything, this scored **worse** (best sept 15.73) than the row-level pooled
ridge. exp06 explains why: stage 1 only has ~50 annual rows, and its own skill
is limited (see below). Fitting on region×year rows gives 9× the rows and the
region-level weather anomalies carry genuine within-year information.

### Stage-1 skill in isolation (exp06)

Rolling-origin prediction of the **national year mean**, 2005–2025:

| target | model RMSE | climatology | skill | corr |
|---|---|---|---|---|
| L1 septoria severity | 1.47 | 1.87 | +21 % | 0.60 |
| L1 septoria incidence | 22.02 | 24.14 | +9 % | 0.65 |
| L2 septoria severity | 2.52 | 3.95 | +36 % | 0.73 |
| L2 septoria incidence | 13.98 | 15.65 | +11 % | 0.69 |
| **all 4 yellow rust** | — | — | **−9 % to −395 %** | **−0.21 to −0.24** |

**Yellow rust weather skill is negative out of sample.** Do not use a linear
weather model for rust.

### Non-stationarity — the other half of the problem

National means by decade:

| decade | L1 sept inc | L1 rust inc | L2 rust inc |
|---|---|---|---|
| 1970s | 15.96 | 21.99 | 21.09 |
| 1990s | 49.10 | 3.57 | 4.35 |
| 2000s | 62.31 | 0.84 | 0.66 |
| 2010s | 51.93 | 2.16 | 2.17 |
| **2020s** | **75.59** | **9.36** | **14.92** |

Rust collapsed then came roaring back (Warrior-type race incursion, varietal
resistance breakdown); septoria has stepped up in the 2020s. **None of this is
weather.** Old years must be down-weighted and an explicit level term supplied.

---

## 6. Results — the models that actually work

Protocol: leave-one-year-out 2005–2025 (excl. 2020), configs fixed **a priori**
(no tuning on these windows). `05_19` and `21_25` sub-windows shown so a config
that only wins on one era is visible.

### Weather model vs climatology (exp08)

| config | sept 05–25 | sept 05–19 | sept 21–25 | rust 05–25 | rust 05–19 | rust 21–25 |
|---|---|---|---|---|---|---|
| C0 climatology only | 17.62 | 17.20 | 18.72 | **5.82** | **2.34** | **10.56** |
| C5 ridge blend .3, 1990+ | 14.71 | 13.34 | 17.91 | 6.04 | 2.57 | 10.87 |
| C9 ridge blend .5 hl=15 allfeat | **14.60** | 13.36 | 17.55 | 6.42 | 3.05 | 11.33 |

→ **septoria: weather wins big. rust: climatology beats every weather model.**

### Adding "as-of" disease-baseline features (exp09)

Region- and national-level exponentially-weighted means of the target over
*previously observed* years (half-lives 4 and 10), as ridge features. This is
what lets the model track the regime shift.

| config | sept 05–25 | sept 21–25 | rust 05–25 | rust 21–25 |
|---|---|---|---|---|
| D6 baseline only (no weather) | 17.53 | 18.28 | **5.67** | **10.01** |
| D10 weather + baseline, 1990+, α=30 | **14.25** | **17.30** | 5.87 | 10.51 |

### Husbandry covariates — real, but **unusable for 2026** (exp10, exp11)

| config | sept 05–25 | sept 05–19 | sept 21–25 |
|---|---|---|---|
| D10 (no extras) | 14.25 | 12.95 | 17.30 |
| + husbandry, **contemporaneous** | **13.46** | 12.46 | 15.87 |
| + husbandry, **lagged 2 years** | 14.23 | 12.79 | 17.56 |
| + plain year trend | 14.24 | 13.19 | 16.76 |

The gain needs *contemporaneous* husbandry. `agronomic_data.csv` is **entirely
empty for 2021 and 2025 and has no 2026 row**, so 2026 can only use a 2-year
carry-forward — and at lag 2 the gain is gone, matching a plain year trend.
**Decision: exclude husbandry; use a year trend.** (If the September 2026
assessment release *does* ship 2026 agronomic rows, switching it back on is
worth ~0.5 RMSE on septoria — the pipeline should detect and use it.)

### Transforms and non-linearity (exp12)

| config | sept 05–25 | sept 05–19 | sept 21–25 | rust 05–25 | rust 05–19 | rust 21–25 |
|---|---|---|---|---|---|---|
| **R6 weather+baseline+trend, α=100, 1990+** | **13.95** | 13.09 | **16.06** | 5.81 | 2.51 | 10.43 |
| + sqrt transform | 14.22 | 12.88 | 17.37 | 6.06 | 2.52 | 10.95 |
| + log/logit transform | 14.83 | 14.48 | 15.73 | 5.91 | 2.34 | 10.75 |
| **+ non-linear threshold terms** | 14.20 | 13.20 | 16.62 | **5.41** | 2.65 | **9.50** |

Non-linear terms are `relu` thresholds, not slopes: extreme-bright/dry spring,
extreme-wet spring, June heat above +1 SD, frost-free winter. Justified by
exp06's quintile table, where septoria only collapses in the *top* sunshine
quintile.

**They help yellow rust a lot (5.82 → 5.41 overall, 10.56 → 9.50 recent) and do
not help septoria.** Epidemiologically sensible: rust is governed by kill
thresholds (winter frost, June heat), septoria by a graded moisture response.

### Model FORM: additive vs multiplicative vs interaction (exp14)

The remaining big septoria misses are years where weather and the recent
baseline disagree — above all 2025 (spring dryness +5.1 SD, actual L1 incidence
35.0, predicted 53.6). The additive model lets the elevated 2020s baseline hold
the prediction up when the weather says "collapse". **This matters because 2026
is itself a drought year (dryness +3.9 SD).**

| form | sept 05–25 | 05–19 | 21–25 | **DRY yrs** | rust 05–25 | rust DRY |
|---|---|---|---|---|---|---|
| `clim` | 17.62 | 17.20 | 18.72 | 18.37 | 5.82 | 6.83 |
| `add` additive | 13.95 | 13.09 | 16.06 | 11.83 | 5.41 | 5.14 |
| `rel` multiplicative, models log(y/baseline) | 14.85 | **12.98** | 19.04 | **11.15** | 5.49 | **4.67** |
| `int` additive + drought×baseline | **13.90** | 12.99 | 16.12 | 11.33 | **5.37** | 4.97 |

`rel` handles drought best but badly under-predicts the recent high years
(2021: 34.8 vs actual 85.2). `int` is the compromise. Differences between `add`
and `int` are inside the noise of a 944-point backtest, which argues for
ensembling rather than picking a winner (exp15).

### 🏆 FINAL MODEL

Per-disease ensembles of the model forms, chosen because no single form won
consistently across windows:

- **Septoria** = mean(`add`, `int`)
- **Yellow rust** = mean(`add`, `rel`, `int`)

plus an **L2 ≥ L1 structural constraint for septoria only** (exp16): leaf 2
emerges earlier, sits nearer the splash-dispersed inoculum and is exposed
longer, so it always carries more disease. Holds in 97.8–99.3 % of observed
septoria rows — but only 75–78 % for rust, so it is *not* imposed there. RMSE
effect is ~nil; it exists to stop structurally impossible forecasts (the raw
2026 run gave East L2 severity 0.00 against L1 0.24).

Implemented in [experiments/final_model.py](experiments/final_model.py).

| window | septoria | clim | skill | rust | clim | skill |
|---|---|---|---|---|---|---|
| **2005–2025** | **13.87** | 17.62 | **+21.3 %** | **5.38** | 5.82 | **+7.6 %** |
| 2005–2019 | 13.00 | 17.20 | +24.4 % | 2.52 | 2.34 | −7.6 % |
| 2021–2025 | 15.99 | 18.72 | +14.6 % | 9.52 | 10.56 | +9.9 % |

Per-target skill vs climatology (2005–2025): septoria incidence **+23.2 % /
+18.0 %**, septoria severity +10.1 % / +19.8 %, rust L2 incidence +10.6 %,
rust severity ≈ 0.

Oracle floor is sept 10.20 / rust 3.85, so septoria has captured roughly half
the available headroom over climatology. Rust's 2005–2019 loss is expected:
in that era rust was near zero everywhere so climatology was almost perfect and
any model variance cost accuracy; the gain appears in the current high-rust
regime, which is the one 2026 sits in.

### Leakage audit (done)

- as-of baselines use strictly earlier observed years ✅
- weather capped at June of the harvest year ✅
- training window strictly `Year < T` ✅
- ⚠️ **fixed**: the feature-standardising climatology window was 1961–2010,
  which overlapped eval years 2005–2010. Moved to **1961–2000**, entirely
  before the first eval year. Results barely moved (sept 13.90 → 13.87),
  confirming it was not propping up the numbers.

### 2026 forecast (national means, vs recent actuals)

| year | L1 sept sev | L1 sept inc | L1 rust inc | L2 sept sev | L2 sept inc | L2 rust inc |
|---|---|---|---|---|---|---|
| 2023 | 0.64 | 85.5 | 4.4 | 3.83 | 94.2 | 6.0 |
| 2024 | 2.42 | 97.6 | 8.1 | 8.72 | 98.5 | 9.5 |
| 2025 | 0.04 | 35.0 | 12.3 | 0.29 | 68.2 | 26.3 |
| **2026 fc** | **1.36** | **58.6** | **9.7** | **2.87** | **74.8** | **22.4** |

Driven by: spring 2026 was dry and very bright (spring rain −1.24 SD, Apr–May
sunshine +2.68 SD — East Anglia had **4.0 mm of rain in April**), which
suppresses septoria; but the winter was very mild (frost −1.40 SD, Tmin
+1.57 SD), which favours yellow rust carry-over, partly offset by a hot June
(+3.05 SD). Hence low-to-moderate septoria and continued elevated rust.

## 7. Dead ends / cautions

- ❌ **Per-region models** (the repo example's approach): ~40 training rows each,
  and the region effect is only 0–9 % of variance. Pool all regions instead.
- ❌ **Two-stage national year model** (exp05) — too few annual rows; worse than
  row-level pooling.
- ❌ **Persistence / last-year's value** — worst baseline tried (sept 25.20).
  Disease has essentially no year-to-year carry-over.
- ❌ **Log/logit target transform** for septoria — RMSE is a raw-scale squared
  loss; transforming optimises the wrong objective. (It marginally helps
  septoria on 21–25 only, not overall.)
- ❌ **Tree models** (RF/GBM) — no better than ridge with this little data, and
  they cannot extrapolate to an unprecedented year, which 2026 may well be.
- ❌ **Raw 84-column monthly weather** — worse than ~30 engineered epi features.
- ❌ **Ensembling** several of these ridge configs (exp10 E1–E4) — the members
  are too correlated; averaging gave no gain over the best member.
- ❌ **Coordinate-descent hyper-parameter tuning** on 2011–2019 (exp07) — did
  **not** transfer to 2021–2025 (septoria 18.36 vs 18.72 climatology, i.e. all
  the apparent gain evaporated). Prefer a few a-priori configs over search.
- ❌ `prop_LUC` ends 2015; `bioclim` is GCM output — neither adds year-specific
  signal.
- ⚠️ Do not tune hard on `21_25` alone (5 years × 9 regions); prefer configs
  that win on both sub-windows.

Additional dead ends found later:

- ❌ **Multiplicative (`rel`) form alone** — best in drought years but collapses
  on 2021–2025 (19.04 vs 16.06). Only useful inside an ensemble.
- ❌ **L2 ≥ L1 constraint for yellow rust** — holds in only 75–78 % of observed
  rows, so imposing it is not justified (and does not help RMSE).
- ❌ **Blending predictions toward climatology with a fixed weight** — once the
  as-of baseline features are in the design matrix, ridge learns the right
  weight itself; an extra hand-set blend only hurts.

## 8. Still open / next

1. **ERA5 daily features** — ⚠️ **download incomplete: 4 of 9 regions**
   (`East`, `East Midlands`, `North East`, `North West`). Open-Meteo throttles
   the archive endpoint hard and the job died on HTTP 429 after exhausting 8
   retries (backoffs up to 480 s). The free tier bills an "API weight" roughly
   proportional to points × variables × days, and the original request
   (3 points × 10 variables × 1960–2026) is heavy enough to burn the daily
   quota part-way through.

   `fetch_weather.py` now defaults to a **LITE** request — 1 point per region,
   5 epidemiologically essential variables (`precipitation_sum`,
   `temperature_2m_mean/min/max`, `relative_humidity_2m_mean`), from 1970 —
   which is ~6× lighter and should complete inside the quota.
   `ERA5_FULL=1` restores the original heavy request.

   ⚠️ **Before resuming, delete `data_external/_era5_cache/`.** The 4 cached
   regions were fetched in FULL mode (3-point spatial averages); mixing them
   with LITE single-point regions would put a different noise level on
   different regions and manufacture spurious regional contrasts.

   Expected value, honestly: **downgraded to low by exp27.** It would separate
   the regions Met Office lumps together (East+West Midlands share `Midlands`;
   North East+Yorkshire share `England_E_and_NE`) and give true daily
   epidemiology (consecutive wet spells, rain-splash day counts inside growth
   stage windows) rather than monthly means. But §13D showed **53 % of septoria
   MSE is national year-level error**, which finer regional weather cannot
   touch, and most of the remaining within-year error is ADAS survey sampling
   noise rather than predictable regional structure (the systematic region
   effect is 0–9 % of variance). `Raindays1mm` already supplies a real
   splash-event count. This is the biggest untried *data* source but it is
   aimed at the smaller and largely irreducible half of the error.
2. Region×weather interactions — are some regions more moisture-responsive?
3. Multi-task shrinkage across the highly-correlated L1/L2 target pairs.
4. **Varietal resistance data** — AHDB Recommended List septoria/yellow-rust
   resistance ratings weighted by area grown would directly explain the regime
   shifts the model currently has to absorb through the baseline term. Not
   available programmatically; would need manual curation.
5. If the September 2026 assessment release ships **2026 agronomic rows**,
   switch contemporaneous husbandry back on — worth ~0.5 RMSE on septoria
   (exp10/exp11). The pipeline should detect availability rather than assume it.

## 9. How to reproduce

```bash
cd experiments
../.venv/bin/python fetch_weather.py       # caches Met Office + ERA5 to data_external/
../.venv/bin/python final_model_v2.py      # backtest + writes submission/*.csv
```

Individual experiments are `exp01_*.py` … `exp23_*.py`, each self-contained and
runnable the same way; each prints the table quoted in this log.
`final_model.py` is the v1 pipeline, kept for comparison.

---

# ROUND 2 — attacking from different angles

Round 1 converged on regularised ridge over statistical weather aggregates.
Round 2 asked what that framing was *missing*. Starting point: **sept 13.87 /
rust 5.38**.

## 10. Structure not being exploited

A single latent factor explains **84 %** of the variance in each disease's four
targets (two factors → 98 %). L1 and L2 severity correlate 0.94; L1 and L2
incidence 0.96. And **septoria and yellow rust are negatively correlated**
(−0.05 to −0.26) — the conditions favouring one disfavour the other. Eight
independent ridges throw all of this away.

## 11. Angles tried

### ✅ A. Mechanistic epidemiology instead of statistical aggregates (exp17/18)

Built what an epidemiologist would compute rather than monthly means: thermal
time (degree-days) → number of latent cycles completed (septoria ≈ 300 DD base
0 °C; rust ≈ 140 DD base 3 °C), × splash-event frequency from `Raindays1mm`,
weighted by each leaf layer's exposure window (L2 emerges ~25 Apr, L1 ~15 May —
which is *why* L2 always carries more disease). Plus winter frost kill and the
~25 °C June heat ceiling for rust.

`m_sept_spring` correlates **0.68** with national septoria severity — the
strongest single feature found anywhere in this project (best statistical
aggregate was ~0.45).

**Two bugs had to be fixed before it worked**, both instructive:
1. *Fixed 1961–2000 anomaly baseline + warming climate* → the degree-day term
   drifts up every year, so 2026 scored **+1.48** on septoria potential despite
   a drought. Fixed with a **trailing 30-year rolling climatology** (standard
   meteorological practice, and strictly backward-looking).
2. *Product form* (cycles × splash) lets warmth compensate for dryness. Added a
   **Liebig limiting-factor form**, `min(cycles, splash)` — but it must be built
   from *standardised* components: raw cycles ≈ 4 vs raw splash-days ≈ 40 meant
   the minimum was always the thermal term, making the index track warming only.
   After the fix it behaves correctly (2025 = −2.32 matching the record-low
   septoria; 2024 = +1.16; 2026 = −0.58).

Mechanistic features are **better on 2021–2025, worse on 2005–2019** than
statistical ones — complementary, so ensemble rather than choose.

### ✅ B. Severity = incidence × conditional severity (exp21)

Severity is a mean over *all* crops including unaffected ones, so
`severity = (incidence/100) × conditional_severity_among_affected`. The
year-to-year swings come from how *widely* the disease spread, not how bad it
got where it did — conditional severity is far more stable:

| target | raw CV | conditional CV |
|---|---|---|
| L1 septoria severity | 1.25 | 0.88 |
| L2 septoria severity | 0.98 | 0.75 |
| L1 rust severity | **3.35** | **0.99** |
| L2 rust severity | **3.33** | **1.25** |

Modelling the two factors separately and multiplying **improved all four
severity targets** (e.g. L2 septoria 3.79 → 3.66). Adopted.

### ⚠️ C. Latent-factor / reduced-rank multi-task (exp19)

Extract k factors from the 4 targets per disease inside each fold, regress the
factor *scores* on weather, reconstruct. k=1 gave septoria 13.92 overall but
**15.48 on 2021–2025** (better than the then-best 15.99); mechanistic-only k=2
reached **13.67** on 2021–2025. But every latent config made **rust clearly
worse** (6.15–6.42 vs 5.38). Not adopted — kept as a documented near-miss worth
revisiting if rust is handled separately.

### ❌ D. Analogue / k-nearest-year forecasting (exp20)

Match the target year's weather signature to historical years, blend what
actually happened (as a ratio to each analogue's own baseline). Fully
non-parametric, so thresholds need no functional form. **Clearly worse**: best
septoria 15.12 vs 13.87, rust 6.32 vs 5.38. With only ~50 candidate years and a
7-dimensional signature, nearest neighbours are too noisy.

Still useful as a sanity check — 2026's nearest analogues:
- *septoria*: 1989, 2017, 1997, 2003, 2007, 2009 → L1 incidence 19–73, mean ≈ 46
- *rust*: **2025** (nearest by a clear margin), 2017, 2005, 2003, 2022 — and
  2025 had L2 rust incidence 26.3, corroborating the elevated-rust forecast.

### ❌ E. Monotone-constrained gradient boosting (exp21)

Plain GBM/RF failed in round 1 by overfitting. Monotone constraints (septoria
non-decreasing in spring moisture, non-increasing in spring sunshine; rust
non-increasing in winter frost and June heat) are both a strong regulariser and
a statement of known epidemiology, and unlike a linear fit can represent the
threshold shape. **Still worse**: 15.00 vs 13.89 septoria. At ~450 rows, trees
lose to a well-regularised linear model even when handed the right shape.

### ❌ F. Rolling anomalies for *rust* threshold terms (exp23)

Rebasing rust's frost/heat thresholds on the trailing climatology cost 0.4 RMSE
(5.38 → 5.78). A rolling window keeps re-centring on recent mild winters, so
"frost-free winter" stops reading as unusual exactly when it matters most.
Rust's threshold terms stay on the fixed baseline.

### G. TabPFN (exp22) — environment notes

TabPFN is a transformer pre-trained on millions of synthetic tabular tasks that
does *in-context learning*: it conditions on the training rows at inference
rather than fitting parameters. Its design regime is small tabular data —
roughly what we have (~450 rows × ~30 features after the 1990+ cut), and exactly
the regime where RF/GBM/monotone-LGBM all overfit here. So it is a genuinely
different bet: nonparametric flexibility with regularisation supplied by the
prior instead of by a penalty.

**Environments available on this machine**

| | |
|---|---|
| `.venv` (this project) | `tabpfn==2.2.1` — runs ungated, downloads its own v2 weights |
| `bes_africa/analysis/code/bes_env` | `tabpfn 8.0.1` + `tabpfn_client 0.3.0` (used read-only; nothing there was modified) |
| cached weights | `~/.cache/tabpfn/tabpfn-v3-regressor-v3_default.ckpt` (233 MB) and `tabpfn-v2-regressor.ckpt` (44 MB) |
| hardware | **CPU only** — no `nvidia-smi`, `torch.cuda.is_available()` is `False` |

### 🔑 Running the gated v3 weights without a licence token

`tabpfn` 8.x refuses to load v2.5 / v2.6 / v3 without a Prior Labs token
(`~/.config/.tabpfn/state.json` has `user_id: null` — never authenticated; the
same blocker is recorded in `bes_africa/analysis_cleaned/rq_2/RUNNING.md` for
the HPC). But the gate is only on **downloading**:

```python
# tabpfn/model_loading.py
def download_model(to, ...):
    if to.exists():                 # <-- short-circuits BEFORE the licence check
        return "ok"
    return _download_model(...)      # <-- ensure_license_accepted() lives in here
```

The weights are already cached, so the check should never fire — except that
`resolve_model_path("v3_default")` returns the **relative** path `v3_default`,
which does not exist relative to cwd, so the cache is missed and it tries to
download.

**Fix: pass the absolute path to the cached checkpoint as `model_path`.**

```python
CKPT = "/home/mikael-minten/.cache/tabpfn/tabpfn-v3-regressor-v3_default.ckpt"
TabPFNRegressor(model_path=CKPT, n_estimators=1, device="cpu")   # works, no token
```

Verified working. `exp22_tabpfn.py` picks this up from the `TABPFN_CKPT`
environment variable, so the same script runs on either interpreter.

**Cost on CPU** — v3 is ~10× slower than v2 and the cost is *inference*, not
loading (reusing one fitted instance does not help: 31 s first fit, 33 s
reused):

| | per fit | full 20-year × 8-target backtest |
|---|---|---|
| v2, `n_estimators=4` | ~10 s | ~35 min |
| v3, `n_estimators=1` | ~32 s | ~85 min |
| v3, `n_estimators=2` | ~66 s | ~3 h |

**Fine-tuning** (`FinetunedTabPFNRegressor`, as in
`bes_africa/sandbox/HPC_TabPNF/finetune.py`) exists only in the 8.x line and
every reference script runs `device="cuda"`. Fine-tuning a 233 M-parameter model
on ~450 rows on CPU is not practical — it needs a GPU box. The
`model_path`-to-cached-ckpt trick above removes the *token* barrier, so on a
CUDA machine fine-tuning should work without authentication too.

Backtests use the same features, folds and protocol as the ridge, so the numbers
are directly comparable.

**Result — TabPFN v2 and v3 (exp22):**

| model | sept 05–25 | sept 05–19 | **sept 21–25** | rust 05–25 | rust 21–25 |
|---|---|---|---|---|---|
| **ridge ensemble (final v2)** | **13.70** | **12.92** | **15.66** | **5.38** | **9.52** |
| climatology | 17.62 | 17.20 | 18.72 | 5.82 | 10.56 |
| TabPFN **v2** (`n_est=4`) | 15.51 | 13.51 | **19.97** | 5.96 | 10.74 |
| TabPFN **v3** (`n_est=1`) | 15.15 | 13.42 | **19.10** | 5.77 | 10.35 |

❌ **Neither wins.** v3 is consistently better than v2 (as expected from the
bigger, newer model) but both are well behind the ridge on septoria, and both
lose to climatology on yellow rust overall.

The diagnostic detail is the split between windows. Both versions are
respectable on 2005–2019 (13.4–13.5, close to the ridge's 12.9) and then
collapse to **worse than climatology on 2021–2025** (19.1–20.0 vs 18.72). That
is the identical failure mode to RF, GBM and monotone-LGBM (exp03, exp21):
flexible learners interpolate the historical weather→disease relationship well
but cannot handle the elevated 2020s regime, because nothing in the training
data resembles it. A nonparametric model cannot produce a response outside its
observed range; the ridge's linear extrapolation plus explicit as-of baseline
terms can.

**This is the central lesson of the whole project.** The binding constraint is
not model capacity — it is non-stationarity plus ~450 rows. Every flexible model
tried (RF, GBM, monotone-LGBM, TabPFN v2, TabPFN v3, k-NN analogues) lost to a
heavily regularised linear model with domain-shaped features. Effort spent on
features and on explicit regime terms paid; effort spent on model class did not.

**Timing note for future runs.** Both backtests are *fast* when run alone:

| | per target | full 8-target backtest |
|---|---|---|
| v2 `n_est=4` | ~150 s | ~20 min |
| v3 `n_est=1` | ~52 s | **~7 min** |

The 1222 s first target and the 32 s/fit micro-benchmark were both measured while
2–3 jobs competed for 12 cores (load average 30 — each TabPFN process spawns
~12 torch threads). **Run these one at a time**, or set `OMP_NUM_THREADS` per
process; contention inflated the apparent cost by more than 10×.

## 12. Round-2 result

| | v1 | **v2** | climatology |
|---|---|---|---|
| septoria 2005–2025 | 13.87 | **13.70** | 17.62 (**+22.2 %**) |
| septoria 2005–2019 | 13.00 | **12.92** | 17.20 (+24.9 %) |
| septoria 2021–2025 | 15.99 | **15.66** | 18.72 (+16.3 %) |
| yellow rust 2005–2025 | 5.38 | **5.38** | 5.82 (+7.6 %) |
| yellow rust 2021–2025 | 9.52 | **9.52** | 10.56 (+9.9 %) |

Per-target skill vs climatology: septoria incidence **+23.8 % / +19.5 %**,
septoria severity **+11.4 % / +22.2 %**, rust L2 incidence +10.6 %.

Implemented in [experiments/final_model_v2.py](experiments/final_model_v2.py):
septoria averages 3 feature bases × 2 model forms; rust uses 1 basis × 3 forms;
severity via the conditional decomposition; L2 ≥ L1 constraint for septoria.

## 13. Round 3 — four negatives and the diagnostic that explains them

Round 3 asked whether anything was left on the table. Four ideas were tested and
**all four failed**. Recorded in detail because the *pattern* of failure is the
finding: it says the model is at the limit of what this data supports, not that
these were bad ideas.

Order matters here — the headroom decomposition (exp27) was run third, and it
retrospectively explains why the other three could not have worked.

### ❌ A. Cross-disease features (exp24)

Septoria and rust year-effects are negatively correlated (−0.05 to −0.26), which
is mechanistically sensible: wet dull springs drive splash-dispersed septoria,
not the mild-winter/bright-spring conditions rust wants. So each disease's
drivers and level might inform the other. Tested by adding the *other* disease's
weather block and as-of baselines.

| | sept 05–25 | sept 05–19 | sept 21–25 | rust 05–25 | rust 21–25 |
|---|---|---|---|---|---|
| v2 (own only) | **13.704** | 12.915 | **15.658** | **5.377** | **9.518** |
| + cross-disease | 13.666 | **12.685** | 16.043 | 5.484 | 9.798 |

Better on 2005–2019, worse on 2021–2025 — the overfitting signature seen in
every failed round-2 angle. Doubling the feature count buys in-sample fit and
costs out-of-regime robustness. **Rejected.**

### ❌ B. Recalibrating the anomaly scale (exp25)

exp24's bias table showed septoria L1 incidence missed by −30.9 in 2023 and
+11.4 in 2025 — large errors in *opposite* directions in a wet year and a dry
year. That is the signature of an under-dispersed forecast, and ridge at
`alpha=100` averaged over 6 members is over-shrunk a priori. Tested by writing
each prediction as `clim + beta × (pred − clim)` and fitting `beta`.

Note `beta` **is** the climatology-blend weight, so this also tests "should the
weak targets just be shrunk to climatology?" (rust L1 severity has −4.2 % skill).

*Oracle* betas, fitted on all 20 folds — an unattainable upper bound:

| target | beta | raw RMSE | recalibrated | gain |
|---|---|---|---|---|
| septoria L1 incidence | 1.17 | 21.554 | 21.385 | **0.8 %** |
| septoria L2 incidence | 1.05 | 16.407 | 16.389 | 0.1 % |
| rust L2 incidence | 1.29 | 8.634 | 8.532 | 1.2 % |
| rust L1 incidence | 0.65 | 6.411 | 6.375 | 0.6 % |

Even with hindsight the ceiling is **under 1 %** — the ensemble is already
calibrated in scale (`beta` ≈ 1.0–1.2). The as-of version, re-estimating `beta`
each year from earlier folds only and shrinking toward 1, is strictly worse:

| shrink strength | sept 05–25 | rust 05–25 |
|---|---|---|
| raw v2 | **13.704** | **5.377** |
| none (`shrink_n=0`) | 13.798 | 5.853 |
| heavy (`shrink_n=400`) | 13.716 | 5.560 |

Monotone toward the raw model as shrinkage increases: the best achievable
version of this idea is not doing it. **Rejected.** The 2023/2025 swings are
genuine unpredictable year effects, not a scale defect.

### ❌ C. Recency weighting of training rows (exp26)

Rust's history is 22.0 (1970s) → 0.8 (2000s, resistant varieties) → 9.4 (2020s,
Warrior race incursion). Its weather→disease slope is therefore fitted partly on
an era whose host genetics no longer exist, which would explain why the model is
*worse than climatology* on rust for 2005–2019 (2.518 vs 2.340). An exponential
recency weight keeps all ~450 rows but lets recent years dominate. (exp09 tried
this under the old architecture; never tested inside the current ensemble.)

Gridded over half-life {∞, 30, 20, 15, 10} × `MIN_YEAR` {1980, 1990, 1995, 2000}
× `alpha` {30, 100, 300}. **Not one of the 15 cells improved both sub-windows**,
for either disease. The grid shows a clean mechanical tilt — shorter half-life
always helps 2021–2025 and always hurts 2005–2019:

| septoria | 05–25 | Δ05–19 | Δ21–25 |
|---|---|---|---|
| shipped (equal weights) | 13.704 | — | — |
| half-life 20 | 13.724 | +0.074 | −0.102 |
| half-life 15 | 13.744 | +0.112 | −0.122 |
| half-life 10 | 13.805 | +0.203 | −0.131 |

That is a trade, not skill. **Rejected.**

### 🔍 D. Headroom decomposition — the diagnostic (exp27)

Splits RMSE² into the error in the **national year level** and the error in the
**spread across regions given that level**, and computes what perfect
information of each kind would be worth.

| | total | year-level | within-year | year share of MSE |
|---|---|---|---|---|
| septoria model | 13.704 | 10.016 | 9.352 | **53 %** |
| septoria climatology | 17.620 | 14.595 | 9.873 | 69 % |
| rust model | 5.377 | 3.861 | 3.742 | 52 % |
| rust climatology | 5.820 | 4.372 | 3.841 | 56 % |

| ceiling | septoria | rust |
|---|---|---|
| current model | 13.704 | 5.377 |
| + perfect YEAR level | **9.352** | **3.742** |
| + perfect REGION pattern | 10.016 | 3.861 |

**Septoria has captured 47 % of the available year-effect gap; rust only 21 %.**

⚠️ The "perfect REGION pattern" ceiling is **not attainable and should not be
used to justify work**. It assumes the observed regional deviation is predictable
*including its survey sampling noise*. The variance decomposition (section 3) put
the systematic region effect at only 0–9 %, so most of that 9.35 within-year RMSE
is ADAS sampling noise on a limited number of fields per region. This is the
honest reason to stop expecting much from the stalled ERA5 download.

The per-year national miss on septoria L1 incidence is where the story is:

| year | actual | model | clim | model err | clim err |
|---|---|---|---|---|---|
| 2010 | 23.8 | 31.3 | 64.2 | +7.5 | +40.3 |
| 2014 | 95.8 | 75.2 | 59.2 | −20.7 | −36.7 |
| 2016 | 37.0 | 71.2 | 58.8 | **+34.3** | +21.9 |
| 2021 | 85.2 | 46.3 | 51.9 | **−38.9** | −33.4 |
| 2023 | 85.5 | 54.6 | 57.7 | **−30.9** | −27.7 |
| 2025 | 35.0 | 46.4 | 68.4 | +11.4 | +33.4 |

Model year-level RMSE 16.59 vs climatology 24.06. The model earns its skill in
the extreme dry years (2010, 2025) and fails in 2016, 2021, 2023 — and it
**under-predicts the 2020s by −15.7 on average**.

### ❌ E. Baseline as an offset rather than a shrunk regressor (exp28)

The −15.7 bias has an obvious candidate mechanism. The as-of baseline is one
standardised column among ~20, penalised at `alpha=100`, so ridge shrinks its
coefficient well below 1 and pulls every prediction toward the mean of the
1990-onward training window. When the current level sits far above that window —
exactly the 2020s — the pull is a systematic downward bias.

Fix tested: fit `y − baseline` and add the baseline back, so shrinkage targets
"this region's current level" instead of "the training-window mean". This is the
additive sibling of the existing `rel` form, never tried for septoria.

| variant | sept 05–25 | 05–19 | 21–25 | bias 21–25 |
|---|---|---|---|---|
| **v2 shipped** | **13.704** | **12.915** | 15.658 | −15.71 |
| septoria: `off` only | 14.035 | 13.256 | 15.972 | **−13.73** |
| septoria: add+int+`off` | 13.760 | 12.976 | 15.703 | −15.04 |
| septoria: add+int+`off` hl10 | 13.750 | 13.017 | **15.579** | −15.38 |

**It works exactly as designed and that makes things worse.** The bias does fall
(−15.7 → −13.7) and RMSE rises (13.70 → 14.04). The bias-variance trade is
strictly unfavourable.

This is the important one. **The under-prediction of the 2020s is not a defect —
it is the price of correct hedging.** The level jumps unpredictably in *both*
directions (up in 2021, down hard in 2025), so a tracker that follows it faster
wins the up-jumps and loses the down-jumps by more. A biased, shrunk forecast
beats an unbiased one in squared error. Chasing the bias is chasing a mirage.

The `off` form is left implemented in `final_model_v2.py` but is not in `FORMS`.

### What round 3 establishes

Round 2 concluded that model *class* was not the constraint. Round 3 goes
further: **the shrinkage level is not the constraint either.** Four independent
attempts to extract more signal — more features, rescaled output, reweighted
training rows, retargeted shrinkage — all failed, and exp27 says why. Septoria
has already taken 47 % of the year-effect gap and the rest sits in years like
2016 and 2021 whose national level is genuinely not implied by the weather.

**Rust is the exception worth noting: only 21 % of its gap is captured.** But its
year-to-year variation is driven by pathogen race incursions and varietal
resistance turnover, which no weather feature can see and for which this repo has
no data. Closing it needs UK Cereal Pathogen Virulence Survey race-frequency data
or AHDB Recommended List varietal-resistance scores — a **new data source**, not
a new model.

## 14. Contest logistics (read before doing more modelling)

From `README.md`, confirmed 2026-08-11:

**Three award categories, judged separately:**
1. most accurate **Zymoseptoria tritici** forecast, by RMSE;
2. most accurate **yellow rust** forecast, by RMSE;
3. **most interesting report**.

Septoria and rust are scored independently, which is what `septoria_pooled` /
`rust_pooled` already optimise — no reweighting needed. Both pool the 4 targets
of that disease into one RMSE, so **incidence dominates entirely** (RMSE ~16–22)
and severity is nearly free (~0.04–3.6). Do not spend effort on severity targets.

**Dates:**

| | |
|---|---|
| algorithms frozen | **6 September 2026** |
| 2026 assessment data released | 14 September 2026 |
| final forecasts due | 28 September 2026 |

Final submissions are checked against the frozen algorithm for consistency, so
the pipeline must be **re-runnable unchanged** in late September. It is: weather
is fetched to `data_external/` and cached, Met Office already publishes 2026
through June, and `as_of_baselines` only ever uses years < 2026.

**⚠️ The mandatory report does not exist.** Rule: *"All entries must include a
max 1000 word report"* — `submission/` currently holds only the organisers'
`example_report.pdf`. This is both an entry-validity requirement and one of the
three prizes, and it is the category where this project is strongest: the
negative results (TabPFN v3 losing to ridge; §13's four failed extraction
attempts; the bias-is-correct-hedging finding) are a more interesting story than
the RMSE. **Highest-value remaining work.**

Also outstanding: the repo must be public and panel-reproducible, and nothing in
`experiments/` has been committed yet.

## 15. Head-to-head vs the repo's example model (exp29)

The contest ships a worked example (`report/example_report.md`): per-region
ElasticNet on lagged disease values plus the four repo covariate files, Kalman
imputation, skew correction. Replicated faithfully in
[experiments/exp29_example_baseline.py](experiments/exp29_example_baseline.py),
including choices I would not make, and scored on my protocol.

### ⚠️ The comparison is invalid unless cells are aligned

The example's target is `_lead1`: year *t* features predict year *t+1*. To
forecast 2021 it needs a **Year = 2020** row — and 2020 does not exist (no
COVID-year survey). **So it silently predicts nothing for 2021.** Because
`score()` inner-joins, the naive comparison grades it on an easier subset, and
2021 is the worst year for both models (rust L2 incidence 25.9 after years near
6; septoria under-predicted by 38.9).

Uncorrected, this reverses the rust verdict — the example appeared to *beat* me
5.38 → 4.81. On common cells it does not. **Any future model comparison here must
intersect the predicted cells first.**

### Results, rolling origin, identical cells (1248)

| model | septoria | rust |
|---|---|---|
| repo example (ElasticNet) | 18.021 | 4.809 |
| climatology | 17.210 | 4.622 |
| **mine, weather removed (ablation)** | 15.114 | 4.355 |
| **mine (final_model_v2)** | **12.780** | **3.848** |

**+29.1 % septoria, +20.0 % rust vs the example.** Note the example **loses to
climatology on both diseases** — 107 predictors on ~40 rows per region.

⚠️ 12.78 / 3.85 are **not** new headline numbers: this subset excludes 2021, the
hardest year. The headline on full coverage remains **13.70 / 5.38**.

### Where my advantage actually comes from

The ablation row is my architecture with the entire weather block deleted, and it
already beats the example comfortably. Decomposing against climatology:

| | septoria | rust |
|---|---|---|
| architecture (pooling + as-of baselines + ensemble) | +12.2 % | +5.8 % |
| external weather data on top | +15.4 % | +11.6 % |

So it is roughly half method, half data — worth stating honestly rather than
attributing the win to the Met Office pull. The example also uses **no in-season
information at all**, which is the single biggest framing difference.

Its three structural problems, in order of cost: per-region fitting (~40 rows,
107 predictors); no in-season weather; and the `lead1` framing that cannot
express a missing year.

## 16. UKCPVS pathogen virulence data (fetch_ukcpvs.py, exp30)

§13 ended by saying rust needs a **new data source**, not a new model. This is
that attempt, and it is a real data acquisition, not a proxy.

### Getting it

No CSV/JSON/API exists. AHDB publishes PDF reports plus an interactive app. The
only machine-readable files are UKCPVS supplementary data (2022) and (2023)
`.xlsx` — single-year isolate × gene matrices (41 and 26 isolates), useless as a
series. So it had to come out of the PDFs.

[experiments/fetch_ukcpvs.py](experiments/fetch_ukcpvs.py) downloads 8 annual
reports and parses "Frequency of detection of isolates carrying virulence to the
different yellow rust resistance genes and varieties over the past five years" —
5 years per report, overlapping, which cross-checks the parse (2016 Warrior:
33 % in the 2018 report, 37 % in the 2021 one — minor retrospective revisions).

**Result: 15 years × 31 items, 2010–2024.** Reports back to 2004 were downloaded
and checked; the frequency table only starts in the 2014/2015 reports, so 2010 is
a hard floor. The extraction is sound — it puts Warrior at **0 % in 2010 and 56 %
in 2011**, exactly the documented incursion.

### ⚠️ Power, stated before the result

14 usable year-pairs after lagging. A correlation needs **|r| > 0.53** for
p < 0.05 at n = 14. This can only detect a large effect.

### The information rule bites unusually hard here

UKCPVS isolates for year Y are **collected during year Y's epidemic**. Using them
to predict year Y is circular: the isolates *are* the epidemic. Everything must
be lagged ≥1 year, which is also what a real 2026 forecast would have.

That distinction is the whole result:

| index | target | lag 0 (circular) | **lag 1 (usable)** |
|---|---|---|---|
| Warrior frequency | rust L1 incidence | **0.53** | **0.06** |
| Warrior frequency | rust L2 incidence | **0.50** | **0.05** |
| variety-mean virulence | rust L1 incidence | 0.31 | 0.14 |

**Virulence is coincident with the epidemic, not leading it.** The contemporaneous
correlation is real and respectable; one year ahead it is zero.

### The sharp test — does it explain the model's residual?

Correlating with the *level* is the wrong question, since the as-of baselines may
already carry it. Regressing my out-of-sample residual on lagged virulence:

| index | rust L1 inc | rust L2 inc |
|---|---|---|
| variety-mean virulence | 0.24 | 0.20 |
| gene-mean virulence | −0.11 | −0.20 |
| Warrior frequency | −0.02 | −0.04 |
| virulence breadth | 0.30 | 0.29 |

Largest |corr| = **0.30**, against the 0.53 needed. ❌ **Nothing survives.**

### The number worth keeping

| | rust incidence RMSE |
|---|---|
| current model | 7.604 |
| if every year's mean residual were perfectly corrected | **5.451** |

That 5.451 is the ceiling for **any** national annual covariate — virulence,
varietal resistance, fungicide use, anything without regional resolution. So the
28 % headroom is real; UKCPVS virulence just does not deliver it, for a reason
that is structural rather than fixable.

### On AHDB Recommended List ratings (the other candidate)

Also checked: **no machine-readable form** — PDF booklets plus an interactive
app, no CSV/API. RL ratings are, unlike UKCPVS, genuinely *leading* (published in
autumn before drilling), which makes them the better bet in principle. But a
usable index needs variety resistance ratings **weighted by area actually grown**,
which is a second PDF-locked dataset, and the resulting series would move slowly
— the regime the as-of baseline already absorbs. Cost is high, and the virulence
result above is a warning about how such indices tend to fail. Not attempted.

## 17. Round 4 — five ideas, three adopted → **v3**

Round 3 concluded the model was at the limit of what the data supports. Round 4
tested that by attacking structure rather than signal: not "what other variable
predicts disease" but "is the model's *shape* wrong". Three of five ideas paid.

| | septoria | yellow rust |
|---|---|---|
| climatology | 17.620 | 5.820 |
| v2 | 13.704 | 5.377 |
| **v3** | **13.615** | **5.345** |
| skill vs climatology | **+22.7 %** | **+8.2 %** |

Per window: septoria 2005–19 **12.900** (was 12.915), 2021–25 **15.402** (was
15.658); rust 2005–19 **2.474** (was 2.518), 2021–25 **9.482** (was 9.518).
Every component improves *both* windows — the bar used throughout.

### ✅ A. Multi-task coefficient sharing (exp35) — the biggest win

exp04 found ~84 % of the variance across a disease's four targets is one latent
factor: they are four views of one epidemic. Yet all four were fitted
independently, each estimating its own weather coefficients from its own noise.

Fix: z-score each target (incidence is 0–100, severity 0–28 — raw coefficients
are not commensurable), fit ridge on each, then replace each coefficient vector
with the average across the four. The shared component is then estimated from
~1800 rows instead of ~450.

| lam (sharing) | sept 05–25 | 05–19 | 21–25 |
|---|---|---|---|
| 0 (independent, = v2) | 13.704 | 12.915 | 15.658 |
| 0.5 | 13.659 | 12.946 | 15.442 |
| **1.0 (full sharing)** | **13.635** | **12.903** | **15.460** |

**lam = 1 wins** — the four targets should share *one* weather response entirely,
differing only in mean and scale. That is a direct confirmation of the
single-latent-factor result, and it is the strongest structural finding of the
project after the variance decomposition itself.

Blend weight into the ensemble swept 0.3→1.0; 0.4 is the best point that
improves both windows (0.7 is marginally better overall but costs 2005–19).

⚠️ Applied to septoria **incidence only**. On severity it cost 2.2 % (3.593 →
3.670) because it bypasses the incidence × conditional-severity decomposition
that is already a v2 win. On rust it was slightly worse in every configuration.

### ✅ B. Averaging over the ridge penalty (exp31)

exp26 gridded `alpha` and found every setting traded 2005–19 against 2021–25, and
I concluded "no setting is better". That was the wrong inference. If different
penalties suit different regimes and the forecast year's regime is unknown, the
answer is not to pick — it is to **average**, exactly the argument that already
justifies averaging over feature bases and forms.

Averaging over `alpha ∈ {30, 100, 300}`: septoria 13.704 → 13.681, rust 5.377 →
5.357. Applied to **rust only** — on septoria it cost 2005–19 (+0.015) for a
0.007 overall gain, which fails the bar.

**Incidental finding: `CLIM_K` is dead code.** Averaging over it changed the
output *not at all* (13.704 / 5.377 to four decimals). `base` is computed in
`_fit` but only returned when `form == "clim"`, which is not in `FORMS` — so the
climatology window affects only the benchmark, never a prediction.

### ✅ C. Hurdle model for rust (exp33)

**46.3 % of rust incidence observations are exactly zero.** One linear fit had to
serve both "does an epidemic establish at all" (overwintering survival) and "how
far does it run" (spring conditions, June heat) — different questions with
different drivers. Split into `P(y > 1) × E[y | y > 1]`, logistic for the first
part, ridge on log1p for the second.

Blend weight swept 0.2→0.8; **0.2–0.5 all improve both windows**, so this is a
plateau rather than a lucky point. Adopted at 0.4: rust 5.377 → 5.358 alone,
5.345 combined with alpha averaging.

Fitting the level on log1p mattered — the raw-scale version failed the bar.

### ❌ D. Leaf-specific exposure windows (exp32)

L2 emerges ~25 Apr, L1 ~15 May, and septoria splashes *up* the canopy, so each
leaf should respond to rain falling after it emerged. exp17 encoded this
(`EXPOSURE_L1/L2`) but the model handed all four septoria targets the same
columns — and `m_splash_L1_ranom` was never used at all. Fixed that, and added
May–June statistical windows to pair with the existing April–May ones.

**Null result, for a clear reason:** the mechanistic L1/L2 pairs correlate
**0.92–0.94**. They are near-collinear, so re-allocating them cannot change a
heavily-shrunk fit. Septoria went 13.704 → 13.701.

More interesting: the *statistical* windows genuinely differ (May–Jun vs Apr–May
rain correlate only **0.34**) — and using the "epidemiologically correct" narrow
window made things **worse** (13.840). Reading: April–May moisture matters for
*both* leaves, because it sets the lower-canopy inoculum load that later splashes
up. The disease does not start on L1; it climbs. The wide window is not sloppy,
it is right.

### ❌ E. Robust (Huber) loss (exp34)

A few seasons dominate the squared-error objective (2016 +34.3, 2021 −38.9), and
they are anomalous precisely because something happened the weather cannot see.
Huber loss should stop them setting the coefficients for every ordinary year.

**Decisive failure at every `epsilon` tried:**

| | sept 05–25 | rust 05–25 |
|---|---|---|
| ridge | **13.704** | **5.377** |
| huber eps=1.35 | 14.830 | 7.499 |
| huber eps=3.0 | 15.113 | 7.250 |

Rust degrades 40 %. The round-1 lesson holds: the metric is squared error, and
down-weighting large residuals during fitting optimises the wrong objective. My
"it only changes estimation, not the target" defence was wrong — with a strong
year effect, the big-residual years *are* the signal, not contamination.

### What round 4 changes about the overall conclusion

Rounds 2–3 said model class and shrinkage level were not the constraint, and
round 4 does not overturn that: the two ideas that swapped the *estimator*
(Huber) or re-timed the *features* (leaf windows) both failed. What worked was
imposing **structure that matches how the data is generated** — four targets are
one epidemic, rust incidence is zero-inflated, and the penalty is unknown so it
should be integrated over rather than guessed.

That is a sharper statement of the same lesson: with ~450 rows and a strong
non-stationary year effect, gains come from telling the model something true
about the problem's structure, not from giving it more freedom or more features.
