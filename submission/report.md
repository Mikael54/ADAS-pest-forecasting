# Forecasting wheat disease incidence and severity in the UK, 2026

Author: Mikael Minten. August 2026.
Code in `experiments/`. Full experiment log in `EXPERIMENTS.md`.

## Summary

This report forecasts eight targets for nine UK regions: incidence and severity
of *Zymoseptoria tritici* and yellow rust on wheat leaves 1 and 2. The final
model is an ensemble of ridge regressions pooled across regions, driven by
observed Met Office weather and by strictly backward-looking disease baselines.
On a 2005 to 2025 rolling-origin backtest it improves septoria RMSE by 22.7% and
yellow rust RMSE by 8.2% against a recent-climatology benchmark.

## Data

I did not use the supplied `bioclim_data.csv`. It contains HadGEM2 (CMIP5) model
output rather than observations, and its regional series correlate between 0.90
and 1.00, so it carries almost no cross-region signal. In its place I used the
Met Office HadUK-Grid areal series: monthly rainfall, rain days, sunshine,
temperature and air frost for seven climate districts, from 1836 to the present.
Observations for 2026 through June are already published, which is what makes a
2026 forecast possible from real data.

The information rule is that predicting year Y may use weather from September of
year Y-1 to June of year Y, and disease observations up to year Y-1. June is the
cut-off because the ADAS survey scores leaves at GS71 to GS75 in late June and
July.

## Why the model has this shape

A variance decomposition determined the design. Of the variance in each target,
51% to 79% is a national year effect and only 0% to 9% is a region effect. An
oracle given the perfect regional pattern but no year information scores 16.7 on
septoria, barely better than climatology's 17.6. Almost all the winnable error
sits in the year effect.

Two consequences follow. The task is a national weather-driven year-level model
distributed across regions, not nine regional models, because per-region fitting
gives roughly 40 training rows against more than 100 candidate predictors. And
persistence is worthless: last year's value is the weakest baseline I tested
(septoria RMSE 25.2), because disease has almost no year-to-year carry-over.

## The model

For each target, ridge regression pooled over all nine regions:

> y ~ weather block + as-of disease baselines + year trend + region dummies

Predictions are averaged over three feature bases (anomalies against a fixed
1961 to 2000 climatology, anomalies against a trailing 30-year climatology, and
a mechanistic block) and over two or three model forms: purely additive, an
additive form with drought by baseline interactions, and for rust a
multiplicative form. These components fail in different eras, so averaging beats
selecting.

The mechanistic block encodes the epidemiology directly. It combines degree-day
infection cycles (about 300 °C·d base 0 °C for septoria, about 140 °C·d base
3 °C for rust) with rain-splash event counts inside leaf-exposure windows, plus
a Liebig limiting-factor index taking the minimum of the thermal and moisture
terms. Its strongest single feature correlates 0.68 with national septoria
severity.

Four further structural elements each earned their place on the backtest:

- severity is modelled as (incidence/100) times conditional severity, which
  improved all four severity targets over fitting severity directly;
- multi-task coefficient sharing across a disease's four targets. About 84% of
  their joint variance is a single latent factor, and forcing them to share one
  weather-response vector outright beat independent fitting;
- a hurdle decomposition for rust incidence, P(present) times E[level given
  present], because 46% of rust observations are exactly zero and one linear fit
  cannot serve both "does an epidemic establish" and "how far does it run";
- averaging over the ridge penalty, since different penalties suit different
  eras and the forecast year's era is unknown.

Leaf 2 is constrained to carry at least as much septoria as leaf 1, which holds
in 97.8% to 99.3% of observed rows. I did not impose the constraint for rust,
where it holds in only 75% to 78%.

## Validation

All figures come from rolling-origin backtesting: for each evaluation year T the
model is refitted on years strictly before T. Evaluation spans 2005 to 2025,
excluding 2020, for which no survey exists. Because the disease regime shifted in
the 2020s, every candidate change had to improve both the 2005 to 2019 and the
2021 to 2025 sub-windows before I adopted it. That rule rejected most of what I
tried.

| Pooled RMSE, 2005 to 2025 | Septoria | Yellow rust |
|---|---|---|
| Recent climatology | 17.62 | 5.82 |
| Repo example pipeline | 18.02 | 4.81 |
| This model | 13.62 | 5.35 |

The repo example is scored only on the cells it covers. It cannot produce a 2021
forecast at all, because its year-t to year-t+1 framing requires a 2020 row.

## 2026 forecast

Spring 2026 was dry and exceptionally bright, with East Anglia recording 4.0 mm
of rain in April, following a very mild and largely frost-free winter, and then a
hot June. That combination suppresses splash-dispersed septoria while favouring
rust. Forecast national means for 2026 are septoria incidence 60.6 on leaf 1 and
78.5 on leaf 2, and rust incidence 9.1 on leaf 1 and 20.4 on leaf 2.

## What did not work

Model capacity was never the binding constraint. Random forests, gradient
boosting, monotone-constrained boosting and TabPFN v2 and v3 all lost to
regularised ridge, and all failed the same way: respectable on 2005 to 2019,
worse than climatology on 2021 to 2025. With roughly 450 rows and a
non-stationary year effect, flexible learners cannot extrapolate to a regime with
no precedent in the training data. UKCPVS pathogen virulence data, which I
extracted from annual report PDFs, also failed, because it is coincident with the
epidemic rather than leading it.

## Reproducibility

Running `cd experiments && ../.venv/bin/python final_model_v2.py` fetches and
caches the weather, refits the model, and writes both submission files. All
experiments, including the failures, are documented in `EXPERIMENTS.md`.
