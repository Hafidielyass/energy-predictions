# Day-Ahead Solar PV Forecasting

Forecasts Belgian grid-scale photovoltaic generation 24 hours ahead from
numerical weather predictions, with calibrated P10/P50/P90 uncertainty bands.

The benchmark throughout is Elia's own operational day-ahead forecast, which
ships in the same dataset — so every number below is measured against what a
transmission system operator actually runs in production, not against a
strawman baseline.

## Data

| Source | What it provides | Cadence |
|---|---|---|
| [Elia ODS032](https://opendata.elia.be/explore/dataset/ods032/) | Measured Belgian PV output **and** Elia's own day-ahead forecast on the same row, plus installed capacity | 15 min |
| [Elia ODS087](https://opendata.elia.be/explore/dataset/ods087/) | Same fields looking forward up to a week — used only at serving time | 15 min |
| [Open-Meteo previous-runs](https://open-meteo.com/en/docs/previous-runs-api) | Archived NWP at a fixed one-day lead: irradiance, temperature, cloud, wind | hourly |

Weather is sampled at **all eleven Belgian provinces** and averaged by each
one's installed PV capacity, so the forecast reflects conditions where the
panels actually are rather than at an arbitrary national centroid. The spread
of irradiance between provinces is kept as its own feature: when it is high the
country is partly clouded and aggregate output is smoother than any single
location would imply.

Because the weather columns are *archived forecasts* rather than observations,
the day-ahead framing is leak-free by construction — the model only ever sees
what was genuinely knowable the day before.

**Usable window: 2024-01-19 → present** (~2.6 years). Elia reaches back to
2020, but Open-Meteo's forecast archive starts in 2024, and that is the binding
constraint. Rows without an archived forecast are dropped, never imputed.

## Pipeline

Stages depend in one direction only — later stages import earlier ones.

```
src/
├── data/           fetch Elia + Open-Meteo, align, join, integrity-check
├── preprocessing/  chronological and rolling-origin splits
├── features/       solar geometry, clear-sky reference, physics, lags
├── models/         tuning, training, quantiles, walk-forward backtest
├── evaluation/     point and probabilistic metrics, skill scores
└── deployment/     live forecast for a future date
```

```bash
pip install -r requirements.txt

python -m src.data.build_dataset      # fetch + join + integrity report
python -m src.features.build_features # 27 features
python -m src.models.tune             # hyperparameters, CV folds only
python -m src.models.train            # fit + single test evaluation
python -m src.models.quantile         # P10/P50/P90
python -m src.models.backtest         # retraining-policy comparison
python -m src.deployment.predict      # live forecast for tomorrow
```

## Results

Test period is a full 12 months (2025-09 → 2026-08) never seen during
development. Errors are percentages of installed capacity, daylight hours only
— including night would add thousands of trivially correct zeros.

| Model | nMAE | nRMSE | R² | Skill vs Elia |
|---|---|---|---|---|
| **Elia day-ahead** (baseline) | 2.788 | 4.191 | 0.9489 | — |
| Ridge | 2.832 | 4.197 | 0.9487 | −0.14% |
| LightGBM (tuned) | 2.949 | 4.340 | 0.9452 | −3.55% |
| Clear-sky persistence | 14.726 | 22.845 | −0.519 | −445% |

Probabilistic, same period:

| | Coverage (target 80%) | Mean width | Mean pinball |
|---|---|---|---|
| LightGBM quantile | 83.6% | 13.24% | 1.035 |
| Elia's own band | 82.1% | 12.56% | 0.947 |

**Ridge reaches parity with Elia; nothing beats it.** A −0.14% skill gap is
indistinguishable from a tie, and the quarterly breakdown shows why that
average is not the whole story — Ridge wins Q4 (+4.88%) and Q3 (+5.89%) but
loses Q1 (−6.20%). Matching a TSO's production forecast from free public data
is a reasonable outcome; claiming to beat it would not be supported.

> **Methodological caveat.** The test year was evaluated four times across the
> project as the data pipeline changed. Each change was selected on
> cross-validation rather than on test performance, but repeated looks still
> bias the estimate optimistically. Treat −0.14% as "roughly parity", not as a
> precise measurement.

## What the build surfaced

**Elia's forecast is a moving target.** Their error fell ~10% between the
development period and the test year — 19% year-on-year in Q1, 15% in Q4. A
model trained to correct their 2024 residuals is fixing mistakes they have
since stopped making, which is why cross-validation showed +2.3% skill and the
test year showed −5%.

**Retraining helps less than expected.** Walk-forward backtesting found an
expanding window (−2.61% vs Elia) beats both a static fit (−4.11%) and a
trailing 12-month window (−5.26%). With so little data, discarding history to
chase recency costs more than staleness does.

**Open-Meteo radiation is backward-averaged; Elia's is forward-labelled.**
Joining them naively offsets sunshine from power by a full hour. Detected by
scanning shifts for peak correlation, using Elia's own forecast as a control:
0.896 → 0.938 once corrected.

**48-hour lags are trainable but not servable.** The historical archive
publishes ~1.5 days in arrears while the near-real-time feed only reaches back
about a day, leaving a ~24 h blind spot. Only building the serving path exposed
it; the minimum lag is 72 h.

**Where the panels are beats where the country's middle is.** Moving from a
single centroid grid point to eleven capacity-weighted provincial points lifted
the weather/generation correlation from 0.938 to 0.958, and closed most of the
remaining gap to Elia — Ridge went from −1.06% to −0.14% skill. It was the
single largest improvement in the project, and it came from geography rather
than modelling.

## Known limitations

- Grid-aggregate data, so no module temperature or DC power. Cell temperature
  is estimated from air temperature and irradiance via a NOCT-style model.
- ~7,500 daylight training rows is thin for the model class.
- Provinces are sampled at their capitals, a proxy for the true
  capacity-weighted centroid of panels within each one.
- The probabilistic models reuse regularisation tuned for squared error rather
  than pinball loss, and did not improve when the point forecast did — tuning
  them on their own objective is the obvious next step.
