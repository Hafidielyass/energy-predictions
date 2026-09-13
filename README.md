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
| Ridge | 2.860 | 4.236 | 0.9478 | −1.06% |
| LightGBM (tuned) | 3.011 | 4.403 | 0.9436 | −5.06% |
| Clear-sky persistence | 14.726 | 22.845 | −0.519 | −445% |

Probabilistic, same period:

| | Coverage (target 80%) | Mean width | Mean pinball |
|---|---|---|---|
| LightGBM quantile | 83.3% | 13.00% | 1.029 |
| Elia's own band | 82.1% | 12.56% | 0.947 |

**The models do not beat Elia.** They land within a few percent of a utility's
production system and the uncertainty bands are properly calibrated, but the
honest headline is that a single free weather grid point and 2.6 years of
history do not outperform a TSO's operational forecasting stack.

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

## Known limitations

- One Open-Meteo grid point at the Belgian centroid represents a fleet spread
  across the whole country. Spatially distributed forecasts would likely close
  much of the gap to Elia.
- Grid-aggregate data, so no module temperature or DC power. Cell temperature
  is estimated from air temperature and irradiance via a NOCT-style model.
- ~7,500 daylight training rows is thin for the model class.
