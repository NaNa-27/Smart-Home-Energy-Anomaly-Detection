# Data Understanding, Cleaning and EDA Report — Corrected Version

## 1. Project objective

The project analyzes one-minute smart-home electricity and weather measurements for Home C. Its practical goals are:

1. identify unusually high household load using a transparent statistical proxy;
2. determine when these events occur;
3. identify the appliance groups most associated with household energy use and anomaly periods;
4. examine weather-energy relationships;
5. provide consistent artifacts for model training and the Dash dashboard.

## 2. Research questions

- **RQ1:** At which hours, days and months are high-consumption proxy anomalies most frequent?
- **RQ2:** Which appliance groups account for the largest annual energy consumption and which groups dominate anomaly rows?
- **RQ3:** How strongly are weather variables associated with household consumption and solar generation?

## 3. Dataset and time reconstruction

The final compressed dataset contains **503,910 rows and 40 columns**. The raw `time` counter increases by one unit per record. Direct conversion as Unix seconds would incorrectly compress the data to roughly six days. Because the dataset design represents one observation per minute, the project reconstructs a regular one-minute timeline:

- start: `2016-01-01 00:00:00`;
- end: `2016-12-15 22:29:00`.

This is an explicit analytical assumption. The reconstructed dates are used for relative time-series analysis and chronological validation, not claimed as independently verified timestamps.

## 4. Cleaning and reproducibility corrections

The preprocessing pipeline performs:

- missing-value inspection and median/mode imputation when required;
- duplicate-row detection;
- duplicate-column detection;
- constant-column removal;
- time-feature engineering;
- appliance grouping;
- KPI, table and chart generation.

### Idempotence correction

Before rebuilding features, the script removes all previously engineered columns:

```text
hour, dayofweek, month, is_weekend, season, time_period,
Kitchen [kW], Furnace [kW], total_appliance
```

It then reconstructs them only from the original circuit columns. This prevents Furnace and Kitchen totals from being counted again when the script is rerun on an already processed dataset.

## 5. Correct treatment of units

The source fields are power readings in kW sampled once per minute. Summing them directly does not produce kWh. The corrected conversion is:

```text
Energy (kWh) = Σ Power (kW) × 1/60 hour
```

For monthly MWh:

```text
Energy (MWh) = Σ Power (kW) / 60 / 1000
```

All daily, monthly, appliance and KPI energy values in this revision use these conversions.

## 6. Verified descriptive results

| Indicator | Correct value |
|---|---:|
| Total household energy | 7,214.00 kWh |
| Average household power | 0.8590 kW |
| Peak household power | 14.7146 kW |
| Peak reconstructed time | 2016-07-30 16:04 |
| Total solar generation | 640.21 kWh |
| Proxy anomaly threshold | 4.0336 kW |
| Proxy anomaly rows | 12,418 |
| Proxy anomaly rate | 2.46% |
| IQR outlier rows | 34,211 |

The proxy anomaly label is defined as:

```text
use [kW] > mean(use [kW]) + 3 × std(use [kW])
```

It identifies high-load observations, not verified faults.

## 7. RQ1 — Temporal pattern of proxy anomalies

### Hour of day

The highest anomaly rates occur in the afternoon:

| Hour | Anomaly count | Rate within hour |
|---:|---:|---:|
| 15:00 | 1,417 | 6.75% |
| 16:00 | 1,224 | 5.83% |
| 17:00 | 1,217 | 5.80% |
| 14:00 | 1,096 | 5.22% |
| 18:00 | 1,079 | 5.14% |

The concentration from 14:00–18:00 suggests that high household loads are strongly associated with afternoon appliance use.

### Day of week

| Day | Anomaly count | Rate |
|---|---:|---:|
| Monday | 2,727 | 3.79% |
| Saturday | 2,022 | 2.81% |
| Friday | 1,938 | 2.69% |

Monday has the highest rate. Tuesday is the lowest at approximately 1.03%.

### Month

| Month | Anomaly count | Rate |
|---|---:|---:|
| August | 5,316 | 11.91% |
| July | 4,408 | 9.87% |
| September | 780 | 1.81% |

July and August account for most proxy anomalies. This temporal concentration also explains why a random train/test split produced unrealistically high scores: records from the same seasonal regime were mixed across training and testing.

## 8. RQ2 — Appliance contribution

### Annual measured appliance energy

| Appliance group | Energy | Share of grouped appliance energy |
|---|---:|---:|
| Furnace | 1,981.96 kWh | 39.41% |
| Home office | 682.69 kWh | 13.58% |
| Fridge | 533.78 kWh | 10.62% |
| Barn | 491.56 kWh | 9.78% |
| Wine cellar | 353.88 kWh | 7.04% |

Furnace is the dominant measured appliance group. The previous doubled value was caused by rerunning feature engineering on a file that already contained `Furnace [kW]`.

### Largest appliance reading during anomaly rows

Among 12,418 proxy-anomaly rows, the highest appliance-group reading is:

| Appliance group | Rows where it is largest |
|---|---:|
| Furnace | 10,332 |
| Barn | 791 |
| Dishwasher | 340 |
| Well | 267 |
| Living room | 204 |

This supports the descriptive conclusion that Furnace activity is the strongest appliance-level signature associated with the high-load proxy label. It does not prove a fault in the furnace.

## 9. RQ3 — Weather and energy

Daily-aggregated correlations show:

- household use has weak linear correlation with the measured weather variables;
- solar generation correlates positively with temperature (`r ≈ 0.356`);
- solar generation also correlates positively with apparent temperature (`r ≈ 0.355`);
- solar generation has a negative association with wind speed (`r ≈ -0.145`).

Average weather conditions differ between normal and anomaly rows:

| Variable | Normal rows | Anomaly rows |
|---|---:|---:|
| Temperature | 50.30 °F | 68.25 °F |
| Humidity | 0.663 | 0.693 |
| Cloud cover | 0.227 | 0.180 |
| Wind speed | 6.67 mph | 5.84 mph |

These are associations. Seasonal appliance behavior, particularly Furnace activity, may confound the relationship, so the analysis does not claim weather causality.

## 10. Corrected modeling handoff

The modeling stage now uses the following deployable features:

```text
gen_kw, total_appliance, temperature, humidity,
hour, dayofweek, month, is_weekend
```

The direct label source `use [kW]` is excluded.

The evaluation strategy is chronological:

- 70% train;
- 15% validation;
- 15% held-out test.

Model and decision-threshold selection use validation data only. The final selected model is LightGBM. Its held-out chronological test results are:

| Metric | Value |
|---|---:|
| F1 | 0.2912 |
| ROC-AUC | 0.8987 |
| Precision | 0.2648 |
| Recall | 0.3235 |

The lower F1 relative to the old random split is expected and more credible because the test period represents future data with seasonal distribution shift.

## 11. Visualization outputs

The preprocessing script regenerates 14 corrected images:

1. daily household energy time series;
2. one-minute power histogram;
3. power boxplot;
4. correlation heatmap;
5. weather-energy scatter/regression plots;
6. Top-5 appliance energy bar chart;
7. appliance energy-share pie chart;
8. hourly average power;
9. daily use-versus-solar energy area chart;
10. monthly appliance energy stacked area chart;
11. daily energy calendar heatmap;
12. proxy anomaly time series;
13. day-of-week average power;
14. monthly average power.

Daily/appliance values are in kWh and monthly stacked values are in MWh. Instantaneous or average readings remain in kW.

## 12. Limitations and next steps

1. Replace the statistical proxy with verified fault or event labels when possible.
2. Evaluate rolling-window or walk-forward validation in addition to a single chronological holdout.
3. Test household-specific and season-adaptive thresholds.
4. Compare performance with and without `total_appliance` to quantify its dominance.
5. Validate on additional homes before making broader claims.
6. Add precision-recall curves and calibration plots to the final presentation.

## 13. Reproduction commands

```bash
pip install -r requirements.txt
python src/HomeC_preprocess.py
python src/train_model.py
python app/app.py
```

The dashboard reads the compressed dataset and model metadata directly; no manual extraction is required.
