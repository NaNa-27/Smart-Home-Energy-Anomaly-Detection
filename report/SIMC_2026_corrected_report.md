# Bug-Fix Report - `SIMC_2026_revised` Project

## 1. Summary of Work Done

The project was fixed directly from the `SIMC_2026_revised.zip` file: preprocessing was re-run, all KPI tables and 14 charts were regenerated, the model was retrained, and the dashboard, notebook, README, internal report, and architecture diagram were all updated.

The final fixed submission contains 34 files, totaling roughly 27 MB. The dataset continues to be stored as a ZIP file, so the submission package does not exceed 160 MB.

## 2. Bugs Fixed

### 2.1. Fixed the kW / kWh / MWh unit error

The data represents average power per minute, in kW. The previous version summed the raw kW values directly and recorded the result as kWh/MWh, which inflated many figures by a factor of roughly 60.

The correct formula has now been applied:

```text
Energy (kWh) = sum(Power kW) / 60
Energy (MWh) = sum(Power kW) / 60 / 1000
```

The following components were corrected:

- `data/top5_appliances.csv`.
- `data/kpi_summary.json`.
- the total-energy-by-day chart.
- the Use vs Solar area chart.
- the monthly stacked area chart.
- the calendar heatmap.
- the Top-5 bar chart and pie chart.
- all corresponding charts on the dashboard.

### 2.2. Fixed the Kitchen and Furnace double-counting bug

The previous version could re-read `HomeC_cleaned_final.csv`, which already contained:

- `Kitchen [kW]`.
- `Furnace [kW]`.
- `total_appliance`.
- the engineered time features.

The script then fed these columns back into the group-summing step, causing Furnace/Kitchen to be added a second time.

The fix removes all previously engineered features before regenerating them:

```text
hour, dayofweek, month, is_weekend, season, time_period,
Kitchen [kW], Furnace [kW], total_appliance
```

The groups are now computed only from the 14 raw circuits. Preprocessing was run twice in a row and produced identical `top5_appliances.csv` and `kpi_summary.json` outputs both times.

### 2.3. No more manual dataset extraction required

The following files now read directly from:

```text
data/HomeC_cleaned_final.zip
```

- `src/HomeC_preprocess.py`.
- `src/train_model.py`.
- `app/app.py`.
- `notebooks/model_pipeline.ipynb`.

Preprocessing also writes results back into the same ZIP file safely, using a temporary file that is then swapped in.

### 2.4. Fixed the model's data-splitting method

The previous version used `train_test_split(..., stratify=y)`, which randomly shuffled timestamps across the entire year between train and test sets. Since this is time-series data with anomalies concentrated heavily in July–August, this approach produced overly optimistic results.

The fix uses a chronological split:

| Set | Ratio | Reconstructed Time Range |
|---|---:|---|
| Train | 70% | 2016-01-01 -> 2016-09-01 |
| Validation | 15% | 2016-09-01 -> 2016-10-24 |
| Test | 15% | 2016-10-24 -> 2016-12-15 |

The model and decision threshold were selected using only the validation set. The test set was used only after the final model had already been locked in.

### 2.5. Removed SMOTE from the time-series pipeline

SMOTE is no longer applied. Synthetic samples can break the time-series structure and make evaluation harder to interpret. Class imbalance is now handled via class weight / scale weight and by calibrating the decision threshold on the validation set.

### 2.6. Reduced features and synchronized them with the dashboard

The previous version trained on 60 features, but the dashboard only accepted three input values and filled the rest with zeros. As a result, predictions did not reflect real input.

The new model uses exactly eight deployable features:

```text
gen_kw
total_appliance
temperature
humidity
hour
dayofweek
month
is_weekend
```

The dashboard now accepts full input for:

- temperature.
- humidity.
- solar generation.
- total appliance power.
- hour.
- day of week.
- month.

`is_weekend` is now automatically derived from `dayofweek`.

### 2.7. Fixed a mismatch between the saved model artifact and its evaluation results

In the previous version:

1. the model was evaluated on scaled + SMOTE-resampled data.
2. tuning caused the Random Forest's F1 score to drop from 0.9412 to 0.8477.
3. a new pipeline was then created and refit using a different method.
4. that model was still saved as `best_model.pkl`.

The fix now saves everything consistently:

- `best_model.pkl`.
- `feature_columns.pkl`.
- `feature_defaults.json`.
- `model_metadata.json`.
- `model_comparison.csv`.

`model_metadata.json` clearly records the model, threshold, split, test metrics, confusion matrix, and research limitations. The dashboard reads this metadata directly.

### 2.8. Fixed the dashboard

Key changes in `app/app.py`:

- reads data directly from the ZIP file.
- converts daily/monthly energy using the correct units.
- uses the corrected Top-5 data.
- reads model metadata and feature defaults.
- displays the model name, feature count, and test F1 score.
- uses `predict_proba` together with a calibrated decision threshold.
- displays the anomaly score and threshold.
- no longer fills dozens of missing features with zero.
- renamed the `Average Energy` KPI to `Average Power`.
- fixed the run instructions and dataset error messages.

### 2.9. Added missing dependencies

The following missing libraries were added to `requirements.txt`:

- `seaborn`.
- `python-dotenv`.
- `nbformat`.

### 2.10. Synchronized documentation

The following were updated or regenerated:

- `README.md`.
- `report/Day2_MemberB_Report.md`.
- `report/architecture_pipeline_mermaid_source.md`.
- `report/architecture.png`.
- `report/architecture.svg`.
- `notebooks/model_pipeline.ipynb`.

The updated documents no longer reference deleted artifacts, no longer require extracting RAR/CSV files, and no longer use the incorrect Furnace figures.

## 3. Data Statistics After the Fix

| Metric | Value |
|---|---:|
| Number of rows | 503,910 |
| Number of columns | 40 |
| Reconstructed time range | 2016-01-01 00:00 → 2016-12-15 22:29 |
| Total household electricity | 7,214.00 kWh |
| Average power | 0.8590 kW |
| Peak | 14.7146 kW |
| Total solar generation | 640.21 kWh |
| Proxy anomaly threshold | 4.0336 kW |
| Proxy anomalies | 12,418 |
| Proxy anomaly rate | 2.46% |

### Corrected Top-5 Appliances

| Appliance | Total Energy | Share | Peak Hour | Peak Date |
|---|---:|---:|---:|---|
| Furnace | 1,981.96 kWh | 39.41% | 05:00 | 2016-02-14 |
| Home office | 682.69 kWh | 13.58% | 21:00 | 2016-08-09 |
| Fridge | 533.78 kWh | 10.62% | 20:00 | 2016-08-14 |
| Barn | 491.56 kWh | 9.78% | 16:00 | 2016-09-11 |
| Wine cellar | 353.88 kWh | 7.04% | 17:00 | 2016-08-14 |

## 4. Research Question Results After Re-Verification

### RQ1 - When anomalies occur most frequently

- 15:00 has the highest anomaly rate, at roughly 6.75%.
- 16:00 and 17:00 follow at roughly 5.83% and 5.80%, respectively.
- August has an anomaly rate of roughly 11.91%.
- July has an anomaly rate of roughly 9.87%.
- Monday has the highest anomaly rate by day of week, at roughly 3.79%.

### RQ2 - Main contributing appliance

- Furnace accounts for 39.41% of total measured appliance energy.
- Furnace is the largest-value appliance in 10,332 of 12,418 anomaly rows.

This indicates that Furnace is strongly associated with the proxy high-load anomaly, but this does not mean the device is malfunctioning.

### RQ3 - Weather and energy

- Weather shows a weak linear correlation with household usage at the daily aggregation level.
- Solar generation is positively correlated with temperature, `r ≈ 0.356`.
- Average temperature in anomaly rows is roughly 68.25°F, compared with 50.30°F in normal rows.

This is a descriptive association, not evidence of causation.

## 5. New Model Results

### Validation Comparison

| Model | F1 | AUC | Precision | Recall | Decision Threshold |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.2987 | 0.9511 | 0.2241 | 0.4477 | 0.450795 |
| Random Forest | 0.2424 | 0.9287 | 0.1445 | 0.7503 | 0.139608 |
| XGBoost | 0.2222 | 0.9435 | 0.1427 | 0.5017 | 0.644966 |
| Logistic Regression | 0.2131 | 0.9230 | 0.9304 | 0.1204 | 0.999999 |
| Isolation Forest | 0.1381 | 0.7529 | 0.1453 | 0.1316 | 0.602938 |

### Selected Final Model

Selected model: **LightGBM**

Held-out chronological test results:

| Metric | Value |
|---|---:|
| F1 | 0.2912 |
| ROC-AUC | 0.8987 |
| Precision | 0.2648 |
| Recall | 0.3235 |

Confusion matrix:

```text
[[74748, 397],
 [  299, 143]]
```

Results are lower than under the previous random split, but more trustworthy, since the model must predict a future period that did not appear in training.

## 6. Key Files Fixed or Added

| File | Change |
|---|---|
| `src/HomeC_preprocess.py` | Rewrote preprocessing to be idempotent, read/write ZIP directly, fix units, regenerate KPIs and 14 charts |
| `src/train_model.py` | Added chronological split pipeline, model comparison, threshold calibration, and artifact export |
| `app/app.py` | Fixed data/units, model input, prediction threshold, and metadata |
| `notebooks/model_pipeline.ipynb` | Synced workflow and new results |
| `notebooks/best_model.pkl` | New LightGBM model |
| `notebooks/feature_columns.pkl` | 8 features matching the dashboard |
| `notebooks/feature_defaults.json` | Default values as needed |
| `notebooks/model_metadata.json` | Split, threshold, test metrics, and limitations |
| `notebooks/model_comparison.csv` | New validation comparison |
| `data/HomeC_cleaned_final.zip` | Cleanly regenerated 40-column dataset, no double-counting |
| `data/kpi_summary.json` | Correct-unit KPIs |
| `data/top5_appliances.csv` | Correct-unit Top-5, no double-counting |
| `visualization/*.png` | 14 regenerated figures |
| `README.md` | Updated instructions and results |
| `report/Day2_MemberB_Report.md` | New RQ analysis and limitations |
| `report/architecture.*` | New pipeline architecture |
| `requirements.txt` | Added missing dependencies |

## 7. Quality Checks Performed

- `HomeC_preprocess.py`: passed `py_compile`.
- `train_model.py`: passed `py_compile`.
- `app.py`: passed `py_compile`.
- Notebook: valid per the nbformat v4 standard, containing 13 cells.
- Dataset ZIP: read successfully, exactly 503,910 * 40.
- Dataset ZIP: passed integrity check.
- Kitchen group: matches the sum of its three circuits, floating-point error under `1e-15`.
- Furnace group: matches the sum of its two circuits, error under `1e-15`.
- `total_appliance`: matches the sum of all 14 circuits, error under `1e-15`.
- Preprocessing run twice in a row: Top-5 and KPI results unchanged.
- Model artifact: loaded successfully.
- Feature count: exactly 8.
- Test prediction with median defaults: successful.
- Project contains no extracted 162 MB CSV and no `__pycache__`/`.pyc` files.

The dashboard was not launched via a web server in the test environment because the Dash package was not pre-installed in the current runtime. However, the file passed syntax checks, the model prediction logic was tested separately, and `dash` is now listed in `requirements.txt`.

## 8. How to Run the Fixed Version

```bash
pip install -r requirements.txt
python src/HomeC_preprocess.py
python src/train_model.py
python app/app.py
```

There is no need to extract `HomeC_cleaned_final.zip`.

## 9. Remaining Limitations

The following are research limitations, not code bugs that can be fully fixed in the current version:

1. the anomaly label is still a proxy (`mean + 3×std`), not a ground-truth fraud/fault label.
2. the timeline is reconstructed under the assumption of one sample per minute.
3. `total_appliance` is strongly correlated with household usage and may dominate the model.
4. the data covers only a single household.
5. the chronological test F1 shows a significant distribution shift.
6. further rolling/walk-forward validation, verified event labels, and external household validation would be needed to develop this into a paper or conference submission.
