# DAP391m — Smart Home Energy Anomaly Detection

Dự án phân tích dữ liệu điện năng và thời tiết của **Home C**, xây dựng proxy anomaly label, so sánh năm mô hình và triển khai dashboard Plotly Dash.

## Current corrected version

Bản này đã sửa các lỗi kỹ thuật quan trọng của bản trước:

- mọi phép cộng công suất theo phút được quy đổi đúng: `kWh = sum(kW) / 60`;
- preprocessing có thể chạy lại nhiều lần mà không cộng lặp `Kitchen`, `Furnace` hoặc `total_appliance`;
- script và dashboard đọc trực tiếp `data/HomeC_cleaned_final.zip`, không cần giải nén thủ công;
- model dùng chronological split thay vì random split;
- không dùng SMOTE trên chuỗi thời gian;
- model cuối được lưu đúng với model và decision threshold đã đánh giá;
- dashboard cung cấp đủ toàn bộ feature mà model cần;
- README, notebook, model artifacts, KPI và biểu đồ đã được đồng bộ.

## Project structure

```text
SIMC_2026/
├── app/
│   └── app.py
├── data/
│   ├── HomeC_cleaned_final.zip
│   ├── kpi_summary.json
│   ├── top5_appliances.csv
│   └── weather_energy_correlation.csv
├── notebooks/
│   ├── model_pipeline.ipynb
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   ├── feature_defaults.json
│   ├── model_metadata.json
│   └── model_comparison.csv
├── report/
│   ├── Day2_MemberB_Report.md
│   ├── architecture.png
│   ├── architecture.svg
│   └── architecture_pipeline_mermaid_source.md
├── src/
│   ├── HomeC_preprocess.py
│   └── train_model.py
├── visualization/
│   └── chart1...chart14.png
├── requirements.txt
└── README.md
```

## Dataset facts

| Item | Verified value |
|---|---:|
| Rows | 503,910 |
| Columns after feature engineering | 40 |
| Reconstructed period | 2016-01-01 00:00 to 2016-12-15 22:29 |
| Sampling assumption | One reading per minute |
| Total household energy | 7,214.00 kWh |
| Average household power | 0.8590 kW |
| Peak power | 14.7146 kW |
| Proxy anomaly threshold | 4.0336 kW |
| Proxy anomalies | 12,418 rows (2.46%) |
| Total solar generation | 640.21 kWh |

### Timestamp limitation

The raw `time` counter increases by 1 per row even though the dataset represents one-minute measurements. Interpreting it directly as Unix seconds compresses the data to roughly six days. This project therefore reconstructs a one-minute timeline from the row order. Dates are suitable for relative time-series analysis under that assumption, but they are not treated as independently verified timestamps.

## Correct energy units

Each row contains average power in kW for one minute. Therefore:

```text
energy_kWh = sum(power_kW) × (1 / 60 hour)
```

Monthly values displayed in MWh use:

```text
energy_MWh = sum(power_kW) / 60 / 1000
```

Correct Top-5 appliance totals:

| Appliance | Total energy | Share |
|---|---:|---:|
| Furnace | 1,981.96 kWh | 39.41% |
| Home office | 682.69 kWh | 13.58% |
| Fridge | 533.78 kWh | 10.62% |
| Barn | 491.56 kWh | 9.78% |
| Wine cellar | 353.88 kWh | 7.04% |

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Rebuild preprocessing and charts

From the repository root:

```bash
python src/HomeC_preprocess.py
```

The script:

1. reads `HomeC.csv`, `HomeC_cleaned_final.csv`, or the shipped ZIP;
2. removes stale engineered columns before rebuilding them;
3. reconstructs the one-minute timeline;
4. cleans missing/duplicate/constant data;
5. creates time, appliance-group and aggregate features;
6. writes the cleaned dataset back to `HomeC_cleaned_final.zip`;
7. regenerates KPI/CSV artifacts and all 14 charts.

It is **idempotent**: rerunning it does not double-count grouped appliances.

## Rebuild model artifacts

```bash
python src/train_model.py
```

The corrected model workflow uses:

- chronological split: 70% train, 15% validation, 15% test;
- Logistic Regression, Random Forest, XGBoost, LightGBM and Isolation Forest;
- no random train/test shuffling;
- no SMOTE;
- validation-based probability-threshold calibration;
- final refit followed by one evaluation on the untouched test period.

### Deployable model features

The saved model uses exactly eight features:

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

`use [kW]` is excluded because it directly creates the proxy label.

### Latest verified model result

Selected model: **LightGBM**

| Metric | Chronological test result |
|---|---:|
| F1 | 0.2912 |
| ROC-AUC | 0.8987 |
| Precision | 0.2648 |
| Recall | 0.3235 |

This score is lower than the old random-split result but is more credible because future records are never mixed into training.

Full details are stored in:

- `notebooks/model_comparison.csv` — validation comparison;
- `notebooks/model_metadata.json` — test metrics, periods, threshold and limitations;
- `notebooks/feature_defaults.json` — safe defaults used by the dashboard.

## Run dashboard

```bash
python app/app.py
```

Open the local URL printed by Dash, normally `http://127.0.0.1:8050`.

The dashboard reads the ZIP directly and includes:

- KPI cards;
- recent proxy-anomaly alerts;
- corrected energy visualizations;
- weather-energy plots;
- a prediction form covering all model features;
- calibrated anomaly score and decision threshold;
- optional Gemini explanation when `GEMINI_API_KEY` is configured.

Optional `.env` file:

```env
GEMINI_API_KEY=your_key_here
```

Without the key, the app uses a local rule-based explanation.

## Research-question findings

### RQ1 — When are high-consumption anomalies most frequent?

- 15:00 has the highest anomaly rate at approximately 6.75%.
- July and August dominate, with anomaly rates of approximately 9.87% and 11.91%.
- Monday has the highest day-of-week anomaly rate at approximately 3.79%.

### RQ2 — Which appliances contribute most?

- Furnace is the largest annual appliance group at 39.41% of measured appliance energy.
- Furnace is also the largest appliance reading in 10,332 of the 12,418 proxy-anomaly rows.

### RQ3 — How does weather relate to energy?

- Daily weather variables have weak direct linear correlation with household use.
- Solar generation has a moderate positive daily correlation with temperature (`r ≈ 0.356`).
- Anomaly rows are warmer on average than normal rows, but this is descriptive association rather than proof of causality.

## Limitations

1. The target is a statistical high-load proxy, not a ground-truth equipment-fault label.
2. `total_appliance` is closely related to household consumption and can dominate prediction.
3. The reconstructed timeline depends on the one-reading-per-minute assumption.
4. The model is evaluated on one household only.
5. The final chronological test F1 shows meaningful temporal distribution shift; further work should consider rolling validation, adaptive thresholds and verified labels.
