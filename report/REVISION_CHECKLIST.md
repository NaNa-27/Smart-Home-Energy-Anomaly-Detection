# Revision checklist

- [x] Checked repository for committed `.env` or exposed Gemini key; none was present.
- [x] Global threshold calculated from train only.
- [x] Added majority baseline.
- [x] Added one-column `total_appliance` baseline.
- [x] Added leaky vs leakage-free comparison.
- [x] Added prior 30-day rolling label with current row excluded.
- [x] Kept chronological 70/15/15 split.
- [x] Calibrated decision threshold on validation only.
- [x] Excluded `total_appliance` from deployable model.
- [x] Updated dashboard KPI/anomaly alerts to use the prior 30-day local label.
- [x] Removed API-key suffix logging from the Gemini integration.
- [x] Fixed duplicate metric columns in `model_comparison.csv`.
- [x] Exported experiment results and updated metadata.
- [x] Added revised research report.
- [x] Updated README and notebook.
