# Smart Home Energy Anomaly Detection

This revision focuses on methodological validity rather than maximizing a misleading score. It corrects target leakage, seasonal label bias and missing baselines in the HomeC anomaly-detection pipeline.

## Core correction

The original proxy label was derived from `use [kW]`, while `total_appliance` - an additive component of `use [kW]` - was used as a model feature. The revised project now:

- Calculates the global threshold from train only.
- Includes majority and one-column `total_appliance` baselines.
- Compares leaky and leakage-free feature sets.
- Defines the primary label using the previous 30 days only.
- Excludes `total_appliance` from the deployable model.
- Preserves chronological 70/15/15 evaluation.
- Calibrates thresholds on validation only.
- Uses the same prior-30-day label for dashboard KPIs and anomaly alerts.

## Verified finding

The one-column `total_appliance` rule achieves test F1 **0.4866**, outperforming all machine-learning configurations in the rerun. Once that leakage channel is removed, F1 falls sharply. The project therefore presents leakage and seasonality diagnosis as its main research contribution.

## Rebuild

```bash
pip install -r requirements.txt
python src/HomeC_preprocess.py
python src/train_model.py
```

`src/train_model.py` regenerates:

- `notebooks/experiment_results.csv`
- `notebooks/model_comparison.csv`
- `notebooks/model_metadata.json`
- `notebooks/best_model.pkl`
- `notebooks/feature_columns.pkl`
- `notebooks/feature_defaults.json`

## Main files

- `src/train_model.py`: corrected experiments and artifact export.
- `notebooks/model_pipeline.ipynb`: reproducibility notebook.
- `report/revised_research_report.md`: updated research report.
- `report/REVISION_CHECKLIST.md`: completion checklist.
- `app/app.py`: dashboard using the exported model metadata.

## Limitations

The label remains a statistical high-load proxy rather than a verified fault label. The data covers one household, the timeline is reconstructed under a one-reading-per-minute assumption, and the current weather/calendar features are insufficient for reliable anomaly deployment.
