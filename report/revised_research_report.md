# Smart Home Energy Anomaly Detection: Revised Research Report

## 1. Revision objective

This revision addresses the methodological issues identified in the review file. The central problem was target leakage: `total_appliance` is an additive component of `use [kW]`, while the original label was calculated directly from `use [kW]`. A model using `total_appliance` was therefore partly reconstructing the target rather than learning an independent anomaly pattern.

## 2. Changes implemented

1. The global 3-sigma threshold is calculated from the training period only.
2. A leakage diagnostic is included with the original eight-feature configuration.
3. A required one-column baseline using only `total_appliance` is included.
4. The deployable model excludes `total_appliance`.
5. A causal 30-day rolling label is added: the current row is compared with the previous 30 days, and the current value is excluded from its own threshold.
6. Train, validation and test sets remain chronological at 70/15/15.
7. Probability thresholds are calibrated on validation only and then frozen for test.
8. Experiment outputs are saved in `notebooks/experiment_results.csv`.

## 3. Label definitions

### 3.1 Train-only global label

`anomaly = use [kW] > mean(train use) + 3 × std(train use)`

The resulting train threshold is 4.3952 kW.

### 3.2 Local 30-day label

`anomaly_t = use_t > rolling_mean(previous 30 days) + 3 × rolling_std(previous 30 days)`

This definition asks whether a reading is unusually high relative to its recent regime rather than relative to the whole year.

## 4. Distribution-shift diagnosis

| Label | Train anomaly rate | Validation anomaly rate | Test anomaly rate |
|---|---:|---:|---:|
| Train-only global 3-sigma | 2.527% | 0.983% | 0.368% |
| Local 30-day 3-sigma | 2.505% | 0.693% | 2.355% |

The global label is strongly seasonal. Its anomaly rate falls from 2.527% in train to 0.368% in test. The local label reduces the summer concentration and produces a substantially more meaningful winter test prevalence.

### Monthly anomaly rates

| Month | Global label | Local 30-day label |
|---:|---:|---:|
| 1 | 0.361% | 1.396% |
| 2 | 0.584% | 1.830% |
| 3 | 0.291% | 1.228% |
| 4 | 0.292% | 1.748% |
| 5 | 0.361% | 1.427% |
| 6 | 0.218% | 1.803% |
| 7 | 8.107% | 7.807% |
| 8 | 9.810% | 2.789% |
| 9 | 1.512% | 0.204% |
| 10 | 0.302% | 1.810% |
| 11 | 0.421% | 2.528% |
| 12 | 0.237% | 1.469% |

July remains elevated under the local label. This indicates a sustained consumption regime that a 30-day window does not immediately absorb. It is a substantive finding rather than a preprocessing failure.

## 5. Required baselines and leakage experiments

| Experiment | Model / rule | Validation F1 | Test F1 | Test AUC | Test precision | Test recall |
|---|---|---:|---:|---:|---:|---:|
| Baseline: majority normal | Rule | — | 0.0000 | 0.5000 | 0.0000 | 0.0000 |
| Baseline: total_appliance threshold | Rule | — | 0.4866 | 0.9235 | 0.9479 | 0.3273 |
| Global label + leaky features | LogisticRegression | 0.2259 | 0.3931 | 0.9020 | 1.0000 | 0.2446 |
| Global label + leaky features | LightGBM | 0.2539 | 0.0000 | 0.8747 | 0.0000 | 0.0000 |
| Global label + leakage removed | LogisticRegression | 0.0235 | 0.0000 | 0.5992 | 0.0000 | 0.0000 |
| Global label + leakage removed | LightGBM | 0.0545 | 0.0000 | 0.6981 | 0.0000 | 0.0000 |
| Primary: local 30-day label + leakage removed | LogisticRegression | 0.0284 | 0.0000 | 0.6974 | 0.0000 | 0.0000 |
| Primary: local 30-day label + leakage removed | RandomForest | 0.0614 | 0.0752 | 0.6791 | 0.0537 | 0.1253 |
| Primary: local 30-day label + leakage removed | XGBoost | 0.1105 | 0.0000 | 0.7674 | 0.0000 | 0.0000 |
| Primary: local 30-day label + leakage removed | LightGBM | 0.0653 | 0.0795 | 0.7525 | 0.0600 | 0.1174 |


## 6. Main findings

The one-column `total_appliance` baseline achieves test F1 = 0.4866, which is higher than every machine-learning configuration in this rerun. This confirms that the original predictive performance was dominated by a feature that directly overlaps with the label-generating quantity.

After removing `total_appliance`, performance drops sharply. Under the train-only global label, the seven-feature models produce test F1 of 0.0000. Under the local 30-day label, Random Forest reaches test F1 0.0752 and LightGBM reaches 0.0795. The validation-selected XGBoost model has AUC 0.7674 on test but predicts no positives at the frozen validation threshold, giving F1 0.0000.

These results mean the current weather and calendar variables have limited operational value for detecting high-load events in this dataset. The honest contribution of the project is therefore the diagnosis and correction of leakage and seasonality, not a claim of strong anomaly classification.

## 7. Selected deployable artifact

The saved model is XGBoost, selected strictly by validation F1 under the local 30-day label and leakage-free feature set. Its test metrics are:

| Metric | Value |
|---|---:|
| F1 | 0.0000 |
| ROC-AUC | 0.7674 |
| Precision | 0.0000 |
| Recall | 0.0000 |

The poor thresholded test result is retained rather than replaced by a test-optimized threshold, because tuning on test would invalidate the evaluation.

## 8. Interpretation and implications

The project should not describe `total_appliance` as merely "strongly related" to household load. It is an additive component of `use [kW]` and is therefore a direct leakage channel for this proxy label.

The project should also avoid claiming that chronological splitting alone explains lower F1. The larger cause is label prevalence shift produced by a global threshold on seasonal consumption. The local rolling threshold improves the definition of anomaly, although it does not create strong predictive information in the remaining features.

## 9. Limitations

- The anomaly target is a statistical high-load proxy, not a verified equipment-fault label.
- The analysis covers one household only.
- The local threshold adapts to recent history but can lag sustained regime changes.
- The reconstructed timeline assumes one observation per minute.
- Current exogenous features are insufficient for reliable deployment.

## 10. Recommended next work

Future work should add lagged consumption, rolling load statistics, appliance-state transitions and verified event labels. These features must be constructed from information available before prediction time. Evaluation should use rolling-origin validation across multiple seasons and households.

## 11. Reproducibility

Run:

```bash
python src/HomeC_preprocess.py
python src/train_model.py
```

The second command regenerates the baseline table, leakage experiments, local-label experiments, model metadata and deployable artifacts.
