"""Train and export a chronologically evaluated anomaly classifier.

This revision avoids random train/test shuffling and SMOTE. It uses a chronological
70/15/15 split, selects both the model and probability threshold on validation data,
and evaluates the final refitted artifact once on the held-out test period.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "HomeC_cleaned_final.zip"
OUT_DIR = BASE_DIR / "notebooks"
RANDOM_STATE = 42
SOURCE_FEATURES = ["gen [kW]", "total_appliance", "temperature", "humidity", "hour", "dayofweek", "month", "is_weekend"]
RENAME = {"gen [kW]": "gen_kw"}
MODEL_FEATURES = [RENAME.get(c, c) for c in SOURCE_FEATURES]


def choose_threshold(y_true, score):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1[:-1])) if len(thresholds) else 0
    return float(thresholds[idx]), float(f1[idx])


def evaluate(y_true, score, threshold):
    pred = (score >= threshold).astype("int8")
    return {
        "F1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "AUC": round(float(roc_auc_score(y_true, score)), 4),
        "Precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
    }, pred


def score_model(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise TypeError("Model has no probability/decision score")


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    missing = [c for c in SOURCE_FEATURES + ["use [kW]"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing}; run src/HomeC_preprocess.py first")

    threshold_kw = float(df["use [kW]"].mean() + 3 * df["use [kW]"].std())
    y = (df["use [kW]"] > threshold_kw).astype("int8")
    X = df[SOURCE_FEATURES].rename(columns=RENAME).astype("float32")

    n = len(df); train_end = int(n * 0.70); val_end = int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    # Evenly spaced sampling reduces training time while retaining the entire time span.
    X_train_fit, y_train_fit = X_train.iloc[::3], y_train.iloc[::3]
    pos_weight = float((len(y_train_fit) - y_train_fit.sum()) / max(y_train_fit.sum(), 1))
    models = {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=700, class_weight="balanced", random_state=RANDOM_STATE))]),
        "RandomForest": RandomForestClassifier(n_estimators=70, max_depth=18, min_samples_leaf=1, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.08, subsample=0.85, colsample_bytree=0.9, scale_pos_weight=pos_weight, tree_method="hist", eval_metric="logloss", n_jobs=4, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.08, class_weight="balanced", n_jobs=4, random_state=RANDOM_STATE, verbose=-1),
    }

    rows = []; fitted = {}; thresholds = {}
    for name, model in models.items():
        model.fit(X_train_fit, y_train_fit)
        score = score_model(model, X_val)
        decision_threshold, _ = choose_threshold(y_val, score)
        result, _ = evaluate(y_val, score, decision_threshold)
        rows.append({"Model": name, **result, "DecisionThreshold": round(decision_threshold, 6)})
        fitted[name] = model; thresholds[name] = decision_threshold
        print(name, rows[-1], flush=True)

    iso = Pipeline([("scaler", StandardScaler()), ("model", IsolationForest(n_estimators=80, contamination=float(y_train_fit.mean()), n_jobs=4, random_state=RANDOM_STATE))])
    iso.fit(X_train_fit)
    scaled_val = iso.named_steps["scaler"].transform(X_val)
    iso_score = -iso.named_steps["model"].score_samples(scaled_val)
    iso_threshold, _ = choose_threshold(y_val, iso_score)
    iso_result, _ = evaluate(y_val, iso_score, iso_threshold)
    rows.append({"Model": "IsolationForest", **iso_result, "DecisionThreshold": round(iso_threshold, 6)})

    comparison = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    best_name = str(comparison[comparison["Model"] != "IsolationForest"].iloc[0]["Model"])
    best_threshold = thresholds[best_name]
    best_model = fitted[best_name]

    # Refit the selected configuration on an evenly spaced sample from train+validation.
    X_train_val = pd.concat([X_train, X_val]); y_train_val = pd.concat([y_train, y_val])
    X_final_fit, y_final_fit = X_train_val.iloc[::2], y_train_val.iloc[::2]
    final_model = clone(best_model)
    final_model.fit(X_final_fit, y_final_fit)
    test_score = score_model(final_model, X_test)
    test_result, test_pred = evaluate(y_test, test_score, best_threshold)
    cm = confusion_matrix(y_test, test_pred).tolist()

    defaults = {c: float(X_train_val[c].median()) for c in MODEL_FEATURES}
    metadata = {
        "selected_model": best_name,
        "decision_threshold": round(best_threshold, 8),
        "selection_metric": "validation F1 after probability-threshold calibration",
        "split_strategy": "chronological 70% train / 15% validation / 15% test",
        "training_sampling": "every 3rd train row for comparison; every 2nd train+validation row for final refit",
        "label_definition": "use [kW] > mean(use [kW]) + 3*std(use [kW])",
        "anomaly_threshold_kw": round(threshold_kw, 6),
        "source_features": SOURCE_FEATURES,
        "model_features": MODEL_FEATURES,
        "train_rows": len(X_train), "validation_rows": len(X_val), "test_rows": len(X_test),
        "final_fit_rows": len(X_final_fit),
        "train_period": [str(df.loc[0, "datetime"]), str(df.loc[train_end - 1, "datetime"])],
        "validation_period": [str(df.loc[train_end, "datetime"]), str(df.loc[val_end - 1, "datetime"])],
        "test_period": [str(df.loc[val_end, "datetime"]), str(df.loc[n - 1, "datetime"])],
        "test_metrics": test_result,
        "confusion_matrix": cm,
        "limitations": [
            "The anomaly label is a statistical proxy rather than a verified equipment-fault label.",
            "total_appliance is strongly related to household load and can dominate the model.",
            "The timeline is reconstructed under a one-reading-per-minute assumption.",
        ],
    }

    joblib.dump(final_model, OUT_DIR / "best_model.pkl")
    joblib.dump(MODEL_FEATURES, OUT_DIR / "feature_columns.pkl")
    comparison.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    (OUT_DIR / "feature_defaults.json").write_text(json.dumps(defaults, indent=2), encoding="utf-8")
    (OUT_DIR / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSelected", best_name, "threshold", best_threshold, flush=True)
    print("Test", test_result, "CM", cm, flush=True)

if __name__ == "__main__":
    main()
