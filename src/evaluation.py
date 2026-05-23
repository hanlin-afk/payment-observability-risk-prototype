"""Evaluation metrics for binary latency-risk prediction."""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def evaluate_classifier(name: str, model, x_test, y_test) -> dict:
    """Evaluate a classifier with ranking and threshold-based metrics."""
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
        "average_precision": average_precision_score(y_test, y_score),
    }


def metrics_to_frame(metrics: list[dict]) -> pd.DataFrame:
    """Convert metric dictionaries to a rounded DataFrame."""
    return pd.DataFrame(metrics).round(4)
