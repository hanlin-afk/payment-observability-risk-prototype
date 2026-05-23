"""Baseline model for payment latency-risk prediction."""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression


def train_baseline_model(x_train, y_train) -> LogisticRegression:
    """Train a transparent logistic regression baseline."""
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return model
