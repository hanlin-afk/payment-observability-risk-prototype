"""Improved model for latency-risk prediction."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


def train_improved_model(x_train, y_train) -> RandomForestClassifier:
    """Train a lightweight nonlinear model.

    Random forests handle interactions such as traffic spikes during deployment
    windows without requiring deep learning frameworks.
    """
    model = RandomForestClassifier(
        n_estimators=160,
        max_depth=8,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return model


def get_feature_importance(model: RandomForestClassifier, feature_names) -> list[tuple[str, float]]:
    """Return sorted feature importances from the improved model."""
    pairs = list(zip(feature_names, model.feature_importances_))
    return sorted(pairs, key=lambda item: item[1], reverse=True)
