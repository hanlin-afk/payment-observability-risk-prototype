"""Preprocessing utilities for the latency-risk experiment."""
from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "latency_risk"
FEATURES = [
    "hour",
    "day_of_week",
    "is_peak_hour",
    "is_weekend",
    "deployment_window",
    "requests_per_min",
    "cpu_utilization",
    "memory_utilization",
    "dependency_error_rate",
    "cold_start_rate",
    "queue_depth",
    "db_connection_saturation",
    "cache_hit_rate",
    "p95_latency_ms",
]


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix and binary target vector."""
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df[FEATURES].copy(), df[TARGET].astype(int).copy()


def make_train_test_split(
    df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    x, y = split_features_target(df)
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)


def scale_for_linear_model(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Standardize numeric inputs for the baseline logistic model."""
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
    test_scaled = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)
    return train_scaled, test_scaled, scaler
