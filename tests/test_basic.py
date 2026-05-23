"""Basic tests for the research prototype."""
from __future__ import annotations

import pandas as pd

from src.data_generator import GeneratorConfig, generate_payment_observability_data
from src.preprocessing import FEATURES, TARGET, make_train_test_split, scale_for_linear_model
from src.baseline import train_baseline_model
from src.model import train_improved_model
from src.evaluation import evaluate_classifier


def test_data_generator_has_expected_columns():
    df = generate_payment_observability_data(GeneratorConfig(n_samples=100, random_state=7))
    for column in FEATURES + [TARGET]:
        assert column in df.columns
    assert len(df) == 100
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_train_test_split_shapes():
    df = generate_payment_observability_data(GeneratorConfig(n_samples=200, random_state=8))
    x_train, x_test, y_train, y_test = make_train_test_split(df)
    assert len(x_train) + len(x_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert list(x_train.columns) == FEATURES


def test_models_produce_metrics():
    df = generate_payment_observability_data(GeneratorConfig(n_samples=300, random_state=9))
    x_train, x_test, y_train, y_test = make_train_test_split(df)
    x_train_scaled, x_test_scaled, _ = scale_for_linear_model(x_train, x_test)
    baseline = train_baseline_model(x_train_scaled, y_train)
    improved = train_improved_model(x_train, y_train)
    baseline_metrics = evaluate_classifier("baseline", baseline, x_test_scaled, y_test)
    improved_metrics = evaluate_classifier("improved", improved, x_test, y_test)
    for metrics in [baseline_metrics, improved_metrics]:
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["average_precision"] <= 1
