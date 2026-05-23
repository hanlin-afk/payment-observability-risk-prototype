"""Run the full reproducible experiment.

Usage:
    python experiments/run_experiment.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.baseline import train_baseline_model
from src.data_generator import GeneratorConfig, save_dataset
from src.evaluation import evaluate_classifier, metrics_to_frame
from src.model import get_feature_importance, train_improved_model
from src.preprocessing import make_train_test_split, scale_for_linear_model
from src.visualization import plot_curves, plot_feature_importance, plot_metric_comparison


def main() -> None:
    data_path = save_dataset(GeneratorConfig(n_samples=3000, random_state=42, output_path="data/payment_observability.csv"))
    df = pd.read_csv(data_path)

    x_train, x_test, y_train, y_test = make_train_test_split(df, test_size=0.25, random_state=42)
    x_train_scaled, x_test_scaled, _ = scale_for_linear_model(x_train, x_test)

    baseline = train_baseline_model(x_train_scaled, y_train)
    improved = train_improved_model(x_train, y_train)

    metrics = metrics_to_frame(
        [
            evaluate_classifier("Logistic Regression Baseline", baseline, x_test_scaled, y_test),
            evaluate_classifier("Random Forest Improved", improved, x_test, y_test),
        ]
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    importances = get_feature_importance(improved, x_train.columns)
    importance_df = pd.DataFrame(importances, columns=["feature", "importance"]).round(5)
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    plot_metric_comparison(metrics, output_dir / "metric_comparison.png")
    plot_curves(
        {
            "Logistic Regression Baseline": baseline,
            "Random Forest Improved": improved,
        },
        {
            "Logistic Regression Baseline": x_test_scaled,
            "Random Forest Improved": x_test,
        },
        y_test,
        output_dir,
    )
    plot_feature_importance(importances, output_dir / "feature_importance.png")

    print("Experiment complete.")
    print(f"Dataset: {data_path}")
    print(f"Metrics: {metrics_path}")
    print(metrics.to_string(index=False))
    print("Top 5 improved-model features:")
    print(importance_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
