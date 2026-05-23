"""Visualization helpers for experiment outputs."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay


def plot_metric_comparison(metrics_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Create a simple bar chart comparing key model metrics."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = metrics_df.set_index("model")[["f1", "roc_auc", "average_precision"]]
    ax = plot_df.plot(kind="bar", figsize=(8, 5))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Latency-risk prediction performance")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_curves(models: dict, x_sets: dict, y_test, output_dir: str | Path) -> list[Path]:
    """Save ROC and precision-recall curves for each model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for curve_name, display_class in {
        "roc_curve": RocCurveDisplay,
        "precision_recall_curve": PrecisionRecallDisplay,
    }.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, model in models.items():
            display_class.from_estimator(model, x_sets[name], y_test, ax=ax, name=name)
        ax.set_title(curve_name.replace("_", " ").title())
        plt.tight_layout()
        path = output_dir / f"{curve_name}.png"
        plt.savefig(path, dpi=160)
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths


def plot_feature_importance(importances: list[tuple[str, float]], output_path: str | Path, top_n: int = 10) -> Path:
    """Plot the top feature importances from the improved model."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_items = importances[:top_n]
    names = [item[0] for item in top_items][::-1]
    values = [item[1] for item in top_items][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values)
    ax.set_xlabel("Importance")
    ax.set_title("Top Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
