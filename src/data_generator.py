"""Synthetic observability data generator for cloud-native payment APIs.

The data represents service-level telemetry collected at fixed time windows.
It is synthetic by design: the goal is to support a reproducible research
prototype without relying on private infrastructure logs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for synthetic telemetry generation."""

    n_samples: int = 3000
    random_state: int = 42
    output_path: str = "data/payment_observability.csv"


def generate_payment_observability_data(config: GeneratorConfig = GeneratorConfig()) -> pd.DataFrame:
    """Generate synthetic payment-service observability windows.

    The binary label `latency_risk` indicates whether the next short window is
    likely to exceed a high-latency threshold. The label is constructed from a
    transparent risk process involving traffic pressure, dependency failures,
    cold starts, queue depth, and deployment changes.
    """
    rng = np.random.default_rng(config.random_state)
    n = config.n_samples

    hour = rng.integers(0, 24, size=n)
    day_of_week = rng.integers(0, 7, size=n)
    is_peak_hour = np.isin(hour, [10, 11, 12, 13, 18, 19, 20]).astype(int)
    is_weekend = (day_of_week >= 5).astype(int)
    deployment_window = rng.binomial(1, 0.09, size=n)

    requests_per_min = rng.normal(800 + 260 * is_peak_hour - 120 * is_weekend, 150, size=n)
    requests_per_min = np.clip(requests_per_min, 80, None)

    cpu_utilization = 38 + 0.045 * requests_per_min + 11 * deployment_window + rng.normal(0, 8, size=n)
    cpu_utilization = np.clip(cpu_utilization, 5, 99)

    memory_utilization = 45 + 0.020 * requests_per_min + 5 * deployment_window + rng.normal(0, 7, size=n)
    memory_utilization = np.clip(memory_utilization, 10, 98)

    dependency_error_rate = rng.beta(1.2, 35, size=n) + 0.015 * deployment_window + 0.010 * is_peak_hour
    dependency_error_rate = np.clip(dependency_error_rate, 0, 0.25)

    cold_start_rate = rng.beta(1.5, 45, size=n) + 0.025 * deployment_window + 0.012 * is_peak_hour
    cold_start_rate = np.clip(cold_start_rate, 0, 0.30)

    queue_depth = rng.poisson(10 + requests_per_min / 110 + 18 * dependency_error_rate + 8 * deployment_window)
    db_connection_saturation = 0.25 + 0.0038 * queue_depth + 0.45 * dependency_error_rate + rng.normal(0, 0.05, size=n)
    db_connection_saturation = np.clip(db_connection_saturation, 0.05, 0.98)

    cache_hit_rate = 0.88 - 0.10 * deployment_window - 0.08 * dependency_error_rate + rng.normal(0, 0.035, size=n)
    cache_hit_rate = np.clip(cache_hit_rate, 0.35, 0.99)

    p95_latency_ms = (
        90
        + 0.10 * requests_per_min
        + 1.4 * cpu_utilization
        + 2.1 * queue_depth
        + 360 * dependency_error_rate
        + 190 * cold_start_rate
        + 80 * db_connection_saturation
        - 60 * cache_hit_rate
        + rng.normal(0, 35, size=n)
    )
    p95_latency_ms = np.clip(p95_latency_ms, 40, None)

    risk_score = (
        -7.0
        + 0.018 * (p95_latency_ms - 180)
        + 3.0 * dependency_error_rate
        + 2.6 * cold_start_rate
        + 2.1 * db_connection_saturation
        + 0.015 * queue_depth
        + 0.75 * deployment_window
        + 0.35 * is_peak_hour
    )
    probability = 1 / (1 + np.exp(-risk_score))
    latency_risk = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "hour": hour,
            "day_of_week": day_of_week,
            "is_peak_hour": is_peak_hour,
            "is_weekend": is_weekend,
            "deployment_window": deployment_window,
            "requests_per_min": requests_per_min.round(2),
            "cpu_utilization": cpu_utilization.round(2),
            "memory_utilization": memory_utilization.round(2),
            "dependency_error_rate": dependency_error_rate.round(4),
            "cold_start_rate": cold_start_rate.round(4),
            "queue_depth": queue_depth,
            "db_connection_saturation": db_connection_saturation.round(4),
            "cache_hit_rate": cache_hit_rate.round(4),
            "p95_latency_ms": p95_latency_ms.round(2),
            "latency_risk": latency_risk,
        }
    )


def save_dataset(config: GeneratorConfig = GeneratorConfig()) -> Path:
    """Generate and save the synthetic dataset to CSV."""
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_payment_observability_data(config)
    df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    path = save_dataset()
    print(f"Synthetic dataset written to {path}")
