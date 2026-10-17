"""Shared time-series inference utilities."""

import numpy as np


def automatic_bandwidth(values: np.ndarray) -> int:
    """Newey-West (1994) automatic bandwidth for one series."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    n = len(centered)
    lead = min(int(4.0 * (n / 100.0) ** (2.0 / 9.0)), n - 1)
    autocovariances = np.array(
        [centered[lag:] @ centered[: n - lag] / n for lag in range(lead + 1)]
    )
    long_run = autocovariances[0] + 2.0 * autocovariances[1:].sum()
    if long_run == 0.0:
        return 0
    weighted = 2.0 * (np.arange(1, lead + 1) * autocovariances[1:]).sum()
    estimate = int(1.1447 * ((weighted / long_run) ** 2 * n) ** (1.0 / 3.0))
    return min(estimate, n - 1)


def long_run_covariance(values: np.ndarray, bandwidth: int) -> np.ndarray:
    """Bartlett long-run covariance with a fixed bandwidth."""
    centered = np.asarray(values, dtype=float)
    if centered.ndim == 1:
        centered = centered[:, None]
    centered = centered - centered.mean(axis=0)
    covariance = centered.T @ centered / len(centered)
    for lag in range(1, bandwidth + 1):
        autocovariance = centered[lag:].T @ centered[:-lag] / len(centered)
        weight = 1.0 - lag / (bandwidth + 1.0)
        covariance += weight * (autocovariance + autocovariance.T)
    return covariance


def long_run_variance(values: np.ndarray) -> float:
    """Bartlett long-run variance with automatic Newey-West bandwidth."""
    bandwidth = automatic_bandwidth(values)
    return float(long_run_covariance(values, bandwidth)[0, 0])
