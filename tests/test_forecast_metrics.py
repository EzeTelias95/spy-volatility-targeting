"""Forecast metrics against hand-computed values, plus the degenerate cases.

Expected values were computed by hand from the formulas, on variances:
  qlike = proxy/forecast - ln(proxy/forecast) - 1
  rmse  = sqrt(mean((forecast - proxy)^2))
  realized utility = mean(0.08 r - 0.04 r^2), r = sqrt(proxy / forecast)
"""

import numpy as np
import pandas as pd
import pytest

from data.realized_vol import VARIANCE_FLOOR
from evaluation.forecasts import diebold_mariano, evaluate_forecasts


def test_qlike_and_rmse_match_hand_computed_values():
    index = pd.bdate_range("2024-01-01", periods=2)
    proxy = pd.Series([0.01, 0.02], index=index)
    forecasts = {"ewma": pd.Series([0.012, 0.015], index=index)}
    row = evaluate_forecasts(forecasts, proxy).loc["ewma"]
    assert row["qlike"] == pytest.approx(0.13075059545328493, rel=1e-12)
    assert row["rmse"] == pytest.approx(0.00012759506260040003, rel=1e-12)


def test_realized_utility_matches_hand_computed_values():
    index = pd.bdate_range("2024-01-01", periods=2)
    proxy = pd.Series([0.01, 0.02], index=index)
    forecasts = {"ewma": pd.Series([0.012, 0.015], index=index)}
    row = evaluate_forecasts(forecasts, proxy).loc["ewma"]
    assert row["realized_utility"] == pytest.approx(0.037222222222222226, rel=1e-12)
    assert row["realized_utility_share"] == pytest.approx(0.9305555555555556, rel=1e-12)


def test_perfect_forecast_earns_the_ceiling():
    index = pd.bdate_range("2024-01-01", periods=3)
    proxy = pd.Series([0.01, 0.02, 0.015], index=index)
    row = evaluate_forecasts({"perfect": proxy.copy()}, proxy).loc["perfect"]
    assert row["realized_utility"] == pytest.approx(0.04, rel=1e-12)
    assert row["realized_utility_share"] == pytest.approx(1.0, rel=1e-12)


def test_identical_losses_give_a_zero_statistic():
    rng = np.random.default_rng(0)
    losses = rng.gamma(2.0, 0.5, 500)
    assert diebold_mariano(losses, losses) == (0.0, 1.0)


def test_flat_day_proxy_is_floored_and_counted():
    index = pd.bdate_range("2024-01-01", periods=3)
    proxy = pd.Series([0.01, 0.0, np.sqrt(VARIANCE_FLOOR)], index=index)
    forecasts = {"ewma": pd.Series([0.011, 0.012, 0.009], index=index)}
    row = evaluate_forecasts(forecasts, proxy).loc["ewma"]
    assert row["proxy_floored_days"] == 2
    assert row["qlike"] == pytest.approx(5.5306207893531925, rel=1e-12)


def test_newey_west_statistic_on_a_non_degenerate_differential():
    rng = np.random.default_rng(7)
    a = pd.Series(rng.gamma(2.0, 0.5, 600))
    b = a + rng.normal(0.05, 0.2, 600)
    stat, pvalue = diebold_mariano(b, a)
    assert stat == pytest.approx(5.4460674261572644, rel=1e-12)
    assert pvalue == pytest.approx(5.149562264943534e-08, rel=1e-12)
