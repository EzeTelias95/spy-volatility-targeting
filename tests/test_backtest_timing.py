"""Operational execution lag applied to target weights."""

import numpy as np
import pandas as pd
import pytest

from backtest.timing import apply_execution_lag


@pytest.fixture
def weights():
    index = pd.bdate_range("2024-01-02", periods=3)
    return pd.Series([0.5, 1.0, 1.5], index=index, name="target_weight")


def test_one_day_lag_preserves_target_dated_weights(weights):
    pd.testing.assert_series_equal(apply_execution_lag(weights, 1), weights)


def test_two_day_lag_moves_weights_one_day_later(weights):
    shifted = apply_execution_lag(weights, 2)
    expected = pd.Series([np.nan, 0.5, 1.0], index=weights.index, name=weights.name)
    pd.testing.assert_series_equal(shifted, expected)


def test_zero_lag_moves_weights_to_the_decision_date(weights):
    shifted = apply_execution_lag(weights, 0)
    expected = pd.Series([1.0, 1.5, np.nan], index=weights.index, name=weights.name)
    pd.testing.assert_series_equal(shifted, expected)


def test_headline_weight_receives_the_return_after_its_target_date(weights):
    returns = pd.Series([0.10, 0.20, 0.30], index=weights.index)
    contributions = apply_execution_lag(weights, 2) * returns
    assert np.isnan(contributions.iloc[0])
    assert contributions.iloc[1] == pytest.approx(0.5 * 0.20)
    assert contributions.iloc[2] == pytest.approx(1.0 * 0.30)


@pytest.mark.parametrize("lag", [-1, 0.5, 1.0, 3, True])
def test_invalid_lags_are_rejected(weights, lag):
    with pytest.raises(ValueError, match="0, 1 or 2"):
        apply_execution_lag(weights, lag)


@pytest.mark.parametrize("order", ["duplicate", "unsorted"])
def test_invalid_weight_indexes_are_rejected(weights, order):
    if order == "duplicate":
        weights.index = pd.DatetimeIndex(
            [weights.index[0], weights.index[0], weights.index[2]]
        )
    else:
        weights = weights.iloc[[1, 0, 2]]
    with pytest.raises(ValueError, match="index"):
        apply_execution_lag(weights, 1)


def test_non_datetime_weight_index_is_rejected():
    weights = pd.Series([0.5, 1.0], index=[0, 1])
    with pytest.raises(ValueError, match="DatetimeIndex"):
        apply_execution_lag(weights, 1)


def test_two_day_forecasts_need_no_shift_under_the_headline_lag(weights):
    pd.testing.assert_series_equal(apply_execution_lag(weights, 2, 2), weights)


@pytest.mark.parametrize("horizon", [0, 3, 1.0, True])
def test_invalid_horizons_are_rejected(weights, horizon):
    with pytest.raises(ValueError, match="1 or 2"):
        apply_execution_lag(weights, 2, horizon)
