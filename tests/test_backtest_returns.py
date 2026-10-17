"""Adjusted SPY return construction and target-date alignment."""

import numpy as np
import pandas as pd
import pytest

from backtest.returns import build_instrument_returns, prepare_backtest_data


def market(open_, close, adjusted_close=None):
    index = pd.bdate_range("2024-01-02", periods=len(close))
    adjusted_close = close if adjusted_close is None else adjusted_close
    data = pd.DataFrame(
        {"open": open_, "close": close, "adj_close": adjusted_close},
        index=index,
    )
    data["ret"] = data["adj_close"].pct_change()
    data["rf"] = 0.0001
    return data


def test_constant_prices_have_zero_returns_after_the_first_day():
    returns = build_instrument_returns(market([100.0] * 3, [100.0] * 3))
    assert returns.iloc[1:].to_numpy() == pytest.approx(np.zeros((2, 3)))


def test_overnight_and_intraday_moves_are_separated():
    data = market([100.0, 110.0, 110.0], [100.0, 110.0, 121.0])
    returns = build_instrument_returns(data)
    assert returns["return_overnight"].iloc[1] == pytest.approx(0.10)
    assert returns["return_intraday"].iloc[1] == pytest.approx(0.0)
    assert returns["return_overnight"].iloc[2] == pytest.approx(0.0)
    assert returns["return_intraday"].iloc[2] == pytest.approx(0.10)


def test_return_components_reconcile():
    returns = build_instrument_returns(
        market([100.0, 105.0, 108.0], [100.0, 108.0, 102.0])
    )
    valid = returns.iloc[1:]
    composed = (1.0 + valid["return_overnight"]) * (1.0 + valid["return_intraday"])
    assert composed.to_numpy() == pytest.approx(
        (1.0 + valid["return_close_to_close"]).to_numpy()
    )


def test_adjusted_open_uses_the_daily_adjustment_factor():
    data = market(
        [100.0, 90.0, 100.0],
        [100.0, 100.0, 100.0],
        adjusted_close=[50.0, 60.0, 70.0],
    )
    returns = build_instrument_returns(data)
    assert returns["return_overnight"].iloc[1] == pytest.approx(0.08)
    assert returns["return_intraday"].iloc[1] == pytest.approx(1 / 0.9 - 1)


def test_first_oos_return_keeps_the_previous_in_sample_close():
    data = market([100.0, 100.0, 110.0], [100.0, 110.0, 121.0])
    forecasts = pd.Series([0.01, 0.02], index=data.index[1:])
    prepared = prepare_backtest_data(data, forecasts)
    assert prepared["return_close_to_close"].iloc[0] == pytest.approx(0.10)
    assert prepared["forecast_vol"].equals(forecasts)
    assert prepared["rf"].equals(data.loc[forecasts.index, "rf"])


def test_close_to_close_return_must_match_the_schema():
    data = market([100.0, 100.0], [100.0, 110.0])
    data.loc[data.index[1], "ret"] = 0.20
    with pytest.raises(ValueError, match="reconciliation failed"):
        build_instrument_returns(data)


def test_internal_return_nan_is_rejected():
    data = market([100.0] * 3, [100.0] * 3)
    data.loc[data.index[0], "ret"] = 0.0
    data.loc[data.index[1], "ret"] = np.nan
    with pytest.raises(ValueError, match="only on the first date"):
        build_instrument_returns(data)


def test_multiple_return_nans_are_rejected():
    data = market([100.0] * 3, [100.0] * 3)
    data.loc[data.index[1], "ret"] = np.nan
    with pytest.raises(ValueError, match="at most one NaN"):
        build_instrument_returns(data)


def test_non_finite_return_after_first_date_is_rejected():
    data = market([100.0] * 3, [100.0] * 3)
    data.loc[data.index[1], "ret"] = np.inf
    with pytest.raises(ValueError, match="ret must be finite"):
        build_instrument_returns(data)


def test_missing_forecast_date_is_rejected():
    data = market([100.0, 100.0], [100.0, 100.0])
    forecasts = pd.Series([0.01], index=[pd.Timestamp("2024-01-10")])
    with pytest.raises(ValueError, match="missing from market_data"):
        prepare_backtest_data(data, forecasts)


def test_skipped_market_date_in_forecasts_is_rejected():
    data = market([100.0] * 4, [100.0] * 4)
    forecasts = pd.Series([0.01, 0.01], index=data.index[[1, 3]])

    with pytest.raises(ValueError, match="must be consecutive market dates"):
        prepare_backtest_data(data, forecasts)


def test_first_market_date_has_no_return_and_is_rejected_as_a_target():
    data = market([100.0, 100.0], [100.0, 100.0])
    forecasts = pd.Series([0.01], index=data.index[:1])
    with pytest.raises(ValueError, match="returns unavailable"):
        prepare_backtest_data(data, forecasts)


@pytest.mark.parametrize("column", ["open", "close", "adj_close"])
@pytest.mark.parametrize("value", [0.0, np.nan, np.inf])
def test_invalid_prices_are_rejected(column, value):
    data = market([100.0, 100.0], [100.0, 100.0])
    data.loc[data.index[1], column] = value
    with pytest.raises(ValueError, match="finite and positive"):
        build_instrument_returns(data)


@pytest.mark.parametrize("order", ["duplicate", "unsorted"])
def test_invalid_market_indexes_are_rejected(order):
    data = market([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    if order == "duplicate":
        data.index = pd.DatetimeIndex([data.index[0], data.index[0], data.index[2]])
    else:
        data = data.iloc[[1, 0, 2]]
    with pytest.raises(ValueError, match="index"):
        build_instrument_returns(data)


@pytest.mark.parametrize("forecast", [0.0, -0.01, np.nan, np.inf])
def test_invalid_forecast_is_rejected(forecast):
    data = market([100.0, 100.0], [100.0, 100.0])
    forecasts = pd.Series([forecast], index=data.index[1:])
    with pytest.raises(ValueError, match="finite and positive"):
        prepare_backtest_data(data, forecasts)


@pytest.mark.parametrize("rf", [np.nan, np.inf])
def test_invalid_rf_is_rejected(rf):
    data = market([100.0, 100.0], [100.0, 100.0])
    data.loc[data.index[1], "rf"] = rf
    forecasts = pd.Series([0.01], index=data.index[1:])
    with pytest.raises(ValueError, match="rf unavailable"):
        prepare_backtest_data(data, forecasts)


@pytest.mark.parametrize("order", ["duplicate", "unsorted"])
def test_invalid_forecast_indexes_are_rejected(order):
    data = market([100.0, 100.0, 100.0], [100.0, 100.0, 100.0])
    forecasts = pd.Series([0.01, 0.02], index=data.index[1:])
    if order == "duplicate":
        forecasts.index = pd.DatetimeIndex([data.index[1], data.index[1]])
    else:
        forecasts = forecasts.iloc[::-1]
    with pytest.raises(ValueError, match="forecasts index"):
        prepare_backtest_data(data, forecasts)
