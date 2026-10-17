"""Daily volatility targeting and gross SPY-cash accounting."""

from dataclasses import FrozenInstanceError, replace
from math import sqrt

import numpy as np
import pandas as pd
import pytest

from backtest.targeting import (
    BASIS_POINTS_PER_UNIT,
    TRADING_DAYS,
    WEEKDAYS,
    BacktestConfig,
    _cash_contribution,
    _drift_weight,
    build_ledger,
)

SAME_CLOSE = BacktestConfig(execution_lag=1)


def market(returns, rf=0.0001):
    index = pd.bdate_range("2024-01-02", periods=len(returns) + 1)
    prices = 100.0 * np.cumprod([1.0, *(1.0 + np.asarray(returns))])
    data = pd.DataFrame(
        {"open": prices, "close": prices, "adj_close": prices},
        index=index,
    )
    data["ret"] = pd.Series([np.nan, *returns], index=index)
    data["rf"] = rf
    return data


def forecasts_for_weights(data, weights, config=None):
    config = SAME_CLOSE if config is None else config
    target_daily = config.target_vol_annual / sqrt(TRADING_DAYS)
    return pd.Series(target_daily / np.asarray(weights), index=data.index[1:])


def test_config_is_frozen():
    config = BacktestConfig()
    with pytest.raises(FrozenInstanceError):
        config.max_weight = 2.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("target_vol_annual", 0.0, "target_vol_annual must be positive"),
        ("max_weight", 0.0, "max_weight must be positive"),
        ("sizing_exponent", 0.0, "sizing_exponent must be positive"),
        ("vol_cutoff_annual", 0.0, "vol_cutoff_annual must be positive or None"),
        ("funding_spread_annual", -0.01, "must be non-negative"),
        ("execution_lag", 3, "execution_lag must be 0, 1 or 2"),
        ("forecast_horizon", 3, "forecast_horizon must be 1 or 2"),
        ("forecast_horizon", 2, "forecast_horizon 2 requires execution_lag 2"),
        ("apply_cap", 1, "apply_cap must be boolean"),
        ("no_trade_band", -0.01, "no_trade_band must be non-negative"),
        ("transaction_cost_bps", -1.0, "transaction_cost_bps must be non-negative"),
        ("apply_band", 1, "apply_band must be boolean"),
        ("apply_costs", 1, "apply_costs must be boolean"),
        ("initial_capital", 0.0, "initial_capital must be positive"),
        ("decision_frequency", "quarterly", "must be daily, weekly or monthly"),
        ("weekly_anchor", "SAT", "weekly_anchor must be one of MON, TUE"),
        ("weekly_anchor", "MON", "weekly_anchor applies to weekly decisions"),
    ],
)
def test_config_rejects_invalid_values(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(SAME_CLOSE, **{field: value})


def test_two_day_horizon_applies_each_weight_to_its_own_day():
    config = BacktestConfig(forecast_horizon=2, apply_band=False, apply_costs=False)
    data = market([0.01, 0.02, 0.03])
    forecasts = forecasts_for_weights(data, [0.5, 1.0, 1.5], config)
    ledger = build_ledger(data, forecasts, config)
    assert np.allclose(ledger["return_weight"], [0.5, 1.0, 1.5])
    assert ledger["held_weight"].iloc[0] == pytest.approx(0.5)


def test_forecast_to_raw_weight_is_inverse():
    config = replace(SAME_CLOSE, apply_cap=False)
    data = market([0.01, 0.01, 0.01])
    forecasts = forecasts_for_weights(data, [1.0, 0.5, 2.0], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["raw_weight"].to_numpy() == pytest.approx([1.0, 0.5, 2.0])


def test_variance_sizing_squares_the_vol_rule_weight():
    config = replace(SAME_CLOSE, apply_cap=False, sizing_exponent=2.0)
    data = market([0.01, 0.01, 0.01])
    forecasts = forecasts_for_weights(
        data, [1.0, 0.5, 2.0], replace(config, sizing_exponent=1.0)
    )
    ledger = build_ledger(data, forecasts, config)
    assert ledger["raw_weight"].to_numpy() == pytest.approx([1.0, 0.25, 4.0])


def test_volatility_cutoff_zeroes_the_weight_above_it():
    config = replace(SAME_CLOSE, apply_cap=False, vol_cutoff_annual=0.15)
    data = market([0.01, 0.01, 0.01])
    forecasts = forecasts_for_weights(data, [1.0, 0.5, 2.0], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["raw_weight"].to_numpy() == pytest.approx([1.0, 0.0, 2.0])


def test_cap_can_be_applied_or_disabled():
    config = SAME_CLOSE
    data = market([0.01])
    forecasts = forecasts_for_weights(data, [2.0], config)
    capped = build_ledger(data, forecasts, config)
    uncapped = build_ledger(data, forecasts, replace(config, apply_cap=False))
    assert capped["target_weight"].iloc[0] == config.max_weight
    assert uncapped["target_weight"].iloc[0] == pytest.approx(2.0)
    assert (capped["target_weight"] >= 0.0).all()
    assert (uncapped["target_weight"] >= 0.0).all()


def test_execution_lag_uses_target_date_alignment():
    config = replace(SAME_CLOSE, apply_cap=False, execution_lag=1)
    data = market([0.01, 0.02, 0.03])
    forecasts = forecasts_for_weights(data, [0.5, 1.0, 2.0], config)
    same_close = build_ledger(data, forecasts, config)
    headline = build_ledger(data, forecasts, replace(config, execution_lag=2))
    counterfactual = build_ledger(
        data,
        forecasts,
        replace(config, execution_lag=0),
    )
    assert same_close["return_weight"].to_numpy() == pytest.approx([0.5, 1.0, 2.0])
    assert headline["return_weight"].iloc[1:].to_numpy() == pytest.approx([0.5, 1.0])
    assert counterfactual["return_weight"].iloc[:2].to_numpy() == pytest.approx(
        [1.0, 2.0]
    )
    spread = (1.0 + config.funding_spread_annual) ** (1.0 / TRADING_DAYS) - 1.0
    assert same_close["gross_return"].to_numpy() == pytest.approx(
        [0.5 * 0.01 + 0.5 * 0.0001, 0.02, 2.0 * 0.03 - (0.0001 + spread)]
    )
    assert headline["gross_return"].iloc[1:].to_numpy() == pytest.approx(
        [0.5 * 0.02 + 0.5 * 0.0001, 0.03]
    )
    assert counterfactual["gross_return"].iloc[:2].to_numpy() == pytest.approx(
        [0.01, 2.0 * 0.02 - (0.0001 + spread)]
    )
    assert np.isnan(headline["return_weight"].iloc[0])
    assert np.isnan(headline["turnover"].iloc[0])
    assert np.isnan(headline["portfolio_value_gross"].iloc[0])
    assert headline["turnover"].iloc[1] == pytest.approx(0.5)
    assert headline["transaction_cost"].iloc[1] == pytest.approx(0.5 * 0.0002)
    assert headline["portfolio_value_gross"].iloc[1] == pytest.approx(
        1.0 + 0.5 * 0.02 + 0.5 * 0.0001
    )
    assert np.isnan(counterfactual["return_weight"].iloc[-1])
    assert np.isnan(counterfactual["portfolio_value_gross"].iloc[-1])


def test_headline_lag_holds_inside_the_band_after_the_first_trade():
    config = replace(BacktestConfig(), apply_cap=False, no_trade_band=0.05)
    data = market([0.0, 0.0, 0.0, 0.0])
    forecasts = forecasts_for_weights(data, [1.0, 1.02, 1.5, 1.0], config)
    ledger = build_ledger(data, forecasts, config)
    assert np.isnan(ledger["held_weight"].iloc[0])
    assert ledger["turnover"].iloc[1] == pytest.approx(1.0)
    assert ledger["turnover"].iloc[2] == 0.0
    assert ledger["held_weight"].iloc[2] == ledger["drifted_weight"].iloc[2]
    assert ledger["held_weight"].iloc[3] == pytest.approx(1.5)
    assert ledger["turnover"].iloc[3] == pytest.approx(
        1.5 - ledger["drifted_weight"].iloc[3]
    )


def test_headline_lag_covers_leverage_and_funding():
    config = replace(BacktestConfig(), apply_cap=False, apply_band=False)
    data = market([0.01, 0.02, 0.03, 0.04], rf=0.001)
    forecasts = forecasts_for_weights(data, [1.5, 0.5, 1.2, 1.0], config)
    ledger = build_ledger(data, forecasts, config)
    spread = (1.0 + config.funding_spread_annual) ** (1.0 / TRADING_DAYS) - 1.0
    assert ledger["return_weight"].iloc[1:].to_numpy() == pytest.approx([1.5, 0.5, 1.2])
    assert ledger["gross_return"].iloc[1:].to_numpy() == pytest.approx(
        [
            1.5 * 0.02 - 0.5 * (0.001 + spread),
            0.5 * 0.03 + 0.5 * 0.001,
            1.2 * 0.04 - 0.2 * (0.001 + spread),
        ]
    )


def test_cash_and_funding_formulas_cover_all_weight_regions():
    index = pd.bdate_range("2024-01-02", periods=4)
    weights = pd.Series([0.0, 0.5, 1.0, 1.5], index=index)
    rf = pd.Series(0.001, index=index)
    spread = 0.0002
    cash = _cash_contribution(weights, rf, spread)
    assert cash.to_numpy() == pytest.approx(
        [0.001, 0.5 * 0.001, 0.0, -0.5 * (0.001 + spread)]
    )
    assert 0.0 * 0.02 + cash.iloc[0] == rf.iloc[0]


def test_gross_return_matches_spy_cash_and_funding_manually():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        funding_spread_annual=0.05,
    )
    data = market([0.02, 0.02, 0.02], rf=0.001)
    forecasts = forecasts_for_weights(data, [0.5, 1.0, 1.5], config)
    ledger = build_ledger(data, forecasts, config)
    spread = (1.0 + config.funding_spread_annual) ** (1.0 / TRADING_DAYS) - 1.0
    expected = [
        0.5 * 0.02 + 0.5 * 0.001,
        0.02,
        1.5 * 0.02 - 0.5 * (0.001 + spread),
    ]
    assert ledger["gross_return"].to_numpy() == pytest.approx(expected)


def test_gross_portfolio_value_compounds_three_dates():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_costs=False,
        initial_capital=100.0,
    )
    data = market([0.10, -0.05, 0.02], rf=0.0)
    forecasts = forecasts_for_weights(data, [1.0, 1.0, 1.0], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["portfolio_value_gross"].to_numpy() == pytest.approx(
        [110.0, 104.5, 106.59]
    )


@pytest.mark.parametrize(
    ("spy_return", "expected"),
    [
        (0.10, 0.5 * 1.10 / 1.05),
        (-0.10, 0.5 * 0.90 / 0.95),
    ],
)
def test_weight_drifts_with_spy_relative_to_cash(spy_return, expected):
    assert _drift_weight(0.5, spy_return, 0.0, 0.0, 0.0) == pytest.approx(expected)


def test_leveraged_weight_drifts_with_funding():
    gross_return = 1.5 * 0.10 - 0.5 * (0.001 + 0.0002)
    expected = 1.5 * 1.10 / (1.0 + gross_return)
    assert _drift_weight(1.5, 0.10, 0.001, 0.0002, 0.0) == pytest.approx(expected)


def test_transaction_cost_reduces_equity_used_for_next_drift():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        transaction_cost_bps=10.0,
    )
    data = market([0.0, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [0.5, 0.5], config)
    ledger = build_ledger(data, forecasts, config)
    first_cost = 0.5 * config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
    expected = 0.5 / (1.0 - first_cost)
    assert ledger["drifted_weight"].iloc[1] == pytest.approx(expected)
    assert ledger["held_weight"].iloc[1] == pytest.approx(expected)
    assert ledger["turnover"].iloc[1] == 0.0


def test_first_day_trades_from_cash_to_the_full_target():
    config = replace(SAME_CLOSE, apply_cap=False, apply_costs=False)
    data = market([0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [0.8], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["drifted_weight"].iloc[0] == 0.0
    assert ledger["held_weight"].iloc[0] == pytest.approx(0.8)
    assert ledger["turnover"].iloc[0] == pytest.approx(0.8)


def test_band_holds_inside_and_rebalances_fully_outside():
    config = replace(SAME_CLOSE, apply_cap=False, apply_costs=False)
    data = market([0.0, 0.0, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [0.5, 0.54, 0.61], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["drifted_weight"].to_numpy() == pytest.approx([0.0, 0.5, 0.5])
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5, 0.5, 0.61])
    assert ledger["turnover"].to_numpy() == pytest.approx([0.5, 0.0, 0.11])


def test_weight_exactly_on_band_does_not_trade():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_costs=False,
        no_trade_band=0.25,
    )
    data = market([0.0, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [0.5, 0.75], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5, 0.5])
    assert ledger["turnover"].to_numpy() == pytest.approx([0.5, 0.0])


def test_cap_breach_forces_rebalance_inside_band():
    config = replace(SAME_CLOSE, funding_spread_annual=0.0)
    data = market([-0.02, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [config.max_weight, config.max_weight])
    ledger = build_ledger(data, forecasts, config)
    first_cost = config.max_weight * config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
    drifted = config.max_weight * 0.98 / (1.0 - config.max_weight * 0.02 - first_cost)
    turnover = drifted - config.max_weight
    assert turnover < config.no_trade_band
    assert ledger["drifted_weight"].iloc[1] == pytest.approx(drifted)
    assert ledger["held_weight"].iloc[1] == pytest.approx(config.max_weight)
    assert ledger["turnover"].iloc[1] == pytest.approx(turnover)
    assert ledger["transaction_cost"].iloc[1] == pytest.approx(
        turnover * config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
    )


def test_disabling_band_rebalances_every_day():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
    )
    data = market([0.0, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [0.5, 0.52], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5, 0.52])
    assert ledger["turnover"].to_numpy() == pytest.approx([0.5, 0.02])


def test_weekly_frequency_uses_the_forecast_after_the_last_trading_day():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
    )
    data = market([0.0] * 7, rf=0.0)
    forecasts = forecasts_for_weights(
        data,
        [0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3],
        config,
    )
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].to_list() == [
        True,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert ledger["held_weight"].to_numpy() == pytest.approx(
        [0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 0.8]
    )


def test_weekly_schedule_shifts_with_the_headline_lag():
    config = replace(
        BacktestConfig(),
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
    )
    data = market([0.0] * 7, rf=0.0)
    forecasts = forecasts_for_weights(
        data,
        [0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3],
        config,
    )
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].to_list() == [
        False,
        True,
        False,
        False,
        True,
        False,
        False,
    ]
    assert np.isnan(ledger["held_weight"].iloc[0])
    assert ledger["held_weight"].iloc[1:].to_numpy() == pytest.approx(
        [0.5, 0.5, 0.5, 0.8, 0.8, 0.8]
    )


def test_weekly_schedule_with_a_two_day_horizon_flags_one_decision_per_week():
    config = replace(
        BacktestConfig(),
        forecast_horizon=2,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
    )
    data = market([0.0] * 7, rf=0.0)
    forecasts = forecasts_for_weights(
        data, [0.5, 0.6, 0.7, 0.8, 1.1, 1.2, 1.3], config
    ).iloc[1:]
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].to_list() == [
        True,
        False,
        False,
        True,
        False,
        False,
    ]


def test_weekly_anchor_moves_the_decision_to_that_weekday():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
        weekly_anchor="MON",
    )
    data = market([0.0] * 7, rf=0.0)
    forecasts = forecasts_for_weights(
        data,
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0],
        config,
    )
    ledger = build_ledger(data, forecasts, config)

    assert data.index[4].day_name() == "Monday"
    assert ledger["scheduled_rebalance"].to_list() == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert ledger["held_weight"].to_numpy() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5, 0.9, 0.9, 0.9]
    )


@pytest.mark.parametrize("anchor", WEEKDAYS)
def test_weekly_anchor_decides_once_a_week_on_the_anchor(anchor):
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
        weekly_anchor=anchor,
    )
    data = market([0.0] * 19, rf=0.0)
    forecasts = forecasts_for_weights(data, [1.0] * 19, config)
    ledger = build_ledger(data, forecasts, config)

    decisions = data.index[data.index.get_indexer(ledger.index) - 1]
    scheduled = ledger["scheduled_rebalance"].to_numpy()
    after_first_week = decisions >= data.index[4]
    decision_days = decisions[scheduled & after_first_week]
    assert len(decision_days) == 3
    assert set(decision_days.strftime("%a").str.upper()) == {anchor}


def test_weekly_anchor_with_a_two_day_horizon_flags_one_decision_per_week():
    config = replace(
        BacktestConfig(),
        forecast_horizon=2,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
        weekly_anchor="MON",
    )
    data = market([0.0] * 7, rf=0.0)
    forecasts = forecasts_for_weights(
        data, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0], config
    ).iloc[1:]
    ledger = build_ledger(data, forecasts, config)

    assert data.index[4].day_name() == "Monday"
    assert ledger["scheduled_rebalance"].to_list() == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]


def test_weekly_anchor_on_a_holiday_decides_on_the_prior_trading_day():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
        weekly_anchor="MON",
    )
    index = pd.DatetimeIndex(["2024-01-11", "2024-01-12", "2024-01-16", "2024-01-17"])
    data = market([0.0] * 3, rf=0.0).set_axis(index)
    forecasts = forecasts_for_weights(data, [0.5, 0.9, 1.2], config)
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].to_list() == [True, True, False]
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5, 0.9, 0.9])


def test_monthly_frequency_uses_the_forecast_after_month_end():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="monthly",
    )
    index = pd.DatetimeIndex(
        ["2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02"]
    )
    data = market([0.0] * 4, rf=0.0).set_axis(index)
    forecasts = forecasts_for_weights(data, [0.5, 0.6, 1.0, 1.2], config)
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].to_list() == [True, False, True, False]
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5, 0.5, 1.0, 1.0])


def test_weekly_frequency_handles_a_holiday_shortened_week():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency="weekly",
    )
    index = pd.DatetimeIndex(["2024-03-27", "2024-03-28", "2024-04-01", "2024-04-02"])
    data = market([0.0] * 3, rf=0.0).set_axis(index)
    forecasts = forecasts_for_weights(data, [0.5, 0.9, 1.2], config)
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].to_list() == [True, True, False]
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5, 0.9, 0.9])


def test_band_remains_active_on_scheduled_decisions():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_costs=False,
        decision_frequency="weekly",
    )
    data = market([0.0] * 5, rf=0.0)
    forecasts = forecasts_for_weights(data, [0.5, 0.9, 0.9, 0.54, 0.9], config)
    ledger = build_ledger(data, forecasts, config)

    assert ledger["scheduled_rebalance"].iloc[-2]
    assert ledger["held_weight"].to_numpy() == pytest.approx([0.5] * 5)
    assert ledger["turnover"].to_numpy() == pytest.approx([0.5, 0.0, 0.0, 0.0, 0.0])


def test_cap_breach_waits_for_the_next_scheduled_decision():
    config = replace(
        SAME_CLOSE,
        funding_spread_annual=0.0,
        apply_costs=False,
        decision_frequency="weekly",
    )
    data = market([-0.02, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [config.max_weight, config.max_weight])
    ledger = build_ledger(data, forecasts, config)

    assert not ledger["scheduled_rebalance"].iloc[1]
    assert ledger["drifted_weight"].iloc[1] > config.max_weight
    assert ledger["held_weight"].iloc[1] == ledger["drifted_weight"].iloc[1]
    assert ledger["turnover"].iloc[1] == 0.0


@pytest.mark.parametrize("frequency", ["weekly", "monthly"])
def test_lower_frequency_paths_do_not_use_future_forecasts(frequency):
    config = replace(
        BacktestConfig(),
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency=frequency,
    )
    data = market([0.0] * 70, rf=0.0)
    baseline = forecasts_for_weights(data, [0.5] * 70, config)
    changed = baseline.copy()
    first_changed = 20
    changed.iloc[first_changed:] = forecasts_for_weights(data, [1.2] * 70, config).iloc[
        first_changed:
    ]

    baseline_ledger = build_ledger(data, baseline, config)
    changed_ledger = build_ledger(data, changed, config)

    columns = ["held_weight", "turnover", "gross_return", "net_return"]
    pd.testing.assert_frame_equal(
        baseline_ledger.loc[:, columns].iloc[:first_changed],
        changed_ledger.loc[:, columns].iloc[:first_changed],
    )
    assert (
        not baseline_ledger["held_weight"]
        .iloc[first_changed:]
        .equals(changed_ledger["held_weight"].iloc[first_changed:])
    )


@pytest.mark.parametrize("frequency", ["weekly", "monthly"])
def test_lower_frequency_paths_do_not_use_future_market_returns(frequency):
    config = replace(
        BacktestConfig(),
        apply_cap=False,
        apply_band=False,
        apply_costs=False,
        decision_frequency=frequency,
    )
    first_changed = 20
    baseline_data = market([0.0] * 70, rf=0.0)
    changed_data = market(
        [0.0] * first_changed + [0.02, -0.01] * 25,
        rf=0.0,
    )
    forecasts = forecasts_for_weights(baseline_data, [0.5] * 70, config)

    baseline_ledger = build_ledger(baseline_data, forecasts, config)
    changed_ledger = build_ledger(changed_data, forecasts, config)

    columns = ["held_weight", "turnover", "gross_return", "net_return"]
    pd.testing.assert_frame_equal(
        baseline_ledger.loc[:, columns].iloc[:first_changed],
        changed_ledger.loc[:, columns].iloc[:first_changed],
    )
    assert (
        not baseline_ledger["gross_return"]
        .iloc[first_changed:]
        .equals(changed_ledger["gross_return"].iloc[first_changed:])
    )


def test_transaction_costs_reduce_gross_returns_to_net_returns():
    config = replace(
        SAME_CLOSE,
        apply_cap=False,
        apply_band=False,
        transaction_cost_bps=10.0,
        initial_capital=100.0,
    )
    data = market([0.0, 0.0], rf=0.0)
    forecasts = forecasts_for_weights(data, [0.5, 0.8], config)
    ledger = build_ledger(data, forecasts, config)
    first_cost = 0.5 * config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
    second_drift = 0.5 / (1.0 - first_cost)
    second_turnover = 0.8 - second_drift
    second_cost = second_turnover * config.transaction_cost_bps / BASIS_POINTS_PER_UNIT
    assert ledger["turnover"].to_numpy() == pytest.approx([0.5, second_turnover])
    assert ledger["transaction_cost"].to_numpy() == pytest.approx(
        [first_cost, second_cost]
    )
    assert ledger["net_return"].to_numpy() == pytest.approx(
        ledger["gross_return"] - ledger["transaction_cost"]
    )
    assert ledger["portfolio_value_gross"].to_numpy() == pytest.approx([100.0, 100.0])
    assert ledger["portfolio_value_net"].to_numpy() == pytest.approx(
        [99.95, 99.95 * (1.0 - second_cost)]
    )


def test_disabling_costs_sets_cost_to_zero():
    config = replace(SAME_CLOSE, apply_costs=False)
    data = market([0.01])
    forecasts = forecasts_for_weights(data, [1.0], config)
    ledger = build_ledger(data, forecasts, config)
    assert ledger["transaction_cost"].iloc[0] == 0.0
    assert ledger["net_return"].iloc[0] == ledger["gross_return"].iloc[0]


def test_ledger_has_the_required_columns():
    data = market([0.01])
    forecasts = forecasts_for_weights(data, [1.0])
    ledger = build_ledger(data, forecasts)
    assert list(ledger.columns) == [
        "forecast_vol",
        "rf",
        "return_close_to_close",
        "return_overnight",
        "return_intraday",
        "raw_weight",
        "target_weight",
        "return_weight",
        "scheduled_rebalance",
        "drifted_weight",
        "held_weight",
        "turnover",
        "spy_contribution",
        "cash_contribution",
        "transaction_cost",
        "gross_return",
        "net_return",
        "portfolio_value_gross",
        "portfolio_value_net",
    ]
