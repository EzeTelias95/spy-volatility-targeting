"""Mechanical tests for strategy evaluation metrics."""

from math import sqrt

import numpy as np
import pandas as pd
import pytest

import evaluation.strategy as strategy_module
from evaluation.strategy import (
    TRADING_DAYS,
    annualized_volatility,
    evaluate_strategy,
    maximum_drawdown,
    rolling_volatility,
    sharpe_difference_test,
    sharpe_ratio,
    spanning_test,
    tail_metrics,
    volatility_control,
)


def series(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-02", periods=len(values)))


def test_sharpe_ratio_matches_daily_excess_returns():
    returns = series([0.01, -0.01, 0.02])
    rf = series([0.001, 0.001, 0.001])
    excess = returns - rf
    expected = excess.mean() / excess.std(ddof=1) * sqrt(TRADING_DAYS)
    assert sharpe_ratio(returns, rf) == pytest.approx(expected)


def test_maximum_drawdown_compounds_wealth():
    returns = series([0.10, -0.20, 0.05])
    assert maximum_drawdown(returns) == pytest.approx(-0.20)
    assert maximum_drawdown(series([-0.10, 0.05])) == pytest.approx(-0.10)


def test_hong_rolling_volatility_control_and_vol_of_vol_match_manual_values():
    returns = series([0.01, -0.01, 0.02, -0.02, 0.01, -0.01])
    target = 0.10
    rolling = rolling_volatility(returns, window=3)
    expected = np.log1p(returns).rolling(3).std(ddof=1).dropna() * sqrt(TRADING_DAYS)
    assert rolling.to_numpy() == pytest.approx(expected.to_numpy())
    summary_returns = series(np.resize(returns.to_numpy(), 21))
    metrics = evaluate_strategy(
        summary_returns,
        series([0.0] * 21),
        series([0.1] * 21),
        target,
    )
    default_rolling = rolling_volatility(summary_returns)
    expected_control = np.sqrt(((default_rolling - target) ** 2).mean())
    assert metrics["vol_of_vol"] == pytest.approx(expected_control)
    assert volatility_control(summary_returns, target) == pytest.approx(
        expected_control
    )
    assert metrics["target_volatility_gap"] == pytest.approx(
        metrics["realized_volatility"] - target
    )


def test_annualized_volatility_uses_log_returns():
    returns = series([0.01, -0.02, 0.03])
    expected = np.log1p(returns).std(ddof=1) * sqrt(TRADING_DAYS)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_strategy_summary_annualizes_turnover():
    returns = series(np.resize([0.01, -0.01], 21))
    rf = series([0.0] * 21)
    turnover = series([0.02] * 21)
    metrics = evaluate_strategy(returns, rf, turnover)
    assert metrics["annualized_turnover"] == pytest.approx(0.02 * TRADING_DAYS)


def test_tail_metrics_report_the_five_worst_days():
    returns = series([0.03, -0.04, 0.01, -0.02, 0.02, -0.01])
    metrics = tail_metrics(returns)
    assert metrics[
        [f"worst_return_{i}" for i in range(1, 6)]
    ].to_numpy() == pytest.approx([-0.04, -0.02, -0.01, 0.01, 0.02])


def test_spanning_regression_recovers_alpha_and_beta():
    benchmark = series(np.linspace(-0.02, 0.02, 101))
    noise = series(np.sin(np.arange(101)) * 0.0001)
    rf = series([0.0001] * 101)
    strategy = rf + 0.0002 + 0.8 * (benchmark - rf) + noise
    result = spanning_test(strategy, benchmark, rf)
    assert result["alpha_daily"] == pytest.approx(0.0002, abs=2e-6)
    assert result["beta"] == pytest.approx(0.8, abs=4e-4)
    assert 0.0 <= result["alpha_pvalue"] <= 1.0


def test_spanning_bandwidth_uses_alpha_influence(monkeypatch):
    benchmark = series(np.linspace(-0.02, 0.02, 20))
    rf = series([0.0] * 20)
    strategy = 0.0002 + 0.8 * benchmark + series(np.sin(np.arange(20)) * 0.001)
    received = None

    def capture_bandwidth(values):
        nonlocal received
        received = values
        return 0

    monkeypatch.setattr(strategy_module, "automatic_bandwidth", capture_bandwidth)
    spanning_test(strategy, benchmark, rf)

    x = np.column_stack([np.ones(len(benchmark)), benchmark.to_numpy()])
    y = strategy.to_numpy()
    residuals = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    scores = x * residuals[:, None]
    bread = np.linalg.inv(x.T @ x / len(x))
    assert received == pytest.approx(scores @ bread[0])


def test_sharpe_difference_is_reproducible_and_paired():
    rng = np.random.default_rng(7)
    common = rng.normal(0.0002, 0.01, 200)
    returns_a = series(common + 0.0003)
    returns_b = series(common)
    rf = series([0.0] * 200)
    first = sharpe_difference_test(returns_a, returns_b, rf, bootstrap_reps=49)
    second = sharpe_difference_test(returns_a, returns_b, rf, bootstrap_reps=49)
    assert first.equals(second)
    assert first["sharpe_difference"] > 0.0
    assert first["statistic"] > 0.0
    assert 0.0 <= first["pvalue"] <= 1.0
    assert first["block_size"] >= 1.0


def test_identical_sharpe_series_give_a_zero_statistic():
    returns = series(np.resize([0.01, -0.01], 20))
    rf = series([0.0] * 20)
    result = sharpe_difference_test(returns, returns, rf, bootstrap_reps=9)
    assert result["sharpe_difference"] == 0.0
    assert result["statistic"] == 0.0
    assert result["pvalue"] == 1.0


def test_sharpe_difference_redraws_degenerate_bootstrap_samples(monkeypatch):
    class BootstrapWithDegenerateFirstSample:
        def __init__(self, block_size, values, seed):
            self.values = values
            self.first = True

        def bootstrap(self, reps):
            for _ in range(reps):
                if self.first:
                    sample = np.repeat(self.values[:1], len(self.values), axis=0)
                    self.first = False
                else:
                    sample = self.values
                yield (sample,), {}

    monkeypatch.setattr(
        strategy_module,
        "StationaryBootstrap",
        BootstrapWithDegenerateFirstSample,
    )
    returns_a = series(np.resize([0.01, -0.01, 0.02, -0.02], 20))
    returns_b = series(np.resize([-0.01, 0.01, -0.02, 0.02], 20))
    result = sharpe_difference_test(
        returns_a,
        returns_b,
        series([0.0] * 20),
        bootstrap_reps=3,
    )
    assert 0.0 <= result["pvalue"] <= 1.0


def test_spanning_rejects_a_constant_benchmark():
    values = series([0.01, -0.01, 0.02])
    with pytest.raises(ValueError, match="benchmark excess returns"):
        spanning_test(values, series([0.0] * 3), series([0.0] * 3))


@pytest.mark.parametrize(
    ("returns", "message"),
    [
        ([0.0, -1.0, 0.0], "greater than -1"),
        ([0.0, np.nan, 0.0], "finite"),
    ],
)
def test_log_volatility_rejects_invalid_returns(returns, message):
    with pytest.raises(ValueError, match=message):
        annualized_volatility(series(returns))


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (pd.Index([1, 2, 3]), "DatetimeIndex"),
        (pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-04"]), "sorted"),
        (
            pd.DatetimeIndex(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "duplicates",
        ),
    ],
)
def test_strategy_metrics_reject_invalid_indexes(index, message):
    returns = pd.Series([0.01, -0.01, 0.02], index=index)
    with pytest.raises(ValueError, match=message):
        maximum_drawdown(returns)
