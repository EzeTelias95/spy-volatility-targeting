"""Economic evaluation of daily strategy returns."""

import math
from numbers import Integral, Real

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from evaluation.statistics import automatic_bandwidth, long_run_covariance

TRADING_DAYS = 252
VOLATILITY_WINDOW = 21


def evaluate_strategy(
    returns: pd.Series,
    rf: pd.Series,
    turnover: pd.Series,
    target_vol_annual: float = 0.10,
) -> pd.Series:
    """Summarize the economic performance of one daily strategy."""
    _aligned_series({"returns": returns, "rf": rf, "turnover": turnover})
    _positive_target(target_vol_annual)
    if (turnover < 0.0).any():
        raise ValueError("turnover must be non-negative")
    realized_volatility = annualized_volatility(returns)
    tails = tail_metrics(returns)
    return pd.Series(
        {
            "sharpe": sharpe_ratio(returns, rf),
            "maximum_drawdown": maximum_drawdown(returns),
            "annualized_turnover": float(turnover.mean() * TRADING_DAYS),
            "realized_volatility": realized_volatility,
            "target_volatility_gap": realized_volatility - target_vol_annual,
            "vol_of_vol": volatility_control(returns, target_vol_annual),
            **tails.to_dict(),
        },
        dtype=float,
    )


def sharpe_ratio(returns: pd.Series, rf: pd.Series) -> float:
    """Annualized Sharpe ratio of daily excess simple returns."""
    _aligned_series({"returns": returns, "rf": rf})
    excess = returns - rf
    volatility = float(excess.std(ddof=1))
    if not math.isfinite(volatility) or volatility <= 0.0:
        raise ValueError("excess returns must have positive volatility")
    return float(excess.mean() / volatility * math.sqrt(TRADING_DAYS))


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily log returns."""
    values = _log_returns(returns)
    if len(values) < 2:
        raise ValueError("returns must contain at least two observations")
    return float(values.std(ddof=1) * math.sqrt(TRADING_DAYS))


def rolling_volatility(
    returns: pd.Series,
    window: int = VOLATILITY_WINDOW,
) -> pd.Series:
    """Hong rolling annualized volatility of daily log returns."""
    _positive_integer("window", window)
    values = _log_returns(returns)
    if len(values) < window:
        raise ValueError("returns must contain at least window observations")
    return values.rolling(window).std(ddof=1).dropna() * math.sqrt(TRADING_DAYS)


def volatility_control(
    returns: pd.Series,
    target_vol_annual: float,
    window: int = VOLATILITY_WINDOW,
) -> float:
    """Hong RMSE of rolling volatility from target, including its bias."""
    _positive_target(target_vol_annual)
    rolling = rolling_volatility(returns, window)
    return float(np.sqrt(((rolling - target_vol_annual) ** 2).mean()))


def maximum_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough loss in compounded wealth."""
    _returns(returns)
    wealth = (1.0 + returns).cumprod().to_numpy(dtype=float)
    wealth = np.concatenate([[1.0], wealth])
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    return float(drawdown.min())


def tail_metrics(returns: pd.Series) -> pd.Series:
    """Five worst daily returns, skewness and excess kurtosis."""
    _returns(returns)
    if len(returns) < 5:
        raise ValueError("returns must contain at least five observations")
    worst = returns.nsmallest(5).to_numpy(dtype=float)
    values = {f"worst_return_{rank}": value for rank, value in enumerate(worst, 1)}
    values["skew"] = float(returns.skew())
    values["kurtosis"] = float(returns.kurt())
    return pd.Series(values, dtype=float)


def spanning_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf: pd.Series,
) -> pd.Series:
    """Moreira-Muir spanning regression with a Newey-West alpha test."""
    _aligned_series(
        {
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "rf": rf,
        }
    )
    y = (strategy_returns - rf).to_numpy(dtype=float)
    benchmark = (benchmark_returns - rf).to_numpy(dtype=float)
    if len(y) < 3:
        raise ValueError("returns must contain at least three observations")
    if np.std(benchmark, ddof=1) <= 0.0:
        raise ValueError("benchmark excess returns must have positive volatility")
    x = np.column_stack([np.ones(len(y)), benchmark])
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coefficients
    scores = x * residuals[:, None]
    bread = np.linalg.inv(x.T @ x / len(x))
    alpha_influence = scores @ bread[0]
    bandwidth = automatic_bandwidth(alpha_influence)
    long_run = long_run_covariance(scores, bandwidth)
    covariance = bread @ long_run @ bread / len(x)
    alpha_se = math.sqrt(max(float(covariance[0, 0]), 0.0))
    alpha = float(coefficients[0])
    if alpha_se == 0.0:
        alpha_stat = 0.0 if alpha == 0.0 else math.copysign(math.inf, alpha)
    else:
        alpha_stat = alpha / alpha_se
    return pd.Series(
        {
            "alpha_daily": alpha,
            "alpha_annual": alpha * TRADING_DAYS,
            "beta": float(coefficients[1]),
            "alpha_stat": alpha_stat,
            "alpha_pvalue": _normal_pvalue(alpha_stat),
            "newey_west_lag": float(bandwidth),
        },
        dtype=float,
    )


def sharpe_difference_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    rf: pd.Series,
    bootstrap_reps: int = 1_000,
    seed: int = 42,
) -> pd.Series:
    """Ledoit-Wolf studentized stationary-bootstrap Sharpe comparison."""
    _aligned_series({"returns_a": returns_a, "returns_b": returns_b, "rf": rf})
    _positive_integer("bootstrap_reps", bootstrap_reps)
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    paired = pd.DataFrame(
        {
            "a": returns_a.to_numpy(dtype=float) - rf.to_numpy(dtype=float),
            "b": returns_b.to_numpy(dtype=float) - rf.to_numpy(dtype=float),
        }
    )
    if len(paired) < 3:
        raise ValueError("returns must contain at least three observations")
    difference, standard_error = _sharpe_difference(paired.to_numpy())
    if standard_error == 0.0:
        statistic = 0.0 if difference == 0.0 else math.copysign(math.inf, difference)
        return pd.Series(
            {
                "sharpe_difference": difference * math.sqrt(TRADING_DAYS),
                "statistic": statistic,
                "pvalue": 1.0 if difference == 0.0 else 0.0,
                "block_size": 1.0,
                "bootstrap_reps": float(bootstrap_reps),
            },
            dtype=float,
        )
    block_estimates = optimal_block_length(paired**2)["stationary"]
    finite_blocks = block_estimates[np.isfinite(block_estimates)]
    block_size = (
        max(1, math.ceil(float(finite_blocks.max()))) if len(finite_blocks) else 1
    )
    bootstrap = StationaryBootstrap(block_size, paired.to_numpy(), seed=seed)
    statistic = difference / standard_error
    observed = abs(statistic)
    exceedances = 0
    completed = 0
    while completed < bootstrap_reps:
        for positional, _ in bootstrap.bootstrap(bootstrap_reps - completed):
            try:
                sample_difference, sample_se = _sharpe_difference(positional[0])
            except ValueError:
                continue
            bootstrap_statistic = (
                abs((sample_difference - difference) / sample_se)
                if sample_se > 0.0
                else (0.0 if sample_difference == difference else math.inf)
            )
            exceedances += bootstrap_statistic >= observed
            completed += 1
    pvalue = (exceedances + 1.0) / (bootstrap_reps + 1.0)
    return pd.Series(
        {
            "sharpe_difference": difference * math.sqrt(TRADING_DAYS),
            "statistic": statistic,
            "pvalue": pvalue,
            "block_size": float(block_size),
            "bootstrap_reps": float(bootstrap_reps),
        },
        dtype=float,
    )


def _sharpe_difference(paired: np.ndarray) -> tuple[float, float]:
    means = paired.mean(axis=0)
    second_moments = (paired**2).mean(axis=0)
    variances = second_moments - means**2
    if (variances <= 0.0).any():
        difference = 0.0 if np.array_equal(paired[:, 0], paired[:, 1]) else math.nan
        if math.isnan(difference):
            raise ValueError("both excess return series must have positive volatility")
        return difference, 0.0
    standard_deviations = np.sqrt(variances)
    difference = float(
        means[0] / standard_deviations[0] - means[1] / standard_deviations[1]
    )
    moments = np.column_stack([paired, paired**2])
    gradient = np.array(
        [
            second_moments[0] / standard_deviations[0] ** 3,
            -second_moments[1] / standard_deviations[1] ** 3,
            -means[0] / (2.0 * standard_deviations[0] ** 3),
            means[1] / (2.0 * standard_deviations[1] ** 3),
        ]
    )
    bandwidth = automatic_bandwidth(moments @ gradient)
    long_run_variance = float(
        gradient @ long_run_covariance(moments, bandwidth) @ gradient
    )
    standard_error = math.sqrt(max(long_run_variance, 0.0) / len(paired))
    return difference, standard_error


def _normal_pvalue(statistic: float) -> float:
    if math.isinf(statistic):
        return 0.0
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def _log_returns(returns: pd.Series) -> pd.Series:
    _returns(returns)
    if (returns <= -1.0).any():
        raise ValueError("returns must be greater than -1")
    return np.log1p(returns)


def _returns(returns: pd.Series) -> None:
    _aligned_series({"returns": returns})


def _aligned_series(series: dict[str, pd.Series]) -> None:
    index = None
    for name, values in series.items():
        if not isinstance(values, pd.Series):
            raise ValueError(f"{name} must be a Series")
        if not isinstance(values.index, pd.DatetimeIndex):
            raise ValueError(f"{name} must have a DatetimeIndex")
        if not values.index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be sorted")
        if not values.index.is_unique:
            raise ValueError(f"{name} index must not contain duplicates")
        if index is None:
            index = values.index
        elif not values.index.equals(index):
            raise ValueError(f"{name} is not aligned")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} must be finite")


def _positive_target(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError("target_vol_annual must be finite and positive")


def _positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
