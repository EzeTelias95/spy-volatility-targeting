"""Statistical evaluation of daily volatility forecasts.

QLIKE is the headline metric and RMSE the secondary one, both computed on
variances. H1 is a two-sided Diebold-Mariano test on the QLIKE loss
differentials against a benchmark model. The proxy robustness check is this
same entry point called with |log_ret| as the proxy.

Realized utility follows Bollerslev, Hood, Huss and Pedersen (2018): a
mean-variance investor with risk aversion RISK_AVERSION targets an asset with
constant Sharpe ratio SHARPE_RATIO, holding SR / (gamma * forecast_vol). Per
day the utility is SR^2 / gamma * r - SR^2 / (2 gamma) * r^2 with r the ratio
of realized to forecast volatility, so a perfect forecast earns SR^2 / (2 gamma)
of wealth per year and the ratio to that ceiling needs neither parameter. It
is descriptive: it scores the square root of the proxy, which is not robust to
proxy noise the way QLIKE is.
"""

import math

import numpy as np
import pandas as pd

from data.realized_vol import VARIANCE_FLOOR
from evaluation.statistics import long_run_variance

SHARPE_RATIO = 0.4
RISK_AVERSION = 2.0
PERFECT_UTILITY = SHARPE_RATIO**2 / (2.0 * RISK_AVERSION)


def evaluate_forecasts(
    forecasts: dict[str, pd.Series],
    proxy_vol: pd.Series,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Score daily volatility forecasts against a realized volatility proxy.

    Every series holds daily volatility indexed by the day being forecast.
    Alignment belongs to the walk-forward loop, so an index that differs from
    the proxy's is an error rather than something to reconcile silently, and so
    are missing values. Variances are floored at VARIANCE_FLOOR before scoring
    and the floored days are reported alongside the metrics. Given a benchmark
    name, every other model also gets a Diebold-Mariano test of its QLIKE
    losses against the benchmark's; a positive statistic means the model loses
    more than the benchmark. Realized utility is reported in wealth per year
    and as a share of the perfect-forecast ceiling.
    """
    _check("proxy", proxy_vol, proxy_vol)
    proxy_var, proxy_floored = _variances(proxy_vol)

    losses, rows = qlike_losses(forecasts, proxy_vol), {}
    for name, vol in forecasts.items():
        forecast_var, forecast_floored = _variances(vol)
        utility = _realized_utility(forecast_var, proxy_var)
        rows[name] = {
            "qlike": float(losses[name].mean()),
            "rmse": float(np.sqrt(((forecast_var - proxy_var) ** 2).mean())),
            "realized_utility": utility,
            "realized_utility_share": utility / PERFECT_UTILITY,
            "forecast_floored_days": forecast_floored,
            "proxy_floored_days": proxy_floored,
        }
    table = pd.DataFrame.from_dict(rows, orient="index").rename_axis("model")

    if benchmark is not None:
        if benchmark not in losses:
            raise ValueError(f"benchmark {benchmark} is not among the forecasts")
        tests = {
            name: diebold_mariano(loss, losses[benchmark])
            for name, loss in losses.items()
            if name != benchmark
        }
        table = table.join(pd.DataFrame(tests, index=["dm_stat", "dm_pvalue"]).T)
    return table


def qlike_losses(forecasts: dict[str, pd.Series], proxy_vol: pd.Series) -> pd.DataFrame:
    """Daily QLIKE losses on floored variances, one column per model."""
    _check("proxy", proxy_vol, proxy_vol)
    proxy_var, _ = _variances(proxy_vol)
    losses = {}
    for name, vol in forecasts.items():
        _check(name, vol, proxy_vol)
        losses[name] = _qlike(_variances(vol)[0], proxy_var)
    return pd.DataFrame(losses)


def diebold_mariano(loss_a, loss_b) -> tuple[float, float]:
    """Two-sided Diebold-Mariano statistic and its normal p-value.

    The long-run variance of the loss differential uses a Bartlett kernel with
    the automatic bandwidth of Newey and West (1994), which adapts to how little
    autocorrelation a one-day-ahead differential actually has. A degenerate
    differential, identical loss series included, gives a zero statistic.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    mean = float(d.mean())
    variance = long_run_variance(d - mean)
    if variance <= 0.0:
        return 0.0, 1.0
    stat = mean / math.sqrt(variance / len(d))
    return stat, math.erfc(abs(stat) / math.sqrt(2.0))


def _check(name: str, vol: pd.Series, proxy_vol: pd.Series) -> None:
    if not vol.index.equals(proxy_vol.index):
        raise ValueError(f"{name} is not indexed like the proxy")
    if not np.isfinite(vol.astype(float)).all():
        raise ValueError(f"{name} has missing values")


def _variances(vol: pd.Series) -> tuple[pd.Series, int]:
    variance = vol.astype(float) ** 2
    floored = int((variance <= VARIANCE_FLOOR).sum())
    return variance.clip(lower=VARIANCE_FLOOR), floored


def _qlike(forecast_var: pd.Series, proxy_var: pd.Series) -> pd.Series:
    ratio = proxy_var / forecast_var
    return ratio - np.log(ratio) - 1.0


def _realized_utility(forecast_var: pd.Series, proxy_var: pd.Series) -> float:
    ratio = np.sqrt(proxy_var / forecast_var)
    scale = SHARPE_RATIO**2 / RISK_AVERSION
    return float((scale * (ratio - 0.5 * ratio**2)).mean())
