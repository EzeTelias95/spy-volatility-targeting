"""Look-ahead ceilings for the volatility targeting strategy.

Neither series is attainable: both size a day with volatility observed on that
day or later. They bound what any forecast could deliver under the same sizing
rule, cap and costs, and belong in the results chapter as diagnostics, never in
the tables that rank models. Both are indexed by the day being forecast, the
convention the walk-forward loop uses, so they feed the backtester unchanged.

Most of that bound is return timing rather than risk control, because a
volatility peak is a drawdown. `sign_randomized_market` builds the counterfactual
that separates the two, and is a diagnostic on the same footing.
"""

import numpy as np
import pandas as pd

REGIME_WINDOW = 21
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close")


def daily_oracle(realized_vol: pd.Series) -> pd.Series:
    """Perfect foresight of the volatility of the day being sized.

    The loosest ceiling, and partly unattainable for a reason beyond foresight:
    a day's realized volatility carries the sign of that day's return, so the
    weights also time returns, which no honest forecast can do.
    """
    _check(realized_vol)
    return realized_vol.rename("forecast")


def regime_oracle(
    realized_vol: pd.Series,
    window: int = REGIME_WINDOW,
) -> pd.Series:
    """Perfect foresight of the mean variance over the next window days.

    Knows the volatility regime a month ahead but nothing about any single day,
    which strips most of the return timing out of the daily oracle. Every value
    averages exactly window days, so the last window - 1 days are dropped.
    """
    _check(realized_vol)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError("window must be a positive integer")
    if len(realized_vol) < window:
        raise ValueError("realized_vol must be at least window long")
    forward = realized_vol[::-1] ** 2
    mean_variance = forward.rolling(window).mean()[::-1].dropna()
    return np.sqrt(mean_variance).rename("forecast")


def sign_randomized_market(
    market_data: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Market data with the link between volatility and return sign removed.

    Flips the sign of demeaned close-to-close returns over dates and rescales
    every price of a day by one factor, so intraday ratios survive and with them
    the Garman-Klass estimator, the OHLC sanity bounds and the magnitude of every
    move. Volatility is therefore as clustered and as predictable as before, and
    the forecasts of every model are unchanged; what dies is the contemporaneous
    link between the size of a move and its direction. Running the backtester on
    the result separates what a forecast earns by controlling risk from what the
    oracles earn by knowing the sign of the future.
    """
    _check_market(market_data, dates)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy Generator")
    returns = market_data["ret"].astype(float)
    window = returns.loc[dates].to_numpy()
    mean = float(window.mean())
    signs = rng.choice((-1.0, 1.0), size=len(dates))
    flipped = returns.copy()
    flipped.loc[dates] = mean + signs * (window - mean)
    first_close = float(market_data["adj_close"].iloc[0])
    path = first_close * (1.0 + flipped.fillna(0.0)).cumprod()
    scale = path / market_data["adj_close"].astype(float)
    result = market_data.copy()
    for column in PRICE_COLUMNS:
        result[column] = market_data[column].astype(float) * scale
    result["ret"] = result["adj_close"] / result["adj_close"].shift(1) - 1.0
    result.iloc[0, result.columns.get_loc("ret")] = returns.iloc[0]
    result["log_ret"] = np.log1p(result["ret"])
    return result


def _check(realized_vol: pd.Series) -> None:
    if not isinstance(realized_vol, pd.Series):
        raise ValueError("realized_vol must be a Series")
    if not isinstance(realized_vol.index, pd.DatetimeIndex):
        raise ValueError("realized_vol must have a DatetimeIndex")
    if not realized_vol.index.is_monotonic_increasing:
        raise ValueError("realized_vol index must be sorted")
    if not realized_vol.index.is_unique:
        raise ValueError("realized_vol index must not contain duplicates")
    values = realized_vol.to_numpy(dtype=float)
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise ValueError("realized_vol must be finite and positive")


def _check_market(market_data: pd.DataFrame, dates: pd.DatetimeIndex) -> None:
    if not isinstance(market_data, pd.DataFrame):
        raise ValueError("market_data must be a DataFrame")
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise ValueError("market_data must have a DatetimeIndex")
    if not market_data.index.is_monotonic_increasing:
        raise ValueError("market_data index must be sorted")
    if not market_data.index.is_unique:
        raise ValueError("market_data index must not contain duplicates")
    missing = sorted({*PRICE_COLUMNS, "ret"} - set(market_data.columns))
    if missing:
        raise ValueError(f"market_data is missing columns: {missing}")
    prices = market_data[list(PRICE_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or not (prices > 0.0).all():
        raise ValueError("prices must be finite and positive")
    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("dates must be a DatetimeIndex")
    if len(dates) == 0:
        raise ValueError("dates must not be empty")
    if len(dates.difference(market_data.index)) > 0:
        raise ValueError("dates must be contained in the market_data index")
    window = market_data.loc[dates, "ret"].to_numpy(dtype=float)
    if not np.isfinite(window).all():
        raise ValueError("ret must be finite over dates")
