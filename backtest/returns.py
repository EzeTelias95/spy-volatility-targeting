"""Adjusted instrument returns aligned with volatility forecasts."""

import numpy as np
import pandas as pd

RECONCILIATION_RTOL = 1e-12
RECONCILIATION_ATOL = 1e-12


def build_instrument_returns(market_data: pd.DataFrame) -> pd.DataFrame:
    """Build adjusted close-to-close, overnight and intraday returns."""
    _validate_index("market_data", market_data.index)
    adjusted_open, adjusted_close = _adjusted_prices(market_data)
    if "ret" not in market_data.columns:
        raise ValueError("market_data is missing columns: ['ret']")
    close_to_close = market_data["ret"].astype(float)
    _validate_close_to_close(close_to_close)
    previous_close = adjusted_close.shift(1)
    returns = pd.DataFrame(
        {
            "return_close_to_close": close_to_close,
            "return_overnight": adjusted_open / previous_close - 1.0,
            "return_intraday": adjusted_close / adjusted_open - 1.0,
        },
        index=market_data.index,
    )
    _validate_reconciliation(returns)
    return returns


def prepare_backtest_data(
    market_data: pd.DataFrame,
    forecasts: pd.Series,
) -> pd.DataFrame:
    """Align forecasts after computing returns on the full market history.

    Computing before selection preserves the previous close needed by the first
    forecast date and refuses missing dates instead of intersecting silently.
    """
    returns = build_instrument_returns(market_data)
    _validate_index("forecasts", forecasts.index)
    forecast_values = forecasts.to_numpy(dtype=float)
    if not np.isfinite(forecast_values).all() or not (forecast_values > 0.0).all():
        raise ValueError("forecasts must be finite and positive")
    if "rf" not in market_data.columns:
        raise ValueError("market_data is missing columns: ['rf']")
    missing = forecasts.index.difference(market_data.index)
    if len(missing) > 0:
        raise ValueError(f"forecast dates missing from market_data: {_dates(missing)}")
    positions = market_data.index.get_indexer(forecasts.index)
    if len(positions) > 1 and not (np.diff(positions) == 1).all():
        raise ValueError("forecast dates must be consecutive market dates")
    aligned_returns = returns.loc[forecasts.index]
    if not np.isfinite(aligned_returns.to_numpy(dtype=float)).all():
        bad = aligned_returns.index[aligned_returns.isna().any(axis=1)]
        raise ValueError(f"returns unavailable for forecast dates: {_dates(bad)}")
    rf = market_data.loc[forecasts.index, "rf"].astype(float)
    if not np.isfinite(rf.to_numpy()).all():
        bad = rf.index[~np.isfinite(rf.to_numpy())]
        raise ValueError(f"rf unavailable for forecast dates: {_dates(bad)}")
    result = aligned_returns.copy()
    result.insert(0, "forecast_vol", forecasts.astype(float))
    result.insert(1, "rf", rf)
    return result


def _adjusted_prices(market_data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return open and close on one adjusted price scale.

    The close adjustment factor is applied to the raw open so overnight and
    intraday returns compose to the adjusted close-to-close return.
    """
    required = {"open", "close", "adj_close"}
    missing = required - set(market_data.columns)
    if missing:
        raise ValueError(f"market_data is missing columns: {sorted(missing)}")
    raw_open = market_data["open"].astype(float)
    raw_close = market_data["close"].astype(float)
    adjusted_close = market_data["adj_close"].astype(float)
    adjusted_open = raw_open * adjusted_close / raw_close
    prices = np.column_stack([adjusted_open, adjusted_close])
    if not np.isfinite(prices).all() or not (prices > 0.0).all():
        raise ValueError("adjusted open and close prices must be finite and positive")
    return adjusted_open, adjusted_close


def _validate_close_to_close(returns: pd.Series) -> None:
    missing = returns.isna()
    if missing.sum() > 1:
        raise ValueError("ret must contain at most one NaN")
    if missing.iloc[1:].any():
        raise ValueError("ret NaN is allowed only on the first date")
    if not np.isfinite(returns.dropna().to_numpy()).all():
        raise ValueError("ret must be finite")


def _validate_reconciliation(returns: pd.DataFrame) -> None:
    valid = returns.dropna()
    composed = (1.0 + valid["return_overnight"]) * (1.0 + valid["return_intraday"])
    total = 1.0 + valid["return_close_to_close"]
    matches = np.isclose(
        composed,
        total,
        rtol=RECONCILIATION_RTOL,
        atol=RECONCILIATION_ATOL,
    )
    if not matches.all():
        bad = valid.index[~matches]
        raise ValueError(f"return reconciliation failed: {_dates(bad)}")


def _validate_index(label: str, index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"{label} must have a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be sorted")
    if not index.is_unique:
        raise ValueError(f"{label} index must not contain duplicates")


def _dates(index: pd.DatetimeIndex) -> list[str]:
    return [date.date().isoformat() for date in index]
