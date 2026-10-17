"""Garman-Klass + overnight realized volatility estimator."""

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GK_CO_WEIGHT = 2.0 * np.log(2.0) - 1.0
PARKINSON_SCALE = 4.0 * np.log(2.0)
VARIANCE_FLOOR = 1e-8


def _log_days(label: str, days: pd.DatetimeIndex) -> None:
    log.log(
        logging.WARNING if len(days) > 0 else logging.INFO,
        "%s on %d days: %s",
        label,
        len(days),
        [d.date().isoformat() for d in days],
    )


def realized_vol_gk(df: pd.DataFrame) -> pd.Series:
    """Daily total volatility (standard deviation, not annualized).

    Variance = Garman-Klass intraday term on raw OHLC plus the squared
    dividend-adjusted overnight log return. If a day's variance is <= 0 the
    intraday term is replaced with Parkinson's, keeping the overnight term;
    remaining values below VARIANCE_FLOOR are floored. Both cases are logged.
    The first row is NaN (no previous close).

    Expects rows that pass the OHLC sanity filter. On consistent rows
    |ln(C/O)| <= ln(H/L), so the intraday term is non-negative and the
    Parkinson branch is reachable only on flat days.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    intraday = 0.5 * log_hl**2 - GK_CO_WEIGHT * log_co**2
    overnight = np.log((df["open"] + df["dividend"]) / df["close"].shift(1)) ** 2
    variance = intraday + overnight

    parkinson_days = variance.index[variance <= 0]
    variance.loc[parkinson_days] = (
        log_hl.loc[parkinson_days] ** 2 / PARKINSON_SCALE
        + overnight.loc[parkinson_days]
    )
    _log_days("parkinson substitution", parkinson_days)

    floored_days = variance.index[variance < VARIANCE_FLOOR]
    variance = variance.clip(lower=VARIANCE_FLOOR)
    _log_days("variance floor applied", floored_days)
    return np.sqrt(variance).rename("realized_vol_gk")
