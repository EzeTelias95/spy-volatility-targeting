"""Execution timing utilities."""

from numbers import Integral

import pandas as pd

from .returns import _validate_index


def apply_execution_lag(
    target_weights: pd.Series,
    execution_lag: int,
    forecast_horizon: int = 1,
) -> pd.Series:
    """Align target-dated weights with their return dates.

    The input index is the forecast target date, not the decision date. With
    one-day forecasts a lag of two is the headline case: the weight decided at
    the close of t trades at the close of t+1 and earns the return of t+2, so the
    first weight is unavailable. A lag of one executes at the close of t and
    leaves weights aligned with the next day's return. A lag of zero is the
    deliberate look-ahead counterfactual and moves each target weight to the
    decision date. A two-day forecast is already labelled by its return day, so
    under the headline lag it needs no shift. The original index is preserved
    and unavailable weights remain NaN.
    """
    if (
        isinstance(execution_lag, bool)
        or not isinstance(execution_lag, Integral)
        or execution_lag not in {0, 1, 2}
    ):
        raise ValueError("execution_lag must be 0, 1 or 2")
    if (
        isinstance(forecast_horizon, bool)
        or not isinstance(forecast_horizon, Integral)
        or forecast_horizon not in {1, 2}
    ):
        raise ValueError("forecast_horizon must be 1 or 2")
    _validate_index("target_weights", target_weights.index)
    return target_weights.shift(execution_lag - forecast_horizon)
