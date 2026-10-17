"""Expanding-window walk-forward loop shared by every consumer of a model."""

import logging

import numpy as np
import pandas as pd

from models.base import VolatilityModel

log = logging.getLogger(__name__)


def refit_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each month."""
    last = pd.Series(index, index=index).groupby(index.to_period("M")).last()
    return pd.DatetimeIndex(last)


def _params(model: VolatilityModel):
    """What the model exposes as params, None when it estimates nothing.

    Looked up without getattr's default, so a failure inside a params property
    surfaces instead of silently dropping a refit.
    """
    if hasattr(type(model), "params") or "params" in vars(model):
        return model.params
    return None


def walk_forward(
    model: VolatilityModel, df: pd.DataFrame, train_end: str | pd.Timestamp
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily forecasts over the days after train_end, labelled by the day forecast.

    The fixed sequence: an initial fit on the training window, then for
    every day update(t), a refit on the last trading day of each month, and
    predict() for t+horizon. Each forecast is stored under the day it forecasts,
    so the series lines up directly with the realized volatility it is scored
    against. A horizon beyond one moves every label that many days after the
    decision and drops the forecasts that land past the end of the sample.

    train_end is resolved to a timestamp before slicing: a partial date such as
    "2011" would otherwise end the training window at the close of the period and
    open the out-of-sample window at its start, replaying a year of training days
    as forecasts.

    The second frame holds whatever the model exposes as params at each refit,
    empty for models that estimate nothing.
    """
    cut = pd.Timestamp(train_end)
    train = df.loc[:cut]
    days = df.index[df.index > cut]
    if len(train) + len(days) != len(df):
        raise ValueError(f"train_end {cut.date()} does not split the sample cleanly")
    model.fit(train)
    refits = set(refit_dates(days))
    forecasts = {train.index[-1]: model.predict()}
    params = {}
    done = 0
    for previous in days[:-1]:
        model.update(df.loc[previous])
        if previous in refits:
            model.fit(df.loc[:previous])
            value = _params(model)
            if value is not None:
                params[previous] = value
            done += 1
            log.info(
                "%s: refit %d of %d at %s",
                type(model).__name__,
                done,
                len(refits),
                previous.date(),
            )
        forecasts[previous] = model.predict()
    return _label_by_target(forecasts, df.index, model.horizon), pd.DataFrame(params).T


def _label_by_target(
    forecasts: dict, index: pd.DatetimeIndex, horizon: int
) -> pd.Series:
    """Move each forecast from its decision day to the day it is for."""
    positions = index.get_indexer(list(forecasts)) + horizon
    inside = positions < len(index)
    values = np.asarray(list(forecasts.values()), dtype=float)[inside]
    return pd.Series(values, index=index[positions[inside]], name="forecast")
