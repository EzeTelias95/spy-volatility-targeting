"""Walk-forward alignment, refit schedule and the absence of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from evaluation.walk_forward import refit_dates, walk_forward
from models.base import VolatilityModel

TRAIN_END = "2001-12-31"


class Recorder(VolatilityModel):
    """Accumulates every ingested day, so a repeat or a skip changes the forecast.

    predict returns the running sum of the returns seen so far. fit rebuilds that
    sum from df, which is what the interface promises update would have produced,
    so ingesting a day twice or refitting before updating shifts the series.
    """

    def __init__(self):
        self.total = None
        self.fit_windows = []
        self.seen = []

    def fit(self, df):
        self.fit_windows.append(df.index[-1])
        self.total = float(df["log_ret"].sum())

    def update(self, row):
        self.seen.append(row.name)
        self.total += float(row["log_ret"])

    def predict(self):
        return self.total


@pytest.fixture
def df():
    index = pd.bdate_range("2000-01-03", "2003-12-31")
    return pd.DataFrame(
        {"log_ret": np.arange(len(index), dtype=float), "realized_vol_gk": 0.01},
        index=index,
    )


def test_each_forecast_sums_the_days_before_it(df):
    """One day of drift in either direction, or a double ingestion, breaks this."""
    forecast, _ = walk_forward(Recorder(), df, TRAIN_END)
    expected = df["log_ret"].cumsum().shift(1).reindex(forecast.index)
    assert np.allclose(forecast, expected)


def test_covers_every_day_after_the_training_window(df):
    forecast, _ = walk_forward(Recorder(), df, TRAIN_END)
    assert forecast.index.equals(df.index[df.index > pd.Timestamp(TRAIN_END)])


def test_refits_on_month_ends_only(df):
    model = Recorder()
    walk_forward(model, df, TRAIN_END)
    oos = df.index[df.index > pd.Timestamp(TRAIN_END)]
    scheduled = model.fit_windows[1:]
    assert scheduled == sorted(set(refit_dates(oos)) - {oos[-1]})


def test_every_day_is_ingested_exactly_once(df):
    model = Recorder()
    walk_forward(model, df, TRAIN_END)
    oos = df.index[df.index > pd.Timestamp(TRAIN_END)]
    assert model.seen == list(oos[:-1])


def test_a_partial_train_end_is_not_taken_as_a_period(df):
    """df.loc[:"2001"] would end in December while the split starts in January."""
    forecast, _ = walk_forward(Recorder(), df, "2001")
    assert forecast.index[0] == pd.Timestamp("2001-01-02")


def test_refit_dates_are_the_last_trading_day_of_each_month(df):
    dates = refit_dates(df.index)
    assert (dates.to_period("M").value_counts() == 1).all()
    for day in dates:
        same_month = df.index.to_period("M") == day.to_period("M")
        assert len(df.index[(df.index > day) & same_month]) == 0


def test_a_two_day_horizon_labels_each_forecast_two_days_after_its_decision(df):
    model = Recorder()
    model.horizon = 2
    forecast, _ = walk_forward(model, df, TRAIN_END)
    oos = df.index[df.index > pd.Timestamp(TRAIN_END)]
    assert forecast.index.equals(oos[1:])
    expected = df["log_ret"].cumsum().shift(2).reindex(forecast.index)
    assert np.allclose(forecast, expected)
