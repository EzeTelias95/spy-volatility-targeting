"""VIX feature alignment and the buffer the daily path feeds predict."""

import numpy as np
import pandas as pd
import pytest

from models.gbm_vix import TRADING_DAYS, VIX_SCALE, GradientBoostingVIX
from tests.test_gbm import frame


def vix_frame(n, seed):
    """The base frame plus a VIX path that is not a function of the target."""
    df = frame(n, seed)
    rng = np.random.default_rng(seed + 1000)
    df["vix"] = 20.0 * np.exp(0.3 * rng.standard_normal(n).cumsum() / np.sqrt(n))
    return df


def design(df):
    model = GradientBoostingVIX()
    return model.design(model.daily(df))


def test_features_use_no_data_after_day_t():
    full = design(vix_frame(400, 3))
    truncated = design(vix_frame(400, 3).iloc[:-50])
    shared = full.index.intersection(truncated.index)
    pd.testing.assert_frame_equal(full.loc[shared], truncated.loc[shared])


def test_vix_variance_is_the_quote_as_a_daily_log_variance():
    df = vix_frame(200, 7)
    model = GradientBoostingVIX()
    built = model.features(model.daily(df))
    expected = np.log((df["vix"] / VIX_SCALE) ** 2 / TRADING_DAYS)
    assert np.allclose(built["vix_var"], expected)


def test_premium_is_vix_variance_over_realized_variance():
    df = vix_frame(200, 8)
    model = GradientBoostingVIX()
    built = model.features(model.daily(df))
    assert np.allclose(
        built["vix_premium"].dropna(),
        (built["vix_var"] - built["mean_1"]).dropna(),
    )


def test_update_feeds_predict_the_same_features_as_fit():
    df = vix_frame(400, 11)
    model = GradientBoostingVIX()
    model.fit(df.iloc[:-1])
    for _, row in df.iloc[-model.days.maxlen - 1 : -1].iterrows():
        model.update(row)
    daily = pd.DataFrame(list(model.days), columns=model.DAILY)
    streamed = model.features(daily).iloc[-1]
    batch = model.features(model.daily(df.iloc[:-1])).iloc[-1]
    assert streamed.to_numpy() == pytest.approx(batch.to_numpy())


def test_predict_returns_a_positive_daily_volatility():
    df = vix_frame(400, 12)
    model = GradientBoostingVIX()
    model.fit(df)
    forecast = model.predict()
    assert np.isfinite(forecast) and forecast > 0.0
