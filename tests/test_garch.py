"""GARCH(1,1) state handling: the hand recursion, the scale and the refit."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from models.garch import GARCH, SCALE


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    returns = rng.standard_normal(800) * 0.01
    return pd.DataFrame(
        {"log_ret": returns}, index=pd.bdate_range("2020-01-01", periods=800)
    )


def test_update_follows_the_variance_recursion(df):
    model = GARCH()
    model.fit(df.iloc[:-1])
    previous = model.variance
    model.update(df.iloc[-1])
    residual = df["log_ret"].iloc[-1] * SCALE - model.params["mu"]
    expected = (
        model.params["omega"]
        + model.params["alpha[1]"] * residual**2
        + model.params["beta[1]"] * previous
    )
    assert model.variance == pytest.approx(expected, rel=1e-12)


def test_fit_leaves_the_next_day_forecast_rescaled_to_daily_vol(df):
    model = GARCH()
    model.fit(df)
    result = arch_model(
        df["log_ret"] * SCALE, mean="Constant", vol="GARCH", p=1, q=1, dist="t"
    ).fit(disp="off")
    residual = df["log_ret"].iloc[-1] * SCALE - result.params["mu"]
    expected = (
        result.params["omega"]
        + result.params["alpha[1]"] * residual**2
        + result.params["beta[1]"] * result.conditional_volatility.iloc[-1] ** 2
    )
    assert model.variance == pytest.approx(expected, rel=1e-10)
    assert model.predict() == pytest.approx(np.sqrt(expected) / SCALE, rel=1e-10)


def test_refit_does_not_ingest_the_refit_day_twice(df):
    refit = GARCH()
    refit.fit(df.iloc[:-1])
    refit.update(df.iloc[-1])
    refit.fit(df)

    fresh = GARCH()
    fresh.fit(df)
    assert refit.variance == pytest.approx(fresh.variance, rel=1e-12)


def test_two_day_horizon_iterates_the_recursion_once(df):
    model = GARCH(horizon=2)
    model.fit(df)
    expected = (
        model.params["omega"]
        + (model.params["alpha[1]"] + model.params["beta[1]"]) * model.variance
    )
    assert model.predict() == pytest.approx(np.sqrt(expected) / SCALE, rel=1e-12)
