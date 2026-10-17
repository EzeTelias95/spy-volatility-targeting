"""HAR-RV alignment, the validation split and the lognormal correction.

The variance paths are built from the HAR relation itself, one day at a time, so
the expected values come from the generating process rather than from a second
copy of the model's own windowing.
"""

import numpy as np
import pandas as pd
import pytest

from models.har import EMBARGO, HAR, WINDOWS, _design

COEFFICIENTS = (-0.9, 0.15, 0.45, 0.30)


def har_path(shocks):
    """Path where tomorrow's log variance is the HAR relation plus a shock."""
    variance = list(np.exp(np.linspace(-9.5, -8.5, EMBARGO)))
    for day in range(EMBARGO, len(shocks)):
        history = np.asarray(variance)
        features = np.array([1.0] + [np.log(history[-w:].mean()) for w in WINDOWS])
        variance.append(np.exp(features @ np.asarray(COEFFICIENTS) + shocks[day]))
    return np.asarray(variance)


def frame(variance):
    return pd.DataFrame(
        {"realized_vol_gk": np.sqrt(variance)},
        index=pd.bdate_range("2000-01-03", periods=len(variance)),
    )


@pytest.fixture
def shocks():
    return np.random.default_rng(1).standard_normal(12000) * 0.3


def test_recovers_the_generating_har_slopes(shocks):
    """Pairing the regressors with the same day would drive the daily slope to 1."""
    model = HAR()
    model.fit(frame(har_path(shocks)))
    assert model.params.to_numpy()[1:] == pytest.approx(COEFFICIENTS[1:], abs=0.05)
    assert model.residual_variance == pytest.approx(0.3**2, rel=0.2)


def test_predict_extends_the_generating_path(shocks):
    variance = har_path(np.append(shocks, 0.0))
    model = HAR()
    model.fit(frame(variance[:-1]))
    model.params[:] = COEFFICIENTS
    model.residual_variance = 0.0
    assert model.predict() == pytest.approx(np.sqrt(variance[-1]), rel=1e-10)


def test_predict_adds_half_the_residual_variance(shocks):
    model = HAR()
    model.fit(frame(har_path(shocks)))
    model.residual_variance = 0.0
    uncorrected = model.predict()
    model.residual_variance = 0.4
    assert model.predict() == pytest.approx(uncorrected * np.exp(0.1), rel=1e-12)


def test_correction_uses_the_validation_residuals(shocks):
    scale = np.where(np.arange(len(shocks)) < 0.75 * len(shocks), 0.05, 0.4)
    path = har_path(np.random.default_rng(2).standard_normal(len(shocks)) * scale)
    model = HAR()
    model.fit(frame(path))
    assert model.residual_variance == pytest.approx(0.4**2, rel=0.15)


def test_fit_equals_replaying_updates(shocks):
    variance = har_path(shocks)
    replayed = HAR()
    replayed.fit(frame(variance[:-1]))
    replayed.update(frame(variance).iloc[-1])
    params, correction = replayed.params, replayed.residual_variance

    fresh = HAR()
    fresh.fit(frame(variance))
    fresh.params, fresh.residual_variance = params, correction
    assert replayed.predict() == pytest.approx(fresh.predict(), rel=1e-12)


def test_two_day_horizon_targets_the_log_variance_two_days_ahead(shocks):
    data = frame(har_path(shocks[:300]))
    variance = data["realized_vol_gk"] ** 2
    design = _design(variance, 2)
    expected = np.log(variance).shift(-2).loc[design.index]
    assert np.allclose(design["target"], expected)
    assert design.index[-1] == variance.index[-3]


def test_two_day_horizon_fits_a_flatter_and_noisier_relation(shocks):
    """Two steps ahead the same path is less predictable than one step ahead."""
    data = frame(har_path(shocks))
    one, two = HAR(), HAR(horizon=2)
    one.fit(data)
    two.fit(data)
    assert two.params["daily"] < one.params["daily"]
    assert two.residual_variance > one.residual_variance
