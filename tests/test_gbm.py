"""GBM feature alignment, the validation split and the lognormal correction."""

import numpy as np
import pandas as pd
import pytest

from models.gbm import (
    BUFFER,
    DAILY,
    EMBARGO,
    FEATURES,
    TRAIN_FRACTION,
    GradientBoosting,
    _daily,
    _features,
    _split,
)

MEAN_LOG_VARIANCE = -9.0


def frame(n, seed, scale=0.3):
    """Log variance follows an AR(1); returns are shocks with that variance."""
    rng = np.random.default_rng(seed)
    scale = np.broadcast_to(scale, n)
    log_variance = np.empty(n)
    log_variance[0] = MEAN_LOG_VARIANCE
    for day in range(1, n):
        log_variance[day] = (
            MEAN_LOG_VARIANCE
            + 0.9 * (log_variance[day - 1] - MEAN_LOG_VARIANCE)
            + scale[day] * rng.standard_normal()
        )
    vol = np.sqrt(np.exp(log_variance))
    log_ret = vol * rng.standard_normal(n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    gap = 0.3 * vol * rng.standard_normal(n)
    open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(gap)
    stretch = np.exp(vol * np.abs(rng.standard_normal(n)))
    return pd.DataFrame(
        {
            "realized_vol_gk": vol,
            "log_ret": log_ret,
            "open": open_,
            "close": close,
            "high": np.maximum(open_, close) * stretch,
            "low": np.minimum(open_, close) / stretch,
            "dividend": 0.0,
        },
        index=pd.bdate_range("2000-01-03", periods=n),
    )


def design(df):
    return GradientBoosting().design(_daily(df))


def test_features_use_no_data_after_day_t():
    full = design(frame(400, 3))
    truncated = design(frame(400, 3).iloc[:-50])
    shared = full.index.intersection(truncated.index)
    pd.testing.assert_frame_equal(full.loc[shared], truncated.loc[shared])


def test_target_is_the_next_days_log_variance():
    df = frame(400, 4)
    matrix = design(df)
    expected = np.log(df["realized_vol_gk"] ** 2).shift(-1)
    assert np.allclose(matrix["target"], expected.loc[matrix.index])


def test_two_day_horizon_targets_the_log_variance_two_days_ahead():
    df = frame(400, 4)
    matrix = GradientBoosting(horizon=2).design(_daily(df))
    expected = np.log(df["realized_vol_gk"] ** 2).shift(-2)
    assert np.allclose(matrix["target"], expected.loc[matrix.index])
    assert matrix.index[-1] == df.index[-3]


def test_validation_starts_an_embargo_after_the_train_cut():
    matrix = design(frame(400, 10))
    x_train, y_train, x_valid, y_valid = _split(matrix, FEATURES)
    cut = int(len(matrix) * TRAIN_FRACTION)
    assert len(y_train) == cut
    assert len(y_valid) == len(matrix) - cut - EMBARGO
    assert np.allclose(x_valid[0], matrix.iloc[cut + EMBARGO, :-1])


def test_two_day_horizon_validation_shares_no_raw_day_with_training():
    matrix = GradientBoosting(horizon=2).design(_daily(frame(400, 10)))
    x_train, _, x_valid, _ = _split(matrix, FEATURES, 2)
    cut = int(len(matrix) * TRAIN_FRACTION)
    last_train_target = matrix.index[cut - 1] + pd.offsets.BDay(2)
    first_valid_feature = matrix.index[cut + EMBARGO + 1] - pd.offsets.BDay(EMBARGO - 1)
    assert len(x_valid) == len(matrix) - cut - EMBARGO - 1
    assert last_train_target < first_valid_feature


def test_correction_uses_the_validation_residuals():
    n = 4000
    scale = np.where(np.arange(n) < 0.75 * n, 0.05, 0.6)
    model = GradientBoosting()
    model.fit(frame(n, 11, scale))
    assert 0.5 * 0.6**2 < model.residual_variance < 4.0 * 0.6**2


def test_predict_adds_half_the_residual_variance():
    model = GradientBoosting()
    model.fit(frame(600, 5))
    model.residual_variance = 0.0
    uncorrected = model.predict()
    model.residual_variance = 0.4
    assert model.predict() == pytest.approx(uncorrected * np.exp(0.1), rel=1e-12)


def test_final_refit_reuses_the_early_stopped_tree_count():
    model = GradientBoosting()
    model.fit(frame(600, 6))
    assert model.model.num_trees() == model.best_iteration


def test_fit_equals_replaying_updates():
    df = frame(600, 7)
    replayed = GradientBoosting()
    replayed.fit(df.iloc[:-1])
    replayed.update(df.iloc[-1])

    fresh = GradientBoosting()
    fresh.fit(df)
    fresh.model = replayed.model
    fresh.residual_variance = replayed.residual_variance
    assert replayed.predict() == pytest.approx(fresh.predict(), rel=1e-12)


def test_update_matches_the_vectorized_daily_values():
    df = frame(300, 9)
    model = GradientBoosting()
    model.fit(df.iloc[:-1])
    model.update(df.iloc[-1])
    assert model.days[-1] == pytest.approx(_daily(df).iloc[-1].to_numpy(), rel=1e-12)


def test_predict_features_are_complete():
    """A window longer than the buffer would reach LightGBM as a silent NaN."""
    df = frame(300, 12)
    model = GradientBoosting()
    model.fit(df.iloc[:-1])
    model.update(df.iloc[-1])
    row = _features(pd.DataFrame(list(model.days), columns=DAILY)).iloc[-1]
    assert row.notna().all()


def test_buffers_keep_the_longest_feature_window():
    df = frame(600, 8)
    model = GradientBoosting()
    model.fit(df)
    daily = pd.DataFrame(list(model.days))
    assert len(daily) == BUFFER
    assert np.allclose(daily[0], (df["realized_vol_gk"] ** 2).iloc[-BUFFER:])
