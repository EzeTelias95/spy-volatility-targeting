"""Look-ahead direction, ceiling arithmetic and the sign randomized counterfactual."""

import numpy as np
import pandas as pd
import pytest

from evaluation.oracle import daily_oracle, regime_oracle, sign_randomized_market


@pytest.fixture
def vol():
    values = [0.01, 0.02, 0.005, 0.03, 0.015, 0.008]
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=6))


def test_daily_oracle_is_the_realized_series(vol):
    assert daily_oracle(vol).equals(vol.rename("forecast"))


def test_regime_oracle_matches_a_hand_computed_forward_mean(vol):
    result = regime_oracle(vol, window=3)
    expected = np.sqrt(np.mean([0.01**2, 0.02**2, 0.005**2]))
    assert result.iloc[0] == pytest.approx(expected, rel=1e-12)
    assert result.index[0] == vol.index[0]


def test_regime_oracle_looks_forward_not_backward(vol):
    result = regime_oracle(vol, window=3)
    assert len(result) == len(vol) - 2
    assert result.index[-1] == vol.index[-3]
    assert result.iloc[-1] == pytest.approx(
        np.sqrt(np.mean([0.03**2, 0.015**2, 0.008**2])), rel=1e-12
    )


def test_regime_oracle_ignores_days_outside_the_window():
    index = pd.bdate_range("2020-01-01", periods=4)
    base = pd.Series([0.01, 0.01, 0.01, 0.01], index=index)
    moved = base.copy()
    moved.iloc[3] = 0.5
    assert regime_oracle(base, window=3).iloc[0] == pytest.approx(
        regime_oracle(moved, window=3).iloc[0], rel=1e-12
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.01, 0.0, 0.01], "finite and positive"),
        ([0.01, np.nan, 0.01], "finite and positive"),
        ([0.01, -0.01, 0.01], "finite and positive"),
    ],
)
def test_oracles_reject_invalid_volatility(values, message):
    series = pd.Series(values, index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(ValueError, match=message):
        daily_oracle(series)
    with pytest.raises(ValueError, match=message):
        regime_oracle(series, window=2)


def test_regime_oracle_rejects_a_short_sample(vol):
    with pytest.raises(ValueError, match="at least window long"):
        regime_oracle(vol, window=10)


def test_oracles_reject_an_unsorted_index(vol):
    shuffled = vol.iloc[[1, 0, 2, 3, 4, 5]]
    with pytest.raises(ValueError, match="must be sorted"):
        daily_oracle(shuffled)


@pytest.fixture
def market():
    index = pd.bdate_range("2020-01-01", periods=6)
    close = np.array([100.0, 102.0, 99.0, 103.0, 101.0, 104.0])
    frame = pd.DataFrame(
        {
            "open": close * 0.995,
            "high": close * 1.012,
            "low": close * 0.988,
            "close": close,
            "adj_close": close * 0.9,
        },
        index=index,
    )
    frame["ret"] = frame["adj_close"] / frame["adj_close"].shift(1) - 1.0
    frame["log_ret"] = np.log1p(frame["ret"])
    return frame


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_sign_randomization_preserves_intraday_ratios(market, rng):
    result = sign_randomized_market(market, market.index[1:], rng)
    for column in ("open", "high", "low", "adj_close"):
        original = market[column] / market["close"]
        assert np.allclose(result[column] / result["close"], original, rtol=1e-12)


def test_sign_randomization_preserves_the_size_of_every_move(market, rng):
    dates = market.index[1:]
    result = sign_randomized_market(market, dates, rng)
    mean = market.loc[dates, "ret"].mean()
    assert np.allclose(
        np.sort(np.abs(result.loc[dates, "ret"] - mean)),
        np.sort(np.abs(market.loc[dates, "ret"] - mean)),
        rtol=1e-12,
    )


def test_sign_randomization_flips_at_least_one_sign(market, rng):
    result = sign_randomized_market(market, market.index[1:], rng)
    assert not np.allclose(result["ret"].iloc[1:], market["ret"].iloc[1:])


def test_sign_randomization_reconciles_returns_with_prices(market, rng):
    result = sign_randomized_market(market, market.index[1:], rng)
    ratio = result["adj_close"] / result["adj_close"].shift(1) - 1.0
    assert np.allclose(result["ret"].iloc[1:], ratio.iloc[1:], rtol=1e-12)
    assert np.allclose(result["log_ret"].iloc[1:], np.log1p(result["ret"].iloc[1:]))


def test_sign_randomization_keeps_ohlc_sanity(market, rng):
    result = sign_randomized_market(market, market.index[1:], rng)
    assert (result["high"] >= result[["open", "close"]].max(axis=1)).all()
    assert (result["low"] <= result[["open", "close"]].min(axis=1)).all()


def test_sign_randomization_leaves_days_outside_the_window_alone(market, rng):
    dates = market.index[1:4]
    result = sign_randomized_market(market, dates, rng)
    untouched = market.index[4:]
    assert np.allclose(result.loc[untouched, "ret"], market.loc[untouched, "ret"])


def test_sign_randomization_rejects_dates_outside_the_index(market, rng):
    outside = pd.DatetimeIndex(["2019-12-01"])
    with pytest.raises(ValueError, match="contained in the market_data index"):
        sign_randomized_market(market, outside, rng)


def test_sign_randomization_rejects_a_missing_column(market, rng):
    with pytest.raises(ValueError, match="missing columns"):
        sign_randomized_market(market.drop(columns="adj_close"), market.index[1:], rng)


def test_sign_randomization_rejects_a_missing_return(market, rng):
    broken = market.copy()
    broken.loc[broken.index[2], "ret"] = np.nan
    with pytest.raises(ValueError, match="ret must be finite"):
        sign_randomized_market(broken, broken.index[1:], rng)
