"""Trailing benchmark arithmetic and the fit/update refit contract."""

import numpy as np
import pandas as pd
import pytest

from models.trailing import TrailingVol


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"realized_vol_gk": rng.uniform(0.004, 0.02, 400)},
        index=pd.bdate_range("2020-01-01", periods=400),
    )


def test_predict_matches_hand_computed_value():
    vols = [0.01, 0.02, 0.005]
    index = pd.bdate_range("2020-01-01", periods=3)
    model = TrailingVol(window=3)
    model.fit(pd.DataFrame({"realized_vol_gk": vols}, index=index))
    assert model.predict() == pytest.approx(0.013228756555322952, rel=1e-12)


def test_only_the_last_window_days_count(df):
    short, long = TrailingVol(), TrailingVol()
    short.fit(df.iloc[-21:])
    long.fit(df)
    assert short.predict() == long.predict()


def test_fit_equals_replaying_updates(df):
    updated, refitted = TrailingVol(), TrailingVol()
    updated.fit(df.iloc[:300])
    for _, row in df.iloc[300:].iterrows():
        updated.update(row)
    refitted.fit(df)
    assert updated.predict() == refitted.predict()
