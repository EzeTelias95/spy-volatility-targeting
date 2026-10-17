"""EWMA recursion arithmetic and the fit/update refit contract."""

import numpy as np
import pandas as pd
import pytest

from models.ewma import EWMA


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"log_ret": rng.normal(0.0, 0.01, 400)},
        index=pd.bdate_range("2020-01-01", periods=400),
    )


def test_recursion_matches_hand_computed_value():
    returns = [0.01, -0.02, 0.015]
    index = pd.bdate_range("2020-01-01", periods=3)
    model = EWMA()
    model.fit(pd.DataFrame({"log_ret": returns}, index=index))
    assert model.predict() == pytest.approx(0.018411614088214352, rel=1e-12)


def test_fit_equals_replaying_updates(df):
    updated, refitted = EWMA(), EWMA()
    updated.fit(df.iloc[:300])
    for _, row in df.iloc[300:].iterrows():
        updated.update(row)
    refitted.fit(df.iloc[:300])
    refitted.fit(df)
    assert updated.predict() == refitted.predict()


def test_refit_day_is_not_ingested_twice(df):
    sequenced, refitted = EWMA(), EWMA()
    sequenced.fit(df.iloc[:300])
    sequenced.update(df.iloc[-1])
    sequenced.fit(df)
    refitted.fit(df.iloc[:300])
    refitted.fit(df)
    assert sequenced.predict() == refitted.predict()
