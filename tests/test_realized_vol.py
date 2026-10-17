"""GK + overnight estimator against hand-computed values.

Expected variances were computed by hand from the formulas:
  gk = 0.5*ln(H/L)^2 - (2*ln2 - 1)*ln(C/O)^2
  overnight = ln((O_t + div_t) / C_{t-1})^2
  parkinson = ln(H/L)^2 / (4*ln2)
"""

import math

import pandas as pd
import pytest

from data.realized_vol import realized_vol_gk


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "open": [99.0, 101.0, 101.2, 101.6, 103.0],
            "high": [100.5, 103.0, 102.0, 101.6, 103.0],
            "low": [98.5, 100.5, 100.8, 101.4, 103.0],
            "close": [100.0, 102.5, 101.5, 103.0, 103.0],
            "dividend": [0.0, 0.0, 1.5, 0.0, 0.0],
        },
        index=pd.bdate_range("2024-01-01", periods=5),
    )


def test_normal_day(df):
    assert realized_vol_gk(df).iloc[1] == pytest.approx(0.017802451403068464, rel=1e-12)


def test_ex_dividend_day_adjusts_overnight_gap(df):
    vol = realized_vol_gk(df)
    assert vol.iloc[2] == pytest.approx(0.008392996132426072, rel=1e-12)
    assert vol.iloc[2] ** 2 < 0.00022956330318572214


def test_non_positive_gk_uses_parkinson_keeping_overnight(df):
    hl2 = math.log(101.6 / 101.4) ** 2
    assert 0.5 * hl2 - (2 * math.log(2) - 1) * math.log(103.0 / 101.6) ** 2 < 0
    assert realized_vol_gk(df).iloc[3] == pytest.approx(
        0.0015395051451336575, rel=1e-12
    )


def test_flat_day_floored_at_epsilon(df):
    assert realized_vol_gk(df).iloc[4] == pytest.approx(1e-4, rel=1e-12)
