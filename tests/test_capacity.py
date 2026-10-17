"""Capacity against SPY dollar volume."""

import numpy as np
import pandas as pd
import pytest

from evaluation.capacity import average_dollar_volume, capacity_table

INDEX = pd.bdate_range("2024-01-02", periods=5)


def test_average_dollar_volume_ends_the_day_before():
    dollar_volume = pd.Series([1.0, 3.0, 5.0, 7.0, 9.0], index=INDEX)

    average = average_dollar_volume(dollar_volume, window=2)

    assert average.isna().to_list() == [True, True, False, False, False]
    assert average.iloc[2:].to_list() == pytest.approx([2.0, 4.0, 6.0])


def test_capacity_scales_volume_by_participation_over_turnover():
    turnover = pd.DataFrame({"a": [0.9, 0.5, 0.0, 0.1, 0.0]}, index=INDEX)
    volume = pd.Series([1.0, 100.0, 100.0, 200.0, 100.0], index=INDEX)

    table = capacity_table(turnover, volume, participation=0.01, reference_aum=10.0)

    assert table.loc["a", "capacity_min"] == pytest.approx(2.0)
    assert table.loc["a", "capacity_median"] == pytest.approx(11.0)
    assert table.loc["a", "median_trade_size"] == pytest.approx(0.3)
    assert table.loc["a", "trade_days_per_year"] == pytest.approx(126.0)
    assert table.loc["a", "reference_share_max"] == pytest.approx(0.05)
    assert table.loc["a", "reference_share_median"] == pytest.approx(0.0275)


def test_capacity_ignores_days_without_a_trade():
    turnover = pd.DataFrame({"a": [0.9, 0.2, 0.0, 0.0, 0.0]}, index=INDEX)
    volume = pd.Series([1.0, 100.0, 1.0, 1.0, 1.0], index=INDEX)

    table = capacity_table(turnover, volume)

    assert table.loc["a", "capacity_min"] == pytest.approx(5.0)
    assert table.loc["a", "capacity_p05"] == pytest.approx(5.0)


def test_capacity_rejects_missing_volume():
    turnover = pd.DataFrame({"a": [0.9, 0.2, 0.0, 0.1, 0.0]}, index=INDEX)
    volume = pd.Series([1.0, 100.0, 100.0, np.nan, 100.0], index=INDEX)

    with pytest.raises(ValueError, match="cover every strategy date"):
        capacity_table(turnover, volume)


def test_capacity_rejects_a_strategy_that_never_trades():
    turnover = pd.DataFrame({"a": [0.9, 0.0, 0.0, 0.0, 0.0]}, index=INDEX)
    volume = pd.Series(100.0, index=INDEX)

    with pytest.raises(ValueError, match="never trades"):
        capacity_table(turnover, volume)
