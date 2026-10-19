import numpy as np
import pandas as pd
import pytest

from forecast import DataValidationError, forecast_cash, validate_transactions


def history(weeks=20):
    dates = pd.date_range("2025-01-01", periods=weeks, freq="7D")
    return pd.DataFrame({"date": dates, "amount": np.tile([1000.0, -500.0], weeks // 2)})


def test_validation_sorts_and_normalizes_rows():
    result = validate_transactions(pd.DataFrame({"date": ["2025-02-01", "2025-01-01"], "amount": [2, -1]}))
    assert result.iloc[0]["amount"] == -1


def test_rejects_missing_columns():
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_transactions(pd.DataFrame({"date": ["2025-01-01"]}))


def test_rejects_nonfinite_amount():
    with pytest.raises(DataValidationError, match="finite number"):
        validate_transactions(pd.DataFrame({"date": ["2025-01-01"], "amount": [np.inf]}))


def test_forecast_is_reproducible_and_quantiles_ordered():
    args = (history(), 10_000.0)
    first = forecast_cash(*args, weeks=8, simulations=500, seed=9)
    second = forecast_cash(*args, weeks=8, simulations=500, seed=9)
    pd.testing.assert_frame_equal(first.weekly_cash, second.weekly_cash)
    assert (first.weekly_cash.p10 <= first.weekly_cash.p50).all()
    assert (first.weekly_cash.p50 <= first.weekly_cash.p90).all()


def test_breach_probability_is_bounded():
    result = forecast_cash(history(), 10_000.0, weeks=4, threshold=0, simulations=500)
    assert 0 <= result.probability_below_threshold <= 1


def test_already_below_threshold_is_reported_as_breach():
    result = forecast_cash(history(), 500.0, weeks=4, threshold=1_000.0, simulations=500)
    assert result.probability_below_threshold == 1.0
