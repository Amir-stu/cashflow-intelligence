"""Transparent weekly cash-flow bootstrap forecast."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"date", "amount"}


class DataValidationError(ValueError):
    """Raised when a transaction file cannot be interpreted safely."""


@dataclass(frozen=True)
class ForecastResult:
    weekly_cash: pd.DataFrame
    probability_below_threshold: float
    threshold: float
    median_minimum_cash: float
    simulations: int


def validate_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized transaction table or raise a useful validation error."""
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Missing required columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise DataValidationError("The transaction file has no rows.")

    result = frame.loc[:, ["date", "amount"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce", utc=True).dt.tz_convert(None)
    result["amount"] = pd.to_numeric(result["amount"], errors="coerce")
    if result["date"].isna().any():
        raise DataValidationError("Every date must be a valid calendar date.")
    if result["amount"].isna().any() or not np.isfinite(result["amount"].to_numpy()).all():
        raise DataValidationError("Every amount must be a finite number.")
    if (result["date"] > pd.Timestamp.now().normalize()).any():
        raise DataValidationError("Transactions dated in the future are not allowed in historical data.")
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def forecast_cash(
    transactions: pd.DataFrame,
    starting_cash: float,
    weeks: int = 13,
    threshold: float = 0.0,
    simulations: int = 5_000,
    block_size: int = 4,
    seed: int = 42,
) -> ForecastResult:
    """Simulate future cash using a circular moving-block bootstrap of weekly net flows."""
    data = validate_transactions(transactions)
    if not np.isfinite(starting_cash) or starting_cash < 0:
        raise DataValidationError("Starting cash must be a finite, non-negative amount.")
    if not np.isfinite(threshold):
        raise DataValidationError("Cash threshold must be finite.")
    if not 1 <= weeks <= 52:
        raise DataValidationError("Forecast horizon must be between 1 and 52 weeks.")
    if not 500 <= simulations <= 100_000:
        raise DataValidationError("Simulations must be between 500 and 100,000.")

    # Weeks close on Sunday. Empty weeks are explicit zero movements.
    weekly = data.set_index("date")["amount"].resample("W-SUN").sum().asfreq("W-SUN", fill_value=0.0)
    if len(weekly) < 12:
        raise DataValidationError("At least 12 weeks of transaction history are required.")
    if block_size < 1 or block_size > min(12, len(weekly)):
        raise DataValidationError("Block size must be between 1 and 12 weeks and fit the history.")

    history = weekly.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    future_changes = np.empty((simulations, weeks), dtype=float)
    for path in range(simulations):
        offset = 0
        while offset < weeks:
            start = int(rng.integers(0, len(history)))
            length = min(block_size, weeks - offset)
            indices = (start + np.arange(length)) % len(history)
            future_changes[path, offset : offset + length] = history[indices]
            offset += length

    paths = starting_cash + np.cumsum(future_changes, axis=1)
    last_week = weekly.index[-1]
    dates = pd.date_range(last_week + pd.Timedelta(days=7), periods=weeks, freq="W-SUN")
    summary = pd.DataFrame(
        {
            "date": dates,
            "p10": np.quantile(paths, 0.10, axis=0),
            "p50": np.quantile(paths, 0.50, axis=0),
            "p90": np.quantile(paths, 0.90, axis=0),
        }
    )
    return ForecastResult(
        weekly_cash=summary,
        probability_below_threshold=float(starting_cash < threshold or np.mean(np.min(paths, axis=1) < threshold)),
        threshold=float(threshold),
        median_minimum_cash=float(np.median(np.min(paths, axis=1))),
        simulations=simulations,
    )
