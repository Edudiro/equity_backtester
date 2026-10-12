import numpy as np
import pandas as pd
import pytest

from backtester.engine import BacktestResult
from backtester.metrics import summary


def result(returns, held=None, turnover=None, capital=1.0) -> BacktestResult:
    idx = pd.bdate_range("2024-01-01", periods=len(returns))
    r = pd.Series(returns, index=idx, dtype=float)
    held = pd.DataFrame({"X": held if held is not None else [1.0] * len(r)}, index=idx)
    turnover = pd.Series(turnover if turnover is not None else [0.0] * len(r), index=idx)
    return BacktestResult(held, held, turnover, turnover * 0, r, capital * (1 + r).cumprod(), capital)


def test_total_return_and_drawdown_by_hand():
    # equity 1 -> 1.1 -> 0.55 -> 0.66; peak 1.1, trough 0.55 -> -50%
    m = summary(result([0.1, -0.5, 0.2]))
    assert m["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert m["max_drawdown"] == pytest.approx(-0.5)


def test_loss_on_first_day_counts_as_drawdown():
    assert summary(result([-0.1, 0.05]))["max_drawdown"] == pytest.approx(-0.1)


def test_cagr_equals_total_return_over_exactly_one_year():
    m = summary(result([0.001] * 252))
    assert m["cagr"] == pytest.approx(m["total_return"])


def test_cagr_over_two_years_is_the_annual_rate():
    daily = 1.1 ** (1 / 252) - 1  # grows 10% per year
    assert summary(result([daily] * 504))["cagr"] == pytest.approx(0.10)


def test_volatility_and_sharpe():
    r = [0.01, -0.01] * 50
    m = summary(result(r))
    std = np.std(r, ddof=1)
    assert m["volatility"] == pytest.approx(std * np.sqrt(252))
    assert m["sharpe"] == pytest.approx(0.0)


def test_exposure_and_trades():
    m = summary(result([0, 0, 0, 0], held=[0, 1, 1, 0], turnover=[0, 1, 0, 1]))
    assert m["exposure"] == 0.5
    assert m["n_trades"] == 2
