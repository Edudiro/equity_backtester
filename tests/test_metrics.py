import numpy as np
import pandas as pd
import pytest

from backtester.engine import BacktestResult
from backtester.metrics import drawdown_episodes, relative, summary


def result(returns, held=None, turnover=None, capital=1.0) -> BacktestResult:
    idx = pd.bdate_range("2024-01-01", periods=len(returns))
    r = pd.Series(returns, index=idx, dtype=float)
    held = pd.DataFrame({"X": held if held is not None else [1.0] * len(r)}, index=idx)
    turnover = pd.Series(turnover if turnover is not None else [0.0] * len(r), index=idx)
    return BacktestResult(
        target_weights=held,
        held_weights=held,
        turnover=turnover,
        costs=turnover * 0,
        returns=r,
        equity=capital * (1 + r).cumprod(),
        initial_capital=capital,
        execution="close",
        cost=0.0,
    )


def returns_for_equity(path) -> list[float]:
    """Daily returns that produce the given equity path, starting from 1."""
    eq = [1.0, *path]
    return [b / a - 1 for a, b in zip(eq, eq[1:])]


# --- summary --------------------------------------------------------------------


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


def test_exposure():
    assert summary(result([0, 0, 0, 0], held=[0, 1, 1, 0]))["exposure"] == 0.5


# --- drawdown episodes ------------------------------------------------------------


def test_drawdown_episodes_by_hand():
    #  day:     start  0     1      2     3     4
    #  equity:  1.0    1.1   0.99   1.21  1.0   1.05
    #  episode A: peak 1.1 (day 0), trough 0.99 (day 1), recovered day 2: -10%
    #  episode B: peak 1.21 (day 2), trough 1.0 (day 3), not recovered: -17.4%
    res = result(returns_for_equity([1.1, 0.99, 1.21, 1.0, 1.05]))
    ep = drawdown_episodes(res)
    d = res.equity.index

    deepest, other = ep.iloc[0], ep.iloc[1]
    assert deepest["depth"] == pytest.approx(1.0 / 1.21 - 1)
    assert (deepest["peak_date"], deepest["trough_date"]) == (d[2], d[3])
    assert pd.isna(deepest["recovery_date"])
    assert deepest["days_underwater"] == 2  # day 2 -> day 4 (end), still underwater

    assert other["depth"] == pytest.approx(-0.1)
    assert (other["peak_date"], other["trough_date"], other["recovery_date"]) == (d[0], d[1], d[2])
    assert (other["days_to_trough"], other["days_underwater"]) == (1, 2)

    assert summary(res)["longest_drawdown_days"] == 2


def test_no_drawdown_when_equity_only_rises():
    res = result([0.01] * 5)
    assert drawdown_episodes(res).empty
    assert summary(res)["longest_drawdown_days"] == 0


# --- relative to a benchmark ------------------------------------------------------


def test_relative_to_itself():
    res = result([0.01, -0.02, 0.015, 0.0, -0.005])
    rel = relative(res, res)
    assert rel["beta"] == pytest.approx(1)
    assert rel["alpha"] == pytest.approx(0)
    assert rel["correlation"] == pytest.approx(1)
    assert rel["tracking_error"] == 0
    assert np.isnan(rel["information_ratio"])


def test_leveraged_copy_has_beta_two_and_no_alpha():
    b = [0.01, -0.02, 0.015, 0.0, -0.005]
    rel = relative(result([2 * x for x in b]), result(b))
    assert rel["beta"] == pytest.approx(2)
    assert rel["alpha"] == pytest.approx(0)
    assert rel["correlation"] == pytest.approx(1)


def test_constant_extra_return_is_pure_alpha():
    b = [0.01, -0.02, 0.015, 0.0, -0.005]
    rel = relative(result([x + 0.001 for x in b]), result(b))
    assert rel["beta"] == pytest.approx(1)
    assert rel["alpha"] == pytest.approx(0.001 * 252)
