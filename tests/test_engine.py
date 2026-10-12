"""Engine tests on a 4-day toy market, every number computed by hand."""

import pandas as pd
import pytest

from backtester import engine

IDX = pd.bdate_range("2024-01-01", periods=4)


def market(close, open_=None, adj_factor=1.0) -> dict[str, pd.DataFrame]:
    close = pd.DataFrame({"X": close}, index=IDX, dtype=float)
    open_ = close if open_ is None else pd.DataFrame({"X": open_}, index=IDX, dtype=float)
    return {"open": open_, "close": close, "adj_close": close * adj_factor}


def w(values) -> pd.DataFrame:
    return pd.DataFrame({"X": values}, index=IDX, dtype=float)


CLOSE = [100, 110, 99, 99]
OPEN = [100, 105, 100, 99]
SIGNAL = [0, 1, 1, 0]  # decided at each day's close
COST = 0.01


def test_close_execution_by_hand():
    #            day0   day1          day2           day3
    # held h      0      1             1              0       (= signal, traded at the close)
    # h[t-1]      0      0             1              1
    # r           -     +10%          -10%            0%
    # turnover    0      1             0              1
    # growth      1     1*(1-.01)     (1-.1)*1       (1+0)*(1-.01)
    res = engine.run(w(SIGNAL), market(CLOSE), cost=COST, execution="close")
    assert res.returns.tolist() == pytest.approx([0, -0.01, -0.1, -0.01])
    assert res.equity.iloc[-1] == pytest.approx(0.99 * 0.9 * 0.99)


def test_signal_day_return_is_not_earned():
    """The +10% on day 1 produced the signal; it must not be captured by it."""
    res = engine.run(w(SIGNAL), market(CLOSE), execution="close")
    assert res.returns.iloc[1] == 0


def test_next_open_execution_by_hand():
    # held h = signal shifted one day:   0, 0, 1, 1   (bought at day 2's open = 100)
    # day 2: overnight with old h=0 -> 1; pay 1% on turnover 1; intraday 99/100 - 1 = -1%
    # day 3: overnight 99/99, intraday 99/99 -> 0%; no trade (sell would be at day 4's open)
    res = engine.run(w(SIGNAL), market(CLOSE, OPEN), cost=COST, execution="next_open")
    assert res.held_weights["X"].tolist() == [0, 0, 1, 1]
    assert res.returns.tolist() == pytest.approx([0, 0, 0.99 * 0.99 - 1, 0])


def test_next_open_overnight_return_goes_to_the_old_position():
    # Hold from day 0 (h=1 on days 1..3). Day 1: overnight 105/100, intraday 110/105.
    res = engine.run(w([1, 1, 1, 1]), market(CLOSE, OPEN), execution="next_open")
    assert res.returns.iloc[2] == pytest.approx(99 / 110 - 1)  # overnight 100/110 * intraday 99/100
    assert res.equity.iloc[-1] == pytest.approx(99 / 105)  # bought at day 1's open


def test_buy_and_hold_at_close_earns_price_ratio():
    res = engine.run(w([1, 1, 1, 1]), market(CLOSE), execution="close", initial_capital=1000)
    assert res.equity.iloc[-1] == pytest.approx(1000 * 99 / 100)


def test_adjusted_open_uses_dividend_factor():
    # adj_close = close * 0.5 -> adjusted open must be on the same scale.
    m = market(CLOSE, OPEN, adj_factor=0.5)
    assert engine.adjusted(m, "open")["X"].tolist() == [50, 52.5, 50, 49.5]


def test_missing_price_while_invested_raises():
    m = market(CLOSE)
    m["adj_close"].iloc[2] = float("nan")
    with pytest.raises(ValueError, match="price is missing"):
        engine.run(w([1, 1, 1, 1]), m, execution="close")


def test_unknown_execution_rejected():
    with pytest.raises(ValueError):
        engine.run(w(SIGNAL), market(CLOSE), execution="whenever")
