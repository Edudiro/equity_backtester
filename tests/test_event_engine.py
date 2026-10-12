"""Event engine: agreement with the vectorized engine, then stops checked by hand."""

import numpy as np
import pandas as pd
import pytest

from backtester import engine, event_engine
from backtester.risk import RiskRules
from backtester.trades import extract_trades


def market(bars) -> dict[str, pd.DataFrame]:
    """bars: list of (open, high, low, close) for a single ticker X."""
    idx = pd.bdate_range("2024-01-01", periods=len(bars))
    o, h, l, c = (pd.DataFrame({"X": col}, index=idx, dtype=float) for col in zip(*bars))
    return {"open": o, "high": h, "low": l, "close": c, "adj_close": c}


def weights(values, m) -> pd.DataFrame:
    return pd.DataFrame({"X": values}, index=m["close"].index, dtype=float)


def run(bars, signal, execution="next_open", **risk):
    m = market(bars)
    return event_engine.run(weights(signal, m), m, risk=RiskRules(**risk), execution=execution)


# --- 1. Without stops it must agree with the vectorized engine ------------------------


def random_market(n=400, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.008, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.01, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.01, n)))
    m = market(list(zip(open_, high, low, close)))
    signal = (rng.random(n) > 0.5).astype(float)
    return m, weights(signal, m)


@pytest.mark.parametrize("execution", ["next_open", "close"])
def test_matches_vectorized_engine_without_stops(execution):
    m, w = random_market()
    vec = engine.run(w, m, cost=0.002, execution=execution, initial_capital=1000)
    evt = event_engine.run(w, m, cost=0.002, execution=execution, initial_capital=1000)

    np.testing.assert_allclose(evt.equity, vec.equity, rtol=1e-10)
    np.testing.assert_allclose(evt.turnover, vec.turnover, atol=1e-10)
    np.testing.assert_allclose(evt.held_weights, vec.held_weights, atol=1e-10)

    t_evt, t_vec = evt.trade_log, extract_trades(vec, m)
    cols = ["entry_date", "exit_date", "days_held", "is_open", "exit_reason"]
    pd.testing.assert_frame_equal(t_evt[cols], t_vec[cols])
    np.testing.assert_allclose(t_evt["net_return"], t_vec["net_return"], rtol=1e-10)


# --- 2. Stops, by hand (execution = next_open: decide at close, buy at next open) -------
# Signal is 1 from day 0, so we buy at the OPEN of day 1.

LONG = [1, 1, 1, 1, 1]


def test_stop_loss_fills_at_stop_level():
    #        open  high  low  close
    bars = [(100, 100, 100, 100),
            (100, 102,  98, 101),   # buy at 100 -> stop at 95
            ( 99, 100,  94,  96),   # low 94 touches 95 -> sold at 95
            ( 96,  97,  95,  97),
            ( 97,  98,  96,  98)]  # fmt: skip
    res = run(bars, LONG, stop_loss=0.05)
    trade = res.trade_log.iloc[0]
    assert (trade["exit_price"], trade["exit_reason"]) == (95, "stop_loss")
    assert trade["exit_date"] == res.equity.index[2]
    assert res.equity.iloc[-1] == pytest.approx(0.95)  # and we stay out afterwards
    assert len(res.trade_log) == 1


def test_gap_through_stop_fills_at_the_open():
    bars = [(100, 100, 100, 100),
            (100, 102,  98, 101),
            ( 90,  92,  88,  91),   # opens at 90, below the 95 stop: filled at 90, not 95
            ( 91,  92,  90,  92),
            ( 92,  93,  91,  93)]  # fmt: skip
    res = run(bars, LONG, stop_loss=0.05)
    assert res.trade_log.iloc[0]["exit_price"] == 90
    assert res.equity.iloc[-1] == pytest.approx(0.90)


def test_take_profit_intraday_and_on_a_gap():
    bars = [(100, 100, 100, 100),
            (100, 105,  99, 104),   # buy at 100 -> take profit at 110
            (104, 112, 103, 108),   # high 112 -> sold at 110
            (108, 109, 107, 108),
            (108, 109, 107, 108)]  # fmt: skip
    res = run(bars, LONG, take_profit=0.10)
    trade = res.trade_log.iloc[0]
    assert trade["exit_price"] == pytest.approx(110)  # 100 * 1.1 isn't exactly 110 in floating point
    assert trade["exit_reason"] == "take_profit"

    gap = [b if i != 2 else (115, 116, 114, 115) for i, b in enumerate(bars)]  # opens above 110
    assert run(gap, LONG, take_profit=0.10).trade_log.iloc[0]["exit_price"] == 115


def test_stop_assumed_first_when_both_levels_touched_same_day():
    bars = [(100, 100, 100, 100),
            (100, 101,  99, 100),
            (100, 112,  94, 105),   # range covers both 95 and 110: unknowable order -> stop
            (105, 106, 104, 105),
            (105, 106, 104, 105)]  # fmt: skip
    trade = run(bars, LONG, stop_loss=0.05, take_profit=0.10).trade_log.iloc[0]
    assert (trade["exit_price"], trade["exit_reason"]) == (95, "stop_loss")


def test_trailing_stop_follows_highest_close():
    bars = [(100, 100, 100, 100),
            (100, 111,  99, 110),   # buy 100; close 110 -> trail level 99
            (110, 121, 109, 120),   # close 120 -> trail level 108
            (119, 120, 107, 112),   # low 107 <= 108 -> sold at 108
            (112, 113, 111, 112)]  # fmt: skip
    trade = run(bars, LONG, trailing_stop=0.10).trade_log.iloc[0]
    assert (trade["exit_price"], trade["exit_reason"]) == (108, "trailing_stop")


def test_tighter_of_stop_loss_and_trailing_binds():
    # After a rise to 120, the trailing level 108 is above the fixed stop at 95.
    bars = [(100, 100, 100, 100),
            (100, 121,  99, 120),
            (119, 120, 107, 110),   # low 107: trailing (108) fires, fixed stop (95) wouldn't
            (110, 111, 109, 110),
            (110, 111, 109, 110)]  # fmt: skip
    trade = run(bars, LONG, stop_loss=0.05, trailing_stop=0.10).trade_log.iloc[0]
    assert (trade["exit_price"], trade["exit_reason"]) == (108, "trailing_stop")


# --- 3. Re-entry after a stop -----------------------------------------------------------

FLAT_BARS = [(100, 100, 100, 100),
             (100, 101,  99, 100),
             (100, 101,  94, 96),    # stopped at 95
             (96,  97,  95, 96),
             (96,  97,  95, 96),
             (96,  97,  95, 96)]  # fmt: skip


def test_no_reentry_while_strategy_still_long():
    res = run(FLAT_BARS, [1, 1, 1, 1, 1, 1], stop_loss=0.05)
    assert len(res.trade_log) == 1
    assert (res.held_weights["X"].iloc[2:] == 0).all()


def test_reentry_after_strategy_goes_flat_and_signals_again():
    # target 0 on day 3 re-arms; target 1 on day 4 -> buy at day 5's open.
    res = run(FLAT_BARS, [1, 1, 1, 0, 1, 1], stop_loss=0.05)
    assert len(res.trade_log) == 2
    assert res.trade_log.iloc[1]["entry_date"] == res.equity.index[5]


# --- 4. Close execution -----------------------------------------------------------------


def test_close_execution_enters_at_close_and_stops_from_next_day():
    bars = [(100, 100, 100, 100),   # buy at this close (100) -> stop 95
            ( 99, 100,  94,  96),   # sold at 95
            ( 96,  97,  95,  97)]  # fmt: skip
    res = run(bars, [1, 1, 1], execution="close", stop_loss=0.05)
    trade = res.trade_log.iloc[0]
    assert (trade["entry_price"], trade["exit_price"]) == (100, 95)
    assert trade["entry_date"] == res.equity.index[0]


# --- 5. Validation ------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [dict(stop_loss=0), dict(stop_loss=1.5), dict(trailing_stop=-0.1), dict(take_profit=0)])
def test_invalid_risk_rules(bad):
    with pytest.raises(ValueError):
        RiskRules(**bad)
