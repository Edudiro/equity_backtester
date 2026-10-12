"""End-to-end: spec -> data -> rules -> weights -> engine -> metrics, on fake data."""

from datetime import date

import pandas as pd
import pytest

from backtester.backtest import run_backtest
from backtester.spec import StrategySpec
from tests.fakes import FakeProvider, make_history


def spec(**overrides) -> StrategySpec:
    base = {
        "tickers": ["X"],
        "start": "2024-06-03",
        "end": "2024-06-28",
        "strategy": {
            "type": "rule_based",
            # Prices rise every day, so price > SMA(200) is always true once SMA(200) exists.
            "entry": {"left": {"indicator": "price"}, "op": ">", "right": {"indicator": "sma", "window": 200}},
            "exit": {"left": {"indicator": "price"}, "op": "<", "right": {"indicator": "sma", "window": 200}},
        },
    }
    return StrategySpec.from_dict({**base, **overrides})


@pytest.fixture
def provider():
    return FakeProvider(make_history("2023-01-02", "2024-12-31"))


def test_warm_up_comes_from_history_before_start(provider):
    """SMA(200) needs 200 days; the backtest window has 20. It must still be valid on day one."""
    bt = run_backtest(spec(), provider)
    assert bt.result.target_weights.index[0] == pd.Timestamp("2024-06-03")
    assert bt.result.target_weights["X"].iloc[0] == 1.0


def test_requests_full_history_up_to_end(provider):
    run_backtest(spec(), provider)
    (start, end), = provider.requests
    assert start < date(1901, 1, 1) and end == date(2024, 6, 28)


def test_end_to_end_return_matches_hand_calculation(provider):
    """Decide at the close of 06-03, buy at the open of 06-04, hold to the close of 06-28.

    open = close in the fake data, so the return is P(06-28) / P(06-04) - 1.
    """
    bt = run_backtest(spec(initial_capital=1000), provider)
    h = provider.history["adj_close"]
    expected = h.loc["2024-06-28"] / h.loc["2024-06-04"] - 1
    assert bt.metrics["total_return"] == pytest.approx(expected)
    assert bt.result.equity.iloc[-1] == pytest.approx(1000 * (1 + expected))
    assert len(bt.trades) == 1 and bt.trades["is_open"].iloc[0]  # bought once, never sold


def test_costs_reduce_return(provider):
    free = run_backtest(spec(), provider).metrics["total_return"]
    costly = run_backtest(spec(fee_bps=10, slippage_bps=10), provider).metrics["total_return"]
    assert costly == pytest.approx((1 + free) * (1 - 0.002) - 1)


def test_no_trading_days_in_range_raises(provider):
    with pytest.raises(ValueError, match="no trading days"):
        run_backtest(spec(start="2024-06-08", end="2024-06-09"), provider)  # a weekend


def test_benchmark_is_buy_and_hold_through_the_same_engine(provider):
    bt = run_backtest(spec(fee_bps=10), provider)
    # Our strategy is long the whole time, so it IS buy-and-hold: identical results.
    pd.testing.assert_series_equal(bt.result.equity, bt.benchmark.equity)
    assert bt.relative["beta"] == pytest.approx(1)


def test_benchmark_can_be_another_ticker():
    hist = make_history("2023-01-02", "2024-12-31")
    provider = FakeProvider({"X": hist, "SPY": hist * 2})
    bt = run_backtest(spec(benchmark="spy"), provider)
    assert list(bt.benchmark.held_weights.columns) == ["SPY"]
    assert list(bt.result.held_weights.columns) == ["X"]  # the strategy never trades SPY


def test_forcing_either_engine_gives_the_same_result_without_stops(provider):
    vec = run_backtest(spec(engine="vectorized", fee_bps=5), provider)
    evt = run_backtest(spec(engine="event", fee_bps=5), provider)
    pd.testing.assert_series_equal(vec.result.equity, evt.result.equity, check_exact=False, rtol=1e-12)


def test_take_profit_runs_through_the_event_engine(provider):
    # Prices rise 1 point a day from ~460 (~0.2%/day), so a 2% take-profit fires in ~10 days.
    bt = run_backtest(spec(risk={"take_profit": 0.02}), provider)
    first = bt.trades.iloc[0]
    assert first["exit_reason"] == "take_profit"
    # Fake bars have open == close, so the price gaps from 480 to 481 overnight,
    # over the 480.42 level: filled at the open (481), as a gap should be.
    assert first["exit_price"] == 481 > first["entry_price"] * 1.02
    assert bt.result.trade_log is not None
