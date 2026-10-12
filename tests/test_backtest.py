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
    assert bt.metrics["n_trades"] == 1


def test_costs_reduce_return(provider):
    free = run_backtest(spec(), provider).metrics["total_return"]
    costly = run_backtest(spec(fee_bps=10, slippage_bps=10), provider).metrics["total_return"]
    assert costly == pytest.approx((1 + free) * (1 - 0.002) - 1)


def test_no_trading_days_in_range_raises(provider):
    with pytest.raises(ValueError, match="no trading days"):
        run_backtest(spec(start="2024-06-08", end="2024-06-09"), provider)  # a weekend
