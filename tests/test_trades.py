import numpy as np
import pandas as pd
import pytest

from backtester import engine
from backtester.trades import extract_trades, trade_stats

IDX = pd.bdate_range("2024-01-01", periods=6)
CLOSE = [100, 110, 99, 105, 120, 115]
OPEN = [100, 104, 108, 100, 110, 118]


def market(close=CLOSE, open_=OPEN) -> dict[str, pd.DataFrame]:
    c = pd.DataFrame({"X": close}, index=IDX, dtype=float)
    return {"open": pd.DataFrame({"X": open_}, index=IDX, dtype=float), "close": c, "adj_close": c}


def run(signal, execution, cost=0.0, m=None):
    m = m or market()
    w = pd.DataFrame({"X": signal}, index=IDX, dtype=float)
    res = engine.run(w, m, cost=cost, execution=execution)
    return res, extract_trades(res, m)


def test_close_execution_trades_by_hand():
    # signal 0 1 1 0 1 1 -> held = signal (trade at that close)
    # trade 1: buy close day1 (110), sell close day3 (105)
    # trade 2: buy close day4 (120), still open at day5 close (115)
    _, t = run([0, 1, 1, 0, 1, 1], "close")
    assert t[["entry_price", "exit_price", "is_open"]].values.tolist() == [[110, 105, False], [120, 115, True]]
    assert t["days_held"].tolist() == [2, 1]
    assert t["gross_return"].tolist() == pytest.approx([105 / 110 - 1, 115 / 120 - 1])


def test_next_open_execution_trades_by_hand():
    # held = signal shifted: 0 0 1 1 0 1
    # trade 1: buy open day2 (108), sell open day4 (110)
    # trade 2: buy open day5 (118), open at the end -> marked at close day5 (115)
    _, t = run([0, 1, 1, 0, 1, 1], "next_open")
    assert t[["entry_price", "exit_price", "is_open"]].values.tolist() == [[108, 110, False], [118, 115, True]]
    assert t["entry_date"].tolist() == [IDX[2], IDX[5]]


def test_costs_on_both_sides_for_closed_and_entry_only_for_open():
    _, t = run([0, 1, 1, 0, 1, 1], "close", cost=0.01)
    assert t["net_return"].tolist() == pytest.approx([105 / 110 * 0.99**2 - 1, 115 / 120 * 0.99 - 1])


@pytest.mark.parametrize("execution", ["close", "next_open"])
def test_trades_compound_to_engine_equity(execution):
    """Single stock at 0/100%: the product of trade returns IS the equity curve."""
    rng = np.random.default_rng(1)
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    open_ = close * np.exp(rng.normal(0, 0.005, n))
    m = {k: pd.DataFrame({"X": v}, index=idx) for k, v in [("open", open_), ("close", close), ("adj_close", close)]}
    w = pd.DataFrame({"X": (rng.random(n) > 0.5).astype(float)}, index=idx)

    res = engine.run(w, m, cost=0.002, execution=execution)
    t = extract_trades(res, m)

    assert len(t) > 20
    assert (1 + t["net_return"]).prod() == pytest.approx(res.equity.iloc[-1] / res.initial_capital)


def test_trade_stats_by_hand():
    trades = pd.DataFrame(
        {"net_return": [0.1, -0.05, 0.2, -0.05, -0.1, 0.3], "days_held": [5, 3, 10, 2, 4, 1],
         "is_open": [False] * 5 + [True]}  # the last one is open and excluded
    )  # fmt: skip
    s = trade_stats(trades)
    assert s["n_trades"] == 5
    assert s["win_rate"] == pytest.approx(0.4)
    assert s["avg_win"] == pytest.approx(0.15)
    assert s["avg_loss"] == pytest.approx(-0.2 / 3)
    assert s["payoff_ratio"] == pytest.approx(0.15 / (0.2 / 3))
    assert s["profit_factor"] == pytest.approx(0.3 / 0.2)
    assert s["expectancy"] == pytest.approx(0.1 / 5)
    assert (s["best_trade"], s["worst_trade"]) == (0.2, -0.1)
    assert s["avg_days_held"] == pytest.approx(24 / 5)
    assert (s["max_consecutive_wins"], s["max_consecutive_losses"]) == (1, 2)


def test_trade_stats_with_no_closed_trades():
    s = trade_stats(pd.DataFrame({"net_return": [0.1], "days_held": [3], "is_open": [True]}))
    assert s["n_trades"] == 0
    assert np.isnan(s["win_rate"])
