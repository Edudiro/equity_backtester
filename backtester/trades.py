"""
Trades: break the held-weights matrix into individual round trips.

The equity curve says how the strategy did overall. The trade list says
*how* it got there: a strategy can win through many small gains or through
a few huge ones that pay for many small losses, and those feel very different
to run.

What counts as one trade
------------------------
One continuous stretch of holding a ticker: it opens on the first day the
held weight becomes non-zero and closes on the first day it is back to zero.
Resizing a position that stays open (0.5 -> 0.3) stays inside the same
trade. Each trade is measured on the stock alone, independent of position
size, so a 10% sleeve and a 100% position that both bought at 100 and sold
at 110 are both +10% trades.

Entry and exit prices follow the engine's execution setting:
    next_open -> the adjusted open of the day the position changes
    close     -> the adjusted close of the day the position changes

    gross_return = exit_price / entry_price - 1
    net_return   = (exit_price / entry_price) * (1 - cost)^2 - 1    (cost paid on both sides)

A trade still open on the last day is marked at the last close, with only
the entry cost paid (nothing has been sold yet), and flagged `is_open`.
`exit_reason` is "signal" (the strategy exited), "end" (still open), or, with
the event engine, "stop_loss" / "take_profit" / "trailing_stop".
For a single stock traded at 0 or 100%, compounding all trades' net returns
reproduces the engine's equity exactly; tests/test_trades.py checks this.

Trade statistics (closed trades only)
-------------------------------------
win_rate       = share of trades with net_return > 0
avg_win / avg_loss
payoff_ratio   = avg_win / |avg_loss|
expectancy     = average net_return per trade
               = win_rate * avg_win + (1 - win_rate) * avg_loss
profit_factor  = sum of winning returns / |sum of losing returns|; > 1 means winners outweigh losers

A 30% win rate can be very profitable with a payoff ratio of 4 (trend
following), and a 90% win rate can lose money if the rare losses are huge
(selling insurance). Neither number means much without the other.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult, adjusted
from backtester.metrics import as_python

COLUMNS = [
    "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
    "days_held", "gross_return", "net_return", "is_open", "exit_reason",
]  # fmt: skip


def extract_trades(result: BacktestResult, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if result.trade_log is not None:  # the engine recorded its trades exactly
        return result.trade_log
    held = result.held_weights
    dates = held.index
    fill = adjusted(prices, "open") if result.execution == "next_open" else prices["adj_close"]
    last_close = prices["adj_close"].reindex(dates).iloc[-1]
    c = result.cost

    rows = []
    for ticker in held.columns:
        in_pos = (held[ticker] != 0).to_numpy()
        fill_px = fill[ticker].reindex(dates).to_numpy()
        # A trade starts where in_pos turns on and ends where it turns off.
        # Padding with False on both sides makes every start have a matching end.
        edges = np.diff(np.concatenate([[False], in_pos, [False]]).astype(int))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)  # index of the first day flat again (== len for open trades)

        for s, e in zip(starts, ends):
            is_open = e == len(dates)
            entry = fill_px[s]
            exit_ = last_close[ticker] if is_open else fill_px[e]
            ratio = exit_ / entry
            rows.append(
                {
                    "ticker": ticker,
                    "entry_date": dates[s],
                    "exit_date": dates[-1] if is_open else dates[e],
                    "entry_price": entry,
                    "exit_price": exit_,
                    "days_held": (len(dates) - 1 if is_open else e) - s,
                    "gross_return": ratio - 1,
                    "net_return": ratio * (1 - c) ** (1 if is_open else 2) - 1,
                    "is_open": is_open,
                    "exit_reason": "end" if is_open else "signal",
                }
            )

    trades = pd.DataFrame(rows, columns=COLUMNS)
    return trades.sort_values(["entry_date", "ticker"], ignore_index=True)


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    closed = trades.loc[~trades["is_open"], "net_return"].astype(float)
    wins, losses = closed[closed > 0], closed[closed <= 0]
    nan = float("nan")
    return as_python({
        "n_trades": len(closed),
        "win_rate": len(wins) / len(closed) if len(closed) else nan,
        "avg_win": wins.mean() if len(wins) else nan,
        "avg_loss": losses.mean() if len(losses) else nan,
        "payoff_ratio": wins.mean() / -losses.mean() if len(wins) and losses.sum() < 0 else nan,
        "expectancy": closed.mean() if len(closed) else nan,
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() < 0 else nan,
        "best_trade": closed.max() if len(closed) else nan,
        "worst_trade": closed.min() if len(closed) else nan,
        "avg_days_held": trades.loc[~trades["is_open"], "days_held"].mean() if len(closed) else nan,
        "max_consecutive_wins": _longest_run(closed > 0),
        "max_consecutive_losses": _longest_run(closed <= 0),
    })


def _longest_run(flags: pd.Series) -> int:
    """Longest streak of True values, e.g. [T, T, F, T] -> 2."""
    longest = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest
