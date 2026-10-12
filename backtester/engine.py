"""
Engine: target weights + prices -> held weights, costs, daily returns, equity.

This is where look-ahead bias is won or lost, so the timing is spelled out.

Timeline of one day
-------------------
    open[t] ......... trading session ......... close[t]
                                                   ^ strategy sees close[t] and decides w[t]

w[t] uses the close of day t, so it cannot be held *during* day t. When the
trade actually happens is the `execution` setting:

- "next_open" (default, realistic): trade at the open of t+1.
  Held weights h[t] = w[t-1]. Overnight from close[t-1] to open[t] we still
  hold the old weights h[t-1]; from open[t] to close[t] we hold h[t].

      growth[t] = (1 + sum_i h[t-1,i] * r_overnight[t,i])
                * (1 - cost * turnover[t])                  <- pay costs when trading at the open
                * (1 + sum_i h[t,i]   * r_intraday[t,i])

- "close" (idealised): trade at the close of t itself, the same price the
  signal was computed from. Only feasible with market-on-close orders
  placed just before the close, but it is the textbook formula:

      h[t] = w[t]
      growth[t] = (1 + sum_i h[t-1,i] * r[t,i]) * (1 - cost * turnover[t])

  Note the h[t-1]: today's return is earned by YESTERDAY's position. This
  one-day lag is the most important line of any backtester.

with
    r[t]           = adj_close[t] / adj_close[t-1] - 1     (close to close)
    r_overnight[t] = adj_open[t]  / adj_close[t-1] - 1
    r_intraday[t]  = adj_close[t] / adj_open[t]    - 1
    turnover[t]    = sum_i |h[t,i] - h[t-1,i]|              (fraction of capital traded)
    cost           = (fee + slippage) as a fraction, charged on turnover

Equity is the compounded product: equity[t] = capital * prod(growth[1..t]).
Costs and returns are multiplied rather than added, so they are exact,
not approximations.

Assumptions (made explicit so they can be questioned)
-----------------------------------------------------
- Uninvested capital (1 - sum of weights) earns 0%. A cash rate comes later.
- Holding constant weights from day to day means quietly rebalancing back
  to target as prices drift. Exact for a single stock at 0 or 100%; for a
  multi-asset portfolio this hides small rebalancing trades. We'll deal
  with drift when we add portfolios.
- We can trade any size at the open/close price plus slippage (no market impact).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

EXECUTIONS = ("next_open", "close")


@dataclass
class BacktestResult:
    target_weights: pd.DataFrame  # w[t]: decided at close of t
    held_weights: pd.DataFrame  # h[t]: actually held after trading on day t
    turnover: pd.Series
    costs: pd.Series  # fraction of equity paid in costs each day
    returns: pd.Series  # net daily portfolio return
    equity: pd.Series
    initial_capital: float
    execution: str
    cost: float  # per unit of turnover, as a fraction
    # Trades recorded by an engine that knows them exactly (the event engine,
    # e.g. stop fills). None -> trades.py reconstructs them from held_weights.
    trade_log: pd.DataFrame | None = None


def adjusted(prices: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    """open/high/low on the same adjusted scale as adj_close.

    adj_close / close is the dividend adjustment factor for each day (see
    data.py: Yahoo's close is already split-adjusted). Applying the same factor
    to the open, high or low puts them on the adj_close scale.
    """
    return prices[field] * prices["adj_close"] / prices["close"]


def run(
    weights: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    cost: float = 0.0,
    execution: str = "next_open",
    initial_capital: float = 1.0,
) -> BacktestResult:
    """Simulate holding `weights` (dates x tickers). `prices` may contain extra
    history before the first date; it is used for the first day's returns."""
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}, got {execution!r}")

    dates, tickers = weights.index, weights.columns
    adj_close = prices["adj_close"][tickers]
    prev_close = adj_close.shift(1).reindex(dates)
    close = adj_close.reindex(dates)

    if execution == "close":
        held = weights
    else:
        held = weights.shift(1).fillna(0.0)
    held_before = held.shift(1).fillna(0.0)  # we start in cash
    turnover = (held - held_before).abs().sum(axis=1)
    trade_factor = 1 - cost * turnover

    if execution == "close":
        r = close / prev_close - 1
        _check_no_missing_returns(r, held_before)
        growth = (1 + (held_before * r.fillna(0)).sum(axis=1)) * trade_factor
    else:
        open_ = adjusted(prices, "open")[tickers].reindex(dates)
        r_overnight = open_ / prev_close - 1
        r_intraday = close / open_ - 1
        _check_no_missing_returns(r_overnight, held_before)
        _check_no_missing_returns(r_intraday, held)
        growth = (
            (1 + (held_before * r_overnight.fillna(0)).sum(axis=1))
            * trade_factor
            * (1 + (held * r_intraday.fillna(0)).sum(axis=1))
        )

    equity = initial_capital * growth.cumprod()
    return BacktestResult(
        target_weights=weights,
        held_weights=held,
        turnover=turnover,
        costs=cost * turnover,
        returns=growth - 1,
        equity=equity,
        initial_capital=initial_capital,
        execution=execution,
        cost=cost,
    )


def _check_no_missing_returns(returns: pd.DataFrame, held: pd.DataFrame) -> None:
    """A missing return is only harmless if we hold nothing in that ticker."""
    bad = returns.isna() & (held != 0)
    if bad.any().any():
        date, ticker = bad.stack().loc[lambda s: s].index[0]
        raise ValueError(f"position held in {ticker} on {date.date()} but its price is missing")
