"""
run_backtest: the single entry point, StrategySpec in, results out.

    spec -> data -> indicators/rules -> target weights -> engine -> metrics, trades
                                      buy-and-hold weights -> engine -> benchmark

Warm-up handling
----------------
Prices are loaded from the start of each ticker's history (cheap, since the
cache holds full history anyway), not from `spec.start`. Indicators are
computed on all of it, so an SMA(200) already has a value on the first day
of the backtest. Only then is everything cut to [start, end].

Which engine
------------
Without risk rules the vectorized engine runs (fast, every formula visible).
Stops are path-dependent, so with risk rules the event-driven engine runs.
`spec.engine` can force either one, e.g. to compare them.

A fair benchmark
----------------
The benchmark is buy-and-hold run through the SAME engine with the same
costs and execution: decided on the first day, bought at the next open,
charged the same fees. Comparing a strategy with costs to a benchmark
without them (or bought a day earlier) would bias the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester import engine, event_engine, metrics
from backtester.data import COLUMNS, FULL_HISTORY_START, DataProvider, default_provider
from backtester.rules import Context
from backtester.spec import StrategySpec
from backtester.trades import extract_trades, trade_stats


@dataclass
class Backtest:
    spec: StrategySpec
    result: engine.BacktestResult
    benchmark: engine.BacktestResult
    metrics: dict[str, float]  # strategy KPIs
    benchmark_metrics: dict[str, float]  # same KPIs for buy-and-hold
    relative: dict[str, float]  # beta, alpha, ... vs the benchmark
    trades: pd.DataFrame  # one row per round trip
    trade_stats: dict[str, float]
    drawdowns: pd.DataFrame  # one row per drawdown episode, deepest first


def load_prices(provider: DataProvider, tickers, end) -> dict[str, pd.DataFrame]:
    """Wide frames (dates x tickers) for each price field, full history up to `end`."""
    frames = {t: provider.get_prices(t, FULL_HISTORY_START, end) for t in tickers}
    # Outer join on dates: a ticker that wasn't trading yet gets NaN, not a fake value.
    return {field: pd.DataFrame({t: df[field] for t, df in frames.items()}) for field in COLUMNS}


def run_backtest(spec: StrategySpec, provider: DataProvider | None = None) -> Backtest:
    provider = provider or default_provider()
    prices = load_prices(provider, spec.tickers, spec.end)

    dates = prices["adj_close"].loc[spec.start.isoformat() : spec.end.isoformat()].index
    if dates.empty:
        raise ValueError(f"no trading days between {spec.start} and {spec.end}")

    settings = dict(cost=spec.cost_per_unit_turnover, execution=spec.execution, initial_capital=spec.initial_capital)

    weights = spec.strategy.target_weights(Context(prices), dates[0]).loc[dates]
    if spec.uses_event_engine:
        result = event_engine.run(weights, prices, risk=spec.risk, **settings)
    else:
        result = engine.run(weights, prices, **settings)

    bench_tickers = [spec.benchmark] if spec.benchmark else list(spec.tickers)
    bench_prices = load_prices(provider, bench_tickers, spec.end) if spec.benchmark else prices
    bench_weights = pd.DataFrame(1 / len(bench_tickers), index=dates, columns=bench_tickers)
    benchmark = engine.run(bench_weights, bench_prices, **settings)  # plain buy-and-hold, no stops

    trades = extract_trades(result, prices)
    return Backtest(
        spec=spec,
        result=result,
        benchmark=benchmark,
        metrics=metrics.summary(result),
        benchmark_metrics=metrics.summary(benchmark),
        relative=metrics.relative(result, benchmark),
        trades=trades,
        trade_stats=trade_stats(trades),
        drawdowns=metrics.drawdown_episodes(result),
    )
