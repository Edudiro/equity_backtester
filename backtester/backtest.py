"""
run_backtest: the single entry point, StrategySpec in, results out.

    spec -> data -> indicators/rules -> target weights -> engine -> metrics

Warm-up handling
----------------
Prices are loaded from the start of each ticker's history (cheap, since the
cache holds full history anyway), not from `spec.start`. Indicators are
computed on all of it, so an SMA(200) already has a value on the first day
of the backtest. Only then is everything cut to [start, end].
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester import engine, metrics
from backtester.data import COLUMNS, FULL_HISTORY_START, DataProvider, default_provider
from backtester.rules import Context
from backtester.spec import StrategySpec


@dataclass
class Backtest:
    spec: StrategySpec
    result: engine.BacktestResult
    metrics: dict[str, float]


def load_prices(provider: DataProvider, spec: StrategySpec) -> dict[str, pd.DataFrame]:
    """Wide frames (dates x tickers) for each price field, full history up to spec.end."""
    frames = {t: provider.get_prices(t, FULL_HISTORY_START, spec.end) for t in spec.tickers}
    # Outer join on dates: a ticker that wasn't trading yet gets NaN, not a fake value.
    return {field: pd.DataFrame({t: df[field] for t, df in frames.items()}) for field in COLUMNS}


def run_backtest(spec: StrategySpec, provider: DataProvider | None = None) -> Backtest:
    prices = load_prices(provider or default_provider(), spec)

    dates = prices["adj_close"].loc[spec.start.isoformat() : spec.end.isoformat()].index
    if dates.empty:
        raise ValueError(f"no trading days between {spec.start} and {spec.end}")

    weights = spec.strategy.target_weights(Context(prices), dates[0]).loc[dates]
    result = engine.run(
        weights,
        prices,
        cost=spec.cost_per_unit_turnover,
        execution=spec.execution,
        initial_capital=spec.initial_capital,
    )
    return Backtest(spec, result, metrics.summary(result))
