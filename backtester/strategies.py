"""
Strategies: turn a strategy definition into TARGET WEIGHTS.

The central abstraction
-----------------------
Whatever the user asks for, the output of a strategy is one table:

    weights: one row per date, one column per ticker,
             value = fraction of capital we WANT in that ticker after the close of that day.

- SMA crossover on AAPL: one column, values 0 or 1.
- "Hold last month's best S&P 500 stock, rebalanced monthly": 500 columns,
  a single 1 per row that moves on month-ends.

The engine only ever sees this table. So adding a new family of strategies
means writing one new builder here; the engine, costs and metrics stay
untouched.

These are targets decided at the close of day t using data up to t. The
engine is responsible for the delay before they are actually held.

Registry
--------
Like indicators, strategy types are registered by name so the spec can say
{"type": "rule_based", ...} and the UI can list what is available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtester.rules import Condition, Context, parse_condition

STRATEGIES: dict[str, type[Strategy]] = {}


def register(name: str):
    def deco(cls):
        cls.type_name = name
        STRATEGIES[name] = cls
        return cls

    return deco


class Strategy(ABC):
    type_name: str

    @abstractmethod
    def target_weights(self, ctx: Context, start: pd.Timestamp) -> pd.DataFrame:
        """Target weights for every date from `start` on, decided with data up to that date."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> Strategy: ...

    @abstractmethod
    def to_dict(self) -> dict: ...


def parse_strategy(d: dict) -> Strategy:
    d = dict(d)
    kind = d.pop("type", None)
    if kind not in STRATEGIES:
        raise ValueError(f"unknown strategy type {kind!r}; available: {sorted(STRATEGIES)}")
    return STRATEGIES[kind].from_dict(d)


# --- Rule-based: enter / exit on conditions ----------------------------------


@register("rule_based")
@dataclass(frozen=True)
class RuleBased(Strategy):
    """Long when `entry` fires, flat when `exit` fires, otherwise keep the current state.

    From events to a position (a vectorised state machine)
    ------------------------------------------------------
        raw[t] = 1    if entry[t]
                 0    if exit[t]        (exit wins if both fire: the conservative choice)
                 NaN  otherwise         ("no news")
        position = raw.ffill().fillna(0)

    `ffill` carries the last decision forward until the next one, which is
    exactly "hold until told otherwise". `fillna(0)` means we start flat.

    The state machine starts at `start`, not at the beginning of the price
    history. Indicators use the full history (so an SMA(200) is ready on
    day one), but the backtest begins in cash and waits for a fresh entry
    signal, as a real investor starting that day would.

    With several tickers, each gets an equal sleeve of 1/N of capital and
    runs the rules independently.
    """

    entry: Condition
    exit: Condition

    def target_weights(self, ctx: Context, start: pd.Timestamp) -> pd.DataFrame:
        entry = self.entry.evaluate(ctx).loc[start:]
        exit_ = self.exit.evaluate(ctx).loc[start:]

        raw = pd.DataFrame(np.nan, index=entry.index, columns=entry.columns)
        raw[entry] = 1.0
        raw[exit_] = 0.0  # applied second, so exit wins on a tie
        position = raw.ffill().fillna(0.0)
        return position / len(position.columns)

    @classmethod
    def from_dict(cls, d: dict) -> RuleBased:
        if set(d) != {"entry", "exit"}:
            raise ValueError(f"rule_based needs 'entry' and 'exit', got {sorted(d)}")
        return cls(parse_condition(d["entry"]), parse_condition(d["exit"]))

    def to_dict(self) -> dict:
        return {"type": self.type_name, "entry": self.entry.to_dict(), "exit": self.exit.to_dict()}
