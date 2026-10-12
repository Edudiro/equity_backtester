"""
Rules: user-defined conditions, as data, evaluated into True/False frames.

The building blocks
-------------------
An *operand* produces a number per (date, ticker):
    {"indicator": "sma", "window": 50}   -> an indicator from indicators.py
    {"value": 30}                        -> a constant

A *condition* produces True/False per (date, ticker):
    {"left": <operand>, "op": ">", "right": <operand>}
    {"all": [<condition>, ...]}          -> AND
    {"any": [<condition>, ...]}          -> OR

Conditions nest, so "SMA50 > SMA200 and RSI < 30" is an `all` of two
comparisons. The UI builds these dicts; nothing here knows about the UI.

Level vs event comparisons
--------------------------
`>` and `<` are *levels*: true on every day the relation holds.
`crosses_above` is an *event*: true only on the day it starts to hold:

    crosses_above[t] = (L[t] > R[t]) and (L[t-1] <= R[t-1])

Both look only at t and t-1, so no look-ahead. Any NaN (an indicator still
warming up) makes the comparison False, so nothing fires on half-built data.

Parsing into frozen dataclasses means a bad rule is rejected when the spec is
created, not halfway through a backtest.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Union

import pandas as pd

from backtester import indicators


class Context:
    """The price data a backtest runs on, plus a cache of computed indicators.

    The cache matters because the same indicator often appears twice (e.g. in
    the entry and the exit rule) and should only be computed once.
    """

    def __init__(self, prices: dict[str, pd.DataFrame]):
        self.prices = prices
        self.index = prices["adj_close"].index
        self.columns = prices["adj_close"].columns
        self._cache: dict = {}

    def indicator(self, name: str, params: dict) -> pd.DataFrame:
        key = (name, tuple(sorted(params.items())))
        if key not in self._cache:
            self._cache[key] = indicators.compute(name, params, self.prices)
        return self._cache[key]


# --- Operands -----------------------------------------------------------------


@dataclass(frozen=True)
class IndicatorRef:
    name: str
    params: tuple[tuple[str, object], ...]  # a tuple, not a dict, so it is hashable/immutable

    def __post_init__(self):
        indicators.validate(self.name, dict(self.params))

    def evaluate(self, ctx: Context) -> pd.DataFrame:
        return ctx.indicator(self.name, dict(self.params))

    def to_dict(self) -> dict:
        return {"indicator": self.name, **dict(self.params)}


@dataclass(frozen=True)
class Value:
    value: float

    def evaluate(self, ctx: Context) -> pd.DataFrame:
        # A full frame rather than a scalar, so `shift` works for crossings.
        return pd.DataFrame(float(self.value), index=ctx.index, columns=ctx.columns)

    def to_dict(self) -> dict:
        return {"value": self.value}


Operand = Union[IndicatorRef, Value]


def parse_operand(d: dict) -> Operand:
    d = dict(d)
    if "indicator" in d:
        name = d.pop("indicator")
        return IndicatorRef(name, tuple(sorted(d.items())))
    if set(d) == {"value"}:
        if isinstance(d["value"], bool) or not isinstance(d["value"], (int, float)):
            raise ValueError(f"value must be a number, got {d['value']!r}")
        return Value(d["value"])
    raise ValueError(f"operand must have 'indicator' or 'value', got {d}")


# --- Conditions ---------------------------------------------------------------


def _crosses_above(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return (left > right) & (left.shift(1) <= right.shift(1))


def _crosses_below(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    return (left < right) & (left.shift(1) >= right.shift(1))


OPS = {
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
    "crosses_above": _crosses_above,
    "crosses_below": _crosses_below,
}


@dataclass(frozen=True)
class Compare:
    left: Operand
    op: str
    right: Operand

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}; available: {list(OPS)}")

    def evaluate(self, ctx: Context) -> pd.DataFrame:
        return OPS[self.op](self.left.evaluate(ctx), self.right.evaluate(ctx))

    def to_dict(self) -> dict:
        return {"left": self.left.to_dict(), "op": self.op, "right": self.right.to_dict()}


@dataclass(frozen=True)
class AllOf:
    conditions: tuple[Condition, ...]

    def evaluate(self, ctx: Context) -> pd.DataFrame:
        result = self.conditions[0].evaluate(ctx)
        for c in self.conditions[1:]:
            result = result & c.evaluate(ctx)
        return result

    def to_dict(self) -> dict:
        return {"all": [c.to_dict() for c in self.conditions]}


@dataclass(frozen=True)
class AnyOf:
    conditions: tuple[Condition, ...]

    def evaluate(self, ctx: Context) -> pd.DataFrame:
        result = self.conditions[0].evaluate(ctx)
        for c in self.conditions[1:]:
            result = result | c.evaluate(ctx)
        return result

    def to_dict(self) -> dict:
        return {"any": [c.to_dict() for c in self.conditions]}


Condition = Union[Compare, AllOf, AnyOf]


def parse_condition(d: dict) -> Condition:
    if set(d) == {"left", "op", "right"}:
        return Compare(parse_operand(d["left"]), d["op"], parse_operand(d["right"]))
    for key, cls in (("all", AllOf), ("any", AnyOf)):
        if set(d) == {key}:
            if not d[key]:
                raise ValueError(f"'{key}' needs at least one condition")
            return cls(tuple(parse_condition(c) for c in d[key]))
    raise ValueError(f"condition must be {{left, op, right}}, {{all: [...]}} or {{any: [...]}}, got {d}")
