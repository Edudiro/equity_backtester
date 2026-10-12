"""
StrategySpec: the contract between the frontend and the engine.

Why this module exists
----------------------
A backtest must be *reproducible*: the same inputs must always produce the
same outputs. If the engine read settings directly from UI widgets, results
would depend on how they were entered (phone app, script, test), and a test
could never check exactly what the user ran.

So every input that affects a backtest lives in a single, immutable object.
The UI's only job is to build one. The engine's only input is one. That gives
us three properties for free:

1. Reproducibility: save the spec as JSON and you can re-run the backtest later.
2. Testability: tests build specs directly, with no UI involved.
3. Validation in one place: bad inputs (start after end, unknown indicator,
   negative costs) are rejected here, before any data is downloaded.

`frozen=True` makes the object immutable, so nothing downstream can quietly
change a parameter halfway through a run.

The spec has two parts: general settings (universe, dates, capital, costs,
execution) shared by every strategy, and a `strategy` block whose `type`
selects a builder from strategies.py. Example JSON:

    {"tickers": ["AAPL"], "start": "2015-01-01", "end": "2024-12-31",
     "fee_bps": 1, "slippage_bps": 2,
     "strategy": {"type": "rule_based",
                  "entry": {"left": {"indicator": "sma", "window": 50}, "op": "crosses_above",
                            "right": {"indicator": "sma", "window": 200}},
                  "exit":  {"left": {"indicator": "sma", "window": 50}, "op": "crosses_below",
                            "right": {"indicator": "sma", "window": 200}}}}

Costs are in basis points (1 bp = 0.01% = 0.0001), the usual convention on
trading desks. They are charged per unit of turnover; see engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from backtester.engine import EXECUTIONS
from backtester.strategies import Strategy, parse_strategy


@dataclass(frozen=True)
class StrategySpec:
    tickers: tuple[str, ...]
    start: date
    end: date  # inclusive: the last day we want data for
    strategy: Strategy
    initial_capital: float = 10_000.0
    fee_bps: float = 0.0  # broker commission per unit traded, in bps
    slippage_bps: float = 0.0  # execution worse than the quoted price, in bps
    execution: str = "next_open"  # when a decision taken at the close is traded; see engine.py

    def __post_init__(self) -> None:
        # Normalise tickers so "aapl" and "AAPL " hit the same cache file.
        # (object.__setattr__ is the standard way to set a field on a frozen dataclass.)
        tickers = tuple(t.strip().upper() for t in self.tickers)
        object.__setattr__(self, "tickers", tickers)

        if not tickers or not all(tickers):
            raise ValueError("tickers must be a non-empty list of non-empty symbols")
        if len(set(tickers)) != len(tickers):
            raise ValueError(f"duplicate tickers: {tickers}")
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        if self.end > date.today():
            raise ValueError(f"end ({self.end}) is in the future")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be non-negative")
        if self.execution not in EXECUTIONS:
            raise ValueError(f"execution must be one of {EXECUTIONS}, got {self.execution!r}")

    @property
    def cost_per_unit_turnover(self) -> float:
        """Total proportional cost as a fraction, e.g. 5 bps fee + 5 bps slippage -> 0.001."""
        return (self.fee_bps + self.slippage_bps) / 10_000

    # --- JSON round-trip: how the spec crosses the UI/engine boundary --------

    def to_dict(self) -> dict:
        return {
            "tickers": list(self.tickers),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "strategy": self.strategy.to_dict(),
            "initial_capital": self.initial_capital,
            "fee_bps": self.fee_bps,
            "slippage_bps": self.slippage_bps,
            "execution": self.execution,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StrategySpec:
        d = dict(d)
        d["tickers"] = tuple(d["tickers"])
        d["start"] = date.fromisoformat(d["start"])
        d["end"] = date.fromisoformat(d["end"])
        d["strategy"] = parse_strategy(d["strategy"])
        return cls(**d)
