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
3. Validation in one place: bad inputs (start after end, negative costs) are
   rejected here, before any data is downloaded.

`frozen=True` makes the object immutable, so nothing downstream can quietly
change a parameter halfway through a run.

Costs are in basis points (1 bp = 0.01% = 0.0001), the usual convention on
trading desks. They are applied per unit of turnover; see engine.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


@dataclass(frozen=True)
class StrategySpec:
    ticker: str
    start: date
    end: date  # inclusive: the last day we want data for
    initial_capital: float = 10_000.0
    fee_bps: float = 0.0  # broker commission per unit traded, in bps
    slippage_bps: float = 0.0  # execution worse than the quoted price, in bps

    def __post_init__(self) -> None:
        # Normalise the ticker so "aapl" and "AAPL " hit the same cache file.
        # (object.__setattr__ is the standard way to set a field on a frozen dataclass.)
        object.__setattr__(self, "ticker", self.ticker.strip().upper())

        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        if self.end > date.today():
            raise ValueError(f"end ({self.end}) is in the future")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.fee_bps < 0 or self.slippage_bps < 0:
            raise ValueError("costs must be non-negative")

    @property
    def cost_per_unit_turnover(self) -> float:
        """Total proportional cost as a fraction, e.g. 5 bps fee + 5 bps slippage -> 0.001."""
        return (self.fee_bps + self.slippage_bps) / 10_000

    # --- JSON round-trip: how the spec crosses the UI/engine boundary --------

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> StrategySpec:
        d = dict(d)
        d["start"] = date.fromisoformat(d["start"])
        d["end"] = date.fromisoformat(d["end"])
        return cls(**d)
