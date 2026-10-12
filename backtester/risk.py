"""
Risk rules: stop-loss, take-profit and trailing stop, applied per position.

These sit on top of ANY strategy: the strategy says what it wants to hold,
the risk rules can force an exit earlier. They are fractions of the entry
price (0.05 = 5%); None means off.

    stop_loss      exit if price <= entry * (1 - stop_loss)
    take_profit    exit if price >= entry * (1 + take_profit)
    trailing_stop  exit if price <= highest close since entry * (1 - trailing_stop)

Because whether a stop fires depends on the path the price took since
entry, these need the event-driven engine (event_engine.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskRules:
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None

    def __post_init__(self):
        for name in ("stop_loss", "trailing_stop"):
            v = getattr(self, name)
            if v is not None and not 0 < v < 1:
                raise ValueError(f"{name} must be between 0 and 1 (a fraction), got {v}")
        if self.take_profit is not None and not self.take_profit > 0:
            raise ValueError(f"take_profit must be > 0 (a fraction), got {self.take_profit}")

    @property
    def active(self) -> bool:
        return any(v is not None for v in asdict(self).values())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> RiskRules:
        return cls(**(d or {}))
