"""Test doubles shared across test files: no network, fully deterministic."""

from datetime import date

import pandas as pd

from backtester.data import DataProvider


def make_history(first: str, last: str) -> pd.DataFrame:
    """Business-day bars with prices 100, 101, 102, ... (open = close = adj_close)."""
    idx = pd.bdate_range(first, last, name="date")
    px = 100.0 + pd.Series(range(len(idx)), index=idx, dtype=float)
    return pd.DataFrame({c: px for c in ["open", "high", "low", "close", "adj_close"]} | {"volume": 1e6})


class FakeProvider(DataProvider):
    """Serves slices of in-memory histories and records every request."""

    def __init__(self, history: pd.DataFrame | dict[str, pd.DataFrame]):
        self.history = history
        self.requests: list[tuple[date, date]] = []

    def get_prices(self, ticker, start, end):
        self.requests.append((start, end))
        h = self.history[ticker] if isinstance(self.history, dict) else self.history
        return h.loc[start.isoformat() : end.isoformat()].copy()
