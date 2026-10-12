"""
Data layer: fetch daily price bars and cache them on disk.

Why this design
---------------
Free data sources are unreliable: yfinance scrapes Yahoo, and it breaks or
gets rate-limited from time to time. So the rest of the code never calls
yfinance directly. It talks to a `DataProvider`, an interface with one method:

    get_prices(ticker, start, end) -> DataFrame

Anything that implements that method can be plugged in: yfinance today,
Stooq or a CSV folder tomorrow, a fake provider in tests.

Caching is a separate wrapper (`CachedProvider`) rather than being baked into
the yfinance code. This is the decorator pattern: it wraps *any* provider, so
switching source doesn't mean rewriting the cache. It also keeps the app
fast and avoids hammering Yahoo, which rate-limits heavy users.

The standard schema
-------------------
Every provider must return the same shape, so downstream code never cares
where the data came from:

    index:   DatetimeIndex named "date", tz-naive, sorted, unique
    columns: open, high, low, close, adj_close, volume

Prices are adjusted *backwards*: history is rescaled so that today's price
stays as traded and corporate actions don't show up as fake jumps.

- A split (e.g. 4-for-1) would otherwise look like a -75% crash overnight.
- A dividend: the price drops by roughly the dividend on the ex-date, but the
  holder received that cash, so it is not a loss.

Watch out: Yahoo's `close` is NOT the raw traded price. It is already
split-adjusted, but not dividend-adjusted. `adj_close` is adjusted for both,
so its percentage changes are the true total return of holding the stock.
Example: AAPL closed at ~$499 on 2020-08-28, before its 4-for-1 split, but
Yahoo reports close = 124.81 and adj_close = 120.96 (the gap is the
dividends paid since then). Returns therefore use adj_close.

This module only fetches and standardises the format. Data quality checks
(gaps, bad prints) belong in processing.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


class DataProvider(ABC):
    """Interface every data source implements."""

    @abstractmethod
    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Daily bars for `ticker` from `start` to `end`, both inclusive, in the standard schema."""


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a raw provider frame into the standard schema, or fail loudly."""
    df = df.rename(columns=lambda c: str(c).strip().lower().replace(" ", "_"))
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"data is missing columns: {sorted(missing)}")

    df = df[COLUMNS].copy()
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        # Yahoo stamps daily bars with the exchange timezone. For daily data
        # only the calendar date matters, so drop the timezone to avoid
        # mismatches when joining series from different sources.
        idx = idx.tz_localize(None)
    df.index = idx.normalize().rename("date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.astype(float)


class YFinanceProvider(DataProvider):
    """Yahoo Finance via the unofficial `yfinance` package."""

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        import yfinance as yf  # imported here so tests never need the network

        raw = yf.Ticker(ticker).history(
            start=start.isoformat(),
            # yfinance treats `end` as exclusive; add a day to make ours inclusive.
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
            # auto_adjust=False returns raw Close *and* Adj Close. With True,
            # Yahoo overwrites OHLC with adjusted values and raw prices are lost.
            auto_adjust=False,
            actions=False,
        )
        if raw.empty:
            raise ValueError(f"no data returned for {ticker!r} between {start} and {end}")
        return standardise(raw)


class CachedProvider(DataProvider):
    """Wraps another provider and keeps one CSV per ticker with its full history.

    How a request is served
    -----------------------
    1. No file yet: download the ticker's entire history once and save it.
    2. File exists and is up to date: just read it and slice [start, end].
    3. File exists but `end` is after the last cached day: fetch only the
       recent days and append them.

    The cache is checked against the source at most once per day (using the
    file's modification date), so a weekend or holiday request doesn't
    trigger a pointless download every time.

    The adjustment trap when appending
    ----------------------------------
    adj_close is adjusted *backwards*. When a stock goes ex-dividend or
    splits, Yahoo rescales its whole adj_close history. Appending new rows
    (on the new scale) to old cached rows (on the old scale) would put a fake
    jump in the series on the day where they meet.

    To catch this, every update re-downloads a short window that overlaps the
    cache and compares adj_close on the overlapping days. If they disagree,
    history was rescaled, and the whole file is downloaded again. The last
    cached day is left out of the comparison: it may have been saved during
    the trading session as an unfinished bar, which is simply overwritten.
    """

    FULL_HISTORY_START = date(1900, 1, 1)  # earlier than any listing: "give me everything"
    OVERLAP_DAYS = 10  # calendar days re-downloaded to check for rescaling

    def __init__(
        self,
        inner: DataProvider,
        cache_dir: str | Path = "data_cache",
        today: Callable[[], date] = date.today,  # injectable so tests can fake time passing
    ):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.today = today

    def _path(self, ticker: str) -> Path:
        return self.cache_dir / f"{ticker}.csv"

    def get_prices(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        path = self._path(ticker)
        if path.exists():
            df = standardise(pd.read_csv(path, index_col="date", parse_dates=True))
            if end > df.index[-1].date() and not self._checked_today(path):
                df = self._update(ticker, df)
                self._save(df, path)
        else:
            df = self._download_all(ticker)
            self._save(df, path)

        # Slicing a DatetimeIndex with strings is inclusive on both ends.
        return df.loc[start.isoformat() : end.isoformat()]

    def _download_all(self, ticker: str) -> pd.DataFrame:
        return self.inner.get_prices(ticker, self.FULL_HISTORY_START, self.today())

    def _update(self, ticker: str, cached: pd.DataFrame) -> pd.DataFrame:
        last = cached.index[-1]
        fresh = self.inner.get_prices(
            ticker, (last - pd.Timedelta(days=self.OVERLAP_DAYS)).date(), self.today()
        )

        # Days present in both, excluding the last cached day (possibly an unfinished bar).
        overlap = cached.index[:-1].intersection(fresh.index)
        consistent = np.allclose(
            cached.loc[overlap, "adj_close"], fresh.loc[overlap, "adj_close"], rtol=1e-6
        )
        if not consistent:
            return self._download_all(ticker)

        # Fresh rows win on overlapping days (standardise keeps the last duplicate).
        return standardise(pd.concat([cached, fresh]))

    def _checked_today(self, path: Path) -> bool:
        return date.fromtimestamp(path.stat().st_mtime) >= self.today()

    def _save(self, df: pd.DataFrame, path: Path) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)


def default_provider(cache_dir: str | Path = "data_cache") -> DataProvider:
    """The provider the app uses unless told otherwise."""
    return CachedProvider(YFinanceProvider(), cache_dir)
