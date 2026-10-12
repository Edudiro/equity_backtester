import os
from datetime import date, datetime

import pandas as pd
import pytest

from backtester.data import COLUMNS, CachedProvider, DataProvider, standardise


def raw_yahoo_like_frame() -> pd.DataFrame:
    """Mimics yfinance output: capitalised names, tz-aware index, unsorted, a duplicate row."""
    idx = pd.DatetimeIndex(["2024-01-03", "2024-01-02", "2024-01-03"], tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [11, 10, 12],
            "High": [11, 10, 12],
            "Low": [11, 10, 12],
            "Close": [11, 10, 12],
            "Adj Close": [5.5, 5, 6],
            "Volume": [100, 100, 100],
        },
        index=idx,
    )


def make_history(first: str, last: str) -> pd.DataFrame:
    """Business-day bars with prices 100, 101, 102, ... (adj_close = close)."""
    idx = pd.bdate_range(first, last, name="date")
    px = 100.0 + pd.Series(range(len(idx)), index=idx, dtype=float)
    return pd.DataFrame({c: px for c in ["open", "high", "low", "close", "adj_close"]} | {"volume": 1e6})


class FakeProvider(DataProvider):
    """Serves slices of an in-memory history and records every request."""

    def __init__(self, history: pd.DataFrame):
        self.history = history
        self.requests: list[tuple[date, date]] = []

    def get_prices(self, ticker, start, end):
        self.requests.append((start, end))
        return self.history.loc[start.isoformat() : end.isoformat()].copy()


class Clock:
    """A settable 'today' so tests can simulate days passing."""

    def __init__(self, today: date):
        self.value = today

    def __call__(self) -> date:
        return self.value


def age_file(path, day: date) -> None:
    """Pretend the cache file was last written on `day`."""
    ts = datetime(day.year, day.month, day.day).timestamp()
    os.utime(path, (ts, ts))


# --- standardise ------------------------------------------------------------


def test_standardise_schema():
    df = standardise(raw_yahoo_like_frame())
    assert list(df.columns) == COLUMNS
    assert df.index.name == "date"
    assert df.index.tz is None
    assert df.index.is_monotonic_increasing and df.index.is_unique
    # Duplicate 2024-01-03 rows: the last one wins.
    assert df.loc["2024-01-03", "close"] == 12


def test_standardise_rejects_missing_columns():
    with pytest.raises(ValueError, match="adj_close"):
        standardise(raw_yahoo_like_frame().drop(columns="Adj Close"))


# --- CachedProvider ---------------------------------------------------------


@pytest.fixture
def setup(tmp_path):
    inner = FakeProvider(make_history("2024-01-01", "2024-03-29"))
    clock = Clock(date(2024, 3, 29))
    provider = CachedProvider(inner, cache_dir=tmp_path, today=clock)
    return inner, clock, provider, tmp_path / "AAPL.csv"


def test_first_request_downloads_full_history_once(setup):
    inner, _, provider, path = setup

    provider.get_prices("AAPL", date(2024, 2, 1), date(2024, 2, 29))
    provider.get_prices("AAPL", date(2024, 1, 10), date(2024, 3, 15))  # different range

    assert inner.requests == [(CachedProvider.FULL_HISTORY_START, date(2024, 3, 29))]
    assert path.exists()


def test_requested_range_is_sliced_inclusively(setup):
    inner, _, provider, _ = setup
    df = provider.get_prices("AAPL", date(2024, 2, 1), date(2024, 2, 29))

    pd.testing.assert_frame_equal(df, inner.history.loc["2024-02-01":"2024-02-29"], check_freq=False)
    assert df.index[0] == pd.Timestamp("2024-02-01")
    assert df.index[-1] == pd.Timestamp("2024-02-29")


def test_no_redownload_when_already_checked_today(setup):
    inner, _, provider, _ = setup
    provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    # 2024-03-30 is a Saturday: end is past the last bar, but we already checked today.
    provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 3, 30))

    assert len(inner.requests) == 1


def test_new_days_are_appended_with_small_overlap_fetch(setup):
    inner, clock, provider, path = setup
    provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    age_file(path, date(2024, 3, 29))

    # A week passes and the source has new bars.
    inner.history = make_history("2024-01-01", "2024-04-05")
    clock.value = date(2024, 4, 5)
    df = provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 4, 5))

    # Second request is a short window, not the full history.
    start, _ = inner.requests[1]
    assert start == date(2024, 3, 19)  # last cached day minus OVERLAP_DAYS
    assert len(inner.requests) == 2
    pd.testing.assert_frame_equal(df, inner.history, check_freq=False)


def test_rescaled_history_triggers_full_redownload(setup):
    inner, clock, provider, path = setup
    provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    age_file(path, date(2024, 3, 29))

    # A dividend goes ex: the source rescales ALL past adj_close by 0.98.
    new = make_history("2024-01-01", "2024-04-05")
    new.loc[:"2024-04-01", "adj_close"] *= 0.98
    inner.history = new
    clock.value = date(2024, 4, 5)
    df = provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 4, 5))

    assert inner.requests[-1] == (CachedProvider.FULL_HISTORY_START, date(2024, 4, 5))
    # Cache now holds the consistent, rescaled series.
    pd.testing.assert_frame_equal(df, new, check_freq=False)


def test_unfinished_last_bar_is_overwritten_without_full_redownload(setup):
    inner, clock, provider, path = setup
    provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 3, 29))
    age_file(path, date(2024, 3, 29))

    # The cached 03-29 bar was taken mid-session; the final close was different.
    new = make_history("2024-01-01", "2024-04-05")
    new.loc["2024-03-29", ["close", "adj_close"]] = 250.0
    inner.history = new
    clock.value = date(2024, 4, 5)
    df = provider.get_prices("AAPL", date(2024, 1, 1), date(2024, 4, 5))

    assert len(inner.requests) == 2  # only the overlap fetch
    assert df.loc["2024-03-29", "adj_close"] == 250.0
