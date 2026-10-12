from datetime import date

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


class FakeProvider(DataProvider):
    def __init__(self):
        self.calls = 0

    def get_prices(self, ticker, start, end):
        self.calls += 1
        return standardise(raw_yahoo_like_frame())


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


def test_cache_hits_inner_provider_only_once(tmp_path):
    inner = FakeProvider()
    provider = CachedProvider(inner, cache_dir=tmp_path)
    args = ("AAPL", date(2024, 1, 1), date(2024, 1, 31))

    first = provider.get_prices(*args)
    second = provider.get_prices(*args)

    assert inner.calls == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)
