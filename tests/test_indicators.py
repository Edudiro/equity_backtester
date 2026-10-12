import numpy as np
import pandas as pd
import pytest

from backtester import indicators
from backtester.indicators import REGISTRY, compute


def prices_from(values) -> dict[str, pd.DataFrame]:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return {"adj_close": pd.DataFrame({"X": values}, index=idx, dtype=float)}


def col(name, values, **params) -> list:
    """Compute an indicator on one series and return plain floats (NaN kept)."""
    return compute(name, params, prices_from(values))["X"].round(10).tolist()


def assert_same(actual, expected):
    np.testing.assert_allclose(actual, expected, equal_nan=True)


# --- Hand-computed values ------------------------------------------------------


def test_sma():
    assert_same(col("sma", [1, 2, 3, 4, 5], window=3), [np.nan, np.nan, 2, 3, 4])


def test_ema():
    # span 3 -> a = 2 / (3 + 1) = 0.5.  EMA: 1, 1.5, 2.25, 3.125; first 2 hidden.
    assert_same(col("ema", [1, 2, 3, 4], window=3), [np.nan, np.nan, 2.25, 3.125])


def test_momentum():
    assert_same(col("momentum", [100, 110, 121, 133.1], window=2), [np.nan, np.nan, 0.21, 0.21])


def test_volatility_is_zero_for_constant_growth():
    vol = col("volatility", [100 * 1.01**i for i in range(10)], window=5)
    assert_same(vol[5:], [0, 0, 0, 0, 0])
    assert all(np.isnan(vol[:5]))  # 4 returns available on day 4: not enough for window 5


def test_rsi_by_hand():
    # changes: _, +1, -1, +1   gains: _, 1, 0, 1   losses: _, 0, 1, 0
    # Wilder smoothing a = 1/2:  avg_gain 1, .5, .75   avg_loss 0, .5, .25
    # RSI = 100 * g / (g + l): day 2 -> 50, day 3 -> 75
    assert_same(col("rsi", [10, 11, 10, 11], window=2), [np.nan, np.nan, 50, 75])


def test_rsi_extremes():
    assert col("rsi", list(range(1, 30)), window=14)[-1] == 100
    assert col("rsi", list(range(30, 1, -1)), window=14)[-1] == 0


def test_works_column_by_column_on_wide_frames():
    idx = pd.bdate_range("2024-01-01", periods=4)
    wide = {"adj_close": pd.DataFrame({"A": [1, 2, 3, 4], "B": [4, 3, 2, 1]}, index=idx, dtype=float)}
    out = compute("sma", {"window": 2}, wide)
    assert out["A"].tolist()[1:] == [1.5, 2.5, 3.5]
    assert out["B"].tolist()[1:] == [3.5, 2.5, 1.5]


# --- Validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, params",
    [
        ("nope", {}),
        ("sma", {}),
        ("sma", {"window": 5, "extra": 1}),
        ("sma", {"window": 5.0}),
        ("sma", {"window": True}),
        ("sma", {"window": 0}),
    ],
)
def test_invalid_indicator_requests_are_rejected(name, params):
    with pytest.raises(ValueError):
        indicators.validate(name, params)


# --- No look-ahead: checked for EVERY registered indicator -----------------------


def random_walk_prices(n=300, seed=0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    adj = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=(n, 2)), axis=0)), index=idx, columns=["A", "B"]
    )
    return {"adj_close": adj}


def default_params(name: str) -> dict:
    return {key: 10 for key in REGISTRY[name].params}


def has_lookahead(func_of_prices, prices: dict[str, pd.DataFrame]) -> bool:
    """True if values up to day t change when data after t is added.

    An honest indicator computed on data truncated at t must equal the
    full-history indicator up to t. If adding the future changes the past,
    the indicator was using the future.
    """
    full = func_of_prices(prices)
    for cut in (30, 120, 250):
        truncated = func_of_prices({k: v.iloc[:cut] for k, v in prices.items()})
        if not np.allclose(truncated, full.iloc[:cut], rtol=1e-12, equal_nan=True):
            return True
    return False


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_lookahead(name):
    assert not has_lookahead(lambda p: compute(name, default_params(name), p), random_walk_prices())


def test_lookahead_check_catches_a_cheating_indicator():
    """Proof the test above has teeth: a centred window uses the future."""
    centred = lambda p: p["adj_close"].rolling(10, center=True).mean()
    assert has_lookahead(centred, random_walk_prices())
