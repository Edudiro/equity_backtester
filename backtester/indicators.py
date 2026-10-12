"""
Indicators: our own implementations, registered by name.

Why a registry
--------------
The user picks indicators in the UI, so the strategy arrives as *data*
("sma with window 50"), not code. The registry maps that name to a function,
and knows each function's parameters, so we can:

- compute any indicator by name,
- validate a spec before running it ("sma needs an int `window` >= 1"),
- let the UI build its menus straight from the registry.

Adding an indicator means writing one decorated function. Nothing else in
the codebase changes.

Shape convention: wide frames
-----------------------------
Every input and output is a "wide" DataFrame: one row per date, one column
per ticker. pandas operations like `rolling().mean()` work column by column,
so the same `sma` serves a single-stock strategy and a 500-stock ranking.

No look-ahead
-------------
The value on day t may only use data up to and including day t. That means
trailing windows only: no `center=True`, no `shift(-k)`, no full-sample
statistics like `x.mean()`. tests/test_indicators.py checks this
automatically for every registered indicator.

Warm-up
-------
The first `window - 1` values of a rolling indicator are NaN: there isn't
enough history yet. Comparisons with NaN are False, so rules cannot fire
before their indicators are ready. Every indicator below uses
`min_periods` so it is NaN (not a half-baked number) until fully warmed up.

Indicators read `adj_close` so they are on the same scale as returns
(see data.py for why adjusted prices).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from backtester.constants import TRADING_DAYS


@dataclass(frozen=True)
class IndicatorDef:
    name: str
    func: Callable[..., pd.DataFrame]
    inputs: tuple[str, ...]  # price fields the function receives, e.g. ("adj_close",)
    params: dict[str, type]  # parameter name -> type, e.g. {"window": int}
    doc: str


REGISTRY: dict[str, IndicatorDef] = {}


def indicator(inputs: tuple[str, ...] = ("adj_close",), **params: type):
    """Decorator that registers an indicator under its function name."""

    def register(func):
        expected = [*inputs, *params]
        actual = list(inspect.signature(func).parameters)
        if actual != expected:
            raise TypeError(f"{func.__name__}: signature {actual} != declared {expected}")
        REGISTRY[func.__name__] = IndicatorDef(
            func.__name__, func, inputs, params, inspect.getdoc(func) or ""
        )
        return func

    return register


def validate(name: str, params: dict) -> None:
    """Raise ValueError if `name` is unknown or `params` don't match its definition."""
    if name not in REGISTRY:
        raise ValueError(f"unknown indicator {name!r}; available: {sorted(REGISTRY)}")
    spec = REGISTRY[name].params
    if set(params) != set(spec):
        raise ValueError(f"{name} expects parameters {sorted(spec)}, got {sorted(params)}")
    for key, typ in spec.items():
        value = params[key]
        # bool is a subclass of int in Python; reject it explicitly.
        if not isinstance(value, typ) or isinstance(value, bool):
            raise ValueError(f"{name}.{key} must be {typ.__name__}, got {value!r}")
        if typ is int and value < 1:
            raise ValueError(f"{name}.{key} must be >= 1, got {value}")


def compute(name: str, params: dict, prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute indicator `name` from wide price frames keyed by field."""
    validate(name, params)
    d = REGISTRY[name]
    return d.func(*(prices[field] for field in d.inputs), **params)


# --- Indicators ---------------------------------------------------------------


@indicator()
def price(adj_close):
    """The (adjusted) price itself, so rules like "price > sma(200)" are expressible."""
    return adj_close


@indicator(window=int)
def sma(adj_close, window):
    """Simple moving average: mean of the last `window` prices.

    SMA[t] = (P[t] + P[t-1] + ... + P[t-window+1]) / window
    """
    return adj_close.rolling(window, min_periods=window).mean()


@indicator(window=int)
def ema(adj_close, window):
    """Exponential moving average with span `window`.

    EMA[t] = a * P[t] + (1 - a) * EMA[t-1],  a = 2 / (window + 1)

    Recent prices weigh more than in an SMA, so it reacts faster. We hide the
    first `window - 1` values, when the average is still dominated by its
    starting point.
    """
    return adj_close.ewm(span=window, adjust=False, min_periods=window).mean()


@indicator(window=int)
def momentum(adj_close, window):
    """Trailing return over `window` days: P[t] / P[t-window] - 1.

    About 21 trading days = 1 month, 63 = 3 months, 252 = 1 year.
    """
    return adj_close / adj_close.shift(window) - 1


@indicator(window=int)
def volatility(adj_close, window):
    """Annualised volatility: std of the last `window` daily returns * sqrt(252).

    Daily returns are roughly independent, so variance grows linearly with
    time, and std with its square root: sigma_annual = sigma_daily * sqrt(252).
    """
    returns = adj_close.pct_change(fill_method=None)
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


@indicator(window=int)
def rsi(adj_close, window):
    """Relative Strength Index (Wilder), between 0 and 100.

    gain[t] = max(P[t] - P[t-1], 0),   loss[t] = max(P[t-1] - P[t], 0)
    RS      = avg_gain / avg_loss,     RSI = 100 - 100 / (1 + RS)

    Averages use Wilder's smoothing, an EMA with a = 1 / window. (Wilder
    started from a simple average of the first `window` values; pandas starts
    from the first value. The difference fades out within a few windows.)
    Only gains, no losses -> RSI = 100. Only losses -> 0.
    """
    change = adj_close.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    # Written as 100 * gain / (gain + loss), which is the same formula but
    # avoids dividing by zero when there are no losses.
    return 100 * avg_gain / (avg_gain + avg_loss)
