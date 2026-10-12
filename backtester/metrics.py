"""
Performance metrics (KPIs) computed from daily net returns.

Conventions
-----------
- 252 trading days per year, used to annualise.
- Risk-free rate = 0 for now, so Sharpe = mean / std. With a positive cash
  rate, Sharpe should use excess returns (r - rf). This comes later.
- All figures are fractions (0.12 = 12%).

Formulas (N = number of daily returns r[1..N])
---------------------------------------------
total_return  = prod(1 + r) - 1
CAGR          = (1 + total_return) ^ (252 / N) - 1
                Compound annual growth rate: the constant yearly rate that
                would turn the start value into the end value.
volatility    = std(r) * sqrt(252)
sharpe        = mean(r) / std(r) * sqrt(252)
                Mean scales with time (x252) and std with its square root
                (x sqrt(252)), so their ratio scales with sqrt(252).
drawdown[t]   = equity[t] / max(equity[..t]) - 1
                How far below the previous peak we are on day t.
max_drawdown  = min(drawdown), the worst peak-to-trough loss.
exposure      = share of days with any capital invested.

CAGR (geometric) vs mean*252 (arithmetic)
-----------------------------------------
For volatile series the arithmetic average overstates what you actually
earn: +50% then -50% averages 0% but loses 25%. CAGR is what the investor
experiences; mean*252 is what Sharpe uses by convention. The gap is
approximately sigma^2 / 2 ("volatility drag").

Drawdown episodes
-----------------
An episode starts at a peak, reaches a trough, and ends when equity gets
back above the peak (recovery). Depth is how much it hurt; duration is how
long it hurt. A -20% drawdown that recovers in 2 months and one that takes
5 years are very different experiences, so we report both.

Relative to a benchmark (s = strategy, b = benchmark daily returns)
-------------------------------------------------------------------
beta              = cov(s, b) / var(b)
                    Sensitivity to the benchmark: beta 0.5 means the
                    strategy moves half as much as the market on average.
alpha             = (mean(s) - beta * mean(b)) * 252
                    Annual return NOT explained by market exposure (CAPM, rf = 0).
                    You could get beta * market for free with an index fund;
                    alpha is what the strategy adds on top.
correlation       = corr(s, b)
tracking_error    = std(s - b) * sqrt(252)   how far the strategy wanders from the benchmark
information_ratio = mean(s - b) * 252 / tracking_error
                    Sharpe ratio of the active bet: excess return per unit of deviation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.constants import TRADING_DAYS
from backtester.engine import BacktestResult


def as_python(d: dict) -> dict:
    """numpy scalars (np.float64) -> plain Python numbers, for clean printing and JSON."""
    return {k: v.item() if isinstance(v, np.generic) else v for k, v in d.items()}


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def equity_with_start(result: BacktestResult) -> pd.Series:
    """Equity with the starting capital prepended (dated the day before), so a
    loss on day one counts as a drawdown from the initial capital."""
    start = pd.Series([result.initial_capital], index=[result.equity.index[0] - pd.Timedelta(days=1)])
    return pd.concat([start, result.equity])


def summary(result: BacktestResult) -> dict[str, float]:
    r = result.returns
    n = len(r)
    total = float((1 + r).prod()) - 1
    std = r.std()
    episodes = drawdown_episodes(result)
    return as_python({
        "total_return": total,
        "cagr": (1 + total) ** (TRADING_DAYS / n) - 1,
        "volatility": std * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else float("nan"),
        "max_drawdown": float(drawdown(equity_with_start(result)).min()),
        "longest_drawdown_days": int(episodes["days_underwater"].max()) if len(episodes) else 0,
        "exposure": float((result.held_weights.abs().sum(axis=1) > 0).mean()),
        "total_costs": float(result.costs.sum()),
    })


def drawdown_episodes(result: BacktestResult) -> pd.DataFrame:
    """One row per drawdown, deepest first.

    days_to_trough and days_underwater are counted in trading days.
    recovery_date is NaT and days_underwater runs to the last day if the
    drawdown has not recovered by the end of the backtest.
    """
    equity = equity_with_start(result)
    dd = drawdown(equity).to_numpy()
    dates = equity.index
    underwater = dd < 0
    # Same edge trick as trades.py: runs of consecutive underwater days.
    edges = np.diff(np.concatenate([[False], underwater, [False]]).astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)

    rows = []
    for s, e in zip(starts, ends):
        peak = s - 1  # the last day at the high, just before going underwater
        trough = s + int(np.argmin(dd[s:e]))
        recovered = e < len(dates)
        rows.append(
            {
                "peak_date": dates[peak],
                "trough_date": dates[trough],
                "recovery_date": dates[e] if recovered else pd.NaT,
                "depth": dd[trough],
                "days_to_trough": trough - peak,
                "days_underwater": (e if recovered else len(dates) - 1) - peak,
            }
        )
    columns = ["peak_date", "trough_date", "recovery_date", "depth", "days_to_trough", "days_underwater"]
    return pd.DataFrame(rows, columns=columns).sort_values("depth", ignore_index=True)


def relative(strategy: BacktestResult, benchmark: BacktestResult) -> dict[str, float]:
    s, b = strategy.returns, benchmark.returns
    active = s - b
    beta = s.cov(b) / b.var()
    te = active.std() * np.sqrt(TRADING_DAYS)
    return as_python({
        "beta": beta,
        "alpha": (s.mean() - beta * b.mean()) * TRADING_DAYS,
        "correlation": s.corr(b),
        "tracking_error": te,
        "information_ratio": active.mean() * TRADING_DAYS / te if te > 0 else float("nan"),
    })
