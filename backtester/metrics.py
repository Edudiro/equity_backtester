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
n_trades      = number of days with any turnover. Each entry and each exit
                counts once, so a round trip counts 2.

CAGR (geometric) vs mean*252 (arithmetic)
-----------------------------------------
For volatile series the arithmetic average overstates what you actually
earn: +50% then -50% averages 0% but loses 25%. CAGR is what the investor
experiences; mean*252 is what Sharpe uses by convention. The gap is
approximately sigma^2 / 2 ("volatility drag").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import BacktestResult

TRADING_DAYS = 252


def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summary(result: BacktestResult) -> dict[str, float]:
    r = result.returns
    n = len(r)
    total = float((1 + r).prod() - 1)
    std = r.std()
    # Prepend the starting capital so a loss on day one counts as a drawdown.
    equity = pd.concat([pd.Series([result.initial_capital]), result.equity], ignore_index=True)
    return {
        "total_return": total,
        "cagr": (1 + total) ** (TRADING_DAYS / n) - 1,
        "volatility": std * np.sqrt(TRADING_DAYS),
        "sharpe": r.mean() / std * np.sqrt(TRADING_DAYS) if std > 0 else float("nan"),
        "max_drawdown": float(drawdown(equity).min()),
        "exposure": float((result.held_weights.abs().sum(axis=1) > 0).mean()),
        "n_trades": int((result.turnover > 0).sum()),
        "total_costs": float(result.costs.sum()),
    }

