"""
Event-driven engine: walk forward one day at a time, like a real trader.

Why a second engine
-------------------
The vectorized engine (engine.py) computes everything as column arithmetic.
That only works when positions are known in advance from the signals.
Stops break this: whether a 5% stop fires depends on the entry price, which
depends on earlier trades, which depend on earlier stops. That is
path-dependent, so we need a loop that carries state from one day to the next.

This engine keeps real accounting: cash and units per ticker. Positions
drift with prices; we only trade when the target changes or a stop fires.

With no risk rules and a single ticker at 0/100%, it must reproduce the
vectorized engine exactly (tests/test_event_engine.py checks this). Two
independent implementations agreeing is strong evidence both are right.

One day, in order (execution = "next_open")
-------------------------------------------
1. Gap check at the open: if a position held overnight opens through its
   stop (or take-profit), it is filled AT THE OPEN, which can be much worse
   than the stop level. A stop order caps nothing when the market gaps.
2. Trade at the open towards yesterday's target weights.
3. Intraday check against the day's low and high:
       low  <= stop level -> filled at the stop level
       high >= take-profit -> filled at the take-profit level
   If both happen on the same day we cannot know which came first from a
   daily bar, so we assume the stop came first (conservative).
4. At the close: update the highest close (for the trailing stop) and record equity.

With execution = "close", steps 1 and 3 run for positions held from the
previous close, then trades happen at today's close towards today's target.

Re-entry after a stop
---------------------
After a stop exit the ticker is blocked: we stay out even if the strategy
still wants it, until the strategy's target for it drops to 0. Only a fresh
signal after that can re-open it. Without this, a rule like "price > SMA"
would buy straight back the next morning.

Known simplification
--------------------
When several tickers are held and only some targets change, the others are
left to drift; with sleeves this can leave cash slightly negative (a small
implicit leverage). Proper portfolio rebalancing comes with multi-asset
support (Phase 6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.engine import EXECUTIONS, BacktestResult, adjusted
from backtester.risk import RiskRules
from backtester.trades import COLUMNS as TRADE_COLUMNS


def run(
    weights: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    risk: RiskRules | None = None,
    cost: float = 0.0,
    execution: str = "next_open",
    initial_capital: float = 1.0,
) -> BacktestResult:
    if execution not in EXECUTIONS:
        raise ValueError(f"execution must be one of {EXECUTIONS}, got {execution!r}")
    sim = _Simulation(weights, prices, risk or RiskRules(), cost, initial_capital)
    sim.run(execution)
    return sim.result(execution)


class _Simulation:
    def __init__(self, weights, prices, risk, cost, capital):
        self.weights = weights
        self.dates, self.tickers = weights.index, list(weights.columns)
        self.risk, self.cost, self.capital = risk, cost, capital

        def px(field):
            frame = prices["adj_close"] if field == "close" else adjusted(prices, field)
            return frame[self.tickers].reindex(self.dates).to_numpy(dtype=float)

        self.O, self.H, self.L, self.C = px("open"), px("high"), px("low"), px("close")
        self.W = weights.to_numpy(dtype=float)

        n, k = self.W.shape
        self.cash = capital
        self.units = np.zeros(k)
        self.last_target = np.zeros(k)  # target last traded towards, to detect changes
        self.blocked = np.zeros(k, dtype=bool)  # stopped out, waiting for a fresh signal
        self.entry_px = np.full(k, np.nan)
        self.entry_day = np.full(k, -1)
        self.peak = np.full(k, np.nan)  # highest close since entry

        self.equity = np.zeros(n)
        self.held = np.zeros((n, k))
        self.turnover = np.zeros(n)
        self.trades: list[dict] = []

    # --- the daily loop -------------------------------------------------------------

    def run(self, execution: str) -> None:
        for t in range(len(self.dates)):
            self._check_prices(t)
            if execution == "next_open":
                self._gap_stops(t)
                if t > 0:
                    self._trade(t, self.W[t - 1], self.O[t])
                self._intraday_stops(t)
                self._update_peaks(t)
            else:
                self._gap_stops(t)
                self._intraday_stops(t)
                self._update_peaks(t)
                self._trade(t, self.W[t], self.C[t])
            self.equity[t] = self._value(self.C[t])
            self.held[t] = self.units * np.nan_to_num(self.C[t]) / self.equity[t]

        for i in np.flatnonzero(self.units != 0):  # still open at the end
            self._record(i, len(self.dates) - 1, self.C[-1, i], "end", is_open=True)

    def _value(self, px: np.ndarray) -> float:
        return self.cash + float(np.sum(np.where(self.units != 0, self.units * px, 0.0)))

    def _check_prices(self, t: int) -> None:
        for arr in (self.O, self.H, self.L, self.C):
            bad = (self.units != 0) & np.isnan(arr[t])
            if bad.any():
                ticker = self.tickers[int(np.flatnonzero(bad)[0])]
                raise ValueError(f"position held in {ticker} on {self.dates[t].date()} but its price is missing")

    # --- trading towards target weights ---------------------------------------------

    def _trade(self, t: int, target: np.ndarray, px: np.ndarray) -> None:
        target = target.copy()
        self.blocked[target == 0] = False  # strategy went flat: a stopped ticker is re-armed
        target[self.blocked] = 0.0

        changed = target != self.last_target
        if not changed.any():
            return

        value = self._value(px)
        current = np.where(self.units != 0, self.units * px, 0.0) / value
        turnover = float(np.abs(target - current)[changed].sum())
        value_after_costs = value * (1 - self.cost * turnover)

        for i in np.flatnonzero(changed):
            if target[i] == 0 and self.units[i] != 0:
                self._record(i, t, px[i], "signal")
            if target[i] != 0 and self.units[i] == 0:
                self.entry_px[i], self.entry_day[i], self.peak[i] = px[i], t, px[i]
            self.units[i] = target[i] * value_after_costs / px[i] if target[i] != 0 else 0.0

        held_value = float(np.sum(np.where(self.units != 0, self.units * px, 0.0)))
        self.cash = value_after_costs - held_value
        self.last_target[changed] = target[changed]
        self.turnover[t] += turnover

    # --- stops ----------------------------------------------------------------------

    def _levels(self, i: int) -> tuple[float, str | None, float]:
        """(stop level, which stop it is, take-profit level); nan when off."""
        r, entry = self.risk, self.entry_px[i]
        stop, kind = np.nan, None
        if r.stop_loss is not None:
            stop, kind = entry * (1 - r.stop_loss), "stop_loss"
        if r.trailing_stop is not None:
            trail = self.peak[i] * (1 - r.trailing_stop)
            if np.isnan(stop) or trail > stop:  # the tighter (higher) stop binds
                stop, kind = trail, "trailing_stop"
        tp = entry * (1 + r.take_profit) if r.take_profit is not None else np.nan
        return stop, kind, tp

    def _gap_stops(self, t: int) -> None:
        """Positions held overnight that open beyond a level are filled at the open."""
        # Runs before today's trades, so every open position was held overnight.
        for i in np.flatnonzero(self.units != 0):
            stop, kind, tp = self._levels(i)
            o = self.O[t, i]
            if o <= stop:
                self._stop_out(i, t, o, kind)
            elif o >= tp:
                self._stop_out(i, t, o, "take_profit")

    def _intraday_stops(self, t: int) -> None:
        # With next_open this includes positions bought at today's open (they live
        # through today's range). With close, today's trades haven't happened yet.
        for i in np.flatnonzero(self.units != 0):
            stop, kind, tp = self._levels(i)
            if self.L[t, i] <= stop:  # checked first: if both were touched, assume the stop hit first
                self._stop_out(i, t, stop, kind)
            elif self.H[t, i] >= tp:
                self._stop_out(i, t, tp, "take_profit")

    def _stop_out(self, i: int, t: int, px: float, reason: str) -> None:
        position_value = self.units[i] * px
        marks = self.O[t].copy()  # value the rest of the portfolio at the open
        marks[i] = px
        value = self._value(marks)
        self.cash += position_value * (1 - self.cost)
        self.turnover[t] += position_value / value
        self._record(i, t, px, reason)
        self.units[i] = 0.0
        self.last_target[i] = 0.0
        self.blocked[i] = True

    def _update_peaks(self, t: int) -> None:
        held = self.units != 0
        self.peak[held] = np.fmax(self.peak[held], self.C[t, held])

    # --- bookkeeping ----------------------------------------------------------------

    def _record(self, i: int, t: int, exit_px: float, reason: str, is_open: bool = False) -> None:
        ratio = exit_px / self.entry_px[i]
        self.trades.append(
            {
                "ticker": self.tickers[i],
                "entry_date": self.dates[self.entry_day[i]],
                "exit_date": self.dates[t],
                "entry_price": self.entry_px[i],
                "exit_price": exit_px,
                "days_held": t - self.entry_day[i],
                "gross_return": ratio - 1,
                "net_return": ratio * (1 - self.cost) ** (1 if is_open else 2) - 1,
                "is_open": is_open,
                "exit_reason": reason,
            }
        )

    def result(self, execution: str) -> BacktestResult:
        equity = pd.Series(self.equity, index=self.dates)
        returns = equity / equity.shift(1, fill_value=self.capital) - 1
        turnover = pd.Series(self.turnover, index=self.dates)
        trades = pd.DataFrame(self.trades, columns=TRADE_COLUMNS)
        return BacktestResult(
            target_weights=self.weights,
            held_weights=pd.DataFrame(self.held, index=self.dates, columns=self.tickers),
            turnover=turnover,
            costs=self.cost * turnover,
            returns=returns,
            equity=equity,
            initial_capital=self.capital,
            execution=execution,
            cost=self.cost,
            trade_log=trades.sort_values(["entry_date", "ticker"], ignore_index=True),
        )
