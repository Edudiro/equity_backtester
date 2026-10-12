# Equity Backtester

A daily-frequency stock backtesting framework. The user defines a simple strategy in a web frontend and the engine replays it on historical data. It must be usable from a phone at any time.

## Purpose and how to work with the user

- **This is a learning project.** The user wants to understand how a correct backtester is built, because it may come up in quant/trading interviews. Code must be clear and explained: each module starts with a docstring explaining *why* it is designed that way, and the maths (formulas, bias pitfalls) is explained alongside the code.
- **Engine and maths matter. The frontend does not.** Keep the UI thin and simple, and spend the effort on correctness and explanation in the engine.
- The user is comfortable with pandas. Explain concepts, not every pandas line.
- Start slow with a robust structure, then add complexity phase by phase (see the roadmap). Plan together before big steps.

## Scope and constraints

- **Asset class:** stocks only.
- **Frequency:** daily bars only, with no intraday logic.
- **Data:** free sources. Default is `yfinance`, which is unofficial and can break, so it sits behind a `DataProvider` interface with a local disk cache, and a source such as Stooq can be swapped in.
- **Direction:** long-only at first. Shorting comes in Phase 5.
- **Stack:** Python 3.13, pandas, numpy, yfinance, pytest. Frontend is Streamlit, deployed on Streamlit Community Cloud from GitHub so it runs from a phone browser.

## Architecture

```
equity_backtester/
├── backtester/
│   ├── spec.py          # StrategySpec dataclass: the contract between UI and engine
│   ├── data.py          # DataProvider interface, YFinanceProvider, disk cache
│   ├── processing.py    # cleaning, adjusted prices, returns, indicators (SMA, RSI, ...)
│   ├── signals.py       # rules -> target positions (1 long, 0 flat, -1 short)
│   ├── engine.py        # target positions -> trades, costs, positions, equity curve
│   └── metrics.py       # KPIs: returns, vol, Sharpe, drawdown, trade stats, benchmark
├── app.py               # Streamlit frontend (thin): builds a StrategySpec, plots results
├── tests/               # hand-computed toy price series with known answers
├── notebooks/           # explorations / learning notes
└── requirements.txt
```

Data flows one way: `StrategySpec` → `data` → `processing` → `signals` → `engine` → `metrics`.

**Key rule:** the engine never talks to the UI. It only reads a `StrategySpec`. The same spec run from the app, a script or a test must give identical results.

## Core principles (correctness and interview material)

1. **No look-ahead bias.** A signal computed from the close of day *t* can only be traded on *t+1*.
   `strategy_return[t] = position[t-1] * asset_return[t] - costs[t]`
   The `t-1` lag is non-negotiable.
2. **Signal ≠ position ≠ trade.** A signal is the rule firing, a position is what is held, and a trade is a change in position. Keep these as separate series.
3. **Adjusted prices.** Use split- and dividend-adjusted prices for returns. Unadjusted splits look like crashes.
4. **Transaction costs and slippage.** `cost[t] = |position[t] - position[t-1]| * (fee + slippage)`.
5. **Survivorship bias.** Free data only covers stocks that still exist. Document this limitation; it can't be fully fixed.
6. **Overfitting.** Later phases add in-sample/out-of-sample splits, walk-forward testing and parameter sweeps.
7. **Test with known answers.** Every engine feature gets a small hand-made price series where the correct result can be computed by hand.

## Engine strategy

- **Phase A, vectorized (pandas):** the whole backtest as column arithmetic, so the maths is fully visible.
- **Phase B, event-driven (day-by-day loop):** for path-dependent rules such as stop-loss, take-profit, trailing stops and equity-based sizing.
- Cross-check: both engines must produce the same equity curve on strategies that are not path-dependent.

## Roadmap

| Phase | Goal |
|---|---|
| 0 | Project setup, `StrategySpec`, data download and cache |
| 1 | One stock, long/flat, SMA crossover, vectorized engine, basic metrics, tests |
| 2 | Costs and slippage, trade list, buy-and-hold benchmark, drawdown |
| 3 | Streamlit UI with a rule builder (indicator / comparison / threshold), deployed for phone use |
| 4 | Event-driven engine with stop-loss and take-profit, cross-checked against the vectorized engine |
| 5 | Short selling, position sizing (fixed fraction, volatility targeting) |
| 6 | Multiple assets, portfolio rebalancing |
| 7 | Robustness: parameter sweeps, walk-forward, out-of-sample, Monte Carlo on trades |

## Current status

- **Phase 0 done:** `spec.py` (immutable `StrategySpec`, validation, JSON round-trip), `data.py` (`DataProvider` interface, `YFinanceProvider`, `CachedProvider`: one CSV per ticker with full history, incremental updates, and rescaling detection), and offline tests using a fake provider.
- Data note: Yahoo's `close` is already split-adjusted but not dividend-adjusted. `adj_close` is adjusted for both, so returns use `adj_close`.
- **Next:** Phase 1. `processing.py` (returns, SMA), `signals.py` (SMA crossover giving long/flat targets), vectorized `engine.py`, basic `metrics.py`, and hand-computed tests.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest                      # run tests
streamlit run app.py        # run the frontend locally (from Phase 3)
```
