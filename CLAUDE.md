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
│   ├── spec.py          # StrategySpec: general settings + a typed `strategy` block (the UI/engine contract)
│   ├── data.py          # DataProvider interface, YFinanceProvider, per-ticker CSV cache
│   ├── indicators.py    # our own indicator implementations, registered by name
│   ├── rules.py         # operands/conditions as data (compare, crosses, all/any) -> True/False frames
│   ├── strategies.py    # strategy builders, registered by type -> TARGET WEIGHTS
│   ├── engine.py        # target weights -> held weights, costs, returns, equity
│   ├── metrics.py       # KPIs: CAGR, vol, Sharpe, drawdown, exposure, trades
│   └── backtest.py      # run_backtest(spec): wires everything together
├── app.py               # Streamlit frontend (thin): builds a StrategySpec, plots results (Phase 3)
├── tests/               # hand-computed toy series with known answers; tests/fakes.py has no-network fakes
└── requirements.txt
```

Data flows one way: `StrategySpec` → `data` → `indicators`/`rules` → strategy builder → **target weights** → `engine` → `metrics`.

**Key rules:**
- The engine never talks to the UI. It only reads a `StrategySpec`. The same spec run from the app, a script or a test must give identical results.
- **Strategies are data, not code.** The user composes them in the UI from indicators and conditions. The spec's `strategy` block has a `type` (`rule_based` now; `rotation` and others later), and each type has a registered builder.
- **Every strategy reduces to a target-weights table** (dates × tickers, fraction of capital). The engine only understands weights, so new strategy families never touch the engine, costs or metrics.
- **Indicators are our own implementations** in pandas/numpy (no TA-Lib), registered with `@indicator`, and work on wide frames, so one implementation serves one stock or 500. We only skip one if it needs data we can't get for free (e.g. fundamentals).
- Indicators are computed on full history (warm-up), but each backtest's state machine starts flat at `spec.start`.

Strategy families the design must support (the user's examples):
- Time-series / rule-based: "long when SMA50 crosses above SMA200, flat when it crosses below".
- Cross-sectional / rotation: "each month, hold last month's best-performing S&P 500 stock" (rank by `momentum`, select top N, rebalance on a schedule).

## Core principles (correctness and interview material)

1. **No look-ahead bias.** A decision taken with the close of day *t* is held from *t+1* on. Default execution is `next_open` (trade at the next open); `close` (the same close, idealised) is available. Textbook form:
   `portfolio_return[t] = Σ_i w[t-1,i] × r[t,i]  −  cost × Σ_i |w[t-1,i] − w[t-2,i]|`
   The engine compounds returns and costs multiplicatively (exact). A generic test checks every registered indicator for look-ahead (truncated history must equal full history up to t).
2. **Signal ≠ position ≠ trade.** A signal is the rule firing, a position is what is held, and a trade is a change in position. Keep these as separate series.
3. **Adjusted prices.** Use split- and dividend-adjusted prices for returns. Unadjusted splits look like crashes.
4. **Transaction costs and slippage.** `cost[t] = |position[t] - position[t-1]| * (fee + slippage)`.
5. **Survivorship bias.** For S&P 500 universes, use historical membership from github.com/fja05680/sp500 (MIT, since 1996). Yahoo lacks prices for roughly half of the removed members, and some tickers were **reused by other companies** (e.g. EC = Engelhard in the index, Ecopetrol on Yahoo). Only accept a ticker's prices if they cover its membership dates, report what was dropped, and document the remaining bias.
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
| 6 | Multiple assets: `rotation` strategy type, S&P 500 historical membership, portfolio drift and rebalancing |
| 7 | Robustness: parameter sweeps, walk-forward, out-of-sample, Monte Carlo on trades |

## Current status

- **Phase 0 done:** `StrategySpec`, the data layer with a per-ticker cache (full history, incremental updates, re-download when Yahoo rescales `adj_close`), and the cache directory fixed to `<project>/data_cache`.
- **Phase 1 done:** indicators (`price`, `sma`, `ema`, `momentum`, `volatility`, `rsi`), rules, the `rule_based` strategy, the weights engine (`next_open`/`close`), metrics, and `run_backtest`. There are 80 offline tests.
- Sanity check on SPY 2005–2024, with 3 bps total costs: buy-and-hold CAGR 10.3%, max drawdown −55%. Golden cross CAGR 8.4%, max drawdown −34%, Sharpe 0.68 vs 0.61.
- Data note: Yahoo's `close` is already split-adjusted but not dividend-adjusted. `adj_close` is adjusted for both. The engine derives the adjusted open as `open × adj_close / close`.
- **Next:** Phase 2. A trade list (entry/exit dates, per-trade return), a buy-and-hold benchmark, the drawdown series in results, and possibly a cash rate for Sharpe.

## Commands

```bash
source .venv/bin/activate
pip install -r requirements.txt
pytest                      # run tests
streamlit run app.py        # run the frontend locally (from Phase 3)
```
