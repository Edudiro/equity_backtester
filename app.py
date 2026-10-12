"""
Streamlit frontend. Deliberately thin: it builds a StrategySpec, calls
run_backtest, and draws the results. No backtesting logic lives here.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtester.backtest import Backtest, run_backtest
from backtester.indicators import REGISTRY
from backtester.rules import OPS
from backtester.spec import StrategySpec

BLUE, ORANGE = "#2a78d6", "#eb6834"  # strategy, benchmark (validated categorical pair)
MUTED, GRID = "#898781", "rgba(137,135,129,0.25)"

st.set_page_config(page_title="Equity Backtester", page_icon="📈", layout="centered")

# --- Presets: starting points the user can edit -------------------------------------


def sma(n):
    return {"indicator": "sma", "window": n}


def rsi(n):
    return {"indicator": "rsi", "window": n}


PRICE = {"indicator": "price"}

PRESETS = {
    "Golden cross (SMA 50 / 200)": {
        "entry": {"all": [{"left": sma(50), "op": "crosses_above", "right": sma(200)}]},
        "exit": {"all": [{"left": sma(50), "op": "crosses_below", "right": sma(200)}]},
    },
    "Trend filter (price vs SMA 200)": {
        "entry": {"all": [{"left": PRICE, "op": ">", "right": sma(200)}]},
        "exit": {"all": [{"left": PRICE, "op": "<", "right": sma(200)}]},
    },
    "RSI mean reversion (30 / 70)": {
        "entry": {"all": [{"left": rsi(14), "op": "crosses_above", "right": {"value": 30}}]},
        "exit": {"all": [{"left": rsi(14), "op": "crosses_above", "right": {"value": 70}}]},
    },
}

# --- Rule builder widgets -----------------------------------------------------------


def operand_input(key: str, default: dict) -> dict:
    kinds = ["indicator", "value"]
    kind = st.selectbox("Type", kinds, index=kinds.index("value" if "value" in default else "indicator"), key=f"{key}.kind")
    if kind == "value":
        return {"value": st.number_input("Value", value=float(default.get("value", 0.0)), key=f"{key}.value")}

    names = sorted(REGISTRY)
    name = st.selectbox("Indicator", names, index=names.index(default.get("indicator", "sma")), key=f"{key}.name")
    params = {}
    for p, typ in REGISTRY[name].params.items():
        params[p] = int(st.number_input(p, min_value=1, value=int(default.get(p, 20)), step=1, key=f"{key}.{name}.{p}"))
    return {"indicator": name, **params}


def condition_input(key: str, default: dict) -> dict:
    """Builds {"all"|"any": [compare, ...]} from 1-4 comparisons."""
    combine = "any" if "any" in default else "all"
    rows = default[combine]
    n = st.number_input("Number of conditions", 1, 4, len(rows), key=f"{key}.n")
    mode = st.radio(
        "Combine with", ["all", "any"], index=["all", "any"].index(combine), horizontal=True, key=f"{key}.mode",
        format_func={"all": "AND (all true)", "any": "OR (any true)"}.get,
    )  # fmt: skip
    conditions = []
    for i in range(int(n)):
        d = rows[i] if i < len(rows) else {"left": PRICE, "op": ">", "right": sma(50)}
        st.markdown(f"**Condition {i + 1}**")
        left, op, right = st.columns([5, 3, 5])
        with left:
            l_ = operand_input(f"{key}.{i}.left", d["left"])
        with op:
            ops = list(OPS)
            o = st.selectbox("Operator", ops, index=ops.index(d["op"]), key=f"{key}.{i}.op")
        with right:
            r_ = operand_input(f"{key}.{i}.right", d["right"])
        conditions.append({"left": l_, "op": o, "right": r_})
    return {mode: conditions}


# --- Results ------------------------------------------------------------------------


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(spec_json: str) -> Backtest:
    return run_backtest(StrategySpec.from_dict(json.loads(spec_json)))


def pct(x) -> str:
    return "–" if pd.isna(x) else f"{x:+.1%}" if x != 0 else "0.0%"


def num(x, digits=2) -> str:
    return "–" if pd.isna(x) else f"{x:.{digits}f}"


def base_layout(fig: go.Figure, title: str, yformat: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=False, linecolor=MUTED, showspikes=True, spikemode="across", spikethickness=1, spikecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickformat=yformat)
    return fig


def equity_chart(bt: Backtest, log: bool) -> go.Figure:
    fig = go.Figure()
    bench_name = bt.spec.benchmark or "Buy & hold"
    for name, series, color in [("Strategy", bt.result.equity, BLUE), (bench_name, bt.benchmark.equity, ORANGE)]:
        fig.add_trace(go.Scatter(x=series.index, y=series, name=name, line=dict(color=color, width=2),
                                 hovertemplate="%{y:,.0f}"))  # fmt: skip
    base_layout(fig, "Equity", ",.0f")
    if log:
        fig.update_yaxes(type="log")
    return fig


def drawdown_chart(bt: Backtest) -> go.Figure:
    fig = go.Figure()
    for name, res, color, fill in [
        ("Strategy", bt.result, BLUE, "tozeroy"),
        (bt.spec.benchmark or "Buy & hold", bt.benchmark, ORANGE, None),
    ]:
        dd = res.equity / res.equity.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, line=dict(color=color, width=2 if fill else 1.5),
                                 fill=fill, fillcolor="rgba(42,120,214,0.15)", hovertemplate="%{y:.1%}"))  # fmt: skip
    return base_layout(fig, "Drawdown", ".0%")


def show_results(bt: Backtest) -> None:
    m, b = bt.metrics, bt.benchmark_metrics
    st.caption(f"Engine: {'event-driven (day by day)' if bt.spec.uses_event_engine else 'vectorized'}")

    c1, c2, c3 = st.columns(3)
    c1.metric("CAGR", pct(m["cagr"]), f"{(m['cagr'] - b['cagr']) * 100:+.1f} pp vs bench")
    c2.metric("Sharpe", num(m["sharpe"]), f"{m['sharpe'] - b['sharpe']:+.2f} vs bench")
    c3.metric("Max drawdown", pct(m["max_drawdown"]), f"{(m['max_drawdown'] - b['max_drawdown']) * 100:+.1f} pp")

    log = st.toggle("Log scale", value=True, help="Equal vertical distances = equal % changes. Best for long periods.")
    st.plotly_chart(equity_chart(bt, log), width="stretch")
    st.plotly_chart(drawdown_chart(bt), width="stretch")

    st.subheader("Performance")
    rows = [
        ("Total return", pct, "total_return"), ("CAGR", pct, "cagr"), ("Volatility", pct, "volatility"),
        ("Sharpe", num, "sharpe"), ("Max drawdown", pct, "max_drawdown"),
        ("Longest drawdown (days)", lambda x: f"{x:,}", "longest_drawdown_days"),
        ("Exposure", pct, "exposure"), ("Costs paid", pct, "total_costs"),
    ]  # fmt: skip
    st.dataframe(
        pd.DataFrame({"Strategy": [f(m[k]) for _, f, k in rows], "Benchmark": [f(b[k]) for _, f, k in rows]},
                     index=[label for label, _, _ in rows]),
        width="stretch",
    )  # fmt: skip

    r = bt.relative
    st.subheader("Vs benchmark")
    c1, c2, c3 = st.columns(3)
    c1.metric("Beta", num(r["beta"]))
    c2.metric("Alpha (annual)", pct(r["alpha"]))
    c3.metric("Information ratio", num(r["information_ratio"]))
    st.caption(f"Correlation {num(r['correlation'])} · Tracking error {pct(r['tracking_error'])}")

    s = bt.trade_stats
    st.subheader(f"Trades ({s['n_trades']} closed)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Win rate", "–" if pd.isna(s["win_rate"]) else f"{s['win_rate']:.0%}")
    c2.metric("Profit factor", num(s["profit_factor"]))
    c3.metric("Payoff ratio", num(s["payoff_ratio"]))
    st.caption(
        f"Avg win {pct(s['avg_win'])} · Avg loss {pct(s['avg_loss'])} · Expectancy {pct(s['expectancy'])} · "
        f"Avg holding {num(s['avg_days_held'], 0)} days · Streaks: {s['max_consecutive_wins']} wins / "
        f"{s['max_consecutive_losses']} losses"
    )
    trades = bt.trades.assign(
        entry_date=bt.trades["entry_date"].dt.date,
        exit_date=bt.trades["exit_date"].dt.date,
        net_return=bt.trades["net_return"].map(pct),
    )[["ticker", "entry_date", "exit_date", "entry_price", "exit_price", "days_held", "net_return", "exit_reason"]]
    st.dataframe(trades, width="stretch", hide_index=True,
                 column_config={"entry_price": st.column_config.NumberColumn(format="%.2f"),
                                "exit_price": st.column_config.NumberColumn(format="%.2f")})  # fmt: skip

    st.subheader("Worst drawdowns")
    dd = bt.drawdowns.head(5)
    st.dataframe(
        pd.DataFrame({
            "Depth": dd["depth"].map(pct),
            "Peak": dd["peak_date"].dt.date,
            "Trough": dd["trough_date"].dt.date,
            "Recovered": dd["recovery_date"].dt.date.astype(str).replace("NaT", "not yet"),
            "Days underwater": dd["days_underwater"],
        }),
        width="stretch", hide_index=True,
    )  # fmt: skip


# --- Page ---------------------------------------------------------------------------

st.title("📈 Equity Backtester")
st.caption("Daily data from Yahoo Finance. Decisions use the close; trades fill at the next open by default.")

preset = st.selectbox("Start from a preset", list(PRESETS))
p = PRESETS[preset]

with st.expander("Market & costs", expanded=True):
    tickers = st.text_input("Tickers (comma-separated)", "SPY")
    c1, c2 = st.columns(2)
    start = c1.date_input("Start", date(2005, 1, 1), min_value=date(1970, 1, 1), max_value=date.today())
    end = c2.date_input("End", date.today(), min_value=date(1970, 1, 1), max_value=date.today())
    c1, c2 = st.columns(2)
    fee = c1.number_input("Fee (bps)", 0.0, 100.0, 1.0, 0.5)
    slip = c2.number_input("Slippage (bps)", 0.0, 100.0, 2.0, 0.5)
    c1, c2 = st.columns(2)
    execution = c1.radio("Execution", ["next_open", "close"], horizontal=True,
                         format_func={"next_open": "Next open", "close": "Same close"}.get)  # fmt: skip
    benchmark = c2.text_input("Benchmark", "", placeholder="own tickers", help="Leave empty for buy & hold of the tickers above, or e.g. SPY.")

# Widget keys include the preset, so switching preset resets the rule widgets to its defaults.
with st.expander("Entry rule (go long when...)", expanded=True):
    entry = condition_input(f"{preset}.entry", p["entry"])
with st.expander("Exit rule (go flat when...)", expanded=True):
    exit_ = condition_input(f"{preset}.exit", p["exit"])

with st.expander("Risk: stops (0 = off)"):
    st.caption("Per position, as % of the entry price. Stops switch to the day-by-day engine. "
               "A gap through a stop fills at the open; if a day touches both stop and target, the stop counts.")  # fmt: skip
    c1, c2, c3 = st.columns(3)
    sl = c1.number_input("Stop-loss %", 0.0, 99.0, 0.0, 1.0)
    tp = c2.number_input("Take-profit %", 0.0, 1000.0, 0.0, 1.0)
    ts = c3.number_input("Trailing stop %", 0.0, 99.0, 0.0, 1.0)

spec_dict = {
    "tickers": [t for t in tickers.split(",") if t.strip()],
    "start": start.isoformat(),
    "end": end.isoformat(),
    "fee_bps": fee,
    "slippage_bps": slip,
    "execution": execution,
    "benchmark": benchmark or None,
    "strategy": {"type": "rule_based", "entry": entry, "exit": exit_},
    "risk": {"stop_loss": sl / 100 or None, "take_profit": tp / 100 or None, "trailing_stop": ts / 100 or None},
}

with st.expander("Advanced: spec as JSON"):
    st.caption("This is exactly what the engine receives. Edit it to run strategies the form can't express.")
    edited = st.text_area("Spec", json.dumps(spec_dict, indent=2), height=300, label_visibility="collapsed")

if st.button("Run backtest", type="primary", width="stretch"):
    try:
        spec = StrategySpec.from_dict(json.loads(edited))  # validates before any download
        st.session_state["result"] = cached_backtest(json.dumps(spec.to_dict(), sort_keys=True))
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        st.session_state.pop("result", None)
        st.error(f"Invalid strategy: {e}")

if "result" in st.session_state:
    show_results(st.session_state["result"])
