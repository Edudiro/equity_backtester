from datetime import date

import pytest

from backtester.spec import StrategySpec
from backtester.strategies import parse_strategy

GOLDEN_CROSS = {
    "type": "rule_based",
    "entry": {"left": {"indicator": "sma", "window": 50}, "op": "crosses_above", "right": {"indicator": "sma", "window": 200}},
    "exit": {"left": {"indicator": "sma", "window": 50}, "op": "crosses_below", "right": {"indicator": "sma", "window": 200}},
}


def make(**overrides):
    base = dict(
        tickers=("aapl ",),
        start=date(2020, 1, 1),
        end=date(2021, 1, 1),
        strategy=parse_strategy(GOLDEN_CROSS),
    )
    return StrategySpec(**{**base, **overrides})


def test_tickers_are_normalised():
    assert make().tickers == ("AAPL",)


def test_json_round_trip_gives_identical_spec():
    spec = make(fee_bps=5, slippage_bps=2, execution="close")
    assert StrategySpec.from_dict(spec.to_dict()) == spec


def test_from_json_dict():
    spec = StrategySpec.from_dict(
        {"tickers": ["MSFT"], "start": "2015-01-01", "end": "2020-01-01", "strategy": GOLDEN_CROSS}
    )
    assert spec.strategy.to_dict() == GOLDEN_CROSS


def test_cost_per_unit_turnover_in_bps():
    assert make(fee_bps=5, slippage_bps=5).cost_per_unit_turnover == pytest.approx(0.001)


def test_spec_is_immutable():
    with pytest.raises(AttributeError):
        make().tickers = ("MSFT",)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(tickers=()),
        dict(tickers=("  ",)),
        dict(tickers=("AAPL", "aapl")),
        dict(start=date(2021, 1, 1), end=date(2020, 1, 1)),
        dict(end=date(2999, 1, 1)),
        dict(initial_capital=0),
        dict(fee_bps=-1),
        dict(execution="sometime"),
    ],
)
def test_invalid_specs_are_rejected(overrides):
    with pytest.raises(ValueError):
        make(**overrides)


def test_invalid_strategy_rejected_when_parsing():
    bad = {**GOLDEN_CROSS, "entry": {**GOLDEN_CROSS["entry"], "op": "explodes"}}
    with pytest.raises(ValueError):
        StrategySpec.from_dict({"tickers": ["X"], "start": "2020-01-01", "end": "2021-01-01", "strategy": bad})


def test_engine_selection():
    from backtester.risk import RiskRules

    assert not make().uses_event_engine
    assert make(risk=RiskRules(stop_loss=0.05)).uses_event_engine
    assert make(engine="event").uses_event_engine
    with pytest.raises(ValueError, match="event engine"):
        make(engine="vectorized", risk=RiskRules(stop_loss=0.05))


def test_round_trip_with_risk():
    from backtester.risk import RiskRules

    spec = make(risk=RiskRules(stop_loss=0.05, trailing_stop=0.1), engine="event")
    assert StrategySpec.from_dict(spec.to_dict()) == spec
