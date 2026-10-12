from datetime import date

import pytest

from backtester.spec import StrategySpec


def make(**overrides):
    base = dict(ticker="aapl ", start=date(2020, 1, 1), end=date(2021, 1, 1))
    return StrategySpec(**{**base, **overrides})


def test_ticker_is_normalised():
    assert make().ticker == "AAPL"


def test_json_round_trip_gives_identical_spec():
    spec = make(fee_bps=5, slippage_bps=2)
    assert StrategySpec.from_dict(spec.to_dict()) == spec


def test_cost_per_unit_turnover_in_bps():
    assert make(fee_bps=5, slippage_bps=5).cost_per_unit_turnover == pytest.approx(0.001)


def test_spec_is_immutable():
    with pytest.raises(AttributeError):
        make().ticker = "MSFT"


@pytest.mark.parametrize(
    "overrides",
    [
        dict(ticker="  "),
        dict(start=date(2021, 1, 1), end=date(2020, 1, 1)),
        dict(end=date(2999, 1, 1)),
        dict(initial_capital=0),
        dict(fee_bps=-1),
    ],
)
def test_invalid_specs_are_rejected(overrides):
    with pytest.raises(ValueError):
        make(**overrides)
