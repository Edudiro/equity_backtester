import pandas as pd
import pytest

from backtester.rules import Context, parse_condition

PRICE = {"indicator": "price"}


def ctx(values) -> Context:
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return Context({"adj_close": pd.DataFrame({"X": values}, index=idx, dtype=float)})


def run(cond: dict, values) -> list[bool]:
    return parse_condition(cond).evaluate(ctx(values))["X"].tolist()


PRICES = [1, 3, 2, 4, 1]


def test_level_comparison():
    assert run({"left": PRICE, "op": ">", "right": {"value": 2.5}}, PRICES) == [False, True, False, True, False]


def test_crosses_above_fires_only_on_the_crossing_day():
    # Day 0 has no "yesterday", so it can never be a crossing.
    cond = {"left": PRICE, "op": "crosses_above", "right": {"value": 2.5}}
    assert run(cond, PRICES) == [False, True, False, True, False]
    assert run(cond, [1, 3, 4, 5]) == [False, True, False, False]  # stays above: no new cross


def test_crosses_below():
    cond = {"left": PRICE, "op": "crosses_below", "right": {"value": 2.5}}
    assert run(cond, PRICES) == [False, False, True, False, True]


def test_warm_up_nan_never_fires():
    # SMA(3) is NaN for 2 days; "sma > 0" would otherwise be trivially true.
    cond = {"left": {"indicator": "sma", "window": 3}, "op": ">", "right": {"value": 0}}
    assert run(cond, PRICES) == [False, False, True, True, True]


def test_all_and_any():
    above_1 = {"left": PRICE, "op": ">", "right": {"value": 1.5}}
    below_3 = {"left": PRICE, "op": "<", "right": {"value": 3.5}}
    assert run({"all": [above_1, below_3]}, PRICES) == [False, True, True, False, False]
    assert run({"any": [above_1, below_3]}, PRICES) == [True] * 5


def test_round_trip_to_dict():
    d = {
        "all": [
            {"left": {"indicator": "sma", "window": 50}, "op": "crosses_above", "right": {"indicator": "sma", "window": 200}},
            {"left": {"indicator": "rsi", "window": 14}, "op": "<", "right": {"value": 70}},
        ]
    }
    assert parse_condition(d).to_dict() == d


@pytest.mark.parametrize(
    "bad",
    [
        {"left": PRICE, "op": "!=", "right": {"value": 1}},  # unknown op
        {"left": {"indicator": "sma"}, "op": ">", "right": {"value": 1}},  # missing window
        {"left": {"indicator": "foo"}, "op": ">", "right": {"value": 1}},  # unknown indicator
        {"left": PRICE, "op": ">", "right": {"value": "1"}},  # value not a number
        {"left": PRICE, "op": ">"},  # missing right
        {"all": []},
    ],
)
def test_invalid_conditions_are_rejected_at_parse_time(bad):
    with pytest.raises(ValueError):
        parse_condition(bad)


def test_context_computes_each_indicator_once():
    c = ctx(PRICES)
    assert c.indicator("sma", {"window": 2}) is c.indicator("sma", {"window": 2})
