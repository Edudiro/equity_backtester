import pandas as pd
import pytest

from backtester.rules import Context
from backtester.strategies import parse_strategy

PRICE = {"indicator": "price"}
PRICES = [1, 3, 2, 4, 1]


def ctx(columns: dict) -> Context:
    idx = pd.bdate_range("2024-01-01", periods=len(next(iter(columns.values()))))
    return Context({"adj_close": pd.DataFrame(columns, index=idx, dtype=float)})


def rule(entry_op, entry_val, exit_op, exit_val) -> dict:
    return {
        "type": "rule_based",
        "entry": {"left": PRICE, "op": entry_op, "right": {"value": entry_val}},
        "exit": {"left": PRICE, "op": exit_op, "right": {"value": exit_val}},
    }


def weights(spec: dict, columns: dict, start_pos: int = 0) -> pd.DataFrame:
    c = ctx(columns)
    return parse_strategy(spec).target_weights(c, c.index[start_pos])


def test_state_machine_holds_between_events():
    # Enter on crossing above 2.5, exit on crossing below 2.5.
    w = weights(rule("crosses_above", 2.5, "crosses_below", 2.5), {"X": PRICES})
    assert w["X"].tolist() == [0, 1, 0, 1, 0]


def test_position_persists_with_no_new_events():
    # Enter when price > 2.5, exit only when price < 1.5 (hysteresis band).
    w = weights(rule(">", 2.5, "<", 1.5), {"X": [1, 3, 2, 2, 1]})
    assert w["X"].tolist() == [0, 1, 1, 1, 0]


def test_exit_wins_when_both_fire():
    # Entry is always true; exit fires on day 3 only.
    w = weights(rule(">", 0, ">", 3.5), {"X": PRICES})
    assert w["X"].tolist() == [1, 1, 1, 0, 1]


def test_state_machine_starts_flat_at_start_date():
    # Entry at day 1 happened before start (day 2), so we begin flat and wait.
    w = weights(rule("crosses_above", 2.5, "crosses_below", 2.5), {"X": PRICES}, start_pos=2)
    assert w["X"].tolist() == [0, 1, 0]


def test_multiple_tickers_get_equal_sleeves():
    w = weights(rule(">", 0, "<", 0), {"A": PRICES, "B": PRICES})
    assert (w == 0.5).all().all()


def test_round_trip_and_validation():
    spec = rule("crosses_above", 2.5, "crosses_below", 2.5)
    assert parse_strategy(spec).to_dict() == spec
    with pytest.raises(ValueError):
        parse_strategy({"type": "magic"})
    with pytest.raises(ValueError):
        parse_strategy({"type": "rule_based", "entry": spec["entry"]})  # no exit
