"""Phase 7 cost monotonicity tests: with cash never binding (these
scenarios keep the portfolio far below the cash budget, so every order is
fillable under every cost assumption), higher cost assumptions can never
make the simulated portfolio better off: final equity weakly decreases,
fees weakly increase, and the zero-cost run dominates every positive-cost
run. Monotonicity is guaranteed for the executor's arithmetic (fills are
priced with costs that only subtract); when cash binds, a higher-cost
scenario could fill fewer shares and the comparison would no longer be
clean - hence the unconstrained setups below."""

from __future__ import annotations

from datetime import date

import pytest

from orbit.backtest import BacktestConfig, CostConfig
from orbit.backtest.config import ExecutionConfig

from phase7_testutils import make_bars, run_default, signals, weekdays

DATES = weekdays(date(2024, 1, 2), 10)
_BARS = make_bars(DATES, drift=0.002)
_SIGNALS = signals("INS-000001", DATES[:5], target=1000) + signals(
    "INS-000001", DATES[6:7], direction="flat", start_index=5
)


def _final_equity(costs: CostConfig) -> float:
    res = run_default(_BARS, [], config=BacktestConfig(costs=costs)).run(
        _BARS, _SIGNALS
    )
    res.assert_accounting_clean()
    return res.summary()["final_equity"]


def _fees(costs: CostConfig) -> float:
    res = run_default(_BARS, [], config=BacktestConfig(costs=costs)).run(
        _BARS, _SIGNALS
    )
    return res.summary()["total_fees"]


def test_zero_cost_baseline_dominates_positive_costs():
    zero = _final_equity(CostConfig())
    for costs in [
        CostConfig(spread_bps=1),
        CostConfig(fees_bps=1),
        CostConfig(slippage_bps=1),
        CostConfig(spread_bps=2, fees_bps=1, slippage_bps=2),
        CostConfig(fixed_fee_per_order=5.0),
        CostConfig(fee_minimum=1.0),
    ]:
        assert _final_equity(costs) <= zero, costs


def test_equity_weakly_decreases_in_spread_bps():
    equities = [_final_equity(CostConfig(spread_bps=b)) for b in (0, 5, 10, 20)]
    for lower, higher in zip(equities, equities[1:]):
        assert higher <= lower + 1e-9
    assert equities[-1] < equities[0]  # and strictly so with these trades


def test_equity_weakly_decreases_in_fees_bps():
    equities = [_final_equity(CostConfig(fees_bps=b)) for b in (0, 5, 10, 20)]
    for lower, higher in zip(equities, equities[1:]):
        assert higher <= lower + 1e-9


def test_equity_weakly_decreases_in_slippage_bps():
    equities = [_final_equity(CostConfig(slippage_bps=b)) for b in (0, 5, 10, 20)]
    for lower, higher in zip(equities, equities[1:]):
        assert higher <= lower + 1e-9


def test_fees_weakly_increase_with_cost_parameters():
    assert _fees(CostConfig(fees_bps=10)) >= _fees(CostConfig(fees_bps=1))
    assert _fees(CostConfig(fixed_fee_per_order=5.0)) >= _fees(CostConfig())
    assert _fees(CostConfig(fee_minimum=2.0)) >= _fees(CostConfig(fee_minimum=0.1))


def test_spread_and_slippage_are_costs_not_phantom_fees():
    # spread/slippage reduce equity but must NOT be counted as fees
    res = run_default(_BARS, [], config=BacktestConfig(costs=CostConfig(spread_bps=10))).run(
        _BARS, _SIGNALS
    )
    summary = res.summary()
    assert summary["total_fees"] == 0.0
    assert summary["total_spread_cost"] > 0.0
    assert summary["total_slippage_cost"] == 0.0
    # the spread run is never better than the zero-cost run (the price
    # drift can make both profitable, but the spread only subtracts)
    zero = _final_equity(CostConfig())
    assert summary["final_equity"] <= zero
    assert summary["final_equity"] == pytest.approx(
        zero - summary["total_spread_cost"], rel=1e-9
    )


def test_cost_reduction_never_improves_equity():
    # each cost component removed independently cannot hurt equity
    base = _final_equity(CostConfig(spread_bps=3, fees_bps=2, slippage_bps=1))
    assert _final_equity(CostConfig(fees_bps=2, slippage_bps=1)) >= base - 1e-9
    assert _final_equity(CostConfig(spread_bps=3, slippage_bps=1)) >= base - 1e-9
    assert _final_equity(CostConfig(spread_bps=3, fees_bps=2)) >= base - 1e-9


def test_delay_and_price_configuration_change_run_identity():
    a = run_default(_BARS, [], config=BacktestConfig()).run(_BARS, _SIGNALS)
    b = run_default(
        _BARS,
        [],
        config=BacktestConfig(execution=ExecutionConfig(execution_delay=2)),
    ).run(_BARS, _SIGNALS)
    assert a.manifest.run_id != b.manifest.run_id