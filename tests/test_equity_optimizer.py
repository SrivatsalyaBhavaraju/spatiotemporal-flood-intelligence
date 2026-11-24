"""
Unit tests for src/optimizer/equity_optimizer.py (task 7.2).

Covers the solver's guarantees (budget respected, need caps, equity floors
honoured, rho=0 reduces to the demand-driven optimum, floors feasible for
every rho), the real-signal filters (synthetic smoke-test rows and known
geoparsing false positives dropped, one count per post per ward), the
proportional water-filling reference allocators, and the evaluation metrics,
all on small synthetic wards. The real headline (demand-driven starves 12
of 16 wards incl. 3 robust-core wards; equity model starves none) is
validated against the actual data (see developing.md's task 7.2 notes).

Run with:
    pytest tests/test_equity_optimizer.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimizer.equity_optimizer import (  # noqa: E402
    OptimizerError,
    build_ward_inputs,
    cheapest_no_starvation_rho,
    count_real_distress_signals,
    equity_floors,
    evaluate_allocation,
    flooded_share_by_ward,
    gini,
    price_of_fairness,
    proportional_allocation,
    run_models,
    solve_allocation,
)


def _inputs():
    # ward 1: loud, moderately vulnerable; ward 2: silent, most vulnerable;
    # ward 3: silent, least vulnerable; ward 4: some signal
    return pd.DataFrame({
        "need": [1000, 1000, 1000, 1000],
        "signals": [5, 0, 0, 1],
        "vulnerability": [0.4, 0.6, 0.2, 0.3],
    }, index=pd.Index([1, 2, 3, 4], name="Ward_No"))


# --------------------------------------------------------------- the solver

def test_demand_driven_optimum_starves_silent_wards():
    units = solve_allocation(_inputs(), 1500, rho=0.0)
    assert units.sum() == 1500
    assert units[1] == 1000 and units[4] == 500
    assert units[2] == 0 and units[3] == 0


def test_equity_floors_are_honoured_and_budget_respected():
    inp = _inputs()
    for rho in [0.25, 0.5, 1.0]:
        units = solve_allocation(inp, 1500, rho)
        floors = equity_floors(inp, 1500, rho)
        assert (units >= floors).all()
        assert (units <= inp["need"]).all()
        assert units.sum() <= 1500


def test_rho_one_gives_most_vulnerable_at_least_average_coverage():
    inp = _inputs()
    units = solve_allocation(inp, 1500, 1.0)
    c_bar = 1500 / inp["need"].sum()
    assert units[2] / inp.loc[2, "need"] >= c_bar - 1e-3


def test_floors_feasible_for_every_rho():
    inp = _inputs()
    for rho in np.linspace(0, 1, 11):
        assert equity_floors(inp, 400, rho).sum() <= 400


def test_budget_above_total_need_is_clipped():
    units = solve_allocation(_inputs(), 10_000, 0.5)
    assert units.sum() == 4000


def test_invalid_rho_rejected():
    with pytest.raises(OptimizerError):
        solve_allocation(_inputs(), 100, 1.5)


def test_zero_signal_everywhere_still_solves_deterministically():
    inp = _inputs().assign(signals=0)
    a = solve_allocation(inp, 1500, 0.5)
    b = solve_allocation(inp, 1500, 0.5)
    assert a.equals(b) and a.sum() == 1500


# ------------------------------------------------------- reference allocators

def test_proportional_allocation_water_fills_capped_surplus():
    need = pd.Series([100, 1000, 1000], index=[1, 2, 3])
    w = pd.Series([8, 1, 1], index=[1, 2, 3])
    units = proportional_allocation(w, need, 1000)
    assert units[1] == 100                    # capped at need
    assert units[2] == units[3] == 450        # surplus re-split
    assert units.sum() == 1000


def test_proportional_allocation_ignores_zero_weight_wards():
    need = pd.Series([100, 100], index=[1, 2])
    units = proportional_allocation(pd.Series([1, 0], index=[1, 2]), need, 150)
    assert units[2] == 0 and units[1] == 100  # can't place more than the one signalling ward needs


# --------------------------------------------------------------------- inputs

def test_count_real_distress_signals_applies_both_filters():
    corpus = ["Real post about Velachery and Saidapet flooding.", "Nandambakkam was cut off.",
              "Calm day in Adyar."]
    loc = lambda place, lon: {"place": place, "lon": lon, "lat": 0.0, "score": 100.0}  # noqa: E731
    pipeline = pd.DataFrame([
        {"text": corpus[0], "distress_label": "distress",
         "resolved_locations": json.dumps([loc("Velachery", 1), loc("Velachery", 1), loc("Saidapet", 2)])},
        {"text": corpus[1], "distress_label": "distress",
         "resolved_locations": json.dumps([loc("Adambakkam", 3)])},
        {"text": corpus[2], "distress_label": "not_distress",
         "resolved_locations": json.dumps([loc("Adyar", 4)])},
        {"text": "Synthetic smoke-test post in Velachery.", "distress_label": "distress",
         "resolved_locations": json.dumps([loc("Velachery", 1)])},
    ])
    counts, audit = count_real_distress_signals(pipeline, corpus, lambda lon, lat: {1: 179, 2: 142, 3: 177}.get(lon))
    assert counts.to_dict() == {179: 1, 142: 1}   # one count per post per ward
    assert audit["dropped_not_in_corpus"] == 1
    assert audit["real_distress_posts"] == 2
    assert audit["dropped_false_positive_matches"] == [{"row": 1, "place": "Adambakkam"}]
    assert audit["real_distress_posts_unlocated"] == 1


def test_flooded_share_is_length_weighted():
    seg = pd.DataFrame({"ward_no": [1, 1, 2], "length_m": [300.0, 100.0, 50.0], "true_flood": [1, 0, 0]})
    share = flooded_share_by_ward(seg, "true_flood")
    assert share[1] == pytest.approx(0.75) and share[2] == 0.0


def test_build_ward_inputs_need_and_missing_signals():
    vuln = pd.DataFrame({"Ward_No": [1, 2], "population": [1000.0, 2000.0], "vulnerability_index": [0.5, 0.2]})
    inp = build_ward_inputs(vuln, pd.Series({1: 0.5, 2: 0.25}), pd.Series({1: 3.0}))
    assert inp["need"].tolist() == [500, 500]
    assert inp["signals"].tolist() == [3, 0]
    with pytest.raises(OptimizerError):
        build_ward_inputs(vuln, pd.Series({1: 0.5}), pd.Series(dtype=float))


# ------------------------------------------------------------------ metrics

def test_gini_bounds():
    assert gini([0.25, 0.25, 0.25], [1, 1, 1]) == 0.0
    assert gini([0, 0, 0, 1], [1, 1, 1, 1]) == pytest.approx(0.75)


def test_evaluate_allocation_flags_starved_core_wards():
    inp = _inputs()
    m = evaluate_allocation(inp, solve_allocation(inp, 1500, 0.0), core_wards=[2, 1])
    assert m["core_wards_min_coverage"] == 0.0
    assert 2 in m["starved_wards"] and 3 in m["starved_wards"]
    assert m["signal_weighted_coverage"] == pytest.approx((5 * 1.0 + 1 * 0.5) / 6)


def test_run_models_and_frontier_shape():
    inp = _inputs()
    units, metrics = run_models(inp, 1500, core_wards=[2])
    assert set(units.columns) == {"demand_driven_optimum", "signal_proportional",
                                  "need_proportional", "equity_constrained"}
    assert (units.sum() <= 1500).all()
    assert metrics.loc["equity_constrained", "n_starved_wards"] == 0
    frontier = price_of_fairness(inp, 1500, core_wards=[2], rhos=[0.0, 0.5, 1.0])
    assert frontier["signal_coverage_loss_vs_rho0"].iloc[0] == 0.0
    assert frontier["core_wards_min_coverage"].is_monotonic_increasing
    assert cheapest_no_starvation_rho(frontier) in (0.5, 1.0)
