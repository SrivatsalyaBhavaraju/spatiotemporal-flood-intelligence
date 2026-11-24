"""
Unit tests for src/optimizer/build_resource_plan.py (task 7.3).

Covers tranche decomposition, dispatch ordering (floors first by
vulnerability, then the demand tranche by reported distress), Sphere-standard
supply conversion, coverage-band firmness, rationale text, and the
concentration metric, on small synthetic wards. The real plan (rho=0.6,
157,035 units, 3 of 4 robust-core wards "firm") is validated against the
actual data (see developing.md's task 7.3 notes).

Run with:
    pytest tests/test_build_resource_plan.py -v
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimizer.build_resource_plan import (  # noqa: E402
    ResourcePlanError,
    build_plan,
    concentration,
    coverage_bands,
    dispatch_sequence,
    rationale,
    sphere_supplies,
    split_tranches,
)


def _inputs():
    return pd.DataFrame({
        "population": [2000.0, 2000.0, 2000.0, 2000.0],
        "flooded_share": [0.5, 0.5, 0.5, 0.2],
        "need": [1000, 1000, 1000, 400],
        "signals": [5, 0, 0, 1],
        "vulnerability": [0.4, 0.6, 0.2, 0.3],
    }, index=pd.Index([1, 2, 3, 4], name="Ward_No"))


def test_split_tranches_and_inconsistency_guard():
    units = pd.Series([10, 5], index=[1, 2])
    a, b = split_tranches(units, pd.Series([4, 5], index=[1, 2]))
    assert a.tolist() == [4, 5] and b.tolist() == [6, 0]
    with pytest.raises(ResourcePlanError):
        split_tranches(units, pd.Series([11, 0], index=[1, 2]))


def test_dispatch_floors_first_by_vulnerability_then_demand_by_signal():
    inp = _inputs()
    seq = dispatch_sequence(inp, pd.Series([10, 20, 5, 0], index=inp.index), pd.Series([50, 0, 0, 7], index=inp.index))
    assert seq["tranche"].tolist() == ["A_equity_floor"] * 3 + ["B_demand_responsive"] * 2
    assert seq["Ward_No"].tolist() == [2, 1, 3, 1, 4]       # ward 4's zero floor is skipped
    assert seq["cumulative_units"].iloc[-1] == 92
    assert seq["step"].tolist() == [1, 2, 3, 4, 5]


def test_sphere_supplies_minimum_standards():
    s = sphere_supplies(pd.Series([100, 0, 51]), days=3)
    assert s["water_litres"].tolist() == [4500, 0, 2295]     # 15 L/person/day
    assert s["food_ration_days"].tolist() == [300, 0, 153]
    assert s["emergency_toilets"].tolist() == [2, 0, 2]      # 1 per 50 people, rounded up


def test_coverage_bands_flag_judgment_sensitive_wards():
    inp = _inputs()
    louder = inp.assign(signals=[0, 5, 0, 1])               # moves the demand tranche to another ward
    bands = coverage_bands({"primary": (inp, 0.5), "alt": (louder, 0.5)}, rho=0.5, names=["primary", "alt"])
    assert set(bands.columns) == {"coverage_min", "coverage_max", "relative_swing", "firmness"}
    assert bands.loc[1, "firmness"] == "judgment-sensitive"
    assert (bands["coverage_min"] <= bands["coverage_max"]).all()


def test_rationale_mentions_why():
    base = {"robust_core": True, "vulnerability_rank": 1, "signals": 0, "tranche_b_units": 0, "flooded_share": 0.2}
    text = rationale(pd.Series(base))
    assert "robust-core" in text and "zero reported distress" in text and "20%" in text
    text2 = rationale(pd.Series({**base, "robust_core": False, "signals": 2, "flooded_share": 0.9}))
    assert "demand tranche ran out" in text2


def test_build_plan_end_to_end_on_synthetic_wards():
    plan, seq, budget = build_plan(_inputs(), core=[2], rho=1.0)
    assert budget == round(0.25 * 3400)
    assert plan["total_units"].sum() <= budget
    assert (plan["tranche_a_units"] + plan["tranche_b_units"] == plan["total_units"]).all()
    assert plan.index[0] == 2                                # most vulnerable dispatched first
    assert plan["dispatch_priority"].is_monotonic_increasing
    assert {"water_litres", "food_ration_days", "emergency_toilets", "rationale"} <= set(plan.columns)


def test_concentration_share():
    c = concentration(pd.Series([60, 30, 10], index=[1, 2, 3]), pd.Series([10, 10, 80], index=[1, 2, 3]), k=2)
    assert c["top_wards"] == [1, 2]
    assert c["share_of_units"] == pytest.approx(0.9) and c["share_of_need"] == pytest.approx(0.2)
