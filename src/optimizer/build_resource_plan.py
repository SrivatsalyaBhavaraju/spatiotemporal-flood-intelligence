"""
Task 7.3 -- turn task 7.2's equity-constrained optimizer into an actual,
ranked relief plan per ward: who gets what, in what order, how firm each
number is.

*** WHICH ALLOCATION ***
rho = task 7.2's `cheapest_no_starvation_rho` (0.6 on the real data): the
smallest equity-floor strength at which no ward is starved, costing only
7.2% of reported-distress coverage vs. the demand-driven optimum. Read from
7.2's own report, not hard-coded, so the plan follows the analysis. rho = 1
(the most protective setting) is carried alongside for comparison.

*** TWO TRANCHES, STRAIGHT FROM THE MODEL'S STRUCTURE ***
    Tranche A -- equity floor: each ward's floor_i, guaranteed regardless of
                 reported distress. Dispatched FIRST, most-vulnerable ward first.
    Tranche B -- demand-responsive: x_i - floor_i, the part of the budget that
                 follows reported distress. Dispatched in the order the
                 optimizer fills it (highest share of reported distress first).
This is not an invented heuristic: it is how the solution decomposes -- the
floors are the binding constraints, the remainder is the objective's greedy fill.

*** SUPPLIES: SPHERE MINIMUM STANDARDS, NOT INVENTED RATIOS ***
One relief unit = relief for one affected person for one RELIEF_CYCLE_DAYS
cycle, converted with the Sphere Handbook (2018) minimum standards:
    water    15 L / person / day      (WASH: water supply standard 2.1)
    food     2,100 kcal / person / day = 1 standard ration-day (food security)
    toilets  1 communal toilet / 50 people at the onset of a crisis
             (WASH: excreta management standard 3.2; 1:20 is the mid-term target)
The 3-day cycle is a planning assumption (a typical first relief cycle),
stated as such and kept as a parameter.

*** HOW FIRM IS EACH WARD'S NUMBER? ***
Re-solves the same rho across every 7.2 sensitivity scenario at the same
25% budget share (GNN-forecast exposure, dasymetric population, index
without sanitation, PCA-weighted index) and reports each ward's coverage
band. A ward is "firm" if its coverage moves by <= FIRM_TOLERANCE (relative)
across scenarios, else "judgment-sensitive" -- the plan says which numbers
to trust and which depend on a method choice.

Usage:
    python src/optimizer/build_resource_plan.py

Required inputs: everything task 7.2 reads, plus
    data/processed/optimizer/equity_optimizer_report.json  (task 7.2)

Outputs:
    data/processed/optimizer/resource_plan.csv            (one row per ward)
    data/processed/optimizer/resource_plan.geojson        (same, with ward geometry -- dashboard)
    data/processed/optimizer/dispatch_sequence.csv        (ordered dispatch steps)
    data/processed/optimizer/resource_plan_report.json
    docs/figures/phase7_resource_plan.png
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.optimizer.equity_optimizer import (  # noqa: E402
    BUDGET_SHARE,
    OPT_DIR,
    STUDY_WARDS_PATH,
    equity_floors,
    evaluate_allocation,
    load_inputs_and_scenarios,
    solve_allocation,
)

FIGURE_PATH = REPO_ROOT / "docs" / "figures" / "phase7_resource_plan.png"

RELIEF_CYCLE_DAYS = 3
WATER_L_PER_PERSON_DAY = 15        # Sphere 2018, water supply standard 2.1
PEOPLE_PER_TOILET_ONSET = 50       # Sphere 2018, excreta management standard 3.2 (onset)
FIRM_TOLERANCE = 0.25              # relative coverage swing allowed to still call a ward "firm"
COMPARISON_RHO = 1.0
# Scenarios that keep the 25% budget share -- budget-size scenarios change the
# plan's scale by design, so they're not a fair "is this number firm?" test.
BAND_SCENARIOS = ["primary", "exposure_gnn_forecast", "population_dasymetric",
                  "index_without_sanitation", "index_pca_weighted"]

# Display names, each taken from that ward's own gazetteer locality entries (task 2.9).
WARD_LOCALITY = {
    142: "Saidapet", 168: "Ullagaram", 169: "Puzhuthivakkam", 170: "Ekkattuthangal",
    171: "Little Mount", 172: "Kotturpuram", 173: "Kesavaperumalpuram", 174: "Guindy",
    175: "Gandhinagar / Indira Nagar", 176: "Besant Nagar / Adyar", 177: "Adambakkam",
    178: "Vijaynagar", 179: "Velachery", 180: "Taramani", 181: "Thiruvanmiyur",
    182: "Valmiki Nagar / Kamaraj Nagar",
}


class ResourcePlanError(RuntimeError):
    pass


def split_tranches(units, floors):
    """Tranche A = floor, tranche B = everything above it."""
    if (units < floors).any():
        raise ResourcePlanError("Allocation below its equity floor -- solver output is inconsistent")
    return floors.astype(int), (units - floors).astype(int)


def dispatch_sequence(inputs, tranche_a, tranche_b):
    """Ordered dispatch steps: all of tranche A (most vulnerable first), then
    tranche B in the optimizer's fill order (largest share of reported
    distress first; ties -> larger need, matching 7.2's tie-break)."""
    a = pd.DataFrame({"Ward_No": inputs.index, "tranche": "A_equity_floor", "units": tranche_a.values,
                      "order_key": -inputs["vulnerability"].values, "tie": -inputs["need"].values})
    b = pd.DataFrame({"Ward_No": inputs.index, "tranche": "B_demand_responsive", "units": tranche_b.values,
                      "order_key": -inputs["signals"].values, "tie": -inputs["need"].values})
    seq = pd.concat([a.sort_values(["order_key", "tie"]), b.sort_values(["order_key", "tie"])])
    seq = seq[seq["units"] > 0].drop(columns=["order_key", "tie"]).reset_index(drop=True)
    seq.insert(0, "step", np.arange(1, len(seq) + 1))
    seq["cumulative_units"] = seq["units"].cumsum()
    return seq


def sphere_supplies(units, days=RELIEF_CYCLE_DAYS):
    """Relief units (people covered) -> Sphere-minimum supplies for one cycle."""
    units = pd.Series(units).astype(int)
    return pd.DataFrame({
        "water_litres": units * WATER_L_PER_PERSON_DAY * days,
        "food_ration_days": units * days,
        "emergency_toilets": units.apply(lambda u: math.ceil(u / PEOPLE_PER_TOILET_ONSET) if u > 0 else 0),
    }, index=units.index)


def coverage_bands(scenarios, rho, names=BAND_SCENARIOS, tolerance=FIRM_TOLERANCE):
    """Per-ward coverage across scenarios at the same budget share -> band + firmness."""
    cov = {}
    for name in names:
        inp, share = scenarios[name]
        units = solve_allocation(inp, int(round(share * inp["need"].sum())), rho)
        cov[name] = units / inp["need"].replace(0, np.nan)
    cov = pd.DataFrame(cov).fillna(0.0)
    primary = cov["primary"]
    swing = (cov.max(axis=1) - cov.min(axis=1)) / primary.replace(0, np.nan)
    return pd.DataFrame({
        "coverage_min": cov.min(axis=1), "coverage_max": cov.max(axis=1),
        "relative_swing": swing.fillna(0.0),
        "firmness": np.where(swing.fillna(0.0) <= tolerance, "firm", "judgment-sensitive"),
    })


def rationale(row):
    """One plain-language line per ward explaining its allocation."""
    parts = []
    if row["robust_core"]:
        parts.append(f"robust-core vulnerable ward (#{int(row['vulnerability_rank'])})")
    if row["signals"] == 0:
        parts.append("zero reported distress -- relief comes only from the equity floor")
    elif row["tranche_b_units"] > 0:
        parts.append(f"{int(row['signals'])} distress post(s) -- also receives demand-responsive relief")
    else:
        parts.append(f"{int(row['signals'])} distress post(s), but the demand tranche ran out before this ward")
    if row["flooded_share"] < 0.5:
        parts.append(f"only {row['flooded_share']:.0%} of its roads flooded -- small affected population")
    return "; ".join(parts)


def build_plan(inputs, core, rho, scenarios=None):
    budget = int(round(BUDGET_SHARE * inputs["need"].sum()))
    units = solve_allocation(inputs, budget, rho)
    floors = equity_floors(inputs, budget, rho)
    tranche_a, tranche_b = split_tranches(units, floors)

    plan = inputs.copy()
    plan["locality"] = [WARD_LOCALITY.get(w, "") for w in plan.index]
    plan["vulnerability_rank"] = plan["vulnerability"].rank(ascending=False, method="min").astype(int)
    plan["robust_core"] = plan.index.isin(core)
    plan["tranche_a_units"] = tranche_a
    plan["tranche_b_units"] = tranche_b
    plan["total_units"] = units
    plan["coverage"] = units / plan["need"].replace(0, np.nan)
    plan[f"total_units_rho{COMPARISON_RHO:g}"] = solve_allocation(inputs, budget, COMPARISON_RHO)
    plan = plan.join(sphere_supplies(units))
    if scenarios is not None:
        plan = plan.join(coverage_bands(scenarios, rho))
    plan["rationale"] = plan.apply(rationale, axis=1)

    seq = dispatch_sequence(inputs, tranche_a, tranche_b)
    plan["dispatch_priority"] = seq.groupby("Ward_No")["step"].min().reindex(plan.index).astype(int)
    return plan.sort_values("dispatch_priority"), seq, budget


def concentration(units, need, k=3):
    """Share of relief going to the k best-served wards vs. their share of need."""
    top = units.sort_values(ascending=False).index[:k]
    return {"top_wards": [int(w) for w in top],
            "share_of_units": round(float(units[top].sum() / units.sum()), 4),
            "share_of_need": round(float(need[top].sum() / need.sum()), 4)}


def _plot(plan, rho, path):
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    wards = gpd.read_file(STUDY_WARDS_PATH).to_crs("EPSG:32644")
    wards["Ward_No"] = wards["Ward_No"].astype(int)
    g = wards.merge(plan[["coverage", "dispatch_priority"]], left_on="Ward_No", right_index=True)

    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={"width_ratios": [1, 1.3]})
    g.plot(column="coverage", cmap="YlGn", vmin=0, vmax=1, edgecolor="#444", linewidth=0.6, legend=True,
           legend_kwds={"label": "Coverage (relief units / affected people)", "shrink": 0.7}, ax=ax_map)
    for _, r in g.iterrows():
        p = r.geometry.representative_point()
        ax_map.annotate(f"{r.Ward_No}\n{r.coverage:.0%}", (p.x, p.y), ha="center", va="center", fontsize=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
    ax_map.set_title(f"Relief coverage per ward (rho = {rho:g})", fontsize=11)
    ax_map.set_axis_off()

    order = plan.sort_values("dispatch_priority", ascending=False)
    y = np.arange(len(order))
    ax_bar.barh(y, order["tranche_a_units"] / 1000, color="#6a040f", label="Tranche A: equity floor (dispatched first)")
    ax_bar.barh(y, order["tranche_b_units"] / 1000, left=order["tranche_a_units"] / 1000, color="#f4a261",
                label="Tranche B: demand-responsive")
    for i, (w, r) in enumerate(order.iterrows()):
        band = f"  {r.coverage:.0%}"
        if "coverage_min" in r:
            band += f" [{r.coverage_min:.0%}-{r.coverage_max:.0%}]" + ("" if r.firmness == "firm" else " *")
        ax_bar.text(r.total_units / 1000, i, band, va="center", fontsize=7)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([f"#{int(r.dispatch_priority)}  {w} {r.locality}"
                            f"{'  (core)' if r.robust_core else ''}  sig={int(r.signals)}"
                            for w, r in order.iterrows()], fontsize=8)
    ax_bar.set_xlabel("Relief units (thousand people covered)")
    ax_bar.set_title("Ranked plan: dispatch priority, tranches, coverage [scenario band] (* = judgment-sensitive)",
                     fontsize=10)
    ax_bar.legend(fontsize=8, loc="lower right")
    ax_bar.set_xlim(0, order["total_units"].max() / 1000 * 1.45)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    import geopandas as gpd

    loaded = load_inputs_and_scenarios()
    inputs, core, scenarios = loaded["inputs"], loaded["core"], loaded["scenarios"]
    eo_report = json.loads((OPT_DIR / "equity_optimizer_report.json").read_text())
    rho = eo_report["cheapest_no_starvation_rho"]
    if rho is None:
        raise ResourcePlanError("Task 7.2 found no rho without starved wards -- rerun equity_optimizer.py")

    plan, seq, budget = build_plan(inputs, core, rho, scenarios)
    metrics = evaluate_allocation(inputs, plan["total_units"].reindex(inputs.index), core)

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    plan.round(4).to_csv(OPT_DIR / "resource_plan.csv")
    seq.to_csv(OPT_DIR / "dispatch_sequence.csv", index=False)
    wards = gpd.read_file(STUDY_WARDS_PATH)[["Ward_No", "geometry"]]
    wards["Ward_No"] = wards["Ward_No"].astype(int)
    wards.merge(plan.reset_index(), on="Ward_No").to_file(OPT_DIR / "resource_plan.geojson", driver="GeoJSON")
    _plot(plan, rho, FIGURE_PATH)

    supplies = plan[["water_litres", "food_ration_days", "emergency_toilets"]].sum()
    report = {
        "task": "7.3",
        "rho": rho,
        "rho_source": "task 7.2 cheapest_no_starvation_rho",
        "budget_units": budget,
        "budget_share_of_need": BUDGET_SHARE,
        "relief_cycle_days": RELIEF_CYCLE_DAYS,
        "sphere_standards": {"water_l_per_person_day": WATER_L_PER_PERSON_DAY,
                             "food": "2,100 kcal per person per day = 1 ration-day",
                             "people_per_toilet_onset": PEOPLE_PER_TOILET_ONSET,
                             "source": "Sphere Handbook 2018 minimum standards"},
        "totals": {"tranche_a_units": int(plan["tranche_a_units"].sum()),
                   "tranche_b_units": int(plan["tranche_b_units"].sum()),
                   "water_litres": int(supplies["water_litres"]),
                   "food_ration_days": int(supplies["food_ration_days"]),
                   "emergency_toilets": int(supplies["emergency_toilets"])},
        "metrics": {k: (v if isinstance(v, list) else round(float(v), 4)) for k, v in metrics.items()},
        "concentration": {
            f"rho_{rho:g}": concentration(plan["total_units"], plan["need"]),
            f"rho_{COMPARISON_RHO:g}": concentration(plan[f"total_units_rho{COMPARISON_RHO:g}"], plan["need"]),
            "note": ("The equity floor prevents starvation but does not make the ABOVE-floor remainder equitable: "
                     "the linear objective fills the demand tranche greedily, so the loudest wards reach full "
                     "coverage first. Reported here rather than hidden; raising rho shifts relief from the "
                     "demand tranche into the floors."),
        },
        "firmness": {"firm": [int(w) for w in plan.index[plan["firmness"] == "firm"]],
                     "judgment_sensitive": [int(w) for w in plan.index[plan["firmness"] != "firm"]],
                     "tolerance": FIRM_TOLERANCE, "scenarios": BAND_SCENARIOS},
        "plan": [{"ward": int(w), "locality": r.locality, "dispatch_priority": int(r.dispatch_priority),
                  "tranche_a_units": int(r.tranche_a_units), "tranche_b_units": int(r.tranche_b_units),
                  "total_units": int(r.total_units), "coverage": round(float(r.coverage), 4),
                  "coverage_band": [round(float(r.coverage_min), 4), round(float(r.coverage_max), 4)],
                  "firmness": r.firmness, "rationale": r.rationale} for w, r in plan.iterrows()],
        "dispatch_steps": len(seq),
    }
    (OPT_DIR / "resource_plan_report.json").write_text(json.dumps(report, indent=2))

    cols = ["locality", "vulnerability_rank", "signals", "need", "tranche_a_units", "tranche_b_units", "total_units",
            "coverage", "coverage_min", "coverage_max", "firmness", "water_litres", "food_ration_days",
            "emergency_toilets"]
    print(f"rho = {rho} (from task 7.2); budget = {budget:,} units")
    print(plan[cols].round(3).to_string())
    print("\n" + seq.to_string(index=False))
    print("\nTotals:", json.dumps(report["totals"]))
    print("Metrics:", json.dumps(report["metrics"]))
    print("Rationales:")
    for w, r in plan.iterrows():
        print(f"  {w}: {r.rationale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
