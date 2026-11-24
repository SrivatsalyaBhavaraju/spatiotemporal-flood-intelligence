"""
Task 7.2 -- equity-constrained relief allocation optimizer (OR-Tools MIP),
working.md 3.2: allocate limited relief across the 16 study wards so that
wards with high social vulnerability "aren't deprioritized just because they
generate less social-media signal".

*** THE BIAS THIS EXISTS TO FIX IS REAL IN THIS PROJECT'S OWN DATA ***
Joining Objective 2's real distress posts to wards: reported distress is
concentrated in Velachery (179) and Kotturpuram (172), the places news
coverage of the 2015 flood talks about. Three of task 7.1's four robustly
most-vulnerable wards (180 Taramani, 171 Little Mount, 170 Ekkattuthangal)
produce ZERO distress signal. Any allocator driven by reported demand alone
gives them nothing. Measured below, not asserted.

*** INPUTS (all real, upstream task outputs) ***
    need_i     affected people = population_i (7.1) x flooded share_i
               flooded share = road-length-weighted share of ward i's segments
               flooded at PEAK in the fused ground truth (3.4/4.4). Ground truth,
               not the GNN forecast, is the primary exposure: relief is allocated
               during/after the peak, when the observed (SAR/news) extent exists,
               and task 5.1/5.3 showed the GNN's per-ward probabilities barely
               discriminate. The GNN forecast is run as a sensitivity variant.
    signals_i  number of distinct REAL distress posts (task 4.5 pipeline, task 1.13
               corpus only) resolving to a place in ward i. Two filters, both
               found while building this, not assumed:
                 - the pipeline output also contains task 4.5's 4 synthetic
                   smoke-test sentences (one is a located "Velachery" distress
                   post) -- dropped: only rows whose text is in the real corpus
                   count;
                 - task 5.5's known false positive (Nandambakkam fuzzy-matched to
                   Adambakkam, ward 177) -- dropped.
    v_i        vulnerability index (7.1)

*** MODEL ***
Integer relief units x_i (1 unit = relief for 1 affected person), budget B:
    maximise   sum_i  p_i x_i                  p_i = signals_i / sum(signals) + tie-break
    subject to sum_i x_i <= B
               0 <= x_i <= need_i
               x_i >= floor_i                  (equity floor -- the constraint under test)
    floor_i = rho * c_bar * (v_i / v_max) * need_i,   c_bar = B / sum(need)
c_bar is the coverage a perfectly even split would give everyone, so rho = 1
guarantees the MOST vulnerable ward at least average coverage and everyone
else a vulnerability-proportional share of that. sum(floor) <= B for every
rho in [0, 1] because v_i / v_max <= 1 -- feasible by construction. Above
the floors, the budget still follows reported demand: the constraint protects
the silent without ignoring the loud. rho = 0 is the same model with the
constraint switched off ("demand-driven optimum").

Tie-break: wards with no signal have p_i = 0, so without a tie-break the
solver may split leftover budget between them arbitrarily. A 1e-6-scale term
proportional to need_i makes the solution deterministic WITHOUT leaking
vulnerability into the demand-driven model.

Because a greedy LP optimum is an extreme version of "allocate by demand",
two non-optimised references are also reported so the comparison isn't
against a strawman: signal-proportional (what a dispatcher splitting resources
by call volume does) and need-proportional (ignores both signal and
vulnerability).

Usage:
    python src/optimizer/equity_optimizer.py

Required inputs:
    data/processed/optimizer/ward_vulnerability_index.csv   (task 7.1)
    data/processed/dashboard/segments_peak.geojson          (task 6.1: fused labels + GNN probs per segment)
    data/processed/nlp/objective2_pipeline_output.csv       (task 4.5)
    data/raw/distress_text/corpus_draft.csv                 (task 1.13)
    data/raw/wards/study_wards.geojson                      (task 1.3)

Outputs:
    data/processed/optimizer/ward_allocation_inputs.csv
    data/processed/optimizer/allocation_comparison.csv
    data/processed/optimizer/equity_optimizer_report.json
    docs/figures/phase7_allocation_comparison.png
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OPT_DIR = REPO_ROOT / "data" / "processed" / "optimizer"
VULN_PATH = OPT_DIR / "ward_vulnerability_index.csv"
SEGMENTS_PEAK_PATH = REPO_ROOT / "data" / "processed" / "dashboard" / "segments_peak.geojson"
PIPELINE_OUTPUT_PATH = REPO_ROOT / "data" / "processed" / "nlp" / "objective2_pipeline_output.csv"
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "distress_text" / "corpus_draft.csv"
STUDY_WARDS_PATH = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
FIGURE_PATH = REPO_ROOT / "docs" / "figures" / "phase7_allocation_comparison.png"

METRIC_CRS = "EPSG:32644"
# (post-text substring, resolved place) pairs known to be geoparsing errors -- task 5.5.
KNOWN_FALSE_POSITIVE_MATCHES = [("Nandambakkam", "Adambakkam")]
BUDGET_SHARE = 0.25           # primary scenario: relief for 25% of affected people
BUDGET_SHARES_SENSITIVITY = [0.10, 0.25, 0.50]
PRIMARY_RHO = 1.0
RHO_GRID = [round(r, 2) for r in np.linspace(0, 1, 11)]
STARVED_COVERAGE = 0.05       # a ward getting relief for <5% of its affected people is "starved"
TIE_BREAK_SCALE = 1e-6


class OptimizerError(RuntimeError):
    pass


# ------------------------------------------------------------------- inputs

def count_real_distress_signals(pipeline_output, corpus_texts, ward_lookup,
                                false_positives=KNOWN_FALSE_POSITIVE_MATCHES):
    """Distinct real distress posts per ward.

    `pipeline_output`: task 4.5 rows [text, distress_label, resolved_locations(JSON)].
    `corpus_texts`: the real corpus passages; rows whose text isn't among them
    (task 4.5's synthetic smoke tests) are dropped. `ward_lookup(lon, lat)`
    returns a ward number or None. A post mentioning 3 places in one ward
    counts once for that ward. Returns (Series ward -> count, audit dict).
    """
    real = {t.strip() for t in corpus_texts}
    audit = {"pipeline_rows": int(len(pipeline_output)), "dropped_not_in_corpus": 0,
             "real_distress_posts": 0, "real_distress_posts_unlocated": 0,
             "dropped_false_positive_matches": [], "matches_outside_study_wards": 0}
    counts = {}
    for idx, row in pipeline_output.iterrows():
        text = str(row["text"]).strip()
        if text not in real:
            audit["dropped_not_in_corpus"] += 1
            continue
        if row["distress_label"] != "distress":
            continue
        audit["real_distress_posts"] += 1
        locs = json.loads(row["resolved_locations"]) if isinstance(row["resolved_locations"], str) else []
        wards = set()
        for loc in locs:
            if any(fp_text.lower() in text.lower() and loc["place"] == fp_place
                   for fp_text, fp_place in false_positives):
                audit["dropped_false_positive_matches"].append({"row": int(idx), "place": loc["place"]})
                continue
            w = ward_lookup(loc["lon"], loc["lat"])
            if w is None:
                audit["matches_outside_study_wards"] += 1
            else:
                wards.add(int(w))
        if not wards:
            audit["real_distress_posts_unlocated"] += 1
        for w in wards:
            counts[w] = counts.get(w, 0) + 1
    return pd.Series(counts, dtype=float), audit


def flooded_share_by_ward(segments, label_col):
    """Road-length-weighted share of each ward's segments with `label_col` (0/1 or probability)."""
    seg = segments.copy()
    seg["w"] = seg["length_m"] * seg[label_col]
    g = seg.groupby("ward_no")
    return (g["w"].sum() / g["length_m"].sum()).rename(f"flooded_share_{label_col}")


def build_ward_inputs(vuln, flooded_share, signals, population_col="population",
                      vulnerability_col="vulnerability_index"):
    """One row per ward: need, signals, vulnerability."""
    df = pd.DataFrame(index=vuln["Ward_No"].astype(int).values)
    df.index.name = "Ward_No"
    v = vuln.set_index("Ward_No")
    df["population"] = v[population_col].values
    df["flooded_share"] = flooded_share.reindex(df.index).values
    if df["flooded_share"].isna().any():
        raise OptimizerError("Missing flooded share for some wards")
    df["need"] = np.round(df["population"] * df["flooded_share"]).astype(int)
    df["signals"] = signals.reindex(df.index).fillna(0).astype(int).values
    df["vulnerability"] = v[vulnerability_col].values
    return df


# ----------------------------------------------------------------- allocators

def equity_floors(inputs, budget, rho):
    """floor_i = rho * c_bar * (v_i / v_max) * need_i (floored to whole units)."""
    c_bar = budget / inputs["need"].sum()
    rel_v = inputs["vulnerability"] / inputs["vulnerability"].max()
    return np.floor(rho * c_bar * rel_v * inputs["need"]).astype(int)


def solve_allocation(inputs, budget, rho, solver_name="SCIP"):
    """Solve the MIP; returns a Series of integer units per ward."""
    from ortools.linear_solver import pywraplp

    if not 0 <= rho <= 1:
        raise OptimizerError(f"rho must be in [0, 1], got {rho}")
    need = inputs["need"].astype(int)
    budget = int(min(budget, need.sum()))
    floors = equity_floors(inputs, budget, rho)
    total_signals = inputs["signals"].sum()
    p = inputs["signals"] / total_signals if total_signals > 0 else inputs["signals"] * 0.0
    p = p + TIE_BREAK_SCALE * need / need.sum()

    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        raise OptimizerError(f"OR-Tools solver {solver_name} unavailable")
    x = {w: solver.IntVar(int(floors[w]), int(need[w]), f"x_{w}") for w in inputs.index}
    solver.Add(sum(x.values()) <= budget)
    solver.Maximize(sum(float(p[w]) * x[w] for w in inputs.index))
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise OptimizerError(f"Solver did not reach optimality (status {status})")
    return pd.Series({w: int(round(x[w].solution_value())) for w in inputs.index}, name="units")


def proportional_allocation(weights, need, budget):
    """Split `budget` in proportion to `weights`, capped at `need`; capped
    surplus is re-split among uncapped wards (water-filling). Integer units via
    largest remainder. Wards with weight 0 get nothing."""
    weights = weights.astype(float).clip(lower=0)
    need = need.astype(float)
    alloc = pd.Series(0.0, index=need.index)
    remaining = float(min(budget, need[weights > 0].sum()))
    active = weights[weights > 0].index.tolist()
    while remaining > 1e-9 and active:
        share = weights[active] / weights[active].sum() * remaining
        room = need[active] - alloc[active]
        capped = share >= room
        if capped.any():
            for w in share[capped].index:
                remaining -= room[w]
                alloc[w] = need[w]
            active = [w for w in active if not capped[w]]
        else:
            alloc[active] += share
            remaining = 0.0
    base = np.floor(alloc)
    leftover = int(round(alloc.sum() - base.sum()))
    order = (alloc - base).sort_values(ascending=False).index[:leftover]
    base[order] += 1
    return base.astype(int).rename("units")


# ---------------------------------------------------------------- evaluation

def gini(values, weights):
    """Weighted Gini coefficient of `values` (0 = perfectly even)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    if w.sum() == 0 or v.max() == v.min():
        return 0.0
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cvw = np.cumsum(v * w)
    area = np.sum(w * (cvw - v * w / 2)) / (cw[-1] * cvw[-1])
    return float(1 - 2 * area)


def evaluate_allocation(inputs, units, core_wards):
    need = inputs["need"]
    cov = (units / need.replace(0, np.nan)).fillna(0.0)
    zero_sig = inputs["signals"] == 0
    total_signals = inputs["signals"].sum()
    return {
        "units_allocated": int(units.sum()),
        "signal_weighted_coverage": float((inputs["signals"] * cov).sum() / total_signals) if total_signals else 0.0,
        "vulnerability_weighted_coverage": float((inputs["vulnerability"] * units).sum()
                                                 / (inputs["vulnerability"] * need).sum()),
        "n_starved_wards": int(((cov < STARVED_COVERAGE) & (need > 0)).sum()),
        "starved_wards": [int(w) for w in cov[(cov < STARVED_COVERAGE) & (need > 0)].index],
        "core_wards_min_coverage": float(cov.loc[core_wards].min()),
        "core_wards_mean_coverage": float(cov.loc[core_wards].mean()),
        "zero_signal_wards_share_of_units": float(units[zero_sig].sum() / units.sum()) if units.sum() else 0.0,
        "zero_signal_wards_share_of_need": float(need[zero_sig].sum() / need.sum()),
        "coverage_gini": gini(cov.values, need.values),
    }


def run_models(inputs, budget, core_wards, rho=PRIMARY_RHO):
    """All four allocations for one scenario -> (units DataFrame, metrics DataFrame)."""
    units = pd.DataFrame({
        "demand_driven_optimum": solve_allocation(inputs, budget, 0.0),
        "signal_proportional": proportional_allocation(inputs["signals"], inputs["need"], budget),
        "need_proportional": proportional_allocation(inputs["need"], inputs["need"], budget),
        "equity_constrained": solve_allocation(inputs, budget, rho),
    })
    metrics = pd.DataFrame({m: evaluate_allocation(inputs, units[m], core_wards) for m in units.columns}).T
    return units, metrics


def price_of_fairness(inputs, budget, core_wards, rhos=RHO_GRID):
    """Signal-weighted coverage lost vs. vulnerable coverage gained, across rho."""
    rows = []
    for rho in rhos:
        m = evaluate_allocation(inputs, solve_allocation(inputs, budget, rho), core_wards)
        rows.append({"rho": rho, **{k: v for k, v in m.items() if k != "starved_wards"}})
    df = pd.DataFrame(rows)
    base = df.loc[df["rho"] == 0, "signal_weighted_coverage"].iloc[0]
    df["signal_coverage_loss_vs_rho0"] = 1 - df["signal_weighted_coverage"] / base if base else 0.0
    return df


def cheapest_no_starvation_rho(frontier):
    """Smallest rho on the frontier at which no ward is starved (None if never)."""
    ok = frontier[frontier["n_starved_wards"] == 0]
    return float(ok["rho"].min()) if len(ok) else None


# --------------------------------------------------------------------- main

def _load_segments():
    import geopandas as gpd

    seg = gpd.read_file(SEGMENTS_PEAK_PATH).to_crs(METRIC_CRS)
    seg["length_m"] = seg.length
    return seg[["segment_id", "ward_no", "true_flood", "gnn_prob", "length_m"]]


def _ward_lookup():
    import geopandas as gpd
    from shapely.geometry import Point

    wards = gpd.read_file(STUDY_WARDS_PATH)[["Ward_No", "geometry"]]

    def lookup(lon, lat):
        hit = wards[wards.contains(Point(lon, lat))]
        return int(hit["Ward_No"].iloc[0]) if len(hit) else None
    return lookup


def _plot(inputs, units, frontier, path, core_wards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = inputs.sort_values("vulnerability", ascending=False).index
    cov = units.div(inputs["need"], axis=0).loc[order]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
    labels = {"demand_driven_optimum": "Demand-driven optimum (rho=0)",
              "signal_proportional": "Signal-proportional",
              "need_proportional": "Need-proportional (reference)",
              "equity_constrained": f"Equity-constrained (rho={PRIMARY_RHO:g})"}
    colors = {"demand_driven_optimum": "#9e9e9e", "signal_proportional": "#f4a261",
              "need_proportional": "#8ecae6", "equity_constrained": "#6a040f"}
    xs = np.arange(len(order))
    width = 0.2
    for k, m in enumerate(units.columns):
        ax1.bar(xs + (k - 1.5) * width, cov[m], width, label=labels[m], color=colors[m])
    ax1.set_xticks(xs)
    ax1.set_xticklabels([f"{w}{'*' if w in core_wards else ''}\nv={inputs.loc[w, 'vulnerability']:.2f}\n"
                         f"sig={inputs.loc[w, 'signals']}" for w in order], fontsize=7)
    ax1.set_ylabel("Coverage = relief units / affected people")
    ax1.set_title("Ward coverage by allocation model (wards sorted most -> least vulnerable; * = robust core)",
                  fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.axhline(units["need_proportional"].sum() / inputs["need"].sum(), color="black", lw=0.8, ls=":")
    ax1.text(len(order) - 0.5, units["need_proportional"].sum() / inputs["need"].sum() + 0.01,
             "even split", ha="right", fontsize=7)

    ax2.plot(frontier["signal_coverage_loss_vs_rho0"] * 100, frontier["core_wards_min_coverage"] * 100,
             marker="o", color="#6a040f")
    for _, r in frontier.iterrows():
        ax2.annotate(f"rho={r.rho:g}", (r.signal_coverage_loss_vs_rho0 * 100, r.core_wards_min_coverage * 100),
                     fontsize=7, xytext=(4, -3), textcoords="offset points")
    knee = cheapest_no_starvation_rho(frontier)
    if knee is not None:
        k = frontier[frontier["rho"] == knee].iloc[0]
        ax2.scatter([k.signal_coverage_loss_vs_rho0 * 100], [k.core_wards_min_coverage * 100], s=180,
                    facecolors="none", edgecolors="#2a9d8f", linewidths=2, zorder=5)
        ax2.annotate(f"first rho with 0 starved wards\n(costs {k.signal_coverage_loss_vs_rho0:.1%} of "
                     f"reported-distress coverage)", (k.signal_coverage_loss_vs_rho0 * 100,
                                                       k.core_wards_min_coverage * 100),
                     xytext=(25, -40), textcoords="offset points", fontsize=8, color="#2a9d8f",
                     arrowprops={"arrowstyle": "->", "color": "#2a9d8f"})
    ax2.set_xlabel("Reported-distress coverage given up vs. demand-driven optimum (%)")
    ax2.set_ylabel("Lowest coverage among robust-core vulnerable wards (%)")
    ax2.set_title("Price of fairness frontier", fontsize=10)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    vuln = pd.read_csv(VULN_PATH)
    vuln_report = json.loads((OPT_DIR / "vulnerability_index_report.json").read_text())
    core = vuln_report["robustness"]["robust_core"]
    seg = _load_segments()
    corpus = pd.read_csv(CORPUS_PATH)["paragraph_text"].astype(str)
    signals, signal_audit = count_real_distress_signals(pd.read_csv(PIPELINE_OUTPUT_PATH), corpus, _ward_lookup())

    share_gt = flooded_share_by_ward(seg, "true_flood")
    share_gnn = flooded_share_by_ward(seg, "gnn_prob")
    inputs = build_ward_inputs(vuln, share_gt, signals)
    budget = int(round(BUDGET_SHARE * inputs["need"].sum()))

    units, metrics = run_models(inputs, budget, core)
    frontier = price_of_fairness(inputs, budget, core)

    # sensitivity: does the headline survive the judgment calls?
    scenarios = {"primary": (inputs, BUDGET_SHARE)}
    for b in BUDGET_SHARES_SENSITIVITY:
        if b != BUDGET_SHARE:
            scenarios[f"budget_{int(b * 100)}pct"] = (inputs, b)
    scenarios["exposure_gnn_forecast"] = (build_ward_inputs(vuln, share_gnn, signals), BUDGET_SHARE)
    scenarios["population_dasymetric"] = (build_ward_inputs(vuln, share_gt, signals,
                                                            population_col="population_dasymetric"), BUDGET_SHARE)
    scenarios["index_without_sanitation"] = (build_ward_inputs(
        vuln, share_gt, signals, vulnerability_col="vulnerability_index_without_sanitation"), BUDGET_SHARE)
    scenarios["index_pca_weighted"] = (build_ward_inputs(
        vuln, share_gt, signals, vulnerability_col="vulnerability_index_pca_weighted"), BUDGET_SHARE)
    sens = []
    for name, (inp, b) in scenarios.items():
        bud = int(round(b * inp["need"].sum()))
        _, m = run_models(inp, bud, core)
        dd, eq = m.loc["demand_driven_optimum"], m.loc["equity_constrained"]
        sp = m.loc["signal_proportional"]
        sens.append({
            "scenario": name, "budget_units": bud,
            "demand_driven_starved_core_wards": sum(1 for w in dd["starved_wards"] if w in core),
            "signal_proportional_starved_core_wards": sum(1 for w in sp["starved_wards"] if w in core),
            "equity_starved_core_wards": sum(1 for w in eq["starved_wards"] if w in core),
            "equity_core_min_coverage": round(float(eq["core_wards_min_coverage"]), 4),
            "equity_n_starved": int(eq["n_starved_wards"]),
            "demand_driven_n_starved": int(dd["n_starved_wards"]),
            "price_of_fairness_signal_coverage_loss": round(
                1 - float(eq["signal_weighted_coverage"]) / float(dd["signal_weighted_coverage"]), 4)
            if dd["signal_weighted_coverage"] else 0.0,
        })
    sens = pd.DataFrame(sens)

    OPT_DIR.mkdir(parents=True, exist_ok=True)
    inputs.assign(flooded_share_gnn_forecast=share_gnn.reindex(inputs.index).values,
                  equity_floor=equity_floors(inputs, budget, PRIMARY_RHO)).to_csv(OPT_DIR / "ward_allocation_inputs.csv")
    cov = units.div(inputs["need"], axis=0).add_suffix("_coverage")
    units.join(cov).join(inputs).to_csv(OPT_DIR / "allocation_comparison.csv")
    _plot(inputs, units, frontier, FIGURE_PATH, core)

    rho_s = pd.Series(inputs["vulnerability"]).rank().corr(pd.Series(inputs["signals"]).rank())
    report = {
        "task": "7.2",
        "solver": "OR-Tools pywraplp / SCIP (integer relief units)",
        "budget": {"share_of_need": BUDGET_SHARE, "units": budget, "total_need": int(inputs["need"].sum())},
        "primary_rho": PRIMARY_RHO,
        "robust_core_wards": core,
        "signal_audit": signal_audit,
        "signal_gap": {
            "signals_per_ward": {int(w): int(s) for w, s in inputs["signals"].items()},
            "zero_signal_wards": [int(w) for w in inputs.index[inputs["signals"] == 0]],
            "robust_core_wards_with_zero_signal": [int(w) for w in core if inputs.loc[w, "signals"] == 0],
            "spearman_vulnerability_vs_signals": round(float(rho_s), 4),
        },
        "models": {m: {k: (v if isinstance(v, list) else round(float(v), 4)) for k, v in metrics.loc[m].items()}
                   for m in metrics.index},
        "price_of_fairness_frontier": frontier.round(4).to_dict(orient="records"),
        "cheapest_no_starvation_rho": cheapest_no_starvation_rho(frontier),
        "aggregate_metric_caveat": (
            "vulnerability_weighted_coverage is an AGGREGATE (sum v*x / sum v*need) and rewards concentrating "
            "relief on moderately vulnerable, high-signal wards: the demand-driven optimum scores higher on it than "
            "the equity model while giving 3 robust-core wards zero relief. Per-ward floors -- not an aggregate "
            "fairness objective -- are what prevent starvation; judge fairness by starved wards / core min coverage."
        ),
        "sensitivity": sens.to_dict(orient="records"),
    }
    (OPT_DIR / "equity_optimizer_report.json").write_text(json.dumps(report, indent=2))

    print(inputs.to_string())
    print(f"\nBudget {budget:,} units = {BUDGET_SHARE:.0%} of need {inputs['need'].sum():,}")
    print("Signal audit:", json.dumps(signal_audit))
    print(f"Spearman(vulnerability, signals) = {rho_s:.3f}")
    print(units.join(cov.round(3)).to_string())
    print(metrics.to_string())
    print(frontier[["rho", "signal_weighted_coverage", "signal_coverage_loss_vs_rho0", "core_wards_min_coverage",
                    "n_starved_wards", "coverage_gini"]].round(3).to_string(index=False))
    print(sens.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
