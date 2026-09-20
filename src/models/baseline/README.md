# Baseline rule-based propagation model (P1/P2)

## `rule_based_propagation.py` — task 3.6

The static, non-learned counterpart the GNN (task 4.1) has to beat
(working.md §1.6). Two rules, combined with OR, exactly as working.md's
own diagram specifies:

- **Rule A (direct rainfall trigger):** segment floods if the current
  phase's rainfall *intensity* exceeds `RAINFALL_INTENSITY_THRESHOLD_MM_PER_DAY=100`.
- **Rule B (neighbor cascading):** segment floods if a graph-adjacent
  neighbor is flooded in the current phase (real ground truth, task 3.4)
  AND its own elevation is at/below `ELEVATION_THRESHOLD_M=8.5` (the real
  median segment elevation).

```bash
python src/models/baseline/rule_based_propagation.py
```

**Why rainfall *intensity*, not the raw `rainfall_t` already in task 2.7's
`X_t`:** checked the real numbers first. `rainfall_t` is a raw phase-
window total, and developing.md's task 2.5 notes already flag that
pre_event's total (1055.7mm over a 20-day window) exceeds peak's (410.1mm
over 2 days) purely from window-length, not because pre_event was
wetter. Recomputed intensity from task 2.5's own `n_hours` column instead:
pre_event=50.3mm/day, rising=15.6, **peak=205.1** (the clear outlier, 4x
the next-highest), receding=10.5. `100mm/day` is IMD's own "Heavy"
rainfall boundary (Heavy: 64.5–119.5, Very Heavy: 119.6–244.4) — peak
falls solidly in "Very Heavy," everything else falls below "Heavy."

**Adjacency is undirected for Rule B**, unlike task 2.2/2.7's `edge_index`
(deliberately directed there, respecting OSM one-way traversal) —
floodwater doesn't respect traffic direction, so "neighbor" here means
graph-adjacent either way. A deliberate choice specific to this
hand-written physical rule, not a change to the GNN's own edge_index.

**Neighbor state comes from real ground truth (task 3.4), not the
baseline's own prior prediction** — matches working.md's "F1/accuracy PER
PHASE TRANSITION" framing and task 2.8's `transition_pairs()` design
(independent pairs, not a chained rollout), rather than compounding the
baseline's own errors across phases.

## Result (19 Sep 2026) — a genuinely revealing, not just passing, result

| Transition | F1 | Accuracy | Precision | Recall |
|---|---|---|---|---|
| pre_event → rising | 0.0 | 1.0 | 0.0 | 0.0 |
| rising → peak | **0.0** | **0.125** | 0.0 | 0.0 |
| peak → receding | 0.933 | 0.875 | 0.875 | 1.0 |

- **pre_event→rising** is a trivial no-op: both phases are dry by
  construction (task 3.4), so "predict nothing floods" is 100% accurate —
  precision/recall/F1 default to 0 (no positive class exists to score
  against, not a bug — see `binary_classification_metrics()`'s zero-
  division guards).
- **rising→peak is where the baseline completely fails** (F1=0, 12.5%
  accuracy) — and this is the actually informative result. At x_t=rising,
  rainfall intensity (15.6mm/day) is far below the trigger, and the
  ground truth going into Rule B is entirely dry (rising has no flooded
  segments), so BOTH rules stay silent and the baseline predicts
  all-zero. But 87.46% of segments really do flood by peak. A static,
  reactive rule structurally **cannot** anticipate a future rainfall
  spike from a currently-dry state — it has no mechanism to look ahead.
  This is the measured version of working.md §1.6's claim ("the learned
  model captures propagation patterns the static baseline misses"), not
  just an assertion of it.
- **peak→receding looks great (F1=0.933) but is a structurally easy
  case, not evidence of good propagation modeling:** at x_t=peak,
  rainfall intensity (205.1mm/day) blows past the threshold, so Rule A
  floods every segment — and because task 3.4's fusion assigns the exact
  same `ever_flooded` set to both peak and receding, "flood everything"
  trivially matches the receding target too. High recall (1.0) is
  real; the precision cost (0.875) is just the 2,157 segments that never
  flooded at all getting swept in by Rule A's blanket trigger.

**Overall: F1=0.636, accuracy=0.667** — a middling aggregate that hides
the real story (a trivial win, a genuine failure, and an easy win) more
than it reveals it. Task 4.1's GNN should be compared per-transition
against this table, not just against the aggregate number.

Outputs (`data/processed/ground_truth/`, gitignored): `baseline_predictions.csv`
(segment_id, phase pair, y_true, y_pred) and `baseline_evaluation_report.json`.

**Updated 20 Sep 2026 (task 4.4):** the numbers above predate task 4.4's
ground-truth fix (peak/receding no longer share an identical label set).
`peak→receding` F1 is now **0.919** (was 0.933) — still a structurally
easy case for the same reason (Rule A's rainfall trigger floods nearly
every segment at x_t=peak, regardless of elevation), just slightly less
so. `pre_event→rising` and `rising→peak` are unaffected. See task 4.4's
`detect_flood_recession.py` section above for the full story.

## `evaluate_per_transition.py` — task 5.2

Computes F1/accuracy per transition **per split** (train/val/test) against
the current `baseline_predictions.csv`, in the same report shape as task
5.1's GNN report, for task 5.3's direct comparison. Reuses task 3.7's
`evaluate_split()`, not reimplemented — this just formalizes into its own
dedicated, re-runnable artifact what was previously only a smoke-test side
effect buried inside `build_training_harness.py`, which had gone stale
(it was computed before task 4.4's ground-truth fix).

```bash
python src/models/baseline/evaluate_per_transition.py
```

**Real result (20 Sep 2026)** — reproduces task 3.7's own smoke-test
numbers exactly for `rising→peak` (train/val/test accuracy 6.99%/53.54%/
3.54%, confirming no drift there), plus the post-4.4 `peak→receding`
numbers:

| Transition | Train F1 | Val F1 | Test F1 |
|---|---|---|---|
| pre_event→rising | 0.0 | 0.0 | 0.0 |
| rising→peak | 0.0 | 0.0 | 0.0 |
| peak→receding | 0.950 | 0.604 | 0.973 |

**Worth reading alongside task 5.1's GNN finding, not in isolation:** on
`peak→receding`, the baseline predicts "flooded" for essentially 100% of
segments in every split too (Rule A's rainfall trigger fires uniformly at
x_t=peak, independent of elevation) — the SAME "matches the base rate by
predicting everyone positive" pattern task 5.1 found in the tuned GNN, not
a coincidence. Unlike the GNN, this is fully expected and disclosed for
the baseline (it's a deterministic rule, not something meant to show
learned per-segment discrimination) — the concerning finding is that the
GNN's equivalent numbers reflect the same shallow behavior, not genuine
propagation learning. Task 5.4 is where this gets sanity-checked properly.

Outputs (`data/processed/ground_truth/`, gitignored):
`baseline_final_per_transition_report.json`.
