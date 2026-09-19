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
