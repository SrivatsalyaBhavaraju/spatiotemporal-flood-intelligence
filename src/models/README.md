# Model comparison (P1 + P3)

Scripts here compare across model families (`gnn/`, `baseline/`) rather
than belonging to either one. Per-model scripts live in their own
subdirectories — see `gnn/README.md` and `baseline/README.md`.

## `compare_baseline_vs_gnn.py` — task 5.3

Tests working.md §1.6's core claim ("the learned model captures
propagation patterns a static baseline misses") with plots/tables built
from task 5.1 (GNN) and 5.2 (baseline)'s per-transition/per-split reports.

```bash
python src/models/compare_baseline_vs_gnn.py
```

**A methodological precision correction, found while building this, not
hidden:** task 5.1's "final tuned" model (task 4.2) was retrained on
train+val *combined* — its "val" AUC/F1 is **not a fair holdout
measurement**, the model was fit to those exact segments. Citing it as
generalization evidence (as task 5.1's own note arguably implied) would
overstate the case. This script instead also computes AUC for task 4.1's
**original** model — trained on "train" only, so val *and* test are both
genuinely unseen — as the methodologically clean comparison:

| Model | train AUC | val AUC (fair) | test AUC (fair) |
|---|---|---|---|
| 4.1 original (train-only) | ~0.73–0.78 | **~0.59–0.60** | ~0.42–0.49 |
| 4.2 final tuned (train+val) | ~0.70–0.76 | ~0.70–0.72 (not fair — fit to it) | ~0.46–0.50 |

**Real result (20 Sep 2026) — the honest core-claim verdict, not a simple
yes/no:** the fairly-held-out val AUC (~0.59–0.60, both flood-relevant
transitions) is real, if weak, evidence the GNN generalizes — and it
lands specifically on ward 170, the one ward task 5.4 found has genuine
elevation-flood variance. Test is **inconclusive, not negative**: task
5.4 found zero informative wards there, so near-random AUC is expected
regardless of model quality. The baseline has no equivalent AUC at all —
`predict_transition()` (task 3.6) outputs hard 0/1 decisions, not a
ranked probability, so it structurally cannot express the graded
uncertainty the GNN can, even where post-threshold F1 looks similar
(`peak→receding`).

Outputs:
- `data/processed/ground_truth/baseline_vs_gnn_comparison.json` (gitignored)
  — consolidated comparison table, fair-holdout AUC, and the structured
  per-transition verdict.
- `docs/figures/phase5_f1_comparison.png` (committed) — baseline vs. tuned
  GNN F1, grouped by transition and split.
- `docs/figures/phase5_auc_fair_holdout.png` (committed) — task 4.1's
  train-only model's fair AUC per split, with a random-chance reference
  line, showing the train→val→test degradation directly.
