# Results Summary

**Task 6.4.** A standalone, polished summary of what this project actually found — distinct from `developing.md` (the day-to-day task tracker, 30+ dated log entries) and the eventual full final report (task 8.1, with methodology/literature review/etc.). Every number below is pulled directly from committed, regenerable reports, not transcribed from memory — see "Where to look for more" at the end for exact file paths.

Live, interactive version of most of this: `streamlit run src/dashboard/app.py` (see `src/dashboard/README.md`).

---

## Objective 1 — Flood propagation: baseline vs. GNN

**The question (working.md §1.6):** does a spatiotemporal GNN capture flood propagation a static, rule-based baseline structurally cannot?

**Setup:** 17,195 road-segment line-graph, 16 study wards, 4 event phases (Pre-Event / Rising / Peak / Receding) anchored to the real Nov–Dec 2015 Chennai flood, 3 usable phase transitions. Ground truth fuses Sentinel-1 SAR, Bhuvan/NRSC, and news/advisory sources.

### Headline numbers

| Transition | Baseline F1 (test) | GNN F1 (test, tuned) | GNN AUC — fair holdout (val / test) |
|---|---|---|---|
| Pre-Event → Rising | 0.0 (trivial — both always dry) | 0.0 | n/a (single class) |
| **Rising → Peak** | **0.0** | 0.982 | **0.59** / 0.42 |
| Peak → Receding | 0.973 | 0.973 | **0.60** / 0.49 |

*Baseline overall: F1=0.624, accuracy=0.659 (`baseline_evaluation_report.json`). Test-split F1/AUC above from `baseline_vs_gnn_comparison.json`.*

### The real, measured finding

**The baseline completely fails to anticipate flood onset.** At `rising→peak`, it scores F1=0.0 on every split — a static, reactive rule has no mechanism to foresee a rainfall spike from a currently-dry state. This is the measured version of working.md's core claim, not an assumption: it's a genuine structural failure, confirmed on real data.

**The GNN's reported F1=0.982 win is real arithmetic, but not proof of learned skill on its own.** F1 at one threshold can look excellent purely from matching a transition's base rate. Digging past the headline number (task 5.1) found the tuned model predicts "flooded" for 100% of test segments on *every* transition — a threshold/generalization artifact, not nuanced prediction. AUC-ROC (threshold-independent) tells the real story:

- **Task 4.1's original, properly held-out model** (trained on the *train* wards only — val and test both genuinely unseen) scores AUC **0.59–0.60 on val** (real, if modest, above-chance discrimination) but collapses to **0.42–0.49 on test** (indistinguishable from random).

### Why — root-caused, not hand-waved (task 5.4)

Checked the actual per-ward relationship between elevation and the flood label across all 16 wards. **Only 1 of 16 wards (ward 170) has genuine within-ward flood/no-flood variance** — every other ward, including *both* test wards, is >90% one-sided. Task 3.4/4.4's ground-truth fusion (dominated by news' broad ward-level fallback — only 5.4% of the flooded set is SAR-covered, the fine-grained source) pushes most wards toward near-total inundation, leaving almost no signal for *any* model to rank against outside that one ward. That one informative ward landed in **val, not test** — a coin flip with only 16 wards to split across.

**Honest verdict, not a simple yes/no:** the core claim is **weakly supported** where the ground truth actually has signal to test it (val), and **inconclusive — not negative** — on test, because the test wards themselves carry no exploitable signal for any model. The baseline's failure on `rising→peak`, by contrast, is a clean, unambiguous, fully-supported result.

*Recommendation for anyone extending this: stratify the train/val/test ward split by within-ward label variance, not just segment count, so an informative ward isn't left to chance (`sanity_check_comparison_anomalies_report.json`). Not implemented here — a new split-methodology decision, out of scope for a sanity check.*

---

## Objective 2 — Distress classification + geoparsing

**The question (working.md §2.4):** can a lightly-adapted multilingual classifier plus gazetteer geoparsing turn free text into located distress signals?

**Reality check on scope:** working.md's original plan assumed "a few hundred" labeled social-media posts. Live X/Twitter historical search turned out to be Enterprise-only (~$42k/month) with no substitute dataset — verified, not assumed (task 1.13). The real corpus is **19 passages** from real public reporting (news, academic retrospective, ReliefWeb), 17 binary-labeled (2 "uncertain" excluded).

### Headline numbers

| Metric | Value | Source |
|---|---|---|
| Classification precision / recall / F1 | 0.889 / 1.0 / 0.941 | 17-example leave-one-out CV (task 3.8) |
| Geoparsing recall | 1.0 (27/31 known + 5 extra) | vs. task 1.13's real corpus (task 2.10) |
| Geoparsing precision | **0.973** (36/37) | reviewed every real match, not just recall (task 5.5) |
| End-to-end pipeline precision / recall | 0.889 / 1.0 | task 4.5 composition, honest LOOCV predictions |

### What was actually found, not just measured

**The classifier genuinely works on this tiny sample** — MuRIL (frozen body, linear probe head — full fine-tuning of 237M params on 17 examples would just memorize) correctly separates distress from non-distress text, verified on genuinely new, unseen sentences at inference time. Caveat stated plainly: with n=17, every fold's error swings the metric ~5.9% — a real but rough estimate.

**A real geoparsing error, found by reviewing every match, not assumed correct:** "Nandambakkam" — a real, distinct Chennai locality confirmed from its own source sentence — is absent from the gazetteer, so fuzzy matching resolves it to "Adambakkam," a *different* real place, purely on 90.9% string similarity. One false positive out of 37 real matches.

**The end-to-end number matching the classifier-only number is disclosed, not a free lunch:** task 1.13's own corpus was collected by filtering for passages that already mention a gazetteer place, so geoparsing can't fail as a bottleneck on this specific 17-example set. Task 4.5's own smoke test on genuinely new text found a real case where it does matter — a distress post ("we are stranded...") with no place name, correctly flagged as distress but correctly resolved to no location.

---

## Integration

Objective 1 and Objective 2 visibly become one system in `src/dashboard/`: an animated map toggles ground truth / baseline / GNN predictions across all 4 phases (elevation-staggered transitions — real per-segment elevation data, not a literal flow simulation, since no sub-phase timing data exists), real distress markers resolved from Objective 2's pipeline, interactive comparison charts, and the same disclosed limitations above rendered as readable insight cards — not hidden behind the polish.

---

## Limitations that shaped these results (consolidated)

- **Ground truth is coarse and phase-unresolved at the source.** None of SAR/Bhuvan/news individually resolve which *phase* a segment flooded in — task 3.4 had to make a judgment call (confirmed with the user) to assign flooding to peak+receding only. Task 4.4 later used a 4th, previously-unused Sentinel-1 pass to partially (not fully) differentiate the two.
- **Only 16 wards exist to split spatially** — every train/val/test-dependent result in this project carries real variance from *which* wards land where (tasks 3.7, 4.2, 5.4 all had to account for this directly).
- **One real event, ever.** There is no second Chennai-scale flood in this dataset to validate against — every finding here is internal-consistency-checked, not cross-event-validated.
- **The NLP corpus is real text, not real social media.** News/report register, not the informal, code-mixed text MuRIL's core strength targets — proves the pipeline architecture, not production readiness.
- **`A3TGCN(periods=1)`** tests whether spatial graph convolution captures propagation, not genuine multi-step temporal memory (this dataset — one event, 4 phases — can't support that regardless of architecture).

None of these were discovered late or hidden — each was found via a real, run-it-and-check diagnostic during the task that surfaced it, and disclosed in that task's own note.

---

## Where to look for more

| Want... | Look at |
|---|---|
| The full, dated, task-by-task build log | `developing.md` |
| Original technical rationale, data sources, methodology | `working.md` |
| Per-module details (data, bugs found, exact commands) | `src/*/README.md` |
| To see it running | `streamlit run src/dashboard/app.py` |
| Raw report JSON behind every number above | `data/processed/ground_truth/*.json`, `data/processed/nlp/*.json` (gitignored — regenerate via each task's script) |
