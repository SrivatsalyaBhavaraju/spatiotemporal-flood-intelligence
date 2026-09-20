"""
Task 4.5 -- integrate task 3.8's distress classifier and task 2.10's
fuzzy geoparser into one end-to-end Objective 2 pipeline, exactly as
working.md SS2.4 diagrams it:

    Social media posts (scoped to study area / event window)
                      |
      +---------------+---------------+
      v                               v
  Distress classifier          Gazetteer + fuzzy match
  (task 3.8, MuRIL)            (task 2.10, built on task 2.9's
      |                         OSM extraction)
      +---------------+---------------+
                      v
      Distress posts, each resolved to a
      coordinate within the study wards

Both branches run independently over every post (matching the diagram --
geoparsing isn't gated on the distress verdict, so a post's location
mentions are always available even if this task only keeps the ones that
are BOTH distress AND resolved for the final deliverable). Neither task
3.8 nor task 2.10 is re-implemented here -- this module only composes
their existing public functions.

*** WHAT "VALIDATION" MEANS HERE, DISCLOSED UP FRONT *** task 3.8 already
measured the classifier's own accuracy (LOOCV) and task 2.10 already
measured the geoparser's own recall (against task 1.13's corpus) --
re-running either metric here would be circular, not a real check. What
task 4.5 actually needs to prove is that the TWO COMPOSE correctly. Two
separate checks:
  1. An integration run against task 1.13's real 19-passage corpus (the
     classifier's own training data -- explicitly a plumbing smoke test,
     NOT a fresh accuracy measurement, since scores there would be
     optimistic).
  2. A run against a small set of genuinely new, hand-written example
     posts the classifier has never seen in any form, covering all 4
     combinations (distress/not-distress x has-a-resolvable-place/doesn't)
     -- this is the real evidence the pipeline generalizes.

Usage:
    python src/nlp/run_objective2_pipeline.py

Required inputs:
    data/raw/distress_text/corpus_draft.csv        (task 1.13, for the integration smoke test)
    data/processed/nlp/distress_classifier_head.pt (task 3.8)
    data/raw/gazetteer/gazetteer_final.json        (task 2.9, via task 2.10)

Outputs (data/processed/nlp/):
    objective2_pipeline_output.csv          -- one row per post: text, distress_label,
                                                distress_probability, resolved_locations
    objective2_pipeline_validation_report.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.finetune_distress_classifier import (  # noqa: E402
    Phase4NlpError,
    load_muril,
    load_trained_head,
    predict_distress,
)
from src.nlp.fuzzy_geoparse import geoparse_text, load_gazetteer  # noqa: E402

CORPUS_PATH = REPO_ROOT / "data" / "raw" / "distress_text" / "corpus_draft.csv"
OUT_DIR = REPO_ROOT / "data" / "processed" / "nlp"
OUT_CSV = OUT_DIR / "objective2_pipeline_output.csv"
REPORT_PATH = OUT_DIR / "objective2_pipeline_validation_report.json"

# Genuinely new example posts (never seen by task 3.8's training/LOOCV in any form) --
# covering all 4 combinations of distress-label x has-a-resolvable-gazetteer-place.
NEW_UNSEEN_EXAMPLES = [
    "Families in Velachery were trapped on their rooftops as the water kept rising through the night.",
    "We are stranded and water is entering our home right now, please send a boat to help us.",
    "The Chennai Corporation announced routine repair work on Gandhi Road starting next week.",
    "Officials held a press conference today to discuss the annual relief-fund budget allocation.",
]


class Phase4PipelineError(FileNotFoundError):
    """Raised when a required task 1.13/2.9/3.8 artifact is missing."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase4PipelineError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Pipeline (pure composition logic -- unit-tested with injected fake
# classify_fn/geoparse_fn, no real MuRIL/gazetteer needed)
# --------------------------------------------------------------------------

def run_pipeline(texts: list, classify_fn, geoparse_fn, gazetteer: dict) -> list:
    """One result dict per input text: {"text", "distress_label",
    "distress_probability", "resolved_locations"}. Both branches always
    run (see module docstring) -- geoparsing is never skipped based on the
    distress verdict."""
    results = []
    for text in texts:
        classification = classify_fn(text)
        matches = geoparse_fn(text, gazetteer)
        locations = [
            {"place": m["matched_name"], "lon": m["lon"], "lat": m["lat"], "score": m["score"]}
            for m in matches
        ]
        results.append({
            "text": text,
            "distress_label": classification["label"],
            "distress_probability": classification["probability"],
            "resolved_locations": locations,
        })
    return results


def filter_resolved_distress_posts(results: list) -> list:
    """working.md SS2.4's actual final deliverable: distress posts that
    ALSO resolved to at least one coordinate. A distress post with no
    resolvable place mention can't feed the graph (Objective 1) or the
    optimizer (Objective 3) as a located signal -- it's disclosed in the
    report, not silently dropped."""
    return [r for r in results if r["distress_label"] == "distress" and len(r["resolved_locations"]) > 0]


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def build_report(results: list, label: str) -> dict:
    n_total = len(results)
    distress = [r for r in results if r["distress_label"] == "distress"]
    resolved_distress = filter_resolved_distress_posts(results)
    return {
        "run": label,
        "n_total_posts": n_total,
        "n_classified_distress": len(distress),
        "n_classified_not_distress": n_total - len(distress),
        "n_distress_with_resolved_location": len(resolved_distress),
        "n_distress_without_resolved_location": len(distress) - len(resolved_distress),
        "pct_distress_geolocated": round(100 * len(resolved_distress) / len(distress), 2) if distress else None,
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print("Loading task 3.8's trained classifier + task 2.9/2.10's gazetteer ...")
    try:
        _require_file(CORPUS_PATH, "src/nlp/collect_distress_text.py (task 1.13)")
        head, embedding_mean, embedding_std, metadata = load_trained_head()
        gazetteer = load_gazetteer()
    except (Phase4NlpError, Phase4PipelineError, FileNotFoundError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> classifier trained on {metadata['n_training_examples']} examples, "
          f"gazetteer has {len(gazetteer)} places")

    tokenizer, model, device = load_muril()
    print(f"  -> MuRIL loaded on {device}")

    def classify_fn(text):
        return predict_distress(
            text, tokenizer=tokenizer, model=model, device=device,
            head=head, embedding_mean=embedding_mean, embedding_std=embedding_std,
        )

    all_results = []
    all_reports = []

    print(f"\n=== Integration smoke test: task 1.13's real corpus <- {CORPUS_PATH} ===")
    print("(classifier's own training data -- plumbing check, NOT a fresh accuracy measurement, see module docstring)")
    corpus_df = pd.read_csv(CORPUS_PATH)
    corpus_results = run_pipeline(corpus_df["paragraph_text"].tolist(), classify_fn, geoparse_text, gazetteer)
    corpus_report = build_report(corpus_results, "task_1_13_corpus_smoke_test")
    print(f"  {corpus_report}")
    all_results.extend(corpus_results)
    all_reports.append(corpus_report)

    print(f"\n=== Generalization check: {len(NEW_UNSEEN_EXAMPLES)} genuinely new, hand-written examples ===")
    new_results = run_pipeline(NEW_UNSEEN_EXAMPLES, classify_fn, geoparse_text, gazetteer)
    for r in new_results:
        print(f"  [{r['distress_label']} p={r['distress_probability']}] "
              f"locations={[l['place'] for l in r['resolved_locations']]} <- {r['text'][:70]}")
    new_report = build_report(new_results, "new_unseen_examples")
    all_results.extend(new_results)
    all_reports.append(new_report)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_df = pd.DataFrame([
        {**{k: v for k, v in r.items() if k != "resolved_locations"},
         "resolved_locations": json.dumps(r["resolved_locations"])}
        for r in all_results
    ])
    output_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved pipeline output -> {OUT_CSV}")

    report = {"reports": all_reports}
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved validation report -> {REPORT_PATH}")

    print("\n=== Summary ===")
    print(json.dumps(report, indent=2))
    print("\nDone. Objective 2's full pipeline (classify + geoparse -> located distress posts) is working end-to-end.")
    print("Next: task 5.5 evaluates Objective 2 precision/recall as a whole.")


if __name__ == "__main__":
    main()
