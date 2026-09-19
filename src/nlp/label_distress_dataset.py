"""
Task 2.11 -- hand-label task 1.13's real-text corpus as distress /
not-distress.

*** THIS IS A HUMAN-JUDGMENT TASK. THIS SCRIPT IS THE LABELING TOOL, NOT A
LABELER. *** It presents each passage to a human annotator one at a time
and records their judgment -- it does not, and must not, assign labels
itself. Fabricating labels for a real disaster corpus would corrupt exactly
the ground truth this task exists to produce.

*** SIZE REALITY CHECK, DISCLOSED UP FRONT *** developing.md's task
description says "few hundred posts" -- that referred to the ORIGINAL plan
(live social-media collection), which task 1.13's own writeup documents as
infeasible (X/Twitter's archive search is Enterprise-only now; no
redistributable public dataset of 2015 Chennai flood tweets exists). What
actually exists to label is task 1.13's real alternative corpus:
data/raw/distress_text/corpus_draft.csv -- 19 real passages (news,
academic, encyclopedia, ReliefWeb) as of the 01 Sep 2026 run. This script
hand-labels whatever is actually in that file (`--report` at any time shows
the exact count) rather than assuming "a few hundred" exists.

CODEBOOK (for the human annotator):
  DISTRESS      -- the passage describes a person/people/household actually
                    experiencing acute flood impact -- stranded, trapped,
                    evacuated, water inside the home, needing rescue -- as a
                    firsthand or directly-reported account of the event
                    unfolding or its immediate aftermath. A retrospective
                    interview describing what happened TO THEM AT THE TIME
                    still counts (e.g. "the water crossed the first floor").
  NOT_DISTRESS  -- general reporting, statistics, political statements,
                    institutional/relief announcements, or commentary that
                    does not itself describe someone's acute impact -- even
                    if the surrounding article is about the flood.
  UNCERTAIN     -- genuinely ambiguous after reading; better to mark this
                    and add a note than force a guess.

Usage:
    python src/nlp/label_distress_dataset.py                    # interactive labeling session
    python src/nlp/label_distress_dataset.py --report           # print label counts/progress, no prompts
    python src/nlp/label_distress_dataset.py --relabel <passage_id>  # re-open one specific passage

Input:
    data/raw/distress_text/corpus_draft.csv          -- task 1.13

Outputs:
    data/raw/distress_text/labeled_distress_dataset.csv   -- passage_id, label, annotator_notes,
                                                              labeled_at. Written after EVERY single
                                                              label (not just at exit), so nothing is
                                                              lost if the session is interrupted.
    data/raw/distress_text/labeling_progress_report.json
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "distress_text" / "corpus_draft.csv"
OUT_DIR = REPO_ROOT / "data" / "raw" / "distress_text"
LABELED_PATH = OUT_DIR / "labeled_distress_dataset.csv"
REPORT_PATH = OUT_DIR / "labeling_progress_report.json"

VALID_CHOICES = {"d": "distress", "n": "not_distress", "u": "uncertain"}
ORIGINAL_SCOPE_TEXT = "a few hundred"  # developing.md's pre-1.13-pivot task description, for the honesty note


class Phase2LabelingError(FileNotFoundError):
    """Raised only for a genuinely missing/malformed required input file --
    never for 'not enough passages exist', which is a disclosed, expected
    reality (see module docstring), not an error condition."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2LabelingError(
            f"Required artifact not found: {path}\nThis file is produced by {produced_by}."
        )
    return path


def passage_id(paragraph_text: str) -> str:
    """Stable id derived from the passage's own text (not row index), so
    re-running task 1.13's collector -- which can reorder rows or add new
    ones -- never orphans an existing label as long as a given passage's
    text is unchanged.
    """
    return hashlib.sha1(paragraph_text.strip().encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_corpus(path: Path = CORPUS_PATH) -> pd.DataFrame:
    _require_file(path, "src/nlp/collect_distress_text.py (task 1.13)")
    df = pd.read_csv(path)
    required = {"source_url", "source_label", "paragraph_text", "matched_locations"}
    missing = required - set(df.columns)
    if missing:
        raise Phase2LabelingError(f"{path} is missing required column(s) {sorted(missing)}.")
    df = df.copy()
    df["passage_id"] = df["paragraph_text"].apply(passage_id)
    dup_mask = df["passage_id"].duplicated(keep=False)
    if dup_mask.any():
        dupe_ids = sorted(set(df.loc[dup_mask, "passage_id"]))
        print(f"  WARNING: {len(dupe_ids)} passage(s) have byte-identical text (same passage_id) "
              f"-- they'll share one label: {dupe_ids}")
    return df


def load_existing_labels(path: Path = LABELED_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["passage_id", "label", "annotator_notes", "labeled_at"])
    return pd.read_csv(path, dtype={"passage_id": str})


def save_labels(labels_df: pd.DataFrame, path: Path = LABELED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Interactive labeling
# --------------------------------------------------------------------------

def _upsert_label(labels_df: pd.DataFrame, new_row: dict) -> pd.DataFrame:
    without = labels_df[labels_df["passage_id"] != new_row["passage_id"]]
    return pd.concat([without, pd.DataFrame([new_row])], ignore_index=True)


def interactive_session(
    corpus_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    only_passage_id: str = None,
    input_fn=input,
    output_path: Path = LABELED_PATH,
) -> pd.DataFrame:
    """Runs the labeling loop. `input_fn` is injectable (defaults to the
    builtin `input`) purely so tests can drive it without a real terminal --
    production callers should never pass it.
    """
    already_labeled = set(labels_df["passage_id"])
    if only_passage_id is not None:
        to_label = corpus_df[corpus_df["passage_id"] == only_passage_id]
    else:
        to_label = corpus_df[~corpus_df["passage_id"].isin(already_labeled)]

    if len(to_label) == 0:
        msg = f"for passage_id {only_passage_id}" if only_passage_id else "-- every passage already has a label."
        print(f"Nothing to label {msg}")
        print("Run with --report to see the current label distribution, or --relabel <passage_id> to redo one.")
        return labels_df

    print(f"\n{len(to_label)} passage(s) to label. For each: [d]istress / [n]ot-distress / "
          f"[u]ncertain / [s]kip-for-now / [q]uit-and-save.\n")

    n_new = 0
    for _, row in to_label.iterrows():
        print("=" * 80)
        print(f"passage_id: {row.passage_id}   source: {row.source_label}   matched_locations: {row.matched_locations}")
        print(f"url: {row.source_url}")
        print("-" * 80)
        print(row.paragraph_text)
        print("-" * 80)
        while True:
            choice = input_fn("Label [d/n/u/s/q]: ").strip().lower()
            if choice in VALID_CHOICES:
                notes = input_fn("Notes (optional, Enter to skip): ").strip()
                new_row = {
                    "passage_id": row.passage_id,
                    "label": VALID_CHOICES[choice],
                    "annotator_notes": notes,
                    "labeled_at": datetime.now(timezone.utc).isoformat(),
                }
                labels_df = _upsert_label(labels_df, new_row)
                save_labels(labels_df, output_path)  # persist after EVERY label -- nothing lost on interruption
                n_new += 1
                print(f"  saved ({n_new} this session) -> {output_path}\n")
                break
            elif choice == "s":
                print("  skipped for now.\n")
                break
            elif choice == "q":
                print(f"\nQuitting. {n_new} new label(s) saved this session -> {output_path}")
                return labels_df
            else:
                print("  Please enter d, n, u, s, or q.")

    print(f"\nDone with this pass. {n_new} new label(s) saved this session -> {output_path}")
    return labels_df


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def build_report(corpus_df: pd.DataFrame, labels_df: pd.DataFrame) -> dict:
    merged = corpus_df.merge(labels_df[["passage_id", "label"]], on="passage_id", how="left")
    n_total = len(corpus_df)
    n_labeled = int(merged["label"].notna().sum())
    counts = merged["label"].value_counts(dropna=True).to_dict()
    return {
        "corpus_total_passages": int(n_total),
        "labeled_count": n_labeled,
        "unlabeled_count": int(n_total - n_labeled),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
        "original_scope_note": (
            f"developing.md's task 2.11 description says '{ORIGINAL_SCOPE_TEXT} posts' -- that "
            f"referred to the original live-social-media plan, which task 1.13 documents as "
            f"infeasible. The actual corpus (task 1.13's real-text alternative) has {n_total} "
            f"passages total; that's what's being labeled here, not a few hundred."
        ),
        "unlabeled_passage_ids": merged.loc[merged["label"].isna(), "passage_id"].tolist(),
    }


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="store_true", help="Print label counts/progress only, no prompts.")
    parser.add_argument("--relabel", metavar="PASSAGE_ID", help="Re-open one specific passage for (re)labeling.")
    args = parser.parse_args()

    print(f"Loading task 1.13 corpus <- {CORPUS_PATH}")
    try:
        corpus_df = load_corpus()
    except Phase2LabelingError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(corpus_df)} passages")

    labels_df = load_existing_labels()

    if args.report:
        report = build_report(corpus_df, labels_df)
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"\nreport saved -> {REPORT_PATH}")
        return

    if args.relabel:
        if args.relabel not in set(corpus_df["passage_id"]):
            print(f"ERROR: passage_id {args.relabel!r} not found in the corpus.", file=sys.stderr)
            raise SystemExit(1)
        labels_df = interactive_session(corpus_df, labels_df, only_passage_id=args.relabel)
    else:
        labels_df = interactive_session(corpus_df, labels_df)

    report = build_report(corpus_df, labels_df)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Progress ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
