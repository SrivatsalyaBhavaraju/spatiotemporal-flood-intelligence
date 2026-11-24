"""
Lightweight (no torch) helpers deciding which Objective 2 pipeline outputs
are REAL distress signals -- the single source of truth shared by task 5.5's
evaluation, task 7.2's optimizer, and the dashboard's Distress tab.

Why this exists (found during task 7.2, not assumed):
`data/processed/nlp/objective2_pipeline_output.csv` (task 4.5) holds both
task 1.13's real corpus AND task 4.5's 4 synthetic smoke-test sentences
(one is a located "Families in Velachery were trapped..." distress post).
Downstream consumers that read the file directly were silently treating the
synthetic sentences as real -- the dashboard plotted one as a real marker.
Separately, task 5.5 found one real geoparsing false positive that should
never be plotted or counted as a location.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = REPO_ROOT / "data" / "raw" / "distress_text" / "corpus_draft.csv"

# (lower-cased text span, gazetteer place it was wrongly resolved to) -> why.
KNOWN_FALSE_POSITIVES = {
    ("nandambakkam", "Adambakkam"): (
        "Nandambakkam is a real, distinct Chennai locality (confirmed from its source "
        "sentence, listed alongside Guindy/Adyar/Porur/Meenambakkam) that is absent from "
        "the gazetteer -- fuzz.ratio matched it to the different, present place 'Adambakkam' "
        "purely on 90.9% string similarity, clearing SCORE_CUTOFF=85."
    ),
}


def load_real_corpus_texts(path=CORPUS_PATH):
    """Stripped passage texts of task 1.13's real corpus."""
    return {str(t).strip() for t in pd.read_csv(path)["paragraph_text"]}


def is_real_corpus_text(text, corpus_texts):
    return str(text).strip() in corpus_texts


def is_known_false_positive(post_text, place, known=KNOWN_FALSE_POSITIVES):
    """True if `place` was resolved from a span task 5.5 found to be a geoparsing error.

    The pipeline output keeps only the resolved place, not the matched span,
    so this checks that the erroneous span appears in the post AND the place
    matches -- a genuine "Adambakkam" mention elsewhere is unaffected.
    """
    lowered = str(post_text).lower()
    return any(span in lowered and place == wrong_place for span, wrong_place in known)
