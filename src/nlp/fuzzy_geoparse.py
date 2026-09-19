"""
Task 2.10 -- fuzzy string matching pipeline (rapidfuzz) against the
gazetteer (task 2.9). Objective 2's toponym resolution step (working.md
SS2.2): given free text, find place-name mentions and resolve each one to
a coordinate.

Method -- sliding n-gram window + fuzzy match + overlap resolution, the
standard "gazetteer + fuzzy match, for bounded regions" approach working.md
names, not invented fresh here:
  1. Tokenize the text (word spans with character offsets).
  2. Slide a window of 1..MAX_NGRAM words over the tokens; for each window,
     find the gazetteer's best fuzzy match (rapidfuzz).
  3. Keep only matches scoring >= SCORE_CUTOFF.
  4. Resolve overlaps: a text span can only resolve to ONE place, so when
     multiple windows' matches overlap (very common -- see step 2's own
     windows), keep the highest-scoring one (longer, more specific window
     as the tie-break), drop the rest.
  5. Attach each surviving match's gazetteer coordinate.

Scorer choice -- tested and rejected the more "obvious" one first, on real
examples, before picking this one:
  - `fuzz.WRatio` (rapidfuzz's own general-purpose default) rewards partial
    containment for length-mismatched strings -- great for typo tolerance
    ("velacherry" vs "Velachery" -> 94.7), but that same behavior makes a
    single generic word spuriously match any longer name containing it:
    "road" vs "Gandhi Road" scores 90.0, "adyar river" vs "Adyar" scores
    90.0. Given how many gazetteer entries are literally "<name> Road" or
    "<name> Nagar" (~1,758 road entries, task 1.14), this would flood real
    text with false positives on ordinary words.
  - `fuzz.ratio` (plain length-sensitive Levenshtein ratio) rejects those
    same false positives ("road" vs "Gandhi Road" -> 53.3, "adyar river" vs
    "Adyar" -> 62.5) while still tolerating realistic typos ("tansi ngr" vs
    "tansi nagar" -> 90.0). Used here instead.
SCORE_CUTOFF=85 and MAX_NGRAM=4 were checked directly against a sample of
ordinary English words (this, and, flooding, water, help, residents, ...)
-- none scored above 85 against any of the 1,777 real gazetteer names --
and against real typo/short-form examples from task 1.13's corpus, not
picked blind.

Usage (as a library, imported by task 3.3 -- also runnable directly as a
validation check against real committed data):
    from src.nlp.fuzzy_geoparse import load_gazetteer, geoparse_text
    gazetteer = load_gazetteer()
    geoparse_text("Tansi Nagar in Velachery was severely inundated.", gazetteer)
    # -> [{"text": "Tansi Nagar", "start": 0, "end": 11, "matched_name": "Tansi Nagar",
    #      "score": 100.0, "lon": ..., "lat": ...}, {"text": "Velachery", ...}]

    python src/nlp/fuzzy_geoparse.py

Required inputs:
    data/raw/gazetteer/gazetteer_final.json  (task 2.9)

Outputs (data/raw/gazetteer/, when run as a script):
    fuzzy_geoparse_validation_report.json -- precision check against task
        1.13's exact-substring matches over the real distress-text corpus
        (see validate_against_corpus())
"""
import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz, process

REPO_ROOT = Path(__file__).resolve().parents[2]
GAZETTEER_JSON = REPO_ROOT / "data" / "raw" / "gazetteer" / "gazetteer_final.json"
CORPUS_CSV = REPO_ROOT / "data" / "raw" / "distress_text" / "corpus_draft.csv"
REPORT_PATH = REPO_ROOT / "data" / "raw" / "gazetteer" / "fuzzy_geoparse_validation_report.json"

MAX_NGRAM = 4          # see module docstring -- most gazetteer names are 1-3 words
SCORE_CUTOFF = 85.0    # see module docstring's scorer-choice note for how this was checked
TOKEN_RE = re.compile(r"[A-Za-z']+")


class Phase2GeoparseError(FileNotFoundError):
    """Raised when a required task 2.9 artifact is missing. Matches the
    *ArtifactError/*GazetteerError convention used across src/graph,
    src/ground_truth, src/nlp."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2GeoparseError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_gazetteer(path: Path = GAZETTEER_JSON) -> dict:
    """Load task 2.9's {name: [lon, lat]} dict into a lowercase-keyed
    lookup: {name.lower(): {"name": original_name, "lon": ..., "lat": ...}}.
    """
    _require_file(path, "src/nlp/finalize_gazetteer.py (task 2.9)")
    raw = json.loads(path.read_text(encoding="utf-8"))
    gazetteer = {}
    for name, (lon, lat) in raw.items():
        gazetteer[name.lower()] = {"name": name, "lon": lon, "lat": lat}
    return gazetteer


# --------------------------------------------------------------------------
# Tokenizing / windowing (pure text logic -- unit-tested)
# --------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """Word tokens with character offsets: [(word, start, end), ...]."""
    return [(m.group(0), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def generate_candidate_windows(tokens: list, text: str, max_ngram: int = MAX_NGRAM) -> list:
    """Every contiguous window of 1..max_ngram tokens, longest first (so
    overlap resolution's tie-break naturally favors more specific matches).
    Returns [{"text": original-case span, "start": int, "end": int, "n_words": int}, ...].
    """
    windows = []
    for n in range(min(max_ngram, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            start, end = tokens[i][1], tokens[i + n - 1][2]
            windows.append({"text": text[start:end], "start": start, "end": end, "n_words": n})
    return windows


# --------------------------------------------------------------------------
# Fuzzy matching + overlap resolution
# --------------------------------------------------------------------------

def fuzzy_match_windows(windows: list, gazetteer: dict, score_cutoff: float = SCORE_CUTOFF) -> list:
    """For each window, find the gazetteer's best fuzzy match (see module
    docstring for the `fuzz.ratio` scorer-choice reasoning); keep only
    windows scoring >= score_cutoff. Returns windows augmented with
    matched_name/score/lon/lat, still possibly overlapping (see
    resolve_overlaps()).
    """
    names = list(gazetteer.keys())
    matches = []
    for w in windows:
        result = process.extractOne(w["text"].lower(), names, scorer=fuzz.ratio, score_cutoff=score_cutoff)
        if result is None:
            continue
        matched_key, score, _ = result
        entry = gazetteer[matched_key]
        matches.append({**w, "matched_name": entry["name"], "score": score, "lon": entry["lon"], "lat": entry["lat"]})
    return matches


def resolve_overlaps(matches: list) -> list:
    """A text span can only resolve to one place. Sort by (score desc,
    n_words desc) and greedily keep non-overlapping matches -- this
    naturally prefers an exact short match ("Tansi Nagar", score 100) over
    a padded longer window that also happened to match ("Tansi Nagar in",
    score 88), and prefers a longer match over a shorter one when scores
    tie (e.g. "Adyar River" over just "Adyar" when both score 100).
    """
    ordered = sorted(matches, key=lambda m: (-m["score"], -m["n_words"]))
    accepted = []
    for m in ordered:
        if not any(m["start"] < a["end"] and a["start"] < m["end"] for a in accepted):
            accepted.append(m)
    return sorted(accepted, key=lambda m: m["start"])


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def geoparse_text(text: str, gazetteer: dict, max_ngram: int = MAX_NGRAM, score_cutoff: float = SCORE_CUTOFF) -> list:
    """Find place-name mentions in `text` and resolve each to a
    gazetteer coordinate. Returns a list of non-overlapping matches, each
    {"text", "start", "end", "matched_name", "score", "lon", "lat"},
    ordered by position in the text.
    """
    tokens = tokenize(text)
    windows = generate_candidate_windows(tokens, text, max_ngram)
    matches = fuzzy_match_windows(windows, gazetteer, score_cutoff)
    return resolve_overlaps(matches)


# --------------------------------------------------------------------------
# Validation against task 1.13's exact-substring corpus
# --------------------------------------------------------------------------

def validate_against_corpus(gazetteer: dict, corpus_path: Path = CORPUS_CSV) -> dict:
    """Run geoparse_text() over every passage in task 1.13's corpus and
    compare against its `matched_locations` column (built with an exact,
    word-boundary substring match against the SAME gazetteer's names --
    see collect_distress_text.py). This is a real, checkable baseline: does
    the fuzzy pipeline at least recover what exact matching already found,
    and does it find anything extra (typo/short-form tolerance)?
    """
    import pandas as pd

    _require_file(corpus_path, "src/nlp/collect_distress_text.py (task 1.13)")
    corpus = pd.read_csv(corpus_path)

    total_known, total_recovered, total_absorbed, total_extra = 0, 0, 0, 0
    per_passage = []
    for _, row in corpus.iterrows():
        known = {n.strip() for n in str(row["matched_locations"]).split(";") if n.strip()}
        found = geoparse_text(row["paragraph_text"], gazetteer)
        found_names = {m["matched_name"] for m in found}

        recovered = known & found_names
        missed = known - found_names
        # A "miss" isn't always a real gap: task 1.13's baseline checks each
        # gazetteer name independently, so a single mention of "Adyar river"
        # counts as BOTH "Adyar" and "Adyar River" hits, while this script's
        # overlap resolution correctly keeps only the more specific one --
        # see module docstring. Don't count that as a miss.
        absorbed = {m for m in missed if _is_substring_of_any(m, found_names)}
        real_misses = missed - absorbed
        extra = found_names - known

        total_known += len(known)
        total_recovered += len(recovered)
        total_absorbed += len(absorbed)
        total_extra += len(extra)
        per_passage.append({
            "known": sorted(known), "found": sorted(found_names), "recovered": sorted(recovered),
            "absorbed_by_more_specific_match": sorted(absorbed), "real_misses": sorted(real_misses), "extra": sorted(extra),
        })

    denominator = total_known - total_absorbed
    return {
        "n_passages": int(len(corpus)),
        "total_known_locations": total_known,
        "total_recovered": total_recovered,
        "total_absorbed_by_more_specific_match": total_absorbed,
        "recall_excluding_absorbed": round(total_recovered / denominator, 3) if denominator else None,
        "total_extra_found": total_extra,
        "per_passage_examples": per_passage[:10],
    }


def _is_substring_of_any(name: str, candidates: set) -> bool:
    """True if `name` is a strict word-boundary substring of some other
    (different) name in `candidates` -- see validate_against_corpus()'s
    note on why this isn't counted as a real miss."""
    low = name.lower()
    return any(low != c.lower() and re.search(r"\b" + re.escape(low) + r"\b", c.lower()) for c in candidates)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 2.9 gazetteer <- {GAZETTEER_JSON}")
    try:
        gazetteer = load_gazetteer()
    except Phase2GeoparseError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    print(f"  -> {len(gazetteer)} names")

    print(f"\nValidating against task 1.13's exact-substring corpus <- {CORPUS_CSV}")
    try:
        report = validate_against_corpus(gazetteer)
    except Phase2GeoparseError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)

    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved validation report -> {REPORT_PATH}")

    print("\n=== Summary ===")
    summary = {k: v for k, v in report.items() if k != "per_passage_examples"}
    print(json.dumps(summary, indent=2, default=str))

    print("\nDone. Next: task 3.3 (news cross-check) uses geoparse_text() to resolve")
    print("named roads/localities mentioned in distress/news text to coordinates.")


if __name__ == "__main__":
    main()
