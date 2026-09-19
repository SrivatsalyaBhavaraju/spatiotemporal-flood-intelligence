"""
Task 2.9 -- finalize the gazetteer (task 1.14's draft) into the
`name -> coordinate` lookup task 2.10's fuzzy matcher needs, once task
1.13's real distress text can actually be used to check it.

The draft (1.14) was built purely from OSM's own name tags. This task closes
the loop the other direction: does real text ABOUT this event mention place
names the OSM-only pull missed (colloquial micro-locality names, informal
usage) -- and if so, can those be resolved to a real coordinate at all?

Important nuance this script works around: task 1.13's `collect_distress_text.py`
only KEPT paragraphs that already matched a gazetteer name (`if matches:
rows.append(...)`), so its output corpus structurally cannot contain a
place name the gazetteer was missing -- anything not already in the
gazetteer would have been silently dropped before ever reaching
`corpus_draft.csv`. So this script goes back to the SOURCE pages and
re-fetches every paragraph (reusing `collect_distress_text.fetch_paragraphs`
directly, not duplicating that logic), not just the pre-filtered subset.

Method:
  1. Re-fetch every paragraph from task 1.13's source pages (not just the
     gazetteer-matching subset already saved in corpus_draft.csv).
  2. Extract capitalized 2+-word phrase candidates (a plain regex, not an
     NER model -- spaCy is listed in requirements.txt but isn't installed
     or used by any script yet, and pulling in a model download for this
     one task isn't worth it; see the "Why 2+ words, not 1" note below).
  3. Drop candidates whose name already exists in the draft gazetteer
     (exact, case-insensitive) -- nothing to add there.
  4. For everything left, check it against every named OSM feature in the
     study wards for an EXACT name match. This is the real filter that
     matters more than the regex: a bogus extraction ("The Prime
     Minister", a sentence-initial fragment, ...) simply won't match any
     OSM feature inside the 16 study wards and gets dropped here, not
     hand-curated away by a denylist. Fetched as ONE bulk Overpass query
     (`fetch_named_features_index()`), not one query per candidate -- an
     earlier version of this script queried Overpass once per candidate
     and found osmnx's Overpass rate limiter (`overpass_rate_limit=True`,
     which pauses between requests based on the shared public server's
     load) made each of 234 candidates' queries take well over a minute in
     practice. One bulk fetch pays that cost once; every candidate after
     that is a local dict lookup.
  5. Anything that DOES match becomes a new gazetteer row (type=
     "distress_text_candidate"), merged into the draft.

Why 2+ words, not 1: single capitalized words are extremely common at
sentence starts ("The...", "It...", "According...") -- task 1.13 already
hit this exact false-positive class for single-word gazetteer names
(`GENERIC_SINGLE_WORD_STOP`) and disclosed it rather than hiding it. This
script leans the other direction: it only considers 2+-word phrases,
disclosed here as a real, deliberate scope limit -- a genuine new
single-word colloquial place name (if any exists beyond what OSM's own
`place=` tags in the draft already cover) would be missed. Not solved here;
worth revisiting only if task 2.10's fuzzy-match precision turns out to
need it.

A real OSM name match isn't automatically a genuine flood-relevant place --
disclosed, not silently filtered: a real run (19 Sep 2026) resolved "World
Bank" and "Royal Enfield" to actual OSM features inside the study wards
(most plausibly a coincidentally-named local shop and a motorcycle
showroom respectively, not what those distress-text mentions meant --
almost certainly the international org and the motorcycle brand used for
flood rescue, per common reporting patterns). Kept in the output rather
than hand-excluded, on purpose: the whole point of using OSM as the truth
filter (see step 4 above) was avoiding a hand-curated semantic denylist,
and excluding these two on my own judgment would just be that denylist by
another name. `resolved_examples` in the report is short (5 entries this
run) specifically so a human can spot-check it before task 2.10 leans on
it, the same way task 1.14's own draft was "spot-checked and correct"
rather than trusted blindly.

Batching (NAME_BATCH_SIZE): a single query listing all 234 real candidate
names hit `413 Request Entity Too Large` on the one Overpass mirror that
accepted a connection during development -- the actual root cause
underneath what first looked like generic network flakiness. Candidates
are checked in batches of NAME_BATCH_SIZE instead of one giant request.

Graceful degradation, at the batch level, if OSM is unreachable: step 1
(re-fetching task 1.13's source pages) and candidate extraction/
classification always run and are saved regardless. Each batch that fails
on every OVERPASS_MIRRORS entry does NOT block the others -- the script
ships gazetteer_final.* built from the draft plus whatever batches DID
succeed (nothing silently added or dropped), and records candidates from
any failed batch as UNVERIFIED (not confirmed-absent) in
gazetteer_finalization_report.json's `novel_candidates_unverified` field.
This means task 2.10/3.3 are never blocked waiting on a flaky network, and
re-running this script later re-attempts only what's still unverified.

Usage:
    python src/nlp/finalize_gazetteer.py

Required inputs:
    data/raw/gazetteer/gazetteer_draft.csv  (task 1.14)
    data/raw/wards/study_wards.geojson       (task 1.3)
    Network access to task 1.13's source pages (required) + OSM Overpass
    (best-effort -- see "Graceful degradation" above)

Outputs (data/raw/gazetteer/):
    gazetteer_final.csv               -- draft rows + any resolved new candidates
    gazetteer_final.json              -- flat {name: [lon, lat]} dict, the
                                          literal task 2.9 deliverable for
                                          task 2.10's fuzzy matcher
    gazetteer_finalization_report.json -- candidate counts, resolved/dropped/
                                          unverified examples
"""
import json
import re
import sys
import time
from pathlib import Path

import geopandas as gpd
import osmnx as ox
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nlp.build_gazetteer import load_aoi_and_wards, tag_ward  # noqa: E402
from src.nlp.collect_distress_text import HEADERS, SOURCES, fetch_paragraphs  # noqa: E402

# One bulk query, not one per candidate (see fetch_named_features_index()).
# overpass_rate_limit=True makes osmnx do a separate pre-flight GET to the
# server's /status endpoint before EVERY request, including this one -- on
# this network that status check proved to be its own independent failure
# point (timing out even when the main /api/interpreter endpoint was
# reachable), so it's disabled; a single one-shot bulk query doesn't need
# the pacing that setting exists for anyway (that matters for many
# sequential requests, which this script no longer makes).
ox.settings.requests_timeout = 60
ox.settings.overpass_rate_limit = False

GAZETTEER_DRAFT = REPO_ROOT / "data" / "raw" / "gazetteer" / "gazetteer_draft.csv"
OUT_DIR = REPO_ROOT / "data" / "raw" / "gazetteer"
OUT_CSV = OUT_DIR / "gazetteer_final.csv"
OUT_JSON = OUT_DIR / "gazetteer_final.json"
REPORT_PATH = OUT_DIR / "gazetteer_finalization_report.json"

# Sentence-initial / connector capitalized words that would otherwise lead a
# false 2-word match (e.g. "The Adyar River" -> strip "The" -> "Adyar River",
# a genuine 2-word candidate; "According Officials" -> strip "According" ->
# "Officials", now 1 word, correctly dropped by MIN_CANDIDATE_WORDS below).
LEADING_STOPWORDS = {
    "the", "this", "that", "these", "those", "it", "in", "on", "at", "by",
    "for", "with", "from", "after", "before", "during", "while", "when",
    "according", "several", "many", "most", "a", "an", "as", "but", "and",
}
MIN_CANDIDATE_WORDS = 2  # see module docstring's "Why 2+ words" note
CAPITALIZED_PHRASE_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}\b")


class Phase2GazetteerError(FileNotFoundError):
    """Raised when a required task 1.3/1.14 artifact is missing. Matches
    the *ArtifactError convention used across src/graph, src/ground_truth."""


def _require_file(path: Path, produced_by: str) -> Path:
    if not path.exists():
        raise Phase2GazetteerError(
            f"Required artifact not found: {path}\n"
            f"This file is produced by {produced_by}. Run it first."
        )
    return path


# --------------------------------------------------------------------------
# Candidate extraction (pure text logic -- unit-tested)
# --------------------------------------------------------------------------

def extract_candidate_phrases(text: str) -> list[str]:
    """2+-word capitalized phrase candidates from `text`, with a leading
    stopword stripped (not the whole phrase dropped) if present. See module
    docstring's "Why 2+ words" note for the single-word scope limit.
    """
    candidates = []
    for match in CAPITALIZED_PHRASE_RE.finditer(text):
        words = match.group(0).split()
        while words and words[0].lower() in LEADING_STOPWORDS:
            words = words[1:]
        if len(words) >= MIN_CANDIDATE_WORDS:
            candidates.append(" ".join(words))
    return candidates


def classify_candidates(candidates: list[str], existing_names: set) -> tuple:
    """Split `candidates` into (novel, covered) against `existing_names`
    (exact, case-insensitive). Returns (novel_unique_sorted, covered_count).
    """
    existing_lower = {n.lower() for n in existing_names}
    seen, novel = set(), []
    covered = 0
    for c in candidates:
        key = c.lower()
        if key in existing_lower:
            covered += 1
            continue
        if key not in seen:
            seen.add(key)
            novel.append(c)
    return sorted(novel), covered


# --------------------------------------------------------------------------
# OSM resolution (network-dependent -- validated by a real run, not mocked)
# --------------------------------------------------------------------------

# Public Overpass mirrors to try, in order. `overpass.kumi.systems` first:
# during development it was the only one of the three that ever actually
# accepted a TCP connection on this network (the other two consistently
# hit connect-level timeouts, not slow responses -- see developing.md's
# task 2.9 notes) -- trying the two broken ones first would waste up to two
# minutes per batch before ever reaching the one that works.
OVERPASS_MIRRORS = [
    "https://overpass.kumi.systems/api",
    "https://overpass-api.de/api",
    "https://lz4.overpass-api.de/api",
]
FETCH_RETRIES_PER_MIRROR = 2

# A single query listing all 234 real candidate names hit `413 Request
# Entity Too Large` on the one mirror that accepted a connection during
# development -- that was the actual root cause underneath the earlier
# connectivity trouble, not more flakiness. Batching avoids it.
NAME_BATCH_SIZE = 25


def fetch_named_features_index(aoi_polygon, candidate_names: list) -> tuple:
    """Query OSM (Overpass) for features within `aoi_polygon` named one of
    `candidate_names`, batched (see NAME_BATCH_SIZE). Returns
    (index, unresolved_names): `index` is {name.lower(): (lon, lat)} built
    from every batch that was successfully queried (whether or not each
    individual name matched a feature); `unresolved_names` lists candidates
    from batches that failed on every mirror/attempt -- those were never
    actually checked against OSM, so the caller must not report them as
    "confirmed no match" (see main()'s handling of this).
    """
    index: dict = {}
    unresolved_names: list = []
    batches = [candidate_names[i:i + NAME_BATCH_SIZE] for i in range(0, len(candidate_names), NAME_BATCH_SIZE)]
    for batch_num, batch in enumerate(batches, 1):
        try:
            batch_index = _fetch_one_batch(aoi_polygon, batch)
            index.update(batch_index)
            print(f"    batch {batch_num}/{len(batches)} ({len(batch)} names) OK -> {len(batch_index)} matched")
        except Phase2GazetteerError as e:
            unresolved_names.extend(batch)
            print(f"    batch {batch_num}/{len(batches)} ({len(batch)} names) FAILED on every mirror: {e}")
    return index, unresolved_names


def _fetch_one_batch(aoi_polygon, batch_names: list) -> dict:
    last_error = None
    for mirror in OVERPASS_MIRRORS:
        ox.settings.overpass_url = mirror
        for attempt in range(FETCH_RETRIES_PER_MIRROR):
            try:
                gdf = ox.features_from_polygon(aoi_polygon, tags={"name": batch_names})
                return _build_named_index(gdf)
            except Exception as e:
                last_error = e
    raise Phase2GazetteerError(f"all mirrors failed; last error: {last_error}")


def _build_named_index(gdf) -> dict:
    index = {}
    for name, geom in zip(gdf.get("name"), gdf.geometry):
        if not isinstance(name, str) or geom is None:
            continue
        key = name.strip().lower()
        if key and key not in index:
            pt = geom.representative_point()
            index[key] = (pt.x, pt.y)
    return index


def geocode_candidate(name: str, named_features_index: dict):
    """Look up `name` (exact, case-insensitive) in a pre-fetched named-
    features index (see fetch_named_features_index()). Returns (lon, lat)
    or None -- this is the real truth filter for candidates (see module
    docstring), not the regex. Most candidates are expected to return None
    (person names, government bodies, etc. picked up by the permissive
    regex extractor) -- that's correct, not a failure.
    """
    return named_features_index.get(name.strip().lower())


def fetch_all_paragraphs(sources: list = SOURCES) -> list[dict]:
    """Every paragraph (not just gazetteer-matching ones) from task 1.13's
    source pages -- reuses that module's own fetch_paragraphs()."""
    rows = []
    for url, label in sources:
        print(f"  fetching [{label}] {url}")
        try:
            paragraphs = fetch_paragraphs(url)
        except Exception as e:
            print(f"    FAILED: {e}")
            continue
        rows.extend({"source_url": url, "source_label": label, "paragraph_text": p} for p in paragraphs)
        time.sleep(1)  # polite, matching task 1.13's convention
    return rows


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------

def build_new_rows(resolved: list[dict], wards: gpd.GeoDataFrame) -> pd.DataFrame:
    """resolved: [{"name": str, "lon": float, "lat": float}, ...] ->
    a gazetteer-schema DataFrame (type="distress_text_candidate"), ward-
    tagged via task 1.14's own tag_ward()."""
    if not resolved:
        return pd.DataFrame(columns=["name", "name_ta", "type", "subtype", "ward_no", "lon", "lat"])
    gdf = gpd.GeoDataFrame(
        resolved, geometry=gpd.points_from_xy([r["lon"] for r in resolved], [r["lat"] for r in resolved]), crs="EPSG:4326"
    )
    gdf = tag_ward(gdf, wards)
    return pd.DataFrame({
        "name": gdf["name"],
        "name_ta": None,
        "type": "distress_text_candidate",
        "subtype": None,
        "ward_no": gdf["ward_no"],
        "lon": gdf["lon"],
        "lat": gdf["lat"],
    })


def to_name_coordinate_dict(gaz: pd.DataFrame) -> dict:
    """Flatten the gazetteer table to {name: [lon, lat]} -- the literal
    task 2.9 deliverable. A name can legitimately appear multiple times in
    the table (e.g. a road spans several wards as distinct segments); this
    picks one representative coordinate per unique name, preferring a
    locality/landmark/waterway row (a more specific "place") over a plain
    road segment when both exist, else the first occurrence. Approximate by
    design -- task 2.10's fuzzy matcher already works at ward-level
    precision, not exact-point precision.
    """
    type_priority = {"locality": 0, "landmark": 1, "distress_text_candidate": 1, "waterway": 2, "road": 3}
    ranked = gaz.assign(_rank=gaz["type"].map(type_priority).fillna(9)).sort_values("_rank")
    best = ranked.drop_duplicates(subset="name", keep="first")
    return {row["name"]: [round(float(row["lon"]), 6), round(float(row["lat"]), 6)] for _, row in best.iterrows()}


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    print(f"Loading task 1.14 draft gazetteer <- {GAZETTEER_DRAFT}")
    try:
        _require_file(GAZETTEER_DRAFT, "src/nlp/build_gazetteer.py (task 1.14)")
    except Phase2GazetteerError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        raise SystemExit(1)
    draft = pd.read_csv(GAZETTEER_DRAFT)
    existing_names = set(draft["name"].dropna().astype(str))
    print(f"  -> {len(draft)} rows, {len(existing_names)} unique names")

    aoi, wards = load_aoi_and_wards()

    print(f"\nRe-fetching ALL paragraphs from task 1.13's {len(SOURCES)} sources (not just gazetteer-matching ones) ...")
    paragraphs = fetch_all_paragraphs()
    print(f"  -> {len(paragraphs)} paragraphs total")

    print("\nExtracting 2+-word capitalized phrase candidates ...")
    all_candidates = []
    for row in paragraphs:
        all_candidates.extend(extract_candidate_phrases(row["paragraph_text"]))
    novel, covered_count = classify_candidates(all_candidates, existing_names)
    print(f"  -> {len(all_candidates)} raw candidate mentions, {covered_count} already covered by the draft gazetteer")
    print(f"  -> {len(novel)} unique NOVEL candidates to check against OSM: {novel[:10]}{'...' if len(novel) > 10 else ''}")

    n_batches = (len(novel) + NAME_BATCH_SIZE - 1) // NAME_BATCH_SIZE
    print(f"\nFetching OSM features within the study wards named one of the {len(novel)} candidates "
          f"({n_batches} batch(es) of up to {NAME_BATCH_SIZE}, avoiding the 413 a single big query hit) ...")
    named_index, unresolved = fetch_named_features_index(aoi, novel)
    checked = [n for n in novel if n not in unresolved]
    print(f"  -> {len(named_index)} uniquely-named features indexed; {len(checked)}/{len(novel)} candidates actually checked")

    resolved, failed = [], []
    print(f"\nResolving the {len(checked)} checked candidates against that index (local lookups) ...")
    for i, name in enumerate(checked, 1):
        coord = geocode_candidate(name, named_index)
        if coord:
            resolved.append({"name": name, "lon": coord[0], "lat": coord[1]})
            print(f"  [{i}/{len(checked)}] RESOLVED  {name} -> {coord}")
        else:
            failed.append(name)
    print(f"\n{len(resolved)} candidate(s) resolved to a real OSM feature, {len(failed)} confirmed no match in study wards")
    osm_resolution_available = len(checked) > 0
    if unresolved:
        print(f"WARNING: {len(unresolved)} candidate(s) could not be checked (every mirror failed for their batch) "
              f"-- see the report's `novel_candidates_unverified` list.", file=sys.stderr)

    new_rows = build_new_rows(resolved, wards)
    final_gaz = pd.concat([draft, new_rows], ignore_index=True)
    final_gaz = final_gaz.drop_duplicates(subset=["name", "type", "ward_no"]).sort_values(["type", "name"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_gaz.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved final gazetteer table -> {OUT_CSV} ({len(final_gaz)} rows)")

    name_coord_dict = to_name_coordinate_dict(final_gaz)
    OUT_JSON.write_text(json.dumps(name_coord_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved name->coordinate dict -> {OUT_JSON} ({len(name_coord_dict)} unique names)")

    report = {
        "draft_gazetteer_rows": int(len(draft)),
        "paragraphs_rescanned": len(paragraphs),
        "raw_candidate_mentions": len(all_candidates),
        "already_covered_mentions": covered_count,
        "novel_unique_candidates": len(novel),
        "osm_resolution_available": osm_resolution_available,
        "candidates_actually_checked": len(checked),
        "resolved_to_osm_feature": len(resolved),
        "resolved_examples": resolved[:20],
        "dropped_no_osm_match": len(failed),
        "dropped_examples": failed[:20],
        "novel_candidates_unverified": unresolved,
        "final_gazetteer_rows": int(len(final_gaz)),
        "final_unique_names": len(name_coord_dict),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved finalization report -> {REPORT_PATH}")

    print("\n=== Summary ===")
    print(json.dumps(report, indent=2, default=str))
    print("\nDone. Next: task 2.10 (fuzzy matching) looks up distress-text place mentions")
    print(f"  against {OUT_JSON.name}'s name->coordinate dict.")


if __name__ == "__main__":
    main()
