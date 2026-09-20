# P4 — NLP (gazetteer, fuzzy geoparsing, distress classifier)

## `build_gazetteer.py` — task 1.14

Builds a draft lookup table of names Objective 2's geoparser will need to
resolve out of distress text: roads, waterways, localities, and landmarks
(hospitals, bus/railway stations, colleges, schools, places of worship)
within the 16 study wards. Task 2.10's fuzzy matcher looks a place-name
mention up against this table to get a ward/coordinate — no gazetteer, no
way to turn "flooding near Velachery bus depot" into something the graph
can use.

Roads/waterways reuse the `name` column already in `data/raw/osm/` (tasks
1.1/1.2); localities and landmarks are a new OSM Overpass pull
(`place=neighbourhood/suburb/quarter/locality/village` and
`amenity=hospital/bus_station/college/school/place_of_worship` +
`railway=station`). Each row also carries OSM's `name:ta` (Tamil name) when
present — this is what makes the gazetteer usable for Tamil-language
distress messages, not just English, which is the actual point of
Objective 2 being multilingual.

```bash
python src/nlp/build_gazetteer.py
```

Output: `data/raw/gazetteer/gazetteer_draft.csv` (gitignored — rerun to
regenerate). Columns: `name, name_ta, type, subtype, ward_no, lon, lat`.

**Result (01 Sep 2026):** 2,168 unique entries after de-duplication — 1,758
roads, 307 landmarks, 89 localities, 14 named waterways; 88 entries have a
Tamil name from OSM. Spot-checked and correct: real Chennai localities
(Adyar, Besant Nagar, Adambakkam...), landmarks with sensible subtypes
(ARK Hospital/hospital, Adhi Arunachaleshwar Temple/place_of_worship...).
120 rows (mostly boundary-adjacent road segments) didn't match a ward via
the spatial join — expected at polygon edges, not a data quality problem.

**Network note:** this machine sits behind a Sophos TLS-inspection proxy
that re-signs HTTPS certs with its own CA, which Python's bundled `certifi`
store doesn't trust by default — Overpass API calls failed with
`SSLCertVerificationError` until `pip install pip-system-certs` was run
(makes Python defer to the Windows system cert store, where the Sophos CA
is already trusted). If OSM/Overpass calls suddenly fail with a cert error
on a machine that previously worked, check for this before assuming the
data source is down.

**What this is, and isn't:** a *draft*. Task 2.9 finalizes it once task
1.13's real distress-message corpus shows which names people actually use
(colloquial names, abbreviations, misspellings) versus this OSM-only pull.

## `collect_distress_text.py` — task 1.13

**The original plan (live social-media collection) turned out not to be
feasible, and this was verified rather than assumed:** X/Twitter's
historical full-archive search is Enterprise-only now (~$42k/month contract
pricing, no research tier since the 2023 Academic API shutdown), and no
redistributable public dataset of 2015 Chennai flood tweets exists —
academic papers that analyzed some used a since-dead scraping tool
(`GetOldTweets`) and never published their data. Checked directly, not
assumed: `working.md` §2 had already flagged that *"a ready-made labeled
code-mixed disaster dataset for Indian languages does not exist."*

**Chosen alternative:** mine real, publicly accessible reporting about the
event — ReliefWeb situation reports, an academic retrospective, and news
coverage including resident interviews — filtered to passages that mention
a place from the task 1.14 gazetteer. ReliefWeb's own API now requires a
pre-approved `appname` (as of Nov 2025, checked live) — rather than wait on
approval, the script fetches the same report pages directly instead
(public web pages, not gated).

```bash
python src/nlp/collect_distress_text.py
```

Output: `data/raw/distress_text/corpus_draft.csv` (gitignored). Columns:
`source_url, source_label, paragraph_text, matched_locations`.

**Result (01 Sep 2026):** 19 location-mentioning passages from 7 sources —
11 from news (including first-person resident accounts: *"Tansi Nagar in
Velachery was severely inundated... families... had to be evacuated by
boat"*), 4 encyclopedia, 3 academic retrospective, 1 ReliefWeb sitrep. One
real precision bug caught and fixed along the way: naive substring matching
against gazetteer names produced false positives from landmarks with
generic single-word OSM names ("Temple", "Church", "HEALTH") matching
ordinary English usage — fixed with a word-boundary regex plus a small stop
list for exactly this class of name (see `GENERIC_SINGLE_WORD_STOP` in the
script). Verified the fix by hand against the output.

**What this is, and isn't:** REAL text about the real event, not synthetic
— but from professional reporting, not informal social media. That's a
more formal register than what Objective 2 ultimately targets, and it's a
deliberate, disclosed choice (the user picked this path explicitly over a
synthetic code-mixed alternative) — not hidden as equivalent to tweets.
Task 2.11 hand-labels a subset of this corpus (distress / not-distress);
if the register gap turns out to matter in practice, revisit then rather
than guessing now.

## `finalize_gazetteer.py` — task 2.9

Closes the loop `build_gazetteer.py`'s docstring promised: does real text
about this event (task 1.13) mention place names the OSM-only draft
missed? Re-fetches **every** paragraph from task 1.13's source pages (not
just the subset `corpus_draft.csv` kept — that file only contains
paragraphs that already matched a gazetteer name, so it structurally can't
surface a name the gazetteer was missing), extracts 2+-word capitalized
phrase candidates (a plain regex, no NER model), and checks the ones not
already in the draft against real OSM data for the study wards.

```bash
python src/nlp/finalize_gazetteer.py
```

**Result (19 Sep 2026):** 146 paragraphs re-scanned (vs. the 19 that had
already survived task 1.13's gazetteer-match filter), 397 raw candidate
mentions, 234 unique novel candidates (i.e. not already in the 1,787-name
draft) — real place names ("Anna Nagar", "Cooum River", "Meenambakkam
Airport", "Okkiyam Thoraipakkam") mixed with the expected noise from a
disaster-coverage corpus (person names, government bodies, generic
institutional phrases — "Prime Minister Modi", "World Bank", "Red Cross").

**OSM verification hit two real, disclosed problems, not code bugs, and
both are fixed:** (1) a single query listing all 234 candidate names hit
`413 Request Entity Too Large` on the one mirror that accepted a
connection — fixed by batching into groups of 25 (`NAME_BATCH_SIZE`); (2)
`overpass-api.de` and `lz4.overpass-api.de` both consistently hit TCP-
connect timeouts on this network throughout development, so
`overpass.kumi.systems` is tried first now.

**Result after batching (19 Sep 2026):** even the working mirror is
intermittently flaky — of 10 batches, 6 failed on every mirror/attempt and
4 succeeded (a real ~50/50 pattern across the run, not a one-off). Of the
109 candidates actually checked: **5 resolved** to a real OSM feature
("Chennai Corporation", "Gandhi Road", "Kuberan Nagar", "Royal Enfield",
"World Bank"), 104 confirmed no match. The remaining 125 candidates
(batches that failed every attempt) are recorded as unverified, not
confirmed-absent.

**A real OSM match isn't automatically a genuine flood-relevant place —
disclosed, not silently filtered:** "World Bank" and "Royal Enfield" almost
certainly matched a coincidentally-named local shop and a motorcycle
showroom respectively — the distress text likely meant the international
org and the rescue-vehicle brand, not those specific OSM features. Kept in
the output anyway rather than hand-excluded: the whole point of using OSM
as the truth filter was avoiding a hand-curated semantic denylist, and
picking off inconvenient matches after the fact would just be that
denylist under a different name. `resolved_examples` in the report is
short enough to spot-check by hand before task 2.10 leans on it.

**Ships a partial, honestly-labeled result, not blocked on the remaining
flakiness:** `gazetteer_final.csv`/`.json` include the 5 newly-resolved
rows; nothing fabricated for the 125 still-unverified candidates, and
nothing silently dropped. Re-running this script will only re-attempt what
`novel_candidates_unverified` still lists.

**Outputs** (`data/raw/gazetteer/` — committed, like the draft, not
gitignored; see task 1.13/1.14's own commit for why):

| File | Contents |
|---|---|
| `gazetteer_final.csv` | Draft rows + OSM-verified new candidates (5 this run) |
| `gazetteer_final.json` | Flat `{name: [lon, lat]}` dict — task 2.9's literal deliverable for task 2.10's fuzzy matcher |
| `gazetteer_finalization_report.json` | Candidate counts, resolved/dropped/**unverified** examples |

## `fuzzy_geoparse.py` — task 2.10

Objective 2's toponym-resolution step (working.md §2.2): given free text,
find place-name mentions and resolve each to a gazetteer coordinate —
sliding n-gram window + `rapidfuzz` fuzzy match + overlap resolution (one
text span, one place).

```bash
python src/nlp/fuzzy_geoparse.py
```

**Scorer choice, tested on real examples before picking it:** `fuzz.WRatio`
(rapidfuzz's own general-purpose default) rewards partial containment for
length-mismatched strings — great for typos, but it also makes a single
generic word spuriously match any longer name containing it: "road" vs
"Gandhi Road" scored 90.0, which would flood real text with false
positives given how many gazetteer entries are literally "`<name>` Road" or
"`<name>` Nagar" (~1,758 road entries). Plain `fuzz.ratio` rejects those
same false positives (53.3, 62.5) while still tolerating realistic typos
("tansi ngr" vs "tansi nagar" → 90.0) — used instead. `SCORE_CUTOFF=85`
was checked against a sample of ordinary English words (none scored above
85 against any of the 1,777 real gazetteer names), not picked blind.

**Result (19 Sep 2026), validated against task 1.13's exact-substring
corpus** (`validate_against_corpus()`): 31 known locations across 19
passages, **100% recovered** once 4 "misses" are correctly excluded — those
were never real gaps, just the exact-substring baseline double-counting
one mention twice (e.g. "the Adyar river" independently matches both
"Adyar" and "Adyar River" as substrings; this script's overlap resolution
correctly keeps only the more specific "Adyar River"). Plus **5 genuine
extra matches** beyond the baseline: typo/spacing variants ("Vijayanagar" →
"Vijaynagar", 95.2; "Ramnagar" → "Ram Nagar", 94.1; "storm water drains" →
"Stormwater Drain", 94.1) and one name task 2.9 added to the gazetteer
after task 1.13's corpus was originally built ("Kuberan Nagar", exact
100.0) — a direct demonstration of the fuzzy pipeline picking up real
improvements the rigid exact-match baseline can't.

**Outputs** (`data/raw/gazetteer/`, committed): `fuzzy_geoparse_validation_report.json`
— per-passage known/found/recovered/absorbed/real_misses/extra breakdown.

## `label_distress_dataset.py` — task 2.11

Interactive hand-labeling tool for task 1.13's real 19-passage corpus (the
"few hundred posts" in working.md's original description referred to the
infeasible live-social-media plan — see task 1.13's own note; this labels
what actually exists). A labeling TOOL, not a labeler: presents each
passage to a human one at a time and records their DISTRESS / NOT_DISTRESS
/ UNCERTAIN judgment — never assigns a label itself.

```bash
python src/nlp/label_distress_dataset.py                    # interactive session
python src/nlp/label_distress_dataset.py --report           # progress only, no prompts
```

Stable text-hash `passage_id`s (re-running task 1.13's collector can't
silently orphan a label) and a write-after-every-label design (nothing
lost on interruption). **All 19 real passages labeled (19 Sep 2026): 8
distress, 9 not_distress, 2 uncertain.** Labels spot-checked against the
underlying text before merging — firsthand/directly-reported acute impact
(evacuated by boat, water inside homes) marked distress; political
statements and institutional announcements marked not_distress even when
the surrounding article is flood-related.

Outputs (`data/raw/distress_text/`, committed): `labeled_distress_dataset.csv`,
`labeling_progress_report.json`.

## `finetune_distress_classifier.py` — task 3.8

Lightly fine-tunes a pretrained multilingual transformer (MuRIL,
working.md §2.3's own choice) into a distress/not-distress classifier on
task 2.11's hand-labeled data.

```bash
python src/nlp/finetune_distress_classifier.py
```

**Real data-size reality check:** working.md specifies "a few hundred
posts" — the real corpus has 19, 17 binary-labeled (2 "uncertain" excluded
— this task is binary per working.md). Fine-tunes what actually exists,
disclosed as such.

**Adaptation strategy — confirmed with the user (20 Sep 2026), no spec
exists in working.md at this sample size:** with n=17 vs. MuRIL-base's
237M parameters, full end-to-end fine-tuning would just memorize the
training set. Chosen instead: freeze the entire MuRIL body, extract a
768-dim attention-mask-weighted mean-pooled sentence embedding, train only
a linear classification head — standard "linear probing" practice for
tiny-data regimes, and the most defensible reading of working.md's
"lightly fine-tuned" at n=17.

**A real bug found and fixed, same class as task 4.1's GNN bug, not
hidden:** the first real run's leave-one-out CV (LOOCV, k=n=17 — the only
honest evaluation at this sample size) came back completely degenerate:
F1=0.0, recall=0.0, predicting "not distress" for every held-out example.
Diagnosed rather than accepted: full-data train accuracy was only 53%
with every predicted probability clustered at ~0.47 regardless of label —
the head barely moved off its initial output. Raw MuRIL embeddings have
tiny per-dimension scale (std ~0.023), starving gradient flow exactly like
task 4.1's unnormalized GNN features did. Ruled out "the embeddings just
aren't separable" first: 1-NN classification using raw cosine similarity
on the SAME embeddings got 88% LOOCV accuracy, so the signal was there —
the raw-scale linear head just couldn't reach it. Fixed with per-fold,
train-only z-score standardization (never leaking the held-out example's
own statistics into its own prediction, same discipline as task 4.1's
`compute_feature_stats()`), which took LOOCV F1 from 0.0 to **0.9412**.

**Real result (20 Sep 2026):** LOOCV — precision 0.889, recall 1.0 (catches
every real distress example), F1 **0.9412**, accuracy 94.12% (1 false
positive out of 17 folds). **Heavily caveated, not oversold:** with n=17,
each fold's error swings the metric by ~5.9% — this is a rough estimate,
not a production-grade confidence number. Verified end-to-end on genuinely
new text after training the final head: correctly classified an unseen
first-person distress account (p=0.998) and an unseen institutional
announcement (p=0.0002) it had never encountered in any form.

**Disclosed, inherited limitation (not new here):** MuRIL's core strength
is code-mixed Indian-language text; the real corpus is English news/report
register (task 1.13's infeasibility pivot). This proves the pipeline
architecture task 4.5 needs, not that MuRIL is the ideal model for this
specific text.

`predict_distress(text, ...)` is the reusable inference entry point task
4.5 imports directly.

Outputs (`data/processed/nlp/`, gitignored): `distress_classifier_head.pt`
(head weights + standardization stats), `distress_classifier_metadata.json`,
`distress_classifier_loocv_report.json`.

## `run_objective2_pipeline.py` — task 4.5

Composes task 3.8's classifier and task 2.10's geoparser into working.md
§2.4's full Objective 2 pipeline, exactly as diagrammed: both branches run
independently on every post (classification isn't gated on geoparsing or
vice versa), then merge into the final deliverable — distress posts that
also resolved to a coordinate.

```bash
python src/nlp/run_objective2_pipeline.py
```

**What "validation" means here, disclosed up front:** task 3.8 already
measured the classifier's own accuracy (LOOCV) and task 2.10 already
measured the geoparser's own recall — re-measuring either here would be
circular. What this task actually needs to prove is that the two **compose
correctly**, checked two ways: (1) an integration smoke test against task
1.13's real corpus (the classifier's own training data — plumbing check
only, not a fresh accuracy number), and (2) a generalization check against
4 genuinely new, hand-written posts covering all 4 combinations of
distress-label × has-a-resolvable-place.

**Real result (20 Sep 2026):** the generalization check landed exactly on
all 4 combinations, correctly: a real-sounding distress post ("Families in
Velachery were trapped on their rooftops...") → distress (p=0.998),
resolved to Velachery; a distress post with no place name ("We are
stranded...") → distress (p=0.876), correctly resolved to **no**
location — a real, disclosed gap (an unlocatable distress signal can't
feed the graph/optimizer), not silently hidden; an institutional
announcement mentioning two real places ("Chennai Corporation announced
routine repair work on Gandhi Road...") → not_distress (p=0.007), but
still geoparsed to both places — proving the two branches genuinely run
independently, not gated on each other; and a generic institutional
statement → not_distress, no locations. On the 1.13 corpus smoke test,
10/19 real passages classified distress, and **100% of those resolved to
a location** (expected — task 1.13's own corpus was already filtered to
location-mentioning passages).

Outputs (`data/processed/nlp/`, gitignored): `objective2_pipeline_output.csv`,
`objective2_pipeline_validation_report.json`.

## `evaluate_objective2_precision_recall.py` — task 5.5

Consolidates what tasks 2.10/3.8 already measured piecemeal into one
Objective 2-level report, and fills a real gap neither of them checked:
geoparsing **precision** (2.10 only ever measured recall against a
known-substring baseline — never "of everything matched, how much was
actually correct?").

```bash
python src/nlp/evaluate_objective2_precision_recall.py
```

**A real false positive found while building this, not hidden:** ran the
geoparser over every real passage (not just the known-substring subset
2.10 checked) and read all 37 real matches in context. 36 are correct.
One is not: **"Nandambakkam"** (a real, distinct Chennai locality,
confirmed from its source sentence — listed alongside Guindy/Adyar/Porur/
Meenambakkam, all different real areas) is **absent from the gazetteer**,
so `fuzz.ratio` fuzzy-matched it to **"Adambakkam"** — a different real
place that IS present — purely on 90.9% string similarity, clearing
`SCORE_CUTOFF=85`. A genuine precision failure: an absent place silently
resolves to the WRONG coordinate instead of being flagged unknown.
Disclosed as `KNOWN_FALSE_POSITIVES`, not excluded (same precedent as
task 2.9's "World Bank"/"Royal Enfield"). One more case ("storm water
drains" → the specific OSM feature "Stormwater Drain") reviewed and
judged borderline — geographically fine (right neighborhood) but
conceptually a generic phrase over-matched to one specific named entity —
counted as correct for the headline number, flagged in the report.

**Real result (20 Sep 2026):**

| Metric | Value |
|---|---|
| Classification precision/recall (task 3.8 LOOCV) | 0.889 / 1.0 (F1=0.9412) |
| Geoparsing recall (task 2.10) | 1.0 (27/31 known + 5 extra) |
| Geoparsing precision (new) | **0.973** (36/37) |
| End-to-end pipeline precision/recall | 0.889 / 1.0 |

**The end-to-end number is identical to the classifier's own number —
disclosed why, not presented as a free lunch:** joining task 3.8's honest
LOOCV predictions (never fit on the held-out example) with geoparsing on
the same 17 passages gives EXACTLY the classifier-only precision/recall.
Reason: task 1.13's own collection method only kept passages that already
mention a gazetteer place, so every one of the 17 labeled examples has
≥1 resolvable location by construction — geoparsing can't be a
bottleneck on *this* corpus. Not evidence geoparsing failure never
matters — task 4.5's own smoke test already found a real case ("We are
stranded...") where a genuine distress post has no resolvable location.

Outputs (`data/processed/nlp/`, gitignored): `objective2_precision_recall_report.json`.
