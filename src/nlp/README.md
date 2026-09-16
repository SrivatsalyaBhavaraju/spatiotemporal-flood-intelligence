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
