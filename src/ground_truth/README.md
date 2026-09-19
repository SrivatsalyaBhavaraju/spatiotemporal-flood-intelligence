# P2 — Ground Truth & Data (satellite/rainfall processing, label fusion)

## `fetch_dem.py` — tasks 1.5 / 1.6

Downloads SRTM GL1 (30m) elevation for the study wards from OpenTopography's
public S3 mirror (two tiles, N12E080 + N13E080 — the AOI straddles the
tile boundary) and derives a slope raster from it via `numpy.gradient`
(avoids depending on the `gdaldem` CLI, which the README already flags as
finicky on Windows).

```bash
python src/ground_truth/fetch_dem.py
```

Outputs `data/raw/dem/srtm_dem.tif` and `srtm_slope.tif` (gitignored).
Result (01 Sep 2026): elevation 0–40m dropping toward the coast, St. Thomas
Mount visible as a real local high point near Guindy — physically sensible.
Preview: `data/raw/dem/dem_slope_preview.png`. These are two of the model's
MUST-HAVE static node features (`working.md` §1.7), sampled per road segment
once the line-graph exists (task 2.3).

## `fetch_rainfall.py` / `fetch_rainfall_imd.py` — tasks 1.7 / 1.8

Pull hourly (Open-Meteo/ERA5-Land) and daily (IMD gridded) rainfall for the
event window. **Read the discrepancy note in `working.md` §1.8 before using
either for task 2.5** — the two sources disagree by 2–6× on magnitude, and
neither cleanly matches the literature's claimed peaks. Recommended fix:
IMD for magnitude, Open-Meteo for within-day hourly shape.

```bash
python src/ground_truth/fetch_rainfall.py       # -> data/raw/rainfall/open_meteo_hourly.csv
python src/ground_truth/fetch_rainfall_imd.py   # -> data/raw/rainfall/imd_daily.csv
```

Note in `fetch_rainfall_imd.py`: the IMD grid cell nearest the AOI is
coastal and fully masked (no data any day in 2015) — the script falls back
to the nearest cell that actually has data (26km inland) rather than
silently returning an empty series.

## `query_sentinel1_gee.py` — task 1.9

Confirms the 4 Sentinel-1 scenes identified in the feasibility check
(12/24 Nov, 6/18 Dec 2015) are reachable through Google Earth Engine, which
is what task 3.1's actual SAR change detection will run on.

**Done (01 Sep 2026).** GEE project: `flood-intelligence-507219` (hardcoded
as `GEE_PROJECT` in the script — use this same project ID for any other GEE
work on this project, `ee.Initialize(project="flood-intelligence-507219")`).
All 4 confirmed scenes came back from GEE, exact match to the Copernicus
catalog check — image IDs are in `developing.md`'s Phase 1 notes.

One-time setup for anyone else on the team who needs to run this:
```bash
earthengine authenticate   # opens a browser; needs a Google account with GEE access on this project
python src/ground_truth/query_sentinel1_gee.py
```

## `georeference_nrsc_simulation.py` — task 0.6 follow-up

Georeferences Fig. 8 (the flood-depth map) from an NRSC/ISRO hydrological
simulation report the team obtained directly — *not* a RISAT SAR product,
see the caveat below — into an approximate raster clipped to the study wards.

**Setup:** extract the figure image from the source PDF and save it as
`data/raw/bhuvan/fig8_raw.jpg` (933×487 JPEG; the source PDF is
`Adayar &Cooum Rivers.pdf`, "Hydrological Simulation Study of Flood Disaster
in Adyar and Cooum Rivers, Tamilnadu", NRSC/ISRO v1.2, 07 Dec 2015).

**Run:**
```bash
python src/ground_truth/georeference_nrsc_simulation.py
```

**Outputs** (`data/raw/bhuvan/`, gitignored):
| File | Contents |
|---|---|
| `fig8_georeferenced.tif` | Water-depth-in-meters raster, EPSG:4326, clipped to the study wards |
| `fig8_georeferenced.png` | Preview with ward boundaries |
| `qa_overlay.png` | Same raster with OSM waterways overlaid — the alignment check that matters |
| `homography_accuracy.json` | The 9 control points used and per-point reprojection error |

### What this is, and isn't

This is a **bare-earth hydrological simulation** (rainfall-runoff + DEM
model), not an observed/measured flood extent — the source report says so
explicitly ("approximate," "without considering urban infrastructure"). It
is a materially different evidentiary tier from Sentinel-1 SAR change
detection, which stays the project's **primary** ground truth
(`working.md` §1.5).

**Georeferencing method:** Fig. 8 is an oblique 3D perspective screenshot
(Google-Earth-style), not an orthophoto. A planar homography was fit from 9
labeled-locality control points (marker pixels found by automated dark-red
detection, matched to known lon/lat) — this only works as an approximation
because Chennai's terrain is genuinely flat, which the source report also
notes. **Reprojection error: RMSE 563 m, max 1,045 m** — 10-20% of the study
cluster's own span. This is not segment-level ground truth.

**Cross-check that makes it worth keeping anyway:** overlaying the
raster's darkest-depth band against the **independently-sourced OSM
waterway layer** (`data/raw/osm/waterways.geojson`, task 1.2 — a completely
different source, georeferenced a completely different way) shows the
simulated deep-water channel tracking the OSM-tagged Adyar river closely —
see `qa_overlay.png`. That's real corroboration between two independent
sources, despite the coarse positional accuracy.

**Bottom line:** use this as a coarse, ward-scale plausibility check when
fusing labels in task 3.4 — e.g. "does the fused label agree this ward's
river-adjacent segments were the deepest-hit" — never as a precise
per-segment label on its own, and never in place of Sentinel-1.

## `verify_bhuvan_wms.py` — task 0.6, round 2

The PDF above came from a real historical-flood archive on the Bhuvan portal
(`bhuvan-app1.nrsc.gov.in/disaster/usrtasks/flood/flood.php`) that the first
0.6 pass missed — it's a year/state layer tree that **does** list Chennai-2015
entries: a RISAT-1 cumulative-inundation layer and a Cartosat-2 post-event
layer, both with working-looking checkboxes in the UI.

This script tests whether those checkboxes actually work — traces each one's
`onclick` handler through the portal's own JS (`loadfloodmap()` /
`loadlayer()` → `loadmap()` in `disaster/lib/uncomp/disaster09122022_v2.js`)
to the real WMS backend URL, then issues a direct `GetMap` request.

**Run:**
```bash
python src/ground_truth/verify_bhuvan_wms.py
```

**Result (01 Sep 2026):** every Chennai-2015 layer the UI advertises is dead
— the RISAT-1 and Cartosat-2 layers return a WMS `ServiceException: invalid
layer` from `bhuvan-ras2.nrsc.gov.in/mapcache`; the Tamil Nadu date-stamped
extents are absent from `bhuvan-gp1.nrsc.gov.in/bhuvan/wms`'s own
`GetCapabilities` and serve blank placeholder tiles. Full results in
`data/raw/bhuvan/wms_verification.json` (gitignored — rerun to regenerate).

**Why this matters:** it upgrades the 0.6 verdict from "we navigated the UI
and didn't find an export option" to "we tested the actual data layer at the
protocol level and it doesn't exist on the server" — same conclusion
(Bhuvan/RISAT is not usable ground truth), much stronger evidence. If NRSC
is approached with a formal data request, this is exactly what to describe:
the specific dead layer names, so they know what to actually provide.

## `run_sentinel1_change_detection.py` — task 3.1

Objective 1's PRIMARY ground truth: SAR change-detection flood mapping
between the 24 Nov 2015 (pre-event) and 6 Dec 2015 (post-event) passes task
1.9 already confirmed reachable in GEE.

**Run:**
```bash
python src/ground_truth/run_sentinel1_change_detection.py
```

**Method:** pre/post VV backscatter (dB, speckle-smoothed) → per-pixel
change (post − pre) thresholded at `mean − 2·std` (a self-calibrating
robust-outlier cutoff on THIS pass pair's own noise floor) → JRC Global
Surface Water permanent-water exclusion → sieve-filtered (GDAL minimum
mapping unit, 8px) → vectorized, clipped to the study wards.

**Why not a fixed literature dB threshold:** tried UN-SPIDER's commonly-cited
−17dB VV default first — only 0.08% of the AOI's post-event pixels fall
below it at all in this dense-urban scene (double-bounce off building
facades runs backscatter higher than the open/rural terrain that default
assumes), producing a near-empty ~0.01 km² result. Otsu's method on the
difference histogram was tried next — picked −0.77dB and flagged 54% of the
AOI, because the change values are one noisy mode with a skewed tail, not a
clean bimodal split. Full reasoning and both rejected alternatives' numbers
are in the script's own docstring and printed in the validation report.

**Result (19 Sep 2026):** 449 flood polygons, 0.60 km² total (~1.3% of the
AOI), touching **all 16 of 16 study wards** — Perungudi (Ward 169) highest
at 0.0941 km², down to Adyar (Ward 181) lowest at 0.0054 km². (An earlier
version of this note claimed only 3 wards showed any flooding — that was a
bug in `validate_flood_extent()`'s per-ward breakdown, not the underlying
detection: it keyed flooded-area-per-ward on `Zone_Name` alone, and 13 of
this study's 16 wards all share `Zone_Name="ADYAR"`, so each subsequent
Adyar ward's area silently overwrote the previous one instead of summing.
Fixed in `per_ward_area_breakdown()`, task 3.2's PR — the polygon geometries
and 0.60 km² total were never affected, only this one report field.)
Cross-checked against task 0.6's georeferenced NRSC simulation raster: mean
depth at flood-polygon centroids (2.94m) is higher than at random AOI points
(2.62m) — corroborating, not proof, given that raster's own ~563m RMSE.

**Read as a floor, not the full picture:** this is a known, expected SAR
limitation (working.md §1.5), not a bug to chase — dense urban canopy
genuinely suppresses the water-like backscatter signal, and this pass pair
brackets the 30 Nov–2 Dec peak rather than capturing it (nearest pass is +4
days post-peak). 0.60 km² across the whole 16-ward cluster is still a small
fraction of the ~46.6 km² AOI. Task 3.3 (news/advisory cross-check) is
explicitly the step that's supposed to catch what SAR misses.

Outputs (`data/processed/ground_truth/`, gitignored): the three downloaded
intermediate rasters (pre/post VV dB, JRC occurrence — kept for QA/rerun),
`sentinel1_flood_extent.geojson`, and the validation report.

## `extract_bhuvan_flood_footprint.py` — task 3.2

Objective 1's SECONDARY/opportunistic flood footprint (task 0.6's verdict —
Bhuvan/RISAT is not primary ground truth, unchanged here).

**Run:**
```bash
python src/ground_truth/extract_bhuvan_flood_footprint.py
```

**Re-verified directly (19 Sep 2026), not just cited:** re-ran
`verify_bhuvan_wms.py` as part of this task — Chennai's actual RISAT-1/
Cartosat-2 WMS layers (`ch_exp_0306dec15`, `ch_c2_sat`) are still dead
(`ServiceException: invalid layer`), unchanged from task 0.6's finding.
There is no live, exportable RISAT SAR footprint for this event to extract.

**What this extracts instead:** the one real Bhuvan-portal-sourced artifact
this project actually has — task 0.6's follow-up NRSC/ISRO hydrological-
simulation flood-depth raster (`fig8_georeferenced.tif`, see
`georeference_nrsc_simulation.py` above). Thresholded at `MIN_DEPTH_M=0.1`
(a noise/edge floor, not a "significant flooding" cutoff — 95.8% of the
raster's valid pixels already show >0.1m modeled depth, confirming the
source figure's whole colored region already **is** the simulation's
claimed inundation zone) and vectorized into the same polygon shape as
task 3.1's Sentinel-1 output, reusing that module's `sieve_mask()`/
`vectorize_mask()`/`clip_to_wards()`/`per_ward_area_breakdown()` directly
rather than reimplementing them.

**Result (19 Sep 2026):** 35 polygons, 21.66 km² total (much larger than
Sentinel-1's 0.60 km² — expected, since this raster is a whole-domain depth
surface, not a discriminating change-detection result), touching 14 of 16
wards. Cross-checked against task 3.1's Sentinel-1 result: 38.9% of the
Sentinel-1 flood area falls inside this Bhuvan/NRSC zone — meaningful
overlap given the two methods' opposite biases (SAR underestimates urban
flooding; this coarse ~563m-RMSE simulation likely overestimates extent).

**Read this as "was this ward inside the simulation's modeled flood zone,"
never a segment-level signal** — same caveat as `georeference_nrsc_simulation.py`
above. Task 3.3/3.4 should weight Sentinel-1 (task 3.1) as primary and use
this only as a coarse, ward-scale corroboration.

Outputs (`data/processed/ground_truth/`, gitignored):
`bhuvan_nrsc_flood_extent.geojson` and the validation report.

## `cross_check_news_advisories.py` — task 3.3

Cross-checks task 3.1/3.2's satellite-derived flood extents against real
news/advisory text, geocoding named roads/localities via task 2.9's
gazetteer + task 2.10's fuzzy matcher (working.md §1.5: "News/advisory
cross-check — segments matching named flooded roads/localities are
manually confirmed or added, catching what SAR misses").

```bash
python src/ground_truth/cross_check_news_advisories.py
```

**Resolution rule:** for each uniquely-mentioned place, try an EXACT match
against every line-graph segment's OSM `name` field first — regardless of
the gazetteer's `type` (not just `type=="road"`; task 2.9's own OSM-
verified additions are generically tagged `distress_text_candidate` even
when the matched feature actually is a road, e.g. "Gandhi Road" — an exact
string match against a real segment name has no real false-positive risk,
so trying it broadly is strictly better than gating on type). If that finds
nothing, falls back to every segment in the place's containing study ward
— coarser, same tier as task 3.2's raster cross-check.

**A real dtype bug found and fixed while building this:** the gazetteer's
`ward_no` column is `float64` (pandas' default once any row has a `NaN` —
task 2.9's unresolved candidates), while `study_wards.geojson`'s `Ward_No`
is `int32`. Naive string comparison (`str(177.0) == str(177)` → `"177.0" ==
"177"` → `False`) meant the ward-fallback silently matched ZERO segments
for every one of the 22 ward-level resolutions on the first real run.
Fixed by comparing as numbers (`int(float(ward_no))`), with a regression
test using the exact real-data shape (float ward_no vs. int Ward_No) that
would have caught it.

**Result (19 Sep 2026):** 69 resolved mentions across 23 unique places (22
via ward fallback, 1 exact segment-name match, 1 unresolved — no ward on
record) → **13,939 news-flagged segments, 81% of the entire 17,195-segment
graph.** That's broad, and disclosed as such: the real corpus's retrospective/
encyclopedia sources discuss the flood citywide, naming localities spread
across nearly all 16 study wards, and the ward-level fallback (by design,
same coarseness as task 3.2) flags every segment in a mentioned ward, not
just the specific spot. Cross-checked against task 3.1/3.2: 5.6% of
news-flagged segments overlap Sentinel-1, 47.8% overlap the Bhuvan/NRSC
zone, and **48.3% (6,736 segments) are news-only** — flagged by neither
satellite source. That news-only set is the concrete "catches what SAR
misses" this task is named for, not just a restated citation of the idea.

**Not phase-resolved, by design — disclosed, not glossed over:** checked
directly whether the real corpus carries per-passage dates; it doesn't
(only the ReliefWeb sitrep source's own URL has any date range, one of 19
original passages). A news-flagged segment means "flooded at some point
during the event," not "flooded in phase X." Task 3.4's fusion has to
decide how to fold in a phase-unresolved signal — this task doesn't guess.

Outputs (`data/processed/ground_truth/`, gitignored): `news_cross_check_places.csv`
(one row per resolved place), `news_cross_check_segments.geojson` (the
13,939 flagged segments), and the validation report.

## `fuse_flood_labels.py` — task 3.4

**Phase 3's exit criterion.** Fuses tasks 3.1 (Sentinel-1, primary), 3.2
(Bhuvan/NRSC, secondary), and 3.3 (news cross-check, secondary) into the
4-phase flood label scheme (working.md §1.5) — one binary `flood_label`
per line-graph segment per phase, in exactly the shape task 2.7's
`schema.json` `y_t1_contract` and task 2.8's `GraphSnapshotDataset.attach_labels()`
expect.

```bash
python src/ground_truth/fuse_flood_labels.py
```

**Fusion rule:** `ever_flooded = intersects(3.1) OR intersects(3.2) OR flagged-by(3.3)`
— working.md's own rule ("segment labeled flooded if it intersects the
polygon"), extended across all three sources with a plain OR.

**Phase-assignment decision — confirmed with the user, not derivable from
the data alone:** none of the three sources are phase-resolved (3.1 is a
single before/after change detection spanning the whole event; 3.2 has no
time axis at all; 3.3's real corpus has no reliable per-passage dates, per
that task's own note). So `ever_flooded` gets assigned to **peak and
receding only** — pre_event and rising stay dry. Grounded in task 2.5's
own rainfall numbers: rising totals just 31mm over 2 days vs. peak's
410mm — a 13x contrast suggesting flooding hadn't materialized yet during
rising. pre_event is always dry by construction (it's the SAR change
detection's own "before" reference snapshot).

**Result (19 Sep 2026):** SAR (primary) flags 932 segments (5.4%), Bhuvan
7,642 (44.4%), news 13,939 (81.1%) — the union is **15,038 segments
(87.46%) flooded at peak/receding.** That's dominated by the two coarse
secondary sources, not the primary one — checked and disclosed, not
hidden: only 241 segments (1.4%) have all 3 sources agreeing, 6,993
(40.7%) have 2, and 7,804 (45.4%) rest on exactly 1 source alone (mostly
news' broad ward-level fallback). **Discussed directly with the user
whether to keep the literal OR rule or require 2+ sources to agree
instead (which would give a tighter, more discriminating 42.1%) — kept
the OR rule**, since it's what working.md actually specifies rather than
a new judgment call layered on top. The per-source columns
(`sar_flagged`/`bhuvan_flagged`/`news_flagged`/`n_sources_agreeing`) are
saved in the output specifically so task 4.x can re-derive a stricter
subset later without re-running this script, if the 87% rate turns out to
limit what the GNN can learn.

**Validated beyond "does it run"** — an actual integration check against
task 2.8's real `GraphSnapshotDataset`: `attach_labels()` accepted the
fused table, and `transition_pairs()` produced exactly the 3 usable
`(X_t, Y_{t+1})` pairs task 2.7's `schema.json` promises, each with the
correct `(17195, 6)` / `(17195, 1)` shapes. This is the concrete proof
that "Phase 4 cannot meaningfully start without fused ground truth"
(developing.md §7) is now satisfied, not just an assertion.

Outputs (`data/processed/ground_truth/`, gitignored): `fused_flood_labels.csv`
(68,780 rows = 17,195 segments × 4 phases) and the validation report.

