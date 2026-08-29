# Spatiotemporal Digital Twins for Predictive Disaster Intelligence
### Revised Scope, Objectives, and Technical Working Document

---

## 0. Why this document exists

The original proposal had 4 objectives, each broad enough to be its own thesis (multilingual NLP fine-tuning, GNN cascading-failure modeling, a full replay engine, and an equity optimizer — all treated as independent, equally-weighted deliverables). For a 4-person team with a 3-month window, that scope was not defensible.

This document redesigns the project into **3 objectives**, where:
- **Objective 1** and **Objective 2** are the **priority, must-deliver** objectives.
- **Objective 3** is explicitly **optional / stretch goal** — attempt it only if Objectives 1 and 2 are complete and stable with time remaining.

The redesign also narrows scope along three axes that were previously too broad:

| Axis | Old scope | New scope | Why |
|---|---|---|---|
| Study area | Entire city | **10–20 contiguous wards** | City-scale OSM graphs have tens of thousands of nodes — uninspectable and untrainable in 3 months alongside everything else. A bounded ward cluster is small enough to fully debug by hand. |
| Infrastructure layers | Roads + drainage + utilities | **Roads + drainage only** | Utility network data (power/water) isn't reliably public for Indian cities. Drainage itself is already the harder half of this ask (see §1.3). |
| Objective 1 core claim | "Build a GNN for cascading failure" (no falsifiable claim) | **Rule-based baseline vs. learned temporal GNN**, compared head-to-head | Gives a defensible, measurable research claim: *"the learned model captures propagation patterns the static baseline misses."* This is also one arm of the ablation study already promised in the abstract. |

All three objectives still map directly onto the research gaps identified in the literature review (Slide 7/8 of the original deck): static one-shot cascade modeling, English-only crisis NLP pipelines, and equity-blind resource optimization.

---

## 1. Objective 1 (Priority, ~70% of project difficulty) — Ward-Scale Spatiotemporal Flood Propagation Model

### 1.1 What this objective actually is

Build a road-network graph for a bounded 10–20 ward area of one Indian city, replay one real historical flood event over it, and train a temporal Graph Neural Network to predict which road segments flood next, given the current state of the network and the rainfall driving it. Compare this learned model against a simple rule-based propagation baseline to show it captures something the baseline can't.

This is the actual "spatiotemporal graph" core the project is named after — the other two objectives feed into or consume this one.

**Research gap it fills:** existing GNN cascade-modeling work (flagged in the literature review) treats infrastructure failure as a static, one-shot classification problem, ignoring how failure actually propagates over time through a connected network. This objective treats propagation as the central object of study, not an afterthought.

### 1.2 Graph representation — the design decision, explained

There are two ways to turn OSM's road network into a graph:

- **Primal graph** (what `osmnx` gives you by default): node = intersection, edge = road segment.
- **Line graph / dual graph**: node = road segment, edge = "these two segments share an intersection."

**We use the line graph.** Reason: our prediction target is "is this road flooded" — that is a property of a *segment*, not a *point*. News reports, traffic advisories, and satellite flood polygons all describe flooding in terms of roads/segments, not intersection coordinates. Keeping intersections as nodes would force "segment flooded" to become an edge-level prediction, which is a more awkward setup in most GNN frameworks (including GraphSAGE) than node-level prediction.

Converting primal → line graph is a **standard, well-understood transform**, not a research problem — `networkx.line_graph()` does it directly, or it can be built manually in about 20 lines by treating every original edge as a new node and connecting new nodes that shared an original intersection.

```
OSM road network (osmnx)
        │
        ▼
Primal graph: node = intersection, edge = road segment
        │  (networkx.line_graph transform)
        ▼
Working graph: node = road segment, edge = shared-intersection adjacency
```

### 1.3 Node and edge definitions

**Node = one road segment.**

**Edge = adjacency (two segments share a physical intersection).** This is the channel through which flood/failure "propagates" — if segment A floods and segment B is adjacent, B's risk should be informed by A's state.

**Node features — static (fixed for the whole event):**

```
elevation_mean        — average elevation along the segment (DEM)
slope                 — along-segment gradient (derived from DEM)
length                — segment length in meters (OSM geometry)
distance_to_drain     — distance to nearest mapped drainage line (OSM)
impervious_pct        — % built-up/paved surface in a buffer around the segment (optional layer)
ward_population_density — Census 2011 ward-level density, via spatial join (optional layer)
```

**Node features — dynamic (recomputed at every timestep):**

```
rainfall_t             — rainfall in the current time window
cumulative_rainfall_t  — running total since event start
```

**Honest caveat, stated up front:** at ward-cluster scale (a few km across), rainfall data sources have grid cells of ~9–28 km — meaning every node in the study area will likely receive the *same* rainfall value at a given timestep. Rainfall does not spatially differentiate risk between segments; elevation, slope, and distance-to-drain do that. Rainfall's role is as a **global time-varying driver broadcast to every node**, exactly how weather features are used in published spatiotemporal traffic/flood GNN work. This should be stated explicitly in the report, not glossed over.

**Edge features:** none required for a first version — plain adjacency (`edge_index`) is sufficient input for GraphSAGE. An optional edge attribute (elevation difference between connected segments, as a crude flow-direction proxy) can be added later if time allows.

### 1.4 What literally goes into the model

```
N = number of segment-nodes (expect ~300–1500 for a 10–20 ward clip — verify this
    number early; if it exceeds ~2000, tighten the ward selection before proceeding)
F = 7 features (must-have set, see §1.7)
E = number of adjacency edges

X_t        = [N × F]     node feature matrix at timestep t
edge_index = [2 × E]     adjacency list (which segments touch which)
Y_{t+1}    = [N × 1]     binary flood label per segment at t+1
```

**Toy example (5 segments, illustrative):**

```
Segment   elev   slope  length  dist_drain  rain_t  cum_rain   →  flood_{t+1}
  A       912    1.2     180        40        22       58              1
  B       905    0.4      95        15        22       58              1
  C       918    2.1     220        90        22       58              0
  D       901    0.2     140         8        22       58              1
  E       915    1.8     160        60        22       58              0

edges (shared intersections): A–B, B–D, A–C, C–E, B–C
```

Segment D has the lowest elevation, flattest slope, and is closest to a drain — but note that proximity to a drain does not save it, since an overwhelmed drain doesn't help; that non-linearity is exactly what a GNN can learn and a static threshold rule can't. D's only neighbor, B, is also low-lying and flooded — GraphSAGE aggregates D's own features with B's state to predict `flood_{t+1}[D] = 1`. C sits highest and steepest with mixed-risk neighbors (A, E) and stays dry. **This neighbor-aggregation mechanism is the actual definition of "cascading" in this project** — it's what a per-segment-in-isolation rule-based model structurally cannot represent.

### 1.5 Ground truth — the hardest and most important part

**Verified conclusion: hourly, per-road-segment ground truth is not realistic to obtain for any Indian urban flood event.** This was checked directly, not assumed:

- **Satellite (Sentinel-1 SAR):** nominal constellation revisit is 6 days, but only when both satellites are flying. Sentinel-1B was lost in December 2021, so throughout 2022–2024 (covering both Bengaluru-2022 and Chennai-Dec-2023 as candidate events) only Sentinel-1A was operational, giving a **~12-day revisit**. This means at best one or two SAR passes land anywhere near an event's peak — a snapshot, not a time series.
- **News/traffic advisories:** genuinely useful, but report at *day / multi-hour* granularity for *named roads and localities* — not GPS-coordinate segments at exact clock times. Even the best-documented case found (Bengaluru, Sept 2022, dozens of articles) only resolves to "this road was advised-against by Monday morning," not exact timestamps.
- **NRSC/Bhuvan (ISRO):** confirmed to hold an official RISAT-derived flood footprint for the 2015 Chennai floods — again, one event-level map, not a time series.
- **No open dataset exists** (checked GitHub, Kaggle, academic data registries) that provides road-segment-level, timestamped flood state for any Indian city event.

**Redesigned ground truth — fusion of three coarse-but-real signals, at phase-level rather than hour-level:**

```
Phase 0: Pre-event   (dry baseline)
Phase 1: Rising      (derived from the hourly rainfall ramp-up)
Phase 2: Peak        (aligned to nearest available satellite pass / peak rainfall / news reports)
Phase 3: Receding     (post-peak, rainfall tapering, "still waterlogged" reports)
```

Each phase's label per segment is built by combining:
1. **Satellite inundation polygon** (nearest Sentinel-1 pass via change detection, or the Bhuvan RISAT footprint for Chennai 2015) — segment labeled flooded if it intersects the polygon. Known limitation: SAR underestimates flood extent in dense urban areas (confirmed in a published study of the 2015 Chennai floods) — state this explicitly as a limitation.
2. **News/advisory cross-check** — segments matching named flooded roads/localities are manually confirmed or added, catching what SAR misses.
3. **Hourly rainfall time series** (the only genuinely fine-grained signal available) — used to define the phase boundaries themselves.

This is not a compromise to apologize for — the published academic paper on the 2015 Chennai flood used exactly this kind of before/after change-detection approach, not continuous monitoring. This project is applying an already-established method, not inventing an under-tested one.

### 1.6 The core experiment: baseline vs. learned model

```
                 ┌─────────────────────────┐
                 │   Graph state at time t  │
                 │  X_t, edge_index         │
                 └────────────┬─────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                  ▼
  ┌─────────────────────┐          ┌───────────────────────────┐
  │  BASELINE (rule-based)│          │  LEARNED (temporal GNN)    │
  │  if rainfall > thresh │          │  GraphSAGE (spatial)       │
  │   → segment floods    │          │   + GRU/temporal layer     │
  │  if neighbor flooded  │          │   (via PyTorch Geometric   │
  │   & elevation below   │          │    Temporal, e.g. A3TGCN)  │
  │   threshold → floods  │          │  → per-segment flood prob. │
  └──────────┬───────────┘          └─────────────┬───────────────┘
             │                                     │
             └──────────────┬──────────────────────┘
                             ▼
              Compare both against fused ground truth
              (F1 / accuracy per phase transition)
                             │
                             ▼
        Claim: "the learned model captures propagation
         patterns the static baseline misses" — measured,
                     not just asserted
```

This comparison is deliberately one of the "three-way ablation" arms already promised in the abstract.

### 1.7 Final feature set for Objective 1

**MUST HAVE (build and validate the model with only these first):**
```
- road topology (line-graph: segment = node, shared intersection = edge)
- elevation (SRTM 30m)
- slope (derived from SRTM)
- distance to nearest drain (OSM waterway, geodesic distance)
- rainfall_t (hourly, aggregated to phase windows)
- cumulative_rainfall_t (derived)
- flood label per segment per phase (fused: satellite polygon + news cross-check)
```

**OPTIONAL (add only after the core comparison in §1.6 is working):**
```
- land cover / impervious % (ESA WorldCover)
- population density (Census 2011 ward-level — dated, more useful for Objective 3 than Objective 1)
- known flood-prone-point overlays (city-specific, needs direct verification before use)
```

### 1.8 Data source verification table

| Feature | Source | Resolution | Temporal? | Accessibility | Processing needed | Status |
|---|---|---|---|---|---|---|
| Road network | OSM via `osmnx` | segment-level | Static (current snapshot used as proxy) | 🟢 Free | Convert primal → line graph | **Keep** |
| Drainage/waterway | OSM `waterway=drain/ditch/stream` | segment-level | Static | 🟡 Free but sparse/inconsistent tagging in India (confirmed via OSM community discussion) | Manual coverage check for chosen wards before committing | **Keep, verify first** |
| Elevation (DEM) | SRTM GL1 30m, OpenTopography (AWS S3, no login) | 30m | Static | 🟢 Free, bulk-downloadable | Sample raster along segments (`rasterio`) | **Keep** |
| Slope | Derived from SRTM via `gdaldem slope` | 30m | Static | 🟢 | Pure processing | **Keep** |
| Rainfall (primary) | Open-Meteo Historical Weather API (ERA5/ERA5-Land) | hourly, ~9–28km | **Hourly** | 🟢 Free, no auth, simple REST call | Aggregate to phase windows | **Keep — primary driver** |
| Rainfall (cross-check) | IMD gridded 0.25° (`imdlib`/`imddaily` on PyPI) | daily, ~28km | Daily | 🟢 Free, working Python wrappers confirmed | Cross-validate against Open-Meteo | **Keep as secondary** |
| Distance to drainage | Derived (OSM roads ↔ OSM waterways) | meters | Static | 🟢 | `geopandas.sjoin_nearest` | **Keep** |
| Land cover | ESA WorldCover 10m (2021) | 10m | Static | 🟢 Free, CC-BY-4.0, available via Google Earth Engine | Buffer + zonal stats | **Optional** |
| Impervious surface | Derived from WorldCover "built-up" class | 10m | Static | 🟢 | Buffer + zonal stats | **Optional** |
| Population | Census 2011 Primary Census Abstract, ward-level | ward-level | Static, 15 yrs old | 🟢 Free (censusindia.gov.in / data.gov.in) | Spatial join segment→ward | **Optional** |
| Ward boundaries | data.gov.in / DataMeet / city GIS portals | polygon | Static | 🟢 Confirmed for Bengaluru & Chennai | Direct use | **Keep** |
| Satellite flood extent (Sentinel-1) | Google Earth Engine, SAR change detection | 10m | 1–2 snapshots per event | 🟢 Free access, established methodology (UN-SPIDER, ESA tutorials) | Change detection script (before/after backscatter) | **Keep, primary quantitative ground truth** |
| Satellite flood extent (Indian source) | NRSC Bhuvan — confirmed RISAT footprint exists for **2015 Chennai floods** | event-specific | Snapshot | 🟢 Referenced directly in academic literature | Extract via Bhuvan WebGIS | **Keep — strongest ground-truth candidate found** |
| News/advisory flood reports | Local news archives, traffic police advisories | road/locality-level, day/multi-hour | Coarse temporal | 🟢 Freely readable | Manual extraction + geocoding via gazetteer | **Keep — qualitative cross-check** |
| Copernicus EMS official maps | EU Copernicus Emergency Management Service | — | — | 🔴 **Verified: no EMSR activation exists for any of the three candidate cities** | — | **Drop** |

### 1.9 City & event recommendation

**Recommendation: Chennai, November–December 2015 floods.**

| Criterion | Bengaluru (Sept 2022) | **Chennai (Nov–Dec 2015)** | Hyderabad (Oct 2020) |
|---|---|---|---|
| Ward boundaries | 🟢 Confirmed | 🟢 Confirmed | 🟡 Less clean ward-level data |
| Official satellite flood footprint | 🔴 None found | 🟢 **NRSC/Bhuvan RISAT footprint confirmed** | 🟡 HEC-RAS modeled extent in a paper, not a released shapefile |
| Existing academic SAR replication | — | 🟢 **A published IEEE paper already did Sentinel-1 SAR change-detection mapping for this exact event** | — |
| News timeline detail | 🟢 Best found (day/multi-hour, named roads) | 🟡 Extensively covered (India's most-documented urban flood) | 🟡 Colony-level detail |
| Sentinel-1 revisit risk | 🟡 Only S1A flying in 2022 (~12-day revisit), unverified whether a pass landed near peak | 🟡 Same constraint, but **already proven usable by an existing paper** | 🔴 Not verified |

The deciding factor is not flood severity — it's that Chennai 2015 is the only candidate with **two independent, verified ground-truth trails**: an official ISRO satellite footprint, and a separate published paper that already applied the SAR methodology to this exact event. That gives the riskiest deliverable in the whole project (ground truth) a template to follow and a second source to cross-check against.

**Before committing:** spend half a day verifying directly — (1) check the Copernicus Browser for a usable Sentinel-1 pass near the event dates over the chosen wards, and (2) confirm the Bhuvan RISAT layer is actually exportable, not just referenced in a figure caption. Bengaluru 2022 is the fallback if Chennai's ~10-year-old OSM-as-proxy assumption breaks down for the chosen wards.

### 1.10 Tech stack — Objective 1

```
Data acquisition:   osmnx, requests (Open-Meteo API), imdlib/imddaily, rasterio
Geospatial processing: geopandas, shapely, networkx, rasterio, GDAL (gdaldem)
Ground truth (satellite): Google Earth Engine (Python API), Sentinel-1 GRD
Graph ML:           PyTorch, PyTorch Geometric, PyTorch Geometric Temporal (A3TGCN/MPNN-LSTM)
Baseline model:      plain Python / networkx BFS-style propagation
Evaluation/visualization: scikit-learn (F1/accuracy), Folium (map animation)
```

Note: PyTorch Geometric Temporal already implements the class of model this objective needs (A3TGCN, MPNN-LSTM, DCRNN) — the team is **adapting** an existing spatiotemporal GNN layer to this graph, not writing one from scratch. This is the single biggest feasibility de-risking factor for Objective 1.

---

## 2. Objective 2 (Priority) — Gazetteer-Grounded Multilingual Distress Detection

### 2.1 What this objective actually is

Detect distress signals (people reporting being stuck, flooded, needing help) in code-mixed regional-language social media posts within the same 10–20 ward study area, and resolve each detected post to a location on the map — feeding into Objective 1's graph (as a real-world signal layer) and Objective 3's optimizer (as demand input).

**Research gap it fills:** existing crisis-informatics pipelines (e.g. CrisisNLP-based systems) are built and benchmarked on English-language social media, systematically under-serving regions and populations that post in regional/code-mixed languages.

### 2.2 Two separate sub-problems (this is the part that was previously unclear)

Geoparsing is not one step — it's two, and conflating them is what made this objective look harder than it is:

```
"stuck near Ejipura signal, paani bohot zyada hai"
        │
        ▼
Step A — Toponym recognition: find the place-mentioning span → "Ejipura signal"
        │
        ▼
Step B — Toponym resolution: turn that string into a lat/long
```

**Step A + B combined, scoped approach:**

```
OSM pull (already done for Objective 1)
        │
        ▼
Extract every named entity in the study wards: road names,
locality names, landmarks, water bodies
        │
        ▼
Build a GAZETTEER: {name → coordinates}, scoped to the study area
        │
        ▼
For each social media post: fuzzy string match (rapidfuzz) between
tokens in the text and gazetteer entries
        │
        ▼
Matched → resolved coordinate. Unmatched → fall back to a
lightweight NER pass (only for the residual cases)
```

This works because code-mixed text usually keeps proper nouns (place names) untouched even when the surrounding sentence switches language — "Ejipura" stays "Ejipura" whether the sentence is Kannada, Hindi, or English. This is standard practice in crisis-informatics literature (gazetteer + fuzzy match, for bounded regions) and **directly reuses the OSM extraction already done for Objective 1** — no separate data-collection pipeline required.

### 2.3 Distress classification

```
Raw code-mixed social media post
        │
        ▼
Pretrained multilingual transformer (IndicBERT / MuRIL)
        │
        ▼
Lightly fine-tuned (NOT trained from scratch) on a small
hand-labeled set (a few hundred posts: distress / not-distress)
        │
        ▼
distress / not-distress label
```

Hand-labeling a few hundred examples for one bounded study area is realistic for a 4-person team in 1–2 weeks. This is deliberately scoped down from "build a full custom multilingual NLP pipeline" (which would itself be a thesis-scale problem, and a ready-made labeled code-mixed disaster dataset for Indian languages does not exist) to "apply and lightly adapt an existing pretrained model."

### 2.4 Full Objective 2 pipeline

```
        Social media posts (scoped to study area / event window)
                          │
          ┌───────────────┴───────────────┐
          ▼                                ▼
  Distress classifier              Gazetteer + fuzzy match
  (MuRIL/IndicBERT,                (built from Objective 1's
   lightly fine-tuned)              OSM extraction)
          │                                │
          └───────────────┬────────────────┘
                           ▼
          Distress posts, each resolved to a
          coordinate within the study wards
                           │
                           ▼
        Feeds into Objective 1's graph (as an observed
        signal layer) and Objective 3's optimizer (as demand)
```

### 2.5 Tech stack — Objective 2

```
NLP:            HuggingFace Transformers (IndicBERT / MuRIL), spaCy (fallback NER)
Matching:       rapidfuzz (fuzzy string matching against gazetteer)
Data handling:  pandas, geopandas
Labeling:       manual annotation (small team-labeled dataset, few hundred posts)
```

---

## 3. Objective 3 (OPTIONAL — attempt only if time remains) — Equity-Constrained Resource Allocation

> **This objective is explicitly a stretch goal.** Objectives 1 and 2 are the priority and should consume the majority of the 3-month timeline. Objective 3 should only be started once Objective 1's core baseline-vs-GNN comparison is working and Objective 2's distress+geoparsing pipeline is producing usable output. If time runs short, the project is still complete and defensible with just Objectives 1 and 2.

### 3.1 What this objective actually is (if attempted)

Using the fused graph state from Objective 1 and the distress locations from Objective 2, allocate limited relief resources across the study wards subject to a constraint: wards with a high vulnerability index (derived from Census 2011 demographic data) should not be deprioritized just because they generate less social-media signal.

**Research gap it fills:** traditional optimization models allocate purely by demand volume or speed, which systematically penalizes under-reporting / low-connectivity populations — exactly the bias the project's problem statement calls out.

### 3.2 Pipeline (if attempted)

```
Objective 1 output:            Objective 2 output:
graph state per phase    +     resolved distress locations
        │                              │
        └───────────────┬──────────────┘
                         ▼
        Ward-level vulnerability index
        (Census 2011: population density,
         literacy, sanitation access as proxies)
                         │
                         ▼
        Constrained optimizer (OR-Tools / PuLP)
        objective: allocate resources
        constraint: vulnerability-weighted floor,
                    so low-signal wards aren't starved
                         │
                         ▼
        Ranked/allocated resource plan per ward
```

### 3.3 Why this is lower priority than Objectives 1 and 2

Constraint optimization with OR-Tools/PuLP is a mature, well-documented technique — the risk here is genuinely low compared to Objective 1 (temporal GNN + ground-truth construction) or Objective 2 (multilingual NLP). That's precisely why it's the right thing to cut first if the timeline gets tight: it's valuable to have, but it isn't where the project's technical risk or its central "spatiotemporal" claim lives.

### 3.4 Tech stack — Objective 3 (if attempted)

```
Optimization:   Google OR-Tools or PuLP
Data handling:  pandas, geopandas (Census join)
```

---

## 4. How the three objectives connect

```
┌─────────────────────────────────────────────────────────────┐
│                     Objective 1 (priority)                    │
│   OSM road+drainage graph → temporal GNN flood propagation    │
│           (the spatiotemporal digital twin substrate)         │
└───────────────────────────┬────────────────────────────────┘
                             │  graph + gazetteer reused
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                     Objective 2 (priority)                    │
│   Multilingual distress detection + geoparsing,                │
│    grounded in Objective 1's OSM-derived gazetteer             │
└───────────────────────────┬────────────────────────────────┘
                             │  graph state + distress locations
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Objective 3 (OPTIONAL / stretch)              │
│   Equity-constrained resource allocation optimizer              │
└─────────────────────────────────────────────────────────────┘
```

Each objective consumes the previous one's output — this is what makes the project read as one integrated system rather than three loosely related pieces, and it's also why cutting Objective 3 under time pressure doesn't break the narrative: Objectives 1 and 2 already form a complete, defensible story (build the digital twin, ground it with real-world distress signal).

---

## 5. Full tech stack summary

```
Language:            Python 3.11
Geospatial:          osmnx, geopandas, shapely, networkx, rasterio, GDAL
Graph ML:            PyTorch, PyTorch Geometric, PyTorch Geometric Temporal
NLP:                 HuggingFace Transformers (IndicBERT/MuRIL), spaCy, rapidfuzz
Optimization (opt.): Google OR-Tools / PuLP
Satellite/EO:        Google Earth Engine (Sentinel-1), rasterio, GDAL
Data access:         Open-Meteo API, imdlib/imddaily (IMD), OpenTopography (SRTM),
                      ESA WorldCover, Census 2011 (censusindia.gov.in), NRSC Bhuvan
Dashboard/backend:   Streamlit, Folium, FastAPI, SQLite/DuckDB
Compute:             Google Colab / Kaggle free-tier GPU (T4/A100) for GNN + transformer fine-tuning
```

---

## 6. Source verification summary (all links checked live during scoping)

- [IMDLIB documentation](https://imdlib.readthedocs.io/en/latest/Usage.html)
- [IMD gridded rainfall download](https://rcc.imdpune.gov.in/download.php)
- [imddaily on PyPI](https://pypi.org/project/imddaily/0.2.1/)
- [OpenTopography SRTM GL1 30m](https://portal.opentopography.org/raster?opentopoID=OTSRTM.082015.4326.1)
- [SRTM 30m tile downloader](https://dwtkns.com/srtm30m/)
- [OSM Tag:waterway=drain wiki](https://wiki.openstreetmap.org/wiki/Tag:waterway=drain)
- [OSM community forum — India drain tagging gaps](https://community.openstreetmap.org/t/drain-covers-and-blocked-drains/106138)
- [BBMP GIS Viewer](https://bbmp.gov.in/gisviewer/)
- [DataMeet Municipal Spatial Data (GitHub)](https://github.com/datameet/Municipal_Spatial_Data)
- [Bharatlas BBMP ward map](https://bharatlas.com/view/wards_bengaluru_bbmp_2022)
- [Chennai Wards.geojson (DataMeet)](https://github.com/datameet/Municipal_Spatial_Data/blob/master/Chennai/Wards.geojson)
- [GCC Ward Information (OpenCity)](https://data.opencity.in/dataset/gcc-ward-information)
- [Copernicus EMS Activations](https://mapping.emergency.copernicus.eu/activations/)
- [Bengaluru 2022 flood technical study](https://www.academia.edu/97563902/Floods_at_Bengaluru_City_A_Technical_Study)
- [Mongabay — Bengaluru floods and lake loss](https://india.mongabay.com/2022/09/bengaluru-floods/)
- [Deccan Herald — ORR flooding Sept 2022](https://www.deccanherald.com/india/karnataka/bengaluru/heavy-rain-in-bengaluru-leaves-outer-ring-road-flooded-4025388)
- [Comparative study 2015 vs 2023 Chennai flooding (MDPI)](https://www.mdpi.com/2073-4441/16/17/2477)
- [Change Detection Flood Mapping of 2015 Chennai Flood using Sentinel-1 (IEEE)](https://ieeexplore.ieee.org/iel7/8891871/8897702/08899282.pdf)
- [UN-SPIDER Sentinel-1 GEE flood mapping recommended practice](https://un-spider.org/advisory-support/recommended-practices/recommended-practice-google-earth-engine-flood-mapping)
- [2020 Hyderabad floods (Wikipedia)](https://en.wikipedia.org/wiki/2020_Hyderabad_floods)
- [Rapid Assessment of Oct 2020 Hyderabad Urban Flood (ResearchGate)](https://www.researchgate.net/publication/352734097_Rapid_Assessment_of_The_October_2020_Hyderabad_Urban_Flood_and_Risk_Analysis_Using_Geospatial_Data)
- [2015 Chennai floods RISAT footprint figure (ResearchGate)](https://www.researchgate.net/figure/Fig-8-a-2015-Chennai-floods-event-foot-print-NRSC-Bhuvan-RISAT-b-AIR-100-year_fig4_325038820)
- [Bhuvan Disaster Management Support Services](https://bhuvan-app1.nrsc.gov.in/bhuvandisaster/)
- [Flood Affected Area Atlas of India (NRSC)](https://ndem.nrsc.gov.in/documents/downloads/Flood%20Affected%20Area%20%20Atlas%20of%20India%20-Satellite%20based%20study.pdf)
- [ESA WorldCover Download](https://worldcover2021.esa.int/download)
- [ESA WorldCover on Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100)
- [Census 2011 ward-level data](https://censusindia.gov.in/census.website/en/data/tables)
- [Sentinel-1 constellation status 2026](https://dataspace.copernicus.eu/news/2026-5-28-sentinel-1-orbital-reconfiguration-dates)
- [Copernicus Data Space Catalog API](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html)
- [India Flood Inventory (GitHub)](https://github.com/hydrosenselab/India-Flood-Inventory)

---

## 7. Bottom line for the guide meeting

- **Objective 1** is the hardest and most central piece — it is the actual spatiotemporal digital twin. It is technically feasible for students because the temporal-GNN component (GraphSAGE + recurrent layer) is available pre-built in PyTorch Geometric Temporal; the team is adapting, not inventing. The real risk is ground truth, which is why it's been redesigned to phase-level (not hourly) labels fused from satellite + news + rainfall, following an already-published methodology (the 2015 Chennai Sentinel-1 SAR paper).
- **Objective 2** is scoped down from "build a multilingual NLP pipeline" to "apply pretrained models + reuse Objective 1's OSM data as a gazetteer" — this is what makes it achievable in the same timeframe without needing a custom-labeled code-mixed dataset from scratch.
- **Objective 3 is optional.** It's real, useful, and not hard (OR-Tools constraint optimization is mature), but it is explicitly the first thing to drop if the 3-month timeline gets tight. The project remains complete and defensible on Objectives 1 and 2 alone.
