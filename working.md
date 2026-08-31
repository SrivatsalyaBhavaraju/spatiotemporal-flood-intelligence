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

- **Satellite (Sentinel-1 SAR):** nominal constellation revisit is 6 days, but only when both satellites are flying. Sentinel-1B was lost in December 2021 (and hadn't even launched yet in 2015), so throughout the Chennai 2015 event only Sentinel-1A was operational, giving a **~12-day revisit**. **Verified directly against the live Copernicus Data Space Ecosystem catalog (31 Aug 2026, not just literature):** exactly 4 S1A passes cover the study wards in the event window — **12 Nov, 24 Nov, 6 Dec, 18 Dec 2015** (all IW mode, GRD+SLC, full swath coverage of the AOI). None falls inside the 30 Nov–2 Dec peak itself; the nearest is Dec 6, four days after. This means at best a before/after pair brackets the peak — a snapshot pairing, not a time series, and confirmed to not include the peak day.
- **News/traffic advisories:** genuinely useful, but report at *day / multi-hour* granularity for *named roads and localities* — not GPS-coordinate segments at exact clock times. Even the best-documented case found (Bengaluru, Sept 2022, dozens of articles) only resolves to "this road was advised-against by Monday morning," not exact timestamps.
- **NRSC/Bhuvan (ISRO):** an official RISAT-derived flood footprint for the 2015 Chennai floods is real and cited in peer-reviewed literature — but **verified directly (31 Aug 2026) that it is not confirmed exportable**: the live Bhuvan disaster portal has no historical-event archive for 2015 (it's oriented at current/recent disasters), and community precedent (DataMeet mailing list) confirms Bhuvan flood-zonation layers are raster WMS tiles, not shapefiles, with export requiring WMS-scraping/georeferencing workarounds unproven for this specific layer. **Downgraded to opportunistic secondary source** — pursue via a formal NRSC data request in parallel with Phase 1, not on Objective 1's critical path. Sentinel-1 is the primary ground-truth path.
- **No open dataset exists** (checked GitHub, Kaggle, academic data registries) that provides road-segment-level, timestamped flood state for any Indian city event.

**Redesigned ground truth — fusion of three coarse-but-real signals, at phase-level rather than hour-level, with concrete anchors now verified for Chennai 2015:**

```
Phase 0: Pre-event   (dry baseline)              — anchored to the 24 Nov 2015 Sentinel-1 pass
Phase 1: Rising      (hourly rainfall ramp-up)    — 30 Nov – 1 Dec 2015 rainfall telemetry
Phase 2: Peak        (peak rainfall / news reports) — 1–2 Dec 2015; no direct SAR pass this close, see above
Phase 3: Receding    (post-peak, tapering)        — anchored to the 6 Dec 2015 Sentinel-1 pass (closest post-peak, +4 days)
```

Each phase's label per segment is built by combining:
1. **Satellite inundation polygon** — **primary:** Sentinel-1 change detection between the 24 Nov (pre) and 6 Dec (post) passes confirmed above; **secondary, opportunistic only:** the Bhuvan RISAT footprint for Chennai 2015, if a formal NRSC request succeeds. Segment labeled flooded if it intersects the polygon. Known limitation: SAR underestimates flood extent in dense urban areas (confirmed in a published study of the 2015 Chennai floods) — state this explicitly as a limitation. A second limitation now confirmed directly: the available passes bracket the peak rather than capturing it, so Rising/Peak boundaries lean more heavily on rainfall + news than on SAR.
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
| Satellite flood extent (Sentinel-1) | Google Earth Engine, SAR change detection | 10m | 2 snapshots per event | 🟢 **Verified via live Copernicus catalog query (31 Aug 2026): 4 S1A passes confirmed over study wards — 12 Nov, 24 Nov, 6 Dec, 18 Dec 2015** | Change detection script (before/after backscatter, Nov 24 → Dec 6 pair) | **Keep — primary quantitative ground truth** |
| Satellite flood extent (Indian source) | NRSC Bhuvan — RISAT footprint exists for **2015 Chennai floods** | event-specific | Snapshot | 🔴 **Verified 31 Aug 2026: live portal has no 2015 historical archive; flood layers are raster WMS, no confirmed vector export** — viewable/citable, not confirmed downloadable | Would require formal NRSC data request, or unproven WMS-scraping workaround | **Downgraded — opportunistic secondary only, not a dependency** |
| News/advisory flood reports | Local news archives, traffic police advisories | road/locality-level, day/multi-hour | Coarse temporal | 🟢 Freely readable | Manual extraction + geocoding via gazetteer | **Keep — qualitative cross-check** |
| Copernicus EMS official maps | EU Copernicus Emergency Management Service | — | — | 🔴 **Verified: no EMSR activation exists for any of the three candidate cities** | — | **Drop** |

### 1.9 City & event recommendation — CONFIRMED (feasibility check completed 31 Aug 2026)

**Confirmed: Chennai, 8 November – 14 December 2015 floods.** This was a recommendation as of initial scoping; it has since been independently verified (ward geometry, event timing, live satellite catalog, live Bhuvan portal) rather than left as an assumption. Full evidence, the polygon-adjacency map, and the decision table are in the [Chennai Study-Area Validation report](https://claude.ai/code/artifact/13744b14-c706-4bde-acf0-0042146e0129).

**Study wards (confirmed geometrically contiguous — 16 wards):** 142, 168, 169, 170–182, forming one connected component along the Adyar river corridor from Saidapet down through Kotturpuram/Adyar to Taramani/Velachery. Verified via true polygon adjacency (Shapely) against the DataMeet ward boundary file — not centroid proximity. Zero outliers. The 168/169 link runs through a narrow neck at 177/178; worth a road-network sanity check once OSM data lands in task 1.1, but not a blocker.

**Event date range:** 8 Nov – 14 Dec 2015 overall, with **30 Nov – 2 Dec 2015 as the Peak phase anchor** — both the rainfall maximum (490mm/24h at Tambaram, the heaviest single day since 1901) and the only point with a direct hydraulic forcing event (the Chembarambakkam reservoir release, ~20,000 cusecs into the Adyar river) tied to the river running through the study wards.

| Criterion | Bengaluru (Sept 2022) | **Chennai (Nov–Dec 2015)** | Hyderabad (Oct 2020) |
|---|---|---|---|
| Ward boundaries | 🟢 Confirmed | 🟢 **Confirmed — 16-ward cluster geometrically verified contiguous** | 🟡 Less clean ward-level data |
| Official satellite flood footprint | 🔴 None found | 🟡 **RISAT footprint exists but verified NOT confirmed exportable** (see below) | 🟡 HEC-RAS modeled extent in a paper, not a released shapefile |
| Existing academic SAR replication | — | 🟢 **A published IEEE paper already did Sentinel-1 SAR change-detection mapping for this exact event** | — |
| News timeline detail | 🟢 Best found (day/multi-hour, named roads) | 🟡 Extensively covered (India's most-documented urban flood) | 🟡 Colony-level detail |
| Sentinel-1 revisit risk | 🟡 Only S1A flying in 2022 (~12-day revisit), unverified whether a pass landed near peak | 🟢 **Verified via live catalog: 4 confirmed S1A passes (12/24 Nov, 6/18 Dec) fully cover the study wards** — though none lands inside the peak window itself | 🔴 Not verified |

**What changed from the original recommendation, now that it's verified rather than assumed:** the Bhuvan/RISAT footprint is real and citable, but checking the live portal directly found no 2015 historical archive and only raster WMS access with no confirmed vector export — so it's **downgraded from primary ground-truth candidate to opportunistic secondary**, pursued via a formal NRSC request off the critical path. **Sentinel-1 change detection (Nov 24 pre → Dec 6 post, both confirmed covering the study wards) is now Objective 1's primary ground-truth path.** Bengaluru 2022 remains the fallback if this ward cluster's OSM-as-proxy assumption breaks down during Phase 1.

**Team sign-off still needed:** this report is the evidence base for `developing.md` task 0.7 (go/no-go), which is an "All" task — check it off once the whole team has actually reviewed and agreed, not just on this verification.

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
- [Copernicus Data Space Ecosystem OData catalog](https://catalogue.dataspace.copernicus.eu/odata/v1/Products) — queried live 31 Aug 2026 to confirm actual Sentinel-1 acquisition dates over the study wards (§1.5, §1.9)
- [Chennai Floods Situation Report No. 1, 2–4 Dec 2015 (ReliefWeb)](https://reliefweb.int/report/india/chennai-floods-situation-report-no-1-chennai-flood-2-4-december-2015)
- [Chennai floods, December 2015 (World Weather Attribution)](https://www.worldweatherattribution.org/chennai-floods-december-2015/)
- [2015 South India floods (Wikipedia)](https://en.wikipedia.org/wiki/2015_South_India_floods)
- [ADRC disaster record — India flood 2015/11/08](https://www.adrc.asia/view_disaster_en.php?Lang=en&Key=2059)
- [DataMeet mailing list — "Looking for flood data from ISRO Bhuvan"](https://groups.google.com/g/datameet/c/eyg-T6w6Gl4/m/TmVlyDaxGAAJ) — community precedent confirming Bhuvan disaster layers are raster WMS, not shapefiles
- [Bhuvan disaster portal — direct check](https://bhuvan-app1.nrsc.gov.in/disaster/disaster.php) — verified live 31 Aug 2026: no 2015 historical-event archive/date picker
- [Chennai Study-Area Validation report (feasibility check, 31 Aug 2026)](https://claude.ai/code/artifact/13744b14-c706-4bde-acf0-0042146e0129) — full ward-contiguity, date-range, Sentinel-1, and Bhuvan verification with decision table

---

## 7. Bottom line for the guide meeting

- **Objective 1** is the hardest and most central piece — it is the actual spatiotemporal digital twin. It is technically feasible for students because the temporal-GNN component (GraphSAGE + recurrent layer) is available pre-built in PyTorch Geometric Temporal; the team is adapting, not inventing. The real risk is ground truth, which is why it's been redesigned to phase-level (not hourly) labels fused from satellite + news + rainfall, following an already-published methodology (the 2015 Chennai Sentinel-1 SAR paper).
- **Objective 2** is scoped down from "build a multilingual NLP pipeline" to "apply pretrained models + reuse Objective 1's OSM data as a gazetteer" — this is what makes it achievable in the same timeframe without needing a custom-labeled code-mixed dataset from scratch.
- **Objective 3 is optional.** It's real, useful, and not hard (OR-Tools constraint optimization is mature), but it is explicitly the first thing to drop if the 3-month timeline gets tight. The project remains complete and defensible on Objectives 1 and 2 alone.
