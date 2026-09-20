# Building the Project — Team Tracker

This is the **builder's document** — companion to [`working.md`](./working.md), which explains *why* the project is scoped this way and the technical rationale for every decision. This document is about *who does what, when, and in what order*. Update the Status column as you go; that's the whole point of this file.

**Team:** 4 people, labeled P1–P4 in every task table below. Who is P1 vs P2 vs P3 vs P4 is for the team to decide — pick based on interest/skill, not assigned here. Roughly:
- **P1's track** = geospatial & graph work (OSM, DEM, line-graph transform, features)
- **P2's track** = ground truth & data work (satellite, rainfall, label fusion)
- **P3's track** = ML/GNN work (model, training, evaluation)
- **P4's track** = NLP / Objective 2 work (gazetteer, geoparsing, distress classifier), and Objective 3 if attempted

These are tracks, not solo ownership — most phases need cross-support, and a "P1 task" doesn't have to be done by the same person all the way through if you'd rather rotate.

**Timeline:** 12 weeks (~3 months), 8 phases, Phase 7 (Objective 3) optional.

---

## 1. How to read this document

Every task is numbered **`<phase>.<feature>`** — e.g. `0.2` is Phase 0's 2nd feature, `3.4` is Phase 3's 4th. That's literally how you talk about it: *"phase 0 feature 2, let's go."* Each task also has an **Owner** column (P1/P2/P3/P4/All — see the Team note at the top for what each track means) and a **Depends on** column naming the exact ID(s) it needs finished first — not just "depends on graph work." When you finish a task, change its Status from `☐` to `✅`. If a task is blocked, leave it `☐` and check whether its dependency is done yet. `🔶` means everything automatable is done and code-ready, but the task is stuck on one specific manual step (e.g. an interactive login only a human can complete) — check the task's notes for exactly what's left.

Phase 7 (`7.1`–`7.3`) is Objective 3 and is optional — see §11.

---

## 2. Phase dependency map

```mermaid
flowchart TD
    P0["Phase 0 — Setup & Foundation\n(Week 1)"]
    P1p["Phase 1 — Data Acquisition\n(Weeks 2-3, 4-way parallel)"]
    P2p["Phase 2 — Graph Construction & Features\n(Weeks 3-4)"]
    P3p["Phase 3 — Ground Truth Construction\n(Weeks 4-5)"]
    P4p["Phase 4 — Model Development & Training\n(Weeks 6-7)"]
    P5p["Phase 5 — Evaluation & Comparison\n(Week 8)"]
    P6p["Phase 6 — Integration\n(Week 9)"]
    P7p["Phase 7 — Objective 3 (OPTIONAL)\n(Week 10)"]
    P8p["Phase 8 — Docs, Report, Presentation\n(Weeks 10-12)"]

    P0 --> P1p
    P1p --> P2p
    P1p --> P3p
    P2p -.overlaps.-> P3p
    P2p --> P4p
    P3p --> P4p
    P4p --> P5p
    P5p --> P6p
    P6p -.-> P7p
    P6p --> P8p
    P7p -.-> P8p

    style P7p stroke-dasharray: 5 5
```

**Reading this:** Phase 0 gates everything. Phase 1 is where all 4 people work independently at the same time. Phases 2 and 3 *overlap* rather than strictly sequence (P2's ground-truth work can start as soon as Phase 1's satellite/news data lands, it doesn't need P1's feature engineering finished). Phase 4 needs both 2 and 3 complete. Objective 2 (P4's track) runs almost entirely on its own rail from Phase 1 through Phase 5, only rejoining Objective 1's output at Phase 6 (Integration). Phase 7 is dashed because it's optional and only starts if Phase 6 is stable with time left.

## 2b. Feature-level dependency map (every task, not just phases)

This is the detailed version of the flowchart above — every individual feature across all 8 phases, with the real edges between them. Grouped into a box per phase so you can still see the phase shape, but arrows cross phase boundaries wherever a feature actually depends on one from an earlier (or the same) phase. This is the picture to check before saying "phase X feature Y, let's go" — trace its incoming arrows to make sure they're all `✅` first.

```mermaid
flowchart TD
    subgraph PH0["Phase 0 — Setup"]
        f0_1["0.1 Repo + scaffold"]
        f0_2["0.2 Python env"]
        f0_3["0.3 City + wards"]
        f0_4["0.4 Event dates"]
        f0_5["0.5 Verify S1 pass"]
        f0_6["0.6 Verify Bhuvan export"]
        f0_7["0.7 Go/no-go"]
    end

    subgraph PH1["Phase 1 — Data Acquisition"]
        f1_1["1.1 OSM roads"]
        f1_2["1.2 OSM drainage"]
        f1_3["1.3 Ward boundaries"]
        f1_4["1.4 Drainage coverage check"]
        f1_5["1.5 SRTM DEM"]
        f1_6["1.6 Slope raster"]
        f1_7["1.7 Hourly rainfall"]
        f1_8["1.8 IMD daily rainfall"]
        f1_9["1.9 GEE access setup"]
        f1_10["1.10 GPU env"]
        f1_11["1.11 PyG install/test"]
        f1_12["1.12 A3TGCN/MPNN prototype"]
        f1_13["1.13 Social media collection"]
        f1_14["1.14 Gazetteer draft"]
    end

    subgraph PH2["Phase 2 — Graph & Features"]
        f2_1["2.1 Primal graph"]
        f2_2["2.2 Line-graph transform"]
        f2_3["2.3 Static node features"]
        f2_4["2.4 Optional features"]
        f2_5["2.5 Rainfall aggregation"]
        f2_6["2.6 Attach dynamic features"]
        f2_7["2.7 Model input schema"]
        f2_8["2.8 Data loader"]
        f2_9["2.9 Gazetteer final"]
        f2_10["2.10 Fuzzy matching"]
        f2_11["2.11 Hand-label distress"]
    end

    subgraph PH3["Phase 3 — Ground Truth"]
        f3_1["3.1 Sentinel-1 SAR change detect"]
        f3_2["3.2 Bhuvan RISAT extract"]
        f3_3["3.3 News cross-check"]
        f3_4["3.4 Fuse 4-phase labels"]
        f3_5["3.5 Freeze graph"]
        f3_6["3.6 Baseline model"]
        f3_7["3.7 Training/eval harness"]
        f3_8["3.8 Fine-tune classifier"]
    end

    subgraph PH4["Phase 4 — Model Training"]
        f4_1["4.1 Train GNN"]
        f4_2["4.2 Hyperparameter tuning"]
        f4_3["4.3 Baseline predictions"]
        f4_4["4.4 GT iteration"]
        f4_5["4.5 Obj2 pipeline integration"]
    end

    subgraph PH5["Phase 5 — Evaluation"]
        f5_1["5.1 GNN metrics"]
        f5_2["5.2 Baseline metrics"]
        f5_3["5.3 Baseline vs GNN comparison"]
        f5_4["5.4 GT sanity-check"]
        f5_5["5.5 Obj2 eval"]
        f5_6["5.6 Guide checkpoint"]
    end

    subgraph PH6["Phase 6 — Integration"]
        f6_1["6.1 Dashboard skeleton"]
        f6_2["6.2 Graph-state layer"]
        f6_3["6.3 Distress marker layer"]
        f6_4["6.4 Polish + writeup"]
    end

    subgraph PH7["Phase 7 — Objective 3 (OPTIONAL)"]
        f7_1["7.1 Vulnerability index"]
        f7_2["7.2 OR-Tools optimizer"]
        f7_3["7.3 Allocation plan"]
    end

    subgraph PH8["Phase 8 — Docs & Report"]
        f8_1["8.1 Final report"]
        f8_2["8.2 Slide deck"]
        f8_3["8.3 Demo rehearsal"]
        f8_4["8.4 Code cleanup"]
    end

    f0_1 --> f0_2
    f0_3 --> f0_4
    f0_4 --> f0_5
    f0_4 --> f0_6
    f0_5 --> f0_7
    f0_6 --> f0_7

    f0_7 --> f1_1
    f0_7 --> f1_2
    f0_7 --> f1_3
    f1_2 --> f1_4
    f0_7 --> f1_5
    f1_5 --> f1_6
    f0_7 --> f1_7
    f0_7 --> f1_8
    f0_7 --> f1_9
    f0_2 --> f1_10
    f1_10 --> f1_11
    f1_11 --> f1_12
    f0_7 --> f1_13
    f1_1 --> f1_14
    f1_3 --> f1_14

    f1_1 --> f2_1
    f1_2 --> f2_1
    f1_4 --> f2_1
    f2_1 --> f2_2
    f2_2 --> f2_3
    f1_5 --> f2_3
    f1_6 --> f2_3
    f2_3 --> f2_4
    f1_7 --> f2_5
    f1_8 --> f2_5
    f2_5 --> f2_6
    f2_2 --> f2_6
    f2_3 --> f2_7
    f2_6 --> f2_7
    f2_7 --> f2_8
    f1_14 --> f2_9
    f2_9 --> f2_10
    f1_13 --> f2_11

    f1_9 --> f3_1
    f0_5 --> f3_1
    f0_6 --> f3_2
    f2_9 --> f3_3
    f3_1 --> f3_3
    f3_2 --> f3_3
    f3_3 --> f3_4
    f2_2 --> f3_4
    f2_3 --> f3_5
    f3_5 --> f3_6
    f2_8 --> f3_7
    f2_11 --> f3_8

    f3_4 --> f4_1
    f3_7 --> f4_1
    f4_1 --> f4_2
    f3_6 --> f4_3
    f3_4 --> f4_3
    f4_1 --> f4_4
    f2_10 --> f4_5
    f3_8 --> f4_5

    f4_2 --> f5_1
    f4_3 --> f5_2
    f5_1 --> f5_3
    f5_2 --> f5_3
    f5_3 --> f5_4
    f4_5 --> f5_5
    f5_3 --> f5_6
    f5_5 --> f5_6

    f5_3 --> f6_1
    f6_1 --> f6_2
    f6_1 --> f6_3
    f5_5 --> f6_3
    f5_3 --> f6_4

    f6_4 -.-> f7_1
    f7_1 -.-> f7_2
    f6_3 -.-> f7_2
    f7_2 -.-> f7_3

    f6_4 --> f8_1
    f8_1 --> f8_2
    f6_1 --> f8_3
    f6_2 --> f8_3
    f6_3 --> f8_3
    f6_4 --> f8_3
    f8_3 --> f8_4

    style f7_1 stroke-dasharray: 5 5
    style f7_2 stroke-dasharray: 5 5
    style f7_3 stroke-dasharray: 5 5
```

## 3. Who's doing what, when (swimlanes)

```mermaid
gantt
    dateFormat  YYYY-MM-DD
    axisFormat  W%W
    title 12-Week Team Timeline
    section P1 Geospatial/Graph
    Setup & city/ward decision      :s1, 2026-09-01, 7d
    OSM roads+drainage+wards        :p1a, after s1, 14d
    Line graph + static features    :p1b, after p1a, 14d
    Baseline model                  :p1c, after p1b, 14d
    Baseline eval                   :p1d, after p1c, 7d
    Dashboard build                 :p1e, after p1d, 7d
    Docs & report                   :p1f, after p1e, 21d

    section P2 Ground Truth/Data
    Setup & verification checks     :s2, 2026-09-01, 7d
    DEM+rainfall+GEE access         :p2a, after s2, 14d
    Rainfall feature aggregation    :p2b, after p2a, 14d
    Satellite+news ground truth     :p2c, after p2b, 14d
    GT iteration support            :p2d, after p2c, 7d
    Writeup support                 :p2e, after p2d, 7d
    Docs & report                   :p2f, after p2e, 21d

    section P3 ML/GNN
    Setup & GPU/PyG env              :s3, 2026-09-01, 7d
    PyG Temporal prototyping         :p3a, after s3, 14d
    Model schema + data loader       :p3b, after p3a, 14d
    Training harness                 :p3c, after p3b, 14d
    GNN training + tuning            :p3d, after p3c, 14d
    Eval + comparison                :p3e, after p3d, 7d
    Docs & report                    :p3f, after p3e, 21d

    section P4 NLP/Obj2(+3)
    Setup & social data scoping      :s4, 2026-09-01, 7d
    Social post collection           :p4a, after s4, 14d
    Gazetteer + fuzzy match          :p4b, after p4a, 14d
    Hand-labeling + fine-tuning      :p4c, after p4b, 14d
    Full Obj2 pipeline integration   :p4d, after p4c, 14d
    Obj2 eval + dashboard layer      :p4e, after p4d, 7d
    Objective 3 (optional)           :crit, o3, after p4e, 7d
    Docs & report                    :p4f, after o3, 14d
```

**Sync points (all 4 people meet):** end of Phase 0 (go/no-go on city/event), end of Phase 3 (ground truth ready), end of Phase 5 (mid-project guide checkpoint), Phase 6 (integration — everyone's output merges), end of Phase 8 (final).

---

## 4. Phase 0 — Setup & Foundation (Week 1)

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 0.1 | Create GitHub repo + local scaffold | P1 | — | ✅ |
| 0.2 | Set up shared Python environment (`requirements.txt`, venv/conda instructions in README) | P1 | 0.1 | ✅ |
| 0.3 | Finalize study city + select 10–20 contiguous wards | All | — | ✅ |
| 0.4 | Confirm flood event date range (default: Chennai, Nov–Dec 2015 — see working.md §1.9) | All | 0.3 | ✅ |
| 0.5 | Verify a usable Sentinel-1 pass exists near event dates over chosen wards (Copernicus Browser) | P2 | 0.4 | ✅ |
| 0.6 | Verify the Bhuvan RISAT flood-footprint layer is actually exportable, not just a figure caption | P2 | 0.4 | ✅ |
| 0.7 | **Go/no-go decision:** confirm Chennai, or fall back to Bengaluru 2022 | All | 0.5, 0.6 | ✅ |

**Feasibility check results (31 Aug 2026)** — 0.3–0.6 completed and verified, not just assumed; full evidence (polygon-adjacency map, live Sentinel-1 catalog query, Bhuvan accessibility check) is in the [Chennai Study-Area Validation report](https://claude.ai/code/artifact/13744b14-c706-4bde-acf0-0042146e0129):
- **Study wards (16, geometrically confirmed contiguous):** 142, 168, 169, 170–182 (Adyar river corridor — Saidapet down through Kotturpuram/Adyar to Taramani/Velachery). No outliers found; the 168/169 link is a narrow neck worth re-checking once OSM roads land in 1.1.
- **Event window:** 8 Nov – 14 Dec 2015, with **30 Nov – 2 Dec 2015** as the Peak phase anchor (rainfall max + Chembarambakkam reservoir release into the Adyar river).
- **Sentinel-1:** 4 confirmed S1A passes covering the full study area — 12 Nov, 24 Nov, 6 Dec, 18 Dec 2015. None fall inside the peak window itself (nearest is +4 days); Nov 24 → Dec 6 is the usable pre/post change-detection pair. This is now Objective 1's **primary** ground-truth path.
- **Bhuvan/RISAT:** downgraded from "verify" to **at-risk / secondary only** — the footprint is real (cited in literature) but the live portal has no 2015 historical archive and flood layers are raster WMS tiles with no confirmed vector export. Not to be relied on as the primary ground-truth source; pursue via a formal NRSC request in parallel with Phase 1, off the critical path.

`working.md` §1.5, §1.8, and §1.9 have been updated to reflect these verified results in detail.

**0.7 — GO, confirmed 31 Aug 2026.** Proceeding with Chennai, the 16-ward cluster, and the 8 Nov–14 Dec 2015 window, with Sentinel-1 (not Bhuvan) as Objective 1's primary ground truth. This closes Phase 0 based on the feasibility report above; if the rest of the team hasn't reviewed it yet, flag anything that changes their mind before Phase 1 work goes too far.

**0.6 update (01 Sep 2026)** — a teammate obtained an NRSC/ISRO PDF report directly from the Bhuvan portal ("Hydrological Simulation Study of Flood Disaster in Adyar and Cooum Rivers," v1.2, 07 Dec 2015). **Correction: this is not RISAT data** — it's a modeled rainfall-runoff/DEM hydrological simulation, not an observed SAR flood footprint, despite where it was found. Its rainfall/discharge timeline (peaks on 23 Nov and 1 Dec 2015) independently cross-validates the event dates already confirmed above. Its flood-depth map (Fig. 8) was georeferenced (`src/ground_truth/georeference_nrsc_simulation.py`) into an approximate raster clipped to the study wards — RMSE ~563m (coarse; it's an oblique 3D screenshot, not an orthophoto), but it cross-checks well structurally against the independently-sourced OSM waterway layer (see `src/ground_truth/README.md`, `data/raw/bhuvan/qa_overlay.png`).

**Follow-up (01 Sep 2026): tracked the PDF's provenance back to a real historical-flood archive** at `bhuvan-app1.nrsc.gov.in/disaster/usrtasks/flood/flood.php` — a year/state layer tree that **does** list Chennai-2015 entries, including an actual RISAT-1 cumulative-inundation layer and a Cartosat-2 post-event layer. This **corrects the 31 Aug finding that the portal has "no 2015 historical archive"** — it exists, we just hadn't found the right URL. However, tested directly at the WMS protocol level (`src/ground_truth/verify_bhuvan_wms.py`, traces each checkbox through the portal's own JS to the real backend and issues a `GetMap` request): every one of those layer names is dead — `ServiceException: invalid layer` or blank placeholder tiles. The UI still shows them; the backing data has been pruned or moved.

**Verdict unchanged, now on much firmer evidence: Bhuvan/RISAT stays secondary/cross-check only, not primary ground truth.** This doesn't reopen 0.6 or 0.7, it replaces a UI-navigation-based finding with a protocol-level-tested one that happens to reach the same conclusion. `working.md` §1.5/§1.8/§6 updated accordingly.

**Exit criterion for Phase 0 — met.** Phase 1 is now unblocked: OSM extent = the 16-ward cluster, rainfall window = 8 Nov–14 Dec 2015, satellite passes = the 4 confirmed Sentinel-1 dates above.

---

## 5. Phase 1 — Data Acquisition (Weeks 2–3, parallel)

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 1.1 | Pull OSM road network for study wards (`osmnx`) | P1 | 0.7 | ✅ |
| 1.2 | Pull OSM drainage/waterway layer for study wards | P1 | 0.7 | ✅ |
| 1.3 | Pull/verify ward boundary polygons (DataMeet/BBMP/GCC per city) | P1 | 0.7 | ✅ |
| 1.4 | Manual coverage check of drainage tagging completeness in chosen wards | P1 | 1.2 | ✅ |
| 1.5 | Download SRTM DEM tiles for study area (OpenTopography) | P2 | 0.7 | ✅ |
| 1.6 | Derive slope raster (`gdaldem slope`) | P2 | 1.5 | ✅ |
| 1.7 | Pull hourly rainfall for event window (Open-Meteo API) | P2 | 0.7 | ✅ |
| 1.8 | Pull daily IMD rainfall as cross-check (`imdlib`/`imddaily`) | P2 | 0.7 | ✅ |
| 1.9 | Set up Google Earth Engine access + Sentinel-1 GRD collection query | P2 | 0.7 | ✅ |
| 1.10 | Set up Colab/Kaggle GPU environment | P3 | 0.2 | ✅ |
| 1.11 | Install & smoke-test PyTorch Geometric + PyTorch Geometric Temporal | P3 | 1.10 | ✅ |
| 1.12 | Prototype A3TGCN vs MPNN-LSTM on toy/synthetic graph data | P3 | 1.11 | ✅ |
| 1.13 | Collect social media posts scoped to study wards + event window | P4 | 0.7 | ✅ |
| 1.14 | Draft initial gazetteer (road/locality/landmark names) | P4 | 1.1, 1.3 | ✅ |

**Note the one cross-track dependency in this "parallel" phase:** 1.14 needs 1.1 and 1.3 to exist first (the gazetteer is built from OSM data). Everything else in Phase 1 is fully independent — this is the biggest parallelism window in the whole project.

**1.9 done (01 Sep 2026).** GEE project: **`flood-intelligence-507219`** (noncommercial/research access) — this is the project ID everyone on the team should pass to `ee.Initialize(project=...)` for GEE work going forward. `src/ground_truth/query_sentinel1_gee.py` confirmed all 4 expected Sentinel-1 scenes are reachable in GEE, exact match to the earlier Copernicus catalog check:

| Date | GEE image ID |
|---|---|
| 12 Nov 2015 | `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20151112T003120_20151112T003149_008564_00C241_37E8` |
| 24 Nov 2015 (pre-event, task 3.1) | `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20151124T003120_20151124T003149_008739_00C723_AB80` |
| 06 Dec 2015 (post-event, task 3.1) | `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20151206T003120_20151206T003149_008914_00CC1F_BEF9` |
| 18 Dec 2015 | `COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20151218T003119_20151218T003148_009089_00D0E6_767A` |

**1.10–1.12 done (01 Sep 2026).** No Colab/Kaggle needed — the study laptop has a local NVIDIA RTX 2050 (4GB VRAM), plenty for a ~17k-node line-graph, so the GPU/PyG stack was set up locally instead: `torch==2.13.0+cu126`, `torch_geometric==2.8.0.post1`, `torch_geometric_temporal==0.56.2`. One real snag worth knowing about: `torch-geometric-temporal`'s pinned `torch-scatter`/`torch-sparse` deps have no prebuilt wheel for this torch/Python combo and would need a full MSVC+CUDA toolkit to build from source, so it was installed with `--no-deps` plus a small documented import shim (PyG's own fallback `SparseTensor` class stood in for the one thing `torch_sparse` is needed for — an unused model, `EvolveGCNH` — never a real functional gap). `src/models/gnn/toy_gnn_prototype.py` then ran a real forward+backward pass through both candidate architectures (A3TGCN, MPNN-LSTM) on synthetic data on the GPU — both completed cleanly with gradients flowing. Full details, exact commands, and the shim explanation: `src/models/gnn/README.md`. **This only proves the environment works — no real features, no real labels, no training yet; that's task 2.1+.**

**1.14 done (01 Sep 2026).** `src/nlp/build_gazetteer.py` — 2,168 unique named entries (1,758 roads, 307 landmarks, 89 localities, 14 waterways) across the study wards, 88 with a Tamil name pulled straight from OSM (`name:ta`). This is what task 2.10's fuzzy geoparser will match distress-message place mentions against. Hit and fixed one real environment snag along the way: this laptop sits behind a Sophos TLS-inspection proxy that Python's `certifi` bundle doesn't trust by default, breaking Overpass API calls with `SSLCertVerificationError` — fixed with `pip install pip-system-certs` (defers to the Windows system cert store instead of disabling verification). Full writeup: `src/nlp/README.md`. Still a **draft** — task 2.9 finalizes it once 1.13's real distress text shows which names people actually use.

**1.13 done (01 Sep 2026), with a scope change worth knowing about.** Live social-media collection turned out not to be feasible and this was verified, not assumed: X/Twitter historical search is Enterprise-only (~$42k/month, no research tier since the 2023 Academic API shutdown), and no redistributable public dataset of 2015 Chennai flood tweets exists. Flagged this to the team and got a direct decision: build the corpus from real, publicly accessible reporting instead (ReliefWeb sitreps, an academic retrospective, news coverage with resident interviews), filtered to passages mentioning a task 1.14 gazetteer place. `src/nlp/collect_distress_text.py` — 19 location-mentioning passages from 7 sources, including real first-person resident accounts (Velachery, Kotturpuram). **Disclosed limitation:** this is real text about the real event, but news/report register, not informal social media — task 2.11's hand-labeling works with this register; revisit only if it turns out to matter. Full writeup: `src/nlp/README.md`.

**Phase 1 is now fully complete (1.1–1.14).** Next: Phase 2 (graph & features) — building the actual line-graph from the OSM road network (task 2.1) and attaching the static/dynamic features gathered across P1-P4.

**1.5–1.8 done (01 Sep 2026)** — see `src/ground_truth/fetch_dem.py`, `fetch_rainfall.py`, `fetch_rainfall_imd.py`. DEM/slope came back physically sensible (elevation 0-40m dropping to the coast, St. Thomas Mount visible as a real local high point). **Rainfall did not** — Open-Meteo and IMD disagree by 2-6x on magnitude, and neither cleanly matches the ~240mm/~340-490mm peaks the literature claims for 23 Nov / 1 Dec. Flagged in detail in `working.md` §1.8 with a recommended fix for task 2.5 (IMD for magnitude, Open-Meteo for hourly shape) — **don't aggregate rainfall for the model without reading that note first.**

**1.1–1.4 done (31 Aug 2026)** — see `src/graph/fetch_osm.py` and `src/graph/README.md` for the script, outputs, and the 1.4 coverage-check writeup. Headline numbers: 6,971 road nodes / 17,195 edges (99.1% in one connected component); 39 waterway features, sparsely tagged inside wards as `working.md` §1.8 already anticipated — coverage concentrated along the ward-boundary river/canals rather than interior storm drains. Sanity-check map: `docs/figures/phase1_osm_coverage.png`. Outputs live in `data/raw/wards/` and `data/raw/osm/` (gitignored — rerun the script to regenerate). `requirements.txt`'s `osmnx`/`geopandas`/`networkx` are now confirmed installable and working on this machine.

---

## 6. Phase 2 — Graph Construction & Features (Weeks 3–4)

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 2.1 | Build primal graph (`networkx`) from OSM road+drainage data | P1 | 1.1, 1.2, 1.4 | ✅ |
| 2.2 | Transform primal → line graph (segment = node) | P1 | 2.1 | ✅ |
| 2.3 | Compute static node features: elevation, slope, length, distance_to_drain | P1 | 2.2, 1.5, 1.6 | ✅ |
| 2.4 | *(Optional)* Compute impervious %, ward population density | P1 | 2.3 | ☐ |
| 2.5 | Aggregate rainfall into phase-window dynamic features (`rainfall_t`, `cumulative_rainfall_t`) | P2 | 1.7, 1.8 | ✅ |
| 2.6 | Attach dynamic features to graph nodes per timestep | P2 | 2.5, 2.2 | ✅ |
| 2.7 | Define model input schema (`X_t`, `edge_index`, `Y_{t+1}` shapes) | P3 | 2.3, 2.6 | ✅ |
| 2.8 | Build graph-snapshot dataset/data-loader class | P3 | 2.7 | ✅ |
| 2.9 | Finalize gazetteer (`name → coordinate` dict) | P4 | 1.14 | ✅ |
| 2.10 | Implement fuzzy string matching pipeline (`rapidfuzz`) against gazetteer | P4 | 2.9 | ✅ |
| 2.11 | Hand-label distress/not-distress dataset (few hundred posts) | P4 | 1.13 | ✅ |

**2.1, 2.2, 2.5, 2.6 done (17 Sep 2026)** — merged via PR [#1](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/1) (2.1, `src/graph/build_primal_graph.py`), [#2](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/2) (2.2, `src/graph/build_line_graph.py`), [#3](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/3) (2.5, `src/features/build_rainfall_phase_features.py`), [#4](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/4) (2.6, `src/features/attach_dynamic_node_features.py`). All four re-run end-to-end against the real committed Phase 1 data before merging, not just trusted from PR descriptions: primal graph reproduces Phase 1's recorded 6,971 nodes/17,195 edges exactly; line graph comes out to 17,195 nodes/48,732 edges with a clean segment_id bijection, 0 duplicate edges, 0 junction inconsistencies; bias-corrected rainfall_t lands at 1055.7/31.2/410.1/126.0mm across pre_event/rising/peak/receding (peak within the literature's ~340–490mm range) with cumulative_rainfall_t as the running sum; dynamic features broadcast correctly to all 17,195 segments × 4 phases (68,780 rows). PR #3 needed one fix before merging — it called `pd.option_context("mode.use_inf_as_na", ...)`, an option pandas 3.0 removes outright, which would crash on any fresh `pip install` since `requirements.txt` pins no pandas version; simplified to the existing `.where()` guard (verified byte-identical output) and pushed directly to the PR branch.

**Note on 2.5's phase-window totals:** pre_event's total (1055.7mm over its 20-day window) is larger than peak's (410.1mm over 2 days) simply because the windows are very different lengths — this doesn't contradict peak being correct (410mm matches the literature), but the script's own printed "phase 2 should now be the clear maximum" sanity message is misleading since it's comparing un-normalized window totals. Cosmetic only, not asserted in code, doesn't affect downstream values — worth fixing the message (or switching to average intensity) whenever 2.5 is next touched, no need to reopen now.

**2.3 done (19 Sep 2026)** — merged via PR [#5](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/5) (`src/graph/build_static_features.py`). Re-run end-to-end against real committed Phase 1/2.1/2.2 data before merging (2.1/2.2 regenerated locally since `data/processed/` is gitignored), reproducing the PR's own claimed numbers exactly: 17,195 segments, 0 missing elevation/slope/distance_to_drain values, 0 segments with zero valid raster samples. Elevation came out 0–20m (mean 8.2m) with one segment averaging slightly negative (-3.7m) near the coast/backwaters — consistent with the raw SRTM DEM's own min of -19m (task 1.5), not a bug in this PR. distance_to_drain ranged 0–2091m (mean 498m), consistent with 1.4's finding that drainage tagging is sparse and concentrated along ward-boundary waterways rather than interior segments.

**2.7 done (19 Sep 2026)** — merged via PR [#6](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/6) (`src/models/gnn/build_model_input_schema.py`). Re-run end-to-end against real committed Phase 1/2.1/2.2/2.3/2.5/2.6 data before merging (2.1/2.2/2.5/2.6 regenerated locally since `data/processed/` is gitignored): `X.npy` shape `(4, 17195, 6)`, `edge_index.npy` shape `(2, 48732)` (reused 2.2's own `node_order()`/`edge_index_array()`, directed per that task's traversal-direction design), 0 missing values across all 6 feature columns, edge indices all in range. `X[0,0,:4]` spot-checked byte-for-byte against `static_features.geojson`'s row for the same segment. Real `Y_{t+1}` labels don't exist yet (task 3.4/Phase 3 hasn't started) — no label file was fabricated; `schema.json` documents the shape/dtype/join-key contract 3.4 must satisfy (3 usable transitions: pre_event→rising, rising→peak, peak→receding).

**Note on `working.md` line 90 ("F = 7 features"):** that line counts §1.7's MUST-HAVE list literally, which bundles "road topology" (structural, = edge_index) and "flood label" (= Y_{t+1}) in with the 5 real per-node scalars, and doesn't itemize `length_m` separately even though §1.3's own node-feature list includes it. Actual per-node feature count in `X_t` is F=6, not 7 — cosmetic doc mismatch, not a functional gap, not fixed upstream, just flagged in the script's docstring.

**2.8 done (19 Sep 2026)** — merged via PR [#7](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/7) (`src/models/gnn/build_graph_snapshot_dataset.py`). `GraphSnapshotDataset` wraps 2.7's outputs into one `Data(x, edge_index, y)` per phase, plus `.to_a3tgcn_input()`/`.to_mpnn_lstm_input()` reproducing task 1.12's `toy_gnn_prototype.py` shape conventions exactly. Validated beyond shape-checking: ran a real forward pass through the actual `A3TGCN`/`MPNNLSTM` model classes (not synthetic data) on the full real 17,195-node graph via this loader — both completed cleanly on GPU, output shapes `(17195, 1)` and `(17195, 25)` (matches `MPNNLSTM`'s own `2*hidden+in_channels+periods-1` formula). `attach_labels()` is ready to accept task 3.4's fused labels once they exist — validated against schema.json's `y_t1_contract` — but no labels are attached yet since Phase 3 hasn't started.

**2.9 done (19 Sep 2026)** — merged via PR [#10](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/10) (`src/nlp/finalize_gazetteer.py`). Re-fetched all 146 paragraphs from task 1.13's source pages (not just the 19 that survived that task's gazetteer-match pre-filter, which structurally can't reveal a missing name) and extracted 234 novel 2+-word candidate place names not already in the 1,787-name draft. Checking them against live OSM hit two real infrastructure problems, both fixed in this PR: a single 234-name query hit `413 Request Entity Too Large` on the one mirror that accepted a connection (fixed by batching into groups of 25), and two of three public Overpass mirrors (`overpass-api.de`, its `lz4` load-balanced node) consistently failed to connect at all on this network throughout development (fixed by trying the working mirror, `overpass.kumi.systems`, first). Even fixed, that connection stayed intermittently flaky — 6 of 10 batches failed on every mirror/attempt in the merged run. Net result: **109/234 candidates actually checked, 5 resolved to a real OSM feature** (Chennai Corporation, Gandhi Road, Kuberan Nagar, Royal Enfield, World Bank), 104 confirmed no match, **125 honestly recorded as unverified** (not confirmed-absent) rather than blocked on or faked.

**Two of the 5 resolved matches are probably not what the source text meant, kept anyway on purpose:** "World Bank" and "Royal Enfield" most likely matched a coincidentally-named local shop and a motorcycle showroom rather than the international org / rescue-vehicle brand the distress text almost certainly referenced. Discussed directly — kept in the output rather than hand-excluded, since picking off inconvenient real OSM matches after the fact would just be the hand-curated semantic denylist this script's design was built to avoid (see its docstring's "Method" step 4). Flagged clearly for a human spot-check, same as task 1.14's own draft was.

**2.10 done (19 Sep 2026)** — merged via PR [#11](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/11) (`src/nlp/fuzzy_geoparse.py`). Sliding n-gram window + `rapidfuzz` fuzzy match against 2.9's gazetteer + overlap resolution (one text span → one place). Tried `fuzz.WRatio` (rapidfuzz's own default) first and rejected it on real data: its partial-containment scoring made generic single words spuriously match any longer name containing them ("road" vs "Gandhi Road" → 90.0) — a real risk given ~1,758 of the gazetteer's entries are literally "`<name>` Road"/"`<name>` Nagar". Plain `fuzz.ratio` fixed that (53.3, 62.5 for the same pairs) while still catching real typos ("tansi ngr" vs "tansi nagar" → 90.0); `SCORE_CUTOFF=85` checked against ordinary English words with none scoring above it. **Validated against task 1.13's real corpus, not synthetic examples:** initial raw recall vs. the exact-substring baseline was 87.1% (27/31) — traced the 4 "misses" to the baseline double-counting one mention twice (e.g. "the Adyar river" independently substring-matches both "Adyar" and "Adyar River"; this pipeline's overlap resolution correctly keeps only the more specific one), and after excluding that artifact recall is **100%**, plus **5 genuine extra matches** (typo/spacing variants, and "Kuberan Nagar" — a name 2.9 added to the gazetteer after this corpus was originally built).

**2.11 done (20 Sep 2026)** — merged via PR [#19](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/19) (`src/nlp/label_distress_dataset.py`). Task description's "few hundred posts" scoping referred to the original live-social-media plan that 1.13 already documented as infeasible; what this actually hand-labels is 1.13's real 19-passage corpus, disclosed up front rather than silently reinterpreted. The script is a labeling tool, not a labeler — it presents each passage to a human one at a time and records their DISTRESS/NOT_DISTRESS/UNCERTAIN judgment, with stable text-hash passage IDs (so re-running 1.13's collector can't silently orphan a label) and a write-after-every-label design so nothing is lost on interruption. All 19 real passages hand-labeled: 8 distress, 9 not_distress, 2 uncertain. Verified before merging: passage IDs match the real corpus exactly (19/19, no drift), and spot-checking the actual label choices against the underlying text confirms the codebook was applied consistently — firsthand/directly-reported acute impact (evacuated by boat, water inside homes, families stranded) marked distress; political statements, institutional announcements, and general drainage-system reporting marked not_distress even when the surrounding article is flood-related. 17 new tests, 243 passing repo-wide (no regressions).

**Next:** 2.4 (optional impervious %/ward density) and Phase 3 (ground truth fusion — 3.3 news cross-check is now fully unblocked, feeding 3.4, the critical-path exit criterion per developing.md §7) are the open items.

---

## 7. Phase 3 — Ground Truth Construction (Weeks 4–5)

This is the highest-risk phase in the project (see working.md §1.5) — treat 3.1–3.4 as the critical path.

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 3.1 | Run Sentinel-1 SAR change detection in GEE (before/after backscatter) | P2 | 1.9, 0.5 | ✅ |
| 3.2 | Extract Bhuvan RISAT flood footprint (if Chennai chosen) | P2 | 0.6 | ✅ |
| 3.3 | Cross-check with news/traffic advisories; geocode named roads via gazetteer | P2 | 2.9, 3.1, 3.2 | ✅ |
| 3.4 | **Fuse into 4-phase label scheme** (Pre-event / Rising / Peak / Receding) per segment | P2 | 3.3, 2.2 | ✅ |
| 3.5 | Freeze graph structure (no more topology changes after this point) | P1 | 2.3 | ✅ |
| 3.6 | Build rule-based baseline propagation model | P1 | 3.5 | ✅ |
| 3.7 | Build training/eval harness skeleton (phase-based train/test split, metrics) | P3 | 2.8 | ✅ |
| 3.8 | Fine-tune MuRIL/IndicBERT distress classifier on labeled data | P4 | 2.11 | ✅ |

**Exit criterion for Phase 3:** 3.4 checked off — this is the mid-project sync point. Phase 4 cannot meaningfully start without fused ground truth.

**3.1 done (19 Sep 2026)** — merged via PR [#8](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/8) (`src/ground_truth/run_sentinel1_change_detection.py`). Pre/post VV backscatter change (24 Nov pre → 6 Dec 2015 post, task 1.9's confirmed pass pair) thresholded at `mean - 2*std` on the AOI's own change distribution — chosen only after two standard alternatives were tried directly on the real data and rejected: a fixed -17dB literature threshold (UN-SPIDER's common VV default) flagged just 0.08% of valid post-event pixels in this dense-urban scene (near-empty ~0.01 km² result — buildings' double-bounce backscatter runs higher than the open terrain that default assumes), and Otsu's method on the diff histogram flagged 54% of the AOI (no real bimodal split to find; flooding is a minority-class tail on one noisy mode, not two separable populations). Real run against live GEE: **449 flood polygons, 0.60 km² total (~1.3% of the ~46.6 km² AOI)**, touching **all 16 of 16 study wards** — Perungudi (Ward 169) highest at 0.0941 km², Adyar (Ward 181) lowest at 0.0054 km² (corrected 19 Sep 2026 in PR #9, see below — an earlier version of this note wrongly said only 3 wards showed flooding). Cross-checked against task 0.6's georeferenced NRSC simulation raster: mean depth at flood-polygon centroids (2.94m) exceeds random-point depth (2.62m) — corroborating, not proof, given that raster's own ~563m RMSE.

**Read this as a floor, not the full picture:** `working.md` §1.5 already documents SAR's dense-urban-canopy blind spot — 0.60 km² across the whole 16-ward cluster is still a small fraction of the ~46.6 km² AOI, and this before/after pass pair also brackets the 30 Nov–2 Dec peak rather than capturing it (nearest pass is +4 days post-peak). Task 3.3 (news/advisory cross-check) is explicitly the step meant to catch what SAR misses.

**3.2 done (19 Sep 2026)** — merged via PR [#9](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/9) (`src/ground_truth/extract_bhuvan_flood_footprint.py`). Re-verified live (re-ran `verify_bhuvan_wms.py`, not just cited the old result) that Chennai's actual RISAT-1/Cartosat-2 WMS layers are still dead — no exportable RISAT SAR footprint exists for this event, unchanged from task 0.6. Extracted a polygon instead from the one real Bhuvan-sourced artifact this project has: task 0.6's follow-up georeferenced NRSC hydrological-simulation depth raster, thresholded at a 0.1m noise floor (95.8% of the raster already shows >0.1m depth — the source figure's colored region already **is** the claimed inundation zone, not something a threshold discriminates within) and vectorized by reusing 3.1's `sieve_mask()`/`vectorize_mask()`/`clip_to_wards()` directly. Real run: **35 polygons, 21.66 km² total, touching 14 of 16 wards**; cross-checked against 3.1's Sentinel-1 result — 38.9% of the Sentinel-1 flood area falls inside this Bhuvan/NRSC zone, meaningful overlap given the two methods' opposite biases (SAR underestimates; this coarse ~563m-RMSE simulation likely overestimates). **Stays secondary/opportunistic only** (task 0.6's verdict) — 3.3/3.4 should weight Sentinel-1 as primary.

**Bug found and fixed while building 3.2, in already-merged 3.1 code:** `validate_flood_extent()`'s per-ward report keyed flooded area on `Zone_Name` alone, but 13 of the 16 study wards all share `Zone_Name="ADYAR"` (distinguished only by `Ward_No`) — each subsequent Adyar ward's area was silently overwriting the previous one instead of summing, which is why 3.1's original merged note above wrongly said flooding was "concentrated in only 3 of 16 wards." Only that report field was wrong — the flood polygon geometries and 0.60 km² total were never affected. Fixed in a shared `per_ward_area_breakdown()` (keyed on a ward-unique label, accumulates) both scripts now use, with a regression test reproducing the exact real-data shape.

**3.3 done (19 Sep 2026)** — merged via PR [#12](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/12) (`src/ground_truth/cross_check_news_advisories.py`). Geoparses the real distress/news corpus (task 2.10's fuzzy matcher) against task 2.9's gazetteer, resolves each uniquely-mentioned place to line-graph segments (exact OSM `name` match tried first regardless of gazetteer `type`, else every segment in the place's containing ward), and cross-checks against 3.1/3.2's satellite extents. **Bug found and fixed on the first real run:** the gazetteer's `ward_no` column is `float64` (pandas' default once any row has a `NaN`) while `study_wards.geojson`'s `Ward_No` is `int32` — naive string comparison (`"177.0" == "177"` → `False`) meant the ward-fallback resolved **zero segments** for all 22 ward-level places despite them appearing to "resolve." Fixed by comparing as numbers, with a regression test using the exact real-data shape. Real result: 69 resolved mentions across 23 unique places → **13,939 of 17,195 segments news-flagged (81% of the graph)** — broad by design, since the retrospective/encyclopedia sources discuss the flood citywide across nearly all 16 wards, and the ward-level fallback (same coarseness as 3.2) flags every segment in a mentioned ward, not just the specific spot. Cross-checked: 5.6% overlap Sentinel-1, 47.8% overlap Bhuvan/NRSC, and **48.3% (6,736 segments) are news-only** — the concrete "catches what SAR misses" value this task is named for.

**Not phase-resolved, by design:** checked directly whether the real corpus carries per-passage dates — it doesn't (only one of 19 original passages' own source URL has any date range). A news-flagged segment means "flooded at some point during the event," not "flooded in phase X." Task 3.4's fusion has to decide how to use a phase-unresolved signal.

**3.4 done (19 Sep 2026) — Phase 3's exit criterion met** — merged via PR [#13](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/13) (`src/ground_truth/fuse_flood_labels.py`). `ever_flooded = intersects(3.1) OR intersects(3.2) OR flagged-by(3.3)` — working.md's own rule, extended across all three sources. **Two decisions confirmed directly with the user before implementing/finalizing, not silently chosen:** (1) phase assignment — none of the three sources are phase-resolved, so `ever_flooded` gets assigned to **peak and receding only** (pre_event/rising stay dry), grounded in task 2.5's rainfall numbers (rising 31mm/2days vs. peak 410mm/2days, a 13x contrast); (2) whether to keep the literal OR-fusion rule despite it producing a very high flooded rate, or require 2+ sources to agree for a tighter signal — kept the literal OR rule since that's what working.md actually specifies. Real result: SAR (primary) flags 5.4% of segments, Bhuvan 44.4%, news 81.1% → union is **15,038/17,195 segments (87.46%) flooded at peak/receding**, dominated by the two coarse secondary sources rather than the primary one (only 1.4% of segments have all 3 sources agreeing, 40.7% have 2, 45.4% rest on exactly 1). Per-source columns (`sar_flagged`/`bhuvan_flagged`/`news_flagged`/`n_sources_agreeing`) are saved in the output specifically so task 4.x can re-derive a stricter subset later without rerunning this script, if 87% turns out to limit what the GNN can learn.

**Validated beyond "does it run":** ran a real integration check against task 2.8's actual `GraphSnapshotDataset` — `attach_labels()` accepted the fused table and `transition_pairs()` produced exactly the 3 usable `(X_t, Y_{t+1})` pairs `schema.json` promises, each with correct `(17195, 6)`/`(17195, 1)` shapes. Concrete proof, not just an assertion, that Phase 4 (GNN training) can now start.

**3.5 done (19 Sep 2026)** — merged via PR [#14](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/14) (`src/graph/verify_graph_freeze.py`). A checksum-based freeze contract (`FROZEN_FINGERPRINT`: node count, edge count, SHA256 of the sorted segment_id set) committed as code rather than a new data file, since `data/processed/` stays regenerable-only throughout this repo. Not a non-determinism safeguard — the 2.1/2.2 pipeline was re-run 5+ times this session with identical `6,971/17,195` primal and `17,195/48,732` line-graph counts every time — this is a **process commitment**: no more edits to `build_primal_graph.py`/`build_line_graph.py`, the AOI, or an OSM re-pull without deliberately updating the fingerprint and re-verifying everything built on top. Real run: exact match, no drift, and both downstream tables checked (task 2.3's `static_features.geojson`, task 2.7's `node_order.json`) carry exactly the frozen segment_id set. **Graph topology is frozen as of this commit.**

**3.6 done (19 Sep 2026)** — merged via PR [#15](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/15) (`src/models/baseline/rule_based_propagation.py`). Two rules per working.md's own diagram, OR'd together: rainfall-intensity trigger (recomputed from task 2.5's `n_hours` as mm/day, not the raw window-total `rainfall_t` already in `X_t` — that total is distorted by wildly different window lengths, already flagged in the 2.5 note above, and would make pre_event look wetter than peak; `100mm/day` is IMD's own "Heavy" rainfall boundary, peak's 205.1mm/day clears it 2x over) + neighbor-cascading (a flooded graph-adjacent neighbor, checked via **undirected** adjacency since floodwater doesn't respect one-way traffic, unlike 2.2/2.7's directed `edge_index` — AND elevation at/below the real median, 8.5m). Real, genuinely revealing result: **rising→peak: F1=0, 12.5% accuracy** — the baseline completely fails to anticipate the actual flood onset, since a static reactive rule structurally cannot foresee a future rainfall spike from a currently-dry state (the measured version of working.md §1.6's claim that a GNN captures propagation a static baseline misses, not just an assertion of it). peak→receding scores misleadingly well (F1=0.933) mostly because task 3.4's fusion gives both phases the *identical* flooded set, making that transition structurally easy rather than a real test of propagation. Task 4.1's GNN should be compared per-transition against this table, not just the overall aggregate (F1=0.636).

**3.7 done (19 Sep 2026) — Phase 3 is now fully complete** — merged via PR [#16](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/16) (`src/models/gnn/build_training_harness.py`). No train/test split spec exists in working.md (checked directly) — this task's own judgment call: a **ward-level (spatial) split, not a temporal (transition) one**, since there are only 3 usable transitions total and holding out a whole one would remove an entire dynamic-feature context (rainfall_t is broadcast per-phase, task 2.6) from an already-small dataset. Greedy assignment targets 70/15/15 by segment count (not ward count, since wards range 639–1,830 segments); ~2.7% boundary-adjacent segments (465/17,195) get a nearest-ward fallback rather than being dropped. Achieved 70.43/13.15/16.42% — close to target.

**Smoke-tested against task 3.6's real baseline predictions, which surfaced a real limitation worth knowing before trusting this split:** with only 16 wards, val/test each land just ~2 of them, so per-split metrics carry real variance from *which specific wards* get held out, not just model quality. Concretely, on the rising→peak transition alone, the exact same baseline model scores: train accuracy 6.99%, val 53.54%, test 3.54% — a huge swing from ward selection alone. Task 4.2's hyperparameter tuning should account for this (e.g. k-fold across wards) rather than trust a single val split's numbers at face value.

**3.8 done (20 Sep 2026) — Phase 3 is now fully complete** — merged via PR [#21](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/21) (`src/nlp/finetune_distress_classifier.py`). Fine-tunes MuRIL (working.md §2.3) on task 2.11's real hand-labeled data — 17 binary-labeled passages (2 "uncertain" excluded), not working.md's "a few hundred." **Adaptation strategy confirmed with the user, no spec exists at this sample size:** freeze MuRIL's 237M-param body, train only a linear head on frozen mean-pooled embeddings ("linear probing") — full fine-tuning at n=17 would just memorize the training set.

**Real bug found and fixed, same class as task 4.1's GNN bug:** first LOOCV run was completely degenerate (F1=0.0, recall=0.0, predicting one constant output for every example) — raw MuRIL embeddings have tiny per-dimension scale (std ~0.023), starving the linear head's gradient. Ruled out "the embeddings aren't separable" first (1-NN cosine similarity on the same embeddings got 88% accuracy, so the signal was there). Fixed with per-fold, train-only z-score standardization — **LOOCV F1 went from 0.0 to 0.9412** (precision 0.889, recall 1.0, 1 false positive out of 17 folds). Heavily caveated: with n=17, each fold's error swings the metric ~5.9% — a rough estimate, not production confidence. Verified end-to-end on genuinely new text: correctly classified an unseen first-person distress account (p=0.998) and an unseen institutional announcement (p=0.0002). **Disclosed, inherited limitation:** MuRIL's strength is code-mixed Indian-language text; the real corpus is English news/report register (task 1.13's infeasibility pivot) — proves the pipeline architecture, not that MuRIL is the ideal model for this specific text. `predict_distress()` is the reusable inference entry point task 4.5 now wires in.

**Next:** Phase 4 (model training, 4.1+) can now start for real — 3.4's fused labels, 2.8's data loader, 3.6's baseline comparison point, and 3.7's train/val/test split are all ready, and 3.5 confirms the graph itself won't shift under them. **All of Phase 3 (3.1–3.8) is done.**

**4.1 done (19 Sep 2026) — working.md §1.6's core experiment** — merged via PR [#17](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/17) (`src/models/gnn/train_gnn.py`). Temporal framing confirmed with the user before implementing (no working.md spec exists for this): one `A3TGCN(periods=1)`, shared weights, trained by pooling all 3 transitions' `(X_t, Y_t+1)` pairs — matching task 3.6's baseline and task 2.8's `transition_pairs()` design exactly, for an apples-to-apples comparison against the rule-based baseline.

**Two real bugs found and fixed on the first training runs, not hidden:** (1) features were never normalized (raw scales span elevation ~0-20m to distance_to_drain_m ~0-2000m) — the first trained model's output had **exactly zero variance across all 17,195 segments**, verified directly; it had learned to ignore every input and output one constant bias. Fixed with train-split-derived z-score normalization. (2) `pos_weight` was pooled across all 3 transitions instead of computed per transition — pre_event→rising is 100% negative while the other two are ~93% positive (task 3.4), and pooling blended these into one misleading weight. Fixed per-transition.

**Real result — the actual baseline-vs-GNN comparison:**

| Transition | Split | Baseline F1 | GNN F1 |
|---|---|---|---|
| **rising→peak** | train/val/test | 0.0 / 0.0 / 0.0 | **0.79 / 0.44 / 0.93** |
| peak→receding | train/val/test | 0.96 / 0.63 / 0.98 | 0.79 / 0.43 / 0.93 |

**rising→peak (predicting flood onset) is the headline result** — the measured version of working.md's core claim, not just an assertion of it: the baseline completely fails (F1=0 everywhere) because a static reactive rule structurally cannot anticipate a future rainfall spike from a currently-dry state, while the GNN learns real spatial+rainfall signal and gets this dramatically right. **peak→receding is more mixed, disclosed honestly rather than only reporting the flattering transition:** the baseline actually matches or beats the GNN there on train/test, because that transition is structurally easy for any model given task 3.4's fusion assigns peak/receding the *identical* flooded set — the baseline's blanket "rain > threshold → flood everything" trivially matches it.

**4.2 done (19 Sep 2026)** — merged via PR [#18](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/18) (`src/models/gnn/tune_hyperparameters.py`). Directly addresses the small-N-of-wards variance task 3.7/4.1 both already disclosed: **k-fold cross-validation (K=4)** over the train+val wards only — official test wards stay completely untouched until the final step, never used for hyperparameter selection. Grid search over learning rate (0.005/0.01/0.02) via CV, then a decision-threshold sweep on pooled held-out CV predictions from the selected LR, then one final retrain + single test evaluation.

**Real result:** selected **LR=0.02** (CV mean flood-relevant F1: 0.673→0.764→**0.768**, ±0.073 fold std — a real, cross-validated improvement over 4.1's untuned default). Selected **threshold=0.1** (down from the naive 0.5). **Final tuned test F1 on rising→peak and peak→receding: 0.982** — up from 4.1's un-tuned 0.93/0.93 on the same test wards.

**A real, disclosed tradeoff, not hidden:** the same global threshold (0.1), tuned specifically for the flood-relevant transitions, **badly hurts `pre_event->rising`** — which should trivially predict "nothing floods" but now predicts *everything* as flooded (test accuracy 0.0, down from 4.1's trivial 1.0). Expected consequence of one global threshold across transitions with wildly different base rates (0% vs. ~93% positive); `pre_event->rising` was deliberately excluded from the tuning objective since it's uninformative for selecting a flood-relevant threshold, but that also means nothing protected it from the chosen one. Worth a per-transition threshold if this model line is developed further (not done here — scope kept to what 3.7/4.1 explicitly flagged as needing tuning).

---

## 8. Phase 4 — Model Development & Training (Weeks 6–7)

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 4.1 | Train GraphSAGE + temporal layer (A3TGCN/MPNN-LSTM) on fused ground truth | P3 | 3.4, 3.7 | ✅ |
| 4.2 | Hyperparameter tuning | P3 | 4.1 | ✅ |
| 4.3 | Run baseline model predictions across all phases | P1 | 3.6, 3.4 | ✅ |
| 4.4 | Iterate/fix ground-truth issues surfaced during training (feedback loop) | P2 | 4.1 | ✅ |
| 4.5 | Integrate classifier + gazetteer resolution into one end-to-end Objective 2 pipeline | P4 | 2.10, 3.8 | ✅ |

**4.3 done (20 Sep 2026)** — no new script needed: task 3.6's `src/models/baseline/rule_based_propagation.py` already *is* "run the baseline across all phases" end to end (that's where 3.6's own F1 numbers above came from), so this task is the confirmation that its outputs are current against the real, committed pipeline rather than stale from an earlier run. Re-ran it fresh against the latest `data/processed/model_input` (task 2.7) and `fused_flood_labels.csv` (task 3.4): produced `baseline_predictions.csv` (51,585 rows = all 17,195 segments × all 3 usable transitions, one row per segment per transition) and `baseline_evaluation_report.json`. Numbers reproduce exactly, deterministically, what 3.6's note already reported (pre_event→rising F1=0/accuracy=1.0, rising→peak F1=0/accuracy=0.1254, peak→receding F1=0.9331, overall F1=0.6362) — confirming nothing upstream (3.4's fusion, 3.7's split, 2.7's schema) has drifted since 3.6 was merged. This is now the frozen baseline comparison point task 5.2 reads from. **Superseded a few hours later by task 4.4 below — see that note for the corrected `peak→receding` numbers.**

**4.4 done (20 Sep 2026) — merged via PR [#20](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/20) (`src/ground_truth/detect_flood_recession.py`).** The ground-truth feedback loop this task exists for: 3.6/4.1's training runs kept surfacing that `peak→receding` scored suspiciously well (baseline F1=0.933) — traced to task 3.4's own design giving `peak` and `receding` **identical** flood labels, since none of the 3 ground-truth sources are individually phase-resolved. That made the transition a copy-the-input exercise, not a real propagation test.

**Fixed with real, previously-unused data, not a synthetic patch or a threshold tweak:** working.md §1.5 already confirmed a **4th Sentinel-1 pass on 18 Dec 2015** reachable in GEE (12 days after the 6 Dec peak pass) that nothing had used yet. Ran the *exact same* change-detection method as task 3.1 (same `CHANGE_STD_MULTIPLIER`, speckle filter, JRC water mask — imported, not reimplemented) against 24 Nov→18 Dec. A segment with a SAR flood signature at peak but not at 18 Dec has genuinely receded. **Real result: 422 of 932 SAR-flagged peak segments (45.28%) recovered.** `fuse_flood_labels.py` now demotes those to `flood_label=0` in receding only (peak untouched): receding dropped from an exact-copy 87.46% to **85.00%** flooded.

**Disclosed scope limit:** this recession signal only exists for SAR-covered segments (5.4% of the 87.46% "ever flooded" set) — Bhuvan/news are single static snapshots with no time axis, so segments flagged only by those sources are deliberately left unchanged rather than guessed at. A real, partial, proportional fix, not a complete one.

**Re-ran the full downstream chain against the corrected ground truth** (not just this script) to check what actually changed:

| `peak→receding` | Before (identical labels) | After (task 4.4 fix) |
|---|---|---|
| Baseline (3.6/4.3) F1 | 0.933 | 0.919 |
| GNN untuned (4.1) test F1 | 0.93 | 0.928 |
| GNN tuned (4.2) test F1 | 0.982 | **0.973** |

`peak→receding` now scores visibly lower than `rising→peak` (still 0.982) instead of the two being suspiciously tied — a more defensible, presentable result, and the actual reason this was worth fixing rather than just documenting. `pre_event→rising` and `rising→peak` numbers are unaffected (recession-check only touches the receding phase). 16 new tests (8 for `detect_flood_recession.py`'s logic, 8 for `fuse_flood_labels.py`'s demotion behavior); 259 passing repo-wide, no regressions.

**4.5 done (20 Sep 2026) — Phase 4 is now fully complete** — merged via PR [#22](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/22) (`src/nlp/run_objective2_pipeline.py`). Composes task 3.8's classifier and task 2.10's geoparser exactly as working.md §2.4 diagrams: both branches run independently on every post (classification never gates geoparsing or vice versa), merging into the final deliverable — distress posts that also resolved to a coordinate. Neither task is reimplemented, only their existing public functions composed.

**Validation approach, disclosed rather than assumed:** re-measuring either component's own accuracy here would be circular (3.8 already ran LOOCV, 2.10 already measured recall against the real corpus) — what 4.5 needs to prove is that the two *compose* correctly. Checked with 4 genuinely new, hand-written posts (never seen by 3.8's training in any form) covering all 4 combinations of distress-label × has-a-resolvable-place. **All 4 landed correctly:** a real-sounding distress post ("Families in Velachery were trapped on their rooftops...") → distress (p=0.998), resolved to Velachery; a distress post with no place name ("We are stranded...") → distress (p=0.876), correctly resolved to **no** location — a real, disclosed gap (an unlocatable distress signal can't feed the graph/optimizer), not hidden; an institutional post mentioning two real places ("Chennai Corporation announced routine repair work on Gandhi Road...") → not_distress (p=0.007), but still geoparsed to both places — concrete proof the two branches run independently, not gated on each other. Separately, an integration smoke test against task 1.13's real corpus (the classifier's own training data — plumbing check only, not a fresh accuracy number) confirmed 10/19 passages classified distress, 100% of those resolved to a location (expected, since that corpus was already filtered to location-mentioning passages). 8 new tests; 285 passing repo-wide, no regressions.

**Objective 2's full pipeline (task 2.9→2.10→3.8→4.5) is now working end-to-end on real data. Phase 4 (4.1–4.5) is fully complete.**

---

## 9. Phase 5 — Evaluation & Comparison (Week 8)
 
| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 5.1 | Compute F1/accuracy per phase transition — GNN model | P3 | 4.2 | ✅ |
| 5.2 | Compute F1/accuracy per phase transition — baseline model | P1 | 4.3 | ✅ |
| 5.3 | **Baseline vs. GNN comparison** — plots/tables, test the core claim (working.md §1.6) | P1 + P3 | 5.1, 5.2 | ✅ |
| 5.4 | Sanity-check ground truth against any comparison anomalies | P2 | 5.3 | ✅ |
| 5.5 | Evaluate Objective 2 precision/recall (classification + geoparsing accuracy) | P4 | 4.5 | ✅ |
| 5.6 | **Mid-project guide checkpoint meeting** | All | 5.3, 5.5 | ☐ |

**5.1, 5.2 done (20 Sep 2026)** — merged via PR [#23](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/23) (`src/models/gnn/evaluate_final_model.py`, `src/models/baseline/evaluate_per_transition.py`). 5.1 reproduces task 4.2's final tuned GNN model exactly (`train_model()` confirmed deterministic — this script's test-split metrics matched task 4.2's saved numbers byte-for-byte) and evaluates it on all three splits, not just test. 5.2 formalizes the baseline's per-transition/per-split evaluation into its own dedicated, current report (the version embedded in `build_training_harness.py` had gone stale after task 4.4's ground-truth fix); reproduces task 3.7's own reference numbers exactly (rising→peak train/val/test accuracy 6.99%/53.54%/3.54%).

**A real, important finding surfaced while doing this — not hidden:** F1 at one tuned threshold can look excellent purely from matching a transition's base rate, and that's what direct inspection found here — the final tuned GNN predicts "flooded" for **100% of test segments on every transition**, zero exceptions. Added AUC-ROC (threshold-independent, hand-rolled to avoid a new dependency) to check whether this reflects real discrimination. Real result: **AUC ~0.70–0.76 on train/val (genuine signal) but collapses to ~0.46–0.50 on test** (statistically indistinguishable from random) for both flood-relevant transitions. The model learned something real — it just doesn't transfer to whichever 2–3 wards land in the test split. This sharpens task 3.7/4.2's already-disclosed small-N-of-wards variance into a concrete, quantified problem: **the reported F1=0.982/0.973 test "wins" over the baseline are correct arithmetic but do not demonstrate learned per-segment discrimination on held-out wards.** Confirmed with the user this should be reported accurately (not hidden, not fixed prematurely) and flagged for task 5.4, which exists specifically to sanity-check exactly this kind of comparison anomaly. 17 new tests; 302 passing repo-wide, no regressions.

**5.4 done (20 Sep 2026)** — merged via PR [#24](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/24) (`src/ground_truth/sanity_check_comparison_anomalies.py`). Done immediately after 5.1/5.2 rather than after 5.3 (its listed dependency) — the anomaly it needed to sanity-check was already found, and 5.3's comparison/plots would have been built on an unexplained coincidence otherwise.

**Root cause, confirmed with real data, not guessed:** checked the actual per-ward relationship between elevation and the fused flood label across all 16 wards. At the `peak` transition, **only 1 of 16 wards (ward 170) has a minority class ≥10% of its segments** — every other ward, including **both** test wards (169, 182), is >90% one-sided (169: 96.5% flooded, 182: 96.4% flooded). Task 3.4/4.4's fusion (dominated by news' broad ward-level fallback — only 5.4% of the flooded set is SAR-covered) pushes most wards toward near-total inundation, leaving almost no within-ward variance for *any* feature-based model — baseline or GNN — to demonstrate skill against outside that one ward. And that one informative ward landed in **val, not test**, purely by chance: test wards show no meaningful elevation-flood relationship (corr +0.025, +0.166 — near zero and the *wrong* sign versus the physically sensible negative correlation the model actually learned from train/val, -0.137/-0.286).

**Conclusion: the test AUC collapse (5.1) is a real, ground-truth-driven limitation — not a GNN bug, not a coding error.** With only 16 wards and a fusion rule that skews most wards toward "everyone floods," whether an informative ward lands in train/val/test is close to a coin flip. **Deliberately not fixed** — a stratified re-split (by within-ward label variance, not just segment count) is recorded as a recommendation for future work, not retrofitted into this sanity-check task. 9 new tests; 311 passing repo-wide, no regressions.

**Next:** task 5.3's baseline-vs-GNN comparison/plots can now cite this root cause directly rather than present the test AUC collapse as an unexplained anomaly.

**5.3 done (20 Sep 2026)** — merged via PR [#25](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/25) (`src/models/compare_baseline_vs_gnn.py`). Builds the plots/tables testing working.md §1.6's core claim, using 5.1/5.2's reports and citing 5.4's root cause directly rather than presenting an unexplained anomaly.

**A methodological precision correction, found while building this, not hidden:** task 4.2's "final tuned" model was retrained on train+val *combined*, so its val AUC/F1 (cited in 5.1's own note) is **not a fair holdout measurement** — the model was fit to those exact labels. Computed AUC for task 4.1's *original* model instead (trained on "train" only, so val **and** test are genuinely unseen) as the methodologically clean comparison, and appended a correction note to 5.1's README rather than silently rewriting it.

| Model | train AUC | val AUC (fair) | test AUC (fair) |
|---|---|---|---|
| 4.1 original (train-only) | ~0.73–0.78 | **~0.59–0.60** | ~0.42–0.49 |
| 4.2 final tuned (train+val) | ~0.70–0.76 | ~0.70–0.72 (not fair — fit to it) | ~0.46–0.50 |

**Real, honest verdict — not a simple yes/no:** val (the fair holdout): **weakly supported** — AUC ~0.59–0.60 on both flood-relevant transitions, real if modest, landing specifically on ward 170, the one ward 5.4 found has genuine elevation-flood variance. Test: **inconclusive, not negative** — 5.4 already explains why. The baseline has no comparable AUC at all (`predict_transition()` outputs hard 0/1 rules, not a ranked probability) — itself part of the comparison: the GNN can express graded uncertainty a fixed rule structurally cannot, even where post-threshold F1 looks similar (`peak→receding`). Plots: `docs/figures/phase5_f1_comparison.png`, `docs/figures/phase5_auc_fair_holdout.png`. 7 new tests; 318 passing repo-wide, no regressions.

**Phase 5's core comparison is now complete and honestly presentable.** Next: 5.5 (Objective 2 precision/recall) is independent of the GNN/baseline thread; 5.6 is a human checkpoint meeting, not a code task.

**5.5 done (20 Sep 2026)** — merged via PR [#26](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/26) (`src/nlp/evaluate_objective2_precision_recall.py`). Consolidates task 2.10/3.8's piecemeal metrics into one Objective 2-level report, and fills a real gap neither of them checked: geoparsing **precision** — 2.10 only ever measured recall against a known-substring baseline.

**A real false positive found while building this, not hidden:** read all 37 real geoparser matches on task 1.13's corpus in context (not just the known-substring subset 2.10 checked). 36 correct. One is not: **"Nandambakkam"** (a real, distinct Chennai locality, confirmed from its source sentence listing it alongside Guindy/Adyar/Porur/Meenambakkam) is **absent from the gazetteer**, so `fuzz.ratio` fuzzy-matches it to **"Adambakkam"** — a different real place that IS present — purely on 90.9% string similarity, clearing `SCORE_CUTOFF=85`. Disclosed as a documented `KNOWN_FALSE_POSITIVES` entry, not excluded (same precedent as task 2.9's "World Bank"/"Royal Enfield").

**Real result:** classification precision/recall (task 3.8 LOOCV) 0.889/1.0; geoparsing recall (task 2.10) 1.0; geoparsing precision (new) **0.973** (36/37); end-to-end pipeline precision/recall 0.889/1.0. **The end-to-end number matching the classifier-only number is disclosed, not presented as a free lunch:** task 1.13's own collection method only kept passages that already mention a gazetteer place, so geoparsing can't be a bottleneck on *this* corpus by construction — task 4.5's own smoke test already found a real case where it would be, on genuinely new text. 9 new tests; 327 passing repo-wide, no regressions.

**Phase 5 is now fully complete except 5.6, a human checkpoint meeting with the project guide — not a code task. Tell me when it happens and I'll mark it done with whatever notes you want recorded.**

---

## 10. Phase 6 — Integration (Week 9)

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 6.1 | Build Streamlit/Folium dashboard skeleton | P1 | 5.3 | ✅ |
| 6.2 | Add graph-state animation layer (baseline vs. GNN over time) | P1 | 6.1 | ✅ |
| 6.3 | Add distress marker layer from Objective 2 | P4 | 6.1, 5.5 | ✅ |
| 6.4 | Polish model outputs and write up results | P2 + P3 | 5.3 | ☐ |

**This is where Objective 1 and Objective 2 visibly become one system** — the dashboard is the artifact that proves it.

**6.1, 6.2, 6.3 done (20 Sep 2026)** — merged via PR [#27](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/27) (`src/dashboard/app.py`, `src/dashboard/prepare_dashboard_data.py`, `src/dashboard/data_loader.py`). One Streamlit app, 5 tabs: Overview (real KPIs from tasks 3.4/4.2/5.1/5.3/5.4/5.5), Flood Propagation (6.2 — animated map, phase slider + play/pause, toggles ground truth/baseline/GNN across all 17,195 segments), Distress Signals (6.3 — real, pulsing markers for every Objective 2 pipeline output that classified as distress AND resolved to a coordinate), Model Comparison (interactive Plotly recreations of task 5.3), Data & Limitations (task 5.4/5.5's findings as designed insight cards, not hidden behind the polish). `prepare_dashboard_data.py` joins real task outputs into per-phase layers, running task 5.1's already-trained model once (not retraining) for per-segment predictions.

**Two real bugs found and fixed while building this, not hidden:** (1) `streamlit-folium`'s `st_folium()` silently hung with this environment's streamlit/streamlit-folium version pairing — zero iframes ever mounted, no console error. Confirmed via a minimal reproduction before concluding it was a real incompatibility. Fixed by dropping the custom-component dependency entirely — `folium`'s own `_repr_html_()` embedded via Streamlit's built-in `components.html()` needs no handshake, and nothing was lost since the app never used `st_folium`'s click-return functionality. (2) CartoDB's free dark tiles now require an API key (a real provider change) — the map rendered blank with only an easy-to-miss `UserWarning` in the server log. Switched to Esri's dark-gray-canvas tiles, which need no key.

**Verified with a real running instance, not just import-checked:** the Claude-in-Chrome extension wasn't available this session, so installed Playwright and drove the actual launched app — all 5 tabs, all 3 prediction views, multiple phases, screenshots inspected at each step, zero console errors. The flood propagation map alone makes working.md §1.6's core claim visually vivid: at `peak`, ground truth and the GNN both turn the city red; the baseline stays entirely blue, visibly failing to anticipate the flood. 10 new tests; 337 passing repo-wide, no regressions.

**Next:** 6.4 (polish model outputs and write up results) is the last Phase 6 task.

**Correction (20 Sep 2026)** — merged via PR [#28](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/28), found from real user feedback ("the latency is too much and its kind of glitching... going till peak sometimes and coming back to pre event"), not a style preference. The Play animation's server-driven design was structurally broken: each tick rebuilt the whole map and remounted a `components.html()` iframe (`time.sleep()` + `st.rerun()` in a loop). Timed directly with Playwright, not assumed: tick intervals grew **6s → 13s → 25s → 39s → 52s** over one Play run — consistent with iframes/Leaflet instances accumulating rather than being torn down. Separately, mutating `st.session_state.phase_idx` after its widget had already rendered raised `StreamlitWidgetAlreadyInstantiatedError` outright — the actual, confirmed cause of phases jumping to peak and back to pre-event. Fixed by rebuilding the animation entirely client-side: the map is built once per model view, and Play/Pause now call Leaflet's own `setStyle()` per feature from a JS `setInterval()` — zero server round-trips per frame. Also fixed a `ReferenceError` this surfaced (Folium's own JS variable wasn't always defined yet when the injected script ran) with a readiness poll. Re-verified via Playwright: phase order confirmed correct frame-by-frame across a full cycle, tick timing stays flat (no escalation), Pause holds correctly. 335 passing repo-wide (2 fewer — removed a now-dead-code style-function test).

**Enhancement (20 Sep 2026)** — merged via PR [#29](https://github.com/SrivatsalyaBhavaraju/spatiotemporal-flood-intelligence/pull/29), requested directly: "is there a way to show the flow of water?" A literal flow simulation isn't possible — task 5.4 already established the real ground truth has no sub-phase timing/velocity/depth data to animate from. Used real per-segment elevation (task 2.3) instead: phase transitions now repaint newly-flooding segments lowest-elevation-first and newly-receding segments highest-elevation-first, batched into 40 steps over ~800ms, rather than flipping all changed segments instantly. Disclosed as an illustrative, physically-motivated choice grounded in real elevation data, not literal flow physics. Verified via Playwright the sweep is visible (a real partial-red "flood front" mid-transition) and that it doesn't reintroduce PR #28's escalating-latency bug — timed 4 full Play cycles, interval stayed flat throughout.

---

## 11. Phase 7 — Objective 3 (OPTIONAL, Week 10)

> Only start this phase if Phases 1–6 are complete and stable with time to spare. If not — skip straight to Phase 8. The project is fully defensible on Objectives 1 and 2 alone (see working.md §3.3, §7).

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 7.1 | Build ward-level vulnerability index (Census 2011 proxies) | P4 | 6.4 | ☐ |
| 7.2 | Set up OR-Tools/PuLP constrained optimizer | P4 | 7.1, 6.3 | ☐ |
| 7.3 | Produce ranked/allocated resource plan per ward | P4 | 7.2 | ☐ |

---

## 12. Phase 8 — Documentation, Report, Presentation (Weeks 10–12)

| # | Task | Owner | Depends on | Status |
|---|---|---|---|---|
| 8.1 | Write final report | All | 6.4 | ☐ |
| 8.2 | Prepare slide deck | All | 8.1 | ☐ |
| 8.3 | Demo rehearsal | All | 6.1, 6.2, 6.3, 6.4 | ☐ |
| 8.4 | Code cleanup + README finalization | All | 8.3 | ☐ |

---

## 13. Git workflow

- **Branch naming:** `p1/feature-name`, `p2/feature-name`, etc. — prefix by owner track, e.g. `p1/line-graph-transform`, `p3/gnn-training-loop`.
- **No direct commits to `main`.** Open a PR, get at least one other teammate to skim it (doesn't need to be a deep review — mainly a sanity check that it runs), then merge.
- **Commit early, commit often** — small commits scoped to one task ID make it easy to trace "which commit closed 3.4" later, for the report.
- Reference the task ID in commit messages/PR titles where possible, e.g. `2.2: primal to line graph transform`.

## 14. Repo structure

```
spatiotemporal-flood-intelligence/
├── README.md
├── working.md              — technical rationale & research (read this first)
├── developing.md            — this file
├── requirements.txt
├── .gitignore                (data/raw, .venv, __pycache__, checkpoints)
├── data/
│   ├── raw/                  (gitignored — too large to commit)
│   ├── processed/            (gitignored, or small samples only)
│   └── README.md             (how to re-fetch each dataset — see working.md §1.8)
├── src/
│   ├── graph/                 (P1 — OSM extraction, line-graph transform, static features)
│   ├── ground_truth/          (P2 — satellite/rainfall processing, label fusion)
│   ├── models/
│   │   ├── baseline/          (P1/P2 — rule-based propagation)
│   │   └── gnn/                (P3 — GraphSAGE / temporal GNN)
│   ├── nlp/                    (P4 — gazetteer, classifier, geoparsing)
│   ├── optimizer/               (P4 — Objective 3, optional)
│   └── dashboard/
├── notebooks/                  (exploratory work, one subfolder per person)
├── tests/
└── docs/figures/
```

## 15. Milestone checkpoints (guide meetings)

| Milestone | When | What to show |
|---|---|---|
| M1 | End of Phase 0 | City/ward/event finalized, verified feasible |
| M2 | End of Phase 3 | Graph built, fused ground truth ready |
| M3 | End of Phase 5 | Baseline vs. GNN comparison results, Objective 2 metrics |
| M4 | End of Phase 8 | Full demo, report, final presentation |
