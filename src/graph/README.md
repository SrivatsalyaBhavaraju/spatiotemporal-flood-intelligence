# P1 — Geospatial & Graph (OSM extraction, line-graph transform, static features)

## `fetch_osm.py` — tasks 1.1, 1.2, 1.3

Pulls the drivable road network and waterway/drainage layer from OSM for the confirmed
16-ward study cluster (142, 168, 169, 170–182 — see `developing.md` Phase 0 notes and
`working.md` §1.9 for how this list was verified), and saves the ward boundary subset
used as the query polygon.

**Setup:** copy the DataMeet Chennai ward file to
`data/raw/wards/chennai_wards_full.geojson` (source: [Chennai Wards.geojson](https://github.com/datameet/Municipal_Spatial_Data/blob/master/Chennai/Wards.geojson)) — not committed, per `data/README.md`.

**Run:**
```bash
python src/graph/fetch_osm.py
```

**Outputs** (`data/raw/`, gitignored):
| File | Contents |
|---|---|
| `wards/study_wards.geojson` | The 16 study ward polygons, subset from the full DataMeet file |
| `osm/roads.graphml` | Drivable road network as an osmnx/networkx `MultiDiGraph` |
| `osm/roads_edges.geojson` | Same network flattened to an edge GeoDataFrame |
| `osm/waterways.geojson` | `waterway=*` features (drain/ditch/stream/canal/river), **clipped to the AOI** |
| `osm/coverage_summary.json` | Node/edge counts, connectivity, waterway tag breakdown |
| `osm/coverage_map.png` | Quick sanity-check plot (also copied to `docs/figures/phase1_osm_coverage.png`) |

**⚠️ Clipping matters:** `osmnx.features_from_polygon` does not clip returned ways to the
query polygon — a way that only partially crosses the AOI (e.g. a long river) comes back
in full. The script clips waterway geometries to the AOI explicitly before saving; don't
remove that step or a single long way can dwarf the study area in any downstream analysis.

## Task 1.4 — drainage coverage check (result, 31 Aug 2026)

Run against the confirmed cluster:

- **Roads:** 6,971 nodes / 17,195 edges. 20 weakly connected components, but the largest
  holds 6,911/6,971 nodes (99.1%) — the rest are small edge-truncation artifacts from
  clipping to the ward polygon, not a real network gap.
- **Waterways:** 39 features total — 25 `drain`, 10 `canal`, 3 `stream`, 1 `river`. Visually
  (`docs/figures/phase1_osm_coverage.png`), coverage is concentrated along the ward
  perimeter (the Adyar river and its main canals largely trace the ward boundaries
  themselves) with very sparse tagging of internal neighborhood storm drains — one long
  `river` way plus a handful of `canal`/`drain` ways per ward, not the dense micro-drainage
  network that actually exists on the ground.
- **Verdict: matches the known risk already flagged in `working.md` §1.8** — OSM drainage
  tagging in Indian cities is confirmed sparse/inconsistent, not a surprise specific to this
  cluster. Proceed, but don't expect `distance_to_drain` (task 2.3) to be meaningfully
  differentiated in wards with near-zero interior tagging; the DEM-derived slope/elevation
  features will be carrying more of the signal there than drainage proximity will.

## Next in this track

- **1.14** (P4, blocked on this): gazetteer draft needs `roads_edges.geojson` (road names)
  and `wards/study_wards.geojson` (locality bounds).
- **2.1**: build the primal graph from `roads.graphml` + `waterways.geojson` (this task's
  output *is* 2.1's input — no further OSM pulling needed, just the primal→line-graph
  transform).
