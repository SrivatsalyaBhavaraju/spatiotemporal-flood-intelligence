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
AOI), concentrated in 3 of 16 wards (Adyar 0.049 km², Perungudi 0.023 km²,
Kodambakkam 0.009 km²) — the other 13 wards show no detected flooding.
Cross-checked against task 0.6's georeferenced NRSC simulation raster: mean
depth at flood-polygon centroids (2.94m) is higher than at random AOI points
(2.62m) — corroborating, not proof, given that raster's own ~563m RMSE.

**Read as a floor, not the full picture:** this is a known, expected SAR
limitation (working.md §1.5), not a bug to chase — dense urban canopy
genuinely suppresses the water-like backscatter signal, and this pass pair
brackets the 30 Nov–2 Dec peak rather than capturing it (nearest pass is +4
days post-peak). Task 3.3 (news/advisory cross-check) is explicitly the
step that's supposed to catch what SAR misses here — don't treat the
13 SAR-silent wards as "confirmed dry" when 3.3/3.4 run.

Outputs (`data/processed/ground_truth/`, gitignored): the three downloaded
intermediate rasters (pre/post VV dB, JRC occurrence — kept for QA/rerun),
`sentinel1_flood_extent.geojson`, and the validation report.

