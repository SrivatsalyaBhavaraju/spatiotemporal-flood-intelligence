# Data acquisition

Most of Phase 1's output *is* committed under `data/raw/` now, so anyone pulling
`main` can start Phase 2 straight away without re-running every fetch script —
that was the whole point of doing Phase 1 once. What's **not** committed is the
bulky, trivially-redownloadable raw source files (SRTM tiles, the IMD gridded
rainfall archive) — regenerate those locally using the sources below (all
verified live; see `working.md` §1.8 for the full accessibility/processing
table and §6 for the complete link list). `data/processed/` stays empty until
Phase 2 actually produces something to put there.

| Not committed (regenerate) | Why |
|---|---|
| `data/raw/dem/N12E080.tif`, `N13E080.tif` | Raw SRTM tiles — `python src/ground_truth/fetch_dem.py` re-downloads them and rebuilds the (committed) clipped `srtm_dem.tif`/`srtm_slope.tif` |
| `data/raw/rainfall/imd_raw/` | Raw IMD gridded archive (25MB) — `python src/ground_truth/fetch_rainfall_imd.py` re-downloads it and rebuilds the (committed) `imd_daily.csv` |

Everything else under `data/raw/` — OSM roads/waterways, ward boundaries, the
committed DEM/slope rasters, rainfall CSVs, the Bhuvan georeferencing outputs,
the gazetteer draft, and the distress-text corpus draft — is now tracked in
git. If a script's output isn't showing up after a fresh run, check
`.gitignore`'s `data/raw/*` block before assuming it's meant to stay local.

| Dataset | Source | Fetched by (task ID) |
|---|---|---|
| Roads + drainage | OSM via `osmnx` | P1.1, P1.2 |
| Ward boundaries | DataMeet / BBMP GIS Viewer / GCC Ward Info | P1.3 |
| DEM (elevation) | SRTM GL1 30m via OpenTopography | P2.1 |
| Slope | Derived from DEM (`gdaldem slope`) | P2.2 |
| Rainfall (hourly) | Open-Meteo Historical Weather API | P2.3 |
| Rainfall (daily cross-check) | IMD gridded 0.25° via `imdlib`/`imddaily` | P2.4 |
| Satellite flood extent | Sentinel-1 GRD via Google Earth Engine | P2.5, P2.8 |
| Satellite flood footprint (Chennai) | NRSC Bhuvan RISAT | P2.9 |
| Land cover (optional) | ESA WorldCover 10m | P1.8 |
| Population (optional) | Census 2011 Primary Census Abstract | P1.8 |

Keep genuinely bulky/regenerable raw downloads out of git (see the table above) — but don't default to gitignoring everything under `data/raw/` just because it's a data folder. If a fetch script's output is small and Phase 2+ needs it, commit it: add a `!data/raw/<dir>/` negation if it's a new subfolder, or a specific `!data/raw/<dir>/<file>` negation if it's a `.tif`/`.tiff` (those are blanket-ignored separately, further down in `.gitignore`).
