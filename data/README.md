# Data acquisition

Nothing in `data/raw/` or `data/processed/` is committed to git — regenerate it locally using the sources below (all verified live; see `working.md` §1.8 for the full accessibility/processing table and §6 for the complete link list).

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

Keep raw downloads out of git — large rasters and satellite scenes will blow up repo size. If a processed artifact is small and needs to be shared (e.g. the final graph pickle, the labeled distress dataset), commit it explicitly under `data/processed/` and remove it from `.gitignore`'s exclusion for that specific file.
