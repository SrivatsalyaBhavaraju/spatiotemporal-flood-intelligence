# Spatiotemporal Digital Twins for Predictive Disaster Intelligence

Capstone project — ward-scale flood propagation modeling with a spatiotemporal GNN, multilingual distress geoparsing, and (optional) equity-constrained resource allocation.

- **Start here:** [`working.md`](./working.md) — full technical rationale: objectives, research gaps, data sources, graph design, model design, ground-truth methodology, verified source list.
- **Then:** [`developing.md`](./developing.md) — the build tracker: phases, task IDs, dependencies, owners, timeline. This is what the team works off day to day.

## Team

| Role | Person |
|---|---|
| P1 — Geospatial & Graph Lead | Indla Sai Sneha |
| P2 — Ground Truth & Data Lead | Srivatsalya Bhavaraju |
| P3 — ML/GNN Lead | Jonathan S Pulickal |
| P4 — NLP / Objective 2 (+3) Lead | Anurag Reddy Thumma |

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

GDAL can be finicky on Windows — if `pip install gdal` fails, use `conda install -c conda-forge gdal` instead, or install via [OSGeo4W](https://trac.osgeo.org/osgeo4w/).

## Repo structure

See `developing.md` §14 for the full layout and what goes in each `src/` subfolder.

## Data

Raw and processed data are **not committed** (see `.gitignore`). See `data/README.md` for how to (re)fetch each dataset — every source is verified and linked in `working.md` §1.8 and §6.
