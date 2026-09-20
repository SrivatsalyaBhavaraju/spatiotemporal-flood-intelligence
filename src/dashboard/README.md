# Integration dashboard (Streamlit + Folium) — tasks 6.1/6.2/6.3

This is where Objective 1 (baseline vs. GNN flood propagation) and
Objective 2 (distress classification + geoparsing) visibly become one
system, per developing.md §10.

## Running it

```bash
python src/dashboard/prepare_dashboard_data.py   # one-time (or after re-running any upstream task)
streamlit run src/dashboard/app.py
```

`prepare_dashboard_data.py` joins task 2.2/2.3/3.4/3.7/4.3/5.1's real
outputs into 4 lightweight per-phase GeoJSON files (`data/processed/dashboard/`,
gitignored) — including running the already-trained task 5.1 GNN model's
forward pass (not retraining it) to get per-segment predictions for every
phase. The Streamlit app never loads a model or touches `data/processed/`
outside these pre-joined files, so it starts fast.

## Structure

- **Overview** — real KPI tiles pulled from tasks 3.4/4.2/5.1/5.3/5.4/5.5's
  own reports (peak flood coverage, fair-holdout AUC, classifier F1,
  geoparsing precision/recall).
- **Flood Propagation** (task 6.2) — an animated map (phase slider +
  play/pause) toggling between real ground truth, the rule-based
  baseline's predictions, and the tuned GNN's predictions, across all
  17,195 segments. Directly visualizes task 5.3's finding: at `peak`,
  ground truth and the GNN both show the city mostly flooded (red), while
  the baseline stays entirely dry (blue) — the measured version of
  working.md §1.6's core claim, not just a number in a table.
- **Distress Signals** (task 6.3) — real, pulsing markers for every
  Objective 2 pipeline output (task 4.5) that classified as distress AND
  resolved to a coordinate, with popups showing the actual passage text.
- **Model Comparison** — interactive Plotly recreations of task 5.3's
  comparison (F1 per transition/split, fair-holdout AUC with a
  random-chance reference line).
- **Data & Limitations** — task 5.4's ward-informativeness finding and
  task 5.5's geoparsing false positive, presented as designed insight
  cards, not a wall of JSON. The dashboard makes the same disclosures the
  written notes do — a polished UI is not an excuse to hide a limitation.

## A real bug found and fixed while building this, not hidden

The first version used `streamlit-folium`'s `st_folium()` component.
It **silently hung** with this environment's `streamlit==1.64.0` /
`streamlit-folium==0.27.4` pairing — no exception, no console error, the
map area just stayed blank forever (confirmed with a Playwright-driven
headless-browser check, since the Claude-in-Chrome extension wasn't
available in this session — 0 iframes ever mounted). Root-caused by
building a minimal reproduction outside the real app before concluding it
was a real incompatibility, not a data problem. Fixed by dropping the
custom component dependency entirely: `folium.Map._repr_html_()` embedded
via Streamlit's own built-in `st.components.v1.html()` needs no custom
component handshake and worked immediately. This app doesn't need
`st_folium`'s click-return functionality anyway (no `returned_objects`
were used), so nothing was lost.

**Also found and fixed:** the free `CartoDB dark_matter` tile set now
requires an API key (a real-world provider change, not a bug in this
project) — the map rendered completely blank with only a `UserWarning` in
the server log, easy to miss. Switched to Esri's dark-gray-canvas tile
service, which needs no key.

**Verified visually, not just "it imports cleanly":** launched the real
app, drove it with Playwright (phase slider, all three prediction views,
all 5 tabs), and inspected real screenshots before considering this done
— per this project's own "test suite verifies correctness, not feature
correctness" standard.

## Performance note

Rendering ~17,195 GeoJSON line features takes a few real seconds the
first time a given phase/view combination is requested (not a toy
dataset). `build_flood_map_html()` is wrapped in `@st.cache_data` so
switching back to an already-viewed combination is instant.
