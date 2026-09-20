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
- **Flood Propagation** (task 6.2) — an animated map (Play/Pause built
  into the map itself) cycling through all 4 phases entirely client-side,
  toggling between real ground truth, the rule-based baseline's
  predictions, and the tuned GNN's predictions, across all 17,195
  segments. Directly visualizes task 5.3's finding: at `peak`, ground
  truth and the GNN both show the city mostly flooded (red), while the
  baseline stays entirely dry (blue) — the measured version of
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

## Real bugs found and fixed while building this, not hidden

**Bug 1 — `streamlit-folium`'s `st_folium()` silently hung.** With this
environment's `streamlit==1.64.0` / `streamlit-folium==0.27.4` pairing,
zero exception, no console error, the map area just stayed blank forever
(confirmed with a Playwright-driven headless-browser check — 0 iframes
ever mounted — since the Claude-in-Chrome extension wasn't available this
session). Root-caused with a minimal reproduction before concluding it
was a real incompatibility, not a data problem. Fixed by dropping the
custom component dependency entirely: `folium.Map._repr_html_()` embedded
via Streamlit's own built-in `st.components.v1.html()` needs no custom
component handshake.

**Bug 2 — CartoDB's free dark tiles now require an API key** (a
real-world provider change, not a bug in this project) — the map
rendered completely blank with only an easy-to-miss `UserWarning` in the
server log. Switched to Esri's dark-gray-canvas tile service, which needs
no key.

**Bug 3 — the first Play/animation design was fundamentally the wrong
architecture, found via real user feedback, not a style preference.**
That version rebuilt the entire ~6MB map HTML server-side and remounted a
brand-new `components.html()` iframe on every animation tick (driven by
`time.sleep()` + `st.rerun()` in a loop). Measured directly with
Playwright, not assumed: each tick got progressively **slower** the
longer Play ran — 6s, then 13s, 25s, 39s, 52s... between ticks in a real
timed run, consistent with iframes/Leaflet instances accumulating rather
than being torn down. On top of that, mutating a widget-keyed
`session_state` entry after the widget existed raised
`StreamlitWidgetAlreadyInstantiatedError` outright — a real Streamlit
constraint (you cannot write to `st.session_state.X` after a widget with
`key="X"` has rendered in the same script run). **Fixed with a proper
client-side animation:** `build_animated_map_html()` builds ONE map per
view (Ground Truth / Baseline / GNN), embeds all 4 phases' colors as a
JS array, and animates with a plain `setInterval()` calling Leaflet's own
`layer.setStyle()` per feature — no Streamlit rerun, no iframe rebuild,
per phase change, ever. Hit one more real bug fixing this: the injected
`<script>` referenced Folium's own `geo_json_xxx` JS variable before
Folium's own script (elsewhere in the page) had defined it yet
(`ReferenceError: geo_json_xxx is not defined`, also caught via
Playwright's console listener) — fixed with a `setTimeout`-based
readiness poll rather than assuming script order.

**Verified with a real running instance, not just "it imports cleanly":**
launched the real app, drove it with Playwright across all 5 tabs, all 3
prediction views, and multiple full Play cycles, watching for both
correctness (phase order Pre-Event→Rising→Peak→Receding→repeat, verified
frame-by-frame) and performance (tick-to-tick timing stayed flat, not
escalating) — per this project's own "test suite verifies correctness,
not feature correctness" standard.

## Performance note

`build_animated_map_html()` is `@st.cache_data`-decorated per view (3
combinations, not 12) — computing each involves reading 4 phase GeoJSONs
(~17,195 segments each) and building one JS color array per phase, a few
real seconds the first time a given view is selected. After that,
switching back to an already-viewed model is instant, and the Play
animation itself never touches the server at all.
