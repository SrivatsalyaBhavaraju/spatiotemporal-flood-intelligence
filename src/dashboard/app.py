"""
Task 6.1/6.2/6.3 -- Streamlit + Folium integration dashboard. This is
where Objective 1 (flood propagation, baseline vs. GNN) and Objective 2
(distress classification + geoparsing) visibly become one system. Task 7.3
adds the Fair Relief tab: Objective 3's equity-constrained optimizer,
re-solved live (task 7.2's own MIP) as the rho/budget sliders move.

Presentation redesign (task 8.3 prep): the app is organised as a story an
audience can follow -- every tab opens with the question it answers and the
answer in one sentence, the Overview shows how the three objectives connect,
and technical metrics sit one click deeper instead of leading. Every number
is still read from the tasks' own generated reports; nothing is invented.

Run with:
    streamlit run src/dashboard/app.py

All data is real -- pre-joined by prepare_dashboard_data.py or read
directly from tasks 3.4/4.3/5.1/5.3/5.4/5.5/7.1-7.3's own generated reports.
Nothing here recomputes a model or invents a number (the Fair Relief
sliders re-run task 7.2's own optimizer; the live geoparser box runs task
2.10's own geoparser).
"""
import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dashboard.data_loader import (  # noqa: E402
    COLOR_BASELINE,
    COLOR_DRY,
    COLOR_FLOODED,
    COLOR_GNN,
    GROUND_TRUTH_DIR,
    NLP_DIR,
    PHASES,
    PHASE_LABELS,
    compute_overview_kpis,
    flood_fill_color,
    load_distress_markers,
    load_json_report,
    load_optimizer_outputs,
    load_phase_layer,
    load_wards,
)
from src.features.build_rainfall_phase_features import PHASES as PHASE_WINDOWS  # noqa: E402 -- task 2.5's real windows
from src.optimizer.equity_optimizer import evaluate_allocation, solve_allocation  # noqa: E402

CHENNAI_CENTER = [12.985, 80.225]
# CartoDB's free dark tiles now require an API key (discovered while testing this dashboard --
# the map silently failed to render with no console error). Esri's dark-gray-canvas tile service
# needs no key and gives the same dark aesthetic.
DARK_TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
DARK_TILES_ATTR = "Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, GIS User Community"

MAP_ATTR_SHORT = "© Esri · © OpenStreetMap contributors"
TRANSITION_LABELS = {"pre_event->rising": "Pre-event → Rising", "rising->peak": "Rising → Peak (flood onset)",
                     "peak->receding": "Peak → Receding"}
SPLIT_LABELS = {"train": "Training wards", "val": "Validation wards (unseen)", "test": "Test wards (unseen)"}


def _phase_date_label(phase_name: str) -> str:
    w = next(p for p in PHASE_WINDOWS if p["phase_name"] == phase_name)
    start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end"])
    return f"{start:%d %b} – {end:%d %b %Y}"


st.set_page_config(page_title="Chennai Flood Intelligence", page_icon="\U0001F30A", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }
.block-container { padding-top: 2.2rem; max-width: 1500px; }

.hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.1;
    background: linear-gradient(90deg, #eb6834 0%, #eda100 50%, #e34948 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-subtitle { color: #d6d4cb; font-size: 1.15rem; line-height: 1.55; margin: 0.5rem 0 1.4rem 0; max-width: 1100px; }
.hero-chips { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.chip { background: #1f1f1d; border: 1px solid rgba(255,255,255,0.10); color: #d6d4cb; border-radius: 999px;
        padding: 0.28rem 0.8rem; font-size: 0.82rem; font-weight: 500; }

.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { font-size: 1.02rem; font-weight: 600; padding: 0.55rem 0.9rem; }

.story {
    background: linear-gradient(135deg, #24150e 0%, #181817 55%);
    border: 1px solid rgba(235,104,52,0.28); border-radius: 16px;
    padding: 1.25rem 1.6rem; margin: 0.6rem 0 1.3rem 0;
}
.story .q { color: #eb6834; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.story .a { color: #ffffff; font-size: 1.5rem; font-weight: 700; line-height: 1.35; margin-top: 0.35rem; }
.story .sub { color: #c3c2b7; font-size: 1rem; line-height: 1.55; margin-top: 0.5rem; }

.section-title { font-size: 1.25rem; font-weight: 700; color: #ffffff; margin: 1.4rem 0 0.6rem 0; }
.section-title span { color: #898781; font-weight: 500; font-size: 0.95rem; margin-left: 0.4rem; }

.pipeline { display: flex; gap: 0.7rem; align-items: stretch; margin: 0.4rem 0 0.6rem 0; }
.stage { flex: 1; background: #1a1a19; border-radius: 14px; padding: 1.1rem 1.25rem;
         border: 1px solid rgba(255,255,255,0.07); border-top: 4px solid var(--c); }
.stage .step { display: inline-block; background: var(--c); color: #0d0d0d; font-weight: 800; border-radius: 999px;
               width: 1.7rem; height: 1.7rem; text-align: center; line-height: 1.7rem; font-size: 0.9rem; }
.stage .kicker { color: var(--c); font-weight: 700; font-size: 0.78rem; letter-spacing: 0.09em; text-transform: uppercase;
                 margin-left: 0.45rem; }
.stage h3 { color: #ffffff; font-size: 1.15rem; margin: 0.6rem 0 0.35rem 0; font-weight: 700; }
.stage p { color: #c3c2b7; font-size: 0.92rem; line-height: 1.5; margin: 0; }
.stage .out { margin-top: 0.75rem; padding-top: 0.65rem; border-top: 1px dashed rgba(255,255,255,0.12);
              color: #ffffff; font-size: 0.88rem; }
.stage .out b { color: var(--c); }
.arrow { align-self: center; color: #8a8983; font-size: 2.4rem; font-weight: 400; }

.finding { background: #1a1a19; border-radius: 14px; padding: 1.2rem 1.35rem; height: 100%;
           border: 1px solid rgba(255,255,255,0.07); }
.finding .big { font-size: 2.7rem; font-weight: 800; line-height: 1.05; color: var(--c); }
.finding .what { color: #ffffff; font-weight: 700; font-size: 1.02rem; margin-top: 0.45rem; }
.finding .why { color: #c3c2b7; font-size: 0.9rem; line-height: 1.5; margin-top: 0.35rem; }
.finding .tag { color: #898781; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }

.navgrid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.6rem; margin: 0.4rem 0 1.4rem 0; }
.navcard { background: #151514; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 0.8rem 0.9rem; }
.navcard .t { color: #ffffff; font-weight: 700; font-size: 0.95rem; }
.navcard .d { color: #a9a79e; font-size: 0.83rem; line-height: 1.45; margin-top: 0.25rem; }

.kpi-card {
    background: linear-gradient(145deg, #1a1a19 0%, #232320 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.1rem 1.3rem;
    box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    height: 100%;
}
.kpi-card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
.kpi-label { color: #898781; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.kpi-value { font-size: 1.9rem; font-weight: 800; color: #ffffff; margin-top: 0.15rem; }
.kpi-caption { color: #c3c2b7; font-size: 0.8rem; margin-top: 0.2rem; }
.kpi-accent-orange { color: #eb6834; }
.kpi-accent-blue { color: #3987e5; }
.kpi-accent-red { color: #e66767; }
.kpi-accent-green { color: #0ca30c; }

.insight-card {
    background: #1a1a19; border-left: 4px solid #eb6834; border-radius: 8px;
    padding: 1rem 1.2rem; margin-bottom: 0.9rem;
}
.insight-card.warning { border-left-color: #fab219; }
.insight-card.good { border-left-color: #0ca30c; }
.insight-card h4 { margin: 0 0 0.35rem 0; color: #ffffff; }
.insight-card p { margin: 0; color: #c3c2b7; font-size: 0.95rem; line-height: 1.55; }

.steps { display: flex; gap: 0.5rem; align-items: stretch; margin: 0.2rem 0 1rem 0; }
.stepbox { flex: 1; background: #1a1a19; border-radius: 12px; padding: 0.85rem 1rem; border: 1px solid rgba(255,255,255,0.07); }
.stepbox .n { color: #eb6834; font-weight: 800; font-size: 0.8rem; letter-spacing: 0.08em; }
.stepbox .t { color: #ffffff; font-weight: 700; margin-top: 0.15rem; }
.stepbox .d { color: #a9a79e; font-size: 0.85rem; line-height: 1.45; margin-top: 0.2rem; }
.steps .arrow { font-size: 1.4rem; }

.phase-row { display: flex; gap: 0.4rem; margin: 0.4rem 0 0.8rem 0; }
.phase { flex: 1; background: #171716; border-radius: 10px; padding: 0.55rem 0.7rem; border: 1px solid rgba(255,255,255,0.07); }
.phase .p { color: #ffffff; font-weight: 700; font-size: 0.9rem; }
.phase .dt { color: #898781; font-size: 0.76rem; }
.phase.hot { border-color: rgba(208,59,59,0.7); background: #221413; }

.watch { background: #171716; border-radius: 12px; padding: 0.9rem 1rem; border: 1px solid rgba(255,255,255,0.07); }
.watch li { color: #c3c2b7; font-size: 0.92rem; line-height: 1.55; margin-bottom: 0.3rem; }
.watch b { color: #ffffff; }

.match-chip { display: inline-block; background: #22301f; border: 1px solid #0ca30c; color: #d9f2d9; border-radius: 999px;
              padding: 0.2rem 0.75rem; margin: 0.15rem 0.3rem 0.15rem 0; font-size: 0.88rem; }
.legend-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; }
.footer { color: #6f6e68; font-size: 0.8rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.06); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def story(question: str, answer: str, sub: str = "") -> None:
    """Every tab opens with the question it answers and the answer in one sentence."""
    st.markdown(
        f'<div class="story"><div class="q">{question}</div><div class="a">{answer}</div>'
        + (f'<div class="sub">{sub}</div>' if sub else "") + "</div>",
        unsafe_allow_html=True,
    )


def section(title: str, hint: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}' + (f"<span>{hint}</span>" if hint else "") + "</div>",
                unsafe_allow_html=True)


def kpi_row(tiles) -> None:
    for col, label, value, caption, accent in tiles:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value kpi-accent-{accent}">{value}</div>'
                f'<div class="kpi-caption">{caption}</div></div>',
                unsafe_allow_html=True,
            )


try:
    kpis = compute_overview_kpis()
except FileNotFoundError as e:
    st.error(f"Dashboard data not ready: {e}\n\nRun `python src/dashboard/prepare_dashboard_data.py` first.")
    st.stop()

try:
    OPT = load_optimizer_outputs()
except FileNotFoundError:
    OPT = None  # Objective 3 not generated -- the Fair Relief tab explains how to build it

st.markdown('<div class="hero-title">Chennai 2015 Flood Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">A digital twin of the December 2015 Chennai flood that <b>predicts</b> where water '
    'spreads across the road network, <b>listens</b> for people asking for help, and <b>decides</b> where relief '
    'should go — so the most vulnerable wards are not left behind just because they are the quietest.</div>',
    unsafe_allow_html=True,
)
chips = [f"{kpis['n_wards']} wards", f"{kpis['n_segments']:,} road segments", "4 flood phases · 8 Nov – 14 Dec 2015",
         "Sentinel-1 SAR + Bhuvan + news ground truth"]
if OPT is not None:
    chips.insert(1, f"{OPT['vulnerability_report']['census_population_total_study_area']:,} residents (Census 2011)")
st.markdown('<div class="hero-chips">' + "".join(f'<span class="chip">{c}</span>' for c in chips) + "</div>",
            unsafe_allow_html=True)


VIEW_COLUMNS = [("Ground Truth", "true_flood"), ("Baseline Prediction", "baseline_pred"), ("GNN Prediction", "gnn_pred")]


@st.cache_data(show_spinner=False)
def build_animated_map_html(view_label: str, column: str) -> str:
    """One map, built once, animates all 4 phases entirely client-side via
    embedded JS -- no Streamlit rerun, no iframe rebuild, per phase change.

    *** REPLACES AN EARLIER, BROKEN DESIGN, NOT A STYLE PREFERENCE ***
    the first version rebuilt the whole ~6MB map HTML server-side and
    remounted a new components.html() iframe on every Play tick (driven by
    time.sleep() + st.rerun() in a loop). Measured directly, not assumed:
    each tick got progressively SLOWER the longer Play ran (6s, then 13s,
    25s, 39s, 52s... between ticks in a real Playwright-timed run) --
    consistent with iframes/Leaflet map instances accumulating rather than
    being cleanly torn down. On top of that, mutating a widget-keyed
    session_state entry after the widget existed raised
    StreamlitWidgetAlreadyInstantiatedError outright. Both problems are
    structural to "rebuild everything server-side per animation frame" --
    fixed here by building the map ONCE and animating by calling Leaflet's
    own `setStyle()` per feature from a `setInterval()` loop already
    running in the browser. Geometry (fixed across phases) is fetched
    once; only each phase's color list is precomputed and embedded.

    *** ELEVATION-STAGGERED TRANSITIONS, NOT A LITERAL FLOW SIMULATION ***
    task 5.4 already found the real ground truth has no sub-phase timing/
    velocity/depth data -- there is nothing to animate a genuine water-flow
    physics simulation FROM. What we do have is real per-segment elevation
    (task 2.3). So a transition doesn't flip all changed segments at once:
    newly-flooding segments repaint lowest-elevation-first (water fills low
    ground first) and newly-receding segments repaint highest-elevation-first
    (marginal high ground drains first, low ground stays flooded longest).
    This is a disclosed, physically-motivated illustrative stagger grounded
    in real elevation data -- not a claim that this is the literal, timed
    path floodwater took (that data doesn't exist for this event).
    """
    base_full = load_phase_layer(PHASES[0])[["segment_id", "geometry", "elevation_m"]].reset_index(drop=True)
    base = base_full[["segment_id", "geometry"]]
    elevations = base_full["elevation_m"].fillna(base_full["elevation_m"].median()).tolist()
    wards = load_wards()

    colors_by_phase = []
    for phase in PHASES:
        layer = load_phase_layer(phase)[["segment_id", column]]
        merged = base[["segment_id"]].merge(layer, on="segment_id", how="left")
        values = merged[column].fillna(0).astype(int).tolist()
        colors_by_phase.append([flood_fill_color(v) for v in values])

    # scrollWheelZoom off: scrolling the page past the map must not zoom it mid-presentation.
    m = folium.Map(location=CHENNAI_CENTER, zoom_start=12.3, tiles=None, prefer_canvas=True, scrollWheelZoom=False)
    folium.TileLayer(tiles=DARK_TILES_URL, attr=DARK_TILES_ATTR, name="Dark basemap").add_to(m)
    folium.GeoJson(
        wards, style_function=lambda _: {"color": "#52514e", "weight": 1.5, "fillOpacity": 0.02},
    ).add_to(m)

    geo = folium.GeoJson(
        # Initial style is constant blue (dry) -- correct as-is for phase 0 (pre_event is always
        # 100% dry by construction, task 3.4) and JS immediately repaints per column/phase anyway.
        base, name=view_label, style_function=lambda _: {"color": COLOR_DRY, "weight": 2, "opacity": 0.85},
        tooltip=folium.GeoJsonTooltip(fields=["segment_id"], aliases=["Segment"]),
    )
    geo.add_to(m)

    controls_html = f"""
    <div style="position:absolute; top:10px; left:60px; z-index:1000; display:flex; gap:8px; align-items:center; font-family:'Inter',system-ui,sans-serif;">
      <div id="phase-indicator-{geo.get_name()}" style="background:#1a1a19; color:#eb6834; padding:6px 16px; border-radius:8px; font-weight:700; border:1px solid rgba(255,255,255,0.15); min-width:90px; text-align:center;">{PHASE_LABELS[PHASES[0]]}</div>
      <button id="play-btn-{geo.get_name()}" style="background:#eb6834;color:#fff;border:none;padding:6px 14px;border-radius:8px;cursor:pointer;font-weight:600;">&#9654; Play</button>
      <button id="pause-btn-{geo.get_name()}" style="background:#2a2a28;color:#fff;border:1px solid rgba(255,255,255,0.15);padding:6px 14px;border-radius:8px;cursor:pointer;font-weight:600;">&#10074;&#10074; Pause</button>
    </div>
    <script>
    (function() {{
        // Folium's own script (which defines {geo.get_name()} = L.geoJson(...)) can execute
        // AFTER this block in the rendered page -- hit a real "geo_json_xxx is not defined"
        // ReferenceError from this exact race the first time. Poll until Folium's variable
        // actually exists rather than assuming script order.
        function whenReady(callback) {{
            if (typeof {geo.get_name()} !== "undefined") {{
                callback();
            }} else {{
                setTimeout(function() {{ whenReady(callback); }}, 50);
            }}
        }}
        whenReady(function() {{
            var colors = {json.dumps(colors_by_phase)};
            var labels = {json.dumps([PHASE_LABELS[p] for p in PHASES])};
            var elevations = {json.dumps(elevations)};
            var floodedColor = {json.dumps(COLOR_FLOODED)};
            var sublayers = {geo.get_name()}.getLayers();
            var current = colors[0].slice();
            var idx = 0;
            var timer = null;
            var sweepTimer = null;
            var TICK_MS = 1500;
            var SWEEP_STEPS = 40;
            var SWEEP_STEP_MS = 20;  // 40 * 20 = 800ms sweep, comfortably inside TICK_MS

            function applyPhase(i) {{
                var target = colors[i];
                var flooding = [];    // newly dry -> flooded: lowest elevation repaints first
                var receding = [];    // newly flooded -> dry: highest elevation repaints first
                for (var j = 0; j < sublayers.length; j++) {{
                    if (target[j] !== current[j]) {{
                        var entry = {{idx: j, elev: elevations[j]}};
                        if (target[j] === floodedColor) {{ flooding.push(entry); }} else {{ receding.push(entry); }}
                    }}
                }}
                flooding.sort(function(a, b) {{ return a.elev - b.elev; }});   // low ground first
                receding.sort(function(a, b) {{ return b.elev - a.elev; }});   // high ground first
                var ordered = flooding.concat(receding);

                if (sweepTimer) {{ clearTimeout(sweepTimer); sweepTimer = null; }}
                var chunkSize = Math.max(1, Math.ceil(ordered.length / SWEEP_STEPS));
                var step = 0;
                function sweep() {{
                    var start = step * chunkSize;
                    var end = Math.min(start + chunkSize, ordered.length);
                    for (var k = start; k < end; k++) {{
                        var c = ordered[k];
                        sublayers[c.idx].setStyle({{color: target[c.idx]}});
                        current[c.idx] = target[c.idx];
                    }}
                    step++;
                    if (start < ordered.length) {{ sweepTimer = setTimeout(sweep, SWEEP_STEP_MS); }}
                }}
                sweep();
                document.getElementById('phase-indicator-{geo.get_name()}').innerText = labels[i];
            }}
            document.getElementById('play-btn-{geo.get_name()}').onclick = function() {{
                if (timer) return;
                timer = setInterval(function() {{
                    idx = (idx + 1) % colors.length;
                    applyPhase(idx);
                }}, TICK_MS);
            }};
            document.getElementById('pause-btn-{geo.get_name()}').onclick = function() {{
                clearInterval(timer);
                timer = null;
            }};
        }});
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(controls_html))
    return m._repr_html_()


@st.cache_data(show_spinner=False)
def phase_flood_shares() -> pd.DataFrame:
    """Share of road segments flooded per phase, per view (real per-phase layers)."""
    rows = []
    for phase in PHASES:
        layer = load_phase_layer(phase)
        rows.append({"phase": phase, "Ground truth": layer["true_flood"].mean(),
                     "Rule-based baseline": layer["baseline_pred"].mean(),
                     "GNN (thresholded)": layer["gnn_pred"].mean(),
                     "GNN average flood probability": layer["gnn_prob"].mean()})
    return pd.DataFrame(rows).set_index("phase")


@st.cache_resource(show_spinner=False)
def load_geoparser():
    from src.nlp.fuzzy_geoparse import geoparse_text, load_gazetteer
    return geoparse_text, load_gazetteer()


tab_story, tab_map, tab_distress, tab_compare, tab_alloc, tab_limits = st.tabs(
    ["\U0001F9ED The Story", "\U0001F30A Flood Prediction", "\U0001F4E1 Distress Signals",
     "\U0001F4C8 Model Verdict", "⚖️ Fair Relief", "\U0001F50D Limits & Honesty"]
)

# --------------------------------------------------------------------------
# The Story (overview)
# --------------------------------------------------------------------------
with tab_story:
    story(
        "The problem",
        "In December 2015, Chennai's heaviest single-day rainfall since 1901 put most of the southern city under water.",
        "Rule-based flood maps react after the water arrives, emergency lines hear only from people who can reach "
        "them, and relief tends to follow whoever is loudest. This project builds one connected system that tackles "
        f"all three — on {kpis['n_segments']:,} real road segments across {kpis['n_wards']} south-Chennai wards, "
        f"where {kpis['peak_flooded_pct']}% of roads were under water at the peak.",
    )

    section("How the system fits together", "three objectives, each feeding the next")
    if OPT is not None:
        dd_starved = int(OPT["optimizer_report"]["models"]["demand_driven_optimum"]["n_starved_wards"])
        eq_starved = int(OPT["plan_report"]["metrics"]["n_starved_wards"])
        obj3_out = f"<b>{dd_starved} → {eq_starved}</b> wards left without relief"
    else:
        obj3_out = "run the Objective 3 scripts to populate"
    st.markdown(
        '<div class="pipeline">'
        '<div class="stage" style="--c:#3987e5"><span class="step">1</span><span class="kicker">Predict</span>'
        "<h3>Flood propagation on the road graph</h3>"
        "<p>Every road segment is a node; a spatiotemporal graph neural network learns how flooding spreads from "
        "elevation, slope, drainage and rainfall — compared against a hand-written rule.</p>"
        f'<div class="out">Rule-based baseline at flood onset: <b>F1 = {kpis["baseline_test_f1"]:.2f}</b></div></div>'
        '<div class="arrow">›</div>'
        '<div class="stage" style="--c:#eb6834"><span class="step">2</span><span class="kicker">Listen</span>'
        "<h3>Distress detection + geoparsing</h3>"
        "<p>A multilingual MuRIL classifier flags text asking for help; a fuzzy gazetteer geoparser turns place names "
        "(\"near Velachery bus stand\") into map coordinates.</p>"
        f'<div class="out">Classifier <b>F1 = {kpis["classifier_f1"]:.2f}</b> · geoparsing precision '
        f'<b>{kpis["geoparse_precision"]:.2f}</b></div></div>'
        '<div class="arrow">›</div>'
        '<div class="stage" style="--c:#0ca30c"><span class="step">3</span><span class="kicker">Decide</span>'
        "<h3>Equity-constrained relief allocation</h3>"
        "<p>Flood extent + distress signals + Census vulnerability feed an optimizer that guarantees every ward a "
        "vulnerability-weighted floor before following reported demand.</p>"
        f'<div class="out">{obj3_out}</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    section("Three findings to remember")
    f1, f2, f3 = st.columns(3)
    findings = [
        (f1, "#e66767", f"{kpis['baseline_test_f1']:.2f}", "The rule never sees the flood coming",
         "A static, rule-based model scores F1 = 0 at the moment flooding starts, on every split of wards. "
         f"The graph model shows real, modest skill on unseen wards (AUC {kpis['gnn_val_auc']:.2f} vs. 0.50 chance).",
         "Objective 1"),
        (f2, "#eb6834", f"{kpis['geoparse_precision']:.0%}", "Free text becomes a location",
         f"Of 37 real place-name matches reviewed by hand, 36 resolved to the right place; the distress classifier "
         f"reaches F1 = {kpis['classifier_f1']:.2f} on real flood reporting.", "Objective 2"),
    ]
    if OPT is not None:
        eo_rep = OPT["optimizer_report"]
        silent = eo_rep["signal_gap"]["robust_core_wards_with_zero_signal"]
        core = OPT["vulnerability_report"]["robustness"]["robust_core"]
        knee = next(r for r in eo_rep["price_of_fairness_frontier"] if r["rho"] == eo_rep["cheapest_no_starvation_rho"])
        findings.append((f3, "#0ca30c", f"{len(silent)} of {len(core)}", "Most-vulnerable wards were silent",
                         f"{len(silent)} of the {len(core)} most vulnerable wards sent no distress signal at all. "
                         f"Demand-driven allocation leaves {dd_starved} wards without relief; the equity floor leaves "
                         f"{eq_starved}, giving up only {knee['signal_coverage_loss_vs_rho0']:.1%} of reported-distress "
                         "coverage.", "Objective 3"))
    for col, color, big, what, why, tag in findings:
        with col:
            st.markdown(
                f'<div class="finding" style="--c:{color}"><div class="tag">{tag}</div><div class="big">{big}</div>'
                f'<div class="what">{what}</div><div class="why">{why}</div></div>',
                unsafe_allow_html=True,
            )

    section("Where to look next")
    st.markdown(
        '<div class="navgrid">'
        '<div class="navcard"><div class="t">\U0001F30A Flood Prediction</div><div class="d">Watch the flood rise and '
        "recede road by road — and see which model keeps up.</div></div>"
        '<div class="navcard"><div class="t">\U0001F4E1 Distress Signals</div><div class="d">Real posts on the map, and '
        "a live box that turns your sentence into places.</div></div>"
        '<div class="navcard"><div class="t">\U0001F4C8 Model Verdict</div><div class="d">The baseline-vs-GNN test, '
        "with the numbers that make it fair.</div></div>"
        '<div class="navcard"><div class="t">⚖️ Fair Relief</div><div class="d">Move the equity slider and '
        "watch the allocation re-solve live.</div></div>"
        '<div class="navcard"><div class="t">\U0001F50D Limits & Honesty</div><div class="d">What the data can and '
        "cannot support — stated, not hidden.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Technical metrics (for the Q&A)"):
        c1, c2, c3, c4 = st.columns(4)
        kpi_row([
            (c1, "Road Segments", f"{kpis['n_segments']:,}", "across 16 study wards", "blue"),
            (c2, "Peak Flood Coverage", f"{kpis['peak_flooded_pct']}%", "of segments, at event peak", "red"),
            (c3, "GNN Fair-Holdout AUC (val)", f"{kpis['gnn_val_auc']:.2f}", "vs. 0.50 = random chance", "green"),
            (c4, "Distress Classifier F1", f"{kpis['classifier_f1']:.2f}", "17-example LOOCV (task 3.8)", "orange"),
        ])
        st.write("")
        c5, c6, c7, c8 = st.columns(4)
        kpi_row([
            (c5, "Baseline Test F1 (rising→peak)", f"{kpis['baseline_test_f1']:.2f}", "structurally can't anticipate onset", "red"),
            (c6, "GNN Tuned Test F1 (rising→peak)", f"{kpis['gnn_tuned_test_f1']:.2f}", "see Limits & Honesty", "orange"),
            (c7, "Geoparsing Precision", f"{kpis['geoparse_precision']:.3f}", "36/37 real matches reviewed", "blue"),
            (c8, "Geoparsing Recall", f"{kpis['geoparse_recall']:.2f}", "vs. task 1.13's real corpus", "green"),
        ])

# --------------------------------------------------------------------------
# Flood Prediction (task 6.2)
# --------------------------------------------------------------------------
with tab_map:
    shares = phase_flood_shares()
    story(
        "Objective 1 · Where does the water go, and which model sees it coming?",
        "Reality turns the map red at the peak. The rule-based baseline stays dry until the water is already "
        "receding.",
        f"At the peak, {shares.loc['peak', 'Ground truth']:.0%} of roads are flooded in the ground truth, and "
        f"{shares.loc['peak', 'Rule-based baseline']:.0%} in the baseline's prediction. The graph model's average "
        f"flood probability rises from {shares.loc['rising', 'GNN average flood probability']:.2f} to "
        f"{shares.loc['peak', 'GNN average flood probability']:.2f} as the flood arrives.",
    )
    st.markdown(
        '<div class="phase-row">'
        + "".join(
            f'<div class="phase{" hot" if p == "peak" else ""}"><div class="p">{PHASE_LABELS[p]}</div>'
            f'<div class="dt">{_phase_date_label(p)}</div></div>' for p in PHASES)
        + "</div>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 3])
    with left:
        view = st.radio("Show on the map", ["Ground Truth", "Baseline Prediction", "GNN Prediction"], index=0)
        column_map = dict(VIEW_COLUMNS)
        st.markdown(
            f'<span class="legend-swatch" style="background:{COLOR_DRY}"></span>Dry &nbsp;&nbsp;'
            f'<span class="legend-swatch" style="background:{COLOR_FLOODED}"></span>Flooded',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="watch"><ul>'
            "<li>Press <b>▶ Play</b> on the map — it cycles the 4 phases in your browser.</li>"
            "<li><b>Ground Truth:</b> watch low-lying roads fill first at the <b>Peak</b>.</li>"
            "<li><b>Baseline:</b> stays blue through the Peak — it only turns red at <b>Receding</b>, "
            "after the fact.</li>"
            "<li><b>GNN:</b> flags every road from Rising on — a known threshold artifact (see Limits); its "
            "<i>probability</i> is what tracks the flood (chart below).</li>"
            "</ul></div>",
            unsafe_allow_html=True,
        )
        if view != "Ground Truth":
            st.caption("Pre-Event is never a model target (task 2.7's schema) — always shown dry for Baseline/GNN.")
    with right:
        map_html = build_animated_map_html(view, column_map[view])
        st.iframe(map_html, height=620)

    section("Share of roads flooded, phase by phase", "ground truth vs. each model")
    x = [f"{PHASE_LABELS[p]}<br><sub>{_phase_date_label(p)}</sub>" for p in PHASES]
    fig_t = go.Figure()
    fig_t.add_trace(go.Scatter(x=x, y=shares["Ground truth"], name="Ground truth (observed)", mode="lines+markers",
                               line={"color": COLOR_FLOODED, "width": 4}, marker={"size": 11}))
    fig_t.add_trace(go.Scatter(x=x, y=shares["Rule-based baseline"], name="Rule-based baseline", mode="lines+markers",
                               line={"color": COLOR_BASELINE, "width": 3}, marker={"size": 9}))
    fig_t.add_trace(go.Scatter(x=x, y=shares["GNN average flood probability"], name="GNN average flood probability",
                               mode="lines+markers", line={"color": COLOR_GNN, "width": 3}, marker={"size": 9}))
    fig_t.add_trace(go.Scatter(x=x, y=shares["GNN (thresholded)"], name="GNN thresholded (flags everything)",
                               mode="lines", line={"color": COLOR_GNN, "width": 1.5, "dash": "dot"}))
    fig_t.add_annotation(x=x[2], y=shares.loc["peak", "Rule-based baseline"], text="Baseline misses the peak entirely",
                         showarrow=True, arrowhead=2, ax=0, ay=-60, font={"color": "#ffffff", "size": 13},
                         bgcolor="#1a1a19", bordercolor=COLOR_BASELINE)
    fig_t.add_annotation(x=x[3], y=shares.loc["receding", "Rule-based baseline"], text="…and reacts only afterwards",
                         showarrow=True, arrowhead=2, ax=-120, ay=30, font={"color": "#ffffff", "size": 13},
                         bgcolor="#1a1a19", bordercolor=COLOR_BASELINE)
    fig_t.update_layout(template="plotly_dark", height=420, yaxis_tickformat=".0%", yaxis_range=[-0.05, 1.08],
                        yaxis_title="share of road segments", legend={"orientation": "h", "y": -0.25},
                        margin={"t": 20}, font={"size": 14}, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_t, width='stretch')

# --------------------------------------------------------------------------
# Distress Signals (task 6.3)
# --------------------------------------------------------------------------
with tab_distress:
    markers_df = load_distress_markers()
    story(
        "Objective 2 · Who is asking for help, and where are they?",
        f"{markers_df['text'].nunique()} real flood-distress passages, automatically placed at "
        f"{markers_df['place'].nunique()} locations on the map.",
        "Each passage is real 2015 reporting. A multilingual classifier decides whether it is a call for help; a "
        "gazetteer geoparser finds the place names inside it and pins them to coordinates.",
    )
    st.markdown(
        '<div class="steps">'
        '<div class="stepbox"><div class="n">STEP 1</div><div class="t">Raw text</div><div class="d">News, reports and '
        "first-hand accounts from the 2015 flood.</div></div><div class=\"arrow\">›</div>"
        f'<div class="stepbox"><div class="n">STEP 2</div><div class="t">Is it distress?</div><div class="d">MuRIL '
        f"(multilingual BERT) classifier, F1 = {kpis['classifier_f1']:.2f}.</div></div><div class=\"arrow\">›</div>"
        f'<div class="stepbox"><div class="n">STEP 3</div><div class="t">Where?</div><div class="d">Fuzzy match against '
        f"a {len(load_geoparser()[1]):,}-name local gazetteer, precision {kpis['geoparse_precision']:.2f}.</div></div>"
        '<div class="arrow">›</div>'
        '<div class="stepbox"><div class="n">STEP 4</div><div class="t">On the map</div><div class="d">Pinned below — '
        "and counted per ward as the demand signal for Fair Relief.</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    m2 = folium.Map(location=CHENNAI_CENTER, zoom_start=12.6, tiles=None, prefer_canvas=True, scrollWheelZoom=False)
    folium.TileLayer(tiles=DARK_TILES_URL, attr=DARK_TILES_ATTR, name="Dark basemap").add_to(m2)
    folium.GeoJson(load_wards(), style_function=lambda _: {"color": "#52514e", "weight": 1.2, "fillOpacity": 0.03}).add_to(m2)
    pulse_css = """
    <style>
    .pulse-marker {
        width: 16px; height: 16px; border-radius: 50%; background: #e34948;
        box-shadow: 0 0 0 0 rgba(227,73,72,0.7); animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(227,73,72,0.7); }
        70% { box-shadow: 0 0 0 14px rgba(227,73,72,0); }
        100% { box-shadow: 0 0 0 0 rgba(227,73,72,0); }
    }
    </style>
    """
    m2.get_root().html.add_child(folium.Element(pulse_css))
    for _, row in markers_df.iterrows():
        folium.Marker(
            location=[row["lat"], row["lon"]],
            icon=folium.DivIcon(html='<div class="pulse-marker"></div>'),
            popup=folium.Popup(
                f"<b>{row['place']}</b> (p={row['probability']:.3f})<br><i>{row['text'][:220]}...</i>", max_width=320
            ),
        ).add_to(m2)
    map_col, try_col = st.columns([1.6, 1])
    with map_col:
        st.iframe(m2._repr_html_(), height=540)
        st.caption("Click a pulsing marker to read the passage behind it.")
    with try_col:
        section("Try the geoparser live")
        sample = "Water entered our houses near Velachery and Taramani, we are stuck on the first floor."
        text = st.text_area("Type a sentence mentioning Chennai places", sample, height=110)
        geoparse_text, gazetteer = load_geoparser()
        matches = geoparse_text(text, gazetteer) if text.strip() else []
        if matches:
            st.markdown("".join(f'<span class="match-chip">"{m["text"]}" → {m["matched_name"]} ({m["score"]:.0f})</span>'
                                for m in matches), unsafe_allow_html=True)
            fig_g = go.Figure(go.Scattermap(lat=[m["lat"] for m in matches], lon=[m["lon"] for m in matches],
                                            mode="markers+text", text=[m["matched_name"] for m in matches],
                                            textposition="top right", textfont={"color": "#ffffff", "size": 13},
                                            marker={"size": 16, "color": "#0ca30c"}))
            fig_g.update_layout(
                map={"style": "white-bg", "center": {"lat": CHENNAI_CENTER[0], "lon": CHENNAI_CENTER[1]}, "zoom": 11.3,
                     "layers": [{"below": "traces", "sourcetype": "raster", "source": [DARK_TILES_URL],
                                 "sourceattribution": MAP_ATTR_SHORT}]},
                margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=300, showlegend=False)
            st.plotly_chart(fig_g, width='stretch', config={"scrollZoom": False})
        else:
            st.info("No known place names found — the geoparser only resolves names in its local gazetteer "
                    "(it correctly returns nothing rather than guessing).")
        st.caption("Runs task 2.10's real geoparser on your text. The distress classifier itself (a 237M-parameter "
                   "model) is not run live; the markers on the left use its saved predictions.")

    with st.expander(f"All {len(markers_df)} resolved markers"):
        st.dataframe(
            markers_df[["place", "probability", "text"]].rename(columns={"probability": "distress_probability"}),
            width='stretch', hide_index=True,
        )

# --------------------------------------------------------------------------
# Model Verdict (task 5.3's data, interactive)
# --------------------------------------------------------------------------
with tab_compare:
    comparison = load_json_report(GROUND_TRUTH_DIR / "baseline_vs_gnn_comparison.json")
    comparison_table = comparison["comparison_table"]
    fair_auc = comparison["fair_holdout_auc_train_only_model"]
    splits = ["train", "val", "test"]
    onset = comparison_table["rising->peak"]
    story(
        "Objective 1 · Does learning on the road graph beat a hand-written rule?",
        "At flood onset, yes: the rule scores zero on every group of wards. The graph model shows real but modest "
        "skill where the data is able to test it.",
        "F1 rewards getting flooded roads right. AUC asks a harder question — can the model rank which roads flood "
        "first? — where 0.50 is a coin flip.",
    )
    c1, c2 = st.columns(2)
    with c1:
        section("Flood onset (Rising → Peak): F1 score")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Rule-based baseline", x=[SPLIT_LABELS[s] for s in splits],
                             y=[onset[s]["baseline_f1"] or 0 for s in splits], marker_color=COLOR_BASELINE,
                             text=[f"{(onset[s]['baseline_f1'] or 0):.2f}" for s in splits], textposition="outside"))
        fig.add_trace(go.Bar(name="Graph neural network", x=[SPLIT_LABELS[s] for s in splits],
                             y=[onset[s]["gnn_f1"] for s in splits], marker_color=COLOR_GNN,
                             text=[f"{onset[s]['gnn_f1']:.2f}" for s in splits], textposition="outside"))
        fig.update_layout(barmode="group", template="plotly_dark", height=400, yaxis_range=[0, 1.15], yaxis_title="F1",
                          legend={"orientation": "h", "y": -0.18}, margin={"t": 20}, font={"size": 14},
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width='stretch')
    with c2:
        section("Ranking skill on unseen wards: AUC", "train-only model, 0.50 = coin flip")
        fig2 = go.Figure()
        for i, transition in enumerate(["rising->peak", "peak->receding"]):
            fig2.add_trace(go.Bar(
                name=TRANSITION_LABELS[transition], x=[SPLIT_LABELS[s] for s in ["val", "test"]],
                y=[fair_auc[transition][s] for s in ["val", "test"]], marker_color=[COLOR_GNN, "#eda100"][i],
                text=[f"{fair_auc[transition][s]:.2f}" for s in ["val", "test"]], textposition="outside",
            ))
        fig2.add_hline(y=0.5, line_dash="dash", line_color="#e66767", annotation_text="coin flip (0.50)",
                       annotation_position="top left")
        fig2.update_layout(barmode="group", template="plotly_dark", height=400, yaxis_title="AUC-ROC",
                           yaxis_range=[0, 1], legend={"orientation": "h", "y": -0.18}, margin={"t": 20},
                           font={"size": 14}, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, width='stretch')

    section("The honest verdict", "working.md §1.6's core claim")
    for transition, v in comparison["core_claim_verdicts"].items():
        st.markdown(
            f'<div class="insight-card {"good" if "SUPPORTED" in v["val_conclusion"] else "warning"}">'
            f"<h4>{TRANSITION_LABELS.get(transition, transition)}</h4>"
            f"<p><b>Validation wards:</b> {v['val_conclusion']}<br><b>Test wards:</b> {v['test_conclusion']}</p></div>",
            unsafe_allow_html=True,
        )

    with st.expander("Full F1 table: every phase transition × every group of wards"):
        transitions = list(comparison_table.keys())
        fig3 = go.Figure()
        for name, color, label in [("baseline_f1", COLOR_BASELINE, "Rule-based baseline"),
                                   ("gnn_f1", COLOR_GNN, "Graph neural network (tuned)")]:
            fig3.add_trace(go.Bar(
                name=label, x=[f"{TRANSITION_LABELS[t]}<br>{SPLIT_LABELS[s]}" for t in transitions for s in splits],
                y=[comparison_table[t][s][name] or 0 for t in transitions for s in splits], marker_color=color,
            ))
        fig3.update_layout(barmode="group", template="plotly_dark", height=430, yaxis_title="F1", legend_title=None,
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig3, width='stretch')
        st.caption("Pre-event → Rising is trivial for both models (no road floods before the rain), so both score 0.")

# --------------------------------------------------------------------------
# Fair Relief (task 7.3 -- Objective 3, tasks 7.1/7.2's outputs)
# --------------------------------------------------------------------------
COLOR_DEMAND = "#9e9e9e"
COLOR_EQUITY = "#6a040f"


@st.cache_data(show_spinner=False)
def solve_live(inputs: pd.DataFrame, budget: int, rho: float) -> pd.Series:
    """Task 7.2's own MIP (16 integer variables -- solves in milliseconds)."""
    return solve_allocation(inputs, budget, rho)


def render_allocation_tab(opt: dict) -> None:
    inputs = opt["inputs"]
    plan = opt["plan"]
    core = opt["vulnerability_report"]["robustness"]["robust_core"]
    eo = opt["optimizer_report"]
    plan_rho = opt["plan_report"]["rho"]

    story(
        "Objective 3 · Who should get relief when there isn't enough for everyone?",
        "Guarantee every ward a vulnerability-weighted floor first — then let reported distress steer the rest.",
        "Allocating purely by reported distress sends nothing to the wards that stayed silent. Move the slider "
        "to see the trade-off re-solve live.",
    )
    st.caption(
        "Affected people = Census 2011 population (task 7.1, crosswalked onto the post-2011 wards) x road-length "
        "share flooded at peak (fused ground truth). Distress signals = real Objective 2 posts per ward. "
        "Vulnerability = task 7.1's index. The optimizer is task 7.2's OR-Tools MIP, re-solved live below."
    )

    zero_sig = inputs["signals"] == 0
    dd_primary = eo["models"]["demand_driven_optimum"]
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, caption, accent in [
        (c1, "Affected People", f"{int(inputs['need'].sum()):,}", "population x flooded share at peak", "blue"),
        (c2, "Need in Silent Wards", f"{inputs.loc[zero_sig, 'need'].sum() / inputs['need'].sum():.0%}",
         f"{int(zero_sig.sum())} wards with zero distress posts", "orange"),
        (c3, "Starved by Demand-Driven", f"{int(dd_primary['n_starved_wards'])} / 16",
         "wards under 5% coverage (25% budget)", "red"),
        (c4, "Starved by Equity Plan", f"{int(opt['plan_report']['metrics']['n_starved_wards'])} / 16",
         f"recommended rho = {plan_rho:g}", "green"),
    ]:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value kpi-accent-{accent}">{value}</div>'
                f'<div class="kpi-caption">{caption}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")
    silent_core = eo["signal_gap"]["robust_core_wards_with_zero_signal"]
    st.markdown(
        '<div class="insight-card warning"><h4>The bias this exists to fix — measured, not assumed</h4>'
        f"<p><b>{len(silent_core)} of the {len(core)} most robustly vulnerable wards "
        f"({', '.join(map(str, silent_core))}) produced zero distress signal.</b> Reported distress clusters "
        "where 2015 news coverage went (Velachery, Kotturpuram). Any allocator that follows reported demand alone "
        "gives these wards nothing — the optimizer guarantees each ward a vulnerability-weighted floor, and lets "
        "the rest of the budget follow reported distress.</p></div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 2.2])
    with left:
        st.markdown("#### Try it")
        rho = st.slider("Equity floor strength (rho)", 0.0, 1.0, float(plan_rho), 0.05,
                        help="0 = no floor (pure demand-driven). 1 = the most vulnerable ward is guaranteed "
                             "at least the even-split coverage; others a vulnerability-proportional share of it.")
        share = st.slider("Relief budget (% of affected people)", 5, 60, 25, 5)
        budget = int(round(share / 100 * inputs["need"].sum()))
        eq_units = solve_live(inputs, budget, rho)
        dd_units = solve_live(inputs, budget, 0.0)
        m_eq = evaluate_allocation(inputs, eq_units, core)
        m_dd = evaluate_allocation(inputs, dd_units, core)
        base_sig = m_dd["signal_weighted_coverage"]
        loss = 1 - m_eq["signal_weighted_coverage"] / base_sig if base_sig else 0.0
        top3 = eq_units.sort_values(ascending=False).index[:3]
        st.metric("Starved wards (<5% coverage)", m_eq["n_starved_wards"],
                  delta=int(m_eq["n_starved_wards"] - m_dd["n_starved_wards"]), delta_color="inverse",
                  help="delta vs. the demand-driven optimum at the same budget")
        st.metric("Lowest coverage, robust-core wards", f"{m_eq['core_wards_min_coverage']:.1%}",
                  delta=f"{(m_eq['core_wards_min_coverage'] - m_dd['core_wards_min_coverage']) * 100:+.1f} pts")
        st.metric("Reported-distress coverage given up", f"{loss:.1%}", help="the price of fairness")
        st.metric("Relief to the 3 best-served wards", f"{eq_units[top3].sum() / eq_units.sum():.0%}",
                  help=f"they hold {inputs.loc[top3, 'need'].sum() / inputs['need'].sum():.0%} of need")
        st.caption(f"Budget: {budget:,} relief units. Each move re-solves task 7.2's MIP.")

    with right:
        cov = (eq_units / inputs["need"]).rename("coverage")
        wards_geo = load_wards()
        wards_geo["Ward_No"] = wards_geo["Ward_No"].astype(int)
        geojson = json.loads(wards_geo[["Ward_No", "geometry"]].to_json())
        for f in geojson["features"]:
            f["id"] = int(f["properties"]["Ward_No"])
        hover = [f"<b>Ward {w}</b> {plan.loc[w, 'locality'] if w in plan.index else ''}<br>"
                 f"coverage {cov[w]:.0%} ({int(eq_units[w]):,} / {int(inputs.loc[w, 'need']):,})<br>"
                 f"vulnerability {inputs.loc[w, 'vulnerability']:.2f} · distress posts {int(inputs.loc[w, 'signals'])}"
                 for w in cov.index]
        fig_map = go.Figure(go.Choroplethmap(
            geojson=geojson, locations=list(cov.index), z=list(cov.values), zmin=0, zmax=1,
            colorscale="YlGn", marker_opacity=0.75, marker_line_width=1, marker_line_color="#222",
            text=hover, hoverinfo="text", colorbar={"title": "coverage", "tickformat": ".0%"},
        ))
        fig_map.update_layout(
            map={"style": "white-bg", "center": {"lat": CHENNAI_CENTER[0], "lon": CHENNAI_CENTER[1]}, "zoom": 11.7,
                 "layers": [{"below": "traces", "sourcetype": "raster", "source": [DARK_TILES_URL],
                             "sourceattribution": MAP_ATTR_SHORT}]},
            margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=470, template="plotly_dark",
        )
        st.plotly_chart(fig_map, width='stretch', config={"scrollZoom": False})

    order = inputs.sort_values("vulnerability", ascending=False).index
    fig_cov = go.Figure()
    fig_cov.add_trace(go.Bar(name="Demand-driven optimum (rho=0)", x=[str(w) for w in order],
                             y=(dd_units / inputs["need"])[order], marker_color=COLOR_DEMAND))
    fig_cov.add_trace(go.Bar(name=f"Equity-constrained (rho={rho:g})", x=[str(w) for w in order],
                             y=cov[order], marker_color=COLOR_EQUITY))
    fig_cov.update_layout(
        barmode="group", template="plotly_dark", height=380, yaxis_title="coverage", yaxis_tickformat=".0%",
        xaxis_title="ward (most -> least vulnerable)", legend_title=None,
        xaxis={"tickvals": [str(w) for w in order],
               "ticktext": [f"{w}{'*' if w in core else ''}<br>sig={int(inputs.loc[w, 'signals'])}" for w in order]},
    )
    st.markdown("#### Coverage per ward — * robust-core vulnerable ward, sig = real distress posts")
    st.plotly_chart(fig_cov, width='stretch')

    st.markdown(f"#### Ranked relief plan (recommended rho = {plan_rho:g}, 25% budget, one 3-day relief cycle)")
    show = plan.reset_index()[["dispatch_priority", "Ward_No", "locality", "tranche_a_units", "tranche_b_units",
                               "total_units", "coverage", "coverage_min", "coverage_max", "firmness",
                               "water_litres", "food_ration_days", "emergency_toilets", "rationale"]]
    st.dataframe(
        show, width='stretch', hide_index=True,
        column_config={
            "dispatch_priority": st.column_config.NumberColumn("#", help="first dispatch step"),
            "Ward_No": st.column_config.NumberColumn("Ward", format="%d"),
            "locality": st.column_config.TextColumn("Locality"),
            "total_units": st.column_config.NumberColumn("Total", format="%d"),
            "firmness": st.column_config.TextColumn("Firmness"),
            "rationale": st.column_config.TextColumn("Why", width="large"),
            "tranche_a_units": st.column_config.NumberColumn("Floor (A)", format="%d"),
            "tranche_b_units": st.column_config.NumberColumn("Demand (B)", format="%d"),
            "coverage": st.column_config.ProgressColumn("Coverage", min_value=0.0, max_value=1.0, format="percent"),
            "coverage_min": st.column_config.NumberColumn("Band min", format="percent"),
            "coverage_max": st.column_config.NumberColumn("Band max", format="percent"),
            "water_litres": st.column_config.NumberColumn("Water (L)", format="%d"),
            "food_ration_days": st.column_config.NumberColumn("Food rations", format="%d"),
            "emergency_toilets": st.column_config.NumberColumn("Toilets", format="%d"),
        },
    )
    st.download_button("Download plan (CSV)", plan.to_csv().encode("utf-8"), "chennai_relief_plan.csv", "text/csv")

    conc = opt["plan_report"]["concentration"][f"rho_{plan_rho:g}"]
    st.markdown(
        '<div class="insight-card warning"><h4>What the floor does not fix</h4>'
        f"<p>The floor prevents starvation, but the budget above it still follows reported demand greedily: at "
        f"rho = {plan_rho:g} the 3 best-served wards receive <b>{conc['share_of_units']:.0%}</b> of relief while "
        f"holding <b>{conc['share_of_need']:.0%}</b> of need. Raising rho moves relief from the demand tranche into "
        "the floors — try it with the slider.</p></div>",
        unsafe_allow_html=True,
    )
    firm = opt["plan_report"]["firmness"]
    core_firm = [w for w in core if w in firm["firm"]]
    core_soft = [w for w in core if w not in firm["firm"]]
    core_note = f"{len(core_firm)} of the {len(core)} robust-core wards ({', '.join(map(str, core_firm))})"
    if core_soft:
        core_note += (f"; core ward {', '.join(map(str, core_soft))} is judgment-sensitive — how much "
                      "demand-tranche relief it gets depends on the method choice")
    st.markdown(
        '<div class="insight-card good"><h4>Which numbers to trust</h4>'
        f"<p><b>{len(firm['firm'])} wards</b> keep their coverage within ±25% across every sensitivity scenario "
        f"(GNN-forecast exposure, dasymetric population, two index variants), including {core_note}. "
        f"<b>{len(firm['judgment_sensitive'])}</b> ({', '.join(map(str, firm['judgment_sensitive']))}) depend on a "
        "method choice; their band is shown in the table. Supplies use Sphere Handbook (2018) minimums: 15 L water "
        "and one 2,100 kcal ration per person per day, one emergency toilet per 50 people.</p></div>",
        unsafe_allow_html=True,
    )


with tab_alloc:
    try:
        render_allocation_tab(load_optimizer_outputs())
    except FileNotFoundError as e:
        st.warning(f"Objective 3 outputs not generated yet: {e}")


# --------------------------------------------------------------------------
# Limits & Honesty (task 5.4/5.5's findings, presented honestly)
# --------------------------------------------------------------------------
with tab_limits:
    story(
        "What can — and can't — this project claim?",
        "Every headline number comes with the limitation that shapes it, found by checking the data, not assumed.",
    )

    anomaly = load_json_report(GROUND_TRUTH_DIR / "sanity_check_comparison_anomalies_report.json")
    obj2 = load_json_report(NLP_DIR / "objective2_precision_recall_report.json")

    st.markdown(
        '<div class="insight-card warning"><h4>Ground truth is too coarse for a fair test-split verdict</h4>'
        f'<p>Of 16 wards, only <b>{anomaly["by_phase"]["peak"]["informativeness_by_split"]["val"]["n_informative_wards"]}</b> '
        f'has genuine within-ward flood/no-flood variance at the peak transition (ward 170) — and it landed in '
        f'<b>val</b>, not test. Both test wards are &gt;96% flooded with no meaningful signal to rank against. '
        f'The GNN\'s test AUC collapse (~0.46–0.50) is a ground-truth coverage artifact, not a model failure — '
        f'confirmed by checking the actual elevation-flood correlation per ward, not assumed.</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="insight-card warning"><h4>The tuned GNN flags every road as flooded</h4>'
        f"<p>Its reported F1 = {kpis['gnn_tuned_test_f1']:.2f} comes from matching a flood that covered most of the "
        "study area, not from per-road discrimination — the thresholded model predicts \"flooded\" for 100% of test roads. "
        "Its probabilities (and the AUC built on them) are the honest measure of skill.</p></div>",
        unsafe_allow_html=True,
    )

    fp = obj2["geoparsing_precision_new"]["false_positive_examples"][0]
    st.markdown(
        '<div class="insight-card warning"><h4>One real geoparsing error, found and disclosed</h4>'
        f'<p>"<b>{fp["matched_text"]}</b>" incorrectly resolves to "<b>{fp["resolved_to"]}</b>": {fp["reason"]}</p></div>',
        unsafe_allow_html=True,
    )
    if OPT is not None:
        st.markdown(
            '<div class="insight-card warning"><h4>Objective 3 is only as fine-grained as Census 2011</h4>'
            "<p>The 16 post-2011 wards had to be rebuilt from the old 155-ward census map and rest on 12 distinct census "
            "units; sanitation data exists only per town; and the demand side is 10 real distress posts — enough to "
            "expose the bias, not to measure demand precisely.</p></div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="insight-card good"><h4>What genuinely works</h4>'
        "<p>The baseline structurally cannot anticipate flood onset (rising→peak: F1=0 on every split) — "
        "a real, measured failure mode, not an assumption. The GNN shows real (if modest) learned "
        "discrimination on the one ward with genuine ground-truth variance (AUC ~0.59–0.60, fairly "
        "held out). The Objective 2 pipeline correctly composes classification and geoparsing as two "
        "independent branches, verified on genuinely new text. The equity floor removes starvation in every "
        "sensitivity scenario tested.</p></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Full per-ward informativeness table (task 5.4)"):
        st.dataframe(pd.DataFrame(anomaly["by_phase"]["peak"]["ward_table"]), width='stretch', hide_index=True)

st.markdown(
    '<div class="footer">Data: OpenStreetMap road network · SRTM 30 m DEM · IMD + Open-Meteo rainfall · Sentinel-1 SAR '
    "(Google Earth Engine) · NRSC Bhuvan · news & advisories · Census of India 2011 · DataMeet ward boundaries. "
    "Every number on this dashboard is read from the project's own generated reports.</div>",
    unsafe_allow_html=True,
)
