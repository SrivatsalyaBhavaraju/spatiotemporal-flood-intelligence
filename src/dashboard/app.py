"""
Task 6.1/6.2/6.3 -- Streamlit + Folium integration dashboard. This is
where Objective 1 (flood propagation, baseline vs. GNN) and Objective 2
(distress classification + geoparsing) visibly become one system. Task 7.3
adds the Resource Allocation tab: Objective 3's equity-constrained
optimizer, re-solved live (task 7.2's own MIP) as the rho/budget sliders move.

Run with:
    streamlit run src/dashboard/app.py

All data is real -- pre-joined by prepare_dashboard_data.py or read
directly from tasks 3.4/4.3/5.1/5.3/5.4/5.5's own generated reports.
Nothing here recomputes a model or invents a number.
"""
import json
import sys
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
from src.optimizer.equity_optimizer import evaluate_allocation, solve_allocation  # noqa: E402

CHENNAI_CENTER = [12.985, 80.225]
# CartoDB's free dark tiles now require an API key (discovered while testing this dashboard --
# the map silently failed to render with no console error). Esri's dark-gray-canvas tile service
# needs no key and gives the same dark aesthetic.
DARK_TILES_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
DARK_TILES_ATTR = "Esri, HERE, Garmin, FAO, NOAA, USGS, © OpenStreetMap contributors, GIS User Community"

st.set_page_config(page_title="Chennai Flood Intelligence", page_icon="\U0001F30A", layout="wide")

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', system-ui, sans-serif; }

.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
    background: linear-gradient(90deg, #eb6834 0%, #eda100 50%, #e34948 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-subtitle { color: #c3c2b7; font-size: 1.05rem; margin-top: 0.2rem; margin-bottom: 1.5rem; }

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
.insight-card p { margin: 0; color: #c3c2b7; font-size: 0.92rem; line-height: 1.5; }

.legend-swatch { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="hero-title">Chennai 2015 Flood Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Spatiotemporal graph learning vs. a rule-based baseline, '
    'plus a distress-classification + geoparsing pipeline — one integrated system.</div>',
    unsafe_allow_html=True,
)

try:
    kpis = compute_overview_kpis()
except FileNotFoundError as e:
    st.error(f"Dashboard data not ready: {e}\n\nRun `python src/dashboard/prepare_dashboard_data.py` first.")
    st.stop()


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

    m = folium.Map(location=CHENNAI_CENTER, zoom_start=12.3, tiles=None, prefer_canvas=True)
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


tab_overview, tab_map, tab_distress, tab_compare, tab_alloc, tab_limits = st.tabs(
    ["\U0001F4CA Overview", "\U0001F30A Flood Propagation", "\U0001F4E1 Distress Signals",
     "\U0001F4C8 Model Comparison", "\u2696\ufe0f Resource Allocation", "\U0001F50D Data & Limitations"]
)

# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    tiles = [
        (c1, "Road Segments", f"{kpis['n_segments']:,}", "across 16 study wards", "blue"),
        (c2, "Peak Flood Coverage", f"{kpis['peak_flooded_pct']}%", "of segments, at event peak", "red"),
        (c3, "GNN Fair-Holdout AUC (val)", f"{kpis['gnn_val_auc']:.2f}", "vs. 0.50 = random chance", "green"),
        (c4, "Distress Classifier F1", f"{kpis['classifier_f1']:.2f}", "17-example LOOCV (task 3.8)", "orange"),
    ]
    for col, label, value, caption, accent in tiles:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value kpi-accent-{accent}">{value}</div>'
                f'<div class="kpi-caption">{caption}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    c5, c6, c7, c8 = st.columns(4)
    tiles2 = [
        (c5, "Baseline Test F1 (rising→peak)", f"{kpis['baseline_test_f1']:.2f}", "structurally can't anticipate onset", "red"),
        (c6, "GNN Tuned Test F1 (rising→peak)", f"{kpis['gnn_tuned_test_f1']:.2f}", "see Data & Limitations tab", "orange"),
        (c7, "Geoparsing Precision", f"{kpis['geoparse_precision']:.3f}", "36/37 real matches reviewed", "blue"),
        (c8, "Geoparsing Recall", f"{kpis['geoparse_recall']:.2f}", "vs. task 1.13's real corpus", "green"),
    ]
    for col, label, value, caption, accent in tiles2:
        with col:
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
                f'<div class="kpi-value kpi-accent-{accent}">{value}</div>'
                f'<div class="kpi-caption">{caption}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown(
        '<div class="insight-card"><h4>What this dashboard shows</h4>'
        "<p>The <b>Flood Propagation</b> tab animates real ground truth and both models' predictions "
        "across the 2015 Chennai flood's four phases. The <b>Distress Signals</b> tab plots real "
        "distress-text passages resolved to coordinates via Objective 2's classifier + geoparser. "
        "<b>Model Comparison</b> tests working.md's core claim with real, honest numbers — including "
        "where the ground truth itself limits what can be concluded (see <b>Data & Limitations</b>). "
        "<b>Resource Allocation</b> closes the loop (Objective 3): flood state + distress signals + Census "
        "vulnerability feed an optimizer that keeps silent, vulnerable wards from being starved of relief.</p></div>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Flood Propagation Map (task 6.2)
# --------------------------------------------------------------------------
with tab_map:
    left, right = st.columns([1, 3])
    with left:
        st.markdown("#### Controls")
        view = st.radio("Show", ["Ground Truth", "Baseline Prediction", "GNN Prediction"], index=0)
        column_map = dict(VIEW_COLUMNS)

        st.caption(
            "Play/Pause on the map animate all 4 phases directly in your browser "
            "(no reloading) — the phase label above the map updates in place."
        )
        if view != "Ground Truth":
            st.info("Pre-Event is never a model target (task 2.7's schema) — always shown dry for Baseline/GNN too.")

        st.markdown("#### Legend")
        st.markdown(
            f'<span class="legend-swatch" style="background:{COLOR_DRY}"></span>Dry / not flooded<br>'
            f'<span class="legend-swatch" style="background:{COLOR_FLOODED}"></span>Flooded',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(f"**{view}** (all 4 phases — use the map's own Play button)")
        map_html = build_animated_map_html(view, column_map[view])
        components.html(map_html, height=620)

# --------------------------------------------------------------------------
# Distress Signals (task 6.3)
# --------------------------------------------------------------------------
with tab_distress:
    st.markdown("#### Objective 2: distress posts resolved to a coordinate")
    st.caption(
        "Real passages from task 1.13's corpus, classified as distress (task 3.8's MuRIL classifier) "
        "and geoparsed to a real location (task 2.10) — working.md §2.4's actual deliverable."
    )
    markers_df = load_distress_markers()

    m2 = folium.Map(location=CHENNAI_CENTER, zoom_start=12.3, tiles=None, prefer_canvas=True)
    folium.TileLayer(tiles=DARK_TILES_URL, attr=DARK_TILES_ATTR, name="Dark basemap").add_to(m2)
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
    components.html(m2._repr_html_(), height=560)

    st.markdown(f"**{len(markers_df)} resolved distress markers** across {markers_df['place'].nunique()} unique places")
    st.dataframe(
        markers_df[["place", "probability", "text"]].rename(columns={"probability": "distress_probability"}),
        width='stretch', hide_index=True,
    )

# --------------------------------------------------------------------------
# Model Comparison (task 5.3's data, interactive)
# --------------------------------------------------------------------------
with tab_compare:
    comparison = load_json_report(GROUND_TRUTH_DIR / "baseline_vs_gnn_comparison.json")
    comparison_table = comparison["comparison_table"]
    fair_auc = comparison["fair_holdout_auc_train_only_model"]

    st.markdown("#### F1 per transition per split — baseline vs. tuned GNN")
    transitions = list(comparison_table.keys())
    splits = ["train", "val", "test"]
    fig = go.Figure()
    for name, color in [("baseline_f1", COLOR_BASELINE), ("gnn_f1", COLOR_GNN)]:
        fig.add_trace(go.Bar(
            name="Baseline" if name == "baseline_f1" else "GNN (tuned)",
            x=[f"{t}<br>{s}" for t in transitions for s in splits],
            y=[comparison_table[t][s][name] or 0 for t in transitions for s in splits],
            marker_color=color,
        ))
    fig.update_layout(barmode="group", template="plotly_dark", height=420, yaxis_title="F1", legend_title=None)
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Fair-holdout AUC-ROC (task 4.1's train-only model — val/test genuinely unseen)")
    fig2 = go.Figure()
    for i, transition in enumerate(["rising->peak", "peak->receding"]):
        fig2.add_trace(go.Bar(
            name=transition, x=splits, y=[fair_auc[transition][s] for s in splits],
            marker_color=[COLOR_BASELINE, COLOR_GNN][i],
        ))
    fig2.add_hline(y=0.5, line_dash="dash", line_color="#e66767", annotation_text="random chance")
    fig2.update_layout(barmode="group", template="plotly_dark", height=420, yaxis_title="AUC-ROC", yaxis_range=[0, 1])
    st.plotly_chart(fig2, width='stretch')

    st.markdown("#### Honest core-claim verdict (working.md §1.6)")
    for transition, v in comparison["core_claim_verdicts"].items():
        st.markdown(
            f'<div class="insight-card {"good" if "SUPPORTED" in v["val_conclusion"] else "warning"}">'
            f"<h4>{transition}</h4>"
            f"<p><b>Val:</b> {v['val_conclusion']}<br><b>Test:</b> {v['test_conclusion']}</p></div>",
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------------------
# Resource Allocation (task 7.3 -- Objective 3, tasks 7.1/7.2's outputs)
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

    st.markdown("#### Objective 3: equity-constrained relief allocation")
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
                             "sourceattribution": DARK_TILES_ATTR}]},
            margin={"l": 0, "r": 0, "t": 0, "b": 0}, height=470, template="plotly_dark",
        )
        st.plotly_chart(fig_map, width='stretch')

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
# Data & Limitations (task 5.4/5.5's findings, presented honestly)
# --------------------------------------------------------------------------
with tab_limits:
    st.markdown("#### Real, disclosed limitations — not hidden behind the headline numbers")

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

    fp = obj2["geoparsing_precision_new"]["false_positive_examples"][0]
    st.markdown(
        '<div class="insight-card warning"><h4>One real geoparsing error, found and disclosed</h4>'
        f'<p>"<b>{fp["matched_text"]}</b>" incorrectly resolves to "<b>{fp["resolved_to"]}</b>": {fp["reason"]}</p></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="insight-card good"><h4>What genuinely works</h4>'
        "<p>The baseline structurally cannot anticipate flood onset (rising→peak: F1=0 on every split) — "
        "a real, measured failure mode, not an assumption. The GNN shows real (if modest) learned "
        "discrimination on the one ward with genuine ground-truth variance (AUC ~0.59–0.60, fairly "
        "held out). The Objective 2 pipeline correctly composes classification and geoparsing as two "
        "independent branches, verified on genuinely new text.</p></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Full per-ward informativeness table (task 5.4)"):
        st.dataframe(pd.DataFrame(anomaly["by_phase"]["peak"]["ward_table"]), width='stretch', hide_index=True)
