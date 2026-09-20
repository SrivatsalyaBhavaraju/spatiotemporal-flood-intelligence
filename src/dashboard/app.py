"""
Task 6.1/6.2/6.3 -- Streamlit + Folium integration dashboard. This is
where Objective 1 (flood propagation, baseline vs. GNN) and Objective 2
(distress classification + geoparsing) visibly become one system.

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
    load_phase_layer,
    load_wards,
)

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
    """
    base = load_phase_layer(PHASES[0])[["segment_id", "geometry"]].reset_index(drop=True)
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
            var sublayers = {geo.get_name()}.getLayers();
            var idx = 0;
            var timer = null;
            function applyPhase(i) {{
                for (var j = 0; j < sublayers.length; j++) {{
                    sublayers[j].setStyle({{color: colors[i][j]}});
                }}
                document.getElementById('phase-indicator-{geo.get_name()}').innerText = labels[i];
            }}
            document.getElementById('play-btn-{geo.get_name()}').onclick = function() {{
                if (timer) return;
                timer = setInterval(function() {{
                    idx = (idx + 1) % colors.length;
                    applyPhase(idx);
                }}, 1200);
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


tab_overview, tab_map, tab_distress, tab_compare, tab_limits = st.tabs(
    ["\U0001F4CA Overview", "\U0001F30A Flood Propagation", "\U0001F4E1 Distress Signals",
     "\U0001F4C8 Model Comparison", "\U0001F50D Data & Limitations"]
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
        "where the ground truth itself limits what can be concluded (see <b>Data & Limitations</b>).</p></div>",
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
