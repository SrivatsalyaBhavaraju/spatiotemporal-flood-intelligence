"""
Unit tests for src/graph/build_static_features.py (task 2.3).

Builds tiny synthetic DEM/slope GeoTIFFs and a synthetic line-graph-nodes
GeoDataFrame directly, so raster sampling and distance_to_drain logic can be
verified without the real (large) SRTM files. Run against the real files via
`python src/graph/build_static_features.py`.

Run with:
    pytest tests/test_build_static_features.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.graph.build_static_features import (  # noqa: E402
    Phase2FeatureError,
    build_static_features,
    compute_distance_to_drain,
    load_drainage,
    load_line_graph_nodes,
    sample_rasters_for_segments,
    validate_static_features,
)

# Synthetic raster: 10x10 grid over lon [80.20, 80.21], lat [12.99, 13.00],
# pixel size 0.001 deg (~111m). Elevation increases linearly eastward (a
# clean, checkable gradient); slope is a flat constant for simplicity.
RASTER_WEST, RASTER_NORTH = 80.20, 13.00
PIXEL_DEG = 0.001
GRID_SIZE = 10


def _write_synthetic_rasters(tmp_path):
    transform = from_origin(RASTER_WEST, RASTER_NORTH, PIXEL_DEG, PIXEL_DEG)
    elev = np.tile(np.arange(GRID_SIZE, dtype="int16") * 10, (GRID_SIZE, 1))  # 0,10,...,90 eastward
    slope = np.full((GRID_SIZE, GRID_SIZE), 5.0, dtype="float32")

    dem_path = tmp_path / "srtm_dem.tif"
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=GRID_SIZE, width=GRID_SIZE, count=1,
        dtype="int16", crs="EPSG:4326", transform=transform, nodata=-32768,
    ) as dst:
        dst.write(elev, 1)

    slope_path = tmp_path / "srtm_slope.tif"
    with rasterio.open(
        slope_path, "w", driver="GTiff", height=GRID_SIZE, width=GRID_SIZE, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(slope, 1)

    return dem_path, slope_path


def _make_segments_gdf():
    # A short segment (~1.1km) running west-to-east through the raster's mid-latitude
    geoms = {
        "seg1": LineString([(80.2005, 12.995), (80.2095, 12.995)]),  # inside raster, spans the gradient
        "seg2": LineString([(80.5, 13.5), (80.501, 13.501)]),  # far outside raster extent -> all nodata
    }
    records = [{"segment_id": k, "length_m": 1100.0 if k == "seg1" else 150.0, "geometry": v} for k, v in geoms.items()]
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def _make_drainage_gdf(empty=False):
    if empty:
        return gpd.GeoDataFrame(columns=["waterway", "geometry"], geometry="geometry", crs="EPSG:4326")
    geom = LineString([(80.2050, 12.990), (80.2050, 13.000)])  # a north-south drain near seg1's midpoint
    return gpd.GeoDataFrame({"waterway": ["drain"]}, geometry=[geom], crs="EPSG:4326")


class TestLoaders:
    def test_missing_line_graph_nodes_raises(self, tmp_path):
        with pytest.raises(Phase2FeatureError, match="build_line_graph.py"):
            load_line_graph_nodes(tmp_path / "does_not_exist.geojson")

    def test_missing_required_column_raises(self, tmp_path):
        p = tmp_path / "bad_nodes.geojson"
        gpd.GeoDataFrame(
            {"segment_id": ["a"], "geometry": [LineString([(0, 0), (1, 1)])]},  # no length_m
            crs="EPSG:4326",
        ).to_file(p, driver="GeoJSON")
        with pytest.raises(Phase2FeatureError, match="length_m"):
            load_line_graph_nodes(p)

    def test_load_drainage_missing_returns_none(self, tmp_path):
        gdf, path = load_drainage(tmp_path / "a.geojson", tmp_path / "b.geojson")
        assert gdf is None
        assert path is None

    def test_load_drainage_prefers_primary_over_fallback(self, tmp_path):
        primary = tmp_path / "primary.geojson"
        fallback = tmp_path / "fallback.geojson"
        _make_drainage_gdf().to_file(primary, driver="GeoJSON")
        _make_drainage_gdf().to_file(fallback, driver="GeoJSON")
        gdf, path = load_drainage(primary, fallback)
        assert path == primary

    def test_load_drainage_falls_back_when_primary_missing(self, tmp_path):
        primary = tmp_path / "does_not_exist.geojson"
        fallback = tmp_path / "fallback.geojson"
        _make_drainage_gdf().to_file(fallback, driver="GeoJSON")
        gdf, path = load_drainage(primary, fallback)
        assert path == fallback


class TestRasterSampling:
    def test_elevation_gradient_sampled_correctly(self, tmp_path):
        dem_path, slope_path = _write_synthetic_rasters(tmp_path)
        segments = _make_segments_gdf().iloc[[0]]  # just seg1, which crosses the gradient
        elevations, slopes, n_elev_valid, n_slope_valid, n_samples = sample_rasters_for_segments(
            segments, dem_path, slope_path
        )
        # seg1 spans roughly the middle of the eastward gradient -> mean should be well within [0, 90]
        assert 0 <= elevations[0] <= 90
        assert n_elev_valid[0] > 0
        assert slopes[0] == pytest.approx(5.0)  # constant slope raster
        assert n_slope_valid[0] > 0

    def test_out_of_extent_segment_returns_nan_with_zero_valid_samples(self, tmp_path):
        dem_path, slope_path = _write_synthetic_rasters(tmp_path)
        segments = _make_segments_gdf().iloc[[1]]  # seg2, far outside raster
        elevations, slopes, n_elev_valid, n_slope_valid, n_samples = sample_rasters_for_segments(
            segments, dem_path, slope_path
        )
        assert np.isnan(elevations[0])
        assert n_elev_valid[0] == 0

    def test_more_samples_for_longer_segments(self, tmp_path):
        dem_path, slope_path = _write_synthetic_rasters(tmp_path)
        short = gpd.GeoDataFrame(
            {"segment_id": ["s"], "length_m": [10.0], "geometry": [LineString([(80.205, 12.995), (80.2051, 12.995)])]},
            crs="EPSG:4326",
        )
        long = gpd.GeoDataFrame(
            {"segment_id": ["l"], "length_m": [900.0], "geometry": [LineString([(80.2005, 12.995), (80.2095, 12.995)])]},
            crs="EPSG:4326",
        )
        _, _, _, _, n_short = sample_rasters_for_segments(short, dem_path, slope_path)
        _, _, _, _, n_long = sample_rasters_for_segments(long, dem_path, slope_path)
        assert n_short[0] == 2  # floor: two endpoints
        assert n_long[0] > n_short[0]


class TestDistanceToDrain:
    def test_none_drainage_returns_all_nan(self):
        segments = _make_segments_gdf()
        distances, utm_crs = compute_distance_to_drain(segments, None)
        assert np.all(np.isnan(distances))
        assert utm_crs is None

    def test_empty_drainage_returns_all_nan(self):
        segments = _make_segments_gdf()
        distances, utm_crs = compute_distance_to_drain(segments, _make_drainage_gdf(empty=True))
        assert np.all(np.isnan(distances))

    def test_nonempty_drainage_returns_positive_finite_distances(self):
        segments = _make_segments_gdf()
        distances, utm_crs = compute_distance_to_drain(segments, _make_drainage_gdf())
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)
        assert utm_crs is not None

    def test_closer_segment_gets_smaller_distance(self):
        # seg1 passes right through the drain's longitude; a segment far
        # away should get a strictly larger distance_to_drain
        near = gpd.GeoDataFrame(
            {"segment_id": ["near"], "length_m": [50.0], "geometry": [LineString([(80.2049, 12.995), (80.2051, 12.995)])]},
            crs="EPSG:4326",
        )
        far = gpd.GeoDataFrame(
            {"segment_id": ["far"], "length_m": [50.0], "geometry": [LineString([(80.22, 12.995), (80.221, 12.995)])]},
            crs="EPSG:4326",
        )
        both = gpd.GeoDataFrame(pd_concat_safe(near, far), crs="EPSG:4326")
        distances, _ = compute_distance_to_drain(both, _make_drainage_gdf())
        assert distances[0] < distances[1]


def pd_concat_safe(a, b):
    import pandas as pd
    return pd.concat([a, b], ignore_index=True)


class TestBuildStaticFeatures:
    def test_end_to_end(self, tmp_path):
        dem_path, slope_path = _write_synthetic_rasters(tmp_path)
        nodes_path = tmp_path / "line_graph_nodes.geojson"
        _make_segments_gdf().to_file(nodes_path, driver="GeoJSON")
        drainage_path = tmp_path / "drainage.geojson"
        _make_drainage_gdf().to_file(drainage_path, driver="GeoJSON")
        missing_fallback = tmp_path / "does_not_exist.geojson"

        features = build_static_features(
            line_graph_nodes_path=nodes_path,
            dem_path=dem_path,
            slope_path=slope_path,
            drainage_primary=drainage_path,
            drainage_fallback=missing_fallback,
        )
        assert len(features) == 2
        assert set(["segment_id", "length_m", "elevation_m", "slope_deg", "distance_to_drain_m"]).issubset(features.columns)
        assert features.loc["seg1", "length_m"] == 1100.0
        assert not np.isnan(features.loc["seg1", "elevation_m"])  # in-extent
        assert np.isnan(features.loc["seg2", "elevation_m"])  # out-of-extent


class TestValidation:
    def test_reports_missing_and_summary_stats(self, tmp_path):
        dem_path, slope_path = _write_synthetic_rasters(tmp_path)
        nodes_path = tmp_path / "line_graph_nodes.geojson"
        _make_segments_gdf().to_file(nodes_path, driver="GeoJSON")
        drainage_path = tmp_path / "drainage.geojson"
        _make_drainage_gdf().to_file(drainage_path, driver="GeoJSON")

        features = build_static_features(
            line_graph_nodes_path=nodes_path, dem_path=dem_path, slope_path=slope_path,
            drainage_primary=drainage_path, drainage_fallback=tmp_path / "nope.geojson",
        )
        report = validate_static_features(features)
        assert report["counts"]["segments"] == 2
        assert report["elevation_m"]["missing_count"] == 1  # seg2, out of extent
        assert "seg2" in report["elevation_m"]["missing_segment_ids_sample"]
        assert report["distance_to_drain_m"]["missing_count"] == 0
        assert report["length_m"]["missing_count"] == 0
