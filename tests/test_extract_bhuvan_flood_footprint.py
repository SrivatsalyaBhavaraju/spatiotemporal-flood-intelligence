"""
Unit tests for src/ground_truth/extract_bhuvan_flood_footprint.py (task 3.2).

Covers the pure array/vector logic specific to this task (depth
thresholding, the Sentinel-1 cross-check, validation) with small synthetic
rasters/GeoDataFrames. The sieve/vectorize/clip helpers are imported and
reused from task 3.1's module -- already covered by
tests/test_run_sentinel1_change_detection.py, not re-tested here.

Run with:
    pytest tests/test_extract_bhuvan_flood_footprint.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.extract_bhuvan_flood_footprint import (  # noqa: E402
    MIN_DEPTH_M,
    cross_check_against_sentinel1,
    threshold_depth_mask,
    validate_extraction,
)


class TestThresholdDepthMask:
    def test_above_threshold_flagged_below_not(self):
        depth = np.array([[0.0, 0.05, 0.1], [0.11, 1.0, 9.0]])
        mask = threshold_depth_mask(depth, min_depth_m=0.1)
        assert not mask[0, 0]  # 0.0
        assert not mask[0, 1]  # 0.05
        assert not mask[0, 2]  # exactly 0.1 -- not strictly greater
        assert mask[1, 0]      # 0.11
        assert mask[1, 1]
        assert mask[1, 2]

    def test_nan_never_flagged(self):
        depth = np.array([[np.nan, 5.0]])
        mask = threshold_depth_mask(depth, min_depth_m=0.1)
        assert not mask[0, 0]
        assert mask[0, 1]

    def test_default_matches_module_constant(self):
        depth = np.array([[MIN_DEPTH_M + 0.01]])
        assert threshold_depth_mask(depth)[0, 0]


class TestCrossCheckAgainstSentinel1:
    def test_unavailable_when_file_missing(self, tmp_path):
        bhuvan_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
        result = cross_check_against_sentinel1(bhuvan_gdf, sentinel1_path=tmp_path / "nope.geojson")
        assert result["available"] is False

    def test_unavailable_when_bhuvan_empty(self, tmp_path):
        s1_path = tmp_path / "s1.geojson"
        gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326").to_file(s1_path, driver="GeoJSON")
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        result = cross_check_against_sentinel1(empty, sentinel1_path=s1_path)
        assert result["available"] is False

    def test_computes_overlap_stats(self, tmp_path):
        s1_path = tmp_path / "s1.geojson"
        # Sentinel-1 polygon: a 2x2 degree box (roughly) -- half inside, half outside the bhuvan zone
        gpd.GeoDataFrame({"geometry": [box(80.20, 13.00, 80.22, 13.02)]}, crs="EPSG:4326").to_file(s1_path, driver="GeoJSON")
        bhuvan_gdf = gpd.GeoDataFrame({"geometry": [box(80.21, 13.00, 80.23, 13.02)]}, crs="EPSG:4326")  # overlaps right half

        result = cross_check_against_sentinel1(bhuvan_gdf, sentinel1_path=s1_path)
        assert result["available"] is True
        assert result["intersection_area_km2"] > 0
        assert result["intersection_area_km2"] < result["sentinel1_area_km2"]
        assert 0 < result["pct_of_sentinel1_area_inside_bhuvan_nrsc_zone"] < 100


class TestValidateExtraction:
    def test_empty_gdf(self):
        wards = gpd.GeoDataFrame({"Zone_Name": ["A"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        depth = np.array([[0.0, 5.0]])
        report = validate_extraction(empty, wards, depth)
        assert report["flood_extent"]["n_polygons"] == 0
        assert report["depth_histogram_m"]["n_valid_pixels"] == 2

    def test_area_and_per_ward_breakdown(self):
        wards = gpd.GeoDataFrame(
            {"Zone_Name": ["WardA", "WardB"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326"
        )
        bhuvan_gdf = gpd.GeoDataFrame({"geometry": [box(0.1, 0.1, 0.2, 0.2)]}, crs="EPSG:4326")  # entirely in WardA
        depth = np.array([[0.0, 5.0], [np.nan, 9.0]])
        report = validate_extraction(bhuvan_gdf, wards, depth)

        assert report["flood_extent"]["n_polygons"] == 1
        assert report["flood_extent"]["total_area_km2"] > 0
        assert "WardA" in report["per_ward_flooded_area_km2"]
        assert "WardB" not in report["per_ward_flooded_area_km2"]
        assert report["depth_histogram_m"]["n_valid_pixels"] == 3  # one NaN excluded
        assert report["depth_histogram_m"]["min_depth_m_used"] == MIN_DEPTH_M
