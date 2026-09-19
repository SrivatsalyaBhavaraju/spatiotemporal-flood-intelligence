"""
Unit tests for src/ground_truth/run_sentinel1_change_detection.py (task 3.1).

Covers the pure array/vector logic (classification, sieving, vectorizing,
clipping, validation, NRSC cross-check) with small synthetic rasters/
GeoDataFrames -- no GEE credentials needed. The GEE fetch/download
functions themselves are validated by an actual live run against real data
(see developing.md's task 3.1 notes), matching this repo's convention of
not mocking external services.

Run with:
    pytest tests/test_run_sentinel1_change_detection.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.run_sentinel1_change_detection import (  # noqa: E402
    CHANGE_STD_MULTIPLIER,
    JRC_OCCURRENCE_THRESHOLD,
    Phase3ArtifactError,
    classify_flood_mask,
    clip_to_wards,
    cross_check_against_nrsc_simulation,
    load_wards_union,
    sieve_mask,
    validate_flood_extent,
    vectorize_mask,
)

GRID = 20
TRANSFORM = from_origin(80.20, 13.00, 0.0001, 0.0001)  # arbitrary small pixel size


def _make_db_arrays():
    """pre/post dB arrays with a clear scenario baked in against a small
    background of checkerboard noise (+/-0.2dB, so mean/std reflect a
    realistic non-degenerate "quiet" AOI rather than a single flat value):
      - rows 2:6, cols 2:6      -> new flood: -10dB drop, far beyond mean-k*std, low JRC
      - rows 10:13, cols 10:13  -> permanent water: same -10dB drop, but high JRC -> excluded
      - (15,15) single isolated pixel -> -10dB drop but too small an area (sieve target)
    """
    pre = np.zeros((GRID, GRID))
    post = np.zeros((GRID, GRID))
    jrc = np.zeros((GRID, GRID))

    i, j = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    post += 0.2 * ((i + j) % 2 * 2 - 1)  # background checkerboard noise, +/-0.2dB

    post[2:6, 2:6] = pre[2:6, 2:6] - 10.0        # new flood: big drop

    post[10:13, 10:13] = pre[10:13, 10:13] - 10.0  # same big drop...
    jrc[10:13, 10:13] = 90.0                        # ...but permanent water per JRC

    post[15, 15] = pre[15, 15] - 10.0            # isolated single-pixel "flood"

    return pre, post, jrc


class TestClassifyFloodMask:
    def test_new_flood_detected_permanent_water_excluded_and_shapes_checked(self):
        pre, post, jrc = _make_db_arrays()
        mask = classify_flood_mask(pre, post, jrc)

        assert mask[2:6, 2:6].all()          # new flood block flagged
        assert not mask[10:13, 10:13].any()  # permanent water excluded
        assert mask[15, 15]                  # isolated pixel still flagged here (sieve handles it separately)
        assert not mask[0, 0]                # untouched dry pixel not flagged

    def test_shape_mismatch_raises(self):
        pre, post, jrc = _make_db_arrays()
        with pytest.raises(Phase3ArtifactError, match="shapes must match"):
            classify_flood_mask(pre, post, jrc[:-1, :])

    def test_nan_pixels_never_flagged(self):
        pre, post, jrc = _make_db_arrays()
        post = post.copy()
        post[2, 2] = np.nan
        mask = classify_flood_mask(pre, post, jrc)
        assert not mask[2, 2]
        assert mask[3, 3]  # rest of the block still flagged

    def test_thresholds_are_module_defaults(self):
        assert CHANGE_STD_MULTIPLIER == 2.0
        assert JRC_OCCURRENCE_THRESHOLD == 50


class TestSieveMask:
    def test_isolated_pixel_removed_block_kept(self):
        pre, post, jrc = _make_db_arrays()
        mask = classify_flood_mask(pre, post, jrc)
        sieved = sieve_mask(mask, min_pixels=8)
        assert not sieved[15, 15]      # isolated pixel gone
        assert sieved[2:6, 2:6].all()  # 16-pixel block survives


class TestVectorizeMask:
    def test_empty_mask_returns_empty_gdf(self):
        gdf = vectorize_mask(np.zeros((GRID, GRID), dtype=bool), TRANSFORM, "EPSG:4326")
        assert len(gdf) == 0
        assert gdf.crs == "EPSG:4326"

    def test_single_block_vectorizes_to_one_polygon_with_correct_area(self):
        mask = np.zeros((GRID, GRID), dtype=bool)
        mask[2:6, 2:6] = True
        gdf = vectorize_mask(mask, TRANSFORM, "EPSG:4326")
        assert len(gdf) == 1
        # 4x4 pixels at 0.0001 deg each -> 0.0004 x 0.0004 deg polygon
        assert gdf.geometry.iloc[0].bounds == pytest.approx((80.2002, 12.9994, 80.2006, 12.9998), abs=1e-9)


class TestClipToWards:
    def test_clips_to_union_geometry(self):
        flood_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 10, 10)]}, crs="EPSG:4326")
        wards_union = box(5, 5, 15, 15)
        clipped = clip_to_wards(flood_gdf, wards_union)
        assert len(clipped) == 1
        assert clipped.geometry.iloc[0].bounds == pytest.approx((5, 5, 10, 10))

    def test_empty_input_returns_empty(self):
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        result = clip_to_wards(empty, box(0, 0, 1, 1))
        assert len(result) == 0

    def test_non_intersecting_polygon_dropped(self):
        flood_gdf = gpd.GeoDataFrame({"geometry": [box(100, 100, 101, 101)]}, crs="EPSG:4326")
        wards_union = box(0, 0, 1, 1)
        clipped = clip_to_wards(flood_gdf, wards_union)
        assert len(clipped) == 0


class TestLoadWardsUnion:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Phase3ArtifactError, match="task 1.3"):
            load_wards_union(tmp_path / "does_not_exist.geojson")

    def test_returns_gdf_and_union(self, tmp_path):
        p = tmp_path / "wards.geojson"
        gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326").to_file(p, driver="GeoJSON")
        wards, union = load_wards_union(p)
        assert len(wards) == 2
        assert union.bounds == pytest.approx((0, 0, 2, 1))


class TestValidateFloodExtent:
    def test_empty_flood_gdf(self):
        wards = gpd.GeoDataFrame({"Zone_Name": ["A"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        report = validate_flood_extent(empty, wards, np.array([[0.0]]), np.array([[0.0]]))
        assert report["flood_extent"]["n_polygons"] == 0

    def test_area_and_per_ward_breakdown(self):
        # ~111km x 111km at the equator per degree -- use a tiny known box for a sanity check on ordering/keys
        wards = gpd.GeoDataFrame(
            {"Zone_Name": ["WardA", "WardB"], "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)]}, crs="EPSG:4326"
        )
        flood_gdf = gpd.GeoDataFrame({"geometry": [box(0.1, 0.1, 0.2, 0.2)]}, crs="EPSG:4326")  # entirely in WardA
        pre_db = np.array([[0.0, -20.0], [-20.0, 0.0]])
        post_db = np.array([[-20.0, -20.0], [0.0, 0.0]])
        report = validate_flood_extent(flood_gdf, wards, pre_db, post_db)

        assert report["flood_extent"]["n_polygons"] == 1
        assert report["flood_extent"]["total_area_km2"] > 0
        assert "WardA" in report["per_ward_flooded_area_km2"]
        assert "WardB" not in report["per_ward_flooded_area_km2"]
        assert report["change_detection_diagnostics"]["threshold_used"]["method"] == f"mean - {CHANGE_STD_MULTIPLIER}*std"


class TestNrscCrossCheck:
    def test_unavailable_when_file_missing(self, tmp_path):
        flood_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
        result = cross_check_against_nrsc_simulation(flood_gdf, nrsc_path=tmp_path / "nope.tif")
        assert result["available"] is False

    def test_unavailable_when_no_flood_polygons(self, tmp_path):
        empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
        # even a real file path shouldn't matter here since flood_gdf is empty
        result = cross_check_against_nrsc_simulation(empty, nrsc_path=tmp_path / "nope.tif")
        assert result["available"] is False

    def test_samples_depth_at_flood_centroids(self, tmp_path):
        # a depth raster that's high (5m) in the top-left quadrant, 0 elsewhere
        depth = np.zeros((20, 20), dtype="float32")
        depth[:10, :10] = 5.0
        transform = from_origin(0, 20, 1, 1)  # 1 unit/pixel, origin (0,20)
        path = tmp_path / "nrsc.tif"
        with rasterio.open(path, "w", driver="GTiff", height=20, width=20, count=1, dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan) as dst:
            dst.write(depth, 1)

        # flood polygon centered in the high-depth quadrant (row/col ~ (5,5) -> x=5, y=15)
        flood_gdf = gpd.GeoDataFrame({"geometry": [box(4, 14, 6, 16)]}, crs="EPSG:4326")
        result = cross_check_against_nrsc_simulation(flood_gdf, nrsc_path=path, n_control=50)

        assert result["available"] is True
        assert result["mean_nrsc_depth_m_at_flood_centroids"] == pytest.approx(5.0)
