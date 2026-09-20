"""
Unit tests for src/ground_truth/detect_flood_recession.py (task 4.4).

Covers the pure segment-flagging/set-difference/validation logic with
small synthetic geometries. The real GEE fetch (fetch_vv_db_image,
download_ee_image) and the SAR classification math (classify_flood_mask,
sieve_mask, vectorize_mask) are reused unchanged from task 3.1 and already
tested there (test_run_sentinel1_change_detection.py) -- not retested here.
This module's real output is validated by an actual run against live GEE
(see developing.md's task 4.4 notes), not mocked.

Run with:
    pytest tests/test_detect_flood_recession.py -v
"""
import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, box

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ground_truth.detect_flood_recession import (  # noqa: E402
    compute_recovered_segments,
    flag_segments_intersecting,
    validate_recession,
)


def _line_graph_nodes():
    # s1/s2 sit inside the peak polygon; s3 is far away and never flooded
    return gpd.GeoDataFrame(
        {
            "segment_id": ["s1", "s2", "s3"],
            "geometry": [
                LineString([(0.1, 0.1), (0.2, 0.2)]),
                LineString([(0.3, 0.3), (0.4, 0.4)]),
                LineString([(10, 10), (10.1, 10.1)]),
            ],
        },
        crs="EPSG:4326",
    )


def _write_extent(path: Path, polygons):
    gpd.GeoDataFrame({"geometry": polygons}, crs="EPSG:4326").to_file(path, driver="GeoJSON")


class TestFlagSegmentsIntersecting:
    def test_missing_file_returns_empty_set(self, tmp_path):
        assert flag_segments_intersecting(_line_graph_nodes(), tmp_path / "nope.geojson") == set()

    def test_intersecting_segments_flagged(self, tmp_path):
        p = tmp_path / "extent.geojson"
        _write_extent(p, [box(0, 0, 0.5, 0.5)])
        assert flag_segments_intersecting(_line_graph_nodes(), p) == {"s1", "s2"}


class TestComputeRecoveredSegments:
    def test_segment_still_flooded_at_18dec_is_not_recovered(self, tmp_path):
        peak_path = tmp_path / "peak.geojson"
        recession_path = tmp_path / "recession.geojson"
        _write_extent(peak_path, [box(0, 0, 0.5, 0.5)])       # s1, s2 flooded at peak
        _write_extent(recession_path, [box(0, 0, 0.5, 0.5)])  # still flooded at 18 Dec -> nobody recovered
        recovered = compute_recovered_segments(_line_graph_nodes(), peak_path, recession_path)
        assert recovered == set()

    def test_segment_no_longer_flooded_at_18dec_is_recovered(self, tmp_path):
        peak_path = tmp_path / "peak.geojson"
        recession_path = tmp_path / "recession.geojson"
        _write_extent(peak_path, [box(0, 0, 0.5, 0.5)])   # s1, s2 flooded at peak
        _write_extent(recession_path, [box(0.15, 0.15, 0.25, 0.25)])  # only s1 still shows a signature
        recovered = compute_recovered_segments(_line_graph_nodes(), peak_path, recession_path)
        assert recovered == {"s2"}

    def test_recovered_is_always_a_subset_of_peak_segments(self, tmp_path):
        # recession-check polygon covers a totally different area than peak
        # (e.g. speckle noise unrelated to the original flood) -- must never
        # introduce a "recovered" segment that wasn't flooded at peak at all
        peak_path = tmp_path / "peak.geojson"
        recession_path = tmp_path / "recession.geojson"
        _write_extent(peak_path, [box(0, 0, 0.5, 0.5)])
        _write_extent(recession_path, [box(9, 9, 9.5, 9.5)])  # near s3, not s1/s2
        recovered = compute_recovered_segments(_line_graph_nodes(), peak_path, recession_path)
        assert recovered == {"s1", "s2"}
        assert "s3" not in recovered  # s3 was never flooded at peak -- can't "recover"

    def test_no_peak_extent_gives_no_recovered_segments(self, tmp_path):
        recession_path = tmp_path / "recession.geojson"
        _write_extent(recession_path, [box(0, 0, 0.5, 0.5)])
        recovered = compute_recovered_segments(_line_graph_nodes(), tmp_path / "nope.geojson", recession_path)
        assert recovered == set()


class TestValidateRecession:
    def test_reports_counts_and_pct(self):
        report = validate_recession(
            peak_segments={"s1", "s2"}, recession_check_segments={"s1"}, recovered_segments={"s2"},
        )
        assert report["peak_sar_flagged_segments"] == 2
        assert report["still_flooded_at_18dec_segments"] == 1
        assert report["recovered_segments"] == 1
        assert report["pct_of_peak_sar_segments_recovered"] == 50.0

    def test_empty_peak_segments_gives_zero_pct_not_a_crash(self):
        report = validate_recession(peak_segments=set(), recession_check_segments=set(), recovered_segments=set())
        assert report["pct_of_peak_sar_segments_recovered"] == 0.0
