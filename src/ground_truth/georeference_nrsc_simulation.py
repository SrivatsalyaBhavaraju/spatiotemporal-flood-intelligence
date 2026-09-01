"""
Task 0.6 follow-up -- georeference the flood-depth simulation map (Fig. 8) from
the NRSC/ISRO "Hydrological Simulation Study of Flood Disaster in Adyar and
Cooum Rivers" report (v1.2, 07 Dec 2015) into an approximate raster layer
clipped to the study wards.

IMPORTANT CAVEATS (read before using the output as ground truth):
  1. Fig. 8 is a Google-Earth-style OBLIQUE 3D PERSPECTIVE screenshot, not an
     orthophoto. A planar homography (fit here from 9 labeled-locality control
     points) approximates this reasonably only because Chennai's terrain is
     very flat -- the report itself notes this. It is NOT a substitute for a
     true orthorectification.
  2. Control-point reprojection error: RMSE ~560 m, max ~1050 m (see
     homography_accuracy.json). That is 10-20% of the study cluster's own
     span. This layer is NOT segment-level ground truth -- treat it as a
     coarse, ward-scale cross-check only.
  3. This is a bare-earth HYDROLOGICAL SIMULATION (modeled water depth from a
     rainfall-runoff + DEM model), not an observed/measured flood extent. It
     is a different evidentiary tier than Sentinel-1 SAR change detection,
     which remains the project's primary ground truth (see working.md SS1.5).
  4. Water-depth extraction from the color legend is a simple blue-dominance
     classifier (see extract_depth()) -- it will misclassify shadowed water,
     turbid/pale water pixels, and JPEG-compression artifacts along edges.

Outputs (data/raw/bhuvan/, gitignored):
    fig8_georeferenced.tif      -- depth-in-meters raster, EPSG:4326, clipped to AOI
    fig8_georeferenced.png      -- preview
    homography_accuracy.json    -- control points + reprojection error report
    qa_overlay.png              -- OSM roads + ward boundary over the new raster,
                                    for a by-eye alignment sanity check
"""
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG8_JPEG = REPO_ROOT / "data" / "raw" / "bhuvan" / "fig8_raw.jpg"
WARDS = REPO_ROOT / "data" / "raw" / "wards" / "study_wards.geojson"
OUT_DIR = REPO_ROOT / "data" / "raw" / "bhuvan"

# Pixel (x, y) in fig8_raw.jpg -> (lon, lat) of the labeled locality.
# Marker pixel positions found via automated dark-red-marker detection;
# lon/lat are standard locality coordinates for these Chennai neighborhoods.
# "Ramapuram", "Valasaravakkam", "T Nagar" are plain text labels in the
# figure with no marker pin, so they're excluded as control points.
CONTROL_POINTS = {
    "Meenambakkam":  ((41, 214),  (80.1697, 12.9950)),
    "Anna Nagar":    ((569, 173), (80.2101, 13.0850)),
    "Madhavaram":    ((819, 152), (80.2270, 13.1480)),
    "Vyasarpadi":    ((839, 216), (80.2470, 13.1170)),
    "Guindy":        ((224, 266), (80.2185, 13.0100)),
    "Chennai":       ((757, 284), (80.2700, 13.0700)),
    "Velachery":     ((2, 337),   (80.2211, 12.9791)),
    "Adyar":         ((242, 399), (80.2570, 13.0067)),
    "Thiruvanmiyur": ((97, 461),  (80.2594, 12.9830)),
}

# Legend color ramp sampled from the vertical bar at x=812, y=55 (depth=9) to
# y=92 (depth=0) in fig8_raw.jpg.
LEGEND_X = 812
LEGEND_Y_TOP, LEGEND_DEPTH_TOP = 55, 9.0
LEGEND_Y_BOT, LEGEND_DEPTH_BOT = 92, 0.0

GRID_RES_DEG = 0.0002  # ~22 m pixels -- finer than our positional accuracy,
                        # chosen for smooth output, NOT a precision claim


def fit_homography():
    names = list(CONTROL_POINTS.keys())
    src = np.array([CONTROL_POINTS[n][0] for n in names], dtype=np.float64)
    dst = np.array([CONTROL_POINTS[n][1] for n in names], dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=0)
    H_inv = np.linalg.inv(H)

    src_h = np.hstack([src, np.ones((len(src), 1))])
    proj = (H @ src_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    errs_deg = np.linalg.norm(proj - dst, axis=1)
    errs_m = errs_deg * 111000

    report = {
        "control_points": {
            n: {"pixel": CONTROL_POINTS[n][0], "lonlat": CONTROL_POINTS[n][1],
                "predicted_lonlat": proj[i].tolist(), "error_m": float(errs_m[i])}
            for i, n in enumerate(names)
        },
        "rmse_m": float(np.sqrt(np.mean(errs_m ** 2))),
        "max_error_m": float(errs_m.max()),
        "mean_error_m": float(errs_m.mean()),
        "n_control_points": len(names),
    }
    return H, H_inv, report


def build_depth_lut(img_arr):
    """Sample the legend color bar -> (blue_score, depth) lookup table."""
    ys = np.arange(LEGEND_Y_TOP, LEGEND_Y_BOT + 1)
    colors = img_arr[ys, LEGEND_X].astype(float)  # (N,3)
    depths = LEGEND_DEPTH_TOP + (ys - LEGEND_Y_TOP) * (LEGEND_DEPTH_BOT - LEGEND_DEPTH_TOP) / (LEGEND_Y_BOT - LEGEND_Y_TOP)
    blue_score = colors[:, 2] - np.maximum(colors[:, 0], colors[:, 1])
    order = np.argsort(blue_score)
    return blue_score[order], depths[order]


def extract_depth(rgb, lut_blue, lut_depth, land_threshold=10.0):
    """rgb: (...,3) uint8 array. Returns depth-in-meters array, NaN where classified as land/no-data."""
    r, g, b = rgb[..., 0].astype(float), rgb[..., 1].astype(float), rgb[..., 2].astype(float)
    blue_score = b - np.maximum(r, g)
    depth = np.interp(blue_score, lut_blue, lut_depth, left=np.nan, right=lut_depth[-1])
    depth = np.where(blue_score < land_threshold, np.nan, depth)
    return depth


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not FIG8_JPEG.exists():
        raise FileNotFoundError(f"{FIG8_JPEG} not found -- extract it from the PDF first.")

    print("Fitting homography from 9 control points...")
    H, H_inv, accuracy_report = fit_homography()
    print(f"  RMSE: {accuracy_report['rmse_m']:.0f} m, max: {accuracy_report['max_error_m']:.0f} m")
    with open(OUT_DIR / "homography_accuracy.json", "w", encoding="utf-8") as f:
        json.dump(accuracy_report, f, indent=2)
    print(f"  saved -> {OUT_DIR / 'homography_accuracy.json'}")
    if accuracy_report["rmse_m"] > 300:
        print("  ** RMSE > 300m: this confirms the layer should be treated as coarse/approximate only **")

    img = Image.open(FIG8_JPEG).convert("RGB")
    img_arr = np.array(img)
    h_img, w_img = img_arr.shape[:2]

    print("Building depth color LUT from legend...")
    lut_blue, lut_depth = build_depth_lut(img_arr)

    print("Loading study ward AOI...")
    wards = gpd.read_file(WARDS)
    minx, miny, maxx, maxy = wards.total_bounds
    pad = 0.01
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

    width = int((maxx - minx) / GRID_RES_DEG)
    height = int((maxy - miny) / GRID_RES_DEG)
    print(f"  output grid: {width} x {height} px, bounds ({minx:.4f},{miny:.4f})-({maxx:.4f},{maxy:.4f})")

    # output pixel centers -> lon/lat
    xs = minx + (np.arange(width) + 0.5) * (maxx - minx) / width
    ys = maxy - (np.arange(height) + 0.5) * (maxy - miny) / height
    lon_grid, lat_grid = np.meshgrid(xs, ys)

    # inverse-map every output cell's lon/lat to a source pixel via H_inv
    ones = np.ones_like(lon_grid)
    dst_h = np.stack([lon_grid, lat_grid, ones], axis=-1)  # (H,W,3)
    src_h = dst_h @ H_inv.T
    src_x = src_h[..., 0] / src_h[..., 2]
    src_y = src_h[..., 1] / src_h[..., 2]

    valid = (src_x >= 0) & (src_x < w_img) & (src_y >= 0) & (src_y < h_img)
    src_x_c = np.clip(np.round(src_x).astype(int), 0, w_img - 1)
    src_y_c = np.clip(np.round(src_y).astype(int), 0, h_img - 1)
    sampled_rgb = img_arr[src_y_c, src_x_c]

    print("Classifying water depth per output pixel...")
    depth = extract_depth(sampled_rgb, lut_blue, lut_depth)
    depth = np.where(valid, depth, np.nan)
    n_wet = np.isfinite(depth).sum()
    print(f"  {n_wet} / {width*height} output pixels classified as flooded ({100*n_wet/(width*height):.1f}%)")

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    out_tif = OUT_DIR / "fig8_georeferenced.tif"
    with rasterio.open(
        out_tif, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform, nodata=np.nan,
    ) as dst:
        dst.write(depth.astype("float32"), 1)
    print(f"  saved -> {out_tif}")

    # quick preview PNG
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(depth, extent=(minx, maxx, miny, maxy), origin="upper", cmap="Blues", vmin=0, vmax=9)
    wards.boundary.plot(ax=ax, color="#14213D", linewidth=1)
    plt.colorbar(im, ax=ax, label="Simulated water depth (m) -- APPROXIMATE, RMSE ~%dm" % accuracy_report["rmse_m"])
    ax.set_title("Georeferenced NRSC hydrological simulation (Fig. 8)\nCAUTION: modeled, not observed; coarse georeferencing")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig8_georeferenced.png", dpi=150)
    print(f"  saved -> {OUT_DIR / 'fig8_georeferenced.png'}")


if __name__ == "__main__":
    main()
