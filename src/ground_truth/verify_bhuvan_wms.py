"""
Task 0.6, round 2 -- direct protocol-level verification of the Chennai-2015
flood layers advertised by the Bhuvan historical-flood archive UI.

Context: the team found this archive at the URL below (not surfaced by the
general disaster.php portal checked in the first 0.6 pass) --
    https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/flood/flood.php
It's a real year/state-organized tree of flood layers, and it DOES list
Chennai-2015-specific entries: a RISAT-1 cumulative inundation layer, a
Cartosat-2 post-event layer, and several Tamil-Nadu date-stamped extents in
Nov-Dec 2015. That corrects the earlier "no 2015 historical archive" claim.

BUT the UI's checkboxes only prove the front-end still references these
layer names -- not that the backing WMS data still exists. This script tests
that directly: it traces each checkbox's onclick handler back to the actual
WMS endpoint (extracted from the portal's own loadlayer()/loadmap() JS, see
src/ground_truth/README.md for how) and issues a real GetMap request.

Run:
    python src/ground_truth/verify_bhuvan_wms.py

Output: data/raw/bhuvan/wms_verification.json
"""
import json
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "raw" / "bhuvan"

# AOI: bounding box of the study wards, with margin
BBOX = "80.15,12.90,80.35,13.10"

# Layers referenced by checkboxes on the flood.php Chennai-2015 archive page,
# and the real WMS endpoint each one's onclick handler resolves to (traced
# through loadfloodmap()/loadlayer() -> loadmap() in
# disaster/lib/uncomp/disaster09122022_v2.js).
CHECKS = [
    {
        "label": "TN state flood extent, 05/12/2015",
        "wms_url": "https://bhuvan-gp1.nrsc.gov.in/bhuvan/wms",
        "layer": "flood:tn_2015_05_12",
        "via": "loadfloodmap() -> direct WMS URL in the onclick handler",
    },
    {
        "label": "TN state flood extent, 14/11/2015-18Hr",
        "wms_url": "https://bhuvan-gp1.nrsc.gov.in/bhuvan/wms",
        "layer": "flood:tn_2015_14_11_18",
        "via": "loadfloodmap() -> direct WMS URL in the onclick handler",
    },
    {
        "label": "Chennai RISAT-1 Cumulative Inundation (3,4,6 Dec 2015) -- the layer 0.6 originally asked about",
        "wms_url": "https://bhuvan-ras2.nrsc.gov.in/mapcache",
        "layer": "ch_exp_0306dec15",
        "via": "loadlayer(type='tilecache2') -> loadmap() -> urlArray1[0] from disaster09122022_v2.js",
    },
    {
        "label": "Chennai Post-Event Cartosat-2, 04/12/2015",
        "wms_url": "https://bhuvan-ras2.nrsc.gov.in/mapcache",
        "layer": "ch_c2_sat",
        "via": "loadlayer(type='tilecache2') -> loadmap() -> urlArray1[0] from disaster09122022_v2.js",
    },
    {
        "label": "post_risat (generic layer name found in this server's GetCapabilities -- worth checking in case it now carries the Chennai data under a reused name)",
        "wms_url": "https://bhuvan-ras2.nrsc.gov.in/mapcache",
        "layer": "post_risat",
        "via": "found by grepping this server's own GetCapabilities for risat-like names",
    },
]


def test_layer(wms_url, layer_name):
    params = {
        "service": "WMS", "version": "1.1.1", "request": "GetMap",
        "layers": layer_name, "bbox": BBOX, "width": 400, "height": 400,
        "srs": "EPSG:4326", "format": "image/png", "transparent": "true",
    }
    try:
        r = requests.get(wms_url, params=params, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    except requests.RequestException as e:
        return {"ok": False, "reason": f"request failed: {e}"}

    ctype = r.headers.get("Content-Type", "")
    if "xml" in ctype or b"ServiceException" in r.content[:2000]:
        # WMS error response -- layer doesn't exist on this server
        return {"ok": False, "reason": "WMS ServiceException", "status": r.status_code,
                "body_snippet": r.content[:300].decode("utf-8", errors="replace")}
    if "image" not in ctype:
        return {"ok": False, "reason": f"unexpected content-type: {ctype}", "status": r.status_code}

    # A valid PNG that's tiny and near-blank is the server's "no data here" placeholder,
    # not real coverage -- flag it as suspect rather than a clean pass/fail.
    n_bytes = len(r.content)
    suspect_blank = n_bytes < 8000
    return {"ok": True, "status": r.status_code, "content_type": ctype,
            "response_bytes": n_bytes, "suspect_blank_or_placeholder": suspect_blank}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"Testing {len(CHECKS)} layers against AOI bbox={BBOX} ...\n")
    for check in CHECKS:
        print(f"- {check['label']}")
        print(f"  layer={check['layer']!r} on {check['wms_url']}")
        res = test_layer(check["wms_url"], check["layer"])
        verdict = "DEAD (WMS error)" if not res["ok"] else (
            "RETURNED, but tiny/likely-blank -- treat as unconfirmed" if res.get("suspect_blank_or_placeholder")
            else "RETURNED real-looking imagery"
        )
        print(f"  -> {verdict}")
        results.append({**check, "result": res, "verdict": verdict})
        print()

    out = {"aoi_bbox": BBOX, "checks": results}
    with open(OUT_DIR / "wms_verification.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {OUT_DIR / 'wms_verification.json'}")


if __name__ == "__main__":
    main()
