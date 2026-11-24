"""
Task 7.1 (data step) -- fetch the real Census 2011 inputs for the ward-level
vulnerability index, and extract the rows the study area actually needs into
small, committed CSVs.

*** THE STUDY WARDS DO NOT EXIST IN CENSUS 2011 -- FOUND, NOT ASSUMED ***
The project's 16 study wards (data/raw/wards/study_wards.geojson) use Greater
Chennai Corporation's 200-ward numbering, created by the October 2011
expansion. Census 2011 was enumerated in Feb-Mar 2011, on the OLD 155-ward
Chennai (M Corp.) -- confirmed directly from the District Census Handbook
(Part B) appendix, which lists exactly WARD No.-0001 .. WARD No.-0155. So no
study ward can be looked up by number; build_vulnerability_index.py
crosswalks old -> new wards spatially instead, using DataMeet's
digitised pre-2011 ward boundaries (Wards-2008.geojson).

Two study wards (168, 169) were not inside the 2011 corporation at all.
Despite their zone label ("PERUNGUDI"), the gazetteer's own localities
place them in Ullagaram (168) and Puzhuthivakkam (169) -- in 2011 the
separate Puzhithivakkam (Ullagaram) Municipality, Kancheepuram district,
Census town code 803357 (NOT Perungudi TP, 803358, which the zone name
would suggest). Public records confirm that municipality was annexed as
exactly GCC wards 168 + 169. Its PCA row therefore comes from the
Kancheepuram district file, not Chennai's.

*** SANITATION EXISTS ONLY AT TOWN LEVEL -- DISCLOSED, NOT INTERPOLATED ***
working.md 3.2 names "sanitation access" as a proxy. The Census publishes
latrine/sewer data in the Houselisting tables (HH-08), and the official
download only goes down to TOWN level: one value for all of Chennai (M Corp.)
and one for Puzhithivakkam (M). It is kept (the contrast is large and real:
~93% vs ~10% of households with a piped-sewer latrine) but it only
distinguishes two groups of wards. build_vulnerability_index.py reports
the index both with and without it.

Usage:
    python src/optimizer/fetch_census.py

Outputs (committed -- small, and Phase 7 needs them):
    data/raw/census/pca_wards_2011.csv          (155 old Chennai wards + Puzhithivakkam town)
    data/raw/census/sanitation_towns_2011.csv   (HH-08, the 2 towns the study area spans)
    data/raw/census/chennai_wards_2008.geojson  (old ward boundaries, for the crosswalk)
    data/raw/census/census_sources.json         (URLs + sha256 of every download)
Not committed (regenerate by rerunning this script):
    data/raw/census/source/*.xlsx, *.xls        (full district-level source files)
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
CENSUS_DIR = REPO_ROOT / "data" / "raw" / "census"
SOURCE_DIR = CENSUS_DIR / "source"

# Census of India NADA catalog downloads (verified live 24 Nov 2026).
SOURCES = {
    "pca_chennai": {
        "url": "https://censusindia.gov.in/nada/index.php/catalog/6794/download/9871",
        "file": "DDW_PCA3302_2011_chennai.xlsx",
        "description": "PCA TV: Primary census abstract at town, village and ward level, Tamil Nadu - District Chennai - 2011",
    },
    "pca_kancheepuram": {
        "url": "https://censusindia.gov.in/nada/index.php/catalog/6795/download/9872",
        "file": "DDW_PCA3304_2011_kancheepuram.xlsx",
        "description": "PCA TV: Primary census abstract at town, village and ward level, Tamil Nadu - District Kancheepuram - 2011",
    },
    "hh08_latrine": {
        "url": "https://censusindia.gov.in/nada/index.php/catalog/8673/download/11785",
        "file": "HH08_latrine_tamil_nadu_2011.xls",
        "description": "HL-08: Households by availability of type of latrine facility, Tamil Nadu - 2011 (town level)",
    },
    "wards_2008": {
        "url": "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Chennai/Wards-2008.geojson",
        "file": "Wards-2008.geojson",
        "description": "DataMeet / Transparent Chennai: pre-2011 (155-ward) Chennai Corporation ward boundaries, CC BY-SA 2.5 IN",
    },
}

CHENNAI_TOWN_CODE = 803339       # Chennai (M Corp.)
PUZHITHIVAKKAM_TOWN_CODE = 803357  # Puzhithivakkam (Ullagaram) (M) -> GCC wards 168 + 169
N_OLD_CHENNAI_WARDS = 155

# PCA columns the index uses (see build_vulnerability_index.py for how).
PCA_COLUMNS = [
    "No_HH", "TOT_P", "TOT_M", "TOT_F", "P_06", "P_SC", "P_ST", "P_LIT", "P_ILL",
    "TOT_WORK_P", "MAINWORK_P", "MARGWORK_P", "NON_WORK_P",
]


class CensusFetchError(RuntimeError):
    pass


def download(url, dest, timeout=300):
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=timeout)
    if resp.status_code != 200 or not resp.content:
        raise CensusFetchError(f"Download failed ({resp.status_code}): {url}")
    dest.write_bytes(resp.content)
    return hashlib.sha256(resp.content).hexdigest()


def extract_pca_rows(chennai_pca, kancheepuram_pca):
    """Return one row per census unit the crosswalk needs.

    `unit_id` is "chennai_old_ward_<n>" for the 155 old Chennai wards and
    "puzhithivakkam_town" for the municipality that became wards 168 + 169.
    Only `TRU == "Urban"` rows are kept (PCA repeats Total/Urban for towns).
    """
    wards = chennai_pca[
        (chennai_pca["Level"] == "WARD")
        & (chennai_pca["Town/Village"] == CHENNAI_TOWN_CODE)
        & (chennai_pca["TRU"] == "Urban")
    ].copy()
    if len(wards) != N_OLD_CHENNAI_WARDS:
        raise CensusFetchError(f"Expected {N_OLD_CHENNAI_WARDS} old Chennai wards, found {len(wards)}")
    wards["unit_id"] = "chennai_old_ward_" + wards["Ward"].astype(int).astype(str)
    wards["old_ward_no"] = wards["Ward"].astype(int)

    town = kancheepuram_pca[
        (kancheepuram_pca["Level"] == "TOWN")
        & (kancheepuram_pca["Town/Village"] == PUZHITHIVAKKAM_TOWN_CODE)
        & (kancheepuram_pca["TRU"] == "Urban")
    ].copy()
    if len(town) != 1:
        raise CensusFetchError(f"Expected exactly 1 Puzhithivakkam town row, found {len(town)}")
    town["unit_id"] = "puzhithivakkam_town"
    town["old_ward_no"] = pd.NA

    out = pd.concat([wards, town], ignore_index=True)
    out = out[["unit_id", "old_ward_no", "Town/Village", "Name"] + PCA_COLUMNS]
    return out.rename(columns={"Town/Village": "census_town_code", "Name": "census_name"})


def extract_sanitation_rows(hh08_raw):
    """Pull HH-08's town rows for the two towns the study area spans.

    HH-08 has a multi-row merged header, so columns are taken by position
    (verified against the sheet's own 1-indexed column-number row; pandas
    index = sheet number - 1: 4 = town code, 5 = area name, 6 = T/R/U,
    7 = total households, 8 = latrine within premises, 9 = flush latrine
    connected to piped sewer).
    """
    rows = []
    for code, label in [(CHENNAI_TOWN_CODE, "chennai_m_corp"), (PUZHITHIVAKKAM_TOWN_CODE, "puzhithivakkam_town")]:
        match = hh08_raw[(hh08_raw[4].astype(str).str.strip() == str(code)) & (hh08_raw[6] == "Urban")]
        if len(match) != 1:
            raise CensusFetchError(f"Expected 1 HH-08 row for town {code}, found {len(match)}")
        r = match.iloc[0]
        total, in_premises, piped_sewer = int(r[7]), int(r[8]), int(r[9])
        rows.append({
            "town_unit": label,
            "census_town_code": code,
            "census_name": str(r[5]).strip(),
            "total_households": total,
            "households_latrine_in_premises": in_premises,
            "households_piped_sewer_latrine": piped_sewer,
            "share_no_latrine_in_premises": round(1 - in_premises / total, 4),
            "share_no_piped_sewer": round(1 - piped_sewer / total, 4),
        })
    return pd.DataFrame(rows)


def main():
    manifest = {}
    for key, src in SOURCES.items():
        dest = (CENSUS_DIR if key == "wards_2008" else SOURCE_DIR) / src["file"]
        print(f"Downloading {key} ...")
        sha = download(src["url"], dest)
        manifest[key] = {**src, "sha256": sha, "bytes": dest.stat().st_size}

    pca = extract_pca_rows(
        pd.read_excel(SOURCE_DIR / SOURCES["pca_chennai"]["file"]),
        pd.read_excel(SOURCE_DIR / SOURCES["pca_kancheepuram"]["file"]),
    )
    pca.to_csv(CENSUS_DIR / "pca_wards_2011.csv", index=False)

    sanitation = extract_sanitation_rows(pd.read_excel(SOURCE_DIR / SOURCES["hh08_latrine"]["file"], header=None))
    sanitation.to_csv(CENSUS_DIR / "sanitation_towns_2011.csv", index=False)

    (CENSUS_DIR / "Wards-2008.geojson").rename(CENSUS_DIR / "chennai_wards_2008.geojson")
    manifest["wards_2008"]["file"] = "chennai_wards_2008.geojson"
    (CENSUS_DIR / "census_sources.json").write_text(json.dumps(manifest, indent=2))

    print(f"PCA units: {len(pca)} (155 old wards + Puzhithivakkam town); population {int(pca.TOT_P.sum()):,}")
    print(sanitation.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
