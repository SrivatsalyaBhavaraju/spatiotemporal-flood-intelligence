"""
Task 1.13 -- collect real, publicly accessible text about the 2015 Chennai
flood event, scoped to the study wards.

Original plan assumed live social-media collection would be possible. It
isn't, and this was verified rather than assumed (developing.md Phase 1
notes have the detail): X/Twitter's historical full-archive search is
Enterprise-only now (~$42k/month, no research tier since the 2023 Academic
API shutdown), and no redistributable public dataset of 2015 Chennai tweets
exists -- academic papers that analyzed some used a since-dead scraping
tool and never published their data.

**Chosen alternative (real text, not synthetic):** mine real, publicly
accessible reporting about the event -- UN/ReliefWeb situation reports,
news coverage, an academic retrospective -- filtered down to passages that
actually mention a place in our gazetteer (task 1.14). This is REAL text
about the real event, but it comes from professional reporting, not
informal social-media posts -- a different, more formal register than what
Objective 2 ultimately targets. That gap is disclosed here and in
src/nlp/README.md, not hidden. Task 2.11 (hand-labeling) works with
whatever register this corpus actually has.

ReliefWeb's own API now requires a pre-approved appname (as of Nov 2025,
verified live) -- rather than wait on that approval, this script fetches
the same report pages directly (they're public web pages, not gated).

Usage:
    python src/nlp/collect_distress_text.py

Output:
    data/raw/distress_text/corpus_draft.csv (gitignored -- rerun to regenerate)
    columns: source_url, source_label, paragraph_text, matched_locations
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
GAZETTEER = REPO_ROOT / "data" / "raw" / "gazetteer" / "gazetteer_draft.csv"
OUT_DIR = REPO_ROOT / "data" / "raw" / "distress_text"
OUT_CSV = OUT_DIR / "corpus_draft.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) research-project-scraper"}

# Verified accessible (checked live, 01 Sep 2026) real sources about the Nov-Dec 2015
# Chennai flood event. Not an exhaustive news trawl -- a proportionate, disclosed
# starting set; task 2.9/2.11 can grow this once the corpus is actually put to use.
SOURCES = [
    ("https://reliefweb.int/report/india/chennai-floods-situation-report-no-1-chennai-flood-2-4-december-2015",
     "reliefweb_sitrep"),
    ("https://reliefweb.int/report/india/chennai-floods-situation-report-no1-december-2-2015",
     "reliefweb_sitrep"),
    ("https://www.worldweatherattribution.org/chennai-floods-december-2015/",
     "academic_retrospective"),
    ("https://www.environmentandsociety.org/arcadia/rains-floods-case-chennai-2015",
     "academic_retrospective"),
    ("https://www.thenewsminute.com/tamil-nadu/chennai-floods-velachery-residents-forced-to-evacuate-say-little-has-changed-since-2015",
     "news"),
    ("https://www.dtnext.in/news/city/flood-of-memories",
     "news"),
    ("https://en.wikipedia.org/wiki/2015_South_India_floods",
     "encyclopedia"),
]

MIN_NAME_LEN = 4  # skip gazetteer names too short/generic to match reliably (e.g. "Adyar" ok, "GST" would be risky)
MIN_PARAGRAPH_WORDS = 8  # skip nav/caption fragments

# Single-word gazetteer names that are also ordinary English words -- matching these
# by themselves is almost always a false positive ("Temple"/"Church"/"HEALTH" as a
# landmark's literal OSM name vs. the word appearing generically in flood-report
# prose). Multi-word names ("Advent Christian Church") aren't affected -- those are
# specific enough to keep.
GENERIC_SINGLE_WORD_STOP = {
    "temple", "church", "mosque", "health", "school", "hospital", "college",
    "centre", "center", "clinic", "kovil", "mandapam",
}


def load_gazetteer_names() -> list[str]:
    gaz = pd.read_csv(GAZETTEER)
    names = set(gaz["name"].dropna().astype(str).str.strip())
    names = {n for n in names if len(n) >= MIN_NAME_LEN}
    names = {n for n in names if not (" " not in n and n.lower() in GENERIC_SINGLE_WORD_STOP)}
    return sorted(names)


def fetch_paragraphs(url: str) -> list[str]:
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    return [p for p in paragraphs if len(p.split()) >= MIN_PARAGRAPH_WORDS]


def find_matches(paragraph: str, names: list[str]) -> list[str]:
    low = paragraph.lower()
    return [n for n in names if re.search(r"\b" + re.escape(n.lower()) + r"\b", low)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = load_gazetteer_names()
    print(f"Matching against {len(names)} gazetteer names (roads/localities/landmarks/waterways).\n")

    rows = []
    for url, label in SOURCES:
        print(f"Fetching [{label}] {url}")
        try:
            paragraphs = fetch_paragraphs(url)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        kept = 0
        for p in paragraphs:
            matches = find_matches(p, names)
            if matches:
                rows.append({
                    "source_url": url,
                    "source_label": label,
                    "paragraph_text": p,
                    "matched_locations": "; ".join(matches),
                })
                kept += 1
        print(f"  {len(paragraphs)} paragraphs -> {kept} mention a gazetteer place name")
        time.sleep(1)  # be polite between requests

    corpus = pd.DataFrame(rows).drop_duplicates(subset=["paragraph_text"])
    corpus.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(corpus)} location-mentioning passages -> {OUT_CSV}")
    print("\nBy source:")
    print(corpus["source_label"].value_counts().to_string())
    print("\nThis is a DRAFT real-text corpus (task 1.13), news/report register, not social")
    print("media. Task 2.11 hand-labels a subset of it (distress / not-distress).")


if __name__ == "__main__":
    main()
