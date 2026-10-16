"""Corpus loader: Charak Samhita New Edition (carakasamhitaonline.com).

Source
    Charak Samhita Research, Training and Skill Development Centre,
    "Charak Samhita New Edition" — Sanskrit text, transliteration and English
    translation, chapter by chapter, with per-chapter DOIs.
    Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
    (stated on every page footer).  Use is therefore limited to
    non-commercial work, with attribution, and derived datasets must be shared
    under the same licence.  Raw pages are cached locally and must NOT be
    committed to a public repository without the attribution/licence notice.

What is extracted
    Only the English translation paragraphs inside the section
    "Sanskrit Text, Transliteration and English Translation" of each chapter of
    Nidana Sthana and Chikitsa Sthana.  Devanagari lines and transliteration
    lines (which end in ``||n||``) are skipped.  Editorial commentary sections
    ("Tattva Vimarsha", "Vidhi Vimarsha", ...) are excluded because they are
    modern commentary, not translation of the Charaka text.

Every English unit is attached to the verse reference printed at its end
(e.g. ``[3]`` or ``[12-15]``) and keeps chapter title, sthana, chapter number,
URL, page revision id, DOI and licence.

Chapter pages are discovered from the sthana index pages and kept only if the
page itself states "<Sthana> Sthana Chapter <n>"; nothing is assumed from page
titles alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import unquote, urlparse, parse_qs

logger = logging.getLogger(__name__)

BASE_URL = "https://www.carakasamhitaonline.com/index.php"
LICENSE = "CC BY-NC-SA 4.0"
CORPUS_NAME = "Charak Samhita New Edition (carakasamhitaonline.com)"
SECTION_INDEX_PAGES = {"Nidana": "Nidana_Sthana", "Chikitsa": "Chikitsa_Sthana"}
TRANSLATION_HEADING = re.compile(r"sanskrit text.*english translation", re.I)
CHAPTER_HEADER = re.compile(r"\b(Nidana|Chikitsa)\s+Sthana\s+Chapter\s+(\d+)\b", re.I)
VERSE_REF_END = re.compile(r"\[(\d+(?:\s*[-–]\s*\d+)?(?:\s*,\s*\d+)*)\]\s*$")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
TRANSLIT_VERSE = re.compile(r"\|\|\s*\d+")
USER_AGENT = "member2-ayurveda-re/1.0 (academic NLP coursework; contact: team lead)"


@dataclass
class VerseUnit:
    """One English translation unit with provenance."""

    unit_id: str
    sthana: str
    chapter_no: int
    chapter_title: str
    topic: Optional[str]
    verse_ref: str
    text: str
    url: str
    revision_id: Optional[str]
    doi: Optional[str]
    license: str = LICENSE
    corpus: str = CORPUS_NAME

    @property
    def source_ref(self) -> str:
        ref = f"Charak Samhita New Edition, {self.sthana} Sthana {self.chapter_no} ({self.chapter_title}), verse {self.verse_ref}"
        if self.doi:
            ref += f", DOI {self.doi}"
        return ref + f", {self.license}"


# ---------------------------------------------------------------------------
# HTTP with cache
# ---------------------------------------------------------------------------

def _fetch(title: str, cache_dir: Path, delay: float = 1.0, timeout: int = 30) -> str:
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", title)
    path = cache_dir / f"{safe}.html"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    resp = requests.get(BASE_URL, params={"title": title}, timeout=timeout,
                        headers={"User-Agent": USER_AGENT})
    resp.raise_for_status()
    path.write_text(resp.text, encoding="utf-8")
    time.sleep(delay)  # be polite to a volunteer-run site
    return resp.text


def _content_root(html: str):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one("div.mw-parser-output") or soup
    return soup, root


def _revision_id(soup) -> Optional[str]:
    for a in soup.find_all("a", href=True):
        q = parse_qs(urlparse(a["href"]).query)
        if "oldid" in q:
            return q["oldid"][0]
    m = re.search(r'"wgRevisionId"\s*:\s*(\d+)', str(soup))
    return m.group(1) if m else None


def _infobox(root) -> dict[str, str]:
    info: dict[str, str] = {}
    for row in root.find_all("tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) >= 2:
            info[cells[0].get_text(" ", strip=True).lower()] = cells[1].get_text(" ", strip=True)
    return info


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_english(text: str) -> bool:
    if not text or DEVANAGARI.search(text) or TRANSLIT_VERSE.search(text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    ascii_ratio = sum(c.isascii() for c in letters) / len(letters)
    return ascii_ratio > 0.97 and len(text.split()) >= 3


def parse_chapter(html: str, title: str) -> list[VerseUnit]:
    """Extract English verse units from one chapter page."""
    soup, root = _content_root(html)
    full_text = root.get_text(" ", strip=True)
    header = CHAPTER_HEADER.search(full_text)
    if not header:
        return []
    sthana, chapter_no = header.group(1).capitalize(), int(header.group(2))
    info = _infobox(root)
    doi = info.get("doi")
    rev = _revision_id(soup)
    url = f"{BASE_URL}?title={title}" + (f"&oldid={rev}" if rev else "")
    chapter_title = title.replace("_", " ")

    units: list[VerseUnit] = []
    in_section = False
    topic: Optional[str] = None
    buffer: list[str] = []
    for el in root.find_all(["h2", "h3", "p", "li"]):
        if el.name == "h2":
            heading = el.get_text(" ", strip=True)
            if in_section and not TRANSLATION_HEADING.search(heading):
                break
            in_section = bool(TRANSLATION_HEADING.search(heading))
            continue
        if not in_section:
            continue
        if el.name == "h3":
            topic = el.get_text(" ", strip=True)
            continue
        if el.name == "p" and el.find_parent("li") is not None:
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        if not _is_english(text):
            continue
        buffer.append(text)
        m = VERSE_REF_END.search(text)
        if m:
            ref = re.sub(r"\s+", "", m.group(1)).replace("–", "-")
            body = " ".join(buffer)
            body = VERSE_REF_END.sub("", body).strip()
            uid = f"{sthana[:1]}{chapter_no:02d}:{ref}"
            units.append(VerseUnit(uid, sthana, chapter_no, chapter_title, topic, ref, body, url, rev, doi))
            buffer = []
    if buffer:
        logger.info("%s: %d trailing English paragraph(s) without verse reference dropped", title, len(buffer))
    return units


def discover_chapters(sthana: str, cache_dir: Path) -> list[str]:
    """Titles of chapter pages linked from the sthana index page."""
    html = _fetch(SECTION_INDEX_PAGES[sthana], cache_dir)
    _soup, root = _content_root(html)
    titles: list[str] = []
    for a in root.find_all("a", href=True):
        href = a["href"]
        q = parse_qs(urlparse(href).query)
        t = q.get("title", [None])[0]
        if t is None and "/index.php/" in href:
            t = unquote(href.split("/index.php/")[1].split("#")[0])
        if not t or ":" in t or t in titles or t in SECTION_INDEX_PAGES.values():
            continue
        titles.append(t)
    return titles


def load_corpus(
    sthanas: Iterable[str] = ("Nidana", "Chikitsa"), cache_dir: Path | str = "data/raw/charaka",
    max_pages: Optional[int] = None,
) -> list[VerseUnit]:
    """Download (or read from cache), verify and parse the requested sthanas."""
    cache = Path(cache_dir)
    units: list[VerseUnit] = []
    for sthana in sthanas:
        titles = discover_chapters(sthana, cache)
        kept = 0
        for title in titles[: max_pages or None]:
            try:
                html = _fetch(title, cache)
            except Exception as exc:  # network errors are logged, not hidden
                logger.warning("Failed to fetch %s: %s", title, exc)
                continue
            chapter_units = parse_chapter(html, title)
            if chapter_units and chapter_units[0].sthana == sthana:
                units.extend(chapter_units)
                kept += 1
        logger.info("%s Sthana: %d chapter pages verified, %d candidate links", sthana, kept, len(titles))
    return units


def save_units(units: list[VerseUnit], path: Path | str) -> dict:
    """Write JSONL and return a manifest (counts + SHA-256) for provenance."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for u in units:
            fh.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    chapters = sorted({(u.sthana, u.chapter_no, u.chapter_title) for u in units})
    manifest = {
        "corpus": CORPUS_NAME, "license": LICENSE, "file": str(path), "sha256": digest,
        "units": len(units), "chapters": [{"sthana": s, "chapter_no": n, "title": t} for s, n, t in chapters],
        "retrieved": time.strftime("%Y-%m-%d"),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_units(path: Path | str) -> list[VerseUnit]:
    with Path(path).open(encoding="utf-8") as fh:
        return [VerseUnit(**json.loads(line)) for line in fh if line.strip()]


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/processed/charaka_units.jsonl")
    ap.add_argument("--cache", default="data/raw/charaka")
    ap.add_argument("--sthana", nargs="+", default=["Nidana", "Chikitsa"])
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    units = load_corpus(args.sthana, args.cache, args.max_pages)
    if not units:
        raise SystemExit("No verse units parsed. Inspect cached HTML in --cache; the site layout may have changed.")
    manifest = save_units(units, args.out)
    print(json.dumps({k: v for k, v in manifest.items() if k != "chapters"}, indent=2))
    print(f"chapters: {len(manifest['chapters'])}")


if __name__ == "__main__":
    main()
