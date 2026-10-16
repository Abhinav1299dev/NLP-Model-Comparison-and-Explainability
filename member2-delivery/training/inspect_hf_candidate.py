"""Verify whether a Hugging Face dataset is usable as Member 2's Charaka corpus.

Default target: ``shinigamiRaj/IndianVedasOriginal`` (the corpus named in the
team brief).  The script does not trust keyword matches in the text body: a
record only counts as Charaka material if a *metadata* field (title / source /
book / file name / chapter) identifies Charaka Samhita.  It also measures the
script of the text, because PubMedBERT is an English model.

Decision rule (all must hold to be "suitable"):
  * >= MIN_RECORDS records whose metadata identifies Charaka Samhita;
  * >= 80 % of those records are predominantly Latin-script English
    (Devanagari share < 5 % of letters and English stop-word rate >= 8 %);
  * a usable chapter / sthana / verse reference exists in metadata;
  * a licence is declared on the dataset card.

Usage:
    python -m training.inspect_hf_candidate --dataset shinigamiRaj/IndianVedasOriginal
Writes data/processed/hf_candidate_report.json.  Requires internet access and
the ``datasets`` package.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

MIN_RECORDS = 200
DEVANAGARI = re.compile(r"[\u0900-\u097F]")
CHARAKA = re.compile(r"\bc(h)?araka?\s*-?\s*sa[mṃ]hit[aā]\b|\bcharak\b", re.I)
SECTION = re.compile(r"\b(nidana|nidāna|chikitsa|cikitsā|cikitsa)\b", re.I)
META_KEYS = re.compile(r"title|source|book|file|name|chapter|sthana|section|text_name|work", re.I)
STOPWORDS = {"the", "and", "of", "is", "in", "to", "with", "by", "for", "which", "are", "be", "it"}


def _script_profile(text: str) -> dict[str, float]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return {"devanagari": 0.0, "latin": 0.0, "stopword_rate": 0.0}
    dev = sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters)
    latin = sum(c.isascii() for c in letters) / len(letters)
    words = re.findall(r"[A-Za-z]+", text.lower())
    sw = sum(w in STOPWORDS for w in words) / max(1, len(words))
    return {"devanagari": dev, "latin": latin, "stopword_rate": sw}


def inspect(dataset_id: str, max_rows: int = 50000) -> dict[str, Any]:
    from datasets import get_dataset_config_names, load_dataset
    from huggingface_hub import HfApi

    report: dict[str, Any] = {"dataset": dataset_id}
    card = HfApi().dataset_info(dataset_id)
    report["license"] = getattr(card.card_data, "license", None) if card.card_data else None
    report["configs"] = get_dataset_config_names(dataset_id)
    per_config = {}
    for cfg in report["configs"]:
        ds = load_dataset(dataset_id, cfg)
        for split, data in ds.items():
            cols = data.column_names
            meta_cols = [c for c in cols if META_KEYS.search(c)]
            text_cols = [c for c in cols if c not in meta_cols]
            n = min(len(data), max_rows)
            charaka_meta, charaka_textonly, section_meta = 0, 0, 0
            english, profiles = 0, Counter()
            samples = []
            for row in data.select(range(n)):
                meta = " ".join(str(row[c]) for c in meta_cols if row.get(c) is not None)
                body = " ".join(str(row[c]) for c in text_cols if isinstance(row.get(c), str))
                if CHARAKA.search(meta):
                    charaka_meta += 1
                    section_meta += bool(SECTION.search(meta))
                    prof = _script_profile(body)
                    is_en = prof["devanagari"] < 0.05 and prof["stopword_rate"] >= 0.08
                    english += is_en
                    profiles["english" if is_en else "non_english"] += 1
                    if len(samples) < 3:
                        samples.append({"meta": meta[:300], "text": body[:300]})
                elif CHARAKA.search(body):
                    charaka_textonly += 1
            per_config[f"{cfg}/{split}"] = {
                "rows": len(data), "inspected": n, "columns": cols, "metadata_columns": meta_cols,
                "charaka_by_metadata": charaka_meta, "charaka_mentioned_in_text_only": charaka_textonly,
                "charaka_with_section_metadata": section_meta, "charaka_english_records": english,
                "samples": samples,
            }
    report["splits"] = per_config
    total = sum(v["charaka_by_metadata"] for v in per_config.values())
    eng = sum(v["charaka_english_records"] for v in per_config.values())
    sect = sum(v["charaka_with_section_metadata"] for v in per_config.values())
    reasons = []
    if total < MIN_RECORDS:
        reasons.append(f"only {total} records identified as Charaka Samhita by metadata (< {MIN_RECORDS})")
    if total and eng / total < 0.8:
        reasons.append(f"only {eng}/{total} Charaka records are English (PubMedBERT is an English model)")
    if total and sect == 0:
        reasons.append("no sthana/chapter reference in metadata; verse-level provenance impossible")
    if not report["license"]:
        reasons.append("no licence declared on the dataset card")
    report["suitable"] = not reasons
    report["reasons"] = reasons
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", default="shinigamiRaj/IndianVedasOriginal")
    ap.add_argument("--out", default="data/processed/hf_candidate_report.json")
    ap.add_argument("--max-rows", type=int, default=50000)
    args = ap.parse_args()
    report = inspect(args.dataset, args.max_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dataset", "license", "suitable", "reasons")}, indent=2))


if __name__ == "__main__":
    main()
