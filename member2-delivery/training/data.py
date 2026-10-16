"""Dataset assembly: review sheets, gold merge, leakage-safe splits, agreement.

Annotation workflow
    1. ``export_review_sheet`` writes every candidate (weak_positive,
       weak_negative and uncertain) to CSV with empty ``gold_relation``,
       ``reviewer`` and ``review_note`` columns.
    2. Two reviewers label a sample independently (``gold_relation`` must be one
       of the five team labels).  ``cohens_kappa`` reports agreement; conflicts
       are adjudicated into a single file.
    3. ``merge_gold`` overlays adjudicated labels on the candidates
       (``label_origin`` becomes ``gold``).
    4. The final evaluation uses gold-labelled chapters only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

from models.model2 import LABELS

REVIEW_COLUMNS = [
    "candidate_id", "group", "unit_id", "source_ref", "sentence", "head", "head_type", "tail",
    "tail_type", "weak_relation", "label_origin", "rule", "gold_relation", "reviewer", "review_note",
    "head_start", "head_end", "tail_start", "tail_end",
]
TRAINABLE_ORIGINS = {"weak_positive", "weak_negative", "gold"}


def candidate_id(rec: dict) -> str:
    key = f"{rec['unit_id']}|{rec['sentence']}|{rec['head_start']}|{rec['tail_start']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def export_review_sheet(records: Iterable[dict], path: Path | str, seed: int = 42) -> int:
    rows = []
    for r in records:
        row = {k: r.get(k) for k in REVIEW_COLUMNS}
        row["candidate_id"] = candidate_id(r)
        rows.append(row)
    random.Random(seed).shuffle(rows)  # avoid order bias while reviewing
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def read_review_sheet(path: Path | str) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        g = (r.get("gold_relation") or "").strip().upper()
        if g and g not in LABELS:
            raise ValueError(f"Invalid gold_relation {g!r} for candidate {r['candidate_id']}")
        r["gold_relation"] = g or None
    return rows


def merge_gold(records: list[dict], reviewed: Sequence[dict]) -> list[dict]:
    gold = {r["candidate_id"]: r for r in reviewed if r.get("gold_relation")}
    out = []
    for rec in records:
        rec = dict(rec)
        g = gold.get(candidate_id(rec))
        if g:
            rec["gold_relation"] = g["gold_relation"]
            rec["reviewer"] = g.get("reviewer")
            rec["review_note"] = g.get("review_note")
            rec["label_origin"] = "gold"
        out.append(rec)
    return out


def final_label(rec: dict) -> Optional[str]:
    return rec.get("gold_relation") or rec.get("weak_relation")


def trainable(records: Iterable[dict]) -> list[dict]:
    return [r for r in records if r["label_origin"] in TRAINABLE_ORIGINS and final_label(r) in LABELS]


def group_split(records: list[dict], test_frac: float = 0.2, dev_frac: float = 0.1,
                seed: int = 42) -> dict[str, list[dict]]:
    """Split by chapter group so no chapter appears in two splits.

    Groups are shuffled with ``seed`` and greedily assigned to test, then dev,
    until the target fraction of examples is reached.  If every positive label
    cannot be represented in test, the caller is warned through
    ``split_report``.
    """
    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_group[r["group"]].append(r)
    groups = sorted(by_group)
    random.Random(seed).shuffle(groups)
    n = len(records)
    split = {"train": [], "dev": [], "test": []}
    for g in groups:
        if len(split["test"]) < test_frac * n:
            split["test"].extend(by_group[g])
        elif len(split["dev"]) < dev_frac * n:
            split["dev"].extend(by_group[g])
        else:
            split["train"].extend(by_group[g])
    return split


def split_report(split: dict[str, list[dict]]) -> dict:
    rep = {}
    for name, recs in split.items():
        rep[name] = {
            "examples": len(recs),
            "groups": sorted({r["group"] for r in recs}),
            "labels": dict(Counter(final_label(r) for r in recs)),
            "origins": dict(Counter(r["label_origin"] for r in recs)),
        }
    overlap = set(rep["train"]["groups"]) & (set(rep["test"]["groups"]) | set(rep["dev"]["groups"]))
    rep["group_overlap"] = sorted(overlap)
    missing = [l for l in LABELS if l not in rep["test"]["labels"]]
    rep["labels_missing_from_test"] = missing
    return rep


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Need two equally long, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def write_jsonl(records: Iterable[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]
