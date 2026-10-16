"""Build relation candidates (weak labels) from parsed Charaka verse units.

Usage:
    python -m training.build_dataset --units data/processed/charaka_units.jsonl

Outputs
    data/processed/candidates.jsonl          all candidates with label_origin
    data/annotation/review_sheet.csv         sheet for manual review (gold labels)
    data/processed/candidate_stats.json      label / origin / rule counts
If ``--reviewed`` is given, adjudicated gold labels are merged first.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.corpus_charaka import read_units
from training.data import export_review_sheet, merge_gold, read_review_sheet, write_jsonl
from training.weak_labels import build_candidates, to_records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--units", default="data/processed/charaka_units.jsonl")
    ap.add_argument("--out", default="data/processed/candidates.jsonl")
    ap.add_argument("--review-sheet", default="data/annotation/review_sheet.csv")
    ap.add_argument("--reviewed", default=None, help="adjudicated review CSV with gold_relation filled in")
    args = ap.parse_args()

    records = to_records(build_candidates(read_units(args.units)))
    if args.reviewed:
        records = merge_gold(records, read_review_sheet(args.reviewed))
    write_jsonl(records, args.out)
    n_sheet = export_review_sheet(records, args.review_sheet)
    stats = {
        "candidates": len(records),
        "label_origin": dict(Counter(r["label_origin"] for r in records)),
        "weak_relation": dict(Counter(str(r["weak_relation"]) for r in records)),
        "gold_relation": dict(Counter(r["gold_relation"] for r in records if r.get("gold_relation"))),
        "rules": dict(Counter(r["rule"] for r in records).most_common(25)),
        "chapters": len({r["group"] for r in records}),
        "review_sheet_rows": n_sheet,
    }
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("data/processed/candidate_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
