"""Evaluation and error analysis for Member 2.

``evaluate_records`` classifies fixed candidate pairs exactly as inference does
(entity markers + label constraint by entity types) and returns per-label
precision/recall/F1, macro-F1 over the labels present, the confusion matrix and
support.  The caller is responsible for stating whether the reference labels
are weak or gold; ``train.py`` records this in the metadata.

``rule_baseline`` scores the weak labeller itself against gold labels, which is
the minimum a learned model should beat on reviewed data.  (Against weak labels
the rule baseline is trivially perfect, so that comparison is meaningless and is
not reported.)

CLI:
    python -m training.evaluate --weights weights/model2 --data data/annotation/gold.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Optional

from models.model2 import LABELS, allowed_labels, mark_entities
from training.data import final_label, read_jsonl


def _metrics(y_true: list[str], y_pred: list[str]) -> dict:
    labels = [l for l in LABELS if l in set(y_true) | set(y_pred)]
    per = {}
    for l in labels:
        tp = sum(t == l and p == l for t, p in zip(y_true, y_pred))
        fp = sum(t != l and p == l for t, p in zip(y_true, y_pred))
        fn = sum(t == l and p != l for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[l] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                  "support": sum(t == l for t in y_true)}
    positive = [l for l in labels if l != "NO_RELATION"]
    macro = sum(per[l]["f1"] for l in labels) / len(labels) if labels else 0.0
    macro_pos = sum(per[l]["f1"] for l in positive) / len(positive) if positive else 0.0
    confusion = {t: {p: sum(a == t and b == p for a, b in zip(y_true, y_pred)) for p in labels} for t in labels}
    return {"n": len(y_true), "accuracy": round(sum(a == b for a, b in zip(y_true, y_pred)) / max(1, len(y_true)), 4),
            "macro_f1": round(macro, 4), "macro_f1_positive_labels": round(macro_pos, 4),
            "per_label": per, "confusion": confusion, "labels_evaluated": labels}


def classify(model, tokenizer, records: list[dict], max_length: int, device, batch_size: int = 32):
    """Constrained predictions ``[(label, confidence, probs)]`` for fixed candidate pairs."""
    import torch

    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    preds = []
    for i in range(0, len(records), batch_size):
        chunk = records[i:i + batch_size]
        texts = [mark_entities(r["sentence"], (r["head_start"], r["head_end"]), (r["tail_start"], r["tail_end"]))[0]
                 for r in chunk]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu()
        for r, p in zip(chunk, probs):
            allowed = allowed_labels(r["head_type"], r["tail_type"])
            best = max(allowed, key=lambda l: float(p[label2id[l]]))
            preds.append((best, float(p[label2id[best]]), {id2label[j]: float(p[j]) for j in range(len(p))}))
    return preds


def evaluate_records(model, tokenizer, records: list[dict], max_length: int, device,
                     errors_path: Optional[Path] = None) -> dict:
    if not records:
        return {"n": 0, "macro_f1": 0.0, "note": "empty evaluation set"}
    preds = classify(model, tokenizer, records, max_length, device)
    y_true = [final_label(r) for r in records]
    y_pred = [p[0] for p in preds]
    m = _metrics(y_true, y_pred)
    m["reference_label_origins"] = dict(Counter(r["label_origin"] for r in records))
    if errors_path:
        errors_path = Path(errors_path)
        errors_path.parent.mkdir(parents=True, exist_ok=True)
        with errors_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["source_ref", "sentence", "head", "tail", "reference", "reference_origin",
                        "predicted", "confidence", "rule"])
            for r, (lab, conf, _p) in zip(records, preds):
                if lab != final_label(r):
                    w.writerow([r.get("source_ref"), r["sentence"], r["head"], r["tail"], final_label(r),
                                r["label_origin"], lab, round(conf, 4), r.get("rule")])
    return m


def rule_baseline(gold_records: list[dict]) -> dict:
    """Weak-labeller performance on gold data (uncertain -> NO_RELATION)."""
    y_true = [r["gold_relation"] for r in gold_records]
    y_pred = [r.get("weak_relation") or "NO_RELATION" for r in gold_records]
    return _metrics(y_true, y_pred)


def main() -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default="weights/model2")
    ap.add_argument("--data", required=True, help="JSONL candidates; gold_relation used when present")
    ap.add_argument("--errors", default="reports/errors.csv")
    args = ap.parse_args()
    recs = read_jsonl(args.data)
    gold = [r for r in recs if r.get("gold_relation")]
    use = gold or [r for r in recs if r["label_origin"] in {"weak_positive", "weak_negative"}]
    tok = AutoTokenizer.from_pretrained(args.weights)
    model = AutoModelForSequenceClassification.from_pretrained(args.weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    meta = json.loads((Path(args.weights) / "member2_metadata.json").read_text())
    res = {"label_type": "GOLD" if gold else "WEAK (rule-generated; not verified accuracy)",
           "model": evaluate_records(model, tok, use, meta.get("max_length", 256), device, Path(args.errors))}
    if gold:
        res["rule_baseline"] = rule_baseline(gold)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
