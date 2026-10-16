"""Fine-tune PubMedBERT as an entity-marked relation classifier.

Usage (after building candidates, see README):
    python -m training.train --data data/processed/candidates.jsonl --out weights/model2

Writes Hugging Face artifacts (config.json, model.safetensors, tokenizer files)
plus ``member2_metadata.json`` (status, base checkpoint, data hash, label
provenance, hyper-parameters, split report and evaluation summary).  The
metadata states explicitly whether evaluation labels were weak or gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from models.model2 import (BASE_CHECKPOINT, ENTITY_MARKERS, LABELS, LEXICON_VERSION,
                           METADATA_FILENAME, mark_entities)
from training.data import final_label, group_split, read_jsonl, split_report, trainable

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    base_checkpoint: str = BASE_CHECKPOINT
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    patience: int = 2
    seed: int = 42
    test_frac: float = 0.2
    dev_frac: float = 0.1
    class_weighting: bool = True
    min_examples: int = 50


def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def encode(records: list[dict], tokenizer, max_length: int):
    import torch

    texts, labels, keep = [], [], []
    marker_ids = {tokenizer.convert_tokens_to_ids(m) for m in ENTITY_MARKERS}
    for r in records:
        marked, _ = mark_entities(r["sentence"], (r["head_start"], r["head_end"]), (r["tail_start"], r["tail_end"]))
        texts.append(marked)
        labels.append(LABELS.index(final_label(r)))
    enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    for i in range(len(texts)):
        keep.append(len(marker_ids & set(enc["input_ids"][i].tolist())) == 4)
    idx = [i for i, k in enumerate(keep) if k]
    if len(idx) < len(texts):
        logger.warning("Dropped %d examples whose entity markers were truncated", len(texts) - len(idx))
    sel = torch.tensor(idx, dtype=torch.long)
    return {k: v[sel] for k, v in enc.items()}, torch.tensor(labels, dtype=torch.long)[sel]


def train(data_path: str, out_dir: str, cfg: TrainConfig, gold_test: Optional[str] = None) -> dict:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    from training.evaluate import evaluate_records

    set_seed(cfg.seed)
    records = trainable(read_jsonl(data_path))
    if len(records) < cfg.min_examples:
        raise SystemExit(f"Only {len(records)} trainable examples (< {cfg.min_examples}). "
                         "Collect/review more data instead of training on a toy set.")
    split = group_split(records, cfg.test_frac, cfg.dev_frac, cfg.seed)
    report = split_report(split)
    assert not report["group_overlap"], "chapter leakage between splits"
    logger.info("Split: %s", {k: v["examples"] for k, v in report.items() if isinstance(v, dict)})

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_checkpoint)
    tokenizer.add_special_tokens({"additional_special_tokens": list(ENTITY_MARKERS)})
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_checkpoint, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id={l: i for i, l in enumerate(LABELS)},
    )
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x_tr, y_tr = encode(split["train"], tokenizer, cfg.max_length)
    loader = DataLoader(TensorDataset(x_tr["input_ids"], x_tr["attention_mask"], y_tr),
                        batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    counts = np.bincount(y_tr.numpy(), minlength=len(LABELS)).astype(float)
    weights = None
    if cfg.class_weighting:
        w = np.where(counts > 0, counts.sum() / (len(LABELS) * np.maximum(counts, 1)), 0.0)
        weights = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(loader) * cfg.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best, bad_epochs, history = -1.0, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        dev_metrics = evaluate_records(model, tokenizer, split["dev"] or split["test"], cfg.max_length, device)
        history.append({"epoch": epoch + 1, "train_loss": total / max(1, len(loader)),
                        "dev_macro_f1_weak": dev_metrics["macro_f1"]})
        logger.info("epoch %d loss %.4f dev macro-F1 (weak labels) %.3f", epoch + 1, history[-1]["train_loss"],
                    dev_metrics["macro_f1"])
        if dev_metrics["macro_f1"] > best:
            best, bad_epochs = dev_metrics["macro_f1"], 0
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
    test_metrics = evaluate_records(best_model, tokenizer, split["test"], cfg.max_length, device,
                                    errors_path=out / "test_errors_weak.csv")
    gold_metrics = None
    if gold_test:
        gold = [r for r in read_jsonl(gold_test) if r.get("gold_relation")]
        gold_metrics = evaluate_records(best_model, tokenizer, gold, cfg.max_length, device,
                                        errors_path=out / "test_errors_gold.csv")
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()
    meta = {
        "status": "trained",
        "model": "PubMedBERT (Member 2)",
        "base_checkpoint": cfg.base_checkpoint,
        "labels": list(LABELS),
        "entity_markers": list(ENTITY_MARKERS),
        "lexicon_version": LEXICON_VERSION,
        "max_length": cfg.max_length,
        "train_config": asdict(cfg),
        "training_data": {"path": str(data_path), "sha256": data_hash,
                          "label_origins": report["train"]["origins"]},
        "split_report": report,
        "history": history,
        "evaluation": {
            "weak_label_test": {**test_metrics, "label_type": "WEAK (rule-generated) — agreement with rules, "
                                "not verified accuracy"},
            "gold_test": ({**gold_metrics, "label_type": "GOLD (manually reviewed)"} if gold_metrics else None),
        },
        "confidence_calibrated": False,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": "Charak Samhita New Edition (carakasamhitaonline.com), CC BY-NC-SA 4.0",
    }
    (out / METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="data/processed/candidates.jsonl")
    ap.add_argument("--out", default="weights/model2")
    ap.add_argument("--gold-test", default=None, help="JSONL with gold_relation for final evaluation")
    ap.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--lr", type=float, default=TrainConfig.lr)
    ap.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    meta = train(args.data, args.out, cfg, args.gold_test)
    print(json.dumps(meta["evaluation"], indent=2))


if __name__ == "__main__":
    main()
