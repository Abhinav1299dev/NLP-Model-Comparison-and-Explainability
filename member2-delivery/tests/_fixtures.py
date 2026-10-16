"""Shared helpers for building tiny synthetic tokenizers (works on transformers 4.x and 5.x)."""

from __future__ import annotations

from pathlib import Path


def tiny_tokenizer(words: list[str], workdir: Path):
    from transformers import BertTokenizerFast

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [w for w in dict.fromkeys(words)]
    workdir.mkdir(parents=True, exist_ok=True)
    vocab_file = workdir / "vocab.txt"
    vocab_file.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vocab_file), do_lower_case=True)
    if len(tok) < len(vocab):  # transformers 5 ignores vocab_file for fast tokenizers
        tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
    return tok
