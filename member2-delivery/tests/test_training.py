"""Tests for the training pipeline (weak labels, parsing, splits, train/save/reload).

All sentences and HTML below are synthetic strings written for these tests; they
are not quotations of the corpus.  The end-to-end test trains a tiny randomly
initialised BERT for one epoch purely to exercise the save/reload path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import models.model2 as m2  # noqa: E402
from training.data import cohens_kappa, group_split, split_report, trainable  # noqa: E402
from training.weak_labels import candidates_from_text, weak_label  # noqa: E402


def _label(sentence: str, head: str, tail: str):
    hs = sentence.index(head)
    ts = sentence.index(tail)
    ents = {e.text: e for e in m2.find_entities(sentence)}
    pairs = {(h.text, t.text): (ht, tt) for h, t, ht, tt in m2.candidate_pairs(ents.values())}
    ht, tt = pairs[(head, tail)]
    return weak_label(sentence, (hs, hs + len(head)), (ts, ts + len(tail)), ht, tt)


@pytest.mark.parametrize("sentence,head,tail,expected", [
    ("Khadira alleviates kushtha when used regularly.", "Khadira", "kushtha", "TREATS"),
    ("Guduchi alleviates pitta.", "Guduchi", "pitta", "REDUCES"),
    ("Excess fish aggravates kapha.", "fish", "kapha", "AGGRAVATES"),
    ("Kapha is alleviated by pippali.", "pippali", "Kapha", "REDUCES"),
    ("Nimba is tikta in taste.", "Nimba", "tikta", "HAS_PROPERTY"),
])
def test_weak_positive_patterns(sentence, head, tail, expected):
    rel, origin, _rule = _label(sentence, head, tail)
    assert (rel, origin) == (expected, "weak_positive")


def test_negation_blocks_positive():
    rel, origin, _ = _label("Haridra does not alleviate kapha.", "Haridra", "kapha")
    assert origin != "weak_positive"


def test_trigger_not_bound_is_uncertain_not_negative():
    rel, origin, rule = _label("Haridra and vata were discussed; pitta alleviates nothing.", "Haridra", "vata")
    assert origin == "uncertain" and rel is None


def test_causal_sentence_is_out_of_schema():
    rel, origin, rule = _label("Excess pippali causes pitta disorders.", "pippali", "pitta")
    assert (rel, origin, rule) == (None, "uncertain", "causal_out_of_schema")


def test_negative_only_without_any_trigger():
    rel, origin, _ = _label("Haridra and kapha are named in the list.", "Haridra", "kapha")
    assert (rel, origin) == ("NO_RELATION", "weak_negative")


def test_candidates_keep_provenance():
    c = candidates_from_text("Guduchi alleviates pitta.", "N05:3", "ref-x", "Nidana:05")
    assert c and c[0].source_ref == "ref-x" and c[0].group == "Nidana:05"


def _records(n_groups=6, per_group=6):
    recs = []
    for g in range(n_groups):
        for i in range(per_group):
            lab = ["TREATS", "REDUCES", "NO_RELATION"][i % 3]
            recs.append({"group": f"G{g}", "weak_relation": lab, "gold_relation": None,
                         "label_origin": "weak_negative" if lab == "NO_RELATION" else "weak_positive"})
    return recs


def test_group_split_has_no_leakage():
    split = group_split(_records(), 0.2, 0.1, seed=1)
    rep = split_report(split)
    assert rep["group_overlap"] == []
    assert sum(v["examples"] for k, v in rep.items() if isinstance(v, dict)) == 36


def test_uncertain_records_not_trainable():
    recs = _records() + [{"group": "G9", "weak_relation": None, "gold_relation": None, "label_origin": "uncertain"}]
    assert all(r["label_origin"] != "uncertain" for r in trainable(recs))


def test_cohens_kappa():
    assert cohens_kappa(["A", "B", "A", "B"], ["A", "B", "A", "B"]) == 1.0
    assert cohens_kappa(["A", "A", "B", "B"], ["A", "B", "A", "B"]) == pytest.approx(0.0)


SYNTHETIC_HTML = """
<html><body><div class="mw-parser-output">
<p><b>Nidana Sthana Chapter 9. Synthetic test chapter</b></p>
<table><tr><th>DOI</th><td>10.0000/test.doi</td></tr></table>
<h2>Introduction</h2><p>Editorial introduction that must be ignored entirely [99]</p>
<h2>Sanskrit Text, Transliteration and English Translation</h2>
<h3>Test topic</h3>
<p>अथातः परीक्षाध्यायं व्याख्यास्यामः||१||</p>
<p>athātaḥ parīkṣādhyāyaṁ vyākhyāsyāmaḥ||1||</p>
<p>Now we shall describe the synthetic test chapter. [1]</p>
<p>Guduchi alleviates pitta in this invented sentence.</p>
<ul><li>First invented item about kapha.</li><li>Second invented item about vata [2-3]</li></ul>
<h2>Tattva Vimarsha</h2><p>Commentary text that must not be included in output [4]</p>
<a href="https://www.carakasamhitaonline.com/index.php?title=Test&amp;oldid=12345">perm</a>
</div></body></html>
"""


def test_charaka_parser_on_synthetic_html():
    pytest.importorskip("bs4")
    from training.corpus_charaka import parse_chapter

    units = parse_chapter(SYNTHETIC_HTML, "Test_Chapter")
    assert [u.verse_ref for u in units] == ["1", "2-3"]
    assert units[0].sthana == "Nidana" and units[0].chapter_no == 9
    assert units[0].doi == "10.0000/test.doi" and units[0].revision_id == "12345"
    assert "Guduchi alleviates pitta" in units[1].text and "Second invented item" in units[1].text
    assert all("||" not in u.text and "Commentary" not in u.text for u in units)
    assert "Nidana Sthana 9" in units[0].source_ref and "CC BY-NC-SA" in units[0].source_ref


def test_end_to_end_tiny_training_save_and_reload(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from transformers import BertConfig, BertForMaskedLM

    from tests._fixtures import tiny_tokenizer

    from training.data import write_jsonl
    from training.train import TrainConfig, train

    words = ("guduchi alleviates pitta khadira cures kushtha fish aggravates kapha haridra and vata "
             "were named here . nimba is tikta").split()
    base = tmp_path / "tiny_base"
    tok = tiny_tokenizer(sorted(set(words)), tmp_path / "vocab")
    BertForMaskedLM(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                               intermediate_size=64, max_position_embeddings=64)).save_pretrained(base)
    tok.save_pretrained(base)

    sentences = ["Guduchi alleviates pitta.", "Khadira cures kushtha.", "Fish aggravates kapha.",
                 "Haridra and vata were named here."]
    recs = []
    for g in range(8):
        for s in sentences:
            for c in candidates_from_text(s, f"u{g}", f"synthetic-{g}", f"G{g}"):
                recs.append(c.__dict__)
    data = tmp_path / "cands.jsonl"
    write_jsonl(recs, data)
    out = tmp_path / "model2"
    cfg = TrainConfig(base_checkpoint=str(base), epochs=1, batch_size=8, max_length=32, min_examples=10)
    meta = train(str(data), str(out), cfg)
    assert meta["status"] == "trained"
    assert "WEAK" in meta["evaluation"]["weak_label_test"]["label_type"]
    assert not meta["split_report"]["group_overlap"]

    monkeypatch.setenv("MODEL2_WEIGHTS_DIR", str(out))
    monkeypatch.delenv("MODEL2_ALLOW_UNTRAINED", raising=False)
    m2.reset()
    assert m2.is_available()
    result = m2.predict("Guduchi alleviates pitta.")
    assert isinstance(result, list)
    for item in result:
        assert tuple(item) == m2.OUTPUT_KEYS
    m2.reset()
