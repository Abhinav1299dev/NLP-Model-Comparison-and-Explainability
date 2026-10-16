"""Tests for models/model2.py.

These tests validate the integration contract WITHOUT a trained model.  Where a
model is needed, a tiny randomly initialised BERT ("synthetic fixture") is
written to a temporary directory.  Its predictions are meaningless by design;
the classifier bias is set so that TREATS/REDUCES dominate, which makes the
output non-empty and therefore lets the schema be checked.  The fixture is
marked ``status="test_fixture"`` and can only be loaded when
``MODEL2_ALLOW_UNTRAINED=1`` is set, which production code never does.

Run:  pytest -q tests/
Optional real-model check:  MODEL2_TEST_REAL_WEIGHTS=1 pytest -q tests/
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import models.model2 as m2  # noqa: E402
from tests._fixtures import tiny_tokenizer  # noqa: E402

SAMPLE = "Haridra is useful in Kushta and reduces Kapha."
ALLOWED = {"TREATS", "REDUCES", "AGGRAVATES", "HAS_PROPERTY"}

# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

_FIXTURE_WORDS = (
    "haridra is useful in kushta and reduces kapha . khadira cures skin disease "
    "maricha aggravates pitta honey vata the of , turmeric ##s"
).split()


def _build_fixture(path: Path, status: str = "test_fixture", labels=None) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification

    torch.manual_seed(0)
    tok = tiny_tokenizer(_FIXTURE_WORDS, path.parent / (path.name + "_vocab"))
    tok.add_special_tokens({"additional_special_tokens": list(m2.ENTITY_MARKERS)})
    labels = list(labels or m2.LABELS)
    cfg = BertConfig(
        vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, num_labels=len(labels),
        id2label={i: l for i, l in enumerate(labels)}, label2id={l: i for i, l in enumerate(labels)},
    )
    model = BertForSequenceClassification(cfg)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        if "TREATS" in labels:
            model.classifier.bias[labels.index("TREATS")] = 5.0
        if "REDUCES" in labels:
            model.classifier.bias[labels.index("REDUCES")] = 4.0
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / m2.METADATA_FILENAME).write_text(json.dumps({
        "status": status, "base_checkpoint": "synthetic-tiny-bert", "max_length": 64,
        "note": "Synthetic test fixture. Not a trained model. Predictions are meaningless.",
    }))
    return path


@pytest.fixture()
def fixture_weights(tmp_path, monkeypatch):
    path = _build_fixture(tmp_path / "model2")
    monkeypatch.setenv("MODEL2_WEIGHTS_DIR", str(path))
    monkeypatch.setenv("MODEL2_ALLOW_UNTRAINED", "1")
    monkeypatch.setenv("MODEL2_DEVICE", "cpu")
    m2.reset()
    yield path
    m2.reset()


@pytest.fixture()
def missing_weights(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL2_WEIGHTS_DIR", str(tmp_path / "does_not_exist"))
    monkeypatch.delenv("MODEL2_ALLOW_UNTRAINED", raising=False)
    m2.reset()
    yield
    m2.reset()


# ---------------------------------------------------------------------------
# Import and input handling
# ---------------------------------------------------------------------------

def test_import_is_lightweight_and_exposes_predict():
    mod = importlib.reload(m2)
    assert callable(mod.predict)
    assert mod._STATE == {}, "importing must not load or train a model"
    assert list(mod.LABELS) == ["NO_RELATION", "TREATS", "REDUCES", "AGGRAVATES", "HAS_PROPERTY"]


@pytest.mark.parametrize("value", ["", "   ", "\n\t", None])
def test_empty_input_returns_empty_list_without_loading(value, missing_weights):
    assert m2.predict(value) == []
    assert m2._STATE == {}


@pytest.mark.parametrize("value", [123, 4.5, ["text"], {"text": "x"}])
def test_non_string_input_raises_type_error(value):
    with pytest.raises(TypeError):
        m2.predict(value)


def test_missing_artifacts_raise_clear_error(missing_weights):
    assert m2.is_available() is False
    with pytest.raises(m2.Model2NotAvailableError) as exc:
        m2.predict(SAMPLE)
    assert "member2_metadata.json" in str(exc.value)


def test_untrained_fixture_refused_without_explicit_opt_in(tmp_path, monkeypatch):
    path = _build_fixture(tmp_path / "model2")
    monkeypatch.setenv("MODEL2_WEIGHTS_DIR", str(path))
    monkeypatch.delenv("MODEL2_ALLOW_UNTRAINED", raising=False)
    m2.reset()
    assert m2.is_available() is False
    with pytest.raises(m2.Model2NotAvailableError):
        m2.predict(SAMPLE)
    m2.reset()


def test_wrong_label_schema_is_rejected(tmp_path, monkeypatch):
    path = _build_fixture(tmp_path / "model2", labels=["NO_RELATION", "TREATS", "CAUSES"])
    monkeypatch.setenv("MODEL2_WEIGHTS_DIR", str(path))
    monkeypatch.setenv("MODEL2_ALLOW_UNTRAINED", "1")
    m2.reset()
    with pytest.raises(m2.Model2NotAvailableError):
        m2.predict(SAMPLE)
    m2.reset()


def test_text_without_entities_returns_empty_list(fixture_weights):
    assert m2.predict("This sentence mentions no lexicon terms at all.") == []


# ---------------------------------------------------------------------------
# Output schema (synthetic fixture)
# ---------------------------------------------------------------------------

def test_normal_input_schema(fixture_weights):
    out = m2.predict(SAMPLE)
    assert isinstance(out, list) and out, "fixture is biased to produce positive labels"
    for item in out:
        assert isinstance(item, dict)
        assert tuple(item.keys()) == m2.OUTPUT_KEYS
        assert item["relation"] in ALLOWED
        assert isinstance(item["confidence"], float)
        assert 0.0 <= item["confidence"] <= 1.0 and not math.isnan(item["confidence"])
        assert item["evidence"] and item["evidence"] in SAMPLE
        assert item["head"] in item["evidence"] and item["tail"] in item["evidence"]
        assert item["model"] == "PubMedBERT (Member 2)"
        assert isinstance(item["source"], str) and item["source"]
        assert isinstance(item["explanation_cues"], list) and item["explanation_cues"]
        for cue in item["explanation_cues"]:
            assert set(cue) == {"token", "score"}
            assert cue["token"] in item["evidence"]
            assert 0.0 <= cue["score"] <= 1.0
        assert not any(c["token"].startswith("[") for c in item["explanation_cues"])
    json.dumps(out)  # must be JSON serialisable for Gradio


def test_relations_respect_type_signatures(fixture_weights):
    for d in m2.predict_detailed(SAMPLE):
        htypes, ttypes = m2.RELATION_SIGNATURES[d["relation"]]
        assert d["head_type"] in htypes and d["tail_type"] in ttypes


def test_detailed_offsets_align_with_original_text(fixture_weights):
    text = "Note.  Haridra is useful in Kushta; Maricha aggravates Pitta."
    detailed = m2.predict_detailed(text, include_no_relation=True)
    assert detailed
    for d in detailed:
        s, e = d["evidence_span"]
        assert text[s:e] == d["evidence"]
        hs, he = d["head_span"]
        ts, te = d["tail_span"]
        assert text[hs:he] == d["head"] and text[ts:te] == d["tail"]
        total = 0.0
        for w in d["token_attributions"]:
            assert text[w["start"]:w["end"]] == w["token"]
            assert s <= w["start"] < w["end"] <= e
            total += w["score"]
        assert abs(total - 1.0) < 0.01
        assert set(d["probabilities"]) == set(m2.LABELS)
        assert d["explanation_method"].startswith("attention_rollout")


def test_source_override_is_preserved(fixture_weights):
    ref = "Charak Samhita New Edition, Chikitsa Sthana 7, verse 12 (test reference)"
    out = m2.predict(SAMPLE, source=ref)
    assert out and all(o["source"] == ref for o in out)


def test_saved_artifacts_reload(fixture_weights):
    first = m2.predict(SAMPLE)
    m2.reset()
    second = m2.predict(SAMPLE)
    assert first == second
    info = m2.get_model_info()
    assert info["weights_dir"] == str(Path(fixture_weights).resolve())


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_entity_finder_longest_match_and_ambiguity():
    ents = m2.find_entities("Khadira cures kushtha; ushna virya; dry ginger.")
    texts = [e.text for e in ents]
    assert "ushna virya" in texts and "ushna" not in texts
    assert "dry ginger" in texts and "ginger" not in texts
    kushtha = next(e for e in ents if e.text == "kushtha")
    assert {"HERB", "DISEASE"} <= set(kushtha.types)


def test_entity_finder_word_boundaries():
    assert m2.find_entities("vatara kaphaja pittala") == []


def test_mark_entities_mapping_roundtrip():
    text = "Haridra is useful in Kushta."
    marked, mapping = m2.mark_entities(text, (21, 27), (0, 7))
    assert marked.count("[E1]") == 1 and marked.count("[E2]") == 1
    rebuilt = "".join(ch for ch, src in zip(marked, mapping) if src is not None)
    assert rebuilt == text
    assert marked.index("[E2]") < marked.index("[E1]")


def test_candidate_pairs_only_type_compatible():
    pairs = m2.candidate_pairs(m2.find_entities("Haridra reduces Kapha and Pitta."))
    assert {(h.text, t.text) for h, t, _, _ in pairs} == {("Haridra", "Kapha"), ("Haridra", "Pitta")}


def test_attention_rollout_rows_sum_to_one():
    torch = pytest.importorskip("torch")
    att = [torch.softmax(torch.randn(2, 6, 6), dim=-1) for _ in range(3)]
    scores = m2.attention_rollout(att, 6)
    assert len(scores) == 6 and abs(sum(scores) - 1.0) < 1e-5
    uniform = [torch.full((1, 4, 4), 0.25)]
    assert m2.attention_rollout(uniform, 4) == pytest.approx([0.625, 0.125, 0.125, 0.125])


def test_split_sentences_offsets():
    text = "  First one. Second; third!\nFourth"
    for s in m2.split_sentences(text):
        assert text[s.start:s.end] == s.text
    assert [s.text for s in m2.split_sentences(text)] == ["First one.", "Second;", "third!", "Fourth"]


# ---------------------------------------------------------------------------
# Optional: real trained weights (skipped unless explicitly requested)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(os.environ.get("MODEL2_TEST_REAL_WEIGHTS") != "1",
                    reason="set MODEL2_TEST_REAL_WEIGHTS=1 after training to test weights/model2/")
def test_real_weights_schema(monkeypatch):
    monkeypatch.delenv("MODEL2_ALLOW_UNTRAINED", raising=False)
    monkeypatch.delenv("MODEL2_WEIGHTS_DIR", raising=False)
    m2.reset()
    assert m2.is_available(), "weights/model2/ does not contain trained artifacts"
    for item in m2.predict(SAMPLE):
        assert tuple(item.keys()) == m2.OUTPUT_KEYS
        assert item["relation"] in ALLOWED
        assert 0.0 <= item["confidence"] <= 1.0
