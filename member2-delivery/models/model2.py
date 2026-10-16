"""Member 2 inference module — Explainable Ayurvedic Relation Extraction.

Model
    ``microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`` (renamed
    upstream to ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext``),
    fine-tuned as an entity-marked relation classifier.  The fine-tuned artifacts
    are loaded from ``weights/model2/`` (or ``$MODEL2_WEIGHTS_DIR``).

Pipeline per call to :func:`predict`
    1. Split the input into sentences (evidence units) with character offsets.
    2. Rule-based, gazetteer entity recognition (typed: HERB, FORMULATION, DIET,
       DISEASE, DOSHA, PROPERTY).  This is NOT a learned NER model.
    3. Candidate pairs: only type-compatible (head, tail) pairs inside the same
       sentence (see ``RELATION_SIGNATURES``).
    4. Relation classification of each pair with entity markers
       ``[E1] head [/E1]`` / ``[E2] tail [/E2]``.
    5. Explanation: attention rollout (Abnar & Zuidema, 2020) from the ``[CLS]``
       position, mapped back to exact character spans of the original input.

Contract (shared team schema)
    ``predict(text: str) -> list[dict]`` returns dictionaries with exactly the keys
    head, relation, tail, confidence, evidence, model, source, explanation_cues.
    ``NO_RELATION`` pairs are not returned by :func:`predict`; use
    :func:`predict_detailed` for full diagnostic output.

The module never trains, never downloads a base checkpoint as a substitute for
the fine-tuned model, and never fabricates predictions.  If the fine-tuned
artifacts are missing, :class:`Model2NotAvailableError` is raised.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and team schema
# ---------------------------------------------------------------------------

MODEL_NAME = "PubMedBERT (Member 2)"
BASE_CHECKPOINT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
LABELS: tuple[str, ...] = ("NO_RELATION", "TREATS", "REDUCES", "AGGRAVATES", "HAS_PROPERTY")
POSITIVE_LABELS: frozenset[str] = frozenset(LABELS[1:])
OUTPUT_KEYS: tuple[str, ...] = (
    "head", "relation", "tail", "confidence", "evidence", "model", "source", "explanation_cues",
)
ENTITY_MARKERS: tuple[str, str, str, str] = ("[E1]", "[/E1]", "[E2]", "[/E2]")
EXPLANATION_METHOD = (
    "attention_rollout(CLS row; heads averaged; 0.5*A + 0.5*I per layer; "
    "subword scores summed per word; normalised over input words)"
)
DEFAULT_SOURCE = (
    "Input text. Model fine-tuned on Charak Samhita New Edition "
    "(Nidana & Chikitsa Sthana, carakasamhitaonline.com, CC BY-NC-SA 4.0)"
)
DEFAULT_MAX_LENGTH = 256
DEFAULT_TOP_K_CUES = 5
METADATA_FILENAME = "member2_metadata.json"
_WEIGHTS_ENV = "MODEL2_WEIGHTS_DIR"
_ALLOW_UNTRAINED_ENV = "MODEL2_ALLOW_UNTRAINED"  # tests only; never set in production
_DEVICE_ENV = "MODEL2_DEVICE"

DEFAULT_WEIGHTS_DIR = (Path(__file__).resolve().parent.parent / "weights" / "model2")

# Allowed (head type, tail type) per relation.  Candidate pairs are generated only
# for type combinations that at least one positive relation allows.
RELATION_SIGNATURES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "TREATS": (frozenset({"HERB", "FORMULATION"}), frozenset({"DISEASE"})),
    "REDUCES": (frozenset({"HERB", "FORMULATION", "DIET"}), frozenset({"DOSHA"})),
    "AGGRAVATES": (frozenset({"HERB", "FORMULATION", "DIET"}), frozenset({"DOSHA"})),
    "HAS_PROPERTY": (frozenset({"HERB"}), frozenset({"PROPERTY"})),
}

# ---------------------------------------------------------------------------
# Gazetteer (rule-based entity recognition).  Single source of truth: the
# training pipeline imports this lexicon so train/inference stay consistent.
# Surface forms are matched case-insensitively on word boundaries; the longest
# match wins.  Terms with several plausible types are listed under each type.
# ---------------------------------------------------------------------------

LEXICON_VERSION = "2026-10-15.1"

ENTITY_LEXICON: dict[str, dict[str, list[str]]] = {
    "HERB": {
        "Haridra": ["haridra", "turmeric"],
        "Daruharidra": ["daruharidra", "daru haridra"],
        "Guduchi": ["guduchi", "amrita", "giloy"],
        "Amalaki": ["amalaki", "amla", "dhatri", "emblic myrobalan"],
        "Haritaki": ["haritaki", "abhaya", "chebulic myrobalan"],
        "Bibhitaki": ["bibhitaki", "vibhitaki"],
        "Pippali": ["pippali", "long pepper"],
        "Maricha": ["maricha", "black pepper"],
        "Shunthi": ["shunthi", "nagara", "dry ginger", "ginger"],
        "Nimba": ["nimba", "neem"],
        "Khadira": ["khadira"],
        "Vasa": ["vasa", "vasaka"],
        "Kutaja": ["kutaja"],
        "Chitraka": ["chitraka"],
        "Bilva": ["bilva"],
        "Musta": ["musta", "mustaka"],
        "Patola": ["patola"],
        "Aragvadha": ["aragvadha"],
        "Vidanga": ["vidanga"],
        "Bhallataka": ["bhallataka"],
        "Kushtha (Saussurea)": ["kushtha", "kushta", "kustha"],  # ambiguous with the disease
        "Ashwagandha": ["ashwagandha", "ashvagandha"],
        "Bala": ["bala"],
        "Yashtimadhu": ["yashtimadhu", "madhuka", "liquorice", "licorice"],
        "Draksha": ["draksha", "grapes", "raisins"],
        "Dadima": ["dadima", "pomegranate"],
        "Brahmi": ["brahmi"],
        "Tulsi": ["tulsi", "tulasi"],
        "Arjuna": ["arjuna"],
        "Shatavari": ["shatavari"],
        "Punarnava": ["punarnava"],
        "Kantakari": ["kantakari"],
    },
    "FORMULATION": {
        "Triphala": ["triphala"],
        "Trikatu": ["trikatu"],
        "Guggulu": ["guggulu", "guggul"],
        "Shilajatu": ["shilajatu", "shilajit"],
        "Ghrita": ["medicated ghee", "medicated ghrita"],
        "Chyavanaprasha": ["chyavanaprasha", "chyawanprash"],
    },
    "DIET": {
        "Madhu": ["madhu", "honey"],
        "Ghrita": ["ghrita", "ghee"],
        "Ksheera": ["ksheera", "kshira", "milk"],
        "Takra": ["takra", "buttermilk"],
        "Dadhi": ["dadhi", "curd", "yogurt"],
        "Taila": ["taila", "sesame oil"],
        "Matsya": ["matsya", "fish"],
        "Mamsa rasa": ["meat soup", "mamsa rasa"],
        "Yava": ["yava", "barley"],
        "Masha": ["masha", "black gram"],
    },
    "DISEASE": {
        "Kushtha": ["kushtha", "kushta", "kustha", "skin disease", "skin diseases"],
        "Jwara": ["jwara", "jvara", "fever"],
        "Prameha": ["prameha", "diabetes"],
        "Madhumeha": ["madhumeha"],
        "Shwasa": ["shwasa", "shvasa", "dyspnoea", "dyspnea", "asthma"],
        "Kasa": ["kasa", "cough"],
        "Amlapitta": ["amlapitta", "hyperacidity"],
        "Atisara": ["atisara", "diarrhoea", "diarrhea"],
        "Grahani": ["grahani"],
        "Arsha": ["arsha", "piles", "haemorrhoids", "hemorrhoids"],
        "Pandu": ["pandu", "anaemia", "anemia"],
        "Hridroga": ["hridroga", "heart disease"],
        "Raktapitta": ["raktapitta"],
        "Gulma": ["gulma"],
        "Unmada": ["unmada", "insanity"],
        "Apasmara": ["apasmara", "epilepsy"],
        "Shosha": ["shosha"],
        "Rajayakshma": ["rajayakshma", "consumption"],
        "Visarpa": ["visarpa", "erysipelas"],
        "Udara": ["udara roga", "ascites"],
        "Shvayathu": ["shvayathu", "shotha", "oedema", "edema"],
        "Chhardi": ["chhardi", "vomiting"],
        "Trishna": ["trishna", "morbid thirst"],
        "Vatarakta": ["vatarakta", "gout"],
        "Krimi": ["krimi", "worm infestation"],
        "Kilasa": ["kilasa", "shvitra", "vitiligo"],
        "Kandu": ["kandu", "itching"],
        "Daha": ["daha", "burning sensation"],
        "Aruchi": ["aruchi", "anorexia"],
        "Ajirna": ["ajirna", "indigestion"],
        "Medoroga": ["sthaulya", "obesity"],
    },
    "DOSHA": {
        "Vata": ["vata", "vayu", "anila", "maruta"],
        "Pitta": ["pitta"],
        "Kapha": ["kapha", "shleshma", "shleshman", "shleshmā"],
        "Tridosha": ["tridosha", "all three dosha", "all three doshas"],
    },
    "PROPERTY": {
        "Madhura rasa": ["madhura", "sweet taste"],
        "Amla rasa": ["amla rasa", "sour taste"],
        "Lavana rasa": ["lavana", "salty taste"],
        "Katu rasa": ["katu", "pungent"],
        "Tikta rasa": ["tikta", "bitter"],
        "Kashaya rasa": ["kashaya", "astringent"],
        "Ushna virya": ["ushna virya", "ushna", "hot potency"],
        "Sheeta virya": ["sheeta virya", "shita virya", "sheeta", "shita", "cold potency"],
        "Laghu guna": ["laghu"],
        "Guru guna": ["guru"],
        "Snigdha guna": ["snigdha", "unctuous"],
        "Ruksha guna": ["ruksha", "rooksha"],
        "Tikshna guna": ["tikshna"],
        "Madhura vipaka": ["madhura vipaka"],
        "Katu vipaka": ["katu vipaka"],
    },
}

_ENTITY_TYPES = tuple(ENTITY_LEXICON.keys())


class Model2Error(RuntimeError):
    """Base error for Member 2."""


class Model2NotAvailableError(Model2Error):
    """Raised when the fine-tuned artifacts in ``weights/model2/`` are missing or invalid."""


# ---------------------------------------------------------------------------
# Text utilities (no ML dependencies; importable by the training pipeline)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    """An entity mention found by the gazetteer."""

    text: str          # exact surface text from the input
    start: int         # character offset in the input (inclusive)
    end: int           # character offset in the input (exclusive)
    types: tuple[str, ...]
    canonical: dict[str, str] = field(default_factory=dict, compare=False, hash=False)

    def canonical_for(self, etype: str) -> str:
        return self.canonical.get(etype, self.text)


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


_SENT_RE = re.compile(r"[^.!?;\n]+(?:[.!?]+|;|\n|$)")


def split_sentences(text: str) -> list[Sentence]:
    """Split text into evidence units with exact character offsets.

    Boundaries are ``.``, ``!``, ``?``, ``;`` and newlines.  Simple and
    deterministic; abbreviations such as "e.g." can cause over-splitting, which
    only makes evidence spans shorter (never misaligned).
    """
    out: list[Sentence] = []
    for m in _SENT_RE.finditer(text):
        raw = m.group(0)
        lstrip = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if not stripped or not re.search(r"[A-Za-z]", stripped):
            continue
        start = m.start() + lstrip
        out.append(Sentence(stripped, start, start + len(stripped)))
    return out


def _build_patterns() -> list[tuple[re.Pattern[str], str, str, str]]:
    items: list[tuple[str, str, str]] = []
    for etype, entries in ENTITY_LEXICON.items():
        for canonical, forms in entries.items():
            for form in forms:
                items.append((form, etype, canonical))
    items.sort(key=lambda x: len(x[0]), reverse=True)
    pats = []
    for form, etype, canonical in items:
        escaped = r"\s+".join(re.escape(part) for part in form.split())
        pats.append((re.compile(r"(?<![\w-])" + escaped + r"(?![\w-])", re.IGNORECASE), form, etype, canonical))
    return pats


_PATTERNS = _build_patterns()


def find_entities(text: str) -> list[Entity]:
    """Return non-overlapping gazetteer mentions (longest match wins).

    A span matched by forms of several types (e.g. "kushtha" = disease or the
    herb Saussurea) keeps all of those types; pair generation then decides which
    typed reading can participate in which relation.
    """
    hits: dict[tuple[int, int], dict[str, str]] = {}
    for pat, _form, etype, canonical in _PATTERNS:
        for m in pat.finditer(text):
            hits.setdefault((m.start(), m.end()), {}).setdefault(etype, canonical)
    selected: list[tuple[int, int]] = []
    for span in sorted(hits, key=lambda s: (-(s[1] - s[0]), s[0])):
        if all(span[1] <= s[0] or span[0] >= s[1] for s in selected):
            selected.append(span)
    ents = []
    for s, e in sorted(selected):
        canon = hits[(s, e)]
        types = tuple(t for t in _ENTITY_TYPES if t in canon)
        ents.append(Entity(text[s:e], s, e, types, dict(canon)))
    return ents


def candidate_pairs(entities: Iterable[Entity]) -> list[tuple[Entity, Entity, str, str]]:
    """Type-compatible ordered pairs ``(head, tail, head_type, tail_type)``.

    One pair is produced per (head span, tail span); when a span is ambiguous the
    first compatible typed reading is used.  Spans never pair with themselves.
    """
    ents = list(entities)
    pairs: list[tuple[Entity, Entity, str, str]] = []
    for h in ents:
        for t in ents:
            if h is t or not (h.end <= t.start or t.end <= h.start):
                continue
            chosen = None
            for _rel, (htypes, ttypes) in RELATION_SIGNATURES.items():
                ht = next((x for x in h.types if x in htypes), None)
                tt = next((x for x in t.types if x in ttypes), None)
                if ht and tt:
                    chosen = (ht, tt)
                    break
            if chosen:
                pairs.append((h, t, chosen[0], chosen[1]))
    return pairs


def allowed_labels(head_type: str, tail_type: str) -> list[str]:
    """Labels permitted for a typed pair (``NO_RELATION`` is always allowed)."""
    out = ["NO_RELATION"]
    for rel, (htypes, ttypes) in RELATION_SIGNATURES.items():
        if head_type in htypes and tail_type in ttypes:
            out.append(rel)
    return out


def mark_entities(text: str, head: tuple[int, int], tail: tuple[int, int]) -> tuple[str, list[Optional[int]]]:
    """Insert ``[E1] … [/E1]`` / ``[E2] … [/E2]`` markers.

    Returns the marked string and a list mapping every character of the marked
    string to the offset of the same character in ``text`` (``None`` for
    characters that belong to inserted markers or padding spaces).
    """
    (hs, he), (ts, te) = head, tail
    if not (he <= ts or te <= hs):
        raise ValueError("Entity spans overlap")
    inserts = sorted(
        [(hs, "[E1] "), (he, " [/E1]"), (ts, "[E2] "), (te, " [/E2]")],
        key=lambda x: (x[0], 0 if x[1].startswith(" ") else 1),
    )
    out: list[str] = []
    mapping: list[Optional[int]] = []
    pos = 0
    for at, marker in inserts:
        out.append(text[pos:at])
        mapping.extend(range(pos, at))
        out.append(marker)
        mapping.extend([None] * len(marker))
        pos = at
    out.append(text[pos:])
    mapping.extend(range(pos, len(text)))
    return "".join(out), mapping


def attention_rollout(attentions: Any, seq_len: int) -> list[float]:
    """Attention rollout relevance of every token for the ``[CLS]`` position.

    ``attentions`` is a sequence of per-layer tensors shaped
    ``(heads, seq, seq)`` (batch dimension already removed) restricted to the
    real (unpadded) tokens.  For each layer the heads are averaged, the
    residual connection is modelled as ``0.5*A + 0.5*I`` and rows are
    re-normalised; layer matrices are multiplied from the first to the last
    layer.  Row 0 of the product is returned (sums to 1).
    """
    import torch

    joint = torch.eye(seq_len, dtype=torch.float32)
    eye = torch.eye(seq_len, dtype=torch.float32)
    for layer in attentions:
        a = layer[:, :seq_len, :seq_len].float().mean(dim=0).cpu()
        a = 0.5 * a + 0.5 * eye
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        joint = a @ joint
    return joint[0].tolist()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def weights_dir() -> Path:
    """Directory holding the fine-tuned artifacts (``$MODEL2_WEIGHTS_DIR`` overrides)."""
    return Path(os.environ.get(_WEIGHTS_ENV, str(DEFAULT_WEIGHTS_DIR))).expanduser().resolve()


def _read_metadata(path: Path) -> dict[str, Any]:
    meta_path = path / METADATA_FILENAME
    if not meta_path.is_file():
        raise Model2NotAvailableError(
            f"Member 2 artifacts not found: '{meta_path}' is missing. Train the model "
            f"(notebooks/member2.ipynb or `python -m training.train`) and place the output in "
            f"'{path}', or set {_WEIGHTS_ENV}. The base checkpoint is deliberately NOT used as a "
            f"substitute because its classification head is untrained."
        )
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    status = meta.get("status")
    if status != "trained" and os.environ.get(_ALLOW_UNTRAINED_ENV) != "1":
        raise Model2NotAvailableError(
            f"Artifacts in '{path}' have status={status!r}; only status='trained' is loaded."
        )
    return meta


def is_available() -> bool:
    """True if loadable fine-tuned artifacts are present (does not load the model)."""
    path = weights_dir()
    try:
        _read_metadata(path)
    except Model2NotAvailableError:
        return False
    return (path / "config.json").is_file()


def _load() -> dict[str, Any]:
    with _LOCK:
        if _STATE.get("dir") == str(weights_dir()) and "model" in _STATE:
            return _STATE
        path = weights_dir()
        meta = _read_metadata(path)
        if not (path / "config.json").is_file():
            raise Model2NotAvailableError(f"'{path / 'config.json'}' is missing.")
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment issue
            raise Model2Error(
                "Missing dependencies for Member 2. Install requirements.txt (torch, transformers)."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(str(path))
        missing = [m for m in ENTITY_MARKERS if m not in tokenizer.get_vocab()]
        if missing:
            raise Model2NotAvailableError(f"Tokenizer in '{path}' lacks entity markers {missing}.")
        # Attention rollout needs explicit attention probabilities; the default
        # SDPA kernel returns none, so eager attention is required.
        model = AutoModelForSequenceClassification.from_pretrained(
            str(path), attn_implementation="eager"
        )
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        if sorted(id2label.values()) != sorted(LABELS):
            raise Model2NotAvailableError(
                f"Label set {sorted(id2label.values())} in '{path}' does not match the team schema {list(LABELS)}."
            )
        if model.get_input_embeddings().num_embeddings < len(tokenizer):
            raise Model2NotAvailableError("Model embedding matrix is smaller than the tokenizer vocabulary.")
        device_name = os.environ.get(_DEVICE_ENV) or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        _STATE.clear()
        _STATE.update(
            dir=str(path), model=model, tokenizer=tokenizer, device=device, meta=meta,
            id2label=id2label, label2id={v: k for k, v in id2label.items()},
            max_length=int(meta.get("max_length", DEFAULT_MAX_LENGTH)),
            marker_ids={tokenizer.convert_tokens_to_ids(m) for m in ENTITY_MARKERS},
        )
        logger.info("Loaded Member 2 model from %s on %s", path, device)
        return _STATE


def get_model_info() -> dict[str, Any]:
    """Metadata of the loaded artifacts (loads the model if needed)."""
    state = _load()
    return {
        "model": MODEL_NAME,
        "base_checkpoint": state["meta"].get("base_checkpoint", BASE_CHECKPOINT),
        "weights_dir": state["dir"],
        "device": str(state["device"]),
        "labels": list(LABELS),
        "explanation_method": EXPLANATION_METHOD,
        "lexicon_version": LEXICON_VERSION,
        "metadata": state["meta"],
    }


def reset() -> None:
    """Forget the cached model (used by tests or after replacing weights)."""
    with _LOCK:
        _STATE.clear()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _word_cues(
    tokens: list[str], offsets: list[tuple[int, int]], word_ids: list[Optional[int]],
    scores: list[float], mapping: list[Optional[int]], sentence: Sentence, special: set[str],
) -> list[dict[str, Any]]:
    """Merge sub-word rollout scores into words aligned to the original input.

    Sub-words sharing a tokenizer word id are summed.  Special tokens and the
    entity markers are excluded; scores are renormalised over the remaining words.
    """
    words: list[dict[str, Any]] = []
    prev_wid: Optional[int] = None
    for tok, (s, e), wid, score in zip(tokens, offsets, word_ids, scores):
        if tok in special or e <= s or wid is None:
            prev_wid = None
            continue
        orig = [mapping[i] for i in range(s, e) if i < len(mapping) and mapping[i] is not None]
        if not orig:
            prev_wid = None
            continue
        o_start = sentence.start + orig[0]
        o_end = sentence.start + orig[-1] + 1
        if words and prev_wid == wid:      # continuation sub-word of the same word
            words[-1]["end"] = o_end
            words[-1]["score"] += score
        else:
            words.append({"start": o_start, "end": o_end, "score": score})
        prev_wid = wid
    total = sum(w["score"] for w in words)
    for w in words:
        w["score"] = w["score"] / total if total > 0 else 0.0
    return words


def _validate_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    if not isinstance(text, str):
        raise TypeError(f"predict() expects str, got {type(text).__name__}")
    return text if text.strip() else None


def predict_detailed(
    text: str, source: Optional[str] = None, top_k: int = DEFAULT_TOP_K_CUES,
    include_no_relation: bool = False,
) -> list[dict[str, Any]]:
    """Full diagnostic output for every classified candidate pair.

    Each item contains the team-schema keys plus: ``input_text``,
    ``evidence_span``, ``head_span``/``tail_span``, ``head_type``/``tail_type``,
    ``head_canonical``/``tail_canonical``, ``probabilities`` (all labels, raw
    softmax), ``allowed_labels``, ``explanation_method`` and
    ``token_attributions`` (every word with character offsets).
    """
    clean = _validate_text(text)
    if clean is None:
        return []
    state = _load()
    import torch

    model, tokenizer, device = state["model"], state["tokenizer"], state["device"]
    id2label, label2id = state["id2label"], state["label2id"]
    special = set(tokenizer.all_special_tokens) | set(ENTITY_MARKERS)
    results: list[dict[str, Any]] = []
    src = source or DEFAULT_SOURCE

    for sent in split_sentences(clean):
        ents = find_entities(sent.text)
        pairs = candidate_pairs(ents)
        if not pairs:
            continue
        marked, maps = [], []
        for h, t, _ht, _tt in pairs:
            m, mp = mark_entities(sent.text, (h.start, h.end), (t.start, t.end))
            marked.append(m)
            maps.append(mp)
        enc = tokenizer(
            marked, return_tensors="pt", padding=True, truncation=True,
            max_length=state["max_length"], return_offsets_mapping=True,
        )
        word_ids = [enc.word_ids(batch_index=b) for b in range(len(marked))]
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_attentions=True)
        if not out.attentions:
            raise Model2Error("Model returned no attention weights; eager attention is required for rollout.")
        probs = torch.softmax(out.logits.float(), dim=-1).cpu()

        for i, (h, t, ht, tt) in enumerate(pairs):
            ids = enc["input_ids"][i].tolist()
            length = int(enc["attention_mask"][i].sum().item())
            if len(state["marker_ids"] & set(ids[:length])) < 4:
                logger.warning("Skipping pair (%s, %s): entity markers truncated at max_length.", h.text, t.text)
                continue
            allowed = allowed_labels(ht, tt)
            p = probs[i]
            best = max(allowed, key=lambda lab: float(p[label2id[lab]]))
            if best == "NO_RELATION" and not include_no_relation:
                continue
            confidence = float(p[label2id[best]])
            roll = attention_rollout([layer[i] for layer in out.attentions], length)
            toks = tokenizer.convert_ids_to_tokens(ids[:length])
            words = _word_cues(toks, offsets[i][:length], word_ids[i][:length], roll, maps[i], sent, special)
            for w in words:
                w["token"] = clean[w["start"]:w["end"]]
                w["score"] = round(float(w["score"]), 4)
            ranked = sorted(words, key=lambda w: w["score"], reverse=True)
            # Punctuation-only words stay in token_attributions but are not offered as cues.
            cue_words = [w for w in ranked if any(ch.isalnum() for ch in w["token"])]
            cues = [{"token": w["token"], "score": w["score"]} for w in cue_words[:max(0, top_k)]]
            results.append({
                "head": h.text,
                "relation": best,
                "tail": t.text,
                "confidence": round(confidence, 4),
                "evidence": sent.text,
                "model": MODEL_NAME,
                "source": src,
                "explanation_cues": cues,
                "input_text": clean,
                "evidence_span": [sent.start, sent.end],
                "head_span": [sent.start + h.start, sent.start + h.end],
                "tail_span": [sent.start + t.start, sent.start + t.end],
                "head_type": ht, "tail_type": tt,
                "head_canonical": h.canonical_for(ht), "tail_canonical": t.canonical_for(tt),
                "allowed_labels": allowed,
                "probabilities": {id2label[j]: round(float(p[j]), 4) for j in range(len(p))},
                "explanation_method": EXPLANATION_METHOD,
                "token_attributions": [
                    {"token": w["token"], "start": w["start"], "end": w["end"], "score": w["score"]}
                    for w in words
                ],
            })
    return results


def predict(text: str, source: Optional[str] = None) -> list[dict]:
    """Team contract: extract relations from ``text``.

    Returns a list of dictionaries with exactly the keys in ``OUTPUT_KEYS``.
    Empty/whitespace/``None`` input returns ``[]``; non-string input raises
    ``TypeError``; missing artifacts raise :class:`Model2NotAvailableError`.

    ``confidence`` is the raw softmax probability of the returned label from the
    fine-tuned classifier.  It has not been calibrated.
    ``source`` optionally attaches a corpus reference (e.g. a verse id) to every
    prediction; otherwise a description of the model's training source is used.
    """
    detailed = predict_detailed(text, source=source)
    out: list[dict] = []
    seen: set[tuple[str, str, str, str]] = set()
    for d in detailed:
        key = (d["head"].lower(), d["relation"], d["tail"].lower(), d["evidence"])
        if key in seen:
            continue
        seen.add(key)
        item = {k: d[k] for k in OUTPUT_KEYS}
        if not (0.0 <= item["confidence"] <= 1.0) or math.isnan(item["confidence"]):
            raise Model2Error(f"Invalid confidence {item['confidence']!r}")
        out.append(item)
    return out


__all__ = [
    "predict", "predict_detailed", "is_available", "get_model_info", "reset",
    "Model2Error", "Model2NotAvailableError", "LABELS", "OUTPUT_KEYS", "ENTITY_LEXICON",
    "RELATION_SIGNATURES", "find_entities", "candidate_pairs", "split_sentences",
    "mark_entities", "attention_rollout", "allowed_labels", "MODEL_NAME", "BASE_CHECKPOINT",
]
