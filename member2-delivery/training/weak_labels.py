"""Candidate pairs and weak (rule-based) relation labels.

All labels produced here are WEAK labels.  They are not expert annotations and
must not be reported as ground truth.  Every candidate carries
``label_origin``:

``weak_positive``
    A relation trigger occurs *between* the two entities, in the right
    direction, inside one clause (no ``;`` or sentence end in between), within
    ``MAX_GAP_TOKENS`` words.  Passive forms ("kapha is alleviated by haridra")
    are recognised.  The label is decided by the trigger *and* the tail type
    (e.g. "alleviates" + DISEASE -> TREATS, "alleviates" + DOSHA -> REDUCES).

``weak_negative``
    A type-compatible pair in a sentence that contains NO relation trigger at
    all and no causal trigger.  These are plausible negatives, still weak.

``uncertain``
    Everything else: a trigger exists in the sentence but the pattern did not
    bind this pair, or the sentence expresses causation ("X causes kushtha"),
    which the shared schema cannot represent.  Uncertain candidates are NOT used
    for training; they are exported for manual review.

``gold`` (set later by reviewers) overrides any weak label.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

from models.model2 import Sentence, allowed_labels, candidate_pairs, find_entities, split_sentences

MAX_GAP_TOKENS = 12

_REDUCE = r"(?:alleviat\w*|pacif\w*|reduc\w*|subdu\w*|mitigat\w*|balanc\w*|decreas\w*|calm\w*|normali[sz]\w*|correct\w*)"
_TREAT = r"(?:cur\w*|treat\w*|heal\w*|remov\w*|destroy\w*|overcom\w*|reliev\w*|eradicat\w*|useful\s+(?:in|for)|effective\s+(?:in|for|against)|indicated\s+(?:in|for)|beneficial\s+(?:in|for)|good\s+for|best\s+(?:in|for)|prescribed\s+(?:in|for)|(?:is|are)\s+(?:a\s+)?remed\w*\s+for)"
_AGGRAVATE = r"(?:aggravat\w*|provok\w*|vitiat\w*|increas\w*|excit\w*|imbalanc\w*)"
_PROPERTY = r"(?:is|are|has|have|having|possess\w*|with|of)"
_CAUSAL = r"(?:caus\w*|lead\w*\s+to|result\w*\s+in|produc\w*|manifest\w*|etiolog\w*|due\s+to)"

ANY_TRIGGER = re.compile(rf"\b(?:{_REDUCE}|{_TREAT}|{_AGGRAVATE})", re.I)
CAUSAL = re.compile(rf"\b{_CAUSAL}", re.I)
NEGATION = re.compile(r"\b(?:not|never|no|neither|nor|without|contraindicated|avoid\w*)\b", re.I)


@dataclass
class Candidate:
    unit_id: str
    source_ref: str
    group: str                 # split group (chapter) to prevent leakage
    sentence: str
    head: str
    head_start: int
    head_end: int
    head_type: str
    tail: str
    tail_start: int
    tail_end: int
    tail_type: str
    weak_relation: Optional[str]
    label_origin: str
    rule: str
    gold_relation: Optional[str] = None
    reviewer: Optional[str] = None
    review_note: Optional[str] = None

    @property
    def relation(self) -> Optional[str]:
        return self.gold_relation or self.weak_relation


def _relation_from_trigger(trigger: str, tail_type: str) -> Optional[str]:
    t = trigger.lower()
    if re.fullmatch(_AGGRAVATE, t, re.I):
        return "AGGRAVATES" if tail_type == "DOSHA" else None
    if re.fullmatch(_REDUCE, t, re.I):
        return {"DOSHA": "REDUCES", "DISEASE": "TREATS"}.get(tail_type)
    if re.fullmatch(_TREAT, t, re.I):
        return "TREATS" if tail_type == "DISEASE" else ("REDUCES" if tail_type == "DOSHA" else None)
    return None


def weak_label(sentence: str, h_span: tuple[int, int], t_span: tuple[int, int],
               head_type: str, tail_type: str) -> tuple[Optional[str], str, str]:
    """Return ``(relation, label_origin, rule)`` for one candidate pair."""
    hs, he = h_span
    ts, te = t_span
    allowed = set(allowed_labels(head_type, tail_type))

    def bounded(gap: str) -> bool:
        return ";" not in gap and len(gap.split()) <= MAX_GAP_TOKENS

    # Active voice: head ... trigger ... tail
    if he <= ts:
        gap = sentence[he:ts]
        if bounded(gap):
            if tail_type == "PROPERTY" and head_type == "HERB" and len(gap.split()) <= 4 \
                    and re.search(rf"\b{_PROPERTY}\b", gap, re.I):
                rel = "HAS_PROPERTY"
                if rel in allowed and not NEGATION.search(gap):
                    return rel, "weak_positive", "property_copula"
            for m in re.finditer(rf"\b(?:{_AGGRAVATE}|{_TREAT}|{_REDUCE})", gap, re.I):
                rel = _relation_from_trigger(m.group(0), tail_type)
                if rel and rel in allowed and not NEGATION.search(gap):
                    return rel, "weak_positive", f"active:{m.group(0).lower()}"
    # Passive voice: tail ... (is|are) <trigger> by ... head
    if te <= hs:
        gap = sentence[te:hs]
        m = re.search(rf"\b(?:is|are|gets?|becomes?)\s+({_AGGRAVATE}|{_REDUCE}|{_TREAT})\s+(?:\w+\s+){{0,3}}?by\b",
                      gap, re.I)
        if m and bounded(gap) and not NEGATION.search(gap):
            rel = _relation_from_trigger(m.group(1), tail_type)
            if rel and rel in allowed:
                return rel, "weak_positive", f"passive:{m.group(1).lower()}"
    if CAUSAL.search(sentence):
        return None, "uncertain", "causal_out_of_schema"
    if not ANY_TRIGGER.search(sentence):
        return "NO_RELATION", "weak_negative", "no_trigger_in_sentence"
    return None, "uncertain", "trigger_not_bound_to_pair"


def candidates_from_text(text: str, unit_id: str, source_ref: str, group: str) -> list[Candidate]:
    out: list[Candidate] = []
    for sent in split_sentences(text):
        out.extend(candidates_from_sentence(sent, unit_id, source_ref, group))
    return out


def candidates_from_sentence(sent: Sentence, unit_id: str, source_ref: str, group: str) -> list[Candidate]:
    out = []
    for h, t, ht, tt in candidate_pairs(find_entities(sent.text)):
        rel, origin, rule = weak_label(sent.text, (h.start, h.end), (t.start, t.end), ht, tt)
        out.append(Candidate(unit_id, source_ref, group, sent.text, h.text, h.start, h.end, ht,
                             t.text, t.start, t.end, tt, rel, origin, rule))
    return out


def build_candidates(units: Iterable) -> list[Candidate]:
    """Candidates for every verse unit (``training.corpus_charaka.VerseUnit``)."""
    cands: list[Candidate] = []
    for u in units:
        group = f"{u.sthana}:{u.chapter_no:02d}"
        cands.extend(candidates_from_text(u.text, u.unit_id, u.source_ref, group))
    return cands


def to_records(cands: Iterable[Candidate]) -> list[dict]:
    return [asdict(c) for c in cands]
