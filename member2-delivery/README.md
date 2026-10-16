# Member 2 — Explainable Ayurvedic Relation Extraction

PubMedBERT relation classifier for Charaka Samhita (Nidana & Chikitsa Sthana),
with attention-rollout explanations, packaged for the team's shared Gradio app.

| | Member 1 (reference) | **Member 2 (this repo)** |
|---|---|---|
| Model | `emilyalsentzer/Bio_ClinicalBERT` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Corpus | `jaychedaa/Ayurveda-LLM-dataset` | Charak Samhita New Edition — Nidana & Chikitsa Sthana (English translation, verse-referenced) |
| Explainability | Captum Layer Integrated Gradients | Attention rollout + evidence sentence |
| Exports | `models/model1.py`, `weights/model1/` | `models/model2.py`, `weights/model2/` |

> **Status of this delivery.** Code, notebook, tests and documentation are complete.
> **No trained weights are included**: the environment used to prepare it had no
> access to the Hugging Face Hub or the corpus site. `weights/model2/` must be
> produced by running the notebook/scripts (≈ 1–2 h on a Colab T4 GPU — estimate,
> not measured). Until then `predict()` raises `Model2NotAvailableError`.

## 1. Layout

```
member2-delivery/
├── models/model2.py           # production inference module (team contract)
├── weights/model2/            # trained artifacts (not in git)
├── training/                  # corpus loader, weak labels, splits, training, evaluation
├── notebooks/member2.ipynb    # experiment notebook (calls training/ and models/)
├── data/README.md             # source, licence, provenance, annotation guideline
├── tests/                     # pytest: contract, helpers, pipeline, tiny end-to-end train
├── requirements.txt
├── INTEGRATION_GUIDE.md       # instructions for the team lead
└── README.md
```

## 2. Model

* Checkpoint: `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` — BERT-base (uncased, own biomedical WordPiece vocabulary), pretrained from scratch on PubMed abstracts and PubMed Central full text (Gu et al., 2020, arXiv:2007.15779). MIT licence. Upstream it has been renamed *BiomedBERT* (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`); the old id still resolves with transformers ≥ 4.22. Information taken from the model card; the download itself was not tested here.
* Language: **English**. It does not understand Sanskrit or Devanagari, so only English translations are used. Romanised Sanskrit terms (e.g. *haridra*) are split into several word pieces; the notebook (§2) prints this fragmentation.
* Fine-tuning is required: the released checkpoint is a masked-LM; the classification head is new.
* Size ≈ 110 M parameters (≈ 420 MB fp32). CPU inference is feasible (≈ 0.1–0.5 s per sentence, estimate); training needs a GPU for reasonable time.
* Explainability compatibility: standard BERT self-attention, so per-layer attention probabilities are available — **only with eager attention**. Recent transformers default to SDPA, which returns no attention weights; the module loads with `attn_implementation="eager"` and fails loudly if attentions are missing.

## 3. Corpus

**Final source:** *Charak Samhita New Edition* (Charak Samhita Research, Training and Skill Development Centre), carakasamhitaonline.com — English translation paragraphs of Nidana Sthana and Chikitsa Sthana, each attached to its verse reference, with chapter DOI, page revision id and URL. Licence **CC BY-NC-SA 4.0** (non-commercial, attribution, share-alike). See `data/README.md`.

**Why not `shinigamiRaj/IndianVedasOriginal`?** It could not be inspected here (Hub unreachable). The only public description found (of the same author's model) points to Vedas in Griffith translations and bilingual Sanskrit/Hindi text with "structural chapters" for Ayurvedic works — no evidence of an English, verse-referenced Charaka Samhita. Keyword-filtering a mixed corpus for "Charaka" does not prove provenance. `training/inspect_hf_candidate.py` implements the required check (metadata-based identification, script/language, references, licence); notebook §3 runs it. Switch only if it reports `suitable: true`.

Both sources are distinct from Member 1's corpus.

## 4. Pipeline

1. **Corpus** (`training/corpus_charaka.py`): discover chapter pages from the sthana index pages, verify the in-page header "<Sthana> Sthana Chapter n", cache HTML, extract English verse units with provenance.
2. **Entities** (`models/model2.py`): typed gazetteer — HERB, FORMULATION, DIET, DISEASE, DOSHA, PROPERTY; case-insensitive, word-boundary, longest match. **Rule-based; not a learned NER.** Ambiguous strings keep several types (e.g. *kushtha*: disease or the herb *Saussurea*).
3. **Pair generation:** only same-sentence pairs whose types fit a relation signature.
4. **Weak labels** (`training/weak_labels.py`): see §5.
5. **Classifier:** `[E1] head [/E1]` / `[E2] tail [/E2]` markers added as special tokens; `BertForSequenceClassification` on `[CLS]`; at inference the argmax is taken only over labels allowed for the pair's types (plus `NO_RELATION`).
6. **Training** (`training/train.py`): chapter-level train/dev/test split, class-weighted CE, AdamW (lr 2e-5, wd 0.01), linear warm-up 10 %, ≤ 5 epochs with early stopping on dev macro-F1, seed 42, max length 256.
7. **Evaluation** (`training/evaluate.py`): per-label P/R/F1, macro-F1, confusion matrix, error CSV; rule baseline on gold data.
8. **Inference/explanations** (`models/model2.py`).

### Relation labels

| Label | Head type(s) | Tail type(s) | Meaning |
|---|---|---|---|
| `TREATS` | HERB, FORMULATION | DISEASE (incl. symptoms) | stated to cure, alleviate, or be indicated for |
| `REDUCES` | HERB, FORMULATION, DIET | DOSHA | stated to pacify / alleviate / reduce the dosha |
| `AGGRAVATES` | HERB, FORMULATION, DIET | DOSHA | stated to increase / provoke / vitiate the dosha |
| `HAS_PROPERTY` | HERB | PROPERTY (rasa, guna, virya, vipaka value) | stated property |
| `NO_RELATION` | any compatible pair | | none of the above is stated between the pair |

Schema gap (for the team): Nidana Sthana is mostly **etiology** ("X causes kushtha"). The shared schema has no CAUSES label; such pairs are flagged `causal_out_of_schema`, excluded from training, and should be `NO_RELATION` (with a note) in gold review.

## 5. Training data quality

* All automatically produced labels are **weak**. `label_origin` is kept on every example: `weak_positive`, `weak_negative`, `uncertain` (excluded), `gold` (reviewed).
* Negatives are not "every pair without a keyword": `weak_negative` is assigned only when the sentence contains no relational and no causal language at all; all other undecided pairs are `uncertain` and go to review.
* Leakage: splits are by chapter; the split report (stored in `member2_metadata.json`) asserts zero chapter overlap and lists labels missing from the test split.
* Review workflow: `data/annotation/review_sheet.csv` → two reviewers → Cohen's κ (`training.data.cohens_kappa`) → adjudicated `reviewed.csv` → gold test set. Recommended: ≥ 200 gold pairs from held-out chapters, covering every label.
* `HAS_PROPERTY` statements are rare in Nidana/Chikitsa Sthana (they are mostly in Sutra Sthana). Expect few or no examples; report per-label support and do not interpret F1 for labels with tiny support.
* Training aborts with fewer than 50 trainable examples rather than producing a meaningless model.

## 6. Explainability

**Attention rollout** (Abnar & Zuidema, 2020): for each layer, attention is averaged over heads, the residual path is modelled as `0.5·A + 0.5·I`, rows are re-normalised, and the layer matrices are multiplied from the first to the last layer. The `[CLS]` row of the product is the relevance of each input token to the representation used by the classifier. Sub-word scores are summed per word; `[CLS]`, `[SEP]` and the entity markers are removed and the remaining word scores are renormalised to sum to 1. Every cue is an exact substring of the input; `predict_detailed()` also returns character offsets.

What it is not: rollout is **class-agnostic** (the same map for every label of one input), ignores value vectors and feed-forward layers, and is not proof of causal reasoning. Attention usually concentrates on special tokens and entity mentions. If a class-specific attribution is needed, the recommended alternative is gradient-weighted attention or Layer Integrated Gradients on the embeddings (Member 1 already uses the latter). Evidence highlighting (the supporting sentence with offsets) is independent of the model and always exact.

## 7. Installation

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q tests/                                       # ~10 s, no download needed
```

## 8. Reproduce

```bash
python -m training.inspect_hf_candidate                               # optional: check the originally proposed corpus
python -m training.corpus_charaka --out data/processed/charaka_units.jsonl
python -m training.build_dataset                                      # candidates + review sheet
# (manual review -> data/annotation/reviewed.csv, then rerun build_dataset --reviewed ... )
python -m training.train --data data/processed/candidates.jsonl --out weights/model2 \
       [--gold-test data/annotation/gold_test.jsonl]
python -m training.evaluate --weights weights/model2 --data data/annotation/gold_test.jsonl
MODEL2_TEST_REAL_WEIGHTS=1 pytest -q tests/
```
or run `notebooks/member2.ipynb` top to bottom.

## 9. Inference

```python
from models.model2 import predict, predict_detailed
predict("Haridra is useful in Kushta and reduces Kapha.")
# -> [{"head", "relation", "tail", "confidence", "evidence", "model", "source", "explanation_cues"}, ...]
```
* `confidence`: raw softmax probability of the returned label — **not calibrated**.
* `evidence`: the exact sentence/clause of the input containing both entities.
* `source`: description of the training corpus, or the reference passed as `predict(text, source=...)`.
* `NO_RELATION` pairs are omitted from `predict()`; `predict_detailed(..., include_no_relation=True)` shows them.

## 10. Audit of the original notebooks

Member 2 notebook (never executed):
1. Attention rollout used `output_attentions=True` with the default SDPA kernel → empty attentions → `explanation_cues` silently `[]`.
2. Fallback to the base checkpoint when weights were missing → random classification head, labels `LABEL_n` (not team labels) could be returned.
3. Model downloaded/loaded and embeddings resized at import time.
4. Corpus filtered by keyword hits in the text body; provenance not verified; language not checked.
5. No negatives: pairs without a keyword were dropped, so `NO_RELATION` never occurred in training.
6. Random row split (leakage across near-identical passages).
7. Evidence = whole input; cues = WordPiece tokens (`##…`), not aligned to input; markers could leak into cues.
8. Case-sensitive type checks (`tail in DISEASE_TERMS` with `"kushta"` vs `"Kushta"`) dropped valid predictions.
9. `display()` and notebook state required by the export cell.

Member 1 notebook (reference only; not modified): adds hand-written "anchor" sentences to training (including the demo sentence used for testing → leakage), assigns `NO_RELATION` to every pair without a keyword, has no held-out evaluation, omits `HAS_PROPERTY`, and falls back to the base checkpoint. Its environment (torch 2.14.1, transformers 5.19.0) matches the versions this module was tested with.

## 11. Limitations

* Gazetteer NER: misses unlisted names and spelling variants; some forms are ambiguous in context (*bala* = herb or "strength", *guru* = "heavy" or "teacher", *amla* = sour or amalaki). Recall of the whole system is bounded by the lexicon.
* Pairs across sentences are not considered; long lists in one sentence create many candidates.
* Weak labels encode the rules' blind spots; agreement with them is not accuracy.
* Translation layer: relations are extracted from a modern English translation, not from the Sanskrit; translation choices affect results.
* Confidence is uncalibrated; small data makes it over-confident.
* Attention rollout limitations as in §6.

## 12. Ethical and scientific caveats

* Output describes statements in a classical text; it is **not** medical advice and has no clinical validation.
* Results must be reported as text-mining of Charaka Samhita, not as evidence of efficacy.
* Respect CC BY-NC-SA 4.0: attribute the Charak Samhita New Edition, no commercial use, share derived data under the same licence.
* Do not present weak-label scores as accuracy; report gold-label results with κ and support.

## 13. Verification status of this delivery

Executed here (Python 3.12, CPU, torch 2.14.1):
* `pytest -q tests/` — 39 passed, 1 skipped (real-weights test) on transformers 5.19.0 **and** 4.46.3.
* Tests cover: import without loading, empty/None/non-string input, missing artifacts, refusal of untrained artifacts, wrong label schema, output keys/types, label set, confidence range, evidence and cue alignment to the input, source override, save/reload, entity finder, markers, rollout maths, weak-label rules, chapter split, Cohen's κ, HTML parser (synthetic page), and a 1-epoch train → save → reload → `predict()` run on a tiny random BERT.
* Notebook: valid nbformat; every code cell parses.

Not executed here: download of PubMedBERT, corpus download/parsing of the live site, `shinigamiRaj/IndianVedasOriginal` inspection, real fine-tuning, any real evaluation. No accuracy figures exist yet.
