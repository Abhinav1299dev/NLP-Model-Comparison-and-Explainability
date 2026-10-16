# Integration guide — Member 2 (PubMedBERT, Charaka Samhita)

## 1. Copy the module

```
<team-repo>/
├── app.py
├── models/
│   ├── __init__.py          # keep your existing one (may be empty)
│   ├── model1.py
│   └── model2.py            # <- copy member2-delivery/models/model2.py here
└── weights/
    ├── model1/
    └── model2/              # <- trained artifacts (step 2)
```

`model2.py` is self-contained (no imports from `training/`). It locates weights
at `<repo>/weights/model2/` relative to its own file, or at
`$MODEL2_WEIGHTS_DIR` if that is set.

## 2. Place the weights

Copy the trained folder so that these files exist:

```
weights/model2/config.json
weights/model2/model.safetensors
weights/model2/tokenizer.json
weights/model2/tokenizer_config.json
weights/model2/vocab.txt
weights/model2/member2_metadata.json      # must contain "status": "trained"
```

The weights are not in git (≈ 420 MB); I will share them as a release asset /
Drive folder once training is complete.

## 3. Dependencies

```bash
pip install "torch>=2.2" "transformers>=4.46,<6" "tokenizers>=0.20" safetensors numpy
```
Tested with torch 2.14.1 and transformers 5.19.0 (same as Member 1's
environment) and with transformers 4.46.3. No Captum needed. Inference runs
on CPU; set `MODEL2_DEVICE=cuda` or `cpu` to force a device.

## 4. Import

```python
from models.model2 import predict
results = predict("Haridra is useful in Kushta and reduces Kapha.")
```

* Importing is cheap: nothing is loaded until the first `predict()` call
  (thread-safe, loaded once, reused).
* `predict("")`, whitespace or `None` → `[]` (no model load).
* Non-string input → `TypeError`.
* Missing/invalid weights → `models.model2.Model2NotAvailableError` with a message
  naming the missing file. No fabricated or untrained output is ever returned.
* `NO_RELATION` pairs are not returned (same as Member 1).
* Extra helpers: `is_available()`, `get_model_info()`, `predict_detailed(text)`
  (adds character offsets, entity types, all class probabilities and per-word
  attention-rollout scores for heatmaps).

## 5. Check the output schema

```bash
pytest -q tests/test_model2.py                         # contract tests, no weights needed
MODEL2_TEST_REAL_WEIGHTS=1 pytest -q tests/test_model2.py   # also checks weights/model2/
```
or quickly:

```python
from models.model2 import predict, OUTPUT_KEYS
for r in predict("Haridra is useful in Kushta and reduces Kapha."):
    assert tuple(r) == OUTPUT_KEYS
    assert r["relation"] in {"TREATS", "REDUCES", "AGGRAVATES", "HAS_PROPERTY"}
    assert 0.0 <= r["confidence"] <= 1.0
```

Each item has exactly: `head`, `relation`, `tail`, `confidence` (uncalibrated
softmax probability), `evidence` (exact sentence of the input), `model`
(`"PubMedBERT (Member 2)"`), `source`, `explanation_cues`
(`[{"token": <word from input>, "score": <rollout share 0–1>}]`, top 5).

## 6. Use in the shared Gradio app

```python
import gradio as gr
from models import model1, model2

MODELS = {"Member 1 — Bio_ClinicalBERT": model1.predict,
          "Member 2 — PubMedBERT": model2.predict}

def run_all(text):
    rows, errors = [], []
    for name, fn in MODELS.items():
        try:
            for r in fn(text):
                rows.append([name, r["head"], r["relation"], r["tail"], r["confidence"], r["evidence"],
                             ", ".join(f'{c["token"]} ({c["score"]})' for c in r["explanation_cues"])])
        except Exception as exc:            # e.g. Model2NotAvailableError
            errors.append(f"{name}: {exc}")
    return rows, "\n".join(errors) or "OK"

demo = gr.Interface(
    fn=run_all,
    inputs=gr.Textbox(lines=4, label="Ayurvedic text (English)"),
    outputs=[gr.Dataframe(headers=["model", "head", "relation", "tail", "confidence", "evidence", "cues"]),
             gr.Textbox(label="status")],
)
demo.launch()
```

Heatmap: `model2.predict_detailed(text)[i]["token_attributions"]` gives
`{"token", "start", "end", "score"}` for every word of the evidence sentence,
ready for `gr.HighlightedText` (map each word span to its score).

Wording for the UI: label the cues "attention rollout (indicative, not causal)"
and confidence "model probability (uncalibrated)".

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `Model2NotAvailableError: ... member2_metadata.json is missing` | Weights not copied, or wrong folder. Check `weights/model2/` or set `MODEL2_WEIGHTS_DIR=/abs/path/to/model2`. |
| `... status='test_fixture'` / not `'trained'` | You copied a test fixture or an unfinished run. Use the trained folder. |
| `Label set ... does not match the team schema` | Weights from another model/member. Use Member 2's folder. |
| `Tokenizer ... lacks entity markers` | Tokenizer files missing from `weights/model2/`. Copy all files. |
| `Model returned no attention weights` | Model loaded with SDPA elsewhere; use `model2.predict` (it forces eager attention). |
| `ModuleNotFoundError: torch/transformers` | `pip install` step 3. |
| Empty list for a valid sentence | No lexicon entity pair in the same sentence, or the model predicted `NO_RELATION`. Inspect with `predict_detailed(text, include_no_relation=True)`. |
| Slow first call | One-time model load (~420 MB); later calls reuse it. |
| Hugging Face download attempted | Should not happen; the module loads only local files. Set `HF_HUB_OFFLINE=1` to be sure. |
