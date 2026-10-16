# weights/model2 — trained artifacts go here

**Status: no trained model is included in this delivery.** Training could not be
run where the delivery was prepared (no access to the Hugging Face Hub or the
corpus site, no GPU). `models/model2.py` therefore raises
`Model2NotAvailableError` until this folder is populated. It never falls back to
the untrained base checkpoint.

After running training (`notebooks/member2.ipynb` §7 or
`python -m training.train --out weights/model2`) this folder must contain:

| File | Produced by |
|---|---|
| `config.json` | `save_pretrained` — includes `id2label` with the five team labels |
| `model.safetensors` | `save_pretrained` (≈ 420 MB) |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` (+ `special_tokens_map.json` on transformers 4.x) | `tokenizer.save_pretrained` — includes `[E1] [/E1] [E2] [/E2]` |
| `member2_metadata.json` | `training/train.py` — `"status": "trained"`, data hash, split report, evaluation (weak vs gold) |
| `test_errors_weak.csv` / `test_errors_gold.csv` | error analysis |

Do not commit these files to git (size). Share them as a release asset or a
shared-drive folder and copy them here.
