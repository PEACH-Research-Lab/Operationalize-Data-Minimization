# Data Minimization Pipeline

This pipeline hides PII in prompts while keeping the task solvable.
It was built for **WildChat** (open‑ended). It also works for similar open‑ended datasets.
For close‑ended datasets (e.g., MedQA), see the note at the end.

---

## What it does

* Get a baseline answer from your chosen model.
* Try masking each PII item:

  * **redact** → e.g., `[PERSON1]`
  * **abstract** → e.g., `an individual`
* Keep changes that still let the model answer well.
* Search small edits to reduce exposure further.
* Save the final masked prompt and the mapping.

---

## Files

* `data_minimization_pipeline_clean_anon.py` — main script and CLI.
* `Prefiltered datasets/` — your dataset JSONL files (use these as `--dataset`).
* `data_minimization_results/` — outputs will be written here.

---

## Install

```bash
pip install -r requirements.txt
# please set keys only for the providers you use
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export TOGETHER_API_KEY=...
export OPENROUTER_API_KEY=...
export FIREWORKS_API_KEY=...
```

---

## Run

```bash
# single model
python data_minimization_pipeline_clean_anon.py \
  --models gpt-4o \
  --dataset "Prefiltered datasets/wildchat.jsonl" \
  --output-dir data_minimization_results

**Dataset format (per line JSON):**

```json
{
  "index": 1,
  "conversation_hash": "abc123",
  "user_message": "...",
  "pii_dict": {"Alice": "PERSON", "NYC": "GPE"},
  "variants_map": {"Alice": ["Alicia"], "NYC": ["New York"]},
  "redact_map": {"Alice": "[PERSON1]", "NYC": "[GPE1]"},
  "abstract_map": {"Alice": "an individual", "NYC": "a large US city"}
}
```

---

## Outputs

* `data_minimization_results/dm_of_<MODEL>/auto_masked_messages_<TIME>.jsonl`

  * `masked_message`
  * `transformation` (what we did per PII)
  * `replacement_mapping`
  * `transformation_stats`, timings, call counts
* `.../_summary.json` — log with cost/latency and decisions

---

## Note on datasets

* **WildChat / open‑ended**: keep the LLM utility judge (already in the script).
* **MedQA / close‑ended**: do not use an LLM judge. Compare answers directly to gold labels (exact match or EM/F1). This change is limited to the `utility_eval(...)` path.



---

## Heads-up 

Our sensitivity model is deployed on a private cloud. As a result, this pipeline may not fully run end-to-end in your environment. FYI. Please contact us if needed.


## Other files

human_annotation_vs_o3mini.jsonl includes human annotator's choices on messages that they think are more privacy preserving.
