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
* `run_pipeline.py` — main runner script.
* `Prefiltered datasets/` — curated JSONL datasets used as `--dataset` inputs in our pipeline:
  * `wildchat.jsonl`
  * `ShareGPT.jsonl`
  * `medQA.jsonl`
  * `casehold.jsonl`
* `human_annotation_vs_o3mini.jsonl` — an example analysis file showing how we compare model decisions against a reference teacher (o3mini) for evaluation/debugging (not the released human ground-truth dataset).
* `human_labeled_datasets/` - released datasets
* `README.md` — this document.

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

## Note on datasets used for evaluation

* **WildChat / open‑ended**: keep the LLM utility judge (already in the script).
* **MedQA / close‑ended**: do not use an LLM judge. Compare answers directly to gold labels (exact match or EM/F1). This change is limited to the `utility_eval(...)` path.

---

## Heads-up (sensitivity comparator)

The [sensitivity comparator](https://huggingface.co/seazon96/privacy-comparator) used in this pipeline is now publicly available:

This model can be used as a drop-in replacement for the internal comparator originally deployed in our experiments.

Note that certain infrastructure details (e.g., original deployment setup) are not included, but the released model enables reproducible sensitivity ranking for the pipeline.

---

## Human annotation files

This repo includes two JSONL files related to human preference annotations over A/B privacy-variant pairs:

- `human_labeled_datasets/question-submissions_data_with_messages_anonymized.jsonl` (**released dataset; full votes**)  
  Contains the full anonymized participant vote dictionary per pair:
  `answers` = {`participant_1`: "A"/"B"/"SAME", ...}, along with `consensus`, `consensus_ratio`, `message_A`, and `message_B`.

- `human_annotation_vs_o3mini.jsonl` (**analysis/example file; consensus + teacher prediction**)  
  Contains only the aggregated human label (`consensus`, `consensus_ratio`) plus a reference model’s prediction (e.g., `gpt_response`), an optional explanation (`reason`), and correctness (`is_correct`).  
  This file is provided as an example of how we evaluate model/teacher alignment against human consensus; it is not the full vote-level dataset.

---

### Human-labeled pairwise privacy-preference dataset (released)

We release a human-labeled dataset of **150 A/B pairs** constructed from ShareGPT-derived prompts under `human_labeled_datasets/`.  See `human_labeled_datasets/DATASET_CARD.md` for details.
For each pair, annotators choose which message variant is more privacy-preserving (**A**, **B**, or **SAME**).  
Each pair has **at least 5** votes from qualified participants (52 unique participants total). We include:
- anonymized per-participant votes (`participant_1`, `participant_2`, ...)
- `consensus` and `consensus_ratio`
- `message_A` and `message_B`

The dataset file will be added to this repository (JSONL; one example per line).
For full details on pair construction and annotation procedures, please refer to the paper.

---

## License
This repository is released under the Apache 2.0 License.
