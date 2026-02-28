# Human-Labeled Privacy Preference Dataset

## Overview

This dataset contains **150 human-labeled A/B comparisons** constructed from ShareGPT-derived prompts.

Each example presents two variants of the same prompt (Message A and Message B), where sensitive information has been transformed differently (e.g., retained, abstracted, or redacted). Human annotators were asked:

> Which version is more privacy-preserving?

Possible responses:

- `A`
- `B`
- `SAME`

Each pair was annotated by **at least 5 qualified participants**  
(52 unique participants in total).

The dataset includes:
- anonymized participant votes
- majority consensus label
- consensus ratio

This dataset is released to support research on privacy-preserving prompting and preference learning for data minimization.

---

## File
question-submissions_data_with_messages_anonymized.jsonl

Format: JSONL (one example per line)

---

## Fields

Each entry contains:

| Field | Description |
|------|-------------|
| `survey_id` | Survey batch identifier |
| `conversation_id` | Source prompt identifier |
| `pair_index` | Pair index within survey |
| `answers` | Dictionary of anonymized participant votes |
| `consensus` | Majority preference label |
| `consensus_ratio` | Fraction of participants agreeing with consensus |
| `message_A` | Variant A of the prompt |
| `message_B` | Variant B of the prompt |

Example:

```json
{
  "survey_id": "survey_1",
  "conversation_id": "fCldHm3",
  "pair_index": 3,
  "answers": {
    "participant_1": "A",
    "participant_2": "SAME",
    "participant_3": "A",
    "participant_4": "A",
    "participant_5": "A"
  },
  "consensus": "A",
  "consensus_ratio": 0.8,
  "message_A": "...",
  "message_B": "..."
}
```

## Annotation Procedure

Prompts were sampled from ShareGPT under filtering criteria (e.g., English, task-oriented, containing multiple sensitive spans). For more info, please refer to our paper.

Each prompt was transformed into two variants with different privacy treatments.

Participants selected which version revealed less sensitive information.

All annotations were collected via survey batches.

## Anonymization
Participant identifiers have been replaced with stable pseudonyms:
```participant_1, participant_2, ...```
No Prolific IDs or personal identifiers are included.

## Citation
If you use this dataset, please cite the associated paper.