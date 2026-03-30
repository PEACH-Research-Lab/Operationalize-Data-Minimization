#!/usr/bin/env python3
"""Build website-ready JSON files for the project explorer.

This script standardizes four datasets into one JSON file per dataset.
It uses prediction files as the set of valid records, then joins:

- oracle data minimization results from data_minimization_results/dm_of_{model}
- prediction results from prediction_runs_{dataset}_{model}.jsonl

Output files are written to website/data/{dataset}.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
WEBSITE_DIR = ROOT / "website"
OUTPUT_DIR = WEBSITE_DIR / "data"

DATASET_CONFIGS = {
    "ShareGPT": {
        "dir_name": "ShareGPT",
        "oracle_index_key": "conversation_id",
        "prediction_index_key": "conversation_id",
        "display_id_label": "conversation_id",
        "original_text_field": "original_message",
        "masked_text_field": "masked_message",
        "choices_field": None,
        "correct_choice_field": None,
        "prediction_prefix": "prediction_runs_ShareGPT_",
    },
    "wildchat": {
        "dir_name": "wildchat",
        "oracle_index_key": "conversation_hash",
        "prediction_index_key": "conversation_hash",
        "display_id_label": "conversation_hash",
        "original_text_field": "original_message",
        "masked_text_field": "masked_message",
        "choices_field": None,
        "correct_choice_field": None,
        "prediction_prefix": "prediction_runs_wildchat_",
    },
    "medQA": {
        "dir_name": "medQA",
        "oracle_index_key": "index",
        "prediction_index_key": "question_id",
        "display_id_label": "index",
        "original_text_field": "original_question",
        "masked_text_field": "masked_question",
        "choices_field": "choices",
        "correct_choice_field": "correct_choice",
        "prediction_prefix": "prediction_runs_medQA_",
    },
    "casehold": {
        "dir_name": "casehold",
        "oracle_index_key": "index",
        "prediction_index_key": "index",
        "display_id_label": "index",
        "original_text_field": "original_question",
        "masked_text_field": "masked_question",
        "choices_field": "choices",
        "correct_choice_field": "correct_choice",
        "prediction_prefix": "prediction_runs_casehold_",
    },
}

MODEL_ORDER = [
    "claude-3-7-sonnet-20250219",
    "claude-sonnet-4-20250514",
    "gpt-4.1-nano",
    "gpt-4.1",
    "gpt-5",
    "lgai_exaone-deep-32b",
    "mistralai_mistral-small-3.1-24b-instruct",
    "qwen_qwen-2.5-7b-instruct",
    "local_qwen2.5-0.5b-instruct",
]

PREDICTION_UNAVAILABLE_REASONS = {
    "local_qwen2.5-0.5b-instruct": (
        "Prediction unavailable: this model could not reliably perform the "
        "prediction task."
    )
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dir_name: str
    oracle_index_key: str
    prediction_index_key: str
    display_id_label: str
    original_text_field: str
    masked_text_field: str
    choices_field: str | None
    correct_choice_field: str | None
    prediction_prefix: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc
    return rows


def choose_oracle_file(dm_dir: Path) -> Path:
    candidates = sorted(dm_dir.glob("auto_masked_messages*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No auto_masked_messages*.jsonl found in {dm_dir}")

    no_variants = [path for path in candidates if path.name.endswith("_no_variants.jsonl")]
    if no_variants:
        return no_variants[0]

    return candidates[0]


def prediction_file_for_model(dataset_dir: Path, prefix: str, model: str) -> Path | None:
    path = dataset_dir / f"{prefix}{model}.jsonl"
    if path.exists():
        return path
    return None


def normalize_choice_index(dataset_name: str, raw_index: Any) -> int | None:
    if raw_index is None:
        return None

    try:
        numeric = int(raw_index)
    except (TypeError, ValueError):
        return None

    if dataset_name == "casehold":
        return numeric + 1
    return numeric


def build_original_block(
    dataset_name: str, config: DatasetConfig, oracle_row: dict[str, Any]
) -> dict[str, Any]:
    original = {
        "text": oracle_row.get(config.original_text_field),
    }

    if config.choices_field:
        original["choices"] = oracle_row.get(config.choices_field, [])
    if config.correct_choice_field:
        original["correct_choice"] = normalize_choice_index(
            dataset_name,
            oracle_row.get(config.correct_choice_field),
        )

    return original


def build_result_block(
    row: dict[str, Any], *, masked_text_field: str, is_prediction: bool
) -> dict[str, Any]:
    pii_dict = row.get("pii_dict") or {}
    raw_transformation = row.get("corrected_transformation") or row.get("transformation") or {}
    transformation = {
        span: raw_transformation.get(span, "retain")
        for span in pii_dict
    }

    block = {
        "masked_text": row.get("masked_message") or row.get(masked_text_field),
        "transformation": transformation,
        "transformation_stats": row.get("transformation_stats"),
        "pii_dict": pii_dict,
        "replacement_map": row.get("replacement_map") or row.get("replacement_mapping"),
        "redact_map": row.get("redact_map"),
        "abstract_map": row.get("abstract_map"),
    }

    if is_prediction:
        block["predict_success"] = row.get("predict_success")
        block["explanation"] = row.get("explanation")
        block["errors"] = row.get("errors")

    return block


def build_record_id(
    dataset_name: str,
    oracle_index_value: Any,
    prediction_index_value: Any,
    index_value: Any,
) -> str:
    return f"{dataset_name}::{oracle_index_value}::{prediction_index_value}::{index_value}"


def build_dataset(config: DatasetConfig) -> dict[str, Any]:
    dataset_dir = ROOT / config.dir_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_dir}")

    predictions_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    oracles_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    all_record_keys: list[str] = []
    seen_record_keys: set[str] = set()

    for model in MODEL_ORDER:
        dm_dir = dataset_dir / "data_minimization_results" / f"dm_of_{model}"
        oracle_file = choose_oracle_file(dm_dir)
        oracle_rows = load_jsonl(oracle_file)

        oracle_map: dict[str, dict[str, Any]] = {}
        for row in oracle_rows:
            key_value = row.get(config.oracle_index_key)
            if key_value is None:
                raise KeyError(
                    f"{oracle_file} row missing oracle index key {config.oracle_index_key}"
                )
            key = str(key_value)
            if key in oracle_map:
                raise ValueError(f"Duplicate oracle key {key} in {oracle_file}")
            oracle_map[key] = row

        oracles_by_model[model] = oracle_map

        prediction_path = prediction_file_for_model(
            dataset_dir,
            config.prediction_prefix,
            model,
        )
        if prediction_path is None:
            predictions_by_model[model] = {}
            continue

        prediction_rows = load_jsonl(prediction_path)
        prediction_map: dict[str, dict[str, Any]] = {}
        for row in prediction_rows:
            key_value = row.get(config.prediction_index_key)
            if key_value is None:
                raise KeyError(
                    f"{prediction_path} row missing prediction index key "
                    f"{config.prediction_index_key}"
                )
            key = str(key_value)
            if key in prediction_map:
                raise ValueError(f"Duplicate prediction key {key} in {prediction_path}")
            prediction_map[key] = row
            if key not in seen_record_keys:
                seen_record_keys.add(key)
                all_record_keys.append(key)

        predictions_by_model[model] = prediction_map

    records: list[dict[str, Any]] = []
    for key in all_record_keys:
        base_prediction_row = None
        for model in MODEL_ORDER:
            row = predictions_by_model[model].get(key)
            if row is not None:
                base_prediction_row = row
                break

        if base_prediction_row is None:
            continue

        base_oracle_row = None
        for model in MODEL_ORDER:
            oracle_lookup_key = str(base_prediction_row.get(config.prediction_index_key))
            row = oracles_by_model[model].get(oracle_lookup_key)
            if row is not None:
                base_oracle_row = row
                break

        if base_oracle_row is None:
            raise KeyError(
                f"No oracle row found for dataset={config.name}, key={key} "
                f"using oracle key {config.oracle_index_key}"
            )

        oracle_index_value = base_oracle_row.get(config.oracle_index_key)
        prediction_index_value = base_prediction_row.get(config.prediction_index_key)
        if config.name == "medQA":
            index_value = base_oracle_row.get("index")
        else:
            index_value = (
                base_oracle_row.get("index")
                if "index" in base_oracle_row
                else base_prediction_row.get("index")
            )

        record = {
            "record_id": build_record_id(
                config.name,
                oracle_index_value,
                prediction_index_value,
                index_value,
            ),
            "dataset": config.name,
            "oracle_index_key": config.oracle_index_key,
            "prediction_index_key": config.prediction_index_key,
            "oracle_index_value": oracle_index_value,
            "prediction_index_value": prediction_index_value,
            "index": index_value,
            "original": build_original_block(config.name, config, base_oracle_row),
            "models": {},
        }

        for model in MODEL_ORDER:
            oracle_lookup_key = str(prediction_index_value)
            oracle_row = oracles_by_model[model].get(oracle_lookup_key)
            if oracle_row is None:
                raise KeyError(
                    f"Missing oracle join for dataset={config.name}, model={model}, "
                    f"key={oracle_lookup_key}"
                )

            prediction_row = predictions_by_model[model].get(str(prediction_index_value))
            model_block: dict[str, Any] = {
                "oracle": build_result_block(
                    oracle_row,
                    masked_text_field=config.masked_text_field,
                    is_prediction=False,
                ),
                "prediction": None,
            }

            if prediction_row is None:
                model_block["prediction_unavailable_reason"] = (
                    PREDICTION_UNAVAILABLE_REASONS.get(model)
                    or "Prediction unavailable for this model."
                )
            else:
                model_block["prediction"] = build_result_block(
                    prediction_row,
                    masked_text_field=config.masked_text_field,
                    is_prediction=True,
                )

            record["models"][model] = model_block

        records.append(record)

    return {
        "dataset": config.name,
        "oracle_index_key": config.oracle_index_key,
        "prediction_index_key": config.prediction_index_key,
        "models": MODEL_ORDER,
        "record_count": len(records),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=list(DATASET_CONFIGS.keys()),
        default=list(DATASET_CONFIGS.keys()),
        help="Datasets to export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        config = DatasetConfig(name=dataset_name, **DATASET_CONFIGS[dataset_name])
        payload = build_dataset(config)
        output_path = OUTPUT_DIR / f"{dataset_name}.json"
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"Wrote {output_path} ({payload['record_count']} records)")


if __name__ == "__main__":
    main()
