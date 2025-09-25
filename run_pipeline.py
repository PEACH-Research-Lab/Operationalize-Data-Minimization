#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import heapq
import warnings
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Tuple, Union, List, Optional
import random
import faulthandler; faulthandler.enable()

import builtins
import traceback

from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

DEBUG = os.getenv("DEBUG", "1") == "1"

def dbg(prefix: str, *args):
    if DEBUG:
        print(f"[{prefix}]", *args, flush=True)

def log_exc(prefix: str):
    import sys, traceback
    tb = "".join(traceback.format_exception(*sys.exc_info()))
    print(f"[{prefix}] EXCEPTION:\n{tb}", flush=True)



from openai import OpenAI, DefaultHttpxClient
from fireworks.client import Fireworks
from together import Together
import anthropic

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
FIREWORKS_KEY = os.getenv("FIREWORKS_API_KEY")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_DETECTION_MODEL = "gpt-4o"
DEFAULT_ABSTRACTION_MODEL = "gpt-4o"
DEFAULT_SENSITIVITY_MODEL = "provider/sensitivity-model-id" # anonymized
DEFAULT_LLM_RESPONSE_MODEL = "" # Select your LLM response model
DEFAULT_UTILITY_MODEL = "gpt-4o"

ABSTRACTION_SYS_PROMPT = """Rewrite the text to abstract the protected information, without changing other parts. Please output in JSON format only.

For example:

Input: 
<Text>I graduated from CMU, and I earn a six-figure salary. Today a lady from China said...</Text><ProtectedInformation1>CMU</ProtectedInformation1><ProtectedInformation2>China</ProtectedInformation2>

Output:
{"results": [{"protected": "CMU", "abstracted": "a prestigious American university"}, {"protected": "China", "abstracted": "A country in Asia"}]}"""



class Logger:
    def __init__(self, index, sensitivity_model, llm_response_model, utility_model):
        self.id = index
        self.phase1_counter = 0
        self.phase2_counter = 0
        self.phase1_counters = {}
        self.phase2_counters = {}
        self.data = {
            "message_index": index,
            "original_message": "",
            "original_response": "",
            "sensitivity_model": sensitivity_model,
            "llm_response_model": llm_response_model,
            "utility_model": utility_model,
            "detect+filter": {},
            "pii_dict": {},
            "redact_map": {},
            "abstract_map": {},
            "abstraction_related": {},
            "phase1": [],
            "phase1_summary": {},
            "phase2": [],
            "final_result": None,
            "transformation": {},
            "duration_sec": 0.0,
            "utility_evaluate_calls": 0,
            "sensitivity_cache_stats": {},
            "total_cost": 0,
            "cost_breakdown": {
                "pii_detection": 0,
                "pii_filtering": 0,
                "abstraction": 0,
                "llm_response": 0,
                "sensitivity_compare": 0,
                "utility_eval": 0
            },
        }
        self.data["latency_summary"] = {}
        self._latency_totals = {
            "pii_detection": 0.0,
            "pii_filtering": 0.0,
            "abstraction": 0.0,
            "llm_response": 0.0,
            "sensitivity_compare": 0.0,
            "utility_eval": 0.0,
        }
        self._latency_counts = {k: 0 for k in self._latency_totals}
        self.pq_state = set()
        self.TOKEN_PRICING = {
            "local:qwen2.5-0.5b-instruct": {"input_per_million": 0.00, "output_per_million": 0.00},
            "qwen/qwen-2.5-7b-instruct": {"input_per_million": 0.04, "output_per_million": 0.1},
            "mistralai/mistral-small-3.1-24b-instruct": {"input_per_million": 0.02, "output_per_million": 0.08},
            "private/sensitivity-model": {"input_per_million": 0.00, "cached_input_per_million": 0.00, "output_per_million": 0.00},
            "lgai/exaone-deep-32b": {"input_per_million": 0.00, "cached_input_per_million": 0.00, "output_per_million": 0.00},
            "claude-3-7-sonnet-20250219": {"input_per_million": 3, "output_per_million": 3.75},
            "claude-sonnet-4-20250514": {"input_per_million": 3, "output_per_million": 3.75},
            "gpt-5-2025-08-07": {"input_per_million": 1.25, "cached_input_per_million": 0.125, "output_per_million": 10.00},
            "gpt-4.1-nano-2025-04-14": {"input_per_million": 0.10, "cached_input_per_million": 0.025, "output_per_million": 0.40},
            "gpt-4.1-2025-04-14": {"input_per_million": 2.00, "cached_input_per_million": 0.5, "output_per_million": 8.00},
            "gpt-4o-2024-08-06": {"input_per_million": 2.50, "cached_input_per_million": 1.25, "output_per_million": 10.00},
            "gpt-4o-mini-2024-07-18": {"input_per_million": 0.15, "cached_input_per_million": 0.075, "output_per_million": 0.60},
            "o1-2024-12-17": {"input_per_million": 15.00, "cached_input_per_million": 7.50, "output_per_million": 60.00},
            "o3-2025-04-16": {"input_per_million": 2.00, "cached_input_per_million": 0.50, "output_per_million": 8.00},
            "o3-mini-2025-01-31": {"input_per_million": 1.10, "cached_input_per_million": 0.55, "output_per_million": 4.40},
        }

    def add_latency(self, category: str, seconds: float):
        if category not in self._latency_totals:
            self._latency_totals[category] = 0.0
            self._latency_counts[category] = 0
        self._latency_totals[category] += float(seconds or 0.0)
        self._latency_counts[category] += 1

    def build_latency_summary(self):
        summary = {}
        for k in self._latency_totals:
            total = round(self._latency_totals[k], 6)
            count = self._latency_counts[k]
            avg = round(total / count, 6) if count > 0 else 0.0
            summary[k] = {"count": count, "total_sec": total, "avg_sec": avg}
        self.data["latency_summary"] = summary

    def _log_action(self, phase, action_name, extra_data):
        if phase not in ("phase1", "phase2"):
            raise ValueError(f"Unknown phase: {phase}")
        action_index = self.phase1_counter if phase == "phase1" else self.phase2_counter
        action_type_index = self._next_action_type_index(phase, action_name)
        entry = {"action_index": action_index,"action_type_index": action_type_index,"action": action_name,**extra_data}
        self.data[phase].append(entry)
        if phase == "phase1":
            self.phase1_counter += 1
        else:
            self.phase2_counter += 1
        return entry

    def _next_action_type_index(self, phase, action_type):
        counters = self.phase1_counters if phase == "phase1" else self.phase2_counters
        counters[action_type] = counters.get(action_type, 0) + 1
        return counters[action_type]

    def _calculate_cost(self, model, input_tokens, output_tokens, cached=False, category=None):
        try:
            if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
                raise TypeError(f"Token values must be numeric. Got input={input_tokens}, output={output_tokens}")
            pricing = self.TOKEN_PRICING.get(model)
            if pricing is None:
                raise ValueError(f"Model '{model}' not found in TOKEN_PRICING")
            per_million = 1_000_000
            input_key = "cached_input_per_million" if cached else "input_per_million"
            input_rate = pricing.get(input_key, 0)
            output_rate = pricing.get("output_per_million", 0)
            cost = (input_rate * input_tokens + output_rate * output_tokens) / per_million
            self.data["total_cost"] += cost
            if category and category in self.data["cost_breakdown"]:
                self.data["cost_breakdown"][category] += cost
            return round(cost, 6)
        except Exception as e:
            print(f"⚠️ [WARNING] Cost calculation failed for model={model}. Reason: {e}")
            return 0.0

    def log_original(self, original_message, original_response):
        self.data["original_message"] = original_message
        self.data["original_response"] = original_response

    def detect_then_filter(
        self, 
        detection_response, 
        detection_model, 
        detection_input_token,
        detection_output_token,
        filtered_response,
        filtered_model,
        filtered_input_token,
        filtered_output_token,
        detection_latency,
        filter_latency,
    ):
        detection_cost = self._calculate_cost(detection_model, detection_input_token, detection_output_token, category="pii_detection")
        filter_cost = self._calculate_cost(filtered_model, filtered_input_token, filtered_output_token, category="pii_filtering")
        
        self.data["detect+filter"] = {
            "detection_response": detection_response,
            "detection_model":detection_model,
            "detection_input_token": detection_input_token,
            "detection_output_token": detection_output_token,
            "filtered_response":filtered_response,
            "filtered_model":filtered_model,
            "filtered_input_token":filtered_input_token,
            "filtered_output_token":filtered_output_token,
            "detection_cost": detection_cost, 
            "filter_cost": filter_cost
        }
        if detection_latency:
            self.add_latency("pii_detection", detection_latency)
        if filter_latency:
            self.add_latency("pii_filtering", filter_latency)

    def log_abstraction_related(self, model, input_tokens, output_tokens, abstraction_time):
        cost = self._calculate_cost(model, input_tokens, output_tokens, category="abstraction")
        self.data["abstraction_related"] = {
            "model": model,
            "input_tokens":input_tokens,
            "output_tokens": output_tokens,
            "cost":cost,
            "time": abstraction_time
        }
        if abstraction_time:
            self.add_latency("abstraction", abstraction_time)
        
    def log_empty_pii(self, message_index, pii_dict, original_message):
        self.data["message_index"] = message_index
        self.data["pii_dict"] = pii_dict
        self.data["original_message"] = original_message
        
    def set_initial_metadata(self, pii_dict, redact_map, abstract_map):
        self.data["pii_dict"] = pii_dict
        self.data["redact_map"] = redact_map
        self.data["abstract_map"] = abstract_map

    def finalize_phase1(self, frozen, redact_failed, abstract_failed, duration):
        summary = {
            "phase1_completion_time": round(duration, 2),
            "phase1_frozen": sorted(frozen),
            "phase1_redact_failed": sorted(redact_failed),
            "phase1_abstract_failed": sorted(abstract_failed)
        }
        self.data["phase1_summary"] = summary

    def log_phase2_initial(self, transformation):
        self.pq_state = {frozenset(transformation.items())}
        extra_data = {"priority_queue_set": [dict(t) for t in self.pq_state]}
        return self._log_action("phase2", "initial_transformation", extra_data)

    def log_phase2_pop(self, popped_transformation, replacement_mapping, popped_message):
        self.pq_state.discard(frozenset(popped_transformation.items()))
        extra_data = {
            "popped_transformation": str(popped_transformation),
            "poppsed_mapping": str(replacement_mapping),
            "popped_message": popped_message,
            "priority_queue_set": [dict(t) for t in self.pq_state]
        }
        return self._log_action("phase2", "pop", extra_data)

    def log_phase2_push(self, transformation):
        self.pq_state.add(frozenset(transformation.items()))
        extra_data = {"transformation": str(transformation),"priority_queue_set": [dict(t) for t in self.pq_state]}
        return self._log_action("phase2", "push", extra_data)

    def log_phase2_sensitivity_cache_hit(self, sensitivity_cache_keys, current_key, cache_result):
        extra_data = {"cache_keys": list(sensitivity_cache_keys),"current_key": current_key,"cache_result": cache_result}
        return self._log_action("phase2", "sensitivity_cache_hit", extra_data)

    def log_phase2_sensitivity_compare(self, gpt_result, output_message, input_data, 
                                       model, raw_response, input_tokens, output_tokens, time_elapsed):
        cost = self._calculate_cost(model, input_tokens, output_tokens, category="sensitivity_compare")
        extra_data = {
            "result": gpt_result,
            "output": output_message,
            "input": input_data,
            "model": model,
            "raw_response": raw_response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "time_taken": time_elapsed,
            "cost": cost
        }
        if time_elapsed:
            self.add_latency("sensitivity_compare", time_elapsed)
        return self._log_action("phase2", "sensitivity_compare", extra_data)

    def log_phase_llm_response(self, phase, replacement_mapping, output,
                       model=None, input_text=None, input_tokens=None, output_tokens=None, time_taken=None):
        cost = self._calculate_cost(model, input_tokens, output_tokens, category="llm_response")
        extra_data = {"replacement_mapping": replacement_mapping,"output": output}
        if model is not None:
            extra_data.update({
                "model": model,
                "input_text": input_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "time_taken": time_taken,
                "cost": cost
            })
        if time_taken:
            self.add_latency("llm_response", time_taken)
        return self._log_action(phase, "llm_response", extra_data)

    def log_phase_utility_check(self, phase, response_of_masked_message, restored_response, original_response, passed, model, input_tokens, output_tokens, time_taken, raw_response):
        cost = self._calculate_cost(model, input_tokens, output_tokens, category="utility_eval")
        extra_data = {
            "passed": passed,
            "response_of_masked_message": response_of_masked_message,
            "restored_response":restored_response,
            "original_response": original_response,
            "raw_response": raw_response
        }
        if time_taken:
            self.add_latency("utility_eval", time_taken)
        return self._log_action(phase, "utility_check", extra_data)

    def log_phase_utility_cache_hit(self, phase, replacement_mapping, cache_result):
        extra_data = {"replacement_mapping": json.dumps(replacement_mapping),"cache_result": cache_result}
        return self._log_action(phase, "utility_cache_hit", extra_data)

    def log_final(self, transformation, duration_sec, utility_evaluate_calls):
        self.data["final_result"] = "PASS" if transformation else "FAIL"
        self.data["transformation"] = transformation
        self.data["duration_sec"] = round(duration_sec, 2)
        self.data["utility_evaluate_calls"] = utility_evaluate_calls
        cost_summary = {
            "total_cost": round(self.data["total_cost"], 6),
            "breakdown": {k: round(v, 6) for k, v in self.data["cost_breakdown"].items()}
        }
        self.build_latency_summary()
        print(f"[FINAL] result={self.data['final_result']}, utility_calls={utility_evaluate_calls}, duration={duration_sec:.2f}s, cost_summary={cost_summary}")

    def save(self, path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.data, f, indent=2)
            print(f"[SAVE] Logger data written to {path}")
        except TypeError as e:
            print(f"❌ [ERROR] Failed to save logger data to {path}: {e}")



def build_prompt_for_abstract(message: str, pii_list: List[str]) -> str:
    tagged = f"<Text>{message}</Text>"
    for i, p in enumerate(pii_list, 1):
        tagged += f"<ProtectedInformation{i}>{p}</ProtectedInformation{i}>"
    return tagged

def get_abstraction_map(message: str, pii_list: List[str], logger: Logger):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if not pii_list: return {}
    tagged = build_prompt_for_abstract(message, pii_list)
    try:
        start = time.time()
        resp = client.chat.completions.create(
            model=DEFAULT_ABSTRACTION_MODEL,
            temperature=0,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":ABSTRACTION_SYS_PROMPT},{"role":"user","content":tagged}]
        )
        elapsed = time.time() - start
        raw = resp.choices[0].message.content.strip()
        parsed = json.loads(raw)
        result = {r["protected"]: r["abstracted"] for r in parsed.get("results", [])}
        logger.log_abstraction_related(resp.model, resp.usage.prompt_tokens, resp.usage.completion_tokens, elapsed)
        return result
    except Exception as e:
        print("❌ get_abstraction_map error:", e)
        traceback.print_exc()
        return {}


def build_fuzzy_regex(pii_text):
    tokens = pii_text.split()
    quote = r"[\"'“”‘’]?"
    sep = r"[ \-_.\u00A0]*"
    if len(tokens) == 1:
        body = rf"{quote}{re.escape(tokens[0])}{quote}"
    else:
        body = sep.join(rf"{quote}{re.escape(t)}{quote}" for t in tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)

def build_replacement_map(transformation: dict, global_redact_map: dict, global_abstract_map: dict, variants_map: dict):
    replacement_map = {}
    for canonical, method in (transformation or {}).items():
        if method not in ("redact", "abstract"):
            continue
        if method == "redact":
            repl = global_redact_map.get(canonical, "[REDACTED]")
        else:
            repl = global_abstract_map.get(canonical, "[ABSTRACT]")
        keys = []
        if variants_map and canonical in variants_map and variants_map[canonical]:
            seen = set()
            for v in variants_map[canonical]:
                if v and v not in seen:
                    seen.add(v)
                    keys.append(v)
        else:
            keys = [canonical]
        for k in keys:
            replacement_map[k] = repl
    return replacement_map

def apply_replacement(replacement_map, original_message):
    replaced = original_message
    for pii_or_variant in sorted(replacement_map.keys(), key=len, reverse=True):
        replacement = replacement_map[pii_or_variant]
        pat = build_fuzzy_regex(pii_or_variant)
        replaced = pat.sub(replacement, replaced)
    return replaced

def restore_response(response, replacement_map):
    restored = response
    for entity, replaced_form in replacement_map.items():
        pat = build_fuzzy_regex(replaced_form)
        restored = pat.sub(entity, restored)
    return restored

def apply_transformation(message, replacement_mapping):
    return apply_replacement(replacement_mapping, message)

def compute_transformation_stats(pii_dict: dict[str, str], transformation: dict[str, str]) -> dict:
    per_type: dict[str, dict[str, int]] = {}
    for text, etype in pii_dict.items():
        method = transformation.get(text, "retain")
        if method not in ("redact", "abstract"):
            method = "retain"
        bucket = per_type.setdefault(etype, {"redact": 0, "abstract": 0, "retain": 0})
        bucket[method] += 1
    totals = {"redact": 0, "abstract": 0, "retain": 0}
    for d in per_type.values():
        for k in totals:
            totals[k] += d[k]
    return {"per_type": per_type, "totals": totals}

# Local Qwen (0.5B Instruct) support
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

_LOCAL_QWEN_CACHE = {}

def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_local_qwen(model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"):
    """Lazy-load and cache local Qwen model and tokenizer."""
    if model_id in _LOCAL_QWEN_CACHE:
        return _LOCAL_QWEN_CACHE[model_id]

    device = _pick_device()
    if device == "mps":
        torch_dtype = torch.bfloat16
    elif device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map="auto" if device in {"cuda", "mps"} else None,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    _LOCAL_QWEN_CACHE[model_id] = (tokenizer, model, device)
    return tokenizer, model, device

@torch.inference_mode()
def local_qwen_chat(
    messages,
    *,
    max_new_tokens: int = 512,
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct",
) -> str:
    tokenizer, model, device = get_local_qwen(model_id)

    gc = model.generation_config
    gc.do_sample = False
    gc.max_new_tokens = 512
    model.generation_config = gc

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    if device in {"cuda", "mps"}:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    out = model.generate(
        **inputs,
        return_dict_in_generate=True,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    seq = out.sequences[0]
    gen_only = seq[inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_only, skip_special_tokens=True).strip()



def _norm_usage(usage_obj: Any) -> Dict[str, int]:
    if not usage_obj:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    pt = getattr(usage_obj, "prompt_tokens", None)
    ct = getattr(usage_obj, "completion_tokens", None)
    tt = getattr(usage_obj, "total_tokens", None)
    if pt is not None or ct is not None or tt is not None:
        pt = int(pt or 0); ct = int(ct or 0)
        return {"prompt_tokens": pt,"completion_tokens": ct,"total_tokens": int(tt) if tt is not None else pt + ct}
    it = getattr(usage_obj, "input_tokens", None)
    ot = getattr(usage_obj, "output_tokens", None)
    if it is not None or ot is not None:
        it = int(it or 0); ot = int(ot or 0)
        return {"prompt_tokens": it,"completion_tokens": ot,"total_tokens": it + ot}
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def gpt_llm_response(user_message: str, llm_response_model: str, OPENAI_KEY: str, OPENROUTER_KEY: str, TOGETHER_KEY: Optional[str], ANTHROPIC_KEY: Optional[str], return_raw: bool = False):
    m = (llm_response_model or "").lower().strip()
    if not m:
        raise ValueError("llm_response_model not defined")
    common_messages = [{"role": "user", "content": f"{user_message}"}]

    if m.startswith("gpt"):
        client = OpenAI(api_key=OPENAI_KEY)
        if m.startswith("gpt-5"):
            response = client.chat.completions.create(model=llm_response_model, messages=common_messages)
        else:
            response = client.chat.completions.create(model=llm_response_model, temperature=0, top_p=1.0, messages=common_messages)
        raw_response = (response.choices[0].message.content or "").strip()
        usage = _norm_usage(getattr(response, "usage", None))
        model = getattr(response, "model", llm_response_model)

    elif m.startswith("qwen") or m.startswith("mistral"):
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY)
        response = client.chat.completions.create(model=llm_response_model, temperature=0, top_p=1.0, messages=common_messages)
        raw_response = (response.choices[0].message.content or "").strip()
        usage = _norm_usage(getattr(response, "usage", None))
        model = getattr(response, "model", llm_response_model)

    elif m.startswith("lgai"):
        client = Together(api_key=TOGETHER_KEY)
        backoff_base = 0.6
        jitter = 0.2
        wait = backoff_base * (1 + jitter * random.random())
        time.sleep(wait)

        response = client.chat.completions.create(
            model=llm_response_model,
            temperature=0,
            top_p=1.0,
            messages=common_messages,
        )
        raw_response = (response.choices[0].message.content or "").strip()
        usage = _norm_usage(getattr(response, "usage", None))
        model = getattr(response, "model", llm_response_model)

    elif m.startswith("claude"):
        if anthropic is None:
            raise RuntimeError("anthropic SDK not installed")
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        time.sleep(random.uniform(0.2,1.0))
        response = client.messages.create(model=llm_response_model, temperature=0, top_p=1.0, max_tokens=1000, messages=[{"role": "user", "content": user_message}])
        raw_response = "".join(block.text for block in getattr(response, "content", []) if getattr(block, "type", None) == "text" and hasattr(block, "text")).strip()
        usage = _norm_usage(getattr(response, "usage", None))
        model = getattr(response, "model", llm_response_model)
    
    elif m in {"local:qwen2.5-0.5b-instruct", "local:qwen-0.5b", "local-qwen-0.5b"}:
        text = local_qwen_chat(
            common_messages,
            max_new_tokens=512,
        )
        if return_raw:
            return text, "local:qwen2.5-0.5b-instruct", {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
            }
        return text
    else:
        raise ValueError(f"Unknown LLM response model: {llm_response_model}")

    if return_raw:
        return raw_response, model, usage
    return raw_response


def extract_json_response(text):
    try:
        matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        for m in matches:
            try:
                parsed = json.loads(m)
                if isinstance(parsed, dict) and "response" in parsed and "reason" in parsed:
                    return parsed
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
    return None


def gpt_sensitivity_with_retry(
    message_A, 
    message_B, 
    sensitivity_model: str, 
    fw_client: Fireworks, 
    return_raw=False, 
    max_attempts: int = 3,
    backoff_base: float = 0.4,
    ):
    sensitivity_system_prompt = """In the following messages, certain spans are annotated to indicate how sensitive information has been treated:

    - <redacted>: Redacted content, replaced with a placeholder such as [NAME1], [GEOLOCATION2], etc.
    - <abstracted>: Abstracted content, where specific PII was generalized (e.g., “a country in Asia”).
    - <retained>: Sensitive information that was detected but left unchanged.

    We use two privacy-preserving techniques:
    - Redaction: replaces PII with type+index placeholders.
    - Abstraction: generalizes PII content.

    These techniques apply to different PII types like names, dates, organizations, locations, etc.

    Please evaluate: which version offers better protection of personal or sensitive information?

    Respond in JSON format with a brief explanation:
    {"reason": YOUR_REASON, "response": "A" | "B" | "SAME"}"""
    user_prompt = f"""A:\n{message_A}\n\nB:\n{message_B}
    Please determine which version better protects personal or sensitive information.
    Output ONLY the following JSON object:
    {{
    "reason": Please state your reason within 2 sentences,
    "response": "A" | "B" | "SAME"
    }}"""
    messages = [
        {"role": "system", "content": sensitivity_system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    allowed = {"A", "B", "SAME"}
    last_raw = ""
    last_model = sensitivity_model
    last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in range(1, max_attempts + 1):
        try:
            response = fw_client.chat.completions.create(
                model=sensitivity_model,
                messages=messages,
                temperature=0.0,
                top_p=1,
            )
            last_model = getattr(response, "model", sensitivity_model)
            usage_obj = getattr(response, "usage", None)
            last_usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }

            raw = response.choices[0].message.content if getattr(response, "choices", None) else ""
            last_raw = raw or ""
            model_response = (last_raw or "").strip()

            json_obj = extract_json_response(model_response)
            result = None
            if isinstance(json_obj, dict):
                cand = json_obj.get("response")
                if isinstance(cand, str):
                    cand_up = cand.strip().upper()
                    if cand_up in allowed:
                        result = cand_up

            if result is not None:
                if return_raw:
                    return result, last_raw, last_model, last_usage
                return result

            if attempt < max_attempts:
                wait = backoff_base * (2 ** (attempt - 1))
                print(f"⚠️ parse failed (attempt {attempt}/{max_attempts}), retrying in {wait:.2f}s…")
                time.sleep(wait)
                continue

            if return_raw:
                return "Sensitivity ERROR", last_raw, last_model, last_usage
            return "ERROR"

        except Exception as e:
            print(f"❌ gpt_sensitivity_with_retry failed on attempt {attempt}/{max_attempts}: {e}")
            print(traceback.format_exc())
            break

    if return_raw:
        return "Sensitivity ERROR", last_raw, last_model, last_usage
    return "ERROR"


def gpt_utility_with_retry(
    user_message: str,
    message_A: str,
    message_B: str,
    utility_model: str,
    OPENAI_KEY: str,
    return_raw: bool = False,
    max_attempts: int = 3,
    backoff_base: float = 0.4,
):
    utility_system_prompt = '''You are a utility evaluator. Your task is to assess whether a second response adequately addresses a user's original message. You will be provided with:
    The user message (what the user is asking or talking about)
    A reference response A(a standard answer that is known to be good)
    A candidate response B(a second version that needs to be evaluated)
    You must decide whether the candidate response is "PASS" or "FAIL".
    This is not a test of exact wording or stylistic similarity. Instead, focus on whether the candidate response addresses all the key points or needs expressed in the user message. If it does, it passes. If it fails to address one or more key points, it fails.

    Input will contain three tagged sections: <user_message>, <response_A>, and <response_B>.

    Return in JSON format with the result and also one line of explanation:
    {"Result": "PASS/FAIL", "Reason": YOUR_EXPLANATION}'''

    messages = [
        {"role": "system", "content": utility_system_prompt},
        {"role": "user", "content": f"<user_message>{user_message}</user_message>\n<response_A>{message_A}</response_A>\n<response_B>{message_B}</response_B>"}
    ]

    allowed = {"PASS", "FAIL"}
    last_raw = ""
    last_model = utility_model
    last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_elapsed = 0.0

    for attempt in range(1, max_attempts + 1):
        try:
            start = time.time()
            client = OpenAI(api_key=OPENAI_KEY, timeout=60.0)
            response = client.chat.completions.create(
                model=utility_model,
                temperature=0,
                top_p=1.0,
                messages=messages,
                response_format={"type": "json_object"},
            )
            last_elapsed = time.time() - start

            last_model = getattr(response, "model", utility_model)
            usage_obj = getattr(response, "usage", None)
            last_usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                "completion_tokens": getattr(usage_obj, "completion_tokens", 0),
                "total_tokens": getattr(usage_obj, "total_tokens", 0),
            }

            raw = (response.choices[0].message.content or "").strip() if getattr(response, "choices", None) else ""
            last_raw = raw

            parsed = {}
            if raw:
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}

            result = None
            if isinstance(parsed, dict):
                cand = parsed.get("Result")
                if isinstance(cand, str):
                    cand_up = cand.strip().upper()
                    if cand_up in allowed:
                        result = cand_up

            if result is not None:
                if return_raw:
                    return result, last_raw, last_model, last_usage, last_elapsed
                return result

            if attempt < max_attempts:
                wait = backoff_base * (2 ** (attempt - 1))
                print(f"⚠️ utility parse failed (attempt {attempt}/{max_attempts}), retrying in {wait:.2f}s…")
                time.sleep(wait)
                continue

            if return_raw:
                return "Utility ERROR", last_raw, last_model, last_usage, last_elapsed
            return "ERROR"

        except Exception as e:
            print(f"❌ gpt_utility_with_retry failed on attempt {attempt}/{max_attempts}: {type(e).__name__}: {e}")
            print(traceback.print_exc())
            break

    if return_raw:
        return "Utility ERROR", last_raw, last_model, last_usage, last_elapsed
    return "ERROR"



def make_LLM(logger: Logger, llm_response_model: str):
    def LLM(msg, phase="phase2", replacement_mapping=None):
        start = time.time()
        raw_response, model, usage = gpt_llm_response(
            msg, llm_response_model, OPENAI_KEY, OPENROUTER_KEY, TOGETHER_KEY, ANTHROPIC_KEY, return_raw=True
        )
        elapsed = time.time() - start
        logger.log_phase_llm_response(
            phase=phase,
            replacement_mapping=replacement_mapping or {},
            output=raw_response,
            model=model,
            input_text=msg,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            time_taken=elapsed
        )
        return raw_response
    return LLM

def make_S(logger: Logger, sensitivity_model: str, fw_client: Fireworks):

    def S_func(M1, M2):
        start = time.time()
        result, raw, model, usage = gpt_sensitivity_with_retry(M1, M2, sensitivity_model, fw_client, return_raw=True)
        elapsed = time.time() - start
        returned_message = M1
        input_data = {"M1": M1,"M2": M2}
        if result == "A":
            returned_message = M1
        elif result == "B":
            returned_message = M2
        else:
            returned_message = random.choice([M1, M2])
        logger.log_phase2_sensitivity_compare(
            gpt_result=result, 
            output_message=returned_message, 
            input_data=input_data,
            model=model,
            raw_response=raw,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            time_elapsed=elapsed
        )
        return returned_message
    return S_func

def make_cached_S(S_func, logger: Logger):
    s_cache = {}; stats = {"hit": 0, "miss": 0}
    def cached_S(M1, M2):
        key = tuple(sorted((M1, M2)))
        if key in s_cache:
            result = s_cache[key]
            stats["hit"] += 1
            logger.log_phase2_sensitivity_cache_hit(s_cache.keys(), key, result)
            return result
        stats["miss"] += 1
        result = S_func(M1, M2)
        s_cache[key] = result
        s_cache[(key[1], key[0])] = result
        return result
    cached_S.stats = stats
    return cached_S

def utility_eval(original_message, original_response, restored_response, utility_model: str):
    return gpt_utility_with_retry(original_message, original_response, restored_response, utility_model, OPENAI_KEY, return_raw=True)


class PriorityWrapper:
    def __init__(self, value, comparator):
        self.value = value
        self.comparator = comparator
    def __lt__(self, other):
        M1 = self.value[1]; M2 = other.value[1]
        return self.comparator(M1, M2) == M1

def check_answer(original_message, original_response, replacement_mapping, LLM, cache=None, logger: Logger=None, phase="phase2", utility_model: str = DEFAULT_UTILITY_MODEL):
    key = json.dumps({"msg": original_response, "replacement_mapping": replacement_mapping}, sort_keys=True)
    if cache is not None and key in cache:
        cache_entry = cache[key]
        passed = cache_entry["passed"]
        prompt = cache_entry["prompt"]
        restored = cache_entry.get("restored_response")
        if logger:
            logger.log_phase_utility_cache_hit(phase, replacement_mapping, cache_entry)
        return passed, prompt, restored
    masked_message = apply_transformation(original_message, replacement_mapping)
    response_of_masked_message = LLM(masked_message, phase, replacement_mapping)
    restored_response = restore_response(response_of_masked_message, replacement_mapping)
    result, raw, model, usage, elapsed = utility_eval(original_message, original_response, restored_response, utility_model)
    passed = result.upper() == "PASS"
    if logger:
        logger.log_phase_utility_check(
            phase=phase,
            passed = passed,
            response_of_masked_message=response_of_masked_message,
            restored_response=restored_response,
            original_response=original_response,
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            time_taken=elapsed,
            raw_response=raw,
        )
    if cache is not None:
        cache[key] = {"passed": passed,"prompt": masked_message,"restored_response": restored_response}
    return passed, masked_message, restored_response

def masking_optimization(original_message, original_response, pii_entities, redact_map, abstract_map, variants_map, LLM, cached_S, logger: Logger, utility_model: str):
    total_start = time.time()
    logger.log_original(original_message=original_message, original_response=original_response)
    utility_evaluate_cache = {}
    frozen_entities = set(); redact_failed = set(); abstract_failed = set()

    phase1_start_time = time.time()
    for e in pii_entities:
        try:
            replacement_mapping = build_replacement_map({e: "redact"}, redact_map, abstract_map, variants_map)
            passed, _, _ = check_answer(original_message, original_response, replacement_mapping, LLM, cache=utility_evaluate_cache, logger=logger, phase="phase1", utility_model=utility_model)
            if not passed: redact_failed.add(e)
        except Exception:
            redact_failed.add(e)
        dbg(logger.data["llm_response_model"], f"phase1 redact test: entity={e}, passed={passed}")
    for e in redact_failed:
        try:
            replacement_mapping = build_replacement_map({e: "abstract"}, redact_map, abstract_map, variants_map)
            passed, _, _ = check_answer(original_message, original_response, replacement_mapping, LLM, cache=utility_evaluate_cache, logger=logger, phase="phase1", utility_model=utility_model)
            if not passed: abstract_failed.add(e)
        except Exception:
            abstract_failed.add(e)
        dbg(logger.data["llm_response_model"], f"phase1 abstract test: entity={e}, passed={passed}")
    for e in redact_failed:
        if e in abstract_failed:
            frozen_entities.add(e)

    logger.finalize_phase1(frozen=frozen_entities, redact_failed=redact_failed, abstract_failed=abstract_failed, duration=time.time() - phase1_start_time)

    transformation_0 = {}
    for e in pii_entities:
        if e in frozen_entities:
            continue
        elif e in redact_failed:
            transformation_0[e] = "abstract"
        else:
            transformation_0[e] = "redact"

    dbg(logger.data["llm_response_model"], f"phase1 frozen={sorted(frozen_entities)}")
    dbg(logger.data["llm_response_model"], f"transformation_0={transformation_0}")

    replace_mapping0 = build_replacement_map(transformation_0, redact_map, abstract_map, variants_map)
    M0 = apply_transformation(original_message, replace_mapping0)
    pq = [PriorityWrapper((transformation_0, M0), cached_S)]
    logger.log_phase2_initial(transformation_0)
    visited = set()

    while pq:
        dbg(logger.data["llm_response_model"], f"PQ init size={len(pq)}")
        wrapper = heapq.heappop(pq)
        T, M_current = wrapper.value
        replacement_mapping = build_replacement_map(T, redact_map, abstract_map, variants_map)
        T_key = frozenset(T.items())
        if T_key in visited:
            continue
        visited.add(T_key)
        logger.log_phase2_pop(T, replacement_mapping, M_current)
        passed, last_prompt, last_restored_response = check_answer(original_message, original_response, replacement_mapping, LLM, cache=utility_evaluate_cache, logger=logger, phase="phase2", utility_model=utility_model)
        dbg(logger.data["llm_response_model"], f"PQ pop | |T|={len(T)} keys={list(T.keys())}")
        dbg(logger.data["llm_response_model"], f"check_answer passed={passed}")
        if passed:
            duration = time.time() - total_start
            logger.log_final(T, duration, len(utility_evaluate_cache))
            return M_current, T, replacement_mapping, duration, len(utility_evaluate_cache), last_prompt, last_restored_response
        for e, method in list(T.items()):
            if e in frozen_entities:
                continue
            T_new = T.copy()
            if method == "redact":
                T_new[e] = "abstract"
            elif method == "abstract":
                del T_new[e]
            else:
                continue
            replacement_mapping_new = build_replacement_map(T_new, redact_map, abstract_map, variants_map)
            M_new = apply_transformation(original_message, replacement_mapping_new)
            logger.log_phase2_push(T_new)
            heapq.heappush(pq, PriorityWrapper((T_new, M_new), cached_S))
            dbg(logger.data["llm_response_model"], f"PQ push | change {e}:{method} -> {T_new.get(e, 'removed')} | new_size={len(pq)}")
    duration = time.time() - total_start
    logger.data["sensitivity_cache_stats"] = cached_S.stats
    logger.log_final({}, duration, len(utility_evaluate_cache))
    return original_message, {}, {}, duration, len(utility_evaluate_cache), "[ORIGINAL_MESSAGE]", "[ORIGINAL_RESPONSE]"

def run_masking_pipeline(
    dataset_name: str,
    log_folder_name: str,
    idx: int,
    conversation_id: str,
    original_message: str,
    pii_dict: dict,
    variants_map: dict,
    redact_map: dict,
    abstract_map: dict,
    sensitivity_model: str,
    llm_response_model: str,
    utility_model: str,
    fw_client: Fireworks
):
    logger = Logger(conversation_id, sensitivity_model, llm_response_model, utility_model)
    S = make_S(logger, sensitivity_model, fw_client)
    cached_S = make_cached_S(S, logger)
    LLM = make_LLM(logger, llm_response_model)
    original_response = LLM(original_message, phase="phase1")
    logger.data["pii_dict"] = pii_dict
    logger.data["variants_map"] = variants_map
    dbg(llm_response_model, f"idx={idx} PII keys={list(pii_dict.keys())}")
    dbg(llm_response_model, f"redact_map={redact_map}")
    dbg(llm_response_model, f"abstract_map={abstract_map}")

    logger.data["redact_map"] = redact_map
    logger.data["abstract_map"] = abstract_map

    masked_message, T, replacement_mapping, duration, utility_evaluate_calls, last_prompt, last_restored_response = masking_optimization(
        original_message, original_response, list(pii_dict.keys()),
        redact_map, abstract_map, variants_map,
        LLM, cached_S, logger, utility_model
    )

    sensitivity_stats = cached_S.stats
    logger.data["sensitivity_cache_stats"] = sensitivity_stats
    logger.save(f"./{log_folder_name}/{dataset_name}_{idx}_{conversation_id}_summary.json")
    transformation_stats = compute_transformation_stats(pii_dict, T)
    return (masked_message, T, replacement_mapping, duration, utility_evaluate_calls, sensitivity_stats.get("miss", 0), transformation_stats, last_prompt, last_restored_response)


def run_for_model(
    llm_response_model: str,
    dataset_path: str,
    output_dir: str,
    start_idx: int,
    end_idx: int,
    target_indices: Optional[List[int]],
    sensitivity_model: str,
    utility_model: str
):

    orig_print = builtins.print
    def model_print(*args, **kwargs):
        prefix = f"[{llm_response_model}]"
        if args:
            args = (prefix,) + args
        else:
            args = (prefix,)
        orig_print(*args, **kwargs)
    builtins.print = model_print



    log_folder_name = f"{output_dir}/dm_of_{llm_response_model}"
    os.makedirs(log_folder_name, exist_ok=True)
    est_time = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%Z")
    output_path = f"{log_folder_name}/auto_masked_messages_{est_time}.jsonl"

    fw = Fireworks(api_key=FIREWORKS_KEY)

    dataset_name = Path(dataset_path).stem

    with open(output_path, "w", encoding="utf-8") as outfile:
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                print(f"Processing line#{line_no}", flush=True)
                record = json.loads(line.strip())
                rec_idx = int(record["index"])
                conversation_id = record["conversation_hash"]
                
                if target_indices:
                    if rec_idx not in target_indices:
                        continue
                else:
                    if rec_idx < start_idx:
                        continue
                    if end_idx is not None and rec_idx > end_idx:
                        continue
                dbg(llm_response_model, f"parsed rec_idx={rec_idx}, conversation_id={conversation_id}")
                dbg(llm_response_model, f"filter: start_idx={start_idx}, end_idx={end_idx}, target_indices={target_indices}")
                original_message = record["user_message"]
                pii_dict = record["pii_dict"]
                variants_map = record["variants_map"]
                redact_map = record["redact_map"]
                abstract_map = record["abstract_map"]

                try:
                    (masked_message, transformation, replacement_mapping,
                     total_elapsed, utility_evaluate_calls, sensitivity_compare_calls,
                     transformation_stats, last_prompt, last_restored_response) = run_masking_pipeline(
                        dataset_name=dataset_name,
                        log_folder_name=log_folder_name,
                        idx=rec_idx,
                        conversation_id=conversation_id,
                        original_message=original_message,
                        pii_dict=pii_dict,
                        variants_map=variants_map,
                        redact_map=redact_map,
                        abstract_map=abstract_map,
                        sensitivity_model=sensitivity_model,
                        llm_response_model=llm_response_model,
                        utility_model=utility_model,
                        fw_client=fw
                    )
                except Exception as e:
                    print(f"[ERROR] Model={llm_response_model} Message {conversation_id} failed: {e}")
                    masked_message, transformation, replacement_mapping = original_message, {}, {}
                    total_elapsed, utility_evaluate_calls, sensitivity_compare_calls = -1, -1, -1
                    transformation_stats = {"per_type": {}, "totals": {"redact": 0, "abstract": 0, "retain": 0}}
                
                
                result_item = {
                    "index": rec_idx,
                    "conversation_hash": conversation_id,
                    "original_message": original_message,
                    "masked_message": masked_message,
                    "pii_dict": pii_dict,
                    "variants_map": variants_map,
                    "transformation": transformation,
                    "transformation_stats": transformation_stats,
                    "replacement_mapping": replacement_mapping,
                    "duration_sec": round(total_elapsed, 2),
                    "utility_evaluate_calls": utility_evaluate_calls,
                    "sensitivity_compare_calls": sensitivity_compare_calls,
                    "redact_map": redact_map,
                    "abstract_map": abstract_map,
                }
                outfile.write(json.dumps(result_item) + "\n")
                outfile.flush()

    try:
        tmp_sorted = output_path.replace(".jsonl", "_sorted.jsonl")
        rows = []
        with open(output_path, "r", encoding="utf-8") as _in:
            for _line in _in:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    obj = json.loads(_line)
                except Exception:
                    continue
                rows.append(obj)

        rows.sort(key=lambda r: int(r.get("index", 0)))

        with open(tmp_sorted, "w", encoding="utf-8") as _out:
            for r in rows:
                _out.write(json.dumps(r, ensure_ascii=False) + "\n")

        os.replace(tmp_sorted, output_path)
        print(f"[SORTED] Rewritten in ascending index order: {output_path}")
    except Exception as e:
        print(f"[WARN] Post-sort failed, keeping unsorted file: {e}")

    print(f"✅ Model={llm_response_model} done. Saved to {output_path}")
    return output_path

# =========================================================
# CLI
# =========================================================

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Run data minimization pipeline over models.")
    p.add_argument("--models", nargs="+", required=True,
                   help="Space-separated list of models to run, e.g. gpt-4o lgai/exaone-deep-32b qwen/qwen-2.5-7b-instruct")
    p.add_argument("--dataset", default="wildchat_sampled_stratified_with_maps.jsonl",
                   help="Path to dataset (.jsonl)")
    p.add_argument("--output-dir", default="data_minimization_results", help="Output root directory")

    p.add_argument("--start-idx", type=int, default=-1,
               help="Start from records where index >= start_idx (inclusive)")
    p.add_argument("--end-idx", type=int, default=10000000,
                help="End at records where index <= end_idx (inclusive)")
    p.add_argument("--target-indices", type=str, default="",
                help="Only run these record indices (comma-separated, e.g. 6,15,18). Takes precedence over start/end range")

    p.add_argument("--sensitivity-model", default=DEFAULT_SENSITIVITY_MODEL)
    p.add_argument("--utility-model", default=DEFAULT_UTILITY_MODEL)
    p.add_argument("-j", "--concurrency", type=int, default=2, help="Number of concurrent processes (by model)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.target_indices.strip():
        target_indices = [int(x) for x in args.target_indices.split(",") if x.strip()]
    else:
        target_indices = None

    from concurrent.futures import ProcessPoolExecutor, as_completed

    models = list(args.models)

    if args.concurrency <= 1:
        for m in models:
            path = run_for_model(
                llm_response_model=m,
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                target_indices=target_indices,
                sensitivity_model=args.sensitivity_model,
                utility_model=args.utility_model,
            )
            print(f"[OK][{m}] {path}")
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            future_to_model = {
                ex.submit(
                    run_for_model,
                    llm_response_model=m,
                    dataset_path=args.dataset,
                    output_dir=args.output_dir,
                    start_idx=args.start_idx,
                    end_idx=args.end_idx,
                    target_indices=target_indices,
                    sensitivity_model=args.sensitivity_model,
                    utility_model=args.utility_model,
                ): m
                for m in models
            }

            for fut in as_completed(future_to_model):
                m = future_to_model[fut]
                try:
                    path = fut.result()
                    print(f"[OK][{m}] {path}")
                except Exception:
                    import sys, traceback
                    tb = "".join(traceback.format_exception(*sys.exc_info()))
                    print(f"[FAILED] model={m}\n{tb}")



if __name__ == "__main__":
    main()
