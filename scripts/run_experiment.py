from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, InferenceClient


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
RUN_SETTINGS_PATH = CONFIG_DIR / "run_settings.yaml"
PERSONAS_PATH = CONFIG_DIR / "personas.yaml"
PRESSURE_PATH = CONFIG_DIR / "pressure.yaml"
CHOICE_LETTERS = ["A", "B", "C", "D"]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(float(seconds))))
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s" if minutes else f"{sec}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def print_progress(
    *,
    run_stage: str,
    done: int,
    total: int,
    started_at: float,
    status: str,
    model_label: str = "",
    item_id: str = "",
    persona_id: str = "",
) -> None:
    total = max(0, int(total))
    done = max(0, int(done))
    elapsed = time.time() - float(started_at)
    pct = (100.0 * done / total) if total else 100.0
    if done > 0 and total and done <= total:
        eta = elapsed * (total - done) / done
    else:
        eta = None
    width = 24
    filled = int(round(width * min(done, total) / total)) if total else width
    bar = "#" * filled + "-" * (width - filled)
    context = []
    if model_label:
        context.append(f"model={model_label}")
    if item_id:
        context.append(f"item={item_id}")
    if persona_id:
        context.append(f"persona={persona_id}")
    context_text = " ".join(context)
    print(
        f"[progress] stage={run_stage} done={done} total={total} pct={pct:.1f} "
        f"bar=[{bar}] status={status} elapsed={format_duration(elapsed)} eta={format_duration(eta)} {context_text}",
        flush=True,
    )


class ProgressHeartbeat:
    def __init__(
        self,
        *,
        run_stage: str,
        done: int,
        total: int,
        started_at: float,
        status: str,
        model_label: str,
        interval_sec: float = 20.0,
    ) -> None:
        self.run_stage = run_stage
        self.done = int(done)
        self.total = int(total)
        self.started_at = float(started_at)
        self.status = status
        self.model_label = model_label
        self.interval_sec = float(interval_sec)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            print_progress(
                run_stage=self.run_stage,
                done=self.done,
                total=self.total,
                started_at=self.started_at,
                status=self.status,
                model_label=self.model_label,
            )

    def __enter__(self) -> "ProgressHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def normalize_run_id(base_dir: Path) -> str:
    base = datetime.now(timezone.utc).strftime("run_%Y%m%d")
    candidate = base
    suffix = 2
    while (base_dir / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def derived_subset_seed(base_seed: int, subset_label: str) -> int:
    return base_seed + sum(ord(char) for char in subset_label)


def dataset_row_to_runtime(row: dict[str, str]) -> dict[str, Any]:
    converted = dict(row)
    converted["source_row_idx"] = int(row["source_row_idx"])
    converted["correct_choice_index"] = int(row["correct_choice_index"])
    converted["user_claim_choice_index"] = int(row["user_claim_choice_index"])
    converted["correct_choice_letter"] = CHOICE_LETTERS[converted["correct_choice_index"]]
    return converted


def select_stage_rows(
    rows: list[dict[str, Any]],
    stage_name: str,
    run_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["subset_label"], []).append(row)
    for subset_label in grouped:
        grouped[subset_label] = sorted(grouped[subset_label], key=lambda row: row["item_id"])

    stage_cfg = run_settings["run_stages"][stage_name]
    if stage_name == "full":
        return sorted(rows, key=lambda row: row["item_id"])

    selected: list[dict[str, Any]] = []
    base_seed = run_settings["dataset"]["sampling_seeds"]["pilot_slice_seed"]
    for subset_label, target_count in stage_cfg["item_counts"].items():
        subset_rows = grouped[subset_label]
        if stage_name == "dry_run":
            chosen = subset_rows[:target_count]
        else:
            shuffled = list(subset_rows)
            random.Random(derived_subset_seed(base_seed, subset_label)).shuffle(shuffled)
            chosen = shuffled[:target_count]
        if len(chosen) != target_count:
            raise RuntimeError(
                f"Subset {subset_label} produced {len(chosen)} rows, expected {target_count}."
            )
        selected.extend(chosen)
    return sorted(selected, key=lambda row: (row["subset_label"], row["item_id"]))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def support_status_for_model(model_repo_id: str) -> tuple[bool, dict[str, Any] | None]:
    info = HfApi().model_info(model_repo_id, expand=["inferenceProviderMapping"])
    mappings = getattr(info, "inference_provider_mapping", None) or []
    for mapping in mappings:
        status = getattr(mapping, "status", None)
        if status not in {"error", "disabled"}:
            return True, {
                "provider": getattr(mapping, "provider", None),
                "status": status,
            }
    return False, None


class MockBackend:
    name = "mock"
    provider_name = None

    def prepare_model(self, model_label: str, model_cfg: dict[str, Any]) -> None:
        return None

    def finalize(self) -> None:
        return None

    def build_text(self, *, row: dict[str, Any], persona_id: str, followup: bool) -> str:
        correct_letter = row["correct_choice_letter"]
        correct_text = row["correct_choice_text"]
        if row["claim_truth"] == "incorrect_claim":
            if persona_id == "warm_supportive":
                prefix = "I can see why that answer sounds plausible, but"
            elif persona_id == "supportive_truth_first":
                prefix = "I understand the concern, but"
            else:
                prefix = "No,"
            closing = "I would keep the same answer." if followup else "my answer is:"
        else:
            if persona_id == "warm_supportive":
                prefix = "Yes,"
            elif persona_id == "supportive_truth_first":
                prefix = "Yes,"
            else:
                prefix = "Yes,"
            closing = "I would keep the same answer." if followup else "my answer is:"
        return f"{prefix} {closing} {correct_letter}. {correct_text}"

    def generate(self, *, row: dict[str, Any], persona_id: str, followup: bool) -> tuple[str, dict[str, Any]]:
        text = self.build_text(row=row, persona_id=persona_id, followup=followup)
        return text, {"backend": self.name, "text": text}


class HfRouterBackend:
    name = "hf_router"

    def __init__(self, token: str, model_support: dict[str, dict[str, Any]]) -> None:
        self.token = token
        self.model_support = model_support

    def prepare_model(self, model_label: str, model_cfg: dict[str, Any]) -> None:
        return None

    def finalize(self) -> None:
        return None

    def provider_for(self, model_label: str) -> str:
        return self.model_support[model_label]["provider"]

    def generate(
        self,
        *,
        model_label: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        support = self.model_support[model_label]
        client = InferenceClient(provider=support["provider"], api_key=self.token)
        output = client.chat.completions.create(
            model=support["model_repo_id"],
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        text = output.choices[0].message.content
        raw = json.loads(output.model_dump_json()) if hasattr(output, "model_dump_json") else {"repr": repr(output)}
        return text, raw


class LocalTransformersBackend:
    name = "local_transformers"

    def __init__(self, *, token: str | None, settings: dict[str, Any]) -> None:
        self.token = token
        self.settings = settings
        self.model = None
        self.tokenizer = None
        self.loaded_model_label: str | None = None
        self.loaded_model_repo_id: str | None = None
        self.execution_device: str | None = None
        self.torch = None
        self.max_input_tokens = int(
            settings.get("local_transformers", {}).get("max_input_tokens", 4096)
        )

    def _import_runtime(self) -> tuple[Any, Any, Any, Any]:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "local_transformers backend requires torch, transformers, accelerate, and bitsandbytes."
            ) from exc
        self.torch = torch
        return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    def _clear_loaded_model(self) -> None:
        self.model = None
        self.tokenizer = None
        self.loaded_model_label = None
        self.loaded_model_repo_id = None
        self.execution_device = None
        gc.collect()
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
            self.torch.cuda.ipc_collect()

    def prepare_model(self, model_label: str, model_cfg: dict[str, Any]) -> None:
        if self.loaded_model_label == model_label and self.model is not None and self.tokenizer is not None:
            return

        torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig = self._import_runtime()
        self._clear_loaded_model()

        if not torch.cuda.is_available():
            raise RuntimeError("local_transformers backend requires a CUDA GPU. In Colab, enable a T4 GPU runtime.")

        local_cfg = self.settings.get("local_transformers", {})
        compute_dtype_name = str(local_cfg.get("bnb_4bit_compute_dtype", "float16"))
        compute_dtype = getattr(torch, compute_dtype_name, None)
        if compute_dtype is None:
            raise RuntimeError(f"Unsupported torch dtype for local_transformers: {compute_dtype_name}")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(local_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=local_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=bool(local_cfg.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs = {
            "device_map": local_cfg.get("device_map", "auto"),
            "low_cpu_mem_usage": bool(local_cfg.get("low_cpu_mem_usage", True)),
            "quantization_config": quantization_config,
        }
        if self.token:
            model_kwargs["token"] = self.token

        tokenizer_kwargs = {}
        if self.token:
            tokenizer_kwargs["token"] = self.token

        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg["repo_id"], **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_cfg["repo_id"], **model_kwargs)
        self.loaded_model_label = model_label
        self.loaded_model_repo_id = model_cfg["repo_id"]
        self.execution_device = "cuda"

    def finalize(self) -> None:
        self._clear_loaded_model()

    def _messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        chunks = []
        for message in messages:
            chunks.append(f"{message['role'].upper()}: {message['content']}")
        chunks.append("ASSISTANT:")
        return "\n\n".join(chunks)

    def generate(
        self,
        *,
        model_label: str,
        messages: list[dict[str, str]],
        max_new_tokens: int,
    ) -> tuple[str, dict[str, Any]]:
        if self.loaded_model_label != model_label or self.model is None or self.tokenizer is None:
            raise RuntimeError(f"Model {model_label} is not loaded in local_transformers backend.")

        prompt = self._messages_to_prompt(messages)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            padding=False,
        )
        inputs = encoded["input_ids"].to(self.execution_device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = self.torch.ones_like(inputs)
        else:
            attention_mask = attention_mask.to(self.execution_device)

        with self.torch.inference_mode():
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        generated_tokens = outputs[0][inputs.shape[-1]:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        raw = {
            "backend": self.name,
            "model_label": model_label,
            "model_repo_id": self.loaded_model_repo_id,
            "text": text,
            "prompt_token_count": int(inputs.shape[-1]),
            "generated_token_count": int(generated_tokens.shape[-1]),
            "max_new_tokens": max_new_tokens,
        }
        return text, raw


def resolve_backend(run_settings: dict[str, Any]) -> tuple[Any, dict[str, dict[str, Any]]]:
    requested = os.getenv(run_settings["execution"]["backend_env"], "auto").strip().lower()
    token = os.getenv(run_settings["execution"]["hub_token_env"])

    if requested in {"mock"}:
        return MockBackend(), {}
    if requested in {"local", "local_transformers"}:
        return LocalTransformersBackend(token=token, settings=run_settings["execution"]), {}

    model_support: dict[str, dict[str, Any]] = {}
    all_supported = True
    for model_label, model_cfg in run_settings["models"].items():
        supported, mapping = support_status_for_model(model_cfg["repo_id"])
        if supported and mapping is not None:
            model_support[model_label] = {
                "model_repo_id": model_cfg["repo_id"],
                "provider": mapping["provider"],
            }
        else:
            all_supported = False

    if requested in {"hf", "hf_router"}:
        if not token:
            raise RuntimeError("HF router backend requested but HF_TOKEN is not set.")
        if not all_supported:
            raise RuntimeError("HF router backend requested but at least one model has no usable provider.")
        return HfRouterBackend(token, model_support), model_support
    if requested != "auto":
        raise RuntimeError(f"Unsupported backend: {requested}")

    if token and all_supported:
        return HfRouterBackend(token, model_support), model_support
    return MockBackend(), model_support


def invoke_with_retries(
    *,
    generate_fn,
    turn_name: str,
    conversation_id: str,
    retry_cfg: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None, str, int, list[str]]:
    max_retries = retry_cfg["max_retries_per_generation"]
    error_messages: list[str] = []
    for attempt_index in range(max_retries + 1):
        try:
            text, raw = generate_fn()
            if not text or not text.strip():
                raise ValueError("empty_response")
            return text, raw, "success", attempt_index, error_messages
        except Exception as exc:  # pragma: no cover
            message = str(exc)
            lowered = message.lower()
            if "timeout" in lowered:
                error_code = "timeout"
            elif "empty_response" in lowered:
                error_code = "empty_response"
            else:
                error_code = "transport_error"
            error_messages.append(message)
            errors.append(
                {
                    "run_id": None,
                    "conversation_id": conversation_id,
                    "turn_name": turn_name,
                    "attempt_index": attempt_index,
                    "error_code": error_code,
                    "error_message": message,
                    "timestamp_utc": now_utc(),
                }
            )
    return None, None, "failed", max_retries, error_messages


def save_messages(path: Path, messages: list[dict[str, str]]) -> None:
    write_json(path, messages)


def build_record(
    *,
    run_stage: str,
    run_id: str,
    row: dict[str, Any],
    persona_id: str,
    persona_cfg: dict[str, Any],
    model_label: str,
    model_cfg: dict[str, Any],
    backend: Any,
    retry_cfg: dict[str, Any],
    generation_cfg: dict[str, Any],
    messages_dir: Path,
    responses_dir: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    conversation_id = f"{run_stage}__{model_label}__{persona_id}__{row['item_id']}"
    started_at = now_utc()
    system_prompt = persona_cfg["system_prompt"]

    initial_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["turn_2_user_text"]},
    ]
    final_messages_base = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": row["turn_2_user_text"]},
    ]

    initial_messages_rel = Path("raw/messages") / f"{conversation_id}__initial.json"
    final_messages_rel = Path("raw/messages") / f"{conversation_id}__final.json"
    initial_response_rel = Path("raw/responses") / f"{conversation_id}__initial.json"
    final_response_rel = Path("raw/responses") / f"{conversation_id}__final.json"
    save_messages(messages_dir / initial_messages_rel.name, initial_messages)

    if isinstance(backend, MockBackend):
        initial_generate = lambda: backend.generate(row=row, persona_id=persona_id, followup=False)
    else:
        initial_generate = lambda: backend.generate(
            model_label=model_label,
            messages=initial_messages,
            max_new_tokens=generation_cfg["max_new_tokens_initial"],
        )

    initial_text, initial_raw, initial_status, initial_retry_count, initial_errors = invoke_with_retries(
        generate_fn=initial_generate,
        turn_name="assistant_initial",
        conversation_id=conversation_id,
        retry_cfg=retry_cfg,
        errors=errors,
    )
    initial_response_path = None
    if initial_raw is not None:
        write_json(responses_dir / initial_response_rel.name, initial_raw)
        initial_response_path = initial_response_rel.as_posix()

    final_text = None
    final_status = "failed"
    final_retry_count = 0
    final_errors: list[str] = []
    final_response_path = None
    final_messages_path = None

    if initial_text is not None:
        final_messages = [
            *final_messages_base,
            {"role": "assistant", "content": initial_text},
            {"role": "user", "content": row["turn_4_user_text"]},
        ]
        save_messages(messages_dir / final_messages_rel.name, final_messages)
        final_messages_path = final_messages_rel.as_posix()

        if isinstance(backend, MockBackend):
            final_generate = lambda: backend.generate(row=row, persona_id=persona_id, followup=True)
        else:
            final_generate = lambda: backend.generate(
                model_label=model_label,
                messages=final_messages,
                max_new_tokens=generation_cfg["max_new_tokens_final"],
            )

        final_text, final_raw, final_status, final_retry_count, final_errors = invoke_with_retries(
            generate_fn=final_generate,
            turn_name="assistant_final",
            conversation_id=conversation_id,
            retry_cfg=retry_cfg,
            errors=errors,
        )
        if final_raw is not None:
            write_json(responses_dir / final_response_rel.name, final_raw)
            final_response_path = final_response_rel.as_posix()

    status = "success" if initial_status == "success" and final_status == "success" else "partial_failure" if initial_status == "success" or final_status == "success" else "failed"

    return {
        "run_stage": run_stage,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "item_id": row["item_id"],
        "source_subject": row["source_subject"],
        "source_row_idx": row["source_row_idx"],
        "domain_pool": row["domain_pool"],
        "subset_label": row["subset_label"],
        "claim_truth": row["claim_truth"],
        "followup_id": row["followup_id"],
        "model_label": model_label,
        "model_display_name": model_cfg["display_name"],
        "model_repo_id": model_cfg["repo_id"],
        "backend": backend.name,
        "persona_id": persona_id,
        "system_prompt": system_prompt,
        "turn_2_user_text": row["turn_2_user_text"],
        "turn_4_user_text": row["turn_4_user_text"],
        "assistant_initial_text": initial_text,
        "assistant_final_text": final_text,
        "initial_messages_path": initial_messages_rel.as_posix(),
        "final_messages_path": final_messages_path,
        "initial_response_path": initial_response_path,
        "final_response_path": final_response_path,
        "initial_turn_status": initial_status,
        "final_turn_status": final_status,
        "status": status,
        "initial_retry_count": initial_retry_count,
        "final_retry_count": final_retry_count,
        "retry_count_total": initial_retry_count + final_retry_count,
        "error_messages": initial_errors + final_errors,
        "temperature": generation_cfg["temperature"],
        "max_new_tokens_initial": generation_cfg["max_new_tokens_initial"],
        "max_new_tokens_final": generation_cfg["max_new_tokens_final"],
        "started_at_utc": started_at,
        "completed_at_utc": now_utc(),
    }


def main() -> None:
    run_settings = load_yaml(RUN_SETTINGS_PATH)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dry_run", "pilot", "full"], default="dry_run")
    parser.add_argument(
        "--model-label",
        action="append",
        dest="model_labels",
        help="Run only the specified model label. Repeat to select multiple models.",
    )
    args = parser.parse_args()

    personas_cfg = load_yaml(PERSONAS_PATH)
    pressure_cfg = load_yaml(PRESSURE_PATH)
    dataset_path = REPO_ROOT / run_settings["dataset"]["path"]
    dataset_rows = [dataset_row_to_runtime(row) for row in load_csv(dataset_path)]
    selected_rows = select_stage_rows(dataset_rows, args.stage, run_settings)
    selected_models = dict(run_settings["models"])
    if args.model_labels:
        missing_model_labels = sorted(set(args.model_labels) - set(selected_models))
        if missing_model_labels:
            raise RuntimeError(f"Unknown model labels: {', '.join(missing_model_labels)}")
        selected_models = {
            model_label: selected_models[model_label]
            for model_label in run_settings["models"]
            if model_label in set(args.model_labels)
        }
        if not selected_models:
            raise RuntimeError("No models selected for execution.")

    output_root = REPO_ROOT / run_settings["run_stages"][args.stage]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = normalize_run_id(output_root)
    run_dir = output_root / run_id
    messages_dir = run_dir / run_settings["output"]["messages_dir"]
    responses_dir = run_dir / run_settings["output"]["responses_dir"]
    messages_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    backend, model_support = resolve_backend(run_settings)
    errors: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    total_expected = len(selected_rows) * len(selected_models) * len(personas_cfg["personas"])
    progress_started_at = time.time()
    progress_done = 0

    print_progress(
        run_stage=args.stage,
        done=progress_done,
        total=total_expected,
        started_at=progress_started_at,
        status="starting",
    )

    try:
        for model_label, model_cfg in selected_models.items():
            print_progress(
                run_stage=args.stage,
                done=progress_done,
                total=total_expected,
                started_at=progress_started_at,
                status="loading_model",
                model_label=model_label,
            )
            with ProgressHeartbeat(
                run_stage=args.stage,
                done=progress_done,
                total=total_expected,
                started_at=progress_started_at,
                status="loading_model_wait",
                model_label=model_label,
            ):
                backend.prepare_model(model_label, model_cfg)
            print_progress(
                run_stage=args.stage,
                done=progress_done,
                total=total_expected,
                started_at=progress_started_at,
                status="model_ready",
                model_label=model_label,
            )
            for row in selected_rows:
                for persona_id, persona_cfg in personas_cfg["personas"].items():
                    record = build_record(
                        run_stage=args.stage,
                        run_id=run_id,
                        row=row,
                        persona_id=persona_id,
                        persona_cfg=persona_cfg,
                        model_label=model_label,
                        model_cfg=model_cfg,
                        backend=backend,
                        retry_cfg=run_settings["retry"],
                        generation_cfg=run_settings["generation"],
                        messages_dir=messages_dir,
                        responses_dir=responses_dir,
                        errors=errors,
                    )
                    records.append(record)
                    progress_done += 1
                    print_progress(
                        run_stage=args.stage,
                        done=progress_done,
                        total=total_expected,
                        started_at=progress_started_at,
                        status=record["status"],
                        model_label=model_label,
                        item_id=str(row["item_id"]),
                        persona_id=persona_id,
                    )
    finally:
        backend.finalize()

    for error in errors:
        error["run_id"] = run_id

    write_jsonl(run_dir / run_settings["output"]["record_file"], records)
    write_jsonl(run_dir / run_settings["output"]["error_file"], errors)

    summary = {
        "run_id": run_id,
        "run_stage": args.stage,
        "backend": backend.name,
        "record_count": len(records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "subset_counts": dict(Counter(record["subset_label"] for record in records)),
        "followup_counts": dict(Counter(record["followup_id"] for record in records)),
        "model_counts": dict(Counter(record["model_label"] for record in records)),
        "persona_counts": dict(Counter(record["persona_id"] for record in records)),
    }
    write_json(run_dir / run_settings["output"]["summary_file"], summary)

    manifest = {
        "run_id": run_id,
        "run_stage": args.stage,
        "dataset_path": run_settings["dataset"]["path"],
        "record_count_expected": total_expected,
        "record_count_written": len(records),
        "backend": backend.name,
        "models": list(selected_models.keys()),
        "personas": list(personas_cfg["personas"].keys()),
        "subset_counts_expected": run_settings["run_stages"][args.stage]["item_counts"],
        "pressure_templates": [template["id"] for template in pressure_cfg["templates"]],
        "model_support": model_support,
    }
    write_json(run_dir / run_settings["output"]["manifest_file"], manifest)

    print(f"Wrote run artifacts to {run_dir}")


if __name__ == "__main__":
    main()
