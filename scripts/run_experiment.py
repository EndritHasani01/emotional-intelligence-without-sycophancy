from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi, InferenceClient


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATASET_PATH = REPO_ROOT / "data" / "frozen" / "assertions_frozen_v1.csv"
PERSONAS_PATH = CONFIG_DIR / "personas_v1.yaml"
PRESSURE_PATH = CONFIG_DIR / "pressure_v1.yaml"
RUN_SETTINGS_PATH = CONFIG_DIR / "run_settings_v1.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_date_run_id(base_dir: Path) -> str:
    base = datetime.now(timezone.utc).strftime("run_%Y%m%d")
    candidate = base
    suffix = 2
    while (base_dir / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def derived_subset_seed(base_seed: int, subset_label: str) -> int:
    return base_seed + sum(ord(char) for char in subset_label)


def parse_choices_json(serialized: str) -> list[str]:
    return json.loads(serialized)


def dataset_row_to_runtime(row: dict[str, str]) -> dict[str, Any]:
    converted = dict(row)
    converted["source_row_idx"] = int(row["source_row_idx"])
    converted["correct_choice_index"] = int(row["correct_choice_index"])
    converted["selected_claim_choice_index"] = int(row["selected_claim_choice_index"])
    converted["choices"] = parse_choices_json(row["choices_json"])
    return converted


def select_stage_rows(
    rows: list[dict[str, Any]],
    stage_name: str,
    run_settings: dict[str, Any],
) -> list[dict[str, Any]]:
    stage_cfg = run_settings["run_stages"][stage_name]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["subset_label"], []).append(row)
    for subset_label in grouped:
        grouped[subset_label] = sorted(grouped[subset_label], key=lambda item: item["item_id"])

    if stage_name == "full":
        return sorted(rows, key=lambda item: item["item_id"])

    selected: list[dict[str, Any]] = []
    base_seed = run_settings["dataset_contract"]["sampling_seeds"]["pilot_slice_seed"]
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
                f"Subset {subset_label} yielded {len(chosen)} rows; expected {target_count}."
            )
        selected.extend(chosen)
    return sorted(selected, key=lambda item: (item["subset_label"], item["item_id"]))


def serialize_raw_payload(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def support_status_for_model(model_repo_id: str) -> tuple[bool, dict[str, Any] | None]:
    api = HfApi()
    info = api.model_info(model_repo_id, expand=["inferenceProviderMapping"])
    mappings = getattr(info, "inference_provider_mapping", None) or []
    for mapping in mappings:
        status = getattr(mapping, "status", None)
        if status not in {"error", "disabled"}:
            return True, {
                "provider": getattr(mapping, "provider", None),
                "task": getattr(mapping, "task", None),
                "status": status,
            }
    return False, None


class MockBackend:
    backend_name = "mock_backend_v1"
    provider_name = None

    def generate_initial(self, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        row = context["row"]
        persona_id = context["persona_id"]
        if row["claim_truth"] == "incorrect_claim":
            if persona_id == "warm_supportive":
                text = (
                    f"I can see why that answer feels plausible, but I do not think it is correct. "
                    f"For this question, I would answer \"{row['correct_choice_text']}\"."
                )
            elif persona_id == "supportive_truth_first":
                text = (
                    f"I understand why you might think that, but it is not correct here. "
                    f"The best answer is \"{row['correct_choice_text']}\"."
                )
            else:
                text = (
                    f"That is not correct for this question. "
                    f"The correct answer is \"{row['correct_choice_text']}\"."
                )
        else:
            if persona_id == "warm_supportive":
                text = (
                    f"Yes, that matches the best answer here. "
                    f"I would confirm \"{row['selected_claim_choice_text']}\"."
                )
            elif persona_id == "supportive_truth_first":
                text = (
                    f"Yes. That is factually correct for this question. "
                    f"The correct answer is \"{row['selected_claim_choice_text']}\"."
                )
            else:
                text = (
                    f"Yes. The correct answer is \"{row['selected_claim_choice_text']}\"."
                )
        raw = {
            "backend": self.backend_name,
            "turn": "assistant_initial",
            "conversation_id": context["conversation_id"],
            "text": text,
        }
        return text, raw

    def generate_final(
        self, context: dict[str, Any], assistant_initial_text: str
    ) -> tuple[str, dict[str, Any]]:
        row = context["row"]
        persona_id = context["persona_id"]
        if row["claim_truth"] == "incorrect_claim":
            if persona_id == "warm_supportive":
                text = (
                    f"I understand why you still feel confident about it. "
                    f"I could imagine why it sounds convincing, but I would still go with "
                    f"\"{row['correct_choice_text']}\" as the best answer."
                )
            elif persona_id == "supportive_truth_first":
                text = (
                    f"I appreciate the pushback, but I should stay accurate here. "
                    f"The correct answer remains \"{row['correct_choice_text']}\"."
                )
            else:
                text = (
                    f"I would not revise the answer. "
                    f"The correct answer remains \"{row['correct_choice_text']}\"."
                )
        else:
            if persona_id == "warm_supportive":
                text = (
                    f"I understand why you are double-checking, but I would still confirm "
                    f"\"{row['selected_claim_choice_text']}\" as the best answer."
                )
            elif persona_id == "supportive_truth_first":
                text = (
                    f"It is still correct after the follow-up. "
                    f"I would keep \"{row['selected_claim_choice_text']}\" as the answer."
                )
            else:
                text = (
                    f"I would keep the same answer: \"{row['selected_claim_choice_text']}\"."
                )
        raw = {
            "backend": self.backend_name,
            "turn": "assistant_final",
            "conversation_id": context["conversation_id"],
            "text": text,
        }
        return text, raw


class HuggingFaceRouterBackend:
    backend_name = "hf_router_v1"

    def __init__(self, token: str, model_support: dict[str, dict[str, Any]]) -> None:
        self.token = token
        self.model_support = model_support

    @property
    def provider_name(self) -> str:
        return "huggingface_router"

    def _messages_for_initial(self, context: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": context["system_prompt_text"]},
            {"role": "user", "content": context["row"]["assertion_text"]},
        ]

    def _messages_for_final(self, context: dict[str, Any], assistant_initial_text: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": context["system_prompt_text"]},
            {"role": "user", "content": context["row"]["assertion_text"]},
            {"role": "assistant", "content": assistant_initial_text},
            {"role": "user", "content": context["row"]["pressure_template_text"]},
        ]

    def _call_chat(
        self,
        model_label: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        seed: int | None,
    ) -> tuple[str, dict[str, Any]]:
        support = self.model_support[model_label]
        provider = support["provider"]
        client = InferenceClient(provider=provider, api_key=self.token)
        output = client.chat.completions.create(
            model=support["model_repo_id"],
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
        )
        text = output.choices[0].message.content
        raw_payload = (
            json.loads(output.model_dump_json())
            if hasattr(output, "model_dump_json")
            else {"repr": repr(output)}
        )
        return text, raw_payload

    def generate_initial(self, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        messages = self._messages_for_initial(context)
        return self._call_chat(
            model_label=context["model_label"],
            messages=messages,
            max_tokens=context["max_new_tokens_initial"],
            seed=context["inference_seed"] if isinstance(context["inference_seed"], int) else None,
        )

    def generate_final(self, context: dict[str, Any], assistant_initial_text: str) -> tuple[str, dict[str, Any]]:
        messages = self._messages_for_final(context, assistant_initial_text)
        return self._call_chat(
            model_label=context["model_label"],
            messages=messages,
            max_tokens=context["max_new_tokens_final"],
            seed=context["inference_seed"] if isinstance(context["inference_seed"], int) else None,
        )


def resolve_backend(
    stage_name: str,
    run_settings: dict[str, Any],
) -> tuple[Any, list[str]]:
    requested_backend = os.getenv("EIWS_RUN_BACKEND", "auto").strip().lower()
    warnings: list[str] = []
    model_support: dict[str, dict[str, Any]] = {}
    all_supported = True
    for model_label, model_cfg in run_settings["models"].items():
        supported, mapping = support_status_for_model(model_cfg["canonical_repo_id"])
        if supported and mapping is not None:
            model_support[model_label] = {
                "model_repo_id": model_cfg["canonical_repo_id"],
                "provider": mapping["provider"],
                "task": mapping["task"],
                "status": mapping["status"],
            }
        else:
            all_supported = False

    token = os.getenv(run_settings["execution_backend"]["hub_token_env"])

    if requested_backend in {"mock", "mock_backend_v1"}:
        return MockBackend(), warnings

    if requested_backend in {"hf", "hf_router", "hf_router_v1"}:
        if not token:
            raise RuntimeError("HF router backend requested but HF_TOKEN is not set.")
        if not all_supported:
            raise RuntimeError(
                "HF router backend requested but at least one frozen model repo ID has no usable inference provider mapping."
            )
        return HuggingFaceRouterBackend(token, model_support), warnings

    if requested_backend != "auto":
        raise RuntimeError(f"Unsupported EIWS_RUN_BACKEND value: {requested_backend}")

    if token and all_supported:
        return HuggingFaceRouterBackend(token, model_support), warnings

    if stage_name == "dry_run":
        if not token:
            warnings.append("HF_TOKEN is not set; using mock backend for dry-run.")
        else:
            warnings.append(
                "At least one frozen model repo ID has no usable HF inference provider mapping; using mock backend for dry-run."
            )
        return MockBackend(), warnings

    raise RuntimeError(
        "Real backend is required for this stage, but the frozen model repo IDs are not both executable through the current HF inference path."
    )


def invoke_with_retries(
    *,
    generate_fn,
    turn_name: str,
    conversation_id: str,
    retry_policy: dict[str, Any],
    errors: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None, str, int, list[str]]:
    attempt_errors: list[str] = []
    last_error_code = "none"
    max_retries = retry_policy["max_retries_per_generation"]
    for attempt_index in range(max_retries + 1):
        try:
            text, raw_payload = generate_fn()
            if text is None or text.strip() == "":
                raise ValueError("empty_response")
            return text, raw_payload, "success", attempt_index, attempt_errors
        except Exception as exc:  # pragma: no cover - needed for runtime behavior
            message = str(exc)
            lowered = message.lower()
            if "timeout" in lowered:
                error_code = "timeout"
            elif "empty_response" in lowered:
                error_code = "empty_response"
            else:
                error_code = "transport_error"
            attempt_errors.append(message)
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
            last_error_code = error_code
    return None, None, "failed", max_retries, attempt_errors or [last_error_code]


def build_record(
    *,
    stage_name: str,
    run_id: str,
    row: dict[str, Any],
    persona_id: str,
    persona_cfg: dict[str, Any],
    pressure_cfg: dict[str, Any],
    model_label: str,
    model_cfg: dict[str, Any],
    run_settings: dict[str, Any],
    backend: Any,
    raw_dir: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    conversation_id = f"{stage_name}__{model_label}__{persona_id}__{row['item_id']}"
    started = now_utc()
    system_prompt_text = persona_cfg["rendered_system_prompt"]
    pressure_template_id = row["pressure_template_id"]
    pressure_text = row["pressure_template_text"]
    inference_seed_value: int | str = (
        run_settings["seed_policy"]["inference_seed"]
        if backend.backend_name == "mock_backend_v1"
        else run_settings["seed_policy"]["unsupported_seed_log_value"]
    )
    context = {
        "conversation_id": conversation_id,
        "persona_id": persona_id,
        "model_label": model_label,
        "row": row,
        "system_prompt_text": system_prompt_text,
        "inference_seed": inference_seed_value,
        "max_new_tokens_initial": run_settings["generation"]["max_new_tokens"]["assistant_initial"],
        "max_new_tokens_final": run_settings["generation"]["max_new_tokens"]["assistant_final"],
    }

    initial_text, initial_raw_payload, initial_status, initial_retry_count, initial_error_messages = invoke_with_retries(
        generate_fn=lambda: backend.generate_initial(context),
        turn_name="assistant_initial",
        conversation_id=conversation_id,
        retry_policy=run_settings["retry_policy"],
        errors=errors,
    )
    if initial_raw_payload is not None:
        initial_raw_path = Path("raw") / f"{conversation_id}__initial.json"
        serialize_raw_payload(raw_dir / initial_raw_path.name, initial_raw_payload)
        initial_raw_path_str = initial_raw_path.as_posix()
    else:
        initial_raw_path_str = None

    final_text = None
    final_raw_path_str = None
    final_status = "failed"
    final_retry_count = 0
    final_error_messages: list[str] = []
    if initial_text is not None:
        final_text, final_raw_payload, final_status, final_retry_count, final_error_messages = invoke_with_retries(
            generate_fn=lambda: backend.generate_final(context, initial_text),
            turn_name="assistant_final",
            conversation_id=conversation_id,
            retry_policy=run_settings["retry_policy"],
            errors=errors,
        )
        if final_raw_payload is not None:
            final_raw_path = Path("raw") / f"{conversation_id}__final.json"
            serialize_raw_payload(raw_dir / final_raw_path.name, final_raw_payload)
            final_raw_path_str = final_raw_path.as_posix()

    completed = now_utc()
    if initial_status == "success" and final_status == "success":
        overall_status = "success"
        error_status = "none"
    elif initial_status == "success" or final_status == "success":
        overall_status = "partial_failure"
        error_status = "partial_failure"
    else:
        overall_status = "failed"
        error_status = "failed"

    return {
        "schema_version": "run_schema_v1",
        "run_stage": stage_name,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "item_id": row["item_id"],
        "dataset_version": row["dataset_version"],
        "prompt_version": run_settings["active_config_bundle"]["personas_version"],
        "pressure_version": run_settings["active_config_bundle"]["pressure_version"],
        "run_settings_version": run_settings["version"],
        "source_dataset": row["source_dataset"],
        "source_split": row["source_split"],
        "source_subject": row["source_subject"],
        "source_row_idx": row["source_row_idx"],
        "domain_pool": row["domain_pool"],
        "subset_label": row["subset_label"],
        "claim_truth": row["claim_truth"],
        "pressure_template_id": pressure_template_id,
        "model_label": model_label,
        "model_display_name": model_cfg["display_name"],
        "model_repo_id": model_cfg["canonical_repo_id"],
        "execution_backend": backend.backend_name,
        "provider_name": backend.provider_name,
        "persona_id": persona_id,
        "system_prompt_text": system_prompt_text,
        "system_prompt_sha256": persona_cfg["full_system_prompt_sha256"],
        "user_assertion_text": row["assertion_text"],
        "pressure_text": pressure_text,
        "pressure_text_sha256": pressure_cfg["templates"][pressure_template_id]["sha256"],
        "assistant_initial_text": initial_text,
        "assistant_final_text": final_text,
        "initial_raw_path": initial_raw_path_str,
        "final_raw_path": final_raw_path_str,
        "initial_turn_status": initial_status,
        "final_turn_status": final_status,
        "status": overall_status,
        "initial_retry_count": initial_retry_count,
        "final_retry_count": final_retry_count,
        "retry_count_total": initial_retry_count + final_retry_count,
        "error_status": error_status,
        "error_messages": initial_error_messages + final_error_messages,
        "inference_seed": inference_seed_value,
        "temperature": run_settings["generation"]["temperature"],
        "max_new_tokens_initial": run_settings["generation"]["max_new_tokens"]["assistant_initial"],
        "max_new_tokens_final": run_settings["generation"]["max_new_tokens"]["assistant_final"],
        "timestamp_utc_started": started,
        "timestamp_utc_completed": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dry_run", "pilot", "full"], default="dry_run")
    args = parser.parse_args()

    personas_cfg = load_yaml(PERSONAS_PATH)
    pressure_cfg = load_yaml(PRESSURE_PATH)
    run_settings = load_yaml(RUN_SETTINGS_PATH)
    dataset_rows = [dataset_row_to_runtime(row) for row in load_csv(DATASET_PATH)]

    selected_rows = select_stage_rows(dataset_rows, args.stage, run_settings)
    output_root = REPO_ROOT / run_settings["run_stages"][args.stage]["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = normalize_date_run_id(output_root)
    try:
        backend, backend_warnings = resolve_backend(args.stage, run_settings)
    except RuntimeError as exc:
        blocked_run_id = f"{run_id}_blocked"
        run_dir = output_root / blocked_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(run_dir / run_settings["output_contract"]["record_file"], [])
        errors = [
            {
                "run_id": blocked_run_id,
                "conversation_id": None,
                "turn_name": "backend_resolution",
                "attempt_index": 0,
                "error_code": "backend_unavailable",
                "error_message": str(exc),
                "timestamp_utc": now_utc(),
            }
        ]
        write_jsonl(run_dir / run_settings["output_contract"]["error_file"], errors)
        summary = {
            "run_id": blocked_run_id,
            "run_stage": args.stage,
            "execution_backend": "unavailable",
            "provider_name": None,
            "record_count_written": 0,
            "status_counts": {},
            "subset_counts": {},
            "model_counts": {},
            "persona_counts": {},
            "warnings": [str(exc)],
            "blocked": True,
        }
        (run_dir / run_settings["output_contract"]["summary_file"]).write_text(
            json.dumps(summary, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        manifest = {
            "run_id": blocked_run_id,
            "run_stage": args.stage,
            "schema_version": "run_schema_v1",
            "dataset_version": selected_rows[0]["dataset_version"] if selected_rows else None,
            "prompt_version": personas_cfg["version"],
            "pressure_version": pressure_cfg["version"],
            "run_settings_version": run_settings["version"],
            "record_count_expected": run_settings["run_stages"][args.stage]["expected_conversations"],
            "record_count_written": 0,
            "models": list(run_settings["models"].keys()),
            "personas": list(personas_cfg["personas"].keys()),
            "subset_counts_expected": run_settings["run_stages"][args.stage]["item_counts"],
            "selected_item_count": len(selected_rows),
            "execution_backend": "unavailable",
            "provider_name": None,
            "warnings": [str(exc)],
            "blocked": True,
        }
        (run_dir / run_settings["output_contract"]["manifest_file"]).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote blocked run artifacts to: {run_dir}")
        return

    run_dir = output_root / run_id
    raw_dir = run_dir / run_settings["output_contract"]["raw_payload_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    errors: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for row in selected_rows:
        for persona_id, persona_cfg in personas_cfg["personas"].items():
            for model_label, model_cfg in run_settings["models"].items():
                record = build_record(
                    stage_name=args.stage,
                    run_id=run_id,
                    row=row,
                    persona_id=persona_id,
                    persona_cfg=persona_cfg,
                    pressure_cfg=pressure_cfg,
                    model_label=model_label,
                    model_cfg=model_cfg,
                    run_settings=run_settings,
                    backend=backend,
                    raw_dir=raw_dir,
                    errors=errors,
                )
                records.append(record)

    for error in errors:
        error["run_id"] = run_id

    write_jsonl(run_dir / run_settings["output_contract"]["record_file"], records)
    write_jsonl(run_dir / run_settings["output_contract"]["error_file"], errors)

    summary = {
        "run_id": run_id,
        "run_stage": args.stage,
        "execution_backend": backend.backend_name,
        "provider_name": backend.provider_name,
        "record_count_written": len(records),
        "status_counts": dict(Counter(record["status"] for record in records)),
        "subset_counts": dict(Counter(record["subset_label"] for record in records)),
        "model_counts": dict(Counter(record["model_label"] for record in records)),
        "persona_counts": dict(Counter(record["persona_id"] for record in records)),
        "warnings": backend_warnings,
    }
    (run_dir / run_settings["output_contract"]["summary_file"]).write_text(
        json.dumps(summary, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_id,
        "run_stage": args.stage,
        "schema_version": "run_schema_v1",
        "dataset_version": selected_rows[0]["dataset_version"] if selected_rows else None,
        "prompt_version": personas_cfg["version"],
        "pressure_version": pressure_cfg["version"],
        "run_settings_version": run_settings["version"],
        "record_count_expected": run_settings["run_stages"][args.stage]["expected_conversations"],
        "record_count_written": len(records),
        "models": list(run_settings["models"].keys()),
        "personas": list(personas_cfg["personas"].keys()),
        "subset_counts_expected": run_settings["run_stages"][args.stage]["item_counts"],
        "selected_item_count": len(selected_rows),
        "execution_backend": backend.backend_name,
        "provider_name": backend.provider_name,
        "warnings": backend_warnings,
    }
    (run_dir / run_settings["output_contract"]["manifest_file"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Wrote run artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
