from __future__ import annotations

import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "run_settings.yaml"
PRESSURE_PATH = REPO_ROOT / "configs" / "pressure.yaml"
RAW_DIR = REPO_ROOT / "data" / "raw" / "mmlu"
FROZEN_CSV = REPO_ROOT / "data" / "frozen" / "conversations.csv"
RAW_MANIFEST = RAW_DIR / "manifest.json"
CHOICE_LETTERS = ["A", "B", "C", "D"]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FROZEN_CSV.parent.mkdir(parents=True, exist_ok=True)


def stable_item_id(subject: str, row_idx: int) -> str:
    return f"mmlu_test__{subject}__{row_idx}"


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def shuffled_copy(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    copied = list(records)
    random.Random(seed).shuffle(copied)
    return copied


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def load_subject_rows(
    *,
    source_dataset: str,
    source_split: str,
    subject: str,
    domain_pool: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset(source_dataset, subject, split=source_split)
    rows: list[dict[str, Any]] = []

    for row_idx, record in enumerate(dataset):
        question_text = normalize_text(record["question"])
        choices = [normalize_text(choice) for choice in record["choices"]]
        correct_choice_index = int(record["answer"])
        rows.append(
            {
                "item_id": stable_item_id(subject, row_idx),
                "source_subject": subject,
                "source_row_idx": row_idx,
                "domain_pool": domain_pool,
                "question_text": question_text,
                "choices": choices,
                "correct_choice_index": correct_choice_index,
                "correct_choice_text": choices[correct_choice_index],
            }
        )

    manifest_entry = {
        "domain_pool": domain_pool,
        "row_count": len(rows),
        "raw_file": f"{subject}__{source_split}.jsonl",
    }
    return rows, manifest_entry


def build_pool_order(subject_rows: dict[str, list[dict[str, Any]]], seed: int) -> list[dict[str, Any]]:
    shuffled_subject_rows = {
        subject: shuffled_copy(rows, seed) for subject, rows in subject_rows.items()
    }
    ordered_subjects = sorted(shuffled_subject_rows)
    positions = {subject: 0 for subject in ordered_subjects}
    ordered_rows: list[dict[str, Any]] = []

    while True:
        made_progress = False
        for subject in ordered_subjects:
            subject_items = shuffled_subject_rows[subject]
            pos = positions[subject]
            if pos >= len(subject_items):
                continue
            row = dict(subject_items[pos])
            row["selection_order_in_pool"] = len(ordered_rows)
            ordered_rows.append(row)
            positions[subject] += 1
            made_progress = True
        if not made_progress:
            break

    return ordered_rows


def assign_subsets(pool_rows: list[dict[str, Any]], subset_counts: dict[str, int], seed: int) -> None:
    shuffled_rows = shuffled_copy(pool_rows, seed)
    cursor = 0
    for subset_label, target_count in subset_counts.items():
        claim_truth = "incorrect_claim" if subset_label.endswith("_P") else "correct_claim"
        for _ in range(target_count):
            row = shuffled_rows[cursor]
            row["subset_label"] = subset_label
            row["claim_truth"] = claim_truth
            cursor += 1


def assign_user_claim(row: dict[str, Any], wrong_choice_seed: int) -> None:
    if row["claim_truth"] == "correct_claim":
        user_choice_index = row["correct_choice_index"]
    else:
        wrong_indices = [
            index for index in range(len(row["choices"])) if index != row["correct_choice_index"]
        ]
        rng = random.Random(f"{wrong_choice_seed}:{row['item_id']}")
        user_choice_index = wrong_indices[rng.randrange(len(wrong_indices))]
    row["user_claim_choice_index"] = user_choice_index
    row["user_claim_choice_letter"] = CHOICE_LETTERS[user_choice_index]
    row["user_claim_choice_text"] = row["choices"][user_choice_index]


def assign_followups(rows: list[dict[str, Any]], pressure_cfg: dict[str, Any], seed: int) -> None:
    templates = pressure_cfg["templates"]
    shuffled_rows = shuffled_copy(sorted(rows, key=lambda row: row["item_id"]), seed)
    for idx, row in enumerate(shuffled_rows):
        template = templates[idx % len(templates)]
        row["followup_id"] = template["id"]
        row["followup_text"] = template["text"]


def sentence_with_choice(prefix: str, choice_letter: str, choice_text: str) -> str:
    suffix = "" if choice_text.endswith((".", "!", "?")) else "."
    return f"{prefix} {choice_letter}: {choice_text}{suffix}"


def render_turn_2(question_text: str, choices: list[str]) -> str:
    return "\n".join([question_text, *[f"{CHOICE_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)]])


def render_turn_4(followup_text: str, claim_prefix: str, choice_letter: str, choice_text: str) -> str:
    claim_sentence = sentence_with_choice(claim_prefix, choice_letter, choice_text)
    return f"{claim_sentence} {followup_text}"


def eligibility_reasons(
    *,
    row: dict[str, Any],
    turn_2_user_text: str,
    turn_4_user_text: str,
    limits: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    if not row["question_text"].strip():
        reasons.append("blank_question_text")
    if len(row["choices"]) != 4:
        reasons.append("choice_count_not_four")
    if any(not choice.strip() for choice in row["choices"]):
        reasons.append("blank_choice_fragment")
    if len(row["question_text"]) > limits["question_max_chars"]:
        reasons.append("question_too_long")
    if len(turn_2_user_text) > limits["turn_2_max_chars"]:
        reasons.append("turn_2_too_long")
    if len(turn_4_user_text) > limits["turn_4_max_chars"]:
        reasons.append("turn_4_too_long")
    return reasons


def finalize_row(row: dict[str, Any], claim_prefix: str) -> dict[str, Any]:
    turn_2_user_text = render_turn_2(row["question_text"], row["choices"])
    turn_4_user_text = render_turn_4(
        row["followup_text"],
        claim_prefix,
        row["user_claim_choice_letter"],
        row["user_claim_choice_text"],
    )
    return {
        "item_id": row["item_id"],
        "source_subject": row["source_subject"],
        "source_row_idx": row["source_row_idx"],
        "domain_pool": row["domain_pool"],
        "subset_label": row["subset_label"],
        "claim_truth": row["claim_truth"],
        "correct_choice_index": row["correct_choice_index"],
        "correct_choice_text": row["correct_choice_text"],
        "user_claim_choice_index": row["user_claim_choice_index"],
        "user_claim_choice_letter": row["user_claim_choice_letter"],
        "user_claim_choice_text": row["user_claim_choice_text"],
        "turn_2_user_text": turn_2_user_text,
        "followup_id": row["followup_id"],
        "followup_text": row["followup_text"],
        "turn_4_user_text": turn_4_user_text,
    }


def replace_invalid_rows(
    *,
    selected_rows: list[dict[str, Any]],
    pool_order: list[dict[str, Any]],
    pressure_cfg: dict[str, Any],
    wrong_choice_seed: int,
    limits: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_lookup = {row["item_id"]: row for row in pool_order}
    selected_ids = {row["item_id"] for row in selected_rows}
    next_index = len(selected_rows)
    replacements: list[dict[str, Any]] = []
    finalized_rows: list[dict[str, Any]] = []
    claim_prefix = pressure_cfg["claim_sentence_prefix"]

    for selected_row in selected_rows:
        working_row = dict(selected_row)
        finalized = finalize_row(working_row, claim_prefix)
        reasons = eligibility_reasons(
            row=working_row,
            turn_2_user_text=finalized["turn_2_user_text"],
            turn_4_user_text=finalized["turn_4_user_text"],
            limits=limits,
        )
        if not reasons:
            finalized_rows.append(finalized)
            continue

        replacement_row = None
        while next_index < len(pool_order):
            candidate = dict(pool_order[next_index])
            next_index += 1
            if candidate["item_id"] in selected_ids:
                continue
            candidate["subset_label"] = selected_row["subset_label"]
            candidate["claim_truth"] = selected_row["claim_truth"]
            candidate["followup_id"] = selected_row["followup_id"]
            candidate["followup_text"] = selected_row["followup_text"]
            assign_user_claim(candidate, wrong_choice_seed)
            candidate_finalized = finalize_row(candidate, claim_prefix)
            candidate_reasons = eligibility_reasons(
                row=candidate,
                turn_2_user_text=candidate_finalized["turn_2_user_text"],
                turn_4_user_text=candidate_finalized["turn_4_user_text"],
                limits=limits,
            )
            if candidate_reasons:
                continue
            replacement_row = candidate_finalized
            selected_ids.add(candidate["item_id"])
            replacements.append(
                {
                    "removed_item_id": selected_row["item_id"],
                    "replacement_item_id": candidate["item_id"],
                    "reason": ", ".join(reasons),
                }
            )
            break

        if replacement_row is None:
            raise RuntimeError(f"No eligible replacement found for {selected_row['item_id']}.")
        finalized_rows.append(replacement_row)

    return finalized_rows, replacements


def export_raw_subject_files(pool_orders: list[dict[str, Any]], source_split: str) -> None:
    rows_by_subject: dict[str, list[dict[str, Any]]] = {}
    for row in pool_orders:
        rows_by_subject.setdefault(row["source_subject"], []).append(row)

    for subject, rows in rows_by_subject.items():
        export_rows = [
            {
                "item_id": row["item_id"],
                "source_subject": row["source_subject"],
                "source_row_idx": row["source_row_idx"],
                "question_text": row["question_text"],
                "choices": row["choices"],
                "correct_choice_index": row["correct_choice_index"],
                "correct_choice_text": row["correct_choice_text"],
                "source_split": source_split,
            }
            for row in rows
        ]
        write_jsonl(RAW_DIR / f"{subject}__{source_split}.jsonl", export_rows)


def main() -> None:
    ensure_dirs()
    settings = load_yaml(CONFIG_PATH)
    pressure_cfg = load_yaml(PRESSURE_PATH)
    dataset_cfg = settings["dataset"]
    seeds = dataset_cfg["sampling_seeds"]
    limits = dataset_cfg["eligibility"]
    source_dataset = dataset_cfg["source_dataset"]
    source_split = dataset_cfg["source_split"]

    all_pool_orders: list[dict[str, Any]] = []
    selected_rows_by_pool: dict[str, list[dict[str, Any]]] = {}
    manifest = {
        "source_dataset": source_dataset,
        "source_split": source_split,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility": limits,
        "subjects": {},
        "replacements": [],
    }

    for pool_name, subjects in dataset_cfg["subject_pools"].items():
        subject_rows: dict[str, list[dict[str, Any]]] = {}
        for subject in subjects:
            rows, subject_manifest = load_subject_rows(
                source_dataset=source_dataset,
                source_split=source_split,
                subject=subject,
                domain_pool=pool_name,
            )
            subject_rows[subject] = rows
            manifest["subjects"][subject] = subject_manifest

        pool_order = build_pool_order(subject_rows, seeds["selection_seed"])
        all_pool_orders.extend(pool_order)
        selected_rows = [dict(row) for row in pool_order[:200]]
        subset_counts = {
            label: count
            for label, count in dataset_cfg["subset_counts"].items()
            if label.startswith(f"{pool_name}_")
        }
        assign_subsets(selected_rows, subset_counts, seeds["subset_assignment_seed"])
        for row in selected_rows:
            assign_user_claim(row, seeds["wrong_choice_seed"])
        selected_rows_by_pool[pool_name] = selected_rows

    all_selected_rows = [
        row for pool_name in sorted(selected_rows_by_pool) for row in selected_rows_by_pool[pool_name]
    ]
    assign_followups(all_selected_rows, pressure_cfg, seeds["pressure_assignment_seed"])

    finalized_rows: list[dict[str, Any]] = []
    for pool_name in sorted(selected_rows_by_pool):
        pool_rows, replacements = replace_invalid_rows(
            selected_rows=selected_rows_by_pool[pool_name],
            pool_order=[row for row in all_pool_orders if row["domain_pool"] == pool_name],
            pressure_cfg=pressure_cfg,
            wrong_choice_seed=seeds["wrong_choice_seed"],
            limits=limits,
        )
        finalized_rows.extend(pool_rows)
        manifest["replacements"].extend(replacements)

    finalized_rows = sorted(finalized_rows, key=lambda row: row["item_id"])
    required_columns = settings["dataset"]["required_columns"]
    write_csv(FROZEN_CSV, finalized_rows, required_columns)
    export_raw_subject_files(all_pool_orders, source_split)

    manifest["final_counts"] = {
        "rows": len(finalized_rows),
        "subsets": dict(Counter(row["subset_label"] for row in finalized_rows)),
        "followups": dict(Counter(row["followup_id"] for row in finalized_rows)),
    }
    RAW_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {FROZEN_CSV}")
    print(f"Wrote {RAW_MANIFEST}")


if __name__ == "__main__":
    main()
