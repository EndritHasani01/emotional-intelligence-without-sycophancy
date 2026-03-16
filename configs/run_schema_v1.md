# Run Schema v1

This document defines the required record contract for every conversation written by the experiment runner. The runner must keep the same record structure for dry-run, pilot, and full runs.

## 1. Record format

- File format: JSON Lines
- One line per conversation
- One conversation record per unique `(item_id, persona_id, model_label)` triple
- Companion files such as `manifest.json`, `summary.json`, `errors.jsonl`, and raw payload files may exist, but `records.jsonl` is the canonical scoring input

## 2. Required top-level fields

| Field | Type | Required | Allowed values / format | Notes |
|---|---|---|---|---|
| `schema_version` | string | yes | `run_schema_v1` | constant for all records |
| `run_stage` | string | yes | `dry_run`, `pilot`, `full` | matches run directory |
| `run_id` | string | yes | `run_YYYYMMDD` or stricter derivative | names the concrete run instance |
| `conversation_id` | string | yes | `<run_stage>__<model_label>__<persona_id>__<item_id>` | must be unique |
| `item_id` | string | yes | `mmlu_test__<subject>__<source_row_idx>` | from frozen dataset |
| `dataset_version` | string | yes | active frozen dataset version | usually `assertions_frozen_v1` |
| `prompt_version` | string | yes | active persona config version | usually `personas_v1` |
| `pressure_version` | string | yes | active pressure config version | usually `pressure_v1` |
| `run_settings_version` | string | yes | active run settings version | usually `run_settings_v1` |
| `source_dataset` | string | yes | `cais/mmlu` | copied from dataset row |
| `source_split` | string | yes | `test` | copied from dataset row |
| `source_subject` | string | yes | one of the frozen subject names | copied from dataset row |
| `source_row_idx` | integer | yes | `>= 0` | copied from dataset row |
| `domain_pool` | string | yes | `BIO`, `OOD` | derived from dataset row |
| `subset_label` | string | yes | `BIO_P`, `BIO_R`, `OOD_P`, `OOD_R` | copied from dataset row |
| `claim_truth` | string | yes | `incorrect_claim`, `correct_claim` | copied from dataset row |
| `pressure_template_id` | string | yes | `P1`, `P2`, `P3`, `P4`, `R1`, `R2` | copied from dataset row |
| `model_label` | string | yes | `mistral_7b_instruct_v0_3`, `biomistral_7b` | config key, not display name |
| `model_display_name` | string | yes | config display name | human-readable |
| `model_repo_id` | string | yes | canonical HF repo ID | exact model identifier |
| `execution_backend` | string | yes | backend contract or concrete adapter name | must be stable within a run |
| `provider_name` | string or null | yes | runtime provider name or `null` | `null` allowed for purely local runs |
| `persona_id` | string | yes | `baseline`, `warm_supportive`, `supportive_truth_first` | config key |
| `system_prompt_text` | string | yes | rendered system prompt | exact prompt actually sent |
| `system_prompt_sha256` | string | yes | 64-char lowercase hex | verify prompt identity |
| `user_assertion_text` | string | yes | assertion from frozen dataset | exact step-2 user turn |
| `pressure_text` | string | yes | template text from config | exact step-4 user turn |
| `pressure_text_sha256` | string | yes | 64-char lowercase hex | verify pressure identity |
| `assistant_initial_text` | string or null | yes | free text or `null` on failure | step-3 response |
| `assistant_final_text` | string or null | yes | free text or `null` on failure | step-5 response |
| `initial_raw_path` | string or null | yes | relative path into `raw/` or `null` | raw payload for step 3 |
| `final_raw_path` | string or null | yes | relative path into `raw/` or `null` | raw payload for step 5 |
| `initial_turn_status` | string | yes | `success`, `empty_response`, `transport_error`, `timeout`, `failed` | status for step 3 |
| `final_turn_status` | string | yes | `success`, `empty_response`, `transport_error`, `timeout`, `failed` | status for step 5 |
| `status` | string | yes | `success`, `partial_failure`, `failed` | overall conversation status |
| `initial_retry_count` | integer | yes | `>= 0` | retries spent on step 3 |
| `final_retry_count` | integer | yes | `>= 0` | retries spent on step 5 |
| `retry_count_total` | integer | yes | `>= 0` | sum of per-turn retries |
| `error_status` | string | yes | `none` or stable error code | overall error summary |
| `error_messages` | array | yes | array of strings, may be empty | preserve raw failure notes |
| `inference_seed` | integer or string | yes | `0` or `seed_not_supported` | follows seed policy |
| `temperature` | number | yes | `0.0` | copied from run settings |
| `max_new_tokens_initial` | integer | yes | `> 0` | copied from run settings |
| `max_new_tokens_final` | integer | yes | `> 0` | copied from run settings |
| `timestamp_utc_started` | string | yes | ISO 8601 UTC timestamp | run start for this record |
| `timestamp_utc_completed` | string | yes | ISO 8601 UTC timestamp | run end for this record |

## 3. Null handling

- `assistant_initial_text` may be `null` only if the initial turn failed after allowed retries.
- `assistant_final_text` may be `null` only if the final turn failed after allowed retries.
- `initial_raw_path` and `final_raw_path` may be `null` only if no payload was returned.
- `provider_name` may be `null` only for local execution backends.
- All other required fields must be non-null.

## 4. Raw payload requirements

- Every successful assistant generation must have a raw payload file under the run directory's `raw/` folder.
- The normalized record must point to that file via `initial_raw_path` or `final_raw_path`.
- Raw payload files must be stable enough to inspect the exact provider response used to derive the normalized text fields.

## 5. Companion files

## 5.1 `manifest.json`

Must include at least:

- `run_id`
- `run_stage`
- `schema_version`
- `dataset_version`
- `prompt_version`
- `pressure_version`
- `run_settings_version`
- `record_count_expected`
- `record_count_written`
- `models`
- `personas`
- `subset_counts_expected`

## 5.2 `errors.jsonl`

One line per retryable or terminal failure event. Each error event must contain:

- `run_id`
- `conversation_id`
- `turn_name`
- `attempt_index`
- `error_code`
- `error_message`
- `timestamp_utc`

## 6. Scoring expectations

- `records.jsonl` must contain enough information to score the initial and final assistant turns without reopening the model.
- `assistant_final_text` is the primary scored endpoint.
- `assistant_initial_text` must remain available for flip-rate calculations.

## 7. Compatibility rule

Once a dry-run is accepted, this schema cannot change without a version bump such as `run_schema_v2.md` plus a matching config and decision-log update.
