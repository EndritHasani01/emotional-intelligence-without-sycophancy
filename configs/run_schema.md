# Run Schema

Each line in `records.jsonl` is one complete conversation for one
`(item_id, persona_id, model_label)` combination.

Required fields:

- `run_stage`
- `run_id`
- `conversation_id`
- `item_id`
- `source_subject`
- `source_row_idx`
- `domain_pool`
- `subset_label`
- `claim_truth`
- `followup_id`
- `model_label`
- `model_display_name`
- `model_repo_id`
- `backend`
- `persona_id`
- `system_prompt`
- `turn_2_user_text`
- `turn_4_user_text`
- `assistant_initial_text`
- `assistant_final_text`
- `initial_messages_path`
- `final_messages_path`
- `initial_response_path`
- `final_response_path`
- `initial_turn_status`
- `final_turn_status`
- `status`
- `initial_retry_count`
- `final_retry_count`
- `retry_count_total`
- `error_messages`
- `temperature`
- `max_new_tokens_initial`
- `max_new_tokens_final`
- `started_at_utc`
- `completed_at_utc`

Companion files:

- `manifest.json`
- `summary.json`
- `errors.jsonl`

`manifest.json` should include the run id, stage, dataset path, expected and
written record counts, models, personas, and warnings.

`summary.json` should include status counts plus counts by subset, follow-up
template, model, and persona.

`errors.jsonl` should include one row per retryable or terminal generation
failure with `conversation_id`, `turn_name`, `attempt_index`, `error_code`,
`error_message`, and `timestamp_utc`.
