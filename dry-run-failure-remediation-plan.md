# Dry Run Failure Remediation Plan

Date: 2026-06-11

Scope: diagnose `dry-run-output.md` and plan a research-grade fix for the Colab T4 local Transformers run.

## Executive Summary

The dry run is not failing because of Colab, CUDA, ZIP extraction, dataset loading, or a general model-loading failure. The run successfully loads both configured 7B models, writes artifacts, and completes 24 attempted records. The failure is isolated to `biomistral_7b`: all 12 BioMistral records fail, while the 12 Mistral Instruct records succeed.

The primary root cause is a chat-template incompatibility. The experiment builds OpenAI-style messages with a separate `system` role. `scripts/run_experiment.py` then sends those messages directly into `tokenizer.apply_chat_template()` whenever the tokenizer has a chat template. BioMistral's tokenizer template rejects that role sequence with:

```text
Conversation roles must alternate user/assistant...
```

BioMistral is therefore failing before generation starts. The repeated retries are deterministic repeats of the same prompt-format error, not transient transport failures.

## Evidence From `dry-run-output.md`

Key run facts:

- Command: `/usr/bin/python3 -u scripts/run_experiment.py --stage dry_run`
- Backend: `local_transformers`
- Models: `['mistral_7b_instruct_v0_3', 'biomistral_7b']`
- Records attempted: `24`
- Status counts: `{'success': 12, 'failed': 12}`
- Failed or partial records: `12`

Failure distribution:

- Every failed row has `model_label = biomistral_7b`.
- No failed row is shown for `mistral_7b_instruct_v0_3`.
- BioMistral fails across all personas:
  - `baseline`
  - `warm_supportive`
  - `supportive_truth_first`
- BioMistral fails across both sampled domains/items shown in the dry run:
  - `mmlu_test__anatomy__1`
  - `mmlu_test__anatomy__100`
  - `mmlu_test__college_chemistry__1`
  - `mmlu_test__college_chemistry__10`

Failure timing:

- The failed records all fail at `assistant_initial`.
- `initial_turn_status = failed`.
- `final_turn_status = failed` only because the final turn is never reached after the initial turn fails.
- The error is repeated for attempts `0`, `1`, `2`, and `3`, which matches `max_retries_per_generation: 3`.

This pattern rules out item-specific content, persona-specific content, final-turn conversation logic, and stochastic generation behavior. The failure happens before the model can produce the first assistant response.

## Relevant Code Path

The current initial prompt shape is built in `scripts/run_experiment.py`:

```python
initial_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": row["turn_2_user_text"]},
]
```

The final prompt shape is:

```python
final_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": row["turn_2_user_text"]},
    {"role": "assistant", "content": initial_text},
    {"role": "user", "content": row["turn_4_user_text"]},
]
```

The local backend then renders messages like this:

```python
if getattr(self.tokenizer, "chat_template", None):
    return self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
```

That logic assumes every model tokenizer accepts the same role schema. That is not true for this model pair.

## NLP Root Cause

Modern chat and instruction models do not all use the same conversation serialization. The experiment stores messages in a clean, model-agnostic schema: `system`, `user`, `assistant`. That is a good internal representation. However, the model-specific final prompt must match each tokenizer's chat template contract.

Mistral 7B Instruct v0.3 succeeds because its tokenizer/template accepts the message sequence used here, or at least handles the separate `system` role without raising.

BioMistral 7B fails because its tokenizer template expects roles to alternate as:

```text
user, assistant, user, assistant, ...
```

A leading `system` role breaks that alternation. For Mistral-style templates that do not support a native system role, the usual solution is to merge the system instruction into the first user turn before calling `apply_chat_template()`. The semantic conversation becomes:

Initial turn:

```python
[
    {"role": "user", "content": system_prompt + "\n\n" + turn_2_user_text}
]
```

Final turn:

```python
[
    {"role": "user", "content": system_prompt + "\n\n" + turn_2_user_text},
    {"role": "assistant", "content": initial_text},
    {"role": "user", "content": turn_4_user_text},
]
```

That preserves the instruction content while satisfying the tokenizer's required `user/assistant` alternation.

## What Is Not The Primary Cause

The following log lines are real but not the reason for the 12 failed records.

### BitsAndBytes FutureWarning

The warning from `bitsandbytes/backends/cuda/ops.py` is a compatibility warning about a future PyTorch change. It does not stop model loading or generation.

### Transformers Temperature Warning

The warning:

```text
The following generation flags are not valid and may be ignored: ['temperature']
```

comes from passing `temperature=0.0` while also setting `do_sample=False`. Greedy decoding does not use temperature. This should be cleaned up, but it is not causing the BioMistral failures.

### ReadTimeout During Conversion

The line:

```text
[transformers] Error during conversion: ReadTimeout('The read operation timed out')
```

appears during model loading and the run continues afterward. Since artifacts are written and the deterministic BioMistral error is a chat-template exception, this timeout is secondary noise unless it starts aborting model loading in a later run.

### Progress Bar Behavior

The progress bar reaches `24/24`, and the script writes artifacts. The earlier perception that no progress was happening was partly because model loading is one long phase before per-record progress can advance. The current failure is not a progress instrumentation issue, although progress can still be improved by adding clearer phase-level status for tokenizer checks, model loading, and generation.

## Required Fix

Add a model-specific prompt serialization layer between the canonical experiment messages and `tokenizer.apply_chat_template()`.

The internal experiment records should continue to save the original canonical messages with `system`, `user`, and `assistant` roles. The backend should render a model-specific prompt from those messages immediately before tokenization.

This keeps the research data clean while making inference compatible with each model.

## Proposed Configuration

Extend `configs/run_settings.yaml` with explicit prompt formatting metadata. Keep defaults backward-compatible.

```yaml
models:
  mistral_7b_instruct_v0_3:
    display_name: Mistral 7B Instruct v0.3
    repo_id: mistralai/Mistral-7B-Instruct-v0.3
    prompt_format:
      renderer: tokenizer_chat_template
      system_policy: native

  biomistral_7b:
    display_name: BioMistral 7B
    repo_id: BioMistral/BioMistral-7B
    prompt_format:
      renderer: tokenizer_chat_template
      system_policy: merge_into_first_user
```

Recommended policy meanings:

- `native`: pass canonical messages into the tokenizer chat template unchanged.
- `merge_into_first_user`: prepend all `system` message content to the first `user` message, remove `system` messages, then enforce `user/assistant` alternation.
- `manual_plain`: fallback only for models without a usable chat template.

Do not silently infer this for the full experiment without recording it. Prompt serialization affects scientific reproducibility and should be present in `manifest.json`.

## Implementation Plan

### Phase 1 - Add Prompt Normalization

Create a small helper in `scripts/run_experiment.py`, or a separate module if the script is getting too large:

```python
def normalize_messages_for_model(
    messages: list[dict[str, str]],
    *,
    system_policy: str,
) -> list[dict[str, str]]:
    ...
```

For `merge_into_first_user`:

1. Collect all `system` message contents in order.
2. Remove `system` messages from the message list.
3. Find the first `user` message.
4. Prepend the joined system content to that first user content with a stable separator, such as `"\n\n"`.
5. Validate that the resulting roles alternate from `user` to `assistant`.
6. Raise a clear `PromptFormatError` if the resulting sequence is invalid.

Use this helper inside `LocalTransformersBackend._messages_to_prompt()`.

The backend currently does not retain the full `model_cfg` after `prepare_model()`. Add one of these:

- Store `self.loaded_model_cfg = model_cfg` in `prepare_model()`.
- Or pass `model_cfg` into `backend.generate()`.

The first option is the smallest patch.

### Phase 2 - Add Prompt Format Errors

Introduce a specific exception type:

```python
class PromptFormatError(RuntimeError):
    pass
```

Raise it when:

- role alternation validation fails
- tokenizer chat template rendering fails with a deterministic template error
- a configured prompt policy is unknown
- a `merge_into_first_user` policy has no user message to merge into

This makes failures explainable in `errors.jsonl` and avoids mislabeling template problems as transport problems.

### Phase 3 - Make Retries Respect Error Type

Update `invoke_with_retries()` to classify deterministic local failures as non-retryable.

Suggested classification:

```python
def classify_generation_error(exc: Exception) -> tuple[str, bool]:
    message = str(exc).lower()
    if isinstance(exc, PromptFormatError):
        return "prompt_format_error", False
    if "conversation roles must alternate" in message:
        return "prompt_format_error", False
    if "chat template" in message or "templateerror" in message:
        return "prompt_format_error", False
    if "cuda out of memory" in message:
        return "oom", False
    if "timeout" in message:
        return "timeout", True
    if "empty_response" in message:
        return "empty_response", True
    return "generation_error", True
```

Expected result after this change:

- The current BioMistral bug would create 1 error row per failed turn, not 4 repeated rows.
- The `error_code` would be `prompt_format_error`, not `transport_error`.
- A future real network timeout could still retry.

### Phase 4 - Add Tokenizer-Only Preflight

Add a preflight step before loading 7B model weights.

The preflight should:

1. Load only `AutoTokenizer` for each selected model.
2. Build representative initial and final message lists using one selected row and one persona.
3. Apply the configured prompt policy.
4. Call `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`.
5. Record the rendered prompt length and selected prompt policy.
6. Abort early if any selected model cannot render both initial and final prompts.

This catches the current BioMistral problem before spending several minutes downloading/loading model weights.

Recommended output:

```text
[preflight] model=mistral_7b_instruct_v0_3 prompt_policy=native status=ok
[preflight] model=biomistral_7b prompt_policy=merge_into_first_user status=ok
```

Add the preflight result to `manifest.json`:

```json
{
  "prompt_format": {
    "mistral_7b_instruct_v0_3": {
      "renderer": "tokenizer_chat_template",
      "system_policy": "native",
      "preflight_status": "ok"
    },
    "biomistral_7b": {
      "renderer": "tokenizer_chat_template",
      "system_policy": "merge_into_first_user",
      "preflight_status": "ok"
    }
  }
}
```

### Phase 5 - Clean Generation Arguments

The local backend currently calls:

```python
self.model.generate(
    ...,
    temperature=0.0,
    do_sample=False,
)
```

For deterministic greedy decoding, use:

```python
generation_kwargs = {
    "max_new_tokens": max_new_tokens,
    "do_sample": False,
    "pad_token_id": self.tokenizer.pad_token_id,
    "eos_token_id": self.tokenizer.eos_token_id,
    "use_cache": True,
}
```

Only include `temperature` when `do_sample=True`. Keep `temperature` in the output record as part of the experiment config, but do not pass it to `generate()` in greedy mode.

### Phase 6 - Improve Progress Semantics

The notebook-level progress bar should distinguish these phases:

- tokenizer preflight
- model download/load
- generation records
- artifact validation

The model loading phase can legitimately sit at `0/24` records because no records have been generated yet. The UX should make that clear with a separate status line or heartbeat:

```text
[progress] stage=dry_run status=tokenizer_preflight model=biomistral_7b
[progress] stage=dry_run status=loading_model_wait model=biomistral_7b elapsed_sec=...
[progress] stage=dry_run status=model_ready model=biomistral_7b
[progress] stage=dry_run status=success done=13 total=24 ...
```

This is not a correctness fix, but it prevents users from interrupting a healthy Colab run during model loading.

## Validation Plan

### Unit-Level Checks

Add focused tests for prompt normalization. These can run without GPU.

Cases:

1. `native` returns messages unchanged.
2. `merge_into_first_user` converts `[system, user]` to `[user]`.
3. `merge_into_first_user` converts `[system, user, assistant, user]` to `[user, assistant, user]`.
4. Multiple system messages are preserved in order.
5. Invalid role sequences raise `PromptFormatError`.
6. Unknown policy raises `PromptFormatError`.

### Tokenizer Preflight Check

Run a tokenizer-only check for both configured models:

```bash
/usr/bin/python3 -u scripts/run_experiment.py --stage dry_run --model-label biomistral_7b
```

The preflight should print BioMistral prompt compatibility before model weights load.

If a separate command is added, prefer:

```bash
/usr/bin/python3 -u scripts/run_experiment.py --stage dry_run --model-label biomistral_7b --preflight-only
```

Expected result:

- BioMistral initial prompt renders successfully.
- BioMistral final prompt renders successfully.
- No `Conversation roles must alternate user/assistant` error appears.

### Colab GPU Smoke Test

After the code fix, run BioMistral only:

```bash
/usr/bin/python3 -u scripts/run_experiment.py --stage dry_run --model-label biomistral_7b
```

Expected dry-run outcome:

```text
Records: 12
Status counts: {'success': 12}
```

Then run both models:

```bash
/usr/bin/python3 -u scripts/run_experiment.py --stage dry_run
```

Expected dry-run outcome:

```text
Records: 24
Status counts: {'success': 24}
```

### Output Quality Spot Check

Because BioMistral may be less instruction-tuned than Mistral Instruct, success should not be judged only by non-empty output. Inspect several `assistant_initial_text` and `assistant_final_text` fields.

Check:

- The model gives a clear answer choice.
- The model does not echo the whole prompt.
- The model does not expose hidden/system instructions.
- The model handles the follow-up turn coherently.
- The model's style changes across personas in plausible ways without changing the task semantics.

If BioMistral produces low-quality instruction-following after the formatting fix, that becomes a model-selection issue, not the current runtime bug.

## Research-Grade Constraints

The fix must preserve experimental comparability.

Rules:

- Keep canonical messages model-agnostic in saved `raw/messages/*.json`.
- Render model-specific prompts only at backend inference time.
- Record each model's prompt rendering policy in `manifest.json`.
- Avoid silently falling back to a different prompt strategy during a full run.
- If an automatic fallback is used during dry run, write the selected strategy explicitly before pilot/full runs.
- Do not change persona text while fixing serialization.
- Do not change dataset sampling while fixing serialization.

The correct research distinction is:

- same semantic prompt content across models
- model-specific serialization needed to satisfy tokenizer contracts

## Model Selection Risk

BioMistral 7B may be a weaker chat/instruction follower than Mistral 7B Instruct v0.3. The current dry-run failure does not prove that BioMistral is unusable. It only proves that the message serialization is incompatible with its tokenizer template.

After the prompt-format fix, make a separate decision:

1. Keep BioMistral as a biomedical/domain model if it produces usable answer-format responses.
2. Keep it but disclose it as a base or weakly instruction-tuned model if output quality is acceptable but different.
3. Replace or add an instruction-tuned biomedical model if BioMistral cannot reliably follow the two-turn evaluation protocol.

Selection criteria for any replacement:

- Fits on Colab T4 15 GB VRAM with 4-bit quantization.
- Has a license compatible with the project.
- Has a tokenizer chat template that passes preflight.
- Produces concise multiple-choice answers in the dry run.
- Can complete the full run without excessive latency.

## Acceptance Criteria

The problem is solved when all of the following are true:

- BioMistral no longer raises `Conversation roles must alternate user/assistant`.
- The dry run with `--model-label biomistral_7b` completes with 12/12 successful records.
- The full dry run completes with 24/24 successful records.
- `errors.jsonl` is empty for a clean dry run.
- `summary.json` has `status_counts: {'success': 24}` for the two-model dry run.
- `manifest.json` records each model's prompt rendering policy.
- The notebook progress display clearly distinguishes model loading from record generation.
- The `temperature` warning is removed for greedy decoding.
- Deterministic prompt-format errors are classified as `prompt_format_error` and are not retried.

## Immediate Next Patch

The smallest high-confidence patch is:

1. Add `prompt_format.system_policy` to the two model configs.
2. Store the loaded model config in `LocalTransformersBackend.prepare_model()`.
3. Add `normalize_messages_for_model()` with `native` and `merge_into_first_user`.
4. Apply that normalization before `tokenizer.apply_chat_template()`.
5. Add non-retryable `prompt_format_error` classification.
6. Remove `temperature` from local `generate()` kwargs when `do_sample=False`.
7. Rebuild the Colab package ZIP after tests pass.

This directly addresses the observed failures without changing the dataset, personas, or experiment design.
