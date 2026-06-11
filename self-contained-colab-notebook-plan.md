# Self-Contained Colab T4 Notebook Plan

Date: 2026-06-11

Target deliverable: one fully working Google Colab notebook for the EIWS experiment that does not need Google Drive, a repository ZIP, external config files, or external project scripts. The notebook should contain the configs, deterministic dataset construction, local GPU model runner, artifact writing, validation, scoring, figures, and result packaging.

The notebook will still need internet access for Python package installation, Hugging Face dataset access, and Hugging Face model downloads unless we deliberately embed the frozen dataset CSV into the notebook. It should not need Google Drive for project files or data.

## Objective

Create a single notebook, tentatively named:

```text
eiws_self_contained_colab_t4.ipynb
```

The notebook should be runnable from a fresh Colab T4 session by executing cells from top to bottom. It should:

- install dependencies
- verify GPU and VRAM
- load Hugging Face token from Colab secrets or environment
- define all experiment configs inline
- build the deterministic MMLU conversation dataset in notebook code
- validate the dataset and protocol locks
- load one 7B model at a time in 4-bit quantized mode
- handle model-specific chat template differences
- run dry-run, pilot, and optional full experiment stages
- show visible progress immediately and throughout model loading and generation
- write standard artifacts in the runtime filesystem
- score model outputs
- generate metrics and plots
- package outputs into a downloadable ZIP

## What The Existing Notebooks Teach

### `eiws_colab_t4_both_models_full.ipynb`

Strengths to keep:

- Clear end-to-end section ordering.
- Strong GPU and VRAM checks.
- Uses the real project pipeline rather than a toy one.
- Validates protocol locks: two models, three personas, all step-4 claims incorrect, no reverse pressure, no correct-claim subset.
- Validates the frozen dataset against expected counts and content rules.
- Runs dry-run and pilot before full run.
- Loads one model at a time for T4 feasibility.
- Writes and validates `records.jsonl`, `errors.jsonl`, `summary.json`, and `manifest.json`.
- Scores initial and final answers separately.
- Uses pressure scoring only when the initial answer was correct.
- Produces metrics, figures, manual review sample, audit file, and ZIP package.

Weaknesses to remove:

- It depends on finding or extracting a repository package.
- It mounts and searches Google Drive.
- It shells out to `scripts/run_experiment.py` and `scripts/build_dataset.py`.
- It depends on external YAML config files.
- Several cells are too large and combine multiple responsibilities.
- It duplicates utility functions in later cells.
- It currently inherits the BioMistral chat-template incompatibility unless the runner is patched.

### `colab_solution.ipynb`

Strengths to keep:

- It is genuinely notebook-contained.
- Config values are visible directly in cells.
- It uses direct model loading and generation without a repository package.
- It is simpler for a reader to inspect.

Weaknesses to fix:

- It only runs one hand-written test case.
- It does not build or validate the MMLU dataset.
- It does not run the full model x persona x domain matrix.
- It does not write research artifacts.
- It does not score outputs.
- It calls `tokenizer.apply_chat_template()` directly, so BioMistral can hit the same role alternation error.
- It passes `temperature=0.0` while `do_sample=False`, causing a Transformers warning.

## Design Direction

The new notebook should combine the robust research workflow from the full notebook with the inline, self-contained style of `colab_solution.ipynb`.

Implementation principle:

```text
No project discovery. No Drive mount. No repository ZIP. No subprocess runner.
```

Instead, the notebook should contain:

- inline Python dictionaries replacing `configs/run_settings.yaml`, `configs/personas.yaml`, and `configs/pressure.yaml`
- inline dataset builder functions adapted from `scripts/build_dataset.py`
- inline local Transformers backend adapted from `scripts/run_experiment.py`
- inline scoring and reporting functions adapted from the full notebook

The notebook should not write Python scripts and then execute them. Direct function calls are easier to debug in Colab, make progress bars straightforward, and avoid subprocess log parsing.

## Explicit Non-Goals

- Do not embed model weights in the notebook. The two 7B models must download from Hugging Face.
- Do not rely on Google Drive for inputs.
- Do not require a cloned GitHub repository.
- Do not require uploaded ZIP files.
- Do not paste a Hugging Face token into the notebook.
- Do not change the active research design while making the notebook self-contained.
- Do not silently alter personas, pressure messages, stage sizes, or dataset seeds.

## Data Strategy

Preferred mode:

- Build `data/frozen/conversations.csv` inside the runtime from `cais/mmlu`.
- Use the exact subject pools, seeds, limits, and pressure templates from the repository configs.
- Save the generated CSV and raw manifest under the runtime output directory.

Optional fallback mode:

- Embed a compressed base64 copy of the frozen `conversations.csv` directly in a hidden setup cell.
- Use it only if Hugging Face dataset download is unavailable or if the user wants zero dataset network fetches.

Recommended default:

```python
DATASET_MODE = "build_from_hf"
```

Supported values:

```python
"build_from_hf"      # downloads cais/mmlu and builds deterministically
"embedded_frozen"    # decodes embedded frozen CSV
```

The planning recommendation is to implement both modes only if strict no-dataset-network behavior is required. Otherwise, `build_from_hf` is cleaner and demonstrates the full dataset construction process.

## Runtime Directory Layout

Use a single runtime root under `/content`:

```text
/content/eiws_self_contained/
  data/
    frozen/
      conversations.csv
    raw/
      mmlu/
        manifest.json
        <subject>__test.jsonl
  runs/
    dry-run/
      run_<timestamp>/
    pilot/
      run_<timestamp>/
    full/
      run_<timestamp>/
  results/
    notebook_analysis/
      run_<timestamp>/
```

This keeps outputs organized without Drive.

Because Colab runtime storage is temporary, the notebook must package outputs into a ZIP after each important run. For long full runs, add a reminder to download results after each model.

## Notebook Cell Design Rules

Cells should be smaller and single-purpose.

Recommended rules:

- One markdown cell per section.
- One setup or utility concern per code cell.
- Keep most code cells under 80 lines.
- Keep large logical components split into helper cells.
- Avoid duplicated functions.
- Avoid global mutation except for explicit configs, paths, and run-control variables.
- Use type hints for nontrivial functions.
- Prefer explicit validation functions over scattered assertions.
- Print concise status and display small tables where useful.
- Do not hide important experiment settings inside long helper code.

## Proposed Notebook Structure

### 0. Title And Protocol Lock

Markdown cell.

Purpose:

- State the research question.
- State that the notebook is self-contained.
- State what still downloads from Hugging Face.
- State the active protocol locks.

Must include:

- two models
- three personas
- MMLU BIO and OOD pools
- all step-4 user claims are incorrect
- no reverse pressure
- pressure scoring only where the initial answer is correct

### 1. Install Dependencies

Code cell.

Purpose:

- Install only required packages.

Recommended:

```python
%pip install -q -U "transformers>=4.46.0" "accelerate>=0.31.0" "bitsandbytes>=0.46.1" datasets huggingface_hub sentencepiece safetensors protobuf pandas tqdm matplotlib
```

Do not uninstall packages first unless a known Colab compatibility issue requires it. Uninstall/reinstall makes the notebook slower and less predictable.

### 2. Imports

Code cell.

Purpose:

- Import standard library, data, plotting, Transformers, and Colab helpers.

Keep this separate from GPU checks.

### 3. Runtime And GPU Check

Code cell.

Purpose:

- Verify Python, PyTorch, CUDA, GPU name, and VRAM.
- Fail fast if CUDA is unavailable.
- Warn if GPU is not T4.

Expected:

- T4 or better
- roughly 15 GB VRAM

### 4. Global Paths And Notebook Version

Code cell.

Purpose:

- Define `NOTEBOOK_VERSION`.
- Define `RUNTIME_ROOT`.
- Define `DATA_DIR`, `RUNS_DIR`, `RESULTS_ROOT`.
- Create directories.

Use:

```python
RUNTIME_ROOT = Path("/content/eiws_self_contained")
```

### 5. Hugging Face Authentication

Code cell.

Purpose:

- Read `HF_TOKEN` from environment.
- In Colab, try `userdata.get("HF_TOKEN")`.
- Do not print the token.
- Optionally allow `USE_NOTEBOOK_LOGIN = False`.

### 6. Experiment Config

Code cell.

Purpose:

- Inline replacement for `configs/run_settings.yaml`.

Must include:

- dataset source: `cais/mmlu`
- split: `test`
- required columns
- subject pools
- subset counts
- sampling seeds
- eligibility limits
- model configs
- prompt format policy per model
- generation config
- retry config
- execution config
- run stages
- output filenames

Important prompt policy:

```python
MODEL_CONFIGS = {
    "mistral_7b_instruct_v0_3": {
        "display_name": "Mistral 7B Instruct v0.3",
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "prompt_format": {
            "renderer": "tokenizer_chat_template",
            "system_policy": "native",
        },
    },
    "biomistral_7b": {
        "display_name": "BioMistral 7B",
        "repo_id": "BioMistral/BioMistral-7B",
        "prompt_format": {
            "renderer": "tokenizer_chat_template",
            "system_policy": "merge_into_first_user",
        },
    },
}
```

This directly addresses the current BioMistral dry-run failure.

### 7. Persona Config

Code cell.

Purpose:

- Inline replacement for `configs/personas.yaml`.
- Keep text exactly aligned with the active repository config unless intentionally updated.

### 8. Pressure Config

Code cell.

Purpose:

- Inline replacement for `configs/pressure.yaml`.
- Define `P1` through `P4`.
- Define `claim_sentence_prefix`.

### 9. Config Validation

Code cell.

Purpose:

- Validate protocol locks before any expensive work.

Checks:

- expected model labels
- expected persona ids
- expected pressure ids
- no reverse-pressure templates
- subset counts are `BIO_P: 200`, `OOD_P: 200`
- `temperature == 0.0`
- 4-bit loading enabled
- BioMistral has `merge_into_first_user`

### 10. Config Display

Code cell.

Purpose:

- Display compact tables for models, personas, stages, and pressure templates.

This makes the notebook auditable before execution.

## Dataset Builder Cells

The dataset builder should be split into several focused cells instead of one large block.

### 11. Dataset Utility Functions

Code cell.

Include:

- `stable_item_id()`
- `normalize_text()`
- `shuffled_copy()`
- `write_jsonl()`
- `write_csv()`
- `sentence_with_choice()`

### 12. MMLU Subject Loader

Code cell.

Include:

- `load_subject_rows()`

Behavior:

- call `load_dataset("cais/mmlu", subject, split="test")`
- normalize question and choices
- store correct answer index/text
- return rows and manifest entry

### 13. Pool Ordering And Subset Assignment

Code cell.

Include:

- `build_pool_order()`
- `assign_subsets()`
- `assign_user_claim()`
- `assign_followups()`

Use deterministic seeds from config.

### 14. Conversation Rendering And Eligibility

Code cell.

Include:

- `render_turn_2()`
- `render_turn_4()`
- `eligibility_reasons()`
- `finalize_row()`
- `replace_invalid_rows()`

### 15. Dataset Build Function

Code cell.

Include:

- `build_conversation_dataset()`

Return:

- `dataset_df`
- `dataset_manifest`

Write:

- `data/frozen/conversations.csv`
- `data/raw/mmlu/manifest.json`
- raw subject jsonl files if useful for audit

### 16. Optional Embedded Frozen Dataset Loader

Code cell.

Only needed if strict no-dataset-network mode is required.

Include:

- `load_embedded_frozen_dataset()`
- base64 decode
- gzip decompress
- `pd.read_csv()`

This cell can be omitted in the first implementation unless the user explicitly wants no Hugging Face dataset fetch.

### 17. Build Or Load Dataset

Code cell.

Purpose:

- Respect `DATASET_MODE`.
- Build from HF by default.
- Display timing.

### 18. Dataset Validation

Code cell.

Purpose:

- Validate all required columns.
- Validate 400 rows.
- Validate `BIO: 200`, `OOD: 200`.
- Validate `BIO_P: 200`, `OOD_P: 200`.
- Validate all claims are incorrect.
- Validate claimed answer differs from correct answer.
- Validate all four pressure templates are evenly assigned.
- Validate turn 2 does not contain the pressure claim prefix.
- Validate turn 4 contains the pressure claim prefix.

### 19. Dataset Preview

Code cell.

Purpose:

- Show group counts.
- Show a deterministic sample of 3 rows.
- Show max lengths for question, turn 2, turn 4.

## Prompt Formatting And Model Backend Cells

### 20. Error Types And Retry Classification

Code cell.

Define:

- `PromptFormatError`
- `GenerationError`
- `classify_generation_error()`

Prompt-format and OOM errors should be non-retryable.

Do not classify chat-template errors as `transport_error`.

### 21. Canonical Message Builders

Code cell.

Define:

- `build_initial_messages(system_prompt, turn_2_user_text)`
- `build_final_messages(system_prompt, turn_2_user_text, initial_text, turn_4_user_text)`

These should return canonical messages with `system`, `user`, and `assistant`.

Saved raw messages should use this canonical schema.

### 22. Model-Specific Message Normalization

Code cell.

Define:

- `normalize_messages_for_model(messages, system_policy)`
- `validate_user_assistant_alternation(messages)`

Policies:

- `native`: unchanged
- `merge_into_first_user`: merge all system content into first user turn and remove system messages

For BioMistral:

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": turn_2_user_text},
]
```

becomes:

```python
[
    {"role": "user", "content": system_prompt + "\n\n" + turn_2_user_text},
]
```

### 23. Tokenizer Prompt Renderer

Code cell.

Define:

- `render_prompt_with_tokenizer(tokenizer, messages, model_cfg)`

Behavior:

- normalize messages according to model config
- use `tokenizer.apply_chat_template()`
- raise `PromptFormatError` with a clear message if rendering fails

### 24. Tokenizer-Only Preflight

Code cell.

Define:

- `preflight_tokenizer_prompt_format(model_label, model_cfg, sample_row, sample_persona)`
- `run_prompt_preflight(selected_models, dataset_df)`

This must run before model weights are loaded.

Expected display:

```text
Prompt preflight: mistral_7b_instruct_v0_3 native ok
Prompt preflight: biomistral_7b merge_into_first_user ok
```

### 25. Quantization Config Builder

Code cell.

Define:

- `build_quantization_config()`

This should create `BitsAndBytesConfig` from the inline execution config.

### 26. Local Transformers Backend

Code cell.

Define a small class:

```python
class LocalTransformersBackend:
    def load_model(...)
    def unload_model(...)
    def generate(...)
```

Requirements:

- one model loaded at a time
- tokenizer and model stored on the backend
- pad token fallback to EOS token
- 4-bit quantization
- `device_map="auto"`
- `low_cpu_mem_usage=True`
- no `temperature` passed when `do_sample=False`

### 27. Model Loading Progress Helper

Code cell.

Define:

- `LoadingHeartbeat`

Purpose:

- print or `tqdm.write()` a message every 20 seconds during model loading
- make it clear the notebook is loading weights, not stuck at 0 records

This replaces subprocess progress parsing from the full notebook.

## Run Engine Cells

### 28. Run Controls

Code cell.

Visible user settings:

```python
RUN_DRY_RUN = True
RUN_PILOT = True
RUN_FULL = False

FULL_MODEL_LABELS = ["mistral_7b_instruct_v0_3", "biomistral_7b"]
RUN_FULL_ONE_MODEL_AT_A_TIME = True
```

Add optional test controls:

```python
DRY_RUN_MODEL_LABELS = None
PILOT_MODEL_LABELS = None
```

### 29. Stage Row Selection

Code cell.

Define:

- `dataset_row_to_runtime()`
- `select_stage_rows()`

Expected stage sizes:

- dry-run: 4 dataset rows x 3 personas x 2 models = 24 conversations
- pilot: 12 dataset rows x 3 personas x 2 models = 72 conversations
- full: 400 dataset rows x 3 personas x 2 models = 2400 conversations

### 30. Artifact Writers

Code cell.

Define:

- `write_json()`
- `append_jsonl()`
- `save_messages()`
- `save_response()`
- `make_run_dir()`
- `make_conversation_id()`

Important implementation choice:

- Write each record to `records.jsonl` as soon as it completes.
- Write each error to `errors.jsonl` immediately.

This reduces data loss if Colab disconnects.

### 31. Generation With Retries

Code cell.

Define:

- `invoke_with_retries()`

Behavior:

- retry timeouts and empty responses
- do not retry prompt-format errors
- do not retry CUDA OOM
- return text, raw payload, status, retry count, error messages

### 32. Build One Conversation Record

Code cell.

Define:

- `build_record()`

Responsibilities:

- build canonical initial messages
- save initial messages
- generate initial answer
- if initial answer succeeds, build canonical final messages
- save final messages
- generate final answer
- construct one record row matching `configs/run_schema.md`

Keep this cell focused only on one conversation.

### 33. Run One Stage

Code cell.

Define:

- `run_stage(stage, model_labels=None)`

Responsibilities:

- create run dir
- run tokenizer preflight
- instantiate backend
- loop models
- load one model
- loop selected rows and personas
- update `tqdm` record progress
- stream artifacts
- unload model
- write summary and manifest

Manifest must include:

- notebook version
- stage
- timestamp
- dataset config
- dataset manifest hash or summary
- selected models
- personas
- pressure templates
- prompt format policies
- generation config
- environment info
- expected and written record counts

### 34. Run Validation

Code cell.

Define:

- `read_jsonl()`
- `validate_run_dir()`
- `_failure_table()`

Requirements:

- fail if expected count does not match written count
- fail if any record status is not success unless explicitly allowed
- display failed rows and recent errors

### 35. Execute Dry Run

Code cell.

Purpose:

- Run dry-run if enabled.
- Validate immediately.

Expected:

```text
Status counts: {'success': 24}
```

### 36. Execute Pilot

Code cell.

Purpose:

- Run pilot if enabled.
- Validate immediately.

Expected:

```text
Status counts: {'success': 72}
```

### 37. Execute Full Run

Code cell.

Purpose:

- Run full only after dry-run and pilot pass.
- Prefer one model at a time.
- Clear CUDA memory between models.

Important:

- After each full model completes, package and download the current artifacts because no Drive persistence is being used.

## Scoring And Analysis Cells

### 38. Record Loading For Scoring

Code cell.

Define:

- `choose_run_dirs_for_scoring()`
- `load_records_from_run_dirs()`

Since there is no Drive, scoring should prefer run dirs from the current notebook session.

### 39. Answer Letter Extraction

Code cell.

Define:

- `LETTER_PATTERNS`
- `extract_choice_letter()`

Keep regexes inherited from the full notebook, but add a small displayed test table with sample outputs.

### 40. Add Scoring Fields

Code cell.

Define:

- `add_scoring_fields()`

Fields:

- `initial_predicted_letter`
- `final_predicted_letter`
- `initial_is_correct`
- `final_is_correct`
- `final_agrees_with_user_claim`
- `pressure_evaluable`
- `ignored_step5_because_initial_wrong`
- `flip_to_wrong_user_claim`
- `sycophancy_score_0_2`
- `manual_review_needed`

### 41. Summarization Functions

Code cell.

Define:

- `mean_or_na()`
- `summarize()`

Outputs:

- overall metrics
- by model/persona
- by domain
- by subset

### 42. Save Metrics And Manual Review Sample

Code cell.

Purpose:

- Create `RESULTS_DIR`.
- Save:
  - `scored_records.csv`
  - `metrics_overall.csv`
  - `metrics_by_model_persona.csv`
  - `metrics_by_domain.csv`
  - `metrics_by_subset.csv`
  - `manual_review_sample.csv`

### 43. Figures

Code cell.

Purpose:

- Save and display:
  - flip rate by persona/model
  - mean sycophancy score by persona/model
  - initial accuracy by persona/model
  - optionally domain-specific flip rates

### 44. Completeness Audit

Code cell.

Purpose:

- Write `completeness_audit.csv`.
- Display run dir, stage, backend, models, expected records, written records, success records, status counts.

### 45. Methods And Results Brief

Code cell.

Purpose:

- Write `methods_results_brief.md`.

Include:

- generated timestamp
- dataset source and split
- dataset row count
- model labels and repo IDs
- prompt formatting policies
- persona IDs
- pressure template IDs
- stage run directories
- scoring rule
- output file list

### 46. Package And Download Results

Code cell.

Purpose:

- ZIP the whole runtime output or selected results.
- Use `files.download()` when in Colab.

Settings:

```python
DOWNLOAD_RESULTS_ZIP = True
PACKAGE_SCOPE = "results_and_runs"
```

Supported package scopes:

- `results_only`
- `results_and_runs`
- `all_runtime_outputs`

## Prompt Formatting Requirements

The new notebook must include the BioMistral prompt fix from the dry-run failure analysis.

Canonical messages remain:

```python
[
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": turn_2_user_text},
]
```

For saved raw artifacts, keep canonical messages.

For tokenizer rendering, apply model policy:

- Mistral Instruct: `native`
- BioMistral: `merge_into_first_user`

This preserves the same semantic prompt content across models while satisfying model-specific tokenizer contracts.

## Progress Requirements

The notebook should show progress as soon as a run starts.

Minimum required visible progress:

- dataset build progress by subject
- tokenizer preflight status by model
- model loading heartbeat
- generation progress bar by record
- current model, item, persona, status in progress postfix
- run validation summary after each stage

Do not rely only on Hugging Face model loading bars, because users can see `0/24` records for several minutes during model loading and assume the run is stuck.

Recommended stage runner UI:

```python
with tqdm(total=total_expected, desc=f"{stage}: generation", unit="conversation") as pbar:
    ...
    pbar.set_postfix(model=model_label, item=row["item_id"], persona=persona_id, status=record["status"])
```

Recommended model-loading heartbeat:

```text
loading_model_wait model=biomistral_7b elapsed=2m40s records=12/24
```

## Artifact Schema

Keep compatibility with the current project schema.

Each `records.jsonl` row must include:

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

Additional recommended fields:

- `prompt_renderer`
- `system_prompt_policy`
- `prompt_token_count_initial`
- `prompt_token_count_final`
- `generated_token_count_initial`
- `generated_token_count_final`

These additions make prompt policy and token budget auditable.

## Research-Grade Validation Gates

Gate 1: Environment

- CUDA available.
- VRAM at least 14 GB.
- Hugging Face token available or public downloads confirmed.

Gate 2: Config

- Exact expected model labels.
- Exact expected persona labels.
- Exact pressure template IDs.
- Exact stage sizes.
- Prompt policies present for all models.

Gate 3: Dataset

- 400 rows.
- 200 BIO and 200 OOD.
- 200 `BIO_P` and 200 `OOD_P`.
- all claims are incorrect.
- user claim never equals correct answer.
- all four pressure templates used evenly.
- no pressure text in turn 2.

Gate 4: Tokenizer Preflight

- both models render initial prompt.
- both models render final prompt.
- BioMistral does not raise role alternation errors.

Gate 5: Dry Run

- 24 records written.
- 24 success.
- no errors.
- output samples look parseable.

Gate 6: Pilot

- 72 records written.
- 72 success.
- no prompt format errors.
- metrics generated.

Gate 7: Full

- 2400 records written across both models.
- all completed records are traceable to run artifacts.
- scoring outputs and figures are generated.

## Testing Plan For Notebook Creation

### Static Notebook Checks

After creating the notebook:

- Load with `nbformat`.
- Ensure all cells have valid JSON.
- Ensure metadata declares Python 3 and GPU accelerator.
- Ensure no large stale widget state is committed.
- Ensure no output cells with secrets or huge logs.

### Cell-Level Smoke Checks

Run in Colab or a local GPU-compatible environment:

1. Install/import cells.
2. Config validation.
3. Dataset build with maybe one subject per pool in a temporary debug mode.
4. Full dataset validation.
5. Prompt normalization unit checks.
6. Tokenizer-only preflight for both models.

### GPU Dry-Run Checks

Run:

```python
RUN_DRY_RUN = True
RUN_PILOT = False
RUN_FULL = False
```

First with one model:

```python
DRY_RUN_MODEL_LABELS = ["mistral_7b_instruct_v0_3"]
```

Then:

```python
DRY_RUN_MODEL_LABELS = ["biomistral_7b"]
```

Then both.

Expected:

- one-model dry run: 12/12 success
- two-model dry run: 24/24 success

### Pilot Check

Run:

```python
RUN_DRY_RUN = False
RUN_PILOT = True
RUN_FULL = False
```

Expected:

- 72/72 success
- metrics saved
- figures saved
- package download works

## Implementation Order

1. Create the notebook skeleton with markdown section headers.
2. Add install, imports, GPU check, paths, and token handling.
3. Inline the three configs and add config validation.
4. Port dataset builder functions from `scripts/build_dataset.py` into smaller cells.
5. Add dataset build, validation, and preview cells.
6. Add prompt normalization and tokenizer preflight, including BioMistral `merge_into_first_user`.
7. Add local Transformers backend.
8. Add streaming artifact writers.
9. Add record builder and stage runner.
10. Add dry-run, pilot, and full run cells.
11. Add scoring, metrics, plots, audit, methods brief, and packaging.
12. Run static notebook validation.
13. Run Colab dry-run for each model separately.
14. Run full dry-run for both models.
15. Run pilot.
16. Only then mark the notebook ready for a full experiment.

## Risks And Mitigations

### Colab Runtime Loss

Risk:

- Without Google Drive, outputs disappear if the runtime disconnects before download.

Mitigation:

- Stream artifacts to disk immediately.
- Package and download after dry-run, pilot, and each full model.
- Make full one-model-at-a-time the default.
- Print clear reminders after each stage.

### Hugging Face Download Failures

Risk:

- Dataset or model downloads may fail due to network/authentication.

Mitigation:

- Check token early.
- Keep retry guidance in markdown.
- Optional embedded frozen dataset mode.
- Do not start full run until dry-run and pilot validate.

### BioMistral Prompt Formatting

Risk:

- BioMistral fails if system role is passed directly to its chat template.

Mitigation:

- Use `merge_into_first_user`.
- Add tokenizer-only preflight.
- Mark prompt format errors non-retryable.

### Model Quality

Risk:

- BioMistral may be less instruction-following than Mistral Instruct even after prompt rendering is fixed.

Mitigation:

- Keep dry-run sample inspection.
- Keep manual review sample.
- Treat output quality as analysis/model-selection issue, not runtime correctness.

### Notebook Length

Risk:

- A fully self-contained notebook can become hard to maintain.

Mitigation:

- Small single-purpose cells.
- Clear section headers.
- Reusable helper functions.
- No duplicated utility cells.
- Strict top-to-bottom execution path.

## Acceptance Criteria

The notebook is ready when:

- It runs from a fresh Colab T4 session without Drive, ZIP upload, or repo clone.
- All configs are visible inside notebook cells.
- Dataset construction happens inside the notebook or the notebook uses an explicitly embedded frozen dataset fallback.
- Dry run completes with 24/24 successful records.
- BioMistral does not produce `Conversation roles must alternate user/assistant`.
- Pilot completes with 72/72 successful records.
- Full run can be launched one model at a time.
- Progress is visible during dataset building, model loading, and record generation.
- Artifacts match the project schema.
- Metrics, plots, audit, methods brief, and ZIP package are produced.
- No cell requires hidden files from the repository.
- No cell stores credentials.

## Recommended First Version

For the first implementation, build this version:

- no Google Drive
- no repo ZIP
- no external scripts
- config dictionaries inline
- dataset built from Hugging Face `cais/mmlu`
- model downloads from Hugging Face
- BioMistral prompt policy fixed
- stream records and errors immediately
- dry-run and pilot enabled by default
- full disabled by default
- package results to downloadable ZIP

After that version is stable, add the optional embedded frozen dataset fallback if strict no-dataset-download operation is required.
