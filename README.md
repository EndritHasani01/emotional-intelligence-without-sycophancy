# Emotional Intelligence Without Sycophancy

This repository studies a simple question:

How does an LLM's persona affect sycophantic behavior, and does domain specialization help the model resist social pressure?

In plain terms, the project checks whether a model becomes more likely to agree with a user when the conversation is warm and emotionally supportive, even when the user is wrong.

## What this repo contains

The experiment is frozen around:

- 2 models
- 3 personas
- 400 converted `cais/mmlu` items
- 4 balanced subsets: `BIO_P`, `BIO_R`, `OOD_P`, `OOD_R`
- a 5-turn conversation format:
  1. system prompt
  2. user assertion
  3. assistant answer
  4. user pressure follow-up
  5. assistant final answer

The current frozen design produces:

- `2400` conversations total
- `4800` assistant generations total

## Current status

What works today:

- protocol is frozen
- configs are frozen
- dataset artifacts are built
- the experiment runner exists
- the dry-run passed

What does not work yet:

- the real pilot is blocked in the current backend path
- the checked-in runner can use either:
  - a deterministic mock backend for dry-runs
  - the Hugging Face router path for real runs
- on `2026-03-16`, the frozen model IDs were not both executable through that hosted Hugging Face path

So this repo is infrastructure-ready, but not yet ready for a real pilot or full experiment with the current execution backend.

## Repo map

- `configs/`: persona prompts, pressure prompts, run settings, output schema
- `data/raw/mmlu/`: cached source rows from `cais/mmlu`
- `data/interim/`: selected source items before final conversion
- `data/frozen/`: final frozen assertion dataset used by the runner
- `scripts/build_dataset.py`: rebuilds the dataset artifacts
- `scripts/run_experiment.py`: runs `dry_run`, `pilot`, or `full`
- `runs/`: saved run artifacts
- `todos/`: stage-by-stage work packages

## Beginner quick start

If you just want to understand the project, read these in order:

1. `README.md`

If you want to run the parts that already work:

### 1. Install Python dependencies

This repo currently uses scripts directly. There is no packaged installer yet.

```bash
python -m pip install datasets pyyaml huggingface_hub
```

### 2. Rebuild the dataset

```bash
python scripts/build_dataset.py
```

This creates or refreshes:

- `data/raw/mmlu/`
- `data/interim/mmlu_selected_v1.csv`
- `data/frozen/assertions_frozen_v1.csv`

### 3. Run the accepted smoke test

```bash
python scripts/run_experiment.py --stage dry_run
```

This should create a run directory under `runs/dry-run/`.

In the current repo state, the dry-run is allowed to fall back to a deterministic mock backend if the frozen Hugging Face model IDs are not available through the hosted inference route.

## What the dataset means

The 4 subset labels are:

- `BIO_P`: biomedical question, user starts with an incorrect claim
- `BIO_R`: biomedical question, user starts with a correct claim and then pushes back anyway
- `OOD_P`: out-of-domain question, user starts with an incorrect claim
- `OOD_R`: out-of-domain question, user starts with a correct claim and then pushes back anyway

Each subset contains exactly `100` items.

## Personas

The experiment compares 3 system-prompt personas:

- `baseline`: neutral and matter-of-fact
- `warm_supportive`: warm and validating
- `supportive_truth_first`: warm, but explicitly instructed to prioritize accuracy over agreement

## Models

The frozen model IDs are:

- `mistralai/Mistral-7B-Instruct-v0.3`
- `BioMistral/BioMistral-7B`

## Run outputs

Each run writes a directory with:

- `records.jsonl`: one record per conversation
- `errors.jsonl`: retry and failure information
- `manifest.json`: run-level metadata and expected counts
- `summary.json`: a compact status summary
- `raw/`: raw payloads for each assistant turn

## Important limitation

The current pilot artifact is intentionally marked blocked.

Why:

- the real hosted Hugging Face path did not expose a usable serving route for both frozen model IDs
- the checked-in runner does not yet include a local `transformers` GPU backend

That means:

- `dry_run` is valid for pipeline validation
- `pilot` and `full` are not yet valid scientific runs in the current repo state

