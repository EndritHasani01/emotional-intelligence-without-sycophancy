# Emotional Intelligence Without Sycophancy

This repository runs a five-turn multiple-choice evaluation:

1. system prompt
2. user question with all answer choices
3. assistant answer
4. user pressure follow-up with the user's asserted answer in the same message
5. assistant final answer

The active pipeline uses:

- `configs/personas.yaml`
- `configs/pressure.yaml`
- `configs/run_settings.yaml`
- `configs/run_schema.md`
- `scripts/build_dataset.py`
- `scripts/run_experiment.py`
- `data/frozen/conversations.csv`

## Dataset

The dataset is built from `cais/mmlu` test questions across the fixed BIO and
OOD subject pools. Each row stores only the fields needed to run and analyze
the conversation:

- item identity and source row
- pool and subset labels
- whether the user's asserted answer is correct or incorrect
- the correct answer
- the user's claimed answer
- the rendered turn-2 and turn-4 user messages
- the follow-up template id and text

`scripts/build_dataset.py` deterministically selects the items, assigns subset
membership, assigns a claimed answer, assigns one of the pressure templates,
filters unusable rows, and writes `data/frozen/conversations.csv`.

## Runner

`scripts/run_experiment.py` reads the frozen dataset directly and executes the
exact conversation format above. It writes:

- `records.jsonl`
- `errors.jsonl`
- `summary.json`
- `manifest.json`
- raw message payloads
- raw response payloads

## Usage

Build the dataset:

```powershell
python scripts/build_dataset.py
```

Run a dry-run:

```powershell
python scripts/run_experiment.py --stage dry_run
```

If `HF_TOKEN` is not set or the models do not have a usable hosted provider,
the runner falls back to the mock backend.
