# Colab T4 Single-Notebook Runbook

This repository evaluates whether assistant persona instructions make models more likely to accept an incorrect user claim after social pressure.

The active experiment uses:

- `cais/mmlu` test questions
- 400 frozen conversation rows in `data/frozen/conversations.csv`
- two models: `mistralai/Mistral-7B-Instruct-v0.3` and `BioMistral/BioMistral-7B`
- three personas: baseline, warm supportive, supportive truth-first
- one incorrect user claim in every step-4 pressure message
- no reverse-pressure or correct-claim control subset

The Colab entry point is:

- `eiws_colab_t4_both_models_full.ipynb`

The upload package is:

- `dist/eiws_colab_t4_package.zip`

## Recommended Colab Setup

Use a Google Colab runtime with:

- runtime type: GPU
- GPU: T4 or larger
- optional but recommended: a Hugging Face token in Colab secrets named `HF_TOKEN`
- recommended for long full runs: keep the package in `My Drive/Colab Notebooks/nlp-eiws`

The notebook runs local GPU inference with the repository backend:

```text
EIWS_RUN_BACKEND=local_transformers
```

That backend loads one 7B model at a time in 4-bit quantized mode with `bitsandbytes`, runs its assigned conversations, unloads it, then loads the next model. Do not try to keep both models loaded at once on a 15 GB T4.

## ZIP Workflow

1. Upload `dist/eiws_colab_t4_package.zip` to `My Drive/Colab Notebooks/nlp-eiws`.
2. Open `eiws_colab_t4_both_models_full.ipynb` in Google Colab.
3. Set the runtime to GPU/T4.
4. Run the install and GPU-check cells.
5. Run the project-location cell. It mounts Drive, finds `My Drive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package.zip`, unzips it, and switches into the extracted project folder.
6. Keep `FORCE_REBUILD_DATASET = False` unless you specifically want to rebuild from Hugging Face.
7. Run the config and dataset validation cells.
8. Run dry-run and pilot first.
9. After dry-run and pilot validate cleanly, set `RUN_FULL = True` in the run-control cell and rerun from that cell onward.

The package includes the frozen CSV, so the notebook can run without rebuilding the dataset. Rebuilding is deterministic, but it downloads `cais/mmlu` again and takes extra time.

## Full Run Strategy

The full matrix is:

- 400 dataset rows
- 3 personas
- 2 models
- 2400 conversations
- 4800 total generations

On a T4 this can take several hours. The notebook defaults to:

```python
RUN_FULL_ONE_MODEL_AT_A_TIME = True
FULL_MODEL_LABELS = ["mistral_7b_instruct_v0_3", "biomistral_7b"]
```

This creates one full run directory per model. That is easier to inspect and safer than trying to run both models inside one long process.

For even safer long sessions, run one model per Colab session:

```python
RUN_FULL = True
FULL_MODEL_LABELS = ["mistral_7b_instruct_v0_3"]
```

Then repeat with:

```python
RUN_FULL = True
FULL_MODEL_LABELS = ["biomistral_7b"]
```

If you split the two full models across separate sessions, keep the extracted project in Google Drive. The notebook scoring cell will look for the latest full `local_transformers` run directories and combine the latest runs that cover both configured models.

With the recommended folder, the extracted project path is:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package
```

## Outputs

Run artifacts are written under:

- `runs/dry-run/`
- `runs/pilot/`
- `runs/full/`

Each run directory contains:

- `records.jsonl`
- `errors.jsonl`
- `summary.json`
- `manifest.json`
- `raw/messages/`
- `raw/responses/`

Notebook analysis outputs are written under:

- `results/notebook_analysis/<timestamp>/`

The final notebook cell creates a ZIP archive of that results directory.

## Common Problems

If the notebook says no CUDA GPU is available, switch Colab to `Runtime -> Change runtime type -> T4 GPU`, then restart and rerun.

If model loading runs out of memory, keep `RUN_FULL_ONE_MODEL_AT_A_TIME = True`, restart the runtime, and run only one model label in that session.

If scoring says no `local_transformers` run artifacts exist, run dry-run or pilot in the current notebook session. Old mock-backend runs are intentionally ignored.

If scoring says the run artifacts do not match the all-incorrect-claim design, ignore old run folders and rerun dry-run, pilot, or full with the current package and notebook.

If downloads fail, add a Colab secret named `HF_TOKEN` or set `USE_NOTEBOOK_LOGIN = True` in the token cell and authenticate interactively.

If the notebook says it extracted the ZIP but then asks you to upload a file, use the updated notebook. The project-location cell now searches recursively under `My Drive/Colab Notebooks/nlp-eiws` and prints a debug tree instead of falling through to the upload prompt.
