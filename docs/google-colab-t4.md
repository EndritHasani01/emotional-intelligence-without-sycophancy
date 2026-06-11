# Running This Repo In Google Colab On A 15 GB T4

## Summary

The practical way to run this repository on a Google Colab T4 is:

1. use the new `local_transformers` backend
2. load only one 7B model at a time
3. load that model in 4-bit quantized mode with `bitsandbytes`
4. let the runner process the full dataset sequentially

This repo now supports that path directly. The local backend keeps VRAM usage low enough for a 15 GB T4 by unloading one model before loading the next one.

## Why This Is The Right Approach

The original runner only supported:

- `mock`
- `hf_router`

`hf_router` uses hosted inference and does not use the Colab GPU. On a T4, you want local inference with:

- `transformers`
- `accelerate`
- `bitsandbytes`
- 4-bit quantization

That is what `EIWS_RUN_BACKEND=local_transformers` now does.

## Expected Run Size

From `configs/run_settings.yaml`:

- `dry_run`: 4 dataset rows x 3 personas x 2 models = 24 conversations
- `pilot`: 12 dataset rows x 3 personas x 2 models = 72 conversations
- `full`: 400 dataset rows x 3 personas x 2 models = 2400 conversations

All step-4 user claims are incorrect in the active design. The run no longer
uses correct-claim control subsets.

Each conversation makes 2 generations:

- initial answer
- final answer after user pressure

So the `full` stage is 4800 generations. On a T4 this is feasible, but it is slow. Start with `dry_run`, then `pilot`, before attempting `full`.

## Recommended Colab Runtime

- Runtime type: `GPU`
- GPU class: `T4`
- Use Drive mounting if you want outputs to survive runtime resets

High RAM is helpful but not strictly required for this repo.

## Colab Notebook Steps

### 1. Recommended: use your Google Drive folder

Upload `dist/eiws_colab_t4_package.zip` to:

```text
My Drive/Colab Notebooks/nlp-eiws
```

Then open `eiws_colab_t4_both_models_full.ipynb` in Colab and run it from the top. The notebook now mounts Drive, finds:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package.zip
```

unzips it, and switches into:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package
```

This Drive-based path is recommended for the full run because outputs survive runtime resets.

### 2. Confirm the GPU

```bash
!nvidia-smi
```

You should see a T4 with about 15 GB VRAM.

### 3. Install runtime dependencies

```bash
!pip install -q transformers accelerate bitsandbytes datasets pyyaml huggingface_hub sentencepiece
```

Colab usually already has PyTorch installed. If the runtime image is broken or mismatched, reinstall CUDA-enabled PyTorch separately.

### 4. Optional: authenticate with Hugging Face

Both configured models are public, but authentication is still useful for more reliable downloads.

```python
from huggingface_hub import login
login()
```

Or set the token as an environment variable:

```python
import os
os.environ["HF_TOKEN"] = "hf_xxx"
```

### 5. Build the dataset

```bash
!python scripts/build_dataset.py
```

This writes the frozen conversation file used by the runner:

- `data/frozen/conversations.csv`

### 6. Enable the local Colab backend

```python
import os
os.environ["EIWS_RUN_BACKEND"] = "local_transformers"
```

### 7. Run a smoke test first

```bash
!python scripts/run_experiment.py --stage dry_run
```

If that succeeds, run the pilot:

```bash
!python scripts/run_experiment.py --stage pilot
```

### 8. Run the full experiment

```bash
!python scripts/run_experiment.py --stage full
```

## Safer Option For Long Colab Sessions

If you are worried about Colab disconnects, run one model at a time. The runner now supports that directly:

```bash
!python scripts/run_experiment.py --stage full --model-label mistral_7b_instruct_v0_3
```

```bash
!python scripts/run_experiment.py --stage full --model-label biomistral_7b
```

This is slower overall than a large dedicated machine, but it is safer on Colab because:

- only one model is loaded in VRAM at a time
- each notebook session can focus on one model
- you avoid large GPU memory spikes from trying to keep both models resident

## What The Local Backend Does

The `local_transformers` backend in `scripts/run_experiment.py` uses:

- `AutoTokenizer`
- `AutoModelForCausalLM`
- `BitsAndBytesConfig(load_in_4bit=True)`
- `device_map="auto"`
- `do_sample=False`
- `temperature=0.0`

The execution flow is:

1. load model A in 4-bit
2. run all assigned conversations for model A
3. free GPU memory
4. load model B in 4-bit
5. run all assigned conversations for model B

That is the key design choice that makes a T4 workable.

## Output Location

Runs are written under:

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

## Practical Notes

- Do not try to hold both 7B models in memory at the same time on a 15 GB T4.
- Do not start with `full`; first verify `dry_run` and `pilot`.
- Keep the notebook attached during long runs.
- If Colab disconnect risk is high, prefer the per-model runs shown above.
- The local backend is for GPU inference. If Colab is not in GPU mode, it will fail intentionally.

## Minimal Colab Command Sequence

If you want the shortest working path:

```bash
!pip install -q transformers accelerate bitsandbytes datasets pyyaml huggingface_hub sentencepiece
!python scripts/build_dataset.py
```

```python
import os
os.environ["EIWS_RUN_BACKEND"] = "local_transformers"
```

```bash
!python scripts/run_experiment.py --stage dry_run
!python scripts/run_experiment.py --stage pilot
!python scripts/run_experiment.py --stage full
```

## Recommended Conclusion

For this repository, a Google Colab T4 is good enough if you use quantized local inference and sequential model loading. The setup above is the lowest-friction path that still uses the GPU properly and stays within the T4 memory limit.
