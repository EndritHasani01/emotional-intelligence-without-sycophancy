# Exact Google Colab T4 Steps

This file explains exactly how to run this project on Google Colab with a T4 GPU.

Short answer: yes, this setup is the right way to run the two models for this project on a 15 GB T4. The notebook uses the repository runner, not a separate simplified implementation. It runs the configured conversation format, loads each 7B model locally on the Colab GPU in 4-bit mode, uses `temperature = 0.0`, writes the standard run artifacts, then scores the outputs.

The only thing I cannot guarantee in advance is that Colab itself will stay connected for the whole full run. That is why the instructions below use Google Drive and recommend running one full model at a time.

## Files You Need

Use these two files from the repository:

- `eiws_colab_t4_both_models_full.ipynb`
- `dist/eiws_colab_t4_package.zip`

The ZIP contains the project files needed by Colab:

- configs
- scripts
- frozen dataset CSV
- docs
- notebook

It intentionally does not include old `runs/` folders, because old local or mock outputs could confuse scoring.

## Why The Models Are Run Correctly

The notebook sets:

```text
EIWS_RUN_BACKEND=local_transformers
```

That makes `scripts/run_experiment.py` use the local Transformers backend.

For each configured model, the runner:

1. loads the tokenizer from Hugging Face
2. loads the model with `AutoModelForCausalLM`
3. uses 4-bit `bitsandbytes` quantization
4. uses `device_map="auto"`
5. applies the tokenizer chat template when available
6. runs deterministic generation with `do_sample=False` and `temperature=0.0`
7. unloads the model before moving to the next model

This is the correct approach for a T4 because both 7B models should not be kept in VRAM at the same time.

The experiment itself is also run correctly because the notebook calls:

```bash
python scripts/run_experiment.py --stage dry_run
python scripts/run_experiment.py --stage pilot
python scripts/run_experiment.py --stage full --model-label ...
```

So it uses the same frozen configs, personas, pressure templates, dataset, and logging schema as the repository.

## Recommended Method: Google Drive

Use this method for the real full run. It keeps the project and outputs after Colab disconnects.

### Step 1: Upload The ZIP To Your Google Drive Folder

1. Open Google Drive in your browser.
2. Go to this folder:

```text
My Drive/Colab Notebooks/nlp-eiws
```

3. Upload this file:

```text
dist/eiws_colab_t4_package.zip
```

After upload, the file should be here:

```text
My Drive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package.zip
```

Do not unzip it manually in Google Drive. The notebook will unzip it.

### Step 2: Open The Notebook In Colab

1. Go to:

```text
https://colab.research.google.com
```

2. Click `File`.
3. Click `Upload notebook`.
4. Upload:

```text
eiws_colab_t4_both_models_full.ipynb
```

### Step 3: Select A T4 GPU

In Colab:

1. Click `Runtime`.
2. Click `Change runtime type`.
3. Set `Hardware accelerator` to `GPU`.
4. Set `GPU type` to `T4`.
5. Click `Save`.

Then restart the runtime if Colab asks.

### Step 4: Add A Hugging Face Token

This is recommended because the notebook downloads both models from Hugging Face.

In Colab:

1. Click the key icon in the left sidebar.
2. Add a new secret.
3. Name it:

```text
HF_TOKEN
```

4. Paste your Hugging Face token as the value.
5. Enable notebook access for that secret.

Do not paste the token directly into the notebook.

### Step 5: Let The Notebook Mount Drive And Unzip The Project

Run the notebook cells in order. In the `Locate The Project` section, the notebook will:

1. mount Google Drive
2. look for this ZIP:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package.zip
```

3. unzip it into:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package
```

4. switch into that extracted project folder

You do not need to add a separate mount/unzip cell anymore.

If you want to do it manually anyway, this is the equivalent code:

```python
from google.colab import drive
drive.mount("/content/drive")

!unzip -q -o "/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package.zip" -d "/content/drive/MyDrive/Colab Notebooks/nlp-eiws/"
%cd "/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package"
```

Expected output should end with something like:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package
```

### Step 6: Run The Notebook From The Top

Now run the notebook cells in order.

The project-location cell should print:

```text
PROJECT_ROOT: /content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package
Files checked: True
```

If it prints that, the notebook found the project correctly.

### Step 7: Keep The Dataset Setting As-Is

In the dataset section, keep:

```python
FORCE_REBUILD_DATASET = False
```

The ZIP already includes:

```text
data/frozen/conversations.csv
```

You only need to set `FORCE_REBUILD_DATASET = True` if you deliberately want to download `cais/mmlu` again and rebuild the same frozen dataset.

### Step 8: Run Dry-Run And Pilot First

In the run-control cell, start with:

```python
RUN_DRY_RUN = True
RUN_PILOT = True
RUN_FULL = False
```

Run the dry-run and pilot cells.

Do not start the full run until these validate successfully.

Expected run sizes:

- dry-run: 24 conversations
- pilot: 72 conversations

Each conversation has two generations:

- initial answer
- final answer after pressure

### Step 9: Run The Full Experiment One Model At A Time

After dry-run and pilot pass, change the run-control cell to run Mistral first:

```python
RUN_DRY_RUN = False
RUN_PILOT = False
RUN_FULL = True

FULL_MODEL_LABELS = ["mistral_7b_instruct_v0_3"]
RUN_FULL_ONE_MODEL_AT_A_TIME = True
```

Run the full-run cell.

When it finishes, run BioMistral:

```python
RUN_DRY_RUN = False
RUN_PILOT = False
RUN_FULL = True

FULL_MODEL_LABELS = ["biomistral_7b"]
RUN_FULL_ONE_MODEL_AT_A_TIME = True
```

Run the full-run cell again.

This creates two full run directories, one for each model.

### Step 10: Score And Package Results

After both full model runs finish, run the remaining notebook sections:

1. `Load Run Records And Score Responses`
2. `Aggregate Metrics And Save Results`
3. `Figures`
4. `Completeness Audit And Methods Summary`
5. `Package Results`

The scoring cell will look for full `local_transformers` runs and combine the latest run directories that cover both configured models.

## Output Locations

Because you are using Drive, outputs will be saved here:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package/runs/
```

Full runs will be under:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package/runs/full/
```

Analysis outputs will be under:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package/results/notebook_analysis/
```

The final notebook cell creates a ZIP archive of the analysis results folder.

## Fallback Method: Upload ZIP When Prompted

Use this only for dry-run or pilot, not for the real full run, because `/content` is temporary.

1. Open `eiws_colab_t4_both_models_full.ipynb` in Colab.
2. Set runtime to T4 GPU.
3. Run cells from the top.
4. When the project-location cell says the project was not found, upload:

```text
dist/eiws_colab_t4_package.zip
```

5. The notebook extracts the ZIP into `/content/eiws_project_upload`.
6. Continue running the notebook.

This works, but if Colab resets, outputs under `/content` are lost.

## What Success Looks Like

The dry-run or pilot validation should print:

```text
Backend: local_transformers
Status counts: {'success': ...}
```

For the active protocol, scoring should only see:

```text
claim_truth = incorrect_claim
backend = local_transformers
```

If it sees old mock runs or old correct-claim subsets, it will stop instead of silently scoring the wrong artifacts.

## Common Problems

### No CUDA GPU Found

Fix:

1. Go to `Runtime`.
2. Click `Change runtime type`.
3. Choose GPU/T4.
4. Restart runtime.
5. Rerun from the top.

### Out Of Memory

Fix:

1. Restart runtime.
2. Keep `RUN_FULL_ONE_MODEL_AT_A_TIME = True`.
3. Run only one model label in that session.

Example:

```python
FULL_MODEL_LABELS = ["mistral_7b_instruct_v0_3"]
```

Then later:

```python
FULL_MODEL_LABELS = ["biomistral_7b"]
```

### Hugging Face Download Fails

Fix:

1. Add `HF_TOKEN` in Colab secrets.
2. Make sure notebook access is enabled for that secret.
3. Restart runtime and rerun.

### Colab Disconnects

If you used Google Drive, your completed run directory should still exist.

Rerun the notebook, mount Drive again, change to:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package
```

Then continue with the missing model or with scoring.

### ZIP Extracted But Notebook Still Asked For Upload

This means the notebook extracted the ZIP but did not recognize the project root afterward. Use the updated notebook from this repository; its project-location cell now searches recursively under:

```text
/content/drive/MyDrive/Colab Notebooks/nlp-eiws
```

and prints a debug tree instead of opening the upload prompt blindly.

If you need to inspect the folder manually in Colab, run:

```python
from pathlib import Path

base = Path("/content/drive/MyDrive/Colab Notebooks/nlp-eiws")
for path in sorted(base.rglob("*"))[:120]:
    print(path.relative_to(base))
```

The correct extracted folder should contain:

```text
eiws_colab_t4_package/main.md
eiws_colab_t4_package/MUST.md
eiws_colab_t4_package/configs/run_settings.yaml
eiws_colab_t4_package/scripts/run_experiment.py
eiws_colab_t4_package/scripts/build_dataset.py
```

## Final Recommendation

For the real experiment, use Google Drive, run dry-run, run pilot, then run the full stage one model at a time.

That is the most reliable way to run this project on a 15 GB Colab T4 without changing the experiment design.
