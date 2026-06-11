# Cell Explanations For `eiws_colab_t4_both_models_full.ipynb`

This file explains what each notebook cell does. The explanations follow the same order as the notebook.

## Opening

### Cell 0 - Markdown

I introduce the project and explain the main experimental contract. This cell states the important rules: two models, three personas, original MMLU questions in step 2, only incorrect user claims in step 4, and scoring step 5 only when step 3 was already correct.

## 1. Install Dependencies

### Cell 1 - Markdown

I mark the setup section of the notebook. This makes it clear that the next cell prepares the Colab runtime.

### Cell 2 - Code

I install the Python packages needed for the experiment. These include Transformers, Accelerate, bitsandbytes for 4-bit loading, datasets, Hugging Face utilities, pandas, tqdm, and matplotlib.

## 2. Imports And GPU Check

### Cell 3 - Markdown

I start the environment-check section. This separates package installation from the actual Python imports and runtime validation.

### Cell 4 - Code

I import the libraries used throughout the notebook and detect whether the notebook is running inside Colab. I also check that CUDA is available and print the GPU name and available VRAM, because the experiment needs a T4 GPU.

### Cell 5 - Code

I run `nvidia-smi` so I can see the GPU status directly from the notebook. This is a quick sanity check before loading any large model.

## 3. Locate The Project

### Cell 6 - Markdown

I explain that the notebook needs to find the repository folder. If the notebook is opened outside the repo, this section lets me upload a project ZIP in Colab.

### Cell 7 - Code

I define helper functions that search for the project root by checking for files like `main.md`, `MUST.md`, configs, and scripts. If the project is not found, the cell asks for a ZIP upload, extracts it, switches into the project directory, and prints the final project path.

## 4. Hugging Face Token And Backend

### Cell 8 - Markdown

I describe how Hugging Face authentication should be handled. The point is to avoid hardcoding a token in the notebook.

### Cell 9 - Code

I set the experiment backend to `local_transformers`, which runs the models on the Colab GPU. I also try to load `HF_TOKEN` from the environment or Colab secrets, and leave an optional interactive login switch if needed.

## 5. Load And Validate Project Configuration

### Cell 10 - Markdown

I start the configuration section. This is where the notebook reads the frozen project settings before building data or running models.

### Cell 11 - Code

I load `run_settings.yaml`, `personas.yaml`, and `pressure.yaml`. Then I display small tables for the configured models, personas, and pressure templates so I can quickly inspect what will be used.

### Cell 12 - Code

I assert that the loaded configuration matches the current project protocol. This checks the two exact models, three personas, four normal pressure templates, no reverse-pressure templates, `cais/mmlu`, all-incorrect claim subset counts, temperature `0.0`, and 4-bit local loading.

## 6. Rebuild The Frozen Dataset

### Cell 13 - Markdown

I explain that the dataset is rebuilt from the current repository script. This matters because the script now follows the updated rule where step 2 is the original MMLU question and the user claim appears only in step 4.

### Cell 14 - Code

I run `scripts/build_dataset.py` when `REBUILD_DATASET` is enabled. After that, I check that the expected frozen CSV exists.

### Cell 15 - Code

I load the frozen dataset and validate its structure. This cell checks the required columns, 400 total rows, balanced BIO/OOD pools, all incorrect user claims, balanced pressure templates, and confirms that the claim text appears in turn 4 but not in turn 2.

## 7. Run Controls

### Cell 16 - Markdown

I explain the size of the run and the intended execution order. The notebook is set up to run dry-run, pilot, and then full experiment.

### Cell 17 - Code

I define the run switches. By default, dry-run, pilot, and full run are enabled, and the full run processes Mistral and BioMistral one at a time.

## 8. Runner Helpers

### Cell 18 - Markdown

I start the runner-helper section. These helper functions keep the later execution cells short.

### Cell 19 - Code

I define functions for reading JSON, finding the newest run directory, running a project stage through `scripts/run_experiment.py`, and validating the run artifacts. I also create the `run_dirs` dictionary that stores the dry-run, pilot, and full run directories created during the notebook session.

## 9. Dry Run

### Cell 20 - Markdown

I mark the dry-run section. This stage is the first smoke test before doing heavier runs.

### Cell 21 - Code

I run the `dry_run` stage if it is enabled. The cell then validates that the expected records, manifest, summary, and output files were written.

## 10. Pilot Run

### Cell 22 - Markdown

I mark the pilot section. This is the intermediate test before committing to the full experiment.

### Cell 23 - Code

I run the `pilot` stage if it is enabled. The cell validates the pilot output in the same way as the dry run.

## 11. Full Run

### Cell 24 - Markdown

I explain the full-run strategy. The notebook runs one model per run directory by default, which is safer on a 15 GB T4 and easier to resume after a Colab disconnect.

### Cell 25 - Code

I run the full experiment if it is enabled. By default it runs Mistral first, clears GPU memory, then runs BioMistral, validating each run directory after completion.

## 12. Load Run Records And Score Responses

### Cell 26 - Markdown

I introduce the scoring section. The scoring is automatic and based on extracting the answer letter from each assistant response.

### Cell 27 - Code

I load records from the latest available run directories and merge them back with the frozen dataset. Then I extract the initial and final answer letters, mark correctness, identify pressure-evaluable rows where the initial answer was correct, calculate flip behavior, score sycophancy on a 0-2 scale, and flag rows that need manual review.

## 13. Aggregate Metrics And Save Results

### Cell 28 - Markdown

I start the results aggregation section. This separates raw scoring from summary tables.

### Cell 29 - Code

I create a timestamped results folder and build summary tables at several levels: overall, by model and persona, by domain, and by subset. I also save scored records, metric CSVs, and a manual-review sample.

## 14. Figures

### Cell 30 - Markdown

I mark the plotting section. These figures make the main comparison easier to inspect.

### Cell 31 - Code

I create bar charts for flip rate, mean sycophancy score, and initial accuracy. The figures are saved as PNG files inside the results folder.

## 15. Completeness Audit And Methods Summary

### Cell 32 - Markdown

I introduce the final audit section. This checks whether the run artifacts are complete before relying on the scores.

### Cell 33 - Code

I build a completeness audit from the manifest and summary files for each scored run. I also write a short methods-and-results brief that records the dataset, models, personas, conversation contract, scoring rule, and output files.

## 16. Package Results

### Cell 34 - Markdown

I mark the packaging section. This is the final step after scoring and analysis.

### Cell 35 - Code

I zip the results folder so it can be downloaded or saved easily. Downloading is controlled by `DOWNLOAD_RESULTS_ZIP`, so the notebook does not automatically start downloads unless I enable it.
