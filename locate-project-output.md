Mounted at /content/drive
Extracting package: /content/drive/MyDrive/Colab Notebooks/nlp-eiws/eiws_colab_t4_package.zip
ZIP top-level entries: ['eiws_colab_t4_package\\MUST.md', 'eiws_colab_t4_package\\README.md', 'eiws_colab_t4_package\\configs\\personas.yaml', 'eiws_colab_t4_package\\configs\\pressure.yaml', 'eiws_colab_t4_package\\configs\\run_schema.md', 'eiws_colab_t4_package\\configs\\run_settings.yaml', 'eiws_colab_t4_package\\data\\', 'eiws_colab_t4_package\\data\\frozen\\conversations.csv', 'eiws_colab_t4_package\\docs\\colab-t4-exact-steps.md', 'eiws_colab_t4_package\\docs\\colab-t4-single-notebook-run.md', 'eiws_colab_t4_package\\docs\\google-colab-t4.md', 'eiws_colab_t4_package\\eiws_colab_t4_both_models_full.ipynb', 'eiws_colab_t4_package\\main.md', 'eiws_colab_t4_package\\scripts\\build_dataset.py', 'eiws_colab_t4_package\\scripts\\run_experiment.py', 'eiws_colab_t4_package\\work-divided-example.md']
Debug tree for: /content/drive/MyDrive/Colab Notebooks/nlp-eiws
  - eiws_colab_t4_package.zip
  - eiws_colab_t4_package\data\
  - eiws_colab_t4_package\eiws_colab_t4_both_models_full.ipynb
  - eiws_colab_t4_package\main.md
  - eiws_colab_t4_package\MUST.md
  - eiws_colab_t4_package\README.md
  - eiws_colab_t4_package\work-divided-example.md
  - eiws_colab_t4_package\configs\personas.yaml
  - eiws_colab_t4_package\configs\pressure.yaml
  - eiws_colab_t4_package\configs\run_schema.md
  - eiws_colab_t4_package\configs\run_settings.yaml
  - eiws_colab_t4_package\data\frozen\conversations.csv
  - eiws_colab_t4_package\docs\colab-t4-exact-steps.md
  - eiws_colab_t4_package\docs\colab-t4-single-notebook-run.md
  - eiws_colab_t4_package\docs\google-colab-t4.md
  - eiws_colab_t4_package\scripts\build_dataset.py
  - eiws_colab_t4_package\scripts\run_experiment.py
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_8475/1039978436.py in <cell line: 0>()
    150     PROJECT_ROOT, saw_drive_zip = extract_drive_project_zip()
    151 if PROJECT_ROOT is None and saw_drive_zip:
--> 152     raise RuntimeError(
    153         "The Drive ZIP was found and extracted, but the project root was not found. "
    154         "Check the debug tree above and make sure the ZIP contains main.md, MUST.md, configs/, and scripts/."

RuntimeError: The Drive ZIP was found and extracted, but the project root was not found. Check the debug tree above and make sure the ZIP contains main.md, MUST.md, configs/, and scripts/.
