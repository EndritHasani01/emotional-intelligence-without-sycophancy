# Project explanation

In this project, I test if different assistant personas make language models more likely to change a correct answer when the user pushes back.

I use MMLU multiple-choice questions, so every question has a known correct answer.

The model first answers the normal question, then I add a pressure message where the user says they think a wrong answer is right.

I compare a general Mistral model with BioMistral, and I also compare three personas: baseline, warm supportive, and supportive truth-first.

At the end, I score the first answer and the final answer to see when the model stayed correct and when it gave in to the pressure.

## More technical explanation

The dataset is built with `scripts/build_dataset.py`, which takes questions from `cais/mmlu` and saves the final rows in `data/frozen/conversations.csv`.

Each row keeps the original question, the answer choices, the correct answer, the user's incorrect claimed answer, and the pressure message that will be used later.

The experiment runner reads the frozen CSV and creates the same five-step conversation every time, using the prompts from `configs/personas.yaml` and the pressure templates from `configs/pressure.yaml`.

For every question, the runner stores the first model answer and the final answer after pressure, plus metadata like the model, persona, subject pool, and subset.

When I score the results, I first check if the model answered correctly before pressure. I only use the final answer for pressure scoring if the first answer was already correct, because otherwise the model did not really "flip" under pressure.
