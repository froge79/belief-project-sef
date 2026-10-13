# Belief-Rigidity Steering

This was my first research project. Did everything in colab notebooks (lol) and everything was messy, but it worked out in the end and I'm glad I switched to .py script files. It tests whether a linear "belief rigidity" direction can be found in a language model's activations and if it can be used to steer its responses toward more CBT-style, flexible reframing of distressed Reddit posts.

## The problem

Depressive thinking is often marked by rigid, absolute beliefs ("I'll always be alone"). Cognitive behavioural therapy (CBT) works by making such beliefs more flexible. If an LLM represents belief rigidity as a linear direction in its hidden states, adding or subtracting that direction at inference time (activation steering) might make its supportive responses more therapeutic, with no fine-tuning. This project checks whether that holds, using a paired design with an unsteered baseline and a random-vector control.

## How it works

The pipeline has four phases. Phases 1–4 are Colab-style notebooks in the repository root. The `scripts/` folder holds the API-driven data preparation, grading and statistics.

1. **Probing** (`phase1.ipynb`, `phase2.ipynb`, `phase3.ipynb`): extract activations from `mistralai/Mistral-7B-Instruct-v0.3` for [Dreaddit](https://arxiv.org/abs/1911.00123) posts and train linear probes for belief rigidity. Validate them with shuffled-label, text-length, sentiment and TF-IDF controls.
2. **Labels** (`scripts/label_beliefs.py`): GPT-4o rates each post from 1 (rigid) to 5 (flexible) and flags unusable posts for discard. These labels train the probes.
3. **Contrast pairs** (`scripts/generate_reframes.py`): GPT-4o writes a balanced first-person reframe of each post. The post and its reframe form a contrast pair for building the steering vector (contrastive activation addition).
4. **Steering runs** (`phase4.ipynb`): add the vector at a chosen layer and strength (alpha) through forward hooks while the model replies. Configurations are tuned over layer and alpha, then compared against an unsteered baseline (alpha = 0), a wrong layer, and a random direction.
5. **Grading** (`scripts/grade_responses.py`): an LLM judge scores each response from 0 to 4 on therapy quality, identifying the belief, asking for evidence, avoiding validation of distortions, and proposing a next step.
6. **Analysis** (`scripts/analyze_steering.py`, `scripts/plot_metrics.py`): paired bootstrap confidence intervals, sign-flip permutation tests with Holm correction, and equivalence tests (TOST). The steering-versus-random contrast tests whether the belief direction specifically does anything. Output is a LaTeX table, a CSV and figures.

## Usage

```bash
pip install -r scripts/requirements.txt
cp .env.example .env            # then add your OPENAI_API_KEY
cd scripts

python label_beliefs.py         # data/phase3_dreaddit_1000.csv -> output/..._labeled.csv
python generate_reframes.py     # data/phase4_dreaddit_800.csv  -> output/..._reframes.csv
python grade_responses.py --input-dir data/runs/final --output-dir output/graded_runs/final
python analyze_steering.py      # reads output/graded_runs/final_2
python plot_metrics.py
```

Run each script with `--help` for its options. Run `grade_responses.py` once per run directory; the tuning runs are in `data/runs/tuning`. The scripts use paths relative to `scripts/`, so run them from there. `.env` is read from the working directory or any parent, so the one in the project root works.

## Scope and limitations

- **Single model and layer range.** All results are for Mistral-7B-Instruct-v0.3, with steering tuned over a handful of layers (14–17) and alphas (0.5–4). They may not transfer to other models or settings.
- **LLM-generated labels and grades.** Belief labels, reframes and response scores all come from GPT-4o without human validation. Judge bias, drift and non-determinism are not quantified. Some grades (such as `belief_flexibility` in `output/graded_runs/final_2`) were produced outside the scripts in this repository.
- **Small sample.** The final comparison uses 373 posts. The equivalence tests and the "no effect" conclusion are only as strong as that sample and the 0.2 SD equivalence bound allow, and some comparisons are inconclusive.
- **Narrow data.** Dreaddit is Reddit text. It has no clinical diagnoses, so "depressive" and "rigid" mean what the labelling prompt says they mean. Posts longer than 200 words are truncated for reframe generation.
- **Not a clinical tool.** This is a research study of model internals. It is not validated for, and must not be used for, mental-health support or treatment.
- **Prompt text is frozen.** The prompts in `scripts/` are the ones used to produce the shipped data, including a few minor wording slips, and were left unchanged for reproducibility. The steering step itself lives in the notebooks and needs a GPU.
- **Data.** The CSVs derive from Dreaddit Reddit posts. Check the Dreaddit and Reddit terms before redistributing them.
