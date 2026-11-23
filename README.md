# MNIST ANN Hyperparameter Optimization

Compare six hyperparameter search algorithms using the same two-hidden-layer
PyTorch ANN and MNIST classification task:

- Grid search
- Random search
- Gaussian-process Bayesian optimization (expected improvement)
- Hyperband-style successive halving
- Genetic algorithm
- Particle swarm optimization (PSO)

The network takes 28 × 28 images (784 input values), uses two ReLU hidden
layers, and produces 10 class logits for cross-entropy loss. The first hidden
layer width is optimized; the second is half that width (128 → 64 by default).

## Setup

Use Python 3.10 or later. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

MNIST is downloaded automatically to `data/` on the first run.

## Quick run

The default is intended for development: 8 evaluations per algorithm, 3
epochs, and a fixed stratified 10,000-image subset for hyperparameter search.
Each winning configuration is then retrained on all 60,000 MNIST training
images and evaluated once on the official test set.

```powershell
python -m experiments.benchmark
```

Select algorithms or adjust the compute budget:

```powershell
python -m experiments.benchmark --algorithms random_search bayesian hyperband --budget 12 --epochs 5
```

For a fuller research run, use all training examples in the search as well as a
larger evaluation budget and epoch count:

```powershell
python -m experiments.benchmark --subset-size 0 --budget 30 --epochs 10
```

The full run can take substantially longer. To use a GPU when available, add
`--device cuda`; `--device auto` selects CUDA when available and otherwise CPU.
Run `python -m experiments.benchmark --help` for all options.

## Experimental fairness

- Every optimizer uses the same search space, fixed validation split, seed,
  maximum evaluation budget, and maximum epoch count.
- Learning rate and weight decay are sampled logarithmically; batch size is
  categorical; hidden width is an integer; dropout is continuous.
- The search objective is validation accuracy (validation loss breaks ties).
- Hyperband promotes the same candidate model to more epochs rather than
  restarting its training; other algorithms evaluate each candidate from
  scratch.
- The official test set is excluded from selection and used only to report the
  final retrained model's accuracy, macro precision, macro recall, macro F1,
  and loss.
- Record the command, hardware, Python/package versions, and seed when
  reporting results. For stronger conclusions, repeat the comparison across
  multiple seeds.

Results are written to `results/benchmark.json`, `results/benchmark.csv`, and
`results/comparison.png`. The JSON includes each trial's validation metrics and
epoch budget; the CSV and plot summarize final test performance and search
runtime.
