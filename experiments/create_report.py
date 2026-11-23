"""Generate report-ready tables, plots, and diagrams from benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.visualization import (
    plot_ann_architecture,
    plot_experiment_workflow,
    plot_metric_dashboard,
    plot_validation_trials,
)


def create_report(
    results_path: Path,
    output_dir: Path,
    seed: int,
    subset_size: int,
    epochs: int,
    budget: int,
    device: str,
) -> Path:
    with results_path.open(encoding="utf-8") as file:
        summary = json.load(file)
    if not isinstance(summary, list) or not summary:
        raise ValueError(f"{results_path} must contain a non-empty benchmark result list.")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric_dashboard(summary, output_dir / "metrics_dashboard.png")
    plot_validation_trials(summary, output_dir / "validation_trials.png")
    plot_ann_architecture(output_dir / "mnist_ann_architecture.png")
    plot_experiment_workflow(output_dir / "experiment_workflow.png")

    ranked = sorted(
        summary,
        key=lambda item: item["test_metrics"]["test_accuracy"],
        reverse=True,
    )
    report_path = output_dir / "experiment_report.md"
    report_path.write_text(
        _render_report(ranked, seed, subset_size, epochs, budget, device),
        encoding="utf-8",
    )
    return report_path


def _render_report(
    ranked: list[dict[str, Any]],
    seed: int,
    subset_size: int,
    epochs: int,
    budget: int,
    device: str,
) -> str:
    rows = []
    config_rows = []
    for rank, item in enumerate(ranked, start=1):
        metrics = item["test_metrics"]
        rows.append(
            "| {rank} | {algorithm} | {accuracy:.4f} | {precision:.4f} | "
            "{recall:.4f} | {f1:.4f} | {loss:.4f} | {runtime:.1f} | {trials} |".format(
                rank=rank,
                algorithm=item["algorithm"],
                accuracy=metrics["test_accuracy"],
                precision=metrics["test_precision_macro"],
                recall=metrics["test_recall_macro"],
                f1=metrics["test_f1_macro"],
                loss=metrics["test_loss"],
                runtime=item["search_seconds"],
                trials=len(item["trials"]),
            )
        )
        config = json.dumps(item["best_config"], sort_keys=True, separators=(",", ":"))
        config_rows.append(
            f"| {item['algorithm']} | `{config}` | "
            f"{item['best_validation_metrics']['val_accuracy']:.4f} |"
        )

    max_used_trials = max(len(item["trials"]) for item in ranked)
    return f"""# MNIST ANN Hyperparameter Optimization — Benchmark Report

## Summary

This report compares six hyperparameter optimizers on MNIST using a PyTorch
two-hidden-layer fully connected neural network. Each optimizer selects a
configuration using validation accuracy (validation loss breaks ties). The
selected configuration is then retrained on all 60,000 MNIST training images
and evaluated on the official 10,000-image test set. The test set is not used
to select hyperparameters.

This is a **preliminary, single-seed comparison**, not evidence that one method
is statistically superior. Read the limitations below before citing the
results as a final conclusion.

## Experimental setup

| Setting | Value |
|---|---|
| Dataset | MNIST, 28 × 28 grayscale images, 10 classes |
| Search data | Fixed stratified subset of {subset_size:,} training images; 10% held out for validation |
| Final fit | All 60,000 training images |
| Test set | Official 10,000 MNIST test images |
| Network | 784 inputs → H ReLU units → H/2 ReLU units → 10 logits |
| Optimizer | Adam; cross-entropy objective |
| Tuned parameters | Learning rate, batch size, H, dropout, weight decay |
| Maximum evaluations per optimizer | {budget} |
| Search epoch limit | {epochs} per candidate; Hyperband allocates epochs across rungs |
| Random seed | {seed} |
| Device | {device} |

Learning rate and weight decay are searched logarithmically; batch size is
categorical. Hidden width is an integer from 32 to 256, and dropout is between
0 and 0.5. The second hidden layer has half as many neurons as the first.

## Results

Ranked by final test accuracy. All metric values are proportions from 0 to 1.
Search runtime is wall-clock time for the optimizer's validation evaluations;
it excludes final full-training-set retraining.

| Rank | Optimizer | Test accuracy | Macro precision | Macro recall | Macro F1 | Test loss | Search seconds | Evaluations |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

### Selected configurations

| Optimizer | Best configuration | Best validation accuracy |
|---|---|---:|
{chr(10).join(config_rows)}

![Test metric comparison](metrics_dashboard.png)

![Validation accuracy by candidate evaluation](validation_trials.png)

## Diagrams

![MNIST ANN architecture](mnist_ann_architecture.png)

![Experiment workflow and test-set isolation](experiment_workflow.png)

## Interpretation and limitations

- Each optimizer was run once with seed {seed}; performance variation across
  seeds is not measured.
- The search used {subset_size:,} training examples rather than the full
  training set, and only {epochs} search epochs. These choices keep the
  comparison practical but can change which configurations perform best after
  full-data retraining.
- Each optimizer had a maximum of {budget} candidate evaluations. The actual
  number used by each method is shown in the results table (maximum observed:
  {max_used_trials}). Hyperband deliberately assigns different epoch counts to
  different candidates, so equal candidate counts do not imply equal training
  compute; search runtime is reported separately.
- Final test metrics are for each optimizer's single selected configuration
  retrained for {epochs} epochs. The test set is reserved for reporting, but
  avoid repeatedly changing the search based on these test results.
- A stronger final study should repeat with several predeclared seeds, increase
  the search budget and epoch count, and report mean ± standard deviation.
  For strict compute-budget fairness, also compare total training examples
  processed or total epochs, not only candidate evaluations.

## Reproducibility

Command used:

```powershell
python -m experiments.benchmark --budget {budget} --epochs {epochs} --subset-size {subset_size} --seed {seed} --device {device} --output-dir results\\report_run
```

Raw per-trial results are in `benchmark.json`; a flat summary is in
`benchmark.csv`. This report and its figures were generated from the JSON
results, without manually editing metric values.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/report_run/benchmark.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/report_run"))
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report_path = create_report(
        args.results,
        args.output_dir,
        args.seed,
        args.subset_size,
        args.epochs,
        args.budget,
        args.device,
    )
    print(f"Generated report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
