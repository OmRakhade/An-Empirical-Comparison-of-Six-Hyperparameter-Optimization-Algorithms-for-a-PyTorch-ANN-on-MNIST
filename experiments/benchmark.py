"""Run a repeatable, equal-budget MNIST comparison of search algorithms."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

from algorithms import bayesian, genetic_algorithm, grid_search, hyperband, pso, random_search
from algorithms.search_space import SearchSpace
from evaluation.visualization import plot_comparison
from experiments.mnist_experiment import MNISTObjective, fit_and_test, load_mnist, set_seed


ALGORITHMS = {
    "grid_search": grid_search,
    "random_search": random_search,
    "bayesian": bayesian,
    "hyperband": hyperband,
    "genetic_algorithm": genetic_algorithm,
    "pso": pso,
}


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    set_seed(args.seed)
    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Loading MNIST (search subset: {args.subset_size or 'full training set'})...")
    data = load_mnist(args.data_dir, args.subset_size or None, args.seed)
    summary = []

    for name in args.algorithms:
        print(
            f"\n[{name}] budget={args.budget} trials, "
            f"search epochs={args.epochs}, seed={args.seed}",
            flush=True,
        )
        objective = MNISTObjective(
            data.search_train,
            data.validation,
            device,
            args.seed,
            verbose=not args.quiet,
        )
        started = time.perf_counter()
        search = ALGORITHMS[name].run(
            SearchSpace(), objective, args.budget, args.seed, args.epochs
        )
        search_seconds = time.perf_counter() - started
        print(f"  Best validation accuracy: {search['best_metrics']['val_accuracy']:.4f}")
        print(f"  Best configuration: {search['best_config']}")
        print("  Retraining best configuration on all 60,000 training images...")
        test_metrics = fit_and_test(
            search["best_config"],
            data.full_train,
            data.test,
            args.epochs,
            device,
            args.seed,
        )
        print(f"  Final test accuracy: {test_metrics['test_accuracy']:.4f}")
        summary.append(
            {
                "algorithm": name,
                "search_seconds": search_seconds,
                "best_config": search["best_config"],
                "best_validation_metrics": search["best_metrics"],
                "test_metrics": test_metrics,
                "trials": search["trials"],
            }
        )

    save_results(summary, Path(args.output_dir))
    return summary


def save_results(summary: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    csv_path = output_dir / "benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        columns = [
            "algorithm",
            "search_seconds",
            "test_accuracy",
            "test_precision_macro",
            "test_recall_macro",
            "test_f1_macro",
            "test_loss",
            "best_config",
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for item in summary:
            writer.writerow(
                {
                    "algorithm": item["algorithm"],
                    "search_seconds": item["search_seconds"],
                    **item["test_metrics"],
                    "best_config": json.dumps(item["best_config"], sort_keys=True),
                }
            )
    plot_comparison(summary, output_dir / "comparison.png")
    print(f"\nSaved results to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=list(ALGORITHMS))
    parser.add_argument("--budget", type=int, default=8, help="Maximum candidate evaluations per algorithm.")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum search and final-fit epochs.")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Stratified training subset used during search; use 0 for all 60,000 examples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-trial progress.")
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.subset_size < 0:
        parser.error("--subset-size cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available.")
    return args


if __name__ == "__main__":
    run_benchmark(parse_args())
