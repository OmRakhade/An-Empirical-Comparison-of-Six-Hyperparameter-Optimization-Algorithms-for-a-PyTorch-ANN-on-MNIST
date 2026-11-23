"""Plots for comparing optimizer results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def plot_comparison(summary: list[dict[str, Any]], output_path: str | Path) -> None:
    if not summary:
        raise ValueError("Cannot plot an empty experiment summary.")
    names = [item["algorithm"] for item in summary]
    accuracies = [item["test_metrics"]["test_accuracy"] for item in summary]
    runtimes = [item["search_seconds"] for item in summary]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, accuracies, color="#4472C4")
    axes[0].set_title("Best configuration: test accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(names, runtimes, color="#70AD47")
    axes[1].set_title("Hyperparameter search runtime")
    axes[1].set_ylabel("Seconds")
    for axis in axes:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def plot_metric_dashboard(summary: list[dict[str, Any]], output_path: str | Path) -> None:
    if not summary:
        raise ValueError("Cannot plot an empty experiment summary.")
    names = [item["algorithm"] for item in summary]
    metrics = (
        ("test_accuracy", "Accuracy"),
        ("test_precision_macro", "Macro precision"),
        ("test_recall_macro", "Macro recall"),
        ("test_f1_macro", "Macro F1"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, title) in zip(axes.flat, metrics):
        values = [item["test_metrics"][key] for item in summary]
        bars = axis.bar(names, values, color="#4472C4")
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=30)
        axis.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    figure.suptitle("MNIST test-set metrics for each optimizer", fontsize=14)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_validation_trials(summary: list[dict[str, Any]], output_path: str | Path) -> None:
    if not summary:
        raise ValueError("Cannot plot an empty experiment summary.")
    figure, axis = plt.subplots(figsize=(11, 6))
    for item in summary:
        scores = [trial["metrics"]["val_accuracy"] for trial in item["trials"]]
        axis.scatter(
            range(1, len(scores) + 1),
            scores,
            label=item["algorithm"],
            alpha=0.8,
        )
    axis.set_title("Validation accuracy by candidate evaluation")
    axis.set_xlabel("Evaluation number within optimizer")
    axis.set_ylabel("Validation accuracy")
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_ann_architecture(output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 8))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 12)
    axis.axis("off")
    layers = [
        (9.5, "Input", "784 values (28 × 28 pixels)", "#D9EAF7"),
        (7.2, "Hidden layer 1", "H neurons · ReLU · Dropout", "#DDEBF7"),
        (4.9, "Hidden layer 2", "H / 2 neurons · ReLU · Dropout", "#DDEBF7"),
        (2.6, "Output", "10 logits · Cross-entropy loss", "#E2F0D9"),
    ]
    for y, title, detail, color in layers:
        box = FancyBboxPatch(
            (1, y - 0.75),
            8,
            1.5,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=color,
            edgecolor="#506070",
            linewidth=1.2,
        )
        axis.add_patch(box)
        axis.text(5, y + 0.2, title, ha="center", va="center", fontsize=13, weight="bold")
        axis.text(5, y - 0.25, detail, ha="center", va="center", fontsize=10)
    for upper, lower in zip(layers, layers[1:]):
        axis.annotate(
            "",
            xy=(5, lower[0] + 0.85),
            xytext=(5, upper[0] - 0.85),
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#506070"},
        )
    axis.text(5, 11.1, "MNIST ANN architecture", ha="center", fontsize=16, weight="bold")
    axis.text(5, 0.8, "H is tuned from 32 to 256; the second layer uses half of H.", ha="center", fontsize=9)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_experiment_workflow(output_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 5)
    axis.axis("off")
    steps = [
        (0.25, "MNIST\ntraining data", "#D9EAF7"),
        (2.55, "Shared 10,000-image\nsearch subset + validation", "#D9EAF7"),
        (5.0, "Six optimizers\nsame search space", "#FFF2CC"),
        (7.45, "Select best config\nusing validation only", "#FFF2CC"),
        (9.9, "Retrain on all\n60,000 train images", "#E2F0D9"),
        (12.0, "Evaluate once on\n10,000 test images", "#FCE4D6"),
    ]
    widths = [1.65, 2.0, 1.8, 1.8, 1.8, 1.7]
    for (x, label, color), width in zip(steps, widths):
        box = FancyBboxPatch(
            (x, 2.1),
            width,
            1.25,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=color,
            edgecolor="#506070",
            linewidth=1.1,
        )
        axis.add_patch(box)
        axis.text(x + width / 2, 2.725, label, ha="center", va="center", fontsize=9)
    for index in range(len(steps) - 1):
        x, _, _ = steps[index]
        next_x, _, _ = steps[index + 1]
        axis.annotate(
            "",
            xy=(next_x - 0.08, 2.725),
            xytext=(x + widths[index] + 0.08, 2.725),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#506070"},
        )
    axis.text(7, 4.35, "Experimental workflow and test-set isolation", ha="center", fontsize=15, weight="bold")
    axis.text(7, 1.1, "Validation drives tuning; the official test set is reserved for final reporting.", ha="center", fontsize=10)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
