"""Shared types and trial bookkeeping for search algorithms."""

from __future__ import annotations

from typing import Any, Callable

Config = dict[str, int | float]
Metrics = dict[str, float]
Evaluator = Callable[[Config, int, str | None], Metrics]
Trial = dict[str, Any]


def evaluate(evaluator: Evaluator, config: Config, epochs: int, trial_id: str | None = None) -> Trial:
    metrics = evaluator(config, epochs, trial_id)
    return {"config": dict(config), "metrics": metrics, "epochs": epochs}


def best_trial(trials: list[Trial]) -> Trial:
    if not trials:
        raise ValueError("An optimizer must evaluate at least one configuration.")
    return max(
        trials,
        key=lambda trial: (
            trial["metrics"]["val_accuracy"],
            -trial["metrics"]["val_loss"],
        ),
    )


def result(trials: list[Trial]) -> dict[str, Any]:
    best = best_trial(trials)
    return {"best_config": best["config"], "best_metrics": best["metrics"], "trials": trials}
