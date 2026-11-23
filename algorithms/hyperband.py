"""Successive-halving search that increases training epochs for survivors."""

from __future__ import annotations

import math
import random
from typing import Any

from algorithms.base import Evaluator, evaluate, result
from algorithms.search_space import SearchSpace


def run(
    space: SearchSpace, evaluator: Evaluator, budget: int, seed: int, epochs: int
) -> dict[str, Any]:
    if budget < 1:
        raise ValueError("budget must be at least 1")
    rng = random.Random(seed)
    initial_count = max(1, budget // 2)
    candidates = [
        (f"hyperband-{index}", space.sample(rng))
        for index in range(initial_count)
    ]
    trials = []
    used = 0
    rung = 0

    while candidates and used < budget:
        remaining = budget - used
        if len(candidates) > remaining:
            candidates = candidates[:remaining]
        fraction = min(1.0, 2**rung / max(1, initial_count))
        resource = max(1, math.ceil(epochs * fraction))
        rung_trials = [
            (trial_id, config, evaluate(evaluator, config, resource, trial_id))
            for trial_id, config in candidates
        ]
        trials.extend(item[2] for item in rung_trials)
        used += len(rung_trials)
        if len(rung_trials) <= 1 or used >= budget:
            break

        rung_trials.sort(
            key=lambda item: (
                item[2]["metrics"]["val_accuracy"],
                -item[2]["metrics"]["val_loss"],
            ),
            reverse=True,
        )
        survivors = {item[0] for item in rung_trials[: max(1, math.ceil(len(rung_trials) / 2))]}
        for trial_id, _, _ in rung_trials:
            if trial_id not in survivors:
                release = getattr(evaluator, "release_trial", None)
                if release is not None:
                    release(trial_id)
        candidates = [(trial_id, config) for trial_id, config, _ in rung_trials if trial_id in survivors]
        rung += 1

    return result(trials)
