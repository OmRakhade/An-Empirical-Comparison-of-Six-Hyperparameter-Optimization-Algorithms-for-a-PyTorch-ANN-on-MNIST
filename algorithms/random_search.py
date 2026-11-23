"""Uniform random search over the shared hyperparameter space."""

from __future__ import annotations

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
    trials = [
        evaluate(evaluator, space.sample(rng), epochs)
        for _ in range(budget)
    ]
    return result(trials)
