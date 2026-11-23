"""Budget-limited exhaustive search over a discretized parameter grid."""

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
    candidates = space.grid(budget, random.Random(seed))
    trials = [evaluate(evaluator, config, epochs) for config in candidates]
    return result(trials)
