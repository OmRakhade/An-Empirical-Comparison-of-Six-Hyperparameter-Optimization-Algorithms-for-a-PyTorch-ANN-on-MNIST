"""Particle swarm optimization over normalized hyperparameter coordinates."""

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
    swarm_size = min(budget, max(4, round(budget**0.5)))
    positions = [space.encode(space.sample(rng)) for _ in range(swarm_size)]
    velocities = [[rng.uniform(-0.1, 0.1) for _ in space.names] for _ in positions]
    personal_positions = [position[:] for position in positions]
    personal_scores = [-1.0] * swarm_size
    global_position = positions[0][:]
    global_score = -1.0
    trials = []

    while len(trials) < budget:
        for index, position in enumerate(positions):
            if len(trials) >= budget:
                break
            config = space.decode(position)
            trial = evaluate(evaluator, config, epochs)
            trials.append(trial)
            score = float(trial["metrics"]["val_accuracy"])
            if score > personal_scores[index]:
                personal_scores[index] = score
                personal_positions[index] = position[:]
            if score > global_score:
                global_score = score
                global_position = position[:]

        for index, position in enumerate(positions):
            for dimension in range(len(position)):
                velocities[index][dimension] = (
                    0.5 * velocities[index][dimension]
                    + 1.2 * rng.random() * (personal_positions[index][dimension] - position[dimension])
                    + 1.2 * rng.random() * (global_position[dimension] - position[dimension])
                )
                position[dimension] = min(
                    1.0, max(0.0, position[dimension] + velocities[index][dimension])
                )

    return result(trials)
