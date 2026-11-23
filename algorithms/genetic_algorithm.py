"""A small real-coded genetic optimizer."""

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
    population_size = min(budget, max(4, round(budget**0.5)))
    population = [space.sample(rng) for _ in range(population_size)]
    trials = []

    while len(trials) < budget:
        config = population[len(trials) % len(population)] if len(trials) < population_size else _offspring(population, trials, space, rng)
        trial = evaluate(evaluator, config, epochs)
        trials.append(trial)
        population.append(config)
        if len(population) > population_size * 2:
            population.sort(
                key=lambda item: _score(item, trials),
                reverse=True,
            )
            population = population[:population_size]

    return result(trials)


def _offspring(population, trials, space: SearchSpace, rng: random.Random):
    score_by_key = {
        _key(trial["config"]): trial["metrics"]["val_accuracy"]
        for trial in trials
    }

    def tournament():
        pool = rng.sample(population, min(3, len(population)))
        return max(pool, key=lambda config: score_by_key.get(_key(config), -1.0))

    first, second = tournament(), tournament()
    left, right = space.encode(first), space.encode(second)
    child = [
        left[index] if rng.random() < 0.5 else right[index]
        for index in range(len(left))
    ]
    for index in range(len(child)):
        if rng.random() < 0.25:
            child[index] += rng.gauss(0, 0.18)
    return space.decode(child)


def _score(config, trials):
    for trial in reversed(trials):
        if _key(trial["config"]) == _key(config):
            return trial["metrics"]["val_accuracy"]
    return -1.0


def _key(config):
    return tuple(round(float(config[name]), 10) for name in SearchSpace.names)
