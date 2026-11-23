"""Shared search-space definition used by every optimizer."""

from __future__ import annotations

import itertools
import math
import random
from typing import Any


class SearchSpace:
    """Encode and decode the five MNIST ANN hyperparameters."""

    names = ("learning_rate", "batch_size", "hidden_neurons", "dropout", "weight_decay")
    batch_sizes = (16, 32, 64, 128)

    def sample(self, rng: random.Random) -> dict[str, int | float]:
        return {
            "learning_rate": 10 ** rng.uniform(-4, -1),
            "batch_size": rng.choice(self.batch_sizes),
            "hidden_neurons": rng.randint(32, 256),
            "dropout": rng.uniform(0.0, 0.5),
            "weight_decay": 10 ** rng.uniform(-5, -2),
        }

    def encode(self, config: dict[str, Any]) -> list[float]:
        batch_index = self.batch_sizes.index(int(config["batch_size"]))
        return [
            (math.log10(float(config["learning_rate"])) + 4) / 3,
            batch_index / (len(self.batch_sizes) - 1),
            (int(config["hidden_neurons"]) - 32) / 224,
            float(config["dropout"]) / 0.5,
            (math.log10(float(config["weight_decay"])) + 5) / 3,
        ]

    def decode(self, vector: list[float]) -> dict[str, int | float]:
        values = [min(1.0, max(0.0, float(value))) for value in vector]
        batch_index = round(values[1] * (len(self.batch_sizes) - 1))
        return {
            "learning_rate": 10 ** (-4 + 3 * values[0]),
            "batch_size": self.batch_sizes[batch_index],
            "hidden_neurons": round(32 + 224 * values[2]),
            "dropout": 0.5 * values[3],
            "weight_decay": 10 ** (-5 + 3 * values[4]),
        }

    def grid(self, budget: int, rng: random.Random) -> list[dict[str, int | float]]:
        """Build a bounded grid and select at most `budget` grid points."""
        levels = max(2, math.ceil(max(1, budget) ** (1 / len(self.names))))
        axes: list[list[int | float]] = [
            _log_levels(-4, -1, min(levels, 4)),
            _evenly_spaced(list(self.batch_sizes), min(levels, len(self.batch_sizes))),
            _evenly_spaced(list(range(32, 257)), min(levels, 8)),
            _evenly_spaced([i / 100 for i in range(51)], min(levels, 6)),
            _log_levels(-5, -2, min(levels, 4)),
        ]
        points = [
            dict(zip(self.names, values))
            for values in itertools.product(*axes)
        ]
        if len(points) > budget:
            points = rng.sample(points, budget)
        return points


def _log_levels(low: int, high: int, count: int) -> list[float]:
    if count == 1:
        return [10**low]
    return [10 ** (low + (high - low) * index / (count - 1)) for index in range(count)]


def _evenly_spaced(values: list[int | float], count: int) -> list[int | float]:
    if count >= len(values):
        return values
    indices = [round(index * (len(values) - 1) / (count - 1)) for index in range(count)]
    return [values[index] for index in indices]
