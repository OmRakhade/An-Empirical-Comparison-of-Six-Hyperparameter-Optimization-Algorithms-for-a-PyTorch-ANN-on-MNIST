"""Gaussian-process Bayesian optimization with expected improvement."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from algorithms.base import Evaluator, evaluate, result
from algorithms.search_space import SearchSpace


def run(
    space: SearchSpace, evaluator: Evaluator, budget: int, seed: int, epochs: int
) -> dict[str, Any]:
    if budget < 1:
        raise ValueError("budget must be at least 1")
    rng = random.Random(seed)
    trials = []
    points: list[list[float]] = []
    scores: list[float] = []
    seen: set[tuple[int | float, ...]] = set()
    warmup = min(5, budget)

    for index in range(budget):
        if index < warmup:
            config = _unique_sample(space, rng, seen)
        else:
            config = _expected_improvement_candidate(space, points, scores, rng, seen)
        seen.add(_key(config))
        trial = evaluate(evaluator, config, epochs)
        trials.append(trial)
        points.append(space.encode(config))
        scores.append(float(trial["metrics"]["val_accuracy"]))

    return result(trials)


def _unique_sample(space: SearchSpace, rng: random.Random, seen: set[tuple[int | float, ...]]):
    for _ in range(100):
        config = space.sample(rng)
        if _key(config) not in seen:
            return config
    return space.sample(rng)


def _expected_improvement_candidate(
    space: SearchSpace,
    points: list[list[float]],
    scores: list[float],
    rng: random.Random,
    seen: set[tuple[int | float, ...]],
) -> dict[str, int | float]:
    pool = []
    for _ in range(256):
        config = space.sample(rng)
        if _key(config) not in seen:
            pool.append(config)
    if not pool:
        return space.sample(rng)

    x_train = np.asarray(points, dtype=float)
    y_train = np.asarray(scores, dtype=float)
    baseline = float(np.mean(y_train))
    centered_scores = y_train - baseline
    x_candidates = np.asarray([space.encode(config) for config in pool], dtype=float)
    kernel = _rbf(x_train, x_train) + np.eye(len(x_train)) * 1e-6
    cross = _rbf(x_candidates, x_train)
    try:
        alpha = np.linalg.solve(kernel, centered_scores)
        mean = baseline + cross @ alpha
        solved = np.linalg.solve(kernel, cross.T)
        variance = np.maximum(1e-12, 1.0 - np.sum(cross * solved.T, axis=1))
    except np.linalg.LinAlgError:
        return pool[0]

    sigma = np.sqrt(variance)
    improvement = mean - float(np.max(y_train))
    z = improvement / sigma
    cdf = np.asarray([0.5 * (1 + math.erf(value / math.sqrt(2))) for value in z])
    pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    expected_improvement = improvement * cdf + sigma * pdf
    return pool[int(np.argmax(expected_improvement))]


def _rbf(left: np.ndarray, right: np.ndarray, length_scale: float = 0.35) -> np.ndarray:
    squared_distances = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
    return np.exp(-0.5 * squared_distances / (length_scale**2))


def _key(config: dict[str, int | float]) -> tuple[int | float, ...]:
    return tuple(round(float(config[name]), 10) for name in SearchSpace.names)
