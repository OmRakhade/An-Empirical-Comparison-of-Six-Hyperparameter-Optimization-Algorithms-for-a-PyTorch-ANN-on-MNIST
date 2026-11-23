from __future__ import annotations

import unittest

from algorithms import bayesian, genetic_algorithm, grid_search, hyperband, pso, random_search
from algorithms.search_space import SearchSpace


class SearchSpaceTests(unittest.TestCase):
    def test_encode_decode_preserves_search_space(self):
        space = SearchSpace()
        config = {
            "learning_rate": 0.001,
            "batch_size": 64,
            "hidden_neurons": 128,
            "dropout": 0.25,
            "weight_decay": 0.0001,
        }
        decoded = space.decode(space.encode(config))
        self.assertAlmostEqual(decoded["learning_rate"], config["learning_rate"])
        self.assertEqual(decoded["batch_size"], config["batch_size"])
        self.assertEqual(decoded["hidden_neurons"], config["hidden_neurons"])
        self.assertAlmostEqual(decoded["dropout"], config["dropout"])
        self.assertAlmostEqual(decoded["weight_decay"], config["weight_decay"])


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.space = SearchSpace()
        self.evaluated_epochs = {}

    def objective(self, config, epochs, trial_id=None):
        if trial_id is not None:
            self.evaluated_epochs.setdefault(trial_id, []).append(epochs)
        accuracy = config["hidden_neurons"] / 300
        return {
            "val_accuracy": accuracy,
            "val_loss": 1 - accuracy,
            "duration_seconds": 0.01,
        }

    def test_each_optimizer_respects_evaluation_budget(self):
        for optimizer in (
            grid_search,
            random_search,
            bayesian,
            hyperband,
            genetic_algorithm,
            pso,
        ):
            with self.subTest(optimizer=optimizer.__name__):
                result = optimizer.run(self.space, self.objective, 6, 21, 4)
                self.assertGreaterEqual(len(result["trials"]), 1)
                self.assertLessEqual(len(result["trials"]), 6)
                self.assertIn("hidden_neurons", result["best_config"])

    def test_hyperband_promotes_trials_with_increasing_resources(self):
        hyperband.run(self.space, self.objective, 8, 21, 4)
        promoted = [resources for resources in self.evaluated_epochs.values() if len(resources) > 1]
        self.assertTrue(promoted)
        self.assertTrue(all(left < right for resources in promoted for left, right in zip(resources, resources[1:])))


if __name__ == "__main__":
    unittest.main()
