"""Configurable two-hidden-layer ANN for MNIST classification."""

from __future__ import annotations

import torch.nn as nn


class MNISTANN(nn.Module):
    def __init__(self, hidden_neurons: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        if not 32 <= hidden_neurons <= 256:
            raise ValueError("hidden_neurons must be between 32 and 256")
        if not 0.0 <= dropout <= 0.5:
            raise ValueError("dropout must be between 0 and 0.5")
        second_layer_neurons = max(16, hidden_neurons // 2)
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_neurons, second_layer_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_layer_neurons, 10),
        )

    def forward(self, inputs):
        return self.network(inputs)
