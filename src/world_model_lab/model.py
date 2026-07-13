"""Neural network used as the learned one-step world model."""

from __future__ import annotations

from torch import nn


class WorldModelMLP(nn.Module):
    """Predict four normalized state deltas from seven normalized inputs."""

    input_size = 7
    output_size = 4

    def __init__(self, *, hidden_size: int = 128) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.output_size),
        )

    def forward(self, inputs):
        return self.network(inputs)
