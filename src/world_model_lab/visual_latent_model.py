"""Neural networks for the first visual latent world-model baseline."""

from __future__ import annotations

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """Compress and reconstruct fixed-size RGB observations."""

    image_size = 64
    image_channels = 3

    def __init__(self, *, latent_dim: int = 32, base_channels: int = 16) -> None:
        super().__init__()
        if latent_dim <= 0 or base_channels <= 0:
            raise ValueError("latent_dim and base_channels must be positive")
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        encoded_channels = 4 * self.base_channels
        self.encoder_convolutions = nn.Sequential(
            nn.Conv2d(3, self.base_channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(
                self.base_channels,
                2 * self.base_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                2 * self.base_channels,
                encoded_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                encoded_channels,
                encoded_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
        )
        self.encoder_projection = nn.Linear(
            encoded_channels * 4 * 4,
            self.latent_dim,
        )
        self.decoder_projection = nn.Linear(
            self.latent_dim,
            encoded_channels * 4 * 4,
        )
        self.decoder_convolutions = nn.Sequential(
            nn.ConvTranspose2d(
                encoded_channels,
                encoded_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                encoded_channels,
                2 * self.base_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2 * self.base_channels,
                self.base_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(self.base_channels, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("images must have shape [B, 3, 64, 64]")
        features = self.encoder_convolutions(images)
        return self.encoder_projection(features.flatten(start_dim=1))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 2 or latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"latents must have shape [B, {self.latent_dim}]"
            )
        encoded_channels = 4 * self.base_channels
        features = self.decoder_projection(latents).reshape(
            latents.shape[0],
            encoded_channels,
            4,
            4,
        )
        return self.decoder_convolutions(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))


class LatentDynamicsMLP(nn.Module):
    """Predict a residual next latent from recent latents and aligned actions."""

    action_size = 2

    def __init__(
        self,
        *,
        latent_dim: int = 32,
        hidden_size: int = 256,
        context_frames: int = 4,
    ) -> None:
        super().__init__()
        if latent_dim <= 0 or hidden_size <= 0:
            raise ValueError("latent_dim and hidden_size must be positive")
        if context_frames < 2:
            raise ValueError("context_frames must be at least two")
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.context_frames = int(context_frames)
        input_size = (
            self.context_frames * self.latent_dim
            + self.context_frames * self.action_size
        )
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.latent_dim),
        )

    def forward(
        self,
        context_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        if context_latents.ndim != 3:
            raise ValueError("context_latents have an invalid shape")
        batch_size = context_latents.shape[0]
        if tuple(context_latents.shape) != (
            batch_size,
            self.context_frames,
            self.latent_dim,
        ):
            raise ValueError("context_latents have an invalid shape")
        if tuple(history_actions.shape) != (
            batch_size,
            self.context_frames - 1,
            self.action_size,
        ):
            raise ValueError("history_actions have an invalid shape")
        if tuple(current_action.shape) != (batch_size, self.action_size):
            raise ValueError("current_action has an invalid shape")
        model_input = torch.cat(
            (
                context_latents.flatten(start_dim=1),
                history_actions.flatten(start_dim=1),
                current_action,
            ),
            dim=1,
        )
        return context_latents[:, -1] + self.network(model_input)
