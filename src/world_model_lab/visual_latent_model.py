"""Neural networks for the first visual latent world-model baseline."""

from __future__ import annotations

import math

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


class SpatialConvAutoencoder(nn.Module):
    """Reconstruct RGB observations without collapsing the latent grid."""

    image_size = 64
    image_channels = 3
    latent_size = 8

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        base_channels: int = 16,
        object_residual_decoder: bool = False,
        object_head_channels: int | None = None,
        object_initial_alpha: float = 0.01,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or base_channels <= 0:
            raise ValueError(
                "latent_channels and base_channels must be positive"
            )
        if not isinstance(object_residual_decoder, bool):
            raise ValueError("object_residual_decoder must be boolean")
        if (
            not math.isfinite(object_initial_alpha)
            or object_initial_alpha <= 0.0
            or object_initial_alpha >= 1.0
        ):
            raise ValueError("object_initial_alpha must be between zero and one")
        if object_residual_decoder:
            selected_head_channels = (
                base_channels
                if object_head_channels is None
                else object_head_channels
            )
            if (
                isinstance(selected_head_channels, bool)
                or not isinstance(selected_head_channels, int)
                or selected_head_channels <= 0
            ):
                raise ValueError("object_head_channels must be positive")
        else:
            if object_head_channels is not None:
                raise ValueError(
                    "object_head_channels requires object residual decoding"
                )
            selected_head_channels = 0
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.object_residual_decoder = object_residual_decoder
        self.object_head_channels = int(selected_head_channels)
        self.latent_dim = (
            self.latent_channels * self.latent_size * self.latent_size
        )
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
                self.latent_channels,
                4,
                2,
                1,
            ),
            nn.ReLU(),
        )
        self.decoder_convolutions = nn.Sequential(
            nn.ConvTranspose2d(
                self.latent_channels,
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
        if self.object_residual_decoder:
            self.object_decoder_convolutions: nn.Sequential | None = (
                nn.Sequential(
                    nn.ConvTranspose2d(
                        self.latent_channels,
                        2 * self.object_head_channels,
                        4,
                        2,
                        1,
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        2 * self.object_head_channels,
                        self.object_head_channels,
                        4,
                        2,
                        1,
                    ),
                    nn.ReLU(),
                    nn.ConvTranspose2d(
                        self.object_head_channels,
                        4,
                        4,
                        2,
                        1,
                    ),
                )
            )
            final_convolution = self.object_decoder_convolutions[-1]
            assert isinstance(final_convolution, nn.ConvTranspose2d)
            with torch.no_grad():
                final_convolution.weight[:, 3].zero_()
                final_convolution.bias[3] = math.log(
                    object_initial_alpha / (1.0 - object_initial_alpha)
                )
        else:
            self.object_decoder_convolutions = None

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("images must have shape [B, 3, 64, 64]")
        return self.encoder_convolutions(images)

    def _latent_grid(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 2:
            if latents.shape[1] != self.latent_dim:
                raise ValueError(
                    "latents must have shape "
                    f"[B, {self.latent_channels}, 8, 8] or "
                    f"[B, {self.latent_dim}]"
                )
            latent_grid = latents.reshape(
                latents.shape[0],
                self.latent_channels,
                self.latent_size,
                self.latent_size,
            )
        elif latents.ndim == 4 and tuple(latents.shape[1:]) == (
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        ):
            latent_grid = latents
        else:
            raise ValueError(
                "latents must have shape "
                f"[B, {self.latent_channels}, 8, 8] or "
                f"[B, {self.latent_dim}]"
            )
        return latent_grid

    def decode_components(
        self,
        latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode frozen-base and learned object-composite components."""

        if (
            not self.object_residual_decoder
            or self.object_decoder_convolutions is None
        ):
            raise ValueError("object residual decoder is not enabled")
        latent_grid = self._latent_grid(latents)
        base = self.decoder_convolutions(latent_grid)
        object_output = self.object_decoder_convolutions(latent_grid)
        foreground = torch.sigmoid(object_output[:, :3])
        mask_logits = object_output[:, 3:4]
        alpha = torch.sigmoid(mask_logits)
        composite = base * (1.0 - alpha) + foreground * alpha
        return base, foreground, mask_logits, composite

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latent_grid = self._latent_grid(latents)
        if (
            not self.object_residual_decoder
            or self.object_decoder_convolutions is None
        ):
            return self.decoder_convolutions(latent_grid)
        return self.decode_components(latent_grid)[-1]

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


class SpatialLatentDynamicsCNN(nn.Module):
    """Predict a local residual on a reversibly flattened latent grid."""

    action_size = 2
    latent_size = 8

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        hidden_channels: int = 64,
        context_frames: int = 4,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or hidden_channels <= 0:
            raise ValueError(
                "latent_channels and hidden_channels must be positive"
            )
        if context_frames < 2:
            raise ValueError("context_frames must be at least two")
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.hidden_size = self.hidden_channels
        self.context_frames = int(context_frames)
        self.latent_dim = (
            self.latent_channels * self.latent_size * self.latent_size
        )
        input_channels = self.context_frames * (
            self.latent_channels + self.action_size
        )
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                3,
                1,
                1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                self.hidden_channels,
                self.latent_channels,
                3,
                1,
                1,
            ),
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

        context_grid = context_latents.reshape(
            batch_size,
            self.context_frames,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        aligned_actions = torch.cat(
            (history_actions, current_action[:, None, :]),
            dim=1,
        )
        action_maps = aligned_actions.flatten(start_dim=1)[:, :, None, None]
        action_maps = action_maps.expand(
            -1,
            -1,
            self.latent_size,
            self.latent_size,
        )
        model_input = torch.cat(
            (context_grid.flatten(start_dim=1, end_dim=2), action_maps),
            dim=1,
        )
        next_grid = context_grid[:, -1] + self.network(model_input)
        return next_grid.flatten(start_dim=1)


class SpatialLatentDynamicsConvGRU(nn.Module):
    """Predict a local residual while recurrently consuming latent grids."""

    action_size = 2
    latent_size = 8

    def __init__(
        self,
        *,
        latent_channels: int = 8,
        hidden_channels: int = 40,
        context_frames: int = 4,
    ) -> None:
        super().__init__()
        if latent_channels <= 0 or hidden_channels <= 0:
            raise ValueError(
                "latent_channels and hidden_channels must be positive"
            )
        if context_frames < 2:
            raise ValueError("context_frames must be at least two")
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.hidden_size = self.hidden_channels
        self.context_frames = int(context_frames)
        self.latent_dim = (
            self.latent_channels * self.latent_size * self.latent_size
        )
        recurrent_input_channels = (
            self.latent_channels
            + self.action_size
            + self.hidden_channels
        )
        self.gate_convolution = nn.Conv2d(
            recurrent_input_channels,
            2 * self.hidden_channels,
            3,
            1,
            1,
        )
        self.candidate_convolution = nn.Conv2d(
            recurrent_input_channels,
            self.hidden_channels,
            3,
            1,
            1,
        )
        self.output_convolution = nn.Conv2d(
            self.hidden_channels,
            self.latent_channels,
            3,
            1,
            1,
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

        context_grid = context_latents.reshape(
            batch_size,
            self.context_frames,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
        )
        aligned_actions = torch.cat(
            (history_actions, current_action[:, None, :]),
            dim=1,
        )
        hidden = context_grid.new_zeros(
            (
                batch_size,
                self.hidden_channels,
                self.latent_size,
                self.latent_size,
            )
        )
        for frame_index in range(self.context_frames):
            action_map = aligned_actions[:, frame_index, :, None, None].expand(
                -1,
                -1,
                self.latent_size,
                self.latent_size,
            )
            recurrent_input = torch.cat(
                (context_grid[:, frame_index], action_map),
                dim=1,
            )
            reset_gate, update_gate = torch.sigmoid(
                self.gate_convolution(
                    torch.cat((recurrent_input, hidden), dim=1)
                )
            ).chunk(2, dim=1)
            candidate = torch.tanh(
                self.candidate_convolution(
                    torch.cat(
                        (recurrent_input, reset_gate * hidden),
                        dim=1,
                    )
                )
            )
            hidden = update_gate * hidden + (1.0 - update_gate) * candidate

        next_grid = (
            context_grid[:, -1] + self.output_convolution(hidden)
        )
        return next_grid.flatten(start_dim=1)
