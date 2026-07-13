"""Differentiable free-rollout operations used during world-model training."""

from __future__ import annotations

import torch
from torch import nn


def _model_inputs(
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    return torch.stack(
        (
            states[:, 0],
            states[:, 1],
            torch.sin(states[:, 2]),
            torch.cos(states[:, 2]),
            states[:, 3],
            actions[:, 0],
            actions[:, 1],
        ),
        dim=1,
    )


def _wrap_angle(values: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(values), torch.cos(values))


def rollout_states(
    model: nn.Module,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    *,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Recursively predict physical states while preserving autograd."""

    if initial_states.ndim != 2 or initial_states.shape[1] != 4:
        raise ValueError("initial_states must have shape [B, 4]")
    if (
        actions.ndim != 3
        or actions.shape[0] != initial_states.shape[0]
        or actions.shape[2] != 2
    ):
        raise ValueError("actions must have shape [B, H, 2]")
    if actions.shape[1] == 0:
        raise ValueError("actions must contain at least one rollout step")

    current_states = initial_states
    predicted_states = []
    for offset in range(actions.shape[1]):
        raw_inputs = _model_inputs(current_states, actions[:, offset])
        normalized_inputs = (raw_inputs - input_mean) / input_std
        normalized_deltas = model(normalized_inputs)
        deltas = normalized_deltas * target_std + target_mean
        next_states = current_states + deltas
        next_states = torch.cat(
            (
                next_states[:, :2],
                _wrap_angle(next_states[:, 2:3]),
                next_states[:, 3:4],
            ),
            dim=1,
        )
        predicted_states.append(next_states)
        current_states = next_states
    return torch.stack(predicted_states, dim=1)


def wrapped_state_errors(
    predicted_states: torch.Tensor,
    true_states: torch.Tensor,
) -> torch.Tensor:
    """Return physical state errors with heading wrapped around the circle."""

    if (
        predicted_states.shape != true_states.shape
        or predicted_states.ndim < 2
        or predicted_states.shape[-1] != 4
    ):
        raise ValueError(
            "predicted and true states must have matching shape [..., 4]"
        )
    difference = predicted_states - true_states
    return torch.cat(
        (
            difference[..., :2],
            _wrap_angle(difference[..., 2:3]),
            difference[..., 3:4],
        ),
        dim=-1,
    )


def rollout_state_loss(
    model: nn.Module,
    *,
    initial_states: torch.Tensor,
    actions: torch.Tensor,
    true_next_states: torch.Tensor,
    input_mean: torch.Tensor,
    input_std: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    """Return normalized state MSE across every recursive rollout step."""

    predictions = rollout_states(
        model,
        initial_states,
        actions,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
    )
    errors = wrapped_state_errors(predictions, true_next_states)
    return torch.mean(torch.square(errors / target_std))
