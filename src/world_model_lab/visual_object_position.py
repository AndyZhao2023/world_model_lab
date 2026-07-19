"""Frozen linear object-position supervision for visual latent dynamics."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
from torch import nn


def _validated_world_bounds(world_bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(world_bounds, dtype=np.float64)
    if (
        bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
        or bounds[2] >= bounds[3]
    ):
        raise ValueError(
            "world_bounds must contain finite "
            "(min_x, max_x, min_y, max_y)"
        )
    return bounds


def normalize_object_positions(
    states: np.ndarray,
    world_bounds: np.ndarray,
) -> np.ndarray:
    """Map finite physical-state car centres to normalized XY coordinates."""

    values = np.asarray(states)
    bounds = _validated_world_bounds(world_bounds)
    if (
        values.ndim < 2
        or values.shape[-1] != 4
        or values.size == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("states must be finite non-empty [..., 4]")
    positions = np.asarray(values[..., :2], dtype=np.float64)
    minima = np.asarray([bounds[0], bounds[2]], dtype=np.float64)
    extents = np.asarray(
        [bounds[1] - bounds[0], bounds[3] - bounds[2]],
        dtype=np.float64,
    )
    normalized = 2.0 * (positions - minima) / extents - 1.0
    if not np.all(np.isfinite(normalized)):
        raise ValueError("normalized object positions must be finite")
    return normalized.astype(np.float32)


def world_position_errors(
    predicted_normalized: np.ndarray,
    true_normalized: np.ndarray,
    world_bounds: np.ndarray,
) -> np.ndarray:
    """Return Euclidean car-centre errors after restoring world-axis scales."""

    predicted = np.asarray(predicted_normalized, dtype=np.float64)
    target = np.asarray(true_normalized, dtype=np.float64)
    bounds = _validated_world_bounds(world_bounds)
    if (
        predicted.shape != target.shape
        or predicted.ndim < 2
        or predicted.shape[-1] != 2
        or predicted.size == 0
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError(
            "predicted and true positions must be finite matching [..., 2]"
        )
    half_extents = np.asarray(
        [
            (bounds[1] - bounds[0]) / 2.0,
            (bounds[3] - bounds[2]) / 2.0,
        ],
        dtype=np.float64,
    )
    return np.linalg.norm((predicted - target) * half_extents, axis=-1)


class LinearObjectPositionProbe(nn.Module):
    """Frozen affine map from normalized visual latents to normalized XY."""

    def __init__(
        self,
        *,
        weight: np.ndarray,
        bias: np.ndarray,
    ) -> None:
        super().__init__()
        weight_values = np.asarray(weight, dtype=np.float32)
        bias_values = np.asarray(bias, dtype=np.float32)
        if (
            weight_values.ndim != 2
            or weight_values.shape[0] != 2
            or weight_values.shape[1] == 0
            or bias_values.shape != (2,)
            or not np.all(np.isfinite(weight_values))
            or not np.all(np.isfinite(bias_values))
        ):
            raise ValueError(
                "probe weight and bias must be finite [2, D] and [2]"
            )
        self.latent_dim = int(weight_values.shape[1])
        self.linear = nn.Linear(self.latent_dim, 2)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(weight_values))
            self.linear.bias.copy_(torch.from_numpy(bias_values))
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def forward(self, normalized_latents: torch.Tensor) -> torch.Tensor:
        values = normalized_latents
        if (
            values.ndim < 2
            or values.shape[-1] != self.latent_dim
            or values.numel() == 0
            or not torch.is_floating_point(values)
            or not bool(torch.all(torch.isfinite(values)))
        ):
            raise ValueError(
                "normalized_latents must be finite floating [..., latent_dim]"
            )
        return self.linear(values)


def fit_linear_object_position_probe(
    normalized_latents: np.ndarray,
    normalized_positions: np.ndarray,
    *,
    ridge: float,
) -> LinearObjectPositionProbe:
    """Fit one deterministic affine ridge probe in float64."""

    latents = np.asarray(normalized_latents)
    positions = np.asarray(normalized_positions)
    if (
        latents.ndim != 2
        or latents.shape[0] == 0
        or latents.shape[1] == 0
        or positions.shape != (latents.shape[0], 2)
        or not np.all(np.isfinite(latents))
        or not np.all(np.isfinite(positions))
    ):
        raise ValueError(
            "probe inputs must be finite [N, D] latents and [N, 2] positions"
        )
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    latent_values = np.asarray(latents, dtype=np.float64)
    position_values = np.asarray(positions, dtype=np.float64)
    design = np.concatenate(
        (
            latent_values,
            np.ones((latent_values.shape[0], 1), dtype=np.float64),
        ),
        axis=1,
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    regularizer[-1, -1] = 0.0
    try:
        solution = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ position_values,
        )
    except np.linalg.LinAlgError:
        raise ValueError("probe ridge system is singular") from None
    if not np.all(np.isfinite(solution)):
        raise ValueError("probe fit produced non-finite coefficients")
    return LinearObjectPositionProbe(
        weight=solution[:-1].T,
        bias=solution[-1],
    )


def object_position_probe_sha256(
    probe: LinearObjectPositionProbe,
) -> str:
    """Hash the exact fitted probe tensors and their names."""

    if not isinstance(probe, LinearObjectPositionProbe):
        raise TypeError("probe must be a LinearObjectPositionProbe")
    digest = hashlib.sha256()
    for name, tensor in sorted(probe.state_dict().items()):
        values = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def predict_normalized_object_positions(
    probe: LinearObjectPositionProbe,
    normalized_latents: np.ndarray,
) -> np.ndarray:
    """Apply a frozen probe to finite NumPy latent arrays."""

    values = np.asarray(normalized_latents)
    if (
        values.ndim < 2
        or values.shape[-1] != probe.latent_dim
        or values.size == 0
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "normalized_latents must be finite [..., probe.latent_dim]"
        )
    with torch.no_grad():
        predicted = probe(
            torch.as_tensor(values, dtype=torch.float32)
        )
    return predicted.cpu().numpy().astype(np.float32, copy=False)


def evaluate_object_position_probe(
    probe: LinearObjectPositionProbe,
    *,
    normalized_latents: np.ndarray,
    normalized_positions: np.ndarray,
    world_bounds: np.ndarray,
) -> dict[str, float | int]:
    """Measure one frozen probe on aligned finite latent-position rows."""

    target = np.asarray(normalized_positions)
    predicted = predict_normalized_object_positions(
        probe,
        normalized_latents,
    )
    if target.shape != predicted.shape or target.ndim != 2:
        raise ValueError(
            "normalized_positions must match probe predictions [N, 2]"
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("normalized_positions must be finite")
    errors = predicted.astype(np.float64) - target
    world_errors = world_position_errors(
        predicted,
        target,
        world_bounds,
    )
    return {
        "frames": int(target.shape[0]),
        "normalized_position_mse": float(np.mean(np.square(errors))),
        "mean_world_position_error": float(np.mean(world_errors)),
        "p95_world_position_error": float(
            np.quantile(world_errors, 0.95)
        ),
    }
