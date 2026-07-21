"""Frozen V-JEPA representation-probe contracts and metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .visual_dataset import CONTEXT_FRAMES, IMAGE_SIZE, validate_visual_dataset
from .visual_windows import VisualWindowIndex


_CLIP_ORDERS = frozenset({"recorded", "reversed", "repeat_last"})
DEFAULT_VJEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
VJEPA_POOLING = "last_tubelet_mean_plus_last_minus_first"


def _readonly_copy(
    values: np.ndarray,
    *,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ProbeClipBatch:
    """Owned, immutable four-frame clips and their recorded final states."""

    frames: np.ndarray
    states: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        states = np.asarray(self.states)
        episode_ids = np.asarray(self.episode_ids)
        step_ids = np.asarray(self.step_ids)
        sample_count = int(frames.shape[0]) if frames.ndim > 0 else 0
        if (
            frames.shape
            != (sample_count, CONTEXT_FRAMES, IMAGE_SIZE, IMAGE_SIZE, 3)
            or frames.dtype != np.dtype(np.uint8)
            or sample_count == 0
        ):
            raise ValueError(
                "frames must have dtype uint8 and shape [N, 4, 64, 64, 3]"
            )
        if (
            states.shape != (sample_count, 4)
            or states.dtype.kind not in "fiu"
            or not np.all(np.isfinite(states))
        ):
            raise ValueError("states must be finite numeric [N, 4]")
        for name, values in (
            ("episode_ids", episode_ids),
            ("step_ids", step_ids),
        ):
            if values.shape != (sample_count,) or values.dtype.kind not in "iu":
                raise ValueError(f"{name} must be an integer [N] array")
        object.__setattr__(self, "frames", _readonly_copy(frames))
        object.__setattr__(
            self,
            "states",
            _readonly_copy(states, dtype=np.float64),
        )
        object.__setattr__(
            self,
            "episode_ids",
            _readonly_copy(episode_ids, dtype=np.int64),
        )
        object.__setattr__(
            self,
            "step_ids",
            _readonly_copy(step_ids, dtype=np.int64),
        )


def _validated_probe_positions(
    positions: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    values = np.asarray(positions)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError("positions must be a non-empty one-dimensional integer array")
    integer_values = [int(value) for value in values.tolist()]
    if min(integer_values) < 0 or max(integer_values) >= count:
        raise ValueError("positions must refer to available visual windows")
    result = np.asarray(integer_values, dtype=np.int64)
    if np.unique(result).size != result.size:
        raise ValueError("positions must not contain duplicates")
    return result


def select_evenly_spaced_positions(count: int, *, limit: int) -> np.ndarray:
    """Return stable, unique positions spanning the complete index."""

    sample_count = int(count)
    maximum = int(limit)
    if sample_count <= 0:
        raise ValueError("count must be positive")
    if maximum <= 0:
        raise ValueError("limit must be positive")
    if maximum >= sample_count:
        return np.arange(sample_count, dtype=np.int64)
    positions = np.rint(
        np.linspace(0, sample_count - 1, num=maximum)
    ).astype(np.int64)
    if np.unique(positions).size != maximum:
        raise RuntimeError("evenly spaced selection produced duplicate positions")
    return positions


def _apply_clip_order(frames: np.ndarray, *, order: str) -> np.ndarray:
    if order not in _CLIP_ORDERS:
        listed = ", ".join(sorted(_CLIP_ORDERS))
        raise ValueError(f"order must be one of: {listed}")
    if order == "recorded":
        return frames.copy()
    if order == "reversed":
        return frames[::-1].copy()
    return np.repeat(frames[-1:], CONTEXT_FRAMES, axis=0)


def build_probe_clip_batch(
    dataset: Mapping[str, np.ndarray],
    index: VisualWindowIndex,
    positions: np.ndarray,
    *,
    order: str,
) -> ProbeClipBatch:
    """Materialize selected four-frame clips without crossing episodes."""

    validate_visual_dataset(dataset)
    selected = _validated_probe_positions(positions, count=index.count)
    frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
    episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    source_frames = np.asarray(dataset["frames"])
    source_states = np.asarray(dataset["states"])
    clips: list[np.ndarray] = []
    states: list[np.ndarray] = []
    selected_episode_ids: list[int] = []
    for position in selected.tolist():
        episode_index = int(index.episode_indices[position])
        step_id = int(index.step_ids[position])
        frame_start = (
            int(frame_offsets[episode_index])
            + step_id
            - (CONTEXT_FRAMES - 1)
        )
        recorded = source_frames[frame_start : frame_start + CONTEXT_FRAMES]
        if recorded.shape[0] != CONTEXT_FRAMES:
            raise ValueError("visual window does not contain four context frames")
        clips.append(_apply_clip_order(recorded, order=order))
        states.append(source_states[frame_start + CONTEXT_FRAMES - 1])
        selected_episode_ids.append(int(episode_ids[episode_index]))
    return ProbeClipBatch(
        frames=np.stack(clips),
        states=np.stack(states),
        episode_ids=np.asarray(selected_episode_ids, dtype=np.int64),
        step_ids=index.step_ids[selected],
    )


def state_to_probe_targets(states: np.ndarray) -> np.ndarray:
    """Encode physical states as x, y, sin-heading, cos-heading, velocity."""

    values = np.asarray(states)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or values.shape[0] == 0
        or values.dtype.kind not in "fiu"
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("states must be finite non-empty numeric [N, 4]")
    numeric = np.asarray(values, dtype=np.float64)
    targets = np.column_stack(
        (
            numeric[:, :2],
            np.sin(numeric[:, 2]),
            np.cos(numeric[:, 2]),
            numeric[:, 3],
        )
    )
    return targets.astype(np.float32)


def _validated_world_bounds(world_bounds: np.ndarray) -> np.ndarray:
    bounds = np.asarray(world_bounds, dtype=np.float64)
    if (
        bounds.shape != (4,)
        or not np.all(np.isfinite(bounds))
        or bounds[0] >= bounds[1]
        or bounds[2] >= bounds[3]
    ):
        raise ValueError(
            "world_bounds must contain finite min_x, max_x, min_y, max_y"
        )
    return bounds


def state_probe_metrics(
    predicted_targets: np.ndarray,
    true_targets: np.ndarray,
    *,
    world_bounds: np.ndarray,
) -> dict[str, float | int]:
    """Measure centre pixels, circular heading degrees, and velocity error."""

    predicted = np.asarray(predicted_targets, dtype=np.float64)
    target = np.asarray(true_targets, dtype=np.float64)
    bounds = _validated_world_bounds(world_bounds)
    if (
        predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != 5
        or predicted.shape[0] == 0
        or not np.all(np.isfinite(predicted))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("probe targets must be finite matching non-empty [N, 5]")
    target_heading_norms = np.linalg.norm(target[:, 2:4], axis=1)
    if np.any(target_heading_norms <= 0.0):
        raise ValueError("true heading vectors must be non-zero")

    world_errors = np.linalg.norm(predicted[:, :2] - target[:, :2], axis=1)
    scale = min(
        (IMAGE_SIZE - 1) / (bounds[1] - bounds[0]),
        (IMAGE_SIZE - 1) / (bounds[3] - bounds[2]),
    )
    centre_errors = world_errors * scale

    predicted_heading = predicted[:, 2:4]
    predicted_norms = np.linalg.norm(predicted_heading, axis=1, keepdims=True)
    target_heading = target[:, 2:4] / target_heading_norms[:, None]
    predicted_heading = predicted_heading / np.maximum(predicted_norms, 1e-12)
    heading_cosines = np.clip(
        np.sum(predicted_heading * target_heading, axis=1),
        -1.0,
        1.0,
    )
    heading_errors = np.degrees(np.arccos(heading_cosines))
    velocity_errors = np.abs(predicted[:, 4] - target[:, 4])
    return {
        "samples": int(target.shape[0]),
        "mean_centre_error_pixels": float(np.mean(centre_errors)),
        "p95_centre_error_pixels": float(np.quantile(centre_errors, 0.95)),
        "mean_heading_error_degrees": float(np.mean(heading_errors)),
        "p95_heading_error_degrees": float(np.quantile(heading_errors, 0.95)),
        "mean_velocity_error": float(np.mean(velocity_errors)),
        "p95_velocity_error": float(np.quantile(velocity_errors, 0.95)),
    }


def pool_vjepa_tokens(
    tokens: torch.Tensor,
    *,
    tubelet_count: int,
    spatial_tokens: int,
) -> torch.Tensor:
    """Pool encoder tokens into current-state and temporal-delta features."""

    tubelets = int(tubelet_count)
    patches = int(spatial_tokens)
    if tubelets <= 0 or patches <= 0:
        raise ValueError("tubelet_count and spatial_tokens must be positive")
    if (
        tokens.ndim != 3
        or tokens.shape[0] == 0
        or tokens.shape[1] != tubelets * patches
        or tokens.shape[2] == 0
        or not bool(torch.all(torch.isfinite(tokens)))
    ):
        raise ValueError(
            "tokens must be finite [B, tubelet_count * spatial_tokens, D]"
        )
    grids = tokens.reshape(tokens.shape[0], tubelets, patches, tokens.shape[2])
    tubelet_means = torch.mean(grids, dim=2)
    first = tubelet_means[:, 0]
    last = tubelet_means[:, -1]
    return torch.cat((last, last - first), dim=1)


class FrozenVJEPAEncoder:
    """Lazy, frozen adapter around the Hugging Face V-JEPA 2 encoder."""

    def __init__(
        self,
        *,
        processor: Any,
        model: torch.nn.Module,
        model_id: str,
        revision: str,
        device: str,
    ) -> None:
        if not str(model_id).strip():
            raise ValueError("model_id must be non-empty")
        if not str(revision).strip():
            raise ValueError("revision must be non-empty")
        if not hasattr(model, "config") or not hasattr(
            model,
            "get_vision_features",
        ):
            raise ValueError(
                "model must expose config and get_vision_features"
            )
        self.processor = processor
        self.model = model
        self.model_id = str(model_id)
        self.revision = str(revision)
        self.device = torch.device(device)
        config = model.config
        contract = {
            "tubelet_size": getattr(config, "tubelet_size", None),
            "crop_size": getattr(config, "crop_size", None),
            "patch_size": getattr(config, "patch_size", None),
            "hidden_size": getattr(config, "hidden_size", None),
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for value in contract.values()
        ):
            raise ValueError("model config has an invalid V-JEPA shape contract")
        if contract["crop_size"] % contract["patch_size"] != 0:
            raise ValueError("crop_size must be divisible by patch_size")
        self.tubelet_size = int(contract["tubelet_size"])
        self.crop_size = int(contract["crop_size"])
        self.patch_size = int(contract["patch_size"])
        self.hidden_size = int(contract["hidden_size"])
        resolved = getattr(config, "_commit_hash", None)
        self.resolved_revision = (
            str(resolved) if resolved is not None else self.revision
        )
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_VJEPA_MODEL_ID,
        *,
        revision: str = "main",
        device: str = "cpu",
    ) -> FrozenVJEPAEncoder:
        """Load the optional Transformers integration only when requested."""

        try:
            from transformers import AutoModel, AutoVideoProcessor
        except ImportError as error:
            raise RuntimeError(
                "V-JEPA support requires `pip install world-model-lab[vjepa]`"
            ) from error
        processor = AutoVideoProcessor.from_pretrained(
            model_id,
            revision=revision,
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            attn_implementation="sdpa",
        )
        return cls(
            processor=processor,
            model=model,
            model_id=model_id,
            revision=revision,
            device=device,
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        """Return the exact frozen-encoder contract used for extraction."""

        return {
            "model_id": self.model_id,
            "requested_revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "device": str(self.device),
            "tubelet_size": self.tubelet_size,
            "crop_size": self.crop_size,
            "patch_size": self.patch_size,
            "hidden_size": self.hidden_size,
            "feature_dim": self.hidden_size * 2,
            "pooling": VJEPA_POOLING,
        }

    def encode(self, clips: np.ndarray) -> np.ndarray:
        """Return owned float32 pooled features for uint8 four-frame clips."""

        values = np.asarray(clips)
        if (
            values.ndim != 5
            or values.shape[0] == 0
            or values.shape[1:] != (
                CONTEXT_FRAMES,
                IMAGE_SIZE,
                IMAGE_SIZE,
                3,
            )
            or values.dtype != np.dtype(np.uint8)
        ):
            raise ValueError(
                "clips must have dtype uint8 and shape [B, 4, 64, 64, 3]"
            )
        if CONTEXT_FRAMES % self.tubelet_size != 0:
            raise ValueError("context frames must be divisible by tubelet size")
        videos = [clip.copy() for clip in values]
        processed = self.processor(videos, return_tensors="pt")
        try:
            pixel_values = processed["pixel_values_videos"]
        except (KeyError, TypeError):
            raise ValueError(
                "processor must return pixel_values_videos"
            ) from None
        if (
            not isinstance(pixel_values, torch.Tensor)
            or pixel_values.ndim != 5
            or pixel_values.shape[0] != values.shape[0]
            or pixel_values.shape[1] != CONTEXT_FRAMES
            or pixel_values.shape[2] != 3
            or pixel_values.shape[3:] != (self.crop_size, self.crop_size)
            or not bool(torch.all(torch.isfinite(pixel_values)))
        ):
            raise ValueError(
                "processor output must be finite [B, 4, 3, crop, crop]"
            )
        pixel_values = pixel_values.to(self.device)
        with torch.inference_mode():
            tokens = self.model.get_vision_features(pixel_values)
        if not isinstance(tokens, torch.Tensor):
            raise ValueError("model must return a token tensor")
        tubelet_count = CONTEXT_FRAMES // self.tubelet_size
        grid_size = self.crop_size // self.patch_size
        features = pool_vjepa_tokens(
            tokens,
            tubelet_count=tubelet_count,
            spatial_tokens=grid_size * grid_size,
        )
        if features.shape != (values.shape[0], self.hidden_size * 2):
            raise ValueError("pooled V-JEPA feature shape is inconsistent")
        return np.asarray(
            features.detach().cpu(),
            dtype=np.float32,
        ).copy()
