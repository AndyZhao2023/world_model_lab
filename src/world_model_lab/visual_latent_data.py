"""PyTorch adapters and compact arrays for visual latent-model training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import operator

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import Normalizer
from .visual_dataset import (
    CONTEXT_FRAMES,
    IMAGE_SIZE,
    validate_visual_dataset,
)
from .visual_latent_model import ConvAutoencoder
from .visual_windows import VisualWindowIndex


def _selected_episode_positions(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(selected_episode_ids)
    if raw.ndim != 1:
        raise ValueError("selected_episode_ids must be one-dimensional")
    if raw.size == 0:
        raise ValueError("selected_episode_ids must be non-empty")
    if raw.dtype.kind not in "iu":
        raise ValueError("selected_episode_ids must be an integer array")
    limits = np.iinfo(np.int64)
    integer_values = [int(value) for value in raw.tolist()]
    if min(integer_values) < limits.min or max(integer_values) > limits.max:
        raise ValueError("selected_episode_ids values must fit in int64")
    selected = np.asarray(integer_values, dtype=np.int64)
    if np.unique(selected).size != selected.size:
        raise ValueError("selected_episode_ids must not contain duplicates")

    available = np.asarray(dataset["episode_ids"], dtype=np.int64)
    positions = {int(value): index for index, value in enumerate(available)}
    missing = [value for value in selected.tolist() if int(value) not in positions]
    if missing:
        raise ValueError(
            f"episode {int(missing[0])} is missing from the visual dataset"
        )
    selected_positions = np.asarray(
        [positions[int(value)] for value in selected.tolist()],
        dtype=np.int64,
    )
    return selected, selected_positions


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert uint8 HWC/NHWC images to owned float32 CHW/NCHW tensors."""

    values = np.asarray(frames)
    single = values.ndim == 3
    if single:
        values = values[None, ...]
    if (
        values.ndim != 4
        or values.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3)
        or values.dtype != np.dtype(np.uint8)
    ):
        raise ValueError(
            "frames must have dtype uint8 and shape "
            "[64, 64, 3] or [N, 64, 64, 3]"
        )
    contiguous = np.ascontiguousarray(values)
    tensor = (
        torch.from_numpy(contiguous.copy())
        .permute(0, 3, 1, 2)
        .to(dtype=torch.float32)
        .div_(255.0)
    )
    return tensor[0] if single else tensor


def frame_indices_for_episode_ids(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> np.ndarray:
    """Return canonical global frame indexes in selected episode order."""

    validate_visual_dataset(dataset)
    _, positions = _selected_episode_positions(dataset, selected_episode_ids)
    offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
    indexes = [
        np.arange(offsets[position], offsets[position + 1], dtype=np.int64)
        for position in positions.tolist()
    ]
    return np.concatenate(indexes)


def transition_indices_for_episode_ids(
    dataset: Mapping[str, np.ndarray],
    selected_episode_ids: np.ndarray,
) -> np.ndarray:
    """Return canonical transition indexes in selected episode order."""

    validate_visual_dataset(dataset)
    _, positions = _selected_episode_positions(dataset, selected_episode_ids)
    offsets = np.asarray(dataset["transition_offsets"], dtype=np.int64)
    indexes = [
        np.arange(offsets[position], offsets[position + 1], dtype=np.int64)
        for position in positions.tolist()
    ]
    return np.concatenate(indexes)


class VisualFrameDataset(Dataset):
    """Map-style access to unique normalized frames from selected episodes."""

    def __init__(
        self,
        dataset: Mapping[str, np.ndarray],
        selected_episode_ids: np.ndarray,
    ) -> None:
        self.frame_indices = frame_indices_for_episode_ids(
            dataset,
            selected_episode_ids,
        )
        self.frame_indices.setflags(write=False)
        self._frames = np.asarray(dataset["frames"])

    def __len__(self) -> int:
        return int(self.frame_indices.size)

    def __getitem__(self, item: int) -> torch.Tensor:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("frame dataset index must be an integer")
        try:
            position = operator.index(item)
        except TypeError:
            raise TypeError("frame dataset index must be an integer") from None
        if position < 0:
            position += len(self)
        if position < 0 or position >= len(self):
            raise IndexError("frame dataset index out of range")
        return frames_to_tensor(self._frames[int(self.frame_indices[position])])


def fit_safe_normalizer(
    values: np.ndarray,
    *,
    minimum_std: float = 1e-6,
) -> Normalizer:
    """Fit finite per-feature statistics while tolerating constants."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(
            "normalizer values must be a non-empty two-dimensional array"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("normalizer values must contain only finite values")
    if not np.isfinite(minimum_std) or minimum_std <= 0.0:
        raise ValueError("minimum_std must be finite and positive")
    mean = array.mean(axis=0)
    raw_std = array.std(axis=0)
    std = np.where(raw_std < minimum_std, 1.0, raw_std)
    return Normalizer(mean=mean, std=std)


def encode_all_frames(
    model: ConvAutoencoder,
    frames: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    """Encode every canonical frame once in deterministic global order."""

    values = np.asarray(frames)
    if (
        values.ndim != 4
        or values.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3)
        or values.dtype != np.dtype(np.uint8)
        or values.shape[0] == 0
    ):
        raise ValueError("frames must be non-empty uint8 [F, 64, 64, 3]")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            images = frames_to_tensor(values[start : start + batch_size])
            latents = model.encode(images)
            if not torch.all(torch.isfinite(latents)):
                raise ValueError("encoder produced non-finite latents")
            batches.append(latents.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)


@dataclass(frozen=True)
class LatentWindowArrays:
    """Compact aligned arrays for one-step latent dynamics training."""

    context_latents: np.ndarray
    history_actions: np.ndarray
    current_actions: np.ndarray
    target_latents: np.ndarray
    last_frame_indices: np.ndarray
    target_frame_indices: np.ndarray
    episode_ids: np.ndarray
    step_ids: np.ndarray

    def __post_init__(self) -> None:
        context = np.asarray(self.context_latents)
        if context.ndim != 3 or context.shape[1] != CONTEXT_FRAMES:
            raise ValueError(
                "context_latents must have shape [N, 4, latent_dim]"
            )
        count = context.shape[0]
        latent_dim = context.shape[2]
        expected_shapes = {
            "history_actions": (count, CONTEXT_FRAMES - 1, 2),
            "current_actions": (count, 2),
            "target_latents": (count, latent_dim),
            "last_frame_indices": (count,),
            "target_frame_indices": (count,),
            "episode_ids": (count,),
            "step_ids": (count,),
        }
        for name, shape in expected_shapes.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {list(shape)}")
        for name in (
            "context_latents",
            "history_actions",
            "current_actions",
            "target_latents",
        ):
            values = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
        for name in (
            "last_frame_indices",
            "target_frame_indices",
            "episode_ids",
            "step_ids",
        ):
            values = np.asarray(getattr(self, name))
            if values.dtype != np.dtype(np.int64):
                raise ValueError(f"{name} must have dtype int64")
        for name in self.__dataclass_fields__:
            values = np.asarray(getattr(self, name)).copy()
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def count(self) -> int:
        return int(self.context_latents.shape[0])


def build_latent_window_arrays(
    dataset: Mapping[str, np.ndarray],
    index: VisualWindowIndex,
    latent_frames: np.ndarray,
) -> LatentWindowArrays:
    """Resolve one validated visual index into compact latent/action arrays."""

    validate_visual_dataset(dataset)
    latents = np.asarray(latent_frames)
    frame_count = int(np.asarray(dataset["frames"]).shape[0])
    if (
        latents.ndim != 2
        or latents.shape[0] != frame_count
        or latents.shape[1] == 0
    ):
        raise ValueError(
            "latent_frames must have shape [F, latent_dim] matching frames"
        )
    if not np.all(np.isfinite(latents)):
        raise ValueError("latent_frames must contain only finite values")

    frame_offsets = np.asarray(dataset["frame_offsets"], dtype=np.int64)
    transition_offsets = np.asarray(
        dataset["transition_offsets"],
        dtype=np.int64,
    )
    history = CONTEXT_FRAMES - 1
    episode_indices = np.asarray(index.episode_indices, dtype=np.int64)
    step_ids = np.asarray(index.step_ids, dtype=np.int64)
    frame_starts = (
        frame_offsets[episode_indices] + step_ids - history
    )
    action_starts = (
        transition_offsets[episode_indices] + step_ids - history
    )
    context_indices = (
        frame_starts[:, None]
        + np.arange(CONTEXT_FRAMES, dtype=np.int64)[None, :]
    )
    history_action_indices = (
        action_starts[:, None]
        + np.arange(history, dtype=np.int64)[None, :]
    )
    current_action_indices = action_starts + history
    target_frame_indices = frame_starts + CONTEXT_FRAMES
    last_frame_indices = target_frame_indices - 1
    actions = np.asarray(dataset["actions"], dtype=np.float64)
    episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)[
        episode_indices
    ]
    return LatentWindowArrays(
        context_latents=np.asarray(latents[context_indices], dtype=np.float32),
        history_actions=np.asarray(
            actions[history_action_indices],
            dtype=np.float64,
        ),
        current_actions=np.asarray(
            actions[current_action_indices],
            dtype=np.float64,
        ),
        target_latents=np.asarray(
            latents[target_frame_indices],
            dtype=np.float32,
        ),
        last_frame_indices=np.asarray(last_frame_indices, dtype=np.int64),
        target_frame_indices=np.asarray(target_frame_indices, dtype=np.int64),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        step_ids=np.asarray(step_ids, dtype=np.int64),
    )
