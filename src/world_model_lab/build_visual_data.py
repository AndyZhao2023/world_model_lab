"""Build deterministic episode-oriented RGB data from transition NPZ input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from .visual_dataset import (
    build_visual_dataset,
    load_transition_dataset,
    load_visual_dataset,
    save_visual_dataset,
    summarize_visual_dataset,
    validate_visual_dataset,
)


def _resolved_paths(
    *,
    data_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
) -> tuple[Path, Path, Path]:
    source = Path(data_path).expanduser().resolve(strict=False)
    output = Path(output_path).expanduser().resolve(strict=False)
    preview = Path(preview_path).expanduser().resolve(strict=False)
    if len({source, output, preview}) != 3:
        raise ValueError(
            "data, output, and preview paths must be pairwise distinct"
        )
    if not source.is_file():
        raise FileNotFoundError(
            f"transition dataset is not a regular file: {source}"
        )
    for label, path in (("output", output), ("preview", preview)):
        if path.is_dir():
            raise IsADirectoryError(f"{label} path is a directory: {path}")
        if path.exists():
            raise FileExistsError(f"{label} path already exists: {path}")
        if path.parent.exists() and not path.parent.is_dir():
            raise NotADirectoryError(
                f"{label} parent is not a directory: {path.parent}"
            )
    return source, output, preview


def _preview_episode_index(
    dataset: Mapping[str, np.ndarray],
    requested_episode_id: int | None,
) -> int:
    episode_ids = np.asarray(dataset["episode_ids"], dtype=np.int64)
    transition_lengths = np.diff(dataset["transition_offsets"])
    if requested_episode_id is None:
        return int(np.argmax(transition_lengths))
    matches = np.flatnonzero(episode_ids == int(requested_episode_id))
    if matches.size != 1:
        raise ValueError(
            f"preview episode is unavailable: {requested_episode_id}"
        )
    return int(matches[0])


def write_preview_gif(
    dataset: Mapping[str, np.ndarray],
    output_path: Path | str,
    *,
    episode_id: int | None = None,
) -> int:
    """Write all frames from one deterministic episode preview."""

    validate_visual_dataset(dataset)
    episode_index = _preview_episode_index(dataset, episode_id)
    selected_episode_id = int(dataset["episode_ids"][episode_index])
    frame_start = int(dataset["frame_offsets"][episode_index])
    frame_stop = int(dataset["frame_offsets"][episode_index + 1])
    frames = np.asarray(dataset["frames"][frame_start:frame_stop])
    duration_ms = max(
        1,
        int(np.rint(float(dataset["scene_dt"].item()) * 1000.0)),
    )

    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"preview path already exists: {path}")
    if path.parent.exists() and not path.parent.is_dir():
        raise NotADirectoryError(
            f"preview parent is not a directory: {path.parent}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [
        Image.fromarray(frame).convert(
            "P",
            palette=Image.Palette.ADAPTIVE,
            colors=256,
        )
        for frame in frames
    ]
    with path.open("xb") as handle:
        images[0].save(
            handle,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
    return selected_episode_id


def run_visual_data_build(
    *,
    data_path: Path | str,
    output_path: Path | str,
    preview_path: Path | str,
    preview_episode_id: int | None = None,
) -> dict[str, object]:
    """Validate source, build both artifacts, and return JSON-safe metadata."""

    source, output, preview = _resolved_paths(
        data_path=data_path,
        output_path=output_path,
        preview_path=preview_path,
    )
    transitions = load_transition_dataset(source)
    visual_dataset = build_visual_dataset(transitions)
    selected_index = _preview_episode_index(
        visual_dataset,
        preview_episode_id,
    )
    selected_episode_id = int(visual_dataset["episode_ids"][selected_index])

    save_visual_dataset(visual_dataset, output)
    write_preview_gif(
        visual_dataset,
        preview,
        episode_id=selected_episode_id,
    )
    persisted_dataset = load_visual_dataset(output)
    summary: dict[str, object] = summarize_visual_dataset(persisted_dataset)
    summary.update(
        {
            "source": str(source),
            "output": str(output),
            "preview": str(preview),
            "output_bytes": int(output.stat().st_size),
            "preview_episode_id": selected_episode_id,
        }
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/transitions.npz"),
        help="source transition NPZ",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/visual_episodes.npz"),
        help="new schema-v1 visual NPZ",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=Path("artifacts/visual_episode_preview.gif"),
        help="new episode preview GIF",
    )
    parser.add_argument(
        "--preview-episode-id",
        type=int,
        help="explicit episode ID; default is the longest episode",
    )
    args = parser.parse_args()

    try:
        summary = run_visual_data_build(
            data_path=args.data,
            output_path=args.output,
            preview_path=args.preview,
            preview_episode_id=args.preview_episode_id,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
