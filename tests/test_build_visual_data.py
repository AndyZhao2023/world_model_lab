import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageSequence

from tests.visual_fixtures import make_transition_source
from world_model_lab import build_visual_data
from world_model_lab.build_visual_data import run_visual_data_build
from world_model_lab.visual_dataset import load_visual_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_source(path: Path) -> None:
    np.savez_compressed(path, **make_transition_source())


class BuildVisualDataTest(unittest.TestCase):
    def test_pyproject_registers_visual_data_command(self):
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn(
            'world-model-build-visual-data = '
            '"world_model_lab.build_visual_data:main"',
            pyproject,
        )

    def test_cli_help_lists_all_visual_data_parameters(self):
        standard_output = io.StringIO()
        with patch.object(
            sys,
            "argv",
            ["world-model-build-visual-data", "--help"],
        ):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    build_visual_data.main()

        self.assertEqual(context.exception.code, 0)
        help_text = standard_output.getvalue()
        for flag in (
            "--data",
            "--output",
            "--preview",
            "--preview-episode-id",
        ):
            self.assertIn(flag, help_text)

    def test_cli_prints_sorted_indented_json_without_nan(self):
        standard_output = io.StringIO()
        returned = {"z": 1, "a": 2}
        with patch.object(
            build_visual_data,
            "run_visual_data_build",
            return_value=returned,
        ):
            with patch.object(sys, "argv", ["world-model-build-visual-data"]):
                with redirect_stdout(standard_output):
                    build_visual_data.main()

        self.assertEqual(
            standard_output.getvalue(),
            json.dumps(
                returned,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )

    def test_missing_input_becomes_argument_error(self):
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-build-visual-data",
                    "--data",
                    str(root / "missing.npz"),
                    "--output",
                    str(root / "visual.npz"),
                    "--preview",
                    str(root / "preview.gif"),
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        build_visual_data.main()

        self.assertEqual(context.exception.code, 2)
        self.assertIn("not a regular file", standard_error.getvalue())

    def test_resolved_paths_must_be_pairwise_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)

            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                run_visual_data_build(
                    data_path=source,
                    output_path=root / "nested" / ".." / "source.npz",
                    preview_path=root / "preview.gif",
                )

            same_artifact = root / "artifact.bin"
            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                run_visual_data_build(
                    data_path=source,
                    output_path=same_artifact,
                    preview_path=same_artifact,
                )

    def test_existing_output_or_preview_fails_before_any_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)

            existing_output = root / "visual.npz"
            existing_output.write_bytes(b"keep-output")
            preview = root / "preview.gif"
            with self.assertRaises(FileExistsError):
                run_visual_data_build(
                    data_path=source,
                    output_path=existing_output,
                    preview_path=preview,
                )
            self.assertEqual(existing_output.read_bytes(), b"keep-output")
            self.assertFalse(preview.exists())

            existing_output.unlink()
            preview.write_bytes(b"keep-preview")
            with self.assertRaises(FileExistsError):
                run_visual_data_build(
                    data_path=source,
                    output_path=existing_output,
                    preview_path=preview,
                )
            self.assertFalse(existing_output.exists())
            self.assertEqual(preview.read_bytes(), b"keep-preview")

    def test_output_and_preview_directory_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)
            output_directory = root / "visual.npz"
            output_directory.mkdir()
            with self.assertRaisesRegex(IsADirectoryError, "output path"):
                run_visual_data_build(
                    data_path=source,
                    output_path=output_directory,
                    preview_path=root / "preview.gif",
                )

            output_directory.rmdir()
            preview_directory = root / "preview.gif"
            preview_directory.mkdir()
            with self.assertRaisesRegex(IsADirectoryError, "preview path"):
                run_visual_data_build(
                    data_path=source,
                    output_path=root / "visual.npz",
                    preview_path=preview_directory,
                )

    def test_tiny_end_to_end_build_writes_valid_npz_and_full_gif(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            save_source(source)
            source_before = source.read_bytes()

            summary = run_visual_data_build(
                data_path=source,
                output_path=output,
                preview_path=preview,
            )
            loaded = load_visual_dataset(output)
            with Image.open(preview) as image:
                durations = [
                    int(frame.info["duration"])
                    for frame in ImageSequence.Iterator(image)
                ]
                frame_count = int(image.n_frames)

            self.assertEqual(summary["episodes"], 2)
            self.assertEqual(summary["transitions"], 6)
            self.assertEqual(summary["frames"], 8)
            self.assertEqual(summary["four_frame_eligible_episodes"], 1)
            self.assertEqual(summary["one_step_visual_samples"], 1)
            self.assertEqual(summary["preview_episode_id"], 7)
            self.assertEqual(summary["output_bytes"], output.stat().st_size)
            self.assertEqual(frame_count, 5)
            self.assertEqual(durations, [100, 100, 100, 100, 100])
            self.assertEqual(loaded["frames"].shape, (8, 64, 64, 3))
            self.assertEqual(source.read_bytes(), source_before)

    def test_explicit_preview_episode_uses_its_complete_frame_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            save_source(source)

            summary = run_visual_data_build(
                data_path=source,
                output_path=output,
                preview_path=preview,
                preview_episode_id=3,
            )
            with Image.open(preview) as image:
                frame_count = int(image.n_frames)

            self.assertEqual(summary["preview_episode_id"], 3)
            self.assertEqual(frame_count, 3)

    def test_same_source_and_versions_produce_identical_gif_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)

            first_preview = root / "first.gif"
            second_preview = root / "second.gif"
            run_visual_data_build(
                data_path=source,
                output_path=root / "first.npz",
                preview_path=first_preview,
            )
            run_visual_data_build(
                data_path=source,
                output_path=root / "second.npz",
                preview_path=second_preview,
            )

            self.assertEqual(
                first_preview.read_bytes(),
                second_preview.read_bytes(),
            )

    def test_unavailable_preview_episode_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            save_source(source)

            with self.assertRaisesRegex(
                ValueError,
                "preview episode is unavailable: 999",
            ):
                run_visual_data_build(
                    data_path=source,
                    output_path=output,
                    preview_path=preview,
                    preview_episode_id=999,
                )

            self.assertFalse(output.exists())
            self.assertFalse(preview.exists())

    def test_invalid_preview_episode_becomes_argument_error(self):
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-build-visual-data",
                    "--data",
                    str(source),
                    "--output",
                    str(root / "visual.npz"),
                    "--preview",
                    str(root / "preview.gif"),
                    "--preview-episode-id",
                    "999",
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        build_visual_data.main()

        self.assertEqual(context.exception.code, 2)
        self.assertIn(
            "preview episode is unavailable: 999",
            standard_error.getvalue(),
        )
