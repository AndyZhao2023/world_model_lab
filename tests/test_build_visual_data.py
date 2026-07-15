import gc
import io
import json
import sys
import tempfile
import unittest
import weakref
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageSequence

from tests.visual_fixtures import clone_arrays, make_transition_source
from world_model_lab import build_visual_data
from world_model_lab.build_visual_data import (
    run_visual_data_build,
    write_preview_gif,
)
from world_model_lab.visual_dataset import (
    build_visual_dataset,
    load_visual_dataset,
)


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

    def test_readme_documents_schema_v1_source_provenance_boundary(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("`CarEnv` 和 `collect_transitions`", readme)
        self.assertIn(
            "`transitions.npz` 不保存场景或 `dt` provenance",
            readme,
        )
        self.assertIn("`world_bounds`、障碍物/目标、各类半径或 `dt`", readme)
        self.assertIn("不得转换为 schema version 1", readme)

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

    def test_malformed_source_becomes_argument_error_without_traceback(self):
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "malformed.npz"
            output = root / "visual.npz"
            preview = root / "preview.gif"
            source.write_bytes(b"PK\x03\x04garbage")
            with patch.object(
                sys,
                "argv",
                [
                    "world-model-build-visual-data",
                    "--data",
                    str(source),
                    "--output",
                    str(output),
                    "--preview",
                    str(preview),
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        build_visual_data.main()

            self.assertFalse(output.exists())
            self.assertFalse(preview.exists())

        self.assertEqual(context.exception.code, 2)
        self.assertIn(
            "malformed transition dataset NPZ",
            standard_error.getvalue(),
        )
        self.assertNotIn("Traceback", standard_error.getvalue())

    def test_unrepresentable_episode_id_is_clean_argument_error(self):
        standard_error = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            arrays = clone_arrays(make_transition_source())
            arrays["episode_ids"] = arrays["episode_ids"].astype(np.uint64)
            arrays["episode_ids"][0] = np.uint64(2**63)
            np.savez_compressed(source, **arrays)
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
                ],
            ):
                with redirect_stderr(standard_error):
                    with self.assertRaises(SystemExit) as context:
                        build_visual_data.main()

        self.assertEqual(context.exception.code, 2)
        self.assertIn(
            "episode_ids values must fit in int64",
            standard_error.getvalue(),
        )
        self.assertNotIn("Traceback", standard_error.getvalue())

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

    def test_case_only_prospective_targets_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "Artifact.bin"
            preview = root / "artifact.BIN"
            save_source(source)

            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                run_visual_data_build(
                    data_path=source,
                    output_path=output,
                    preview_path=preview,
                )

            self.assertFalse(output.exists())
            self.assertFalse(preview.exists())

    def test_nfc_nfd_prospective_targets_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "artifact-\u00e9.bin"
            preview = root / "artifact-e\u0301.bin"
            save_source(source)

            with self.assertRaisesRegex(ValueError, "pairwise distinct"):
                run_visual_data_build(
                    data_path=source,
                    output_path=output,
                    preview_path=preview,
                )

            self.assertFalse(output.exists())
            self.assertFalse(preview.exists())

    def test_normalized_prospective_ancestor_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            output = root / "Caf\u00e9"
            preview = root / "cafe\u0301" / "preview.gif"
            save_source(source)

            with self.assertRaisesRegex(ValueError, "ancestor"):
                run_visual_data_build(
                    data_path=source,
                    output_path=output,
                    preview_path=preview,
                )

            self.assertFalse(output.exists())
            self.assertFalse(preview.exists())

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

    def test_output_and_preview_must_not_be_ancestors_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            save_source(source)
            cases = (
                (
                    root / "output-parent",
                    root / "output-parent" / "preview.gif",
                ),
                (
                    root / "preview-parent" / "visual.npz",
                    root / "preview-parent",
                ),
            )

            for output, preview in cases:
                with self.subTest(output=output, preview=preview):
                    with self.assertRaises(Exception) as context:
                        run_visual_data_build(
                            data_path=source,
                            output_path=output,
                            preview_path=preview,
                        )

                    self.assertFalse(output.exists())
                    self.assertFalse(preview.exists())
                    self.assertIsInstance(context.exception, ValueError)
                    self.assertRegex(str(context.exception), "ancestor")

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

    def test_gif_encoder_failure_leaves_no_final_or_temporary_file(self):
        dataset = build_visual_dataset(make_transition_source())

        def fail_after_partial_write(image, handle, *args, **kwargs):
            del image, args, kwargs
            handle.write(b"partial-gif")
            handle.flush()
            raise OSError("injected GIF encoder failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview = root / "preview.gif"
            with patch.object(
                Image.Image,
                "save",
                new=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "injected GIF"):
                    write_preview_gif(dataset, preview, episode_id=7)

            self.assertFalse(preview.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_atomic_gif_preserves_normal_file_creation_mode(self):
        dataset = build_visual_dataset(make_transition_source())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.bin"
            reference.open("xb").close()
            preview = root / "preview.gif"

            write_preview_gif(dataset, preview, episode_id=7)

            self.assertEqual(
                preview.stat().st_mode & 0o777,
                reference.stat().st_mode & 0o777,
            )

    def test_gif_publish_race_preserves_existing_final_file(self):
        dataset = build_visual_dataset(make_transition_source())
        original_save = Image.Image.save

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview = root / "preview.gif"

            def create_racing_final(image, handle, *args, **kwargs):
                preview.write_bytes(b"racing-writer")
                return original_save(image, handle, *args, **kwargs)

            with patch.object(
                Image.Image,
                "save",
                new=create_racing_final,
            ):
                with self.assertRaises(FileExistsError):
                    write_preview_gif(dataset, preview, episode_id=7)

            self.assertEqual(preview.read_bytes(), b"racing-writer")
            self.assertEqual(set(root.iterdir()), {preview})

    def test_gif_preserves_identical_logical_frames_without_pixel_changes(self):
        dataset = build_visual_dataset(make_transition_source())
        episode_index = int(np.flatnonzero(dataset["episode_ids"] == 7)[0])
        frame_start = int(dataset["frame_offsets"][episode_index])
        frame_stop = int(dataset["frame_offsets"][episode_index + 1])
        stationary_frame = dataset["frames"][frame_start].copy()
        dataset["frames"][frame_start:frame_stop] = stationary_frame

        with tempfile.TemporaryDirectory() as directory:
            preview = Path(directory) / "stationary.gif"
            write_preview_gif(dataset, preview, episode_id=7)
            with Image.open(preview) as image:
                decoded_frames = [
                    np.asarray(frame.convert("RGB")).copy()
                    for frame in ImageSequence.Iterator(image)
                ]
                durations = [
                    int(frame.info["duration"])
                    for frame in ImageSequence.Iterator(image)
                ]

        self.assertEqual(len(decoded_frames), frame_stop - frame_start)
        self.assertEqual(durations, [100] * (frame_stop - frame_start))
        for decoded_frame in decoded_frames:
            np.testing.assert_array_equal(decoded_frame, stationary_frame)

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

    def test_releases_source_and_built_arrays_before_persisted_reload(self):
        references: dict[str, weakref.ReferenceType[np.ndarray]] = {}

        def fake_load_transitions(path):
            del path
            sentinel = np.empty((1,), dtype=np.uint8)
            references["transitions"] = weakref.ref(sentinel)
            return {"sentinel": sentinel}

        def fake_build_visual(transitions):
            self.assertIsNotNone(references["transitions"]())
            self.assertIn("sentinel", transitions)
            sentinel = np.empty((1,), dtype=np.uint8)
            references["visual"] = weakref.ref(sentinel)
            return {
                "episode_ids": np.asarray([7], dtype=np.int64),
                "transition_offsets": np.asarray([0, 1], dtype=np.int64),
                "sentinel": sentinel,
            }

        def fake_save_visual(dataset, path):
            self.assertIsNotNone(references["visual"]())
            self.assertIn("sentinel", dataset)
            Path(path).write_bytes(b"npz")

        def fake_write_preview(dataset, path, *, episode_id):
            self.assertIsNotNone(references["visual"]())
            self.assertIn("sentinel", dataset)
            Path(path).write_bytes(b"gif")
            return episode_id

        def fake_load_persisted(path):
            del path
            gc.collect()
            self.assertIsNone(references["transitions"]())
            self.assertIsNone(references["visual"]())
            return {}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            source.write_bytes(b"source")
            with (
                patch.object(
                    build_visual_data,
                    "load_transition_dataset",
                    new=fake_load_transitions,
                ),
                patch.object(
                    build_visual_data,
                    "build_visual_dataset",
                    new=fake_build_visual,
                ),
                patch.object(
                    build_visual_data,
                    "save_visual_dataset",
                    new=fake_save_visual,
                ),
                patch.object(
                    build_visual_data,
                    "write_preview_gif",
                    new=fake_write_preview,
                ),
                patch.object(
                    build_visual_data,
                    "load_visual_dataset",
                    new=fake_load_persisted,
                ),
                patch.object(
                    build_visual_data,
                    "summarize_visual_dataset",
                    new=lambda dataset: {"released": dataset == {}},
                ),
            ):
                summary = run_visual_data_build(
                    data_path=source,
                    output_path=root / "visual.npz",
                    preview_path=root / "preview.gif",
                )

        self.assertTrue(summary["released"])
