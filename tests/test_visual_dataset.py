import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tests.visual_fixtures import (
    append_duplicate_npz_array,
    clone_arrays,
    corrupt_compressed_npz_member,
    make_transition_source,
)
from world_model_lab.car_env import CarEnv
from world_model_lab.visual_dataset import (
    REQUIRED_VISUAL_ARRAYS,
    build_visual_dataset,
    load_transition_dataset,
    load_visual_dataset,
    reconstruct_episodes,
    save_visual_dataset,
    summarize_visual_dataset,
    validate_visual_dataset,
)
from world_model_lab.visual_observation import (
    PILLOW_VERSION,
    RENDERER_VERSION,
    render_observation,
    scene_from_env,
)


class VisualSourceValidationTest(unittest.TestCase):
    def setUp(self):
        self.source = make_transition_source()

    def test_reconstructs_episodes_in_canonical_id_and_step_order(self):
        episodes = reconstruct_episodes(self.source)

        self.assertEqual([episode.episode_id for episode in episodes], [3, 7])
        np.testing.assert_array_equal(episodes[0].step_ids, [0, 1])
        np.testing.assert_array_equal(episodes[1].step_ids, [0, 1, 2, 3])
        self.assertEqual(episodes[0].states.shape, (2, 4))
        self.assertEqual(episodes[1].actions.shape, (4, 2))

    def test_shuffled_rows_reconstruct_identically(self):
        permutation = np.asarray([5, 1, 4, 0, 3, 2])
        shuffled = {
            name: values[permutation]
            for name, values in self.source.items()
        }

        expected = reconstruct_episodes(self.source)
        actual = reconstruct_episodes(shuffled)

        for expected_episode, actual_episode in zip(
            expected,
            actual,
            strict=True,
        ):
            self.assertEqual(actual_episode.episode_id, expected_episode.episode_id)
            for field in (
                "states",
                "actions",
                "next_states",
                "rewards",
                "dones",
                "step_ids",
                "terminal_reasons",
            ):
                np.testing.assert_array_equal(
                    getattr(actual_episode, field),
                    getattr(expected_episode, field),
                )

    def test_loader_uses_allow_pickle_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "object-source.npz"
            unsafe = clone_arrays(self.source)
            unsafe["states"] = np.asarray([object()], dtype=object)
            np.savez_compressed(path, **unsafe)

            with self.assertRaisesRegex(ValueError, "Object arrays"):
                load_transition_dataset(path)

    def test_loader_rejects_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.npz"
            with self.assertRaisesRegex(FileNotFoundError, "regular file"):
                load_transition_dataset(path)

    def test_numeric_npy_is_rejected_as_non_npz_transition_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric.npy"
            np.save(path, np.arange(4, dtype=np.int64))

            with self.assertRaisesRegex(
                ValueError,
                "transition dataset must be an NPZ archive",
            ):
                load_transition_dataset(path)

    def test_duplicate_transition_member_is_rejected_before_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-source.npz"
            np.savez_compressed(path, **self.source)
            append_duplicate_npz_array(
                path,
                "actions",
                np.full_like(self.source["actions"], 99.0),
            )

            with self.assertRaisesRegex(
                ValueError,
                "transition dataset has duplicate arrays: actions",
            ):
                load_transition_dataset(path)

    def test_corrupted_transition_member_is_reported_as_malformed_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupted-source.npz"
            np.savez_compressed(path, **self.source)
            corrupt_compressed_npz_member(path, "states")

            with self.assertRaisesRegex(
                ValueError,
                "malformed transition dataset NPZ",
            ):
                load_transition_dataset(path)

    def test_loader_normalizes_valid_integer_ids_to_int64(self):
        source = clone_arrays(self.source)
        source["episode_ids"] = source["episode_ids"].astype(np.uint16)
        source["step_ids"] = source["step_ids"].astype(np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez_compressed(path, **source)

            loaded = load_transition_dataset(path)

        self.assertEqual(loaded["episode_ids"].dtype, np.dtype(np.int64))
        self.assertEqual(loaded["step_ids"].dtype, np.dtype(np.int64))

    def test_malformed_transition_npz_is_reported_as_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed-source.npz"
            path.write_bytes(b"PK\x03\x04garbage")

            with self.assertRaisesRegex(
                ValueError,
                "malformed transition dataset NPZ",
            ):
                load_transition_dataset(path)

    def test_missing_required_array_names_the_array(self):
        source = clone_arrays(self.source)
        source.pop("actions")

        with self.assertRaisesRegex(ValueError, "missing arrays: actions"):
            reconstruct_episodes(source)

    def test_non_finite_numeric_value_names_the_array(self):
        source = clone_arrays(self.source)
        source["states"][0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "states.*finite"):
            reconstruct_episodes(source)

    def test_invalid_action_shape_names_the_array(self):
        source = clone_arrays(self.source)
        source["actions"] = source["actions"][:, :1]

        with self.assertRaisesRegex(ValueError, r"actions.*\[6, 2\]"):
            reconstruct_episodes(source)

    def test_float_step_ids_are_rejected(self):
        source = clone_arrays(self.source)
        source["step_ids"] = source["step_ids"].astype(np.float64)

        with self.assertRaisesRegex(ValueError, "step_ids.*integer"):
            reconstruct_episodes(source)

    def test_episode_ids_must_fit_int64_before_canonicalization(self):
        source = clone_arrays(self.source)
        source["episode_ids"] = source["episode_ids"].astype(np.uint64)
        source["episode_ids"][0] = np.uint64(2**63)

        with self.assertRaisesRegex(
            ValueError,
            "episode_ids values must fit in int64",
        ):
            reconstruct_episodes(source)

    def test_step_ids_must_fit_int64_before_canonicalization(self):
        source = clone_arrays(self.source)
        source["step_ids"] = source["step_ids"].astype(np.uint64)
        source["step_ids"][0] = np.uint64(2**63)

        with self.assertRaisesRegex(
            ValueError,
            "step_ids values must fit in int64",
        ):
            reconstruct_episodes(source)

    def test_duplicate_missing_or_negative_steps_are_rejected(self):
        for replacement in (
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int64),
            np.asarray([-1, 0], dtype=np.int64),
        ):
            with self.subTest(replacement=replacement):
                source = clone_arrays(self.source)
                mask = source["episode_ids"] == 3
                source["step_ids"][mask] = replacement
                with self.assertRaisesRegex(
                    ValueError,
                    "episode 3 step_ids.*contiguous",
                ):
                    reconstruct_episodes(source)

    def test_transition_discontinuity_names_episode_and_step(self):
        source = clone_arrays(self.source)
        first_row = np.flatnonzero(
            (source["episode_ids"] == 7) & (source["step_ids"] == 0)
        )[0]
        source["next_states"][first_row, 0] += 1e-3

        with self.assertRaisesRegex(
            ValueError,
            "episode 7 is discontinuous after step 0",
        ):
            reconstruct_episodes(source)

    def test_early_terminal_names_episode_and_step(self):
        source = clone_arrays(self.source)
        first_row = np.flatnonzero(
            (source["episode_ids"] == 7) & (source["step_ids"] == 0)
        )[0]
        source["dones"][first_row] = True

        with self.assertRaisesRegex(
            ValueError,
            "episode 7 terminates.*step 0",
        ):
            reconstruct_episodes(source)

    def test_non_terminal_final_row_names_episode(self):
        source = clone_arrays(self.source)
        final_row = np.flatnonzero(
            (source["episode_ids"] == 3) & (source["step_ids"] == 1)
        )[0]
        source["dones"][final_row] = False
        source["terminal_reasons"][final_row] = ""

        with self.assertRaisesRegex(ValueError, "episode 3 final row"):
            reconstruct_episodes(source)

    def test_invalid_terminal_reason_names_episode(self):
        source = clone_arrays(self.source)
        final_row = np.flatnonzero(
            (source["episode_ids"] == 3) & (source["step_ids"] == 1)
        )[0]
        source["terminal_reasons"][final_row] = "unknown"

        with self.assertRaisesRegex(
            ValueError,
            "episode 3 final terminal reason is invalid",
        ):
            reconstruct_episodes(source)


class VisualArtifactTest(unittest.TestCase):
    def setUp(self):
        self.source = make_transition_source()
        self.dataset = build_visual_dataset(self.source)

    def test_builds_n_plus_e_frames_and_exact_offsets(self):
        self.assertEqual(self.dataset["episode_ids"].tolist(), [3, 7])
        np.testing.assert_array_equal(
            self.dataset["frame_offsets"],
            [0, 3, 8],
        )
        np.testing.assert_array_equal(
            self.dataset["transition_offsets"],
            [0, 2, 6],
        )
        self.assertEqual(self.dataset["frames"].shape, (8, 64, 64, 3))
        self.assertEqual(self.dataset["states"].shape, (8, 4))
        self.assertEqual(self.dataset["actions"].shape, (6, 2))

    def test_frames_actions_and_states_preserve_causal_alignment(self):
        scene = scene_from_env(CarEnv())
        episodes = reconstruct_episodes(self.source)

        for episode_index, episode in enumerate(episodes):
            frame_start, frame_stop = self.dataset["frame_offsets"][
                episode_index : episode_index + 2
            ]
            transition_start, transition_stop = self.dataset[
                "transition_offsets"
            ][episode_index : episode_index + 2]
            aligned_states = np.concatenate(
                (episode.states[:1], episode.next_states),
                axis=0,
            )

            np.testing.assert_array_equal(
                self.dataset["states"][frame_start:frame_stop],
                aligned_states,
            )
            np.testing.assert_array_equal(
                self.dataset["actions"][transition_start:transition_stop],
                episode.actions,
            )
            for local_index, state in enumerate(aligned_states):
                expected_frame = render_observation(state, scene=scene)
                np.testing.assert_array_equal(
                    self.dataset["frames"][frame_start + local_index],
                    expected_frame,
                )

    def test_terminal_metadata_stays_on_transitions(self):
        episodes = reconstruct_episodes(self.source)

        for episode_index, episode in enumerate(episodes):
            start, stop = self.dataset["transition_offsets"][
                episode_index : episode_index + 2
            ]
            np.testing.assert_array_equal(
                self.dataset["rewards"][start:stop],
                episode.rewards,
            )
            np.testing.assert_array_equal(
                self.dataset["dones"][start:stop],
                episode.dones,
            )
            np.testing.assert_array_equal(
                self.dataset["terminal_reasons"][start:stop],
                episode.terminal_reasons,
            )

    def test_schema_and_scene_metadata_are_exact(self):
        self.assertEqual(self.dataset["schema_version"].item(), 1)
        self.assertEqual(self.dataset["image_size"].item(), 64)
        self.assertEqual(self.dataset["context_frames"].item(), 4)
        self.assertEqual(
            self.dataset["renderer_version"].item(),
            RENDERER_VERSION,
        )
        self.assertEqual(
            self.dataset["pillow_version"].item(),
            PILLOW_VERSION,
        )
        np.testing.assert_array_equal(
            self.dataset["scene_world_bounds"],
            [0.0, 10.0, 0.0, 8.0],
        )
        self.assertEqual(self.dataset["scene_dt"].item(), 0.1)

    def test_short_episodes_are_retained_but_not_eligible(self):
        summary = summarize_visual_dataset(self.dataset)

        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["transitions"], 6)
        self.assertEqual(summary["frames"], 8)
        self.assertEqual(summary["four_frame_eligible_episodes"], 1)
        self.assertEqual(summary["one_step_visual_samples"], 1)

    def test_same_source_produces_identical_arrays_and_metadata(self):
        second = build_visual_dataset(self.source)

        self.assertEqual(set(self.dataset), set(second))
        for name in self.dataset:
            np.testing.assert_array_equal(self.dataset[name], second[name])

    def test_shuffled_source_rows_produce_the_same_complete_artifact(self):
        permutation = np.asarray([5, 1, 4, 0, 3, 2])
        shuffled = {
            name: values[permutation]
            for name, values in self.source.items()
        }

        actual = build_visual_dataset(shuffled)

        self.assertEqual(set(self.dataset), set(actual))
        for name in self.dataset:
            np.testing.assert_array_equal(self.dataset[name], actual[name])

    def test_visual_artifact_round_trips_without_pickle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-episodes.npz"
            save_visual_dataset(self.dataset, path)
            loaded = load_visual_dataset(path)

        self.assertEqual(set(loaded), set(self.dataset))
        for name in self.dataset:
            np.testing.assert_array_equal(loaded[name], self.dataset[name])

    def test_save_refuses_to_overwrite_existing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-episodes.npz"
            save_visual_dataset(self.dataset, path)
            before = path.read_bytes()

            with self.assertRaises(FileExistsError):
                save_visual_dataset(self.dataset, path)

            self.assertEqual(path.read_bytes(), before)

    def test_atomic_npz_preserves_normal_file_creation_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference.bin"
            reference.open("xb").close()
            path = root / "visual-episodes.npz"

            save_visual_dataset(self.dataset, path)

            self.assertEqual(
                path.stat().st_mode & 0o777,
                reference.stat().st_mode & 0o777,
            )

    def test_validator_rejects_unknown_fields_in_stable_order(self):
        broken = clone_arrays(self.dataset)
        broken["z_extra"] = np.asarray(1, dtype=np.int64)
        broken["a_extra"] = np.asarray(object(), dtype=object)

        with self.assertRaisesRegex(
            ValueError,
            "visual dataset has unknown arrays: a_extra, z_extra",
        ):
            validate_visual_dataset(broken)

    def test_loader_rejects_unknown_archive_arrays_before_loading_values(self):
        extras = (
            ("numeric_extra", np.asarray(1, dtype=np.int64)),
            ("object_extra", np.asarray(object(), dtype=object)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, values in extras:
                with self.subTest(name=name):
                    path = root / f"{name}.npz"
                    archive = clone_arrays(self.dataset)
                    archive[name] = values
                    np.savez_compressed(path, **archive)

                    with self.assertRaisesRegex(
                        ValueError,
                        f"visual dataset has unknown arrays: {name}",
                    ):
                        load_visual_dataset(path)

    def test_save_rejects_object_extra_without_creating_artifact(self):
        broken = clone_arrays(self.dataset)
        broken["object_extra"] = np.asarray(object(), dtype=object)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "visual-episodes.npz"
            with self.assertRaisesRegex(
                ValueError,
                "visual dataset has unknown arrays: object_extra",
            ):
                save_visual_dataset(broken, path)

            self.assertFalse(path.exists())

    def test_save_uses_required_archive_order_for_identical_bytes(self):
        reversed_dataset = {
            name: self.dataset[name]
            for name in reversed(REQUIRED_VISUAL_ARRAYS)
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.npz"
            second = root / "second.npz"
            save_visual_dataset(self.dataset, first)
            save_visual_dataset(reversed_dataset, second)

            with np.load(second, allow_pickle=False) as loaded:
                self.assertEqual(loaded.files, list(REQUIRED_VISUAL_ARRAYS))
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_npz_encoder_failure_leaves_no_final_or_temporary_file(self):
        def fail_after_partial_write(handle, *args, **kwargs):
            del args, kwargs
            handle.write(b"partial-npz")
            handle.flush()
            raise OSError("injected NPZ encoder failure")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "visual-episodes.npz"
            with patch(
                "world_model_lab.visual_dataset.np.savez_compressed",
                side_effect=fail_after_partial_write,
            ):
                with self.assertRaisesRegex(OSError, "injected NPZ"):
                    save_visual_dataset(self.dataset, path)

            self.assertFalse(path.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_npz_publish_race_preserves_existing_final_file(self):
        original_save = np.savez_compressed

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "visual-episodes.npz"

            def create_racing_final(handle, *args, **kwargs):
                path.write_bytes(b"racing-writer")
                return original_save(handle, *args, **kwargs)

            with patch(
                "world_model_lab.visual_dataset.np.savez_compressed",
                side_effect=create_racing_final,
            ):
                with self.assertRaises(FileExistsError):
                    save_visual_dataset(self.dataset, path)

            self.assertEqual(path.read_bytes(), b"racing-writer")
            self.assertEqual(set(root.iterdir()), {path})

    def test_loader_uses_allow_pickle_false(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe-visual.npz"
            unsafe = clone_arrays(self.dataset)
            unsafe["renderer_version"] = np.asarray(object(), dtype=object)
            np.savez_compressed(path, **unsafe)

            with self.assertRaisesRegex(ValueError, "Object arrays"):
                load_visual_dataset(path)

    def test_malformed_visual_npz_is_reported_as_value_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "malformed-visual.npz"
            path.write_bytes(b"PK\x03\x04garbage")

            with self.assertRaisesRegex(
                ValueError,
                "malformed visual dataset NPZ",
            ):
                load_visual_dataset(path)

    def test_numeric_npy_is_rejected_as_non_npz_visual_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "numeric.npy"
            np.save(path, np.arange(4, dtype=np.int64))

            with self.assertRaisesRegex(
                ValueError,
                "visual dataset must be an NPZ archive",
            ):
                load_visual_dataset(path)

    def test_duplicate_visual_member_is_rejected_before_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate-visual.npz"
            np.savez_compressed(path, **self.dataset)
            append_duplicate_npz_array(
                path,
                "states",
                self.dataset["states"],
            )

            with self.assertRaisesRegex(
                ValueError,
                "visual dataset has duplicate arrays: states",
            ):
                load_visual_dataset(path)

    def test_corrupted_visual_member_is_reported_as_malformed_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupted-visual.npz"
            np.savez_compressed(path, **self.dataset)
            corrupt_compressed_npz_member(path, "states")

            with self.assertRaisesRegex(
                ValueError,
                "malformed visual dataset NPZ",
            ):
                load_visual_dataset(path)

    def test_missing_visual_field_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken.pop("actions")

        with self.assertRaisesRegex(
            ValueError,
            "visual dataset is missing arrays: actions",
        ):
            validate_visual_dataset(broken)

    def test_unsupported_schema_version_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["schema_version"] = np.asarray(2, dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "unsupported schema_version: 2"):
            validate_visual_dataset(broken)

    def test_frame_dtype_and_shape_are_strict(self):
        wrong_dtype = clone_arrays(self.dataset)
        wrong_dtype["frames"] = wrong_dtype["frames"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "frames must have dtype uint8"):
            validate_visual_dataset(wrong_dtype)

        wrong_shape = clone_arrays(self.dataset)
        wrong_shape["frames"] = wrong_shape["frames"][:, :, :, :2]
        with self.assertRaisesRegex(
            ValueError,
            r"frames must have shape \[F, 64, 64, 3\]",
        ):
            validate_visual_dataset(wrong_shape)

    def test_empty_renderer_or_pillow_version_is_rejected(self):
        for field in ("renderer_version", "pillow_version"):
            with self.subTest(field=field):
                broken = clone_arrays(self.dataset)
                broken[field] = np.asarray("", dtype=np.str_)
                with self.assertRaisesRegex(ValueError, f"{field}.*non-empty"):
                    validate_visual_dataset(broken)

    def test_noncanonical_renderer_version_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["renderer_version"] = np.asarray(
            "different-renderer",
            dtype=np.str_,
        )

        with self.assertRaisesRegex(
            ValueError,
            "schema version 1 requires renderer_version=pillow-raster-v1",
        ):
            validate_visual_dataset(broken)

    def test_invalid_offsets_are_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["frame_offsets"] = np.asarray([0, 4, 8], dtype=np.int64)

        with self.assertRaisesRegex(
            ValueError,
            "episode 3 must own T plus one frames",
        ):
            validate_visual_dataset(broken)

    def test_offset_checks_reject_modular_overflow_before_slicing(self):
        broken = clone_arrays(self.dataset)
        broken["actions"] = broken["actions"][:4]
        broken["rewards"] = broken["rewards"][:4]
        broken["dones"] = np.zeros((4,), dtype=np.bool_)
        broken["terminal_reasons"] = np.asarray(
            ["", "", "", ""],
            dtype=broken["terminal_reasons"].dtype,
        )
        broken["episode_ids"] = np.arange(4, dtype=np.int64)
        broken["frame_offsets"] = np.asarray(
            [0, 2**62 + 2, -(2**63) + 4, -(2**62) + 6, 8],
            dtype=np.int64,
        )
        broken["transition_offsets"] = np.asarray(
            [0, 2**62 + 1, -(2**63) + 2, -(2**62) + 3, 4],
            dtype=np.int64,
        )

        with self.assertRaisesRegex(
            ValueError,
            "frame_offsets must cover every frame exactly once",
        ):
            validate_visual_dataset(broken)

    def test_noncanonical_episode_ids_are_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["episode_ids"] = np.asarray([7, 3], dtype=np.int64)

        with self.assertRaisesRegex(
            ValueError,
            "episode_ids must be strictly increasing",
        ):
            validate_visual_dataset(broken)

    def test_nondefault_scene_metadata_is_rejected(self):
        broken = clone_arrays(self.dataset)
        broken["scene_goal"] = np.asarray([8.0, 7.0], dtype=np.float64)

        with self.assertRaisesRegex(
            ValueError,
            "scene_goal does not match the schema-v1 default scene",
        ):
            validate_visual_dataset(broken)
