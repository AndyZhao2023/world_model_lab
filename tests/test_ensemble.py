import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from world_model_lab.dataset import Normalizer
from world_model_lab.ensemble import (
    build_ensemble,
    load_ensemble,
    predict_ensemble_next_states,
    rollout_ensemble,
)
from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import (
    LoadedWorldModel,
    TrainingResult,
    save_checkpoint,
)


def make_member(seed: int, delta: np.ndarray) -> LoadedWorldModel:
    model = WorldModelMLP(hidden_size=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(
            torch.as_tensor(delta, dtype=torch.float32)
        )
    return LoadedWorldModel(
        model=model,
        input_normalizer=Normalizer(np.zeros(7), np.ones(7)),
        target_normalizer=Normalizer(np.zeros(4), np.ones(4)),
        split_episode_ids={
            "train": np.asarray([0, 1]),
            "validation": np.asarray([2]),
            "test": np.asarray([3]),
        },
        training_config={
            "seed": seed,
            "split_seed": 0,
            "hidden_size": 4,
            "rollout_horizon": 10,
            "rollout_loss_weight": 1.0,
        },
        train_losses=[1.0],
        validation_losses=[1.0],
        best_epoch=1,
        test_metrics={},
    )


def save_member_checkpoint(path: Path, member: LoadedWorldModel) -> Path:
    result = TrainingResult(
        model=member.model,
        input_normalizer=member.input_normalizer,
        target_normalizer=member.target_normalizer,
        train_losses=member.train_losses,
        validation_losses=member.validation_losses,
        best_epoch=member.best_epoch,
    )
    return save_checkpoint(
        path,
        result,
        split_episode_ids=member.split_episode_ids,
        training_config=member.training_config,
        test_metrics=member.test_metrics,
    )


class EnsembleTest(unittest.TestCase):
    def test_prediction_uses_arithmetic_mean_and_rms_disagreement(self):
        ensemble = build_ensemble(
            (
                make_member(1, np.asarray([3.0, 4.0, 0.0, 2.0])),
                make_member(0, np.asarray([1.0, 2.0, 0.0, 0.0])),
            )
        )
        prediction = predict_ensemble_next_states(
            ensemble,
            np.zeros((1, 4)),
            np.zeros((1, 2)),
        )

        self.assertEqual(ensemble.seeds, (0, 1))
        np.testing.assert_allclose(
            prediction.mean_next_states[0],
            [2.0, 3.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            prediction.disagreement["position"], math.sqrt(2.0)
        )
        np.testing.assert_allclose(prediction.disagreement["velocity"], 1.0)

    def test_heading_mean_is_circular_across_wrap_boundary(self):
        ensemble = build_ensemble(
            (
                make_member(
                    0,
                    np.asarray([0.0, 0.0, math.radians(179.0), 0.0]),
                ),
                make_member(
                    1,
                    np.asarray([0.0, 0.0, math.radians(-179.0), 0.0]),
                ),
            )
        )
        prediction = predict_ensemble_next_states(
            ensemble,
            np.zeros((1, 4)),
            np.zeros((1, 2)),
        )

        self.assertAlmostEqual(
            abs(math.degrees(prediction.mean_next_states[0, 2])),
            180.0,
            places=5,
        )
        self.assertAlmostEqual(
            prediction.disagreement["heading_degrees"][0],
            1.0,
            places=5,
        )

    def test_build_rejects_incompatible_or_invalid_members(self):
        valid = make_member(0, np.zeros(4))
        cases = (
            ((valid,), "at least two"),
            ((valid, make_member(0, np.ones(4))), "unique"),
        )
        for members, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_ensemble(members)

        for field, value, message in (
            ("split_seed", 1, "split_seed"),
            ("rollout_horizon", 1, "rollout_horizon"),
            ("rollout_loss_weight", 2.0, "rollout_loss_weight"),
            ("hidden_size", 8, "hidden_size"),
        ):
            incompatible = make_member(1, np.ones(4))
            incompatible.training_config[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, message):
                    build_ensemble((valid, incompatible))

        wrong_objective = make_member(1, np.ones(4))
        valid.training_config["rollout_horizon"] = 1
        wrong_objective.training_config["rollout_horizon"] = 1
        with self.assertRaisesRegex(ValueError, "rollout_horizon must equal 10"):
            build_ensemble((valid, wrong_objective))

    def test_build_rejects_model_hidden_size_metadata_mismatch(self):
        valid = make_member(0, np.zeros(4))
        inconsistent_model = make_member(1, np.ones(4))
        inconsistent_model.model = WorldModelMLP(hidden_size=8)
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            build_ensemble((valid, inconsistent_model))

    def test_build_requires_rollout_horizon_to_be_exact_integer_ten(self):
        for invalid_horizon in (10.5, 10.0, "10", True, 9, 11):
            first = make_member(0, np.zeros(4))
            second = make_member(1, np.ones(4))
            first.training_config["rollout_horizon"] = invalid_horizon
            second.training_config["rollout_horizon"] = invalid_horizon
            with self.subTest(invalid_horizon=invalid_horizon):
                with self.assertRaisesRegex(
                    ValueError,
                    "rollout_horizon must equal 10",
                ):
                    build_ensemble((first, second))

        first = make_member(0, np.zeros(4))
        second = make_member(1, np.ones(4))
        first.training_config["rollout_horizon"] = np.int64(10)
        second.training_config["rollout_horizon"] = np.int64(10)
        self.assertEqual(build_ensemble((first, second)).seeds, (0, 1))

    def test_build_rejects_normalizer_and_split_mismatches(self):
        valid = make_member(0, np.zeros(4))

        normalizer_mismatch = make_member(1, np.ones(4))
        normalizer_mismatch.input_normalizer = Normalizer(
            np.ones(7), np.ones(7)
        )
        with self.assertRaisesRegex(ValueError, "input mean"):
            build_ensemble((valid, normalizer_mismatch))

        split_mismatch = make_member(1, np.ones(4))
        split_mismatch.split_episode_ids["test"] = np.asarray([4])
        with self.assertRaisesRegex(ValueError, "test split episode IDs"):
            build_ensemble((valid, split_mismatch))

    def test_build_rejects_missing_required_split_arrays(self):
        first = make_member(0, np.zeros(4))
        second = make_member(1, np.ones(4))
        del first.split_episode_ids["test"]
        del second.split_episode_ids["test"]
        with self.assertRaisesRegex(ValueError, "test split episode IDs"):
            build_ensemble((first, second))

    def test_build_rejects_invalid_normalizer_arrays(self):
        invalid_cases = (
            (
                Normalizer(np.zeros(6), np.ones(7)),
                Normalizer(np.zeros(4), np.ones(4)),
                "input mean",
            ),
            (
                Normalizer(np.zeros(7), np.ones(7)),
                Normalizer(np.zeros(4), np.asarray([1.0, 1.0, 1.0, 0.0])),
                "target std",
            ),
            (
                Normalizer(np.zeros(7), np.ones(7)),
                Normalizer(
                    np.asarray([0.0, 0.0, np.nan, 0.0]),
                    np.ones(4),
                ),
                "target mean",
            ),
        )
        for input_normalizer, target_normalizer, message in invalid_cases:
            invalid = make_member(0, np.zeros(4))
            invalid.input_normalizer = input_normalizer
            invalid.target_normalizer = target_normalizer
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_ensemble((invalid, make_member(1, np.ones(4))))

    def test_build_rejects_missing_or_negative_seed(self):
        missing = make_member(0, np.zeros(4))
        del missing.training_config["seed"]
        with self.assertRaisesRegex(ValueError, "missing seed"):
            build_ensemble((missing, make_member(1, np.ones(4))))

        negative = make_member(-1, np.zeros(4))
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            build_ensemble((negative, make_member(1, np.ones(4))))

    def test_prediction_rejects_non_finite_member_output(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.zeros(4)),
                make_member(1, np.asarray([np.nan, 0.0, 0.0, 0.0])),
            )
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            predict_ensemble_next_states(
                ensemble,
                np.zeros((1, 4)),
                np.zeros((1, 2)),
            )

    def test_load_sorts_checkpoints_and_rejects_duplicate_or_missing_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            seed_one = save_member_checkpoint(
                root / "seed_1.pt",
                make_member(1, np.ones(4)),
            )
            seed_zero = save_member_checkpoint(
                root / "seed_0.pt",
                make_member(0, np.zeros(4)),
            )

            ensemble = load_ensemble((seed_one, seed_zero))

            self.assertEqual(ensemble.seeds, (0, 1))
            self.assertEqual(
                ensemble.checkpoint_paths,
                (seed_zero.resolve(), seed_one.resolve()),
            )
            with self.assertRaisesRegex(ValueError, "unique"):
                load_ensemble((seed_zero, seed_zero))
            with self.assertRaisesRegex(FileNotFoundError, "regular file"):
                load_ensemble((seed_zero, root / "missing.pt"))

    def test_load_expands_user_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            home = Path(temporary_directory)
            checkpoint_directory = home / "models"
            checkpoint_directory.mkdir()
            seed_one = save_member_checkpoint(
                checkpoint_directory / "seed_1.pt",
                make_member(1, np.ones(4)),
            )
            seed_zero = save_member_checkpoint(
                checkpoint_directory / "seed_0.pt",
                make_member(0, np.zeros(4)),
            )

            with patch.dict("os.environ", {"HOME": str(home)}):
                try:
                    ensemble = load_ensemble(
                        ("~/models/seed_1.pt", "~/models/seed_0.pt")
                    )
                except FileNotFoundError as error:
                    self.fail(f"load_ensemble did not expand user paths: {error}")

            self.assertEqual(
                ensemble.checkpoint_paths,
                (seed_zero.resolve(), seed_one.resolve()),
            )

    def test_rollout_recursively_advances_each_member_independently(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        rollout = rollout_ensemble(
            ensemble,
            np.zeros(4),
            np.zeros((2, 2)),
        )

        np.testing.assert_allclose(
            rollout.member_states[:, :, 0],
            [[0, 1, 2], [0, 3, 6]],
        )
        np.testing.assert_allclose(rollout.mean_states[:, 0], [0, 2, 4])
        np.testing.assert_allclose(rollout.disagreement["position"], [0, 1, 2])

    def test_rollout_rejects_invalid_inputs(self):
        ensemble = build_ensemble(
            (make_member(0, np.zeros(4)), make_member(1, np.ones(4)))
        )
        for initial_state, actions, message in (
            (np.zeros(3), np.zeros((1, 2)), "initial_state"),
            (np.zeros(4), np.zeros((1, 3)), "actions"),
            (
                np.asarray([0.0, 0.0, np.nan, 0.0]),
                np.zeros((1, 2)),
                "finite",
            ),
            (
                np.zeros(4),
                np.asarray([[0.0, np.inf]]),
                "finite",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    rollout_ensemble(ensemble, initial_state, actions)
