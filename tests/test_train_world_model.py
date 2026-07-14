import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from world_model_lab import train_world_model
from world_model_lab.dataset import (
    Normalizer,
    build_model_arrays,
    build_sequence_windows,
    fit_normalizer,
)
from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import (
    evaluate_model,
    load_checkpoint,
    predict_deltas,
    run_training,
    save_checkpoint,
    train_model,
)


def make_linear_dynamics(count: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(23)
    inputs = rng.normal(size=(count, 7))
    weights = rng.normal(size=(7, 4))
    targets = inputs @ weights + np.asarray([0.2, -0.1, 0.05, 0.3])
    return inputs, targets


def make_sequence_dynamics(
    episodes: int = 4,
    steps: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    states = []
    actions = []
    next_states = []
    episode_ids = []
    step_ids = []
    for episode_id in range(episodes):
        state = np.asarray(
            [float(episode_id), 0.2 * episode_id, 0.03 * episode_id, 0.2]
        )
        for step in range(steps):
            action = np.asarray(
                [0.03 * episode_id + 0.01 * step, 0.05 + 0.02 * step]
            )
            delta = np.asarray(
                [
                    0.1 + 0.02 * state[3],
                    0.01 + 0.005 * state[0],
                    0.01 * action[0] + 0.002 * state[3],
                    0.01 + 0.02 * action[1],
                ]
            )
            next_state = state + delta
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            episode_ids.append(episode_id)
            step_ids.append(step)
            state = next_state
    return tuple(
        np.asarray(values)
        for values in (states, actions, next_states, episode_ids, step_ids)
    )


def _save_sequence_dataset(path: Path) -> None:
    states, actions, next_states, episode_ids, step_ids = (
        make_sequence_dynamics(episodes=10, steps=12)
    )
    np.savez_compressed(
        path,
        states=states,
        actions=actions,
        next_states=next_states,
        episode_ids=episode_ids,
        step_ids=step_ids,
    )


class TrainWorldModelTest(unittest.TestCase):
    def test_cli_help_lists_bootstrap_seed(self):
        standard_output = io.StringIO()
        with patch.object(sys, "argv", ["world-model-train", "--help"]):
            with redirect_stdout(standard_output):
                with self.assertRaises(SystemExit) as context:
                    train_world_model.main()

        self.assertEqual(context.exception.code, 0)
        self.assertIn("--bootstrap-seed", standard_output.getvalue())

    def test_bootstrap_training_rejects_negative_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            _save_sequence_dataset(data_path)

            with self.assertRaisesRegex(ValueError, "bootstrap seed"):
                run_training(
                    data_path=data_path,
                    output_path=root / "world_model.pt",
                    hidden_size=8,
                    epochs=1,
                    batch_size=32,
                    seed=3,
                    split_seed=0,
                    rollout_horizon=10,
                    rollout_loss_weight=1.0,
                    bootstrap_seed=-1,
                )

    def test_bootstrap_training_preserves_split_and_shared_normalizers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            _save_sequence_dataset(data_path)
            baseline_path = root / "baseline.pt"
            bootstrap_path = root / "bootstrap.pt"

            baseline_summary = run_training(
                data_path=data_path,
                output_path=baseline_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=3,
                split_seed=0,
                rollout_horizon=10,
                rollout_loss_weight=1.0,
            )
            bootstrap_summary = run_training(
                data_path=data_path,
                output_path=bootstrap_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=3,
                split_seed=0,
                rollout_horizon=10,
                rollout_loss_weight=1.0,
                bootstrap_seed=3,
            )
            baseline = load_checkpoint(baseline_path)
            bootstrap = load_checkpoint(bootstrap_path)

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                baseline.split_episode_ids[name],
                bootstrap.split_episode_ids[name],
            )
        np.testing.assert_array_equal(
            baseline.input_normalizer.mean,
            bootstrap.input_normalizer.mean,
        )
        np.testing.assert_array_equal(
            baseline.input_normalizer.std,
            bootstrap.input_normalizer.std,
        )
        np.testing.assert_array_equal(
            baseline.target_normalizer.mean,
            bootstrap.target_normalizer.mean,
        )
        np.testing.assert_array_equal(
            baseline.target_normalizer.std,
            bootstrap.target_normalizer.std,
        )
        config = bootstrap.training_config
        self.assertEqual(config["bootstrap_seed"], 3)
        self.assertEqual(
            sum(config["bootstrap_episode_counts"].values()),
            config["bootstrap_episode_draws"],
        )
        self.assertEqual(
            sum(
                value > 0
                for value in config["bootstrap_episode_counts"].values()
            ),
            config["bootstrap_unique_episodes"],
        )
        self.assertNotIn("bootstrap_seed", baseline.training_config)
        self.assertEqual(
            bootstrap_summary["split_transitions"]["validation"],
            baseline_summary["split_transitions"]["validation"],
        )
        self.assertEqual(
            bootstrap_summary["split_transitions"]["test"],
            baseline_summary["split_transitions"]["test"],
        )
        self.assertEqual(
            bootstrap_summary["bootstrap_train_transitions"],
            12 * config["bootstrap_episode_draws"],
        )
        self.assertEqual(
            bootstrap_summary["train_sequence_windows"],
            3 * config["bootstrap_episode_draws"],
        )

    def test_bootstrap_training_is_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            _save_sequence_dataset(data_path)
            checkpoints = []
            summaries = []
            for repeat in range(2):
                checkpoint_path = root / f"bootstrap-{repeat}.pt"
                summaries.append(
                    run_training(
                        data_path=data_path,
                        output_path=checkpoint_path,
                        hidden_size=8,
                        epochs=2,
                        batch_size=32,
                        learning_rate=1e-3,
                        seed=3,
                        split_seed=0,
                        rollout_horizon=10,
                        rollout_loss_weight=1.0,
                        bootstrap_seed=7,
                    )
                )
                checkpoints.append(load_checkpoint(checkpoint_path))

        first, second = checkpoints
        self.assertEqual(
            first.training_config["bootstrap_seed"],
            second.training_config["bootstrap_seed"],
        )
        self.assertEqual(
            first.training_config["bootstrap_episode_counts"],
            second.training_config["bootstrap_episode_counts"],
        )
        self.assertEqual(first.train_losses, second.train_losses)
        self.assertEqual(first.validation_losses, second.validation_losses)
        self.assertEqual(
            first.train_one_step_losses,
            second.train_one_step_losses,
        )
        self.assertEqual(
            first.train_rollout_losses,
            second.train_rollout_losses,
        )
        self.assertEqual(
            first.validation_one_step_losses,
            second.validation_one_step_losses,
        )
        self.assertEqual(
            first.validation_rollout_losses,
            second.validation_rollout_losses,
        )
        np.testing.assert_array_equal(
            first.input_normalizer.mean,
            second.input_normalizer.mean,
        )
        np.testing.assert_array_equal(
            first.input_normalizer.std,
            second.input_normalizer.std,
        )
        np.testing.assert_array_equal(
            first.target_normalizer.mean,
            second.target_normalizer.mean,
        )
        np.testing.assert_array_equal(
            first.target_normalizer.std,
            second.target_normalizer.std,
        )
        self.assertEqual(
            summaries[0]["bootstrap_train_transitions"],
            summaries[1]["bootstrap_train_transitions"],
        )
        for name, first_tensor in first.model.state_dict().items():
            torch.testing.assert_close(
                first_tensor,
                second.model.state_dict()[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_train_model_uses_explicit_shared_normalizers(self):
        inputs, targets = make_linear_dynamics(count=80)
        shared_input = fit_normalizer(inputs[:64])
        shared_target = fit_normalizer(targets[:64])

        result = train_model(
            inputs[:32],
            targets[:32],
            validation_inputs=inputs[64:],
            validation_targets=targets[64:],
            input_normalizer=shared_input,
            target_normalizer=shared_target,
            hidden_size=8,
            epochs=1,
            batch_size=16,
            seed=3,
        )

        np.testing.assert_array_equal(
            result.input_normalizer.mean,
            shared_input.mean,
        )
        np.testing.assert_array_equal(
            result.input_normalizer.std,
            shared_input.std,
        )
        np.testing.assert_array_equal(
            result.target_normalizer.mean,
            shared_target.mean,
        )
        np.testing.assert_array_equal(
            result.target_normalizer.std,
            shared_target.std,
        )

    def test_train_model_requires_both_explicit_normalizers(self):
        inputs, targets = make_linear_dynamics(count=64)
        shared_input = fit_normalizer(inputs[:48])
        shared_target = fit_normalizer(targets[:48])
        cases = (
            {"input_normalizer": shared_input},
            {"target_normalizer": shared_target},
        )

        for provided in cases:
            with self.subTest(provided=tuple(provided)):
                with self.assertRaisesRegex(ValueError, "provided together"):
                    train_model(
                        inputs[:48],
                        targets[:48],
                        validation_inputs=inputs[48:],
                        validation_targets=targets[48:],
                        hidden_size=8,
                        epochs=1,
                        **provided,
                    )

    def test_train_model_rejects_invalid_normalizer_shapes(self):
        inputs, targets = make_linear_dynamics(count=64)
        shared_input = fit_normalizer(inputs[:48])
        shared_target = fit_normalizer(targets[:48])
        cases = (
            (
                "input mean",
                Normalizer(shared_input.mean[:-1], shared_input.std),
                shared_target,
                "input_normalizer.*shape \\[7\\]",
            ),
            (
                "input std",
                Normalizer(shared_input.mean, shared_input.std[:-1]),
                shared_target,
                "input_normalizer.*shape \\[7\\]",
            ),
            (
                "target mean",
                shared_input,
                Normalizer(shared_target.mean[:-1], shared_target.std),
                "target_normalizer.*shape \\[4\\]",
            ),
            (
                "target std",
                shared_input,
                Normalizer(shared_target.mean, shared_target.std[:-1]),
                "target_normalizer.*shape \\[4\\]",
            ),
        )

        for name, input_normalizer, target_normalizer, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    train_model(
                        inputs[:48],
                        targets[:48],
                        validation_inputs=inputs[48:],
                        validation_targets=targets[48:],
                        input_normalizer=input_normalizer,
                        target_normalizer=target_normalizer,
                        hidden_size=8,
                        epochs=1,
                    )

    def test_train_model_rejects_non_finite_normalizer_values(self):
        inputs, targets = make_linear_dynamics(count=64)
        shared_input = fit_normalizer(inputs[:48])
        shared_target = fit_normalizer(targets[:48])

        def changed(values, *, index, value):
            result = values.copy()
            result[index] = value
            return result

        cases = (
            (
                "input mean",
                Normalizer(
                    changed(shared_input.mean, index=0, value=np.nan),
                    shared_input.std,
                ),
                shared_target,
                "input_normalizer.*finite",
            ),
            (
                "input std",
                Normalizer(
                    shared_input.mean,
                    changed(shared_input.std, index=0, value=np.inf),
                ),
                shared_target,
                "input_normalizer.*finite",
            ),
            (
                "target mean",
                shared_input,
                Normalizer(
                    changed(shared_target.mean, index=0, value=np.nan),
                    shared_target.std,
                ),
                "target_normalizer.*finite",
            ),
            (
                "target std",
                shared_input,
                Normalizer(
                    shared_target.mean,
                    changed(shared_target.std, index=0, value=np.inf),
                ),
                "target_normalizer.*finite",
            ),
        )

        for name, input_normalizer, target_normalizer, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    train_model(
                        inputs[:48],
                        targets[:48],
                        validation_inputs=inputs[48:],
                        validation_targets=targets[48:],
                        input_normalizer=input_normalizer,
                        target_normalizer=target_normalizer,
                        hidden_size=8,
                        epochs=1,
                    )

    def test_train_model_rejects_non_positive_normalizer_std(self):
        inputs, targets = make_linear_dynamics(count=64)
        shared_input = fit_normalizer(inputs[:48])
        shared_target = fit_normalizer(targets[:48])

        def changed(values, *, value):
            result = values.copy()
            result[0] = value
            return result

        cases = (
            (
                "zero input std",
                Normalizer(shared_input.mean, changed(shared_input.std, value=0.0)),
                shared_target,
                "input_normalizer.*positive std",
            ),
            (
                "negative input std",
                Normalizer(shared_input.mean, changed(shared_input.std, value=-1.0)),
                shared_target,
                "input_normalizer.*positive std",
            ),
            (
                "zero target std",
                shared_input,
                Normalizer(shared_target.mean, changed(shared_target.std, value=0.0)),
                "target_normalizer.*positive std",
            ),
            (
                "negative target std",
                shared_input,
                Normalizer(shared_target.mean, changed(shared_target.std, value=-1.0)),
                "target_normalizer.*positive std",
            ),
        )

        for name, input_normalizer, target_normalizer, message in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    train_model(
                        inputs[:48],
                        targets[:48],
                        validation_inputs=inputs[48:],
                        validation_targets=targets[48:],
                        input_normalizer=input_normalizer,
                        target_normalizer=target_normalizer,
                        hidden_size=8,
                        epochs=1,
                    )

    def test_multistep_training_records_finite_component_histories(self):
        states, actions, next_states, episode_ids, step_ids = (
            make_sequence_dynamics()
        )
        inputs, targets = build_model_arrays(states, actions, next_states)
        train_mask = episode_ids < 3
        validation_mask = episode_ids == 3
        train_sequences = build_sequence_windows(
            states,
            actions,
            next_states,
            episode_ids=episode_ids,
            step_ids=step_ids,
            selected_episode_ids=np.asarray([0, 1, 2]),
            horizon=3,
        )
        validation_sequences = build_sequence_windows(
            states,
            actions,
            next_states,
            episode_ids=episode_ids,
            step_ids=step_ids,
            selected_episode_ids=np.asarray([3]),
            horizon=3,
        )

        result = train_model(
            inputs[train_mask],
            targets[train_mask],
            validation_inputs=inputs[validation_mask],
            validation_targets=targets[validation_mask],
            train_sequences=train_sequences,
            validation_sequences=validation_sequences,
            rollout_loss_weight=1.0,
            hidden_size=16,
            epochs=4,
            batch_size=4,
            learning_rate=1e-3,
            seed=9,
        )

        self.assertEqual(len(result.train_losses), 4)
        self.assertTrue(np.all(np.isfinite(result.train_one_step_losses)))
        self.assertTrue(np.all(np.isfinite(result.train_rollout_losses)))
        self.assertTrue(np.all(np.asarray(result.train_rollout_losses) > 0.0))
        np.testing.assert_allclose(
            result.train_losses,
            np.asarray(result.train_one_step_losses)
            + np.asarray(result.train_rollout_losses),
            rtol=1e-6,
        )

    def test_predict_deltas_returns_denormalized_physical_values(self):
        inputs, targets = make_linear_dynamics(count=64)
        result = train_model(
            inputs[:48],
            targets[:48],
            validation_inputs=inputs[48:],
            validation_targets=targets[48:],
            hidden_size=16,
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            seed=3,
        )

        predictions = predict_deltas(result, inputs[:5])

        self.assertEqual(predictions.shape, (5, 4))
        self.assertTrue(np.all(np.isfinite(predictions)))

    def test_mlp_maps_each_input_row_to_four_deltas(self):
        model = WorldModelMLP(hidden_size=16)

        output = model(torch.zeros((5, 7), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (5, 4))

    def test_training_reduces_normalized_loss_on_deterministic_dynamics(self):
        inputs, targets = make_linear_dynamics()
        train_inputs, validation_inputs = inputs[:192], inputs[192:]
        train_targets, validation_targets = targets[:192], targets[192:]

        result = train_model(
            train_inputs,
            train_targets,
            validation_inputs=validation_inputs,
            validation_targets=validation_targets,
            hidden_size=32,
            epochs=100,
            batch_size=64,
            learning_rate=1e-2,
            seed=11,
        )

        self.assertEqual(len(result.train_losses), 100)
        self.assertEqual(len(result.validation_losses), 100)
        self.assertEqual(result.train_one_step_losses, result.train_losses)
        self.assertEqual(result.validation_one_step_losses, result.validation_losses)
        self.assertEqual(result.train_rollout_losses, [0.0] * 100)
        self.assertEqual(result.validation_rollout_losses, [0.0] * 100)
        self.assertLess(result.train_losses[-1], result.train_losses[0] * 0.1)
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertLessEqual(result.best_epoch, 100)
        validation_metrics = evaluate_model(
            result, validation_inputs, validation_targets
        )
        self.assertAlmostEqual(
            validation_metrics["normalized_mse"],
            min(result.validation_losses),
            places=6,
        )
        metrics = evaluate_model(result, inputs, targets)
        self.assertLess(metrics["mae_x"], 0.12)
        self.assertLess(metrics["mae_y"], 0.12)
        self.assertLess(metrics["mae_heading_radians"], 0.12)
        self.assertLess(metrics["mae_velocity"], 0.12)
        self.assertAlmostEqual(
            metrics["mae_heading_degrees"],
            np.degrees(metrics["mae_heading_radians"]),
        )

    def test_checkpoint_round_trip_preserves_predictions_and_metadata(self):
        inputs, targets = make_linear_dynamics(count=64)
        result = train_model(
            inputs[:48],
            targets[:48],
            validation_inputs=inputs[48:],
            validation_targets=targets[48:],
            hidden_size=12,
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            seed=5,
        )
        split_ids = {
            "train": np.asarray([0, 1, 2]),
            "validation": np.asarray([3]),
            "test": np.asarray([4]),
        }
        test_metrics = {
            "normalized_mse": 0.25,
            "mae_x": 0.1,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "world_model.pt"
            save_checkpoint(
                path,
                result,
                split_episode_ids=split_ids,
                training_config={"epochs": 2, "seed": 5},
                test_metrics=test_metrics,
            )
            loaded = load_checkpoint(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)

        self.assertEqual(loaded.model.hidden_size, 12)
        self.assertEqual(loaded.training_config, {"epochs": 2, "seed": 5})
        self.assertEqual(loaded.train_losses, result.train_losses)
        self.assertEqual(loaded.validation_losses, result.validation_losses)
        self.assertEqual(
            loaded.train_one_step_losses,
            result.train_one_step_losses,
        )
        self.assertEqual(
            loaded.train_rollout_losses,
            result.train_rollout_losses,
        )
        self.assertEqual(
            loaded.validation_one_step_losses,
            result.validation_one_step_losses,
        )
        self.assertEqual(
            loaded.validation_rollout_losses,
            result.validation_rollout_losses,
        )
        self.assertEqual(payload["format_version"], 3)
        self.assertEqual(loaded.best_epoch, result.best_epoch)
        self.assertEqual(loaded.test_metrics, test_metrics)
        for name in split_ids:
            np.testing.assert_array_equal(loaded.split_episode_ids[name], split_ids[name])
        np.testing.assert_allclose(loaded.input_normalizer.mean, result.input_normalizer.mean)
        np.testing.assert_allclose(loaded.target_normalizer.std, result.target_normalizer.std)

        normalized = result.input_normalizer.normalize(inputs[:8])
        tensor = torch.as_tensor(normalized, dtype=torch.float32)
        with torch.no_grad():
            expected = result.model(tensor).numpy()
            actual = loaded.model(tensor).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_checkpoint_loader_supports_versions_one_and_two(self):
        inputs, targets = make_linear_dynamics(count=64)
        result = train_model(
            inputs[:48],
            targets[:48],
            validation_inputs=inputs[48:],
            validation_targets=targets[48:],
            hidden_size=12,
            epochs=2,
            batch_size=32,
            learning_rate=1e-3,
            seed=5,
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            current_path = directory_path / "world_model_v3.pt"
            save_checkpoint(
                current_path,
                result,
                split_episode_ids={
                    "train": np.asarray([0]),
                    "validation": np.asarray([1]),
                    "test": np.asarray([2]),
                },
                training_config={"epochs": 2},
                test_metrics={"normalized_mse": 0.25},
            )
            payload = torch.load(
                current_path,
                map_location="cpu",
                weights_only=True,
            )
            component_keys = (
                "train_one_step_losses",
                "train_rollout_losses",
                "validation_one_step_losses",
                "validation_rollout_losses",
            )

            version_two_payload = dict(payload)
            version_two_payload["format_version"] = 2
            for key in component_keys:
                version_two_payload.pop(key)
            version_two_path = directory_path / "world_model_v2.pt"
            torch.save(version_two_payload, version_two_path)
            loaded_v2 = load_checkpoint(version_two_path)

            version_one_payload = dict(payload)
            version_one_payload["format_version"] = 1
            version_one_payload["losses"] = version_one_payload.pop(
                "train_losses"
            )
            for key in (
                "validation_losses",
                *component_keys,
                "best_epoch",
                "test_metrics",
            ):
                version_one_payload.pop(key)
            version_one_path = directory_path / "world_model_v1.pt"
            torch.save(version_one_payload, version_one_path)
            loaded_v1 = load_checkpoint(version_one_path)

        self.assertEqual(
            loaded_v2.train_one_step_losses,
            loaded_v2.train_losses,
        )
        self.assertEqual(
            loaded_v2.train_rollout_losses,
            [0.0] * len(loaded_v2.train_losses),
        )
        self.assertEqual(
            loaded_v2.validation_one_step_losses,
            loaded_v2.validation_losses,
        )
        self.assertEqual(
            loaded_v2.validation_rollout_losses,
            [0.0] * len(loaded_v2.validation_losses),
        )
        self.assertEqual(
            loaded_v1.train_one_step_losses,
            loaded_v1.train_losses,
        )
        self.assertEqual(
            loaded_v1.train_rollout_losses,
            [0.0] * len(loaded_v1.train_losses),
        )
        self.assertEqual(loaded_v1.validation_losses, [])

    def test_run_training_splits_npz_by_episode_and_saves_checkpoint(self):
        rng = np.random.default_rng(41)
        episode_ids = np.repeat(np.arange(10), 8)
        count = episode_ids.size
        states = np.column_stack(
            (
                rng.uniform(0.5, 9.5, count),
                rng.uniform(0.5, 7.5, count),
                rng.uniform(-np.pi, np.pi, count),
                rng.uniform(0.1, 2.9, count),
            )
        )
        actions = rng.uniform([-0.5, -1.0], [0.5, 1.0], size=(count, 2))
        deltas = np.column_stack(
            (
                0.03 * states[:, 3] + 0.01 * actions[:, 0],
                -0.02 * states[:, 3] + 0.01 * actions[:, 1],
                0.04 * actions[:, 0] + 0.01 * states[:, 3],
                0.05 * actions[:, 1] - 0.01 * states[:, 3],
            )
        )
        next_states = states + deltas

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            dataset_path = directory_path / "transitions.npz"
            checkpoint_path = directory_path / "world_model.pt"
            np.savez_compressed(
                dataset_path,
                states=states,
                actions=actions,
                next_states=next_states,
                episode_ids=episode_ids,
            )

            summary = run_training(
                data_path=dataset_path,
                output_path=checkpoint_path,
                hidden_size=16,
                epochs=3,
                batch_size=32,
                learning_rate=1e-3,
                seed=7,
            )
            loaded = load_checkpoint(checkpoint_path)
            with self.assertRaisesRegex(ValueError, "step_ids"):
                run_training(
                    data_path=dataset_path,
                    output_path=directory_path / "world_model_h2.pt",
                    hidden_size=8,
                    epochs=1,
                    batch_size=16,
                    rollout_horizon=2,
                )

        self.assertEqual(summary["transitions"], count)
        self.assertEqual(summary["rollout_horizon"], 1)
        self.assertEqual(summary["rollout_loss_weight"], 0.0)
        self.assertEqual(summary["train_sequence_windows"], 0)
        self.assertEqual(summary["split_episodes"], {"train": 8, "validation": 1, "test": 1})
        self.assertEqual(loaded.training_config["epochs"], 3)
        self.assertEqual(len(loaded.train_losses), 3)
        self.assertEqual(len(loaded.validation_losses), 3)
        self.assertEqual(loaded.test_metrics, summary["test"])
        self.assertEqual(summary["best_epoch"], loaded.best_epoch)
        self.assertAlmostEqual(
            summary["best_validation_loss"], min(loaded.validation_losses)
        )
        self.assertTrue(np.isfinite(summary["validation"]["normalized_mse"]))
        self.assertTrue(np.isfinite(summary["test"]["normalized_mse"]))

    def test_run_training_supports_multistep_sequences(self):
        states, actions, next_states, episode_ids, step_ids = (
            make_sequence_dynamics(episodes=10, steps=4)
        )

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            dataset_path = directory_path / "transitions.npz"
            checkpoint_path = directory_path / "world_model.pt"
            np.savez_compressed(
                dataset_path,
                states=states,
                actions=actions,
                next_states=next_states,
                episode_ids=episode_ids,
                step_ids=step_ids,
            )

            summary = run_training(
                data_path=dataset_path,
                output_path=checkpoint_path,
                hidden_size=16,
                epochs=2,
                batch_size=8,
                learning_rate=1e-3,
                seed=7,
                rollout_horizon=2,
                rollout_loss_weight=0.5,
            )
            loaded = load_checkpoint(checkpoint_path)

        self.assertEqual(summary["rollout_horizon"], 2)
        self.assertEqual(summary["rollout_loss_weight"], 0.5)
        self.assertGreater(summary["train_sequence_windows"], 0)
        self.assertGreater(summary["validation_sequence_windows"], 0)
        self.assertEqual(loaded.training_config["rollout_horizon"], 2)
        self.assertEqual(loaded.training_config["rollout_loss_weight"], 0.5)
        self.assertGreater(summary["final_train_rollout_loss"], 0.0)

    def test_explicit_split_seed_keeps_episode_split_fixed_across_training_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            _save_sequence_dataset(data_path)
            checkpoints = []
            for seed in (3, 9):
                checkpoint = root / f"seed-{seed}.pt"
                run_training(
                    data_path=data_path,
                    output_path=checkpoint,
                    hidden_size=8,
                    epochs=1,
                    batch_size=32,
                    seed=seed,
                    split_seed=5,
                )
                checkpoints.append(load_checkpoint(checkpoint))

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                checkpoints[0].split_episode_ids[name],
                checkpoints[1].split_episode_ids[name],
            )
        self.assertEqual(checkpoints[0].training_config["split_seed"], 5)
        self.assertEqual(checkpoints[1].training_config["split_seed"], 5)

    def test_omitted_split_seed_preserves_seed_based_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "transitions.npz"
            _save_sequence_dataset(data_path)
            seed = 7
            omitted_path = root / "omitted.pt"
            explicit_path = root / "explicit.pt"
            omitted_summary = run_training(
                data_path=data_path,
                output_path=omitted_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=seed,
            )
            explicit_summary = run_training(
                data_path=data_path,
                output_path=explicit_path,
                hidden_size=8,
                epochs=1,
                batch_size=32,
                seed=seed,
                split_seed=seed,
            )
            checkpoints = (
                load_checkpoint(omitted_path),
                load_checkpoint(explicit_path),
            )

        for name in ("train", "validation", "test"):
            np.testing.assert_array_equal(
                checkpoints[0].split_episode_ids[name],
                checkpoints[1].split_episode_ids[name],
            )
        self.assertEqual(omitted_summary["split_seed"], seed)
        self.assertEqual(explicit_summary["split_seed"], seed)


if __name__ == "__main__":
    unittest.main()
