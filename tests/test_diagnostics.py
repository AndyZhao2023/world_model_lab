import math
import json
import unittest

import numpy as np
import torch

from world_model_lab.dataset import Normalizer
from world_model_lab.diagnostics import (
    build_diagnostic_metrics,
    build_feature_slice,
    build_xy_grid,
    compute_normalized_squared_errors,
    compute_state_errors,
    linear_bin_edges,
    select_rollout_windows,
    summarize_values,
)
from world_model_lab.model import WorldModelMLP
from world_model_lab.train_world_model import LoadedWorldModel


def make_state_sequence(
    initial_state: np.ndarray,
    delta: np.ndarray,
    steps: int,
) -> np.ndarray:
    states = [np.asarray(initial_state, dtype=np.float64)]
    for _ in range(steps):
        states.append(states[-1] + delta)
    return np.asarray(states)


def make_error_components(values: list[float]) -> dict[str, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "position": array,
        "heading_degrees": array + 10.0,
        "velocity": array + 20.0,
    }


def make_constant_delta_world_model(
    delta: np.ndarray,
    *,
    test_episode_ids: np.ndarray,
    target_std: np.ndarray | None = None,
) -> LoadedWorldModel:
    target_std_array = (
        np.ones(4, dtype=np.float64)
        if target_std is None
        else np.asarray(target_std, dtype=np.float64)
    )
    model = WorldModelMLP(hidden_size=4)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(
            torch.as_tensor(delta / target_std_array, dtype=torch.float32)
        )
    model.eval()
    return LoadedWorldModel(
        model=model,
        input_normalizer=Normalizer(mean=np.zeros(7), std=np.ones(7)),
        target_normalizer=Normalizer(
            mean=np.zeros(4),
            std=target_std_array,
        ),
        split_episode_ids={
            "train": np.asarray([0]),
            "validation": np.asarray([99]),
            "test": np.asarray(test_episode_ids),
        },
        training_config={},
        train_losses=[],
        validation_losses=[],
        best_epoch=0,
        test_metrics={},
    )


def arrays_from_episodes(
    episodes: dict[int, np.ndarray],
) -> dict[str, np.ndarray]:
    states = []
    actions = []
    next_states = []
    episode_ids = []
    step_ids = []
    for episode_id, sequence in episodes.items():
        transition_count = sequence.shape[0] - 1
        states.append(sequence[:-1])
        actions.append(np.zeros((transition_count, 2), dtype=np.float64))
        next_states.append(sequence[1:])
        episode_ids.extend([episode_id] * transition_count)
        step_ids.extend(range(transition_count))
    return {
        "states": np.vstack(states),
        "actions": np.vstack(actions),
        "next_states": np.vstack(next_states),
        "episode_ids": np.asarray(episode_ids),
        "step_ids": np.asarray(step_ids),
    }


class DiagnosticsTest(unittest.TestCase):
    def test_public_normalized_squared_errors_matches_component_contract(self):
        result = compute_normalized_squared_errors(
            np.asarray([[2.0, 4.0, 0.2, 3.0]]),
            np.asarray([[1.0, 2.0, 0.1, 1.0]]),
            np.asarray([1.0, 2.0, 0.1, 2.0]),
        )
        np.testing.assert_allclose(
            [
                result[name][0]
                for name in ("x", "y", "heading", "velocity", "total")
            ],
            [1.0, 1.0, 1.0, 1.0, 1.0],
        )

    def test_state_errors_use_euclidean_position_and_wrapped_heading(self):
        true = np.asarray([[0.0, 0.0, math.radians(-179.0), 1.0]])
        predicted = np.asarray([[3.0, 4.0, math.radians(179.0), 1.25]])

        errors = compute_state_errors(predicted, true)

        np.testing.assert_allclose(errors["position"], [5.0])
        np.testing.assert_allclose(
            errors["heading_degrees"],
            [2.0],
            atol=1e-10,
        )
        np.testing.assert_allclose(errors["velocity"], [0.25])

    def test_normalized_squared_errors_use_target_std_and_wrapped_heading(self):
        true = np.asarray([[0.0, 0.0, math.radians(-179.0), 1.0]])
        predicted = np.asarray([[3.0, 4.0, math.radians(179.0), 1.25]])
        target_std = np.asarray([2.0, 4.0, math.radians(1.0), 0.5])

        errors = compute_normalized_squared_errors(
            predicted,
            true,
            target_std,
        )

        np.testing.assert_allclose(errors["x"], [2.25])
        np.testing.assert_allclose(errors["y"], [1.0])
        np.testing.assert_allclose(errors["heading"], [4.0], atol=1e-10)
        np.testing.assert_allclose(errors["velocity"], [0.25])
        np.testing.assert_allclose(errors["total"], [1.875], atol=1e-10)

    def test_normalized_squared_errors_reject_invalid_target_std(self):
        states = np.zeros((1, 4), dtype=np.float64)
        invalid_values = (
            np.ones(3),
            np.asarray([1.0, 1.0, 0.0, 1.0]),
            np.asarray([1.0, 1.0, np.inf, 1.0]),
        )

        for target_std in invalid_values:
            with self.subTest(target_std=target_std):
                with self.assertRaisesRegex(
                    ValueError,
                    "target_std must have shape.*finite positive",
                ):
                    compute_normalized_squared_errors(
                        states,
                        states,
                        target_std,
                    )

    def test_summary_reports_distribution_statistics(self):
        summary = summarize_values(np.asarray([1.0, 2.0, 3.0, 10.0]))

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean"], 4.0)
        self.assertEqual(summary["median"], 2.5)
        self.assertAlmostEqual(summary["p90"], 7.9)
        self.assertEqual(summary["max"], 10.0)

    def test_rollout_windows_are_evenly_spaced_and_skip_short_episodes(self):
        episode_3 = make_state_sequence(
            np.asarray([0.0, 0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.1]),
            steps=7,
        )
        episode_4 = make_state_sequence(
            np.asarray([10.0, 0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.1]),
            steps=2,
        )
        states = np.vstack((episode_3[:-1], episode_4[:-1]))
        next_states = np.vstack((episode_3[1:], episode_4[1:]))
        actions = np.zeros((states.shape[0], 2), dtype=np.float64)
        episode_ids = np.asarray([3] * 7 + [4] * 2)
        step_ids = np.asarray(list(range(7)) + list(range(2)))

        selection = select_rollout_windows(
            states=states,
            actions=actions,
            next_states=next_states,
            episode_ids=episode_ids,
            step_ids=step_ids,
            selected_episode_ids=np.asarray([3, 4]),
            max_horizon=3,
            windows_per_episode=3,
        )

        self.assertEqual(
            [
                (window.episode_id, window.start_step)
                for window in selection.windows
            ],
            [(3, 0), (3, 2), (3, 4)],
        )
        np.testing.assert_array_equal(selection.eligible_episode_ids, [3])
        np.testing.assert_array_equal(selection.skipped_episode_ids, [4])
        for window in selection.windows:
            self.assertEqual(window.true_states.shape, (4, 4))
            self.assertEqual(window.actions.shape, (3, 2))

    def test_feature_slice_masks_sparse_bins_and_includes_maximum(self):
        values = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0])
        errors = make_error_components([1.0, 2.0, 3.0, 4.0, 5.0])

        result = build_feature_slice(
            values,
            errors,
            edges=np.asarray([0.0, 1.0, 2.0]),
            min_bin_count=3,
        )

        self.assertEqual(result["edges"], [0.0, 1.0, 2.0])
        self.assertEqual([cell["count"] for cell in result["bins"]], [2, 3])
        self.assertIsNone(result["bins"][0]["position"])
        self.assertEqual(result["bins"][1]["position"]["count"], 3)
        self.assertEqual(result["bins"][1]["position"]["max"], 5.0)

    def test_xy_grid_keeps_counts_when_error_metrics_are_masked(self):
        xy = np.asarray(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [1.5, 0.5],
                [2.0, 2.0],
            ]
        )
        errors = make_error_components([1.0, 2.0, 3.0, 4.0])

        result = build_xy_grid(
            xy,
            errors,
            x_edges=np.asarray([0.0, 1.0, 2.0]),
            y_edges=np.asarray([0.0, 1.0, 2.0]),
            min_bin_count=2,
        )

        self.assertEqual(result["cells"][0][0]["count"], 2)
        self.assertEqual(result["cells"][0][1]["count"], 1)
        self.assertEqual(result["cells"][1][1]["count"], 1)
        self.assertEqual(result["cells"][0][0]["position"]["mean"], 1.5)
        self.assertIsNone(result["cells"][0][1]["position"])

    def test_linear_bin_edges_expand_a_constant_range(self):
        edges = linear_bin_edges(np.asarray([0.5, 0.5, 0.5]), bin_count=2)

        self.assertEqual(edges.shape, (3,))
        self.assertLess(edges[0], 0.5)
        self.assertGreater(edges[-1], 0.5)

    def test_free_rollout_exposes_compounding_error(self):
        train_episode = make_state_sequence(
            np.asarray([-2.0, 0.0, 0.0, 0.0]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            steps=3,
        )
        test_episode = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
                [6.0, 0.0, 0.0, 0.0],
            ]
        )
        arrays = arrays_from_episodes({0: train_episode, 1: test_episode})
        world_model = make_constant_delta_world_model(
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            test_episode_ids=np.asarray([1]),
            target_std=np.asarray([2.0, 1.0, 1.0, 1.0]),
        )

        metrics = build_diagnostic_metrics(
            world_model,
            arrays=arrays,
            split_episode_ids=world_model.split_episode_ids,
            horizons=(1, 2, 3),
            windows_per_episode=1,
            xy_bins=2,
            feature_bins=2,
            min_bin_count=1,
        )

        horizon_1 = metrics["rollout"]["horizons"]["1"]
        horizon_3 = metrics["rollout"]["horizons"]["3"]
        self.assertAlmostEqual(
            horizon_1["teacher_forcing"]["position"]["mean"],
            horizon_1["free_rollout"]["position"]["mean"],
        )
        self.assertEqual(
            horizon_3["teacher_forcing"]["position"]["mean"],
            2.0,
        )
        self.assertEqual(
            horizon_3["free_rollout"]["position"]["mean"],
            3.0,
        )
        self.assertEqual(metrics["schema_version"], 2)
        curves = metrics["rollout"]["step_curves"]
        self.assertEqual(curves["steps"], [1, 2, 3])
        self.assertEqual(curves["aggregation"], "episode_macro_mean")
        np.testing.assert_allclose(
            curves["teacher_forcing"]["physical"]["position"],
            [0.0, 1.0, 2.0],
        )
        np.testing.assert_allclose(
            curves["free_rollout"]["physical"]["position"],
            [0.0, 1.0, 3.0],
        )
        np.testing.assert_allclose(
            curves["teacher_forcing"]["normalized_mse"]["x"],
            [0.0, 0.25, 1.0],
        )
        np.testing.assert_allclose(
            curves["free_rollout"]["normalized_mse"]["x"],
            [0.0, 0.25, 2.25],
        )
        np.testing.assert_allclose(
            curves["teacher_forcing"]["normalized_mse"]["total"],
            [0.0, 0.0625, 0.25],
        )
        np.testing.assert_allclose(
            curves["free_rollout"]["normalized_mse"]["total"],
            [0.0, 0.0625, 0.5625],
        )
        for horizon in (1, 2, 3):
            sparse = metrics["rollout"]["horizons"][str(horizon)]
            for mode_name in ("teacher_forcing", "free_rollout"):
                self.assertAlmostEqual(
                    sparse[mode_name]["position"]["mean"],
                    curves[mode_name]["physical"]["position"][horizon - 1],
                )
        for mode_name in ("teacher_forcing", "free_rollout"):
            for group in ("physical", "normalized_mse"):
                for values in curves[mode_name][group].values():
                    self.assertEqual(len(values), 3)
                    self.assertTrue(np.all(np.isfinite(values)))
        json.dumps(metrics, allow_nan=False)

    def test_rollout_metrics_weight_episodes_equally(self):
        train_episode = make_state_sequence(
            np.asarray([-2.0, 0.0, 0.0, 0.0]),
            np.zeros(4),
            steps=1,
        )
        high_error_episode = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ]
        )
        zero_error_episode = make_state_sequence(
            np.asarray([10.0, 0.0, 0.0, 0.0]),
            np.zeros(4),
            steps=3,
        )
        arrays = arrays_from_episodes(
            {0: train_episode, 1: high_error_episode, 2: zero_error_episode}
        )
        world_model = make_constant_delta_world_model(
            np.zeros(4),
            test_episode_ids=np.asarray([1, 2]),
        )

        metrics = build_diagnostic_metrics(
            world_model,
            arrays=arrays,
            split_episode_ids=world_model.split_episode_ids,
            horizons=(1,),
            windows_per_episode=3,
            xy_bins=2,
            feature_bins=2,
            min_bin_count=1,
        )

        summary = metrics["rollout"]["horizons"]["1"]["free_rollout"]
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["windows"], 4)
        self.assertEqual(summary["position"]["mean"], 1.0)

        curves = metrics["rollout"]["step_curves"]
        self.assertEqual(
            curves["free_rollout"]["physical"]["position"],
            [1.0],
        )
        self.assertEqual(
            curves["free_rollout"]["normalized_mse"]["x"],
            [2.0],
        )
        self.assertEqual(
            curves["free_rollout"]["normalized_mse"]["total"],
            [0.5],
        )

    def test_build_diagnostic_metrics_rejects_invalid_checkpoint_target_std(self):
        train_episode = make_state_sequence(
            np.asarray([-1.0, 0.0, 0.0, 0.0]),
            np.zeros(4),
            steps=1,
        )
        test_episode = make_state_sequence(
            np.asarray([0.0, 0.0, 0.0, 0.0]),
            np.zeros(4),
            steps=1,
        )
        arrays = arrays_from_episodes({0: train_episode, 1: test_episode})
        world_model = make_constant_delta_world_model(
            np.zeros(4),
            test_episode_ids=np.asarray([1]),
        )
        world_model.target_normalizer = Normalizer(
            mean=np.zeros(4),
            std=np.asarray([1.0, 1.0, 0.0, 1.0]),
        )

        with self.assertRaisesRegex(
            ValueError,
            "target_std must have shape.*finite positive",
        ):
            build_diagnostic_metrics(
                world_model,
                arrays=arrays,
                split_episode_ids=world_model.split_episode_ids,
                horizons=(1,),
                windows_per_episode=1,
                xy_bins=2,
                feature_bins=2,
                min_bin_count=1,
            )


if __name__ == "__main__":
    unittest.main()
