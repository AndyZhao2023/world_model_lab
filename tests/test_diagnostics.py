import math
import unittest

import numpy as np

from world_model_lab.diagnostics import (
    build_feature_slice,
    build_xy_grid,
    compute_state_errors,
    linear_bin_edges,
    select_rollout_windows,
    summarize_values,
)


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


class DiagnosticsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
