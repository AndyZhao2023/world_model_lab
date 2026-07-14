import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from tests.test_ensemble import make_member
from world_model_lab.diagnose_ensemble import (
    build_calibration_bins,
    evaluate_one_step_calibration,
    evaluate_ensemble_rollouts,
    pearson_correlation,
    plot_one_step_calibration,
    plot_rollout_uncertainty,
)
from world_model_lab.diagnostics import RolloutWindow
from world_model_lab.ensemble import build_ensemble


PLOT_METRIC_NAMES = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)


def make_one_step_metrics():
    return {
        "metrics": {
            name: {
                "calibration_bins": [
                    {
                        "count": 2,
                        "disagreement_mean": 0.25,
                        "error_mean": 0.5,
                    },
                    {
                        "count": 2,
                        "disagreement_mean": 0.75,
                        "error_mean": 1.5,
                    },
                ]
            }
            for name in PLOT_METRIC_NAMES
        }
    }


def make_rollout_metrics():
    return {
        "steps": [1, 2],
        "metrics": {
            name: {
                "ensemble_error_mean": [0.5, 1.0],
                "mean_member_error_mean": [0.75, 1.5],
                "min_member_error_mean": [0.25, 0.5],
                "max_member_error_mean": [1.0, 2.0],
                "disagreement_mean": [0.2, 0.4],
                "pearson_correlation": [0.1, 0.2],
            }
            for name in PLOT_METRIC_NAMES
        },
    }


class DiagnoseEnsembleTest(unittest.TestCase):
    def test_pearson_returns_none_for_constant_values(self):
        self.assertIsNone(
            pearson_correlation(np.ones(3), np.asarray([1.0, 2.0, 3.0]))
        )

    def test_pearson_returns_none_when_finite_subnormals_produce_nan(self):
        subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
        values = np.asarray([0.0, subnormal])

        self.assertTrue(np.all(np.isfinite(values)))
        self.assertIsNone(pearson_correlation(values, values))

    def test_pearson_rejects_invalid_shape_or_non_finite_values(self):
        for left, right, message in (
            (np.ones((1, 3)), np.ones((1, 3)), "matching non-empty vectors"),
            (np.ones(2), np.ones(3), "matching non-empty vectors"),
            (np.asarray([]), np.asarray([]), "matching non-empty vectors"),
            (
                np.asarray([1.0, np.nan]),
                np.asarray([1.0, 2.0]),
                "finite",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    pearson_correlation(left, right)

    def test_calibration_bins_are_equal_count_and_sorted_by_disagreement(self):
        bins = build_calibration_bins(
            np.asarray([4.0, 1.0, 3.0, 2.0]),
            np.asarray([40.0, 10.0, 30.0, 20.0]),
            bin_count=2,
        )
        self.assertEqual([item["count"] for item in bins], [2, 2])
        self.assertEqual(
            [item["disagreement_mean"] for item in bins],
            [1.5, 3.5],
        )
        self.assertEqual([item["error_mean"] for item in bins], [15.0, 35.0])

    def test_calibration_bins_reject_invalid_inputs(self):
        for disagreement, errors, bin_count, message in (
            (np.ones((1, 2)), np.ones((1, 2)), 1, "matching non-empty vectors"),
            (np.ones(2), np.ones(3), 1, "matching non-empty vectors"),
            (np.asarray([]), np.asarray([]), 1, "matching non-empty vectors"),
            (np.asarray([1.0, np.inf]), np.ones(2), 1, "finite"),
            (np.ones(2), np.asarray([1.0, np.nan]), 1, "finite"),
            (np.ones(2), np.ones(2), 0, "positive"),
            (np.ones(2), np.ones(2), -1, "positive"),
        ):
            with self.subTest(message=message, bin_count=bin_count):
                with self.assertRaisesRegex(ValueError, message):
                    build_calibration_bins(
                        disagreement,
                        errors,
                        bin_count=bin_count,
                    )

    def test_calibration_bins_cap_count_to_sample_count(self):
        bins = build_calibration_bins(
            np.asarray([3.0, 1.0, 2.0]),
            np.asarray([30.0, 10.0, 20.0]),
            bin_count=5,
        )

        self.assertEqual([item["count"] for item in bins], [1, 1, 1])
        self.assertEqual(
            [item["disagreement_mean"] for item in bins],
            [1.0, 2.0, 3.0],
        )

    def test_one_step_reports_ensemble_gain_and_calibration(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([0.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([2.0, 0.0, 0.0, 0.0])),
            )
        )
        result = evaluate_one_step_calibration(
            ensemble,
            states=np.zeros((2, 4)),
            actions=np.zeros((2, 2)),
            true_next_states=np.asarray(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0, 0.0],
                ]
            ),
            calibration_bins=2,
        )

        position = result["metrics"]["position"]
        self.assertEqual(result["samples"], 2)
        self.assertEqual(position["ensemble_error"]["mean"], 1.0)
        self.assertEqual(position["mean_member_error"]["mean"], 1.5)
        self.assertEqual(position["ensemble_gain_mean"], 0.5)
        self.assertIsNone(position["pearson_correlation"])

    def test_rollout_metrics_weight_episodes_equally_and_keep_member_divergence(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        windows = (
            RolloutWindow(
                10,
                0,
                np.asarray(
                    [[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((2, 2)),
            ),
            RolloutWindow(
                10,
                1,
                np.asarray(
                    [[0, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((2, 2)),
            ),
            RolloutWindow(
                20,
                0,
                np.asarray(
                    [[0, 0, 0, 0], [4, 0, 0, 0], [8, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((2, 2)),
            ),
        )

        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=windows,
            eligible_episode_ids=np.asarray([10, 20]),
        )

        self.assertEqual(result["steps"], [1, 2])
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(result["windows"], 3)
        self.assertEqual(
            result["metrics"]["position"]["ensemble_error_mean"],
            [1.0, 2.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["mean_member_error_mean"],
            [1.5, 3.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["min_member_error_mean"],
            [1.0, 2.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["max_member_error_mean"],
            [2.0, 4.0],
        )
        self.assertEqual(
            result["metrics"]["position"]["disagreement_mean"],
            [1.0, 2.0],
        )

    def test_rollout_metrics_keep_heading_circular_across_wrap_boundary(self):
        ensemble = build_ensemble(
            (
                make_member(
                    0,
                    np.asarray([0.0, 0.0, np.deg2rad(179.0), 0.0]),
                ),
                make_member(
                    1,
                    np.asarray([0.0, 0.0, np.deg2rad(-179.0), 0.0]),
                ),
            )
        )
        window = RolloutWindow(
            10,
            0,
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, np.pi, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            np.zeros((2, 2)),
        )

        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=(window,),
            eligible_episode_ids=np.asarray([10]),
        )

        heading = result["metrics"]["heading_degrees"]
        np.testing.assert_allclose(
            heading["ensemble_error_mean"],
            [0.0, 0.0],
            atol=1e-5,
        )
        np.testing.assert_allclose(
            heading["disagreement_mean"],
            [1.0, 2.0],
            atol=1e-5,
        )

    def test_rollout_metrics_use_json_null_for_constant_correlations(self):
        ensemble = build_ensemble(
            (
                make_member(0, np.asarray([1.0, 0.0, 0.0, 0.0])),
                make_member(1, np.asarray([3.0, 0.0, 0.0, 0.0])),
            )
        )
        windows = tuple(
            RolloutWindow(
                episode_id,
                0,
                np.asarray(
                    [[0, 0, 0, 0], [target, 0, 0, 0]],
                    dtype=float,
                ),
                np.zeros((1, 2)),
            )
            for episode_id, target in ((10, 2.0), (20, 4.0))
        )

        result = evaluate_ensemble_rollouts(
            ensemble,
            windows=windows,
            eligible_episode_ids=np.asarray([10, 20]),
        )

        for metric in result["metrics"].values():
            self.assertEqual(metric["pearson_correlation"], [None])

    def test_ensemble_plots_save_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = plot_one_step_calibration(
                make_one_step_metrics(),
                root / "calibration.png",
            )
            rollout_path = plot_rollout_uncertainty(
                make_rollout_metrics(),
                root / "rollout.png",
            )

            self.assertGreater(calibration_path.stat().st_size, 0)
            self.assertGreater(rollout_path.stat().st_size, 0)

    def test_ensemble_plots_label_all_metric_panels_and_secondary_axes(self):
        labels = (
            ("Position", "error (m)", "disagreement (m)"),
            ("Heading", "error (degrees)", "disagreement (degrees)"),
            ("Velocity", "error (m/s)", "disagreement (m/s)"),
            (
                "Normalized total",
                "normalized MSE",
                "normalized RMS disagreement",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_figure, calibration_axes = plt.subplots(2, 2)
            with patch.object(
                plt,
                "subplots",
                return_value=(calibration_figure, calibration_axes),
            ):
                plot_one_step_calibration(
                    make_one_step_metrics(),
                    root / "calibration.png",
                )

            rollout_figure, rollout_axes = plt.subplots(2, 2)
            with patch.object(
                plt,
                "subplots",
                return_value=(rollout_figure, rollout_axes),
            ):
                plot_rollout_uncertainty(
                    make_rollout_metrics(),
                    root / "rollout.png",
                )

        for axis, (title, error_label, disagreement_label) in zip(
            calibration_axes.flat,
            labels,
        ):
            self.assertEqual(axis.get_title(), title)
            self.assertEqual(axis.get_xlabel(), disagreement_label)
            self.assertEqual(axis.get_ylabel(), error_label)
        for axis, (title, error_label, _) in zip(rollout_axes.flat, labels):
            self.assertEqual(axis.get_title(), title)
            self.assertEqual(axis.get_xlabel(), "Rollout step")
            self.assertEqual(axis.get_ylabel(), error_label)
        self.assertEqual(
            [axis.get_ylabel() for axis in rollout_figure.axes[4:]],
            [item[2] for item in labels],
        )

    def test_ensemble_plots_reject_missing_fields_by_name(self):
        one_step = make_one_step_metrics()
        del one_step["metrics"]["position"]["calibration_bins"]
        rollout = make_rollout_metrics()
        del rollout["metrics"]["velocity"]["disagreement_mean"]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(
                ValueError,
                "position.*calibration_bins",
            ):
                plot_one_step_calibration(one_step, root / "calibration.png")
            with self.assertRaisesRegex(
                ValueError,
                "velocity.*disagreement_mean",
            ):
                plot_rollout_uncertainty(rollout, root / "rollout.png")


if __name__ == "__main__":
    unittest.main()
