import unittest

import numpy as np

from tests.test_ensemble import make_member
from world_model_lab.diagnose_ensemble import (
    build_calibration_bins,
    evaluate_one_step_calibration,
    pearson_correlation,
)
from world_model_lab.ensemble import build_ensemble


class DiagnoseEnsembleTest(unittest.TestCase):
    def test_pearson_returns_none_for_constant_values(self):
        self.assertIsNone(
            pearson_correlation(np.ones(3), np.asarray([1.0, 2.0, 3.0]))
        )

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


if __name__ == "__main__":
    unittest.main()
