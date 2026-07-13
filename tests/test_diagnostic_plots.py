import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from world_model_lab.diagnostic_plots import (
    _physical_rollout_curve,
    plot_diagnostic_overview,
    plot_rollout_errors,
    plot_rollout_loss_components,
)


def make_summary(mean: float, count: int = 2) -> dict[str, float | int]:
    return {
        "count": count,
        "mean": mean,
        "median": mean,
        "p90": mean,
        "max": mean,
    }


def make_metrics() -> dict[str, object]:
    feature_slice = {
        "edges": [0.0, 1.0, 2.0],
        "bins": [
            {"count": 2, "position": make_summary(0.1)},
            {"count": 3, "position": make_summary(0.2, count=3)},
        ],
    }
    return {
        "coverage": {
            "x_edges": [0.0, 1.0, 2.0],
            "y_edges": [0.0, 1.0, 2.0],
            "train_counts": [[2, 1], [0, 3]],
            "test_counts": [[1, 0], [2, 1]],
        },
        "one_step": {
            "xy_grid": {
                "cells": [
                    [
                        {"count": 1, "position": make_summary(0.1, count=1)},
                        {"count": 0, "position": None},
                    ],
                    [
                        {"count": 2, "position": make_summary(0.2)},
                        {"count": 1, "position": make_summary(0.3, count=1)},
                    ],
                ]
            },
            "feature_slices": {
                "velocity": feature_slice,
                "steering": feature_slice,
                "acceleration": feature_slice,
            },
        },
        "rollout": {
            "protocol": {"horizons": [1, 2]},
            "horizons": {
                "1": {
                    "teacher_forcing": {
                        "position": make_summary(0.1),
                        "heading_degrees": make_summary(0.2),
                        "velocity": make_summary(0.3),
                    },
                    "free_rollout": {
                        "position": make_summary(0.1),
                        "heading_degrees": make_summary(0.2),
                        "velocity": make_summary(0.3),
                    },
                },
                "2": {
                    "teacher_forcing": {
                        "position": make_summary(0.15),
                        "heading_degrees": make_summary(0.25),
                        "velocity": make_summary(0.35),
                    },
                    "free_rollout": {
                        "position": make_summary(0.4),
                        "heading_degrees": make_summary(0.5),
                        "velocity": make_summary(0.6),
                    },
                },
            },
        },
    }


def make_dense_metrics() -> dict[str, object]:
    metrics = copy.deepcopy(make_metrics())
    metrics["schema_version"] = 2
    rollout = metrics["rollout"]
    rollout["protocol"]["max_horizon"] = 3
    rollout["step_curves"] = {
        "steps": [1, 2, 3],
        "aggregation": "episode_macro_mean",
        "teacher_forcing": {
            "physical": {
                "position": [0.1, 0.2, 0.3],
                "heading_degrees": [1.0, 2.0, 3.0],
                "velocity": [0.01, 0.02, 0.03],
            },
            "normalized_mse": {
                "x": [0.01, 0.02, 0.03],
                "y": [0.02, 0.03, 0.04],
                "heading": [0.03, 0.04, 0.05],
                "velocity": [0.04, 0.05, 0.06],
                "total": [0.025, 0.035, 0.045],
            },
        },
        "free_rollout": {
            "physical": {
                "position": [0.1, 0.4, 0.9],
                "heading_degrees": [1.0, 4.0, 9.0],
                "velocity": [0.01, 0.04, 0.09],
            },
            "normalized_mse": {
                "x": [0.01, 0.04, 0.09],
                "y": [0.02, 0.05, 0.10],
                "heading": [0.03, 0.06, 0.11],
                "velocity": [0.04, 0.07, 0.12],
                "total": [0.025, 0.055, 0.105],
            },
        },
    }
    return metrics


class DiagnosticPlotsTest(unittest.TestCase):
    def test_plot_diagnostic_overview_saves_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overview.png"

            returned = plot_diagnostic_overview(make_metrics(), output)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")

    def test_physical_rollout_curve_prefers_dense_steps_and_falls_back_to_v1(self):
        dense_rollout = make_dense_metrics()["rollout"]
        dense_steps, dense_values = _physical_rollout_curve(
            dense_rollout,
            "free_rollout",
            "position",
        )
        np.testing.assert_array_equal(dense_steps, [1, 2, 3])
        np.testing.assert_allclose(dense_values, [0.1, 0.4, 0.9])

        sparse_rollout = make_metrics()["rollout"]
        sparse_steps, sparse_values = _physical_rollout_curve(
            sparse_rollout,
            "free_rollout",
            "position",
        )
        np.testing.assert_array_equal(sparse_steps, [1, 2])
        np.testing.assert_allclose(sparse_values, [0.1, 0.4])

    def test_plot_rollout_errors_saves_v1_and_v2_png_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v1_output = root / "rollout-errors-v1.png"
            v2_output = root / "rollout-errors-v2.png"

            self.assertEqual(plot_rollout_errors(make_metrics(), v1_output), v1_output)
            self.assertEqual(
                plot_rollout_errors(make_dense_metrics(), v2_output),
                v2_output,
            )
            signatures = (v1_output.read_bytes()[:8], v2_output.read_bytes()[:8])

        self.assertEqual(signatures, (b"\x89PNG\r\n\x1a\n",) * 2)

    def test_plot_rollout_loss_components_draws_four_named_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rollout-loss-components.png"
            with patch.object(plt, "close"):
                returned = plot_rollout_loss_components(
                    make_dense_metrics(),
                    output,
                )
                figure = plt.gcf()
                titles = [axis.get_title() for axis in figure.axes]
                labels = [line.get_label() for line in figure.axes[0].lines]
                y_bottoms = [axis.get_ylim()[0] for axis in figure.axes]
            plt.close(figure)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")
        self.assertEqual(titles, ["x", "y", "heading", "velocity"])
        self.assertEqual(labels, ["Teacher forcing", "Free rollout"])
        self.assertEqual(y_bottoms, [0.0, 0.0, 0.0, 0.0])

    def test_schema_v2_rollout_plot_requires_step_curves(self):
        metrics = make_dense_metrics()
        del metrics["rollout"]["step_curves"]

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "schema version 2.*step_curves"):
                plot_rollout_errors(
                    metrics,
                    Path(directory) / "rollout-errors.png",
                )


if __name__ == "__main__":
    unittest.main()
