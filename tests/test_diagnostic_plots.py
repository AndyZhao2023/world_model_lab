import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from world_model_lab.diagnostic_plots import (
    plot_diagnostic_overview,
    plot_rollout_errors,
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


class DiagnosticPlotsTest(unittest.TestCase):
    def test_plot_diagnostic_overview_saves_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "overview.png"

            returned = plot_diagnostic_overview(make_metrics(), output)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")

    def test_plot_rollout_errors_saves_png(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rollout-errors.png"

            returned = plot_rollout_errors(make_metrics(), output)
            signature = output.read_bytes()[:8]

        self.assertEqual(returned, output)
        self.assertEqual(signature, b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
