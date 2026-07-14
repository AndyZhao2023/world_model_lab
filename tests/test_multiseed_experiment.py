import tempfile
import unittest
from pathlib import Path

import numpy as np

from world_model_lab.multiseed_experiment import (
    build_experiment_summary,
    plot_multiseed_comparison,
    write_summary_csv,
)


def make_metrics(position, heading, velocity, total):
    return {
        "schema_version": 2,
        "rollout": {
            "curves": {
                "steps": [1, 2],
                "free_rollout": {
                    "physical": {
                        "position": position,
                        "heading_degrees": heading,
                        "velocity": velocity,
                    },
                    "normalized_mse": {"total": total},
                },
            }
        },
    }


def make_records():
    return [
        {
            "seed": 0,
            "h1_metrics": make_metrics(
                [1.0, 2.0],
                [1.0, 2.0],
                [0.4, 0.6],
                [0.8, 1.0],
            ),
            "h10_metrics": make_metrics(
                [0.8, 2.5],
                [0.5, 2.5],
                [0.3, 0.5],
                [0.7, 0.9],
            ),
            "h1_metrics_path": "runs/seed_0/h1/diagnostics/metrics.json",
            "h10_metrics_path": "runs/seed_0/h10/diagnostics/metrics.json",
        },
        {
            "seed": 1,
            "h1_metrics": make_metrics(
                [1.2, 2.2],
                [1.0, 2.0],
                [0.5, 0.7],
                [0.9, 1.1],
            ),
            "h10_metrics": make_metrics(
                [1.0, 2.3],
                [0.5, 1.5],
                [0.4, 0.6],
                [0.8, 1.0],
            ),
            "h1_metrics_path": "runs/seed_1/h1/diagnostics/metrics.json",
            "h10_metrics_path": "runs/seed_1/h10/diagnostics/metrics.json",
        },
    ]


class MultiseedExperimentTest(unittest.TestCase):
    def test_build_experiment_summary_aggregates_paired_curves(self):
        summary = build_experiment_summary(
            make_records(),
            snapshot_horizons=(1, 2),
        )

        position = summary["metrics"]["position"]
        np.testing.assert_allclose(position["h1"]["mean"], [1.1, 2.1])
        np.testing.assert_allclose(
            position["h1"]["std"],
            [np.sqrt(0.02), np.sqrt(0.02)],
        )
        np.testing.assert_allclose(position["paired_delta"]["mean"], [-0.2, 0.3])
        np.testing.assert_allclose(
            position["paired_delta"]["std"],
            [0.0, np.sqrt(0.08)],
            atol=1e-15,
        )
        self.assertEqual(position["paired_delta"]["improved_seed_count"][1], 0)
        self.assertEqual(position["paired_delta"]["worse_seed_count"][1], 2)
        self.assertEqual(
            summary["per_seed"]["0"]["first_heading_regression_step"],
            2,
        )
        self.assertIsNone(
            summary["per_seed"]["1"]["first_heading_regression_step"]
        )

    def test_build_experiment_summary_accepts_schema_v2_step_curves(self):
        records = make_records()
        for record in records:
            for model_name in ("h1", "h10"):
                rollout = record[f"{model_name}_metrics"]["rollout"]
                rollout["step_curves"] = rollout.pop("curves")

        summary = build_experiment_summary(records, snapshot_horizons=(1, 2))

        np.testing.assert_allclose(
            summary["metrics"]["position"]["paired_delta"]["mean"],
            [-0.2, 0.3],
        )

    def test_build_experiment_summary_rejects_duplicate_or_invalid_seeds(self):
        records = make_records()
        records[1]["seed"] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            build_experiment_summary(records, snapshot_horizons=(1, 2))

        records = make_records()
        records[0]["seed"] = -1
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_experiment_summary(records, snapshot_horizons=(1, 2))

        with self.assertRaisesRegex(ValueError, "at least two"):
            build_experiment_summary(records[:1], snapshot_horizons=(1, 2))

    def test_build_experiment_summary_rejects_inconsistent_steps(self):
        records = make_records()
        records[1]["h10_metrics"]["rollout"]["curves"]["steps"] = [1, 3]

        with self.assertRaisesRegex(ValueError, "identical steps"):
            build_experiment_summary(records, snapshot_horizons=(1, 2))

    def test_summary_outputs_csv_and_png(self):
        summary = build_experiment_summary(
            make_records(),
            snapshot_horizons=(1, 2),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "summary.csv"
            plot_path = root / "comparison.png"

            self.assertEqual(write_summary_csv(summary, csv_path), csv_path)
            self.assertEqual(len(csv_path.read_text().splitlines()), 17)
            self.assertEqual(
                plot_multiseed_comparison(summary, plot_path),
                plot_path,
            )
            self.assertEqual(plot_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
