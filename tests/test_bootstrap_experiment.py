import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from world_model_lab import bootstrap_experiment
from world_model_lab.bootstrap_experiment import (
    build_bootstrap_comparison,
    plot_bootstrap_comparison,
    write_comparison_csv,
)


METRICS = (
    "position",
    "heading_degrees",
    "velocity",
    "normalized_total",
)


def make_ensemble_metrics(
    *,
    error: float,
    correlation: float | None,
    disagreement: float = 0.2,
):
    one_step = {
        name: {
            "ensemble_error": {"mean": error},
            "pearson_correlation": correlation,
        }
        for name in METRICS
    }
    rollout_metrics = {
        name: {
            "ensemble_error_mean": [error, error + 1.0],
            "disagreement_mean": [disagreement, disagreement * 2.0],
            "pearson_correlation": [correlation, correlation],
        }
        for name in METRICS
    }
    return {
        "schema_version": 1,
        "one_step": {"metrics": one_step},
        "rollout": {
            "steps": [1, 2],
            "metrics": rollout_metrics,
            "horizons": {
                str(horizon): {
                    "metrics": {
                        name: {
                            "ensemble_error_mean": error + horizon - 1,
                            "disagreement_mean": disagreement * horizon,
                            "pearson_correlation": correlation,
                        }
                        for name in METRICS
                    }
                }
                for horizon in (1, 2)
            },
        },
    }


class BootstrapExperimentTest(unittest.TestCase):
    def test_comparison_csv_has_exact_header_and_stable_row_order(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=0.1),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_comparison_csv(
                comparison,
                Path(directory) / "nested" / "comparison.csv",
            )
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                handle.seek(0)
                fieldnames = next(csv.reader(handle))

        self.assertEqual(
            fieldnames,
            [
                "evaluation",
                "horizon",
                "metric",
                "baseline_error",
                "bootstrap_error",
                "error_delta",
                "baseline_disagreement",
                "bootstrap_disagreement",
                "disagreement_delta",
                "baseline_correlation",
                "bootstrap_correlation",
                "correlation_delta",
            ],
        )
        self.assertEqual(len(rows), 12)
        self.assertEqual(
            [
                (row["evaluation"], row["horizon"], row["metric"])
                for row in rows
            ],
            [
                ("one_step", "", metric)
                for metric in METRICS
            ]
            + [
                ("rollout", str(horizon), metric)
                for horizon in (1, 2)
                for metric in METRICS
            ],
        )

    def test_comparison_csv_leaves_inapplicable_and_null_cells_empty(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=None),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = write_comparison_csv(
                comparison,
                Path(directory) / "comparison.csv",
            )
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        for row in rows[:4]:
            self.assertEqual(row["horizon"], "")
            self.assertEqual(row["baseline_disagreement"], "")
            self.assertEqual(row["bootstrap_disagreement"], "")
            self.assertEqual(row["disagreement_delta"], "")
        for row in rows:
            self.assertEqual(row["baseline_correlation"], "")
            self.assertEqual(row["correlation_delta"], "")
            self.assertNotIn("None", row.values())
            self.assertNotIn("nan", row.values())

    def test_comparison_plot_is_four_panels_with_null_correlation_gaps(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(error=2.0, correlation=None),
            make_ensemble_metrics(error=1.5, correlation=0.4),
            horizons=(1, 2),
        )
        real_subplots = bootstrap_experiment.plt.subplots
        captured = {}

        def capture_subplots(*args, **kwargs):
            figure, axes = real_subplots(*args, **kwargs)
            captured["figure"] = figure
            captured["axes"] = axes
            return figure, axes

        figure_numbers_before = set(bootstrap_experiment.plt.get_fignums())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "comparison.png"
            with patch.object(
                bootstrap_experiment.plt,
                "subplots",
                side_effect=capture_subplots,
            ) as subplots_spy:
                path = plot_bootstrap_comparison(comparison, output)
            contents = path.read_bytes()

        subplots_spy.assert_called_once_with(2, 2, figsize=(11, 8))
        self.assertEqual(captured["axes"].shape, (2, 2))
        self.assertEqual(len(captured["figure"].axes), 8)
        for correlation_axis in captured["figure"].axes[4:]:
            self.assertEqual(len(correlation_axis.lines), 2)
            self.assertTrue(
                np.isnan(correlation_axis.lines[0].get_ydata()).all()
            )
            np.testing.assert_allclose(
                correlation_axis.lines[1].get_ydata(),
                [0.4, 0.4],
            )
        self.assertEqual(contents[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(contents), 1000)
        self.assertEqual(
            set(bootstrap_experiment.plt.get_fignums()),
            figure_numbers_before,
        )

    def test_comparison_uses_bootstrap_minus_baseline_finite_deltas(self):
        comparison = build_bootstrap_comparison(
            make_ensemble_metrics(
                error=2.0,
                correlation=0.1,
                disagreement=0.2,
            ),
            make_ensemble_metrics(
                error=1.5,
                correlation=0.4,
                disagreement=0.3,
            ),
            horizons=(1, 2),
        )

        self.assertEqual(comparison["schema_version"], 1)
        self.assertEqual(
            comparison["delta_definition"],
            "bootstrap minus baseline",
        )
        self.assertEqual(comparison["horizons"], [1, 2])

        one_step = comparison["one_step"]["position"]
        self.assertEqual(one_step["baseline_error"], 2.0)
        self.assertEqual(one_step["bootstrap_error"], 1.5)
        self.assertEqual(one_step["error_delta"], -0.5)
        self.assertAlmostEqual(one_step["correlation_delta"], 0.3)

        rollout = comparison["rollout"]["2"]["position"]
        self.assertEqual(rollout["baseline_error"], 3.0)
        self.assertEqual(rollout["bootstrap_error"], 2.5)
        self.assertEqual(rollout["error_delta"], -0.5)
        self.assertAlmostEqual(rollout["disagreement_delta"], 0.2)
        self.assertAlmostEqual(rollout["correlation_delta"], 0.3)
        json.dumps(comparison, allow_nan=False)

    def test_comparison_keeps_null_correlation_delta_json_safe(self):
        cases = (
            (None, 0.4),
            (0.1, None),
        )
        for baseline_correlation, bootstrap_correlation in cases:
            with self.subTest(
                baseline=baseline_correlation,
                bootstrap=bootstrap_correlation,
            ):
                comparison = build_bootstrap_comparison(
                    make_ensemble_metrics(
                        error=2.0,
                        correlation=baseline_correlation,
                    ),
                    make_ensemble_metrics(
                        error=1.5,
                        correlation=bootstrap_correlation,
                    ),
                    horizons=(1, 2),
                )

                self.assertIsNone(
                    comparison["one_step"]["position"][
                        "correlation_delta"
                    ]
                )
                self.assertIsNone(
                    comparison["rollout"]["1"]["position"][
                        "correlation_delta"
                    ]
                )
                json.dumps(comparison, allow_nan=False)

    def test_comparison_requires_schema_version_one(self):
        for side in ("baseline", "bootstrap"):
            with self.subTest(side=side):
                baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
                bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
                target = baseline if side == "baseline" else bootstrap
                target["schema_version"] = 2

                with self.assertRaisesRegex(
                    ValueError,
                    rf"{side}\.schema_version",
                ):
                    build_bootstrap_comparison(
                        baseline,
                        bootstrap,
                        horizons=(1, 2),
                    )

    def test_comparison_requires_strictly_increasing_positive_unique_horizons(
        self,
    ):
        invalid_horizons = (
            (),
            (1, 1),
            (2, 1),
            (0, 1),
            (-1, 1),
            (True,),
            (1.5,),
        )
        for horizons in invalid_horizons:
            with self.subTest(horizons=horizons):
                with self.assertRaisesRegex(ValueError, "horizons"):
                    build_bootstrap_comparison(
                        make_ensemble_metrics(error=2.0, correlation=0.1),
                        make_ensemble_metrics(error=1.5, correlation=0.4),
                        horizons=horizons,
                    )

    def test_comparison_rejects_missing_metric_with_field_name(self):
        cases = (
            (
                "baseline.one_step.metrics.position",
                lambda baseline, bootstrap: baseline["one_step"]["metrics"].pop(
                    "position"
                ),
            ),
            (
                "bootstrap.rollout.horizons.1.metrics.position",
                lambda baseline, bootstrap: bootstrap["rollout"]["horizons"][
                    "1"
                ]["metrics"].pop("position"),
            ),
        )
        for expected_field, mutate in cases:
            with self.subTest(field=expected_field):
                baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
                bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
                mutate(baseline, bootstrap)

                with self.assertRaisesRegex(
                    ValueError,
                    expected_field.replace(".", r"\."),
                ):
                    build_bootstrap_comparison(
                        baseline,
                        bootstrap,
                        horizons=(1, 2),
                    )

    def test_comparison_rejects_non_finite_numeric_fields(self):
        cases = (
            (
                "baseline.one_step.metrics.position.ensemble_error.mean",
                lambda baseline, bootstrap: baseline["one_step"]["metrics"][
                    "position"
                ]["ensemble_error"].__setitem__("mean", float("nan")),
            ),
            (
                "bootstrap.one_step.metrics.position.pearson_correlation",
                lambda baseline, bootstrap: bootstrap["one_step"]["metrics"][
                    "position"
                ].__setitem__("pearson_correlation", float("inf")),
            ),
            (
                "baseline.rollout.horizons.1.metrics.position.ensemble_error_mean",
                lambda baseline, bootstrap: baseline["rollout"]["horizons"][
                    "1"
                ]["metrics"]["position"].__setitem__(
                    "ensemble_error_mean",
                    float("-inf"),
                ),
            ),
            (
                "bootstrap.rollout.horizons.1.metrics.position.disagreement_mean",
                lambda baseline, bootstrap: bootstrap["rollout"]["horizons"][
                    "1"
                ]["metrics"]["position"].__setitem__(
                    "disagreement_mean",
                    float("nan"),
                ),
            ),
            (
                "baseline.rollout.horizons.1.metrics.position.pearson_correlation",
                lambda baseline, bootstrap: baseline["rollout"]["horizons"][
                    "1"
                ]["metrics"]["position"].__setitem__(
                    "pearson_correlation",
                    float("inf"),
                ),
            ),
        )
        for expected_field, mutate in cases:
            with self.subTest(field=expected_field):
                baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
                bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
                mutate(baseline, bootstrap)

                with self.assertRaisesRegex(
                    ValueError,
                    expected_field.replace(".", r"\."),
                ):
                    build_bootstrap_comparison(
                        baseline,
                        bootstrap,
                        horizons=(1, 2),
                    )

    def test_comparison_rejects_non_finite_delta(self):
        baseline = make_ensemble_metrics(error=-1e308, correlation=0.1)
        bootstrap = make_ensemble_metrics(error=1e308, correlation=0.4)

        with self.assertRaisesRegex(
            ValueError,
            r"one_step\.position\.error_delta",
        ):
            build_bootstrap_comparison(
                baseline,
                bootstrap,
                horizons=(1, 2),
            )

    def test_comparison_rejects_missing_horizon_snapshot_with_field_name(self):
        baseline = make_ensemble_metrics(error=2.0, correlation=0.1)
        bootstrap = make_ensemble_metrics(error=1.5, correlation=0.4)
        del bootstrap["rollout"]["horizons"]["2"]

        with self.assertRaisesRegex(
            ValueError,
            r"bootstrap\.rollout\.horizons\.2",
        ):
            build_bootstrap_comparison(
                baseline,
                bootstrap,
                horizons=(1, 2),
            )
