from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from tests.test_visual_windows import make_visual_dataset
from world_model_lab.vjepa_probe import (
    FrozenVJEPAEncoder,
    build_probe_clip_batch,
    fit_linear_state_probe,
    mean_target_predictions,
    pool_vjepa_tokens,
    representation_probe_gates,
    select_evenly_spaced_positions,
    state_probe_metrics,
    state_to_probe_targets,
)
from world_model_lab.visual_windows import build_visual_window_index


class ProbeClipBatchTest(unittest.TestCase):
    def setUp(self):
        self.visual = make_visual_dataset((5, 4, 2))
        self.index = build_visual_window_index(
            self.visual,
            np.asarray([10, 11, 12], dtype=np.int64),
        )

    def test_recorded_clips_follow_exact_window_frame_and_state_offsets(self):
        positions = np.asarray([0, 2], dtype=np.int64)

        batch = build_probe_clip_batch(
            self.visual,
            self.index,
            positions,
            order="recorded",
        )

        self.assertEqual(batch.frames.shape, (2, 4, 64, 64, 3))
        self.assertEqual(batch.frames.dtype, np.uint8)
        self.assertEqual(batch.states.shape, (2, 4))
        self.assertEqual(batch.states.dtype, np.float64)
        np.testing.assert_array_equal(
            batch.frames[0],
            self.visual["frames"][0:4],
        )
        second_start = int(self.visual["frame_offsets"][1])
        np.testing.assert_array_equal(
            batch.frames[1],
            self.visual["frames"][second_start : second_start + 4],
        )
        np.testing.assert_array_equal(
            batch.states,
            self.visual["states"][[3, second_start + 3]],
        )
        np.testing.assert_array_equal(batch.episode_ids, [10, 11])
        np.testing.assert_array_equal(batch.step_ids, [3, 3])
        with self.assertRaises(ValueError):
            batch.frames[0, 0, 0, 0, 0] = 1

    def test_temporal_ablations_change_frames_but_keep_recorded_targets(self):
        position = np.asarray([0], dtype=np.int64)
        recorded = build_probe_clip_batch(
            self.visual,
            self.index,
            position,
            order="recorded",
        )
        reversed_batch = build_probe_clip_batch(
            self.visual,
            self.index,
            position,
            order="reversed",
        )
        repeated = build_probe_clip_batch(
            self.visual,
            self.index,
            position,
            order="repeat_last",
        )

        np.testing.assert_array_equal(
            reversed_batch.frames[0],
            recorded.frames[0, ::-1],
        )
        np.testing.assert_array_equal(
            repeated.frames[0],
            np.repeat(recorded.frames[0, -1:], 4, axis=0),
        )
        np.testing.assert_array_equal(reversed_batch.states, recorded.states)
        np.testing.assert_array_equal(repeated.states, recorded.states)

    def test_invalid_positions_and_orders_are_rejected(self):
        invalid_positions = (
            np.asarray([], dtype=np.int64),
            np.asarray([[0]], dtype=np.int64),
            np.asarray([0.0]),
            np.asarray([True]),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([-1], dtype=np.int64),
            np.asarray([self.index.count], dtype=np.int64),
        )
        for positions in invalid_positions:
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    build_probe_clip_batch(
                        self.visual,
                        self.index,
                        positions,
                        order="recorded",
                    )
        with self.assertRaises(ValueError):
            build_probe_clip_batch(
                self.visual,
                self.index,
                np.asarray([0], dtype=np.int64),
                order="shuffled",
            )


class ProbeSelectionTest(unittest.TestCase):
    def test_even_spacing_is_deterministic_unique_and_includes_boundaries(self):
        first = select_evenly_spaced_positions(10, limit=4)
        second = select_evenly_spaced_positions(10, limit=4)

        np.testing.assert_array_equal(first, [0, 3, 6, 9])
        np.testing.assert_array_equal(second, first)
        np.testing.assert_array_equal(
            select_evenly_spaced_positions(4, limit=8),
            [0, 1, 2, 3],
        )

    def test_even_spacing_rejects_non_positive_counts_or_limits(self):
        for count, limit in ((0, 1), (-1, 1), (1, 0), (1, -1)):
            with self.subTest(count=count, limit=limit):
                with self.assertRaises(ValueError):
                    select_evenly_spaced_positions(count, limit=limit)


class StateProbeMetricTest(unittest.TestCase):
    def test_state_targets_encode_heading_without_angle_discontinuity(self):
        states = np.asarray(
            [
                [1.0, 2.0, 0.0, 0.5],
                [3.0, 4.0, np.pi / 2.0, 1.5],
            ],
            dtype=np.float64,
        )

        targets = state_to_probe_targets(states)

        np.testing.assert_allclose(
            targets,
            [
                [1.0, 2.0, 0.0, 1.0, 0.5],
                [3.0, 4.0, 1.0, 0.0, 1.5],
            ],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(targets.dtype, np.float32)

    def test_metrics_restore_pixels_circular_heading_and_velocity(self):
        target = np.asarray(
            [[5.0, 4.0, 0.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        predicted = np.asarray(
            [[6.0, 4.0, 1.0, 0.0, 1.5]],
            dtype=np.float32,
        )

        metrics = state_probe_metrics(
            predicted,
            target,
            world_bounds=np.asarray([0.0, 10.0, 0.0, 8.0]),
        )

        self.assertEqual(metrics["samples"], 1)
        self.assertAlmostEqual(metrics["mean_centre_error_pixels"], 6.3)
        self.assertAlmostEqual(metrics["p95_centre_error_pixels"], 6.3)
        self.assertAlmostEqual(metrics["mean_heading_error_degrees"], 90.0)
        self.assertAlmostEqual(metrics["p95_heading_error_degrees"], 90.0)
        self.assertAlmostEqual(metrics["mean_velocity_error"], 0.5)
        self.assertAlmostEqual(metrics["p95_velocity_error"], 0.5)

    def test_state_and_metric_validation_fail_closed(self):
        with self.assertRaises(ValueError):
            state_to_probe_targets(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            state_to_probe_targets(
                np.asarray([[0.0, 0.0, np.nan, 0.0]])
            )
        target = np.zeros((1, 5), dtype=np.float32)
        with self.assertRaises(ValueError):
            state_probe_metrics(
                np.zeros((2, 5), dtype=np.float32),
                target,
                world_bounds=np.asarray([0.0, 10.0, 0.0, 8.0]),
            )
        with self.assertRaises(ValueError):
            state_probe_metrics(
                target,
                target,
                world_bounds=np.asarray([0.0, 0.0, 0.0, 8.0]),
            )


class FakeVideoProcessor:
    def __init__(self):
        self.videos_seen: list[np.ndarray] | None = None

    def __call__(
        self,
        videos: list[np.ndarray],
        *,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        self.videos_seen = videos
        if return_tensors != "pt":
            raise AssertionError("expected PyTorch tensors")
        return {
            "pixel_values_videos": torch.zeros(
                (len(videos), 4, 3, 256, 256),
                dtype=torch.float32,
            )
        }


class FakeVJEPAConfig:
    tubelet_size = 2
    crop_size = 256
    patch_size = 16
    hidden_size = 4
    _commit_hash = "fake-revision"


class FakeVJEPAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = FakeVJEPAConfig()
        self.get_vision_features_called = False

    def get_vision_features(
        self,
        pixel_values_videos: torch.Tensor,
    ) -> torch.Tensor:
        self.get_vision_features_called = True
        batch_size = int(pixel_values_videos.shape[0])
        first = torch.ones((batch_size, 256, 4), dtype=torch.float32)
        last = torch.full((batch_size, 256, 4), 3.0, dtype=torch.float32)
        return torch.cat((first, last), dim=1) * self.weight


class FrozenVJEPAEncoderTest(unittest.TestCase):
    def test_pooling_keeps_last_tubelet_and_temporal_delta(self):
        tokens = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
        )

        pooled = pool_vjepa_tokens(
            tokens,
            tubelet_count=2,
            spatial_tokens=2,
        )

        torch.testing.assert_close(
            pooled,
            torch.tensor([[6.0, 7.0, 4.0, 4.0]]),
        )

    def test_encoder_freezes_model_and_returns_owned_finite_features(self):
        processor = FakeVideoProcessor()
        model = FakeVJEPAModel()
        encoder = FrozenVJEPAEncoder(
            processor=processor,
            model=model,
            model_id="fake/vjepa",
            revision="fake-revision",
            device="cpu",
        )
        clips = np.zeros((2, 4, 64, 64, 3), dtype=np.uint8)

        features = encoder.encode(clips)

        self.assertEqual(features.shape, (2, 8))
        self.assertEqual(features.dtype, np.float32)
        np.testing.assert_array_equal(features[:, :4], 3.0)
        np.testing.assert_array_equal(features[:, 4:], 2.0)
        self.assertTrue(model.get_vision_features_called)
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(len(processor.videos_seen or []), 2)
        features[0, 0] = 100.0
        self.assertEqual(float(model.weight), 1.0)

    def test_encoder_metadata_exposes_the_effective_contract(self):
        encoder = FrozenVJEPAEncoder(
            processor=FakeVideoProcessor(),
            model=FakeVJEPAModel(),
            model_id="fake/vjepa",
            revision="requested-revision",
            device="cpu",
        )

        self.assertEqual(
            encoder.metadata,
            {
                "model_id": "fake/vjepa",
                "requested_revision": "requested-revision",
                "resolved_revision": "fake-revision",
                "device": "cpu",
                "tubelet_size": 2,
                "crop_size": 256,
                "patch_size": 16,
                "hidden_size": 4,
                "feature_dim": 8,
                "pooling": "last_tubelet_mean_plus_last_minus_first",
            },
        )

    def test_pooling_and_encoder_validation_fail_closed(self):
        with self.assertRaises(ValueError):
            pool_vjepa_tokens(
                torch.zeros((2, 5, 4)),
                tubelet_count=2,
                spatial_tokens=2,
            )
        encoder = FrozenVJEPAEncoder(
            processor=FakeVideoProcessor(),
            model=FakeVJEPAModel(),
            model_id="fake/vjepa",
            revision="fake-revision",
            device="cpu",
        )
        invalid_clips = (
            np.zeros((0, 4, 64, 64, 3), dtype=np.uint8),
            np.zeros((1, 3, 64, 64, 3), dtype=np.uint8),
            np.zeros((1, 4, 64, 64, 3), dtype=np.float32),
        )
        for clips in invalid_clips:
            with self.subTest(shape=clips.shape, dtype=clips.dtype):
                with self.assertRaises(ValueError):
                    encoder.encode(clips)


class LinearStateProbeTest(unittest.TestCase):
    def test_primal_ridge_recovers_an_affine_state_mapping(self):
        features = np.asarray(
            [
                [-2.0, 1.0],
                [-1.0, -1.0],
                [0.0, 2.0],
                [1.0, -2.0],
                [2.0, 0.5],
                [3.0, 1.5],
            ],
            dtype=np.float32,
        )
        weights = np.asarray(
            [
                [1.0, 0.0, 0.5, -0.25, 2.0],
                [0.0, 2.0, -1.0, 0.75, 0.5],
            ],
            dtype=np.float64,
        )
        bias = np.asarray([0.5, -1.0, 0.2, 0.4, 1.0])
        targets = features @ weights + bias

        probe = fit_linear_state_probe(features, targets, ridge=0.0)
        predictions = probe.predict(features)

        self.assertEqual(probe.feature_dim, 2)
        np.testing.assert_allclose(predictions, targets, rtol=0.0, atol=1e-10)

    def test_dual_ridge_is_deterministic_and_returns_finite_predictions(self):
        features = np.asarray(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        targets = np.asarray(
            [
                [1.0, 2.0, 0.0, 1.0, 0.5],
                [2.0, 3.0, 1.0, 0.0, 1.0],
                [3.0, 4.0, 0.0, -1.0, 1.5],
            ],
            dtype=np.float32,
        )

        first = fit_linear_state_probe(features, targets, ridge=1e-3)
        second = fit_linear_state_probe(features, targets, ridge=1e-3)

        np.testing.assert_array_equal(first.weight, second.weight)
        np.testing.assert_array_equal(first.bias, second.bias)
        self.assertTrue(np.all(np.isfinite(first.predict(features))))
        self.assertEqual(first.solver, "dual")

    def test_probe_and_fit_validation_fail_closed(self):
        features = np.ones((3, 2), dtype=np.float32)
        targets = np.ones((3, 5), dtype=np.float32)
        invalid = (
            (features[:0], targets[:0], 1e-3),
            (features, targets[:2], 1e-3),
            (features[:, :0], targets, 1e-3),
            (features, targets[:, :4], 1e-3),
            (features, targets, -1.0),
            (features, targets, float("inf")),
        )
        for x, y, ridge in invalid:
            with self.subTest(x_shape=x.shape, y_shape=y.shape, ridge=ridge):
                with self.assertRaises(ValueError):
                    fit_linear_state_probe(x, y, ridge=ridge)
        probe = fit_linear_state_probe(features, targets, ridge=1e-3)
        with self.assertRaises(ValueError):
            probe.predict(np.ones((2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            probe.predict(np.asarray([[np.nan, 0.0]], dtype=np.float32))

    def test_mean_baseline_and_registered_gates_are_exact(self):
        train_targets = np.asarray(
            [
                [1.0, 2.0, 0.0, 1.0, 0.5],
                [3.0, 4.0, 1.0, 0.0, 1.5],
            ],
            dtype=np.float32,
        )

        baseline = mean_target_predictions(train_targets, count=3)
        gates = representation_probe_gates(
            recorded={
                "mean_centre_error_pixels": 3.0,
                "mean_heading_error_degrees": 44.0,
                "mean_velocity_error": 0.90,
            },
            reversed_metrics={"mean_velocity_error": 1.00},
            repeat_last_metrics={"mean_velocity_error": 1.10},
        )

        np.testing.assert_allclose(
            baseline,
            np.repeat(np.mean(train_targets, axis=0, keepdims=True), 3, axis=0),
        )
        self.assertEqual(
            gates,
            {
                "centre_mean_le_3px": True,
                "heading_mean_lt_45deg": True,
                "velocity_beats_reversed_5pct": True,
                "velocity_beats_repeat_last_5pct": True,
            },
        )
        failing = representation_probe_gates(
            recorded={
                "mean_centre_error_pixels": 3.01,
                "mean_heading_error_degrees": 45.0,
                "mean_velocity_error": 0.96,
            },
            reversed_metrics={"mean_velocity_error": 1.00},
            repeat_last_metrics={"mean_velocity_error": 1.00},
        )
        self.assertFalse(any(failing.values()))


if __name__ == "__main__":
    unittest.main()
