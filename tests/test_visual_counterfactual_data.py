from __future__ import annotations

import unittest

import numpy as np

from world_model_lab.car_env import CarEnv
from world_model_lab.collect_data import _terminal_reason
from world_model_lab.diagnose_visual_rollout import (
    VisualRolloutWindow,
    select_visual_rollout_windows,
)
from world_model_lab.visual_counterfactual_data import (
    MatchedCounterfactualBatch,
    build_matched_counterfactual_batch,
)
from world_model_lab.visual_dataset import build_visual_dataset


def make_physical_visual_dataset(
    *,
    episodes: int = 10,
    transitions: int = 6,
    episode_id_start: int = 10,
) -> dict[str, np.ndarray]:
    records: dict[str, list] = {
        "states": [],
        "actions": [],
        "next_states": [],
        "rewards": [],
        "dones": [],
        "episode_ids": [],
        "step_ids": [],
        "terminal_reasons": [],
    }
    for episode_offset in range(episodes):
        episode_id = episode_id_start + episode_offset
        initial_state = (
            1.2 + 0.12 * (episode_offset % 5),
            1.2 + 0.12 * (episode_offset // 5),
            -0.15 + 0.03 * episode_offset,
            0.6,
        )
        environment = CarEnv(
            initial_state=initial_state,
            max_steps=transitions,
        )
        for step in range(transitions):
            steering = -0.2 + 0.04 * episode_offset
            acceleration = -0.1 + 0.04 * (step % 4)
            state = environment.state
            next_state, reward, done, info = environment.step(
                steering,
                acceleration,
            )
            records["states"].append(state)
            records["actions"].append(
                [
                    info["applied_steering"],
                    info["applied_acceleration"],
                ]
            )
            records["next_states"].append(next_state)
            records["rewards"].append(reward)
            records["dones"].append(done)
            records["episode_ids"].append(episode_id)
            records["step_ids"].append(step)
            records["terminal_reasons"].append(_terminal_reason(info))
            if done:
                break
    source = {
        "states": np.asarray(records["states"], dtype=np.float64),
        "actions": np.asarray(records["actions"], dtype=np.float64),
        "next_states": np.asarray(
            records["next_states"],
            dtype=np.float64,
        ),
        "rewards": np.asarray(records["rewards"], dtype=np.float64),
        "dones": np.asarray(records["dones"], dtype=np.bool_),
        "episode_ids": np.asarray(
            records["episode_ids"],
            dtype=np.int64,
        ),
        "step_ids": np.asarray(records["step_ids"], dtype=np.int64),
        "terminal_reasons": np.asarray(
            records["terminal_reasons"],
            dtype=np.str_,
        ),
    }
    return build_visual_dataset(source)


def make_manual_window(
    *,
    episode_id: int,
    initial_frame_index: int,
    future_actions: np.ndarray,
) -> VisualRolloutWindow:
    horizon = int(future_actions.shape[0])
    return VisualRolloutWindow(
        episode_id=episode_id,
        start_step=3,
        context_latents=np.zeros((4, 2), dtype=np.float32),
        history_actions=np.zeros((3, 2), dtype=np.float64),
        future_actions=future_actions,
        target_latents=np.zeros((horizon, 2), dtype=np.float32),
        initial_frame_index=initial_frame_index,
        target_frame_indices=np.zeros(horizon, dtype=np.int64),
    )


class MatchedCounterfactualBranchTest(unittest.TestCase):
    def test_branch_executes_donor_actions_from_recipient_state(self):
        visual = make_physical_visual_dataset(
            episodes=2,
            transitions=6,
        )
        latents = np.zeros(
            (visual["frames"].shape[0], 2),
            dtype=np.float32,
        )
        selection = select_visual_rollout_windows(
            dataset=visual,
            latent_frames=latents,
            selected_episode_ids=visual["episode_ids"],
            max_horizon=2,
            windows_per_episode=1,
        )
        windows = selection.windows

        batch = build_matched_counterfactual_batch(
            visual,
            windows,
            np.asarray([1, 0], dtype=np.int64),
        )

        manual = CarEnv(
            initial_state=visual["states"][
                windows[0].initial_frame_index
            ]
        )
        expected_states = []
        for action in windows[1].future_actions:
            state, _, done, _ = manual.step(*action)
            self.assertFalse(done)
            expected_states.append(state)

        self.assertIsInstance(batch, MatchedCounterfactualBatch)
        np.testing.assert_allclose(
            batch.true_states[0],
            expected_states,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_array_equal(
            batch.donor_window_indices,
            [1, 0],
        )
        np.testing.assert_array_equal(batch.valid_steps, True)
        np.testing.assert_array_equal(
            batch.applied_actions,
            batch.requested_actions,
        )
        np.testing.assert_array_equal(batch.terminal_steps, [-1, -1])
        np.testing.assert_array_equal(batch.terminal_reasons, ["", ""])
        for values in batch.__dict__.values():
            self.assertFalse(np.asarray(values).flags.writeable)

    def test_terminal_transition_is_valid_and_later_steps_are_masked(self):
        records: dict[str, list] = {
            "states": [],
            "actions": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
            "episode_ids": [],
            "step_ids": [],
            "terminal_reasons": [],
        }
        for episode_id, initial_state in (
            (0, (9.75, 1.0, 0.0, 1.0)),
            (1, (1.0, 1.0, 0.0, 0.5)),
        ):
            environment = CarEnv(initial_state=initial_state, max_steps=1)
            state = environment.state
            next_state, reward, done, info = environment.step(0.0, 0.0)
            records["states"].append(state)
            records["actions"].append([0.0, 0.0])
            records["next_states"].append(next_state)
            records["rewards"].append(reward)
            records["dones"].append(done)
            records["episode_ids"].append(episode_id)
            records["step_ids"].append(0)
            records["terminal_reasons"].append(_terminal_reason(info))
        visual = build_visual_dataset(
            {
                "states": np.asarray(records["states"], dtype=np.float64),
                "actions": np.asarray(records["actions"], dtype=np.float64),
                "next_states": np.asarray(
                    records["next_states"],
                    dtype=np.float64,
                ),
                "rewards": np.asarray(
                    records["rewards"],
                    dtype=np.float64,
                ),
                "dones": np.asarray(records["dones"], dtype=np.bool_),
                "episode_ids": np.asarray(
                    records["episode_ids"],
                    dtype=np.int64,
                ),
                "step_ids": np.asarray(
                    records["step_ids"],
                    dtype=np.int64,
                ),
                "terminal_reasons": np.asarray(
                    records["terminal_reasons"],
                    dtype=np.str_,
                ),
            }
        )
        zero_actions = np.zeros((2, 2), dtype=np.float64)
        windows = (
            make_manual_window(
                episode_id=0,
                initial_frame_index=int(visual["frame_offsets"][0]),
                future_actions=zero_actions,
            ),
            make_manual_window(
                episode_id=1,
                initial_frame_index=int(visual["frame_offsets"][1]),
                future_actions=zero_actions,
            ),
        )

        batch = build_matched_counterfactual_batch(
            visual,
            windows,
            np.asarray([1, 0], dtype=np.int64),
        )

        np.testing.assert_array_equal(
            batch.valid_steps[0],
            [True, False],
        )
        self.assertEqual(int(batch.terminal_steps[0]), 1)
        self.assertEqual(str(batch.terminal_reasons[0]), "out_of_bounds")
        self.assertTrue(np.all(np.isnan(batch.true_states[0, 1])))
        self.assertTrue(np.all(batch.true_frames[0, 1] == 0))

    def test_builder_rejects_invalid_windows_or_permutations(self):
        visual = make_physical_visual_dataset(
            episodes=2,
            transitions=6,
        )
        actions = np.zeros((2, 2), dtype=np.float64)
        windows = (
            make_manual_window(
                episode_id=0,
                initial_frame_index=0,
                future_actions=actions,
            ),
            make_manual_window(
                episode_id=1,
                initial_frame_index=int(visual["frame_offsets"][1]),
                future_actions=actions,
            ),
        )
        invalid = (
            np.asarray([0, 1], dtype=np.int64),
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([[1, 0]], dtype=np.int64),
            np.asarray([1.0, 0.0]),
        )
        for donor_indices in invalid:
            with self.subTest(donor_indices=donor_indices):
                with self.assertRaises(ValueError):
                    build_matched_counterfactual_batch(
                        visual,
                        windows,
                        donor_indices,
                    )

        with self.assertRaises(ValueError):
            build_matched_counterfactual_batch(
                visual,
                (),
                np.asarray([], dtype=np.int64),
            )
        mismatched = (
            windows[0],
            make_manual_window(
                episode_id=1,
                initial_frame_index=int(visual["frame_offsets"][1]),
                future_actions=np.zeros((3, 2), dtype=np.float64),
            ),
        )
        with self.assertRaisesRegex(ValueError, "horizon"):
            build_matched_counterfactual_batch(
                visual,
                mismatched,
                np.asarray([1, 0], dtype=np.int64),
            )

    def test_models_receive_the_actions_actually_applied_by_car_env(self):
        visual = make_physical_visual_dataset(
            episodes=2,
            transitions=6,
        )
        extreme = np.asarray(
            [[10.0, -10.0], [10.0, -10.0]],
            dtype=np.float64,
        )
        windows = (
            make_manual_window(
                episode_id=10,
                initial_frame_index=0,
                future_actions=np.zeros((2, 2), dtype=np.float64),
            ),
            make_manual_window(
                episode_id=11,
                initial_frame_index=int(visual["frame_offsets"][1]),
                future_actions=extreme,
            ),
        )

        batch = build_matched_counterfactual_batch(
            visual,
            windows,
            np.asarray([1, 0], dtype=np.int64),
        )

        np.testing.assert_array_equal(batch.requested_actions[0], extreme)
        np.testing.assert_array_equal(
            batch.applied_actions[0],
            [[0.5, -1.0], [0.5, -1.0]],
        )


if __name__ == "__main__":
    unittest.main()
