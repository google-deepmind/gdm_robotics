# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from unittest import mock
import dm_env
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import episodic_logger as gdmr_logger
from gdm_robotics.interfaces import policy as gdmr_policy
from gdm_robotics.runtime import runloop
import numpy as np
from absl.testing import absltest


class RunloopTest(absltest.TestCase):

  def test_single_episode(self):
    env = mock.create_autospec(gdmr_env.Environment)
    policy = mock.create_autospec(gdmr_policy.Policy)
    logger = mock.create_autospec(gdmr_logger.EpisodicLogger)

    env_reset_timestep = dm_env.restart({"obs1": "value1"})
    env_step_timesteps = [
        dm_env.transition(0.9, {"obs1": "value2", "obs2": "value2"}, 1.0),
        dm_env.transition(0.95, {"obs1": "value3"}, 1.0),
        dm_env.termination(1.0, {"obs1": "value4"}),
    ]
    env.reset_with_options.return_value = env_reset_timestep
    env.step.side_effect = env_step_timesteps

    policy_initial_state = {"a_state": 0}
    policy_step_outputs = [
        ((np.array([1]), {"key": 11}), {"a_state": -1}),
        ((np.array([2]), {"key": 12}), {"a_state": -2}),
        ((np.array([3]), {"key": 13}), {"a_state": -3}),
    ]
    policy.initial_state.return_value = policy_initial_state
    policy.step.side_effect = policy_step_outputs

    runloop.Runloop(env, policy, [logger]).run_single_episode()
    # Reset.
    env.reset_with_options.assert_called_once()
    policy.initial_state.assert_called_once()
    logger.reset.assert_called_once_with(env_reset_timestep)

    # There should be 3 steps.
    expected_policy_step_calls = [
        mock.call(env_reset_timestep, policy_initial_state),
    ]
    for timestep, policy_out in zip(
        env_step_timesteps[0:-1], policy_step_outputs[0:-1]
    ):
      expected_policy_step_calls.append(mock.call(timestep, policy_out[1]))

    policy.step.assert_has_calls(expected_policy_step_calls)

    expected_env_step_calls = [
        mock.call(action) for (action, _), _ in policy_step_outputs
    ]
    env.step.assert_has_calls(expected_env_step_calls)

    expected_logger_calls = [
        mock.call(action, timestep, policy_extra)
        for timestep, ((action, policy_extra), _) in zip(
            env_step_timesteps, policy_step_outputs
        )
    ]
    logger.record_action_and_next_timestep.assert_has_calls(
        expected_logger_calls
    )

    # Logger write.
    logger.write.assert_called_once()

  def test_multiple_episodes(self):
    num_episodes = 2
    env = mock.create_autospec(gdmr_env.Environment)
    policy = mock.create_autospec(gdmr_policy.Policy)
    logger = mock.create_autospec(gdmr_logger.EpisodicLogger)

    env.reset_with_options.return_value = dm_env.restart({})
    env.step.side_effect = [
        dm_env.transition(1.0, {}, 1.0),
        dm_env.transition(1.0, {}, 1.0),
        dm_env.termination(1.0, {}),
    ] * num_episodes
    policy.step.return_value = (np.array([]), {}), {}

    # Also add a default Runloop operations object as it should be a no-op.
    runloop.Runloop(
        env,
        policy,
        [logger],
        runloop_runtime_operations=[runloop.RunloopRuntimeOperations()],
    ).run(num_episodes=2)
    # Reset. Called once per episode.
    self.assertEqual(env.reset_with_options.call_count, num_episodes)
    self.assertEqual(policy.initial_state.call_count, num_episodes)
    self.assertEqual(logger.reset.call_count, num_episodes)

    # There should be 3 steps per episode.
    self.assertEqual(policy.step.call_count, 3 * num_episodes)
    self.assertEqual(env.step.call_count, 3 * num_episodes)
    self.assertEqual(
        logger.record_action_and_next_timestep.call_count, 3 * num_episodes
    )

    # Logger write.
    self.assertEqual(logger.write.call_count, num_episodes)

  def test_runtime_operations(self):
    num_episodes = 3
    env = mock.create_autospec(gdmr_env.Environment)
    policy = mock.create_autospec(gdmr_policy.Policy)
    logger = mock.create_autospec(gdmr_logger.EpisodicLogger)

    env_reset_timestep = dm_env.restart({"obs1": "value1"})
    env.reset_with_options.return_value = env_reset_timestep

    env_timesteps = [
        dm_env.transition(0.9, {"obs1": "value2", "obs2": "value2"}, 1.0),
        dm_env.transition(0.95, {"obs1": "value3"}, 1.0),
        dm_env.termination(1.0, {"obs1": "value4"}),
    ] * num_episodes
    env.step.side_effect = env_timesteps
    policy.step.return_value = (np.array([]), {}), {}

    operations = mock.create_autospec(runloop.RunloopRuntimeOperations)
    operations.before_episode_reset.side_effect = [True] * num_episodes + [
        False
    ]

    # Run an infinite number of episodes. It will be the runtime operations that
    # will trigger the termination.
    runloop.Runloop(
        env, policy, [logger], runloop_runtime_operations=[operations]
    ).run(num_episodes=None)
    # `before_episode_reset` is called once per episode, plus an additional time
    # before triggering the termination.
    self.assertEqual(
        operations.before_episode_reset.call_count, num_episodes + 1
    )

    # Assert that inspect_timestep is called for each timestep, with the reset
    # timestep and the with the other timesteps.
    expected_inspect_timestep_calls = []
    for i in range(num_episodes):
      # Reset.
      expected_inspect_timestep_calls.append(mock.call(env_reset_timestep))
      # Steps.
      expected_inspect_timestep_calls.extend(
          mock.call(timestep) for timestep in env_timesteps[i * 3 : (i + 1) * 3]
      )

    operations.inspect_timestep.assert_has_calls(
        expected_inspect_timestep_calls
    )

  def test_reset_with_non_none_default_reset_options(self):

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class TestResetOptions(gdmr_env.Options):

      option_a: int
      option_b: str

    env = mock.create_autospec(gdmr_env.Environment)
    policy = mock.create_autospec(gdmr_policy.Policy)
    logger = mock.create_autospec(gdmr_logger.EpisodicLogger)

    env.default_reset_options.return_value = TestResetOptions(
        option_a=1, option_b="default"
    )

    env_reset_timestep = dm_env.restart({"obs1": "value1"})

    env.reset_with_options.return_value = env_reset_timestep

    # These definitions are needed to run the runloop even if we do not assert
    # on them.
    policy_initial_state = {"a_state": 0}
    policy_step_outputs = [
        ((np.array([1]), {"key": 11}), {"a_state": -1}),
        ((np.array([2]), {"key": 12}), {"a_state": -2}),
        ((np.array([3]), {"key": 13}), {"a_state": -3}),
    ]
    policy.initial_state.return_value = policy_initial_state
    policy.step.side_effect = policy_step_outputs

    runloop.Runloop(env, policy, [logger]).run_single_episode()

    env.reset_with_options.assert_called_once_with(
        options=TestResetOptions(option_a=1, option_b="default")
    )

  def test_reset_with_custom_reset_options_provider(self):

    @dataclasses.dataclass(frozen=True, kw_only=True)
    class TestResetOptions(gdmr_env.Options):

      option_a: int
      option_b: str

    env = mock.create_autospec(gdmr_env.Environment)
    policy = mock.create_autospec(gdmr_policy.Policy)
    logger = mock.create_autospec(gdmr_logger.EpisodicLogger)

    reset_option = TestResetOptions(option_a=2, option_b="custom")

    env_reset_timestep = dm_env.restart({"obs1": "value1"})
    env.reset_with_options.return_value = env_reset_timestep

    # These definitions are needed to run the runloop even if we do not assert
    # on them.
    policy_initial_state = {"a_state": 0}
    policy_step_outputs = [
        ((np.array([1]), {"key": 11}), {"a_state": -1}),
        ((np.array([2]), {"key": 12}), {"a_state": -2}),
        ((np.array([3]), {"key": 13}), {"a_state": -3}),
    ]
    policy.initial_state.return_value = policy_initial_state
    policy.step.side_effect = policy_step_outputs

    runloop.Runloop(
        env, policy, [logger], reset_options_provider=lambda: reset_option
    ).run_single_episode()

    env.reset_with_options.assert_called_once_with(options=reset_option)


if __name__ == "__main__":
  absltest.main()
