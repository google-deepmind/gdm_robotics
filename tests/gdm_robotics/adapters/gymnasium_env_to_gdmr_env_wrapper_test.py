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
from unittest import mock

import dm_env
from dm_env import specs
from gdm_robotics.adapters import gymnasium_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import types as gdmr_types
import gymnasium
from gymnasium import spaces
import numpy as np
import tree

from absl.testing import absltest
from absl.testing import parameterized


class GymEnvToGdmrEnvWrapperTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="discrete",
          gym_space=spaces.Discrete(3),
          expected_spec=specs.DiscreteArray(num_values=3, dtype=np.int64),
      ),
      dict(
          testcase_name="box",
          gym_space=spaces.Box(low=-1.0, high=1.0, shape=(2, 3)),
          expected_spec=specs.BoundedArray(
              shape=(2, 3),
              dtype=np.float32,
              minimum=-1.0,
              maximum=1.0,
          ),
      ),
      dict(
          testcase_name="tuple",
          gym_space=spaces.Tuple((
              spaces.Box(low=-1.0, high=1.0, shape=(2, 3)),
              spaces.Discrete(3),
          )),
          expected_spec=(
              specs.BoundedArray(
                  shape=(2, 3),
                  dtype=np.float32,
                  minimum=-1.0,
                  maximum=1.0,
              ),
              specs.DiscreteArray(num_values=3, dtype=np.int64),
          ),
      ),
      dict(
          testcase_name="dict",
          gym_space=spaces.Dict({
              "a": spaces.Box(low=-1.0, high=1.0, shape=(2, 3)),
              "b": spaces.Discrete(3),
          }),
          expected_spec={
              "a": specs.BoundedArray(
                  shape=(2, 3),
                  dtype=np.float32,
                  minimum=-1.0,
                  maximum=1.0,
              ),
              "b": specs.DiscreteArray(num_values=3, dtype=np.int64),
          },
      ),
      dict(
          testcase_name="multi_binary",
          gym_space=spaces.MultiBinary(3),
          expected_spec=specs.BoundedArray(
              shape=(3,),
              dtype=np.int8,
              minimum=0,
              maximum=1,
          ),
      ),
      dict(
          testcase_name="multi_discrete",
          gym_space=spaces.MultiDiscrete(nvec=[5, 3]),
          expected_spec=specs.BoundedArray(
              shape=(2,),
              dtype=np.int64,
              minimum=[0, 0],
              maximum=[5, 3],
          ),
      ),
      dict(
          testcase_name="text",
          gym_space=spaces.Text(max_length=10),
          expected_spec=specs.StringArray(shape=(), name="text"),
      ),
  )
  def test_spec_conversion(
      self,
      gym_space: gymnasium.Space,
      expected_spec: tree.Structure[specs.Array],
  ):
    self.assertEqual(
        gymnasium_env_to_gdmr_env_wrapper.convert_gym_space_to_spec(gym_space),
        expected_spec,
    )

  def test_environment_returns_specs(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    wrapped_env.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(2, 3), dtype=np.float32
    )
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )
    self.assertEqual(
        env.action_spec(),
        specs.BoundedArray(
            shape=(2, 3),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
        ),
    )
    self.assertEqual(
        env.observation_spec(),
        {
            "a": specs.BoundedArray(
                shape=(4, 3),
                dtype=np.float64,
                minimum=-5.0,
                maximum=5.0,
            ),
            "b": specs.DiscreteArray(num_values=3, dtype=np.int64),
        },
    )
    self.assertEqual(
        env.reward_spec(),
        specs.Array(
            shape=(),
            dtype=np.float32,
        ),
    )
    self.assertEqual(
        env.discount_spec(),
        specs.Array(
            shape=(),
            dtype=np.float32,
        ),
    )

    self.assertEqual(
        env.timestep_spec(),
        gdmr_types.TimeStepSpec(
            step_type=gdmr_types.STEP_TYPE_SPEC,
            reward=specs.Array(
                shape=(),
                dtype=np.float32,
            ),
            discount=specs.Array(shape=(), dtype=np.float32),
            observation={
                "a": specs.BoundedArray(
                    shape=(4, 3),
                    dtype=np.float64,
                    minimum=-5.0,
                    maximum=5.0,
                ),
                "b": specs.DiscreteArray(num_values=3, dtype=np.int64),
            },
        ),
    )

  def test_environment_resets(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    wrapped_env.reset.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
        {},
    )

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )
    timestep = env.reset()

    default_options = (
        gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvResetOptions()
    )
    wrapped_env.reset.assert_called_once_with(
        seed=default_options.seed, options=default_options.options
    )

    self.assertTrue(timestep.first())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.FIRST, dtype=np.uint8)
    )

    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
    )
    np.testing.assert_array_equal(
        timestep.reward, np.zeros((), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.zeros((), dtype=np.float32)
    )

  def test_environment_resets_with_custom_options(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    wrapped_env.reset.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
        {},
    )

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )
    options = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvResetOptions(
        seed=123, options={"some_option": "some_value"}
    )

    timestep = env.reset_with_options(options=options)

    wrapped_env.reset.assert_called_once_with(
        seed=options.seed, options=options.options
    )

    self.assertTrue(timestep.first())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.FIRST, dtype=np.uint8)
    )

    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
    )
    np.testing.assert_array_equal(
        timestep.reward, np.zeros((), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.zeros((), dtype=np.float32)
    )

  def test_environment_steps(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    reward = 0.6
    info = {"some_data": "some_value"}
    wrapped_env.step.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
        reward,
        False,
        False,
        info,
    )

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )

    action = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, -0.6]], dtype=np.float32)
    timestep = env.step(action)

    step_call_args, _ = wrapped_env.step.call_args
    # For some reasons, the mock adds an additional leading dimension to the
    # action.
    np.testing.assert_equal(np.squeeze(step_call_args), action)

    self.assertTrue(timestep.mid())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.MID, dtype=np.uint8)
    )

    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
    )
    np.testing.assert_array_equal(
        timestep.reward, np.array(reward, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.array(1.0, dtype=np.float32)
    )

  def test_environment_terminates(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    reward = 0.6
    info = {"some_data": "some_value"}
    wrapped_env.step.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
        reward,
        True,  # terminated.
        False,  # truncated.
        info,
    )

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )

    action = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, -0.6]], dtype=np.float32)
    timestep = env.step(action)

    step_call_args, _ = wrapped_env.step.call_args
    # For some reasons, the mock adds an additional leading dimension to the
    # action.
    np.testing.assert_equal(np.squeeze(step_call_args), action)

    self.assertTrue(timestep.last())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.LAST, dtype=np.uint8)
    )

    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
    )
    np.testing.assert_array_equal(
        timestep.reward, np.array(reward, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.array(0.0, dtype=np.float32)
    )

  def test_environment_truncates(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    reward = 0.6
    info = {"some_data": "some_value"}
    wrapped_env.step.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
        reward,
        False,  # terminated.
        True,  # truncated.
        info,
    )

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )

    action = np.array([[0.5, 0.6], [0.7, 0.8], [0.9, -0.6]], dtype=np.float32)
    timestep = env.step(action)

    step_call_args, _ = wrapped_env.step.call_args
    # For some reasons, the mock adds an additional leading dimension to the
    # action.
    np.testing.assert_equal(np.squeeze(step_call_args), action)

    self.assertTrue(timestep.last())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.LAST, dtype=np.uint8)
    )

    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
    )
    np.testing.assert_array_equal(
        timestep.reward, np.array(reward, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.array(1.0, dtype=np.float32)
    )

  def test_observations_are_arrays(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Dict({
        "a": spaces.Box(low=-5.0, high=5.0, shape=(4, 3), dtype=np.float64),
        "b": spaces.Discrete(3),
    })

    wrapped_env.reset.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": 1,
        },
        {},
    )

    wrapped_env.step.return_value = (
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": 2,
        },
        0.6,
        False,
        False,
        {"some_data": "some_value"},
    )

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )

    timestep = env.reset()
    self.assertFalse(np.isscalar(timestep.observation["b"]))
    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(1, dtype=np.int64),
        },
    )

    timestep = env.step(
        np.array([[0.5, 0.6], [0.7, 0.8], [0.9, -0.6]], dtype=np.float32)
    )
    self.assertFalse(np.isscalar(timestep.observation["b"]))
    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.zeros((4, 3), dtype=np.float64),
            "b": np.asarray(2, dtype=np.int64),
        },
    )

  def test_environment_closes(self):
    wrapped_env = mock.create_autospec(gymnasium.Env, instance=True)

    # We need to define the action, observation spaces and reward range as these
    # are used in the wrapper init.
    wrapped_env.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))
    wrapped_env.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2, 3))

    env = gymnasium_env_to_gdmr_env_wrapper.GymnasiumEnvToGdmrEnvWrapper(
        wrapped_env
    )
    env.close()
    wrapped_env.close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
