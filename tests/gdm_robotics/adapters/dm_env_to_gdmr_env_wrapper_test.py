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
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
from absl.testing import absltest


class DmEnvToGdmrEnvWrapperTest(absltest.TestCase):

  def test_not_dm_env_raises_type_error(self):
    """Tests error is raised if the wrapped environment is not a dm_env.Environment."""

    class NotDmEnv:

      def reset(self) -> dm_env.TimeStep:
        return dm_env.restart(np.array([0.0, 0.0], dtype=np.float32))

      def step(self, unused_action: np.ndarray) -> dm_env.TimeStep:
        reward = np.float32(0)
        observation = np.array([0, 0], dtype=np.float32)
        return dm_env.transition(
            reward=reward, observation=observation, discount=np.float32(0.9)
        )

    with self.assertRaises(TypeError):
      # Disable pytype as we are forcing a runtime check and we know the type is
      # wrong.
      dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(NotDmEnv())  # pyrefly: ignore[bad-argument-type]

  def test_specs(self):
    """Tests if the specs are correctly propagated."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    wrapped_env.action_spec.return_value = specs.BoundedArray(
        shape=(1,), dtype=np.float32, minimum=0.0, maximum=1.0, name="action"
    )
    wrapped_env.observation_spec.return_value = specs.Array(
        shape=(2,), dtype=np.float32, name="observation"
    )
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )

    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)

    self.assertEqual(env.action_spec(), wrapped_env.action_spec())
    self.assertEqual(env.observation_spec(), wrapped_env.observation_spec())
    self.assertEqual(env.reward_spec(), wrapped_env.reward_spec())
    self.assertEqual(env.discount_spec(), wrapped_env.discount_spec())

    expected_timestep_spec = gdmr_types.TimeStepSpec(
        step_type=dm_env_to_gdmr_env_wrapper.gdmr_types.STEP_TYPE_SPEC,
        reward=wrapped_env.reward_spec(),
        discount=wrapped_env.discount_spec(),
        observation=wrapped_env.observation_spec(),
    )
    self.assertEqual(env.timestep_spec(), expected_timestep_spec)

  def test_reset_is_forwarded(self):
    """Tests that reset is correctly forwarded."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )
    wrapped_env.observation_spec.return_value = specs.Array(
        shape=(2,), dtype=np.float32, name="observation"
    )

    wrapped_env.reset.return_value = dm_env.restart(
        np.array([0.0, 0.0], dtype=np.float32)
    )
    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)

    timestep = env.reset()
    wrapped_env.reset.assert_called_once()
    self.assertTrue(timestep.first())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.FIRST, dtype=np.uint8)
    )

    np.testing.assert_array_equal(
        timestep.observation, np.array([0.0, 0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.reward, np.zeros((), dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.zeros((), dtype=np.float32)
    )

  def test_step_is_forwarded(self):
    """Tests that step is correctly forwarded."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )
    wrapped_env.observation_spec.return_value = specs.Array(
        shape=(2,), dtype=np.float32, name="observation"
    )

    wrapped_env.step.return_value = dm_env.transition(
        reward=np.float32(0.5),
        observation=np.array([1.0, 2.0], dtype=np.float32),
        discount=np.float32(0.9),
    )

    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)
    action = np.array([0.5], dtype=np.float32)
    timestep = env.step(action)
    wrapped_env.step.assert_called_once_with(action)

    self.assertTrue(timestep.mid())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.MID, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        timestep.observation, np.array([1.0, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.reward, np.array(0.5, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.array(0.9, dtype=np.float32)
    )

  def test_step_terminates(self):
    """Tests that step is correctly forwarded."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )

    wrapped_env.step.return_value = dm_env.termination(
        reward=np.float32(0.5),
        observation=np.array([1.0, 2.0], dtype=np.float32),
    )
    wrapped_env.observation_spec.return_value = specs.Array(
        shape=(2,), dtype=np.float32, name="observation"
    )

    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)
    action = np.array([0.5], dtype=np.float32)
    timestep = env.step(action)
    wrapped_env.step.assert_called_once_with(action)

    self.assertTrue(timestep.last())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.LAST, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        timestep.observation, np.array([1.0, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.reward, np.array(0.5, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.array(0, dtype=np.float32)
    )

  def test_step_truncates(self):
    """Tests that step is correctly forwarded."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )

    wrapped_env.step.return_value = dm_env.truncation(
        reward=np.float32(0.5),
        observation=np.array([1.0, 2.0], dtype=np.float32),
        discount=np.float32(0.9),
    )
    wrapped_env.observation_spec.return_value = specs.Array(
        shape=(2,), dtype=np.float32, name="observation"
    )

    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)
    action = np.array([0.5], dtype=np.float32)
    timestep = env.step(action)
    wrapped_env.step.assert_called_once_with(action)

    self.assertTrue(timestep.last())
    np.testing.assert_array_equal(
        timestep.step_type, np.array(dm_env.StepType.LAST, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        timestep.observation, np.array([1.0, 2.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.reward, np.array(0.5, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        timestep.discount, np.array(0.9, dtype=np.float32)
    )

  def test_observations_are_arrays(self):
    """Tests that the observations are arrays."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    wrapped_env.observation_spec.return_value = {
        "a": specs.Array(shape=(2,), dtype=np.float32, name="a"),
        "b": specs.Array(shape=(), dtype=np.int64, name="b"),
    }
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )

    wrapped_env.reset.return_value = dm_env.restart(
        {"a": np.array([0.0, 0.0], dtype=np.float32), "b": 1}
    )
    wrapped_env.step.return_value = dm_env.transition(
        reward=np.float32(0.5),
        observation={"a": np.array([1.0, 2.0], dtype=np.float32), "b": 2},
        discount=np.float32(0.9),
    )

    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)
    timestep = env.reset()
    self.assertFalse(np.isscalar(timestep.observation["b"]))
    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.array([0.0, 0.0], dtype=np.float32),
            "b": np.array(1, dtype=np.int64),
        },
    )

    action = np.array([0.5], dtype=np.float32)
    timestep = env.step(action)
    self.assertFalse(np.isscalar(timestep.observation["b"]))
    np.testing.assert_equal(
        timestep.observation,
        {
            "a": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array(2, dtype=np.int64),
        },
    )

  def test_close(self):
    """Tests that the close method is correctly propagated."""
    wrapped_env = mock.create_autospec(dm_env.Environment, instance=True)
    # Create reward and discount specs as these are used in the wrapper init.
    wrapped_env.reward_spec.return_value = specs.Array(
        shape=(), dtype=np.float32, name="reward"
    )
    wrapped_env.discount_spec.return_value = specs.BoundedArray(
        shape=(), dtype=np.float32, minimum=0.0, maximum=1.0, name="discount"
    )

    wrapped_env.close.return_value = None
    env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(wrapped_env)
    env.close()
    wrapped_env.close.assert_called_once()


if __name__ == "__main__":
  absltest.main()
