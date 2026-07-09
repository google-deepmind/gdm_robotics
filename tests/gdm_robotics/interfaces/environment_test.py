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
from dm_env import specs
from gdm_robotics.interfaces import environment
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np

from absl.testing import absltest


class _TestEnvironment(environment.Environment):
  """Test environment implementing the abstract methods.

  Implementation is forwarded to the wrapped mock object.
  """

  def __init__(self, mock_env: mock.Mock):
    self._mock = mock_env

  def reset_with_options(
      self,
      *,
      options: environment.ResetOptions,
  ) -> dm_env.TimeStep:
    print(f"Hello {options}")
    return self._mock.reset_with_options(options=options)

  def timestep_spec(self) -> gdmr_types.TimeStepSpec:
    return self._mock.timestep_spec()

  def step(self, action: gdmr_types.ActionType) -> dm_env.TimeStep:  # pyrefly: ignore[invalid-type-var]
    return self._mock.step(action)

  def action_spec(self) -> gdmr_types.ActionSpec:  # pyrefly: ignore[invalid-type-var]
    return self._mock.action_spec()

  # We also need to implement `default_reset_options` as it is called on the
  # base env and not the mock otherwise.
  def default_reset_options(self) -> environment.ResetOptions:
    return self._mock.default_reset_options()


class EnvironmentTest(absltest.TestCase):

  def test_dmenv_reset_uses_correct_options(self):
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class MyOptions(environment.Options):
      my_option: str

    mock_env = mock.create_autospec(environment.Environment, instance=True)
    mock_env.default_reset_options.return_value = MyOptions(
        my_option="in the test"
    )

    env = _TestEnvironment(mock_env)
    env.reset()

    mock_env.reset_with_options.assert_called_once_with(
        options=MyOptions(my_option="in the test")
    )

  def test_timestep_spec_are_correct(self):
    mock_env = mock.create_autospec(environment.Environment, instance=True)
    mock_env.timestep_spec.return_value = gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        observation=specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-3.14, maximum=1.72
        ),
        reward=specs.Array(shape=(1,), dtype=np.float32),
        discount=specs.Array(shape=(2,), dtype=np.float16),
    )

    env = _TestEnvironment(mock_env)
    self.assertEqual(
        specs.BoundedArray(
            shape=(3,), dtype=np.float32, minimum=-3.14, maximum=1.72
        ),
        env.observation_spec(),
    )
    self.assertEqual(
        specs.Array(shape=(1,), dtype=np.float32), env.reward_spec()
    )
    self.assertEqual(
        specs.Array(shape=(2,), dtype=np.float16), env.discount_spec()
    )


if __name__ == "__main__":
  absltest.main()
