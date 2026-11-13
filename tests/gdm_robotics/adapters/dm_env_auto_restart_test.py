# Copyright 2026 Google LLC
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
"""Test automatic dm_env episode restarts through the robotics adapter."""

from absl.testing import absltest
from absl.testing import parameterized
import dm_env
from dm_env import specs
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper
import numpy as np


class AutoResetEnvironment(dm_env.Environment):
  """An environment that returns FIRST on step() after LAST."""

  def __init__(self, nested_reward=False, truncated=False):
    self.nested_reward = nested_reward
    self.truncated = truncated
    self.episode = 0
    self.steps = 0

  def reset(self):
    self.episode += 1
    self.steps = 0
    return dm_env.restart(np.array([self.episode], dtype=np.int32))

  def step(self, action):
    del action
    if self.steps >= 2 or self.episode == 0:
      return self.reset()
    self.steps += 1
    value = np.float32(self.steps)
    reward = {'cost': np.array([-value], dtype=np.float32),
              'benefit': np.asarray(value, dtype=np.float64)
              } if self.nested_reward else value
    obs = np.array([self.episode], dtype=np.int32)
    if self.steps == 2:
      if self.truncated:
        return dm_env.truncation(reward, obs, discount=0.75)
      return dm_env.termination(reward, obs)
    return dm_env.transition(reward, obs, discount=0.9)

  def observation_spec(self):
    return specs.Array((1,), np.int32)

  def action_spec(self):
    return specs.DiscreteArray(2)

  def reward_spec(self):
    return ({
        'cost': specs.Array((1,), np.float32),
        'benefit': specs.Array((), np.float64),
    } if self.nested_reward else specs.Array((), np.float32))

  def discount_spec(self):
    return specs.BoundedArray((), np.float32, minimum=0.0, maximum=1.0)

  def close(self):
    pass


class AutoResetAdapterTest(parameterized.TestCase):

  @parameterized.product(nested_reward=(False, True),
                         truncated=(False, True))
  def test_step_after_last_returns_typed_first(
      self, nested_reward, truncated
  ):
    source = AutoResetEnvironment(
        nested_reward=nested_reward, truncated=truncated
    )
    adapter = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(source)
    for episode in (1, 2, 3):
      if episode == 1:
        first = adapter.reset()
      else:
        first = adapter.step(np.int64(0))
      self.assertTrue(first.first())
      self.assertEqual(first.observation.tolist(), [episode])
      if nested_reward:
        np.testing.assert_array_equal(first.reward['cost'], [0.0])
        np.testing.assert_array_equal(first.reward['benefit'], 0.0)
        self.assertEqual(first.reward['cost'].dtype, np.float32)
        self.assertEqual(first.reward['benefit'].dtype, np.float64)
      else:
        np.testing.assert_array_equal(first.reward, 0.0)
        self.assertEqual(first.reward.dtype, np.float32)
      np.testing.assert_array_equal(first.discount, 0.0)
      self.assertEqual(first.discount.dtype, np.float32)

      mid = adapter.step(np.int64(1))
      self.assertTrue(mid.mid())
      self.assertAlmostEqual(float(mid.discount), 0.9, places=6)
      last = adapter.step(np.int64(1))
      self.assertTrue(last.last())
      self.assertAlmostEqual(float(last.discount), 0.75 if truncated else 0.0, places=6)

  def test_reset_converts_first_and_preserves_observation(self):
    source = AutoResetEnvironment()
    adapter = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(source)
    first = adapter.reset()
    self.assertTrue(first.first())
    self.assertEqual(first.observation.tolist(), [1])
    self.assertEqual(first.reward.dtype, np.float32)
    self.assertEqual(first.discount.dtype, np.float32)


if __name__ == '__main__':
  absltest.main()
