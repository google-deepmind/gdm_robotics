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
"""MultiDiscrete specs must preserve Gymnasium's exact integer bounds."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_env import specs
from gdm_robotics.adapters import gymnasium_env_to_gdmr_env_wrapper as adapters
from gymnasium import spaces
import numpy as np


class MultiDiscreteBoundsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ("default_start", [5, 3], None, [0, 0], [4, 2]),
      ("offset_start", [5, 3], [2, -3], [2, -3], [6, -1]),
      ("singleton_range", [1, 2], [4, 0], [4, 0], [4, 1]),
  )
  def test_conversion_preserves_inclusive_bounds(
      self, nvec, start, minimum, maximum
  ):
    kwargs = {} if start is None else {"start": np.asarray(start)}
    space = spaces.MultiDiscrete(nvec, **kwargs)
    converted = adapters.convert_gym_space_to_spec(space)
    self.assertIsInstance(converted, specs.BoundedArray)
    self.assertEqual(converted.shape, space.shape)
    self.assertEqual(converted.dtype, space.dtype)
    np.testing.assert_array_equal(converted.minimum, minimum)
    np.testing.assert_array_equal(converted.maximum, maximum)

  def test_converted_spec_does_not_accept_outside_values(self):
    space = spaces.MultiDiscrete([2, 3])
    converted = adapters.convert_gym_space_to_spec(space)
    self.assertTrue(space.contains(np.asarray([1, 2])))
    self.assertFalse(space.contains(np.asarray([2, 3])))
    self.assertTrue(converted.validate(np.asarray([1, 2], dtype=space.dtype)) is not None)
    with self.assertRaises(ValueError):
      converted.validate(np.asarray([2, 3], dtype=space.dtype))

  def test_conversion_supports_multidimensional_nvec(self):
    space = spaces.MultiDiscrete(np.asarray([[2, 1], [3, 4]]))
    converted = adapters.convert_gym_space_to_spec(space)
    self.assertEqual(converted.shape, (2, 2))
    np.testing.assert_array_equal(converted.maximum, [[1, 0], [2, 3]])


if __name__ == "__main__":
  absltest.main()
