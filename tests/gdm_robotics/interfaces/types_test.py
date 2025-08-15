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
import dm_env
from dm_env import specs
from gdm_robotics.interfaces import types
import numpy as np
import tree
from absl.testing import absltest


class GenerateTimestepFromSpecTest(absltest.TestCase):

  def test_generate_simple_timestep(self):
    """Tests generating a timestep with simple scalar specs."""
    spec = types.TimeStepSpec(
        step_type=types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32, name='reward'),
        discount=specs.BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name='discount',
        ),
        observation={
            'image': specs.Array(
                shape=(64, 64, 3), dtype=np.uint8, name='image'
            ),
            'state': specs.Array(shape=(4,), dtype=np.float64, name='state'),
        },
    )

    timestep = types.generate_valid_timestep_from_spec(spec)

    self.assertIsInstance(timestep, dm_env.TimeStep)
    self.assertEqual(timestep.step_type, dm_env.StepType.FIRST)

    # Validate reward, discount, and observation against their specs.
    spec.reward.validate(timestep.reward)
    spec.discount.validate(timestep.discount)
    tree.map_structure(
        lambda spec, value: spec.validate(value),
        spec.observation,
        timestep.observation,
    )

  def test_generate_nested_timestep(self):
    """Tests generating a timestep with nested observation specs."""
    spec = types.TimeStepSpec(
        step_type=types.STEP_TYPE_SPEC,
        reward=specs.Array(shape=(), dtype=np.float32, name='reward'),
        discount=specs.BoundedArray(
            shape=(),
            dtype=np.float32,
            minimum=0.0,
            maximum=1.0,
            name='discount',
        ),
        observation={
            'pixels': {
                'front': specs.Array(shape=(128, 128, 3), dtype=np.uint8),
                'wrist': specs.Array(shape=(64, 64, 3), dtype=np.uint8),
            },
            'proprio': specs.Array(shape=(7,), dtype=np.float64),
        },
    )

    timestep = types.generate_valid_timestep_from_spec(spec)

    self.assertIsInstance(timestep, dm_env.TimeStep)
    self.assertEqual(timestep.step_type, dm_env.StepType.FIRST)
    tree.map_structure(
        lambda spec, value: spec.validate(value),
        spec.observation,
        timestep.observation,
    )


if __name__ == '__main__':
  absltest.main()
