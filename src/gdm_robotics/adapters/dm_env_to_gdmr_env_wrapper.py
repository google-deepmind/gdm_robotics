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
"""Wrapper to convert a dm_env.Environment to a gdmr_environment.Environment."""

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import environment as gdmr_environment
from gdm_robotics.interfaces import types as gdmr_types
import numpy as np
import tree
from typing_extensions import override


class DmEnvToGdmrEnvWrapper(
    gdmr_environment.Environment[gdmr_environment.ResetOptions]
):
  """Wraps a dm_env.Environment to a gdmr_environment.Environment interface.

  The GDMR Environment interface is very similar to the dm_env.Environment
  interface (indeed it inherits from dm_env.Environment) but it enforces a
  stricter type checking, e.g. all TimeStep elements are Numpy-like arrays.
  """

  def __init__(self, env: dm_env.Environment):
    """Initializes the wrapper.

    Args:
      env: The dm_env.Environment instance to wrap.
    """
    if not isinstance(env, dm_env.Environment):
      raise TypeError(
          'Wrapped environment must be an instance of dm_env.Environment.'
      )
    self._env = env
    # Store specs for timestep_spec construction.
    self._action_spec = self._env.action_spec()
    self._observation_spec = self._env.observation_spec()
    self._reward_spec = self._env.reward_spec()
    self._discount_spec = self._env.discount_spec()

    # Save the zero-like reward and discount values.
    def _zero_like_spec(spec: specs.Array) -> gdmr_types.ArrayType:
      arr = np.zeros(spec.shape, dtype=spec.dtype)
      arr.flags.writeable = False
      return arr

    self._zero_reward = tree.map_structure(_zero_like_spec, self._reward_spec)
    self._zero_discount = tree.map_structure(
        _zero_like_spec, self._discount_spec
    )

  @override
  def reset_with_options(
      self, *, options: gdmr_environment.ResetOptions
  ) -> dm_env.TimeStep:
    """Resets the underlying dm_env environment.

    Note: The `options` argument is ignored as dm_env.Environment.reset()
    does not accept options.

    Args:
      options: Reset options (ignored).

    Returns:
      The first TimeStep from the underlying environment.
    """
    del options  # dm_env.Environment.reset doesn't take options.
    # Force the reward and discount to be type-correct.
    timestep = self._env.reset()
    timestep = timestep._replace(
        reward=self._zero_reward, discount=self._zero_discount
    )
    return self._enforce_timestep(timestep)

  @override
  def step(self, action: gdmr_types.ActionType) -> dm_env.TimeStep:
    """Steps the underlying dm_env environment."""
    return self._enforce_timestep(self._env.step(action))

  @override
  def action_spec(self) -> gdmr_types.ActionSpec:
    """Returns the action spec from the underlying environment."""
    return self._action_spec

  @override
  def timestep_spec(self) -> gdmr_types.TimeStepSpec:
    """Constructs and returns the TimeStepSpec."""
    return gdmr_types.TimeStepSpec(
        step_type=gdmr_types.STEP_TYPE_SPEC,
        reward=self._reward_spec,
        discount=self._discount_spec,
        observation=self._observation_spec,
    )

  # observation_spec, reward_spec, and discount_spec are automatically
  # implemented by the gdmr_environment.Environment base class using
  # timestep_spec.

  @override
  def close(self) -> None:
    """Closes the underlying dm_env environment."""
    self._env.close()

  @property
  def wrapped_env(self) -> dm_env.Environment:
    return self._env

  def _enforce_timestep(self, timestep: dm_env.TimeStep) -> dm_env.TimeStep:
    """Enforces the timestep to be type-correct."""
    # Check that the reward and discount are consistent with the specs.
    tree.assert_same_structure(timestep.reward, self._reward_spec)
    tree.assert_same_structure(timestep.discount, self._discount_spec)
    tree.assert_same_structure(timestep.observation, self._observation_spec)

    timestep = timestep._replace(
        step_type=np.asarray(
            timestep.step_type, dtype=gdmr_types.STEP_TYPE_SPEC.dtype
        ),
        reward=tree.map_structure(
            lambda v, s: np.asarray(v, dtype=s.dtype),
            timestep.reward,
            self._reward_spec,
        ),
        discount=tree.map_structure(
            lambda v, s: np.asarray(v, dtype=s.dtype),
            timestep.discount,
            self._discount_spec,
        ),
        observation=tree.map_structure(
            lambda v, s: np.asarray(v, dtype=s.dtype),
            timestep.observation,
            self._observation_spec,
        ),
    )
    return timestep
