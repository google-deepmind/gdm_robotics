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
"""Wrapper to convert a Gym environment to a GDMR environment.

This wrapper is useful for environments that are implemented using the Gym
interface and need to be used with code that expects GDMR environments.

This wrapper will not overcome the limitations of the Gym interface, for example
a single scalar reward or the absence of a discount.
"""

import dataclasses
from typing import Any

import dm_env
from dm_env import specs
from gdm_robotics.interfaces import environment as gdmr_environment
from gdm_robotics.interfaces import types as gdmr_types
import gymnasium
from gymnasium import spaces
import numpy as np
import tree
from typing_extensions import override


@dataclasses.dataclass(frozen=True, kw_only=True)
class GymnasiumEnvResetOptions(gdmr_environment.Options):
  """Options for the GymnasiumEnvToGdmrEnvWrapper."""

  seed: int | None = None
  options: dict[str, Any] | None = None


class GymnasiumEnvToGdmrEnvWrapper(
    gdmr_environment.Environment[GymnasiumEnvResetOptions]
):
  """Wraps a gymnasium.Env to a gdmr_environment.Environment interface.

  This class provides a GDMR-compatible interface for Gymnasium environments.
  It properly converts the Gymnasium action and observation spaces to
  corresponding GDMR specs.

  Note that:
  - Reward will always be a single unbounded float scalar.
  - There is no discount specification in Gymnasium, so the discount will be a
    single bounded [0.0, 1.0] float scalar. Its value will be:
    - 0.0 in reset()
    - 1.0 in step() if step_type is MID.
    - 0.0 in step() if episode is terminated.
    - 1.0 in step() if episode is truncated.
  - If the underlying environment returns True to both truncated and terminated,
    the step will be considered truncated.

  `render` is not forwarded as they are not part of the GDMR Environment
  interface.
  """

  def __init__(self, env: gymnasium.Env):
    """Initializes the wrapper.

    Args:
      env: The gymnasium.Env instance to wrap.
    """
    if not isinstance(env, gymnasium.Env):
      raise TypeError(
          'Wrapped environment must be an instance of gymnasium.Env.'
      )
    self._env = env
    # Store specs for timestep_spec construction.
    self._action_spec = convert_gym_space_to_spec(self._env.action_space)

    # The action spec does not support generic Spec but wants an UnboundedSpec
    # if no bounds are specified.
    def _is_concrete_spec(s: specs.Array) -> bool:
      return (
          isinstance(s, specs.BoundedArray)
          or isinstance(s, specs.DiscreteArray)
          or isinstance(s, specs.StringArray)
      )

    self._action_spec = tree.map_structure(
        lambda s: gdmr_types.UnboundedArraySpec(s.shape, s.dtype, s.name)
        if not _is_concrete_spec(s)
        else s,
        self._action_spec,
    )
    self._observation_spec = convert_gym_space_to_spec(
        self._env.observation_space
    )

    # Reward is not specified but should be a float.
    self._reward_spec = specs.Array(shape=(), dtype=np.float32)

    # There is no discount in Gymnasium.
    self._discount_spec = specs.Array(shape=(), dtype=np.float32)

    self._zero_reward = np.array(0.0, dtype=np.float32)
    self._zero_discount = np.array(0.0, dtype=np.float32)
    self._default_discount = np.array(1.0, dtype=np.float32)

  @override
  def default_reset_options(self) -> GymnasiumEnvResetOptions:
    """Returns the default ResetOptions."""
    return GymnasiumEnvResetOptions()

  @override
  def reset_with_options(
      self, *, options: GymnasiumEnvResetOptions
  ) -> dm_env.TimeStep:
    """Resets the underlying Gymnasium environment.

    Args:
      options: Reset options. If None, the default options will be used.

    Returns:
      The first TimeStep from the underlying environment.
    """
    observation, _ = self._env.reset(seed=options.seed, options=options.options)
    return dm_env.TimeStep(
        step_type=np.asarray(
            dm_env.StepType.FIRST, dtype=gdmr_types.STEP_TYPE_SPEC.dtype
        ),
        reward=self._zero_reward,
        discount=self._zero_discount,
        observation=self._ensure_observation_is_array(observation),
    )

  @override
  def step(self, action: gdmr_types.ActionType) -> dm_env.TimeStep:
    """Steps the underlying gymnasium environment."""
    observation, reward, terminated, truncated, _ = self._env.step(action)

    if truncated:
      step_type = dm_env.StepType.LAST
      discount = self._default_discount
    elif terminated:
      step_type = dm_env.StepType.LAST
      discount = self._zero_discount
    else:
      step_type = dm_env.StepType.MID
      discount = self._default_discount

    step_type = np.asarray(step_type, dtype=gdmr_types.STEP_TYPE_SPEC.dtype)
    reward = np.asarray(reward, dtype=self._reward_spec.dtype)

    return dm_env.TimeStep(
        step_type=step_type,
        reward=reward,
        discount=discount,
        observation=self._ensure_observation_is_array(observation),
    )

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
    """Closes the underlying gym environment."""
    self._env.close()

  @property
  def wrapped_env(self) -> gymnasium.Env:
    return self._env

  def _ensure_observation_is_array(
      self, observation: tree.Structure[Any]
  ) -> tree.Structure[np.ndarray]:
    """Ensures that the observation is an array."""
    tree.assert_same_structure(observation, self._observation_spec)
    return tree.map_structure(
        lambda v, s: np.asarray(v, dtype=s.dtype),
        observation,
        self._observation_spec,
    )


def convert_gym_space_to_spec(
    gym_space: gymnasium.Space,
    name: str | None = None,
) -> tree.Structure[specs.Array]:
  """Converts a Gym space to a GDMR-compatible spec."""
  if isinstance(gym_space, spaces.Discrete):
    return specs.DiscreteArray(
        num_values=gym_space.n, dtype=gym_space.dtype, name=name
    )
  elif isinstance(gym_space, spaces.Box):
    return specs.BoundedArray(
        shape=gym_space.shape,
        dtype=gym_space.dtype,
        minimum=gym_space.low,
        maximum=gym_space.high,
        name=name,
    )
  elif isinstance(gym_space, spaces.MultiBinary):
    return specs.BoundedArray(
        shape=gym_space.shape,
        dtype=gym_space.dtype,
        minimum=0.0,
        maximum=1.0,
        name=name,
    )
  elif isinstance(gym_space, spaces.MultiDiscrete):
    return specs.BoundedArray(
        shape=gym_space.shape,
        dtype=gym_space.dtype,
        minimum=np.zeros(gym_space.shape),
        maximum=gym_space.nvec,
        name=name,
    )
  elif isinstance(gym_space, spaces.Tuple):
    return tuple(convert_gym_space_to_spec(s) for s in gym_space.spaces)
  elif isinstance(gym_space, spaces.Dict):
    return {
        key: convert_gym_space_to_spec(value, key)
        for key, value in gym_space.spaces.items()
    }
  elif isinstance(gym_space, spaces.Text):
    return specs.StringArray(
        shape=(),
        name=name,
    )
  else:
    raise ValueError(f'Unsupported gym space: {gym_space}')
  pass
