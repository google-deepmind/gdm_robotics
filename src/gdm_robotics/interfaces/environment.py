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
"""Robotics RL environments API."""

import abc
import dataclasses
import types
from typing import Generic, TypeVar, final
from typing_extensions import Self

import dm_env
from gdm_robotics.interfaces import types as gdmr_types


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
  """Specific environment reset options."""


ResetOptions = TypeVar('ResetOptions', bound=Options)


class Environment(dm_env.Environment, Generic[ResetOptions]):
  """Abstract base class for Robotics RL environments.

  The differences with dm_env.Environment are the following:
  - New `reset_with_options` reset method accepting a generic option class. The
    `dm_env.Environment.reset` method forward the call to `reset_with_options`
    using the default reset option (i.e. as returned by
    `default_reset_options`).
  - A `default_reset_options` method returning the default `ResetOptions`.
  - A `timestep_spec` method returning the full spec for the `TimeSpec`.
    Observation, reward and discount spec are automatically implemented using
    this method.
  """

  def default_reset_options(self) -> ResetOptions:
    """Returns the default ResetOptions."""
    return Options()

  @final
  def reset(self) -> dm_env.TimeStep:
    """Starts a new sequence and returns the first `TimeStep`."""
    return self.reset_with_options(options=self.default_reset_options())

  @abc.abstractmethod
  def reset_with_options(
      self,
      *,
      options: ResetOptions,
  ) -> dm_env.TimeStep:
    """Starts a new sequence and returns the first `TimeStep`."""

  @abc.abstractmethod
  def timestep_spec(self) -> gdmr_types.TimeStepSpec:
    """Returns the spec associated to the returned TimeStep.

    You can use the constant `STEP_TYPE_SPEC` (defined in this module) to
    specify the spec of the `step_type` field in the `TimeStepSpec` object.
    """

  # Redefine the methods of dm_env.Environment to add typing support. This is
  # necessary until typing is added to dm_env.Environment and the following
  # methods could be removed in the future.

  @abc.abstractmethod
  def step(self, action: gdmr_types.ActionType) -> dm_env.TimeStep:
    """Updates the environment according to action and returns a `TimeStep`."""

  @abc.abstractmethod
  def action_spec(self) -> gdmr_types.ActionSpec:
    """Defines the actions that should be provided to `step`."""

  @final
  def observation_spec(self) -> gdmr_types.ObservationSpec:
    """Defines the observations provided by the environment."""
    return self.timestep_spec().observation

  @final
  def reward_spec(self) -> gdmr_types.RewardSpec:  # pytype: disable=signature-mismatch
    """Describes the reward returned by the environment."""
    return self.timestep_spec().reward

  @final
  def discount_spec(self) -> gdmr_types.DiscountSpec:  # pytype: disable=signature-mismatch
    """Describes the discount returned by the environment."""
    return self.timestep_spec().discount

  def close(self) -> None:
    """Frees any resources used by the environment.

    Implement this method for an environment backed by an external process.

    This method can be used directly

    ```python
    env = Env(...)
    # Use env.
    env.close()
    ```

    or via a context manager

    ```python
    with Env(...) as env:
      # Use env.
    ```
    """

  def __enter__(self) -> Self:
    """Allows the environment to be used in a with-statement context."""
    return self

  def __exit__(
      self,
      exc_type: type[BaseException] | None,
      exc_value: BaseException | None,
      traceback: types.TracebackType | None,
  ) -> None:
    """Allows the environment to be used in a with-statement context."""
    del exc_type, exc_value, traceback  # Unused.
    self.close()
