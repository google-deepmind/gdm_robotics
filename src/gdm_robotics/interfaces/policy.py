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
"""Interface representing an agent stateless policy."""

import abc
from typing import Generic

import dm_env
from gdm_robotics.interfaces import types as gdmr_types


class Policy(abc.ABC, Generic[gdmr_types.PolicyDataType]):
  """Base class for stateless policies.

  Note: this interface is for stateless policies. This means that the state must
  be explicitly passed as input parameter to the `step` function and the next
  state is explicitly returned as one of its returned parameters.
  """

  @abc.abstractmethod
  def step(
      self,
      timestep: dm_env.TimeStep,
      prev_state: gdmr_types.StateStructure[gdmr_types.PolicyDataType],
  ) -> tuple[
      tuple[
          gdmr_types.ActionType,
          gdmr_types.ExtraOutputStructure[gdmr_types.PolicyDataType],
      ],
      gdmr_types.StateStructure[gdmr_types.PolicyDataType],
  ]:
    """Takes a step with the policy given an environment timestep.

    Args:
      timestep: An instance of environment `TimeStep`.
      prev_state: The state of the policy at the time of calling `step`.

    Returns:
      A tuple of ((action, extra), state) with `action` indicating the action to
      be executed, `extra` policy-specific auxiliary information and state the
      policy state that will need to be provided to the next call to `step`.
    """

  @abc.abstractmethod
  def initial_state(
      self,
  ) -> gdmr_types.StateStructure[gdmr_types.PolicyDataType]:
    """Returns the policy initial state."""

  @abc.abstractmethod
  def step_spec(self, timestep_spec: gdmr_types.TimeStepSpec) -> tuple[
      tuple[gdmr_types.ActionSpec, gdmr_types.ExtraOutputSpec],
      gdmr_types.StateSpec,
  ]:
    """Returns the spec of the ((action, extra), state) from `step` method."""
