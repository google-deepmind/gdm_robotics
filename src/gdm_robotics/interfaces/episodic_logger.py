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
"""Interface for the episodic logger."""

from collections.abc import Mapping
from typing import Any, Protocol

import dm_env
from gdm_robotics.interfaces import types as gdmr_types


class EpisodicLogger(Protocol):
  """Protocol for an episodic logger."""

  def reset(self, timestep: dm_env.TimeStep) -> None:
    """Logs a reset TimeStep, i.e. a timestep with step_type == FIRST."""
    ...

  def record_action_and_next_timestep(
      self,
      action: gdmr_types.ActionType,
      next_timestep: dm_env.TimeStep,
      policy_extra: Mapping[str, Any],
  ) -> None:
    """Logs an action and the resulting timestep."""
    ...

  def write(self) -> None:
    """Writes the current episode logged data."""
    ...
