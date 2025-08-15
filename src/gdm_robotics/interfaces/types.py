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
"""Types used throughout GDMR interfaces."""

from typing import final, NamedTuple, TypeVar

import dm_env
from dm_env import specs
import numpy as np
import tree


# Action spec type definition.
@final
class UnboundedArraySpec(specs.Array):
  """Explicit unbounded array spec to be used instead of specs.Array."""


# We define `AnyArraySpec`, a new array spec type that is a strict subset of
# `specs.Array`. This new spec ensures the use of the more explicit
# `UnboundedArraySpec` type for unbounded arrays, limiting the possible types
# that are accepted to a controlled set of `specs.Array` subclasses.
AnyArraySpec = TypeVar(
    'AnyArraySpec',
    UnboundedArraySpec,
    specs.BoundedArray,
    specs.DiscreteArray,
)

# Action type definition.
ArrayType = TypeVar('ArrayType', bound=np.typing.ArrayLike)
ActionType = tree.Structure[ArrayType]

# Policy state and extra output type definition.
PolicyDataType = TypeVar('PolicyDataType')
# These will be generic on the PolicyDataType in the Policy interface class.
ExtraOutputStructure = tree.Structure
StateStructure = tree.Structure

# Environment Spec types.
ObservationSpec = tree.Structure[specs.Array]
RewardSpec = tree.Structure[specs.Array]
DiscountSpec = tree.Structure[specs.Array]
ActionSpec = tree.Structure[AnyArraySpec]

# Spec of `step_type` in TimeStep.
STEP_TYPE_SPEC = specs.BoundedArray(
    shape=(),
    dtype=np.uint8,
    minimum=min(dm_env.StepType),
    maximum=max(dm_env.StepType),
    name='step_type',
)


# NamedTuple for compatibility with tree.map_structure calls.
class TimeStepSpec(NamedTuple):
  step_type: specs.BoundedArray
  reward: RewardSpec
  discount: DiscountSpec
  observation: ObservationSpec


# Policy Spec type.
ExtraOutputSpec = ExtraOutputStructure[specs.Array]
StateSpec = StateStructure[specs.Array]


def generate_valid_timestep_from_spec(spec: TimeStepSpec) -> dm_env.TimeStep:
  """Generates a dm_env.TimeStep conforming to the TimeStepSpec.

  Args:
    spec: The TimeStepSpec to generate a TimeStep for.

  Returns:
    A dm_env.TimeStep object with values generated according to the spec.
    The step_type is defaulted to dm_env.StepType.FIRST.
  """
  # Default to FIRST for generated steps.
  step_type = np.array(dm_env.StepType.FIRST, dtype=np.uint8)
  reward = tree.map_structure(lambda s: s.generate_value(), spec.reward)
  discount = tree.map_structure(lambda s: s.generate_value(), spec.discount)
  observation = tree.map_structure(
      lambda s: s.generate_value(), spec.observation
  )
  return dm_env.TimeStep(step_type, reward, discount, observation)
