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
"""Functions for generating random specs and values."""

from collections.abc import Mapping, Sequence
import dataclasses
import random
import string
import dm_env
from dm_env import specs
import numpy as np
import tree


@dataclasses.dataclass(frozen=True)
class _Unspecified:
  """Placeholder used as a sentinel when None is an acceptable value."""


def random_one_dimensional_array(length: int | None = None) -> np.ndarray:
  if length is None:
    length = random.randint(1, 10)
  return np.random.random(size=(length,))


def random_string(length: int | None = None) -> str:
  length = length or random.randint(5, 10)
  return "".join(random.choice(string.ascii_letters) for _ in range(length))


def random_shape(ndims: int | None = None) -> tuple[int, ...]:
  ndims = ndims or random.randint(1, 3)
  return tuple([random.randint(1, 10) for _ in range(ndims)])


def random_dtype() -> type[np.floating]:
  return random.choice([float, np.float32, np.float64])


def unit_array_spec(
    shape: tuple[int, ...] | None = None, name: str | None = None
) -> specs.BoundedArray:
  shape = shape or random_shape()
  dtype = random.choice([np.float32, np.float64])
  minimum = np.zeros(shape=shape, dtype=dtype)
  maximum = np.ones(shape=shape, dtype=dtype)
  name = name or random_string()
  return specs.BoundedArray(shape, dtype, minimum, maximum, name)


def random_array_spec(
    shape: tuple[int, ...] | None = None,
    name: str | None = None,
    dtype: type[np.floating] | None = None,
    minimum: np.ndarray | None = None,
    maximum: np.ndarray | None = None,
) -> specs.BoundedArray:
  """Create BoundedArray spec with unspecified parts randomized."""

  shape = shape or random_shape()
  name = name or random_string()
  dtype = dtype or random.choice([np.float32, np.float64])
  if minimum is None:
    minimum = np.random.random(size=shape) * random.randint(0, 10)
  minimum = minimum.astype(dtype)
  if maximum is None:
    maximum = np.random.random(size=shape) * random.randint(0, 10) + minimum
  maximum = maximum.astype(dtype)
  return specs.BoundedArray(shape, dtype, minimum, maximum, name)


def valid_primitive_value(spec: specs.Array) -> np.ndarray:
  """Returns a random value that adheres to the spec."""
  if isinstance(spec, specs.StringArray):
    return np.array(random_string(), dtype=object)
  value = np.random.random(size=spec.shape).astype(spec.dtype)
  if isinstance(spec, specs.BoundedArray):
    # Clip specs to handle +/- np.inf in the specs. Cast the spec as the same
    # dtype as the value.
    maximum = np.clip(spec.maximum, -1e10, 1e10).astype(spec.dtype)
    minimum = np.clip(spec.minimum, -1e10, 1e10).astype(spec.dtype)
    value *= maximum - minimum
    value += minimum
  else:
    if np.issubdtype(spec.dtype, np.floating):
      value *= 1e10  # Make range / magnitude assumptions unlikely to hold.
  return value.astype(spec.dtype)


def valid_dict_value(spec: Mapping[str, specs.Array]) -> dict[str, np.ndarray]:
  """Returns a value that adheres to the spec."""
  return {k: valid_primitive_value(v) for k, v in spec.items()}


def valid_value_for_spec(
    spec: tree.Structure[specs.Array],
) -> tree.Structure[np.ndarray]:
  """Returns a value that adheres to the spec."""
  return tree.map_structure(valid_primitive_value, spec)


def zero_value(spec: specs.Array) -> np.ndarray:
  """Returns a value that adheres to the spec."""
  return np.zeros(shape=spec.shape, dtype=spec.dtype)


def random_image(dims: Sequence[int] | None = None) -> np.ndarray:
  if dims is None:
    dims = (random.randint(1, 10), random.randint(1, 10), 3)
  assert len(dims) == 3
  assert dims[2] == 3
  return np.random.random(size=dims).astype(np.float32)


def random_step_type() -> dm_env.StepType:
  return random.choice([
      dm_env.StepType.FIRST,
      dm_env.StepType.MID,
      dm_env.StepType.LAST,
  ])


def empty_timestep(step_type: dm_env.StepType | None = None) -> dm_env.TimeStep:
  return dm_env.TimeStep(
      step_type=step_type or random_step_type(),
      reward=None,
      discount=None,
      observation={},
  )


def random_timestep(
    *,
    step_type: dm_env.StepType | None = None,
    reward: float | np.ndarray | None | _Unspecified = _Unspecified(),
    discount: float | np.ndarray | None | _Unspecified = _Unspecified(),
    observation: dict[str, np.ndarray] | None = None,
) -> dm_env.TimeStep:
  """Returns a random timestep that adheres to the spec."""
  if step_type is None:
    step_type = random_step_type()
  if isinstance(reward, _Unspecified):
    reward = random_one_dimensional_array()
  if isinstance(discount, _Unspecified):
    discount = random_one_dimensional_array()
  if observation is None:
    observation = random_observation()

  return dm_env.TimeStep(
      step_type=step_type,
      reward=reward,
      discount=discount,
      observation=observation,
  )


def random_observation(*keys: str, **kwargs) -> dict[str, np.ndarray]:
  """Returns a random observation.

  Args:
    *keys: Names of observations create.
    **kwargs: supported kwargs are: size - number of observations to create
      (ignored if keys are provided).
  """

  if keys:
    return {key: valid_primitive_value(random_array_spec()) for key in keys}

  if "size" in kwargs:
    size = kwargs["size"]
  else:
    size = random.randint(1, 10)

  return {
      random_string(): valid_primitive_value(random_array_spec())
      for _ in range(size)
  }
