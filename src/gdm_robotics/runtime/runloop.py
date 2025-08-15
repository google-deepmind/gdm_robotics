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
"""Canonical runloop."""

from collections.abc import Callable, Iterable
import signal
import threading
import types
from typing import Any

from absl import logging
import dm_env
from gdm_robotics.interfaces import environment as gdmr_env
from gdm_robotics.interfaces import episodic_logger as gdmr_logger
from gdm_robotics.interfaces import policy as gdmr_policy

from gdm_robotics.runtime import scope


class RunloopRuntimeOperations:
  """Collection of operations to customise the runloop runtime behaviour."""

  @property
  def name(self) -> str:
    return type(self).__name__

  def before_episode_reset(self) -> bool:
    """Called to notify the runtime that a new episode is starting.

    This is called first thing at the beginning of an episode, before the
    environment is reset.

    Returns:
      True if the episode should be started, False if it should be terminated.
    """
    return True

  def inspect_timestep(self, timestep: dm_env.TimeStep) -> None:
    """Inspects the current timestep returned by the environment.

    This is called with the timestep returned by the environment reset and at
    each runloop step with the timestep returned by environment.step.

    Args:
      timestep: The environment timestep to inspect.
    """

    del timestep
    ...


class Runloop:
  """Runloop class."""

  def __init__(
      self,
      environment: gdmr_env.Environment,
      policy: gdmr_policy.Policy,
      loggers: Iterable[gdmr_logger.EpisodicLogger],
      *,
      handle_sigint: bool = False,
      runloop_runtime_operations: Iterable[RunloopRuntimeOperations] = (),
      reset_options_provider: Callable[[], gdmr_env.ResetOptions] | None = None,
  ):
    """Initializes the runloop.

    Args:
      environment: The environment to run the policy in.
      policy: The policy to run against the environment.
      loggers: The loggers to use to log the runloop data.
      handle_sigint: Whether the runloop object should automatically handle
        SIGINT by stopping the runloop.
      runloop_runtime_operations: The runtime operations to use to customize the
        runloop behaviour.
      reset_options_provider: A function that returns the reset options to use
        for the environment. If not provided, the environment's default reset
        options will be used.
    """
    self._environment = environment
    self._policy = policy
    self._loggers = list(loggers)
    self._runloop_runtime_operations = list(runloop_runtime_operations)
    self._reset_options_provider = (
        reset_options_provider
        if reset_options_provider
        else environment.default_reset_options
    )

    # Whether to (cleanly) stop the runloop.
    self._should_stop = threading.Event()
    if handle_sigint:

      def _handler(sig, frame):
        del sig, frame
        logging.info("Received SIGINT, stopping runloop.")
        self._should_stop.set()

      self._sigint_handler = _SigintHandler(_handler)
    else:
      self._sigint_handler = None

  def reset(self) -> None:
    """Resets the runloop state.

    Note that this does not directly reset the environment or the policy.
    """
    self._should_stop.clear()

  def stop(self) -> None:
    """Explicitly stops the runloop."""
    self._should_stop.set()

  
  def run_single_episode(self) -> None:
    """Runs a single episode."""
    self._run_episode()

  
  def _run_episode(self) -> bool:
    """Runs a single episode.

    Returns:
      true if the runloop should continue, false if it should stop.
    """
    # Notify the runtime operations that a new episode is starting and check if
    # they want to stop the runloop.
    with scope.Scope(name="RunloopRuntimeOperations.before_episode_reset"):
      for ops in self._runloop_runtime_operations:
        if not ops.before_episode_reset():
          logging.info(
              "Runtime operations %s requested to quit runloop.", ops.name
          )
          return False

    logging.info("Resetting environment.")

    with scope.Scope(name="Environment.reset"):
      timestep = self._environment.reset_with_options(
          options=self._reset_options_provider()
      )

    # Let the runtime operations examine the reset timestep.
    with scope.Scope(
        name="RunloopRuntimeOperations.inspect_timestep(reset)"
    ):
      for ops in self._runloop_runtime_operations:
        ops.inspect_timestep(timestep)

    with scope.Scope(name="Policy.initial_state"):
      policy_state = self._policy.initial_state()

    with scope.Scope(name="Logger.reset"):
      for logger in self._loggers:
        logger.reset(timestep)

    logging.info("Start episode loop.")
    with scope.Scope(name="EpisodeLoop"):
      while not timestep.last() and not self._should_stop.is_set():
        with scope.Scope(name="Policy.step"):
          (action, policy_extra), policy_state = self._policy.step(
              timestep, policy_state
          )
        with scope.Scope(name="Environment.step"):
          timestep = self._environment.step(action)

        # Let the runtime operations examine the timestep.
        with scope.Scope(
            name="RunloopRuntimeOperations.inspect_timestep(stepƒ)"
        ):
          for ops in self._runloop_runtime_operations:
            ops.inspect_timestep(timestep)

        with scope.Scope(name="Logger.record_action_and_next_timestep"):
          for logger in self._loggers:
            logger.record_action_and_next_timestep(
                action, timestep, policy_extra
            )

    logging.info("Episode terminated.")
    with scope.Scope(name="Logger.write"):
      for logger in self._loggers:
        logger.write()

    # No reasons to terminate. Return true as we expect a new episode to start.
    return True

  
  def run(self, num_episodes: int | None) -> None:
    """Runs the runloop for the specified number of episodes."""
    self._should_stop.clear()
    episode_count = 0
    while not self._should_stop.is_set():
      if num_episodes is not None and episode_count >= num_episodes:
        break

      logging.info("Running episode %s", episode_count + 1)
      # If the episode returns False, quit the runloop.
      if not self._run_episode():
        break
      episode_count += 1


class _SigintHandler:
  """Lightweight utility to call a function on SIGINT.

  The SIGINT handling will be removed once this object is deleted.
  """

  def __init__(self, handler_fn: Callable[[int, types.FrameType | None], Any]):
    """Initializer.

    NOTE: This must be constructed in the main thread of the python process, or
      a ValueError will be raised.

    Args:
      handler_fn: Function to call when SIGINT is received. This will be called
        from the main thread upon receiving a SIGINT, and should accept two
        arguments, an integer representing the signal received, and the current
        stack frame.
    """
    self._prev_sigint_signal = signal.signal(
        signal.SIGINT,
        handler_fn,
    )

  def __del__(self):
    if self._prev_sigint_signal:
      signal.signal(signal.SIGINT, self._prev_sigint_signal)
