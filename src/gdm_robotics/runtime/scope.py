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
"""Context manager for creating a new scope.

This class can be used as context manager to "create" a new code scope, i.e. to
group a set of code lines by indenting them under the "scope" object.
There is no other use for this, and should not be used outside the code
deployed by Google DeepMind.

For example:

some_var = 1
with Scope("scope_1"):
  some_var += 2
  print(some_var)
  call_foo()
...
"""


class Scope:
  """Context manager for creating a new scope."""

  def __init__(self, name: str):
    self._name = name

  def __enter__(self):
    return self._name

  def __exit__(self, exc_type, exc_value, traceback):
    pass
