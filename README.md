# `gdm_robotics`: The Google DeepMind Robotics interfaces

This package describes a set of interfaces for Python reinforcement learning
(RL) environments. It consists of the following core components:

*   `gdm_robotics.interfaces.Environment`: An abstract base class for RL environments.
*   `gdm_robotics.interfaces.Policy`: An abstract base class for Agent policies.
*   `gdm_robotics.interfaces.EpisodicLogger`: An abstract base class for loggers for Agent/Environment interaction.
*   `gdm_robotics.runtime.RunLoop`: A concrete RunLoop class to run a policy against an environment and logging their interaction.

## Adapters

Common RL environment libraries such as `dm_env.Environment` and `gymnasium.Env`
can be exposed as `gdm_robotics.interfaces.Environment`s by using the provided
environment wrappers in the `adapter` sub-package, e.g. to wrap a
`dm_env.Environment` object:

```py
from gdm_robotics.adapters import dm_env_to_gdmr_env_wrapper

original_env: dm_env.Environment = ...
env = dm_env_to_gdmr_env_wrapper.DmEnvToGdmrEnvWrapper(original_env)
```

## Installation

`gdm_robotics` can be installed from PyPI using `pip`:

```bash
pip install gdm_robotics
```

## Licence and Disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you
may not use this file except in compliance with the Apache 2.0 license. You may
obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing 
permissions and limitations under those licenses.

This is not an official Google product.
