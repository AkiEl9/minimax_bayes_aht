from typing import Type

from ray.rllib import MultiAgentEnv, SampleBatch
from gymnasium.spaces import Dict, Discrete

def ScenarioWrapper(env_cls: Type[MultiAgentEnv]) -> Type[MultiAgentEnv]:
    """
    Multi agent env wrapper
    """

    class ScenarioWrapper(env_cls):

        def __init__(self, *args, num_scenarios=1, **kwargs):
            super().__init__(*args, **kwargs)

            base = self.observation_space[0]
            self.observation_space = Dict({
                agent_id : Dict({
                    SampleBatch.OBS: base,
                    "scenario": Discrete(num_scenarios)
                })
                for agent_id in self._agent_ids
            })
            self.current_scenario_id = 0

            self._obs_space_in_preferred_format = True


        def update_observations(self, observations):
            return {
                agent_id: {
                    SampleBatch.OBS: observations[agent_id],
                    "scenario": self.current_scenario_id
                }
                for agent_id in self._agent_ids
            }

        def step(self, actions):
            observations, rewards, dones, truncs, infos = super().step(actions)

            updated_observations = self.update_observations(observations)


            return updated_observations, rewards, dones, truncs, infos

        def reset(self, *args, **kwargs):
            observations, infos = super().reset(*args, **kwargs)
            return self.update_observations(observations), infos

    ScenarioWrapper.__name__ = env_cls.__name__
    ScenarioWrapper.__qualname__ =  env_cls.__name__

    return ScenarioWrapper

def ScenarioWrapperDict(env_cls: Type[MultiAgentEnv]) -> Type[MultiAgentEnv]:
    """
    Multi agent env wrapper
    """

    class ScenarioWrapper(env_cls):

        def __init__(self, *args, num_scenarios=1, **kwargs):
            super().__init__(*args, **kwargs)

            spaces: dict = self.observation_space.spaces
            new_spaces = {}
            for agent_id, space in spaces.items():
                d = space.spaces
                d.update(
                    scenario=Discrete(num_scenarios)
                )
                new_spaces[agent_id] = Dict(d)
            self.observation_space = Dict(new_spaces)
            self.current_scenario_id = 0
            self._obs_space_in_preferred_format = True


        def update_observations(self, observations):
            for agent_id in self.get_agent_ids():
                observations[agent_id].update(
                    scenario=self.current_scenario_id
                )
            return observations

        def step(self, actions):
            observations, rewards, dones, truncs, infos = super().step(actions)

            updated_observations = self.update_observations(observations)


            return updated_observations, rewards, dones, truncs, infos

        def reset(self, *args, **kwargs):
            observations, infos = super().reset(*args, **kwargs)
            return self.update_observations(observations), infos

    ScenarioWrapper.__name__ = env_cls.__name__
    ScenarioWrapper.__qualname__ =  env_cls.__name__

    return ScenarioWrapper