import itertools
import os
from copy import copy
from typing import Dict, List

import fire
import numpy as np
import yaml

from configs import get_exp_config
from constants import PolicyIDs, Paths


class ScenarioSet:

    def __init__(self, scenarios: Dict[str, "Scenario"] = None, eval_config=None):
        self.scenarios = {}
        self.scenario_list = []
        self.eval_config = {}
        self.scenario_to_id = {}

        if scenarios is not None:
            self.scenarios = scenarios
            self.scenario_list = list(scenarios.keys())
            self.scenario_to_id = {
                scenario_name: i for i, scenario_name in enumerate(self.scenario_list)
            }

        if eval_config is not None:
            self.eval_config.update(**eval_config)

    def build_from_population(self, num_players, background_population, min_replacements=1, max_replacements=np.inf):
        """

        :param num_players:
        :param background_population: list of policy names,  must all start with "background_"
        :return:
        """
        assert all([pid.startswith("background_") for pid in background_population]), "background policy ids must start with 'background_'!"

        del self.scenario_list
        self.scenarios = {}

        for num_copies in range(1, num_players + 1):

            for background_policies in itertools.combinations_with_replacement(background_population,
                                                                               num_players - num_copies):
                uniques, counts = np.unique(np.array(list(background_policies)), return_counts=True)
                if np.all(np.logical_and(counts>=min_replacements, counts<=max_replacements)):
                    policies = (PolicyIDs.MAIN_POLICY_ID,) + (PolicyIDs.MAIN_POLICY_COPY_ID,) * (
                            num_copies - 1) + background_policies

                    scenario_name = Scenario.get_scenario_name(policies)

                    self.scenarios[scenario_name] = Scenario(num_copies, list(background_policies))

        self.scenario_list = np.array(list(self.scenarios.keys()))
        self.scenario_to_id = {
            scenario_name: i for i, scenario_name in enumerate(self.scenario_list)
        }

    def sample_scenario(self, distribution):
        try:
            return np.random.choice(self.scenario_list, p=distribution)
        except ValueError as e:
            print(e, self.scenario_list, distribution)

    def __getitem__(self, item):
        return self.scenarios[item]

    def __len__(self):
        return len(self.scenario_list)

    def split(self, n=None):
        if n is None:
            n = len(self.scenario_list)

        subsets = []
        for sublist in np.split(self.scenario_list, n):
            subset = copy(self)
            subset.scenario_list = list(sublist)
            subsets.append(subset)

        return subsets

    @classmethod
    def from_YAML(cls, path):
        """
        :param path: path of the yaml
        :return: the list of scenarios contained in there
        """

        with open(path, 'r') as f:
            d = yaml.safe_load(f)

        eval_config = d.get("eval_config", {"num_episodes": 1000})

        scenarios = {
            scenario_name: Scenario(scenario_config["focal"], scenario_config["background"])
            for scenario_name, scenario_config in d["scenarios"].items()
        }
        return cls(scenarios=scenarios, eval_config=eval_config)

    def to_YAML(self, path: str, eval_config: Dict = None):

        if eval_config is not None:
            eval_config.update(**self.eval_config)
        else:
            eval_config = self.eval_config



        scenario_set = {
            "scenarios": {

                scenario_name: {
                    "focal": scenario.num_copies,
                    "background": [policy_id.removeprefix("background_") for policy_id in scenario.background_policies]
                }
                for scenario_name, scenario in self.scenarios.items()
            },

            "eval_config": eval_config
        }

        spath = path.split(os.sep)
        file, ext = spath[-1].split(".")
        parent_path = os.sep.join(spath[:-1])
        if not os.path.exists(parent_path):
            print("Created directory:", parent_path)
            os.makedirs(parent_path, exist_ok=True)
        i = 2
        while os.path.exists(path):
            path = parent_path + os.sep + file + f"_{i}." + ext
            i += 1

        print("Dumping YAML to", path, "...")
        with open(path, "w") as f:
            yaml.safe_dump(scenario_set, f,
                           default_flow_style=None,
                           width=50, indent=4
                           )
        print("Success.")


    @staticmethod
    def average_distribution(scenario_sets: List["ScenarioSet"]) -> "ScenarioSet":

        print([scenario_set.eval_config["distribution"]
            for scenario_set in scenario_sets])
        mean_distribution = np.mean([
            scenario_set.eval_config["distribution"]
            for scenario_set in scenario_sets
        ], axis=0)

        eval_config = scenario_sets[0].eval_config.copy()
        eval_config["distribution"] = mean_distribution.tolist()

        return ScenarioSet(scenario_sets[0].scenarios, eval_config)


class Scenario:

    def __init__(self, num_copies, background_policies):
        self.num_copies = num_copies
        self.background_policies = background_policies

    def get_policies(self):
        policies = [PolicyIDs.MAIN_POLICY_ID] + [PolicyIDs.MAIN_POLICY_COPY_ID] * (
                    self.num_copies - 1) + self.background_policies

        # We suppose the order of players does not matter, but we shuffle it in cases s0 is different for each player.
        np.random.shuffle(policies)
        return policies

    @staticmethod
    def get_scenario_name(policies):
        np_policies = np.array(policies)
        main_policy_mask = ["background" not in p for p in np_policies]
        num_copies = len(np.argwhere(main_policy_mask))

        background_policies = [
            policy_id.removeprefix("background_") for policy_id in np_policies[np.logical_not(main_policy_mask)]
        ]

        string = f"c={num_copies}, b={list(sorted(background_policies))}".replace("'", "")

        return string

    def can_play_scenario(self, available_policies):
        return set(self.background_policies).issubset(set(available_policies))


class ScenarioMapper:
    def __init__(self, scenarios=None, distribution=None):
        if distribution is None:
            distribution = np.ones(len(scenarios), dtype=np.float32) / len(scenarios)

        self.mappings = {}
        self.distribution = distribution
        self.scenarios = scenarios

    def __call__(self, agent_id, episode, worker, **kwargs):
        if episode.episode_id not in self.mappings:
            scenario_name = worker.config.scenarios.sample_scenario(self.distribution)
            self.mappings[episode.episode_id] = worker.config.scenarios[scenario_name].get_policies()

        mapping = self.mappings[episode.episode_id]
        policy_id = mapping.pop()
        if len(mapping) == 0:
            del self.mappings[episode.episode_id]

        return policy_id


class Main:

    def average_distribution(self, env, sets, n_runs=3, **env_config):

        env_config = get_exp_config(env)(**env_config)
        suffixes = [""] + [f"_{i+1}" for i in range(1, n_runs)]
        for train_set_with_distribution in sets:
            ScenarioSet.average_distribution(
                [ScenarioSet.from_YAML(
                    Paths.TEST_SET.format(
                        env=env_config.get_env_id(),
                        name=train_set_with_distribution + suffix
                    ))
                    for suffix in suffixes]
            ).to_YAML(Paths.TEST_SET.format(
                env=env_config.get_env_id(),
                name=train_set_with_distribution + "_average"
            ))


if __name__ == '__main__':
    fire.Fire(Main)