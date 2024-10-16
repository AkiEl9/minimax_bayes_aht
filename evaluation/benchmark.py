import multiprocessing as mp
import os
import sys
from collections import defaultdict

from utils import SuppressStd

import pickle
from typing import Dict, List

import numpy as np
import yaml
from ray.rllib import Policy, SampleBatch
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.policy.policy import PolicySpec
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn, TextColumn, BarColumn

from best_responses.serialize import load_best_response_utilities
from configs import get_exp_config

from constants import Paths
from policies.rllib_deterministic_policy import RLlibDeterministicPolicy, NoopPolicy
from scenarios.scenarios import ScenarioSet

NUM_CPU = 32


def shuffle_dict(input_dict):
    # Convert the dictionary into a list of key-value pairs
    items = list(input_dict.items())
    # Shuffle the list
    np.random.shuffle(items)
    # Convert the shuffled list back into a dictionary
    shuffled_dict = dict(items)
    return shuffled_dict


def run_episode(policies: Dict[str, Policy], env, n_episodes=10):
    n_focal = len([
        1 for policy_id in policies if "background" not in policy_id
    ])


    episodic_focal_per_capita = []

    stats = defaultdict(lambda: np.int32(0))

    for i in range(n_episodes):
        policies = shuffle_dict(policies)
        agent_to_policy = {
            agent_id: policy_id
            for agent_id, policy_id in zip(env._agent_ids, policies.keys())
        }

        states = {
            agent_id: policies[agent_to_policy[agent_id]].get_initial_state() for agent_id in env._agent_ids
        }

        done = False
        prev_action = {
            agent_id: 0 for agent_id in env._agent_ids
        }
        obs, _ = env.reset()
        prev_pos = defaultdict(lambda: None)
        focal_per_capita = 0
        while not done:
            actions = {}
            for agent_id in env._agent_ids:
                policy_id = agent_to_policy[agent_id]

                input_dict = {
                    SampleBatch.OBS:  # [[obs[agent_id]]]
                        {
                            name: np.array([feature]) for name, feature in obs[agent_id].items() if
                            (
                                    ("INFO" not in name) and ("POSITION" not in name)
                            )
                        },
                }
                for i, s in enumerate(states[agent_id]):
                    input_dict[f"state_in_{i}"] = [s]
                input_dict[SampleBatch.PREV_ACTIONS] = np.array([prev_action[agent_id]])

                action, next_state, _ = policies[policy_id].compute_actions_from_input_dict(
                    SampleBatch(input_dict),
                )
                if next_state is not None:
                    states[agent_id] = [s[0] for s in next_state]
                actions[agent_id] = action[0]

            obs, rewards, dones, truncs, _ = env.step(actions)

            done = dones["__all__"] or truncs["__all__"]
            timestep_focal_rewards = [
                r for agent_id, r in rewards.items() if not "background" in agent_to_policy[agent_id]
            ]
            for agent_id, agent_obs in obs.items():
                if "background" not in agent_to_policy[agent_id]:
                    for name, value in agent_obs.items():
                        if "INFO" in name:
                            stats[name] = np.int32(stats[name] + value)
                        elif "POSITION" in name:
                            if prev_pos[agent_id] is not None and np.any(prev_pos[agent_id] != value):
                                stats["movement"] =  np.int32(stats["movement"] + 1)
                            prev_pos[agent_id] = value

            focal_per_capita += sum(timestep_focal_rewards) / n_focal
        episodic_focal_per_capita.append(focal_per_capita)

    for stat, total_value in stats.items():
        stats[stat] = float(total_value / (n_episodes * n_focal))

    return {
        "focal_per_capita_mean": float(np.mean(episodic_focal_per_capita)),
        "focal_per_capita_ste": float(np.std(episodic_focal_per_capita) / np.sqrt(n_episodes)),
        **stats
    }


class NamedPolicy:
    def __init__(self, name):
        self.name = name


class PolicyCkpt:
    """
    Can either be a deterministic policy,
    or a named policy
    """

    def __init__(self, name, env_name=""):

        self.name = name
        self.env_name = env_name

        if "deterministic" in name:
            try:
                _, policy_seed = name.split("_")
            except ValueError as e:
                raise ValueError(f"Malformed policy name: {name}.") from e

            def make(environment):
                dummy_agent_id = list(environment.get_agent_ids())[0]
                return RLlibDeterministicPolicy(
                    environment.observation_space[dummy_agent_id],
                    environment.action_space[dummy_agent_id],
                    {},
                    seed=int(policy_seed)
                )

            def get_policy_specs(environment):
                dummy_agent_id = list(environment.get_agent_ids())[0]
                return (RLlibDeterministicPolicy,
                        environment.observation_space[dummy_agent_id],
                        environment.action_space[dummy_agent_id],
                        {"seed": int(policy_seed)}
                        )

        elif name == "random":

            def make(environment):
                dummy_agent_id = list(environment.get_agent_ids())[0]
                return RandomPolicy(
                    environment.observation_space[dummy_agent_id],
                    environment.action_space[dummy_agent_id],
                    {},
                )

            def get_policy_specs(environment):
                dummy_agent_id = list(environment.get_agent_ids())[0]
                return (RandomPolicy,
                        environment.observation_space[dummy_agent_id],
                        environment.action_space[dummy_agent_id],
                        {}
                        )

        elif name == "noop":
            def make(environment):
                dummy_agent_id = list(environment.get_agent_ids())[0]
                return NoopPolicy(
                    environment.observation_space[dummy_agent_id],
                    environment.action_space[dummy_agent_id],
                    {},
                )

            def get_policy_specs(environment):
                dummy_agent_id = list(environment.get_agent_ids())[0]
                return (NoopPolicy,
                        environment.observation_space[dummy_agent_id],
                        environment.action_space[dummy_agent_id],
                        {}
                        )
        elif "expert_ant" in name:
            def get_policy_specs(environment):
                from ray.rllib.utils.checkpoints import get_checkpoint_info
                checkpoint_info = get_checkpoint_info(Paths.NAMED_POLICY.format(name=name, env=env_name))
                with open(checkpoint_info["state_file"], "rb") as f:
                    state = pickle.load(f)
                serialized_pol_spec = state.get("policy_spec")
                if serialized_pol_spec is None:
                    raise ValueError(
                        "No `policy_spec` key was found in given `state`! "
                        "Cannot create new Policy."
                    )
                policy_spec = PolicySpec.deserialize(serialized_pol_spec)
                del state

                class DecoratedClass(policy_spec.policy_class, NamedPolicy):
                    def __init__(_self, *args, **kwargs):
                        policy_spec.policy_class.__init__(_self, *args, **kwargs)
                        NamedPolicy.__init__(_self, name=name)

                    def compute_actions(
                            self,
                            obs_batch,
                            state_batches=None,
                            prev_action_batch=None,
                            prev_reward_batch=None,
                            **kwargs,
                    ):
                        actions, states, infos = super().compute_actions(
                            obs_batch, state_batches, prev_action_batch, prev_reward_batch, **kwargs)

                        print(actions.shape)

                        acts1, acts2 = np.split(actions, 2, axis=1)
                        return (
                            [act1 if obs[SampleBatch.OBS][-1] == 0. else act2 for obs, act1, act2 in
                             zip(obs_batch, acts1, acts2)],
                            [],
                            {},
                        )

                DecoratedClass.__name__ = policy_spec.policy_class.__name__
                DecoratedClass.__qualname__ = policy_spec.policy_class.__qualname__

                return PolicySpec(
                    policy_class=DecoratedClass,
                    observation_space=policy_spec.observation_space,
                    action_space=policy_spec.action_space,
                    config=policy_spec.config
                )

            def make(environment):
                p = Policy.from_checkpoint(Paths.NAMED_POLICY.format(name=name, env=env_name))

                def compute_actions(
                        self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        **kwargs,
                ):
                    actions, states, infos = super().compute_actions(
                        obs_batch, state_batches, prev_action_batch, prev_reward_batch, **kwargs)

                    print(actions.shape)

                    acts1, acts2 = np.split(actions, 2, axis=1)
                    return (
                        [act1 if obs[SampleBatch.OBS][-1] == 0. else act2 for obs, act1, act2 in
                         zip(obs_batch, acts1, acts2)],
                        [],
                        {},
                    )

                p.compute_actions = compute_actions
                return p

        else:
            def make(environment):
                return Policy.from_checkpoint(Paths.NAMED_POLICY.format(name=name, env=env_name))

            def get_policy_specs(environment):
                from ray.rllib.utils.checkpoints import get_checkpoint_info
                checkpoint_info = get_checkpoint_info(Paths.NAMED_POLICY.format(name=name, env=env_name))
                with open(checkpoint_info["state_file"], "rb") as f:
                    state = pickle.load(f)
                serialized_pol_spec = state.get("policy_spec")
                if serialized_pol_spec is None:
                    raise ValueError(
                        "No `policy_spec` key was found in given `state`! "
                        "Cannot create new Policy."
                    )
                policy_spec = PolicySpec.deserialize(serialized_pol_spec)
                del state

                class DecoratedClass(policy_spec.policy_class, NamedPolicy):
                    def __init__(_self, *args, **kwargs):
                        policy_spec.policy_class.__init__(_self, *args, **kwargs)
                        NamedPolicy.__init__(_self, name=name)

                DecoratedClass.__name__ = policy_spec.policy_class.__name__
                DecoratedClass.__qualname__ = policy_spec.policy_class.__qualname__

                return PolicySpec(
                    policy_class=DecoratedClass,
                    observation_space=policy_spec.observation_space,
                    action_space=policy_spec.action_space,
                    config=policy_spec.config
                )

        self.make = make
        self.get_policy_specs = get_policy_specs


def eval_policy_on_scenario(
        packed_args
):
    (scenario_name,
     policy_name,
     test_set,
     env_config,
     set_name
     ) = packed_args

    scenario = test_set[scenario_name]
    environment = env_config.get_maker(evaluation=True)()

    env_id = env_config.get_env_id()
    focal_policy = PolicyCkpt(policy_name, env_id).make(environment)
    focal_policies = {
        f"{policy_name}_{i}": focal_policy
        for i in range(scenario.num_copies)
    }

    background_policies = {}
    for bg_policy_name in scenario.background_policies:
        i = 0
        name =  f"background_{bg_policy_name}_{i}"
        while name in background_policies:
            i += 1
            name = f"background_{bg_policy_name}_{i}"
        background_policies[name] = PolicyCkpt(bg_policy_name, env_id).make(environment)


    policies = {
        **focal_policies,
        **background_policies
    }

    return (packed_args,
            {scenario_name: run_episode(policies, environment, n_episodes=test_set.eval_config["num_episodes"]
                                        )}
            )


class Benchmarking:
    """
    We want to benchmark policies in various test sets
    # TODO : we need this if benchmarking is slow
    """

    def __init__(self,
                 policies: List[str],
                 test_sets: List[str],
                 env: str,
                 env_config: Dict,
                 eval_config: Dict
                 ):

        set_evaluations = [
            Evaluation(test_set, env, env_config=env_config, eval_config=eval_config)
            for test_set in test_sets
        ]

        tasks = []

        for evaluation in set_evaluations:
            for policy in policies:
                tasks.extend(evaluation.get_tasks(policy))


class Evaluation:
    """
    feed env and name of the test set
    loads required policies
    """

    def __init__(self, test_set: str, env: str, env_config: Dict):

        self.test_set_name = test_set
        self.env_name = env,
        self.env_config = env_config
        self.env_config = get_exp_config(env)(**env_config)
        self.environment = self.env_config.get_maker(evaluation=True)()
        self.test_set = ScenarioSet.from_YAML(
            Paths.TEST_SET.format(env=self.env_config.get_env_id(), name=test_set)
        )

    def eval_policy_on_scenario(self, policy_name, scenario_name):

        scenario = self.test_set[scenario_name]
        env_maker = self.env_config.get_maker(evaluation=True)
        focal_policy = PolicyCkpt(policy_name).make(self.environment)
        focal_policies = {
            f"name_{i}": focal_policy
            for i in scenario.num_copies
        }
        background_policies = {
            f"background_{bg_policy_name}": PolicyCkpt(bg_policy_name).make(self.environment)
            for bg_policy_name in scenario.background_policies
        }

        policies = {
            **focal_policies,
            **background_policies
        }
        return {scenario_name: run_episode(policies, env_maker, n_episodes=self.eval_config["n_episodes"])}

    def evaluate_policy(self, policy_name):

        jobs = [
            (scenario_name, policy_name, self.test_set, self.env_config, self.test_set_name)
            for scenario_name in self.test_set.scenario_list
        ]

        res = {}

        with Progress() as progress:
            task = progress.add_task(f'[green]Evaluating Policy "{policy_name}" on test set "{self.test_set_name}"',
                                     total=len(jobs))
            mp.set_start_method('spawn')
            with mp.Pool(np.minimum(NUM_CPU, len(self.test_set)), maxtasksperchild=1) as p:
                for out in p.imap_unordered(eval_policy_on_scenario, jobs):
                    res.update(**out)
                    progress.update(task, advance=1)

        best_response_utilities = load_best_response_utilities(self.env_config.get_env_id())
        if "distribution" not in self.test_set.eval_config:
            self.test_set.eval_config["distribution"] = np.ones(len(self.test_set)) / len(self.test_set)

        for scenario in res:
            res[scenario]["regret_mean"] = best_response_utilities.get(scenario, 0.) - res[scenario][
                "focal_per_capita_mean"]

        # we need to ensure that the ordering is the same for the scores and evals
        weighted_scores = [
            res[scenario]["focal_per_capita_mean"] * self.test_set.eval_config["distribution"][i]
            for i, scenario in enumerate(self.test_set.scenario_list)
        ]
        weighted_regrets = [
            res[scenario]["regret_mean"] * self.test_set.eval_config["distribution"][i]
            for i, scenario in enumerate(self.test_set.scenario_list)
        ]
        weighted_stes = [
            res[scenario]["focal_per_capita_ste"] * self.test_set.eval_config["distribution"][i]
            for i, scenario in enumerate(self.test_set.scenario_list)
        ]
        expected_utility = float(np.sum(weighted_scores))
        expected_regret = float(np.sum(weighted_regrets))
        overall_ste = float(np.sum(weighted_stes))

        worst_case_utility = float(np.min([
            scenario_res["focal_per_capita_mean"] for scenario_res in res.values()
        ]))
        worst_case_regret = float(np.max([
            scenario_res["regret_mean"] for scenario_res in res.values()
        ]))

        out_dict = {
            "worst_case_utility": worst_case_utility,
            "worst_case_regret": worst_case_regret,
            "overall_score": expected_utility,
            "expected_regret": expected_regret,
            "overall_standard_error": overall_ste,
            "per_scenario": res
        }
        return {
            policy_name: out_dict
        }

    @staticmethod
    def get_jobs(policies, sets, env_config):
        jobs = []
        for policy_name in policies:
            for set_name in sets:
                test_set = ScenarioSet.from_YAML(
                    Paths.TEST_SET.format(env=env_config.get_env_id(), name=set_name)
                )

                jobs.extend([
                    (scenario_name, policy_name, test_set, env_config, set_name)
                    for scenario_name in test_set.scenario_list
                ])

        return jobs

    @staticmethod
    def post_process_policy_eval(test_set, exp_config, res):
        best_response_utilities = load_best_response_utilities(exp_config.get_env_id())
        if "distribution" not in test_set.eval_config:
            test_set.eval_config["distribution"] = np.ones(len(test_set)) / len(test_set)

        for scenario in res:
            res[scenario]["regret_mean"] = best_response_utilities.get(scenario, 0.) - res[scenario][
                "focal_per_capita_mean"]

        # we need to ensure that the ordering is the same for the scores and evals
        weighted_scores = [
            res[scenario]["focal_per_capita_mean"] * test_set.eval_config["distribution"][i]
            for i, scenario in enumerate(test_set.scenario_list)
        ]
        weighted_regrets = [
            res[scenario]["regret_mean"] * test_set.eval_config["distribution"][i]
            for i, scenario in enumerate(test_set.scenario_list)
        ]
        weighted_stes = [
            res[scenario]["focal_per_capita_ste"] * test_set.eval_config["distribution"][i]
            for i, scenario in enumerate(test_set.scenario_list)
        ]
        expected_utility = float(np.sum(weighted_scores))
        expected_regret = float(np.sum(weighted_regrets))
        overall_ste = float(np.sum(weighted_stes))

        worst_case_utility = float(np.min([
            scenario_res["focal_per_capita_mean"] for scenario_res in res.values()
        ]))
        worst_case_regret = float(np.max([
            scenario_res["regret_mean"] for scenario_res in res.values()
        ]))

        return {
            "worst_case_utility": worst_case_utility,
            "worst_case_regret": worst_case_regret,
            "overall_score": expected_utility,
            "expected_regret": expected_regret,
            "overall_standard_error": overall_ste,
            "per_scenario": res
        }

    @staticmethod
    def run(jobs, exp_config):

        all_evals = defaultdict(lambda: defaultdict(dict))
        with Progress(SpinnerColumn(), TextColumn(f'[green]Evaluating policies...'), MofNCompleteColumn(), BarColumn(),
                      TimeElapsedColumn()) as progress:
            task = progress.add_task("evaluation", total=len(jobs))

            with mp.Pool(np.minimum(NUM_CPU, len(jobs)), maxtasksperchild=1) as p:
                for out in p.imap_unordered(eval_policy_on_scenario, jobs):
                    (_, p_name, _, _, test_set_name), ret = out
                    all_evals[test_set_name][p_name].update(**ret)
                    progress.update(task, advance=1)

        postprocess_eval = {
            set_name: {
                p_name: Evaluation.post_process_policy_eval(
                    ScenarioSet.from_YAML(Paths.TEST_SET.format(env=exp_config.get_env_id(), name=set_name)),
                    exp_config,
                    res
                )
                for p_name, res in all_res.items()
            } for set_name, all_res in all_evals.items()
        }

        return postprocess_eval

    def load(self):
        path = Paths.EVAL.format(env=self.env_config.get_env_id(),
                                 name=self.test_set_name)
        if os.path.exists(path):
            with open(path, 'r') as f:
                evaluation = yaml.safe_load(f)
            if evaluation is None:
                evaluation = {}
        else:
            evaluation = {}
            parent_path = os.sep.join(path.split(os.sep)[:-1])
            os.makedirs(parent_path, exist_ok=True)
            print("Created directory:", parent_path)
            with open(path, 'w') as f:
                yaml.safe_dump(evaluation, f)

        return evaluation

    def save(self, evaluation):
        path = Paths.EVAL.format(env=self.env_config.get_env_id(),
                                 name=self.test_set_name)

        with open(path, 'w') as f:
            yaml.safe_dump(
                evaluation
                , f)
