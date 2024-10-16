from collections import defaultdict, deque

import numpy as np
import ray
from ray.rllib.algorithms import Algorithm

from ray.rllib.utils.typing import ResultDict

from best_responses.serialize import load_best_response_utilities, save_best_response_utilities
from constants import PolicyIDs
from probabilities.utils import project_onto_simplex, distribution_to_hist
from scenarios.scenarios import ScenarioSet, ScenarioMapper
from utils import SmoothMetric


class ABFSGDA:

    def __init__(self, algo: Algorithm):

        self.algo: Algorithm = algo
        self.config: "PPOBFSGDAConfig" = algo.config

        self.weights_history = None
        self.copy_iter = 0
        self.iter = 0
        self.weights_0 = self.algo.get_weights([PolicyIDs.MAIN_POLICY_ID])[PolicyIDs.MAIN_POLICY_ID]
        # Init copy weights:
        self.copy_weights(reset=True)

        self.scenarios: ScenarioSet = self.config.scenarios
        self.current_level = -1
        self.activated_policies = []
        self.activated_scenarios = []
        self.beta_logits = np.zeros(len(self.scenarios), dtype=np.float32)
        self.go_to_next_level()
        self.smoothed_beta = SmoothMetric(self.beta_logits.copy(), lr=self.config.beta_smoothing)

        self.prev_timesteps = 0

        self.scenario_utilities = defaultdict(lambda: SmoothMetric(lr=0.8, init_value=np.nan))
        self.smoothed_scenario_utilities = defaultdict(lambda: SmoothMetric(lr=0.95, init_value=np.nan))
        self.scenario_best_utilities = defaultdict(lambda: -np.inf)

        self.timesteps = [0]
        self.distribution_history = [np.float16(self.smoothed_beta.get())]
        self.level_timesteps = defaultdict(int)

        self.set_matchmaking()

        _base_compile_iteration_results = algo._compile_iteration_results

        def _compile_iteration_results_with_scenario_counts(
                _, *, episodes_this_iter, step_ctx, iteration_results=None
        ):
            r = _base_compile_iteration_results(episodes_this_iter=episodes_this_iter, step_ctx=step_ctx,
                                                iteration_results=iteration_results)

            scenario_counts = defaultdict(int)
            for episode in episodes_this_iter:
                for k in episode.custom_metrics:
                    if "utility" in k:
                        scenario_counts[k.rstrip("_utility")] += 1

            r["custom_metrics"]["scenario_counts"] = scenario_counts

            return r

        self.algo._compile_iteration_results = _compile_iteration_results_with_scenario_counts.__get__(self.algo,
                                                                                                       type(self.algo))

    def go_to_next_level(self):
        self.current_level += 1

        if self.current_level == self.config.end_level:
            return

        new_policy_id = f"background_level_{self.current_level}"
        self.activated_policies.append(new_policy_id)
        self.activated_scenarios = self.get_activated_scenarios()
        self.beta_logits = self.merge_beta_with_new_scenarios()

        d = {new_policy_id: self.algo.get_weights([PolicyIDs.MAIN_POLICY_ID])[PolicyIDs.MAIN_POLICY_ID]}

        self.algo.workers.foreach_worker(
            lambda w: w.set_weights(d)
        )

    def get_activated_scenarios(self):

        return [
            scenario for scenario in self.scenarios.scenario_list
            if self.scenarios[scenario].can_play_scenario(self.activated_policies)
        ]

    def merge_beta_with_new_scenarios(self):
        activated_scenarios = np.array(
            [int(scenario in self.activated_scenarios) for scenario in self.scenarios.scenario_list]
        )
        curr_non_zeros = self.beta_logits > 0.
        next_non_zeros = activated_scenarios > 0.

        prev_num_non_zero = len(np.argwhere(curr_non_zeros))
        next_num_non_zero = len(np.argwhere(next_non_zeros))
        delta_num_non_zero = next_num_non_zero - prev_num_non_zero

        new_scenario_norm = delta_num_non_zero / np.maximum(1., next_num_non_zero)
        other_norm = 1. - new_scenario_norm

        new_beta_logits = self.beta_logits * other_norm + np.float32(
            np.logical_and(np.logical_not(curr_non_zeros), next_non_zeros)
        ) * new_scenario_norm

        return new_beta_logits / np.sum(new_beta_logits)

    def set_matchmaking(self):

        distrib = self.beta_logits.copy()

        self.algo.workers.foreach_worker(
            lambda w: w.set_policy_mapping_fn(
                ScenarioMapper(distribution=distrib)
            ),
        )

    def copy_weights(self, reset=False):
        if reset:
            self.weights_history = [self.weights_0, self.weights_0, self.weights_0]
            self.copy_iter = 0
        else:
            last_weights = self.algo.get_weights([PolicyIDs.MAIN_POLICY_ID])[PolicyIDs.MAIN_POLICY_ID]
            self.weights_history.append(last_weights)
            if len(self.weights_history) > self.config.copy_history_len:
                self.weights_history.pop(0)

        weights = {wid: {PolicyIDs.MAIN_POLICY_COPY_ID: np.random.choice(self.weights_history[:-2])}
                   for wid in self.algo.workers.healthy_worker_ids()}
        if reset:
            for w_dict in weights.values():
                w_dict[PolicyIDs.MAIN_POLICY_ID] = self.weights_0

        self.algo.workers.foreach_worker_with_id(
            lambda wid, w: w.set_weights(weights[wid]),
            local_worker=False
        )

    def beta_gradients(self, loss):
        if self.config.self_play or self.config.beta_lr == 0.:
            return

        # non_zero betas should be corresponding to the activated scenarios, the beta_eps should be > 0 for this
        # to be true
        non_zero_betas = self.beta_logits > 0.
        self.beta_logits[non_zero_betas] = project_onto_simplex(
            (self.beta_logits + loss * self.config.beta_lr)[non_zero_betas])

        # Allow any scenario to be sampled with beta_eps prob

        self.beta_logits[non_zero_betas] = (self.beta_logits[non_zero_betas] * (1 - self.config.beta_eps)
                                            + self.config.beta_eps / non_zero_betas.sum())

        self.smoothed_beta.update(self.beta_logits)

    def update(self, result: ResultDict):
        """
        We should get the regret/utility and update the distribution
        Update the policy mapping function after
        """

        time_steps = result["timesteps_total"]
        iter_data = result["custom_metrics"]
        self.level_timesteps[self.current_level] += time_steps - self.prev_timesteps
        if (
                (self.current_level < self.config.end_level - 1
                 and
                 self.level_timesteps[self.current_level] > self.config.timesteps_per_level)
                or
                (self.level_timesteps[self.current_level] > self.config.timesteps_per_level * 6)
        ):
            self.go_to_next_level()

        for scenario in self.activated_scenarios:
            scenario_utility = iter_data.get(f"{scenario}_utility_mean", np.nan)
            self.scenario_utilities[scenario].update(scenario_utility,
                                                     weight=iter_data["scenario_counts"][scenario])
            smoothed_utility = self.smoothed_scenario_utilities[scenario].update(scenario_utility,
                                                                                 weight=iter_data["scenario_counts"][
                                                                                     scenario])
            if not np.isnan(smoothed_utility):
                self.scenario_best_utilities[scenario] = np.maximum(
                    self.scenario_best_utilities[scenario], smoothed_utility
                )

        pseudo_regrets = np.array([
            self.scenario_best_utilities[scenario]
            -
            self.scenario_utilities[scenario].get()

            for scenario in self.scenarios.scenario_list
        ])
        utilities = np.array([
            self.scenario_utilities[scenario].get()
            for scenario in self.scenarios.scenario_list
        ])
        pseudo_regrets[np.isnan(pseudo_regrets)] = 0.

        self.copy_iter += 1
        self.iter += 1

        for scenario, pseudo_regret, in zip(self.scenarios.scenario_list, pseudo_regrets):
            iter_data[f"{scenario}_regret_mean"] = pseudo_regret

        iter_data[f"worst_case_regret"] = np.max(pseudo_regrets)
        iter_data[f"uniform_regret"] = np.mean(pseudo_regrets)
        iter_data[f"curr_distrib_regret"] = np.sum(pseudo_regrets * self.beta_logits)

        iter_data[f"worst_case_utility"] = np.min(utilities)
        iter_data[f"uniform_utility"] = np.mean(utilities)
        iter_data[f"curr_distrib_utility"] = np.sum(utilities * self.beta_logits)

        self.beta_gradients(pseudo_regrets)
        # Update the matchmaking scheme
        self.set_matchmaking()

        self.distribution_history.append(np.float16(self.smoothed_beta.get()))
        self.timesteps.append(time_steps)

        if self.copy_iter % self.config.copy_weights_freq == 0:
            self.copy_weights()
        self.prev_timesteps = time_steps

        result["custom_metrics"]["current_level"] = self.current_level
        result["custom_metrics"]["level_timesteps"] = self.level_timesteps[self.current_level]
        result["hist_stats"]["scenario_distribution"] = distribution_to_hist(self.beta_logits)
        result["hist_stats"]["smoothed_scenario_distribution"] = distribution_to_hist(self.smoothed_beta.get())

        for scenario, prob, smoothed_prob in zip(self.scenarios.scenario_list, self.beta_logits,
                                                 self.smoothed_beta.get()):
            result["custom_metrics"][f"{scenario}_probability"] = prob
            result["custom_metrics"][f"{scenario}_smoothed_probability"] = smoothed_prob

        # Get rid of duplicates
        result.update(**result["custom_metrics"])
        del result["custom_metrics"]
        del result["sampler_results"]["custom_metrics"]

        result["done"] = self.current_level == self.config.end_level

        return result
