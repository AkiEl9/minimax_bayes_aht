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


class BFSGDA:

    def __init__(self, algo: Algorithm, learn_best_responses=False):

        self.algo: Algorithm = algo
        self.config: "PPOBFSGDAConfig" = algo.config
        self.learn_best_responses = learn_best_responses

        self.best_response_utilities = defaultdict(float)

        self.scenarios: ScenarioSet = self.config.scenarios
        self.beta_logits = np.ones(len(self.scenarios), dtype=np.float32) / len(self.scenarios)
        if self.config.self_play:
            self.beta_logits[:] = [
                float(len(self.scenarios[scenario].background_policies) == 0)
                for scenario in self.scenarios.scenario_list
            ]

        self.smoothed_beta = SmoothMetric(self.beta_logits.copy(), lr=self.config.beta_smoothing)
        self.weights_history = None
        self.copy_iter = 0
        self.iter = 0
        self.prev_timesteps = 0
        self.weights_0 = self.algo.get_weights([PolicyIDs.MAIN_POLICY_ID])[PolicyIDs.MAIN_POLICY_ID]

        # Init copy weights and best response utilities if needed:
        self.copy_weights(reset=True)
        self.missing_best_responses: list = []
        self.current_best_response_scenario = None
        # The smoothing factor really depends on the time taken per iteration (20 secs appears fine)
        self.current_best_response_utility = SmoothMetric(lr=self.config.utility_smoothing)
        self.scenario_utilities = defaultdict(lambda: SmoothMetric(lr=0.8))

        self.timesteps = [0]
        self.distribution_history = [np.float16(self.smoothed_beta.get())]

        self.smoothed_scenario_utilities = defaultdict(lambda: SmoothMetric(lr=0.95, init_value=np.nan))
        self.scenario_best_utilities = defaultdict(lambda: -np.inf)

        self.best_response_timesteps = defaultdict(int)
        self.missing_best_responses: deque = deque(list(self.scenarios.scenario_list))
        self.load_best_response_utilities()
        if len(self.missing_best_responses) > 0 and (not self.config.use_utility):
            self.current_best_response_scenario = self.missing_best_responses.popleft()

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

        self.algo._compile_iteration_results = _compile_iteration_results_with_scenario_counts.__get__(self.algo, type(self.algo))

    def set_matchmaking(self):

        if self.current_best_response_scenario is None:
            distrib = self.beta_logits.copy()

        else:
            # We are learning best responses
            distrib = np.array([
                float(scenario == self.current_best_response_scenario) for scenario in self.scenarios.scenario_list
            ])

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

    def load_best_response_utilities(self):

        best_response_utilities = defaultdict(float, load_best_response_utilities(self.config.env))

        # TODO: this is actually not a good design
        for scenario_name in self.scenarios.scenario_list:
            if scenario_name in best_response_utilities:
                self.best_response_utilities[scenario_name] = best_response_utilities[scenario_name]
                self.missing_best_responses.remove(scenario_name)
        if self.config.learn_best_responses_only and len(self.missing_best_responses) == 0:
            self.current_best_response_scenario = None


    def beta_gradients(self, loss):
        if self.config.self_play or self.config.beta_lr == 0.:
            return

        self.beta_logits[:] = project_onto_simplex(self.beta_logits + loss * self.config.beta_lr)

        # Allow any scenario to be sampled with beta_eps prob
        self.beta_logits[:] = self.beta_logits * (1 - self.config.beta_eps) + self.config.beta_eps / len(
            self.beta_logits)


    def update(self, result: ResultDict):
        """
        We should get the regret/utility and update the distribution
        Update the policy mapping function after
        """

        time_steps = result["timesteps_total"]
        iter_data = result["custom_metrics"]

        # We are learning the best responses here.
        # Test if we are done learning some best responses
        if (not self.config.use_utility) and (self.current_best_response_scenario is not None):

            expected_utility = iter_data.get(f"{self.current_best_response_scenario}_utility_mean", 0.)
            if self.current_best_response_scenario not in self.best_response_utilities:
                self.current_best_response_utility.set(expected_utility)
                self.best_response_utilities[
                    self.current_best_response_scenario] = self.current_best_response_utility.get()
            else:
                self.current_best_response_utility.update(expected_utility)

                self.best_response_utilities[self.current_best_response_scenario] = float(np.maximum(
                    self.current_best_response_utility.get(),
                    self.best_response_utilities[self.current_best_response_scenario]
                ))

            self.best_response_timesteps[self.current_best_response_scenario] += time_steps - self.prev_timesteps
            if self.best_response_timesteps[self.current_best_response_scenario]>=(self.config.best_response_timesteps_max * self.scenarios[self.current_best_response_scenario].num_copies):

                # expected_utility = iter_data[f"{self.current_best_response_scenario}_utility_mean"]
                # self.best_response_utilities[self.current_best_response_scenario] = expected_utility

                save_best_response_utilities(
                    env_name=self.config.env,
                    best_response_utilities=self.best_response_utilities
                )

                if len(self.missing_best_responses) > 0:
                    self.current_best_response_scenario = self.missing_best_responses.popleft()
                    self.set_matchmaking()
                else:
                    # Will move on to learning the minimax regret solution
                    self.current_best_response_scenario = None

                # Reset learn_best_responses policy to 0, along with its history of weights
                self.copy_weights(reset=True)

        self.copy_iter += 1
        self.iter += 1


        if self.config.use_utility or (self.current_best_response_scenario is None):
            if self.config.learn_best_responses_only:
                # We are done computing best responses, stop
                result["done"] = True
                return result
            # Compute lossweight=iter_data["scenario_counts"][scenario]

            utilities = np.array([
                self.scenario_utilities[scenario].update(iter_data.get(f"{scenario}_utility_mean", np.nan),
                                                         weight=iter_data["scenario_counts"][scenario])
                for scenario in self.scenarios.scenario_list
            ])

            if self.config.auto_regret:
                # pseudo regrets
                for scenario in self.scenarios.scenario_list:
                    smoothed_utility = self.smoothed_scenario_utilities[scenario].update(
                        iter_data.get(f"{scenario}_utility_mean", np.nan),
                        weight=iter_data["scenario_counts"][scenario]
                    )

                    if not np.isnan(smoothed_utility):
                        self.scenario_best_utilities[scenario] = np.maximum(
                            self.scenario_best_utilities[scenario], smoothed_utility
                        )
                regrets = np.array([
                    self.scenario_best_utilities[scenario]
                    -
                    self.scenario_utilities[scenario].get()

                    for scenario in self.scenarios.scenario_list
                ])
                regrets[np.logical_or(np.isnan(regrets), np.isinf(regrets))] = 0.
            else:
                regrets = np.array([
                    self.best_response_utilities[scenario] - self.scenario_utilities[scenario].get()
                    for scenario in self.scenarios.scenario_list
                ])

            if self.config.use_utility:
                beta_losses = utilities
                beta_losses = np.max(beta_losses) - beta_losses + np.min(beta_losses)
            else:
                beta_losses = regrets

            for scenario in self.scenarios.scenario_list:
                iter_data[f"{scenario}_regret_mean"] = (self.best_response_utilities[scenario]
                                                        - self.scenario_utilities[scenario].get())
                iter_data[f"{scenario}_smoothed_utility"] = self.smoothed_scenario_utilities[scenario].get()
            iter_data[f"worst_case_regret"] = np.max(regrets)
            iter_data[f"uniform_regret"] = np.mean(regrets)
            iter_data[f"curr_distrib_regret"] = np.sum(regrets * self.beta_logits)
            iter_data[f"worst_case_utility"] = np.min(utilities)
            iter_data[f"uniform_utility"] = np.mean(utilities)
            iter_data[f"curr_distrib_utility"] = np.sum(utilities * self.beta_logits)

            if self.config.warmup_steps < self.iter:
                self.beta_gradients(beta_losses)
                self.smoothed_beta.update(self.beta_logits,
                                          weight=1.#result["episodes_this_iter"]
                                          )

                # Update the matchmaking scheme
                self.set_matchmaking()
            self.distribution_history.append(np.float16(self.smoothed_beta.get()))
            self.timesteps.append(time_steps)

        if self.copy_iter % self.config.copy_weights_freq == 0:
            self.copy_weights()

        self.prev_timesteps = time_steps

        if not self.config.use_utility:
            result["custom_metrics"]["missing_best_response_utilities"] = len(self.missing_best_responses)
            if self.current_best_response_scenario is None:
                result["custom_metrics"]["best_response_timesteps"] = 0

            else:
                result["custom_metrics"]["best_response_timesteps"] = self.best_response_timesteps[
                    self.current_best_response_scenario]

        result["hist_stats"]["scenario_distribution"] = distribution_to_hist(self.beta_logits)
        result["hist_stats"]["smoothed_scenario_distribution"] = distribution_to_hist(self.smoothed_beta.get())

        for scenario, prob, smoothed_prob in zip(self.scenarios.scenario_list, self.beta_logits, self.smoothed_beta.get()):
            result["custom_metrics"][f"{scenario}_probability"] = prob
            result["custom_metrics"][f"{scenario}_smoothed_probability"] = smoothed_prob

        # Get rid of duplicates
        result.update(**result["custom_metrics"])
        del result["custom_metrics"]
        del result["sampler_results"]["custom_metrics"]

        result["done"] = time_steps >= self.config.max_timesteps
        result["num_gda_updates"] = self.iter

        return result
