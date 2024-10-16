from time import time

import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Dict

from rl_algorithms.policy_iteration import PolicyIteration
from tabular_population.bg_population import BackgroundPopulation
from tabular_population.deterministic import DeterministicPoliciesPopulation
from probabilities.learnable_distribution import LearnableDistribution
from environments.matrix_form.mdp import compute_multiagent_mdp
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy


class PolicyGradient(PolicyIteration):


    def exact_pg(self, bg_population, prior: LearnableDistribution, vf, previous_copy = None):

        action_probs = self.policy.get_probs()
        all_policies = [bg_population.policies]
        if previous_copy is not None:
            all_policies.append(previous_copy[np.newaxis])
        else:
            all_policies.append(action_probs[np.newaxis])

        all_policies = np.concatenate(all_policies, axis=0)
        all_rewards = np.concatenate([np.tile([1., 0.], (len(bg_population.policies), 1)), [[0.5, 0.5]]], axis=0)

        gradients = []

        for teammate, reward_weights, V, scenario_prob \
                in zip(all_policies, all_rewards, vf, prior()):

            induced_transition_function, induced_reward_function = compute_multiagent_mdp(
                self.environment.transition_function, self.environment.reward_function,
                teammate, self.environment.curr_state_to_opp_state, joint_rewards=reward_weights
            )

            Q = induced_reward_function + self.environment.gamma * np.sum(induced_transition_function * V[np.newaxis, np.newaxis], axis=-1)

            gradients.append(scenario_prob * self.policy.compute_pg(
                Q, V, transition_function=induced_transition_function, lambda_=self.lambda_
            ))

        self.policy.apply_gradient(sum(gradients), lr=self.lr, clip=self.clip)