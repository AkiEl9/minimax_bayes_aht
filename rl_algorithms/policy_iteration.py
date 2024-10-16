from time import time

import numpy as np
from gymnasium.spaces import Discrete
from gymnasium.spaces import Dict

from tabular_population.bg_population import BackgroundPopulation
from tabular_population.deterministic import DeterministicPoliciesPopulation
from probabilities.learnable_distribution import LearnableDistribution
from environments.matrix_form.mdp import compute_multiagent_mdp
from policies.policy import Policy
from policies.tabular_policy import TabularPolicy


class PolicyIteration:

    def __init__(self, initial_policy : Policy, environment, epsilon=1e-3, learning_rate=1e-3, lambda_=1e-3, clip=None):

        self.policy = initial_policy
        self.n_states = self.policy.action_logits.shape[0]
        self.n_actions = self.policy.action_logits.shape[1]
        self.environment = environment
        self.epsilon = epsilon
        self.n_iter = 100
        self.lr = learning_rate
        self.lambda_ = lambda_
        self.clip = clip


    def policy_evaluation_for_prior(
            self,
            bg_population: BackgroundPopulation,
            prior: LearnableDistribution,
    ):

        values = np.zeros((len(prior.beta_logits), self.n_states))
        action_probs = self.policy.get_probs()

        transition_functions = []
        reward_functions = []
        for i in range(len(bg_population.policies)):
            transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                          self.environment.reward_function,
                                                                          bg_population.policies[i],
                                                                          self.environment.curr_state_to_opp_state)
            transition_functions.append(transition_function)
            reward_functions.append(reward_function)

        sp_transition_function, sp_reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                      self.environment.reward_function,
                                                                      action_probs,
                                                                      self.environment.curr_state_to_opp_state,
                                                                      joint_rewards=(0.5,0.5))
        transition_functions.append(sp_transition_function)
        reward_functions.append(sp_reward_function)

        transition_functions = np.stack(transition_functions, axis=0)
        reward_functions = np.stack(reward_functions, axis=0)

        action_probs_p1 = action_probs[np.newaxis]
        action_probs_p1_p4 = action_probs_p1[:, :, :, np.newaxis]

        for t in range(self.epsilon):

            values[:, :] = np.sum(np.sum(

                transition_functions * action_probs_p1_p4 * (reward_functions[:, :, :, np.newaxis]
                            + self.environment.gamma * values[:, np.newaxis, np.newaxis])

                            , axis=-1), axis=-1)


        return np.sum(values * prior()[:, np.newaxis], axis=0), values


    def policy_evaluation_for_scenario(
            self,
            scenario
    ):
        teammate, rewards = scenario

        value = np.zeros((self.n_states))

        action_probs = self.policy.get_probs()

        transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                      self.environment.reward_function,
                                                                      teammate,
                                                                      self.environment.curr_state_to_opp_state,
                                                                      rewards)

        action_probs_p3 = action_probs[:, :, np.newaxis]

        for t in range(self.epsilon):
            value[:] = np.sum(
                np.sum(

                    transition_function * action_probs_p3 * (reward_function[:, :, np.newaxis]
                       + self.environment.gamma * value[np.newaxis, np.newaxis]
                                                             )

                       , axis=-1), axis=-1
            )

        return value

    def policy_improvement_for_scenario(self, scenario, vf):

        teammate, rewards = scenario
        action_probs = self.policy.get_probs()

        transition_function, reward_function = compute_multiagent_mdp(self.environment.transition_function,
                                                                      self.environment.reward_function,
                                                                      teammate,
                                                                      self.environment.curr_state_to_opp_state,
                                                                      rewards)

        new_policy = np.zeros_like(action_probs)

        action_values = np.sum(transition_function * (
            self.environment.gamma * vf[np.newaxis, np.newaxis] + reward_function[:, :, np.newaxis]
        ), axis=-1)

        new_policy[np.arange(len(new_policy)), np.argmax(action_values, axis=-1)] = 1.

        self.policy.action_logits[:] = self.policy.action_logits * (1-self.lr) + self.lr * new_policy
