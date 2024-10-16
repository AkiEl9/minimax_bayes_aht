import gymnasium
import numpy as np

class BackgroundPopulation:

    def __init__(self, environment: gymnasium.Env):
        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n
        self.policies: np.ndarray = None
        self.environment = environment

    def get_expected_policy(self, trained_policy, prior):

        policy = np.empty_like(self.policies[0])

        policy[:] = np.sum(prior()[:-1] * self.policies, axis=0) + prior[-1] * trained_policy.get_params()

    def build_randomly(self, size):

        policies = np.exp(np.random.exponential(8, (size, self.n_states, self.n_actions)))
        self.policies = np.float32(policies / policies.sum(axis=-1, keepdims=True))
        #policies = np.random.normal(0, 1, (size, self.n_states, self.n_actions))
        #exp_p =  np.exp(policies)

        #self.policies = exp_p / exp_p.sum(axis=-1, keepdims=True)

        return self.policies
