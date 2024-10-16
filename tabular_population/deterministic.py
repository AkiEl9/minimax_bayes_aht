import gymnasium
import numpy as np
from itertools import product

from tabular_population import bg_population


def build_deterministic_policies(n_actions, n_states, size=None):

    sequences = np.array(list(product(range(n_actions), repeat=n_states)))
    if size is not None:
        if size < len(sequences):
            sequences = sequences[np.random.choice(len(sequences), size, replace=False)]

    policies = np.zeros((len(sequences), n_states, n_actions), dtype=np.float32)

    seq_indices, state_indices = np.indices(policies.shape[:2])

    policies[seq_indices, state_indices, sequences] = 1

    return policies


class DeterministicPoliciesPopulation(bg_population.BackgroundPopulation):

    def __init__(self, environment: gymnasium.Env):
        super().__init__(environment)
        self.num_policies = self.n_actions**self.n_states
        self.max_size = 1_000_000
        #self.build_population()

    def build_population(self, size=None):
        if hasattr(self.environment, "s_terminal"):

            policies = build_deterministic_policies(self.n_actions, self.n_states - 1, size)

            self.policies = np.zeros((len(policies), self.n_states, self.n_actions))
            self.policies[:, :-1] = policies

        else:
            self.policies = build_deterministic_policies(self.n_actions, self.n_states, size)

        print("background pop size :", len(self.policies))






if __name__ == "__main__":

    env = RepeatedPrisonersDilemmaEnv(episode_length=2)
    bg = DeterministicPoliciesPopulation(env)

    bg.build_population()
