import gymnasium
import numpy as np

from policies.policy import Policy


class TabularPolicy(Policy):

    def __init__(self, environment: gymnasium.Env, logits = None):
        super().__init__(environment)
        if logits is not None:
            self.action_logits[:] = logits

    def initialize_uniformly(self):

        self.action_logits[:] = 1. / self.n_actions

    def initialize_randomly(self):

        self.action_logits[:] = np.random.random(self.action_logits.shape)
        self.action_logits /= self.action_logits.sum(axis=-1, keepdims=True)


    def get_probs(self):
        return self.action_logits.copy()

    def get_params(self):
        return self.action_logits


class SingleStatePolicy(Policy):

    def __init__(self, environment: gymnasium.Env, bound_state=0):
        super().__init__(environment)

        self.bound_state = bound_state

        self.action_probs = np.full((self.n_actions,), fill_value=np.nan, dtype=np.float32)

    def make_deterministic(self, action_idx):

        self.action_probs[:] = 0
        self.action_probs[action_idx] = 1


