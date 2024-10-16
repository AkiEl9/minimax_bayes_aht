import gymnasium
import numpy as np


class Policy:

    def __init__(self, environment: gymnasium.Env):

        self.n_actions = environment.action_space[0].n
        self.n_states = environment.observation_space[0].n

        self.action_logits: np.ndarray = np.full((self.n_states, self.n_actions), fill_value=np.nan, dtype=np.float32)

        self.environment = environment


    def initialize_uniformly(self):

        self.action_logits[:] = 0

    def initialize_randomly(self):

        self.action_logits[:] = np.random.normal(0, 0.2, self.action_logits.shape)


    def sample_action(self, state):

        return np.random.choice(self.n_actions, p=self.get_probs()[state])

    def sample_deterministic_action(self, state):
        return np.argmax(self.action_logits[state])

    def get_params(self):
        return self.action_logits.copy()

    def get_probs(self, explore=True):
        exp = np.exp(self.action_logits - self.action_logits.max(axis=-1, keepdims=True))
        p = exp / exp.sum(axis=-1, keepdims=True)

        return p

    def compute_pg(self, Q, V, transition_function, lambda_):

        action_probs = self.get_probs()

        state_visitation = self.get_state_visitation(transition_function)

        A = Q - V[:, np.newaxis]

        gradients = state_visitation[:, np.newaxis] * action_probs * (A - lambda_ * np.log(action_probs+1e-8))

        return gradients

    def apply_gradient(self, gradient, lr, normalize=False, clip=None):
        if normalize:
            mean_grad = np.mean(gradient)
            std = np.maximum(np.std(gradient), 1e-2)
            gradient = (gradient - mean_grad) / std
        else:
            gradient -= np.mean(gradient, axis=1, keepdims=True)

        if clip is not None:
            gradient = np.clip(gradient, -clip, clip)


        self.action_logits[:] = np.clip(lr * gradient + self.action_logits, -16., 16.)

    def get_state_visitation(self, transition_function):

        max_iterations = self.environment.episode_length
        state_frequencies = np.zeros(self.n_states)
        state_frequencies[self.environment.s0] = 1.
        action_probs = self.get_probs()

        for iteration in range(max_iterations):

            # Update state frequencies using the transition function and policy
            state_frequencies[:] += ((state_frequencies[:, np.newaxis, np.newaxis] * action_probs[:, :, np.newaxis] * transition_function)
                                     .sum(axis=0).sum(axis=0))


        if hasattr(self.environment, "s_terminal"):
            state_frequencies[self.environment.s_terminal] = 0.
        state_frequencies /= np.sum(state_frequencies)

        return state_frequencies

    def get_state_action_visitation(self, transition_function):
        action_probs = self.get_probs()
        state_frequencies = self.get_state_visitation(transition_function)
        state_action_frequencies = state_frequencies[:, np.newaxis] * action_probs

        return state_action_frequencies



