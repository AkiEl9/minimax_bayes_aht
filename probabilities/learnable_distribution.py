import numpy as np

from probabilities.utils import project_onto_simplex
from utils import SmoothMetric


class LearnableDistribution:

    def __init__(self, dim, learning_rate=5e-2, smoothing_factor=0.9995):
        self.dim = dim
        self.beta_logits : np.ndarray
        self.learning_rate = learning_rate

        self.smoothing_factor = smoothing_factor
        self.smoothed_beta = None
        self.initialize_uniformly()

    def initialize_uniformly(self):

        self.beta_logits = np.full((self.dim,), fill_value=1/self.dim, dtype=np.float32)

        self.smoothed_beta = SmoothMetric(init_value=self.beta_logits.copy(), lr=self.smoothing_factor)

    def initialize_randomly(self):

        self.beta_logits = np.random.random((self.dim,)) * 4
        self.beta_logits /= self.beta_logits.sum(keepdims=True)
        self.smoothed_beta = SmoothMetric(init_value=self.beta_logits.copy(), lr=self.smoothing_factor)


    def sample_test_set(self, num_scenarios=3):

        self.beta_logits[np.random.choice(self.dim, size=num_scenarios, replace=False)] = 1.
        self.beta_logits /= num_scenarios
        self.smoothed_beta = SmoothMetric(init_value=self.beta_logits.copy(), lr=self.smoothing_factor)


    def initialize_certain(self, idx=0):

        self.beta_logits = np.zeros((self.dim,), dtype=np.float32)
        self.beta_logits[idx] = 1
        self.smoothed_beta = SmoothMetric(init_value=self.beta_logits.copy(), lr=self.smoothing_factor)

    def get_probs(self, smooth=False):

        if smooth:
            return self.smoothed_beta.get()
        else:
            return self.beta_logits

    def __call__(self, smooth=False):
        return self.get_probs(smooth=smooth)

    def update_prior(self, loss, regret=True):

        if not regret:
            loss = np.max(loss) - loss + np.min(loss)

        normalized_loss = loss

        self.beta_logits[:] = project_onto_simplex(self.beta_logits + normalized_loss * self.learning_rate)



        self.smoothed_beta.update(self.beta_logits)


    def sample(self, size):
        return np.random.choice(self.dim, size, p=self())



if __name__ == "__main__":

    prior = LearnableDistribution(5, learning_rate=1e-3)
    prior.initialize_uniformly()

    loss = np.array([
        4,-5,1,5,1
    ])

    for j in range(100):

        prior.update_prior(loss)

        print(prior.get_probs())
