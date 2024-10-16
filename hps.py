import numpy as np

from environments.matrix_form.repeated_prisoners import RepeatedPrisonersDilemmaEnv


# For softmax policies
# TODO NNs with proximal policy gradients
def compute_theoretical_learning_rates(environment, epsilon=1e-1):

    episode_max_return = environment.max_score
    n_actions = environment.n_actions
    T = environment.episode_length

    l = (n_actions + 1) * T * episode_max_return
    L = episode_max_return * T * episode_max_return
    D = np.sqrt(2)

    lr_pi = epsilon ** 4 / (l ** 3 * L ** 2 * D **2)
    lr_beta = 1 / l

    return lr_pi, lr_beta


if __name__ == '__main__':

    env = RepeatedPrisonersDilemmaEnv(1)

    print(compute_theoretical_learning_rates(env, epsilon=3.))
