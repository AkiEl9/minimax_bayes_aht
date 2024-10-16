from functools import partial
from typing import Optional

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.from_config import NotProvided

from bf_sgda.rllib_callbacks import BackgroundFocalSGDA
from scenarios.scenarios import ScenarioSet


def make_bf_sgda_config(cls) -> "BFSGDAConfig":
    """

    :param cls:
    :return: config decorated with bf_sgda, plugging the algorithm with a custom callback
    """
    class BFSGDAConfig(cls):

        def __init__(self):
            super().__init__()

            self.beta_lr = 5e-2
            self.beta_eps = 1e-2

            self.beta_smoothing = 0.999
            self.utility_smoothing = 0.8
            self.copy_weights_freq = 5
            self.copy_history_len = 30
            self.best_response_timesteps_max = 1_000_000
            self.max_timesteps = 50_000_000
            self.warmup_steps = 15

            self.use_utility = False
            self.self_play = False
            self.learn_best_responses_only = False

            # Auto sgda
            self.auto_regret = False
            self.auto_population = False
            self.timesteps_per_level = 5_000_000
            self.end_level = 10


            # Must be specified in the training config.
            self.scenarios = None

        def training(
                self,
                *,
                beta_lr: Optional[float] = NotProvided,
                beta_smoothing: Optional[float] = NotProvided,
                utility_smoothing: Optional[float] = NotProvided,
                use_utility: Optional[bool] = NotProvided,
                self_play: Optional[bool] = NotProvided,
                copy_weights_freq: Optional[int] = NotProvided,
                best_response_timesteps_max: Optional[int] = NotProvided,
                learn_best_responses_only: Optional[bool] = NotProvided,
                copy_history_len: Optional[int] = NotProvided,
                warmup_steps: Optional[int] = NotProvided,
                beta_eps: Optional[float] = NotProvided,
                scenarios: Optional[ScenarioSet] = NotProvided,
                max_timesteps: Optional[int] = NotProvided,
                auto_regret: Optional[bool] = NotProvided,
                auto_population: Optional[bool] = NotProvided,
                timesteps_per_level: Optional[int] = NotProvided,
                end_level: Optional[int] = NotProvided,
                **kwargs,
        ) -> "PPOConfig":

            super().training(**kwargs)
            if beta_lr is not NotProvided:
                self.beta_lr = beta_lr
            if beta_smoothing is not NotProvided:
                self.beta_smoothing = beta_smoothing
            if utility_smoothing is not NotProvided:
                self.utility_smoothing = utility_smoothing
            if use_utility is not NotProvided:
                self.use_utility = use_utility
            if self_play is not NotProvided:
                self.self_play = self_play
            if copy_weights_freq is not NotProvided:
                self.copy_weights_freq = copy_weights_freq
            if copy_history_len is not NotProvided:
                self.copy_history_len = copy_history_len
            if learn_best_responses_only is not NotProvided:
                self.learn_best_responses_only = learn_best_responses_only
            if beta_eps is not NotProvided:
                self.beta_eps = beta_eps
            if warmup_steps is not NotProvided:
                self.warmup_steps = warmup_steps
            if best_response_timesteps_max is not NotProvided:
                self.best_response_timesteps_max = best_response_timesteps_max
            if max_timesteps is not NotProvided:
                self.max_timesteps = max_timesteps
            if auto_regret is not NotProvided:
                self.auto_regret = auto_regret
            if auto_population is not NotProvided:
                self.auto_population = auto_population
            if timesteps_per_level is not NotProvided:
                self.timesteps_per_level = timesteps_per_level
            if end_level is not NotProvided:
                self.end_level = end_level

            if scenarios is not NotProvided:
                self.scenarios = scenarios
                self.callbacks_class = BackgroundFocalSGDA

            return self

    return BFSGDAConfig()
