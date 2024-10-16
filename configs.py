from dataclasses import dataclass, asdict, field
from functools import partial
from typing import Type, Any, List

from ray.rllib.algorithms import AlgorithmConfig
from environments.rllib.scenario_wrapper import ScenarioWrapper, ScenarioWrapperDict
from meltingpot_lib.examples.rllib.utils import MeltingPotEnv
from meltingpot_lib.examples.rllib.wrappers.downsamplesubstrate_wrapper import DownSamplingSubstrateWrapper
from meltingpot_lib.meltingpot import substrate
from meltingpot_lib.meltingpot.configs import substrates as substrate_configs



@dataclass
class EnvConfig:

    episode_length: int
    model_path: str
    num_players: int = 2
    algo: str = "undefined"
    _env_name: str = "undefined"

    IMPALA_CONFIG = dict()
    PPO_CONFIG = dict()
    BG_SGDA_CONFIG = dict()

    def env_config(self):
        d = asdict(self)
        to_remove = [
            "_env_name", "model_path", "algo"
        ] + [k for k in d if "_CONFIG" in k]
        for k in to_remove:
            del d[k]
        return d

    def get_env_id(self):
        pass

    def get_maker(self, num_scenarios=1, **kwargs):
        pass

    def setup_training_config(self, config: AlgorithmConfig):
        if self.algo == "IMPALA":
            return config.training(
                ** self.IMPALA_CONFIG
            )
        elif self.algo == "PPO":
            return config.training(
                ** self.PPO_CONFIG
            )
        else:
            raise NotImplementedError

    def setup_sgda_config(self, config: AlgorithmConfig):
        return config.training(
            ** self.BG_SGDA_CONFIG
        )


    @property
    def background_samples(self):
        return 1_000_000

    @property
    def best_response_samples(self):
        return 1_000_000

    @property
    def bf_sdgda_samples(self):
        return 1_000_000

    @property
    def rollout_fragment_length(self):
        return 20

    @property
    def max_seq_lens(self):
        return 20

    @property
    def copy_weights_freq(self):
        return 1
    @property
    def copy_history_len(self):
        return 30

    @property
    def timesteps_per_level(self):
        return 2_500_000

    @property
    def end_level(self):
        return 20

    @staticmethod
    def sampling_dict(random):

        return dict(
        social_weight=partial(random.uniform, -0.2, 1.2),
        risk_aversion=partial(random.uniform, 0.1, 2.)
    )



@dataclass
class CookingCircuit(EnvConfig):

    _env_name: str = "collaborative_cooking__circuit"
    model_path: str = "models.deep_rl.lstm_meltingpot.LSTMMeltingPot"
    algo: str = "PPO"

    PPO_CONFIG = dict(
        lambda_=0.95,
        lr=2e-4,
        sgd_minibatch_size=1000,
        num_sgd_iter=2,
        train_batch_size=1000 * 64, # 126
        use_critic=True,
        use_gae=True,
        grad_clip=4.,

    )

    BG_SGDA_CONFIG = dict(
        beta_lr=4e-1,
        beta_smoothing=0.99,
        warmup_steps=1,
        beta_eps=5e-2,
        utility_smoothing=0.7,
    )


    substrate_name: str = _env_name
    player_roles: List[str] = substrate.get_config(substrate_name).default_player_roles

    episode_length: int = 1000
    num_players: int = len(player_roles)

    def get_env_id(self):
        return self.substrate_name

    def get_maker(self, num_scenarios=1, evaluation=False, **kwargs):

        def env_maker(config=None):
            config = substrate_configs.get_config(self.substrate_name)
            if not evaluation:
                config.cooking_pot_pseudoreward = 1.
            # We are learning background policies
            if num_scenarios == 1:
                config.delivery_global_reward = False

            env = substrate.build_from_config(config, roles=self.player_roles)
            #env = DownSamplingSubstrateWrapper(env)  # vv
            return ScenarioWrapperDict(MeltingPotEnv)(env=env, num_scenarios=num_scenarios)

        return env_maker

    def build_substrate(self):
        return substrate.build(self.substrate_name, roles=self.player_roles)

    @property
    def background_samples(self):
        return 5e6
        
    @property
    def best_response_samples(self):
        return 1e7

    @property
    def bf_sdgda_samples(self):
        return 4e7

    @property
    def timesteps_per_level(self):
        return 4e7 // 15 #1e7

    @property
    def rollout_fragment_length(self):
        return 100

    @property
    def copy_weights_freq(self):
        return 1
    @property
    def copy_history_len(self):
        return 4

    @property
    def end_level(self):
        return 10


@dataclass
class CookingCramped(CookingCircuit):
    _env_name: str = "collaborative_cooking__cramped"
    substrate_name: str = _env_name

    @property
    def background_samples(self):
        return 4e6

    @property
    def best_response_samples(self):
        return 1e7


@dataclass
class CookingForced(CookingCircuit):
    _env_name: str = "collaborative_cooking__forced"
    substrate_name: str = _env_name

    @property
    def background_samples(self):
        return 4e6

    @property
    def best_response_samples(self):
        return 1e7

@dataclass
class CookingAsymmetric(CookingCircuit):
    _env_name: str = "collaborative_cooking__asymmetric"
    substrate_name: str = _env_name

    @property
    def background_samples(self):
        return 4e6

    @property
    def best_response_samples(self):
        return 1e7


@dataclass
class CookingRing(CookingCircuit):
    _env_name: str = "collaborative_cooking__ring"
    substrate_name: str = _env_name



@dataclass
class Cleanup(CookingCircuit):
    _env_name: str = "clean_up"
    model_path: str = "models.deep_rl.lstm_meltingpot.CleanupModel"

    PPO_CONFIG = dict(
        lambda_=0.95,
        lr=2e-4,
        sgd_minibatch_size=1000,
        num_sgd_iter=1,
        train_batch_size=1000 * 32,  # 126 # 64
        use_critic=True,
        use_gae=True,
        grad_clip=4.,
    )

    substrate_name: str = _env_name
    player_roles: List[str] = substrate.get_config(substrate_name).default_player_roles
    episode_length: int = 1000
    num_players: int = len(player_roles)

    def get_maker(self, num_scenarios=1, evaluation=False, **kwargs):

        def env_maker(config=None):
            config = substrate_configs.get_config(self.substrate_name)
            if not evaluation:
                config.cleaning_reward = 0.01

            env = substrate.build_from_config(config, roles=self.player_roles)
            return ScenarioWrapperDict(MeltingPotEnv)(env=env, num_scenarios=num_scenarios)

        return env_maker

    @property
    def max_seq_lens(self):
        return 20

    @property
    def background_samples(self):
        return 5e6

    @property
    def best_response_samples(self):
        return 1e7 / 7

    @property
    def bf_sdgda_samples(self):
        return 8e6

    @property
    def timesteps_per_level(self):
        return 8e6 // 15  # 1e7




ENVS = {
    CookingCircuit._env_name : CookingCircuit,
    CookingForced._env_name : CookingForced,
    CookingRing._env_name : CookingRing,
    CookingCramped._env_name: CookingCramped,
    CookingAsymmetric._env_name: CookingAsymmetric,

    Cleanup._env_name: Cleanup,

}


def get_exp_config(environment_name) -> Type[EnvConfig]:
    """
    :param environment_name: base name of the env
    :return: helper function to build config, env maker, and env_id as a function its config
    """
    env = ENVS.get(environment_name, None)
    if env is None:
        raise ValueError(f"Environment '{environment_name}' could not be found, available environments: {list(ENVS.keys())}")
    return env
