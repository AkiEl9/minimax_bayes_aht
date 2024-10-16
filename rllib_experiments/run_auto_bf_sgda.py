import os

import fire
import ray
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.registry import ALGORITHMS

from ray import tune
from ray.tune import register_env

from scenarios.scenarios import ScenarioMapper, ScenarioSet

from bf_sgda.config import make_bf_sgda_config
from constants import PolicyIDs
from evaluation.benchmark import PolicyCkpt

from configs import get_exp_config

num_workers = min([os.cpu_count() // 2, 64])

def auto_bf_sgda(
        *,
        version="0.17",
        env="collaborative_cooking__circuit",
        gpu=0,
        beta_lr=None,
        **kwargs
):

    exp_config = get_exp_config(
        environment_name=env
    )(**kwargs)

    env_config = exp_config.env_config()
    env_id = exp_config.get_env_id()

    background_population = [
        f"level_{i}" for i in range(exp_config.end_level)
    ]
    scenarios = ScenarioSet()

    scenarios.build_from_population(
        num_players=exp_config.num_players,
        background_population=["background_" + name for name in background_population]
    )

    register_env(env_id, exp_config.get_maker(len(scenarios)))

    dummy_env = exp_config.get_maker(len(scenarios))()
    dummy_agent_id = list(dummy_env.get_agent_ids())[0]
    
    policies = {
        "background_" + name: (
            None,
            dummy_env.observation_space[dummy_agent_id],
            dummy_env.action_space[dummy_agent_id],
            {}
        )
        for name in background_population
    }

    for policy_id in (PolicyIDs.MAIN_POLICY_ID, PolicyIDs.MAIN_POLICY_COPY_ID):
        policies[policy_id] = (
            None,
            dummy_env.observation_space[dummy_agent_id],
            dummy_env.action_space[dummy_agent_id],
            {}
        )

    ALGO = exp_config.algo
    config = make_bf_sgda_config(ALGORITHMS[ALGO]()[1].__class__).training(
        scenarios=scenarios,
        auto_population=True,
        auto_regret=True,
        end_level=exp_config.end_level,
        timesteps_per_level=exp_config.timesteps_per_level,
        copy_weights_freq=exp_config.copy_weights_freq,  # * 20
        copy_history_len=exp_config.copy_history_len,

        model={
            "max_seq_len": exp_config.max_seq_lens,
            "custom_model": exp_config.model_path,
            "custom_model_config": {
                "n_scenarios": len(scenarios),
            }
        }
    ).rollouts(
        num_rollout_workers=num_workers,
        sample_async=False,
        create_env_on_local_worker=False,
        num_envs_per_worker=1,
        rollout_fragment_length=exp_config.rollout_fragment_length,
        batch_mode="truncate_episodes",
        enable_connectors=True,
    ).environment(
        env=env_id,
        env_config=env_config
    ).resources(num_gpus=gpu
    ).framework(framework="tf"
    ).multi_agent(
        policies=policies,
        policies_to_train={PolicyIDs.MAIN_POLICY_ID},
        policy_mapping_fn=ScenarioMapper(
            scenarios=scenarios
        ),
    ).experimental(
        _disable_preprocessor_api=True
    ).reporting(
         min_time_s_per_iteration=0,
         min_train_timesteps_per_iteration=0,
         min_sample_timesteps_per_iteration=num_workers * exp_config.episode_length,
     )

    train_config = exp_config.setup_sgda_config(exp_config.setup_training_config(config))
    if beta_lr is not None:
        train_config.beta_lr = beta_lr
    exp = tune.run(
        ALGO,
        name=f"BF_SGDA_v{version}/FP",
        config=train_config,
        checkpoint_at_end=True,
        checkpoint_freq=3000,
        keep_checkpoints_num=3,
        stop={
            "done": True,
        },
    )



if __name__ == '__main__':
    fire.Fire(auto_bf_sgda)
