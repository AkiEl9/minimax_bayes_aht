import os

import fire
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.registry import ALGORITHMS
from ray import tune
from ray.tune import register_env

from scenarios.scenarios import ScenarioMapper, ScenarioSet

from bf_sgda.config import make_bf_sgda_config
from constants import PolicyIDs
from evaluation.benchmark import PolicyCkpt

from configs import get_exp_config

num_workers = max([os.cpu_count()//2, 32])

def bf_sgda(
        *,
        background=["random", "deterministic_0"],
        version="0.11",
        env="collaborative_cooking__circuit",
        gpu=0,
        use_utility=False,
        self_play=False,
        auto_regret=False,
        beta_lr=None,
        min_replacements=1,
        max_replacements=np.inf,
        **kwargs
):

    exp_config = get_exp_config(
        environment_name=env
    )(**kwargs)

    env_config = exp_config.env_config()
    env_id = exp_config.get_env_id()

    background_population = [
        PolicyCkpt(p_name, env_id)
        for p_name in background
    ]
    scenarios = ScenarioSet()

    scenarios.build_from_population(
        num_players=exp_config.num_players,
        background_population=["background_" + p.name for p in background_population],
        min_replacements=min_replacements,
        max_replacements=max_replacements
    )

    print(scenarios.scenario_list, len(scenarios.scenario_list))

    register_env(env_id, exp_config.get_maker(len(scenarios)))

    dummy_env = exp_config.get_maker(len(scenarios))()
    dummy_agent_id = list(dummy_env.get_agent_ids())[0]
    
    policies = {
        "background_" + p.name: p.get_policy_specs(dummy_env)
        for p in background_population
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
        use_utility=use_utility,
        scenarios=scenarios,
        copy_weights_freq=exp_config.copy_weights_freq, # * 20
        copy_history_len=exp_config.copy_history_len,
        max_timesteps=exp_config.bf_sdgda_samples,
        learn_best_responses_only=False,
        self_play=self_play,
        auto_regret=auto_regret,

        model={
            "max_seq_len": exp_config.max_seq_lens,
            "custom_model": exp_config.model_path,
            "custom_model_config": {
                "n_scenarios": len(scenarios),
                "env": env_id
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

    name = F"BF_SGDA_v{version}/"
    if train_config.self_play:
        name = name + "SP"
    elif train_config.beta_lr == 0.:
        name = name + "uniform"
    elif train_config.use_utility:
        name = name + "maximin_utility"
    else:
        name = name + "minimax_regret"
    exp = tune.run(
        ALGO,
        name=name,
        config=train_config,
        checkpoint_at_end=True,
        checkpoint_freq=3000,
        keep_checkpoints_num=1,
        stop={
            "done": True
        },
    )



if __name__ == '__main__':
    fire.Fire(bf_sgda)
