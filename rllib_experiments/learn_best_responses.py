import os

import fire
import ray

from ray import tune
from ray.rllib.algorithms.registry import ALGORITHMS
from ray.tune import register_env

from scenarios.scenarios import ScenarioMapper, ScenarioSet
from bf_sgda.config import make_bf_sgda_config
from constants import PolicyIDs
from evaluation.benchmark import PolicyCkpt
from configs import get_exp_config

num_workers = min([os.cpu_count() - 2, 64])


def learn_best_responses(
        *,
        background=["random", "deterministic_0"],
        env="collaborative_cooking__circuit",
        version="0.7",
        gpu=0,
        **kwargs
):
    ray.init(

    )
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

    #TODO : automate background naming, this was annoying to debug
    scenarios.build_from_population(
        num_players=exp_config.num_players,
        background_population=["background_" + p.name for p in background_population]
    )

    register_env(env_id, exp_config.get_maker(num_scenarios=len(scenarios)))

    dummy_env = exp_config.get_maker(num_scenarios=len(scenarios))()
    dummy_agent_id = list(dummy_env.get_agent_ids())[0]

    policies = {
        "background_" + p.name : p.get_policy_specs(dummy_env)
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
        learn_best_responses_only=True,
        scenarios=tune.grid_search(scenarios.split()),

        copy_weights_freq=exp_config.copy_weights_freq,
        copy_history_len=exp_config.copy_history_len,
        best_response_timesteps_max=exp_config.best_response_samples,
        model={
            "custom_model": exp_config.model_path,
            "max_seq_len": 20,
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
        policy_states_are_swappable=False,
    ).experimental(
        _disable_preprocessor_api=True
    ).reporting(
         min_time_s_per_iteration=5,
         min_train_timesteps_per_iteration=0,
         min_sample_timesteps_per_iteration=num_workers * exp_config.episode_length,
     )

    exp = tune.run(
        ALGO,
        name=f"best_responses_learning_v{version}",
        config=exp_config.setup_training_config(config),
        checkpoint_at_end=True,
        checkpoint_freq=300,
        keep_checkpoints_num=3,
        stop={
            "done": True
        }
    )

if __name__ == '__main__':
    fire.Fire(learn_best_responses)
