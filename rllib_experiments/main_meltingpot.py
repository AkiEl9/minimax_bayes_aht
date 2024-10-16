import fire
import numpy as np
import ray

from scenarios.scenarios import ScenarioSet
from constants import Paths
from configs import get_exp_config
from utils import run_fire_script


def exp(
        version="1.0",
        env="collaborative_cooking__circuit",
        train_community_sizes=[2,3,5], #2,3,4
        test_community_sizes=[2,3,5],
        gpu=0,
        communities_seed=0,
        n_runs=3,
        **env_dict,
):
    env_dict = dict(
        **env_dict
    )
    env_config = get_exp_config(env)(**env_dict)
    rng = np.random.default_rng(communities_seed)
    train_seeds = rng.integers(0, 1e8, len(train_community_sizes))
    test_seeds = rng.integers(0, 1e8, len(test_community_sizes))

    # Learn some background policies
    train_population = [pid for (community_seed, community_size) in zip(train_seeds, train_community_sizes) for pid in
                        run_fire_script(
                            script_name="rllib_experiments.learn_community",
                            input_dict=dict(
                                size=community_size,
                                env=env,
                                version=version,
                                communities_seed=community_seed,
                                skip=True,
                                gpu=gpu,
                                **env_dict
                            ),
                            output_dtype=list,
                            #print_output=True
                        )
                        ] + ["noop"]


    test_population = [pid for (community_seed, community_size) in zip(test_seeds, test_community_sizes) for pid in
                       run_fire_script(
                           script_name="rllib_experiments.learn_community",
                           input_dict=dict(
                               size=community_size,
                               env=env,
                               version=version,
                               communities_seed=community_seed,
                               skip=True,
                               gpu=gpu,
                               test_policies=True,
                               **env_dict
                           ),
                           output_dtype=list,
                       )
                       ]

    test_set_name = "main_test_set"
    test_set = ScenarioSet()

    test_set.build_from_population(2, [
        "background_" + pid for pid in test_population
    ])
    test_set.to_YAML(
        Paths.TEST_SET.format(env=env_config.get_env_id(), name=test_set_name),
        eval_config={
            "num_episodes": 50
        }
    )

    # Learn their best responses
    run_fire_script(
        script_name="rllib_experiments.learn_best_responses",
        input_dict=dict(background=train_population,
                        env=env,
                        version=version,
                        gpu=gpu,
                        **env_dict),
        print_output=True
    )
    run_fire_script(
        script_name="rllib_experiments.learn_best_responses",
        input_dict=dict(background=test_population,
                        env=env,
                        version=version,
                        gpu=gpu,
                        **env_dict),
        print_output=True
    )

    for run_id in range(n_runs):

        # Run regret, utility, uniform, self_play
        input_dict = dict(
            background=train_population,
            version=version,
            env=env,
            gpu=gpu,
            use_utility=False,
            self_play=False,
            **env_dict
        )
        #
        # -> fixed uniform scenario distribution
        fixed_uniform_dict = input_dict.copy()
        fixed_uniform_dict["beta_lr"] = 0.0
        fixed_uniform_dict["use_utility"] = True
        run_fire_script(
            script_name="rllib_experiments.run_bf_sgda",
            input_dict=fixed_uniform_dict,
            print_output=True
        )

        # -> fixed self-play
        sp_dict = input_dict.copy()
        sp_dict["self_play"] = True
        sp_dict["beta_lr"] = 0.0
        run_fire_script(
            script_name="rllib_experiments.run_bf_sgda",
            input_dict=sp_dict,
            print_output=True
        )

        # -> regret
        run_fire_script(
            script_name="rllib_experiments.run_bf_sgda",
            input_dict=input_dict,
            print_output=True
        ),

        # -> utility
        input_dict["use_utility"] = True
        run_fire_script(
            script_name="rllib_experiments.run_bf_sgda",
            input_dict=input_dict,
        )


        # -> fictitious_play
        run_fire_script(
            script_name="rllib_experiments.run_auto_bf_sgda",
            input_dict=dict(
                version=version,
                env=env,
                gpu=gpu,
                beta_lr=0.0,
                **env_dict
            ),
        )


if __name__ == '__main__':
    fire.Fire(exp)
