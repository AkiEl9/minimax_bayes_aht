import json
import os
import numpy as np
import fire
import ray

from ray import tune
from ray.tune import register_env
from ray.rllib.algorithms.registry import ALGORITHMS

import background_population.reward_shaped
from bf_sgda.rllib_callbacks import SocialRewards
from bf_sgda.utils import RandomMapping
from configs import get_exp_config

num_workers = min([os.cpu_count() - 2, 64])

import nltk
nltk.download("words")
nltk.download("stopwords")

from nltk.corpus import stopwords, words
titles = list(sorted(list(set(words.words()) - set(stopwords.words()))))

def get_community_name(random):

    min_len = 4
    max_len = 7
    word = ""

    while not (min_len < len(word) < max_len):
        word = random.choice(titles)

    return word.upper().replace("-", "")



def learn_community(
        *,
        size=2,
        version="0.23",
        env="collaborative_cooking__circuit",
        communities_seed=0,
        skip=False,
        gpu=0,
        test_policies=False,
        **kwargs
):
    """
    Learns a subpopulation, representing agents used to cooperate with each other.
    Can an external policy zer-shot generalize to policies found in new communities ?
    """

    exp_config = get_exp_config(
        environment_name=env
    )(**kwargs)

    env_config = exp_config.env_config()
    env_id = exp_config.get_env_id()

    register_env(env_id, exp_config.get_maker(num_scenarios=1))

    dummy_env = exp_config.get_maker(num_scenarios=1)()

    ALGO = exp_config.algo

    random = np.random.default_rng(seed=communities_seed)

    community_title = f"{ALGO}_{get_community_name(random)}"
    policies = background_population.reward_shaped.sample(
        dummy_env,
        community_title,
        exp_config.sampling_dict(random),
        size=size,
        test_policies=test_policies
    )

    pids = list(policies)

    bg_policies = [
        pid for pid in policies if "random" not in pid
    ]

    if skip:
        print(json.dumps(bg_policies))
        return



    config = ALGORITHMS[ALGO]()[1].training(
        model={
            "max_seq_len": exp_config.max_seq_lens,
            "custom_model": exp_config.model_path,
            "custom_model_config": {
                "n_scenarios": 1,
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
        policies_to_train={
            pid for pid in policies if pid != "random_policy"
        },
        policy_mapping_fn=RandomMapping(pids, num_players=exp_config.num_players),
    ).experimental(
        _disable_preprocessor_api=True
    ).callbacks(
        callbacks_class=SocialRewards
    )

    exp = tune.run(
        ALGO,
        name=f"learn_background_v{version}",
        config=exp_config.setup_training_config(config),
        checkpoint_at_end=True,
        checkpoint_freq=100,
        keep_checkpoints_num=3,
        stop={
            "timesteps_total": exp_config.background_samples * size,
        },
    )

    print(json.dumps(bg_policies))


if __name__ == '__main__':
    fire.Fire(learn_community)
