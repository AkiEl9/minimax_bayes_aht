# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs the bots trained in self_play_train.py and renders in pygame.

You must provide experiment_state, expected to be
~/ray_results/PPO/experiment_state_YOUR_RUN_ID.json
"""

import argparse
import itertools
import os.path

import dm_env
from dmlab2d.ui_renderer import pygame
import numpy as np
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
import fire

from configs import get_exp_config
from meltingpot_utils.models import RayModelPolicy
import imageio


def main(policies_path, env_name, **env_args):

    exp_config = get_exp_config(env_name)(**env_args)

    env_maker = exp_config.get_maker()
    register_env("meltingpot", env_maker)


    policies = Policy.from_checkpoint(policies_path)



    env = exp_config.build_substrate()

    all_bots = [
      RayModelPolicy(policy_id, policy)
      for policy_id, policy in policies.items()
    ]

    for bot_comp in itertools.combinations(all_bots, exp_config.num_player):
        for bots in [list(bot_comp), list(reversed(bot_comp))]:
            timestep = env.reset()
            states = [bot.initial_state() for bot in bots]
            actions = [0] * len(bots)

            # Configure the pygame display
            scale = 5
            fps = 60
            comp_name = ""
            for bot in bots:
                comp_name += bot._policy_id
            gif_path = f"data/gifs/{env_name}"
            if not os.path.exists(gif_path):
                os.makedirs(gif_path)
            recorder = imageio.get_writer(gif_path + f"/{comp_name}.gif", mode="I", duration=0.15)

            pygame.init()
            clock = pygame.time.Clock()
            pygame.display.set_caption("DM Lab2d")
            obs_spec = env.observation_spec()
            shape = obs_spec[0]["WORLD.RGB"].shape
            game_display = pygame.display.set_mode(
              (int(shape[1] * scale), int(shape[0] * scale)))

            for _ in range(exp_config.episode_length):
                obs = timestep.observation[0]["WORLD.RGB"]
                recorder.append_data(obs)
                obs = np.transpose(obs, (1, 0, 2))
                surface = pygame.surfarray.make_surface(obs)
                rect = surface.get_rect()
                surf = pygame.transform.scale(surface,
                                              (int(rect[2] * scale), int(rect[3] * scale)))

                game_display.blit(surf, dest=(0, 0))
                pygame.display.update()
                clock.tick(fps)

                for i, bot in enumerate(bots):
                    timestep_bot = dm_env.TimeStep(
                      step_type=timestep.step_type,
                      reward=timestep.reward[i],
                      discount=timestep.discount,
                      observation=timestep.observation[i])

                    actions[i], states[i] = bot.step(timestep_bot, states[i])

                timestep = env.step(actions)

                for i, r in enumerate(timestep.reward):
                    if r != 0:
                        print(i, r)
            recorder.close()


if __name__ == "__main__":
    fire.Fire(main)