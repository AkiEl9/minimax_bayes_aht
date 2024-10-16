import os
import pickle
from collections import defaultdict
from typing import Optional, Dict, Tuple, Union

import gymnasium as gym
import numpy as np
from ray.rllib import Policy, SampleBatch, BaseEnv
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env import EnvContext
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.typing import EnvType, AgentID, PolicyID, EpisodeType

import evaluation.distribution
from bf_sgda.auto_bf_sgda import ABFSGDA
from constants import Paths, PolicyIDs
from bf_sgda.bf_sgda import BFSGDA
from evaluation.benchmark import NamedPolicy
from scenarios.scenarios import ScenarioSet, Scenario
from utils import inject_callback


class BackgroundFocalSGDA(DefaultCallbacks):

    def __init__(self):
        super().__init__()

        self.beta: BFSGDA = None
        self.scenarios: ScenarioSet =  None

    def on_sub_environment_created(
        self,
        *,
        worker: "RolloutWorker",
        sub_environment: EnvType,
        env_context: EnvContext,
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.scenarios = worker.config.scenarios

    def on_algorithm_init(
            self,
            *,
            algorithm: "Algorithm",
            **kwargs,
    ) -> None:
        from ray.rllib.utils.checkpoints import get_checkpoint_info

        local_worker = algorithm.workers.local_worker()
        weights = {}
        for pid, policy in local_worker.policy_map.items():

            if not isinstance(policy, NamedPolicy):
                continue

            p_name = policy.name
            print(f"Loading weights for Policy {p_name}...")
            checkpoint_info = get_checkpoint_info(Paths.NAMED_POLICY.format(name=p_name,
                                                                            env=algorithm.config["env"]))
            with open(checkpoint_info["state_file"], "rb") as f:
                state = pickle.load(f)

            weights[pid] = state["weights"]

            print(f"Loaded weights for Policy {p_name}.")

        local_worker.set_weights(weights)
        algorithm.workers.sync_weights()
        print("Synched weights for all workers")

        if algorithm.config.auto_regret and algorithm.config.auto_population:
            self.beta = ABFSGDA(algorithm)
        else:
            self.beta = BFSGDA(algorithm)
        del self.scenarios

    def on_postprocess_trajectory(
            self,
            *,
            worker: "RolloutWorker",
            episode: Episode,
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
            **kwargs,
    ) -> None:
        """
        Swap rewards to mean focal per capita return
        """


        focal_rewards = [
            batch[SampleBatch.REWARDS] for agent_id, (policy_id, policy_cls, batch) in original_batches.items()
            if "background" not in policy_id
        ]

        mean_focal_per_capita = np.mean(focal_rewards, axis=0)
        postprocessed_batch[SampleBatch.REWARDS][:] = mean_focal_per_capita


    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:

        # Force policies to be selected through the mapping function now
        for agent_id in base_env.get_agent_ids():
            episode.policy_for(agent_id)

        sub_env = base_env.get_sub_environments(as_dict=True)[env_index]

        policies = list(episode._agent_to_policy.values())
        scenario_name = Scenario.get_scenario_name(policies)
        scenario_id = self.scenarios.scenario_to_id[scenario_name]
        setattr(episode, "policies", policies)
        setattr(episode, "scenario", scenario_name)
        setattr(sub_env, "current_scenario_id", scenario_id)
        # might be needed.
        #setattr(sub_env, "agent_reward_weights", )

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        if "cooking" in worker.config.env:
            for agent_id in episode.get_agents():
                r =  episode._agent_reward_history[agent_id][-1]
                if r  in [1., 21.]:
                    episode.user_data["pseudo_rewards"][agent_id] += r
        elif "clean_up" in worker.config.env:
            for agent_id in episode.get_agents():
                r =  episode._agent_reward_history[agent_id][-1]
                if r  < 1.:
                    episode.user_data["pseudo_rewards"][agent_id] += r


    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        episode.user_data["pseudo_rewards"] = defaultdict(float)

    def on_episode_end(
            self,
            *,
            worker: "RolloutWorker",
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        focal_rewards = [
            episode.agent_rewards[agent_id, policy_id] - episode.user_data["pseudo_rewards"][agent_id] for agent_id, policy_id in episode.agent_rewards
            if "background" not in policy_id
        ]
        episodic_mean_focal_per_capita = sum(focal_rewards) / len(focal_rewards)

        episode.custom_metrics[f"{episode.scenario}_utility"] = episodic_mean_focal_per_capita

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            result: dict,
            **kwargs,
    ) -> None:
        self.beta.update(result)

        if result["done"] and not self.beta.config.learn_best_responses_only:

            # Dump learned distribution as test_set
            test_set_name = ""
            run_name = ""
            if not self.beta.config.self_play:
                if self.beta.config.auto_population:
                    if self.beta.config.beta_lr == 0.:
                        run_name = "fictitious_play"
                    else:
                        run_name = "auto_sgda"
                elif self.beta.config.auto_regret:
                    run_name = "minimax_auto_regret"
                elif self.beta.config.beta_lr == 0.:
                    run_name = "uniform"
                elif self.beta.config.use_utility:
                    run_name = "maximin_utility"
                else:
                    run_name = "minimax_regret"

                test_set_name = f"train_set_{run_name}"

                dump_path = Paths.TEST_SET.format(
                    env=self.beta.config.env,
                    name=test_set_name,
                )

                self.beta.scenarios.to_YAML(
                    dump_path,
                    eval_config={
                        "distribution": self.beta.smoothed_beta.get().tolist(),
                        "num_episodes": 1000
                    }
                )
            else:
                run_name = "self_play"

            state = algorithm.__getstate__()
            policy_state = state["worker"]["policy_states"][PolicyIDs.MAIN_POLICY_ID]

            policy_name = run_name
            validate_policy_id(policy_name, error=True)
            policy_path = Paths.NAMED_POLICY.format(env=self.beta.config.env, name=policy_name)

            if os.path.exists(policy_path):
                new_path = "{path}_{i}"
                i = 2
                while os.path.exists(new_path.format(path=policy_path, i=i)):
                    i += 1
                policy_path = new_path.format(path=policy_path, i=i)
            os.makedirs(policy_path, exist_ok=True)
            policy = algorithm.get_policy(PolicyIDs.MAIN_POLICY_ID)
            print("Exporting policy to", policy_path, "...")
            policy.export_checkpoint(policy_path, policy_state=policy_state)
            print("Successfully exported policy at", policy_path)

            fig_path = Paths.FIGURES.format(env=self.beta.config.env, name=run_name)
            spath = fig_path.split(os.sep)
            parent_dir = os.sep.join(spath[:-1])
            filename, ext = spath[-1].split(".")
            os.makedirs(
                parent_dir, exist_ok=True
            )
            i = 2
            while os.path.exists(fig_path):
                fig_path = parent_dir + os.sep + filename + f"_{i}." + ext
                i += 1

            evaluation.distribution.render(
                distribution_overtime=self.beta.distribution_history,
                timesteps=self.beta.timesteps,
                name=fig_path
            )


class SocialRewards(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.reward_weights = {}
    
    def export_policies(self, algo, pid_suffix=None):
        state = algo.__getstate__()
        policy_states = state["worker"]["policy_states"]

        for bg_policy_name, policy_state in policy_states.items():
            if "random" in bg_policy_name:
                continue
                
            if pid_suffix is not None:
                file_name = bg_policy_name + "_" + pid_suffix
            else:
                file_name = bg_policy_name
            
            policy_path = Paths.NAMED_POLICY.format(env=algo.config.env, name=file_name)
            if os.path.exists(policy_path):
                new_path = "{path}_{i}"
                i = 2
                while os.path.exists(new_path.format(path=policy_path, i=i)):
                    i += 1
                policy_path = new_path.format(path=policy_path, i=i)
            os.makedirs(policy_path, exist_ok=True)
            policy = algo.get_policy(bg_policy_name)
            print("Exporting policy to", policy_path, "...")
            policy.export_checkpoint(policy_path, policy_state=policy_state)
            print("Successfully exported policy at", policy_path)

    def on_algorithm_init(
        self,
        *,
        algorithm: "Algorithm",
        **kwargs,
    ) -> None:

        setattr(algorithm, "base_cleanup", algorithm.cleanup)

        def on_algorithm_save(algo):

            self.export_policies(algo)
            algo.base_cleanup()

        algorithm.cleanup = on_algorithm_save.__get__(algorithm, type(algorithm))

    def on_postprocess_trajectory(
            self,
            *,
            worker: "RolloutWorker",
            episode: Episode,
            agent_id: AgentID,
            policy_id: PolicyID,
            policies: Dict[PolicyID, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
            **kwargs,
    ) -> None:

        other_rewards = [
            batch[SampleBatch.REWARDS]
            for aid, (pid, policy_cls, batch) in original_batches.items()
            if aid != agent_id
        ]
        own_rewards = original_batches[agent_id][2][SampleBatch.REWARDS]

        num_players = len(original_batches)



        mean_other_rewards = np.mean(other_rewards, axis=0)

        social_weight = policies[policy_id].config["social_weight"]
        assert -1 <= social_weight <= 2, social_weight

        # so that 0.5 reflects equal weight for own and other rewards

        alpha_e = np.log2(num_players)
        selfishness = 1-social_weight
        alpha = np.power(np.abs(selfishness), alpha_e) * np.sign(selfishness)

        beta_e = np.log2(num_players / (num_players-1))
        selflessness = social_weight
        beta = np.power(np.abs(selflessness), beta_e) * np.sign(selflessness)
        norm = np.abs(alpha) + np.abs(beta)

        risk_aversion = policies[policy_id].config["risk_aversion"]

        episode.custom_metrics[policy_id + "_trajectory_other_rewards"] = np.sum(mean_other_rewards)
        episode.custom_metrics[policy_id + "_trajectory_own_rewards"] = np.sum(own_rewards)

        r = (
                own_rewards * alpha
                +
                mean_other_rewards * beta
        ) / norm
        r = np.maximum(0., r) + np.minimum(0., risk_aversion * r)
        postprocessed_batch[SampleBatch.REWARDS][:] = r

        episode.custom_metrics[policy_id + "_trajectory_optimized_rewards"] = np.sum(r)
