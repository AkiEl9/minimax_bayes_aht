from collections import defaultdict

import numpy as np
from ray.rllib.utils import try_import_tf

from scenarios.scenarios import ScenarioSet

tf1, tf, tfv = try_import_tf()
tf1.disable_eager_execution()

from meltingpot_lib.meltingpot.utils.evaluation.evaluation import evaluate_population
from meltingpot_lib.meltingpot.scenario import SCENARIOS
import fire

from constants import Paths
from meltingpot_utils.models import RayModelPolicy
from evaluation.benchmark import PolicyCkpt, Evaluation
from configs import get_exp_config
def main(
        env="collaborative_cooking__circuit",
        policies=["random", "deterministic_0"],
        n_runs=1,
        **env_config
):

    exp_config = get_exp_config(env)(**env_config)
    env_id = exp_config.get_env_id()

    evaluation = Evaluation(
        test_set="meltingpot_test_set",
        env=env_id,
        env_config=env_config
    )

    results = defaultdict(lambda: defaultdict(dict))
    num_episodes = 20 # 50
    for base_pid in policies:
        for run_id in range(n_runs):
            if run_id == 0:
                pid = base_pid
            else:
                pid = base_pid + f"_{run_id+1}"


            policy = PolicyCkpt(name=pid, env_name=env_id).make(env)
            dm_policy = RayModelPolicy(
                policy_id=pid,
                policy=policy
            )
            population = {pid: dm_policy}
            names_by_role = {role: [pid]
                             for role in exp_config.player_roles}
            scenarios = [
                scenario for scenario in SCENARIOS if env_id in scenario
            ]

            for scenario in scenarios:

                print(f"Evaluating policy '{pid}' on scenario '{scenario}'")
                df = evaluate_population(
                    population=population,
                    names_by_role=names_by_role,
                    scenario=scenario,
                    num_episodes=num_episodes,
                    #video_root=Paths.make_path(Paths.VIDEOS, env_name=env_id, obj_name=scenario + "/"+ pid)
                )
                df.to_pickle(Paths.make_path(Paths.DATAFRAMES, env_name=env_id, obj_name=scenario + "_" + pid))
                results[pid][scenario]["focal_per_capita_mean"] =  float(df["focal_per_capita_return"].mean())
                results[pid][scenario]["focal_per_capita_ste"] = float(np.std(df["focal_per_capita_return"]) / np.sqrt(num_episodes))
                for metric, series in df.items():
                    if "INFO" in metric or "movement" in metric:
                        results[pid][scenario][metric] = float(series.mean())

    prev_res  = evaluation.load()
    results = {
        pid: evaluation.post_process_policy_eval(evaluation.test_set, exp_config=exp_config, res=dict(**res))
        for pid, res in results.items()
    }

    prev_res.update(**results)

    evaluation.save(prev_res)


if __name__ == '__main__':
    fire.Fire(main)