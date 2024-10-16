import os
import fire
from configs import get_exp_config
from evaluation.benchmark import Evaluation
import ray
import logging
from evaluation.scores import render as render_scores

class Main:

    def benchmark_policies(
            self,
            policies=["deterministic_0", "deterministic_1", "random"],
            sets=["deterministic_set_0"],
            env="collaborative_cooking__circuit",
            n_runs=1,
            **env_config
    ):
        ray.logger.setLevel(logging.ERROR)

        # TODO : for each set, save the stats in data/

        for test_set in sets:
            evaluation = Evaluation(test_set, env, env_config=env_config)
            set_eval = evaluation.load()

            past_evaluations = list(set_eval.keys())

            for policy_base in policies:
                # We do not want to remove existing scores, probably
                for i in range(n_runs):
                    if i > 0:
                        # policy 0 has no suffix
                        policy = policy_base + f"_{i + 1}"
                    else:
                        policy = policy_base

                    if policy not in past_evaluations:
                        set_eval.update(
                            **evaluation.evaluate_policy(policy)
                        )
                        evaluation.save(set_eval)

        render_scores(
            policies=policies,
            env=env,
            sets=sets,
            n_runs=n_runs,
            **env_config
        )

    def fast_eval(
            self,
            policies=["deterministic_0", "deterministic_1", "random"],
            sets=["deterministic_set_0"],
            env="collaborative_cooking__circuit",
            n_runs=1,

            **env_config
    ):
        ray.logger.setLevel(logging.ERROR)

        exp_config = get_exp_config(env)(**env_config)

        indexed_policies = policies + [
            p + f"_{run_id+1}" for p in policies for run_id in range(1, n_runs)
        ]

        jobs = Evaluation.get_jobs(indexed_policies, sets, exp_config)
        evaluations = {
            set_name: Evaluation(set_name, env, env_config)
            for set_name in sets
        }

        past_evaluations = []
        for set_name, evaluation in evaluations.items():
            past_set_eval = evaluation.load()
            for policy_name in past_set_eval:
                past_evaluations.append((policy_name, set_name))
        jobs = [
            job for job in jobs if (job[1], job[4]) not in past_evaluations
        ]

        print("Running the following evaluation jobs:")
        for i, j in enumerate(jobs):
            print(f"{i}.\t{j}")

        results = Evaluation.run(jobs, exp_config)

        for set_name, evaluation in evaluations.items():
            results[set_name].update(**evaluation.load())
            evaluation.save(results[set_name])

        render_scores(
            policies=policies,
            env=env,
            sets=sets,
            n_runs=n_runs,
            **env_config
        )


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    fire.Fire(Main)
