from collections import defaultdict

import fire
import numpy as np

from configs import get_exp_config
from constants import Paths
from evaluation.benchmark import Evaluation
from scenarios.scenarios import ScenarioSet

from evaluation.pyplot_utils import label_dict


def write_table(
        policies,
        n_runs,
        train_set,
        train_set_distributions,
        test_sets,

        envs,
        **env_config
):

    scores = defaultdict(
        lambda: defaultdict(
            lambda : defaultdict(lambda: np.full((len(envs), n_runs), fill_value=np.nan, dtype=np.float32))
        )
    )
    metrics = {
        "overall_score": "$\\perf$",
        "worst_case_utility": "$U^-$",
        "worst_case_regret": "$R^+$"
    }
    worst_case_utility_metrics = {
        "focal_per_capita_mean": "$\perf(\maximinprior)$"
    }
    worst_case_regret_metrics = {
        "focal_per_capita_mean": "$\perf(\minimaxprior)$"
    }

    def ste(values, axis=None):
        return np.std(values, axis=axis) / np.sqrt(n_runs)

    if train_set is not None:

        for env_num, env in enumerate(envs):
            exp_config = get_exp_config(env)(**env_config)

            eval_obj = Evaluation(
                test_set=train_set,
                env=env,
                env_config=env_config
            )
            distributions = {
                train_set_distribution: ScenarioSet.from_YAML(
                    Paths.TEST_SET.format(env=exp_config.get_env_id(), name=train_set_distribution)
                ).eval_config["distribution"]
                for train_set_distribution in train_set_distributions
            }
            evaluation = eval_obj.load()
            for base_pid in policies:
                for run_id in range(1, 1 + n_runs):
                    if run_id == 1:
                        pid = base_pid
                    else:
                        pid = base_pid + f"_{run_id}"

                    for metric, table_label in metrics.items():
                        scores[train_set][base_pid][table_label][env_num, run_id-1] = evaluation[pid][metric]
                    for distribution_name, distribution in distributions.items():
                        labels = worst_case_regret_metrics if "regret" in distribution_name else worst_case_utility_metrics
                        for metric, table_label in labels.items():
                            score = np.array([
                                evaluation[pid]["per_scenario"][scenario_name][metric] * prob
                                for scenario_name, prob in zip(sorted(evaluation[pid]["per_scenario"].keys()), distribution)
                            ]).mean()
                            scores[train_set][base_pid][table_label][env_num, run_id-1] = score
        string = "& $\perf$  & $\perf(\maximinprior)$ & $\perf(\minimaxprior)$ & $U^-$ & $R^+$ \\\\ \midrule"
        latex_table = string
        for pid in policies:
            row_string = f"""\n
            {label_dict[pid]}"""
            for metric in string.split("&")[1:]:
                metric = metric.strip().rstrip(" \\\\ \midrule")
                row_string += f" & ${scores[train_set][pid][metric].mean():.1f} \pm {ste(scores[train_set][pid][metric], axis=1).mean():.1f}$"
            latex_table += row_string + " \\\\"
        print("Train set :")
        print(
            latex_table
        )
        print()

    for env_num, env in enumerate(envs):
        for test_set in test_sets:
            exp_config = get_exp_config(env)(**env_config)
            eval_obj = Evaluation(
                test_set=test_set,
                env=env,
                env_config=env_config
            )
            evaluation = eval_obj.load()
            for base_pid in policies:
                for run_id in range(1, 1+n_runs):
                    if run_id == 1:
                        pid = base_pid
                    else:
                        pid = base_pid + f"_{run_id}"

                    for metric, table_label in metrics.items():
                        scores[test_set][base_pid][table_label][env_num, run_id-1] = evaluation[pid][metric]

    metric_header = "& $\perf$  & $U^-$ & $R^+$"
    string = metric_header * len(test_sets) + "\\\\ \midrule"
    latex_table = string
    for pid in policies:
        row_string = f"""\n
            {label_dict[pid]}"""
        for test_set in test_sets:
            for metric in metric_header.split("&")[1:]:
                metric = metric.strip().rstrip(" \\\\ \midrule")
                row_string += f" & ${scores[test_set][pid][metric].mean():.1f} \pm {ste(scores[test_set][pid][metric], axis=1).mean():.1f}$"
        latex_table += row_string + " \\\\"
    print("Test sets :")
    print(
        latex_table
    )
    print()





if __name__ == '__main__':
    print("TODO: FIX MELTINGPOT CRAMPED EXPECTED REGRET")
    fire.Fire(write_table)