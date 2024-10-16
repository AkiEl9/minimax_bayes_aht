from collections import defaultdict

import fire
import numpy as np
import os

from constants import Paths
from evaluation.benchmark import Evaluation
from evaluation import pyplot_utils
from probabilities.utils import distribution_to_hist

def render(
        policies=["random"],
        env="collaborative_cooking__circuit",
        sets=["train_set_uniform"],
        n_runs=1,
        **env_config
):
    data_utility = defaultdict(lambda: defaultdict(list))
    data_regret = defaultdict(lambda: defaultdict(list))

    for eval_set in sets:
        Evaluation(eval_set, env, env_config=env_config)
        evaluation = Evaluation(
            test_set=eval_set,
            env=env,
            env_config=env_config
            )
        results = evaluation.load()

        curr_set_utility = defaultdict(list)
        curr_set_regret = defaultdict(list)

        for base_policy in policies:
            for run_id in range(n_runs):
                if run_id > 0:
                    # policy 0 has no suffix
                    policy = base_policy + f"_{run_id + 1}"
                else:
                    policy = base_policy

                stats = results[policy]
                if "distribution" in evaluation.test_set.eval_config:
                    utilities = np.array([
                        stats["per_scenario"][scenario_name]["focal_per_capita_mean"]
                        for scenario_name in sorted(stats["per_scenario"].keys())
                    ])
                    regrets = np.array([
                        stats["per_scenario"][scenario_name]["regret_mean"]
                        for scenario_name in sorted(stats["per_scenario"].keys())
                    ])

                    distrib = np.array(evaluation.test_set.eval_config["distribution"])
                    distrib[distrib <= 1e-2] = 0.
                    distrib /= np.sum(distrib)
                    hist = distribution_to_hist(distrib, precision=100000)
                    sampled_utilities = utilities[hist]

                    sampled_regrets = regrets[hist]

                else:
                    utilities = np.array([
                        stats["per_scenario"][scenario_name]["focal_per_capita_mean"]
                        for scenario_name in sorted(stats["per_scenario"].keys())
                    ])
                    regrets = np.array([
                        stats["per_scenario"][scenario_name]["regret_mean"]
                        for scenario_name in sorted(stats["per_scenario"].keys())
                    ])
                    sampled_utilities = utilities
                    sampled_regrets = regrets

                data_regret[base_policy][eval_set].append(sampled_regrets)
                data_utility[base_policy][eval_set].append(sampled_utilities)
                curr_set_utility[base_policy].append(utilities)
                curr_set_regret[base_policy].append(regrets)

        path = Paths.FIGURES.format(
            env=evaluation.env_config.get_env_id(),
            name=f"barplot_utility_{eval_set}"
        )
        os.makedirs(
            os.sep.join(path.split(os.sep)[:-1]), exist_ok=True
        )

        pyplot_utils.make_grouped_barplot(
            curr_set_utility,
            name=path,
        )

        pyplot_utils.make_grouped_barplot(
            curr_set_regret,
            name=path.replace("utility", "regret"),
            plot_type="regret"
        )



    # utility boxplot
    path = Paths.FIGURES.format(
        env=evaluation.env_config.get_env_id(),
        name="boxplot_utility"
    )

    print(data_utility)
    pyplot_utils.make_grouped_boxplot(
        data_utility,
        name=path,
        whiskers=[0, 50],
        plot_type="utility"
    )

    # regret boxplot
    pyplot_utils.make_grouped_boxplot(
        data_regret,
        name=path.replace("utility", "regret"),
        whiskers=[50, 100],
        plot_type="regret"
    )


if __name__ == '__main__':
    print("TODO: CARE THAT WE NEED TO AUTOMATE PERF ON WORST CASE DISTRIBUTIONS")
    fire.Fire(render)
