import json
import os.path
from collections import defaultdict

import fire
import numpy as np

import utils
from evaluation.pyplot_utils import make_grouped_plot, plot_distribution
from utils import SmoothMetric
import pandas as pd



def load(ray_results_path: str):

    print("opening", ray_results_path)

    with open(ray_results_path, "r") as f:
        results = [json.loads(ff) for ff in f]

    return results


def get_smoothed_metric(data, metric, is_distribution=False, smoothing=0.6):
    """

    :param datas: ray run data
    :param metric: name of the metric of interest
    :param smoothing:
    :return: dict with two entries, "timesteps" and "{metric_name}" containing a list of values
    """

    metric_overtime = []
    timesteps = []
    smoothed_metric = None

    for i, res in enumerate(data):

        if is_distribution:
            curr_metric = np.array([res[m] for m in res if metric in m])
        else:
            curr_metric = res[metric]
        if i == 0:
            smoothed_metric = SmoothMetric(lr=smoothing, init_value=curr_metric)
        else:
            smoothed_metric.update(curr_metric)

        metric_overtime.append(smoothed_metric.get())
        timesteps.append(res["timesteps_total"])

    return {
        "timesteps": timesteps,
        metric: metric_overtime
    }



class Main:


    def plot_curves(self,
                    metric,
                    smoothing,
                    plot_name="plot",
                    **approaches,
                    ):
        data = defaultdict(lambda: {metric: {"values": [], "timesteps": []}})

        for approach_name, parent_path in approaches.items():
            for run_id, subdir in enumerate(utils.get_subdir_names(parent_path)):
                d = get_smoothed_metric(load(os.path.join(parent_path, subdir, "result.json")), metric, smoothing=smoothing)
                data[approach_name][metric]["values"].append(d[metric])
                data[approach_name][metric]["timesteps"].append(d["timesteps"])

        make_grouped_plot(data, plot_name)


    def plot_distribution(self,
                    metric,
                    smoothing,
                    path,
                    plot_name="distribution"
                    ):

        data = defaultdict(lambda: {metric: {"values": [], "timesteps": []}})
        for run_id, subdir in enumerate(utils.get_subdir_names(path)):

            d = get_smoothed_metric(load(os.path.join(path, subdir, "result.json")), metric, is_distribution=True, smoothing=smoothing)
            data[""][metric]["values"].append(d[metric])
            data[""][metric]["timesteps"].append(d["timesteps"])


        plot_distribution(
            np.clip(
                np.stack(data[""][metric]["values"])
                , 0., 1.)

            , np.mean(data[""][metric]["timesteps"], axis=0), plot_name)









if __name__ == '__main__':
    fire.Fire(Main)