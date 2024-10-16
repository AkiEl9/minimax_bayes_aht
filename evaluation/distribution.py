import json

import numpy as np

from evaluation import ray_results, pyplot_utils
import fire

from utils import SmoothMetric


def render(
        distribution_overtime: list = None,
        timesteps: list = None,
        ray_results_path = None,
        name="tmp.png",
        smoothing=0.5
):

    if distribution_overtime is None:
        if isinstance(ray_results_path, list):
            results = [ray_results.load(p) for p in ray_results_path]
        else:
            results = [ray_results.load(ray_results_path)]

        distribution_overtime = []

        for run_id, result in enumerate(results):
            distribution_overtime.append([])
            timesteps = []

            smoothed_distribution = None
            for i, res in enumerate(result):

                cur_distribution = np.array([
                    res[k] for k in sorted(res.keys()) if "smoothed_probability" in k
                ])
                if i == 0:
                    smoothed_distribution = SmoothMetric(lr=smoothing, init_value=cur_distribution)
                else:
                    smoothed_distribution.update(cur_distribution)

                distribution_overtime[run_id].append(smoothed_distribution.get())
                timesteps.append(res["num_agent_steps_trained"])
    else:
        distribution_overtime = [distribution_overtime]

    pyplot_utils.plot_distribution(
        distribution_overtime,
        timesteps=timesteps,
        name=name
    )



if __name__ == '__main__':
    fire.Fire(render)