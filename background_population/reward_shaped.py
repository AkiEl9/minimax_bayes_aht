from typing import Dict


def get_policy_id(config, test_policy=False):

    d = config.copy()

    name = f'{d.pop("title", "untitled")}_'
    for k, v in d.items():
        v = str(v)[:4].replace("-", "m").replace(".", "")
        name += f"_{k}_{v}"

    if test_policy:
        name += "_TEST"

    return name



def sample(environment, title, sampling_dict: Dict[str, callable], size=1, test_policies=False):

    samples = [
        {"title": title,
         **{
            k: sampler()

            for k, sampler in sampling_dict.items()
            },
         }
        for _ in range(size)
    ]

    dummy_agent_id = list(environment.get_agent_ids())[0]

    return {get_policy_id(d, test_policy=test_policies): (
            None,
            environment.observation_space[dummy_agent_id],
            environment.action_space[dummy_agent_id],
            d
        )
        for d in samples
    }
