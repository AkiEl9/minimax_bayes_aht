import os
import yaml
import numpy as np

from constants import Paths


def load_best_response_utilities(env_name: str) -> dict:
    path = Paths.BEST_RESPONSES.format(env=env_name)
    best_response_utilities = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            best_response_utilities = yaml.safe_load(f)
        if best_response_utilities is None:
            best_response_utilities = {}

    return best_response_utilities


def save_best_response_utilities(env_name: str, best_response_utilities: dict):
    path = Paths.BEST_RESPONSES.format(env=env_name)
    parent = os.sep.join(path.split(os.sep)[:-1])
    os.makedirs(parent, exist_ok=True)
    print("Saving bru to", path, "...")

    to_save = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            to_save = yaml.safe_load(f)
            if to_save is None:
                to_save = {}

    # Update the best responses with the better ones found here.
    for scenario, new_value in best_response_utilities.items():
        if to_save.get(scenario, -np.inf) < new_value:
            to_save[str(scenario)] = float(new_value)

    with open(path, "w") as f:
        yaml.safe_dump(to_save, f)
