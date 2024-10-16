import pathlib
import os

class Paths:
    base_path = str(pathlib.Path(__file__).parent.resolve()) + "/data/{dir}"
    identifier_path = "/{env}/{name}"

    TEST_SET = base_path.format(dir="test_sets") + identifier_path + ".YAML"
    NAMED_POLICY = base_path.format(dir="policies") + identifier_path
    EVAL = base_path.format(dir="evaluation") + identifier_path + ".YAML"
    FIGURES = base_path.format(dir="figures") + identifier_path + ".png"
    DATAFRAMES = base_path.format(dir="dataframes") + identifier_path + ".pkl"
    GIFS = base_path.format(dir="gifs") + identifier_path + ".gif"
    VIDEOS = base_path.format(dir="videos") + identifier_path + "/video"

    POLICIES = base_path.format(dir="policies") + "/{env}"
    BEST_RESPONSES = base_path.format(dir="best_response_utilities") + "/{env}.YAML"

    @staticmethod
    def make_path(path: str, env_name: str, obj_name: str):
        if Paths.identifier_path not in path:
            raise ValueError(f"The path should contain '{Paths.identifier_path}' but does not: {path}")

        formated_path = path.format(env=env_name, name=obj_name)
        parent_path = os.sep.join(formated_path.split(os.sep)[:-1])
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
            print("Created path:", parent_path)

        return formated_path


class PolicyIDs:
    MAIN_POLICY_ID = "MAIN_POLICY"
    MAIN_POLICY_COPY_ID = "MAIN_POLICY_COPY"  # An older version of the learn_best_responses policy