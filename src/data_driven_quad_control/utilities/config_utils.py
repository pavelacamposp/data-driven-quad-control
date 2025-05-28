from typing import Any

import yaml


def load_yaml_config(path: str) -> Any:
    with open(path, "r") as file:
        return yaml.safe_load(file)
