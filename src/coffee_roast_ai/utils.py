import yaml


def read_params(config_path="params.yaml"):
    """
    Reads the YAML configuration file and returns a dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
