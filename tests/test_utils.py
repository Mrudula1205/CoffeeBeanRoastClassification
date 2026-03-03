"""
Unit tests for src/coffee_roast_ai/utils.py

Uses the absolute path to params.yaml so these tests pass
regardless of the working directory pytest is invoked from.
"""

import os
import pytest

from src.coffee_roast_ai.utils import read_params

# Resolve params.yaml relative to this test file so the path is always valid
PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "params.yaml")


class TestReadParams:
    def test_returns_dict(self):
        config = read_params(PARAMS_PATH)
        assert isinstance(config, dict)

    def test_has_required_top_level_keys(self):
        config = read_params(PARAMS_PATH)
        for key in ("data", "model", "augmentation"):
            assert key in config, f"Missing top-level config key: '{key}'"

    def test_data_image_size(self):
        config = read_params(PARAMS_PATH)
        assert config["data"]["image_size"] == [224, 224]

    def test_data_batch_size_positive(self):
        config = read_params(PARAMS_PATH)
        assert config["data"]["batch_size"] > 0

    def test_class_names_are_four(self):
        config = read_params(PARAMS_PATH)
        assert len(config["data"]["class_names"]) == 4
        assert set(config["data"]["class_names"]) == {"Dark", "Green", "Light", "Medium"}

    def test_model_learning_rate_positive(self):
        config = read_params(PARAMS_PATH)
        assert config["model"]["learning_rate"] > 0

    def test_model_epochs_positive(self):
        config = read_params(PARAMS_PATH)
        assert config["model"]["epochs"] > 0

    def test_missing_file_raises_error(self):
        with pytest.raises((FileNotFoundError, OSError)):
            read_params("this_file_does_not_exist.yaml")
