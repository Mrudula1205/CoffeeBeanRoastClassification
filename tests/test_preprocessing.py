"""
Unit tests for src/coffee_roast_ai/preprocessing.py

Validates that:
- process_image produces the correct output shape and dtype
- pixel values are normalised to [0, 1] (matching training 1/255 rescale)
- both PIL Image objects and file-path strings are accepted
- custom target sizes work
- get_color_metrics returns the correct per-channel mean
"""

import os
import numpy as np
import pytest
from PIL import Image

from src.coffee_roast_ai.preprocessing import process_image, get_color_metrics


class TestProcessImage:
    def test_output_shape_from_pil(self):
        img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        result = process_image(img)
        assert result.shape == (1, 224, 224, 3), "Output must be (1, 224, 224, 3)"

    def test_pixel_values_normalised(self):
        """Max-white image should map to 1.0 after rescale."""
        img = Image.fromarray(np.uint8(np.ones((224, 224, 3)) * 255))
        result = process_image(img)
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_output_dtype_float32(self):
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        result = process_image(img)
        assert result.dtype == np.float32, "Output must be float32"

    def test_process_image_from_path(self, tmp_path):
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        p = tmp_path / "sample.jpg"
        img.save(p)
        result = process_image(str(p))
        assert result.shape == (1, 224, 224, 3)

    def test_custom_target_size(self):
        img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
        result = process_image(img, target_size=(128, 128))
        assert result.shape == (1, 128, 128, 3)

    def test_batch_dimension_is_one(self):
        img = Image.fromarray(np.uint8(np.random.rand(50, 50, 3) * 255))
        result = process_image(img)
        assert result.shape[0] == 1, "Batch dimension must be 1 for single-image inference"


class TestGetColorMetrics:
    def test_returns_three_channels(self):
        arr = np.ones((1, 224, 224, 3)) * 0.5
        metrics = get_color_metrics(arr)
        assert len(metrics) == 3

    def test_correct_mean_value(self):
        arr = np.ones((1, 224, 224, 3)) * 0.5
        metrics = get_color_metrics(arr)
        assert all(abs(m - 0.5) < 1e-5 for m in metrics), "Mean should be ~0.5"

    def test_black_image_mean_is_zero(self):
        arr = np.zeros((1, 224, 224, 3))
        metrics = get_color_metrics(arr)
        assert all(m == 0.0 for m in metrics)
