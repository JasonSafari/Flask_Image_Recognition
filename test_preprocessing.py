"""Tests for the preprocessing and prediction functions in model.py."""
from io import BytesIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import model


def make_png_bytes(size=(50, 80), mode="RGB", color="white"):
    """Build an in-memory PNG image and return it as a BytesIO object."""
    buffer = BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# ---------- preprocess_img ----------

def test_preprocess_output_shape():
    """Any RGB image is resized to a single-image batch of 224x224x3."""
    result = model.preprocess_img(make_png_bytes(size=(50, 80)))
    assert result.shape == (1, 224, 224, 3)


def test_preprocess_scales_white_to_one():
    """Edge case: an all-white image is normalized to exactly 1.0."""
    result = model.preprocess_img(make_png_bytes(color="white"))
    assert np.allclose(result, 1.0)


def test_preprocess_scales_black_to_zero():
    """Edge case: an all-black image is normalized to exactly 0.0."""
    result = model.preprocess_img(make_png_bytes(color="black"))
    assert np.allclose(result, 0.0)


def test_preprocess_values_in_unit_range():
    """Normalized pixel values stay between 0 and 1."""
    result = model.preprocess_img(make_png_bytes(color=(10, 120, 250)))
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_preprocess_tiny_image_is_upscaled():
    """Edge case: a 1x1 image is still resized to the expected shape."""
    result = model.preprocess_img(make_png_bytes(size=(1, 1)))
    assert result.shape == (1, 224, 224, 3)


@pytest.mark.parametrize("mode, color", [("L", 128), ("RGBA", (10, 20, 30, 255))])
def test_preprocess_rejects_non_rgb_images(mode, color):
    """Existing behaviour: grayscale and RGBA images fail the 3-channel reshape."""
    with pytest.raises(ValueError):
        model.preprocess_img(make_png_bytes(mode=mode, color=color))


def test_preprocess_rejects_non_image_data():
    """Non-image bytes raise an error from Pillow."""
    with pytest.raises(Exception):  # pylint: disable=broad-exception-caught
        model.preprocess_img(BytesIO(b"not an image"))


# ---------- predict_result ----------

def test_predict_result_returns_most_likely_class():
    """The index of the highest probability is returned (model is mocked)."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.05, 0.1, 0.7, 0.15]])
    with patch.object(model, "model", fake_model):
        result = model.predict_result(np.zeros((1, 224, 224, 3)))
    assert result == 2
    fake_model.predict.assert_called_once()


def test_predict_result_tie_returns_first_index():
    """Edge case: equal probabilities resolve to the first class."""
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([[0.5, 0.5, 0.0]])
    with patch.object(model, "model", fake_model):
        assert model.predict_result(np.zeros((1, 224, 224, 3))) == 0
